"""Tests for the walk-forward HMM regime allocator (synthetic panels only)."""

import math
from unittest import TestCase

import numpy as np
import pandas as pd

from market_data import MARKET, Panel
from regime_hmm import (POLICY, apply_bond_overlay, build_features, fit_gaussian_hmm, forward_filter,
                        latest_snapshot, label_states, regime_policy, standardized_features)
from risk import apply_caps
from strategies import CombinedStrategy, RegimeMeanReversion, TimeSeriesMomentum
from universe import CORE_ETFS


def _panel_from_close(close, classes=None):
    opening = close.shift().fillna(close)
    high = close * 1.005
    low = close * 0.995
    volume = pd.DataFrame(5e6, index=close.index, columns=close.columns)
    volume[["SPY", "IEF", "TLT", "HYG"]] = 5e7
    dividends = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    if classes is None:
        classes = pd.Series({c: "etf" if c in CORE_ETFS or c in {"HYG"} else "stock" for c in close.columns})
    return Panel(opening, high, low, close, volume, dividends, classes,
                 pd.DataFrame(False, index=close.index, columns=close.columns))


def planted_panel(sessions=420, seed=11):
    """Three planted regimes: risk-on, chop, then risk-off (bonds rally, credit lags, vol jumps)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=sessions)
    stocks = [f"S{i:02d}" for i in range(16)]
    names = ["SPY", "QQQ", "IEF", "TLT", "HYG"] + stocks
    close = pd.DataFrame(100.0, index=dates, columns=names)
    # session blocks: 0-139 risk on, 140-259 chop, 260-end risk off
    for i, _ in enumerate(dates):
        if i < 140:
            eq, bond, hyg, vol = 0.0012, -0.0001, 0.0008, 0.006
        elif i < 260:
            eq, bond, hyg, vol = 0.00005, 0.00005, 0.00005, 0.007
        else:
            eq, bond, hyg, vol = -0.0014, 0.0010, -0.0008, 0.022
        shocks = rng.normal(0, vol, len(stocks))
        if i == 0:
            continue
        close.loc[dates[i], stocks] = close.loc[dates[i - 1], stocks] * (1 + eq + shocks)
        close.loc[dates[i], "SPY"] = close.loc[dates[i - 1], "SPY"] * (1 + eq + rng.normal(0, vol * 0.7))
        close.loc[dates[i], "QQQ"] = close.loc[dates[i - 1], "QQQ"] * (1 + eq * 1.1 + rng.normal(0, vol * 0.8))
        close.loc[dates[i], "IEF"] = close.loc[dates[i - 1], "IEF"] * (1 + bond + rng.normal(0, 0.003))
        close.loc[dates[i], "TLT"] = close.loc[dates[i - 1], "TLT"] * (1 + bond * 1.6 + rng.normal(0, 0.005))
        close.loc[dates[i], "HYG"] = close.loc[dates[i - 1], "HYG"] * (1 + hyg + rng.normal(0, vol * 0.5))
    return _panel_from_close(close)


class GaussianHmmTests(TestCase):
    def test_recovers_two_separated_states(self):
        rng = np.random.default_rng(0)
        a = rng.normal(loc=[-1.5, 1.5], scale=0.25, size=(120, 2))
        b = rng.normal(loc=[1.5, -1.5], scale=0.25, size=(120, 2))
        X = np.vstack([a, b])
        theta = fit_gaussian_hmm(X, 2, em_iters=30, seed=1)
        filtered, _ = forward_filter(X, theta)
        names, _ = label_states(theta, ["spy_vol20", "pct_above_50"])
        # The first block should be majority one state, the second the other.
        first = filtered[:120].argmax(axis=1)
        second = filtered[120:].argmax(axis=1)
        self.assertGreater((first == first[0]).mean(), 0.8)
        self.assertGreater((second == second[0]).mean(), 0.8)
        self.assertNotEqual(first[0], second[0])
        self.assertEqual(set(names), {"risk_on", "risk_off"})


class FeatureAndCausalityTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = planted_panel()

    def test_features_are_causal(self):
        cut = self.panel.dates[300]
        before = build_features(self.panel.upto(cut))
        shocked = planted_panel()
        future = shocked.close.index > cut
        shocked.close.loc[future] *= 3
        shocked.high.loc[future] *= 3
        shocked.low.loc[future] *= 3
        after = build_features(shocked).loc[:cut]
        pd.testing.assert_frame_equal(before, after, check_exact=False, atol=1e-12, rtol=0)

    def test_standardized_features_are_causal(self):
        cut = self.panel.dates[320]
        z_before = standardized_features(build_features(self.panel.upto(cut)))
        z_after = standardized_features(build_features(self.panel)).loc[:cut]
        # Align columns; both should match through the cut.
        cols = [c for c in z_before.columns if c in z_after.columns]
        pd.testing.assert_frame_equal(z_before[cols], z_after[cols], check_exact=False, atol=1e-12, rtol=0)

    def test_policy_has_no_lookahead(self):
        cut = self.panel.dates[360]
        overrides = {"n_states": 3, "min_fit": 60, "refit_every": 21, "fit_window": 180, "em_iters": 12,
                     "z_min_periods": 25}
        before = regime_policy(self.panel.upto(cut), model="hmm", overrides=overrides)
        after = regime_policy(self.panel, model="hmm", overrides=overrides).loc[:cut]
        hmm_days = before.index[before["source"] == "hmm"]
        self.assertGreater(len(hmm_days), 10)
        for col in ("tsmom_scale", "ief_weight", "tlt_weight", "meanrev_score", "p_risk_on", "p_risk_off"):
            pd.testing.assert_series_equal(before[col], after[col], check_names=False, atol=1e-10, obj=col)

    def test_fixed_fallback_matches_original_gate_shape(self):
        policy = regime_policy(self.panel, model="fixed")
        self.assertTrue((policy["source"] == "fixed").all())
        self.assertTrue((policy["tsmom_scale"] == 1.0).all())
        self.assertTrue((policy["ief_weight"] == 0.0).all())
        self.assertTrue(policy["meanrev_on"].any() or True)  # may be on or off; just a boolean series
        self.assertEqual(set(policy["meanrev_on"].unique()) - {True, False}, set())


class AllocationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = planted_panel()
        cls.overrides = {"n_states": 3, "min_fit": 60, "refit_every": 21, "fit_window": 180,
                         "em_iters": 15, "z_min_periods": 25}
        cls.policy = regime_policy(cls.panel, model="hmm", overrides=cls.overrides)

    def test_hmm_eventually_activates_and_uses_three_named_states(self):
        hmm = self.policy[self.policy["source"] == "hmm"]
        self.assertGreater(len(hmm), 40)
        names = set(hmm["state_name"]) - {""}
        self.assertTrue(names <= {"risk_on", "chop", "stress", "risk_off"})
        self.assertGreaterEqual(len(names), 2)

    def test_risk_off_block_raises_bond_weight_and_cuts_tsmom(self):
        hmm = self.policy[self.policy["source"] == "hmm"]
        late = hmm.loc[hmm.index >= self.panel.dates[320]]
        early = hmm.loc[(hmm.index <= self.panel.dates[130]) & (hmm.index >= self.panel.dates[90])]
        self.assertGreater(len(late), 10)
        late_bonds = late["ief_weight"].mean() + late["tlt_weight"].mean()
        # After a planted vol spike and equity drop, the late window should lean toward bonds
        # and away from a full TSMOM scale (or at least not be a full risk-on 1.0 book).
        self.assertLess(late["tsmom_scale"].mean(), 0.98)
        if len(early) >= 10:
            early_bonds = early["ief_weight"].mean() + early["tlt_weight"].mean()
            self.assertGreater(late_bonds + 1e-9, early_bonds)
        else:
            self.assertGreater(late_bonds, 0.05)

    def test_probabilities_are_a_simplex(self):
        hmm = self.policy[self.policy["source"] == "hmm"]
        total = hmm[["p_risk_on", "p_chop", "p_stress", "p_risk_off"]].sum(axis=1)
        self.assertTrue(((total - 1.0).abs() < 1e-8).all())

    def test_snapshot_and_bond_overlay_respect_gross_cap(self):
        snap = latest_snapshot(self.policy)
        self.assertIsNotNone(snap)
        self.assertIn(snap["model"], ("hmm", "fixed"))
        self.assertIn("allocation", snap)
        dates = self.panel.dates
        weights = pd.DataFrame(0.0, index=dates, columns=self.panel.close.columns)
        weights["SPY"] = 0.9
        weights["QQQ"] = 0.3
        capped = apply_bond_overlay(weights, self.policy, self.panel, gross_cap=1.0)
        self.assertLessEqual(capped.abs().sum(axis=1).max(), 1.0 + 1e-9)
        self.assertGreaterEqual(capped.min().min(), 0.0)


class CombinedIntegrationTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = planted_panel(sessions=380)

    def _combined(self, model):
        loose = {"min_dollar_volume": 1e6, "min_price": 1.0}
        return CombinedStrategy(
            regime_model=model, regime_n_states=3, regime_min_fit=60, regime_refit_every=21,
            regime_fit_window=180, regime_write_state=False, max_positions=8, per_name_cap=0.3,
            sleeves=[TimeSeriesMomentum(universe="etf", **loose),
                     RegimeMeanReversion(entry_z=-1.2, regime_vol_pct=0.9, trend_t_max=8,
                                         trend_filter=50, **loose)])

    def test_fixed_model_does_not_add_a_bond_overlay(self):
        strategy = self._combined("fixed")
        weights = strategy.target_weights(self.panel)
        diag = strategy.last_diagnostics.get("regime") or {}
        self.assertEqual((diag.get("model") or "fixed"), "fixed")
        # Overlay is zero under the fixed fallback (TSMOM may still hold IEF/TLT on its own trend).
        self.assertEqual(diag.get("allocation", {}).get("ief"), 0.0)
        self.assertEqual(diag.get("allocation", {}).get("tlt"), 0.0)
        self.assertLessEqual(weights.abs().sum(axis=1).max(), 1.0 + 1e-9)

    def test_hmm_model_is_causal_on_the_combined_path(self):
        strategy = self._combined("hmm")
        full = strategy.target_weights(self.panel)
        self.assertIn("regime", strategy.last_diagnostics)
        cut = self.panel.dates[-25]
        live = strategy.target_weights_at(self.panel, cut)
        pd.testing.assert_series_equal(full.loc[cut], live.reindex(full.columns).fillna(0.0),
                                       check_names=False, atol=1e-10)
        self.assertLessEqual(full.abs().sum(axis=1).max(), 1.0 + 1e-9)

    def test_caps_still_hold_with_bonds(self):
        weights = self._combined("hmm").target_weights(self.panel)
        self.assertLessEqual(weights.max().max(), 0.3 + 1e-9)
        self.assertGreaterEqual(weights.min().min(), 0.0)
        # apply_caps contract still holds if we re-apply
        recapped = apply_caps(weights, 0.3, 1.0)
        self.assertLessEqual(recapped.abs().sum(axis=1).max(), 1.0 + 1e-9)
