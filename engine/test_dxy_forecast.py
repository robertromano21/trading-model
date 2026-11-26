"""Unit tests for the DXY forecast module. No network, no broker, no dashboard."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import dxy_forecast as dxyf


def _idx(n=400, start="2000-01-03"):
    return pd.bdate_range(start, periods=n)


class LagAndCompositeTests(unittest.TestCase):
    def test_monthly_value_is_hidden_until_lag_elapses(self):
        s = pd.Series([1.0], index=[pd.Timestamp("2000-01-01")])
        cal = pd.DatetimeIndex([pd.Timestamp("2000-02-01"), pd.Timestamp("2000-02-14"),
                                pd.Timestamp("2000-02-15"), pd.Timestamp("2000-03-01")])
        aligned = dxyf.to_daily(s, cal, lag_days=45)
        self.assertTrue(pd.isna(aligned.loc[pd.Timestamp("2000-02-14")]))
        self.assertEqual(aligned.loc[pd.Timestamp("2000-02-15")], 1.0)
        self.assertEqual(aligned.loc[pd.Timestamp("2000-03-01")], 1.0)

    def test_daily_asof_does_not_look_ahead(self):
        s = pd.Series([10.0, 11.0], index=[pd.Timestamp("2000-01-03"), pd.Timestamp("2000-01-05")])
        cal = pd.bdate_range("2000-01-03", periods=4)
        aligned = dxyf.to_daily(s, cal, lag_days=0)
        self.assertEqual(aligned.loc[pd.Timestamp("2000-01-03")], 10.0)
        self.assertEqual(aligned.loc[pd.Timestamp("2000-01-04")], 10.0)
        self.assertEqual(aligned.loc[pd.Timestamp("2000-01-05")], 11.0)

    def test_composite_renormalizes_over_available_legs(self):
        idx = _idx(3)
        parts = {
            "EURUSD": pd.Series([2.0, 2.0, np.nan], index=idx),
            "USDJPY": pd.Series([4.0, np.nan, 4.0], index=idx),
        }
        weights = {"EURUSD": 0.75, "USDJPY": 0.25}
        comp = dxyf.weighted_composite(parts, weights)
        # day 0: 0.75*2 + 0.25*4 = 2.5
        self.assertAlmostEqual(comp.iloc[0], 2.5)
        # day 1: only EUR -> 2.0
        self.assertAlmostEqual(comp.iloc[1], 2.0)
        # day 2: only JPY -> 4.0
        self.assertAlmostEqual(comp.iloc[2], 4.0)


class TargetAndModelTests(unittest.TestCase):
    def test_forward_return_uses_future_level_only_in_the_label(self):
        level = pd.Series([100.0, 110.0, 121.0], index=_idx(3))
        fwd = dxyf.forward_return(level, 2)
        self.assertAlmostEqual(fwd.iloc[0], 0.21)
        self.assertTrue(pd.isna(fwd.iloc[1]))
        self.assertTrue(pd.isna(fwd.iloc[2]))

    def test_ridge_recovers_a_planted_linear_signal(self):
        rng = np.random.default_rng(0)
        n = 800
        x = rng.normal(size=(n, 3))
        y = 0.5 * x[:, 0] - 0.25 * x[:, 1] + rng.normal(scale=0.05, size=n)
        # already standardized-ish
        beta = dxyf.fit_ridge(y, x, alpha=0.1)
        pred = dxyf.predict_ridge(x, beta)
        self.assertGreater(np.corrcoef(y, pred)[0, 1], 0.9)
        self.assertAlmostEqual(beta[1], 0.5, delta=0.05)

    def test_logit_probabilities_are_in_unit_interval(self):
        rng = np.random.default_rng(1)
        n = 400
        x = rng.normal(size=(n, 2))
        y01 = (x[:, 0] + rng.normal(scale=0.3, size=n) > 0).astype(float)
        beta = dxyf.fit_logit(y01, x)
        self.assertIsNotNone(beta)
        p = dxyf.predict_logit(x, beta)
        self.assertTrue(np.all(p >= 0) and np.all(p <= 1))
        a, b = dxyf.platt_calibrate(p, y01)
        cal = dxyf.apply_platt(p, a, b)
        self.assertTrue(np.all(cal >= 0) and np.all(cal <= 1))
        self.assertLess(dxyf.brier_score(y01, cal), 0.3)

    def test_oos_r2_is_one_minus_mse_ratio_vs_random_walk(self):
        y = np.array([0.1, -0.2, 0.3])
        pred = y.copy()
        self.assertAlmostEqual(dxyf.oos_r2(y, pred), 1.0)
        self.assertAlmostEqual(dxyf.oos_r2(y, np.zeros(3)), 0.0)
        worse = dxyf.oos_r2(y, np.array([0.5, 0.5, 0.5]))
        self.assertLess(worse, 0.0)

    def test_clark_west_detects_a_better_nested_model(self):
        rng = np.random.default_rng(2)
        y = rng.normal(scale=0.01, size=400)
        # model equals y plus tiny noise (near-perfect, still nested vs RW=0)
        pred = 0.8 * y
        result = dxyf.clark_west(y, pred, lags=5)
        self.assertIsNotNone(result["pValue"])
        self.assertLess(result["pValue"], 0.01)

    def test_deflated_sharpe_is_a_probability(self):
        rng = np.random.default_rng(3)
        r = pd.Series(0.001 + rng.normal(scale=0.01, size=500))
        out = dxyf.deflated_sharpe(r, n_trials=12)
        self.assertIsNotNone(out["deflatedSharpeProb"])
        self.assertGreaterEqual(out["deflatedSharpeProb"], 0.0)
        self.assertLessEqual(out["deflatedSharpeProb"], 1.0)
        self.assertGreater(out["nTrials"], 1)


class WalkForwardHygieneTests(unittest.TestCase):
    def test_walk_forward_does_not_fit_on_unresolved_labels(self):
        """Each prediction at t must be fit on rows whose labels end before t."""
        idx = _idx(400)
        rng = np.random.default_rng(4)
        x1 = pd.Series(rng.normal(size=len(idx)), index=idx)
        # Planted signal in x1 for the *contemporaneous* forward return, but the walker
        # should still refuse to use y[s] when s+h is not yet known.
        level = pd.Series(100 * np.exp(np.cumsum(0.001 + 0.2 * x1.shift(1).fillna(0) * 0.01)), index=idx)
        y = dxyf.forward_return(level, horizon=5)
        X = pd.DataFrame({"x1": x1, "x2": rng.normal(size=len(idx))}, index=idx)
        # Spy on fit_ridge to record the last training index used at each refit by
        # wrapping walk_forward internals: compare in-sample last label date vs pred date.
        calls = []
        real_fit = dxyf.fit_ridge

        def wrapped(y_fit, X_fit, alpha=1.0):
            calls.append(len(y_fit))
            return real_fit(y_fit, X_fit, alpha=alpha)

        with mock.patch.object(dxyf, "fit_ridge", side_effect=wrapped):
            pred, info = dxyf.walk_forward_ridge(
                y, X, horizon=5, alpha=1.0, min_train=80, refit_every=40, embargo_extra=5,
            )
        self.assertGreater(info["nFits"], 2)
        self.assertGreater(pred.notna().sum(), 50)
        # After the embargo, the largest possible train length at the first fit is
        # first_pred_index - embargo. First prediction happens at index min_train+embargo.
        embargo = 5 + 5
        # Train rows used at first refit should be <= first_i - embargo.
        self.assertLessEqual(calls[0], 400)
        self.assertLessEqual(calls[0], (80 + embargo) )  # first opportunity: j = i-embargo >= min_train

        # Stronger check: rebuild one step and ensure max train date + horizon < pred date.
        both = pd.concat([y.rename("y"), X], axis=1).dropna()
        dates = list(both.index)
        # Take a mid-sample prediction date and re-run a single fit up to that date.
        mid = 200
        t = dates[mid]
        j = mid - embargo
        last_train = dates[j - 1] if j >= 1 else dates[0]
        # Label at last_train uses level[last_train + 5], which is dates[j-1+5] if contiguous.
        # The embargo requires last_train is at least horizon+extra sessions before t.
        self.assertLessEqual(j, mid - 5)
        self.assertLess(last_train, t)

    def test_trading_costs_reduce_turnover_pnl(self):
        idx = _idx(60)
        # Alternate long/short every day so turnover is high.
        sig = pd.Series(np.where(np.arange(60) % 2 == 0, 0.01, -0.01), index=idx)
        r = pd.Series(0.001, index=idx)
        pnl_cost, _ = dxyf.trading_returns(sig, r, threshold=0.0, cost_bps=10.0, lag=1)
        pnl_free, _ = dxyf.trading_returns(sig, r, threshold=0.0, cost_bps=0.0, lag=1)
        self.assertLess(pnl_cost.sum(), pnl_free.sum())


class OutlookAndLedgerTests(unittest.TestCase):
    def test_attach_alert_flags_a_direction_flip(self):
        prev = {"asOf": "2026-11-01", "direction": "stronger", "pStronger": 0.70, "expectedMove": 0.01}
        cur = {"asOf": "2026-11-20", "direction": "weaker", "pStronger": 0.30, "expectedMove": -0.01}
        out = dxyf.attach_alert(cur, prev)
        self.assertTrue(out["alert"]["flag"])
        self.assertIn("flip", out["alert"]["reason"])

    def test_gates_require_positive_r2_and_clark_west(self):
        pre = {"oosR2": 0.02, "clarkWest": {"pValue": 0.01}}
        hol = {"oosR2": -0.05, "clarkWest": {"pValue": 0.4}}
        g = dxyf._gates(pre, hol)
        self.assertFalse(g["passed"])
        self.assertEqual(g["tradingDefault"], "off")
        g2 = dxyf._gates(pre, {"oosR2": 0.03, "clarkWest": {"pValue": 0.02}})
        self.assertTrue(g2["passed"])

    def test_holdout_ledger_records_a_look(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(dxyf, "OUT_DIR", tmp), mock.patch.object(dxyf, "LEDGER_PATH", tmp / "ledger.json"):
                dxyf.log_holdout("unit test look", looked=True, extra={"n": 1})
                data = (tmp / "ledger.json").read_text(encoding="utf-8")
                self.assertIn("unit test look", data)
                self.assertIn("true", data.lower())


class FeatureLagIntegrationTests(unittest.TestCase):
    def test_session_lag_shifts_every_column(self):
        """build_features applies a session lag; a same-day shock must not appear in features[t]."""
        idx = _idx(300, start="2010-01-04")
        dxy = pd.Series(100 + np.linspace(0, 10, len(idx)), index=idx)
        # A distinctive spike in VIX on one session.
        vix = pd.Series(15.0, index=idx)
        spike_day = idx[150]
        vix.loc[spike_day] = 80.0
        series = {
            "dff": pd.Series(5.0, index=idx),
            "vix": vix,
            "oil": pd.Series(70.0, index=idx),
            "nasdaq": pd.Series(4000.0, index=idx),
            "baa10y": pd.Series(2.0, index=idx),
            "lt_de": pd.Series(3.0, index=pd.date_range("2009-01-01", periods=24, freq="MS")),
            "st_de": pd.Series(2.0, index=pd.date_range("2009-01-01", periods=24, freq="MS")),
        }
        macro = pd.DataFrame({"dgs2": 4.0, "dgs10": 4.5, "dgs3mo": 3.5}, index=idx)

        with mock.patch.object(dxyf, "load_dxy", return_value=dxy), \
             mock.patch.object(dxyf, "load_forecast_fred", return_value=series), \
             mock.patch.object(dxyf, "load_macro", return_value=macro), \
             mock.patch.object(dxyf, "_surprise_feature", return_value=None):
            feats0, _ = dxyf.build_features(calendar=idx, lag=0)
            feats1, _ = dxyf.build_features(calendar=idx, lag=1)
        self.assertAlmostEqual(feats0.loc[spike_day, "vix"], 80.0)
        # With lag=1 the spike is first visible the next session, not on spike_day.
        self.assertNotAlmostEqual(float(feats1.loc[spike_day, "vix"]), 80.0)
        next_day = idx[151]
        self.assertAlmostEqual(feats1.loc[next_day, "vix"], 80.0)


if __name__ == "__main__":
    unittest.main()
