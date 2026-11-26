"""Unit tests for regime_allocation (synthetic panels only; no live data, no orders)."""

import math
from unittest import TestCase

import numpy as np
import pandas as pd

from costs import CostModel
from market_data import MARKET, Panel
from universe import CORE_ETFS

import regime_allocation as ra


def _panel_from_close(close, classes=None):
    opening = close.shift().fillna(close)
    high = close * 1.006
    low = close * 0.994
    volume = pd.DataFrame(8e7, index=close.index, columns=close.columns)
    dividends = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    # a small quarterly dividend on SPY so the yield component has something to work with
    if "SPY" in close.columns:
        for i, day in enumerate(close.index):
            if i and i % 63 == 0:
                dividends.loc[day, "SPY"] = 0.4
    if classes is None:
        extra = {"BIL", "SGOV", "SH", "PSQ", "VIXY", "HYG", "LQD", "JNK"}
        classes = pd.Series({c: "etf" if c in CORE_ETFS or c in extra else "stock" for c in close.columns})
    return Panel(opening, high, low, close, volume, dividends, classes,
                 pd.DataFrame(False, index=close.index, columns=close.columns))


def make_panel(sessions=420, seed=3, crash_at=280, stretch=True):
    """Uptrend, then a vol spike/crash, with breadth and credit names."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=sessions)
    names = list(dict.fromkeys(
        ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
         "BIL", "SGOV", "IEF", "TLT", "GLD", "SH", "PSQ", "VIXY", "HYG", "LQD"]
    ))
    close = pd.DataFrame(100.0, index=dates, columns=names)
    close.loc[dates[0], "BIL"] = 91.5
    close.loc[dates[0], "SGOV"] = 100.2
    close.loc[dates[0], "IEF"] = 95.0
    close.loc[dates[0], "TLT"] = 90.0
    close.loc[dates[0], "GLD"] = 180.0
    close.loc[dates[0], "SH"] = 12.0
    close.loc[dates[0], "PSQ"] = 10.0
    close.loc[dates[0], "VIXY"] = 20.0
    for i in range(1, sessions):
        if i < crash_at:
            eq, vol = (0.0018 if stretch else 0.0006), 0.006
        elif i < crash_at + 25:
            eq, vol = -0.012, 0.028
        else:
            eq, vol = 0.0008, 0.010
        shock = rng.normal(0, vol)
        close.loc[dates[i], "SPY"] = close.loc[dates[i - 1], "SPY"] * (1 + eq + shock)
        for t in ("QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP"):
            close.loc[dates[i], t] = close.loc[dates[i - 1], t] * (1 + eq + rng.normal(0, vol * 1.1))
        close.loc[dates[i], "BIL"] = close.loc[dates[i - 1], "BIL"] * (1 + 0.00016 + rng.normal(0, 0.0002))
        close.loc[dates[i], "SGOV"] = close.loc[dates[i - 1], "SGOV"] * (1 + 0.00016 + rng.normal(0, 0.0002))
        close.loc[dates[i], "IEF"] = close.loc[dates[i - 1], "IEF"] * (1 + (0.0001 if i >= crash_at else -0.00005) + rng.normal(0, 0.003))
        close.loc[dates[i], "TLT"] = close.loc[dates[i - 1], "TLT"] * (1 + (0.0003 if i >= crash_at else -0.0001) + rng.normal(0, 0.007))
        close.loc[dates[i], "GLD"] = close.loc[dates[i - 1], "GLD"] * (1 + 0.0003 + rng.normal(0, 0.008))
        # inverse: roughly opposite SPY, with decay
        spy_ret = close.loc[dates[i], "SPY"] / close.loc[dates[i - 1], "SPY"] - 1
        close.loc[dates[i], "SH"] = close.loc[dates[i - 1], "SH"] * (1 - spy_ret - 0.00005)
        close.loc[dates[i], "PSQ"] = close.loc[dates[i - 1], "PSQ"] * (1 - spy_ret * 1.05 - 0.00005)
        close.loc[dates[i], "VIXY"] = max(5.0, close.loc[dates[i - 1], "VIXY"] * (1 - 0.001 + (0.08 if i >= crash_at and i < crash_at + 25 else 0) + rng.normal(0, 0.03)))
        close.loc[dates[i], "HYG"] = close.loc[dates[i - 1], "HYG"] * (1 + eq * 0.4 + rng.normal(0, vol * 0.4))
        close.loc[dates[i], "LQD"] = close.loc[dates[i - 1], "LQD"] * (1 + 0.0001 + rng.normal(0, 0.003))
    return _panel_from_close(close), crash_at


def _vix(panel, crash_at):
    dates = panel.dates
    vix = pd.Series(15.0, index=dates)
    vix3m = pd.Series(17.0, index=dates)
    for i, _d in enumerate(dates):
        if i >= crash_at and i < crash_at + 25:
            vix.iloc[i] = 32 + (i - crash_at)
            vix3m.iloc[i] = 24.0
        elif i > 60:
            vix.iloc[i] = 14.0 + 2 * math.sin(i / 15)
            vix3m.iloc[i] = 17.0
    return pd.DataFrame({"vix": vix, "vix3m": vix3m})


class HelperTests(TestCase):
    def test_clip_weights_normalize(self):
        self.assertEqual(ra._clip_weights(0.5, 0.3, 0.2), (0.5, 0.3, 0.2))
        s = ra._clip_weights(-1, 2, 2)
        self.assertAlmostEqual(sum(s), 1.0)
        self.assertTrue(all(x >= 0 for x in s))

    def test_piecewise_unit_bounds(self):
        self.assertEqual(ra.piecewise_unit(0.1, 0.35, 0.80), 0.0)
        self.assertEqual(ra.piecewise_unit(0.80, 0.35, 0.80), 1.0)
        self.assertAlmostEqual(ra.piecewise_unit(0.575, 0.35, 0.80), 0.5)

    def test_logistic_monotone(self):
        xs = np.linspace(-3, 3, 13)
        ys = ra.logistic(xs)
        self.assertTrue(np.all(np.diff(ys) > 0))


class SignalTests(TestCase):
    def setUp(self):
        self.panel, self.crash = make_panel()
        self.vix = _vix(self.panel, self.crash)
        self.signals = ra.build_signals(self.panel, vix=self.vix)

    def test_no_lookahead_in_signals(self):
        mid = self.panel.dates[250]
        early = ra.build_signals(self.panel.upto(mid), vix=self.vix.loc[:mid])
        later = self.signals.loc[:mid]
        # mutating the future must not change the past
        cols = ["rv21", "overextension", "drawdown", "vix_pct"]
        for c in cols:
            a, b = early[c].dropna(), later[c].reindex(early[c].dropna().index).dropna()
            joined = pd.concat([a, b], axis=1, join="inner").dropna()
            if len(joined) < 10:
                continue
            delta = (joined.iloc[:, 0] - joined.iloc[:, 1]).abs().max()
            self.assertLess(delta, 1e-9, msg=c)

    def test_overextension_rises_when_stretched(self):
        # late in the pre-crash uptrend the score should exceed the post-crash washout
        pre = self.signals["overextension"].iloc[self.crash - 10:self.crash].mean()
        post = self.signals["overextension"].iloc[self.crash + 10:self.crash + 25].mean()
        self.assertGreater(pre, 0.2)
        self.assertGreater(pre, post - 0.05)

    def test_vol_bucket_crisis_during_crash(self):
        during = self.signals["vol_bucket"].iloc[self.crash + 5:self.crash + 20]
        self.assertTrue((during.isin(["high", "crisis", "elevated"])).mean() >= 0.5)

    def test_washout_fires_in_crash(self):
        washed = ra.washout_flag(self.signals)
        self.assertTrue(bool(washed.iloc[self.crash:self.crash + 30].any()))


class SchemeTests(TestCase):
    def setUp(self):
        self.panel, self.crash = make_panel()
        self.vix = _vix(self.panel, self.crash)
        self.signals = ra.build_signals(self.panel, vix=self.vix)
        # cheap sleeve returns (no full TSMOM) for scheme math
        tr = self.panel.total_return()
        self.sleeves = pd.DataFrame({
            "risk_on": tr["SPY"],
            "defensive": tr["BIL"],
            "hedge": tr["SH"],
            "spy": tr["SPY"],
        })

    def test_static_weights_sum_to_one(self):
        w = ra.scheme_static(self.signals, mix=(0.7, 0.2, 0.1))
        self.assertTrue(np.allclose(w.sum(axis=1), 1.0))

    def test_overextension_scheme_monotone_in_oe(self):
        # all else equal, a higher OE score must not reduce the raw hedge target before hysteresis
        sig = self.signals.copy()
        # build a short synthetic path with rising OE, no washout, vol not crisis
        sig = sig.iloc[200:240].copy()
        sig["overextension"] = np.linspace(0.2, 0.9, len(sig))
        sig["vol_bucket"] = "normal"
        sig["vol_direction"] = "falling"
        sig["rsi14"] = 60.0
        sig["drawdown"] = -0.01
        sig["rv21_pct"] = 0.4
        sig["black_swan"] = 0.0
        raw = ra.scheme_overextension(sig, cap=0.25, enter=0.45, exit=0.30, shape="linear")
        # once the gate is on, hedge should be non-decreasing as OE rises
        on = raw["hedge"] > 0
        if on.any():
            h = raw.loc[on, "hedge"].to_numpy()
            self.assertTrue(np.all(np.diff(h) >= -1e-12))

    def test_hysteresis_does_not_flicker(self):
        sig = self.signals.iloc[200:260].copy()
        # oscillate OE around the enter/exit band
        oe = np.where((np.arange(len(sig)) % 2) == 0, 0.46, 0.40)
        sig["overextension"] = oe
        sig["vol_bucket"] = "normal"
        sig["vol_direction"] = "falling"
        sig["rsi14"] = 60.0
        sig["drawdown"] = -0.01
        sig["rv21_pct"] = 0.4
        sig["black_swan"] = 0.0
        raw = ra.scheme_overextension(sig, cap=0.25, enter=0.45, exit=0.30, shape="linear")
        # after first entry, should stay on through the 0.40 prints (above exit 0.30)
        first_on = int(np.argmax(raw["hedge"].to_numpy() > 0))
        self.assertTrue((raw["hedge"].iloc[first_on:] > 0).all())

    def test_washout_releases_hedge(self):
        sig = self.signals.iloc[200:250].copy()
        sig["overextension"] = 0.8
        sig["vol_bucket"] = "high"
        sig["vol_direction"] = "falling"
        sig["rsi14"] = 60.0
        sig["drawdown"] = -0.02
        sig["rv21_pct"] = 0.8
        sig["black_swan"] = 0.0
        # washout in the last 10 days
        sig.iloc[-10:, sig.columns.get_loc("rsi14")] = 25.0
        sig.iloc[-10:, sig.columns.get_loc("drawdown")] = -0.12
        raw = ra.scheme_overextension(sig, cap=0.25, enter=0.45, exit=0.30)
        self.assertGreater(raw["hedge"].iloc[5], 0.0)
        self.assertEqual(raw["hedge"].iloc[-1], 0.0)

    def test_vol_target_cuts_risk_when_vol_is_high(self):
        w = ra.scheme_vol_target(self.signals, self.sleeves, target_vol=0.08)
        pre = w["risk_on"].iloc[self.crash - 30:self.crash - 5].mean()
        during = w["risk_on"].iloc[self.crash + 5:self.crash + 20].mean()
        self.assertLess(during, pre)

    def test_trend_broken_is_risk_on_in_uptrend(self):
        w = ra.scheme_trend_broken(self.signals, ma="ma200", hedge_frac=0.2, def_frac=0.3)
        # well into the uptrend, after MA200 exists, should be 100% risk-on
        mid = self.crash - 20
        if self.signals["ma200"].iloc[mid] == self.signals["ma200"].iloc[mid]:
            if self.signals["above_ma200"].iloc[mid] == 1:
                self.assertAlmostEqual(w["risk_on"].iloc[mid], 1.0)

    def test_smoothing_limits_step(self):
        raw = pd.DataFrame({"risk_on": [1.0, 0.0], "defensive": [0.0, 1.0], "hedge": [0.0, 0.0]},
                           index=pd.bdate_range("2024-01-02", periods=2))
        sm = ra.smooth_weights(raw, max_step=0.08, lam=1.0)
        self.assertLessEqual(abs(sm["risk_on"].iloc[1] - sm["risk_on"].iloc[0]), 0.08 + 1e-9)

    def test_apply_weights_lags_and_charges_costs(self):
        weights = ra.scheme_static(self.signals, mix=(0.6, 0.3, 0.1))
        zero = CostModel(spread_coef=0.0, min_half_spread_bps=0.0, max_half_spread_bps=0.0,
                         impact_coef=0.0, sec_fee_per_million=0.0, taf_per_share=0.0)
        # first two days of a jump in weights incur costs with the default model, not with a zero model
        path_c = ra.apply_weights(self.sleeves, weights, cost_model=CostModel(), lag=1, smooth=True)
        path_z = ra.apply_weights(self.sleeves, weights, cost_model=zero, lag=1, smooth=True)
        self.assertGreater(float(path_c["costs"].sum()), 0.0)
        self.assertGreater(float(path_z["returns"].sum()), float(path_c["returns"].sum()) - 1e-12)
        # lag: applied weights on day t equal smoothed weights of t-1
        sm = ra.smooth_weights(weights)
        self.assertTrue(np.allclose(path_c["applied"].iloc[5].to_numpy(), sm.iloc[4].to_numpy(), atol=1e-12))

    def test_holdout_selection_ignores_holdout_returns(self):
        # evaluate on a short window, then scramble holdout sleeve returns and re-select
        eval_start = self.panel.dates[180]
        holdout_start = self.panel.dates[320]
        study1 = ra.evaluate_candidates(self.panel, self.signals, self.sleeves,
                                        eval_start=eval_start, holdout_start=holdout_start)
        sleeves2 = self.sleeves.copy()
        rng = np.random.default_rng(99)
        hold = sleeves2.loc[holdout_start:]
        for c in ("risk_on", "defensive", "hedge", "spy"):
            sleeves2.loc[holdout_start:, c] = rng.normal(0, 0.02, size=len(hold))
        study2 = ra.evaluate_candidates(self.panel, self.signals, sleeves2,
                                        eval_start=eval_start, holdout_start=holdout_start)
        self.assertEqual(study1["champion"], study2["champion"])

    def test_available_defensive_prefers_bil(self):
        w, present = ra.available_defensive(self.panel)
        self.assertIn("BIL", w)
        self.assertNotIn("SGOV", w)
        self.assertAlmostEqual(sum(w.values()), 1.0)

    def test_candidate_count_is_small(self):
        n = len(ra.candidate_list())
        self.assertGreaterEqual(n, 15)
        self.assertLessEqual(n, 40)


class ResearchSmokeTests(TestCase):
    def test_research_on_synthetic_panel_no_holdout_log(self):
        panel, crash = make_panel(sessions=400)
        # monkeypatch vix/macro loaders via passing panel only; research will try real files, which is fine
        # but we want a closed test: call evaluate + recommend, not the full I/O research()
        vix = _vix(panel, crash)
        signals = ra.build_signals(panel, vix=vix)
        tr = panel.total_return()
        sleeves = pd.DataFrame({"risk_on": tr["SPY"], "defensive": tr["BIL"], "hedge": tr["SH"], "spy": tr["SPY"]})
        study = ra.evaluate_candidates(panel, signals, sleeves,
                                       eval_start=panel.dates[150], holdout_start=panel.dates[320])
        rec = ra.recommend(study)
        self.assertIn(rec["decision"], {"stay_static", "stay_static_with_interest", "recommend_challenger"})
        self.assertTrue(rec["livePaperUnchanged"])
        self.assertIsNotNone(study["champion"])
        self.assertIn("static_100_0_0", study["rankedIds"])
