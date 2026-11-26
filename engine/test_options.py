"""Tests for options pricing, the paper options book, the vol sleeve, overlays and the top detector (no network)."""

import math
import unittest

import numpy as np
import pandas as pd

from options_book import ModelPricer, OptionsBook, OptionsConfig, RiskRefusal, reg_t_requirement
from options_pricing import (OptionFees, analyze_legs, bsm_greeks, bsm_price, black76_price, build_strategy, implied_vol,
                             occ_symbol, parse_occ, price_legs, probability_of_profit, expected_move, STRATEGIES)


class PricingTests(unittest.TestCase):
    def test_put_call_parity_and_black76(self):
        S, K, T, v, r, q = 100, 95, 0.4, 0.25, 0.03, 0.01
        c, p = bsm_price(S, K, T, v, r, q, "call"), bsm_price(S, K, T, v, r, q, "put")
        self.assertAlmostEqual(c - p, S * math.exp(-q * T) - K * math.exp(-r * T), places=9)
        F = S * math.exp((r - q) * T)
        self.assertAlmostEqual(black76_price(F, K, T, v, r, "call"), c, places=9)

    def test_greeks_match_finite_differences(self):
        S, K, T, v = 100, 105, 0.3, 0.3
        g = bsm_greeks(S, K, T, v, 0.02, 0.0, "put")
        h = 1e-3
        delta = (bsm_price(S + h, K, T, v, 0.02, 0, "put") - bsm_price(S - h, K, T, v, 0.02, 0, "put")) / (2 * h)
        vega = (bsm_price(S, K, T, v + h, 0.02, 0, "put") - bsm_price(S, K, T, v - h, 0.02, 0, "put")) / (2 * h) / 100
        theta = (bsm_price(S, K, T - 1 / 365, v, 0.02, 0, "put") - bsm_price(S, K, T, v, 0.02, 0, "put"))
        self.assertAlmostEqual(g["delta"], delta, places=5)
        self.assertAlmostEqual(g["vega"], vega, places=5)
        self.assertAlmostEqual(g["theta"], theta, places=3)

    def test_implied_vol_round_trip(self):
        for kind in ("call", "put"):
            price = bsm_price(50, 52, 0.2, 0.41, 0.04, 0.0, kind)
            self.assertAlmostEqual(implied_vol(price, 50, 52, 0.2, 0.04, 0.0, kind), 0.41, places=5)
        self.assertIsNone(implied_vol(0.0001, 50, 30, 0.2))   # below intrinsic

    def test_occ_symbols_match_schwab_py(self):
        self.assertEqual(occ_symbol("SPY", "2026-12-18", "C", 600), "SPY   261218C00600000")
        self.assertEqual(occ_symbol("SPXW", "2026-12-18", "P", 5040.5), "SPXW  261218P05040500")
        self.assertEqual(parse_occ("QQQ   240420P00500000")["strike"], 500.0)

    def test_fees(self):
        f = OptionFees()
        self.assertAlmostEqual(f.fee(2, 1.5, "buy_to_open"), 2 * (0.65 + 0.011), places=6)
        self.assertGreater(f.fee(2, 1.5, "sell_to_open"), f.fee(2, 1.5, "buy_to_open"))
        self.assertAlmostEqual(f.fee(1, 0.05, "buy_to_close"), 0.011, places=6)   # $0.05 buy-to-close waiver

    def test_payoff_analytics(self):
        legs = [{"kind": "call", "qty": 1, "strike": 100, "expiration": "2026-12-18", "price": 5},
                {"kind": "call", "qty": -1, "strike": 110, "expiration": "2026-12-18", "price": 2}]
        a = analyze_legs(legs, 100)
        self.assertEqual((a["maxLoss"], a["maxGain"], a["breakevens"]), (-300.0, 700.0, [103.0]))
        naked = analyze_legs([{"kind": "call", "qty": -1, "strike": 110, "expiration": "2026-12-18", "price": 2}], 100)
        self.assertTrue(naked["unboundedLoss"])
        self.assertIsNone(naked["maxLoss"])
        for name in STRATEGIES:
            self.assertTrue(build_strategy(name, 100, "2026-12-18", back_expiration="2027-01-15"))
        pop = probability_of_profit(legs, 100, "2026-11-25", 0.25)
        self.assertTrue(0 < pop < 1)
        self.assertAlmostEqual(expected_move(100, 0.2, 365)["oneSigma"], 20.0)

    def test_calendar_is_approximate(self):
        legs = price_legs(build_strategy("calendar", 100, "2026-12-18", back_expiration="2027-01-15"), 100, "2026-11-25", 0.25)
        self.assertTrue(analyze_legs(legs, 100, vol=0.25)["approximate"])


def pricer(vol=0.2):
    return ModelPricer(lambda d, u: vol)


class BookTests(unittest.TestCase):
    def test_10k_realism_and_limits(self):
        b = OptionsBook({"enabled": True})
        with self.assertRaises(RiskRefusal):     # one ATM SPY put costs more than 3% of $10k
            b.open("long_put", "SPY", build_strategy("long_put", 600, "2026-12-18"), pricer(), "2026-11-25", 600)
        with self.assertRaises(RiskRefusal):
            b.open("naked_call", "SPY", build_strategy("naked_call", 600, "2026-12-18"), pricer(), "2026-11-25", 600)
        with self.assertRaises(RiskRefusal):
            b.open("short_strangle", "XYZ", build_strategy("short_strangle", 20, "2026-12-18"), pricer(), "2026-11-25", 20)
        p = b.open("long_put", "XYZ", build_strategy("long_put", 20, "2026-12-18", strikes={"k": 19}), pricer(), "2026-11-25", 20)
        self.assertEqual(p["pricing"], "modeled")
        self.assertLess(p["entryCash"], 0)

    def test_reg_t(self):
        csp = [{"kind": "put", "qty": -1, "strike": 50, "expiration": "2026-12-18", "price": 1.0}]
        self.assertAlmostEqual(reg_t_requirement(csp, 52, "cash")["requirement"], 4900.0)
        margin = reg_t_requirement(csp, 52, "margin")["requirement"]
        self.assertAlmostEqual(margin, 100 * (max(0.2 * 52 - 2, 0.1 * 50) + 1.0))
        call = [{"kind": "call", "qty": -1, "strike": 55, "expiration": "2026-12-18", "price": 1.0}]
        self.assertFalse(reg_t_requirement(call, 52, "cash")["allowed"])
        covered = call + [{"kind": "stock", "qty": 100, "price": 52}]
        self.assertTrue(reg_t_requirement(covered, 52, "cash")["allowed"])

    def test_expiration_exercise_by_exception_and_pin(self):
        b = OptionsBook({"enabled": True})
        b.open("bull_call_spread", "XYZ", build_strategy("bull_call_spread", 20, "2026-12-18", strikes={"k1": 20, "k2": 21}),
               pricer(), "2026-11-25", 20)
        b.daily(pricer(), "2026-12-18", {"XYZ": 22.0})
        self.assertFalse(b.positions)
        rec = b.closed[-1]
        self.assertAlmostEqual(rec["exitCash"] + rec.get("fees", 0) * 0, 100.0 - 2 * 0.661 - 0.0, delta=3.0)  # ~$100 spread value
        b2 = OptionsBook({"enabled": True})
        b2.open("bull_put_spread", "XYZ", build_strategy("bull_put_spread", 20, "2026-12-18", strikes={"k1": 18, "k2": 20}),
                pricer(), "2026-11-25", 20)
        b2.daily(pricer(), "2026-12-18", {"XYZ": 20.05})
        self.assertTrue(any(e["event"] == "pin_risk" for e in b2.events))

    def test_early_assignment_before_ex_dividend(self):
        b = OptionsBook({"enabled": True})
        b.open("covered_call", "XYZ", [{"kind": "call", "qty": -1, "strike": 15, "expiration": "2027-01-15"}], pricer(0.15),
               "2026-11-25", 20, covered_shares=100)
        out = b.daily(pricer(0.15), "2026-12-01", {"XYZ": 20.0}, next_dividends={"XYZ": ("2026-12-03", 0.50)})
        self.assertTrue(out and "early assignment" in out[0]["exitReason"])

    def test_short_stock_borrow_dividend_gap_stop_and_margin(self):
        cfg = {"enabled": True}
        b = OptionsBook(cfg)
        with self.assertRaises(RiskRefusal):
            b.short("XYZ", 10, 50, "2026-12-01")              # disabled by default
        b = OptionsBook({**cfg, "short_stock_enabled": True})
        b.short("XYZ", 10, 50, "2026-12-01", {"XYZ": 50}, pricer())
        cash0 = b.cash
        b.daily(pricer(), "2026-12-02", {"XYZ": 51}, opens={"XYZ": 50.5}, dividends={"XYZ": 0.25})
        self.assertAlmostEqual(cash0 - b.cash, 10 * 51 * 0.005 / 360 + 2.5, places=6)
        out = b.daily(pricer(), "2026-12-03", {"XYZ": 57}, opens={"XYZ": 58})
        self.assertIn("gap through squeeze stop", out[0]["exitReason"])
        self.assertEqual(out[0]["exit"], 58)
        cash_acct = OptionsBook({**cfg, "short_stock_enabled": True, "account_type": "cash"})
        with self.assertRaises(RiskRefusal):
            cash_acct.short("XYZ", 10, 50, "2026-12-01")


def synthetic_panel(n=320, seed=1):
    from market_data import Panel
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n)
    cols = ["VIXY", "SVIX", "SPY"]
    close = pd.DataFrame({c: 50 * np.exp(np.cumsum(rng.normal(0, 0.03, n))) for c in cols}, index=dates)
    o = close.shift().fillna(close) * (1 + rng.normal(0, 0.003, (n, 3)))
    hi, lo = np.maximum(o, close) * 1.01, np.minimum(o, close) * 0.99
    vol = pd.DataFrame(1e6, index=dates, columns=cols)
    return Panel(o, hi, lo, close, vol, pd.DataFrame(0.0, index=dates, columns=cols), pd.Series({c: "etf" for c in cols}))


class VolSleeveTests(unittest.TestCase):
    def setUp(self):
        import vol_sleeve
        self.vs = vol_sleeve
        rng = np.random.default_rng(4)
        idx = pd.bdate_range("2023-06-01", "2026-06-30")
        v = 16 + np.cumsum(rng.normal(0, 0.8, len(idx)))
        v = np.clip(v, 10, 60)
        v[500:510] += 20                                       # a spike
        self.vix = pd.DataFrame({"vix": v, "vix3m": v * 1.05 + 0.5}, index=idx)

    def test_no_lookahead(self):
        import vix_data
        panel = synthetic_panel()
        f = vix_data.features(self.vix)
        full, _, _ = self.vs.simulate(panel, f, start=panel.dates[0], events=[])
        cut = panel.upto(panel.dates[200])
        part, _, _ = self.vs.simulate(cut, f.loc[:panel.dates[200]], start=panel.dates[0], events=[])
        pd.testing.assert_series_equal(full.loc[:panel.dates[200]], part, check_names=False)

    def test_exits_are_explicit(self):
        sl = self.vs.VolSleeve()
        out = sl.step("2026-01-02", {"vix": 12, "ratio": 0.85, "pct": 0.1, "z": -1, "peak20": 13}, {"VIXY": 30, "SVIX": 20})
        self.assertEqual(out[0]["reason"], "calm_low_vol")
        sl.filled(out[0], 30, 10, "2026-01-05")
        out = sl.step("2026-01-06", {"vix": 30, "ratio": 1.1, "pct": 0.97, "z": 3, "peak20": 30}, {"VIXY": 40})
        self.assertEqual(out[0]["reason"], "take_profit_spike")
        sl2 = self.vs.VolSleeve()
        sl2.pos["SVIX"] = {"shares": 5, "entry": 20, "entryDate": "x", "held": 0, "peak": 20, "sawBackwardation": False}
        self.assertEqual(sl2.step("d", {"vix": 20, "ratio": 0.95, "pct": 0.5, "z": 0, "peak20": 25}, {"SVIX": 20}, sessions_to_event=1)[0]["reason"],
                         "macro_event")
        self.assertEqual(sl2.hard_stop("SVIX"), 18.0)


class OverlayTests(unittest.TestCase):
    def test_disabled_by_default_and_sizing(self):
        from strategies.options_overlay import OptionsOverlay
        self.assertEqual(OptionsOverlay().intents("2026-01-02", {"equity": 1e4, "spots": {}}), [])
        ov = OptionsOverlay(enabled=True, mode="inverse_etf", trigger="always")
        out = ov.intents("2026-01-02", {"equity": 10_000, "spots": {"SH": 40.0}, "share_positions": {}})
        self.assertEqual(out[0]["action"], "buy")
        self.assertEqual(out[0]["shares"], math.floor(0.02 * 10_000 / 0.08 / 40))


class TopDetectorTests(unittest.TestCase):
    def test_multiple_testing_helpers(self):
        import top_detector as td
        p = [0.01, 0.04, 0.03, 0.5]
        self.assertTrue(all(a >= b for a, b in zip(td.holm(p), p)))
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(td.bh(p), p)))
        self.assertEqual(td._episodes([0, 1, 1, 0, 0, 0, 1], 2), 2)

    def test_circular_permutation_null(self):
        import top_detector as td
        rng = np.random.default_rng(0)
        y = (rng.random(400) < 0.3).astype(float)
        s = (rng.random(400) < 0.2).astype(float)
        self.assertGreater(td._circular_p(s, y, n=500), 0.05)   # unrelated signal is not significant


if __name__ == "__main__":
    unittest.main()
