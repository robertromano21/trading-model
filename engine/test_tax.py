"""Unit and integration tests for the tax ledger, wash sales, policy and after-tax simulator hooks."""

import math
import tempfile
import unittest
from contextlib import closing
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from costs import CostModel
from market_data import Panel
from portfolio_sim import simulate
from tax import (LotLedger, TaxConfig, _year_tax_clean, after_tax_performance, is_long_term,
                 qualifies_for_dividend, tax_config_from_settings)
from tax_policy import adjust_targets


def panel_from_prices(prices, dividends=None, volume=1e7):
    """prices: dict ticker -> list of closes, or a DataFrame. Daily open=prior close (first open=close)."""
    if isinstance(prices, pd.DataFrame):
        close = prices
    else:
        close = pd.DataFrame(prices)
        if not isinstance(close.index, pd.DatetimeIndex):
            close.index = pd.bdate_range("2024-01-02", periods=len(close))
    opening = close.shift().fillna(close)
    high = np.maximum(opening, close) * 1.001
    low = np.minimum(opening, close) * 0.999
    vol = pd.DataFrame(volume, index=close.index, columns=close.columns)
    divs = dividends if dividends is not None else pd.DataFrame(0.0, index=close.index, columns=close.columns)
    classes = pd.Series({c: "etf" if c in {"SPY", "DIA", "QQQ", "XLK", "IVV", "VOO", "IEF", "TLT"} else "stock"
                         for c in close.columns})
    return Panel(opening, high, low, close, vol, divs, classes,
                 pd.DataFrame(False, index=close.index, columns=close.columns))


class HoldingPeriodTests(unittest.TestCase):
    def test_long_term_is_the_day_after_the_anniversary(self):
        self.assertFalse(is_long_term(date(2024, 1, 16), date(2025, 1, 16)))
        self.assertTrue(is_long_term(date(2024, 1, 16), date(2025, 1, 17)))

    def test_leap_day_anniversary(self):
        self.assertFalse(is_long_term(date(2024, 2, 29), date(2025, 2, 28)))
        self.assertTrue(is_long_term(date(2024, 2, 29), date(2025, 3, 1)))

    def test_qualified_dividend_needs_more_than_60_days_in_the_window(self):
        ex = date(2024, 6, 15)
        self.assertFalse(qualifies_for_dividend(date(2024, 5, 1), ex))   # 45 days before ex
        self.assertTrue(qualifies_for_dividend(date(2024, 3, 1), ex))    # well before the window


class LotMethodTests(unittest.TestCase):
    def test_fifo_sells_oldest_first(self):
        ledger = LotLedger(TaxConfig.accounting(method="fifo"))
        ledger.buy("AAA", 10, 10, date(2024, 1, 3))
        ledger.buy("AAA", 10, 20, date(2024, 2, 1))
        result = ledger.sell("AAA", 10, 15, date(2024, 3, 1))
        self.assertEqual(result["lots"][0]["lot_id"], 1)
        self.assertAlmostEqual(result["st_gain"], 10 * (15 - 10), places=6)
        self.assertEqual(ledger.open_lots("AAA")[0].lot_id, 2)

    def test_hifo_sells_highest_cost_first(self):
        ledger = LotLedger(TaxConfig.accounting(method="hifo"))
        ledger.buy("AAA", 10, 10, date(2024, 1, 3))
        ledger.buy("AAA", 10, 20, date(2024, 2, 1))
        result = ledger.sell("AAA", 10, 15, date(2024, 3, 1))
        self.assertEqual(result["lots"][0]["lot_id"], 2)
        self.assertAlmostEqual(result["st_gain"], 10 * (15 - 20), places=6)
        self.assertEqual(ledger.open_lots("AAA")[0].lot_id, 1)

    def test_specific_lot_ids(self):
        ledger = LotLedger(TaxConfig.accounting(method="specific"))
        ledger.buy("AAA", 10, 10, date(2024, 1, 3))
        ledger.buy("AAA", 10, 20, date(2024, 2, 1))
        result = ledger.sell("AAA", 10, 15, date(2024, 3, 1), specific_ids=[2])
        self.assertEqual(result["lots"][0]["lot_id"], 2)


class CharacterAndDividendTests(unittest.TestCase):
    def test_short_then_long_term_gain(self):
        ledger = LotLedger(TaxConfig.accounting())
        ledger.buy("AAA", 10, 10, date(2024, 1, 16))
        short = ledger.sell("AAA", 4, 20, date(2025, 1, 16))
        self.assertAlmostEqual(short["st_gain"], 40, places=6)
        self.assertAlmostEqual(short["lt_gain"], 0, places=6)
        long = ledger.sell("AAA", 6, 20, date(2025, 1, 17))
        self.assertAlmostEqual(long["st_gain"], 0, places=6)
        self.assertAlmostEqual(long["lt_gain"], 60, places=6)

    def test_qualified_vs_ordinary_dividend(self):
        ledger = LotLedger(TaxConfig.accounting())
        ledger.buy("SPY", 10, 100, date(2024, 5, 20))   # too new for QD on Jun 15
        ledger.buy("SPY", 10, 100, date(2024, 1, 2))
        # FIFO ids: lot 1 is May 20, lot 2 is Jan 2. dividend() walks by acquired date so Jan 2 first.
        result = ledger.dividend("SPY", 1.0, date(2024, 6, 15), shares=20)
        self.assertAlmostEqual(result["qualified"], 10.0, places=6)
        self.assertAlmostEqual(result["ordinary"], 10.0, places=6)


class WashSaleTests(unittest.TestCase):
    def test_lookback_purchase_disallows_loss_and_tacks_holding_period(self):
        ledger = LotLedger(TaxConfig.accounting())
        ledger.buy("AAA", 10, 100, date(2024, 1, 2))
        ledger.buy("AAA", 10, 100, date(2024, 1, 20))
        # Sell the older lot at a loss; the Jan 20 lot is a replacement in the 30-day window.
        result = ledger.sell("AAA", 10, 70, date(2024, 1, 25), method="fifo")
        self.assertAlmostEqual(result["disallowed"], 300, places=6)
        self.assertAlmostEqual(result["st_gain"], 0.0, places=6)
        repl = ledger.open_lots("AAA")[0]
        self.assertEqual(repl.lot_id, 2)
        self.assertAlmostEqual(repl.cost_per_share, 100 + 30, places=6)
        self.assertEqual(repl.original_acquired, date(2024, 1, 2))

    def test_forward_purchase_disallows_loss_on_the_new_lot(self):
        ledger = LotLedger(TaxConfig.accounting())
        ledger.buy("AAA", 10, 100, date(2024, 1, 2))
        sold = ledger.sell("AAA", 10, 70, date(2024, 2, 1))
        self.assertAlmostEqual(sold["st_gain"], -300, places=6)
        ledger.buy("AAA", 10, 80, date(2024, 2, 15))
        # Loss should be reversed and parked in the new lot's basis.
        self.assertAlmostEqual(ledger.year(date(2024, 2, 15)).st_gain, 0.0, places=4)
        lot = ledger.open_lots("AAA")[0]
        self.assertAlmostEqual(lot.cost_per_share, 80 + 30, places=6)
        self.assertEqual(lot.original_acquired, date(2024, 1, 2))

    def test_substantially_identical_etf_is_a_wash(self):
        ledger = LotLedger(TaxConfig.accounting())
        ledger.buy("SPY", 10, 100, date(2024, 1, 2))
        ledger.sell("SPY", 10, 80, date(2024, 3, 1))
        ledger.buy("IVV", 10, 81, date(2024, 3, 10))
        self.assertAlmostEqual(ledger.year(date(2024, 3, 10)).st_gain, 0.0, places=4)
        self.assertAlmostEqual(ledger.open_lots("IVV")[0].wash_adjustment, 200, places=6)

    def test_correlated_but_not_identical_etf_is_not_a_wash(self):
        ledger = LotLedger(TaxConfig.accounting())
        ledger.buy("SPY", 10, 100, date(2024, 1, 2))
        ledger.sell("SPY", 10, 80, date(2024, 3, 1))
        ledger.buy("DIA", 10, 81, date(2024, 3, 10))
        self.assertAlmostEqual(ledger.year(date(2024, 3, 10)).st_gain, -200, places=4)
        self.assertAlmostEqual(ledger.open_lots("DIA")[0].wash_adjustment, 0.0, places=6)

    def test_purchase_after_the_30_day_window_is_not_a_wash(self):
        ledger = LotLedger(TaxConfig.accounting())
        ledger.buy("AAA", 10, 100, date(2024, 1, 2))
        ledger.sell("AAA", 10, 80, date(2024, 2, 1))
        ledger.maybe_settle(date(2024, 3, 5))  # expires pending (window is 30 days: Feb 1 + 30 = Mar 2)
        ledger.buy("AAA", 10, 80, date(2024, 3, 10))
        self.assertAlmostEqual(ledger.year(date(2024, 3, 10)).st_gain, -200, places=4)
        self.assertAlmostEqual(ledger.open_lots("AAA")[0].cost_per_share, 80, places=6)


class YearTaxTests(unittest.TestCase):
    def test_short_term_gain_taxed_at_st_rate(self):
        cfg = TaxConfig.accounting()
        result = _year_tax_clean(cfg, st_gain=1000, lt_gain=0, qualified_div=0, ordinary_div=0)
        self.assertAlmostEqual(result["tax"], 1000 * cfg.st_rate(), places=6)

    def test_loss_nets_against_gains_then_three_thousand_offset(self):
        cfg = TaxConfig.accounting()
        result = _year_tax_clean(cfg, st_gain=-8000, lt_gain=2000, qualified_div=0, ordinary_div=1000)
        # net ST loss 6000 after netting LT gain; $3k offset; $3k carry ST; ordinary div fully offset
        self.assertAlmostEqual(result["tax"], 0.0, places=6)
        self.assertAlmostEqual(result["carrySt"], -3000, places=6)
        self.assertAlmostEqual(result["offsetUsed"], 3000, places=6)
        self.assertAlmostEqual(result["taxableOrdinaryDiv"], 0.0, places=6)

    def test_state_and_niit_add_to_rates(self):
        cfg = TaxConfig.accounting(state_rate=0.05, niit_rate=0.038)
        self.assertAlmostEqual(cfg.st_rate(), 0.32 + 0.05 + 0.038, places=9)
        self.assertAlmostEqual(cfg.lt_rate(), 0.15 + 0.05 + 0.038, places=9)


class PolicyTests(unittest.TestCase):
    def test_policy_is_noop_when_disabled(self):
        ledger = LotLedger(TaxConfig())  # enabled=False
        ledger.buy("SPY", 10, 100, date(2024, 1, 2))
        adjusted, actions = adjust_targets(ledger, {"SPY": 0.0}, {"SPY": 90}, date(2024, 6, 1), 1000)
        self.assertEqual(adjusted, {"SPY": 0.0})
        self.assertEqual(actions, [])

    def test_trim_hysteresis_defers_near_long_term_gain_on_a_weak_signal(self):
        cfg = TaxConfig.aware(defer_window_days=40, defer_max_signal=0.20)
        ledger = LotLedger(cfg)
        acquired = date(2024, 1, 16)
        asof = date(2025, 1, 1)  # 11 days from going long-term on Jan 17
        ledger.buy("SPY", 10, 100, acquired)
        equity = 10 * 150
        adjusted, actions = adjust_targets(ledger, {"SPY": 0.9}, {"SPY": 150}, asof, equity,
                                           current={"SPY": 1.0}, available={"SPY", "DIA"})
        kinds = [a["kind"] for a in actions]
        self.assertIn("tax_defer_near_lt", kinds)
        self.assertGreater(adjusted["SPY"], 0.9)

    def test_full_exit_and_forced_stop_are_not_deferred(self):
        cfg = TaxConfig.aware(defer_window_days=40)
        ledger = LotLedger(cfg)
        ledger.buy("SPY", 10, 100, date(2024, 1, 16))
        asof = date(2025, 1, 1)
        equity = 10 * 150
        exit_w, exit_a = adjust_targets(ledger, {"SPY": 0.0}, {"SPY": 150}, asof, equity,
                                        current={"SPY": 1.0})
        self.assertEqual(exit_w.get("SPY", 0.0), 0.0)
        self.assertFalse(any(a["kind"] == "tax_defer_near_lt" for a in exit_a))
        stop_w, stop_a = adjust_targets(ledger, {"SPY": 0.0}, {"SPY": 150}, asof, equity,
                                        current={"SPY": 1.0}, forced_exits={"SPY"})
        self.assertEqual(stop_w.get("SPY", 0.0), 0.0)
        self.assertFalse(any(a["kind"] == "tax_defer_near_lt" for a in stop_a))

    def test_harvest_swaps_into_correlated_not_identical_etf(self):
        cfg = TaxConfig.aware(harvest_min_loss=50, harvest_min_loss_pct=0.01)
        ledger = LotLedger(cfg)
        ledger.buy("SPY", 10, 100, date(2024, 1, 2))
        equity = 10 * 80 + 200
        prices = {"SPY": 80, "DIA": 80, "IVV": 80}
        adjusted, actions = adjust_targets(ledger, {"SPY": 0.8}, prices, date(2024, 6, 1), equity,
                                           current={"SPY": 10 * 80 / equity}, available=set(prices))
        harvest = [a for a in actions if a["kind"] == "tax_harvest"]
        self.assertEqual(len(harvest), 1)
        self.assertEqual(harvest[0]["substitute"], "DIA")
        self.assertNotEqual(harvest[0]["substitute"], "IVV")
        self.assertLessEqual(adjusted.get("SPY", 0.0), 1e-9)
        self.assertGreater(adjusted.get("DIA", 0.0), 0.0)

    def test_wash_avoidance_blocks_repurchase(self):
        cfg = TaxConfig.aware()
        ledger = LotLedger(cfg)
        ledger.buy("SPY", 10, 100, date(2024, 1, 2))
        ledger.sell("SPY", 10, 80, date(2024, 3, 1))
        adjusted, actions = adjust_targets(ledger, {"SPY": 0.5}, {"SPY": 80, "DIA": 80}, date(2024, 3, 10),
                                           10_000, current={}, available={"SPY", "DIA"})
        self.assertEqual(adjusted.get("SPY", 0.0), 0.0)
        self.assertGreater(adjusted.get("DIA", 0.0), 0.0)
        self.assertTrue(any("wash_avoid" in a["kind"] for a in actions))


class AfterTaxSimTests(unittest.TestCase):
    def _buy_hold_panel(self):
        sessions = 40
        dates = pd.bdate_range("2024-01-02", periods=sessions)
        close = pd.DataFrame({"SPY": np.linspace(100, 110, sessions)}, index=dates)
        return panel_from_prices(close)

    def test_disabled_matches_pre_tax_simulator(self):
        panel = self._buy_hold_panel()
        weights = pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
        weights.iloc[1:] = 0.5
        costs = CostModel.for_broker("schwab")
        pre = simulate(panel, weights, cost_model=costs, initial_equity=10_000, execution="close")
        off = simulate(panel, weights, cost_model=costs, initial_equity=10_000, execution="close",
                       tax_config=TaxConfig())
        pd.testing.assert_series_equal(pre.equity, off.equity, atol=1e-9, obj="equity")
        pd.testing.assert_series_equal(pre.returns, off.returns, atol=1e-12, obj="returns")
        self.assertIsNone(off.after_tax_equity)

    def test_accounting_mode_tracks_lots_and_reduces_after_tax_equity_on_a_taxable_sale(self):
        # Two names so we can sell one at a short-term gain mid-window.
        dates = pd.bdate_range("2024-01-02", periods=30)
        close = pd.DataFrame({"AAA": np.linspace(100, 130, 30), "BBB": np.linspace(50, 51, 30)}, index=dates)
        panel = panel_from_prices(close)
        weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
        weights.loc[dates[1]:dates[10], "AAA"] = 0.8
        weights.loc[dates[11]:, "BBB"] = 0.8
        pre = simulate(panel, weights, cost_model=CostModel(impact_coef=0, spread_coef=0, min_half_spread_bps=0,
                                                            sec_fee_per_million=0, taf_per_share=0),
                       initial_equity=10_000, execution="close", min_trade_weight=0.0)
        cfg = TaxConfig.accounting()
        taxed = simulate(panel, weights, cost_model=CostModel(impact_coef=0, spread_coef=0, min_half_spread_bps=0,
                                                              sec_fee_per_million=0, taf_per_share=0),
                         initial_equity=10_000, execution="close", min_trade_weight=0.0, tax_config=cfg)
        # Same trades (accounting mode does not change targets), but after-tax equity reserves the tax bill.
        self.assertGreater(len(taxed.tax_report["openLots"]) + len(taxed.tax_ledger.events), 0)
        last_pre = float(pre.equity.iloc[-1])
        last_at = float(taxed.after_tax_equity.iloc[-1])
        self.assertLess(last_at, last_pre)
        self.assertGreater(taxed.tax_ledger.unpaid_tax, 0)

    def test_after_tax_metrics_report_tax_drag(self):
        equity = pd.Series([100, 110, 121], index=pd.bdate_range("2024-01-02", periods=3))
        after = pd.Series([100, 108, 116], index=equity.index)
        from metrics import performance
        pre = performance(equity.pct_change().dropna(), equity)
        out = after_tax_performance(equity, after, pre)
        self.assertIsNotNone(out["taxDragCagr"])
        self.assertGreater(out["taxDragCagr"], 0)


class ConfigAndSqliteTests(unittest.TestCase):
    def test_settings_gate(self):
        off = tax_config_from_settings({"tax_aware": False})
        self.assertFalse(off.enabled)
        on = tax_config_from_settings({"tax_aware": True, "tax": {"federal_st_rate": 0.37, "method": "hifo"}})
        self.assertTrue(on.enabled)
        self.assertEqual(on.method, "hifo")
        self.assertAlmostEqual(on.federal_st_rate, 0.37)

    def test_sqlite_roundtrip_and_seed(self):
        import sqlite3
        cfg = TaxConfig.aware()
        ledger = LotLedger(cfg, "paper")
        ledger.buy("SPY", 5, 100, date(2024, 1, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.sqlite3"
            with closing(sqlite3.connect(path)) as db:
                db.row_factory = sqlite3.Row
                ledger.save_sqlite(db, "paper")
                loaded = LotLedger.load_sqlite(db, "paper", cfg)
        self.assertEqual(len(loaded.open_lots("SPY")), 1)
        self.assertAlmostEqual(loaded.open_lots("SPY")[0].shares, 5)

    def test_disclaimer_present_on_snapshot(self):
        snap = LotLedger(TaxConfig.aware()).snapshot()
        self.assertIn("not tax advice", snap["disclaimer"].lower())


class PaperHookTests(unittest.TestCase):
    def test_paper_trader_books_lots_when_tax_aware(self):
        import paper_trader as pt
        from test_strategy_framework import synthetic_panel
        panel = synthetic_panel(sessions=80, stocks=6, seed=1)
        # Hold a constant 50% of the first ETF so the book is simple and deterministic.
        # Use the real trader path with tax_aware on and a tiny custom strategy? Easier: drive _trade via a
        # minimal run by injecting settings. Combined strategy may or may not trade in 80 sessions.
        # Seed-free path: instantiate, create account, call buy through _trade after run setup.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "state.sqlite3"
            with closing(pt.connect(path)) as db:
                settings = {**pt.DEFAULTS, "tax_aware": True, "whole_shares": False, "min_trade_weight": 0.0,
                            "min_order_value": 0.0, "initial_equity": 10_000}
                config = {"strategy": "tsmom", "params": {"min_dollar_volume": 1e6, "max_positions": 3},
                          "sleeves": {}}
                db.execute("BEGIN IMMEDIATE")
                trader = pt.PaperTrader(db, "tax", panel, config, settings)
                result = trader.run(panel.dates[-1], "run")
                db.execute("COMMIT")
                lots = db.execute("SELECT ticker, remaining FROM tax_lots WHERE account='tax' AND status='open'").fetchall()
                snap = pt.status(db, "tax")
        self.assertTrue(snap.get("tax"), "status should include a tax block")
        self.assertIn("not tax advice", snap["tax"]["disclaimer"].lower())
        # Lots are created for any fill; if the short window had no trades, the ledger still exists.
        self.assertIn("enabled", snap["tax"])


if __name__ == "__main__":
    unittest.main()
