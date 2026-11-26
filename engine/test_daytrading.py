"""Tests for the day-trading component, the advisory hook and the Schwab market-data safety guard."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd

import intraday as it
from advisory import adjust_entry, load_verdicts
from costs import CostModel
from schwab_data import _check_url
from test_strategy_framework import synthetic_panel


def synthetic_bars(sessions, symbols=("SPY", "SA", "SB"), seed=5, breakout=True):
    rng = np.random.default_rng(seed)
    rows = []
    for d in sessions:
        for s in symbols:
            price = 100.0
            for k in range(78):
                minute = 570 + 5 * k
                drift = 0.0015 if breakout and k >= 6 else 0.0
                close = price * (1 + drift + rng.normal(0, 0.001))
                rows.append((s, d, minute, price, max(price, close) * 1.0005, min(price, close) * 0.9995, close,
                             1e5 * (3 if k < 6 else 1)))
                price = close
    return pd.DataFrame(rows, columns=["symbol", "session_date", "minute", "open", "high", "low", "close", "volume"])


def context_for(sessions, symbols=("SPY", "SA", "SB")):
    return pd.DataFrame([(d, s, 100.0, True, False, 5e8, 0.01, True, True) for d in sessions for s in symbols],
                        columns=["session_date", "symbol", "prev_close", "trend_up", "trend_down", "adv",
                                 "daily_vol", "liquid", "regime_ok"])


class DayTradingTests(TestCase):
    def setUp(self):
        self.sessions = [str(d.date()) for d in pd.bdate_range("2026-06-01", periods=20)]
        self.bars = synthetic_bars(self.sessions)
        self.params = it.ORBParams(min_rel_volume=0.0, max_positions=3, target_r=None)
        ranges = it.opening_ranges(self.bars, self.params, 5)
        self.candidates = it.candidate_trades(self.bars, ranges, context_for(self.sessions), self.params, 5, CostModel())

    def test_pdt_limit_caps_day_trades_in_any_five_sessions(self):
        result = it.run_sessions(self.candidates, self.sessions, self.params, 10_000.0)
        per_day = result.daily.set_index("session_date")["trades"]
        rolling = per_day.rolling(5, min_periods=1).sum()
        self.assertLessEqual(rolling.max(), 3)
        self.assertGreater(result.skipped["pdt"], 0)
        unlimited = it.run_sessions(self.candidates, self.sessions, it.ORBParams(min_rel_volume=0.0, max_positions=3,
                                                                                 target_r=None, pdt_max_day_trades=None))
        self.assertGreater(len(unlimited.trades), len(result.trades))

    def test_pdt_not_applied_at_or_above_25k(self):
        result = it.run_sessions(self.candidates, self.sessions, self.params, 30_000.0)
        self.assertEqual(result.skipped["pdt"], 0)

    def test_flat_by_close_and_costs(self):
        result = it.run_sessions(self.candidates, self.sessions, self.params, 10_000.0)
        self.assertTrue((result.trades["exit_minute"] <= 960).all())
        self.assertTrue((result.trades["cost"] > 0).all())          # spread + slippage + sell fees
        start = result.daily.set_index("session_date")["start_equity"]
        trades = result.trades.assign(notional=result.trades["shares"] * result.trades["raw_entry"])
        for day, group in trades.groupby("session_date"):
            for minute in group["entry_minute"]:
                live = group[(group["entry_minute"] <= minute) & (group["exit_minute"] > minute)]
                self.assertLessEqual(live["notional"].sum(), start[day] * 1.0001)   # no intraday leverage

    def test_daily_context_uses_previous_session_only(self):
        panel = synthetic_panel()
        params = it.ORBParams()
        context = it.daily_context(panel, ["SPY", "SA"], params)
        day = panel.dates[100]
        row = context[(context.session_date == day.strftime("%Y-%m-%d")) & (context.symbol == "SA")].iloc[0]
        self.assertAlmostEqual(row.prev_close, panel.close["SA"].iloc[99])

    def test_breaker_stops_new_entries_after_daily_loss(self):
        params = it.ORBParams(min_rel_volume=0.0, max_positions=1, target_r=None, daily_loss_limit=0.0,
                              pdt_max_day_trades=None)
        losers = self.candidates.copy()
        losers["raw_exit"] = losers["raw_entry"] * 0.99
        losers["exit_minute"] = losers["entry_minute"] + 5
        result = it.run_sessions(losers, self.sessions, params, 10_000.0)
        self.assertTrue((result.daily["trades"] <= 1).all())
        self.assertGreater(result.skipped["breaker"], 0)


class AdvisoryTests(TestCase):
    def test_rules_only_reduce_risk(self):
        from advisory import Verdict
        self.assertEqual(adjust_entry(None, 0.2, 100), (0.2, None))
        self.assertEqual(adjust_entry(Verdict("A", "RED"), 0.2, 100), (0.0, "advisory_red"))
        self.assertEqual(adjust_entry(Verdict("A", "YELLOW"), 0.2, 100), (0.1, "advisory_yellow"))
        self.assertEqual(adjust_entry(Verdict("A", "GREEN", conviction=1.0), 0.2, 100), (0.2, None))
        self.assertEqual(adjust_entry(Verdict("A", "GREEN", entry_low=90, entry_high=95), 0.2, 100)[0], 0.0)
        self.assertEqual(adjust_entry(Verdict("A", "YELLOW"), 0.2, 100, yellow_scale=5)[0], 0.2)   # never enlarges

    def test_missing_malformed_and_stale_verdicts_change_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            missing, warnings = load_verdicts("2026-11-25", json_path=Path(directory) / "none.json")
            self.assertEqual((missing, warnings), ({}, []))
            bad = Path(directory) / "bad.json"
            bad.write_text("{not json")
            verdicts, warnings = load_verdicts("2026-11-25", json_path=bad)
            self.assertEqual(verdicts, {})
            self.assertTrue(warnings)
            good = Path(directory) / "good.json"
            good.write_text(json.dumps({"verdicts": {"aapl": {"flag": "red", "asof": "2026-11-24"},
                                                     "MSFT": {"flag": "YELLOW", "asof": "2026-10-01"},
                                                     "NVDA": {"flag": "MAYBE"}}}))
            verdicts, _ = load_verdicts("2026-11-25", json_path=good)
            self.assertEqual(list(verdicts), ["AAPL"])                  # stale and invalid flags ignored
            db = sqlite3.connect(":memory:")
            from advisory import TABLE_SQL
            db.execute(TABLE_SQL)
            db.execute("INSERT INTO advisory_verdicts VALUES ('AAPL','2026-11-25','GREEN',0.8,NULL,NULL,NULL,NULL,'ai',NULL)")
            verdicts, _ = load_verdicts("2026-11-25", json_path=good, connection=db)
            self.assertEqual(verdicts["AAPL"].flag, "GREEN")             # newer verdict wins


class SchwabSafetyTests(TestCase):
    def test_only_market_data_endpoints_are_allowed(self):
        _check_url("https://api.schwabapi.com/marketdata/v1/quotes?symbols=SPY")
        _check_url("https://api.schwabapi.com/marketdata/v1/pricehistory?symbol=SPY")
        for url in ("https://api.schwabapi.com/trader/v1/accounts",
                    "https://api.schwabapi.com/trader/v1/accounts/123/orders",
                    "https://api.schwabapi.com/marketdata/../trader/v1/orders",
                    "https://example.com/marketdata/v1/quotes"):
            with self.assertRaises(PermissionError):
                _check_url(url)


class NoAutomatedOrderPathTests(TestCase):
    def test_engine_code_never_references_order_endpoints(self):
        engine = Path(__file__).resolve().parent
        for path in sorted(engine.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if path.name in ("schwab_data.py", "test_daytrading.py"):
                continue                     # schwab_data only lists /trader/ as FORBIDDEN; this test names it
            for needle in ("/trader/", "/api/live/", "previewOrder", "placeOrder", "place_order"):
                self.assertNotIn(needle, text, f"{path.name} references {needle}")
