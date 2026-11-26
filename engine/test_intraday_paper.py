import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

import crypto_intraday as crypto
import daytrader


class CryptoIntradayTest(unittest.TestCase):
    def test_paper_fill_uses_following_open_and_retry_is_idempotent(self):
        step = crypto.INTERVAL
        latest = 100 * step
        bars = {t: (100.0, 100.0 if t < latest - 7 * step else 110.0)
                for t in range(latest - 40 * step, latest + step, step)}
        load = lambda _, cutoff: {t: bar for t, bar in bars.items() if t <= cutoff}
        load_dollar = lambda _: {}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            first = crypto.run(directory, latest + step + 120, load, load_dollar)
            self.assertEqual([a["pending"] for a in first["accounts"][:2]], ["BUY", "BUY"])
            self.assertFalse(first["recentFills"])
            bars[latest + step] = (120.0, 50.0)
            second = crypto.run(directory, latest + 2 * step + 120, load, load_dollar)
            self.assertEqual(len(second["recentFills"]), 2)
            self.assertAlmostEqual(second["recentFills"][0]["price"], 120 * (1 + crypto.SLIP))
            third = crypto.run(directory, latest + 2 * step + 180, load, load_dollar)
            self.assertEqual(second["recentFills"], third["recentFills"])
            with closing(sqlite3.connect(directory / "crypto_intraday.sqlite3")) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0], 2 * len(crypto.ACCOUNTS))

    def test_dollar_comparison_controls_for_coverage_and_ignores_unfinished_minutes(self):
        step = crypto.INTERVAL
        latest = 100 * step
        bars = {t: (100.0, 100.0 if t < latest - 7 * step else 110.0)
                for t in range(latest - 40 * step, latest + step, step)}
        close = latest + step
        dollars = {close - 16 * 60: 100.0, close - 60: 101.0, close: 90.0}
        control, _ = crypto.advance(crypto.new_account(), bars, latest, "matched", dollars)
        filtered, _ = crypto.advance(crypto.new_account(), bars, latest, "dollar", dollars)
        self.assertEqual(control[0][1]["signal"], "BUY")
        self.assertEqual(filtered[0][1]["signal"], "HOLD")
        self.assertAlmostEqual(filtered[0][1]["dollarReturn"], 0.01)
        stale, _ = crypto.advance(crypto.new_account(), bars, latest, "matched", {close - 16 * 60: 100.0})
        self.assertEqual(stale[0][1]["signal"], "HOLD")
        self.assertIsNone(stale[0][1]["dollarReturn"])

    def test_missing_candle_waits_then_discards_the_order_across_the_gap(self):
        step = crypto.INTERVAL
        newest = 100 * step
        state = {"last": newest - 30 * step, "cash": crypto.INITIAL, "qty": 0.0, "pending": "BUY"}
        bars = {t: (100.0, 100.0) for t in range(newest - 24 * step, newest, step)}
        decisions, warning = crypto.advance(state, bars, newest)
        self.assertFalse(decisions)
        self.assertTrue(warning)
        bars[newest] = (100.0, 100.0)
        decisions, warning = crypto.advance(state, bars, newest)
        self.assertIsNone(warning)
        self.assertEqual(decisions[0][1]["signal"], "HOLD")
        self.assertIsNone(decisions[0][1]["fill"])


class StockWatchTest(unittest.TestCase):
    def test_skips_holiday_and_respects_early_close(self):
        eastern = ZoneInfo("America/New_York")
        for date in (datetime(2026, 11, 26, 11, tzinfo=eastern), datetime(2026, 11, 27, 13, tzinfo=eastern)):
            with patch.object(daytrader, "_eastern_now", return_value=date), patch.object(daytrader, "fetch") as fetch:
                self.assertEqual(daytrader.watch("daytrade", "log")["status"], "closed")
                fetch.assert_not_called()

    def test_completed_fresh_bars_drive_stock_and_swing_decisions(self):
        clock = datetime(2026, 11, 25, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        targets = [{"account": "default", "ticker": "NVDA", "stop": 90, "shares": 1, "target_weight": 0.1}]
        bars = pd.DataFrame([("NVDA", 595, 89, 95), ("NVDA", 600, 10, 50), ("SPY", 580, 96, 100)],
                            columns=["symbol", "minute", "low", "close"])
        with (patch.object(daytrader, "_eastern_now", return_value=clock),
              patch.object(daytrader, "swing_targets", return_value=targets),
              patch.object(daytrader, "fetch"),
              patch.object(daytrader, "connect_cache"),
              patch.object(daytrader.it, "load_bars", return_value=bars),
              patch.object(daytrader, "run", return_value={"tradesThisRun": []}) as run):
            status = daytrader.watch("daytrade", "log")
        self.assertEqual(status["freshCount"], 1)
        self.assertEqual(run.call_args.kwargs["symbols"], ["NVDA"])
        self.assertEqual(status["swingTargets"][0]["signal"], "STOP WATCH")
        self.assertEqual(status["swingTargets"][0]["price"], 95)
