import json
import sqlite3
from contextlib import redirect_stderr
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import history_backfill
from price_history import coverage, day_bars, model_history, open_history, save_session, save_ticker_bars, ticker_history


class HistoryTests(TestCase):
    stat = SimpleNamespace(st_size=100, st_mtime_ns=10)
    bar = {"ticker": "AAPL", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100}

    def test_lookup_limits_to_50_trading_sessions_and_honors_as_of(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "prices.sqlite3"
            start = date(2025, 1, 1)
            with open_history(database, writable=True) as history:
                for offset in range(60):
                    day = (start + timedelta(days=offset)).isoformat()
                    save_session(history, day, [self.bar], self.stat)
            with open_history(database) as history:
                rows = ticker_history(history, "AAPL", "2025-02-25")
                self.assertEqual(len(rows), 50)
                self.assertEqual((rows[0]["date"], rows[-1]["date"]), ("2025-02-25", "2025-01-07"))
                self.assertEqual(ticker_history(history, "MSFT", "2025-02-25"), [])
                self.assertEqual(next(model_history(history, "2025-02-25"))[0], "2025-01-01")

    def test_failed_replacement_leaves_previous_session_readable(self):
        with TemporaryDirectory() as directory, open_history(Path(directory) / "prices.sqlite3", writable=True) as history:
            save_session(history, "2025-01-01", [self.bar], self.stat)
            with self.assertRaises(sqlite3.IntegrityError):
                save_session(history, "2025-01-01", [self.bar, self.bar], self.stat)
            self.assertEqual(len(day_bars(history, "2025-01-01")), 1)
            self.assertEqual(coverage(history)["bars"], 1)

    def test_ticker_import_preserves_grouped_data_and_can_be_replaced(self):
        with TemporaryDirectory() as directory, open_history(Path(directory) / "prices.sqlite3", writable=True) as history:
            save_session(history, "2025-01-02", [self.bar], self.stat)
            old_bar = {**self.bar, "date": "2025-01-01"}
            conflict = {**self.bar, "date": "2025-01-02", "close": 99}
            self.assertEqual(save_ticker_bars(history, [old_bar, conflict]), 1)
            self.assertEqual(save_ticker_bars(history, [old_bar, conflict]), 0)
            self.assertEqual(coverage(history)["sessions"], 1)
            save_session(history, "2025-01-01", [{**self.bar, "close": 13}], self.stat)
            rows = ticker_history(history, "AAPL", "2025-01-02")
            self.assertEqual([row["close"] for row in rows], [11, 13])

    def test_backfill_saves_raw_data_and_resumes_empty_weekdays_too(self):
        last = date(2025, 1, 3)
        responses = [
            b'{"status":"OK","resultsCount":0}',
            json.dumps({"status": "OK", "adjusted": True, "results": [
                {"T": "AAPL", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100},
            ]}).encode(),
        ]
        with TemporaryDirectory() as directory:
            database = Path(directory) / "prices.sqlite3"
            raw_dir = Path(directory) / "raw"
            with (
                patch.object(history_backfill, "DATABASE", database),
                patch.object(history_backfill, "HISTORY_JSON", raw_dir),
                patch.object(history_backfill, "available_dates", return_value=[]),
                patch.object(history_backfill, "years_ago", return_value=last - timedelta(days=1)),
                patch.object(history_backfill.MassiveDownloader, "get", side_effect=responses) as provider,
                patch.dict("os.environ", {"MASSIVE_API_KEY": "fixture"}),
                redirect_stderr(StringIO()),
            ):
                options = {"date": last.isoformat(), "mode": "massive"}
                first = history_backfill.backfill(options)
                second = history_backfill.backfill(options)
            self.assertEqual((first["status"], first["downloaded"], first["empty"]), ("complete", 2, 1))
            self.assertEqual((second["downloaded"], second["existing"], provider.call_count), (0, 2, 2))
            self.assertEqual((raw_dir / "prices_2025-01-03.json").read_bytes(), responses[1])
            self.assertEqual(history_backfill.years_ago(date(2024, 2, 29)), date(2021, 2, 28))

    def test_startup_sync_skips_market_closures_and_retries_empty_sessions(self):
        raw = json.dumps({"status": "OK", "adjusted": True, "results": [
            {"T": "AAPL", "o": 10, "h": 12, "l": 9, "c": 11, "v": 100},
            {"T": "OTCM", "o": 20, "h": 22, "l": 19, "c": 21, "v": 200, "otc": True},
        ]}).encode()
        with TemporaryDirectory() as directory:
            database = Path(directory) / "prices.sqlite3"
            with open_history(database, writable=True) as history:
                save_session(history, "2024-12-31", [self.bar], self.stat)
                save_session(history, "2025-01-02", [], self.stat)
            with (
                patch.object(history_backfill, "DATABASE", database),
                patch.object(history_backfill, "HISTORY_JSON", Path(directory) / "raw"),
                patch.object(history_backfill, "available_dates", return_value=[]),
                patch.object(history_backfill, "download", return_value=raw) as download,
                patch.dict("os.environ", {"MASSIVE_API_KEY": "fixture"}),
                redirect_stderr(StringIO()),
            ):
                options = {"date": "2025-01-05", "mode": "massive", "startup": True}
                first = history_backfill.backfill(options)
                second = history_backfill.backfill(options)
            self.assertEqual((first["status"], first["downloaded"], first["lastSaved"]), ("complete", 2, "2025-01-03"))
            self.assertEqual([call.args[0] for call in download.call_args_list], [date(2025, 1, 3), date(2025, 1, 2)])
            self.assertTrue(all(call.kwargs["include_otc"] for call in download.call_args_list))
            self.assertEqual((second["downloaded"], second["existing"]), (0, 3))
            with open_history(database) as history:
                self.assertEqual([row["ticker"] for row in day_bars(history, "2025-01-03")], ["AAPL", "OTCM"])

    def test_startup_sync_continues_when_basic_plan_cannot_fetch_today(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "prices.sqlite3"
            with open_history(database, writable=True) as history:
                save_session(history, "2025-01-03", [self.bar], self.stat)
            with (
                patch.object(history_backfill, "DATABASE", database),
                patch.object(history_backfill, "available_dates", return_value=[]),
                patch.object(history_backfill.MassiveDownloader, "get",
                             side_effect=history_backfill.MassiveError("Unavailable", 403)) as get,
                patch.dict("os.environ", {"MASSIVE_API_KEY": "fixture"}),
                redirect_stderr(StringIO()),
            ):
                result = history_backfill.backfill({"date": "2025-01-05", "today": "2025-01-06",
                                                   "mode": "massive", "startup": True})
            get.assert_called_once_with(date(2025, 1, 6))
            self.assertEqual((result["status"], result["lastSaved"], result["errors"]), ("complete", "2025-01-03", 0))
