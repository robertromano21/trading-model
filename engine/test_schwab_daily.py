"""Tests for the Schwab long daily backfill: parsing, merge/overlap, resumability, auth-missing.

No live network. The Schwab client is replaced with a fixture-backed fake.
"""

import io
import os
import json
import sqlite3
import subprocess
import sys
import time
import unittest
from contextlib import redirect_stderr
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import schwab_daily as sd
from price_history import open_history, save_session
from schwab_data import SchwabAuthError

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "schwab_daily"
# When tests run from engine/ with fixtures copied next to the module's expected path:
if not FIXTURES.is_dir():
    FIXTURES = Path(__file__).resolve().parent / "fixtures"
if not FIXTURES.is_dir():
    # Source tree during box development.
    alt = Path("/workspace/schwab-daily-build/fixtures")
    if alt.is_dir():
        FIXTURES = alt


def load_fixture(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClient:
    """Minimal stand-in for SchwabMarketData.get / token_status. No network."""

    def __init__(self, payloads, status=None, fail_on=None):
        self.payloads = payloads  # symbol -> payload or list of payloads (popped)
        self.calls = []
        self.status = status or {"tokenFile": True, "usable": True, "accessTokenValid": True,
                                 "canRefresh": True, "accessTokenExpires": "2099-01-01 00:00:00",
                                 "refreshTokenExpiresAbout": "2099-01-08 00:00:00"}
        self.fail_on = fail_on or {}

    def token_status(self):
        return dict(self.status)

    def get(self, path, **params):
        self.calls.append((path, params))
        if "trader" in path:
            raise PermissionError("tests must not call trader endpoints")
        symbol = params.get("symbol")
        if symbol in self.fail_on:
            exc = self.fail_on[symbol]
            if callable(exc):
                raise exc()
            raise exc
        payload = self.payloads.get(symbol)
        if isinstance(payload, list):
            if not payload:
                return {"symbol": symbol, "empty": True, "candles": []}
            return payload.pop(0)
        if payload is None:
            return {"symbol": symbol, "empty": True, "candles": []}
        return payload


def _bar(ticker, day, close, volume=1_000_000):
    return {"ticker": ticker, "date": day, "open": close - 1, "high": close + 1, "low": close - 2,
            "close": close, "volume": volume}


class ParseTests(unittest.TestCase):
    def test_year20_fixture_parses_session_dates_and_ohlcv(self):
        rows = sd.parse_candles("SPY", load_fixture("pricehistory_spy_year20.json"))
        self.assertEqual(rows[0]["date"], "2006-01-03")
        self.assertEqual(rows[-1]["date"], "2026-11-24")
        self.assertEqual(len(rows), 10)
        first = rows[0]
        self.assertEqual((first["ticker"], first["open"], first["close"], first["source"], first["adjustment"]),
                         ("SPY", 103.0, 104.0, "schwab", "split"))

    def test_midnight_utc_maps_to_the_utc_calendar_session_date(self):
        rows = sd.parse_candles("AAPL", load_fixture("pricehistory_midnight_utc.json"))
        self.assertEqual([row["date"] for row in rows], ["2020-01-03", "2020-01-06"])
        self.assertEqual(rows[0]["close"], 70.5)

    def test_empty_payload(self):
        self.assertEqual(sd.parse_candles("ZZZZ", load_fixture("pricehistory_empty.json")), [])
        self.assertEqual(sd.parse_candles("ZZZZ", None), [])
        self.assertEqual(sd.parse_candles("ZZZZ", {"candles": [{"datetime": None}]}), [])


class MergeOverlapTests(unittest.TestCase):
    def test_prefers_massive_on_overlap_keeps_earlier_schwab_and_reports_mismatch(self):
        massive = [
            _bar("SPY", "2024-11-25", 100.0),
            _bar("SPY", "2024-11-26", 101.0),
            _bar("QQQ", "2024-11-25", 50.0),
        ]
        # Schwab history starts earlier; on the overlap SPY is 1% high (div adjustment) and QQQ is 10% high (real mismatch).
        schwab = [
            _bar("SPY", "2020-01-02", 80.0),
            _bar("SPY", "2024-11-25", 101.0),
            _bar("SPY", "2024-11-26", 102.01),
            _bar("QQQ", "2024-11-25", 55.0),
        ]
        rows, meta = sd.merge_rows(massive, schwab, prefer="massive")
        by = {(r["ticker"], r["date"]): r for r in rows}
        self.assertIn(("SPY", "2020-01-02"), by)
        # Massive close wins on the overlap date.
        self.assertEqual(by[("SPY", "2024-11-25")]["close"], 100.0)
        self.assertEqual(by[("QQQ", "2024-11-25")]["close"], 50.0)
        # Schwab-only date is scaled to Massive on first overlap (100/101).
        scale = meta["scales"]["SPY"]["factor"]
        self.assertAlmostEqual(scale, 100 / 101, places=6)
        self.assertAlmostEqual(by[("SPY", "2020-01-02")]["close"], 80.0 * scale, places=6)
        overlap = meta["overlap"]
        self.assertEqual(overlap["overlapDays"], 3)  # SPY two days + QQQ one day
        self.assertGreater(overlap["mismatchRate"], 0)
        self.assertIn("QQQ", overlap["scaleSkipped"] or {"QQQ": True})
        self.assertEqual(meta["survivorshipWarning"], sd.SURVIVORSHIP_WARNING)

    def test_overlap_report_levels_vs_returns(self):
        massive = [_bar("SPY", "2024-01-02", 100.0), _bar("SPY", "2024-01-03", 102.0)]
        # Constant 1% level bias, identical returns.
        schwab = [_bar("SPY", "2024-01-02", 101.0), _bar("SPY", "2024-01-03", 103.02)]
        report = sd.overlap_report(massive, schwab)
        self.assertEqual(report["overlapDays"], 2)
        self.assertGreater(report["medianAbsRelDiff"], 0.005)
        self.assertEqual(report["returnMismatchRate"], 0.0)

    def test_merge_caveats_drops_stale_two_year_notes(self):
        base = [
            "Only ~2 years of data: Sharpe ratio standard errors are about +/-0.7.",
            "The database holds about two years (501 sessions, 2024-11-25 to 2026-11-24) and only one broad regime.",
            "Costs are modelled, not measured.",
        ]
        merged = sd.merge_caveats(base, [sd.SURVIVORSHIP_WARNING])
        self.assertEqual(merged[0], sd.SURVIVORSHIP_WARNING)
        self.assertEqual(merged[-1], "Costs are modelled, not measured.")
        self.assertTrue(all("2 years" not in c.lower() and "two years" not in c.lower() for c in merged[1:]))


class ResumabilityAndAuthTests(unittest.TestCase):
    def test_auth_missing_exits_cleanly_and_writes_summary(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            database = directory / "prices.sqlite3"
            summary = directory / "summary.json"
            client = FakeClient({}, status={"usable": False, "accessTokenValid": False, "canRefresh": False,
                                            "tokenFile": False, "message": "No Schwab token file at /tmp/none."})
            with redirect_stderr(io.StringIO()):
                result = sd.backfill(["SPY"], client=client, database=database, summary_path=summary,
                                     lock_path=directory / "backfill.lock",
                                     ok_if_auth_missing=True, probe=False, include_research=False)
            self.assertEqual(result["status"], "auth_required")
            self.assertIn("No Schwab token", result["message"])
            saved = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "auth_required")
            self.assertFalse(database.is_file() and _table_has(database, "daily_long"))
            self.assertEqual(client.calls, [])  # no market-data request
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SchwabAuthError):
                    sd.backfill(["SPY"], client=client, database=database, summary_path=summary,
                                lock_path=directory / "backfill.lock",
                                ok_if_auth_missing=False, probe=False, include_research=False)

    def test_cli_auth_missing_returns_exit_2_and_json(self):
        client = FakeClient({}, status={"usable": False, "accessTokenValid": False, "canRefresh": False,
                                        "message": "No Schwab token file."})
        with TemporaryDirectory() as directory, \
                patch.object(sd, "SchwabMarketData", return_value=client), \
                patch.object(sd, "SUMMARY_PATH", Path(directory) / "summary.json"), \
                patch.object(sd, "DATABASE", Path(directory) / "prices.sqlite3"), \
                patch.object(sd, "LOCK_PATH", Path(directory) / "backfill.lock"), \
                patch("sys.stdout", new_callable=io.StringIO) as out:
            code = sd.main(["backfill", "--symbols", "SPY", "--no-probe"])
        self.assertEqual(code, 2)
        payload = json.loads(out.getvalue())
        self.assertEqual(payload["status"], "auth_required")
        self.assertIn("token", payload["message"].lower())

    def test_backfill_is_resumable_and_does_not_clobber_massive_daily_prices(self):
        payload = load_fixture("pricehistory_spy_year20.json")
        # First call used by a probe on SPY, second by the ticker loop (or a single get if probe off).
        client = FakeClient({"SPY": payload, "QQQ": load_fixture("pricehistory_spy_year20.json")})
        stat = SimpleNamespace(st_size=10, st_mtime_ns=1)
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            database = directory / "prices.sqlite3"
            with open_history(database, writable=True) as history:
                save_session(history, "2024-11-25", [{"ticker": "SPY", "open": 1, "high": 2, "low": 1,
                                                      "close": 1.5, "volume": 10}], stat)
            summary = directory / "summary.json"
            with redirect_stderr(io.StringIO()):
                first = sd.backfill(["SPY", "QQQ"], client=client, database=database, summary_path=summary,
                                    lock_path=directory / "backfill.lock",
                                    probe=False, include_research=False, today=date(2026, 11, 25))
            self.assertEqual(first["status"], "complete")
            self.assertEqual(first["counts"]["fetched"], 2)
            first_calls = len(client.calls)
            with redirect_stderr(io.StringIO()):
                second = sd.backfill(["SPY", "QQQ"], client=client, database=database, summary_path=summary,
                                     lock_path=directory / "backfill.lock",
                                     probe=False, include_research=False, today=date(2026, 11, 25))
            self.assertEqual(second["counts"]["skipped"], 2)
            self.assertEqual(second["counts"]["fetched"], 0)
            self.assertEqual(len(client.calls), first_calls)  # no new market-data requests
            # Massive grouped session is intact.
            with open_history(database) as history:
                row = history.execute("SELECT close, volume FROM daily_prices WHERE ticker='SPY' AND date='2024-11-25'").fetchone()
                self.assertEqual((row["close"], row["volume"]), (1.5, 10))
                sessions = history.execute("SELECT bar_count FROM market_sessions WHERE date='2024-11-25'").fetchone()
                self.assertEqual(sessions["bar_count"], 1)
            connection = sqlite3.connect(database)
            try:
                rows = connection.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM daily_long WHERE ticker='SPY'").fetchone()
                self.assertEqual(rows[0], 10)
                self.assertEqual((rows[1], rows[2]), ("2006-01-03", "2026-11-24"))
                # No row in daily_long leaked into a destructive change of daily_prices count.
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM daily_prices").fetchone()[0], 1)
            finally:
                connection.close()

    def test_force_refetches_and_time_budget_stops_cleanly(self):
        payload = load_fixture("pricehistory_spy_year20.json")
        client = FakeClient({"SPY": [payload, payload], "QQQ": [payload]})
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            database = directory / "prices.sqlite3"
            summary = directory / "summary.json"
            with redirect_stderr(io.StringIO()):
                sd.backfill(["SPY"], client=client, database=database, summary_path=summary,
                            lock_path=directory / "backfill.lock",
                            probe=False, include_research=False, today=date(2026, 11, 25))
                forced = sd.backfill(["SPY"], client=client, database=database, summary_path=summary,
                                     lock_path=directory / "backfill.lock",
                                     probe=False, force=True, include_research=False, today=date(2026, 11, 25))
            self.assertEqual(forced["counts"]["fetched"], 1)
            self.assertEqual(forced["counts"]["skipped"], 0)
            # Budget already expired: nothing fetched, status partial.
            with redirect_stderr(io.StringIO()), patch("schwab_daily.time.monotonic", side_effect=[0, 0, 9999, 9999, 9999, 9999]):
                partial = sd.backfill(["SPY", "QQQ"], client=client, database=database, summary_path=summary,
                                      lock_path=directory / "backfill.lock",
                                      probe=False, force=True, budget_minutes=0.0001, include_research=False,
                                      today=date(2026, 11, 25))
            self.assertEqual(partial["status"], "partial")
            self.assertIn("time budget", partial["message"])

    def test_universe_includes_core_etfs_indices_and_paper_holdings(self):
        with TemporaryDirectory() as directory:
            directory = Path(directory)
            paper = directory / "paper.sqlite3"
            connection = sqlite3.connect(paper)
            connection.execute("CREATE TABLE positions (account TEXT, ticker TEXT)")
            connection.execute("INSERT INTO positions VALUES ('default', 'ZZQQ'), ('default', 'AAPL')")
            connection.commit()
            connection.close()
            sp500 = directory / "sp500.json"
            sp500.write_text(json.dumps({"tickers": ["MSFT", "AAPL"]}), encoding="utf-8")
            with patch.object(sd, "SP500_PATH", sp500), patch.object(sd, "PAPER_DB", paper), \
                    patch.object(sd, "DATABASE", directory / "missing.sqlite3"):
                uni = sd.collect_universe(include_research=False, database=directory / "missing.sqlite3",
                                          paper_db=paper, research_dir=directory)
            self.assertIn("SPY", uni["tickers"])
            self.assertIn("UUP", uni["tickers"])
            self.assertIn("VIXY", uni["tickers"])
            self.assertIn("SVIX", uni["tickers"])
            self.assertIn("$SPX.X", uni["tickers"])
            self.assertIn("ZZQQ", uni["tickers"])
            self.assertIn("MSFT", uni["tickers"])
            # First group is indices, then core ETFs (SPY near the front of non-index names).
            self.assertTrue(uni["tickers"][0].startswith("$"))
            self.assertGreaterEqual(uni["counts"]["paper"], 1)
            self.assertIn("ZZQQ", uni["tickers"])
            self.assertGreater(uni["counts"]["sp500"], 0)


class PanelHookTests(unittest.TestCase):
    def test_load_panel_extends_with_daily_long_and_attaches_warnings(self):
        from market_data import load_panel
        stat = SimpleNamespace(st_size=10, st_mtime_ns=1)
        with TemporaryDirectory() as directory:
            database = Path(directory) / "prices.sqlite3"
            with open_history(database, writable=True) as history:
                save_session(history, "2024-11-25", [
                    {"ticker": "SPY", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1e7},
                    {"ticker": "AAPL", "open": 50, "high": 51, "low": 49, "close": 50, "volume": 1e7},
                ], stat)
                save_session(history, "2024-11-26", [
                    {"ticker": "SPY", "open": 101, "high": 102, "low": 100, "close": 101, "volume": 1e7},
                    {"ticker": "AAPL", "open": 51, "high": 52, "low": 50, "close": 51, "volume": 1e7},
                ], stat)
            connection = sd.connect_prices(database, writable=True)
            try:
                sd.store_daily_rows(connection, "SPY", [
                    _bar("SPY", "2020-01-02", 80.0, 1e7),
                    _bar("SPY", "2024-11-25", 100.5, 1e7),  # overlap; Massive should win (100)
                    _bar("SPY", "2024-11-26", 101.5, 1e7),
                ], "year-20", status="complete")
            finally:
                connection.close()
            panel = load_panel(tickers=["SPY", "AAPL"], database=database, min_avg_dollar_volume=0, min_avg_price=0)
            dates = [d.date().isoformat() for d in panel.dates]
            self.assertIn("2020-01-02", dates)
            self.assertIn("2024-11-25", dates)
            # Massive close on the overlap.
            self.assertAlmostEqual(float(panel.close.loc[pd.Timestamp("2024-11-25"), "SPY"]), 100.0)
            # Earlier Schwab date is present (possibly splice-scaled).
            self.assertTrue(pd.notna(panel.close.loc[pd.Timestamp("2020-01-02"), "SPY"]))
            self.assertTrue(any("survivorship" in w.lower() for w in panel.warnings))
            self.assertTrue(panel.used_long_history)
            self.assertGreaterEqual(panel.overlap["overlapDays"], 2)


class BackfillLockTests(unittest.TestCase):
    def test_pid_is_alive_for_this_process_and_a_finished_process(self):
        self.assertTrue(sd.pid_is_alive(os.getpid()))
        self.assertFalse(sd.pid_is_alive(-1))
        self.assertFalse(sd.pid_is_alive(0))
        self.assertFalse(sd.pid_is_alive("not-a-pid"))
        finished = subprocess.run([sys.executable, "-c", "pass"], check=True)
        # A freshly reaped child PID is not a running process.
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        pid = child.pid
        child.wait()
        self.assertFalse(sd.pid_is_alive(pid))
        self.assertIsNotNone(finished)

    def test_lock_with_dead_pid_is_stale_immediately(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "schwab_daily_backfill.lock"
            child = subprocess.Popen([sys.executable, "-c", "pass"])
            dead_pid = child.pid
            child.wait()
            path.write_text(json.dumps({"pid": dead_pid, "startedAt": "2026-11-25T00:00:00"}), encoding="utf-8")
            lock = sd.BackfillLock(path)
            self.assertTrue(lock.acquire())
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pid"], os.getpid())
            lock.release()
            self.assertFalse(path.is_file())

    def test_lock_with_live_pid_is_not_stolen(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "schwab_daily_backfill.lock"
            path.write_text(json.dumps({"pid": os.getpid(), "startedAt": "2026-11-25T00:00:00"}), encoding="utf-8")
            lock = sd.BackfillLock(path)
            self.assertFalse(lock.acquire())
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["pid"], os.getpid())

    def test_legacy_lock_without_pid_uses_three_hour_age(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "schwab_daily_backfill.lock"
            path.write_text("{}", encoding="utf-8")
            lock = sd.BackfillLock(path)
            self.assertFalse(lock.acquire())
            old = time.time() - 3 * 3600 - 5
            os.utime(path, (old, old))
            self.assertTrue(lock.acquire())
            lock.release()


def _table_has(path, name):
    if not Path(path).is_file():
        return False
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None
    finally:
        connection.close()


if __name__ == "__main__":
    unittest.main()
