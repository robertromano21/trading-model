"""Data-health report for the UI (read-only): session dates, calendar gaps, split/anomaly repairs, excluded
tickers, dividend coverage, the Schwab intraday cache and Schwab token state (no secrets).

  python data_health.py [--check-files 5]
"""

import argparse
import json
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path

from fetch_and_model import DATABASE, DATA_DIR, HISTORY_JSON, check_session_date, read_results

INTRADAY_DB = DATA_DIR / "intraday.sqlite3"
RESEARCH = DATA_DIR / "Research"
EXCLUDING_KINDS = ("jump_reversal", "unadjusted_split", "split_basis_mismatch")


def _ro(path):
    if not Path(path).is_file():
        return None
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _has(connection, name):
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def sessions_report(connection):
    rows = [r[0] for r in connection.execute("SELECT date FROM market_sessions WHERE bar_count > 0 ORDER BY date")]
    out = {"completeSessions": len(rows), "first": rows[0] if rows else None, "last": rows[-1] if rows else None,
           "missingVsCalendar": [], "expectedLatest": None}
    if not rows:
        return out
    try:
        from risk import _nyse_sessions
        today = date.today().isoformat()
        cal = [d.date().isoformat() for d in _nyse_sessions(rows[0], today)]
        now = datetime.now()
        # a session counts as expected once its daily file is normally available (after 20:00 ET that day)
        expected = [d for d in cal if d < today or (d == today and now.hour >= 20)]
        have = set(rows)
        out["missingVsCalendar"] = [d for d in expected if d not in have]
        out["expectedLatest"] = expected[-1] if expected else None
        out["upToDate"] = out["last"] == out["expectedLatest"]
    except Exception as error:   # calendar package missing: report without gap check
        out["calendarError"] = str(error)
    small = [dict(r) for r in connection.execute(
        "SELECT date, bar_count FROM market_sessions WHERE bar_count > 0 ORDER BY bar_count LIMIT 3")]
    out["smallestSessions"] = small
    return out


def file_date_checks(limit):
    """Re-run the timestamp check (bars stamped with the file's session date) on the newest daily JSON files."""
    files = []
    for folder in (DATA_DIR, DATA_DIR / "Processed", HISTORY_JSON):
        files += [p for p in folder.glob("prices_????-??-??.json")] if folder.is_dir() else []
    files = sorted(files, key=lambda p: p.name, reverse=True)[:limit]
    results = []
    for path in files:
        market_date = path.stem.removeprefix("prices_")
        try:
            check_session_date(read_results(path.read_bytes()), market_date)
            results.append({"file": f"{path.parent.name}/{path.name}", "date": market_date, "ok": True})
        except Exception as error:
            results.append({"file": f"{path.parent.name}/{path.name}", "date": market_date, "ok": False, "error": str(error)})
    aside = sorted(p.name for p in (DATA_DIR / "Processed").glob("*.misdated-*")) if (DATA_DIR / "Processed").is_dir() else []
    return {"checked": results, "setAside": aside}


def anomalies_report(connection):
    if not _has(connection, "price_anomalies"):
        return {"available": False}
    by = [dict(r) for r in connection.execute(
        "SELECT kind, status, COUNT(*) AS n, COUNT(DISTINCT ticker) AS tickers FROM price_anomalies GROUP BY 1, 2 ORDER BY 1, 2")]
    marks = ",".join("?" * len(EXCLUDING_KINDS))
    excluded = [dict(r) for r in connection.execute(
        f"""SELECT a.ticker, COUNT(*) AS anomalies, MAX(a.date) AS last_date, GROUP_CONCAT(DISTINCT a.kind) AS kinds,
                   (SELECT AVG(p.close * p.volume) FROM daily_prices p WHERE p.ticker = a.ticker
                     AND p.date >= date((SELECT MAX(date) FROM daily_prices), '-60 day')) AS adv
            FROM price_anomalies a WHERE a.status IN ('open','persists') AND a.kind IN ({marks})
            GROUP BY a.ticker ORDER BY adv DESC""", EXCLUDING_KINDS)]
    masked = [dict(r) for r in connection.execute(
        "SELECT ticker, date, kind, detail FROM price_anomalies WHERE status='masked' ORDER BY date DESC")]
    repairs = {}
    if _has(connection, "ticker_repairs"):
        repairs = dict(connection.execute(
            "SELECT COUNT(DISTINCT ticker) AS tickers, MAX(repaired_at) AS last FROM ticker_repairs").fetchone())
        repairs["largest"] = [dict(r) for r in connection.execute(
            "SELECT ticker, MAX(max_change) AS max_change, MAX(repaired_at) AS repaired_at FROM ticker_repairs "
            "GROUP BY ticker ORDER BY 2 DESC LIMIT 15")]
    splits = {}
    if _has(connection, "corporate_splits"):
        splits = dict(connection.execute(
            "SELECT COUNT(*) AS events, COUNT(DISTINCT ticker) AS tickers, MAX(execution_date) AS latest "
            "FROM corporate_splits").fetchone())
    return {"available": True, "byKindStatus": by, "excludedTickers": excluded, "excludedCount": len(excluded),
            "masked": masked, "repairs": repairs, "splits": splits}


def dividends_report(connection):
    if not _has(connection, "backtest_dividend_coverage"):
        return {"available": False}
    row = connection.execute("SELECT first_date, last_date, events FROM backtest_dividend_coverage WHERE id=1").fetchone()
    out = {"available": row is not None, **(dict(row) if row else {})}
    if _has(connection, "backtest_dividends"):
        out["rows"], out["latestExDate"] = connection.execute(
            "SELECT COUNT(*), MAX(market_date) FROM backtest_dividends").fetchone()
    return out


def intraday_report():
    cache = _ro(INTRADAY_DB)
    if cache is None or not _has(cache, "intraday_bars"):
        return {"available": False}
    with cache:
        freq = [dict(r) for r in cache.execute(
            "SELECT frequency, COUNT(*) AS bars, COUNT(DISTINCT symbol) AS symbols, COUNT(DISTINCT session_date) AS sessions, "
            "MIN(session_date) AS first, MAX(session_date) AS last FROM intraday_bars GROUP BY frequency")]
        fetch = cache.execute("SELECT MAX(fetched_at) FROM intraday_fetches").fetchone()[0] if _has(cache, "intraday_fetches") else None
        quotes = cache.execute("SELECT COUNT(*), MAX(fetched_at) FROM intraday_quotes").fetchone() if _has(cache, "intraday_quotes") else (0, None)
    return {"available": True, "byFrequency": freq, "lastFetch": fetch, "quoteRows": quotes[0], "lastQuote": quotes[1],
            "path": "Massive/intraday.sqlite3"}


def schwab_report():
    try:
        from schwab_data import SchwabMarketData
        return SchwabMarketData().token_status()
    except Exception as error:
        return {"usable": False, "message": str(error)}


def outputs_report():
    names = ["research_summary.json", "research_report.md", "attribution.json", "daytrade_summary.json", "paper_status.json",
             "paper_status_dryrun.json", "daytrade_status.json", "advisory_verdicts.json", "paper_daily.log"]
    out = {}
    for name in names:
        path = RESEARCH / name
        out[name] = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if path.is_file() else None
    return out


def build(check_files=5):
    started = time.time()
    payload = {"generatedAt": datetime.now().isoformat(timespec="seconds")}
    prices = _ro(DATABASE)
    if prices is None:
        payload["error"] = "Massive/prices.sqlite3 not found"
    else:
        with prices:
            payload["sessions"] = sessions_report(prices)
            payload["anomalies"] = anomalies_report(prices)
            payload["dividends"] = dividends_report(prices)
            payload["tickersLatestSession"] = prices.execute(
                "SELECT COUNT(*) FROM daily_prices WHERE date = (SELECT MAX(date) FROM daily_prices)").fetchone()[0]
    payload["files"] = file_date_checks(check_files)
    payload["intraday"] = intraday_report()
    payload["schwab"] = schwab_report()
    payload["outputs"] = outputs_report()
    payload["elapsedSeconds"] = round(time.time() - started, 2)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-files", type=int, default=5)
    args = parser.parse_args(argv)
    json.dump(build(max(0, min(args.check_files, 600))), sys.stdout, default=str)


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("data_health"):
        main()
