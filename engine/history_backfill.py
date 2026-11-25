"""Backfill or sync grouped U.S. stock sessions into SQLite."""

import json
import logging
import os
import sqlite3
import sys
import time
from datetime import date, timedelta

import exchange_calendars

from fetch_and_model import (
    DATABASE, HISTORY_JSON, MassiveError, available_dates, dated_source, download, market_bars,
    read_local, read_results, write_atomically,
)
from price_history import open_history, save_session, session_manifest

LOGGER = logging.getLogger(__name__)


def years_ago(day, years=3):
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        if day.month == 2 and day.day == 29:
            return day.replace(year=day.year - years, day=28)
        raise


def startup_sessions(end, saved, known_files, today=None):
    floor = years_ago(end + timedelta(days=1), years=2)
    known = [date.fromisoformat(day) for day, (count, *_) in saved.items() if count]
    known.extend(known_files)
    first = min((day for day in known if day <= end), default=None)
    start = max(first or end, floor)
    # Include the latest trading session even when the cutoff falls on a holiday or weekend.
    calendar = exchange_calendars.get_calendar("XNYS", start=min(start, end - timedelta(days=7)), end=today or end)
    sessions = [session.date() for session in calendar.sessions]
    last_complete = max(day for day in sessions if day <= end)
    return [day for day in reversed(sessions) if day >= min(start, last_complete)]


class MassiveDownloader:
    def __init__(self, api_key, *, include_otc=False):
        self.api_key = api_key
        self.include_otc = include_otc
        self.next_request = 0
        self.interval = 1

    def get(self, market_date):
        for attempt in range(3):
            time.sleep(max(0, self.next_request - time.monotonic()))
            self.next_request = time.monotonic() + self.interval
            try:
                return download(market_date, self.api_key, include_otc=self.include_otc)
            except MassiveError as error:
                retryable = error.status in (None, 408, 429) or (error.status is not None and 500 <= error.status < 600)
                if not retryable or attempt == 2:
                    raise
                if error.status == 429:
                    self.interval = 13
                    wait = max(60, error.retry_after or 0)
                else:
                    wait = max(5 * (2 ** attempt), error.retry_after or 0)
                if wait > 120:
                    raise
                LOGGER.warning("%s: %s Retrying in %s seconds.", market_date, error, wait)
                self.next_request = time.monotonic() + wait


def ingest_day(history, market_date, known_files, previous, downloader, *, require_prices=False):
    day = market_date.isoformat()
    if require_prices and previous and previous[0]:
        return previous[0], "existing"
    path = dated_source(market_date) if market_date in known_files else None
    records = None
    if path is None and market_date in known_files:
        path, records = read_local(market_date)
    if path is not None:
        stat = path.stat()
        if not require_prices and previous is not None and previous[1:] == (stat.st_size, stat.st_mtime_ns):
            return previous[0], "existing"
        if records is None:
            records = read_results(path.read_bytes())
        bars = market_bars(records, market_date)
        if bars or not require_prices:
            save_session(history, day, bars, stat)
            return len(bars), "indexed"
    if not require_prices and previous is not None:
        return previous[0], "existing"
    if downloader is None:
        return 0, "missing"

    raw = downloader.get(market_date)
    bars = market_bars(read_results(raw, require_adjusted=True), market_date)
    if require_prices and not bars:
        return 0, "missing"
    path = HISTORY_JSON / f"prices_{day}.json"
    write_atomically(path, raw)
    save_session(history, day, bars, path.stat())
    return len(bars), "downloaded"


def backfill(options):
    end = date.fromisoformat(options["date"])
    startup = options.get("startup", False)
    today = date.fromisoformat(options["today"]) if startup and options.get("today") else None
    mode = options["mode"]
    if mode not in ("local", "massive"):
        raise ValueError("Choose existing JSON or a Massive backfill.")
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if mode == "massive" and not key:
        raise ValueError("Set MASSIVE_API_KEY in PowerShell and restart the dashboard to download history.")
    downloader = MassiveDownloader(key, include_otc=startup) if mode == "massive" else None
    known_files = set(available_dates(include_archive=True))
    missing_key = startup and not key
    key_message = "Set MASSIVE_API_KEY before launching to download missing sessions." if missing_key else None
    if missing_key and not DATABASE.is_file() and not any(day <= end for day in known_files):
        return {"status": "failed", "message": key_message}
    counts = dict.fromkeys(("existing", "indexed", "downloaded", "missing", "empty", "sessions", "errors", "bars"), 0)
    fully_blocked = None
    issues = []
    first_saved = last_saved = None
    with open_history(DATABASE, writable=True) as history:
        saved = session_manifest(history)
        if startup:
            dates = startup_sessions(end, saved, known_files, today)
            start = dates[-1]
        else:
            start = years_ago(end)
            days = (start + timedelta(days=offset) for offset in range((end - start).days + 1))
            dates = [day for day in days if day.weekday() < 5]
        for index, market_date in enumerate(dates, start=1):
            day = market_date.isoformat()
            try:
                sources = known_files - {today} if market_date == today else known_files
                bar_count, result = ingest_day(history, market_date, sources, saved.get(day), downloader,
                                               require_prices=startup)
            except (OSError, ValueError, OverflowError, MassiveError) as error:
                if market_date == today and isinstance(error, MassiveError) and error.status in (403, 404):
                    LOGGER.info("%s: Massive has not made this session available yet.", day)
                    continue
                counts["errors"] += 1
                if len(issues) < 6:
                    issues.append(f"{day}: {error}")
                LOGGER.error("%s: %s", day, error)
                if isinstance(error, MassiveError) and error.status not in (400, 404):
                    fully_blocked = str(error)
                    downloader = None
                elif isinstance(error, OSError):
                    raise
            else:
                if market_date == today and result == "missing":
                    continue
                counts[result] += 1
                if result != "missing":
                    if bar_count:
                        counts["sessions"] += 1
                        first_saved = min(first_saved or day, day)
                        last_saved = max(last_saved or day, day)
                    else:
                        counts["empty"] += 1
                    counts["bars"] += bar_count
                    if result in ("indexed", "downloaded"):
                        LOGGER.info("%s: %s %s stock prices in SQLite", day, result, bar_count)

            if index == 1 or index % 10 == 0 or index == len(dates):
                print("PROGRESS " + json.dumps({
                    "checked": index, "total": len(dates), "date": day, "sessions": counts["sessions"],
                    "downloaded": counts["downloaded"], "bars": counts["bars"],
                }), file=sys.stderr, flush=True)

    incomplete = counts["missing"] or counts["errors"]
    return {
        "status": ("partial" if counts["bars"] else "failed") if incomplete else "complete",
        "startDate": start.isoformat(), "endDate": max(end.isoformat(), last_saved or ""), "weekdays": len(dates),
        "firstSaved": first_saved, "lastSaved": last_saved,
        "message": fully_blocked or key_message, "issues": issues, **counts,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(backfill(json.load(sys.stdin)), separators=(",", ":")))
    except (KeyError, TypeError, ValueError, OSError, sqlite3.Error) as error:
        LOGGER.error("%s", error)
        print(json.dumps({"status": "failed", "message": str(error)}))
