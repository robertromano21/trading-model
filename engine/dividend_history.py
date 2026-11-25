"""Cache Massive's cash dividend records for the saved daily price window."""

import json
import logging
import math
import os
import sqlite3
import time
from collections import defaultdict
from datetime import date, timedelta
from urllib.parse import urlencode, urlsplit

from fetch_and_model import DATABASE, MassiveError, download_json
from price_history import coverage, open_history

LOGGER = logging.getLogger(__name__)
DIVIDEND_PATH = "/stocks/v1/dividends"


def dividend_coverage(connection):
    exists = connection.execute(
        "SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'backtest_dividend_coverage'"
    ).fetchone()
    if not exists:
        return None
    row = connection.execute(
        "SELECT first_date AS firstDate, last_date AS lastDate, events FROM backtest_dividend_coverage"
    ).fetchone()
    return dict(row) if row else None


def read_dividends(connection, end, ticker=None):
    operator = " AND ticker = ?" if ticker else ""
    args = (end, ticker) if ticker else (end,)
    rows = connection.execute(
        f"SELECT ticker, market_date, cash, unsupported FROM backtest_dividends WHERE market_date <= ?{operator}", args,
    )
    result = defaultdict(dict)
    for symbol, market_date, cash, unsupported in rows:
        result[symbol][market_date] = (cash, unsupported)
    return result


def fetch_dividends(key, first, last):
    query = urlencode({"ex_dividend_date.gte": first, "ex_dividend_date.lte": last,
                       "sort": "ex_dividend_date.asc", "limit": 5000})
    url = f"https://api.massive.com{DIVIDEND_PATH}?{query}"
    seen_urls, seen_records = set(), set()
    observed_dates = set()
    amounts = defaultdict(lambda: [0.0, 0])
    while url:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc != "api.massive.com" or parts.path != DIVIDEND_PATH or url in seen_urls:
            raise ValueError("Massive returned an invalid dividend pagination link.")
        for attempt in range(3):
            try:
                payload = json.loads(download_json(url, key))
                break
            except MassiveError as error:
                retryable = error.status in (None, 429, 500, 502, 503, 504)
                wait = max(error.retry_after or 0, 60 if error.status == 429 else 5)
                if not retryable or attempt == 2 or wait > 120:
                    raise
                LOGGER.warning("Retrying dividend page in %s seconds.", wait)
                time.sleep(wait)
        seen_urls.add(url)
        if payload.get("status") != "OK" or not isinstance(payload.get("results"), list):
            raise ValueError("Massive did not return a complete dividend page.")
        for record in payload["results"]:
            ticker, day = record.get("ticker"), record.get("ex_dividend_date")
            if not ticker or not isinstance(day, str) or not first <= day <= last:
                raise ValueError("Massive returned a dividend outside the requested ticker/date format.")
            key_id = record.get("id") or json.dumps(record, sort_keys=True)
            if key_id in seen_records:
                continue
            seen_records.add(key_id)
            observed_dates.add(day)
            value = record.get("split_adjusted_cash_amount")
            if record.get("currency") == "USD" and type(value) in (int, float) and math.isfinite(value) and value >= 0:
                amounts[ticker, day][0] += value
            else:
                amounts[ticker, day][1] += 1
        url = payload.get("next_url")
        LOGGER.info("Dividend pages read: %s; unique records: %s", len(seen_urls), len(seen_records))
        if url:
            time.sleep(13)

    start, finish = date.fromisoformat(first), date.fromisoformat(last)
    # A US-wide feed has payouts throughout multi-week windows. A long empty edge
    # can mean the API silently shortened the request to the key's entitlement.
    if (finish - start).days > 21 and (
        not observed_dates or date.fromisoformat(min(observed_dates)) > start + timedelta(days=10)
        or date.fromisoformat(max(observed_dates)) < finish - timedelta(days=10)
    ):
        raise ValueError("Massive dividend results do not span the saved price window. Check historical access.")
    return amounts, len(seen_records)


def sync_dividends():
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if not key:
        raise ValueError("Set MASSIVE_API_KEY before syncing dividend history.")
    if not DATABASE.is_file():
        raise ValueError("Import historical prices before syncing dividends.")
    with open_history(DATABASE) as connection:
        window = coverage(connection)
        first, last = window["firstDate"], window["lastDate"]
    if not first:
        raise ValueError("Import historical prices before syncing dividends.")

    amounts, events = fetch_dividends(key, first, last)
    with open_history(DATABASE, writable=True) as connection, connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS backtest_dividends (
                 ticker TEXT, market_date TEXT, cash REAL NOT NULL, unsupported INTEGER NOT NULL,
                 PRIMARY KEY (ticker, market_date)
               ) WITHOUT ROWID"""
        )
        connection.execute(
            """CREATE TABLE IF NOT EXISTS backtest_dividend_coverage (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 first_date TEXT NOT NULL, last_date TEXT NOT NULL, events INTEGER NOT NULL
               )"""
        )
        connection.execute("DELETE FROM backtest_dividends")
        connection.executemany(
            "INSERT INTO backtest_dividends VALUES (?, ?, ?, ?)",
            ((ticker, day, value, unsupported) for (ticker, day), (value, unsupported) in amounts.items()),
        )
        connection.execute("INSERT OR REPLACE INTO backtest_dividend_coverage VALUES (1, ?, ?, ?)", (first, last, events))
    return {"status": "complete", "firstDate": first, "lastDate": last, "events": events,
            "unsupported": sum(value[1] for value in amounts.values())}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(sync_dividends(), allow_nan=False))
    except (MassiveError, OSError, ValueError, sqlite3.Error) as error:
        LOGGER.error("%s", error)
        print(json.dumps({"status": "failed", "message": str(error)}))
