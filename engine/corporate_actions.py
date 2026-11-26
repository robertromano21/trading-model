"""Splits, split-artifact detection and per-ticker history repair for the local SQLite store.

Massive grouped-daily files are split-adjusted only as of the day they were downloaded. When sessions for one
ticker were downloaded on different days around a split, the stored history mixes share bases and shows fake
jumps (for example SNDK at 8x for 2026-10-26..11-05). This module:

* syncs Massive split records (/stocks/v1/splits) into ``corporate_splits``;
* reports suspicious jumps (split-ratio jumps and jump-then-reverse anomalies) into ``price_anomalies``;
* repairs a ticker by re-downloading its full adjusted daily range and replacing the stored rows.

Spin-offs (for example CTVA on 2026-10-01) are not splits: Massive's split-adjusted prices keep the price drop on
the distribution date. The anomaly report flags them as ``unexplained_jump``; research code masks that one day's
return instead of rewriting prices.

CLI:
    python engine/corporate_actions.py --sync-splits
    python engine/corporate_actions.py --report
    python engine/corporate_actions.py --repair SNDK,QID
    python engine/corporate_actions.py --repair-flagged --max 50
    python engine/corporate_actions.py --mask CTVA:2026-10-01   # ignore one day's return (spin-off) in research
"""

import argparse
import json
import logging
import math
import time
from datetime import date, datetime, timezone
from urllib.parse import urlencode, urlsplit

from fetch_and_model import DATABASE, MASSIVE_AGGS_URL, MassiveError, download_json, market_bars, read_results
from local_env import massive_key
from price_history import coverage, open_history

LOGGER = logging.getLogger(__name__)
SPLITS_PATH = "/stocks/v1/splits"
JUMP = math.log(1.8)          # |log return| above this is a candidate artifact
RATIO_TOLERANCE = 0.03        # 3% in log space for "matches a split ratio"
REVERSAL_WINDOW = 10          # sessions for a jump to be undone
COMMON_RATIOS = [1 / n for n in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100)] + \
                [n for n in (2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100)] + [1.5, 2 / 3]

SCHEMA = """
CREATE TABLE IF NOT EXISTS corporate_splits (
    ticker TEXT NOT NULL, execution_date TEXT NOT NULL, split_from REAL NOT NULL, split_to REAL NOT NULL,
    adjustment_type TEXT, historical_adjustment_factor REAL, PRIMARY KEY (ticker, execution_date, split_from, split_to)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS price_anomalies (
    ticker TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL, ratio REAL NOT NULL, detail TEXT,
    status TEXT NOT NULL DEFAULT 'open', updated_at TEXT NOT NULL, PRIMARY KEY (ticker, date, kind)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS ticker_repairs (
    ticker TEXT NOT NULL, repaired_at TEXT NOT NULL, first_date TEXT, last_date TEXT,
    rows_before INTEGER, rows_after INTEGER, max_change REAL, PRIMARY KEY (ticker, repaired_at)
) WITHOUT ROWID;
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_schema(connection):
    connection.executescript(SCHEMA)


class Throttle:
    """Space Massive requests; back off on 429 like history_backfill.MassiveDownloader."""

    def __init__(self, interval=1.0):
        self.interval = interval
        self.next = 0.0

    def get(self, url, key):
        for attempt in range(4):
            time.sleep(max(0.0, self.next - time.monotonic()))
            self.next = time.monotonic() + self.interval
            try:
                return download_json(url, key)
            except MassiveError as error:
                retry = error.status in (None, 429, 500, 502, 503, 504)
                if not retry or attempt == 3:
                    raise
                if error.status == 429:
                    self.interval = max(self.interval, 13)
                wait = max(error.retry_after or 0, 60 if error.status == 429 else 5 * 2 ** attempt)
                LOGGER.warning("Massive request failed (%s); retrying in %ss.", error.status, wait)
                self.next = time.monotonic() + wait


def fetch_splits(key, first, last, throttle=None):
    throttle = throttle or Throttle()
    query = urlencode({"execution_date.gte": first, "execution_date.lte": last,
                       "sort": "execution_date.asc", "limit": 5000})
    url = f"https://api.massive.com{SPLITS_PATH}?{query}"
    seen, rows = set(), []
    while url:
        parts = urlsplit(url)
        if parts.scheme != "https" or parts.netloc != "api.massive.com" or parts.path != SPLITS_PATH or url in seen:
            raise ValueError("Massive returned an invalid splits pagination link.")
        seen.add(url)
        payload = json.loads(throttle.get(url, key))
        if payload.get("status") not in ("OK", "DELAYED") or not isinstance(payload.get("results"), list):
            raise ValueError("Massive did not return a complete splits page.")
        for record in payload["results"]:
            try:
                rows.append((record["ticker"], record["execution_date"], float(record["split_from"]),
                             float(record["split_to"]), record.get("adjustment_type"),
                             record.get("historical_adjustment_factor")))
            except (KeyError, TypeError, ValueError):
                LOGGER.warning("Skipping malformed split record.")
        url = payload.get("next_url")
    return rows


def sync_splits(key=None):
    key = key or massive_key()
    with open_history(DATABASE) as connection:
        window = coverage(connection)
    rows = fetch_splits(key, window["firstDate"], window["lastDate"])
    with open_history(DATABASE, writable=True) as connection, connection:
        ensure_schema(connection)
        connection.executemany("INSERT OR REPLACE INTO corporate_splits VALUES (?, ?, ?, ?, ?, ?)", rows)
    return {"status": "complete", "splits": len(rows), "firstDate": window["firstDate"], "lastDate": window["lastDate"]}


def _near(value, candidates):
    if value <= 0:
        return None
    logs = [abs(math.log(value) - math.log(c)) for c in candidates]
    best = min(range(len(candidates)), key=logs.__getitem__) if candidates else None
    return candidates[best] if best is not None and logs[best] < RATIO_TOLERANCE else None


def detect_anomalies(series, splits_by_ticker):
    """series: {ticker: [(date, close), ...] sorted}. Returns list of (ticker, date, kind, ratio, detail)."""
    found = []
    for ticker, rows in series.items():
        ticker_ratios = []
        for execution, split_from, split_to in splits_by_ticker.get(ticker, ()):
            if split_from > 0 and split_to > 0:
                ticker_ratios.append((execution, split_from / split_to))   # price multiplier at the split
        candidates = [ratio for _, ratio in ticker_ratios] + [1 / ratio for _, ratio in ticker_ratios]
        jumps = []
        for index in range(1, len(rows)):
            previous, current = rows[index - 1][1], rows[index][1]
            if previous > 0 and current > 0 and abs(math.log(current / previous)) > JUMP:
                jumps.append((index, current / previous))
        for position, (index, ratio) in enumerate(jumps):
            day = rows[index][0]
            reverse = next((other for other_index, other in jumps[position + 1:]
                            if other_index - index <= REVERSAL_WINDOW and abs(math.log(ratio * other)) < 0.1), None)
            split_match = _near(ratio, candidates) if candidates else None
            on_split_day = any(execution == day for execution, _ in ticker_ratios)
            if reverse is not None:
                found.append((ticker, day, "jump_reversal", ratio, f"undone within {REVERSAL_WINDOW} sessions"))
            elif split_match is not None:
                kind = "unadjusted_split" if on_split_day else "split_basis_mismatch"
                found.append((ticker, day, kind, ratio, f"matches split ratio {split_match:.4g}"))
            elif _near(ratio, COMMON_RATIOS) is not None:
                found.append((ticker, day, "split_like_no_record", ratio, "matches a common split ratio"))
            else:
                found.append((ticker, day, "unexplained_jump", ratio, "possible real move, spin-off or bad print"))
    return found


def anomaly_report(min_dollar_volume=0.0):
    with open_history(DATABASE, writable=True) as connection:
        ensure_schema(connection)
        splits = {}
        for ticker, execution, split_from, split_to in connection.execute(
                "SELECT ticker, execution_date, split_from, split_to FROM corporate_splits"):
            splits.setdefault(ticker, []).append((execution, split_from, split_to))
        liquid = None
        if min_dollar_volume:
            liquid = {row[0] for row in connection.execute(
                "SELECT ticker FROM daily_prices GROUP BY ticker HAVING AVG(close * volume) >= ?", (min_dollar_volume,))}
        series = {}
        for ticker, day, close in connection.execute("SELECT ticker, date, close FROM daily_prices ORDER BY ticker, date"):
            if liquid is None or ticker in liquid:
                series.setdefault(ticker, []).append((day, close))
        found = detect_anomalies(series, splits)
        stamp = now()
        with connection:
            connection.execute("DELETE FROM price_anomalies WHERE status = 'open'")
            # A 'repaired' anomaly that is still detected after the re-download is present in Massive's own adjusted
            # data; mark it 'persists' so research loaders keep excluding the ticker (repairs do not retry it).
            connection.execute("UPDATE price_anomalies SET status = 'repaired' WHERE status = 'persists'")
            connection.executemany(
                """INSERT INTO price_anomalies (ticker, date, kind, ratio, detail, status, updated_at)
                   VALUES (?, ?, ?, ?, ?, 'open', ?)
                   ON CONFLICT (ticker, date, kind) DO UPDATE SET ratio = excluded.ratio, detail = excluded.detail,
                     updated_at = excluded.updated_at,
                     status = CASE WHEN price_anomalies.status = 'repaired' THEN 'persists' ELSE price_anomalies.status END
                   WHERE price_anomalies.status IN ('open', 'repaired', 'persists')""",
                [(*row, stamp) for row in found])
    kinds = {}
    for row in found:
        kinds[row[2]] = kinds.get(row[2], 0) + 1
    return {"status": "complete", "anomalies": len(found), "tickers": len({row[0] for row in found}), "byKind": kinds}


def flagged_tickers(connection, kinds=("jump_reversal", "unadjusted_split", "split_basis_mismatch")):
    marks = ",".join("?" for _ in kinds)
    return [row[0] for row in connection.execute(
        f"""SELECT a.ticker FROM price_anomalies a JOIN daily_prices p ON p.ticker = a.ticker
            WHERE a.status = 'open' AND a.kind IN ({marks})
            GROUP BY a.ticker ORDER BY AVG(p.close * p.volume) DESC""", kinds)]


def repair_ticker(ticker, key, throttle):
    with open_history(DATABASE) as connection:
        first, last, before = connection.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM daily_prices WHERE ticker = ?", (ticker,)).fetchone()
        stored = dict(connection.execute("SELECT date, close FROM daily_prices WHERE ticker = ?", (ticker,)))
    if not first:
        return {"ticker": ticker, "status": "no rows"}
    url = f"{MASSIVE_AGGS_URL}/ticker/{ticker}/range/1/day/{first}/{last}?adjusted=true&sort=asc&limit=50000"
    raw = throttle.get(url, key)
    document = json.loads(raw)
    records = read_results(raw, require_adjusted=True)
    if document.get("next_url"):
        raise ValueError(f"{ticker}: the adjusted range was truncated.")
    bars = {}
    for record in records:
        if type(record.get("t")) is not int:
            raise ValueError(f"{ticker}: a bar is missing its timestamp.")
        day = datetime.fromtimestamp(record["t"] / 1000, timezone.utc).date()
        bars[day.isoformat()] = market_bars([{**record, "T": ticker}], day)[0]
    if not bars:
        return {"ticker": ticker, "status": "empty range; left unchanged"}
    changes = [abs(math.log(bars[day]["close"] / stored[day])) for day in bars
               if day in stored and stored[day] > 0 and bars[day]["close"] > 0]
    with open_history(DATABASE, writable=True) as connection, connection:
        ensure_schema(connection)
        connection.execute("DELETE FROM daily_prices WHERE ticker = ? AND date BETWEEN ? AND ?", (ticker, first, last))
        connection.executemany(
            "INSERT INTO daily_prices (ticker, date, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(ticker, day, b["open"], b["high"], b["low"], b["close"], b["volume"]) for day, b in sorted(bars.items())])
        after = len(bars)
        max_change = math.expm1(max(changes)) if changes else None
        connection.execute("INSERT INTO ticker_repairs VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (ticker, now(), first, last, before, after, max_change))
        connection.execute("UPDATE price_anomalies SET status = 'repaired', updated_at = ? WHERE ticker = ? AND status = 'open'",
                           (now(), ticker))
    return {"ticker": ticker, "status": "repaired", "rowsBefore": before, "rowsAfter": after,
            "maxCloseChangePct": None if max_change is None else round(100 * max_change, 2)}


def mask_days(items, reason="spin-off or corporate action; return ignored in research"):
    """items: ['TICKER:YYYY-MM-DD', ...]. Marks those one-day returns as masked for research/backtests."""
    rows = []
    for item in items:
        ticker, day = item.split(":", 1)
        date.fromisoformat(day)
        rows.append((ticker.upper(), day))
    with open_history(DATABASE, writable=True) as connection, connection:
        ensure_schema(connection)
        for ticker, day in rows:
            previous, current = connection.execute(
                """SELECT (SELECT close FROM daily_prices WHERE ticker=? AND date<? ORDER BY date DESC LIMIT 1),
                          (SELECT close FROM daily_prices WHERE ticker=? AND date=?)""", (ticker, day, ticker, day)).fetchone()
            ratio = current / previous if previous and current else 0.0
            connection.execute(
                """INSERT INTO price_anomalies (ticker, date, kind, ratio, detail, status, updated_at)
                   VALUES (?, ?, 'corporate_action', ?, ?, 'masked', ?)
                   ON CONFLICT (ticker, date, kind) DO UPDATE SET status='masked', detail=excluded.detail,
                     updated_at=excluded.updated_at""", (ticker, day, ratio, reason, now()))
    return {"status": "complete", "masked": [f"{t}:{d}" for t, d in rows]}


def repair(tickers, key=None):
    key = key or massive_key()
    throttle = Throttle()
    results = []
    for ticker in tickers:
        try:
            results.append(repair_ticker(ticker, key, throttle))
        except (MassiveError, ValueError) as error:
            results.append({"ticker": ticker, "status": "failed", "message": str(error)})
        LOGGER.info("%s", results[-1])
    return {"status": "complete", "results": results}


if __name__ == "__main__":
    from structured_log import get_logger as _get_logger

    _get_logger("corporate_actions").info("started", command=" ".join(__import__("sys").argv[1:]))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-splits", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--min-dollar-volume", type=float, default=0.0)
    parser.add_argument("--repair", default="")
    parser.add_argument("--repair-flagged", action="store_true")
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--mask", default="", help="comma list of TICKER:YYYY-MM-DD returns to ignore")
    args = parser.parse_args()
    output = {}
    if args.sync_splits:
        output["splits"] = sync_splits()
    if args.report:
        output["report"] = anomaly_report(args.min_dollar_volume)
    if args.mask:
        output["mask"] = mask_days([item.strip() for item in args.mask.split(",") if item.strip()])
    tickers = [t.strip().upper() for t in args.repair.split(",") if t.strip()]
    if args.repair_flagged:
        with open_history(DATABASE) as connection:
            tickers += [t for t in flagged_tickers(connection) if t not in tickers][:args.max]
    if tickers:
        output["repair"] = repair(tickers)
    print(json.dumps(output, indent=1, default=str))
