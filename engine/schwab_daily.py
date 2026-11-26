"""Schwab long-horizon daily price backfill into Massive/prices.sqlite3 (table daily_long).

SAFETY: talks only to Schwab market-data endpoints, via schwab_data.SchwabMarketData (URL
allow-list). Trading and account endpoints are refused in that client. This module never
writes to daily_prices or market_sessions, so Massive grouped bars are left alone.

Schwab /marketdata/v1/pricehistory daily candles are split-adjusted (OHLC and volume) but
are not dividend- or total-return adjusted. There is no adjustment flag on the endpoint.
Massive daily_prices bars are split- and dividend-adjusted. Research loaders prefer Massive
on overlapping dates and may scale the Schwab leg so the splice does not jump. See
ADJUSTMENT_NOTE and SURVIVORSHIP_WARNING.

CLI (from the engine/ folder, or with PYTHONPATH set):

  python schwab_daily.py status
  python schwab_daily.py universe
  python schwab_daily.py backfill [--symbols SPY QQQ] [--limit N] [--force]
                                 [--budget-minutes N] [--ok-if-auth-missing] [--probe]
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from fetch_and_model import DATABASE, DATA_DIR, write_atomically
from local_env import PROJECT_ROOT
from schwab_data import SchwabAuthError, SchwabMarketData
from universe import CORE_ETFS

ET = ZoneInfo("America/New_York")
SUMMARY_PATH = DATA_DIR / "Research" / "schwab_daily_backfill.json"
LOCK_PATH = DATA_DIR / "Research" / "schwab_daily_backfill.lock"
SP500_PATH = PROJECT_ROOT / "public" / "sp500.json"
PAPER_DB = DATA_DIR / "paper_trading.sqlite3"
SOURCE = "schwab"
ADJUSTMENT = "split"  # split-adjusted only; not dividend-adjusted
DEFAULT_PERIOD_YEARS = 20
# Stay under Schwab's ~120 market-data requests/minute and leave headroom for the dashboard.
BACKFILL_MIN_INTERVAL = 0.6
OVERLAP_REL_TOL = 0.005
OVERLAP_ABS_TOL = 0.02
SCALE_MIN, SCALE_MAX = 0.5, 2.0

# Cash / volatility / dollar / credit / overseas / commodity proxies used by research and paper sleeves.
MACRO_PROXIES = (
    "SPY", "QQQ", "IWM", "DIA",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
    "TLT", "IEF", "IEI", "SHY", "TIP", "LQD", "HYG", "JNK", "MUB", "BND", "AGG",
    "EMB", "BIL", "SGOV", "GLD", "SLV", "DBC", "USO", "UUP", "EFA", "EEM",
    "VIXY", "SVIX", "VIXM", "VXX",
    "EWJ", "FXI", "MCHI", "EWY", "EWG", "EWU", "EZU",
)
# Schwab index symbols (pricehistory). Both dotted and bare forms are tried; empty replies are recorded.
INDEX_SYMBOLS = (
    "$SPX.X", "$DJI.X", "$COMPX.X", "$RUT.X", "$VIX.X", "$VIX3M.X", "$VXN.X", "$TNX.X",
    "$SPX", "$DJI", "$COMPX", "$RUT", "$VIX",
)

SURVIVORSHIP_WARNING = (
    "Schwab daily history only includes symbols that are still listed today (delisted names are "
    "missing), so backtests that use this long history have survivorship bias and will look more "
    "optimistic than a point-in-time universe."
)
ADJUSTMENT_NOTE = (
    "Schwab pricehistory daily candles are split-adjusted but not dividend-adjusted. Massive "
    "grouped bars in daily_prices are split- and dividend-adjusted and are preferred on "
    "overlapping dates. Pre-Massive Schwab returns therefore omit dividends unless a dividend "
    "table covers that period."
)

LONG_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_long (
  ticker TEXT NOT NULL,
  date TEXT NOT NULL,
  open REAL NOT NULL,
  high REAL NOT NULL,
  low REAL NOT NULL,
  close REAL NOT NULL,
  volume REAL NOT NULL,
  source TEXT NOT NULL DEFAULT 'schwab',
  adjustment TEXT NOT NULL DEFAULT 'split',
  fetched_at TEXT NOT NULL,
  PRIMARY KEY (ticker, date, source)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS daily_long_by_date ON daily_long (date, ticker);
CREATE TABLE IF NOT EXISTS daily_long_progress (
  ticker TEXT PRIMARY KEY,
  first_date TEXT,
  last_date TEXT,
  rows INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  message TEXT,
  requested_start TEXT,
  requested_end TEXT,
  period_mode TEXT,
  fetched_at TEXT NOT NULL
) WITHOUT ROWID;
"""


def _now_stamp():
    return datetime.now().isoformat(timespec="seconds")


def session_date_from_ms(ms):
    """Map a Schwab candle datetime (epoch ms) to a US/Eastern session date (YYYY-MM-DD).

    Daily bars arrive as midnight ET, the regular-session close (16:00 ET), or midnight UTC
    (which is 19:00 or 20:00 ET the previous calendar day). Times at or after 17:00 ET are
    treated as the next calendar date so UTC-midnight stamps keep the intended session date.
    """
    stamp = datetime.fromtimestamp(int(ms) / 1000, ET)
    if stamp.hour >= 17:
        stamp = stamp + timedelta(days=1)
    return stamp.date().isoformat()


def parse_candles(ticker, payload):
    """Turn a Schwab pricehistory payload into daily_long row dicts (no DB writes)."""
    if not isinstance(payload, dict):
        return []
    candles = payload.get("candles") or []
    rows = []
    seen = set()
    for candle in candles:
        ms = candle.get("datetime")
        if ms is None:
            continue
        try:
            day = session_date_from_ms(ms)
            opening, high, low, close = (float(candle["open"]), float(candle["high"]),
                                         float(candle["low"]), float(candle["close"]))
            volume = float(candle.get("volume") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if day in seen:
            continue
        seen.add(day)
        rows.append({"ticker": ticker, "date": day, "open": opening, "high": high, "low": low,
                     "close": close, "volume": volume, "source": SOURCE, "adjustment": ADJUSTMENT})
    rows.sort(key=lambda row: row["date"])
    return rows


def connect_prices(path=None, writable=False):
    path = Path(path or DATABASE)
    if writable:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(LONG_SCHEMA)
    else:
        if not path.is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection, name):
    return connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (name,)).fetchone() is not None


def store_daily_rows(connection, ticker, rows, period_mode, requested_start=None, requested_end=None,
                     status="complete", message=None, fetched_at=None):
    """Insert or replace daily_long rows for one ticker and update progress. Does not touch daily_prices."""
    fetched_at = fetched_at or _now_stamp()
    values = [(row["ticker"], row["date"], row["open"], row["high"], row["low"], row["close"],
               row["volume"], row.get("source", SOURCE), row.get("adjustment", ADJUSTMENT), fetched_at)
              for row in rows]
    with connection:
        if values:
            connection.executemany(
                """INSERT OR REPLACE INTO daily_long
                   (ticker, date, open, high, low, close, volume, source, adjustment, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                values,
            )
        first, last, count = connection.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM daily_long WHERE ticker=? AND source=?",
            (ticker, SOURCE),
        ).fetchone()
        connection.execute(
            """INSERT INTO daily_long_progress
               (ticker, first_date, last_date, rows, status, message, requested_start, requested_end, period_mode, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (ticker) DO UPDATE SET
                 first_date=excluded.first_date, last_date=excluded.last_date, rows=excluded.rows,
                 status=excluded.status, message=excluded.message, requested_start=excluded.requested_start,
                 requested_end=excluded.requested_end, period_mode=excluded.period_mode, fetched_at=excluded.fetched_at""",
            (ticker, first, last, int(count or 0), status, message, requested_start, requested_end,
             period_mode, fetched_at),
        )
    return {"ticker": ticker, "first": first, "last": last, "rows": int(count or 0), "stored": len(values),
            "status": status, "message": message}


def progress_map(connection):
    if not table_exists(connection, "daily_long_progress"):
        return {}
    return {row["ticker"]: dict(row) for row in connection.execute("SELECT * FROM daily_long_progress")}


def _load_json_tickers(path, key="tickers"):
    path = Path(path)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        values = data.get(key) or data.get("symbols") or []
    else:
        values = data
    out = []
    for item in values:
        symbol = str(item).strip().upper()
        if symbol:
            out.append(symbol)
    return out


def _sqlite_tickers(path, queries):
    path = Path(path)
    if not path.is_file():
        return []
    found = []
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    except sqlite3.Error:
        return []
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table, column, sql in queries:
            if table not in tables:
                continue
            try:
                found.extend(row[0] for row in connection.execute(sql) if row[0])
            except sqlite3.Error:
                continue
    finally:
        connection.close()
    return found


def _csv_tickers(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    found = []
    try:
        import csv
        for path in directory.glob("*.csv"):
            name = path.name.lower()
            if not any(part in name for part in ("trade", "decision", "holding")):
                continue
            try:
                with path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        continue
                    column = next((c for c in reader.fieldnames if c.lower() in ("ticker", "symbol", "t")), None)
                    if not column:
                        continue
                    for i, row in enumerate(reader):
                        if i > 50_000:
                            break
                        value = (row.get(column) or "").strip().upper()
                        if value:
                            found.append(value)
            except (OSError, csv.Error, UnicodeError):
                continue
    except Exception:
        return found
    return found


def _bond_etfs():
    try:
        from strategies.bonds import BOND_ETFS
        return tuple(BOND_ETFS)
    except Exception:
        return ()


def _daytrade_symbols():
    try:
        from schwab_data import DEFAULT_SYMBOLS
        return tuple(DEFAULT_SYMBOLS)
    except Exception:
        return ()


def collect_universe(include_research=True, database=None, paper_db=None, research_dir=None):
    """Priority-ordered unique tickers: indices, core/macro ETFs, S&P 500, paper names, research liquid."""
    database = Path(database or DATABASE)
    paper_db = Path(paper_db or PAPER_DB)
    research_dir = Path(research_dir or (DATA_DIR / "Research"))
    groups = []
    groups.append(("indices", list(INDEX_SYMBOLS)))
    core = list(dict.fromkeys(list(CORE_ETFS) + list(MACRO_PROXIES) + list(_bond_etfs())))
    groups.append(("core_etf_macro", core))
    groups.append(("sp500", _load_json_tickers(SP500_PATH)))
    groups.append(("daytrade", list(_daytrade_symbols())))
    paper = _sqlite_tickers(paper_db, [
        ("positions", "ticker", "SELECT DISTINCT ticker FROM positions WHERE ticker IS NOT NULL"),
        ("signal_state", "ticker", "SELECT DISTINCT ticker FROM signal_state WHERE ticker IS NOT NULL"),
        ("decision_log", "ticker", "SELECT DISTINCT ticker FROM decision_log WHERE ticker IS NOT NULL"),
        ("fills", "ticker", "SELECT DISTINCT ticker FROM fills WHERE ticker IS NOT NULL"),
        ("orders", "ticker", "SELECT DISTINCT ticker FROM orders WHERE ticker IS NOT NULL"),
        ("dt_trades", "symbol", "SELECT DISTINCT symbol FROM dt_trades WHERE symbol IS NOT NULL"),
        ("dt_decision_log", "ticker", "SELECT DISTINCT ticker FROM dt_decision_log WHERE ticker IS NOT NULL"),
        ("advisory_verdicts", "ticker", "SELECT DISTINCT ticker FROM advisory_verdicts WHERE ticker IS NOT NULL"),
    ])
    groups.append(("paper", paper))
    groups.append(("research_csv", _csv_tickers(research_dir)))
    # Symbols picked in the chart picker and fetched on demand (schwab_ondemand.py) stay on the daily update.
    groups.append(("on_demand", _load_json_tickers(research_dir / "ondemand_symbols.json")))
    if include_research and database.is_file():
        liquid = []
        try:
            connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
            try:
                liquid = [row[0] for row in connection.execute(
                    """SELECT ticker FROM daily_prices
                       GROUP BY ticker HAVING AVG(close * volume) >= 5000000 AND AVG(close) >= 2.0
                       ORDER BY AVG(close * volume) DESC""")]
            finally:
                connection.close()
        except sqlite3.Error:
            liquid = []
        groups.append(("research_liquid", liquid))
    ordered, seen, counts = [], set(), {}
    for name, symbols in groups:
        added = 0
        for symbol in symbols:
            ticker = str(symbol).strip().upper()
            if not ticker or ticker in seen:
                continue
            # Allow Schwab index tickers starting with $; otherwise ordinary listed symbols.
            if ticker.startswith("$"):
                if not all(ch.isalnum() or ch in ".-" for ch in ticker[1:]):
                    continue
            elif not (ticker[0].isalpha() and all(ch.isalnum() or ch in ".-" for ch in ticker)):
                continue
            seen.add(ticker)
            ordered.append(ticker)
            added += 1
        counts[name] = added
    return {"tickers": ordered, "counts": counts, "total": len(ordered)}


def should_skip(progress, ticker, today=None, force=False, incremental_days=7):
    """True when a ticker already has a complete recent history and does not need a full refetch."""
    if force:
        return False
    row = progress.get(ticker) or {}
    if row.get("status") not in ("complete", "empty"):
        return False
    if row.get("status") == "empty":
        fetched = row.get("fetched_at") or ""
        # Retry empty tickers after a day (new listings / transient 404s).
        return bool(fetched) and fetched[:10] >= (today or date.today()).isoformat()
    last = row.get("last_date")
    if not last:
        return False
    today = today or date.today()
    try:
        last_day = date.fromisoformat(last)
    except ValueError:
        return False
    return last_day >= today - timedelta(days=incremental_days)


def incremental_start(progress, ticker, fallback_years=DEFAULT_PERIOD_YEARS, overlap_days=5):
    row = progress.get(ticker) or {}
    last = row.get("last_date")
    if row.get("status") == "complete" and last:
        try:
            return date.fromisoformat(last) - timedelta(days=overlap_days)
        except ValueError:
            pass
    return date.today() - timedelta(days=365 * fallback_years + 5)


def pricehistory_params(symbol, *, start=None, end=None, period_years=DEFAULT_PERIOD_YEARS, use_start_end=False):
    params = {
        "symbol": symbol,
        "frequencyType": "daily",
        "frequency": 1,
        "needExtendedHoursData": "false",
    }
    if use_start_end or start is not None:
        start = start or (date.today() - timedelta(days=365 * period_years + 5))
        end = end or date.today()
        start_dt = datetime(start.year, start.month, start.day, tzinfo=ET)
        # End-of-day ET so the current session is included.
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=ET)
        params["startDate"] = int(start_dt.timestamp() * 1000)
        params["endDate"] = int(end_dt.timestamp() * 1000)
        params["period_mode"] = f"start-end:{start.isoformat()}:{end.isoformat()}"
    else:
        params["periodType"] = "year"
        params["period"] = int(period_years)
        params["period_mode"] = f"year-{period_years}"
    return params


def fetch_daily(client, symbol, **kwargs):
    params = pricehistory_params(symbol, **kwargs)
    mode = params.pop("period_mode")
    payload = client.get("/marketdata/v1/pricehistory", **params)
    return payload, mode, params


def probe_range(client, symbol="SPY", years=DEFAULT_PERIOD_YEARS, deeper_start=date(1990, 1, 1)):
    """Compare period=20 vs startDate/endDate and record how far each actually goes (one symbol)."""
    results = {}
    payload, mode, _ = fetch_daily(client, symbol, period_years=years, use_start_end=False)
    rows = parse_candles(symbol, payload)
    results["periodYear20"] = {
        "mode": mode, "rows": len(rows),
        "first": rows[0]["date"] if rows else None, "last": rows[-1]["date"] if rows else None,
        "empty": bool(payload.get("empty")) if isinstance(payload, dict) else None,
    }
    payload2, mode2, _ = fetch_daily(client, symbol, start=deeper_start, end=date.today(), use_start_end=True)
    rows2 = parse_candles(symbol, payload2)
    results["startDate1990"] = {
        "mode": mode2, "rows": len(rows2),
        "first": rows2[0]["date"] if rows2 else None, "last": rows2[-1]["date"] if rows2 else None,
        "empty": bool(payload2.get("empty")) if isinstance(payload2, dict) else None,
    }
    firsts = [r["first"] for r in results.values() if r.get("first")]
    lasts = [r["last"] for r in results.values() if r.get("last")]
    results["earliest"] = min(firsts) if firsts else None
    results["latest"] = max(lasts) if lasts else None
    results["startDateGoesFurther"] = bool(
        results["startDate1990"]["first"] and results["periodYear20"]["first"]
        and results["startDate1990"]["first"] < results["periodYear20"]["first"]
    )
    return results, rows2 if results["startDateGoesFurther"] else rows, (
        mode2 if results["startDateGoesFurther"] else mode
    )


def pid_is_alive(pid):
    """Return True if *pid* is a still-running process. Used to detect a stale backfill lock."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        process_query_limited_information = 0x1000
        still_active = 259
        error_access_denied = 5
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not handle:
            return ctypes.get_last_error() == error_access_denied
        try:
            exit_code = wintypes.DWORD()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            return bool(ok) and exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _lock_payload(path):
    try:
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


class BackfillLock:
    def __init__(self, path=LOCK_PATH):
        self.path = Path(path)
        self.owned = False

    def _stale(self):
        payload = _lock_payload(self.path)
        try:
            pid = int(payload["pid"]) if payload.get("pid") is not None else None
        except (TypeError, ValueError):
            pid = None
        if pid and pid > 0:
            # A recorded PID that is no longer running means the previous backfill crashed
            # or was killed (for example when the dashboard server stopped). Do not wait 3 hours.
            return not pid_is_alive(pid)
        try:
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            age = 0
        # Legacy lock without a usable PID: only treat as stale after 3 hours.
        return age > 3 * 3600

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "startedAt": _now_stamp()}))
            self.owned = True
            return True
        except FileExistsError:
            if self._stale():
                try:
                    self.path.unlink()
                except OSError:
                    return False
                return self.acquire()
            return False

    def release(self):
        if self.owned:
            try:
                self.path.unlink()
            except OSError:
                pass
            self.owned = False

    def __enter__(self):
        if not self.acquire():
            raise RuntimeError(f"A Schwab daily backfill is already running ({self.path}).")
        return self

    def __exit__(self, *exc):
        self.release()

def _summarize_db(connection, extra=None):
    summary = {"table": "daily_long", "source": SOURCE, "adjustment": ADJUSTMENT,
               "adjustmentNote": ADJUSTMENT_NOTE, "survivorshipWarning": SURVIVORSHIP_WARNING}
    if table_exists(connection, "daily_long"):
        row = connection.execute(
            "SELECT COUNT(*) AS rows, COUNT(DISTINCT ticker) AS tickers, MIN(date) AS first, MAX(date) AS last "
            "FROM daily_long WHERE source=?",
            (SOURCE,),
        ).fetchone()
        summary.update({"rows": row["rows"], "tickers": row["tickers"], "first": row["first"], "last": row["last"]})
    else:
        summary.update({"rows": 0, "tickers": 0, "first": None, "last": None})
    if extra:
        summary.update(extra)
    return summary


def write_summary(payload, path=None):
    path = Path(path or SUMMARY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(payload, indent=2).encode())
    return path


def token_status_safe(client=None):
    client = client or SchwabMarketData()
    return client.token_status()


def backfill(symbols=None, *, include_research=True, limit=None, force=False, budget_minutes=None,
             ok_if_auth_missing=False, probe=True, client=None, database=None, summary_path=None,
             log=None, today=None, lock_path=None):
    """Fetch daily history for the project universe. Resumable and incremental. No Massive clobber."""
    log = log or (lambda message, **fields: print(message, file=sys.stderr, flush=True))
    today = today or date.today()
    database = Path(database or DATABASE)
    started = time.monotonic()
    deadline = None if not budget_minutes else started + float(budget_minutes) * 60
    client = client or SchwabMarketData(min_interval=BACKFILL_MIN_INTERVAL)

    status = token_status_safe(client)
    if not status.get("usable") and not (status.get("accessTokenValid") or status.get("canRefresh")):
        # token_status uses usable only in the JS client; the Python client uses accessTokenValid/canRefresh.
        usable = bool(status.get("accessTokenValid") or status.get("canRefresh"))
    else:
        usable = bool(status.get("usable", status.get("accessTokenValid") or status.get("canRefresh")))
    # Python token_status has no "usable" key — compute it the same way as workbench.js.
    if "usable" not in status:
        status["usable"] = bool(status.get("accessTokenValid") or status.get("canRefresh"))
        usable = status["usable"]

    payload = {"generatedAt": _now_stamp(), "status": "running", "auth": {
        "tokenFile": bool(status.get("tokenFile")), "usable": bool(status.get("usable")),
        "accessTokenValid": bool(status.get("accessTokenValid")), "canRefresh": bool(status.get("canRefresh")),
        "accessTokenExpires": status.get("accessTokenExpires"),
        "refreshTokenExpiresAbout": status.get("refreshTokenExpiresAbout"),
        "message": status.get("message") or status.get("reason"),
    }, "adjustment": ADJUSTMENT, "adjustmentNote": ADJUSTMENT_NOTE,
       "survivorshipWarning": SURVIVORSHIP_WARNING}

    if not usable:
        message = status.get("message") or (
            "Schwab is not authorized. Set up the token (and SCHWAB_APP_KEY / SCHWAB_APP_SECRET in "
            ".env.local so the access token can be refreshed) and re-run this backfill. "
            "No prices were requested."
        )
        payload["status"] = "auth_required"
        payload["message"] = message
        write_summary(payload, summary_path)
        log(message)
        if ok_if_auth_missing:
            return payload
        raise SchwabAuthError(message)

    universe = collect_universe(include_research=include_research, database=database)
    tickers = list(symbols) if symbols else list(universe["tickers"])
    if limit is not None:
        tickers = tickers[: int(limit)]
    payload["universe"] = {"total": universe["total"], "counts": universe["counts"], "selected": len(tickers)}

    counts = dict.fromkeys(("fetched", "skipped", "empty", "errors", "incremental", "auth_stopped"), 0)
    issues = []
    probe_info = None
    prefer_start_end = False

    lock = BackfillLock(lock_path or LOCK_PATH)
    if not lock.acquire():
        payload.update({"status": "busy", "message": "A Schwab daily backfill is already running."})
        write_summary(payload, summary_path)
        return payload
    try:
        connection = connect_prices(database, writable=True)
        try:
            progress = progress_map(connection)
            if probe:
                try:
                    probe_info, _, _ = probe_range(client)
                    prefer_start_end = bool(probe_info.get("startDateGoesFurther"))
                    payload["probe"] = probe_info
                    log(f"range probe: period-20 first={probe_info['periodYear20']['first']} "
                        f"startDate first={probe_info['startDate1990']['first']} "
                        f"startDateGoesFurther={prefer_start_end}")
                except SchwabAuthError:
                    raise
                except Exception as error:
                    payload["probe"] = {"error": str(error)}
                    log(f"range probe failed: {error}")

            for index, ticker in enumerate(tickers, start=1):
                if deadline and time.monotonic() >= deadline:
                    payload["status"] = "partial"
                    payload["message"] = f"Stopped after the time budget ({budget_minutes} minutes); re-run to resume."
                    log(payload["message"])
                    break
                if should_skip(progress, ticker, today=today, force=force):
                    # Incremental catch-up when the last bar is a few days stale but still "complete".
                    row = progress.get(ticker) or {}
                    last = row.get("last_date")
                    stale = False
                    if last:
                        try:
                            stale = date.fromisoformat(last) < today - timedelta(days=3)
                        except ValueError:
                            stale = True
                    if not stale or row.get("status") == "empty":
                        counts["skipped"] += 1
                        if index == 1 or index % 25 == 0 or index == len(tickers):
                            log(f"progress {index}/{len(tickers)} skipped={counts['skipped']} fetched={counts['fetched']}")
                        continue

                start = incremental_start(progress, ticker) if not force else date.today() - timedelta(days=365 * DEFAULT_PERIOD_YEARS + 5)
                use_start_end = prefer_start_end or (not force and (progress.get(ticker) or {}).get("status") == "complete")
                period_mode = None
                try:
                    if use_start_end:
                        payload_bars, period_mode, params = fetch_daily(
                            client, ticker, start=start, end=today, use_start_end=True)
                    else:
                        payload_bars, period_mode, params = fetch_daily(
                            client, ticker, period_years=DEFAULT_PERIOD_YEARS, use_start_end=False)
                    rows = parse_candles(ticker, payload_bars)
                    requested_start = date.fromtimestamp(params["startDate"] / 1000).isoformat() if "startDate" in params else None
                    requested_end = date.fromtimestamp(params["endDate"] / 1000).isoformat() if "endDate" in params else None
                    empty = not rows or bool(isinstance(payload_bars, dict) and payload_bars.get("empty"))
                    result = store_daily_rows(
                        connection, ticker, rows, period_mode,
                        requested_start=requested_start, requested_end=requested_end,
                        status="empty" if empty else "complete",
                        message=None if rows else "Schwab returned no daily candles (unknown, delisted, or index symbol).",
                    )
                    progress[ticker] = {"status": result["status"], "last_date": result["last"],
                                        "first_date": result["first"], "fetched_at": _now_stamp()}
                    if empty:
                        counts["empty"] += 1
                    elif use_start_end and (progress.get(ticker) or {}).get("status"):
                        counts["incremental"] += 1
                        counts["fetched"] += 1
                    else:
                        counts["fetched"] += 1
                    log(f"{ticker}: {result['status']} {result['rows']} rows {result['first']} → {result['last']}")
                except SchwabAuthError as error:
                    counts["auth_stopped"] += 1
                    store_daily_rows(connection, ticker, [], period_mode or "auth",
                                     status="auth_required", message=str(error))
                    payload["message"] = str(error)
                    payload["status"] = "auth_required"
                    log(f"{ticker}: auth stopped the run")
                    break
                except Exception as error:
                    counts["errors"] += 1
                    if len(issues) < 20:
                        issues.append(f"{ticker}: {error}")
                    try:
                        store_daily_rows(connection, ticker, [], period_mode or "error",
                                         status="error", message=str(error)[:400])
                    except sqlite3.Error:
                        pass
                    log(f"{ticker}: error {error}")
                if index == 1 or index % 10 == 0 or index == len(tickers):
                    log(f"progress {index}/{len(tickers)} fetched={counts['fetched']} skipped={counts['skipped']} "
                        f"empty={counts['empty']} errors={counts['errors']}")

            coverage = _summarize_db(connection, extra={"counts": counts, "issues": issues,
                                                        "checked": min(index if tickers else 0, len(tickers)),
                                                        "selected": len(tickers)})
            if payload.get("status") in ("running",):
                payload["status"] = "complete" if not counts["errors"] else "partial"
                if counts["errors"] and "message" not in payload:
                    payload["message"] = f"{counts['errors']} ticker(s) failed; re-run to retry them."
            payload["coverage"] = coverage
            payload["counts"] = counts
            payload["issues"] = issues
            payload["elapsedSeconds"] = round(time.monotonic() - started, 1)
            write_summary(payload, summary_path)
            return payload
        finally:
            connection.close()
    finally:
        lock.release()


# --------------------------------------------------------------------------- merge / overlap (used by market_data.load_panel)

def load_long_frame(connection, start=None, end=None, tickers=None):
    if not table_exists(connection, "daily_long"):
        return []
    sql = "SELECT ticker, date, open, high, low, close, volume FROM daily_long WHERE source=?"
    params = [SOURCE]
    if start:
        sql += " AND date >= ?"
        params.append(start)
    if end:
        sql += " AND date <= ?"
        params.append(end)
    if tickers:
        sql += f" AND ticker IN ({','.join('?' * len(tickers))})"
        params.extend(tickers)
    return [dict(row) for row in connection.execute(sql, params)]


def overlap_report(massive_rows, schwab_rows, rel_tol=OVERLAP_REL_TOL, abs_tol=OVERLAP_ABS_TOL):
    """Compare close prices (and one-day returns) on (ticker, date) keys present in both sources."""
    mass = {(row["ticker"], row["date"]): float(row["close"]) for row in massive_rows}
    schw = {(row["ticker"], row["date"]): float(row["close"]) for row in schwab_rows}
    keys = sorted(set(mass) & set(schw))
    if not keys:
        return {"overlapDays": 0, "tickers": 0, "mismatchRate": None, "returnMismatchRate": None,
                "medianAbsRelDiff": None, "p95AbsRelDiff": None, "thresholds": {}}
    rels = []
    mismatches = 0
    by_ticker = {}
    for key in keys:
        m, s = mass[key], schw[key]
        if m == 0 or s == 0:
            rel = 1.0 if m != s else 0.0
        else:
            rel = abs(m - s) / abs(m)
        rels.append(rel)
        if rel > rel_tol and abs(m - s) > abs_tol:
            mismatches += 1
        by_ticker.setdefault(key[0], []).append((key[1], m, s))
    rels.sort()
    mid = rels[len(rels) // 2]
    p95 = rels[min(len(rels) - 1, int(round(0.95 * (len(rels) - 1))))]
    # Daily close-to-close return mismatches (fairer when dividend back-adjustment shifts levels).
    ret_pairs = 0
    ret_mis = 0
    for ticker, items in by_ticker.items():
        items.sort()
        for (d0, m0, s0), (d1, m1, s1) in zip(items, items[1:]):
            if m0 == 0 or s0 == 0:
                continue
            ret_pairs += 1
            if abs((m1 / m0) - (s1 / s0)) > 0.002:  # 20 bps
                ret_mis += 1
    def _rate(count, total, cutoff):
        return round(sum(1 for r in rels if r > cutoff) / total, 4) if total else None

    total = len(keys)
    return {
        "overlapDays": total,
        "tickers": len(by_ticker),
        "mismatchRate": round(mismatches / total, 4) if total else None,
        "returnMismatchRate": round(ret_mis / ret_pairs, 4) if ret_pairs else None,
        "returnPairs": ret_pairs,
        "medianAbsRelDiff": round(mid, 6),
        "p95AbsRelDiff": round(p95, 6),
        "thresholds": {str(c): _rate(None, total, c) for c in (0.001, 0.005, 0.01, 0.02)},
        "relTol": rel_tol, "absTol": abs_tol,
        "note": "Level differences can come from Massive dividend back-adjustment; return mismatches are more informative.",
    }


def splice_scale_factors(massive_rows, schwab_rows):
    """Per-ticker factor (Massive close / Schwab close) on the first overlapping date, when sane."""
    first_mass = {}
    for row in massive_rows:
        key = row["ticker"]
        if key not in first_mass or row["date"] < first_mass[key][0]:
            first_mass[key] = (row["date"], float(row["close"]))
    schw_on = {(row["ticker"], row["date"]): float(row["close"]) for row in schwab_rows}
    scales = {}
    skipped = {}
    for ticker, (day, mass_close) in first_mass.items():
        schw_close = schw_on.get((ticker, day))
        if not schw_close or not mass_close:
            continue
        factor = mass_close / schw_close
        if SCALE_MIN <= factor <= SCALE_MAX:
            scales[ticker] = {"date": day, "factor": factor}
        else:
            skipped[ticker] = {"date": day, "factor": factor, "reason": "scale outside 0.5-2.0 (possible split disagreement)"}
    return scales, skipped


def merge_rows(massive_rows, schwab_rows, prefer="massive"):
    """Union of rows; Massive wins on (ticker, date). Schwab-only earlier (or later) dates are kept.

    Schwab prices are optionally scaled to Massive on the first overlap date so the splice does not jump.
    Returns (rows, meta).
    """
    scales, skipped_scales = splice_scale_factors(massive_rows, schwab_rows)
    out = {}
    source = {}
    for row in schwab_rows:
        item = dict(row)
        factor = (scales.get(item["ticker"]) or {}).get("factor")
        if factor:
            for field in ("open", "high", "low", "close"):
                item[field] = float(item[field]) * factor
        out[(item["ticker"], item["date"])] = item
        source[(item["ticker"], item["date"])] = "schwab"
    if prefer == "massive":
        for row in massive_rows:
            out[(row["ticker"], row["date"])] = dict(row)
            source[(row["ticker"], row["date"])] = "massive"
    else:
        for row in massive_rows:
            key = (row["ticker"], row["date"])
            if key not in out:
                out[key] = dict(row)
                source[key] = "massive"
    overlap = overlap_report(massive_rows, schwab_rows)
    overlap["scaledTickers"] = len(scales)
    overlap["scaleSkipped"] = skipped_scales
    return list(out.values()), {"overlap": overlap, "scales": scales, "prefer": prefer,
                                "survivorshipWarning": SURVIVORSHIP_WARNING,
                                "adjustmentNote": ADJUSTMENT_NOTE}


def data_warnings(meta=None, used_long=False):
    warnings = []
    if used_long:
        warnings.append(SURVIVORSHIP_WARNING)
        warnings.append(ADJUSTMENT_NOTE)
        overlap = (meta or {}).get("overlap") or {}
        if overlap.get("overlapDays"):
            warnings.append(
                f"On {overlap['overlapDays']} overlapping ticker-days, Massive was preferred. "
                f"Close-level mismatch rate (>{OVERLAP_REL_TOL:.1%} and >${OVERLAP_ABS_TOL:.2f}) "
                f"was {overlap.get('mismatchRate')}; close-to-close return mismatch rate was "
                f"{overlap.get('returnMismatchRate')} (median abs close diff {overlap.get('medianAbsRelDiff')})."
            )
    return warnings


def merge_caveats(base, warnings):
    """Prepend long-history warnings and drop stale 'only ~2 years' notes when that history is in use."""
    warnings = [w for w in (warnings or []) if w]
    if not warnings:
        return list(base)
    kept = []
    for line in base or []:
        lower = line.lower()
        if "only ~2 years" in lower or "about two years" in lower or "holds about two years" in lower:
            continue
        if "501 sessions" in lower and "2024-11-25" in line:
            continue
        kept.append(line)
    # Dedup while preserving order.
    seen, out = set(), []
    for line in list(warnings) + kept:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out


def read_summary(path=None):
    path = Path(path or SUMMARY_PATH)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "universe", "backfill"])
    parser.add_argument("--symbols", nargs="*", default=None, help="Limit to these tickers (default: project universe).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Refetch tickers even if already complete.")
    parser.add_argument("--budget-minutes", type=float, default=None)
    parser.add_argument("--ok-if-auth-missing", action="store_true",
                        help="Exit 0 with status auth_required instead of exit 2 (for autopilot).")
    parser.add_argument("--no-probe", action="store_true", help="Skip the period=20 vs startDate range probe.")
    parser.add_argument("--core-only", action="store_true",
                        help="Skip the broad research-liquid sleeve (indices, ETFs, S&P 500, paper names only).")
    args = parser.parse_args(argv)
    if args.command == "status":
        status = token_status_safe()
        if "usable" not in status:
            status["usable"] = bool(status.get("accessTokenValid") or status.get("canRefresh"))
        coverage = None
        if Path(DATABASE).is_file():
            try:
                connection = connect_prices(DATABASE, writable=False)
                try:
                    coverage = _summarize_db(connection)
                finally:
                    connection.close()
            except (OSError, sqlite3.Error) as error:
                coverage = {"error": str(error)}
        print(json.dumps({"auth": {
            "tokenFile": bool(status.get("tokenFile")), "usable": status.get("usable"),
            "accessTokenValid": status.get("accessTokenValid"), "canRefresh": status.get("canRefresh"),
            "accessTokenExpires": status.get("accessTokenExpires"),
            "refreshTokenExpiresAbout": status.get("refreshTokenExpiresAbout"),
            "message": status.get("message"),
        }, "coverage": coverage, "summary": read_summary(),
            "adjustmentNote": ADJUSTMENT_NOTE, "survivorshipWarning": SURVIVORSHIP_WARNING}, indent=2))
        return 0
    if args.command == "universe":
        print(json.dumps(collect_universe(include_research=not args.core_only), indent=2))
        return 0
    try:
        result = backfill(args.symbols, include_research=not args.core_only, limit=args.limit, force=args.force,
                          budget_minutes=args.budget_minutes, ok_if_auth_missing=args.ok_if_auth_missing,
                          probe=not args.no_probe)
    except SchwabAuthError as error:
        print(json.dumps({"status": "auth_required", "message": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") in ("complete", "partial", "auth_required", "busy") else 1


if __name__ == "__main__":
    from structured_log import get_logger, log_run

    with log_run("schwab_daily"):
        log = get_logger("schwab_daily")

        def _log(message, **fields):
            log.info(message, **fields)
            print(message, file=sys.stderr, flush=True)

        # Rebind default log inside main via env? Keep stderr + structured via wrapping backfill in main.
        raise SystemExit(main())
