"""On-demand Schwab history for one symbol picked in the chart picker (read-only market data).

Nothing is bulk-imported: `search` only looks symbols up (Schwab instruments endpoint) and `fetch` runs only for the one
symbol the user selected, or whose local history is short. History lands in the normal store (Massive/prices.sqlite3
table daily_long, same schema/writer as schwab_daily.py; recent 5-minute bars in Massive/intraday.sqlite3), and the
symbol is added to Massive/Research/ondemand_symbols.json, which schwab_daily.collect_universe() includes so the regular
daily update keeps it current.

Uses schwab_data.SchwabMarketData (URL allow-list: market data + token refresh only). It does not take the bulk
BackfillLock: it is a separate small fetch under a per-symbol lock, rate-limited by a per-symbol cooldown.

  python schwab_ondemand.py search BLOZ
  python schwab_ondemand.py fetch BLOZF
  python schwab_ondemand.py status BLOZF
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fetch_and_model import DATABASE, DATA_DIR

ONDEMAND_DIR = DATA_DIR / "Research" / "ondemand"
TRACKED_PATH = DATA_DIR / "Research" / "ondemand_symbols.json"
SEARCH_CACHE = ONDEMAND_DIR / "search_cache.json"
STAMP_PATH = ONDEMAND_DIR / "updated.stamp"     # touched after every store; chart caches key on it
SEARCH_TTL = 24 * 3600
COOLDOWN_OK = 7 * 24 * 3600        # a complete fetch is not repeated for a week (the daily update keeps it current)
COOLDOWN_FAIL = 30 * 60            # failed / invalid symbols may retry after 30 minutes
SHORT_BARS = 750                   # local history shorter than ~3 years (and no Schwab leg) counts as short
SYMBOL_RE = re.compile(r"^\$?[A-Z][A-Z0-9.\-/]{0,11}$")
QUERY_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_symbol(value):
    s = str(value or "").strip().upper()
    if not SYMBOL_RE.match(s):
        raise ValueError("invalid symbol")
    return s


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def status_path(symbol):
    return ONDEMAND_DIR / f"{symbol.replace('/', '_').replace('$', 'IDX_')}.json"


def local_coverage(symbol, database=None):
    """Rows/first/last in daily_prices (Massive) and daily_long (Schwab) for one symbol."""
    path = Path(database or DATABASE)
    out = {"massive": {"rows": 0, "first": None, "last": None}, "schwab": {"rows": 0, "first": None, "last": None}}
    if not path.is_file():
        return out
    con = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        for key, table in (("massive", "daily_prices"), ("schwab", "daily_long")):
            try:
                n, first, last = con.execute(f"SELECT COUNT(*), MIN(date), MAX(date) FROM {table} WHERE ticker=?", (symbol,)).fetchone()
                out[key] = {"rows": int(n or 0), "first": first, "last": last}
            except sqlite3.Error:
                pass
    finally:
        con.close()
    return out


def is_short(cov):
    return cov["schwab"]["rows"] == 0 and cov["massive"]["rows"] < SHORT_BARS


def status(symbol):
    st = _read_json(status_path(symbol), {})
    cov = local_coverage(symbol)
    return {"symbol": symbol, "status": st.get("status", "idle"), **{k: v for k, v in st.items() if k != "status"},
            "local": cov, "short": is_short(cov), "tracked": symbol in tracked()}


def tracked():
    data = _read_json(TRACKED_PATH, {})
    return [str(s).upper() for s in (data.get("tickers") if isinstance(data, dict) else data) or []]


def add_tracked(symbol):
    data = _read_json(TRACKED_PATH, {})
    if not isinstance(data, dict):
        data = {"tickers": list(data or [])}
    tickers = [str(s).upper() for s in data.get("tickers") or []]
    if symbol not in tickers:
        tickers.append(symbol)
    data.update({"tickers": tickers, "note": "Symbols picked in the chart picker; schwab_daily.py keeps them updated.",
                 "updatedAt": _now()})
    _write_json(TRACKED_PATH, data)


def should_fetch(symbol, force=False, now=None):
    """(bool, reason). Respect the per-symbol cooldown so repeated chart opens never hammer Schwab."""
    now = now or time.time()
    st = _read_json(status_path(symbol), {})
    if force:
        return True, "forced"
    if st.get("status") == "running" and now - float(st.get("startedTs") or 0) < 300:
        return False, "already running"
    done = float(st.get("finishedTs") or 0)
    if st.get("status") == "complete" and now - done < COOLDOWN_OK:
        return False, "fetched recently"
    if st.get("status") in ("error", "invalid", "empty") and now - done < COOLDOWN_FAIL:
        return False, "recent failure (cooldown)"
    return True, "ok"


def _client():
    from schwab_data import SchwabMarketData
    return SchwabMarketData()


def search(query, client=None, now=None):
    """Schwab instrument lookup (exact symbol, then prefix regex). Cached 24 h per query. Read-only."""
    q = str(query or "").strip().upper().lstrip("$")
    if not QUERY_RE.match(q):
        return {"query": q, "matches": [], "error": "invalid query"}
    now = now or time.time()
    cache = _read_json(SEARCH_CACHE, {})
    hit = cache.get(q)
    if hit and now - hit.get("ts", 0) < SEARCH_TTL:
        return {"query": q, "matches": hit["matches"], "cached": True}
    client = client or _client()
    found = {}
    for projection, sym in (("symbol-search", q), ("symbol-regex", re.escape(q) + ".*")):
        try:
            payload = client.get("/marketdata/v1/instruments", symbol=sym, projection=projection) or {}
        except Exception as error:  # noqa: BLE001 - surface as a soft error
            if projection == "symbol-search":
                return {"query": q, "matches": [], "error": f"{type(error).__name__}: {str(error)[:120]}"}
            continue
        items = payload.get("instruments") if isinstance(payload, dict) else payload
        if isinstance(items, dict):
            items = list(items.values())
        for it in items or []:
            s = str(it.get("symbol") or "").upper()
            if s and s not in found and it.get("assetType") in (None, "EQUITY", "ETF", "INDEX", "MUTUAL_FUND", "CLOSED_END_FUND"):
                found[s] = {"symbol": s, "description": it.get("description") or "", "exchange": it.get("exchange") or "",
                            "assetType": it.get("assetType") or ""}
        if len(found) >= 12:
            break
    matches = sorted(found.values(), key=lambda m: (m["symbol"] != q, len(m["symbol"]), m["symbol"]))[:12]
    cache[q] = {"ts": now, "matches": matches}
    if len(cache) > 500:
        for k in sorted(cache, key=lambda k: cache[k].get("ts", 0))[:100]:
            cache.pop(k, None)
    _write_json(SEARCH_CACHE, cache)
    return {"query": q, "matches": matches}


class SymbolLock:
    def __init__(self, symbol):
        self.path = ONDEMAND_DIR / f"{status_path(symbol).stem}.lock"
        self.owned = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            self.owned = True
        except FileExistsError:
            if time.time() - self.path.stat().st_mtime > 600:
                self.path.unlink(missing_ok=True)
                return self.__enter__()
        return self

    def __exit__(self, *exc):
        if self.owned:
            self.path.unlink(missing_ok=True)


def fetch(symbol, client=None, force=False, intraday_days=10, log=None):
    """Fetch max daily history (+ recent 5-minute bars) for ONE symbol into the normal store."""
    import schwab_daily as sd
    log = log or (lambda m: print(f"[ondemand] {m}", file=sys.stderr, flush=True))
    ok, reason = should_fetch(symbol, force)
    if not ok:
        return {**status(symbol), "skipped": reason}
    with SymbolLock(symbol) as lock:
        if not lock.owned:
            return {**status(symbol), "skipped": "another fetch holds the lock"}
        started = time.time()
        _write_json(status_path(symbol), {"status": "running", "startedAt": _now(), "startedTs": started})
        result = {"status": "error", "startedAt": _now(), "startedTs": started}
        try:
            client = client or _client()
            payload, mode, _ = sd.fetch_daily(client, symbol, period_years=sd.DEFAULT_PERIOD_YEARS)
            rows = sd.parse_candles(symbol, payload)
            if not rows:   # some symbols only answer to an explicit start/end window
                payload, mode, _ = sd.fetch_daily(client, symbol, start=date(1990, 1, 1), end=date.today())
                rows = sd.parse_candles(symbol, payload)
            if not rows:
                empty = isinstance(payload, dict) and payload.get("empty") is True
                result.update(status="invalid" if empty else "empty",
                              message="Schwab has no daily history for this symbol (check the symbol).")
            else:
                con = sd.connect_prices(writable=True)
                try:
                    stored = sd.store_daily_rows(con, symbol, rows, mode, status="complete", message="on-demand (chart picker)")
                finally:
                    con.close()
                result.update(status="complete", rows=stored["rows"], first=stored["first"], last=stored["last"],
                              message=f"{stored['rows']} daily bars from Schwab ({stored['first']} to {stored['last']}).")
                add_tracked(symbol)
                if intraday_days:
                    try:
                        import schwab_data
                        cache = schwab_data.connect_cache()
                        try:
                            schwab_data.fetch_history([symbol], days=intraday_days, client=client, cache=cache,
                                                      log=lambda *_: None, frequency=5)
                        finally:
                            cache.close()
                        result["intraday"] = f"5-minute bars for the last {intraday_days} days"
                    except Exception as error:  # noqa: BLE001 - daily history is the important part
                        result["intraday"] = f"skipped ({type(error).__name__})"
                STAMP_PATH.parent.mkdir(parents=True, exist_ok=True)
                STAMP_PATH.write_text(_now(), encoding="utf-8")
        except Exception as error:  # noqa: BLE001
            name = type(error).__name__
            result.update(status="error", message=f"{name}: {str(error)[:160]}")
            log(f"{symbol}: {name}")
        result.update(finishedAt=_now(), finishedTs=time.time(), seconds=round(time.time() - started, 1))
        _write_json(status_path(symbol), result)
    return status(symbol)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["search", "fetch", "status"])
    parser.add_argument("symbol")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "search":
            out = search(args.symbol)
        else:
            sym = clean_symbol(args.symbol)
            out = fetch(sym, force=args.force) if args.command == "fetch" else status(sym)
    except ValueError as error:
        out = {"error": str(error), "status": "invalid"}
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
