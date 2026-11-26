"""Long daily index history from Schwab market data (read-only client), back to 1970.

Schwab's /marketdata/v1/pricehistory only honours startDate before the 20-year window when
periodType=year is also sent (without it the API returns HTTP 400). Epoch (1970-01-01) is the floor.
Rows go to Massive/Macro/index_long/<NAME>.csv (date, open, high, low, close, volume). Price indices:
no dividends. This module only calls the market-data allow-list in schwab_data.SchwabMarketData.

Usage (from engine/): python index_long.py update [--symbols $SPX,$DJI]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically

ET = ZoneInfo("America/New_York")
INDEX_DIR = DATA_DIR / "Macro" / "index_long"
SYMBOLS = ("$SPX", "$DJI", "$COMPX", "$RUT", "$VIX", "$NDX", "$OEX")
FLOOR = date(1970, 1, 2)


def safe_name(symbol):
    return symbol.replace("$", "").replace(".", "_").upper()


def path_for(symbol, root=None):
    return Path(root or INDEX_DIR) / f"{safe_name(symbol)}.csv"


def _ms(d, end=False):
    dt = datetime(d.year, d.month, d.day, 23 if end else 0, 59 if end else 0, 59 if end else 0, tzinfo=ET)
    return int(dt.timestamp() * 1000)


def params(symbol, start, end):
    return {"symbol": symbol, "periodType": "year", "frequencyType": "daily", "frequency": 1,
            "startDate": _ms(start), "endDate": _ms(end, True), "needExtendedHoursData": "false"}


def parse(payload):
    rows = []
    for c in (payload or {}).get("candles") or []:
        try:
            day = datetime.fromtimestamp(int(c["datetime"]) / 1000, tz=ET).date()
            rows.append({"date": pd.Timestamp(day), "open": float(c["open"]), "high": float(c["high"]),
                         "low": float(c["low"]), "close": float(c["close"]), "volume": float(c.get("volume") or 0)})
        except (KeyError, TypeError, ValueError):
            continue
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    f = pd.DataFrame(rows).drop_duplicates("date", keep="last").set_index("date").sort_index()
    return f


def load(symbol, root=None):
    p = path_for(symbol, root)
    if not p.is_file():
        return None
    return pd.read_csv(p, index_col=0, parse_dates=True).sort_index()


def update(symbols=SYMBOLS, client=None, root=None, full=False, today=None, log=print):
    if client is None:
        from schwab_data import SchwabMarketData
        client = SchwabMarketData()
    today = today or date.today()
    out = {}
    for sym in symbols:
        try:
            old = None if full else load(sym, root)
            start = FLOOR if old is None or old.empty else (old.index.max() - pd.Timedelta(days=10)).date()
            fresh = parse(client.get("/marketdata/v1/pricehistory", **params(sym, start, today)))
            merged = fresh if old is None or old.empty else pd.concat([old, fresh])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            if merged.empty:
                out[sym] = {"rows": 0, "status": "empty"}
                continue
            p = path_for(sym, root)
            p.parent.mkdir(parents=True, exist_ok=True)
            write_atomically(p, merged.to_csv(date_format="%Y-%m-%d").encode("utf-8"))
            out[sym] = {"rows": int(len(merged)), "first": str(merged.index.min().date()),
                        "last": str(merged.index.max().date()), "status": "ok"}
        except Exception as exc:  # noqa: BLE001
            out[sym] = {"status": "error", "error": str(exc)[:160]}
        if log:
            log(f"{sym:8s} {out[sym]}")
    meta = Path(root or INDEX_DIR) / "manifest.json"
    meta.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(meta, json.dumps({"generatedAt": datetime.now().isoformat(timespec="seconds"),
                                       "note": "Price indices (no dividends). Schwab market data.",
                                       "symbols": out}, indent=2).encode("utf-8"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--symbols", default="")
    u.add_argument("--full", action="store_true")
    a = ap.parse_args(argv)
    syms = [s.strip() for s in a.symbols.split(",") if s.strip()] or SYMBOLS
    update(syms, full=a.full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
