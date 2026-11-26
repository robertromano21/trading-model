"""Crypto data for research and the crypto PAPER shadow accounts (no orders, ever).

Sources (all read-only):
  * Coinbase Exchange public daily candles (keyless; UTC days, 24/7) for spot BTC, ETH, SOL, AVAX, LINK and other
    liquid majors -> Massive/Research/crypto/spot/<SYM>.csv. Long history: BTC from 2015-07, ETH from 2016-05.
  * FRED (keyless fredgraph CSV via macro_data.download_fred): Coinbase reference prices (CBBTCUSD, CBETHUSD, ...)
    as a cross-check, plus the macro drivers used by the dollar-feedback study: DTWEXBGS (broad dollar), DFII10
    (10y real yield), WALCL (Fed balance sheet), WTREGEN (Treasury General Account), RRPONTSYD (reverse repo),
    M2SL, VIXCLS -> Massive/Research/crypto/macro_drivers_raw.csv. If the FRED macro import
    (engine/fred_macro.py, Massive/Research/macro/) exists its files are only READ, never written.
  * Schwab market data (schwab_data.SchwabMarketData; market-data endpoints only; account/order paths are refused in code):
    quotes for crypto ETPs, the $BTC/$ETH indexes and CME futures, and daily history for those plus UUP (a
    real-time dollar proxy) -> Massive/Research/crypto/schwab/*.csv and schwab_offering.json.

Release lags: every macro series is shifted to the first UTC day on which it was actually public (see RELEASE_RULES)
before any strategy can use it, so weekly H.10 dollar data and monthly M2 cannot leak into earlier signals.

  python crypto_data.py update [--full] [--no-schwab]
  python crypto_data.py status
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically

CRYPTO_DIR = Path(os.environ.get("CRYPTO_OUT") or DATA_DIR / "Research" / "crypto")
SPOT_DIR = CRYPTO_DIR / "spot"
SCHWAB_DIR = CRYPTO_DIR / "schwab"
MACRO_RAW = CRYPTO_DIR / "macro_drivers_raw.csv"
FRED_CRYPTO = CRYPTO_DIR / "fred_coinbase_prices.csv"
STATUS = CRYPTO_DIR / "data_status.json"
OFFERING = CRYPTO_DIR / "schwab_offering.json"
SHARED_MACRO_DIR = DATA_DIR / "Research" / "macro"        # parallel FRED macro import (read only)
LEGACY_FRED_DIR = DATA_DIR / "Macro" / "fred"               # macro_data.py store (read only)

COINBASE = "https://api.exchange.coinbase.com/products/{product}/candles"
USER_AGENT = "Trading-Model-main/crypto (local research)"

# schwab: "spot" = Schwab Crypto trades it today (BTC/USD, ETH/USD, 0.75% per side); "announced" = Schwab said
# "in the coming months" (Aug 2026); "no" = research breadth only, never traded by the paper accounts' Schwab set.
COINS = {
    "BTC": {"product": "BTC-USD", "schwab": "spot", "etp": ["IBIT", "FBTC", "BITB", "GBTC", "BTC"]},
    "ETH": {"product": "ETH-USD", "schwab": "spot", "etp": ["ETHA", "FETH", "ETHE"]},
    "SOL": {"product": "SOL-USD", "schwab": "announced", "etp": ["BSOL", "GSOL", "SOLZ"]},
    "AVAX": {"product": "AVAX-USD", "schwab": "announced", "etp": []},
    "LINK": {"product": "LINK-USD", "schwab": "announced", "etp": []},
    "LTC": {"product": "LTC-USD", "schwab": "no", "etp": []},
    "BCH": {"product": "BCH-USD", "schwab": "no", "etp": []},
    "XRP": {"product": "XRP-USD", "schwab": "no", "etp": ["XRPR"]},
    "DOGE": {"product": "DOGE-USD", "schwab": "no", "etp": []},
    "ADA": {"product": "ADA-USD", "schwab": "no", "etp": []},
    "DOT": {"product": "DOT-USD", "schwab": "no", "etp": []},
}
SCHWAB_SPOT = [c for c, v in COINS.items() if v["schwab"] == "spot"]
SCHWAB_SET = [c for c, v in COINS.items() if v["schwab"] in ("spot", "announced")]
ALL_COINS = list(COINS)

FRED_CRYPTO_SERIES = ["CBBTCUSD", "CBETHUSD", "CBLTCUSD", "CBBCHUSD"]
MACRO_SERIES = ["DTWEXBGS", "DFII10", "WALCL", "WTREGEN", "RRPONTSYD", "M2SL", "VIXCLS", "DGS2"]
SCHWAB_QUOTE_SYMBOLS = ["$BTC", "$ETH", "/BTC", "/MBT", "/ETH", "/MET", "IBIT", "FBTC", "BITB", "GBTC", "BTC",
                        "ETHA", "FETH", "ETHE", "BITO", "BSOL", "GSOL", "SOLZ", "XRPR", "UUP", "BTC/USD", "ETH/USD"]
SCHWAB_HISTORY_SYMBOLS = ["UUP", "SPY", "QQQ", "IBIT", "FBTC", "GBTC", "ETHA", "ETHE", "BITO", "BSOL", "$BTC", "$ETH",
                          "/BTC", "/ETH"]

# Availability rule per series: the first calendar day a value dated d is public (US time), plus one day so the
# crypto close (00:00 UTC = 19:00 ET) that uses it is unambiguously after the release.
def _next_monday_after_week(d):
    return d + pd.Timedelta(days=7 - d.weekday())


RELEASE_RULES = {
    "DTWEXBGS": lambda d: _next_monday_after_week(d) + pd.Timedelta(days=1),   # H.10, weekly on Mondays
    "DFII10": lambda d: d + pd.offsets.BDay(1) + pd.Timedelta(days=1),          # H.15, next business day
    "DGS2": lambda d: d + pd.offsets.BDay(1) + pd.Timedelta(days=1),
    "VIXCLS": lambda d: d + pd.Timedelta(days=1),
    "WALCL": lambda d: d + pd.Timedelta(days=2),                                # Wed level, H.4.1 Thu 4:30pm
    "WTREGEN": lambda d: d + pd.Timedelta(days=2),
    "RRPONTSYD": lambda d: d + pd.Timedelta(days=1),
    "M2SL": lambda d: d + pd.DateOffset(months=1) + pd.Timedelta(days=27),       # monthly, ~4th week next month
    "UUP": lambda d: d + pd.Timedelta(days=1),                                  # ETF close 16:00 ET, used next day
    "SPY": lambda d: d + pd.Timedelta(days=1),
    "QQQ": lambda d: d + pd.Timedelta(days=1),
}


def log(msg):
    print(f"[crypto_data] {msg}", file=sys.stderr, flush=True)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(payload, indent=2, default=str).encode())


def _write_csv(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, frame.to_csv(index_label="date").encode())


def read_csv(path):
    if not Path(path).is_file():
        return None
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return frame[~frame.index.duplicated(keep="last")]


# ====================================================================== Coinbase spot candles
def _get_json(url, timeout=30):
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(5):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            code = error.code
            error.close()
            if code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {code} for {url.split('?')[0]}") from None
        except (URLError, TimeoutError) as error:
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"network error: {error}") from None


def parse_candles(rows):
    """Coinbase rows [time, low, high, open, close, volume] -> UTC-date indexed OHLCV frame."""
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    frame["date"] = pd.to_datetime(frame["time"], unit="s", utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame.drop(columns="time").drop_duplicates("date").set_index("date").sort_index()
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


def fetch_coinbase_daily(product, start, end=None, sleep=0.15):
    """Complete UTC days only (today's in-progress candle is dropped)."""
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    end = min(pd.Timestamp(end) if end is not None else today, today)
    cursor, frames = pd.Timestamp(start), []
    while cursor < end:
        stop = min(cursor + pd.Timedelta(days=299), end)
        url = (COINBASE.format(product=product) + f"?granularity=86400&start={cursor.strftime('%Y-%m-%dT00:00:00Z')}"
               f"&end={stop.strftime('%Y-%m-%dT00:00:00Z')}")
        frames.append(parse_candles(_get_json(url)))
        cursor = stop + pd.Timedelta(days=1)
        time.sleep(sleep)
    out = pd.concat(frames) if frames else parse_candles([])
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[out.index < today]


def update_spot(full=False, coins=None):
    report = {}
    for sym in coins or ALL_COINS:
        path = SPOT_DIR / f"{sym}.csv"
        old = None if full else read_csv(path)
        start = "2015-01-01" if old is None or old.empty else old.index[-1] - pd.Timedelta(days=5)
        try:
            fresh = fetch_coinbase_daily(COINS[sym]["product"], start)
        except RuntimeError as error:
            report[sym] = {"error": str(error)}
            log(f"{sym}: {error}")
            continue
        merged = fresh if old is None else pd.concat([old, fresh])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        if len(merged):
            _write_csv(path, merged)
        report[sym] = {"rows": int(len(merged)), "start": str(merged.index[0].date()) if len(merged) else None,
                       "end": str(merged.index[-1].date()) if len(merged) else None}
        log(f"{sym}: {report[sym]}")
    return report


# ====================================================================== FRED (keyless)
def _shared_series(series):
    """Read a series from the FRED macro import or macro_data.py's store if present (never written here)."""
    for path in (SHARED_MACRO_DIR / f"{series}.csv", SHARED_MACRO_DIR / "fred" / f"{series}.csv",
                 LEGACY_FRED_DIR / f"{series}.csv"):
        frame = read_csv(path)
        if frame is not None and series in frame.columns:
            return frame[series].dropna()
    return None


def update_fred():
    from macro_data import download_fred
    report = {}
    try:
        fc = download_fred(FRED_CRYPTO_SERIES, start="2014-12-01")
        _write_csv(FRED_CRYPTO, fc)
        report["fredCoinbase"] = {"rows": int(len(fc)), "end": str(fc.index[-1].date())}
    except Exception as error:   # noqa: BLE001
        report["fredCoinbase"] = {"error": f"{type(error).__name__}: {error}"}
    try:
        raw = download_fred(MACRO_SERIES, start="2010-01-01")
        _write_csv(MACRO_RAW, raw)
        report["macro"] = {s: str(raw[s].dropna().index[-1].date()) for s in raw.columns if raw[s].notna().any()}
    except Exception as error:   # noqa: BLE001
        report["macro"] = {"error": f"{type(error).__name__}: {error}"}
        # fall back to series already on disk from the other importers
        cols = {s: _shared_series(s) for s in MACRO_SERIES}
        cols = {k: v for k, v in cols.items() if v is not None}
        if cols and not MACRO_RAW.exists():
            _write_csv(MACRO_RAW, pd.DataFrame(cols))
    return report


# ====================================================================== Schwab market data (read only)
def _candles_frame(payload):
    rows = (payload or {}).get("candles") or []
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    # Schwab daily candles are stamped at 00:00 US/Central of the session date
    frame["date"] = (pd.to_datetime(frame["datetime"], unit="ms", utc=True).dt.tz_convert("America/Chicago")
                     .dt.tz_localize(None).dt.normalize())
    return frame.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)


def update_schwab(client=None):
    """Market-data endpoints only (quotes, pricehistory). Never an account or order call."""
    from schwab_data import SchwabMarketData
    client = client or SchwabMarketData()
    offering = {"checkedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "quotes": {}, "invalidSymbols": [],
                "history": {}}
    q = client.quotes(SCHWAB_QUOTE_SYMBOLS)
    for key, value in (q or {}).items():
        if key == "errors":
            offering["invalidSymbols"] = sorted(set((value or {}).get("invalidSymbols", [])))
            continue
        quote, ref = value.get("quote") or {}, value.get("reference") or {}
        offering["quotes"][key] = {"assetMainType": value.get("assetMainType"),
                                   "assetSubType": value.get("assetSubType"), "description": ref.get("description"),
                                   "last": quote.get("lastPrice"), "close": quote.get("closePrice")}
    for sym in SCHWAB_HISTORY_SYMBOLS:
        try:
            payload = client.get("/marketdata/v1/pricehistory", symbol=sym, periodType="year", period=20,
                                 frequencyType="daily", frequency=1)
            frame = _candles_frame(payload)
            if frame is None:
                offering["history"][sym] = {"rows": 0}
                continue
            safe = sym.replace("$", "IDX_").replace("/", "FUT_")
            _write_csv(SCHWAB_DIR / f"{safe}.csv", frame)
            offering["history"][sym] = {"rows": int(len(frame)), "start": str(frame.index[0].date()),
                                        "end": str(frame.index[-1].date())}
        except Exception as error:   # noqa: BLE001
            offering["history"][sym] = {"error": f"{type(error).__name__}: {error}"}
    offering["summary"] = (
        "Schwab Crypto (separate account at Charles Schwab Premier Bank) trades spot BTC/USD and ETH/USD at 0.75% "
        "of trade value per buy or sell (no markup/spread per Schwab); SOL, AVAX and LINK were announced in Aug 2026 "
        "for 'the coming months'. The Schwab market-data API does NOT quote spot pairs (BTC/USD, ETH/USD are "
        "invalid symbols); it does serve the $BTC/$ETH indexes (history only, quote 0), CME futures (/BTC, /MBT, "
        "/ETH, /MET) and spot ETPs (IBIT, FBTC, GBTC, BITB, ETHA, FETH, ETHE, BSOL, GSOL, XRPR, ...). Spot history "
        "for research therefore comes from Coinbase public candles.")
    write_json(OFFERING, offering)
    return offering


# ====================================================================== loaders used by research / paper
def load_spot(coins=None, field="close"):
    cols = {}
    for sym in coins or ALL_COINS:
        frame = read_csv(SPOT_DIR / f"{sym}.csv")
        if frame is not None and len(frame):
            cols[sym] = frame[field]
    if not cols:
        raise FileNotFoundError(f"No spot candles under {SPOT_DIR}; run `python crypto_data.py update`.")
    panel = pd.DataFrame(cols).sort_index()
    calendar = pd.date_range(panel.index[0], panel.index[-1], freq="D")     # 24/7: every calendar day
    panel = panel.reindex(calendar)
    # a short outage (<= 5 days) carries the last price; longer gaps (e.g. XRP's 2021-23 Coinbase delisting) stay NaN
    first = panel.apply(lambda s: s.first_valid_index())
    for sym in panel.columns:
        panel.loc[first[sym]:, sym] = panel.loc[first[sym]:, sym].ffill(limit=5)
    return panel


def as_available(series, rule, calendar):
    """Re-index a dated series onto `calendar` by its public-availability date (strictly causal) and ffill."""
    s = pd.Series(series).dropna()
    if s.empty:
        return pd.Series(np.nan, index=calendar)
    avail = pd.DatetimeIndex([pd.Timestamp(rule(pd.Timestamp(d))).normalize() for d in s.index])
    out = pd.Series(s.values, index=avail)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.reindex(out.index.union(calendar)).ffill().reindex(calendar)


def load_schwab_close(sym):
    safe = sym.replace("$", "IDX_").replace("/", "FUT_")
    frame = read_csv(SCHWAB_DIR / f"{safe}.csv")
    return None if frame is None else frame["close"]


def load_drivers(calendar):
    """Daily (calendar-day) macro driver panel, every column lagged to its public-availability date."""
    raw = read_csv(MACRO_RAW)
    cols = {}
    for series in MACRO_SERIES:
        s = raw[series] if raw is not None and series in raw.columns else _shared_series(series)
        if s is not None:
            cols[series] = as_available(s, RELEASE_RULES[series], calendar)
    for sym in ("UUP", "SPY", "QQQ"):
        s = load_schwab_close(sym)
        if s is not None:
            cols[sym] = as_available(s, RELEASE_RULES[sym], calendar)
    out = pd.DataFrame(cols, index=calendar)
    if {"WALCL", "WTREGEN", "RRPONTSYD"} <= set(out.columns):
        # net liquidity in $bn: Fed assets - TGA - overnight reverse repo (WALCL/WTREGEN in $mn, RRP in $bn)
        out["NETLIQ"] = out["WALCL"] / 1000 - out["WTREGEN"] / 1000 - out["RRPONTSYD"].fillna(0)
    return out


def status():
    doc = {"spot": {}, "schwab": {}, "macro": None}
    for sym in ALL_COINS:
        f = read_csv(SPOT_DIR / f"{sym}.csv")
        if f is not None and len(f):
            doc["spot"][sym] = {"start": str(f.index[0].date()), "end": str(f.index[-1].date()), "rows": len(f),
                                "schwab": COINS[sym]["schwab"]}
    if SCHWAB_DIR.exists():
        for p in sorted(SCHWAB_DIR.glob("*.csv")):
            f = read_csv(p)
            doc["schwab"][p.stem] = {"start": str(f.index[0].date()), "end": str(f.index[-1].date()), "rows": len(f)}
    raw = read_csv(MACRO_RAW)
    if raw is not None:
        doc["macro"] = {c: {"start": str(raw[c].dropna().index[0].date()), "end": str(raw[c].dropna().index[-1].date())}
                        for c in raw.columns if raw[c].notna().any()}
    return doc


def update(full=False, schwab=True):
    started = time.time()
    report = {"updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "spot": update_spot(full)}
    report.update(update_fred())
    if schwab:
        try:
            off = update_schwab()
            report["schwab"] = {"invalidSymbols": off["invalidSymbols"], "history": off["history"]}
        except Exception as error:   # noqa: BLE001 - Schwab auth may be expired; research still works
            report["schwab"] = {"error": f"{type(error).__name__}: {error}"}
    report["seconds"] = round(time.time() - started, 1)
    report["coverage"] = status()
    write_json(STATUS, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["update", "status"])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--no-schwab", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "update":
        rep = update(args.full, not args.no_schwab)
        print(json.dumps({"spot": {k: v.get("end") or v.get("error") for k, v in rep["spot"].items()},
                          "seconds": rep["seconds"]}))
    else:
        print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
