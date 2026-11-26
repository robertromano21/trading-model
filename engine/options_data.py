"""Option chains from Schwab's market-data API (read-only) plus a local daily snapshot store, and a MODELED chain for
dates/tickers without real quotes.

Endpoint: GET https://api.schwabapi.com/marketdata/v1/chains (the same endpoint schwab-py's Client.get_option_chain
uses; params symbol, contractType, strikeCount, includeUnderlyingQuote, strategy, range, fromDate, toDate). Requests
go through schwab_data.SchwabMarketData.get, which allow-lists market-data URLs only. This module has no account or
order code. When the Schwab token is expired every call raises SchwabAuthError with the reauthorize instructions; the
dashboard shows its "click to reauthorize" popup and modeled chains keep working.

Snapshots: Massive/Options/chains.sqlite3 (gitignored), one row per contract per snapshot, so a real quote history
builds up going forward. Run daily (after the close) with:

  python options_data.py snapshot [SYMBOLS]            # default: SPY QQQ IWM + paper-book underlyings
  python options_data.py chain SPY [--strikes 20] [--from 2026-12-01 --to 2027-01-31] [--modeled]
  python options_data.py history SPY                    # what has been stored
"""

import argparse
import json
import math
import sqlite3
import sys
from datetime import date, datetime, timedelta

from fetch_and_model import DATA_DIR
from options_pricing import (ModeledIV, OptionSpreadModel, base_vol_forecast, bsm_greeks, occ_symbol, year_fraction)

OPTIONS_DIR = DATA_DIR / "Options"
CHAIN_DB = OPTIONS_DIR / "chains.sqlite3"
CHAINS_PATH = "/marketdata/v1/chains"
DEFAULT_SNAPSHOT = ("SPY", "QQQ", "IWM")
ETF_LIKE = {"SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "GLD", "XLF", "XLK", "XLE", "SMH", "HYG", "EEM", "EFA", "SH", "PSQ", "SDS"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_quotes (
  snapshot_at TEXT NOT NULL, session_date TEXT NOT NULL, underlying TEXT NOT NULL, symbol TEXT NOT NULL,
  kind TEXT NOT NULL, expiration TEXT NOT NULL, dte INTEGER, strike REAL NOT NULL,
  bid REAL, ask REAL, last REAL, mark REAL, volume REAL, open_interest REAL, iv REAL,
  delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL, underlying_price REAL, source TEXT NOT NULL DEFAULT 'schwab',
  PRIMARY KEY (snapshot_at, symbol)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS chain_quotes_day ON chain_quotes (underlying, session_date);
CREATE TABLE IF NOT EXISTS chain_snapshots (
  snapshot_at TEXT NOT NULL, underlying TEXT NOT NULL, contracts INTEGER, underlying_price REAL, status TEXT,
  PRIMARY KEY (snapshot_at, underlying));
"""


def connect(path=CHAIN_DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(v) or v in (-999.0, -999) else v


def fetch_chain(symbol, client=None, strike_count=None, from_date=None, to_date=None, contract_type="ALL",
                strike_range=None):
    """Raw Schwab chain JSON for one underlying (market data only)."""
    from schwab_data import SchwabMarketData
    client = client or SchwabMarketData()
    params = {"symbol": symbol.upper(), "contractType": contract_type, "includeUnderlyingQuote": "true",
              "strategy": "SINGLE", "strikeCount": strike_count, "range": strike_range,
              "fromDate": from_date, "toDate": to_date}
    return client.get(CHAINS_PATH, **params)


def normalize_chain(raw):
    """Flatten Schwab's callExpDateMap / putExpDateMap into rows (iv as a fraction, -999 greeks -> None)."""
    raw = raw or {}
    underlying = str(raw.get("symbol") or "").upper()
    spot = _f(raw.get("underlyingPrice")) or _f((raw.get("underlying") or {}).get("last")) or _f((raw.get("underlying") or {}).get("mark"))
    rows = []
    for key, kind in (("callExpDateMap", "call"), ("putExpDateMap", "put")):
        for exp_key, strikes in (raw.get(key) or {}).items():
            for strike_key, contracts in (strikes or {}).items():
                for c in contracts or []:
                    exp = str(c.get("expirationDate") or exp_key.split(":")[0])[:10]
                    iv = _f(c.get("volatility"))
                    rows.append({
                        "underlying": underlying, "symbol": c.get("symbol"), "kind": kind, "expiration": exp,
                        "dte": c.get("daysToExpiration"), "strike": _f(c.get("strikePrice")) or _f(strike_key),
                        "bid": _f(c.get("bid")), "ask": _f(c.get("ask")), "last": _f(c.get("last")), "mark": _f(c.get("mark")),
                        "volume": _f(c.get("totalVolume")), "openInterest": _f(c.get("openInterest")),
                        "iv": iv / 100.0 if iv is not None else None,
                        "delta": _f(c.get("delta")), "gamma": _f(c.get("gamma")), "theta": _f(c.get("theta")),
                        "vega": _f(c.get("vega")), "rho": _f(c.get("rho")), "inTheMoney": c.get("inTheMoney"),
                        "nonStandard": c.get("nonStandard"), "multiplier": _f(c.get("multiplier")) or 100.0,
                        "exerciseType": c.get("exerciseType"), "settlementType": c.get("settlementType"),
                    })
    rows.sort(key=lambda r: (r["expiration"], r["strike"] or 0, r["kind"]))
    return {"underlying": underlying, "underlyingPrice": spot, "status": raw.get("status"),
            "isDelayed": raw.get("isDelayed"), "interestRate": _f(raw.get("interestRate")),
            "source": "schwab", "pricing": "quoted", "contracts": rows,
            "expirations": sorted({r["expiration"] for r in rows})}


def store_snapshot(con, chain, snapshot_at=None):
    stamp = snapshot_at or datetime.now().isoformat(timespec="seconds")
    day = stamp[:10]
    rows = [(stamp, day, chain["underlying"], r["symbol"], r["kind"], r["expiration"], r["dte"], r["strike"], r["bid"],
             r["ask"], r["last"], r["mark"], r["volume"], r["openInterest"], r["iv"], r["delta"], r["gamma"], r["theta"],
             r["vega"], r["rho"], chain["underlyingPrice"], chain.get("source", "schwab"))
            for r in chain["contracts"] if r.get("symbol")]
    with con:
        con.executemany(f"INSERT OR REPLACE INTO chain_quotes VALUES ({','.join('?' * 22)})", rows)
        con.execute("INSERT OR REPLACE INTO chain_snapshots VALUES (?, ?, ?, ?, ?)",
                    (stamp, chain["underlying"], len(rows), chain["underlyingPrice"], chain.get("status")))
    return len(rows)


def stored_chain(con, underlying, session_date=None):
    """Latest stored snapshot for an underlying (optionally on a given date) -> normalized chain or None."""
    q = "SELECT snapshot_at FROM chain_snapshots WHERE underlying=?" + (" AND substr(snapshot_at,1,10)<=?" if session_date else "") + \
        " ORDER BY snapshot_at DESC LIMIT 1"
    row = con.execute(q, (underlying.upper(), session_date) if session_date else (underlying.upper(),)).fetchone()
    if not row:
        return None
    cols = ["snapshot_at", "session_date", "underlying", "symbol", "kind", "expiration", "dte", "strike", "bid", "ask",
            "last", "mark", "volume", "openInterest", "iv", "delta", "gamma", "theta", "vega", "rho", "underlyingPrice", "source"]
    data = [dict(zip(cols, r)) for r in con.execute("SELECT * FROM chain_quotes WHERE snapshot_at=? AND underlying=?",
                                                     (row[0], underlying.upper()))]
    spot = data[0]["underlyingPrice"] if data else None
    return {"underlying": underlying.upper(), "underlyingPrice": spot, "snapshotAt": row[0], "source": "snapshot",
            "pricing": "quoted", "contracts": data, "expirations": sorted({d["expiration"] for d in data})}


# ---------------------------------------------------------------------------------------------- modeled chain
def standard_expirations(asof, count=8):
    """Weekly Fridays for 5 weeks, then monthly third Fridays (approximation of a listed calendar)."""
    d = datetime.strptime(str(asof)[:10], "%Y-%m-%d").date()
    out = []
    f = d + timedelta(days=(4 - d.weekday()) % 7 or 7)
    for _ in range(5):
        out.append(f)
        f += timedelta(days=7)
    m = date(d.year, d.month, 1)
    while len(out) < count:
        m = date(m.year + (m.month // 12), m.month % 12 + 1, 1)
        third = m + timedelta(days=(4 - m.weekday()) % 7 + 14)
        if third > out[-1]:
            out.append(third)
    return [x.isoformat() for x in out[:count]]


def modeled_chain(underlying, spot, asof, returns, vix_r20=None, strikes_each_side=10, expirations=None,
                  rate=0.04, div_yield=0.0, iv_model=None, spread_model=None, vol_method="blend"):
    """A synthetic chain priced with Black-Scholes on the ModeledIV surface. Labeled pricing='modeled'."""
    iv_model = iv_model or ModeledIV()
    spread_model = spread_model or OptionSpreadModel()
    is_etf = underlying.upper() in ETF_LIKE
    base = base_vol_forecast(returns, vol_method) or 0.25
    atm = iv_model.atm(base, vix_r20)
    step = 0.5 if spot < 25 else 1.0 if spot < 200 else 5.0
    center = round(spot / step) * step
    strikes = [center + i * step for i in range(-strikes_each_side, strikes_each_side + 1) if center + i * step > 0]
    rows = []
    for exp in expirations or standard_expirations(asof):
        t = max(year_fraction(asof, exp), 1 / 365)
        for k in strikes:
            iv = iv_model.iv(atm, spot, k, t, is_etf)
            for kind in ("call", "put"):
                g = bsm_greeks(spot, k, t, iv, rate, div_yield, kind)
                bid, ask = spread_model.quote(g["price"], g["delta"], t * 365, is_etf)
                rows.append({"underlying": underlying.upper(), "symbol": occ_symbol(underlying, exp, kind, k), "kind": kind,
                             "expiration": exp, "dte": round(t * 365), "strike": k, "bid": bid, "ask": ask,
                             "mark": round(g["price"], 4), "last": None, "volume": None, "openInterest": None, "iv": iv,
                             "delta": g["delta"], "gamma": g["gamma"], "theta": g["theta"], "vega": g["vega"], "rho": g["rho"]})
    return {"underlying": underlying.upper(), "underlyingPrice": spot, "asof": str(asof)[:10], "source": "model",
            "pricing": "modeled", "atmIv": atm, "baseVol": base, "volMethod": vol_method,
            "note": "MODELED chain (Black-Scholes on a realized/GARCH vol forecast with skew and spread models); not quotes.",
            "contracts": rows, "expirations": sorted({r["expiration"] for r in rows})}


def local_history(underlying, lookback=260):
    """(spot, asof, daily returns, VIXY 20d log return) from the local price database."""
    from quant_tools import fast_panel
    panel = fast_panel([underlying.upper(), "VIXY"])
    close = panel.close
    if underlying.upper() not in close.columns:
        raise SystemExit(f"no local prices for {underlying}")
    s = close[underlying.upper()].dropna()
    r = panel.total_return()[underlying.upper()].dropna().tail(lookback)
    vix = None
    if "VIXY" in close.columns:
        v = close["VIXY"].dropna()
        if len(v) > 21:
            vix = float(math.log(v.iloc[-1] / v.iloc[-21]))
    return float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d"), r.tolist(), vix


def get_chain(symbol, prefer="schwab", strike_count=20, from_date=None, to_date=None, store=True):
    """Best available chain: live Schwab (stored as a snapshot) -> latest local snapshot -> modeled."""
    symbol = symbol.upper()
    errors = []
    if prefer in ("schwab", "auto"):
        try:
            chain = normalize_chain(fetch_chain(symbol, strike_count=strike_count, from_date=from_date, to_date=to_date))
            if chain["contracts"]:
                if store:
                    with connect() as con:
                        store_snapshot(con, chain)
                return chain
            errors.append("Schwab returned an empty chain")
        except Exception as error:  # noqa: BLE001 - fall back; message is safe (no secrets in SchwabAuthError text)
            errors.append(f"{type(error).__name__}: {str(error)[:300]}")
    if prefer != "model" and CHAIN_DB.exists():
        with connect() as con:
            chain = stored_chain(con, symbol)
        if chain:
            chain["fallbackReason"] = errors
            return chain
    spot, asof, returns, vix = local_history(symbol)
    chain = modeled_chain(symbol, spot, asof, returns, vix)
    chain["fallbackReason"] = errors
    return chain


def paper_underlyings():
    try:
        from options_book import load_book
        return sorted({p["underlying"] for p in load_book().get("positions", [])})
    except Exception:  # noqa: BLE001
        return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["snapshot", "chain", "history"])
    ap.add_argument("symbols", nargs="*")
    ap.add_argument("--strikes", type=int, default=20)
    ap.add_argument("--from", dest="from_date")
    ap.add_argument("--to", dest="to_date")
    ap.add_argument("--modeled", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max contracts in the printed chain (0 = all)")
    a = ap.parse_args(argv)
    if a.command == "snapshot":
        from schwab_data import SchwabAuthError
        symbols = a.symbols or sorted(set(DEFAULT_SNAPSHOT) | set(paper_underlyings()))
        out = {}
        try:
            with connect() as con:
                for s in symbols:
                    chain = normalize_chain(fetch_chain(s, strike_count=a.strikes))
                    out[s] = store_snapshot(con, chain)
        except SchwabAuthError as error:
            print(json.dumps({"status": "auth_required", "message": str(error), "stored": out}))
            return 2
        print(json.dumps({"status": "ok", "stored": out, "db": str(CHAIN_DB)}))
        return 0
    if a.command == "history":
        with connect() as con:
            rows = con.execute("SELECT underlying, COUNT(*), MIN(snapshot_at), MAX(snapshot_at), SUM(contracts) FROM chain_snapshots "
                               + ("WHERE underlying IN (%s) " % ",".join("?" * len(a.symbols)) if a.symbols else "")
                               + "GROUP BY underlying", [s.upper() for s in a.symbols]).fetchall()
        print(json.dumps([dict(zip(("underlying", "snapshots", "first", "last", "contracts"), r)) for r in rows]))
        return 0
    if not a.symbols:
        ap.error("chain needs a symbol")
    chain = get_chain(a.symbols[0], prefer="model" if a.modeled else "auto", strike_count=a.strikes,
                      from_date=a.from_date, to_date=a.to_date)
    if a.limit:
        chain["contracts"] = chain["contracts"][: a.limit]
    sys.stdout.write(json.dumps(chain, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
