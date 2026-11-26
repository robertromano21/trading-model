"""Event study and playbook calibration for the black swan monitor.

Uses the longest lawful price history available, in this order:
  1. Massive/prices.sqlite3 daily_prices (split- and dividend-adjusted)
  2. A GB-16 Schwab long table (`daily_long`, or legacy names) in
     prices.sqlite3, or Massive/Research/black_swan/schwab_long.csv — fills
     dates before Massive (Schwab candles are split-adjusted only)
  3. FRED public index series (SP500, VIXCLS, gold fix, WTI, DGS10, DEXJPUS,
     DEXSZUS) — lawful, no key, already used elsewhere in this project
  4. Optional Stooq daily ETF bars (SPY, sector ETFs, TLT, GLD, USO, HYG, LQD),
     cached under Massive/Research/black_swan/stooq_*.csv, personal-research use

The FRED series are *indices*, not the ETFs. They omit dividends (SP500 is a
price index), tracking error, and — for USO vs WTI — futures roll yield. Delisted
names never appear (survivorship). Few events means wide error bars; we say so.

  python black_swan_study.py
  python black_swan.py study
"""
from __future__ import annotations

import io
import json
import math
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, DATABASE, write_atomically
from black_swan import (
    DEFAULT_THRESHOLDS, confirmation_score, iso_et, judge, load_thresholds,
    out_dir, save_json, step_state,
)
from black_swan_playbook import policy_for

OUT_DIR = DATA_DIR / "Research" / "black_swan"
STUDY_PATH = OUT_DIR / "study.json"
WF_PATH = OUT_DIR / "walkforward.json"
THRESHOLDS_PATH = OUT_DIR / "thresholds.json"
UA = "trading-model/1.0 (personal research; black-swan event study)"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd=1990-01-01"
STOOQ = "https://stooq.com/q/d/l/?s={sym}&i=d"

# Event date = when the shock became public. first_session = first NYSE session
# that could trade it (next open when the event was after the close or on a
# weekend / market holiday). Forward returns are measured from the close of
# the session *before* first_session (so the event-day close-to-close is
# horizon 1 when first_session is the event day).
EVENTS = (
    {"id": "sept11", "name": "9/11 attacks", "eventDate": "2001-09-11",
     "firstSession": "2001-09-17", "direction": "negative", "type": "terror",
     "note": "NYSE closed 11–14 Sep 2001; first session was Monday 17 Sep."},
    {"id": "iraq_2003", "name": "Iraq invasion", "eventDate": "2003-03-20",
     "firstSession": "2003-03-20", "direction": "negative", "type": "war",
     "note": "Invasion began in the evening of 19 Mar US time; 20 Mar was the first full session."},
    {"id": "lehman", "name": "Lehman bankruptcy", "eventDate": "2008-09-15",
     "firstSession": "2008-09-15", "direction": "negative", "type": "financial_contagion",
     "note": "Lehman filed early on Monday 15 Sep 2008."},
    {"id": "flash_crash", "name": "Flash crash", "eventDate": "2010-05-06",
     "firstSession": "2010-05-06", "direction": "negative", "type": "financial_contagion",
     "note": "Intraday event; close-to-close understates the intra-day drawdown."},
    {"id": "us_downgrade", "name": "US S&P downgrade", "eventDate": "2011-08-05",
     "firstSession": "2011-08-08", "direction": "negative", "type": "sovereign_default",
     "note": "S&P downgraded after the Friday close; first session was Monday 8 Aug."},
    {"id": "brexit", "name": "Brexit referendum", "eventDate": "2016-06-23",
     "firstSession": "2016-06-24", "direction": "negative", "type": "other",
     "note": "Leave result came after the UK polls closed; first US session was 24 Jun."},
    {"id": "volmageddon", "name": "Volmageddon", "eventDate": "2018-02-05",
     "firstSession": "2018-02-05", "direction": "negative", "type": "financial_contagion",
     "note": "VIX spike and short-vol ETF collapse on 5 Feb 2018."},
    {"id": "covid", "name": "COVID crash start", "eventDate": "2020-02-24",
     "firstSession": "2020-02-24", "direction": "negative", "type": "pandemic",
     "note": "First large risk-off session of the COVID crash (peak was 19 Feb). 11–16 Mar were later climax days."},
    {"id": "covid_climax", "name": "COVID circuit-breaker week", "eventDate": "2020-03-16",
     "firstSession": "2020-03-16", "direction": "negative", "type": "pandemic",
     "note": "Fed emergency cut and a NYSE level-1/2 circuit breaker; included as a second COVID date."},
    {"id": "ukraine", "name": "Russia–Ukraine invasion", "eventDate": "2022-02-24",
     "firstSession": "2022-02-24", "direction": "negative", "type": "war",
     "note": "Full-scale invasion began in the early hours of 24 Feb 2022."},
    {"id": "svb", "name": "SVB failure", "eventDate": "2023-03-10",
     "firstSession": "2023-03-10", "direction": "negative", "type": "financial_contagion",
     "note": "SVB closed by regulators on Friday 10 Mar 2023."},
    {"id": "israel_hamas", "name": "Israel–Hamas war", "eventDate": "2023-10-07",
     "firstSession": "2023-10-09", "direction": "negative", "type": "war",
     "note": "Attacks were on Saturday 7 Oct; first US session was Monday 9 Oct."},
    {"id": "yen_carry", "name": "Yen carry unwind", "eventDate": "2024-08-05",
     "firstSession": "2024-08-05", "direction": "negative", "type": "financial_contagion",
     "note": "Climax of the early-August 2024 yen-carry / Nikkei unwind (Monday 5 Aug)."},
    {"id": "tariff_2025", "name": "April 2025 tariff shock", "eventDate": "2025-04-02",
     "firstSession": "2025-04-03", "direction": "negative", "type": "trade_war",
     "note": "Broad tariff announcement on 2 Apr 2025; the large US session move was 3 Apr."},
)

# Horizons in trading sessions from the close before first_session.
HORIZONS = (1, 5, 10, 20, 60)
ASSETS = ("SPY", "XLF", "XLE", "XLK", "XLU", "TLT", "GLD", "USO", "VIX")
FRED_MAP = {
    "SP500": "SP500",
    "VIXCLS": "VIX",
    "GOLDAMGBD228NLBM": "GLD",
    "GOLDPMGBD228NLBM": "GLD",
    "DCOILWTICO": "USO",
    "DGS10": "DGS10",
    "DEXJPUS": "USDJPY",
    "DEXSZUS": "USDCHF",
}
STOOQ_SYMS = {
    "SPY": "spy.us", "TLT": "tlt.us", "IEF": "ief.us", "GLD": "gld.us", "USO": "uso.us",
    "XLF": "xlf.us", "XLE": "xle.us", "XLK": "xlk.us", "XLU": "xlu.us",
    "HYG": "hyg.us", "LQD": "lqd.us", "SH": "sh.us", "FXY": "fxy.us",
}
LONG_TABLES = ("daily_long", "schwab_long_prices", "long_daily_prices", "schwab_daily_long")


def _http_get(url, timeout=60):
    req = Request(url, headers={"User-Agent": UA, "Accept": "text/csv,text/plain,*/*"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _http_get_optional(url, timeout=60):
    try:
        return _http_get(url, timeout=timeout)
    except Exception:
        return None


def _parse_fred(text, series_id):
    rows = {}
    for row in (r for r in text.splitlines()[1:] if r.strip()):
        parts = row.split(",")
        if len(parts) < 2:
            continue
        day, raw = parts[0].strip(), parts[-1].strip()
        try:
            val = float(raw)
        except ValueError:
            continue
        if day:
            rows[day] = val
    return rows


def fetch_fred_series(series_id, cache_dir, use_network=True):
    path = Path(cache_dir) / f"fred_{series_id}.csv"
    # Always refresh SP500 when the cache looks 10-year truncated (FRED's graph CSV
    # sometimes ignores a long cosd and returns only the last decade).
    cached_ok = path.is_file() and (time.time() - path.stat().st_mtime) < 24 * 3600
    if cached_ok:
        rows = _parse_fred(path.read_text(encoding="utf-8"), series_id)
        if series_id != "SP500" or (rows and min(rows) <= "1995-01-01"):
            return rows
    if not use_network:
        if path.is_file():
            return _parse_fred(path.read_text(encoding="utf-8"), series_id)
        return {}
    windows = [("1990-01-01", "1999-12-31"), ("2000-01-01", "2009-12-31"),
               ("2010-01-01", "2019-12-31"), ("2020-01-01", None)]
    merged = {}
    # One un-chunked try first (works for VIX, WTI, FX, yields).
    text = _http_get_optional(FRED.format(id=series_id))
    if text and ("DATE" in text[:80].upper() or "OBSERVATION" in text[:200].upper()):
        merged.update(_parse_fred(text, series_id))
        time.sleep(0.4)
    if series_id == "SP500" or (merged and min(merged) > "1995-01-01"):
        for start, end in windows:
            url = FRED.format(id=series_id) if end is None else (
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd={s}&coed={e}"
                .format(id=series_id, s=start, e=end))
            text = _http_get_optional(url)
            if text:
                merged.update(_parse_fred(text, series_id))
            time.sleep(0.4)
    if not merged:
        return {}
    lines = ["date,{0}".format(series_id)] + ["{0},{1}".format(d, merged[d]) for d in sorted(merged)]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, ("\n".join(lines) + "\n").encode("utf-8"))
    return merged


def fetch_stooq(ticker, cache_dir, use_network=True):
    sym = STOOQ_SYMS.get(ticker)
    if not sym:
        return None
    path = Path(cache_dir) / f"stooq_{ticker}.csv"
    if path.is_file() and (time.time() - path.stat().st_mtime) < 24 * 3600:
        try:
            return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        except Exception:
            return None
    if not use_network:
        if path.is_file():
            try:
                return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
            except Exception:
                return None
        return None
    try:
        text = _http_get_optional(STOOQ.format(sym=sym))
        time.sleep(2.0)
    except Exception:
        return None
    if not text:
        return None
    if "Date" not in text[:80] and "date" not in text[:80].lower():
        return None
    try:
        frame = pd.read_csv(io.StringIO(text))
    except Exception:
        return None
    frame.columns = [c.strip().lower() for c in frame.columns]
    if "date" not in frame.columns or "close" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reset_index()
    out.to_csv(path, index=False)
    return frame


def _sqlite_tables(db):
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            return [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    except Exception:
        return []


def _load_sqlite_prices(db, table, tickers):
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        cols = [r[1] for r in con.execute(f"PRAGMA table_info({table})")]
        date_col = "date" if "date" in cols else ("market_date" if "market_date" in cols else cols[0])
        px_col = "close" if "close" in cols else ("adj_close" if "adj_close" in cols else "px")
        tkr_col = "ticker" if "ticker" in cols else "symbol"
        q = f"SELECT {tkr_col} AS ticker, {date_col} AS date, {px_col} AS close FROM {table}"
        frame = pd.read_sql_query(q, con)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    if tickers:
        frame = frame[frame.ticker.isin(set(tickers) | {"SPY", "VIX"})]
    if frame.empty:
        return pd.DataFrame()
    return frame.pivot(index="date", columns="ticker", values="close").sort_index().astype(float)



def _return_splice(index_levels, etf_levels=None):
    """Rebuild a level series from index returns, switching to ETF returns when both exist."""
    idx = index_levels.dropna().sort_index()
    etf = etf_levels.dropna().sort_index() if etf_levels is not None else pd.Series(dtype=float)
    dates = idx.index.union(etf.index).sort_values()
    if not len(dates):
        return None
    out = []
    last = None
    for d in dates:
        has_i = d in idx.index
        has_e = d in etf.index
        prev = dates[dates.get_loc(d) - 1] if len(out) else None
        if last is None:
            if has_e:
                last = float(etf.loc[d]); out.append((d, last)); continue
            if has_i:
                last = float(idx.loc[d]); out.append((d, last)); continue
            continue
        ret = None
        if has_e and prev is not None and prev in etf.index and float(etf.loc[prev]):
            ret = float(etf.loc[d]) / float(etf.loc[prev]) - 1.0
        elif has_i and prev is not None and prev in idx.index and float(idx.loc[prev]):
            ret = float(idx.loc[d]) / float(idx.loc[prev]) - 1.0
        if ret is None:
            out.append((d, last)); continue
        last = last * (1.0 + ret)
        out.append((d, last))
    if not out:
        return None
    return pd.Series({d: v for d, v in out}, dtype=float)


def load_study_prices(tickers=None, use_network=True, database=DATABASE, cache_dir=None):
    """Return (close_df, coverage_notes). close_df columns are canonical asset names."""
    tickers = list(tickers or ASSETS)
    cache_dir = Path(cache_dir) if cache_dir else out_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pieces = []
    notes = []

    tables = _sqlite_tables(database)

    # 1. Massive daily_prices first so dividend-adjusted closes win on overlap.
    if "daily_prices" in tables:
        try:
            daily = _load_sqlite_prices(database, "daily_prices", list(STOOQ_SYMS) + ["SPY", "VIXY", "SVIX"])
            pieces.append(daily)
            if len(daily):
                notes.append(f"Massive daily_prices covers {daily.index.min().date()} to {daily.index.max().date()} "
                             f"({len(daily)} sessions); earlier dates come from Schwab daily_long or FRED proxies.")
        except Exception as exc:
            notes.append(f"daily_prices unreadable: {exc}.")

    # 2. GB-16 long table or cached long CSV (split-adjusted, fills pre-Massive dates).
    long_csv = cache_dir / "schwab_long.csv"
    long_table = next((t for t in LONG_TABLES if t in tables), None)
    if long_table:
        try:
            pieces.append(_load_sqlite_prices(database, long_table, tickers))
            notes.append(f"Used Schwab long table `{long_table}` in prices.sqlite3 "
                         "(split-adjusted, not dividend-adjusted; delisted names are missing).")
        except Exception as exc:
            notes.append(f"Schwab long table `{long_table}` present but unreadable: {exc}.")
    elif long_csv.is_file():
        try:
            frame = pd.read_csv(long_csv, parse_dates=["date"]).set_index("date")
            pieces.append(frame)
            notes.append("Used Massive/Research/black_swan/schwab_long.csv.")
        except Exception as exc:
            notes.append(f"schwab_long.csv unreadable: {exc}.")
    else:
        notes.append("GB-16 Schwab long table is not populated yet.")

    # 3. Local VIX file (FRED already, from 2010).
    vix_path = DATA_DIR / "Options" / "vix_daily.csv"
    if vix_path.is_file():
        try:
            vix = pd.read_csv(vix_path, parse_dates=["date"]).set_index("date").sort_index()
            if "vix" in vix.columns:
                pieces.append(vix.rename(columns={"vix": "VIX"})[["VIX"]])
                notes.append(f"Local VIX file {vix.index.min().date()} to {vix.index.max().date()}.")
        except Exception:
            pass

    # 4. FRED index proxies (lawful, long history).
    if use_network or any((cache_dir / f"fred_{s}.csv").is_file() for s in FRED_MAP):
        for series, asset in FRED_MAP.items():
            rows = fetch_fred_series(series, cache_dir, use_network=use_network)
            if not rows:
                continue
            s = pd.Series(rows, dtype=float)
            s.index = pd.to_datetime(s.index)
            s = s.sort_index().rename(asset)
            pieces.append(s.to_frame())
            notes.append(f"FRED {series} used as a {asset} *index proxy* "
                         f"({s.index.min().date()} to {s.index.max().date()}, {len(s)} observations).")

    # 5. Stooq ETFs (optional; skipped if the host refuses).
    stooq_ok = []
    for t in tickers:
        if t == "VIX":
            continue
        frame = fetch_stooq(t, cache_dir, use_network=use_network)
        if frame is None or "close" not in frame.columns:
            continue
        pieces.append(frame[["close"]].rename(columns={"close": t}))
        stooq_ok.append(f"{t} {frame.index.min().date()}–{frame.index.max().date()}")
    if stooq_ok:
        notes.append("Stooq ETF history (personal research, cached): " + "; ".join(stooq_ok) + ".")
    else:
        notes.append("Stooq ETF history was not available (no cache and/or the host refused).")

    close = None
    for piece in pieces:
        if piece is None or piece.empty:
            continue
        piece = piece.copy()
        piece.index = pd.to_datetime(piece.index)
        close = piece if close is None else close.combine_first(piece)
    if close is None:
        close = pd.DataFrame()
    close = close.sort_index()
    close = close.loc[:, ~close.columns.duplicated()]
    # Never stitch S&P 500 *index levels* onto SPY *ETF prices* (that creates a fake crash
    # at the Massive join). Build one equity series from returns, preferring the ETF when both exist.
    if "SP500" in close.columns:
        eq = _return_splice(close["SP500"], close["SPY"] if "SPY" in close.columns else None)
        if eq is not None:
            close["SPY"] = eq
    coverage = {}
    for c in close.columns:
        s = close[c].dropna()
        coverage[c] = {"first": str(s.index.min().date()) if len(s) else None,
                       "last": str(s.index.max().date()) if len(s) else None,
                       "n": int(s.shape[0])}
    notes.append("Index proxies (FRED SP500 / gold fix / WTI / VIX) are not the ETFs: they omit dividends, "
                 "tracking error and, for USO vs WTI, futures roll yield. Survivorship: delisted names are absent.")
    return close, {"notes": notes, "columns": coverage, "nSessions": int(len(close))}


def _session_index(close, day):
    day = pd.Timestamp(day)
    idx = close.index
    # first session on or after `day`
    later = idx[idx >= day]
    if len(later):
        return later[0]
    return None


def _prev_session(close, session):
    idx = close.index
    earlier = idx[idx < session]
    return earlier[-1] if len(earlier) else None


def _forward_path(series, start, horizons=HORIZONS):
    """Returns from the close *before* start, plus max drawdown and recovery days."""
    s = series.dropna()
    prev = _prev_session(s.to_frame("x") if not isinstance(s, pd.DataFrame) else s.to_frame(), start)
    # `start` here is first_session; base is previous close.
    if prev is None or prev not in s.index or start not in s.index:
        # If start is the first available row we cannot form a return.
        if start not in s.index:
            return None
        # Use start close as base (understates the event-day move).
        base = float(s.loc[start])
        after = s.loc[start:]
        base_loc = 0
    else:
        base = float(s.loc[prev])
        after = s.loc[prev:]
        base_loc = 0
    if not base or not np.isfinite(base):
        return None
    rets = after / base - 1.0
    # Horizon k = k sessions after the base (so k=1 is the first_session close).
    out = {"baseDate": str(prev.date()) if prev is not None else None,
           "firstSession": str(pd.Timestamp(start).date())}
    locs = after.index
    for h in horizons:
        if len(locs) > h:
            out[f"r{h}"] = float(rets.iloc[h])
        else:
            out[f"r{h}"] = None
    # Max drawdown from the base over the next 60 sessions, and sessions to recover.
    window = rets.iloc[:61] if len(rets) else rets
    if len(window):
        min_i = int(window.argmin())
        out["maxDrawdown"] = float(window.iloc[min_i])
        out["drawdownSession"] = int(min_i)
        rec = None
        for i, v in enumerate(window.iloc[min_i + 1:], start=min_i + 1):
            if v >= 0:
                rec = int(i)
                break
        out["recoverSessions"] = rec
    else:
        out["maxDrawdown"] = out["drawdownSession"] = out["recoverSessions"] = None
    return out


def event_windows(close, events=EVENTS, assets=ASSETS):
    rows = []
    for ev in events:
        first = _session_index(close, ev["firstSession"])
        rec = {**ev, "resolvedSession": str(first.date()) if first is not None else None, "assets": {}}
        if first is None:
            rec["missing"] = "no session on/after firstSession in the price history"
            rows.append(rec)
            continue
        for asset in assets:
            if asset not in close.columns or close[asset].dropna().empty:
                rec["assets"][asset] = {"missing": "no price history"}
                continue
            path = _forward_path(close[asset], first)
            rec["assets"][asset] = path or {"missing": "could not form a forward path"}
        rows.append(rec)
    return rows


def _summarize(rows, assets=ASSETS, horizons=HORIZONS):
    summary = {}
    for asset in assets:
        cell = {"n": 0, "horizons": {}, "medianMaxDrawdown": None, "medianRecoverSessions": None}
        dds, recs = [], []
        for ev in rows:
            path = (ev.get("assets") or {}).get(asset) or {}
            if path.get("maxDrawdown") is not None:
                dds.append(path["maxDrawdown"])
            if path.get("recoverSessions") is not None:
                recs.append(path["recoverSessions"])
        for h in horizons:
            vals = []
            for ev in rows:
                path = (ev.get("assets") or {}).get(asset) or {}
                if path.get(f"r{h}") is not None:
                    vals.append(path[f"r{h}"])
            if vals:
                arr = np.array(vals, dtype=float)
                cell["horizons"][f"r{h}"] = {
                    "n": int(len(arr)),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p25": float(np.quantile(arr, 0.25)),
                    "p75": float(np.quantile(arr, 0.75)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
                cell["n"] = max(cell["n"], int(len(arr)))
        if dds:
            cell["medianMaxDrawdown"] = float(np.median(dds))
            cell["meanMaxDrawdown"] = float(np.mean(dds))
        if recs:
            cell["medianRecoverSessions"] = float(np.median(recs))
            cell["shareRecoveredIn60"] = float(len(recs) / max(len(dds), 1))
        summary[asset] = cell
    return summary


def _confirm_on_day(close, vix, fx, session, thresholds=None):
    """Causal confirmation using only data through `session`."""
    session = pd.Timestamp(session)
    # Build a tiny vix frame if we have VIX.
    vix_frame = None
    if vix is not None and len(vix):
        vix_frame = vix.loc[:session].copy()
        if "vix" not in vix_frame.columns and "VIX" in vix_frame.columns:
            vix_frame = vix_frame.rename(columns={"VIX": "vix"})
        if "vix" in vix_frame.columns:
            v = vix_frame["vix"]
            if "vix3m" not in vix_frame.columns:
                vix_frame["vix3m"] = v  # unknown; ratio will be ~1
            vix_frame["ratio"] = v / vix_frame["vix3m"]
            vix_frame["z"] = (v - v.rolling(60, min_periods=20).mean()) / v.rolling(60, min_periods=20).std()
            vix_frame["peak20"] = v.rolling(20, min_periods=1).max()
            vix_frame["pct"] = v.rolling(252, min_periods=60).apply(lambda w: float((w <= w[-1]).mean()), raw=True)
    # Minimal panel-like object.
    class _P:
        pass
    panel = _P()
    use = close.loc[:session]
    panel.close = use
    # Open is unknown for index proxies: use prior close as a conservative "open".
    panel.open = use.shift(1)
    fx_cut = {k: v.loc[:session] for k, v in (fx or {}).items()}
    return confirmation_score(inputs={"vix": vix_frame, "panel": panel, "macro": None, "fx": fx_cut,
                                      "asof": str(session.date()), "sources": {}},
                              thresholds=thresholds)


def calibrate_thresholds(close, events=EVENTS, thresholds=None):
    """Set shockStress near the 20th percentile of event-day stress; report false-positive rate."""
    base = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        base.update(thresholds)
    vix = close[["VIX"]].rename(columns={"VIX": "vix"}) if "VIX" in close.columns else None
    fx = {}
    if "USDJPY" in close.columns:
        fx["USDJPY"] = close["USDJPY"]
    if "USDCHF" in close.columns:
        fx["USDCHF"] = close["USDCHF"]
    event_scores, event_ids = [], []
    for ev in events:
        first = _session_index(close, ev["firstSession"])
        if first is None:
            continue
        rec = _confirm_on_day(close, vix, fx, first, base)
        event_scores.append(float(rec.get("stress") or 0))
        event_ids.append(ev["id"])
    # Non-event days: every 20th session, excluding a 10-session window around events.
    event_sessions = set()
    for ev in events:
        first = _session_index(close, ev["firstSession"])
        if first is None:
            continue
        loc = close.index.get_indexer([first], method="ffill")[0]
        for j in range(max(0, loc - 5), min(len(close.index), loc + 10)):
            event_sessions.add(close.index[j])
    sample = [d for i, d in enumerate(close.index) if i % 20 == 0 and d not in event_sessions]
    # Cap the quiet sample so calibration stays cheap.
    sample = sample[::max(1, len(sample) // 250)] if len(sample) > 250 else sample
    quiet = []
    for d in sample:
        rec = _confirm_on_day(close, vix, fx, d, base)
        quiet.append(float(rec.get("stress") or 0))
    shock = float(np.quantile(event_scores, 0.20)) if event_scores else base["shockStress"]
    # Don't go below 0.40 or we will flag too many quiet days; don't go above 0.70 or we miss events.
    shock = float(min(0.70, max(0.40, shock)))
    watch = float(min(shock - 0.12, 0.35))
    watch = float(max(0.22, watch))
    fp = float(np.mean([s >= shock for s in quiet])) if quiet else None
    hit = float(np.mean([s >= shock for s in event_scores])) if event_scores else None
    cal = dict(base)
    cal.update({
        "shockStress": round(shock, 3),
        "watchStress": round(watch, 3),
        "calibrated": True,
        "calibratedAt": iso_et(),
        "eventStress": {i: round(s, 3) for i, s in zip(event_ids, event_scores)},
        "eventStressP20": round(float(np.quantile(event_scores, 0.20)), 3) if event_scores else None,
        "eventStressMedian": round(float(np.median(event_scores)), 3) if event_scores else None,
        "falsePositiveRate": fp,
        "eventHitRate": hit,
        "nEventScores": len(event_scores),
        "nQuietScores": len(quiet),
        "notes": ("shockStress is the 20th percentile of event-day confirmation scores, clipped to [0.40, 0.70]. "
                  "falsePositiveRate is the share of sampled non-event days that would still have crossed it "
                  "(market confirmation only, no news). Few events: treat these as rough, not fitted law."),
    })
    return cal, event_scores, quiet


def _tsmom_signal(close, lookback=252, skip=21):
    """12-1 month trend on SPY/proxy: +1 if momentum > 0 else 0 (flat). Causal."""
    s = close.dropna()
    mom = s.shift(skip) / s.shift(lookback) - 1.0
    sig = (mom > 0).astype(float)
    return sig.reindex(close.index).fillna(0.0)


def walk_forward_playbook(close, events=EVENTS, thresholds=None, hold=10, cost_bps=1.0):
    """Compare a 12-1 TSMOM baseline on SPY to the same strategy with the shock playbook.

    The playbook is applied causally: on first_session, if confirmation stress
    (data through that close) would have been ACTIONABLE at the *calibrated*
    thresholds *and* we treat the event as known news that day (the curated
    list — we cannot replay GDELT in 2001), we scale the TSMOM position to
    shockGross for `hold` sessions, then restore. This uses knowledge that a
    headline existed; it does not use future returns. With ~14 events the
    standard error is large and is reported as such.
    """
    if "SPY" not in close.columns:
        return {"ok": False, "reason": "no SPY (or SP500 proxy) series for the walk-forward"}
    px = close["SPY"].dropna()
    rets = px.pct_change().fillna(0.0)
    sig = _tsmom_signal(px)
    # Baseline: yesterday's signal * today's return (next-close execution).
    pos_base = sig.shift(1).fillna(0.0)
    cost = float(cost_bps) / 10000.0
    turn_base = pos_base.diff().abs().fillna(0.0)
    r_base = pos_base * rets - turn_base * cost

    th = thresholds or load_thresholds()
    shock_scale = float(th.get("shockGross", 0.50) or 0.50)
    shock_cut = th.get("shockStress", 0.55)
    vix = close[["VIX"]].rename(columns={"VIX": "vix"}) if "VIX" in close.columns else None
    fx = {}
    if "USDJPY" in close.columns:
        fx["USDJPY"] = close["USDJPY"]
    pos = pos_base.copy()
    applied = []
    for ev in events:
        first = _session_index(close, ev["firstSession"])
        if first is None or first not in pos.index:
            continue
        rec = _confirm_on_day(close, vix, fx, first, th)
        stress = float(rec.get("stress") or 0)
        # News is treated as severity 4+ on the curated event day (we do not have
        # the contemporary headline feed). Confirmation must still agree.
        news = {"newsSeverity": 4, "direction": ev.get("direction") or "negative",
                "eventType": ev.get("type") or "other"}
        decision = judge(news, rec, th)
        if decision.get("flag") != "ACTIONABLE" or decision.get("agreedDirection") != "negative":
            applied.append({"id": ev["id"], "applied": False, "stress": stress, "flag": decision.get("flag")})
            continue
        loc = pos.index.get_loc(first)
        end = min(loc + int(hold), len(pos.index) - 1)
        pos.iloc[loc:end + 1] = pos.iloc[loc:end + 1] * shock_scale
        applied.append({"id": ev["id"], "applied": True, "stress": stress, "hold": int(end - loc + 1),
                        "flag": decision.get("flag"), "first": str(pd.Timestamp(first).date())})
    turn = pos.diff().abs().fillna(0.0)
    r_pb = pos * rets - turn * cost

    def _perf(r):
        r = r.dropna()
        if not len(r):
            return None
        eq = (1.0 + r).cumprod()
        mu, sd = float(r.mean()), float(r.std())
        sharpe = (mu / sd * math.sqrt(252)) if sd > 1e-12 else 0.0
        peak = eq.cummax()
        dd = float((eq / peak - 1.0).min())
        years = len(r) / 252.0
        cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 and eq.iloc[-1] > 0 else None
        return {"n": int(len(r)), "first": str(r.index.min().date()), "last": str(r.index.max().date()),
                "cagr": cagr, "sharpe": sharpe, "maxDrawdown": dd, "totalReturn": float(eq.iloc[-1] - 1.0)}

    # Event-window contribution: difference in returns over the hold after each applied event.
    diffs = []
    for a in applied:
        if not a.get("applied") or not a.get("first"):
            continue
        first = pd.Timestamp(a["first"])
        if first not in r_base.index:
            continue
        loc = r_base.index.get_loc(first)
        end = min(loc + int(hold) - 1, len(r_base.index) - 1)
        diffs.append(float(r_pb.iloc[loc:end + 1].sum() - r_base.iloc[loc:end + 1].sum()))
    n = len(diffs)
    mean_diff = float(np.mean(diffs)) if n else None
    se = float(np.std(diffs, ddof=1) / math.sqrt(n)) if n >= 2 else None
    return {
        "ok": True,
        "baseline": _perf(r_base),
        "playbook": _perf(r_pb),
        "deltaCagr": (( _perf(r_pb) or {}).get("cagr") or 0) - ((_perf(r_base) or {}).get("cagr") or 0)
        if _perf(r_pb) and _perf(r_base) else None,
        "deltaMaxDrawdown": (( _perf(r_pb) or {}).get("maxDrawdown") or 0) - ((_perf(r_base) or {}).get("maxDrawdown") or 0)
        if _perf(r_pb) and _perf(r_base) else None,
        "appliedEvents": applied,
        "nApplied": sum(1 for a in applied if a.get("applied")),
        "nConsidered": len(applied),
        "eventWindowExcessSum": diffs,
        "eventWindowExcessMean": mean_diff,
        "eventWindowExcessSE": se,
        "holdSessions": hold,
        "shockScale": shock_scale,
        "honesty": (
            f"Only {n} applied event windows. The standard error of the mean event-window excess "
            f"return is {se:.4f} on a mean of {mean_diff:.4f}." if (n >= 2 and se is not None and mean_diff is not None)
            else "Not enough applied events to estimate a standard error."
        ) + " The test treats the curated headline as known on the first session and still requires "
        "market confirmation; it is not a test of the news classifier. TSMOM here is a single-name "
        "12-1 signal on SPY (or the SP500 proxy), not the live multi-name paper book.",
    }


def run_study(use_network=True, write=True, close=None, coverage=None, events=EVENTS, output_dir=None):
    if close is None:
        close, coverage = load_study_prices(use_network=use_network)
    coverage = coverage or {}
    rows = event_windows(close, events)
    summary = _summarize(rows)
    cal, event_scores, quiet = calibrate_thresholds(close, events)
    wf = walk_forward_playbook(close, events, thresholds=cal)
    honesty = [
        "Few events (about a dozen). Means and Sharpe differences have wide error bars; do not treat them as a fitted law.",
        "Survivorship: delisted names are absent from ETF history.",
        "Where FRED index proxies are used they are not the ETFs (no dividends on SP500, no USO roll yield, gold fix vs GLD).",
        "Massive daily_prices alone only cover about two years and miss every event before late 2024.",
        "The walk-forward assumes the curated headline was known on the first session; it tests the *playbook + confirmation*, not the live news classifier.",
        "Intraday events (the 2010 flash crash) are understated by close-to-close returns.",
    ]
    if coverage.get("notes"):
        honesty = coverage["notes"] + honesty
    doc = {
        "generatedAt": iso_et(),
        "nEvents": len(events),
        "events": rows,
        "summary": summary,
        "calibratedThresholds": cal,
        "walkForward": wf,
        "coverage": coverage,
        "honesty": honesty,
    }
    if write:
        # A non-default output_dir writes a research copy only; the live monitor
        # keeps reading thresholds from the default black_swan out_dir().
        custom = output_dir is not None
        root = Path(output_dir) if custom else out_dir()
        root.mkdir(parents=True, exist_ok=True)
        if custom:
            save_json(root / "study_20y.json", doc)
            save_json(root / "walkforward_20y.json", wf)
            live = {k: cal[k] for k in DEFAULT_THRESHOLDS if k in cal}
            live.update({k: cal.get(k) for k in ("calibrated", "calibratedAt", "falsePositiveRate",
                                                  "eventHitRate", "notes")})
            save_json(root / "thresholds_20y.json", live)
            _write_markdown(root / "study_20y.md", doc)
        else:
            save_json(root / "study.json", doc)
            save_json(root / "walkforward.json", wf)
            # Persist the calibrated numeric thresholds the live monitor will load,
            # plus a short note — not the entire quiet-day diagnostic dump.
            live = {k: cal[k] for k in DEFAULT_THRESHOLDS if k in cal}
            live.update({k: cal.get(k) for k in ("calibrated", "calibratedAt", "falsePositiveRate",
                                                  "eventHitRate", "notes")})
            save_json(root / "thresholds.json", live)
            _write_markdown(root / "study.md", doc)
    return doc


def _pct(x, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "–"
    return f"{100 * float(x):.{digits}f}%"


def _write_markdown(path, doc):
    lines = ["# Black swan event study", "",
             f"Generated {doc.get('generatedAt')} (America/New_York).", ""]
    lines.append("## Honesty")
    for h in doc.get("honesty") or []:
        lines.append(f"- {h}")
    lines += ["", "## Coverage"]
    cov = (doc.get("coverage") or {}).get("columns") or {}
    for asset, c in cov.items():
        lines.append(f"- **{asset}**: {c.get('first')} to {c.get('last')} ({c.get('n')} observations)")
    lines += ["", "## Median outcomes across events", "",
              "| Asset | n | 1d | 5d | 20d | 60d | median max DD | median sessions to recover |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for asset, cell in (doc.get("summary") or {}).items():
        h = cell.get("horizons") or {}
        def md(key):
            rec = h.get(key) or {}
            return _pct(rec.get("median")) if rec else "–"
        recov = cell.get("medianRecoverSessions")
        recov_s = f"{recov:.0f}" if recov is not None else "not within 60d / n/a"
        lines.append(f"| {asset} | {cell.get('n')} | {md('r1')} | {md('r5')} | {md('r20')} | {md('r60')} | "
                     f"{_pct(cell.get('medianMaxDrawdown'))} | {recov_s} |")
    lines += ["", "## Per-event SPY path", "",
              "| Event | first session | 1d | 5d | 20d | max DD | recover (sessions) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for ev in doc.get("events") or []:
        spy = (ev.get("assets") or {}).get("SPY") or {}
        lines.append(f"| {ev.get('name')} | {ev.get('resolvedSession') or ev.get('firstSession')} | "
                     f"{_pct(spy.get('r1'))} | {_pct(spy.get('r5'))} | {_pct(spy.get('r20'))} | "
                     f"{_pct(spy.get('maxDrawdown'))} | {spy.get('recoverSessions') if spy.get('recoverSessions') is not None else '–'} |")
    wf = doc.get("walkForward") or {}
    lines += ["", "## Playbook vs 12-1 TSMOM on SPY (or SP500 proxy)", ""]
    if not wf.get("ok"):
        lines.append(wf.get("reason") or "Walk-forward not available.")
    else:
        b, p = wf.get("baseline") or {}, wf.get("playbook") or {}
        lines.append(f"- Baseline TSMOM: CAGR {_pct(b.get('cagr'))}, Sharpe {b.get('sharpe') and round(b['sharpe'], 2)}, "
                     f"max DD {_pct(b.get('maxDrawdown'))} ({b.get('first')} to {b.get('last')}).")
        lines.append(f"- With shock playbook: CAGR {_pct(p.get('cagr'))}, Sharpe {p.get('sharpe') and round(p['sharpe'], 2)}, "
                     f"max DD {_pct(p.get('maxDrawdown'))}.")
        lines.append(f"- Applied to {wf.get('nApplied')} of {wf.get('nConsidered')} events. "
                     f"Mean event-window excess return { _pct(wf.get('eventWindowExcessMean')) } "
                     f"(SE {_pct(wf.get('eventWindowExcessSE'))}).")
        lines.append(f"- {wf.get('honesty')}")
    cal = doc.get("calibratedThresholds") or {}
    lines += ["", "## Calibrated thresholds", "",
              f"- shockStress = {cal.get('shockStress')} (event-day p20, clipped).",
              f"- watchStress = {cal.get('watchStress')}.",
              f"- Event-day confirmation hit rate at that threshold: {cal.get('eventHitRate')}.",
              f"- Quiet-day false positive rate (confirmation only): {cal.get('falsePositiveRate')}.",
              f"- {cal.get('notes') or ''}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, ("\n".join(lines) + "\n").encode("utf-8"))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--offline", action="store_true", help="Use only cached / local series (no network).")
    parser.add_argument("--out", default=None,
                        help="Write study_20y.* into this directory instead of updating live black_swan outputs.")
    args = parser.parse_args(argv)
    doc = run_study(use_network=not args.offline, output_dir=args.out)
    # Compact stdout for the daily job / operator.
    wf = doc.get("walkForward") or {}
    print(json.dumps({
        "ok": True,
        "generatedAt": doc.get("generatedAt"),
        "nEvents": doc.get("nEvents"),
        "coverage": (doc.get("coverage") or {}).get("columns"),
        "spySummary": (doc.get("summary") or {}).get("SPY"),
        "thresholds": {k: (doc.get("calibratedThresholds") or {}).get(k)
                       for k in ("shockStress", "watchStress", "eventHitRate", "falsePositiveRate", "calibrated")},
        "walkForward": {k: wf.get(k) for k in
                        ("ok", "baseline", "playbook", "nApplied", "nConsidered",
                         "eventWindowExcessMean", "eventWindowExcessSE", "honesty")},
        "path": str((Path(args.out) / "study_20y.json") if args.out else (OUT_DIR / "study.json")),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
