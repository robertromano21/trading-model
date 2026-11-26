"""Dollar and macro factor data: a DXY-style proxy from FX pairs, plus FRED rates and credit.

Everything is stored under Massive/Macro/ (gitignored). The Massive API key is read the same way
as the rest of the engine (local_env.massive_key) and is never logged or printed.

Preferred FX source is Massive forex daily aggregates (C:EURUSD and the other ICE basket pairs).
If the plan does not include FX, or the key is missing, the loader falls back to FRED's keyless
fredgraph.csv daily FX series. FRED is also the source for the broad dollar, yields, the curve,
HY OAS, real yields and breakevens.

FRED daily series are typically published with a one-session lag. Decision-time features default
to lag=1 so a value dated t is only used from session t+1. Massive FX daily bars close after the
US equity close, so they use the same lag.

Schwab's Individual Trader API does not document a DXY ticker. Intraday/real-time DXY from Schwab
FX streaming is left as a later step.

Usage (from engine/):
  python macro_data.py update          # incremental daily refresh (safe to call from the paper run)
  python macro_data.py update --full   # re-pull the full history
  python macro_data.py snapshot        # print the latest macro card JSON (no secrets)
  python macro_data.py status
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, DATABASE, MASSIVE_AGGS_URL, MassiveError, download_json, read_results, write_atomically
from local_env import massive_key

LOGGER = logging.getLogger(__name__)

MACRO_DIR = DATA_DIR / "Macro"
FX_DIR = MACRO_DIR / "fx"
FRED_DIR = MACRO_DIR / "fred"
DXY_PATH = MACRO_DIR / "dxy_proxy.csv"
PANEL_PATH = MACRO_DIR / "macro_daily.csv"
STATUS_PATH = MACRO_DIR / "status.json"
SNAPSHOT_PATH = MACRO_DIR / "snapshot.json"
RESEARCH_SNAPSHOT = DATA_DIR / "Research" / "macro_snapshot.json"

# Official ICE US Dollar Index geometric formula (same weights Robert specified).
ICE_CONSTANT = 50.14348112
ICE_EXPONENTS = {
    "EURUSD": -0.576,
    "USDJPY": 0.136,
    "GBPUSD": -0.119,
    "USDCAD": 0.091,
    "USDSEK": 0.042,
    "USDCHF": 0.036,
}
# Massive / Polygon forex tickers (C: prefix).
MASSIVE_FX = {pair: f"C:{pair}" for pair in ICE_EXPONENTS}
# FRED noon buying rates, same quote convention as the Massive pairs.
FRED_FX = {
    "EURUSD": "DEXUSEU",  # USD per EUR
    "USDJPY": "DEXJPUS",  # JPY per USD
    "GBPUSD": "DEXUSUK",  # USD per GBP
    "USDCAD": "DEXCAUS",  # CAD per USD
    "USDSEK": "DEXSDUS",  # SEK per USD
    "USDCHF": "DEXSZUS",  # CHF per USD
}
# Macro series pulled from FRED (keyless fredgraph.csv).
# Real CDS/CDX and ICE SOFR swap / swap-spread quotes are proprietary. Everything below is a
# free FRED proxy. LIBOR swap rates (DSWP*) were discontinued in 2016 and do not cover 2024-26.
TREASURY_CURVE = {
    "DGS1MO": "dgs1mo", "DGS3MO": "dgs3mo", "DGS6MO": "dgs6mo", "DGS1": "dgs1",
    "DGS2": "dgs2", "DGS3": "dgs3", "DGS5": "dgs5", "DGS7": "dgs7",
    "DGS10": "dgs10", "DGS20": "dgs20", "DGS30": "dgs30",
}
SOFR_SERIES = {
    "SOFR": "sofr", "SOFR30DAYAVG": "sofr30", "SOFR90DAYAVG": "sofr90", "SOFR180DAYAVG": "sofr180",
}
CREDIT_OAS = {
    "BAMLC0A0CM": "ig_oas",       # ICE BofA US Corporate Index OAS (investment grade)
    "BAMLH0A0HYM2": "hy_oas",     # ICE BofA US High Yield Index OAS
    "BAMLC0A4CBBB": "bbb_oas",    # ICE BofA BBB US Corporate OAS
    "BAMLH0A3HYC": "ccc_oas",     # ICE BofA CCC & Lower US High Yield OAS
}
FRED_MACRO = {
    "DTWEXBGS": "broad_dollar",
    "DGS10": "dgs10",
    "DGS2": "dgs2",
    "T10Y2Y": "t10y2y",
    "BAMLH0A0HYM2": "hy_oas",
    "DFII10": "dfii10",
    "T10YIE": "t10yie",
    **TREASURY_CURVE,
    **SOFR_SERIES,
    **CREDIT_OAS,
}
CREDIT_ETF_PAIRS = (("HYG", "IEF"), ("LQD", "IEF"), ("JNK", "IEF"))
PROXY_DISCLAIMER = (
    "Real CDS/CDX quotes and ICE SOFR swap / swap-spread rates are proprietary and are not "
    "used here. These are free-data proxies: the Treasury curve (level, slope, curvature), "
    "SOFR and the SOFR–3M Treasury spread (a short-end funding proxy, not a swap spread), "
    "ICE BofA option-adjusted spreads (IG, HY, BBB, CCC) as a CDS/CDX proxy, and traded "
    "HYG/LQD vs IEF total-return ratios. FRED LIBOR swap rates (DSWP*) ended in 2016 and "
    "do not cover this sample."
)
FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"
# Cover more than the local equity history so later expansions do not need a full re-pull.
FRED_START = "1999-01-01"
MASSIVE_FX_START = "2005-01-01"
USER_AGENT = "Trading-Model-main/macro (local research; no API key)"
UPDATE_TIMEOUT_SEC = 90
PUBLICATION_LAG = 1  # sessions; applied by load_macro()
UUP = "UUP"


class MacroError(RuntimeError):
    pass


def _today():
    return date.today().isoformat()


def _strip_query(url):
    """Drop query/fragment so a Massive next_url that contains a key is never logged or stored."""
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def ice_dxy(fx, exponents=None, constant=ICE_CONSTANT):
    """Geometric ICE-style dollar index from a DataFrame of FX levels (one column per pair)."""
    exponents = exponents or ICE_EXPONENTS
    missing = [p for p in exponents if p not in fx.columns]
    if missing:
        raise ValueError(f"ice_dxy is missing FX columns: {missing}")
    frame = fx[list(exponents)].astype(float)
    valid = frame.gt(0).all(axis=1)
    log_sum = sum(float(exp) * np.log(frame[pair]) for pair, exp in exponents.items())
    level = pd.Series(np.where(valid, constant * np.exp(log_sum), np.nan), index=fx.index, name="dxy")
    return level


def parse_fred_csv(raw, series_ids=None):
    """Parse a FRED fredgraph.csv body. Missing observations are '.' or empty."""
    text = raw.decode("utf-8-sig") if isinstance(raw, (bytes, bytearray)) else raw
    from io import StringIO
    frame = pd.read_csv(StringIO(text))
    date_names = {c.upper() for c in frame.columns}
    if frame.empty or not (date_names & {"DATE", "OBSERVATION_DATE"}):
        raise MacroError("FRED CSV did not contain a DATE or observation_date column.")
    date_col = next(c for c in frame.columns if c.upper() in {"DATE", "OBSERVATION_DATE"})
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    frame.index.name = "date"
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col].replace(".", np.nan), errors="coerce")
    if series_ids:
        rename = {}
        for series in series_ids:
            if series in frame.columns:
                rename[series] = series
        frame = frame.rename(columns=rename)
    return frame


def download_fred(series_ids, start=None, end=None, timeout=60):
    """Keyless FRED graph CSV download. series_ids is a list of FRED series names.

    FRED returns a ZIP once too many series are requested in one call, so long lists are fetched
    in batches of 8 and joined on the date index.
    """
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    if len(series_ids) > 8:
        frames = [download_fred(series_ids[i:i + 8], start=start, end=end, timeout=timeout)
                  for i in range(0, len(series_ids), 8)]
        return pd.concat(frames, axis=1)
    params = {"id": ",".join(series_ids)}
    if start:
        params["cosd"] = str(start)
    if end:
        params["coed"] = str(end)
    url = FREDGRAPH + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as error:
        code = error.code
        error.close()
        raise MacroError(f"FRED returned HTTP {code} for {','.join(series_ids)}.") from None
    except (URLError, TimeoutError) as error:
        reason = error.reason if isinstance(error, URLError) else error
        raise MacroError(f"Could not reach FRED: {reason}") from None
    return parse_fred_body(raw, series_ids)


def parse_fred_body(raw, series_ids=None):
    """Parse a FRED graph payload: a CSV, or a ZIP of CSVs when several series are requested."""
    if raw.startswith(b'PK'):
        import io
        import zipfile
        frames = []
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name in archive.namelist():
                lower = name.lower()
                if not lower.endswith('.csv') or lower.endswith('readme.txt'):
                    continue
                frames.append(parse_fred_csv(archive.read(name), series_ids))
        if not frames:
            raise MacroError('FRED ZIP did not contain a CSV.')
        return pd.concat(frames, axis=1)
    return parse_fred_csv(raw, series_ids)


def _massive_daily_bars(ticker, start, end, api_key, max_pages=20):
    """Daily aggregates for one Massive ticker (e.g. C:EURUSD). Follows next_url without logging it."""
    start = str(start)
    end = str(end)
    url = f"{MASSIVE_AGGS_URL}/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000"
    records = []
    pages = 0
    while url and pages < max_pages:
        pages += 1
        raw = download_json(url, api_key)
        document = json.loads(raw)
        chunk = read_results(raw, require_adjusted=False)
        records.extend(chunk)
        nxt = document.get("next_url")
        url = nxt if isinstance(nxt, str) and nxt.startswith("https://") else None
        # Re-auth via header; drop any key that Massive may have put on next_url.
        if url and "apiKey=" in url:
            url = _strip_query(url) + ("?" + urlsplit(nxt).query.split("apiKey=")[0].rstrip("&"))
            # If stripping leaves a broken URL, just stop; the first page is usually complete at limit=50000.
            if not urlsplit(url).path:
                url = None
    daily = {}
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    for record in records:
        ts = record.get("t")
        close = record.get("c")
        if type(ts) is not int or close is None:
            continue
        day = datetime.fromtimestamp(ts / 1000, timezone.utc).date()
        if not first <= day <= last:
            continue
        try:
            value = float(close)
        except (TypeError, ValueError):
            continue
        if value <= 0 or not math.isfinite(value):
            continue
        daily[day] = value
    series = pd.Series(daily, dtype=float).sort_index()
    series.index = pd.to_datetime(series.index)
    series.index.name = "date"
    return series


def try_massive_fx(start, end):
    """Download the ICE basket from Massive. Returns (fx DataFrame, error or None). Never prints the key."""
    try:
        key = massive_key()
    except ValueError as error:
        return None, f"Massive key not set ({error.__class__.__name__})."
    frames = {}
    failures = []
    for pair, ticker in MASSIVE_FX.items():
        try:
            series = _massive_daily_bars(ticker, start, end, key)
            if series.empty:
                failures.append(f"{pair}: empty")
                continue
            frames[pair] = series
        except (MassiveError, ValueError, json.JSONDecodeError) as error:
            # Message comes from download_json; it never includes the key.
            failures.append(f"{pair}: {error}")
            if getattr(error, "status", None) in (401, 403):
                return None, "Massive FX is not available on this plan or the key was rejected."
    if len(frames) < len(ICE_EXPONENTS):
        return None, "Massive FX incomplete (" + "; ".join(failures[:6]) + ")."
    return pd.concat(frames, axis=1).sort_index(), None


def try_fred_fx(start, end):
    series_ids = list(FRED_FX.values())
    frame = download_fred(series_ids, start=start, end=end)
    rename = {fred: pair for pair, fred in FRED_FX.items()}
    frame = frame.rename(columns=rename)
    missing = [p for p in ICE_EXPONENTS if p not in frame.columns]
    if missing:
        raise MacroError(f"FRED FX response missing {missing}.")
    return frame[list(ICE_EXPONENTS)]


def _write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=True, date_format="%Y-%m-%d")
    write_atomically(path, text.encode("utf-8"))
    return path


def _read_csv(path):
    path = Path(path)
    if not path.is_file():
        return None
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "date"
    return frame.sort_index()


def last_stored_date(path):
    frame = _read_csv(path)
    if frame is None or frame.empty:
        return None
    return frame.index.max().date()


def _extend_start(path, default_start, overlap_days=10):
    last = last_stored_date(path)
    if last is None:
        return default_start
    return (last - timedelta(days=overlap_days)).isoformat()


def _merge_csv(path, fresh):
    if fresh is None or fresh.empty:
        return _read_csv(path)
    existing = _read_csv(path)
    if existing is None or existing.empty:
        combined = fresh
    else:
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    _write_csv(path, combined)
    return combined


def _equity_range():
    """(start, end) of daily_prices, or None if the database is missing."""
    path = Path(DATABASE)
    if not path.is_file():
        return None, None
    import sqlite3
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as con:
        row = con.execute("SELECT MIN(date), MAX(date) FROM daily_prices").fetchone()
    if not row or not row[0]:
        return None, None
    return row[0], row[1]


def load_uup_returns(start=None, end=None):
    path = Path(DATABASE)
    if not path.is_file():
        return None
    import sqlite3
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as con:
        frame = pd.read_sql_query(
            "SELECT date, close FROM daily_prices WHERE ticker = ? ORDER BY date",
            con, params=(UUP,))
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.set_index("date")["close"].astype(float)
    if start:
        series = series.loc[pd.Timestamp(start):]
    if end:
        series = series.loc[:pd.Timestamp(end)]
    return series.pct_change()


def update_fx(full=False):
    """Refresh the ICE basket and rebuild the DXY proxy. Tries Massive, then FRED."""
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    end = _today()
    default_start = MASSIVE_FX_START if not full else MASSIVE_FX_START
    equity_start, equity_end = _equity_range()
    # Always cover at least the local price history, and usually more for later research.
    start = FRED_START if full or last_stored_date(DXY_PATH) is None else _extend_start(DXY_PATH, default_start)
    if equity_start and start > equity_start and last_stored_date(DXY_PATH) is None:
        start = min(start, equity_start)
    end = max(end, equity_end or end)

    source = None
    fx = None
    notes = []
    massive, massive_error = try_massive_fx(start, end)
    if massive is not None:
        fx = massive
        source = "massive"
        notes.append(f"Massive FX daily bars {fx.index[0].date()} to {fx.index[-1].date()} ({len(fx)} rows).")
    else:
        notes.append(f"Massive FX unavailable, using FRED ({massive_error}).")
        fx = try_fred_fx(FRED_START if full or last_stored_date(DXY_PATH) is None else start, end)
        source = "fred"
        notes.append(f"FRED FX {fx.index[0].date()} to {fx.index[-1].date()} ({int(fx.notna().all(axis=1).sum())} complete rows).")

    for pair in ICE_EXPONENTS:
        _merge_csv(FX_DIR / f"{pair}.csv", fx[[pair]].dropna().rename(columns={pair: "close"}))

    dxy = ice_dxy(fx).dropna().to_frame("dxy")
    dxy["source"] = source
    dxy["dxy_ret"] = dxy["dxy"].pct_change()
    combined = _merge_csv(DXY_PATH, dxy)
    # Recompute the index on the merged FX files so a mixed-source history stays consistent per row.
    parts = []
    for pair in ICE_EXPONENTS:
        part = _read_csv(FX_DIR / f"{pair}.csv")
        if part is not None and "close" in part.columns:
            parts.append(part["close"].rename(pair))
    if len(parts) == len(ICE_EXPONENTS):
        rebuilt = ice_dxy(pd.concat(parts, axis=1)).dropna().to_frame("dxy")
        if "source" in combined.columns:
            rebuilt["source"] = combined["source"].reindex(rebuilt.index)
        rebuilt["source"] = rebuilt["source"].fillna(source)
        rebuilt["dxy_ret"] = rebuilt["dxy"].pct_change()
        _write_csv(DXY_PATH, rebuilt)
        combined = rebuilt
    return {"source": source, "rows": int(len(combined)), "start": str(combined.index[0].date()) if len(combined) else None,
            "end": str(combined.index[-1].date()) if len(combined) else None, "notes": notes, "path": str(DXY_PATH)}


def update_fred_macro(full=False):
    start = FRED_START if full or not any((FRED_DIR / f"{s}.csv").is_file() for s in FRED_MACRO) else None
    if start is None:
        starts = [_extend_start(FRED_DIR / f"{s}.csv", FRED_START) for s in FRED_MACRO]
        start = min(starts)
    end = _today()
    frame = download_fred(list(FRED_MACRO), start=start, end=end)
    written = {}
    for series in FRED_MACRO:
        if series not in frame.columns:
            continue
        col = frame[[series]].dropna()
        merged = _merge_csv(FRED_DIR / f"{series}.csv", col)
        written[series] = {"rows": int(len(merged)), "start": str(merged.index[0].date()), "end": str(merged.index[-1].date())}
    # Prefer a computed 10y-2y when both legs exist; keep T10Y2Y as a cross-check.
    dgs10 = _read_csv(FRED_DIR / "DGS10.csv")
    dgs2 = _read_csv(FRED_DIR / "DGS2.csv")
    if dgs10 is not None and dgs2 is not None:
        spread = (dgs10["DGS10"] - dgs2["DGS2"]).to_frame("curve_10y2y")
        _merge_csv(FRED_DIR / "curve_10y2y.csv", spread.dropna())
    return written


def build_panel():
    """Align DXY, FRED macros and derived features on a daily index; write macro_daily.csv."""
    pieces = []
    dxy = _read_csv(DXY_PATH)
    if dxy is not None and "dxy" in dxy.columns:
        pieces.append(dxy[["dxy"]].copy())
        if "dxy_ret" not in dxy.columns:
            dxy["dxy_ret"] = dxy["dxy"].pct_change()
        pieces.append(dxy[["dxy_ret"]])
        if "source" in dxy.columns:
            pieces.append(dxy[["source"]])
    for series, name in FRED_MACRO.items():
        frame = _read_csv(FRED_DIR / f"{series}.csv")
        if frame is None or frame.empty:
            continue
        col = frame.columns[0]
        pieces.append(frame[col].rename(name).to_frame())
    curve = _read_csv(FRED_DIR / "curve_10y2y.csv")
    if curve is not None:
        pieces.append(curve)
    if not pieces:
        raise MacroError("No macro series on disk. Run python macro_data.py update first.")
    panel = pd.concat(pieces, axis=1)
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()
    if "t10y2y" in panel.columns and "curve_10y2y" not in panel.columns:
        panel["curve_10y2y"] = panel["t10y2y"]
    elif "t10y2y" in panel.columns and "curve_10y2y" in panel.columns:
        panel["curve_10y2y"] = panel["curve_10y2y"].fillna(panel["t10y2y"])
    if "dxy" in panel.columns:
        panel["dxy_ret5"] = panel["dxy"].pct_change(5)
        panel["dxy_ret20"] = panel["dxy"].pct_change(20)
        panel["dxy_ret60"] = panel["dxy"].pct_change(60)
        log_ret = np.log(panel["dxy"]).diff()
        panel["dxy_vol20"] = log_ret.rolling(20, min_periods=15).std() * math.sqrt(252)
        panel["dxy_vol60"] = log_ret.rolling(60, min_periods=40).std() * math.sqrt(252)
    if "dgs10" in panel.columns:
        panel["dgs10_chg20"] = panel["dgs10"].diff(20)
    if "hy_oas" in panel.columns:
        panel["hy_oas_chg20"] = panel["hy_oas"].diff(20)
    if "broad_dollar" in panel.columns:
        panel["broad_dollar_ret20"] = panel["broad_dollar"].pct_change(20)
    # Curve shape: level (10Y), slope (10Y-2Y and 10Y-3M), curvature (2*5Y - 2Y - 10Y butterfly).
    if "dgs10" in panel.columns:
        panel["curve_level"] = panel["dgs10"]
    if "curve_10y2y" in panel.columns:
        panel["curve_slope"] = panel["curve_10y2y"]
    elif "dgs10" in panel.columns and "dgs2" in panel.columns:
        panel["curve_slope"] = panel["dgs10"] - panel["dgs2"]
        panel["curve_10y2y"] = panel["curve_slope"]
    if "dgs10" in panel.columns and "dgs3mo" in panel.columns:
        panel["curve_slope_10y3m"] = panel["dgs10"] - panel["dgs3mo"]
    if "dgs5" in panel.columns and "dgs2" in panel.columns and "dgs10" in panel.columns:
        panel["curve_curvature"] = 2.0 * panel["dgs5"] - panel["dgs2"] - panel["dgs10"]
    if "dgs10" in panel.columns and "dgs2" in panel.columns and "dgs30" in panel.columns:
        panel["curve_curvature_2s10s30s"] = 2.0 * panel["dgs10"] - panel["dgs2"] - panel["dgs30"]
    if "dgs10" in panel.columns:
        panel["curve_level_chg20"] = panel["dgs10"].diff(20)
    if "curve_slope" in panel.columns:
        panel["curve_slope_chg20"] = panel["curve_slope"].diff(20)
    # SOFR and a short-end funding proxy (SOFR minus 3M Treasury). Not an ICE swap spread.
    if "sofr" in panel.columns and "dgs3mo" in panel.columns:
        panel["sofr_tbill_spread"] = panel["sofr"] - panel["dgs3mo"]
        panel["sofr_tbill_spread_chg20"] = panel["sofr_tbill_spread"].diff(20)
    if "ig_oas" in panel.columns:
        panel["ig_oas_chg20"] = panel["ig_oas"].diff(20)
    if "hy_oas" in panel.columns and "ig_oas" in panel.columns:
        panel["hy_ig_oas"] = panel["hy_oas"] - panel["ig_oas"]
        panel["hy_ig_oas_chg20"] = panel["hy_ig_oas"].diff(20)
    if "bbb_oas" in panel.columns:
        panel["bbb_oas_chg20"] = panel["bbb_oas"].diff(20)
    if "ccc_oas" in panel.columns:
        panel["ccc_oas_chg20"] = panel["ccc_oas"].diff(20)
        if "hy_oas" in panel.columns:
            panel["ccc_hy_oas"] = panel["ccc_oas"] - panel["hy_oas"]
    # Traded credit proxies from local ETF prices (HYG/LQD vs IEF).
    for col, series in credit_etf_ratios().items():
        aligned = series.reindex(panel.index)
        panel[col] = aligned
        panel[col + "_chg20"] = aligned.diff(20)
    _write_csv(PANEL_PATH, panel)
    return panel



def credit_etf_ratios():
    """Total-return index ratios HYG/IEF, LQD/IEF, JNK/IEF from the local price database."""
    path = Path(DATABASE)
    out = {}
    if not path.is_file():
        return out
    import sqlite3
    tickers = sorted({a for pair in CREDIT_ETF_PAIRS for a in pair})
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as con:
        frame = pd.read_sql_query(
            "SELECT ticker, date, close FROM daily_prices WHERE ticker IN ({})".format(
                ",".join("?" * len(tickers))),
            con, params=tickers)
    if frame.empty:
        return out
    # Dividends are small relative to spread moves; close-to-close is the traded proxy.
    frame["date"] = pd.to_datetime(frame["date"])
    close = frame.pivot(index="date", columns="ticker", values="close").sort_index().astype(float)
    rets = close.pct_change()
    index = (1.0 + rets.fillna(0.0)).cumprod().where(close.notna())
    for risky, safe in CREDIT_ETF_PAIRS:
        if risky in index.columns and safe in index.columns:
            name = f"{risky.lower()}_{safe.lower()}_ratio"
            out[name] = (index[risky] / index[safe]).rename(name)
    return out


def load_treasury_curve(lag=0, asof=None):
    """Date x tenor (years) constant-maturity Treasury yields in percent, optional publication lag."""
    tenors = {
        "dgs1mo": 1 / 12, "dgs3mo": 0.25, "dgs6mo": 0.5, "dgs1": 1.0, "dgs2": 2.0, "dgs3": 3.0,
        "dgs5": 5.0, "dgs7": 7.0, "dgs10": 10.0, "dgs20": 20.0, "dgs30": 30.0,
    }
    panel = load_macro(lag=lag, asof=asof)
    cols = [c for c in tenors if c in panel.columns]
    if not cols:
        # Unlagged raw FRED files, for pricing that wants the published curve.
        pieces = []
        for series, name in TREASURY_CURVE.items():
            frame = _read_csv(FRED_DIR / f"{series}.csv")
            if frame is not None and not frame.empty:
                pieces.append(frame.iloc[:, 0].rename(name))
        if not pieces:
            return pd.DataFrame()
        panel = pd.concat(pieces, axis=1).sort_index()
        if lag:
            panel = panel.shift(int(lag))
        if asof is not None:
            panel = panel.loc[:pd.Timestamp(asof)]
        cols = [c for c in tenors if c in panel.columns]
    curve = panel[cols].copy()
    curve.columns = [tenors[c] for c in cols]
    return curve.astype(float)


def interpolate_par_yield(curve_row, maturity_years):
    """Linear interpolate a constant-maturity par yield (percent) at a given maturity in years."""
    if curve_row is None:
        return float("nan")
    series = pd.Series(curve_row).dropna().sort_index()
    if series.empty or not math.isfinite(float(maturity_years)) or maturity_years <= 0:
        return float("nan")
    points = series.index.astype(float)
    values = series.to_numpy(dtype=float)
    target = float(maturity_years)
    if target <= points[0]:
        return float(values[0])
    if target >= points[-1]:
        return float(values[-1])
    return float(np.interp(target, points, values))


def load_macro(start=None, end=None, lag=PUBLICATION_LAG, asof=None):
    """Date-indexed macro panel. lag shifts values forward so they are available at decision time.

    A lag of 1 means the FRED/DXY value dated t is first used on t+1. asof cuts rows after that date
    (no lookahead).
    """
    panel = _read_csv(PANEL_PATH)
    if panel is None or panel.empty:
        if DXY_PATH.is_file():
            panel = build_panel()
        else:
            return pd.DataFrame()
    value_cols = [c for c in panel.columns if c != "source"]
    if lag and lag > 0:
        shifted = panel[value_cols].shift(int(lag))
        if "source" in panel.columns:
            panel = pd.concat([shifted, panel[["source"]]], axis=1)
        else:
            panel = shifted
    if start:
        panel = panel.loc[pd.Timestamp(start):]
    if end:
        panel = panel.loc[:pd.Timestamp(end)]
    if asof is not None:
        panel = panel.loc[:pd.Timestamp(asof)]
    return panel


def load_dxy(lag=0):
    frame = _read_csv(DXY_PATH)
    if frame is None or frame.empty or "dxy" not in frame.columns:
        return pd.Series(dtype=float, name="dxy")
    series = frame["dxy"].astype(float)
    if lag:
        series = series.shift(int(lag))
    return series


def uup_crosscheck(lag=0):
    dxy = load_dxy(lag=0)
    uup = load_uup_returns()
    if dxy.empty or uup is None or uup.empty:
        return {"available": False, "reason": "UUP or DXY series missing."}
    dxy_ret = dxy.pct_change()
    frame = pd.concat([dxy_ret.rename("dxy"), uup.rename("uup")], axis=1, join="inner").dropna()
    if len(frame) < 20:
        return {"available": False, "reason": f"Only {len(frame)} overlapping sessions.", "n": int(len(frame))}
    corr = float(frame["dxy"].corr(frame["uup"]))
    return {"available": True, "n": int(len(frame)), "start": str(frame.index[0].date()),
            "end": str(frame.index[-1].date()), "corr": corr,
            "dxyVol": float(frame["dxy"].std() * math.sqrt(252)),
            "uupVol": float(frame["uup"].std() * math.sqrt(252))}


def _last_number(series):
    if series is None:
        return None
    series = pd.Series(series).dropna()
    if series.empty:
        return None
    value = float(series.iloc[-1])
    return value if math.isfinite(value) else None


def snapshot(asof=None):
    """Latest dollar / rates / credit card for the UI and the [MACRO] prompt block."""
    panel = _read_csv(PANEL_PATH)
    if panel is None or panel.empty:
        if DXY_PATH.is_file():
            panel = build_panel()
        else:
            return {"available": False, "reason": "Macro panel not built yet. Run python macro_data.py update."}
    if asof is not None:
        panel = panel.loc[:pd.Timestamp(asof)]
    if panel.empty:
        return {"available": False, "reason": "No macro rows on or before as-of."}

    raw = panel  # unlagged published dates, labelled honestly
    last = raw.dropna(how="all")
    day = last.index[-1]
    row = last.iloc[-1]
    dxy = _last_number(raw.get("dxy"))
    dxy_20 = _last_number(raw.get("dxy_ret20"))
    dxy_5 = _last_number(raw.get("dxy_ret5"))
    dxy_vol = _last_number(raw.get("dxy_vol20"))
    trend = None
    if dxy_20 is not None:
        if dxy_20 > 0.01:
            trend = "stronger"
        elif dxy_20 < -0.01:
            trend = "weaker"
        else:
            trend = "sideways"
    out = {
        "available": True,
        "asOf": str(day.date()),
        "publicationLagSessions": PUBLICATION_LAG,
        "dxy": {
            "level": dxy,
            "ret5": dxy_5,
            "ret20": dxy_20,
            "ret60": _last_number(raw.get("dxy_ret60")),
            "vol20": dxy_vol,
            "trend": trend,
            "source": None if "source" not in raw.columns else (None if pd.isna(row.get("source")) else str(row.get("source"))),
        },
        "broadDollar": _last_number(raw.get("broad_dollar")),
        "broadDollarRet20": _last_number(raw.get("broad_dollar_ret20")),
        "dgs10": _last_number(raw.get("dgs10")),
        "dgs2": _last_number(raw.get("dgs2")),
        "dgs10Chg20": _last_number(raw.get("dgs10_chg20")),
        "curve10y2y": _last_number(raw.get("curve_10y2y")),
        "hyOas": _last_number(raw.get("hy_oas")),
        "hyOasChg20": _last_number(raw.get("hy_oas_chg20")),
        "dfii10": _last_number(raw.get("dfii10")),
        "t10yie": _last_number(raw.get("t10yie")),
        "rates": {
            "sofr": _last_number(raw.get("sofr")),
            "sofr30": _last_number(raw.get("sofr30")),
            "dgs3mo": _last_number(raw.get("dgs3mo")),
            "dgs2": _last_number(raw.get("dgs2")),
            "dgs5": _last_number(raw.get("dgs5")),
            "dgs10": _last_number(raw.get("dgs10")),
            "dgs30": _last_number(raw.get("dgs30")),
            "curveLevel": _last_number(raw.get("curve_level")),
            "curveSlope10y2y": _last_number(raw.get("curve_slope")),
            "curveSlope10y3m": _last_number(raw.get("curve_slope_10y3m")),
            "curveCurvature": _last_number(raw.get("curve_curvature")),
            "curveLevelChg20": _last_number(raw.get("curve_level_chg20")),
            "sofrTbillSpread": _last_number(raw.get("sofr_tbill_spread")),
        },
        "credit": {
            "igOas": _last_number(raw.get("ig_oas")),
            "hyOas": _last_number(raw.get("hy_oas")),
            "bbbOas": _last_number(raw.get("bbb_oas")),
            "cccOas": _last_number(raw.get("ccc_oas")),
            "hyIgOas": _last_number(raw.get("hy_ig_oas")),
            "hyOasChg20": _last_number(raw.get("hy_oas_chg20")),
            "hygIefRatio": _last_number(raw.get("hyg_ief_ratio")),
            "lqdIefRatio": _last_number(raw.get("lqd_ief_ratio")),
        },
        "proxyDisclaimer": PROXY_DISCLAIMER,
        "uup": uup_crosscheck(),
        "note": ("Values are the latest published observations. Decision-time features lag them by "
                 f"{PUBLICATION_LAG} session(s) to avoid FRED publication lookahead. "
                 "The dollar index is a DXY-style proxy, not the ICE futures print. "
                 + PROXY_DISCLAIMER),
    }
    try:
        path = MACRO_DIR / "dxy_outlook.json"
        alt = DATA_DIR / "Research" / "dxy_forecast_outlook.json"
        for candidate in (path, alt):
            if candidate.is_file():
                out["dxyOutlook"] = json.loads(candidate.read_text(encoding="utf-8"))
                break
    except Exception:
        pass
    return out


def write_snapshot(payload=None):
    payload = payload or snapshot()
    text = json.dumps(payload, indent=2, default=str).encode()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(SNAPSHOT_PATH, text)
    RESEARCH_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(RESEARCH_SNAPSHOT, text)
    return payload


def status():
    dxy = _read_csv(DXY_PATH)
    panel = _read_csv(PANEL_PATH)
    st = _read_csv(STATUS_PATH) if False else None  # status is JSON
    payload = {
        "macroDir": str(MACRO_DIR),
        "dxy": None if dxy is None or dxy.empty else {
            "rows": int(len(dxy)), "start": str(dxy.index[0].date()), "end": str(dxy.index[-1].date()),
            "sourceCounts": dxy["source"].value_counts().to_dict() if "source" in dxy.columns else {},
        },
        "panel": None if panel is None or panel.empty else {
            "rows": int(len(panel)), "start": str(panel.index[0].date()), "end": str(panel.index[-1].date()),
            "columns": list(panel.columns),
        },
        "equityRange": dict(zip(("start", "end"), _equity_range())),
        "uup": uup_crosscheck(),
        "snapshotPath": str(SNAPSHOT_PATH),
        "schwabRealtimeDxy": "not implemented — Schwab Individual Trader API has no documented DXY ticker; FX streaming is limited. Later step.",
    }
    if STATUS_PATH.is_file():
        try:
            payload["lastUpdate"] = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload["lastUpdate"] = None
    return payload


def update(full=False, timeout=UPDATE_TIMEOUT_SEC):
    """Incremental refresh. Returns a status dict. Designed to be safe to call from the daily paper run."""
    started = time.time()
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    notes = []
    result = {"ok": False, "full": bool(full), "startedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    try:
        fx_info = update_fx(full=full)
        notes.extend(fx_info.get("notes") or [])
        result["fx"] = {k: v for k, v in fx_info.items() if k != "notes"}
        result["fred"] = update_fred_macro(full=full)
        panel = build_panel()
        result["panel"] = {"rows": int(len(panel)), "start": str(panel.index[0].date()),
                           "end": str(panel.index[-1].date()), "columns": list(panel.columns)}
        snap = write_snapshot(snapshot())
        result["snapshot"] = {k: snap[k] for k in ("available", "asOf") if k in snap}
        result["uup"] = snap.get("uup")
        result["ok"] = True
    except (MacroError, MassiveError, ValueError, OSError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        notes.append(result["error"])
        LOGGER.warning("macro update failed: %s", error)
    result["notes"] = notes
    result["seconds"] = round(time.time() - started, 2)
    write_atomically(STATUS_PATH, json.dumps(result, indent=2, default=str).encode())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["update", "snapshot", "status"], nargs="?", default="status")
    parser.add_argument("--full", action="store_true", help="re-pull the full history instead of an incremental update")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "update":
        result = update(full=args.full)
        # Status JSON only — never a key, never a URL with a query string.
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("ok") else 2
    if args.command == "snapshot":
        print(json.dumps(write_snapshot(), indent=2, default=str))
        return 0
    print(json.dumps(status(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
