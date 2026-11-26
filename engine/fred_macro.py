"""FRED macro import with point-in-time (publication-lag / ALFRED first-release) availability.

Raw series live under Massive/Macro/fred_long/<ID>.csv (latest vintage, full history) and, when a
FRED_API_KEY is configured, Massive/Macro/fred_long/vintage/<ID>.csv (ALFRED *initial release* values
with the exact first-publication date). The key is read from .env.local / the environment and is never
printed or written to any output.

Look-ahead control
------------------
Every observation gets an ``available`` date: the first calendar day on which a trader could have known
it. With ALFRED first-release data that is the real publication date. Without a key it is
``period_end + lag_days`` using the conservative lags in SERIES (e.g. CPI: month end + 25 days, PCE +40,
payrolls +10, weekly claims week-end + 5). ``asof_frame`` turns that into a daily panel where the value on
session d is the latest observation with available <= d. Callers that trade on the panel must still shift
positions one session (decide at the close of d, hold from d+1).

Caveat: without ALFRED vintages the *values* are today's revised numbers (payrolls, GDP, INDPRO revise a
lot; CPI/UNRATE little). The report states which mode was used.

Usage (from engine/):
  python fred_macro.py update            # incremental (safe for the daily run; never raises on network)
  python fred_macro.py update --full     # re-pull everything
  python fred_macro.py update --vintages # also pull ALFRED first releases (needs FRED_API_KEY)
  python fred_macro.py status
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

try:
    from fetch_and_model import DATA_DIR, write_atomically
except Exception:  # pragma: no cover - tests may run without the full engine
    DATA_DIR = Path(__file__).resolve().parents[1] / "Massive"

    def write_atomically(path, contents):
        Path(path).write_bytes(contents)

FRED_LONG_DIR = DATA_DIR / "Macro" / "fred_long"
VINTAGE_DIR = FRED_LONG_DIR / "vintage"
MANIFEST_PATH = FRED_LONG_DIR / "manifest.json"
FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
USER_AGENT = "Trading-Model-main/fred-macro (personal research; +local)"
MIN_GAP_SEC = 0.6          # keyless: be polite; keyed: FRED allows 120 req/min
FULL_START = "1900-01-01"

# id: (frequency, lag_days after period end, title, group)
# Frequencies: D daily, W weekly (FRED date = week-end / observation day), M monthly (dated 1st), Q quarterly.
SERIES = {
    # inflation
    "CPIAUCSL": ("M", 25, "CPI all items, SA", "inflation"),
    "CPILFESL": ("M", 25, "Core CPI, SA", "inflation"),
    "PCEPILFE": ("M", 40, "Core PCE price index", "inflation"),
    "T5YIE": ("D", 1, "5y breakeven inflation", "inflation"),
    "T10YIE": ("D", 1, "10y breakeven inflation", "inflation"),
    "DCOILWTICO": ("D", 1, "WTI crude spot", "inflation"),
    "GASREGW": ("W", 2, "US regular gasoline retail, weekly", "inflation"),
    "PPIACO": ("M", 20, "PPI all commodities", "inflation"),
    # labour
    "UNRATE": ("M", 10, "Unemployment rate", "labor"),
    "PAYEMS": ("M", 10, "Nonfarm payrolls (thousands)", "labor"),
    "ICSA": ("W", 5, "Initial jobless claims, SA (week ending Sat, out Thu)", "labor"),
    "CCSA": ("W", 12, "Continuing claims, SA (one extra week behind ICSA)", "labor"),
    "SAHMREALTIME": ("M", 10, "Sahm rule, real-time", "labor"),
    # rates / policy
    "FEDFUNDS": ("M", 1, "Effective fed funds (monthly avg)", "rates"),
    "DFF": ("D", 1, "Effective fed funds (daily)", "rates"),
    "DTB3": ("D", 1, "3-month T-bill (daily)", "rates"),
    "TB3MS": ("M", 1, "3-month T-bill (monthly avg)", "rates"),
    "DGS2": ("D", 1, "2y Treasury", "rates"),
    "DGS10": ("D", 1, "10y Treasury", "rates"),
    "GS10": ("M", 1, "10y Treasury (monthly avg)", "rates"),
    "T10Y2Y": ("D", 1, "10y-2y spread", "rates"),
    "T10Y3M": ("D", 1, "10y-3m spread", "rates"),
    # credit / stress
    "BAMLH0A0HYM2": ("D", 1, "ICE BofA HY OAS (FRED keeps ~3y only)", "credit"),
    "BAA10Y": ("D", 1, "Moody's Baa - 10y", "credit"),
    "BAA": ("M", 1, "Moody's Baa yield (monthly)", "credit"),
    "AAA": ("M", 1, "Moody's Aaa yield (monthly)", "credit"),
    "NFCI": ("W", 5, "Chicago Fed NFCI (week ending Fri, out Wed)", "stress"),
    "STLFSI4": ("W", 6, "St. Louis Fed financial stress", "stress"),
    "VIXCLS": ("D", 1, "VIX close", "stress"),
    # growth / activity
    "INDPRO": ("M", 20, "Industrial production", "growth"),
    "UMCSENT": ("M", 5, "UMich sentiment", "growth"),
    "GDPC1": ("Q", 35, "Real GDP (advance ~30d after quarter)", "growth"),
    "IMPGSC1": ("Q", 45, "Real U.S. imports of goods and services", "growth"),
    "EXPGSC1": ("Q", 45, "Real U.S. exports of goods and services", "growth"),
    # OUTSIDE FORECAST - display/comparison only, never a model input (see OUTSIDE_FORECASTS).
    "GDPNOW": ("Q", 30, "Atlanta Fed GDPNow (final nowcast per quarter)", "outside_forecast"),
    "WEI": ("W", 5, "NY Fed Weekly Economic Index", "growth"),
    # money / balance sheet
    "M2SL": ("M", 30, "M2 money stock", "money"),
    "WALCL": ("W", 1, "Fed total assets (Wed level, out Thu)", "money"),
    # dollar
    "DTWEXBGS": ("D", 7, "Broad trade-weighted dollar (H.10 is weekly)", "dollar"),
    "DTWEXAFEGS": ("D", 7, "Dollar vs advanced foreign economies", "dollar"),
    "DTWEXM": ("D", 7, "Legacy major-currencies dollar (1973-2019)", "dollar"),
    "DTWEXB": ("D", 7, "Legacy broad dollar (1995-2019)", "dollar"),
    # equity (FRED keeps ~10y of SP500)
    "SP500": ("D", 1, "S&P 500 (FRED, ~10y only)", "equity"),
}

# First-release *growth rates* are fetched with FRED's `units` transform inside each ALFRED vintage, so the
# m/m change is computed within one vintage (no base-year rebasing jumps, and payroll changes match the
# headline as reported). Pseudo-ids look like "CPIAUCSL:pch".
VINTAGE_TRANSFORMS = {"CPIAUCSL": "pch", "CPILFESL": "pch", "PCEPILFE": "pch", "PPIACO": "pch",
                      "PAYEMS": "chg", "GDPC1": "pca", "IMPGSC1": "pca", "EXPGSC1": "pca", "INDPRO": "pch"}
TRADE_GROWTH_VINTAGES = frozenset({"IMPGSC1", "EXPGSC1"})

# Outside projections (Street consensus, Fed nowcasts, surveys of forecasters). These are stored for
# display and accuracy comparison only. macro_nowcast refuses to use any of them as a feature.
OUTSIDE_FORECASTS = frozenset({"GDPNOW"})

# Dollar series are observable in real time in the FX market; H.10 is just the record. Event studies
# that treat the dollar as a market price may use this lag instead of the H.10 publication lag.
MARKET_LAG_DAYS = 1


class FredError(RuntimeError):
    pass


# ---------------------------------------------------------------- env (never printed)
def _api_key():
    try:
        from local_env import env_value
        return env_value("FRED_API_KEY")
    except Exception:  # noqa: BLE001
        import os
        return (os.environ.get("FRED_API_KEY") or "").strip()


def has_key():
    return bool(_api_key())


# ---------------------------------------------------------------- period / availability
def period_end(dates, freq):
    idx = pd.DatetimeIndex(dates)
    if freq == "M":
        return idx + pd.offsets.MonthEnd(0)
    if freq == "Q":
        return idx + pd.offsets.QuarterEnd(0)
    return idx


def availability(dates, series_id, lag_days=None):
    freq, lag, *_ = meta(series_id)
    lag = lag if lag_days is None else lag_days
    return period_end(dates, freq) + pd.to_timedelta(int(lag), unit="D")


# ---------------------------------------------------------------- HTTP
_last_call = [0.0]


def _polite():
    wait = _last_call[0] + MIN_GAP_SEC - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _get(url, opener=None, timeout=60, accept="text/csv", safe_name=""):
    """GET with no secret in any error message (the query string is never echoed)."""
    if opener is not None:
        return opener(url)
    _polite()
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except HTTPError as err:
            code = err.code
            err.close()
            if code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 + 3 * attempt)
                continue
            raise FredError(f"FRED HTTP {code} for {safe_name}") from None
        except (URLError, TimeoutError) as err:
            if attempt < 2:
                time.sleep(2 + 3 * attempt)
                continue
            raise FredError(f"Could not reach FRED for {safe_name}: {getattr(err, 'reason', err)}") from None


def parse_graph_csv(raw, series_id):
    from io import StringIO
    text = raw.decode("utf-8-sig") if isinstance(raw, (bytes, bytearray)) else raw
    frame = pd.read_csv(StringIO(text))
    date_col = next((c for c in frame.columns if c.upper() in {"DATE", "OBSERVATION_DATE"}), None)
    if date_col is None:
        raise FredError(f"FRED CSV for {series_id} had no date column")
    val_col = series_id if series_id in frame.columns else [c for c in frame.columns if c != date_col][0]
    s = pd.Series(pd.to_numeric(frame[val_col].replace(".", np.nan), errors="coerce").values,
                  index=pd.to_datetime(frame[date_col], errors="coerce"), name=series_id)
    s = s[s.index.notna()].dropna().sort_index()
    s.index.name = "date"
    return s


def fetch_graph(series_id, start=None, opener=None):
    params = {"id": series_id, "cosd": start or FULL_START}
    return parse_graph_csv(_get(FREDGRAPH + "?" + urlencode(params), opener=opener, safe_name=series_id), series_id)


def fetch_api(series_id, start=None, opener=None, output_type=None, units=None):
    """Official API (needs key). output_type=4 returns ALFRED initial releases with realtime_start."""
    key = _api_key()
    if not key:
        raise FredError("FRED_API_KEY is not set")
    params = {"series_id": series_id, "file_type": "json", "api_key": key, "limit": 100000,
              "observation_start": start or "1776-07-04"}
    if units:
        params["units"] = units
    if output_type:
        params.update({"output_type": output_type, "realtime_start": "1776-07-04", "realtime_end": "9999-12-31"})
    rows, offset = [], 0
    while True:
        params["offset"] = offset
        doc = json.loads(_get(FRED_API + "?" + urlencode(params), opener=opener, accept="application/json",
                              safe_name=series_id).decode("utf-8"))
        obs = doc.get("observations") or []
        rows.extend(obs)
        if len(obs) < params["limit"]:
            break
        offset += len(obs)
    return rows


def parse_api_rows(rows, series_id, vintage=False):
    recs = []
    for r in rows:
        v = r.get("value")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if vintage:
            # output_type=4 names the value column after the series + realtime date; tolerate both shapes
            recs.append((r.get("date"), v, r.get("realtime_start")))
        else:
            recs.append((r.get("date"), v, None))
    frame = pd.DataFrame(recs, columns=["date", "value", "available"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["available"] = pd.to_datetime(frame["available"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["date", "available"])
    if vintage:
        frame = frame.drop_duplicates("date", keep="first")
    return frame.set_index("date")


# ---------------------------------------------------------------- storage
def series_path(series_id, root=None):
    return Path(root or FRED_LONG_DIR) / f"{series_id}.csv"


def vintage_path(series_id, root=None, units=None):
    name = f"{series_id}__{units}" if units else series_id
    return Path(root or FRED_LONG_DIR) / "vintage" / f"{name}.csv"


def _write_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, frame.to_csv(date_format="%Y-%m-%d").encode("utf-8"))


def load_series(series_id, root=None):
    path = series_path(series_id, root)
    if not path.is_file():
        return None
    f = pd.read_csv(path, index_col=0, parse_dates=True)
    s = f.iloc[:, 0].astype(float).dropna().sort_index()
    s.name = series_id
    return s


def load_vintage(series_id, root=None, units=None):
    path = vintage_path(series_id, root, units)
    if not path.is_file():
        return None
    f = pd.read_csv(path, index_col=0, parse_dates=["date", "available"])
    return f.sort_index()


def overlap_days(series_id):
    freq = meta(series_id)[0]
    return {"D": 60, "W": 120, "M": 800, "Q": 1200}.get(freq, 60)


def merge_series(existing, fresh):
    if existing is None or existing.empty:
        return fresh.sort_index()
    if fresh is None or fresh.empty:
        return existing
    out = pd.concat([existing, fresh])
    return out[~out.index.duplicated(keep="last")].sort_index()


def update_series(series_id, full=False, root=None, opener=None, prefer_api=None):
    existing = None if full else load_series(series_id, root)
    start = FULL_START
    if existing is not None and not existing.empty:
        start = (existing.index.max() - pd.Timedelta(days=overlap_days(series_id))).strftime("%Y-%m-%d")
    use_api = has_key() if prefer_api is None else prefer_api
    if use_api:
        fresh = parse_api_rows(fetch_api(series_id, start=start, opener=opener), series_id)["value"].rename(series_id)
        source = "fred-api"
    else:
        fresh = fetch_graph(series_id, start=start, opener=opener)
        source = "fredgraph-keyless"
    merged = merge_series(existing, fresh)
    _write_frame(series_path(series_id, root), merged.to_frame(series_id))
    return merged, source


def _trade_growth_vintages(series_id, root=None, opener=None):
    """Use both quarterly levels from the same first-release vintage."""
    releases = load_vintage(series_id, root)
    existing = load_vintage(series_id, root, "pca")
    if releases is None:
        raise FredError(f"Initial releases are missing for {series_id}.")
    recent = releases["available"] <= period_end(releases.index, "Q") + pd.Timedelta(days=120)
    todo = releases.loc[recent, "available"]
    if existing is not None:
        todo = todo[todo != existing["available"].reindex(todo.index)]
    records = []
    for start in range(0, len(todo), 30):
        block = todo.iloc[start:start + 30]
        prior = block.index - pd.DateOffset(months=3)
        params = {"series_id": series_id, "file_type": "json", "api_key": _api_key(), "output_type": 2,
                  "observation_start": str(prior[0].date()), "observation_end": str(block.index[-1].date()),
                  "vintage_dates": ",".join(block.dt.strftime("%Y-%m-%d").unique())}
        doc = json.loads(_get(FRED_API + "?" + urlencode(params), opener=opener, accept="application/json",
                              safe_name=series_id))
        rows = {row["date"]: row for row in doc.get("observations", [])}
        for day, previous, released in zip(block.index, prior, block):
            key = f"{series_id}_{released:%Y%m%d}"
            try:
                current = float(rows[str(day.date())][key])
                previous_value = float(rows[str(previous.date())][key])
            except (KeyError, ValueError):
                continue
            if current > 0 and previous_value > 0:
                records.append((day, 100 * ((current / previous_value) ** 4 - 1), released))
    fresh = pd.DataFrame(records, columns=["date", "value", "available"]).set_index("date")
    frame = pd.concat([existing, fresh]) if existing is not None and not fresh.empty else (
        existing if existing is not None else fresh)
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def update_vintage(series_id, root=None, opener=None, units=None):
    if units == "pca" and series_id in TRADE_GROWTH_VINTAGES:
        frame = _trade_growth_vintages(series_id, root, opener)
        _write_frame(vintage_path(series_id, root, units), frame)
        return frame
    rows = fetch_api(series_id, opener=opener, output_type=4, units=units)
    frame = parse_api_rows(rows, series_id, vintage=True)
    _write_frame(vintage_path(series_id, root, units), frame)
    return frame


def update_all(full=False, vintages=False, ids=None, root=None, opener=None, log=print):
    root = Path(root or FRED_LONG_DIR)
    manifest = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "mode": None,
                "keyConfigured": has_key(), "series": {}, "errors": {}}
    for sid in ids or SERIES:
        freq, lag, title, group = SERIES.get(sid, ("D", 1, sid, "other"))
        try:
            s, source = update_series(sid, full=full, root=root, opener=opener)
            entry = {"title": title, "group": group, "freq": freq, "lagDays": lag, "source": source,
                     "first": str(s.index.min().date()) if len(s) else None,
                     "last": str(s.index.max().date()) if len(s) else None, "rows": int(len(s))}
            if vintages and has_key() and freq in {"M", "Q", "W"}:
                try:
                    v = update_vintage(sid, root=root, opener=opener)
                    entry["vintage"] = {"rows": int(len(v)),
                                        "firstRelease": str(v["available"].min().date()) if len(v) else None}
                    if sid in VINTAGE_TRANSFORMS:
                        vt = update_vintage(sid, root=root, opener=opener, units=VINTAGE_TRANSFORMS[sid])
                        entry["vintageTransform"] = {"units": VINTAGE_TRANSFORMS[sid], "rows": int(len(vt))}
                except Exception as exc:  # noqa: BLE001
                    entry["vintageError"] = str(exc)[:200]
            manifest["series"][sid] = entry
            if log:
                log(f"{sid:13s} {entry['first']} -> {entry['last']} rows={entry['rows']} ({source})")
        except Exception as exc:  # noqa: BLE001 - one series must not abort the refresh
            manifest["errors"][sid] = str(exc)[:200]
            if log:
                log(f"{sid:13s} ERROR {str(exc)[:120]}")
    manifest["mode"] = "alfred-first-release" if vintages and has_key() else "publication-lag"
    try:
        derived = build_derived(root=root)
        manifest["derived"] = derived
    except Exception as exc:  # noqa: BLE001
        manifest["errors"]["derived"] = str(exc)[:200]
    _write_frame_json(root / "manifest.json", manifest)
    return manifest


def _write_frame_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(obj, indent=2, default=str).encode("utf-8"))


# ---------------------------------------------------------------- derived long series
def chain(old, new):
    """Splice `new` onto `old` by scaling old to new at the first common date (level-ratio chain)."""
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    common = old.index.intersection(new.index)
    if len(common) == 0:
        raise FredError("no overlap to chain")
    pivot = common[min(len(common) - 1, 20)]
    scaled = old * (new.loc[pivot] / old.loc[pivot])
    return pd.concat([scaled[scaled.index < pivot], new[new.index >= pivot]]).sort_index()


def build_derived(root=None):
    out = {}
    major, afe = load_series("DTWEXM", root), load_series("DTWEXAFEGS", root)
    broad_old, broad = load_series("DTWEXB", root), load_series("DTWEXBGS", root)
    if major is not None and afe is not None:
        s = chain(major, afe).rename("DOLLAR_MAJOR_LONG")
        _write_frame(series_path("DOLLAR_MAJOR_LONG", root), s.to_frame())
        out["DOLLAR_MAJOR_LONG"] = {"first": str(s.index.min().date()), "last": str(s.index.max().date()),
                                    "how": "DTWEXM (1973-2019) chained to DTWEXAFEGS"}
    if broad_old is not None and broad is not None:
        s = chain(broad_old, broad).rename("DOLLAR_BROAD_LONG")
        _write_frame(series_path("DOLLAR_BROAD_LONG", root), s.to_frame())
        out["DOLLAR_BROAD_LONG"] = {"first": str(s.index.min().date()), "last": str(s.index.max().date()),
                                    "how": "DTWEXB (1995-2019) chained to DTWEXBGS"}
    return out


DERIVED = {
    "DOLLAR_MAJOR_LONG": ("D", 7, "Dollar vs majors, 1973+ (chained)", "dollar"),
    "DOLLAR_BROAD_LONG": ("D", 7, "Broad dollar, 1995+ (chained)", "dollar"),
}


def meta(series_id):
    return SERIES.get(series_id) or DERIVED.get(series_id) or ("D", 1, series_id, "other")


# ---------------------------------------------------------------- point-in-time panel
def transform(s, units, freq):
    if units == "pch":
        return 100.0 * (s / s.shift(1) - 1.0)
    if units == "chg":
        return s.diff()
    if units == "pca":
        n = {"M": 12, "Q": 4}.get(freq, 1)
        return 100.0 * ((s / s.shift(1)) ** n - 1.0)
    raise FredError(f"unknown units {units}")


def observations(series_id, root=None, prefer_vintage=True, lag_days=None):
    """DataFrame(index=observation date, columns value, available, vintageDated).

    series_id may be "SID:units" (pch/chg/pca) for a growth rate. With ALFRED first releases stored, those
    values and dates are used; observations ALFRED does not carry use the latest-vintage value at the
    publication-lag date."""
    sid, _, units = series_id.partition(":")
    freq = meta(sid)[0]
    s = load_series(sid, root)
    base = None
    if s is not None:
        if units:
            s = transform(s, units, freq).dropna()
        base = pd.DataFrame({"value": s.values, "available": availability(s.index, sid, lag_days)}, index=s.index)
        base["vintageDated"] = False
    if prefer_vintage and lag_days is None:
        v = load_vintage(sid, root, units or None)
        if v is not None and not v.empty:
            out = v[["value", "available"]].copy()
            pe = period_end(out.index, freq)
            lag_avail = availability(out.index, sid)
            # never earlier than the period end (guards odd vintage rows)
            avail = np.maximum(out["available"].values, pe.values)
            # ALFRED starts recording vintages at some date (often the 1990s); older observations all carry that
            # first vintage date. For those, fall back to the publication-lag date.
            stale = avail > (lag_avail + pd.Timedelta(days=90)).values
            out["available"] = np.where(stale, lag_avail.values, avail)
            out["vintageDated"] = ~stale
            if base is not None:
                # stale rows: prefer the latest-vintage value (consistent base) at the lag date
                keep = out[out["vintageDated"]]
                out = pd.concat([keep, base[~base.index.isin(keep.index)]]).sort_index()
            return out
    return base


def asof_series(obs, sessions):
    """Value known at each session date: latest observation with available <= session."""
    sessions = pd.DatetimeIndex(sessions)
    if obs is None or obs.empty:
        return pd.Series(np.nan, index=sessions)
    ordered = obs.sort_values("available")
    # if a later-dated observation became available earlier than an older one (rare), keep the
    # newest observation date among those available.
    ordered = ordered.assign(obs_date=ordered.index)
    ordered["obs_date_max"] = ordered["obs_date"].cummax()
    ordered = ordered[ordered["obs_date"] == ordered["obs_date_max"]]
    pos = np.searchsorted(ordered["available"].values.astype("datetime64[ns]"),
                          sessions.values.astype("datetime64[ns]"), side="right") - 1
    vals = np.where(pos >= 0, ordered["value"].values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=sessions)


def asof_obs_date(obs, sessions):
    """Observation date of the value known at each session (for 'months since' style features)."""
    sessions = pd.DatetimeIndex(sessions)
    if obs is None or obs.empty:
        return pd.Series(pd.NaT, index=sessions)
    ordered = obs.sort_values("available")
    od = pd.Series(ordered.index.values, index=ordered["available"].values).cummax()
    pos = np.searchsorted(od.index.values.astype("datetime64[ns]"), sessions.values.astype("datetime64[ns]"),
                          side="right") - 1
    return pd.Series(np.where(pos >= 0, od.values[np.clip(pos, 0, None)], np.datetime64("NaT")), index=sessions)


def asof_frame(ids, sessions, root=None, prefer_vintage=True, lag_overrides=None):
    lag_overrides = lag_overrides or {}
    cols = {}
    for sid in ids:
        obs = observations(sid, root, prefer_vintage=prefer_vintage, lag_days=lag_overrides.get(sid))
        cols[sid] = asof_series(obs, sessions)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(sessions))


def status(root=None):
    path = Path(root or FRED_LONG_DIR) / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else {"status": "never run"}


def main(argv=None):
    p = argparse.ArgumentParser(description="FRED macro import (point-in-time aware).")
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("update")
    u.add_argument("--full", action="store_true")
    u.add_argument("--vintages", action="store_true")
    u.add_argument("--ids", default="")
    sub.add_parser("status")
    a = p.parse_args(argv)
    if a.cmd == "update":
        ids = [x.strip().upper() for x in a.ids.split(",") if x.strip()] or None
        m = update_all(full=a.full, vintages=a.vintages, ids=ids)
        print(json.dumps({"mode": m["mode"], "keyConfigured": m["keyConfigured"], "series": len(m["series"]),
                          "errors": m["errors"]}, indent=2))
        return 0  # never fail the daily run on a network hiccup
    print(json.dumps(status(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
