"""Dollar-shock detector and event study (alert only).

A sudden move in the ICE-formula DXY proxy (Massive/Macro/dxy_proxy.csv) or in
UUP is treated as a black-swan *alert*, never as a playbook change: no gross
cut, no long freeze, no sleeve permissions. Thresholds are pre-declared (not
searched):

  * 1-day return / rolling 252-session daily sigma  >= 2 in absolute value
  * 5-day return / rolling 252-session std of 5-day returns >= 2 in absolute value

Either rule fires, in either direction (surge = dollar up, weaken = dollar down).

The event study measures what happens AFTER the trigger day, with a sealed
holdout starting 2023-01-03 (same window as dxy_forecast). Overlapping shocks
are clustered (first crossing, 10-session refractory). Multiple testing uses
the project's Holm / Benjamini-Hochberg helper. Intraday lead-lag is estimated
only when minute bars exist.

CLI (from engine/):  python black_swan_dollar.py detect|study
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

# Pre-declared. Do not tune these on the holdout (or at all in this module).
Z_1D = 2.0
Z_5D = 2.0
VOL_WINDOW = 252
VOL_MIN = 126
CLUSTER_GAP = 10          # sessions after a kept event before a new one can start
FORWARD_HORIZONS = (1, 3, 5, 10, 20)
HOLDOUT_START = pd.Timestamp("2023-01-03")
INTRADAY_HORIZONS_MIN = (15, 60, 240)
OPEN_MINUTE = 570         # 09:30 ET

ETFS = ("SPY", "QQQ", "IWM", "EEM", "GLD", "USO", "DBC", "TLT", "UUP")
# Large US names with material foreign exposure; included only if daily_long has them.
MULTINATIONALS = ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JNJ", "KO", "PG", "XOM", "JPM")
STUDY_TICKERS = ETFS + MULTINATIONALS
PRIMARY_TICKER = "SPY"
PRIMARY_HORIZONS = (1, 5)  # close-to-close
PRIMARY_TRADABLE = ("gap", "from_open_1")

# Intraday coverage is short and missing several ETFs; listed for honesty.
INTRADAY_TICKERS = ("SPY", "QQQ", "IWM", "GLD", "TLT", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "XOM")


def _data_dir():
    raw = Path(__file__).resolve().parents[1] / "Massive"
    env = __import__("os").environ.get("PRICES_DATA_DIR")
    return Path(env) if env else raw


def _out_dir(root=None):
    d = Path(root) if root else _data_dir() / "Research" / "black_swan"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _iso_et():
    return datetime.now(ET).strftime("%Y-%m-%dT%H:%M:%S%z")


def _clean(value, digits=6):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return round(value, digits)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)[:19]
    if isinstance(obj, (np.floating, float)):
        return _clean(float(obj))
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def log_holdout(purpose, looked=False, extra=None, root=None):
    """Append to the dollar-shock holdout ledger (America/New_York timestamps)."""
    path = _out_dir(root) / "dollar_holdout_ledger.json"
    ledger = []
    if path.is_file():
        try:
            ledger = json.loads(path.read_text(encoding="utf-8")) or []
        except json.JSONDecodeError:
            ledger = []
    entry = {
        "at": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
        "tz": "America/New_York",
        "purpose": purpose,
        "lookedAtHoldout": bool(looked),
        "holdoutStart": str(HOLDOUT_START.date()),
    }
    if extra:
        entry["extra"] = _json_ready(extra)
    ledger.append(entry)
    path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return entry


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------
def load_dxy_level():
    """ICE-formula DXY proxy. May lag the latest session (FRED publication)."""
    try:
        from macro_data import load_dxy
        series = load_dxy(lag=0)
        if series is not None and len(series.dropna()):
            series = series.astype(float).dropna().sort_index()
            series.name = "dxy"
            return series
    except Exception:
        pass
    path = _data_dir() / "Macro" / "dxy_proxy.csv"
    if not path.is_file():
        return pd.Series(dtype=float, name="dxy")
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    col = "dxy" if "dxy" in frame.columns else frame.columns[0]
    return frame[col].astype(float).dropna().rename("dxy")


def load_uup_level():
    """UUP close, preferring daily_long then overlaying daily_prices for newer days."""
    db = _data_dir() / "prices.sqlite3"
    if not db.is_file():
        return pd.Series(dtype=float, name="UUP")
    parts = []
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
        for table in ("daily_long", "daily_prices"):
            try:
                frame = pd.read_sql_query(
                    f"SELECT date, close FROM {table} WHERE ticker = 'UUP' ORDER BY date",
                    con,
                )
            except Exception:
                continue
            if frame is None or frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"])
            parts.append(frame.set_index("date")["close"].astype(float))
    if not parts:
        return pd.Series(dtype=float, name="UUP")
    series = pd.concat(parts, axis=0)
    series = series[~series.index.duplicated(keep="last")].sort_index().dropna().rename("UUP")
    return series


def features(level, z1=Z_1D, z5=Z_5D, window=VOL_WINDOW, min_periods=VOL_MIN):
    """Causal z-scores. ``level`` is a price/index series (not returns)."""
    level = pd.Series(level).astype(float).dropna().sort_index()
    if level.empty:
        return pd.DataFrame(columns=["level", "ret1", "ret5", "vol", "vol5", "z1", "z5", "shock", "direction"])
    ret1 = level.pct_change()
    ret5 = level.pct_change(5)
    vol = ret1.rolling(window, min_periods=min_periods).std()
    vol5 = ret5.rolling(window, min_periods=min_periods).std()
    z1s = ret1 / vol.replace(0, np.nan)
    z5s = ret5 / vol5.replace(0, np.nan)
    frame = pd.DataFrame({
        "level": level, "ret1": ret1, "ret5": ret5, "vol": vol, "vol5": vol5,
        "z1": z1s, "z5": z5s,
    })
    shock_up = (z1s >= z1) | (z5s >= z5)
    shock_dn = (z1s <= -z1) | (z5s <= -z5)
    frame["shock"] = shock_up.fillna(False) | shock_dn.fillna(False)
    # Direction from the larger |z| that actually breaches.
    def _dir(row):
        if not row["shock"]:
            return None
        candidates = []
        if pd.notna(row["z1"]) and abs(row["z1"]) >= z1:
            candidates.append(("z1", row["z1"]))
        if pd.notna(row["z5"]) and abs(row["z5"]) >= z5:
            candidates.append(("z5", row["z5"]))
        if not candidates:
            return None
        _, z = max(candidates, key=lambda item: abs(item[1]))
        return "surge" if z > 0 else "weaken"
    frame["direction"] = [_dir(row) for _, row in frame.iterrows()]
    return frame


def cluster_events(frame, gap=CLUSTER_GAP):
    """Keep the first crossing of a cluster; require ``gap`` calm sessions after it."""
    events = []
    last_kept = None
    for ts, row in frame.iterrows():
        if not bool(row.get("shock")):
            continue
        if last_kept is not None:
            # session count between last kept and this date
            idx = frame.index
            try:
                i0 = int(idx.get_indexer([last_kept], method="pad")[0])
                i1 = int(idx.get_loc(ts))
            except Exception:
                i0, i1 = -1, 0
            if i0 >= 0 and (i1 - i0) <= gap:
                continue
        rec = {
            "date": pd.Timestamp(ts),
            "direction": row.get("direction"),
            "z1": _clean(row.get("z1")),
            "z5": _clean(row.get("z5")),
            "ret1": _clean(row.get("ret1")),
            "ret5": _clean(row.get("ret5")),
            "level": _clean(row.get("level")),
            "rule": "z1" if pd.notna(row.get("z1")) and abs(float(row["z1"])) >= Z_1D else "z5",
        }
        if pd.notna(row.get("z5")) and abs(float(row["z5"])) >= Z_5D and rec["rule"] == "z1":
            rec["rule"] = "z1+z5" if abs(float(row["z1"])) >= Z_1D else "z5"
        events.append(rec)
        last_kept = pd.Timestamp(ts)
    return events


def detect_latest(asof=None, prev=None, z1=Z_1D, z5=Z_5D):
    """Latest dollar-shock reading. Alert-only; does not touch the playbook.

    Uses UUP when it is more recent than the FRED DXY proxy, and reports both.
    """
    asof_ts = pd.Timestamp(asof) if asof else None
    dxy = load_dxy_level()
    uup = load_uup_level()
    if asof_ts is not None:
        if len(dxy):
            dxy = dxy.loc[:asof_ts]
        if len(uup):
            uup = uup.loc[:asof_ts]
    dxy_f = features(dxy, z1=z1, z5=z5) if len(dxy) else pd.DataFrame()
    uup_f = features(uup, z1=z1, z5=z5) if len(uup) else pd.DataFrame()

    def _last(frame, source):
        if frame is None or frame.empty:
            return None
        row = frame.iloc[-1]
        return {
            "source": source,
            "asof": str(frame.index[-1].date()),
            "level": _clean(row.get("level")),
            "ret1": _clean(row.get("ret1")),
            "ret5": _clean(row.get("ret5")),
            "z1": _clean(row.get("z1")),
            "z5": _clean(row.get("z5")),
            "vol": _clean(row.get("vol")),
            "shock": bool(row.get("shock")),
            "direction": row.get("direction"),
        }

    dxy_last = _last(dxy_f, "dxy_proxy")
    uup_last = _last(uup_f, "UUP")
    # Prefer the most recent observation for the live flag; mention the other.
    candidates = [c for c in (uup_last, dxy_last) if c]
    live = None
    if candidates:
        live = max(candidates, key=lambda c: (c["asof"], 1 if c["source"] == "UUP" else 0))
        # If the freshest print is calm but a lagged DXY print on a later-known
        # date is a shock we have not yet flagged, still surface the DXY shock
        # when it is the newest *shock* and not older than 5 calendar days
        # behind the live print.
        shocks = [c for c in candidates if c.get("shock")]
        if shocks and not live.get("shock"):
            newest_shock = max(shocks, key=lambda c: c["asof"])
            lag = (pd.Timestamp(live["asof"]) - pd.Timestamp(newest_shock["asof"])).days
            if 0 <= lag <= 5:
                live = dict(newest_shock)
                live["note"] = f"Lagged {newest_shock['source']} shock; live print is {candidates[0]['source']} as of {candidates[0]['asof']} (calm)."

    shock = bool(live and live.get("shock"))
    direction = (live or {}).get("direction")
    prev = prev or {}
    is_new = False
    if shock:
        prev_live = prev.get("live") if isinstance(prev.get("live"), dict) else prev
        prev_key = (prev_live.get("asof"), prev_live.get("direction"), prev_live.get("source"))
        new_key = (live.get("asof"), live.get("direction"), live.get("source"))
        is_new = prev_key != new_key
    headline = "No dollar shock."
    if shock:
        side = "surge" if direction == "surge" else "sudden weakening"
        z1v, z5v = live.get("z1"), live.get("z5")
        headline = (
            f"Dollar {side} ({live['source']} as of {live['asof']}: "
            f"1d z={z1v}, 5d z={z5v}). Alert only — no paper-book change."
        )
    return {
        "shock": shock,
        "isNew": is_new,
        "direction": direction,
        "headline": headline,
        "live": live,
        "dxy": dxy_last,
        "uup": uup_last,
        "thresholds": {"z1": z1, "z5": z5, "volWindow": VOL_WINDOW},
        "tradingChange": False,
        "updatedAt": _iso_et(),
        "notes": [
            "Alert only; the playbook is not changed by a dollar shock.",
            "DXY is the FRED ICE-formula proxy and often lags the latest US session.",
            "UUP is the tradable cross-check and is used when it is more recent.",
        ],
    }


def emit_dollar_alert(reading):
    if not reading or not reading.get("shock"):
        return False
    try:
        from alerts_emit import emit
    except Exception:
        return False
    live = reading.get("live") or {}
    return emit({
        "type": "black_swan",
        "severity": "warning",
        "title": "Black swan: dollar shock",
        "text": (reading.get("headline") or "Dollar shock.")[:800],
        "href": "/paper2.html#black-swan",
        "meta": {
            "state": "WATCH",
            "direction": "negative" if live.get("direction") == "surge" else "positive",
            "flag": "WATCH",
            "eventType": "dollar_shock",
            "dollarDirection": live.get("direction"),
            "source": live.get("source"),
            "asof": live.get("asof"),
            "z1": live.get("z1"),
            "z5": live.get("z5"),
            "tradingChange": False,
        },
    })


def log_dollar_chat(reading, path=None):
    if not reading or not reading.get("shock"):
        return False
    try:
        from black_swan import CHAT_DB, CHAT_SCHEMA, iso_et
    except Exception:
        return False
    path = Path(path) if path else CHAT_DB
    live = reading.get("live") or {}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(path))
        db.executescript(CHAT_SCHEMA)
        key = f"black_swan:dollar_shock:{live.get('source')}:{live.get('asof')}:{live.get('direction')}"
        payload = json.dumps(_json_ready(reading))
        db.execute(
            "INSERT OR IGNORE INTO chat_messages (at, role, kind, text, rationale, verdict, data, source_key) "
            "VALUES (?, 'trader', 'black_swan', ?, ?, ?, ?, ?)",
            (reading.get("updatedAt") or iso_et(), reading.get("headline") or "Dollar shock.",
             "Alert only — no paper-book change.", "dollar_shock", payload, key),
        )
        db.commit()
        db.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[black_swan_dollar] chat log skipped: {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------
def load_ohlc(tickers, db=None):
    """Split-adjusted OHLC from daily_long, overlaid with daily_prices for newer days."""
    db = Path(db) if db else _data_dir() / "prices.sqlite3"
    if not db.is_file():
        return {}
    wanted = tuple(tickers)
    out = {}
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
        for table in ("daily_long", "daily_prices"):
            try:
                frame = pd.read_sql_query(
                    f"SELECT ticker, date, open, high, low, close FROM {table} "
                    f"WHERE ticker IN ({','.join('?' * len(wanted))}) ORDER BY date",
                    con, params=wanted,
                )
            except Exception:
                continue
            if frame is None or frame.empty:
                continue
            frame["date"] = pd.to_datetime(frame["date"])
            for ticker, g in frame.groupby("ticker"):
                piece = g.drop(columns=["ticker"]).set_index("date").sort_index()
                if ticker in out:
                    out[ticker] = pd.concat([out[ticker], piece], axis=0)
                    out[ticker] = out[ticker][~out[ticker].index.duplicated(keep="last")].sort_index()
                else:
                    out[ticker] = piece
    return out


def _forward_row(ohlc, event_date):
    """Returns measured *after* the trigger close.

    gap:           next open / event close - 1
    r{h}:          close[t+h] / close[t] - 1
    from_open_{h}: close[t+h] / next open - 1   (tradable if the shock is known by t+1 open)
    """
    if ohlc is None or ohlc.empty:
        return {}
    idx = ohlc.index
    # locate event date or the last session on/before it
    pos = idx.searchsorted(pd.Timestamp(event_date), side="left")
    if pos >= len(idx) or idx[pos] != pd.Timestamp(event_date):
        # require an exact session match so we don't date a shock on a holiday incorrectly
        return {}
    close_t = float(ohlc["close"].iloc[pos])
    if not math.isfinite(close_t) or close_t == 0:
        return {}
    rec = {}
    if pos + 1 < len(idx) and "open" in ohlc.columns:
        nxt_open = float(ohlc["open"].iloc[pos + 1])
        if math.isfinite(nxt_open) and nxt_open != 0:
            rec["gap"] = nxt_open / close_t - 1.0
            rec["nextOpen"] = nxt_open
            rec["nextDate"] = str(idx[pos + 1].date())
    for h in FORWARD_HORIZONS:
        if pos + h < len(idx):
            c_h = float(ohlc["close"].iloc[pos + h])
            if math.isfinite(c_h) and c_h != 0:
                rec[f"r{h}"] = c_h / close_t - 1.0
                if "nextOpen" in rec and rec["nextOpen"]:
                    rec[f"from_open_{h}"] = c_h / rec["nextOpen"] - 1.0
    rec["eventClose"] = close_t
    return rec


def _ttest(values):
    arr = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    n = int(len(arr))
    if n < 3:
        return {"n": n, "mean": _clean(arr.mean()) if n else None, "median": _clean(np.median(arr)) if n else None,
                "sd": _clean(arr.std(ddof=1)) if n > 1 else None, "t": None, "p": None, "se": _clean(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else None}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / math.sqrt(n) if sd else 0.0
    if se == 0:
        tstat = math.inf if mean != 0 else 0.0
        p = 0.0 if mean != 0 else 1.0
    else:
        tstat = mean / se
        try:
            from scipy import stats
            p = float(stats.t.sf(abs(tstat), df=n - 1) * 2)
        except Exception:
            p = float(math.erfc(abs(tstat) / math.sqrt(2.0)))
        tstat = tstat
    return {"n": n, "mean": _clean(mean), "median": _clean(float(np.median(arr))),
            "sd": _clean(sd), "se": _clean(se), "t": _clean(tstat) if math.isfinite(tstat) else ("inf" if tstat > 0 else "-inf"), "p": _clean(p, 6)}


def _multiple_testing(pvalues, alpha=0.05):
    try:
        from macro_event_study import multiple_testing
        return multiple_testing(pvalues, alpha=alpha)
    except Exception:
        p = np.asarray(pvalues, dtype=float)
        n = len(p)
        out = {"n": int(n), "alpha": alpha, "holmAdj": [None] * n, "bhAdj": [None] * n,
               "holmReject": [False] * n, "bhReject": [False] * n, "method": "local"}
        ok = np.where(np.isfinite(p))[0]
        if not len(ok):
            return out
        order = ok[np.argsort(p[ok])]
        m = len(order)
        running = 0.0
        holm = np.ones(m)
        for rank, i in enumerate(order, start=1):
            running = max(running, p[i] * (m - rank + 1))
            holm[rank - 1] = min(1.0, running)
        for i, a in zip(order, holm):
            out["holmAdj"][int(i)] = float(a)
            out["holmReject"][int(i)] = bool(a <= alpha)
        ranked = p[order]
        adj_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
        adj_sorted = np.clip(adj_sorted, 0, 1)
        for i, a in zip(order, adj_sorted):
            out["bhAdj"][int(i)] = float(a)
            out["bhReject"][int(i)] = bool(a <= alpha)
        return out


def summarize_events(events, ohlc_map, tickers, window_label):
    rows = []
    tests = []
    for ticker in tickers:
        ohlc = ohlc_map.get(ticker)
        if ohlc is None:
            continue
        by_dir = {"surge": [], "weaken": []}
        for ev in events:
            rec = _forward_row(ohlc, ev["date"])
            if not rec:
                continue
            by_dir.setdefault(ev["direction"] or "unknown", []).append({**ev, "fwd": rec, "ticker": ticker})
        for direction, items in by_dir.items():
            if not items:
                continue
            keys = ["gap"] + [f"r{h}" for h in FORWARD_HORIZONS] + [f"from_open_{h}" for h in FORWARD_HORIZONS]
            stats = {}
            for key in keys:
                st = _ttest([it["fwd"].get(key) for it in items])
                stats[key] = st
                tests.append({
                    "window": window_label,
                    "ticker": ticker,
                    "direction": direction,
                    "horizon": key,
                    "primary": ticker == PRIMARY_TICKER and (
                        (key in {f"r{h}" for h in PRIMARY_HORIZONS}) or key in PRIMARY_TRADABLE
                    ) and direction in {"surge", "weaken"},
                    **st,
                })
            rows.append({
                "ticker": ticker, "direction": direction, "n": len(items),
                "window": window_label, "stats": stats,
            })
    # Multiple testing within this window, split primary vs secondary.
    def _adjust(subset, family):
        if not subset:
            return []
        mt = _multiple_testing([s.get("p") for s in subset])
        for i, s in enumerate(subset):
            s["family"] = family
            s["holmAdj"] = mt["holmAdj"][i]
            s["bhAdj"] = mt["bhAdj"][i]
            s["holmReject"] = mt["holmReject"][i]
            s["bhReject"] = mt["bhReject"][i]
            s["mtMethod"] = mt.get("method")
        return subset

    primary = _adjust([t for t in tests if t.get("primary")], "primary")
    secondary = _adjust([t for t in tests if not t.get("primary")], "secondary")
    return {"summaries": rows, "tests": primary + secondary}


def _split_events(events):
    pre, hold = [], []
    for ev in events:
        (hold if pd.Timestamp(ev["date"]) >= HOLDOUT_START else pre).append(ev)
    return pre, hold


# ---------------------------------------------------------------------------
# Intraday lead-lag
# ---------------------------------------------------------------------------
def _intraday_db():
    path = _data_dir() / "intraday.sqlite3"
    return path if path.is_file() else None


def describe_intraday():
    db = _intraday_db()
    if not db:
        return {"available": False, "reason": "No Massive/intraday.sqlite3 on disk."}
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT symbol, min(session_date), max(session_date), count(*), min(frequency), max(frequency) "
            "FROM intraday_bars GROUP BY symbol"
        ).fetchall()
    symbols = {r[0]: {"start": r[1], "end": r[2], "n": int(r[3]), "freqMin": r[4], "freqMax": r[5]} for r in rows}
    missing_for_study = [t for t in ("UUP", "EEM", "USO", "DBC") if t not in symbols]
    return {
        "available": True,
        "path": str(db),
        "symbols": sorted(symbols),
        "start": min((s["start"] for s in symbols.values()), default=None),
        "end": max((s["end"] for s in symbols.values()), default=None),
        "barCount": int(sum(s["n"] for s in symbols.values())),
        "missingForStudy": missing_for_study,
        "note": "Minute bars are 09:30–15:59 ET from Schwab. Coverage is recent and does not include UUP, so a same-day *intraday* dollar-shock trigger cannot be dated; we can only measure the next session after a daily DXY/UUP shock.",
    }


def _bar_at_or_before(frame, minute):
    if frame is None or frame.empty:
        return None
    sub = frame[frame["minute"] <= int(minute)]
    if sub.empty:
        return None
    return sub.iloc[-1]


def intraday_after_events(events, tickers=INTRADAY_TICKERS):
    info = describe_intraday()
    if not info.get("available"):
        return {"available": False, "reason": info.get("reason"), "tests": [], "summaries": []}
    start, end = info.get("start"), info.get("end")
    db = _intraday_db()
    usable = [e for e in events if start <= str(pd.Timestamp(e["date"]).date()) <= end]
    if not usable:
        return {"available": True, "coverage": info, "nEventsInCoverage": 0,
                "reason": f"No clustered dollar shocks fall inside the minute-bar window {start}..{end}.",
                "tests": [], "summaries": []}

    wanted = [t for t in tickers if t in (info.get("symbols") or [])]
    with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as con:
        q_marks = ",".join("?" * len(wanted))
        dates = sorted({str(pd.Timestamp(e["date"]).date()) for e in usable})
        # pull a few extra sessions after each event for t+1
        bars = pd.read_sql_query(
            f"SELECT symbol, session_date, minute, open, close FROM intraday_bars "
            f"WHERE symbol IN ({q_marks}) ORDER BY symbol, session_date, minute",
            con, params=wanted,
        )
    if bars.empty:
        return {"available": True, "coverage": info, "reason": "Query returned no bars.", "tests": [], "summaries": []}
    bars["session_date"] = bars["session_date"].astype(str)
    sessions_by_sym = {sym: sorted(g["session_date"].unique()) for sym, g in bars.groupby("symbol")}

    summaries = []
    tests = []
    for ticker in wanted:
        sess = sessions_by_sym.get(ticker) or []
        tbars = bars[bars["symbol"] == ticker]
        for direction in ("surge", "weaken"):
            bucket = {f"m{h}": [] for h in INTRADAY_HORIZONS_MIN}
            bucket["on_open"] = []
            n_used = 0
            for ev in usable:
                if ev.get("direction") != direction:
                    continue
                d0 = str(pd.Timestamp(ev["date"]).date())
                # next session strictly after the event date
                nxt = next((d for d in sess if d > d0), None)
                if nxt is None:
                    continue
                day = tbars[tbars["session_date"] == nxt]
                if day.empty:
                    continue
                first = day.iloc[0]
                open_px = float(first["open"] if first["minute"] <= OPEN_MINUTE + 5 else first["close"])
                if not math.isfinite(open_px) or open_px == 0:
                    continue
                # overnight gap uses first print vs prior daily close when we have it later; here from first open vs last close of event day if present
                ev_day = tbars[tbars["session_date"] == d0]
                if not ev_day.empty:
                    prior_close = float(ev_day.iloc[-1]["close"])
                    if math.isfinite(prior_close) and prior_close:
                        bucket["on_open"].append(open_px / prior_close - 1.0)
                n_used += 1
                for h in INTRADAY_HORIZONS_MIN:
                    bar = _bar_at_or_before(day, OPEN_MINUTE + h)
                    if bar is None:
                        continue
                    px = float(bar["close"])
                    if math.isfinite(px) and open_px:
                        bucket[f"m{h}"].append(px / open_px - 1.0)
            stats = {k: _ttest(v) for k, v in bucket.items()}
            summaries.append({"ticker": ticker, "direction": direction, "n": n_used, "stats": stats})
            for key, st in stats.items():
                tests.append({
                    "ticker": ticker, "direction": direction, "horizon": key,
                    "primary": ticker == PRIMARY_TICKER and key in {"m15", "m60", "on_open"},
                    **st,
                })
    primary = [t for t in tests if t.get("primary")]
    secondary = [t for t in tests if not t.get("primary")]
    def _adj(subset, family):
        if not subset:
            return []
        mt = _multiple_testing([s.get("p") for s in subset])
        for i, s in enumerate(subset):
            s["family"] = family
            s["holmAdj"] = mt["holmAdj"][i]
            s["bhAdj"] = mt["bhAdj"][i]
            s["holmReject"] = mt["holmReject"][i]
            s["bhReject"] = mt["bhReject"][i]
        return subset
    return {
        "available": True,
        "coverage": info,
        "nEventsInCoverage": len(usable),
        "eventDates": [str(pd.Timestamp(e["date"]).date()) + " " + (e.get("direction") or "") for e in usable],
        "summaries": summaries,
        "tests": _adj(primary, "intraday_primary") + _adj(secondary, "intraday_secondary"),
        "honesty": (
            f"Minute bars cover {info.get('start')} to {info.get('end')} only, and there are no UUP/EEM/USO/DBC "
            f"minute bars, so the trigger is still the *daily* DXY or UUP close. We measure the next session's "
            f"open and the first 15/60/240 minutes after 09:30 ET. n={len(usable)} clustered shocks in that window "
            "— too few for tight inference."
        ),
    }


def _honesty(dxy, ohlc_map, intra):
    dxy_start = str(dxy.index[0].date()) if len(dxy) else None
    dxy_end = str(dxy.index[-1].date()) if len(dxy) else None
    spy = ohlc_map.get("SPY")
    notes = [
        "Thresholds (2-sigma 1-day and 2-sigma 5-day) were pre-declared and were not searched on either window.",
        f"Holdout starts {HOLDOUT_START.date()} (same sealed window as the DXY forecast work). Primary tests were pre-registered: SPY 1d and 5d close-to-close, plus the next-open gap and the tradable open-to-close day, split by surge vs weaken.",
        "A prior daily-horizon study already found that a lagged dollar level does not usefully predict stocks. This study only asks whether a *rare 2-sigma shock* leaves a short tradable lag after the trigger is known.",
        f"The DXY series is the FRED ICE-formula proxy ({dxy_start} to {dxy_end}), not ICE futures, and typically lags the latest US session. Live alerts also watch UUP, which is more timely.",
        "Equity paths use Schwab daily_long (split-adjusted) starting 2006-11-24 for names still trading, overlaid with daily_prices for any newer session. Delisted names are missing, so earlier crises lean a little optimistic.",
        "Overlapping shocks are collapsed to the first crossing, with a 10-session refractory period, so a multi-day surge counts once.",
        "Many ticker × horizon tests are secondary and are Holm / BH-FDR adjusted inside each window. Unadjusted p-values will look more significant than they are.",
    ]
    if spy is not None and len(spy):
        notes.append(f"SPY daily_long/prices coverage used here: {spy.index[0].date()} to {spy.index[-1].date()} ({len(spy)} sessions). Dollar shocks before the equity start have no forward equity path.")
    if intra and intra.get("honesty"):
        notes.append(intra["honesty"])
    elif not (intra and intra.get("available")):
        notes.append("No intraday minute bars were available, so a 15m/60m/240m lead-lag test could not be run.")
    return notes


def _write_markdown(report, path):
    lines = ["# Dollar shock event study", ""]
    lines.append(f"Generated {report.get('generatedAt')} (America/New_York). Alert only — no paper-book change.")
    lines.append("")
    lines.append("## Current reading")
    live = (report.get("latest") or {}).get("live") or {}
    lines.append(f"- {(report.get('latest') or {}).get('headline')}")
    if live:
        lines.append(f"- Source {live.get('source')} as of {live.get('asof')}: 1d z={live.get('z1')}, 5d z={live.get('z5')}, 1d return={live.get('ret1')}.")
    lines.append("")
    lines.append("## Sample")
    ev = report.get("events") or {}
    lines.append(f"- Clustered shocks: {ev.get('n')} (surge {ev.get('nSurge')}, weaken {ev.get('nWeaken')}); pre-holdout {ev.get('nPre')}, holdout {ev.get('nHoldout')}.")
    lines.append(f"- DXY range: {report.get('dxyStart')} to {report.get('dxyEnd')}. Equity start typically 2006-11-24. Holdout starts {HOLDOUT_START.date()}.")
    lines.append("")
    def _block(title, tests):
        lines.append(f"## {title}")
        prim = [t for t in tests if t.get("family", "").startswith("primary") or t.get("primary")]
        if not prim:
            prim = tests[:8]
        lines.append("| ticker | dir | horizon | n | mean | t | p | BH adj | Holm adj | reject BH |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|")
        for t in prim:
            lines.append(
                f"| {t.get('ticker')} | {t.get('direction')} | {t.get('horizon')} | {t.get('n')} | "
                f"{t.get('mean')} | {t.get('t')} | {t.get('p')} | {t.get('bhAdj')} | {t.get('holmAdj')} | "
                f"{t.get('bhReject')} |"
            )
        lines.append("")
    pre = report.get("preHoldout") or {}
    hold = report.get("holdout") or {}
    _block("Pre-holdout primary tests (estimation window)", pre.get("tests") or [])
    _block("Holdout primary tests (sealed confirmation)", hold.get("tests") or [])
    intra = report.get("intraday") or {}
    lines.append("## Intraday lead-lag")
    if intra.get("available") and intra.get("tests"):
        lines.append(intra.get("honesty") or "")
        lines.append("")
        _block("Intraday (next session after a daily shock)", intra.get("tests") or [])
    else:
        lines.append(intra.get("reason") or intra.get("honesty") or "Not available.")
        lines.append("")
    lines.append("## Honesty")
    for n in report.get("honesty") or []:
        lines.append(f"- {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(root=None, tickers=None):
    root = _out_dir(root)
    dxy = load_dxy_level()
    if dxy.empty:
        raise RuntimeError("DXY proxy is missing. Run python macro_data.py update first.")
    feat = features(dxy)
    events = cluster_events(feat)
    pre, hold = _split_events(events)
    log_holdout("dollar_shock event-study setup; thresholds pre-declared, no search", looked=False,
                extra={"nEvents": len(events), "nPre": len(pre), "nHoldoutReserved": len(hold)}, root=root)
    tickers = tuple(tickers) if tickers else STUDY_TICKERS
    ohlc_map = load_ohlc(tickers)
    # Estimation window — pre-holdout only.
    pre_res = summarize_events(pre, ohlc_map, tickers, "pre-holdout")
    # Sealed confirmation — one logged look.
    log_holdout("dollar_shock holdout evaluation of pre-declared forward-return tests", looked=True,
                extra={"nHoldoutEvents": len(hold)}, root=root)
    hold_res = summarize_events(hold, ohlc_map, tickers, "holdout")
    intra = intraday_after_events(events)
    latest = detect_latest()
    report = {
        "generatedAt": _iso_et(),
        "thresholds": {"z1": Z_1D, "z5": Z_5D, "volWindow": VOL_WINDOW, "clusterGap": CLUSTER_GAP},
        "holdoutStart": str(HOLDOUT_START.date()),
        "dxyStart": str(dxy.index[0].date()),
        "dxyEnd": str(dxy.index[-1].date()),
        "dxyRows": int(len(dxy)),
        "tradingChange": False,
        "events": {
            "n": len(events),
            "nSurge": int(sum(1 for e in events if e.get("direction") == "surge")),
            "nWeaken": int(sum(1 for e in events if e.get("direction") == "weaken")),
            "nPre": len(pre),
            "nHoldout": len(hold),
            "dates": [{"date": str(e["date"].date()), **{k: e[k] for k in e if k != "date"}} for e in events],
        },
        "coverage": {t: {"start": str(v.index[0].date()), "end": str(v.index[-1].date()), "n": int(len(v))}
                     for t, v in ohlc_map.items() if v is not None and len(v)},
        "preHoldout": pre_res,
        "holdout": hold_res,
        "intraday": intra,
        "latest": latest,
        "honesty": _honesty(dxy, ohlc_map, intra),
        "verdict": _verdict(pre_res, hold_res, intra),
    }
    (root / "dollar_shock.json").write_text(json.dumps(_json_ready(report), indent=2), encoding="utf-8")
    _write_markdown(report, root / "dollar_shock.md")
    return report


def _verdict(pre, hold, intra):
    """Plain-language answer: is there a tradable lag after a 2-sigma dollar shock?"""
    def _pick(res, horizon, direction, ticker="SPY"):
        for t in res.get("tests") or []:
            if t.get("ticker") == ticker and t.get("direction") == direction and t.get("horizon") == horizon:
                return t
        return None

    bits = []
    for window, res in (("pre-holdout", pre), ("holdout", hold)):
        for direction in ("surge", "weaken"):
            gap = _pick(res, "gap", direction)
            r1 = _pick(res, "r1", direction)
            fo1 = _pick(res, "from_open_1", direction)
            def _fmt(t, label):
                if not t or t.get("mean") is None:
                    return f"{label} n/a"
                star = " (BH significant)" if t.get("bhReject") else ""
                return f"{label} mean {t['mean']:.4f} n={t['n']} p={t.get('p')}{star}"
            bits.append(
                f"{window} {direction}: {_fmt(gap, 'next-open gap')}; {_fmt(r1, 'close-to-close 1d')}; "
                f"{_fmt(fo1, 'tradable open-to-close')}."
            )
    lag_yes = False
    lag_maybe = False
    for res in (pre, hold):
        for direction in ("surge", "weaken"):
            # A tradable lag requires the NEXT open-to-close (or later) to be significant,
            # not merely the overnight gap (which you may not catch if DXY is known only after the close).
            fo1 = _pick(res, "from_open_1", direction)
            if fo1 and fo1.get("bhReject"):
                lag_yes = True
            elif fo1 and fo1.get("p") is not None and fo1["p"] < 0.05 and not fo1.get("bhReject"):
                lag_maybe = True
    intra_note = None
    if intra and intra.get("available") and intra.get("tests"):
        sig = [t for t in intra["tests"] if t.get("ticker") == "SPY" and t.get("bhReject")]
        if sig:
            lag_yes = True
            intra_note = "Intraday SPY tests had at least one BH-significant horizon after a daily shock."
        else:
            intra_note = "Intraday SPY tests in the short minute-bar window were not significant after BH/Holm adjustment."
    elif intra and not intra.get("available"):
        intra_note = intra.get("reason") or "No intraday bars."
    else:
        intra_note = (intra or {}).get("reason") or (intra or {}).get("honesty")

    if lag_yes:
        answer = "There is some evidence of a short tradable lag after a 2-sigma dollar shock, but sample sizes are small and it should be treated as tentative."
    elif lag_maybe:
        answer = "Raw p-values hint at a short lag after a 2-sigma dollar shock, but nothing survives multiple-testing correction. Daily lagged dollar already failed as a predictor; do not trade this yet."
    else:
        answer = "No reliable tradable lag showed up after a 2-sigma dollar shock: once the trigger day is complete, next-session open-to-close moves in SPY are not significant after multiple-testing correction. That fits the earlier finding that a lagged dollar does not predict stocks on a daily horizon."
    return {"answer": answer, "details": bits, "intraday": intra_note}


def attach_to_poll(state, asof=None, emit=True, root=None):
    """Called from black_swan.poll. Never mutates playbook fields."""
    prev = (state or {}).get("dollarShock") or {}
    reading = detect_latest(asof=asof, prev=prev)
    if state is not None:
        state["dollarShock"] = reading
        # Guarantee the playbook is not quietly altered by this path.
        if "policy" in state and reading.get("shock"):
            # Leave policy as-is; only annotate.
            policy = dict(state.get("policy") or {})
            policy["dollarShockIgnoredForTrading"] = True
            state["policy"] = policy
    root = _out_dir(root)
    if reading.get("shock") and reading.get("isNew"):
        append_path = root / "events.jsonl"
        try:
            with append_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "at": _iso_et(), "state": (state or {}).get("state"),
                    "eventType": "dollar_shock", "flag": "WATCH",
                    "headline": reading.get("headline"), "dollarShock": _json_ready(reading.get("live")),
                    "tradingChange": False,
                }, default=str) + "\n")
        except Exception:
            pass
        if emit:
            emit_dollar_alert(reading)
            log_dollar_chat(reading)
    return reading


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["detect", "study"], nargs="?", default="detect")
    parser.add_argument("--asof")
    args = parser.parse_args(argv)
    if args.command == "study":
        report = run_study()
        print(json.dumps(_json_ready({
            "ok": True,
            "generatedAt": report.get("generatedAt"),
            "events": {k: report["events"][k] for k in ("n", "nSurge", "nWeaken", "nPre", "nHoldout")},
            "verdict": report.get("verdict"),
            "latest": report.get("latest"),
            "honesty": report.get("honesty"),
            "paths": {
                "json": str(_out_dir() / "dollar_shock.json"),
                "md": str(_out_dir() / "dollar_shock.md"),
                "ledger": str(_out_dir() / "dollar_holdout_ledger.json"),
            },
        }), indent=2))
        return 0
    print(json.dumps(_json_ready(detect_latest(asof=args.asof)), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
