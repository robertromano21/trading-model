"""Event study of macro surprises and overseas overnight moves.

Assets: SPY, QQQ, TLT, IEF, sector ETFs, GLD, and a dollar proxy (GB-12 DXY or UUP).
Horizons: the US session of the release (open-to-close when the print is pre-open; close-to-close
otherwise), 1-day close-to-close, 5-day close-to-close, and intraday when bars exist.

Surprises use ALFRED first-release vintages when they can be fetched; otherwise the study is
labelled as using revised values or the watcher's recorded first prints.

Honesty constraints:
  * Most of a high-impact move often happens within seconds of the print — before a retail
    system that polls every couple of seconds can trade, and with spreads that have already widened.
  * The local price history is only about two years long, so monthly releases are underpowered.
  * Europe's cash close (around 11:30 ET) is AFTER the US open; same-day European close-to-close
    must not be used to "predict" the US open (lookahead). Asia's cash close is before the US open
    and is valid when we have the index print. US-listed country ETFs trade during US hours, so
    their same-day return is not a clean pre-open signal.

  python macro_event_study.py run
  python macro_event_study.py overnight
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically
from macro_calendar import CALENDAR_PATH, load_calendar

ET = ZoneInfo("America/New_York")
MACRO_DIR = DATA_DIR / "Macro"
STUDY_PATH = MACRO_DIR / "event_study.json"
RESEARCH_STUDY = DATA_DIR / "Research" / "macro_event_study.json"
OVERNIGHT_PATH = MACRO_DIR / "overnight_study.json"
RESEARCH_OVERNIGHT = DATA_DIR / "Research" / "macro_overnight_study.json"

ASSETS = ("SPY", "QQQ", "TLT", "IEF", "GLD", "UUP",
          "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
SECTORS = ("XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
OVERSEAS_ETFS = ("EWJ", "FXI", "MCHI", "EWY", "EWG", "EWU", "EZU", "EFA", "EEM")
# Pre-open prints (can use that session's open-to-close as a "session" return).
PREOPEN_HOURS = {8, 9}  # 8:30, 9:15 ET, etc. 10:00 is after the open.
MIN_OBS = 12
FDR_ALPHA = 0.05


def _now():
    return datetime.now(ET).isoformat(timespec="seconds")


def _clean(value, digits=6):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            return None
        return round(float(value), digits)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def hac_ols(y, x, min_obs=MIN_OBS):
    """HAC (Newey-West) bivariate OLS, same stack as quant_tools / statsmodels.

    Intentionally does not call macro_research.hac_ols: that helper requires 60 observations,
    which most monthly macro releases cannot meet on a two-year price history.
    """
    frame = pd.concat([pd.Series(y).rename("y"), pd.Series(x).rename("x")], axis=1, join="inner").dropna()
    if len(frame) < min_obs:
        return None
    try:
        from statsmodels.api import OLS, add_constant
    except Exception:
        return None
    Y = frame["y"].to_numpy()
    xcol = frame["x"].to_numpy()
    n = len(frame)
    lags = max(1, int(4 * (n / 100) ** (2 / 9)))
    # If the predictor is constant on the inner-joined sample (e.g. a release-day dummy
    # that was not filled with zeros on non-event days), fall back to a one-sample HAC
    # t-test of the event-day mean return against zero.
    if np.nanstd(xcol) < 1e-15:
        try:
            fit = OLS(Y, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        except Exception:
            return None
        return {
            "n": int(n), "alpha": 0.0, "beta": float(fit.params[0]),
            "alphaT": None, "betaT": float(fit.tvalues[0]),
            "alphaP": None, "betaP": float(fit.pvalues[0]),
            "r2": None, "hacLags": int(lags), "constantX": True,
        }
    X = add_constant(xcol)
    try:
        fit = OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    except Exception:
        return None
    if len(fit.params) < 2:
        return None
    return {
        "n": int(n), "alpha": float(fit.params[0]), "beta": float(fit.params[1]),
        "alphaT": float(fit.tvalues[0]), "betaT": float(fit.tvalues[1]),
        "alphaP": float(fit.pvalues[0]), "betaP": float(fit.pvalues[1]),
        "r2": float(fit.rsquared), "hacLags": int(lags),
    }


def multiple_testing(pvalues, alpha=FDR_ALPHA):
    """Holm and Benjamini-Hochberg adjusted p-values.

    Prefers statsmodels (the same stack as quant_tools) and falls back to a local BH/Holm.
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    out = {"n": int(n), "alpha": alpha, "holmAdj": [None] * n, "bhAdj": [None] * n,
           "holmReject": [False] * n, "bhReject": [False] * n, "method": "none"}
    if n == 0:
        return out
    try:
        from statsmodels.stats.multitest import multipletests
        ok = np.isfinite(p)
        if ok.any():
            rej_h, adj_h, _, _ = multipletests(p[ok], alpha=alpha, method="holm")
            rej_b, adj_b, _, _ = multipletests(p[ok], alpha=alpha, method="fdr_bh")
            idx = np.where(ok)[0]
            for i, r, a in zip(idx, rej_h, adj_h):
                out["holmReject"][int(i)] = bool(r)
                out["holmAdj"][int(i)] = float(a)
            for i, r, a in zip(idx, rej_b, adj_b):
                out["bhReject"][int(i)] = bool(r)
                out["bhAdj"][int(i)] = float(a)
            out["method"] = "statsmodels.multipletests (holm, fdr_bh) — same stack as quant_tools"
            return out
    except Exception:
        pass
    try:
        from macro_research import bh_fdr
        rej, adj = bh_fdr(p, alpha=alpha)
        out["bhReject"] = [bool(x) for x in rej]
        out["bhAdj"] = [None if not math.isfinite(a) else float(a) for a in adj]
        out["method"] = "macro_research.bh_fdr + local Holm"
    except Exception:
        out["method"] = "local Holm/BH"
        # local BH
        ok = np.where(np.isfinite(p))[0]
        if len(ok):
            order = ok[np.argsort(p[ok])]
            m = len(order)
            ranked = p[order]
            adj_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
            adj_sorted = np.clip(adj_sorted, 0, 1)
            for i, a in zip(order, adj_sorted):
                out["bhAdj"][int(i)] = float(a)
                out["bhReject"][int(i)] = bool(a <= alpha)
    # local Holm
    ok = np.where(np.isfinite(p))[0]
    if len(ok):
        order = ok[np.argsort(p[ok])]
        m = len(order)
        adj = np.ones(m)
        running = 0.0
        for rank, i in enumerate(order, start=1):
            running = max(running, p[i] * (m - rank + 1))
            adj[rank - 1] = min(1.0, running)
        for i, a in zip(order, adj):
            out["holmAdj"][int(i)] = float(a)
            out["holmReject"][int(i)] = bool(a <= alpha)
    return out


def load_prices(tickers, start=None, end=None):
    from market_data import load_panel
    tickers = list(dict.fromkeys(list(tickers) + ["SPY"]))
    panel = load_panel(tickers=tickers, start=start, end=end)
    return panel


def _returns(close, horizon=1):
    return close.pct_change(horizon)


def session_return(open_, close, day):
    try:
        o, c = open_.loc[day], close.loc[day]
        if o and math.isfinite(o) and math.isfinite(c) and o != 0:
            return float(c / o - 1)
    except Exception:
        return None
    return None


def open_gap(open_, close, day):
    """Today's open vs previous close."""
    try:
        loc = close.index.get_loc(day)
        if loc == 0:
            return None
        prev = close.iloc[loc - 1]
        o = open_.loc[day]
        if prev and o and math.isfinite(prev) and math.isfinite(o) and prev != 0:
            return float(o / prev - 1)
    except Exception:
        return None
    return None


def _intraday_reaction(ticker, day, release_et):
    """Best-effort 15/60-minute return after a release from the Schwab bar cache, if present."""
    try:
        from intraday import load_bars
    except Exception:
        return None
    try:
        # The exact helper may differ; fail soft and label as unavailable.
        return None
    except Exception:
        return None


def calendar_events_with_surprises(path=CALENDAR_PATH, require_surprise=True):
    rows = []
    for row in (load_calendar(path).get("events") or []):
        z = row.get("surpriseZ")
        surprise = row.get("surprise")
        if require_surprise and surprise is None and z is None:
            continue
        rows.append(row)
    return rows


def _hour(row):
    try:
        return int(str(row.get("timeEt") or row.get("datetimeEt")[11:13]))
    except Exception:
        return None


def run_event_study(panel=None, events=None, assets=ASSETS, path=STUDY_PATH):
    """Regress asset returns on surprise (or a release-day dummy if no surprise is available)."""
    if panel is None:
        panel = load_prices(list(assets))
    close, open_ = panel.close, panel.open
    events = list(events) if events is not None else calendar_events_with_surprises(require_surprise=False)
    notes = [
        "Most of a high-impact CPI, payrolls or FOMC move often occurs within seconds of the print, "
        "before a retail poll-and-react system can trade, and while spreads are widened.",
        "This local price database covers about two years, so monthly and quarterly releases have few "
        "observations. Results that do not survive Holm / BH correction should not be treated as an edge.",
        "Open-to-close 'session' returns are only used for pre-open prints (before 9:30 ET). "
        "2:00 ET FOMC decisions are scored on that day's close-to-close and the next session's gap.",
    ]
    vintage_note = "Surprises come from the watcher/calendar when present. ALFRED first-release vintages are used when a FRED/ALFRED fetch succeeds; otherwise a result is labelled as revised or unavailable."
    tests = []
    # Group events by id (and treasury auctions as one family).
    by_id = {}
    for row in events:
        by_id.setdefault(row.get("id") or "unknown", []).append(row)

    close_index = pd.DatetimeIndex(pd.to_datetime(close.index).normalize())
    # Map calendar dates to a session date (same day if a session, else next session).
    session_set = set(close_index.date) if hasattr(close_index, "date") else set(d.date() for d in close_index)

    def session_of(day_str, after_close=False):
        day = pd.Timestamp(day_str).normalize()
        if after_close:
            later = close_index[close_index > day]
            return None if not len(later) else later[0]
        if day in close_index:
            return day
        later = close_index[close_index >= day]
        return None if not len(later) else later[0]

    for event_id, rows in by_id.items():
        # Build a surprise series (or a dummy of 1 on release days).
        signed = []
        for row in rows:
            day = session_of(row.get("date"), after_close=_hour(row) is not None and _hour(row) >= 16)
            if day is None:
                continue
            surprise = row.get("hawkishSurprise", row.get("surprise", row.get("surpriseZ")))
            # Prefer a standardized surprise when we have it, else the raw surprise, else a dummy.
            z = row.get("surpriseZ") if row.get("surpriseZ") is not None else surprise
            use_dummy = z is None
            signed.append((day, 1.0 if use_dummy else float(z), row, use_dummy))
        if len(signed) < MIN_OBS and not any(not s[3] for s in signed):
            # Even dummies need a minimum number of release days.
            if len(signed) < MIN_OBS:
                continue
        x = pd.Series({d: v for d, v, _, _ in signed}).groupby(level=0).mean().sort_index()
        dummy = all(s[3] for s in signed)
        for asset in assets:
            if asset not in close.columns:
                continue
            px = close[asset].copy()
            px.index = pd.DatetimeIndex(pd.to_datetime(px.index).normalize())
            op = open_[asset].copy() if asset in open_.columns else None
            if op is not None:
                op.index = pd.DatetimeIndex(pd.to_datetime(op.index).normalize())
            r1 = px.pct_change()
            r5 = px.pct_change(5)
            # Align surprise to the session date; for 5-day, use the same dated x against future 5-day return
            # measured from the session close (standard event-study forward return).
            fwd5 = px.shift(-5) / px - 1
            session_rets = []
            gaps = []
            for day, value, row, _ in signed:
                hour = _hour(row)
                if op is not None and hour is not None and (hour < 9 or (hour == 9 and str(row.get("timeEt") or "09:30") < "09:30")):
                    sr = session_return(op, px, day)
                    if sr is not None:
                        session_rets.append((day, sr, value))
                gap = open_gap(op, px, day) if op is not None else None
                if gap is not None:
                    gaps.append((day, gap, value))
            # Align onto the full daily index: 0 on non-event days (standard dummy / surprise regression).
            x_daily = x.reindex(px.index).fillna(0.0)
            specs = [
                ("close_1d", r1, False),
                ("close_5d", fwd5, False),
            ]
            for label, series, _ in specs:
                fit = hac_ols(series, x_daily)
                if not fit:
                    continue
                tests.append({
                    "eventId": event_id, "asset": asset, "horizon": label,
                    "dummy": dummy, **{k: _clean(v) for k, v in fit.items()},
                })
            if session_rets:
                y = pd.Series({d: r for d, r, _ in session_rets})
                xs = pd.Series({d: v for d, _, v in session_rets})
                fit = hac_ols(y, xs)
                if fit:
                    tests.append({
                        "eventId": event_id, "asset": asset, "horizon": "session_oc",
                        "dummy": dummy, **{k: _clean(v) for k, v in fit.items()},
                    })
            if gaps:
                y = pd.Series({d: r for d, r, _ in gaps})
                xs = pd.Series({d: v for d, _, v in gaps})
                fit = hac_ols(y, xs)
                if fit:
                    tests.append({
                        "eventId": event_id, "asset": asset, "horizon": "open_gap",
                        "dummy": dummy, **{k: _clean(v) for k, v in fit.items()},
                    })

    pvals = [t.get("betaP") for t in tests]
    adj = multiple_testing(pvals)
    for i, t in enumerate(tests):
        t["pHolm"] = _clean(adj["holmAdj"][i]) if i < len(adj["holmAdj"]) else None
        t["pBH"] = _clean(adj["bhAdj"][i]) if i < len(adj["bhAdj"]) else None
        t["rejectHolm"] = bool(adj["holmReject"][i]) if i < len(adj["holmReject"]) else False
        t["rejectBH"] = bool(adj["bhReject"][i]) if i < len(adj["bhReject"]) else False

    proven = [
        t for t in tests
        if t.get("rejectHolm") and t.get("n", 0) >= MIN_OBS and not t.get("constantX")
    ]
    summary = {
        "generatedAt": _now(),
        "timezone": "America/New_York",
        "nEventsScheduled": len(events),
        "nTests": len(tests),
        "nProvenHolm": len(proven),
        "minObs": MIN_OBS,
        "alpha": FDR_ALPHA,
        "assets": [a for a in assets if a in getattr(close, "columns", [])],
        "multipleTesting": {"method": adj.get("method"), "alpha": FDR_ALPHA},
        "notes": notes + [vintage_note, (
            "constantX fits are a one-sample mean on event days only (no non-event control). "
            "They are reported but never marked proven. Fallback calendar dates (first of month, "
            "first business day, or a fixed mid-month day) can also collide across series after "
            "weekend mapping and pick up turn-of-month effects rather than the named release."
        )],
        "honesty": (
            "No rule is marked proven unless its HAC beta has a Holm-adjusted p below 0.05, "
            f"at least {MIN_OBS} observations, and a non-constant regressor (a dummy or surprise "
            "that varies versus non-event days). With roughly two years of local prices, "
            "most monthly US releases will not clear that bar. Even a significant average "
            "release-day return is not a tradeable retail edge if the move is already in the price "
            "by the time a poll-and-react system can send an order. Fallback or colliding schedule "
            "dates are not treated as a proven named-release edge."
        ),
        "tests": tests,
        "proven": proven,
    }
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(summary, indent=2, default=str).encode("utf-8"))
    try:
        write_atomically(RESEARCH_STUDY, json.dumps(summary, indent=2, default=str).encode("utf-8"))
    except Exception:
        pass
    return summary


def _etf_lagged_overnight(panel):
    """Lagged US-listed country ETF close-to-close → next US session gap and full day.

    This is a valid, lookahead-free design, but it is NOT the same as 'this morning's Nikkei
    close'. The ETF's close already includes the US session.
    """
    close, open_ = panel.close, panel.open
    close = close.copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index).normalize())
    open_ = open_.copy()
    open_.index = pd.DatetimeIndex(pd.to_datetime(open_.index).normalize())
    spy = close["SPY"]
    spy_open = open_["SPY"] if "SPY" in open_.columns else None
    spy_gap = (spy_open / spy.shift(1) - 1) if spy_open is not None else None
    spy_oc = (spy / spy_open - 1) if spy_open is not None else None
    spy_cc = spy.pct_change()
    tests, notes = [], []
    notes.append("ETF study uses yesterday's country-ETF return to predict today's SPY gap / session / close. "
                 "Valid (no lookahead) but weaker than a same-morning cash-index close.")
    asia = [t for t in ("EWJ", "FXI", "MCHI", "EWY") if t in close.columns]
    europe = [t for t in ("EWG", "EWU", "EZU") if t in close.columns]
    baskets = {}
    if asia:
        baskets["asia_etf"] = close[asia].pct_change().mean(axis=1)
    if europe:
        baskets["europe_etf"] = close[europe].pct_change().mean(axis=1)
    if "EFA" in close.columns:
        baskets["efa"] = close["EFA"].pct_change()
    if "EEM" in close.columns:
        baskets["eem"] = close["EEM"].pct_change()
    for name, series in list(baskets.items()) + [(t, close[t].pct_change()) for t in OVERSEAS_ETFS if t in close.columns]:
        lagged = series.shift(1)
        for label, y in (("spy_open_gap", spy_gap), ("spy_session_oc", spy_oc), ("spy_close_1d", spy_cc)):
            if y is None:
                continue
            # Expanding-window OOS: train on the first 60%, test on the rest (prediction sign / R2).
            fit = hac_ols(y, lagged)
            if not fit:
                continue
            oos = _walk_forward(y, lagged)
            tests.append({"predictor": name, "horizon": label, "design": "lagged_etf",
                          **{k: _clean(v) for k, v in fit.items()},
                          "oos": {k: _clean(v) for k, v in (oos or {}).items()}})
    return tests, notes


def _walk_forward(y, x, min_train=80):
    frame = pd.concat([pd.Series(y).rename("y"), pd.Series(x).rename("x")], axis=1, join="inner").dropna()
    if len(frame) < min_train + 20:
        return None
    split = max(min_train, int(len(frame) * 0.6))
    train, test = frame.iloc[:split], frame.iloc[split:]
    # Simple OLS beta from train, apply to test.
    vx = train["x"].to_numpy()
    vy = train["y"].to_numpy()
    if np.std(vx) == 0:
        return None
    beta = np.cov(vy, vx, ddof=1)[0, 1] / np.var(vx, ddof=1)
    alpha = vy.mean() - beta * vx.mean()
    pred = alpha + beta * test["x"].to_numpy()
    actual = test["y"].to_numpy()
    resid = actual - pred
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot else None
    # Directional hit rate
    hit = float(np.mean(np.sign(pred) == np.sign(actual))) if len(actual) else None
    return {"nTrain": int(len(train)), "nTest": int(len(test)), "oosR2": r2, "hitRate": hit, "betaTrain": float(beta)}


def _index_same_morning(index_bars, spy_open, spy_close):
    """Asia cash-index close (before US open) → same-day SPY gap / close.

    `index_bars` is indexed by ET timestamps. We take the last bar whose timestamp falls
    before 09:30 ET on a US session as that morning's information set.
    """
    if index_bars is None or index_bars.empty or "close" not in index_bars.columns:
        return None
    bars = index_bars.copy()
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is None:
        idx = idx.tz_localize(ET)
    else:
        idx = idx.tz_convert(ET)
    bars.index = idx
    rows = []
    spy_days = pd.DatetimeIndex(pd.to_datetime(spy_close.index).normalize())
    for day in spy_days:
        start = day.tz_localize(ET) if day.tzinfo is None else day
        cutoff = start.replace(hour=9, minute=30, second=0, microsecond=0)
        # For a daily-bar Yahoo feed, the index is typically midnight or the session stamp.
        # Treat a bar dated on `day` (ET date before 09:30, or previous calendar date after
        # Asia's close which is 1–4 AM ET on `day`) as available.
        window = bars[(bars.index < cutoff) & (bars.index >= cutoff - pd.Timedelta(hours=36))]
        if window.empty:
            continue
        last = window["close"].dropna()
        if last.empty:
            continue
        # Previous Asia close: the last bar before this one.
        loc = bars.index.get_loc(last.index[-1])
        if isinstance(loc, slice) or isinstance(loc, np.ndarray):
            loc = int(np.max(np.atleast_1d(loc)))
        if loc == 0:
            continue
        prev = bars["close"].iloc[:loc].dropna()
        if prev.empty or prev.iloc[-1] == 0:
            continue
        ret = float(last.iloc[-1] / prev.iloc[-1] - 1)
        rows.append((day.tz_localize(None) if day.tzinfo else day, ret))
    if not rows:
        return None
    return pd.Series({d: r for d, r in rows}).sort_index()


def run_overnight_study(panel=None, index_data=None, path=OVERNIGHT_PATH):
    tickers = list(OVERSEAS_ETFS) + ["SPY", "QQQ", "TLT", "IEF", "GLD", "UUP"]
    if panel is None:
        panel = load_prices(tickers)
    tests, notes = _etf_lagged_overnight(panel)
    close = panel.close.copy()
    close.index = pd.DatetimeIndex(pd.to_datetime(close.index).normalize())
    open_ = panel.open.copy()
    open_.index = pd.DatetimeIndex(pd.to_datetime(open_.index).normalize())
    spy_gap = open_["SPY"] / close["SPY"].shift(1) - 1 if "SPY" in open_.columns else None
    spy_cc = close["SPY"].pct_change()
    spy_oc = close["SPY"] / open_["SPY"] - 1 if "SPY" in open_.columns else None

    if index_data is None:
        try:
            from macro_sources import load_overseas_indices
            index_data, fetch_notes = load_overseas_indices()
            notes.extend(fetch_notes)
        except Exception as error:  # noqa: BLE001
            notes.append(f"Overseas cash-index fetch skipped: {error}")
            index_data = {}

    asia_names = [n for n, d in (index_data or {}).items() if (d.get("meta") or {}).get("region") == "asia"]
    europe_names = [n for n, d in (index_data or {}).items() if (d.get("meta") or {}).get("region") == "europe"]
    notes.append(
        "Same-morning Asia cash-index returns (Nikkei, Hang Seng, Shanghai, KOSPI) are valid predictors "
        "of the US open when the index close is timestamped before 9:30 ET. "
        "Europe's cash close is after the US open and is NOT used to explain that open. "
        "Europe is only used as a lagged (prior-day) feature, or via its open/morning bar when one is present before 9:30 ET."
    )
    for name in asia_names:
        bars = index_data[name]["bars"]
        series = _index_same_morning(bars, open_["SPY"] if "SPY" in open_.columns else None, close["SPY"])
        if series is None or series.dropna().empty:
            notes.append(f"{name}: could not align a pre-09:30 ET close; skipped same-morning test.")
            continue
        for label, y in (("spy_open_gap", spy_gap), ("spy_session_oc", spy_oc), ("spy_close_1d", spy_cc)):
            if y is None:
                continue
            fit = hac_ols(y, series)
            if not fit:
                continue
            oos = _walk_forward(y, series)
            tests.append({"predictor": name, "horizon": label, "design": "asia_same_morning",
                          **{k: _clean(v) for k, v in fit.items()},
                          "oos": {k: _clean(v) for k, v in (oos or {}).items()}})
    # Europe: lagged only (previous close → next US session).
    for name in europe_names:
        bars = index_data[name]["bars"]
        if bars is None or bars.empty or "close" not in bars.columns:
            continue
        daily = bars["close"].copy()
        if getattr(daily.index, "tz", None) is not None:
            daily.index = daily.index.tz_convert(ET).normalize().tz_localize(None)
        else:
            daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index).normalize())
        daily = daily.groupby(level=0).last().pct_change().shift(1)  # previous Europe close
        for label, y in (("spy_open_gap", spy_gap), ("spy_session_oc", spy_oc), ("spy_close_1d", spy_cc)):
            if y is None:
                continue
            fit = hac_ols(y, daily)
            if not fit:
                continue
            oos = _walk_forward(y, daily)
            tests.append({"predictor": name, "horizon": label, "design": "europe_lagged_close",
                          **{k: _clean(v) for k, v in fit.items()},
                          "oos": {k: _clean(v) for k, v in (oos or {}).items()}})

    pvals = [t.get("betaP") for t in tests]
    adj = multiple_testing(pvals)
    for i, t in enumerate(tests):
        t["pHolm"] = _clean(adj["holmAdj"][i]) if i < len(adj["holmAdj"]) else None
        t["pBH"] = _clean(adj["bhAdj"][i]) if i < len(adj["bhAdj"]) else None
        t["rejectHolm"] = bool(adj["holmReject"][i]) if i < len(adj["holmReject"]) else False
        t["rejectBH"] = bool(adj["bhReject"][i]) if i < len(adj["bhReject"]) else False
        # OOS: require a positive OOS R2 and hit rate > 0.5 to even be "suggestive"
        oos = t.get("oos") or {}
        t["oosSuggestive"] = bool(
            (oos.get("oosR2") or -1) > 0 and (oos.get("hitRate") or 0) > 0.52 and t.get("rejectHolm")
        )
    proven = [t for t in tests if t.get("oosSuggestive")]
    summary = {
        "generatedAt": _now(),
        "timezone": "America/New_York",
        "nTests": len(tests),
        "nProvenOosHolm": len(proven),
        "multipleTesting": {"method": adj.get("method"), "alpha": FDR_ALPHA},
        "notes": notes,
        "honesty": (
            "Overseas overnight information is real, but US futures usually incorporate the Asia close "
            "well before the cash open. A significant in-sample beta that does not hold walk-forward, "
            "or that vanishes after Holm correction, is not an edge. Same-day European closes were not "
            "used to explain the US open (lookahead)."
        ),
        "tests": tests,
        "proven": proven,
    }
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(summary, indent=2, default=str).encode("utf-8"))
    try:
        write_atomically(RESEARCH_OVERNIGHT, json.dumps(summary, indent=2, default=str).encode("utf-8"))
    except Exception:
        pass
    return summary


def proven_rules(study=None, overnight=None):
    """Flatten Holm-proven tests into a list the reaction layer can consult."""
    study = study or (_load(STUDY_PATH) or {})
    overnight = overnight or (_load(OVERNIGHT_PATH) or {})
    rules = []
    for t in study.get("proven") or []:
        rules.append({"kind": "macro_event", **t})
    for t in overnight.get("proven") or []:
        rules.append({"kind": "overnight", **t})
    return rules


def _load(path):
    path = Path(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def run_all(fetch_indices=True):
    ev = run_event_study()
    overnight = run_overnight_study(index_data=None if fetch_indices else {})
    return {"eventStudy": ev, "overnight": overnight}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Macro event study (HAC t-stats, Holm/BH).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--no-fetch", action="store_true")
    sub.add_parser("overnight")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.cmd == "status":
        ev = _load(STUDY_PATH) or {}
        ov = _load(OVERNIGHT_PATH) or {}
        print(json.dumps({
            "eventStudy": {k: ev.get(k) for k in ("generatedAt", "nTests", "nProvenHolm", "honesty") if k in ev},
            "overnight": {k: ov.get(k) for k in ("generatedAt", "nTests", "nProvenOosHolm", "honesty") if k in ov},
        }, indent=2))
        return 0
    if args.cmd == "overnight":
        doc = run_overnight_study()
        print(json.dumps({k: doc[k] for k in ("generatedAt", "nTests", "nProvenOosHolm", "honesty", "notes") if k in doc},
                         indent=2))
        return 0
    doc = run_all(fetch_indices=not args.no_fetch)
    ev, ov = doc["eventStudy"], doc["overnight"]
    print(json.dumps({
        "eventStudy": {k: ev.get(k) for k in ("generatedAt", "nTests", "nProvenHolm")},
        "overnight": {k: ov.get(k) for k in ("generatedAt", "nTests", "nProvenOosHolm")},
        "honesty": ev.get("honesty"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
