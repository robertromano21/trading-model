"""Daily drift monitor: compare paper (and later live) behavior to walk-forward expected ranges.

After each daily paper run this module:

* loads the combined walk-forward out-of-sample equity and decisions, plus day-trading research
* builds bootstrap / rolling distributions for daily and 20-day return, vol, drawdown, turnover,
  slippage, position count, gross exposure and the day-trader fill rate
* measures the same quantities on the paper account (and, later, a live account with the same schema)
* flags values outside the 5–95% (warning) or 1–99% (severe) bands
* runs two-sided CUSUM and a sequential standardized-mean test for persistent drift
* emits alerts via alerts_emit.emit(), writes the decision chat log, and stores an EVENT in analyst memory
* writes Massive/Research/drift_status.json (UI) and appends Massive/Research/drift_history.jsonl

It does not retune the strategy, place orders, or call any broker. New paper days update CUSUM state
and the history so the monitor learns whether live trading still looks like the backtest.

  python drift_monitor.py run [--account default] [--asof YYYY-MM-DD] [--no-alerts] [--no-chat]
  python drift_monitor.py status
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, PROJECT_ROOT

RESEARCH = DATA_DIR / "Research"
# The real paper account trades plain TSMOM since 2026-11-25, so its expected ranges come from the TSMOM walk-forward.
WF_EQUITY = RESEARCH / "walkforward_tsmom_oos_equity.csv"
WF_DECISIONS = RESEARCH / "walkforward_tsmom_oos_decisions.csv"
DT_TRADES = RESEARCH / "daytrade_trades_defaults.csv"
DT_SUMMARY = RESEARCH / "daytrade_summary.json"
PAPER_DB = DATA_DIR / "paper_trading.sqlite3"
LIVE_DB = DATA_DIR / "live_trading.sqlite3"  # reserved; used only if present, never for orders
STATUS_FILE = RESEARCH / "drift_status.json"
HISTORY_FILE = RESEARCH / "drift_history.jsonl"
STATE_FILE = RESEARCH / "drift_state.json"
CHAT_DB = DATA_DIR / "chat_log.sqlite3"

METRIC_ORDER = (
    "daily_return",
    "roll20_return",
    "roll20_vol",
    "drawdown",
    "turnover",
    "slippage_bps",
    "signal_agreement",
    "position_count",
    "gross_exposure",
    "daytrade_fill_rate",
)
LABELS = {
    "daily_return": "Daily return",
    "roll20_return": "20-day return",
    "roll20_vol": "20-day volatility (ann.)",
    "drawdown": "Drawdown from peak",
    "turnover": "Daily turnover",
    "slippage_bps": "Slippage (bps, signed)",
    "signal_agreement": "Signal agreement vs backtest replay",
    "position_count": "Position count",
    "gross_exposure": "Gross exposure",
    "daytrade_fill_rate": "Day-trader fill rate (20-day)",
}
# Structural bands for paper-vs-replay agreement (same engine; only rounding / stale fills should differ).
AGREEMENT_BANDS = {"p1": 0.60, "p5": 0.80, "p50": 0.98, "p95": 1.0, "p99": 1.0, "mean": 0.98, "std": 0.05, "n": None}
CUSUM_K = 0.5
CUSUM_H = 5.0
SEQ_CRITICAL = 2.58  # ~1% two-sided
RNG_SEED = 10582
BOOTSTRAP = 2000
WEIGHT_EPS = 1e-6


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _pct(series, qs=(1, 5, 50, 95, 99)):
    s = pd.Series(series).dropna().astype(float)
    if s.empty:
        return None
    out = {f"p{q}": _clean(np.nanpercentile(s, q)) for q in qs}
    out["mean"] = _clean(s.mean())
    out["std"] = _clean(s.std(ddof=1) if len(s) > 1 else 0.0)
    out["n"] = int(len(s))
    return out


def _block_bootstrap(values, window, n_boot=BOOTSTRAP, seed=RNG_SEED):
    """Percentiles of the window statistic via moving-block bootstrap of the daily series."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < window:
        return None
    block = max(1, window // 2)
    rng = np.random.default_rng(seed)
    n = len(x)
    n_blocks = int(math.ceil(n / block))
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in starts])[:n]
        if len(sample) < window:
            continue
        roll = pd.Series(sample)
        compounded = (1 + roll).rolling(window).apply(lambda v: float(np.prod(1 + v) - 1), raw=True)
        vol = roll.rolling(window).std(ddof=1) * math.sqrt(252)
        stats.append((float(compounded.iloc[-1]), float(vol.iloc[-1])))
    if not stats:
        return None
    ret, vol = zip(*stats)
    return {"roll20_return": _pct(ret), "roll20_vol": _pct(vol)}


def _compound(returns):
    r = pd.Series(returns).astype(float)
    return float((1 + r).prod() - 1) if len(r) else None


def _ann_vol(returns):
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * math.sqrt(252))


def _drawdown_series(equity):
    eq = pd.Series(equity).astype(float)
    return eq / eq.cummax() - 1


def _read_csv(path, **kwargs):
    path = Path(path)
    if not path.is_file():
        return None
    frame = pd.read_csv(path, **kwargs)
    if frame.columns[:1].tolist() == ["Unnamed: 0"] or (frame.columns.size and frame.columns[0] == "Unnamed: 0"):
        frame = frame.drop(columns=["Unnamed: 0"], errors="ignore")
    return frame


def _as_dates(series):
    return pd.to_datetime(series, errors="coerce")


# --------------------------------------------------------------------------- expected (walk-forward)
def reconstruct_holdings(decisions, equity):
    """Daily position count and gross exposure from walk-forward decision weights."""
    if decisions is None or decisions.empty or equity is None or equity.empty:
        return pd.DataFrame(columns=["position_count", "gross_exposure", "turnover", "slippage_bps"])
    dec = decisions.copy()
    dec["decision_date"] = _as_dates(dec["decision_date"])
    dec["fill_date"] = _as_dates(dec.get("fill_date", dec["decision_date"]))
    dates = pd.DatetimeIndex(equity.index)
    # Latest target weight per ticker as of each date (filled or decided).
    rows = []
    latest = {}
    dec_by_day = {d: g for d, g in dec.groupby(dec["decision_date"].dt.normalize())}
    fills_by_day = {d: g for d, g in dec.groupby(dec["fill_date"].dt.normalize())}
    eq = equity["equity"] if "equity" in equity.columns else equity.iloc[:, 0]
    for day in dates:
        key = pd.Timestamp(day).normalize()
        if key in dec_by_day:
            for _, r in dec_by_day[key].iterrows():
                latest[str(r["ticker"])] = float(r["target_weight"] or 0.0)
        weights = np.array(list(latest.values()), dtype=float) if latest else np.array([])
        active = weights[np.abs(weights) > WEIGHT_EPS]
        notional = 0.0
        slip = []
        if key in fills_by_day:
            part = fills_by_day[key]
            notional = float(part["notional"].abs().sum()) if "notional" in part else 0.0
            if "slippage_bps" in part:
                slip = [float(x) for x in part["slippage_bps"].dropna()]
        equity_val = float(eq.loc[day]) if day in eq.index else float(eq.iloc[-1])
        rows.append({
            "date": key,
            "position_count": int(len(active)),
            "gross_exposure": float(np.abs(active).sum()) if len(active) else 0.0,
            "turnover": (notional / equity_val) if equity_val else 0.0,
            "slippage_bps": float(np.mean(slip)) if slip else np.nan,
        })
    out = pd.DataFrame(rows).set_index("date")
    return out


def daytrade_fill_series(trades, summary=None):
    """Daily fill count and rolling 20-day fill rate (fills per session)."""
    if trades is None or trades.empty or "session_date" not in trades.columns:
        return pd.Series(dtype=float)
    counts = trades.groupby(_as_dates(trades["session_date"]).dt.normalize()).size()
    first = counts.index.min()
    last = counts.index.max()
    if isinstance(summary, dict):
        fp = (summary.get("fullPeriodDefaults") or {}).get("first") or (summary.get("data") or {}).get("first")
        lp = (summary.get("fullPeriodDefaults") or {}).get("last") or (summary.get("data") or {}).get("last")
        if fp:
            first = min(first, pd.Timestamp(fp))
        if lp:
            last = max(last, pd.Timestamp(lp))
    calendar = pd.bdate_range(first, last)
    daily = counts.reindex(calendar, fill_value=0).astype(float)
    return daily.rolling(20, min_periods=1).mean()


def build_expected(paths):
    """Return {metric: {p1,p5,p50,p95,p99,mean,std,n}} and the daily expected series."""
    equity = _read_csv(paths["wf_equity"], index_col=0)
    decisions = _read_csv(paths["wf_decisions"])
    dt_trades = _read_csv(paths["dt_trades"])
    summary = None
    if Path(paths["dt_summary"]).is_file():
        try:
            summary = json.loads(Path(paths["dt_summary"]).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = None
    notes = []
    if equity is None or equity.empty:
        notes.append("Walk-forward equity file is missing; bands cannot be built until research walk-forward is run.")
        return {"bands": {}, "series": {}, "notes": notes, "source": None, "sessions": 0}
    equity.index = _as_dates(equity.index).normalize()
    equity = equity.sort_index()
    if "return" not in equity.columns:
        equity["return"] = equity["equity"].pct_change().fillna(0.0)
    returns = equity["return"].astype(float)
    roll20_ret = (1 + returns).rolling(20, min_periods=20).apply(lambda v: float(np.prod(v) - 1), raw=True)
    roll20_vol = returns.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(252)
    dd = _drawdown_series(equity["equity"])
    holdings = reconstruct_holdings(decisions, equity)
    holdings = holdings.reindex(equity.index)
    fill_rate = daytrade_fill_series(dt_trades, summary)
    bands = {
        "daily_return": _pct(returns),
        "roll20_return": _pct(roll20_ret),
        "roll20_vol": _pct(roll20_vol),
        "drawdown": _pct(dd),
        "turnover": _pct(holdings["turnover"]) if not holdings.empty else None,
        "slippage_bps": _pct(holdings["slippage_bps"].rolling(20, min_periods=1).mean()) if not holdings.empty else None,
        "signal_agreement": dict(AGREEMENT_BANDS),
        "position_count": _pct(holdings["position_count"]) if not holdings.empty else None,
        "gross_exposure": _pct(holdings["gross_exposure"]) if not holdings.empty else None,
        "daytrade_fill_rate": _pct(fill_rate) if fill_rate is not None and len(fill_rate.dropna()) else None,
    }
    boot = _block_bootstrap(returns.values, 20)
    if boot:
        # Blend: keep empirical rolling percentiles as primary; stash bootstrap next to them.
        for key, boot_band in boot.items():
            if bands.get(key) and boot_band:
                bands[key] = {**bands[key], "bootstrap": boot_band}
    series = {
        "daily_return": returns,
        "roll20_return": roll20_ret,
        "roll20_vol": roll20_vol,
        "drawdown": dd,
        "turnover": holdings["turnover"] if not holdings.empty else pd.Series(dtype=float),
        "slippage_bps": holdings["slippage_bps"] if not holdings.empty else pd.Series(dtype=float),
        "position_count": holdings["position_count"] if not holdings.empty else pd.Series(dtype=float),
        "gross_exposure": holdings["gross_exposure"] if not holdings.empty else pd.Series(dtype=float),
        "daytrade_fill_rate": fill_rate if fill_rate is not None else pd.Series(dtype=float),
    }
    source = {
        "equityFile": str(paths["wf_equity"]),
        "decisionsFile": str(paths["wf_decisions"]),
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "sessions": int(len(equity)),
        "daytradeTrades": int(len(dt_trades)) if dt_trades is not None else 0,
    }
    return {"bands": bands, "series": series, "notes": notes, "source": source, "sessions": int(len(equity))}


# --------------------------------------------------------------------------- realized (paper / live)
def _connect_ro(path):
    path = Path(path)
    if not path.is_file():
        return None
    db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def _has(db, name):
    return bool(db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone())


def _table(db, sql, params=()):
    return [dict(r) for r in db.execute(sql, params)]


def wf_targets_on(decisions, asof):
    if decisions is None or decisions.empty:
        return {}
    dec = decisions.copy()
    dec["decision_date"] = _as_dates(dec["decision_date"])
    asof = pd.Timestamp(asof).normalize()
    part = dec[dec["decision_date"] <= asof]
    if part.empty:
        return {}
    latest = part.sort_values("decision_date").groupby("ticker").tail(1)
    return {str(r["ticker"]).upper(): float(r["target_weight"] or 0.0) for _, r in latest.iterrows()
            if abs(float(r["target_weight"] or 0.0)) > WEIGHT_EPS}


def paper_targets_on(db, account, asof):
    if db is None or not _has(db, "decision_log"):
        return {}
    asof = str(pd.Timestamp(asof).date())
    rows = _table(db, """SELECT ticker, target_weight, decision_date FROM decision_log
                         WHERE account=? AND decision_date<=? AND target_weight IS NOT NULL
                         ORDER BY decision_date, rowid""", (account, asof))
    out = {}
    for r in rows:
        out[str(r["ticker"]).upper()] = float(r["target_weight"] or 0.0)
    return {k: v for k, v in out.items() if abs(v) > WEIGHT_EPS}


def signal_agreement(paper_w, wf_w):
    names = sorted(set(paper_w) | set(wf_w))
    if not names:
        return 1.0
    a = np.array([paper_w.get(t, 0.0) for t in names], dtype=float)
    b = np.array([wf_w.get(t, 0.0) for t in names], dtype=float)
    denom = 0.5 * (np.abs(a).sum() + np.abs(b).sum())
    if denom <= 0:
        return 1.0
    overlap = 1.0 - 0.5 * np.abs(a - b).sum() / denom
    return float(max(0.0, min(1.0, overlap)))


def load_realized(paths, account="default", source="paper", asof=None, decisions=None):
    db_path = paths["live_db"] if source == "live" else paths["paper_db"]
    db = _connect_ro(db_path)
    empty = {"sessions": 0, "asof": None, "metrics": {}, "series": {}, "notes": [], "source": source, "account": account}
    if db is None:
        empty["notes"].append(f"No {source} database yet ({Path(db_path).name}).")
        return empty
    notes = []
    with closing(db):
        if source == "live" and not _has(db, "equity_history"):
            notes.append("Live trading database has no equity_history yet; drift will use it when live fills exist.")
            return {**empty, "notes": notes}
        if not _has(db, "equity_history"):
            notes.append("Paper database has no equity history yet.")
            return {**empty, "notes": notes}
        equity_rows = _table(db, "SELECT date, equity, cash, gross, daily_return FROM equity_history WHERE account=? ORDER BY date",
                             (account,))
        if not equity_rows:
            # day-trade-only accounts use dt_daily
            if _has(db, "dt_daily"):
                equity_rows = [{"date": r["session_date"], "equity": r["equity"], "cash": None,
                                "gross": 0.0, "daily_return": (r["pnl"] / r["start_equity"]) if r.get("start_equity") else 0.0}
                               for r in _table(db, "SELECT * FROM dt_daily WHERE account=? ORDER BY session_date", (account,))]
        if not equity_rows:
            notes.append(f"No sessions booked for account '{account}'.")
            return {**empty, "notes": notes}
        eq = pd.DataFrame(equity_rows)
        eq["date"] = _as_dates(eq["date"]).dt.normalize()
        eq = eq.drop_duplicates("date").set_index("date").sort_index()
        if asof:
            eq = eq.loc[:pd.Timestamp(asof).normalize()]
        if eq.empty:
            notes.append("No sessions on or before the requested as-of date.")
            return {**empty, "notes": notes}
        asof_ts = eq.index[-1]
        returns = eq["daily_return"].astype(float) if "daily_return" in eq else eq["equity"].pct_change().fillna(0.0)
        last20 = returns.iloc[-20:]
        metrics = {
            "daily_return": _clean(float(returns.iloc[-1])),
            "roll20_return": _clean(_compound(last20) if len(last20) >= 20 else None),
            "roll20_vol": _clean(_ann_vol(last20) if len(last20) >= 20 else None),
            "drawdown": _clean(float(_drawdown_series(eq["equity"]).iloc[-1])),
            "gross_exposure": _clean(float(eq["gross"].iloc[-1]) if "gross" in eq else None),
        }
        # turnover + slippage from fills / decision_log on the as-of day (and history)
        turnover_s = pd.Series(0.0, index=eq.index)
        slip_s = pd.Series(np.nan, index=eq.index)
        if _has(db, "fills"):
            fills = pd.DataFrame(_table(db, "SELECT fill_date, notional, slippage_bps FROM fills WHERE account=?", (account,)))
            if not fills.empty:
                fills["fill_date"] = _as_dates(fills["fill_date"]).dt.normalize()
                notion = fills.groupby("fill_date")["notional"].apply(lambda s: float(s.abs().sum()))
                turnover_s = (notion.reindex(eq.index).fillna(0.0) / eq["equity"].replace(0, np.nan)).fillna(0.0)
                slip_s = fills.groupby("fill_date")["slippage_bps"].mean().reindex(eq.index)
        elif _has(db, "decision_log"):
            decs = pd.DataFrame(_table(db, """SELECT fill_date, notional, slippage_bps, status FROM decision_log
                                              WHERE account=? AND status='filled' AND fill_date IS NOT NULL""", (account,)))
            if not decs.empty:
                decs["fill_date"] = _as_dates(decs["fill_date"]).dt.normalize()
                notion = decs.groupby("fill_date")["notional"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum()))
                turnover_s = (notion.reindex(eq.index).fillna(0.0) / eq["equity"].replace(0, np.nan)).fillna(0.0)
                slip_s = decs.groupby("fill_date")["slippage_bps"].mean().reindex(eq.index)
        metrics["turnover"] = _clean(float(turnover_s.iloc[-1])) if len(turnover_s) else 0.0
        slip_roll = slip_s.rolling(20, min_periods=1).mean()
        if len(slip_roll.dropna()):
            metrics["slippage_bps"] = _clean(float(slip_roll.dropna().iloc[-1]))
        else:
            metrics["slippage_bps"] = None

        # position count: live positions table, else last weights
        if _has(db, "positions"):
            pos = _table(db, "SELECT ticker, shares FROM positions WHERE account=?", (account,))
            metrics["position_count"] = int(sum(1 for p in pos if abs(float(p["shares"] or 0)) > WEIGHT_EPS))
        else:
            metrics["position_count"] = len(paper_targets_on(db, account, asof_ts))

        # Signal agreement: paper weights vs a same-day shadow backtest replay (reconciliation table),
        # not the multi-year walk-forward book, which started earlier and is not the same path.
        paper_w, replay_w = {}, {}
        if _has(db, "reconciliation"):
            recon = _table(db, "SELECT date, max_weight_diff, details FROM reconciliation WHERE account=? AND date<=? ORDER BY date DESC LIMIT 1",
                           (account, str(asof_ts.date())))
            if recon:
                details = recon[0].get("details")
                if isinstance(details, str):
                    try:
                        details = json.loads(details)
                    except ValueError:
                        details = {}
                details = details or {}
                paper_w = {str(k).upper(): float(v) for k, v in (details.get("paperWeights") or {}).items()}
                replay_w = {str(k).upper(): float(v) for k, v in (details.get("shadowWeights") or {}).items()}
                if not paper_w and not replay_w:
                    # Day-one cash vs cash, or details not stored: treat a zero max-weight-diff as full agreement.
                    md = recon[0].get("max_weight_diff")
                    if md is not None and float(md) == 0.0:
                        metrics["signal_agreement"] = 1.0
        if "signal_agreement" not in metrics:
            if not paper_w:
                paper_w = paper_targets_on(db, account, asof_ts)
            if paper_w or replay_w:
                metrics["signal_agreement"] = _clean(signal_agreement(paper_w, replay_w))
            else:
                metrics["signal_agreement"] = None
                notes.append("No paper-vs-shadow replay weights yet for signal agreement.")

        # day-trader fill rate (20-day) from dt_daily / dt_trades
        dt_account = "daytrade"
        fill_rate_s = pd.Series(dtype=float)
        if _has(db, "dt_daily"):
            daily = pd.DataFrame(_table(db, "SELECT session_date, trades, skipped FROM dt_daily WHERE account=? ORDER BY session_date",
                                        (dt_account,)))
            if not daily.empty:
                daily["session_date"] = _as_dates(daily["session_date"]).dt.normalize()
                daily = daily.set_index("session_date")
                # 20-day mean fills per session, including days with zero trades (matches the walk-forward series).
                count = daily["trades"].astype(float)
                fill_rate_s = count.rolling(20, min_periods=1).mean()
                metrics["daytrade_fill_rate"] = _clean(float(fill_rate_s.iloc[-1])) if len(fill_rate_s) else None
            else:
                metrics["daytrade_fill_rate"] = None
                notes.append("No day-trading sessions booked yet.")
        elif _has(db, "dt_trades"):
            trades = pd.DataFrame(_table(db, "SELECT session_date FROM dt_trades WHERE account=?", (dt_account,)))
            fill_rate_s = daytrade_fill_series(trades, None) if not trades.empty else pd.Series(dtype=float)
            metrics["daytrade_fill_rate"] = _clean(float(fill_rate_s.dropna().iloc[-1])) if len(fill_rate_s.dropna()) else None
        else:
            metrics["daytrade_fill_rate"] = None

        series = {
            "daily_return": returns,
            "roll20_return": (1 + returns).rolling(20, min_periods=20).apply(lambda v: float(np.prod(v) - 1), raw=True),
            "roll20_vol": returns.rolling(20, min_periods=20).std(ddof=1) * math.sqrt(252),
            "drawdown": _drawdown_series(eq["equity"]),
            "turnover": turnover_s,
            "slippage_bps": slip_s,
            "gross_exposure": eq["gross"].astype(float) if "gross" in eq else pd.Series(dtype=float),
            "daytrade_fill_rate": fill_rate_s,
        }
        return {
            "sessions": int(len(eq)),
            "asof": str(asof_ts.date()),
            "account": account,
            "source": source,
            "equity": _clean(float(eq["equity"].iloc[-1])),
            "metrics": metrics,
            "series": series,
            "notes": notes,
            "paperWeights": len(paper_w),
            "replayWeights": len(replay_w),
        }


# --------------------------------------------------------------------------- scoring / CUSUM
def classify(value, band):
    """Return 'ok', 'warning' (outside 5–95) or 'severe' (outside 1–99). None if value or band missing."""
    if value is None or not band or band.get("p5") is None:
        return None
    p1, p5, p95, p99 = band.get("p1"), band.get("p5"), band.get("p95"), band.get("p99")
    if p1 is not None and p99 is not None and (value < p1 or value > p99):
        return "severe"
    if p5 is not None and p95 is not None and (value < p5 or value > p95):
        return "warning"
    return "ok"


def update_cusum(state, metric, value, band):
    """Two-sided CUSUM in sigma units + sequential standardized-mean test. Mutates state[metric]."""
    slot = state.setdefault(metric, {"sPlus": 0.0, "sMinus": 0.0, "n": 0, "sumZ": 0.0, "alarm": False, "seqAlarm": False})
    if value is None or not band or not band.get("std") or band["std"] == 0:
        return slot
    mu = band.get("mean") if band.get("mean") is not None else band.get("p50") or 0.0
    z = (float(value) - float(mu)) / float(band["std"])
    if not math.isfinite(z):
        return slot
    slot["sPlus"] = max(0.0, float(slot.get("sPlus") or 0.0) + z - CUSUM_K)
    slot["sMinus"] = max(0.0, float(slot.get("sMinus") or 0.0) - z - CUSUM_K)
    slot["n"] = int(slot.get("n") or 0) + 1
    slot["sumZ"] = float(slot.get("sumZ") or 0.0) + z
    slot["z"] = _clean(z)
    slot["lastValue"] = _clean(value)
    slot["alarm"] = bool(slot["sPlus"] > CUSUM_H or slot["sMinus"] > CUSUM_H)
    seq = slot["sumZ"] / math.sqrt(slot["n"]) if slot["n"] else 0.0
    slot["seqStat"] = _clean(seq)
    slot["seqAlarm"] = bool(slot["n"] >= 5 and abs(seq) > SEQ_CRITICAL)
    return slot


def load_state(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cusum": {}, "lastAlertKey": None, "lastAsOf": None}


def save_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_history(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, default=str) + "\n")


def overall_status(flags, cusum):
    if any(v == "severe" for v in flags.values()) or any((s or {}).get("alarm") or (s or {}).get("seqAlarm") for s in cusum.values()):
        return "severe"
    if any(v == "warning" for v in flags.values()):
        return "warning"
    if flags and all(v in (None, "ok") for v in flags.values()) and any(v == "ok" for v in flags.values()):
        return "ok"
    return "awaiting"


def format_value(metric, value):
    if value is None:
        return "n/a"
    if metric in ("slippage_bps",):
        return f"{value:.1f} bps"
    if metric in ("position_count",):
        return f"{value:.0f}"
    if metric in ("daytrade_fill_rate",) and abs(value) > 1.5:
        return f"{value:.2f} fills/session"
    if metric in ("signal_agreement", "daily_return", "roll20_return", "roll20_vol", "drawdown", "turnover", "gross_exposure", "daytrade_fill_rate"):
        return f"{value * 100:.2f}%"
    return f"{value:.4f}"


def speech_for(status, asof, breaches, persistent):
    if status == "severe":
        bits = [LABELS.get(m, m) for m in breaches]
        extra = f" Persistent drift on {', '.join(LABELS.get(m, m) for m in persistent)}." if persistent else ""
        named = ", ".join(bits[:3]) if bits else "several metrics"
        return (f"I've detected a serious drift versus the backtest as of {asof}. "
                f"{named} {'is' if len(bits) == 1 else 'are'} outside the 1 to 99 percent band.{extra} "
                f"I'd treat the paper results with caution until this is explained.")
    if status == "warning":
        named = ", ".join(LABELS.get(m, m) for m in breaches[:3]) or "a metric"
        return (f"Paper trading is outside the backtest's usual range as of {asof} on {named}. "
                f"It's a warning, not a halt, but I'll keep watching.")
    return f"Paper trading is inside the backtest's expected range as of {asof}."


def text_for(report):
    status = report["status"]
    asof = report.get("asof") or "today"
    lines = [f"Drift monitor ({report.get('source', 'paper')} / {report.get('account', 'default')}) as of {asof}: {status.upper()}."]
    for name in METRIC_ORDER:
        row = report["metrics"].get(name) or {}
        if not row:
            continue
        flag = row.get("flag") or "n/a"
        lines.append(f"  {LABELS[name]}: {format_value(name, row.get('value'))} "
                     f"(band {format_value(name, (row.get('band') or {}).get('p5'))} to "
                     f"{format_value(name, (row.get('band') or {}).get('p95'))}, {flag})"
                     + (" CUSUM alarm" if row.get("cusumAlarm") else "")
                     + (" sequential drift" if row.get("seqAlarm") else ""))
    if report.get("persistent"):
        lines.append("Persistent drift (CUSUM or sequential test): " + ", ".join(LABELS.get(m, m) for m in report["persistent"]) + ".")
    lines.append(report["learning"]["note"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- side effects
def write_chat(chat_path, report, insert_fn=None):
    if insert_fn:
        return insert_fn(report)
    try:
        from decision_feed import connect_chat, _insert
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    key = f"drift:{report.get('source')}:{report.get('account')}:{report.get('asof')}:{report['status']}"
    chat = connect_chat(Path(chat_path))
    try:
        added = _insert(
            chat, key, role="trader", kind="drift", account=report.get("account"),
            decision_date=report.get("asof"), text=text_for(report),
            rationale="Compares realized paper (or live) metrics to walk-forward 5–95 and 1–99 percent bands, plus CUSUM.",
            data={"status": report["status"], "flags": {k: v.get("flag") for k, v in report["metrics"].items()},
                  "persistent": report.get("persistent"), "href": "/paper2.html#drift"},
        )
        chat.commit()
        return {"ok": True, "added": int(added), "sourceKey": key}
    finally:
        chat.close()


def write_memory(report, ingest_fn=None):
    text = text_for(report)
    meta = {"dedupeKey": f"drift:{report.get('source')}:{report.get('account')}:{report.get('asof')}:{report['status']}",
            "source": "drift_monitor", "status": report["status"], "asof": report.get("asof"),
            "account": report.get("account"), "href": "/paper2.html#drift"}
    if ingest_fn:
        return ingest_fn("EVENT", text, meta=meta)
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "server" / "ai"))
        from memory_ingest import ingest
        item_id, duplicate = ingest("EVENT", text, ts=None, meta=meta)
        return {"ok": True, "id": item_id, "duplicate": duplicate}
    except Exception as exc:  # noqa: BLE001 — memory must never fail the monitor
        return {"ok": False, "error": str(exc)}


def emit_alerts(report, emit_fn=None, force=False):
    """Emit at most one alert per (asof, status, breach-set), speech for severe only."""
    if report["status"] in (None, "awaiting", "ok") and not force:
        return {"ok": True, "emitted": False, "reason": "in range or awaiting paper data"}
    if emit_fn is None:
        from alerts_emit import emit as emit_fn
    status = report["status"]
    breaches = report.get("breaches") or []
    persistent = report.get("persistent") or []
    severity = "critical" if status == "severe" else "warning"
    payload = {
        "type": "drift",
        "severity": severity,
        "title": "Serious backtest drift" if status == "severe" else "Backtest drift warning",
        "text": text_for(report).split("\n")[0] + " " + speech_for(status, report.get("asof"), breaches, persistent),
        "speech": speech_for(status, report.get("asof"), breaches, persistent) if status == "severe" else "",
        "href": "/paper2.html#drift",
        "meta": {
            "asof": report.get("asof"),
            "status": status,
            "breaches": breaches,
            "persistent": persistent,
            "account": report.get("account"),
            "source": report.get("source"),
            "speak": status == "severe",
        },
    }
    ok = bool(emit_fn(payload))
    return {"ok": ok, "emitted": ok, "severity": severity}


# --------------------------------------------------------------------------- report
def build_report(expected, realized, state, update_cusum_state=True):
    cusum_state = state.setdefault("cusum", {})
    metrics = {}
    flags = {}
    breaches_warn, breaches_sev, persistent = [], [], []
    for name in METRIC_ORDER:
        band = (expected.get("bands") or {}).get(name)
        value = (realized.get("metrics") or {}).get(name)
        flag = classify(value, band)
        if update_cusum_state and value is not None:
            slot = update_cusum(cusum_state, name, value, band)
        else:
            slot = cusum_state.get(name) or {}
        if flag == "warning":
            breaches_warn.append(name)
        if flag == "severe":
            breaches_sev.append(name)
        if slot.get("alarm") or slot.get("seqAlarm"):
            persistent.append(name)
        flags[name] = flag
        metrics[name] = {
            "label": LABELS[name],
            "value": _clean(value),
            "flag": flag,
            "band": {k: _clean(v) for k, v in (band or {}).items() if k != "bootstrap"} if band else None,
            "bootstrap": (band or {}).get("bootstrap") if band else None,
            "z": slot.get("z"),
            "cusumPlus": _clean(slot.get("sPlus")),
            "cusumMinus": _clean(slot.get("sMinus")),
            "cusumAlarm": bool(slot.get("alarm")),
            "seqStat": slot.get("seqStat"),
            "seqAlarm": bool(slot.get("seqAlarm")),
            "cusumN": slot.get("n"),
        }
    status = overall_status(flags, cusum_state)
    if not realized.get("sessions"):
        status = "awaiting"
    breaches = breaches_sev or breaches_warn
    sessions = int(realized.get("sessions") or 0)
    compared = [n for n, f in flags.items() if f is not None]
    inside = [n for n, f in flags.items() if f == "ok"]
    note = (
        "Expected ranges come from the walk-forward out-of-sample distribution (empirical percentiles, "
        "with a moving-block bootstrap of 20-day return and vol). Each new paper day updates CUSUM and a "
        "sequential mean test so the monitor learns whether live behavior still matches the backtest. "
        "It does not auto-retune the strategy or shrink positions on its own."
    )
    report = {
        "generatedAt": _now(),
        "status": status,
        "source": realized.get("source") or "paper",
        "account": realized.get("account") or "default",
        "asof": realized.get("asof"),
        "equity": realized.get("equity"),
        "paperSessions": sessions,
        "expected": expected.get("source"),
        "metrics": metrics,
        "breaches": breaches,
        "persistent": persistent,
        "notes": list(expected.get("notes") or []) + list(realized.get("notes") or []),
        "learning": {
            "paperSessions": sessions,
            "metricsCompared": len(compared),
            "metricsInsideP95": len(inside),
            "persistentMetrics": persistent,
            "cusumThreshold": CUSUM_H,
            "sequentialThreshold": SEQ_CRITICAL,
            "note": note,
        },
        "href": "/paper2.html#drift",
    }
    return report, state


def default_paths():
    return {
        "wf_equity": WF_EQUITY,
        "wf_decisions": WF_DECISIONS,
        "dt_trades": DT_TRADES,
        "dt_summary": DT_SUMMARY,
        "paper_db": PAPER_DB,
        "live_db": LIVE_DB,
        "status": STATUS_FILE,
        "history": HISTORY_FILE,
        "state": STATE_FILE,
        "chat": CHAT_DB,
    }


def run(account="default", asof=None, source="paper", paths=None, emit_fn=None, chat_fn=None,
        memory_fn=None, alerts=True, chat=True, memory=True):
    paths = {**default_paths(), **(paths or {})}
    expected = build_expected(paths)
    decisions = _read_csv(paths["wf_decisions"])
    realized = load_realized(paths, account=account, source=source, asof=asof, decisions=decisions)
    state = load_state(paths["state"])
    same_day = bool(realized.get("asof") and state.get("lastAsOf") == realized.get("asof"))
    report, state = build_report(expected, realized, state, update_cusum_state=not same_day)
    state["lastAsOf"] = realized.get("asof")
    save_json(paths["state"], state)

    save_json(paths["status"], report)
    append_history(paths["history"], {k: report[k] for k in
                                      ("generatedAt", "status", "source", "account", "asof", "equity",
                                       "paperSessions", "breaches", "persistent") if k in report} |
                    {"metrics": {k: {"value": v.get("value"), "flag": v.get("flag")} for k, v in report["metrics"].items()}})

    side = {"alerts": None, "chat": None, "memory": None}
    alert_key = f"{report.get('asof')}:{report['status']}:{','.join(report.get('breaches') or [])}:{','.join(report.get('persistent') or [])}"
    should_alert = alerts and report["status"] in ("warning", "severe") and state.get("lastAlertKey") != alert_key
    if should_alert:
        side["alerts"] = emit_alerts(report, emit_fn=emit_fn)
        if side["alerts"].get("emitted"):
            state["lastAlertKey"] = alert_key
            save_json(paths["state"], state)
    elif alerts:
        side["alerts"] = {"ok": True, "emitted": False, "reason": "no new breach" if report["status"] in ("warning", "severe") else report["status"]}
    if chat and report.get("asof"):
        side["chat"] = write_chat(paths["chat"], report, insert_fn=chat_fn)
    if memory and report.get("asof"):
        side["memory"] = write_memory(report, ingest_fn=memory_fn)
    report["sideEffects"] = side
    return report


def status_only(paths=None):
    paths = {**default_paths(), **(paths or {})}
    path = Path(paths["status"])
    if not path.is_file():
        return {"status": "awaiting", "notes": ["No drift report yet. It is written after the daily paper run."]}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("run", "status"), default="run")
    parser.add_argument("--account", default="default")
    parser.add_argument("--asof")
    parser.add_argument("--source", choices=("paper", "live"), default="paper")
    parser.add_argument("--no-alerts", action="store_true")
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        doc = status_only()
        json.dump(doc, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    report = run(account=args.account, asof=args.asof, source=args.source,
                 alerts=not args.no_alerts, chat=not args.no_chat, memory=not args.no_memory)
    # Compact stdout for autopilot / the daily script.
    json.dump({
        "status": report["status"],
        "asof": report.get("asof"),
        "account": report.get("account"),
        "source": report.get("source"),
        "paperSessions": report.get("paperSessions"),
        "breaches": report.get("breaches"),
        "persistent": report.get("persistent"),
        "sideEffects": report.get("sideEffects"),
        "href": report.get("href"),
    }, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("drift_monitor"):
        main()
