"""Score the AI analyst against what actually happened, and gate position shrinkage on live skill.

Every advisory flag (GREEN/YELLOW/RED), stop and buy-opportunity call is logged with its timestamp and
price. Forward returns are measured at 5, 20 and 60 trading days from Massive/prices.sqlite3 against the
same-day S&P 500 equal-weight universe and a sector-ETF baseline. RED is tested for underperformance and
GREEN for outperformance (hit rate, mean excess, bootstrap and sign-flip permutation p-values, Newey-West
t-stats, sample-size guard). Stops are checked for a low that trades through the level. Calibration
(Brier score by flag) and a Spearman information coefficient are reported by horizon. Multiple tests are
adjusted with Holm and Benjamini-Hochberg.

Gating: verdicts may shrink or skip positions only while the LIVE out-of-sample record beats chance at
the configured significance with enough samples. A sealed historical replay (see analyst_replay.py) is
stored as a separate 'prior' and is an informative starting point only — it never opens the gate.
Missing, tiny or insignificant records mean the verdict is logged and has no effect.

Statistics note: engine/quant_tools.py (GB-2, as of 2026-11-25 21:08 ET) exposes moving-block bootstrap
confidence intervals and HAC/Newey-West t-tests as analyst *operations*, but not importable permutation
tests, BH/Holm corrections, Brier scores or information coefficients. Those are implemented here. If
matching helpers land in quant_tools later, prefer them.

Data stays under Massive/ (gitignored): Massive/Research/analyst_score.sqlite3 and analyst_scorecard.json.

  python analyst_score.py rescore
  python analyst_score.py status
  python analyst_score.py gate
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parent
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from fetch_and_model import DATA_DIR, PROJECT_ROOT, write_atomically

HORIZONS = (5, 20, 60)
GATE_HORIZON = 20
MIN_N = 30
ALPHA = 0.05
N_RESAMPLES = 4000
FLAGS = ("GREEN", "YELLOW", "RED")
SIGNAL = {"GREEN": 1.0, "YELLOW": 0.0, "RED": -1.0}
# Predicted P(positive excess) for Brier calibration.
PROB_POS = {"GREEN": 0.75, "YELLOW": 0.50, "RED": 0.25}
OPPORTUNITY_NONE = {"", "NONE", "NULL", "NONE.", "N/A"}
SECTOR_ETF = {
    "SEMIS": "XLK", "SOFTWARE": "XLK", "HARDWARE": "XLK", "TECH": "XLK",
    "FINANCIALS": "XLF", "HEALTHCARE": "XLV", "ENERGY": "XLE", "RETAIL": "XLY",
    "AUTOS": "XLY", "UTILITIES": "XLU", "REAL_ESTATE": "XLRE", "TELECOM": "XLC",
    "STAPLES": "XLP", "INDUSTRIALS": "XLI", "MATERIALS": "XLB", "COMMUNICATIONS": "XLC",
    "DISCRETIONARY": "XLY",
}
SECTOR_RULES = (
    ("SEMIS", "SEMICONDUCTOR|ELECTRONIC COMP"),
    ("SOFTWARE", "SOFTWARE|PREPACKAGED|COMPUTER PROGRAM|DATA PROCESS|INFORMATION RETRIEVAL"),
    ("HARDWARE", "COMPUTER|COMMUNICATIONS EQUIP|ELECTRONIC"),
    ("FINANCIALS", "BANK|SAVINGS|CREDIT|FINANCE|SECURITY BROKER|INSURANCE|INVESTMENT"),
    ("HEALTHCARE", "PHARMA|BIOLOG|MEDICAL|SURGICAL|HEALTH|HOSPITAL"),
    ("ENERGY", "PETROLEUM|CRUDE|OIL|GAS|NATURAL GAS"),
    ("RETAIL", "RETAIL|STORES|CATALOG|MAIL-ORDER"),
    ("AUTOS", "MOTOR VEHICLE|AUTO"),
    ("UTILITIES", "ELECTRIC SERVICES|UTILIT|WATER SUPPLY"),
    ("REAL_ESTATE", "REAL ESTATE|REIT"),
    ("TELECOM", "TELEPHONE|TELECOM|CABLE|RADIOTELEPHONE"),
    ("STAPLES", "FOOD|BEVERAGE|SOAP|HOUSEHOLD|TOBACCO|SUPERMARKET"),
    ("INDUSTRIALS", "INDUSTRIAL|MACHINERY|AEROSPACE|DEFENSE|AIRLINE|RAIL|TRUCK"),
    ("MATERIALS", "CHEMICAL|STEEL|MINING|METAL|PAPER|FOREST"),
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL,
  asof TEXT NOT NULL,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL,
  flag TEXT,
  opportunity TEXT,
  stop REAL,
  ref_price REAL,
  source TEXT,
  origin TEXT NOT NULL,
  model TEXT,
  payload TEXT,
  UNIQUE (ticker, asof, created_at, kind, source, origin)
);
CREATE INDEX IF NOT EXISTS calls_asof ON calls (asof, kind, flag);
CREATE INDEX IF NOT EXISTS calls_ticker ON calls (ticker, asof);
CREATE TABLE IF NOT EXISTS scorecard_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scored_at TEXT NOT NULL,
  asof TEXT NOT NULL,
  origin TEXT NOT NULL,
  payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baseline_cache (
  start TEXT NOT NULL,
  end TEXT NOT NULL,
  name TEXT NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY (start, end, name)
);
"""

DEFAULT_DB = DATA_DIR / "Research" / "analyst_score.sqlite3"
DEFAULT_CARD = DATA_DIR / "Research" / "analyst_score_eval.json"  # not the live gate file (analyst_scorecard.json is owned by analyst_scoring.py)
SP500_JSON = PROJECT_ROOT / "public" / "sp500.json"
ADVISORY_JSON = DATA_DIR / "Research" / "advisory_verdicts.json"
AI_RESEARCH_DB = DATA_DIR / "ai_research.sqlite3"
AI_MEMORY_DB = DATA_DIR / "ai_memory.sqlite3"
PRICES_DB = DATA_DIR / "prices.sqlite3"


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _num(value):
    try:
        if value is None or value == "":
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _clean(value, digits=6):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        return round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: _clean(v, digits) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v, digits) for v in value]
    return value


# ------------------------------------------------------------------ statistics (local; see module docstring)
def hac_lags(n, horizon=1):
    """Newey-West lags: at least horizon-1 for overlapping multi-day returns, else the usual HAC rule."""
    rule = max(1, int(4 * (max(n, 1) / 100) ** (2 / 9)))
    return max(rule, max(0, int(horizon) - 1))


def newey_west_t(values, lags=None, horizon=1):
    """One-sample HAC/Newey-West t-stat and two-sided p-value for the mean. Returns None if n < 3."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return None
    lags = hac_lags(n, horizon) if lags is None else int(lags)
    # OLS of x on a constant, Bartlett HAC.
    mu = float(x.mean())
    u = x - mu
    gamma0 = float(np.dot(u, u) / n)
    var = gamma0
    for lag in range(1, min(lags, n - 1) + 1):
        gamma = float(np.dot(u[lag:], u[:-lag]) / n)
        weight = 1.0 - lag / (lags + 1)
        var += 2.0 * weight * gamma
    se = math.sqrt(max(var, 0.0) / n)
    if se <= 0:
        return {"n": n, "mean": mu, "t": None, "p": None, "lags": lags, "se": 0.0}
    t = mu / se
    # Regularized normal tail (n is often modest; t_{n-1} would need scipy — use a close approximation).
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        p = float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return {"n": n, "mean": mu, "t": t, "p": p, "lags": lags, "se": se}


def _one_sided(p_two, stat, alternative):
    """Convert a two-sided p and a signed statistic into a one-sided p."""
    if p_two is None or stat is None:
        return None
    if alternative == "two-sided":
        return float(min(1.0, p_two))
    if alternative == "greater":
        return float(p_two / 2 if stat >= 0 else 1 - p_two / 2)
    return float(p_two / 2 if stat <= 0 else 1 - p_two / 2)


def bootstrap_mean_p(values, alternative="greater", n_boot=N_RESAMPLES, block=1, seed=7):
    """Moving-block bootstrap p-value for H0: mean = 0 (one- or two-sided)."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return None
    block = max(1, min(int(block), max(1, n // 2)))
    rng = np.random.default_rng(seed)
    k = math.ceil(n / block)
    starts = rng.integers(0, max(1, n - block + 1), size=(n_boot, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    draws = x[np.clip(idx, 0, n - 1)].mean(axis=1)
    mu = float(x.mean())
    # Center under H0: mean = 0 by subtracting mu, then compare the observed mean to the null draws.
    null = draws - draws.mean()
    if alternative == "greater":
        p = (np.sum(null >= mu) + 1) / (n_boot + 1)
    elif alternative == "less":
        p = (np.sum(null <= mu) + 1) / (n_boot + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(mu)) + 1) / (n_boot + 1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n": n, "mean": mu, "p": float(p), "ciLow": float(lo), "ciHigh": float(hi), "block": block, "nBoot": n_boot}


def permutation_mean_p(values, alternative="greater", n_perm=N_RESAMPLES, seed=7):
    """Sign-flip permutation p-value for H0: mean = 0."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return None
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null = (signs * x).mean(axis=1)
    mu = float(x.mean())
    if alternative == "greater":
        p = (np.sum(null >= mu) + 1) / (n_perm + 1)
    elif alternative == "less":
        p = (np.sum(null <= mu) + 1) / (n_perm + 1)
    else:
        p = (np.sum(np.abs(null) >= abs(mu)) + 1) / (n_perm + 1)
    return {"n": n, "mean": mu, "p": float(p), "nPerm": n_perm}


def spearman_ic(scores, returns, n_perm=N_RESAMPLES, seed=7):
    """Spearman information coefficient and a two-sided permutation p-value."""
    s = np.asarray(scores, dtype=float)
    r = np.asarray(returns, dtype=float)
    mask = np.isfinite(s) & np.isfinite(r)
    s, r = s[mask], r[mask]
    n = len(s)
    if n < 5 or np.unique(s).size < 2 or np.unique(r).size < 2:
        return {"n": n, "ic": None, "p": None}
    ic = float(pd.Series(s).corr(pd.Series(r), method="spearman"))
    if not math.isfinite(ic):
        return {"n": n, "ic": None, "p": None}
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = pd.Series(s).corr(pd.Series(rng.permutation(r)), method="spearman")
    p = (np.sum(np.abs(null) >= abs(ic)) + 1) / (n_perm + 1)
    return {"n": n, "ic": ic, "p": float(p), "nPerm": n_perm}


def brier_score(predictions, outcomes):
    """Brier score (mean squared error of predicted probabilities). Lower is better; 0.25 is a coin-flip."""
    p = np.asarray(predictions, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0:
        return None
    return {"n": int(len(p)), "brier": float(np.mean((p - y) ** 2)), "avgPredicted": float(p.mean()),
            "avgOutcome": float(y.mean())}


def holm(pvalues, alpha=ALPHA):
    """Holm step-down adjustment. pvalues: list of (name, p). Returns dict name -> {p, pAdj, reject}."""
    items = [(n, p) for n, p in pvalues if p is not None and math.isfinite(p)]
    m = len(items)
    out = {n: {"p": p, "pAdj": None, "reject": False} for n, p in pvalues}
    if m == 0:
        return out
    items.sort(key=lambda kv: kv[1])
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items, start=1):
        running = max(running, min(1.0, p * (m - i + 1)))
        adj[name] = running
        out[name] = {"p": p, "pAdj": running, "reject": running <= alpha}
    return out


def benjamini_hochberg(pvalues, alpha=ALPHA):
    """BH FDR adjustment. pvalues: list of (name, p)."""
    items = [(n, p) for n, p in pvalues if p is not None and math.isfinite(p)]
    m = len(items)
    out = {n: {"p": p, "pAdj": None, "reject": False} for n, p in pvalues}
    if m == 0:
        return out
    items.sort(key=lambda kv: kv[1], reverse=True)
    running = 1.0
    stored = {}
    # Walk from largest p to smallest so adjusted p is nondecreasing in the raw p.
    for i_from_end, (name, p) in enumerate(items):
        rank = m - i_from_end
        running = min(running, min(1.0, p * m / rank))
        stored[name] = running
    items_asc = sorted(((n, p) for n, p in pvalues if p is not None), key=lambda kv: kv[1])
    cutoff = 0
    for i, (name, p) in enumerate(items_asc, start=1):
        if p <= alpha * i / m:
            cutoff = i
    rejected = {name for name, _ in items_asc[:cutoff]}
    for name, p in pvalues:
        if name in stored:
            out[name] = {"p": p, "pAdj": stored[name], "reject": name in rejected}
    return out


# ------------------------------------------------------------------ storage
def connect_score(path=None):
    path = Path(path or os.environ.get("ANALYST_SCORE_DB") or DEFAULT_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.executescript(SCHEMA)
    return db


def _flag(value):
    flag = str(value or "").upper().strip()
    return flag if flag in FLAGS else None


def _asof(value):
    if not value:
        return None
    text = str(value).strip()
    try:
        return pd.Timestamp(text[:10] if len(text) >= 10 and text[4] == "-" else text).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def log_call(db, ticker, asof, created_at, kind, flag=None, opportunity=None, stop=None, ref_price=None,
             source=None, origin="live", model=None, payload=None):
    """Insert one call. Returns True if a new row was written."""
    ticker = str(ticker or "").upper().strip()
    asof = _asof(asof)
    kind = str(kind or "").upper()
    if not ticker or not asof or kind not in {"FLAG", "STOP", "OPPORTUNITY"}:
        return False
    created_at = created_at or f"{asof}T00:00:00Z"
    opportunity = None if opportunity is None or str(opportunity).upper().strip() in OPPORTUNITY_NONE else str(opportunity).upper()
    try:
        cur = db.execute(
            """INSERT OR IGNORE INTO calls
               (ticker, asof, created_at, kind, flag, opportunity, stop, ref_price, source, origin, model, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, asof, str(created_at), kind, _flag(flag), opportunity, _num(stop), _num(ref_price),
             source, origin, model, json.dumps(payload) if payload is not None and not isinstance(payload, str) else payload))
        return cur.rowcount > 0
    except sqlite3.Error:
        return False


def _expand_verdict(db, ticker, asof, created_at, flag, opportunity, stop, ref_price, source, origin, model, payload):
    n = 0
    if _flag(flag):
        n += log_call(db, ticker, asof, created_at, "FLAG", flag=flag, opportunity=opportunity, stop=stop,
                      ref_price=ref_price, source=source, origin=origin, model=model, payload=payload)
    if _num(stop) is not None:
        n += log_call(db, ticker, asof, created_at, "STOP", flag=flag, stop=stop, ref_price=ref_price,
                      source=source, origin=origin, model=model, payload=payload)
    if opportunity and str(opportunity).upper().strip() not in OPPORTUNITY_NONE:
        n += log_call(db, ticker, asof, created_at, "OPPORTUNITY", flag=flag, opportunity=opportunity, stop=stop,
                      ref_price=ref_price, source=source, origin=origin, model=model, payload=payload)
    return n


def harvest_advisory_json(db, path=None, origin="live"):
    path = Path(path or os.environ.get("ADVISORY_VERDICTS_PATH") or ADVISORY_JSON)
    if not path.is_file():
        return 0, []
    warnings = []
    added = 0
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        return 0, [f"advisory JSON ignored: {error}"]
    generated = data.get("generatedAt") if isinstance(data, dict) else None
    default_asof = _asof(data.get("asof") if isinstance(data, dict) else None)
    items = data.get("verdicts", data) if isinstance(data, dict) else data
    pairs = items.items() if isinstance(items, dict) else ((i.get("ticker", ""), i) for i in (items or []))
    for ticker, item in pairs:
        if not isinstance(item, dict) or not ticker:
            continue
        added += _expand_verdict(
            db, ticker, item.get("asof") or default_asof, item.get("created_at") or generated,
            item.get("flag"), item.get("opportunity"), item.get("stop"),
            item.get("ref_price") or item.get("refPrice"),
            item.get("source") or "advisory_json", origin, item.get("model"), item)
    return added, warnings


def harvest_ai_research(db, path=None, origin="live"):
    path = Path(path or AI_RESEARCH_DB)
    if not path.is_file():
        return 0, []
    added, warnings = 0, []
    try:
        with sqlite3.connect(path, timeout=30) as src:
            src.row_factory = sqlite3.Row
            if not src.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ai_verdicts'").fetchone():
                return 0, []
            for row in src.execute("SELECT * FROM ai_verdicts"):
                payload = None
                try:
                    payload = json.loads(row["payload"]) if row["payload"] else None
                except (TypeError, ValueError):
                    payload = None
                added += _expand_verdict(
                    db, row["ticker"], row["asof"] or row["created_at"], row["created_at"],
                    row["flag"], row["opportunity"], row["stop"], row["ref_price"],
                    row["source"] or "ai_research", origin, row["model"], payload or dict(row))
    except sqlite3.Error as error:
        warnings.append(f"ai_research ignored: {error}")
    return added, warnings


def harvest_memory(db, path=None, origin="live"):
    path = Path(path or AI_MEMORY_DB)
    if not path.is_file():
        return 0, []
    added, warnings = 0, []
    try:
        with sqlite3.connect(path, timeout=30) as src:
            src.row_factory = sqlite3.Row
            if not src.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_items'").fetchone():
                return 0, []
            for row in src.execute("SELECT ticker, ts, text, meta FROM memory_items WHERE kind='VERDICT'"):
                try:
                    meta = json.loads(row["meta"] or "{}")
                except ValueError:
                    meta = {}
                flag = meta.get("flag")
                if not _flag(flag):
                    import re
                    match = re.search(r"\b(GREEN|YELLOW|RED)\b", row["text"] or "", re.I)
                    flag = match.group(1).upper() if match else None
                added += _expand_verdict(
                    db, row["ticker"], meta.get("asof") or row["ts"], row["ts"],
                    flag, meta.get("opportunity"), meta.get("stop"), meta.get("refPrice") or meta.get("ref_price"),
                    meta.get("source") or "memory", origin, meta.get("model"), {"text": row["text"], "meta": meta})
    except sqlite3.Error as error:
        warnings.append(f"memory ignored: {error}")
    return added, warnings


def harvest_all(db, origin="live", json_path=None, research_path=None, memory_path=None):
    added, warnings = 0, []
    for fn, kwargs in (
        (harvest_advisory_json, {"path": json_path, "origin": origin}),
        (harvest_ai_research, {"path": research_path, "origin": origin}),
        (harvest_memory, {"path": memory_path, "origin": origin}),
    ):
        n, w = fn(db, **{k: v for k, v in kwargs.items() if v is not None or k == "origin"})
        added += n
        warnings.extend(w)
    db.commit()
    return added, warnings


# ------------------------------------------------------------------ prices and baselines
def connect_prices(path=None):
    path = Path(path or os.environ.get("PRICES_DB") or PRICES_DB)
    if not path.is_file():
        raise FileNotFoundError(f"price database not found: {path}")
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def load_sp500(path=None):
    path = Path(path or SP500_JSON)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [str(t).upper() for t in data.get("tickers", [])]
    except (OSError, ValueError):
        return []


def session_calendar(prices, ticker="SPY"):
    rows = prices.execute("SELECT DISTINCT date FROM daily_prices WHERE ticker=? ORDER BY date", (ticker,)).fetchall()
    if not rows:
        rows = prices.execute("SELECT DISTINCT date FROM daily_prices ORDER BY date").fetchall()
    return [r[0] if not isinstance(r, sqlite3.Row) else r["date"] for r in rows]


def _session_pos(calendar, asof):
    """Index of the last session on or before asof, or None."""
    asof = _asof(asof)
    if not asof or not calendar:
        return None
    import bisect
    i = bisect.bisect_right(calendar, asof) - 1
    return i if i >= 0 else None


def bar_on(prices, ticker, date):
    row = prices.execute(
        "SELECT date, open, high, low, close, volume FROM daily_prices WHERE ticker=? AND date=?",
        (ticker, date)).fetchone()
    return None if row is None else dict(row)


def load_closes(prices, tickers, start=None, end=None):
    tickers = [t.upper() for t in tickers]
    if not tickers:
        return {}
    q = "SELECT ticker, date, close, low, high FROM daily_prices WHERE ticker IN (%s)" % ",".join("?" * len(tickers))
    args = list(tickers)
    if start:
        q += " AND date>=?"
        args.append(start)
    if end:
        q += " AND date<=?"
        args.append(end)
    q += " ORDER BY ticker, date"
    out = {}
    for row in prices.execute(q, args):
        out.setdefault(row["ticker"], []).append(row)
    return out


def forward_return(calendar, closes, asof, horizon):
    """Close-to-close simple return over `horizon` trading days. closes: list of rows or {date: close}."""
    i = _session_pos(calendar, asof)
    if i is None or i + horizon >= len(calendar):
        return None
    start, end = calendar[i], calendar[i + horizon]
    if isinstance(closes, dict) and "close" not in closes:
        p0, p1 = closes.get(start), closes.get(end)
    else:
        by_date = {r["date"] if isinstance(r, (dict, sqlite3.Row)) else r[0]:
                   (r["close"] if isinstance(r, (dict, sqlite3.Row)) else r[1]) for r in closes}
        p0, p1 = by_date.get(start), by_date.get(end)
    p0, p1 = _num(p0), _num(p1)
    if p0 is None or p1 is None or p0 <= 0:
        return None
    return {"start": start, "end": end, "startPrice": p0, "endPrice": p1, "ret": p1 / p0 - 1, "horizon": horizon}


def path_to(calendar, rows, asof, horizon):
    """Bars from the asof session through asof+horizon (inclusive), if available."""
    i = _session_pos(calendar, asof)
    if i is None:
        return None
    last = min(i + horizon, len(calendar) - 1)
    want = set(calendar[i:last + 1])
    out = [r for r in rows if (r["date"] if isinstance(r, (dict, sqlite3.Row)) else r[0]) in want]
    return out


def stop_hit(calendar, rows, asof, stop, horizon=60, ref_price=None):
    """Whether the low trades through a long-side stop in the next `horizon` sessions (after asof)."""
    stop = _num(stop)
    if stop is None:
        return None
    i = _session_pos(calendar, asof)
    if i is None:
        return None
    # Evaluate on sessions after the decision date (the close used for the call is already known).
    end_i = min(i + horizon, len(calendar) - 1)
    want = calendar[i + 1:end_i + 1]
    by_date = {r["date"] if isinstance(r, (dict, sqlite3.Row)) else r[0]: r for r in rows}
    for d in want:
        bar = by_date.get(d)
        if bar is None:
            continue
        low = _num(bar["low"] if isinstance(bar, (dict, sqlite3.Row)) else bar[3] if len(bar) > 3 else None)
        if low is not None and low <= stop:
            return {"hit": True, "hitDate": d, "hitLow": low, "stop": stop, "horizon": horizon,
                    "daysToHit": calendar.index(d) - i}
    complete = (i + horizon) < len(calendar)
    return {"hit": False, "hitDate": None, "hitLow": None, "stop": stop, "horizon": horizon,
            "daysToHit": None, "complete": complete}


def _ew_universe_return(prices, start, end, tickers, cache_db=None, name="universe_sp500"):
    if cache_db is not None:
        row = cache_db.execute("SELECT value FROM baseline_cache WHERE start=? AND end=? AND name=?",
                               (start, end, name)).fetchone()
        if row:
            return row["value"]
    if tickers:
        chunk = 400
        total, count = 0.0, 0
        for i in range(0, len(tickers), chunk):
            part = tickers[i:i + chunk]
            q = ("SELECT AVG(e.close * 1.0 / s.close - 1) AS r, COUNT(*) AS n "
                 "FROM daily_prices s JOIN daily_prices e ON e.ticker=s.ticker AND e.date=? "
                 "WHERE s.date=? AND s.close>0 AND e.close>0 AND s.ticker IN (%s)" % ",".join("?" * len(part)))
            row = prices.execute(q, [end, start, *part]).fetchone()
            r, n = (row["r"] if row else None), (row["n"] if row else 0)
            if r is not None and n:
                total += float(r) * n
                count += n
        value = (total / count) if count else None
    else:
        value = None
    if cache_db is not None and value is not None:
        cache_db.execute("INSERT OR REPLACE INTO baseline_cache VALUES (?, ?, ?, ?)", (start, end, name, value))
    return value


def _ticker_return(prices, ticker, start, end):
    row = prices.execute(
        "SELECT s.close AS p0, e.close AS p1 FROM daily_prices s JOIN daily_prices e ON e.ticker=s.ticker AND e.date=? "
        "WHERE s.ticker=? AND s.date=? AND s.close>0 AND e.close>0",
        (end, ticker, start)).fetchone()
    if not row:
        return None
    return float(row["p1"]) / float(row["p0"]) - 1


def sector_of_name(name):
    text = f" {str(name or '').upper()} "
    import re
    for label, pattern in SECTOR_RULES:
        if re.search(pattern, text):
            return label
    return None


def load_sector_map(directory_path=None):
    """Map ticker -> sector ETF using the Nasdaq name directory when present; otherwise CORE ETFs only."""
    mapping = {etf: etf for etf in ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")}
    try:
        from universe import REFERENCE, read_directory
        table = read_directory(directory_path or REFERENCE)
    except Exception:
        table = None
    if table is None:
        return mapping
    for ticker, row in table.iterrows():
        label = sector_of_name(row.get("Security Name"))
        if label and label in SECTOR_ETF:
            mapping[str(ticker).upper()] = SECTOR_ETF[label]
    return mapping


# ------------------------------------------------------------------ evaluation
def _series_stats(excess, alternative, horizon, min_n=MIN_N, alpha=ALPHA, seed=7):
    x = [v for v in excess if v is not None and math.isfinite(v)]
    n = len(x)
    out = {"n": n, "sufficient": n >= min_n, "hitRate": None, "meanExcess": None,
           "bootstrap": None, "permutation": None, "neweyWest": None, "beatsChance": False}
    if n == 0:
        return out
    arr = np.asarray(x, dtype=float)
    out["hitRate"] = float((arr > 0).mean()) if alternative == "greater" else float((arr < 0).mean())
    out["meanExcess"] = float(arr.mean())
    if n < 3:
        return out
    boot = bootstrap_mean_p(arr, alternative=alternative, block=horizon, seed=seed)
    perm = permutation_mean_p(arr, alternative=alternative, seed=seed)
    nw = newey_west_t(arr, horizon=horizon)
    nw_p = _one_sided(nw["p"], nw["t"], alternative) if nw and nw["p"] is not None else None
    if nw is not None:
        nw = dict(nw)
        nw["pOneSided"] = nw_p
    out["bootstrap"] = boot
    out["permutation"] = perm
    out["neweyWest"] = nw
    p_boot = boot["p"] if boot else None
    p_perm = perm["p"] if perm else None
    # "Beats chance" requires the sample-size guard and a significant one-sided test in the hypothesized direction,
    # with the mean also in that direction. The gate later applies a multiple-testing correction on these p-values.
    mean_ok = (out["meanExcess"] > 0) if alternative == "greater" else (out["meanExcess"] < 0)
    raw = [p for p in (p_boot, p_perm, nw_p) if p is not None]
    out["beatsChance"] = bool(out["sufficient"] and mean_ok and raw and min(raw) <= alpha)
    out["pMin"] = min(raw) if raw else None
    return out


def evaluate_calls(rows, calendar, prices, sector_map=None, universe_tickers=None, min_n=MIN_N,
                   alpha=ALPHA, origin="live", seed=7, cache_db=None):
    """Score a list of call rows (sqlite Row or dict). Returns a scorecard block (no gate)."""
    sector_map = sector_map or {}
    universe_tickers = universe_tickers if universe_tickers is not None else load_sp500()
    flags = {h: {f: [] for f in FLAGS} for h in HORIZONS}
    raw = {h: [] for h in HORIZONS}          # (signal, excess_universe, flag) for IC / Brier
    stops = {h: [] for h in HORIZONS}
    opportunities = {h: [] for h in HORIZONS}
    pending = {h: 0 for h in HORIZONS}
    by_ticker = {}
    source_counts = {"LLM": 0, "FALLBACK": 0, "other": 0, "FLAG": 0, "STOP": 0, "OPPORTUNITY": 0}

    # Group price loads.
    tickers = sorted({(r["ticker"] if isinstance(r, (dict, sqlite3.Row)) else r[1]) for r in rows})
    # Load each ticker's closes once (full history is OK here: evaluation may use future prices; the model cannot).
    book = load_closes(prices, list(set(tickers) | {"SPY"} | set(SECTOR_ETF.values())))
    closes_by = {}
    for t, recs in book.items():
        closes_by[t] = {r["date"]: r["close"] for r in recs}

    def src_bucket(source):
        s = str(source or "").upper()
        if s == "FALLBACK" or s.startswith("FALLBACK"):
            return "FALLBACK"
        if s == "LLM" or s.startswith("AI:") or s.startswith("LLM"):
            return "LLM"
        return "other"

    for r in rows:
        d = dict(r)
        kind, ticker, asof = d["kind"], d["ticker"], d["asof"]
        source_counts[kind] = source_counts.get(kind, 0) + 1
        source_counts[src_bucket(d.get("source"))] = source_counts.get(src_bucket(d.get("source")), 0) + 1
        recs = book.get(ticker, [])
        ticker_entry = by_ticker.setdefault(ticker, {"calls": [], "flags": {}})
        if kind == "FLAG" and _flag(d.get("flag")):
            flag = d["flag"]
            for h in HORIZONS:
                fwd = forward_return(calendar, closes_by.get(ticker, {}), asof, h)
                if fwd is None:
                    pending[h] += 1
                    ticker_entry["calls"].append({"kind": "FLAG", "flag": flag, "asof": asof, "horizon": h,
                                                  "status": "pending"})
                    continue
                uni = _ew_universe_return(prices, fwd["start"], fwd["end"], universe_tickers, cache_db)
                etf = sector_map.get(ticker)
                sec = _ticker_return(prices, etf, fwd["start"], fwd["end"]) if etf else None
                spy = _ticker_return(prices, "SPY", fwd["start"], fwd["end"])
                excess_u = None if uni is None else fwd["ret"] - uni
                excess_s = None if sec is None else fwd["ret"] - sec
                excess_m = None if spy is None else fwd["ret"] - spy
                flags[h][flag].append({
                    "ticker": ticker, "asof": asof, "ret": fwd["ret"], "universe": uni, "sector": sec, "spy": spy,
                    "excessUniverse": excess_u, "excessSector": excess_s, "excessSpy": excess_m,
                    "end": fwd["end"], "source": d.get("source"),
                })
                raw[h].append((SIGNAL[flag], excess_u, flag, PROB_POS[flag],
                               1.0 if (excess_u is not None and excess_u > 0) else (0.0 if excess_u is not None else None)))
                ticker_entry["calls"].append(_clean({
                    "kind": "FLAG", "flag": flag, "asof": asof, "horizon": h, "status": "scored",
                    "ret": fwd["ret"], "excessUniverse": excess_u, "excessSector": excess_s, "end": fwd["end"],
                }))
        elif kind == "STOP" and _num(d.get("stop")) is not None:
            for h in HORIZONS:
                hit = stop_hit(calendar, recs, asof, d["stop"], horizon=h, ref_price=d.get("ref_price"))
                if hit is None:
                    continue
                if hit.get("hit") or hit.get("complete", True):
                    stops[h].append({**hit, "ticker": ticker, "asof": asof})
                ticker_entry["calls"].append(_clean({"kind": "STOP", "asof": asof, "horizon": h, **hit}))
        elif kind == "OPPORTUNITY":
            for h in HORIZONS:
                fwd = forward_return(calendar, closes_by.get(ticker, {}), asof, h)
                if fwd is None:
                    pending[h] += 1
                    continue
                uni = _ew_universe_return(prices, fwd["start"], fwd["end"], universe_tickers, cache_db)
                excess_u = None if uni is None else fwd["ret"] - uni
                opportunities[h].append({"ticker": ticker, "asof": asof, "opportunity": d.get("opportunity"),
                                         "ret": fwd["ret"], "excessUniverse": excess_u, "end": fwd["end"]})

    if cache_db is not None:
        cache_db.commit()

    def flag_block(flag, h):
        items = flags[h][flag]
        alt = "greater" if flag == "GREEN" else ("less" if flag == "RED" else "two-sided")
        # YELLOW is not given a directional "beats chance" claim for the gate; we still report the mean.
        stats = _series_stats([i["excessUniverse"] for i in items], alternative=alt if flag != "YELLOW" else "two-sided",
                              horizon=h, min_n=min_n, alpha=alpha, seed=seed)
        sec_stats = _series_stats([i["excessSector"] for i in items if i["excessSector"] is not None],
                                  alternative=alt if flag != "YELLOW" else "two-sided",
                                  horizon=h, min_n=min_n, alpha=alpha, seed=seed + 1)
        stats["meanExcessSector"] = sec_stats["meanExcess"]
        stats["nSector"] = sec_stats["n"]
        stats["meanReturn"] = float(np.mean([i["ret"] for i in items])) if items else None
        return _clean(stats)

    flag_scores = {f: {str(h): flag_block(f, h) for h in HORIZONS} for f in FLAGS}

    ics, briers, brier_by_flag = {}, {}, {f: {} for f in FLAGS}
    for h in HORIZONS:
        scores = [row[0] for row in raw[h] if row[1] is not None]
        ex = [row[1] for row in raw[h] if row[1] is not None]
        ics[str(h)] = _clean(spearman_ic(scores, ex, seed=seed))
        preds = [row[3] for row in raw[h] if row[4] is not None]
        ys = [row[4] for row in raw[h] if row[4] is not None]
        briers[str(h)] = _clean(brier_score(preds, ys))
        for f in FLAGS:
            pred = [row[3] for row in raw[h] if row[2] == f and row[4] is not None]
            y = [row[4] for row in raw[h] if row[2] == f and row[4] is not None]
            brier_by_flag[f][str(h)] = _clean(brier_score(pred, y))

    stop_scores = {}
    for h in HORIZONS:
        items = stops[h]
        complete = [i for i in items if i.get("hit") or i.get("complete", True)]
        hits = [i for i in complete if i.get("hit")]
        days = [i["daysToHit"] for i in hits if i.get("daysToHit") is not None]
        stop_scores[str(h)] = _clean({
            "n": len(complete), "hits": len(hits),
            "hitRate": (len(hits) / len(complete)) if complete else None,
            "medianDaysToHit": float(np.median(days)) if days else None,
        })

    opp_scores = {}
    for h in HORIZONS:
        items = opportunities[h]
        opp_scores[str(h)] = _clean(_series_stats(
            [i["excessUniverse"] for i in items], alternative="greater", horizon=h,
            min_n=min_n, alpha=alpha, seed=seed + 2))
        opp_scores[str(h)]["kinds"] = {}
        kinds = {}
        for i in items:
            kinds.setdefault(i.get("opportunity") or "UNKNOWN", []).append(i["excessUniverse"])
        for k, vals in kinds.items():
            opp_scores[str(h)]["kinds"][k] = _clean(_series_stats(vals, "greater", h, min_n, alpha, seed))

    # Multiple-testing family: GREEN (greater) and RED (less) mean-excess p-values at each horizon (permutation).
    family = []
    for f, alt_name in (("GREEN", "outperform"), ("RED", "underperform")):
        for h in HORIZONS:
            block = flag_scores[f][str(h)]
            p = (block.get("permutation") or {}).get("p") if isinstance(block.get("permutation"), dict) else None
            family.append((f"{f}-{h}d-{alt_name}", p))
    holm_adj = holm(family, alpha)
    bh_adj = benjamini_hochberg(family, alpha)

    return _clean({
        "origin": origin,
        "calls": source_counts,
        "pendingHorizons": pending,
        "flags": flag_scores,
        "calibration": {"brierByHorizon": briers, "brierByFlag": brier_by_flag},
        "informationCoefficient": ics,
        "stops": stop_scores,
        "opportunities": opp_scores,
        "multipleTesting": {"holm": holm_adj, "benjaminiHochberg": bh_adj, "family": [n for n, _ in family]},
        "tickers": {t: {"nCalls": len(v["calls"]), "calls": v["calls"][:40]} for t, v in by_ticker.items()},
    })


def _flag_passes(block, adj, key, alternative, min_n=MIN_N, alpha=ALPHA):
    if not block or not block.get("sufficient") or block.get("n", 0) < min_n:
        return False
    mean = block.get("meanExcess")
    if mean is None:
        return False
    if alternative == "greater" and not (mean > 0):
        return False
    if alternative == "less" and not (mean < 0):
        return False
    entry = adj.get(key) or {}
    p_adj = entry.get("pAdj")
    if p_adj is None:
        return False
    return bool(entry.get("reject") or p_adj <= alpha)


def decide_gate(live_block, min_n=MIN_N, alpha=ALPHA, horizon=GATE_HORIZON):
    """Gate from the LIVE block only. Prior is ignored by the caller."""
    closed = {"open": False, "redMaySkip": False, "yellowMayShrink": False, "bandMayDefer": False, "reason": ""}
    if not live_block:
        closed["reason"] = "No live scorecard yet. Verdicts are logged and have no effect."
        return closed
    flags = live_block.get("flags") or {}
    adj = (live_block.get("multipleTesting") or {}).get("holm") or {}
    green = (flags.get("GREEN") or {}).get(str(horizon)) or {}
    red = (flags.get("RED") or {}).get(str(horizon)) or {}
    yellow = (flags.get("YELLOW") or {}).get(str(horizon)) or {}
    green_ok = _flag_passes(green, adj, f"GREEN-{horizon}d-outperform", "greater", min_n, alpha)
    red_ok = _flag_passes(red, adj, f"RED-{horizon}d-underperform", "less", min_n, alpha)
    # YELLOW may shrink only when the flag system itself has skill (both sides) or YELLOW has a significant
    # negative mean excess with enough samples (two-sided is not enough; require mean < 0 and raw pMin <= alpha).
    yellow_self = bool(yellow.get("sufficient") and (yellow.get("meanExcess") or 0) < 0
                       and yellow.get("pMin") is not None and yellow["pMin"] <= alpha)
    red_may = red_ok
    yellow_may = (green_ok and red_ok) or yellow_self
    band_may = green_ok and red_ok
    reasons = []
    if (green.get("n") or 0) < min_n or (red.get("n") or 0) < min_n:
        reasons.append(
            f"Insufficient live out-of-sample sample at {horizon}d "
            f"(GREEN n={green.get('n') or 0}, RED n={red.get('n') or 0}; need {min_n} each for a two-sided system).")
    if (green.get("n") or 0) >= min_n and not green_ok:
        reasons.append(f"GREEN does not significantly outperform the universe at {horizon}d after Holm correction.")
    if (red.get("n") or 0) >= min_n and not red_ok:
        reasons.append(f"RED does not significantly underperform the universe at {horizon}d after Holm correction.")
    if not reasons and not (red_may or yellow_may or band_may):
        reasons.append("Live tests do not yet beat chance; verdicts are logged and have no effect.")
    if not reasons:
        parts = []
        if red_may:
            parts.append("RED may skip new entries")
        if yellow_may:
            parts.append("YELLOW may shrink new entries")
        if band_may:
            parts.append("entry-band deferral may apply")
        reasons.append("Live record beats chance at %dd: %s." % (horizon, "; ".join(parts)))
    return {
        "open": bool(red_may or yellow_may or band_may),
        "redMaySkip": bool(red_may),
        "yellowMayShrink": bool(yellow_may),
        "bandMayDefer": bool(band_may),
        "horizon": horizon, "minN": min_n, "alpha": alpha,
        "reason": " ".join(reasons),
    }


def prior_is_not_a_gate(prior_block):
    reason = ("Replay/prior scores are an informative starting point only. "
              "They never let verdicts shrink or skip positions.")
    return {"open": False, "redMaySkip": False, "yellowMayShrink": False, "bandMayDefer": False,
            "reason": reason, "hasPrior": bool(prior_block)}


def build_scorecard(live, prior=None, asof=None, warnings=None, settings=None, leakage=None):
    settings = settings or {"minN": MIN_N, "alpha": ALPHA, "gateHorizon": GATE_HORIZON, "horizons": list(HORIZONS)}
    live = dict(live or {})
    live["gate"] = decide_gate(live, settings.get("minN", MIN_N), settings.get("alpha", ALPHA),
                               settings.get("gateHorizon", GATE_HORIZON))
    prior_out = None
    if prior:
        prior_out = dict(prior)
        prior_out["gate"] = prior_is_not_a_gate(prior)
        if leakage:
            prior_out["leakage"] = leakage
    return _clean({
        "schema": "analyst_scorecard/v1",
        "scoredAt": _now_iso(),
        "asof": _asof(asof) or datetime.now().strftime("%Y-%m-%d"),
        "settings": settings,
        "statsNote": (
            "Permutation (sign-flip) and moving-block bootstrap p-values, Newey-West t-stats, Holm and "
            "Benjamini-Hochberg adjustments, Brier calibration and Spearman ICs are computed in analyst_score.py. "
            "quant_tools.py (as of 2026-11-25) has related bootstrap CI and HAC t-test *ops* but not these helpers."
        ),
        "warnings": warnings or [],
        "live": live,
        "prior": prior_out,
    })


def write_scorecard(card, path=None):
    path = Path(path or os.environ.get("ANALYST_SCORECARD") or DEFAULT_CARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(card, indent=2, default=str).encode("utf-8"))
    return path


def read_scorecard(path=None):
    path = Path(path or os.environ.get("ANALYST_SCORECARD") or DEFAULT_CARD)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_live_gate(path=None):
    card = read_scorecard(path)
    if not card:
        return {"open": False, "redMaySkip": False, "yellowMayShrink": False, "bandMayDefer": False,
                "reason": "No live scorecard yet. Verdicts are logged and have no effect."}
    return (card.get("live") or {}).get("gate") or {
        "open": False, "redMaySkip": False, "yellowMayShrink": False, "bandMayDefer": False,
        "reason": "Live gate missing from the scorecard. Verdicts are logged and have no effect.",
    }


def live_effect_allowed(verdict, note, gate=None):
    """True only when the live (not prior) record authorizes this kind of shrinkage/skip."""
    if not note:
        return False
    gate = gate if gate is not None else load_live_gate()
    if note == "advisory_red":
        return bool(gate.get("redMaySkip"))
    if note == "advisory_yellow":
        return bool(gate.get("yellowMayShrink"))
    if note in {"advisory_below_band", "advisory_above_band"}:
        return bool(gate.get("bandMayDefer"))
    return False


def memory_text(card):
    live, gate = (card or {}).get("live") or {}, ((card or {}).get("live") or {}).get("gate") or {}
    flags = live.get("flags") or {}
    def n(flag):
        return ((flags.get(flag) or {}).get(str(GATE_HORIZON)) or {}).get("n") or 0
    prior = (card or {}).get("prior")
    prior_note = " No replay prior yet."
    if prior:
        leak = (prior.get("leakage") or {}).get("warning")
        prior_note = " Replay prior is on file (informative only; it does not open the gate)."
        if leak:
            prior_note += " Leakage check: " + leak
    g20 = (flags.get("GREEN") or {}).get("20") or {}
    r20 = (flags.get("RED") or {}).get("20") or {}
    ic = ((live.get("informationCoefficient") or {}).get("20") or {})
    return (
        f"Analyst live scorecard as of {card.get('asof')}: gate "
        f"{'OPEN' if gate.get('open') else 'CLOSED'}. {gate.get('reason') or ''} "
        f"20-day live samples: GREEN n={n('GREEN')} mean excess={g20.get('meanExcess')!s}, "
        f"RED n={n('RED')} mean excess={r20.get('meanExcess')!s}, "
        f"YELLOW n={n('YELLOW')}. Information coefficient at 20d: {ic.get('ic')} (p={ic.get('p')}, n={ic.get('n')})."
        f"{prior_note}"
    ).strip()


def upsert_scorecard_memory(card, db_path=None):
    """Write/replace a single EVENT in the analyst memory for this asof (does not open the gate)."""
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "server" / "ai"))
        from memory_ingest import ingest, memory_hash, default_path, _iso
    except Exception:
        return None
    text = memory_text(card)
    meta = {"dedupeKey": f"analyst-scorecard-live:{card.get('asof')}", "asof": card.get("asof"),
            "gateOpen": bool(((card.get("live") or {}).get("gate") or {}).get("open"))}
    path = Path(db_path or default_path())
    ingest("EVENT", text, ticker=None, ts=card.get("scoredAt"), meta=meta, db_path=path)
    # Refresh the text if the same-day key already existed (ingest is insert-or-ignore).
    digest = memory_hash("EVENT", None, text, meta["dedupeKey"])
    try:
        with sqlite3.connect(path, timeout=30) as db:
            db.execute("UPDATE memory_items SET text=?, meta=?, ts=?, embedded=0 WHERE hash=?",
                       (text, json.dumps(meta), card.get("scoredAt") or _iso(None), digest))
            db.commit()
    except sqlite3.Error:
        pass
    return text


def _filter_origin(db, origin, llm_only=False):
    q = "SELECT * FROM calls WHERE origin=?"
    args = [origin]
    if llm_only:
        q += " AND (UPPER(source) IN ('LLM','AI') OR LOWER(source) LIKE 'ai:%' OR LOWER(source) LIKE 'llm%')"
        # Keep harvest sources that were real model verdicts; exclude FALLBACK from the gate sample.
    rows = [dict(r) for r in db.execute(q, args)]
    if llm_only:
        rows = [r for r in rows if str(r.get("source") or "").upper() not in {"FALLBACK"}
                and not str(r.get("source") or "").upper().startswith("FALLBACK")]
    return rows



def enrich_live_scorecard(doc, prior_path=None, store_path=None):
    """Attach replay prior, leakage, IC, Brier, Newey-West and BH to the live scorecard.

    Mutates `doc` in place and returns it. Never changes `gate` or `gateOpen` — the live risk gate
    stays entirely under analyst_scoring.py (LLM out-of-sample calls only). The prior is shown as
    an informative starting point and cannot open that gate.
    """
    if not isinstance(doc, dict):
        return doc
    rows = []
    path = Path(store_path or DATA_DIR / "analyst_scoring.sqlite3")
    if path.is_file():
        try:
            with sqlite3.connect(path, timeout=15) as src:
                src.row_factory = sqlite3.Row
                if src.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='call_outcomes'").fetchone():
                    for r in src.execute("""SELECT c.flag, c.scorable, o.horizon_days AS horizon, o.excess_spy, o.status
                                            FROM call_outcomes o JOIN analyst_calls c USING (call_id)
                                            WHERE o.status='scored' AND c.scorable=1"""):
                        rows.append(dict(r))
        except sqlite3.Error:
            rows = []
    if not rows:
        for r in doc.get("recent") or []:
            if not r.get("scorable"):
                continue
            for h, o in (r.get("outcomes") or {}).items():
                if (o or {}).get("status") != "scored":
                    continue
                rows.append({"flag": r.get("flag"), "horizon": int(h), "excess_spy": o.get("excess_spy"),
                             "status": "scored"})
    ics, briers, nw = {}, {}, {}
    family = []
    for h in HORIZONS:
        scored = [r for r in rows if int(r.get("horizon") or 0) == h and r.get("excess_spy") is not None]
        scores = [SIGNAL.get(r.get("flag"), 0.0) for r in scored]
        ex = [float(r["excess_spy"]) for r in scored]
        preds = [PROB_POS.get(r.get("flag")) for r in scored]
        ys = [1.0 if e > 0 else 0.0 for e in ex]
        ics[str(h)] = _clean(spearman_ic(scores, ex))
        briers[str(h)] = _clean(brier_score(preds, ys))
        nw[str(h)] = {}
        for flag, alt in (("GREEN", "greater"), ("RED", "less"), ("YELLOW", "two-sided")):
            vals = [float(r["excess_spy"]) for r in scored if r.get("flag") == flag]
            stat = newey_west_t(vals, horizon=h)
            if stat and stat.get("p") is not None:
                stat = dict(stat)
                stat["pOneSided"] = _one_sided(stat["p"], stat["t"], alt)
            nw[str(h)][flag] = _clean(stat)
            if flag in ("GREEN", "RED") and stat and stat.get("pOneSided") is not None:
                family.append((f"{flag}-{h}d-nw", stat["pOneSided"]))
    for c in doc.get("cells") or []:
        if c.get("benchmark") == "spy" and c.get("flag") in ("GREEN", "RED") and c.get("pMean") is not None:
            family.append((f"{c['flag']}-{c['horizon']}d-perm", c["pMean"]))
    doc["informationCoefficient"] = ics
    doc["calibration"] = {"brierByHorizon": briers}
    doc["neweyWest"] = nw
    doc["benjaminiHochberg"] = benjamini_hochberg(family, ALPHA)
    prior_file = Path(prior_path or DATA_DIR / "Research" / "analyst_replay_prior.json")
    if prior_file.is_file():
        try:
            prior_doc = json.loads(prior_file.read_text(encoding="utf-8"))
            prior = prior_doc.get("prior") or prior_doc
            prior = dict(prior)
            prior["gate"] = prior_is_not_a_gate(prior)
            doc["prior"] = _clean(prior)
            doc["leakage"] = prior_doc.get("leakage") or prior.get("leakage")
        except (OSError, ValueError) as error:
            doc.setdefault("warnings", []).append(f"replay prior ignored: {error}")
    ic20 = (ics.get("20") or {}).get("ic")
    brier20 = (briers.get("20") or {}).get("brier")
    extra = (f" 20-day information coefficient (signal vs SPY excess): {ic20}. "
             f"Brier score at 20d: {brier20} (0.25 is a coin-flip).")
    if doc.get("prior"):
        extra += (" A sealed historical-replay prior is on file as an informative starting point only; "
                  "it does not open the gate.")
        leak = (doc.get("leakage") or {}).get("warning")
        if leak:
            extra += " " + leak
    if isinstance(doc.get("promptSummary"), str) and "information coefficient" not in doc["promptSummary"]:
        doc["promptSummary"] = doc["promptSummary"].rstrip() + extra
    doc["statsNote"] = (
        "Permutation tests vs a date-matched random liquid stock, and the Holm-adjusted live gate, come from "
        "analyst_scoring.py. Newey-West t-stats, Benjamini-Hochberg adjustment, Brier calibration and Spearman ICs "
        "are attached here (analyst_score.py); quant_tools.py (as of 2026-11-25) has related bootstrap CI and HAC "
        "t-test ops but not these helpers as importable functions."
    )
    return doc


def rescore(asof=None, prices_path=None, score_db=None, scorecard_path=None, prior_path=None,
            min_n=MIN_N, alpha=ALPHA, seed=7, ingest_memory=True):
    """Harvest live calls, score them, merge any replay prior, write the scorecard. Never raises to the paper trader."""
    warnings = []
    asof = _asof(asof) or datetime.now().strftime("%Y-%m-%d")
    db = connect_score(score_db)
    try:
        added, warnings = harvest_all(db)
        try:
            prices = connect_prices(prices_path)
        except FileNotFoundError as error:
            warnings.append(str(error))
            card = build_scorecard({"flags": {}, "calls": {}}, asof=asof, warnings=warnings,
                                   settings={"minN": min_n, "alpha": alpha, "gateHorizon": GATE_HORIZON,
                                             "horizons": list(HORIZONS)})
            write_scorecard(card, scorecard_path)
            return card
        try:
            calendar = session_calendar(prices)
            sector_map = load_sector_map()
            universe = load_sp500()
            live_rows = [r for r in (dict(x) for x in db.execute("SELECT * FROM calls WHERE origin!='prior'"))]
            # Gate sample: live/harvest, excluding FALLBACK and replay.
            live_block = evaluate_calls(live_rows, calendar, prices, sector_map, universe,
                                        min_n=min_n, alpha=alpha, origin="live", seed=seed, cache_db=db)
            # A narrower LLM-only view is attached so the gate can ignore deterministic fallbacks.
            llm_rows = [r for r in live_rows if str(r.get("source") or "").upper() not in {"FALLBACK"}
                        and not str(r.get("source") or "").upper().startswith("FALLBACK")]
            if llm_rows != live_rows:
                llm_block = evaluate_calls(llm_rows, calendar, prices, sector_map, universe,
                                           min_n=min_n, alpha=alpha, origin="live_llm", seed=seed, cache_db=db)
                live_block["llmOnly"] = {k: llm_block[k] for k in ("flags", "informationCoefficient", "calibration",
                                                                   "multipleTesting", "calls") if k in llm_block}
                # The gate uses LLM-only flags when any exist; otherwise the all-live block (still usually tiny).
                if (llm_block.get("calls") or {}).get("FLAG"):
                    for k in ("flags", "multipleTesting", "informationCoefficient", "calibration"):
                        live_block[k] = llm_block[k]
                    live_block["gateSample"] = "llm"
                else:
                    live_block["gateSample"] = "all_live_no_llm"
            else:
                live_block["gateSample"] = "all_live"
            prior = None
            leakage = None
            prior_file = Path(prior_path or DATA_DIR / "Research" / "analyst_replay_prior.json")
            if prior_file.is_file():
                try:
                    prior_doc = json.loads(prior_file.read_text(encoding="utf-8"))
                    prior = prior_doc.get("prior") or prior_doc.get("live") or prior_doc
                    leakage = (prior_doc.get("leakage") or (prior or {}).get("leakage"))
                except (OSError, ValueError) as error:
                    warnings.append(f"replay prior ignored: {error}")
            settings = {"minN": min_n, "alpha": alpha, "gateHorizon": GATE_HORIZON, "horizons": list(HORIZONS),
                        "nResamples": N_RESAMPLES}
            card = build_scorecard(live_block, prior=prior, asof=asof, warnings=warnings, settings=settings,
                                   leakage=leakage)
            path = write_scorecard(card, scorecard_path)
            db.execute("INSERT INTO scorecard_runs (scored_at, asof, origin, payload) VALUES (?, ?, 'live', ?)",
                       (card["scoredAt"], card["asof"], json.dumps(card)))
            db.commit()
            if ingest_memory:
                try:
                    upsert_scorecard_memory(card)
                except Exception as error:
                    warnings.append(f"memory ingest skipped: {error}")
                    card["warnings"] = warnings
                    write_scorecard(card, path)
            return card
        finally:
            prices.close()
    finally:
        db.close()


def rescore_after_paper_run(asof=None):
    """Hook for the paper trader. Failures are returned as a dict and never abort the run."""
    try:
        card = rescore(asof=asof)
        gate = ((card or {}).get("live") or {}).get("gate") or {}
        return {"ok": True, "asof": card.get("asof"), "gateOpen": gate.get("open"), "reason": gate.get("reason"),
                "path": str(DEFAULT_CARD)}
    except Exception as error:
        return {"ok": False, "error": str(error)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["rescore", "status", "gate", "harvest"])
    parser.add_argument("--asof")
    parser.add_argument("--min-n", type=int, default=MIN_N)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        card = read_scorecard()
        if not card:
            print(json.dumps({"error": "No scorecard yet. Run: python analyst_score.py rescore"}))
            return
        print(json.dumps(_clean(card), indent=2, default=str))
        return
    if args.command == "gate":
        print(json.dumps(load_live_gate(), indent=2))
        return
    if args.command == "harvest":
        db = connect_score()
        added, warnings = harvest_all(db)
        db.close()
        print(json.dumps({"added": added, "warnings": warnings}))
        return
    card = rescore(asof=args.asof, min_n=args.min_n, alpha=args.alpha, ingest_memory=not args.no_memory)
    live = card.get("live") or {}
    print(json.dumps(_clean({
        "asof": card.get("asof"),
        "gate": (live.get("gate") or {}),
        "calls": live.get("calls"),
        "green20": (live.get("flags") or {}).get("GREEN", {}).get("20"),
        "red20": (live.get("flags") or {}).get("RED", {}).get("20"),
        "ic20": (live.get("informationCoefficient") or {}).get("20"),
        "warnings": card.get("warnings"),
        "path": str(DEFAULT_CARD),
    }), indent=2, default=str))


if __name__ == "__main__":
    main()
