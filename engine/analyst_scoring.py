"""Analyst scorecard: log every AI flag (GREEN/YELLOW/RED) and stop, score them against what actually happened,
and decide whether a flag type has earned the right to change position sizes.

Store (local, gitignored): <data dir>/analyst_scoring.sqlite3
  analyst_calls     one row per flag the analyst issued (ticker, issued_at UTC, flag, ref price, stop, band, source)
  call_outcomes     one row per call and horizon (5/20/60 trading days): raw return, SPY and sector-ETF returns,
                    excess returns, cross-sectional base rates, whether the stop was hit.
Scorecard (read by the risk gate, the dashboard and the analyst prompt): <data dir>/Research/analyst_scorecard.json

Method
  * Anchor: the first NYSE session whose official close is at or after the moment the flag was issued (a flag issued
    intraday is anchored to that day's close, one issued after the close to the next session's close) -- you could
    actually have traded at that price, so there is no look-ahead. Returns include cash dividends (Panel.total_return).
  * Forward return over h sessions; excess vs SPY and vs the ticker's sector SPDR ETF. Sector = manual override >
    SIC code from the analyst's cached company details > highest return correlation with the 11 sector ETFs.
  * Stops (long stops below the anchor price): hit if any later low <= stop within h sessions; exit at
    min(stop, that day's open) to model gaps; "after-stop" return = close at h / exit - 1 (positive = the stop cost money).
  * Chance = a date-matched random pick: for every call, the null draws a random liquid stock (price > $5, 20-day
    median dollar volume > $20M, not an ETF) on the same anchor date and horizon. The p-value of the mean excess return
    and of the hit rate comes from 20,000 such permutations. Base rate = share of that universe that would have "hit".
    Wilson 95% interval for the hit rate, bootstrap 95% interval for the mean excess.
  * Effective sample: repeated same-flag calls on a ticker inside one horizon window count once (overlap removed).
  * Multiple testing: Holm adjustment across every flag x horizon cell that reaches the minimum sample.
  * Hits: GREEN = beat the benchmark; YELLOW / RED = lagged it.

Gate (engine/advisory.py gated_adjust_entry): a flag type may shrink or skip positions ONLY if, for at least one of the
strategy's horizons, its LLM verdicts have n_eff >= 30, the mean excess vs SPY points the right way, and the
Holm-adjusted permutation p < 0.05, in a scorecard no older than 7 days. Anything else -- missing, stale or unreadable
scorecard, too few samples, FALLBACK (non-LLM) verdicts -- means the verdict is logged and shown but has no effect.

    python analyst_scoring.py run            # ingest new calls, score, write scorecard + memory notes (exit 0)
    python analyst_scoring.py status         # print the current scorecard summary
    python analyst_scoring.py gate RED --horizons 20,60
"""

import argparse
import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, DATABASE

STORE = DATA_DIR / "analyst_scoring.sqlite3"
SCORECARD = DATA_DIR / "Research" / "analyst_scorecard.json"
AI_RESEARCH = DATA_DIR / "ai_research.sqlite3"
ADVISORY_JSON = DATA_DIR / "Research" / "advisory_verdicts.json"
PAPER_DB = DATA_DIR / "paper_trading.sqlite3"
SECTOR_OVERRIDES = DATA_DIR / "Reference" / "sector_overrides.json"   # optional {"TICKER": "XLK"}

HORIZONS = (5, 20, 60)
FLAGS = ("GREEN", "YELLOW", "RED")
DIRECTION = {"GREEN": 1, "YELLOW": -1, "RED": -1}
MIN_N = 30
ALPHA = 0.05
PERMUTATIONS = 20_000
MAX_SCORECARD_AGE_DAYS = 7
MARKET = "SPY"
SECTOR_ETFS = ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")
# Large names whose SIC code disagrees with their GICS sector (ETF membership is GICS).
KNOWN_SECTORS = {"GOOGL": "XLC", "GOOG": "XLC", "META": "XLC", "NFLX": "XLC", "DIS": "XLC", "CMCSA": "XLC",
                 "T": "XLC", "VZ": "XLC", "TMUS": "XLC", "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY",
                 "V": "XLF", "MA": "XLF", "PYPL": "XLF", "BRK.B": "XLF", "AAPL": "XLK", "MSFT": "XLK",
                 "NVDA": "XLK", "AVGO": "XLK", "AMD": "XLK", "ORCL": "XLK", "CRM": "XLK", "ADBE": "XLK",
                 "LLY": "XLV", "UNH": "XLV", "JNJ": "XLV", "XOM": "XLE", "CVX": "XLE", "JPM": "XLF",
                 "WMT": "XLP", "COST": "XLP", "PG": "XLP", "KO": "XLP", "PEP": "XLP"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyst_calls (
  call_id TEXT PRIMARY KEY, ticker TEXT NOT NULL, issued_at TEXT, asof TEXT, origin TEXT NOT NULL,
  source TEXT, model TEXT, scorable INTEGER NOT NULL, flag TEXT NOT NULL, conviction REAL, horizon TEXT,
  action TEXT, ref_price REAL, stop REAL, entry_low REAL, entry_high REAL, take_profit REAL, payload TEXT,
  logged_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS analyst_calls_ticker ON analyst_calls (ticker, issued_at);
CREATE TABLE IF NOT EXISTS call_outcomes (
  call_id TEXT NOT NULL, horizon_days INTEGER NOT NULL, status TEXT NOT NULL, anchor_date TEXT, anchor_close REAL,
  end_date TEXT, ret REAL, spy_ret REAL, sector_etf TEXT, sector_method TEXT, sector_ret REAL, excess_spy REAL,
  excess_sector REAL, base_rate_spy REAL, base_rate_sector REAL, universe_n INTEGER, stop_hit INTEGER,
  stop_days INTEGER, stop_exit REAL, stop_ret_after REAL, updated_at TEXT NOT NULL,
  PRIMARY KEY (call_id, horizon_days));
CREATE TABLE IF NOT EXISTS scoring_runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, status TEXT NOT NULL, ingested INTEGER,
  scored INTEGER, pending INTEGER, gate TEXT, message TEXT);
"""


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _num(value):
    try:
        out = None if value is None or value == "" else float(value)
        return out if out is None or math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _utc(value):
    """ISO string -> aware UTC datetime (naive strings are taken as UTC), or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def is_llm(source):
    s = str(source or "")
    return s == "LLM" or s.startswith("ai:")


# ---------------------------------------------------------------- store + logging
def connect(path=None):
    path = Path(path or STORE)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    return db


def log_call(db, *, ticker, flag, issued_at=None, asof=None, origin="verdict", source="LLM", model=None,
             conviction=None, horizon=None, action=None, ref_price=None, stop=None, entry_low=None, entry_high=None,
             take_profit=None, payload=None, call_id=None):
    """Durably record one analyst flag. Idempotent (same call_id -> ignored). Returns True if new."""
    flag = str(flag or "").upper()
    ticker = str(ticker or "").upper().strip()
    if flag not in FLAGS or not ticker:
        return False
    issued = _utc(issued_at)
    issued_s = issued.isoformat(timespec="milliseconds").replace("+00:00", "Z") if issued else None
    if call_id is None:
        call_id = f"{origin}|{ticker}|{issued_s or asof}|{flag}"
    cur = db.execute(
        """INSERT OR IGNORE INTO analyst_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (call_id, ticker, issued_s, str(asof)[:10] if asof else (issued_s[:10] if issued_s else None), origin,
         source, model, int(is_llm(source)), flag, _num(conviction), horizon, action, _num(ref_price), _num(stop),
         _num(entry_low), _num(entry_high), _num(take_profit), json.dumps(payload, default=str) if payload else None,
         now_iso()))
    return cur.rowcount > 0


def _ro(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        db = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        db.row_factory = sqlite3.Row
        return db
    except sqlite3.Error:
        return None


def _has(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def ingest_all(db, ai_research=None, advisory_json=None, paper_db=None):
    """Backfill / catch up from every place a flag can live. Returns (new_calls, warnings)."""
    added, warnings = 0, []
    src = _ro(ai_research or AI_RESEARCH)
    if src is not None:
        try:
            if _has(src, "ai_verdicts"):
                for r in src.execute("SELECT * FROM ai_verdicts ORDER BY created_at"):
                    added += log_call(db, call_id=f"verdict|{r['ticker']}|{r['created_at']}", ticker=r["ticker"],
                                      flag=r["flag"], issued_at=r["created_at"], asof=r["asof"], origin="verdict",
                                      source=r["source"], model=r["model"], conviction=r["conviction_score"],
                                      horizon=r["horizon"], action=r["action"], ref_price=r["ref_price"],
                                      stop=r["stop"], entry_low=r["entry_low"], entry_high=r["entry_high"],
                                      take_profit=r["take_profit"])
            if _has(src, "ai_log"):
                for r in src.execute("SELECT * FROM ai_log WHERE kind='ask' AND status='ok' ORDER BY id"):
                    try:
                        inp, out = json.loads(r["input"] or "{}"), json.loads(r["output"] or "{}")
                    except ValueError:
                        continue
                    tickers = inp.get("tickers") or []
                    if len(tickers) != 1 or str(out.get("stance", "")).upper() not in FLAGS:
                        continue   # multi-ticker answers are not attributable to one name
                    stop = None
                    for level in out.get("levels") or []:
                        if isinstance(level, dict) and "stop" in json.dumps(level).lower():
                            stop = _num(level.get("price") or level.get("value") or level.get("level"))
                            break
                    added += log_call(db, call_id=f"ask|{tickers[0]}|{r['at']}", ticker=tickers[0],
                                      flag=out["stance"], issued_at=r["at"], origin="ask", source="LLM",
                                      model=r["model"], horizon=None, stop=stop,
                                      payload={"question": inp.get("question"), "conviction": out.get("conviction")})
        except sqlite3.Error as error:
            warnings.append(f"ai_research ignored: {error}")
        finally:
            src.close()
    path = Path(advisory_json or ADVISORY_JSON)
    if path.is_file():
        try:
            doc = json.loads(path.read_text(encoding="utf-8-sig"))
            items = doc.get("verdicts", {}) if isinstance(doc, dict) else doc
            pairs = items.items() if isinstance(items, dict) else ((i.get("ticker"), i) for i in items)
            for ticker, v in pairs:
                if not isinstance(v, dict) or not ticker:
                    continue
                created = v.get("created_at")
                asof = v.get("asof") or (doc.get("asof") if isinstance(doc, dict) else None)
                cid = f"verdict|{ticker}|{created}" if created else f"advisory|{ticker}|{asof}|{v.get('source')}"
                added += log_call(db, call_id=cid, ticker=ticker, flag=v.get("flag"), issued_at=created, asof=asof,
                                  origin="verdict", source=v.get("source") or "ai:unknown",
                                  conviction=v.get("conviction"), horizon=v.get("horizon"), action=v.get("action"),
                                  ref_price=v.get("ref_price"), stop=v.get("stop"), entry_low=v.get("entry_low"),
                                  entry_high=v.get("entry_high"), take_profit=v.get("take_profit"))
        except (OSError, ValueError, AttributeError, TypeError) as error:
            warnings.append(f"advisory JSON ignored: {error}")
    paper = _ro(paper_db or PAPER_DB)
    if paper is not None:
        try:
            if _has(paper, "advisory_verdicts"):
                for r in paper.execute("SELECT * FROM advisory_verdicts"):
                    r = dict(r)
                    added += log_call(db, call_id=f"advisory|{r['ticker']}|{r['asof']}|{r.get('source')}",
                                      ticker=r["ticker"], flag=r["flag"], issued_at=r.get("created_at"),
                                      asof=r["asof"], origin="verdict", source=r.get("source") or "ai:table",
                                      conviction=r.get("conviction"), stop=r.get("stop"),
                                      entry_low=r.get("entry_low"), entry_high=r.get("entry_high"))
        except sqlite3.Error as error:
            warnings.append(f"paper advisory table ignored: {error}")
        finally:
            paper.close()
    db.commit()
    return added, warnings


# ---------------------------------------------------------------- calendar + sectors
def session_closes(start, end):
    """Series: NYSE session date (naive Timestamp) -> official close (aware UTC). Handles early closes."""
    import exchange_calendars

    cal = exchange_calendars.get_calendar("XNYS", start=str(pd.Timestamp(start).date()),
                                          end=str(pd.Timestamp(end).date()))
    closes = cal.schedule["close"]
    index = pd.DatetimeIndex(closes.index)
    index = index.tz_localize(None) if index.tz is not None else index
    values = pd.DatetimeIndex(closes.values)
    values = values.tz_localize("UTC") if values.tz is None else values.tz_convert("UTC")
    return pd.Series(values, index=index)


def anchor_session(issued_at, asof, closes):
    """First session whose close is >= issued_at (or the first session on/after asof when there is no timestamp)."""
    issued = _utc(issued_at)
    if issued is not None:
        hit = closes[closes >= pd.Timestamp(issued)]
    elif asof:
        hit = closes[closes.index >= pd.Timestamp(str(asof)[:10])]
    else:
        return None
    return hit.index[0] if len(hit) else None


def sic_to_etf(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return None
    rules = [((1311, 1389), "XLE"), ((2900, 2999), "XLE"), ((1000, 1499), "XLB"), ((1500, 1799), "XLI"),
             ((2830, 2836), "XLV"), ((2840, 2844), "XLP"), ((2000, 2199), "XLP"), ((2200, 2399), "XLY"),
             ((2500, 2599), "XLY"), ((2700, 2799), "XLC"), ((2400, 2899), "XLB"), ((3000, 3399), "XLB"),
             ((3570, 3579), "XLK"), ((3630, 3639), "XLY"), ((3600, 3699), "XLK"), ((3710, 3716), "XLY"),
             ((3812, 3812), "XLI"), ((3826, 3826), "XLV"), ((3841, 3851), "XLV"), ((3800, 3869), "XLK"),
             ((3870, 3999), "XLY"), ((3400, 3799), "XLI"), ((4800, 4899), "XLC"), ((4900, 4999), "XLU"),
             ((4000, 4799), "XLI"), ((5122, 5122), "XLV"), ((5140, 5149), "XLP"), ((5000, 5199), "XLI"),
             ((5400, 5499), "XLP"), ((5912, 5912), "XLP"), ((5200, 5999), "XLY"), ((6798, 6798), "XLRE"),
             ((6500, 6599), "XLRE"), ((6000, 6799), "XLF"), ((7310, 7319), "XLC"), ((7370, 7379), "XLK"),
             ((7812, 7841), "XLC"), ((7000, 7099), "XLY"), ((7200, 7299), "XLY"), ((7900, 7999), "XLY"),
             ((8000, 8099), "XLV"), ((7300, 8999), "XLI")]
    for (low, high), etf in rules:
        if low <= s <= high:
            return etf
    return None


def cached_sics(ai_research=None):
    """{ticker: sic} from the analyst's cached company details (no network)."""
    out = {}
    src = _ro(ai_research or AI_RESEARCH)
    if src is None:
        return out
    try:
        if _has(src, "source_cache"):
            for r in src.execute("SELECT ticker, payload FROM source_cache WHERE payload LIKE '%\"sic\"%'"):
                try:
                    value = json.loads(r["payload"])
                except ValueError:
                    continue
                sic = value.get("sic") if isinstance(value, dict) else None
                if r["ticker"] and sic:
                    out[str(r["ticker"]).upper()] = sic
    except sqlite3.Error:
        pass
    finally:
        src.close()
    return out


def correlation_sectors(returns, window=250, min_obs=60):
    """{ticker: sector ETF} by highest daily-return correlation with the 11 sector SPDRs over the last `window` rows.
    Sector membership is treated as a static attribute (classification only, not a signal)."""
    etfs = [e for e in SECTOR_ETFS if e in returns.columns]
    if not etfs:
        return {}
    r = returns.iloc[-window:]
    stocks = [c for c in r.columns if c not in etfs and c != MARKET]
    x = r[stocks].loc[:, r[stocks].notna().sum() >= min_obs] if stocks else pd.DataFrame()
    if x.empty:
        return {}
    corr = pd.DataFrame({e: x.corrwith(r[e]) for e in etfs})
    best = corr.dropna(how="all").idxmax(axis=1)
    return {t: e for t, e in best.items() if isinstance(e, str)}


def sector_for(ticker, sics, corr_map, overrides):
    if ticker in SECTOR_ETFS or ticker == MARKET:
        return None, "etf"
    if ticker in overrides:
        return overrides[ticker], "override"
    if ticker in KNOWN_SECTORS:
        return KNOWN_SECTORS[ticker], "known"
    etf = sic_to_etf(sics.get(ticker))
    if etf:
        return etf, "sic"
    if ticker in corr_map:
        return corr_map[ticker], "correlation"
    return None, "none"


# ---------------------------------------------------------------- outcomes
def eligible_universe(panel):
    """Liquid, non-ETF stocks per date (same screen as the strategy universes)."""
    from universe import liquid_mask

    stocks = [t for t, kind in panel.classes.items() if kind == "stock"] if len(panel.classes) else \
        [c for c in panel.close.columns if c not in SECTOR_ETFS and c != MARKET]
    stocks = [t for t in stocks if t in panel.close.columns and t not in SECTOR_ETFS and t != MARKET]
    return liquid_mask(panel)[stocks]


class OutcomeEngine:
    def __init__(self, panel, sector_map):
        self.panel = panel
        self.dates = panel.close.index
        tr = panel.total_return()
        self.growth = (1 + tr.fillna(0.0)).cumprod()
        self.growth = self.growth.where(panel.close.notna())
        self.elig = eligible_universe(panel)
        self.sector_map = sector_map
        self._null = {}

    def window_return(self, ticker, a, b):
        if ticker not in self.growth.columns:
            return None
        g0, g1 = self.growth[ticker].iat[a], self.growth[ticker].iat[b]
        return float(g1 / g0 - 1) if np.isfinite(g0) and np.isfinite(g1) and g0 > 0 else None

    def null(self, a, h):
        """(excess vs SPY, excess vs own sector ETF) arrays for the eligible universe on anchor row a."""
        key = (a, h)
        if key not in self._null:
            b = a + h
            row = self.elig.iloc[a]
            names = row.index[row.to_numpy(dtype=bool)]
            g0, g1 = self.growth[names].iloc[a], self.growth[names].iloc[b]
            r = (g1 / g0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
            spy = self.window_return(MARKET, a, b)
            ex_spy = (r - spy).to_numpy() if spy is not None else np.array([])
            etf_ret = {e: self.window_return(e, a, b) for e in SECTOR_ETFS}
            sec = np.array([etf_ret.get(self.sector_map.get(t)) if self.sector_map.get(t) else np.nan for t in r.index],
                           dtype=float)
            ex_sec = (r.to_numpy() - sec)
            self._null[key] = (ex_spy, ex_sec[np.isfinite(ex_sec)])
        return self._null[key]

    def score(self, call, h, closes):
        """Outcome dict for one call (sqlite Row / dict) and horizon."""
        out = {"status": "pending"}
        anchor = anchor_session(call["issued_at"], call["asof"], closes)
        if anchor is None:
            return out
        out["anchor_date"] = str(anchor.date())
        if anchor > self.dates[-1]:
            return out                                     # price data not loaded yet
        if anchor not in self.dates:
            return {**out, "status": "no_data"}            # session missing from the database
        a = self.dates.get_loc(anchor)
        ticker = call["ticker"]
        close = self.panel.close[ticker].iat[a] if ticker in self.panel.close.columns else np.nan
        if not np.isfinite(close):
            return {**out, "status": "no_data"}
        out["anchor_close"] = float(close)
        b = a + h
        if b >= len(self.dates):
            return out
        out["end_date"] = str(self.dates[b].date())
        ret = self.window_return(ticker, a, b)
        if ret is None:
            return {**out, "status": "no_data"}
        spy = self.window_return(MARKET, a, b)
        etf, method = self.sector_map.get(ticker), self.sector_map.get(("method", ticker), "none")
        sec = self.window_return(etf, a, b) if etf else None
        ex_spy, ex_sec = self.null(a, h)
        d = DIRECTION[call["flag"]]
        out.update(status="scored", ret=ret, spy_ret=spy, sector_etf=etf, sector_method=method, sector_ret=sec,
                   excess_spy=None if spy is None else ret - spy, excess_sector=None if sec is None else ret - sec,
                   base_rate_spy=float(np.mean(d * ex_spy > 0)) if len(ex_spy) else None,
                   base_rate_sector=float(np.mean(d * ex_sec > 0)) if len(ex_sec) else None,
                   universe_n=int(len(ex_spy)))
        stop = _num(call["stop"])
        if stop is not None and 0 < stop < close:
            lows = self.panel.low[ticker].iloc[a + 1:b + 1].to_numpy(dtype=float)
            touched = np.where(lows <= stop)[0]
            if len(touched):
                k = int(touched[0])
                opening = self.panel.open[ticker].iat[a + 1 + k]
                exit_price = float(min(stop, opening)) if np.isfinite(opening) else stop
                end_close = self.panel.close[ticker].iat[b]
                out.update(stop_hit=1, stop_days=k + 1, stop_exit=exit_price,
                           stop_ret_after=float(end_close / exit_price - 1) if np.isfinite(end_close) else None)
            else:
                out.update(stop_hit=0)
        return out


def save_outcome(db, call_id, h, o):
    cols = ("status", "anchor_date", "anchor_close", "end_date", "ret", "spy_ret", "sector_etf", "sector_method",
            "sector_ret", "excess_spy", "excess_sector", "base_rate_spy", "base_rate_sector", "universe_n",
            "stop_hit", "stop_days", "stop_exit", "stop_ret_after")
    db.execute(f"INSERT OR REPLACE INTO call_outcomes VALUES (?, ?, {', '.join('?' * len(cols))}, ?)",
               (call_id, h, *[o.get(c) for c in cols], now_iso()))


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def holm(pvalues):
    """Holm step-down adjusted p-values (same order as input)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def effective_sample(rows, h):
    """Drop same-ticker calls whose window overlaps an earlier kept call (rows need ticker, anchor_row)."""
    kept, last = [], {}
    for r in sorted(rows, key=lambda r: (r["ticker"], r["anchor_row"])):
        if r["ticker"] in last and r["anchor_row"] < last[r["ticker"]] + h:
            continue
        last[r["ticker"]] = r["anchor_row"]
        kept.append(r)
    return kept


def test_cell(rows, flag, benchmark, permutations=PERMUTATIONS, seed=20261125):
    """rows: dicts with 'excess' (call excess return) and 'null' (date-matched universe excess array)."""
    rows = [r for r in rows if r["excess"] is not None and r["null"] is not None and len(r["null"])]
    n = len(rows)
    out = {"flag": flag, "benchmark": benchmark, "n": n}
    if n == 0:
        return out
    d = DIRECTION[flag]
    x = np.array([r["excess"] for r in rows], dtype=float)
    hits = d * x > 0
    base = float(np.mean([np.mean(d * r["null"] > 0) for r in rows]))
    rng = np.random.default_rng(seed)
    null_sum = np.zeros(permutations)
    null_hits = np.zeros(permutations)
    for r in rows:
        draw = r["null"][rng.integers(0, len(r["null"]), permutations)]
        null_sum += draw
        null_hits += d * draw > 0
    obs_mean, obs_hit = float(x.mean()), float(hits.mean())
    p_mean = (1 + int(np.sum(d * null_sum / n >= d * obs_mean))) / (permutations + 1)
    p_hit = (1 + int(np.sum(null_hits / n >= obs_hit))) / (permutations + 1)
    boot = rng.choice(x, size=(5000, n)).mean(axis=1)
    lo, hi = wilson(int(hits.sum()), n)
    out.update(hits=int(hits.sum()), hitRate=obs_hit, wilson=[lo, hi], baseRate=base, meanExcess=obs_mean,
               medianExcess=float(np.median(x)), ciMean=[float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
               nullMeanExcess=float(np.mean([r["null"].mean() for r in rows])), pMean=p_mean, pHit=p_hit)
    return out


# ---------------------------------------------------------------- gate (read side, used by engine/advisory.py)
_GATE_CACHE = {}


def load_scorecard(path=None):
    path = Path(path or os.environ.get("ANALYST_SCORECARD_PATH") or SCORECARD)
    try:
        stamp = path.stat().st_mtime_ns
    except OSError:
        return None
    hit = _GATE_CACHE.get(str(path))
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
    _GATE_CACHE[str(path)] = (stamp, doc)
    return doc


def gate_open(flag, horizons=(20, 60), path=None, now=None):
    """(True, reason) only if the scorecard proves this flag type at one of `horizons`. Fails closed."""
    flag = str(flag or "").upper()
    doc = load_scorecard(path)
    if not isinstance(doc, dict) or doc.get("schema") != "analyst_scorecard/v1":
        return False, "no analyst scorecard yet (verdicts are log-only until proven)"
    generated = _utc(doc.get("generatedAt"))
    now = now or datetime.now(timezone.utc)
    if generated is None or (now - generated) > timedelta(days=MAX_SCORECARD_AGE_DAYS):
        return False, f"analyst scorecard older than {MAX_SCORECARD_AGE_DAYS} days"
    gate = (doc.get("gate") or {}).get(flag) or {}
    for h in horizons:
        if gate.get(str(int(h))) is True:
            return True, f"{flag} verdicts beat chance at {int(h)}d (scorecard {doc.get('generatedAt')})"
    cells = {(c.get("flag"), c.get("horizon")): c for c in doc.get("cells", []) if c.get("benchmark") == "spy"}
    detail = "; ".join(f"{int(h)}d n={cells.get((flag, int(h)), {}).get('n', 0)}" for h in horizons)
    return False, f"{flag} not proven ({detail}; needs n>={doc.get('params', {}).get('minN', MIN_N)} and Holm p<{ALPHA})"


# ---------------------------------------------------------------- run
def _load_panel(start):
    from market_data import load_panel

    return load_panel(start=start)


def last_session_in_db(database=None):
    db = _ro(database or DATABASE)
    if db is None:
        return None
    try:
        row = db.execute("SELECT MAX(date) FROM market_sessions WHERE bar_count > 0").fetchone()
        return row[0] if row else None
    finally:
        db.close()


def build_scorecard(db, panel=None, closes=None, sector_overrides=None, sics=None, permutations=PERMUTATIONS,
                    min_n=MIN_N, alpha=ALPHA, last_data=None, panel_loader=None):
    calls = [dict(r) for r in db.execute("SELECT * FROM analyst_calls ORDER BY COALESCE(issued_at, asof)")]
    warnings = []
    if closes is None:
        start = min([str(c["asof"] or c["issued_at"] or "2024-01-01")[:10] for c in calls] or ["2024-01-01"])
        closes = session_closes(pd.Timestamp(start) - pd.Timedelta(days=10), pd.Timestamp.today() + pd.Timedelta(days=200))
    last_data = last_data or (str(panel.close.index[-1].date()) if panel is not None else last_session_in_db())
    # Only load the (large) price panel when at least one call has a complete 5-session window.
    ready = []
    for c in calls:
        anchor = anchor_session(c["issued_at"], c["asof"], closes)
        if anchor is not None and last_data and anchor <= pd.Timestamp(last_data):
            ahead = closes.index[(closes.index > anchor) & (closes.index <= pd.Timestamp(last_data))]
            if len(ahead) >= min(HORIZONS):
                ready.append(c)
    engine = None
    if ready and panel is None:
        first = min(anchor_session(c["issued_at"], c["asof"], closes) for c in ready)
        panel = (panel_loader or _load_panel)(str((first - pd.Timedelta(days=420)).date()))
    if panel is not None and ready:
        overrides = sector_overrides if sector_overrides is not None else _read_overrides()
        sics = sics if sics is not None else cached_sics()
        corr = correlation_sectors(panel.total_return())
        sector_map = dict(corr)
        names = set(panel.close.columns) | {c["ticker"] for c in calls}
        for t in names:
            etf, method = sector_for(t, sics, corr, overrides)
            sector_map[t] = etf
            sector_map[("method", t)] = method
        engine = OutcomeEngine(panel, sector_map)
    scored = pending = 0
    ready_ids = {c["call_id"] for c in ready}
    for c in calls:
        anchor = anchor_session(c["issued_at"], c["asof"], closes)
        for h in HORIZONS:
            if engine is not None and c["call_id"] in ready_ids:
                o = engine.score(c, h, closes)
            else:
                o = {"status": "pending", "anchor_date": str(anchor.date()) if anchor is not None else None}
            save_outcome(db, c["call_id"], h, o)
            scored += o["status"] == "scored"
            pending += o["status"] == "pending"
    db.commit()

    # -------- statistics
    outcomes = {(r["call_id"], r["horizon_days"]): dict(r) for r in db.execute("SELECT * FROM call_outcomes")}
    def rows_for(origin, flag, h, benchmark):
        rows = []
        for c in calls:
            if c["origin"] != origin or not c["scorable"] or c["flag"] != flag:
                continue
            o = outcomes.get((c["call_id"], h))
            if not o or o["status"] != "scored" or engine is None:
                continue
            a = engine.dates.get_loc(pd.Timestamp(o["anchor_date"]))
            ex_spy, ex_sec = engine.null(a, h)
            rows.append({"ticker": c["ticker"], "anchor_row": a,
                         "excess": o["excess_spy"] if benchmark == "spy" else o["excess_sector"],
                         "null": ex_spy if benchmark == "spy" else ex_sec})
        return rows

    cells, info = [], []
    for origin, target in (("verdict", cells), ("ask", info)):
        for flag in FLAGS:
            for h in HORIZONS:
                for benchmark in ("spy", "sector"):
                    raw = rows_for(origin, flag, h, benchmark)
                    eff = effective_sample(raw, h)
                    cell = test_cell(eff, flag, benchmark, permutations)
                    cell.update(horizon=h, origin=origin, nRaw=len(raw))
                    target.append(cell)
    tested = [c for c in cells if c["benchmark"] == "spy" and c["n"] >= min_n]
    for c, p in zip(tested, holm([c["pMean"] for c in tested])):
        c["pHolm"] = p
    gate = {f: {str(h): False for h in HORIZONS} for f in FLAGS}
    for c in cells:
        if c["benchmark"] != "spy":
            continue
        d = DIRECTION[c["flag"]]
        if c["n"] < min_n:
            c["passes"], c["reason"] = False, f"n={c['n']} < {min_n}"
        elif d * c["meanExcess"] <= 0:
            c["passes"], c["reason"] = False, "mean excess points the wrong way"
        elif c.get("pHolm", 1.0) >= alpha:
            c["passes"], c["reason"] = False, f"Holm p={c.get('pHolm', 1.0):.3f} >= {alpha}"
        else:
            c["passes"], c["reason"] = True, f"Holm p={c['pHolm']:.4f} < {alpha}, n={c['n']}"
        gate[c["flag"]][str(c["horizon"])] = bool(c["passes"])

    stops = []
    for h in HORIZONS:
        st = [outcomes[(c["call_id"], h)] for c in calls if c["scorable"] and (c["call_id"], h) in outcomes
              and outcomes[(c["call_id"], h)]["status"] == "scored" and outcomes[(c["call_id"], h)]["stop_hit"] is not None]
        k = sum(int(o["stop_hit"]) for o in st)
        after = [o["stop_ret_after"] for o in st if o["stop_hit"] and o["stop_ret_after"] is not None]
        lo, hi = wilson(k, len(st))
        stops.append({"horizon": h, "n": len(st), "hits": k, "hitRate": (k / len(st)) if st else None,
                      "wilson": [lo, hi], "meanReturnAfterStop": float(np.mean(after)) if after else None,
                      "stopsThatCostMoney": int(sum(1 for x in after if x > 0)),
                      "medianDaysToStop": float(np.median([o["stop_days"] for o in st if o["stop_hit"]])) if k else None})

    by_source, by_origin = {}, {}
    for c in calls:
        by_source[c["source"] or "?"] = by_source.get(c["source"] or "?", 0) + 1
        by_origin[c["origin"]] = by_origin.get(c["origin"], 0) + 1
    status_counts = {str(h): {} for h in HORIZONS}
    for (cid, h), o in outcomes.items():
        status_counts[str(h)][o["status"]] = status_counts[str(h)].get(o["status"], 0) + 1
    recent = []
    for c in calls[-200:][::-1]:
        recent.append({k: c[k] for k in ("call_id", "ticker", "issued_at", "asof", "origin", "source", "model",
                                         "scorable", "flag", "conviction", "horizon", "ref_price", "stop")}
                      | {"outcomes": {str(h): {k: v for k, v in (outcomes.get((c["call_id"], h)) or {}).items()
                                               if k not in ("call_id", "horizon_days", "updated_at")} for h in HORIZONS}})
    open_list = [f"{f}@{h}d" for f in FLAGS for h in HORIZONS if gate[f][str(h)]]
    doc = {"schema": "analyst_scorecard/v1", "generatedAt": now_iso(), "dataThrough": last_data,
           "params": {"horizons": list(HORIZONS), "minN": min_n, "alpha": alpha, "permutations": permutations,
                      "benchmarkForGate": "spy", "multipleTesting": "holm", "maxAgeDays": MAX_SCORECARD_AGE_DAYS,
                      "population": "LLM verdicts (FALLBACK/non-LLM calls are logged but never scored for the gate)",
                      "null": "date-matched random liquid stock (price>$5, 20d median $vol>$20M)"},
           "counts": {"calls": len(calls), "scorable": sum(c["scorable"] for c in calls), "bySource": by_source,
                      "byOrigin": by_origin, "outcomes": status_counts},
           "gate": gate, "gateOpen": open_list, "cells": cells, "askCells": info, "stops": stops,
           "recent": recent, "warnings": warnings}
    doc["promptSummary"] = prompt_summary(doc)
    return doc


def prompt_summary(doc):
    gate_open_list = doc.get("gateOpen") or []
    parts = []
    for c in doc["cells"]:
        if c["benchmark"] == "spy" and c["n"]:
            parts.append(f"{c['flag']} {c['horizon']}d: n={c['n']}, hit {c['hitRate'] * 100:.0f}% vs base "
                         f"{c['baseRate'] * 100:.0f}%, mean excess vs SPY {c['meanExcess'] * 100:+.2f}%, p={c['pMean']:.3f}")
    counts = doc["counts"]
    try:
        as_of = str(pd.Timestamp(doc["generatedAt"]).tz_convert("America/New_York").date()) + " ET"
    except Exception:
        as_of = doc["generatedAt"][:10] + " UTC"
    text = (f"[ANALYST_SCORECARD] Your own track record (as of {as_of}, prices through {doc.get('dataThrough')}): "
            f"{counts['calls']} flag(s) logged, {counts['scorable']} from the LLM (only LLM verdicts are scored). ")
    text += ("; ".join(parts) + ". ") if parts else "No flag has a completed 5/20/60-day window yet. "
    stops = [s for s in doc.get("stops", []) if s["n"]]
    if stops:
        text += "Stops: " + "; ".join(f"{s['horizon']}d hit {s['hits']}/{s['n']}" for s in stops) + ". "
    text += (f"Flags currently allowed to change position sizes: {', '.join(gate_open_list)}. " if gate_open_list else
             "No flag type has beaten chance yet (needs n>=30 and Holm-adjusted p<0.05 vs a date-matched random "
             "stock), so your verdicts are logged and shown but do NOT shrink or skip positions. ")
    text += "Be calibrated: say when evidence is thin, and do not overstate conviction."
    return text


def _read_overrides():
    try:
        data = json.loads(SECTOR_OVERRIDES.read_text(encoding="utf-8"))
        return {str(k).upper(): str(v).upper() for k, v in data.items() if str(v).upper() in SECTOR_ETFS}
    except (OSError, ValueError, AttributeError):
        return {}


def write_scorecard(doc, path=None):
    path = Path(path or SCORECARD)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    _GATE_CACHE.pop(str(path), None)
    return path


def feed_memory(doc, db, memory_db=None):
    """Scorecard note + one outcome note per newly scored call/horizon into the analyst's MiniLM memory."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server" / "ai"))
        from memory_ingest import ingest
    except Exception as error:   # memory is optional
        return {"error": f"memory unavailable: {error}"}
    added = 0
    _, dup = ingest("NOTE", doc["promptSummary"], ts=doc["generatedAt"], db_path=memory_db,
                    meta={"dedupeKey": f"scorecard:{doc['generatedAt'][:10]}:{doc['counts']['calls']}",
                          "source": "analyst_scoring", "gateOpen": doc["gateOpen"]})
    added += not dup
    q = """SELECT c.*, o.horizon_days, o.anchor_date, o.anchor_close, o.end_date, o.ret, o.excess_spy, o.excess_sector,
                  o.sector_etf, o.stop_hit FROM call_outcomes o JOIN analyst_calls c USING (call_id)
           WHERE o.status='scored' AND c.scorable=1"""
    for r in db.execute(q):
        d = DIRECTION[r["flag"]]
        hit = r["excess_spy"] is not None and d * r["excess_spy"] > 0
        vs_spy = "" if r["excess_spy"] is None else " ({:+.1f}% vs SPY".format(r["excess_spy"] * 100)
        vs_sec = "" if r["excess_sector"] is None else ", {:+.1f}% vs {}".format(r["excess_sector"] * 100, r["sector_etf"])
        stop_note = "" if r["stop_hit"] is None else (" Stop was hit." if r["stop_hit"] else " Stop held.")
        text = ("Outcome of my {flag} call on {t} ({origin}, anchored {ad} close {ac:.2f}): after {h} trading days ({ed}) "
                "the stock returned {ret:+.1f}%{spy}{sec}{close} -> {verdict} vs the market.{stop}").format(
            flag=r["flag"], t=r["ticker"], origin=r["origin"], ad=r["anchor_date"], ac=r["anchor_close"],
            h=r["horizon_days"], ed=r["end_date"], ret=r["ret"] * 100, spy=vs_spy, sec=vs_sec,
            close=")" if vs_spy else "", verdict="CORRECT" if hit else "WRONG", stop=stop_note)
        try:
            _, dup = ingest("EVENT", text, ticker=r["ticker"], ts=r["end_date"] + "T21:00:00Z", db_path=memory_db,
                            meta={"dedupeKey": f"outcome:{r['call_id']}:{r['horizon_days']}", "source": "analyst_scoring",
                                  "refPrice": r["anchor_close"], "flag": r["flag"], "horizon": r["horizon_days"]})
            added += not dup
        except ValueError:
            continue
    return {"added": added}


def run(store=None, scorecard_path=None, memory=True, permutations=PERMUTATIONS, **sources):
    started = time.time()
    db = connect(store)
    try:
        added, warnings = ingest_all(db, **sources)
        doc = build_scorecard(db, permutations=permutations)
        doc["warnings"] = warnings + doc["warnings"]
        doc["ingested"] = added
        try:
            from analyst_score import enrich_live_scorecard
            enrich_live_scorecard(doc)  # prior, IC, Brier, Newey-West, BH; never mutates gate / gateOpen
        except Exception as error:
            doc.setdefault("warnings", []).append(f"scorecard extras skipped: {error}")
        write_scorecard(doc, scorecard_path)
        mem = feed_memory(doc, db) if memory else None
        s = doc["counts"]["outcomes"]
        db.execute("INSERT INTO scoring_runs (at, status, ingested, scored, pending, gate, message) VALUES (?, 'ok', ?, ?, ?, ?, ?)",
                   (now_iso(), added, sum(v.get("scored", 0) for v in s.values()),
                    sum(v.get("pending", 0) for v in s.values()), json.dumps(doc["gateOpen"]), None))
        db.commit()
        return {"status": "ok", "ingested": added, "calls": doc["counts"]["calls"], "scorable": doc["counts"]["scorable"],
                "outcomes": s, "gateOpen": doc["gateOpen"], "memory": mem, "warnings": doc["warnings"],
                "scorecard": str(scorecard_path or SCORECARD), "seconds": round(time.time() - started, 1),
                "summary": doc["promptSummary"]}
    finally:
        db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="ingest, score and write the scorecard (always exits 0 unless --strict)")
    r.add_argument("--no-memory", action="store_true")
    r.add_argument("--permutations", type=int, default=PERMUTATIONS)
    r.add_argument("--strict", action="store_true")
    sub.add_parser("status")
    g = sub.add_parser("gate")
    g.add_argument("flag")
    g.add_argument("--horizons", default="20,60")
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            result = run(memory=not args.no_memory, permutations=args.permutations)
        except Exception as error:   # never block the paper run that follows
            result = {"status": "error", "message": f"{type(error).__name__}: {error}"}
            print(json.dumps(result))
            return 1 if args.strict else 0
        print(json.dumps(result, default=str))
        return 0
    if args.command == "status":
        doc = load_scorecard()
        print(json.dumps({"generatedAt": doc.get("generatedAt"), "gateOpen": doc.get("gateOpen"),
                          "counts": doc.get("counts"), "summary": doc.get("promptSummary")} if doc else
                         {"status": "missing"}, indent=2))
        return 0
    ok, why = gate_open(args.flag, [int(h) for h in args.horizons.split(",") if h])
    print(json.dumps({"flag": args.flag.upper(), "open": ok, "reason": why}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
