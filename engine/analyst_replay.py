"""Sealed historical replay so the analyst can build a starting (prior) reputation from backtests.

For each replay date the model sees point-in-time data only: prices, indicators, fundamentals and news
with as-of dates on or before that date, and no future rows. Tests assert this. A held-out portion of
the sample is masked (anonymized IDs, no names or sectors) to limit model recall; masked vs unmasked
scores are compared as a leakage check. Non-reasoning models are called through the existing Retort
path (server/ai/replay_verdict.mjs) when a key is available, otherwise a PIT-only heuristic is used.
Cost is capped by a sampled set of dates and tickers (config) and every result is cached.

The replay scorecard is stored as a separate 'prior' record
(Massive/Research/analyst_replay_prior.json). Gating that lets verdicts shrink positions must rely on
live out-of-sample calls (analyst_score.py); the prior never opens that gate.

  python analyst_replay.py run --dry-run          # sample + heuristic, no Retort calls
  python analyst_replay.py run --model heuristic
  python analyst_replay.py run --model retort     # non-reasoning Retort path, cost-capped
  python analyst_replay.py status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

ENGINE = Path(__file__).resolve().parent
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from fetch_and_model import DATA_DIR, PROJECT_ROOT, write_atomically
from analyst_score import (
    ALPHA, DEFAULT_DB, HORIZONS, MIN_N, _asof, _clean, _now_iso, _num, connect_prices,
    connect_score, evaluate_calls, load_sector_map, load_sp500, log_call, session_calendar,
    sector_of_name,
)

DEFAULT_CONFIG = {
    "n_dates": 12,
    "n_tickers": 15,
    "mask_fraction": 0.4,
    "max_calls": 80,
    "lookback": 260,
    "min_history": 80,
    "seed": 7,
    "model": "retort",          # "retort" or "heuristic"; retort falls back to heuristic if no key / failure
    "held_out_after_days": 60,  # last N calendar trading days are reserved for live scoring, not replay labels
    "timeout_sec": 90,
}
CONFIG_PATH = DATA_DIR / "Research" / "analyst_replay_config.json"
CACHE_DB = DATA_DIR / "Research" / "analyst_replay_cache.sqlite3"
PRIOR_PATH = DATA_DIR / "Research" / "analyst_replay_prior.json"
RETORT_SCRIPT = PROJECT_ROOT / "server" / "ai" / "replay_verdict.mjs"
CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS replay_cache (
  cache_key TEXT PRIMARY KEY,
  asof TEXT NOT NULL,
  ticker TEXT,
  alias TEXT,
  masked INTEGER NOT NULL,
  model TEXT NOT NULL,
  feature_hash TEXT NOT NULL,
  verdict TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""
IDENTITY_KEYS = ("ticker", "name", "company", "sector", "industry", "securityName", "gics")


class PitLeak(AssertionError):
    """A point-in-time bundle contained a future row or a masked identity leak."""


def load_config(path=None, overrides=None):
    cfg = dict(DEFAULT_CONFIG)
    path = Path(path or os.environ.get("ANALYST_REPLAY_CONFIG") or CONFIG_PATH)
    if path.is_file():
        try:
            cfg.update({k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if v is not None})
        except (OSError, ValueError):
            pass
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def connect_cache(path=None):
    path = Path(path or CACHE_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(CACHE_SCHEMA)
    return db


# ------------------------------------------------------------------ point-in-time bundle
def load_bars_pit(prices, ticker, asof, lookback=260):
    asof = _asof(asof)
    rows = prices.execute(
        "SELECT date, open, high, low, close, volume FROM daily_prices "
        "WHERE ticker=? AND date<=? ORDER BY date",
        (ticker, asof)).fetchall()
    bars = []
    cols = ("date", "open", "high", "low", "close", "volume")
    for r in rows[-int(lookback):]:
        bars.append(dict(r) if isinstance(r, (dict, sqlite3.Row)) else dict(zip(cols, r)))
    return bars


def compute_features_pit(bars, benchmark=None):
    """A compact PIT feature set (mirrors the spirit of server/ai/features.js, prices through asof only)."""
    if not bars:
        return None
    closes = [b["close"] for b in bars]
    last = bars[-1]
    price = float(last["close"])
    n = len(closes)

    def sma(window):
        if n < window:
            return None
        return float(sum(closes[-window:]) / window)

    def ret(h):
        if n <= h or closes[-1 - h] <= 0:
            return None
        return price / float(closes[-1 - h]) - 1

    daily = [math.log(closes[i] / closes[i - 1]) for i in range(1, n) if closes[i - 1] > 0]
    trs = []
    for i in range(1, len(bars)):
        b, pc = bars[i], bars[i - 1]["close"]
        trs.append(max(b["high"] - b["low"], abs(b["high"] - pc), abs(b["low"] - pc)))
    atr14 = (sum(trs[-14:]) / 14) if len(trs) >= 14 else None
    s20, s50, s100, s200 = sma(20), sma(50), sma(100), sma(200)
    std20 = float(np.std(closes[-20:], ddof=1)) if n >= 20 else None
    vol = (lambda h: float(np.std(daily[-h:], ddof=1) * math.sqrt(252)) if len(daily) >= h else None)
    window = closes[-252:]
    hi, lo = max(window), min(window)
    peak, max_dd = -1e18, 0.0
    for c in window:
        peak = max(peak, c)
        max_dd = min(max_dd, c / peak - 1)
    trend = "SIDEWAYS"
    r100 = ret(100)
    if s50 and price > s50 and (r100 is None or r100 > 0):
        trend = "UP"
    if s50 and price < s50 and (r100 is None or r100 < 0):
        trend = "DOWN"
    if s50 and s100 and s50 > s100 and r100 and r100 > 0.15:
        trend = "STRONG_UP"
    if s50 and s100 and s50 < s100 and r100 and r100 < -0.15:
        trend = "STRONG_DOWN"
    z20 = ((price - s20) / std20) if (s20 and std20) else None
    if z20 is None:
        over = "NEUTRAL"
    elif z20 >= 2:
        over = "OVEREXTENDED"
    elif z20 >= 1:
        over = "EXTENDED"
    elif z20 <= -2:
        over = "DEEPLY_OVERSOLD"
    elif z20 <= -1:
        over = "OVERSOLD"
    else:
        over = "NEUTRAL"
    rs60 = rs120 = beta = corr = None
    if benchmark:
        bmap = {b["date"]: b["close"] for b in benchmark}
        paired = [(b["close"], bmap[b["date"]]) for b in bars if b["date"] in bmap]
        def rel(h):
            if len(paired) <= h or paired[-1][1] <= 0 or paired[-1 - h][0] <= 0 or paired[-1 - h][1] <= 0:
                return None
            return (paired[-1][0] / paired[-1 - h][0]) / (paired[-1][1] / paired[-1 - h][1]) - 1
        rs60, rs120 = rel(60), rel(120)
        ra, rb = [], []
        for i in range(max(1, len(paired) - 120), len(paired)):
            if paired[i - 1][0] > 0 and paired[i - 1][1] > 0:
                ra.append(math.log(paired[i][0] / paired[i - 1][0]))
                rb.append(math.log(paired[i][1] / paired[i - 1][1]))
        if len(ra) > 30 and float(np.std(rb, ddof=1)) > 0:
            beta = float(np.cov(ra, rb, ddof=1)[0, 1] / np.var(rb, ddof=1))
            corr = float(np.corrcoef(ra, rb)[0, 1])
    return {
        "lastBarDate": last["date"], "sessions": n, "price": price, "priceSource": "pit_close",
        "ret1": ret(1), "ret5": ret(5), "ret20": ret(20), "ret60": ret(60), "ret100": ret(100),
        "sma20": s20, "sma50": s50, "sma100": s100, "sma200": s200,
        "distSma20": (price / s20 - 1) if s20 else None, "distSma50": (price / s50 - 1) if s50 else None,
        "zPrice20": z20, "atr14": atr14, "atrPct": (atr14 / price) if atr14 else None,
        "vol20": vol(20), "vol60": vol(60), "high252": hi, "low252": lo,
        "fromHigh252": price / hi - 1, "fromLow252": price / lo - 1, "maxDrawdown252": max_dd,
        "tsmom50": None if ret(50) is None else int(np.sign(ret(50))),
        "tsmom100": None if ret(100) is None else int(np.sign(ret(100))),
        "rs60VsSpy": rs60, "rs120VsSpy": rs120, "beta120": beta, "corrSpy120": corr,
        "trend": trend, "overextension": over,
    }


def alias_for(ticker, salt="replay", length=6):
    digest = hashlib.sha256(f"{salt}|{str(ticker).upper()}".encode("utf-8")).hexdigest()
    return "NAME_" + digest[:length].upper()


def filter_pit(rows, asof, date_keys=("date", "asof", "published", "publishedAt", "filed", "periodEnd", "endDate")):
    """Drop any row whose date-like field is after asof. Used for news and fundamentals."""
    asof = _asof(asof)
    kept = []
    for row in rows or []:
        dates = []
        if isinstance(row, dict):
            for k in date_keys:
                if row.get(k):
                    d = _asof(row[k])
                    if d:
                        dates.append(d)
        if dates and any(d > asof for d in dates):
            continue
        kept.append(row)
    return kept


def build_bundle(prices, ticker, asof, lookback=260, news=None, fundamentals=None,
                 name=None, sector=None, industry=None, masked=False, salt="replay", spy_bars=None):
    asof = _asof(asof)
    bars = load_bars_pit(prices, ticker, asof, lookback)
    bench = spy_bars if spy_bars is not None else load_bars_pit(prices, "SPY", asof, lookback)
    features = compute_features_pit(bars, bench)
    news = filter_pit(news, asof)
    fundamentals = filter_pit(fundamentals, asof)
    alias = alias_for(ticker, salt) if masked else ticker
    bundle = {
        "asof": asof,
        "masked": bool(masked),
        "lookback": lookback,
        "ticker": alias,
        "bars": bars,
        "benchmarkBars": bench,
        "features": features,
        "news": news,
        "fundamentals": fundamentals,
    }
    if not masked:
        bundle["trueTicker"] = ticker
        bundle["name"] = name
        bundle["sector"] = sector
        bundle["industry"] = industry
    else:
        # Identity must not appear in the sealed payload the model sees.
        bundle["trueTicker"] = None
        bundle["name"] = None
        bundle["sector"] = None
        bundle["industry"] = None
        bundle["alias"] = alias
    bundle["prompt"] = format_prompt(bundle)
    bundle["featureHash"] = hashlib.sha256(
        json.dumps(_clean({k: bundle[k] for k in ("asof", "ticker", "masked", "features", "news", "fundamentals")}),
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    return bundle


def format_prompt(bundle):
    """Plain-text context for the Retort verdict task. Masked bundles never include the real ticker or sector."""
    f = bundle.get("features") or {}
    who = bundle.get("ticker")
    title = f"Held-out name {who} (identity withheld)" if bundle.get("masked") else f"Ticker {who}"
    if not bundle.get("masked") and bundle.get("name"):
        title += f" ({bundle['name']}" + (f", sector {bundle['sector']}" if bundle.get("sector") else "") + ")"
    lines = [
        f"{title}. Point-in-time as of {bundle.get('asof')} (US session close). "
        "You have no data after this date. Do not assume future prices, filings or news.",
        "Price and indicators (through as-of only): " + json.dumps(_clean({
            k: f.get(k) for k in (
                "lastBarDate", "price", "ret5", "ret20", "ret60", "ret100", "distSma20", "distSma50",
                "zPrice20", "atrPct", "vol20", "vol60", "fromHigh252", "tsmom50", "tsmom100",
                "rs60VsSpy", "rs120VsSpy", "beta120", "trend", "overextension",
            )
        }), default=str),
    ]
    news = bundle.get("news") or []
    if news:
        headlines = []
        for item in news[-12:]:
            if isinstance(item, dict):
                headlines.append(f"- {item.get('date') or item.get('published') or ''}: {item.get('headline') or item.get('title') or item.get('text') or ''}")
            else:
                headlines.append(f"- {item}")
        lines.append("News available as of this date:\n" + "\n".join(headlines))
    else:
        lines.append("No news items available as of this date.")
    funds = bundle.get("fundamentals") or []
    if funds:
        lines.append("Fundamentals available as of this date: " + json.dumps(_clean(funds[-3:]), default=str)[:2000])
    else:
        lines.append("No dated fundamentals available as of this date.")
    if bundle.get("masked"):
        lines.append("This name is anonymized. You will not be given its ticker, company name or sector.")
    return "\n".join(lines)


def assert_point_in_time(bundle, asof, real_ticker=None):
    """Raise PitLeak if any future row is present, or if a masked bundle leaks identity."""
    asof = _asof(asof)
    if bundle.get("asof") != asof:
        raise PitLeak(f"bundle asof {bundle.get('asof')} != requested {asof}")
    for key in ("bars", "benchmarkBars"):
        for bar in bundle.get(key) or []:
            d = _asof(bar.get("date") if isinstance(bar, dict) else None)
            if d and d > asof:
                raise PitLeak(f"future {key} bar {d} after asof {asof}")
    for key in ("news", "fundamentals"):
        leaked = filter_pit(bundle.get(key) or [], "9999-01-01")  # just iterate
        for row in leaked:
            dates = [_asof(row[k]) for k in ("date", "asof", "published", "publishedAt", "filed", "periodEnd", "endDate")
                     if isinstance(row, dict) and row.get(k)]
            if any(d and d > asof for d in dates):
                raise PitLeak(f"future {key} row after asof {asof}: {row}")
    feats = bundle.get("features") or {}
    if feats.get("lastBarDate") and _asof(feats["lastBarDate"]) and _asof(feats["lastBarDate"]) > asof:
        raise PitLeak(f"feature lastBarDate {feats['lastBarDate']} after asof {asof}")
    if bundle.get("masked"):
        blob = json.dumps(bundle, default=str)
        # The prompt and payload must not contain the real ticker, a name, or a sector label.
        if real_ticker and real_ticker.upper() in blob.upper().replace("NAME_", ""):
            # Avoid false positives when a hash accidentally contains letter runs; require a ticker token.
            import re
            if re.search(r"\b" + re.escape(real_ticker.upper()) + r"\b", blob.upper()):
                raise PitLeak(f"masked bundle leaks ticker {real_ticker}")
        for key in IDENTITY_KEYS:
            if key in ("ticker",):
                continue
            if bundle.get(key):
                raise PitLeak(f"masked bundle still has {key}={bundle.get(key)!r}")
        if "sector " in (bundle.get("prompt") or "").lower():
            raise PitLeak("masked prompt mentions sector")
    return True


# ------------------------------------------------------------------ models
def heuristic_verdict(bundle):
    """Deterministic PIT-only verdict used in tests and when Retort is unavailable. No future data."""
    f = bundle.get("features") or {}
    price = _num(f.get("price")) or 0.0
    atr = _num(f.get("atr14")) or (price * 0.02 if price else None)
    trend, over, tsmom = f.get("trend"), f.get("overextension"), f.get("tsmom100")
    if trend in {"DOWN", "STRONG_DOWN"} or over == "OVEREXTENDED" or tsmom == -1:
        flag, action, opportunity = "RED", "AVOID" if trend == "STRONG_DOWN" else "TRIM", "NONE"
    elif trend in {"UP", "STRONG_UP"} and over in {"OVERSOLD", "DEEPLY_OVERSOLD"}:
        flag, action, opportunity = "GREEN", "BUY", "OVERSOLD_REVERSAL"
    elif trend in {"UP", "STRONG_UP"} and over in {None, "NEUTRAL"} and tsmom == 1:
        flag, action, opportunity = "GREEN", "ADD", "NONE"
    elif over in {"EXTENDED", "OVEREXTENDED"}:
        flag, action, opportunity = "YELLOW", "HOLD", "NONE"
    else:
        flag, action, opportunity = "YELLOW", "HOLD", "NONE"
    stop = (price - 2 * atr) if (price and atr) else None
    band = None
    if flag == "GREEN" and price and atr:
        band = [price - 0.5 * atr, price + 0.25 * atr]
    return {
        "ticker": bundle.get("ticker"), "asof": bundle.get("asof"), "source": "HEURISTIC",
        "flag": flag, "action": action, "opportunity": opportunity, "conviction": "MEDIUM",
        "trend": trend, "overextension": over, "refPrice": price, "stop": stop,
        "entryLow": None if not band else band[0], "entryHigh": None if not band else band[1],
        "summary": f"Heuristic PIT verdict {flag} from trend={trend}, stretch={over}, tsmom100={tsmom}.",
        "model": "heuristic-pit",
    }


def retort_available(root=None):
    script = Path(root or PROJECT_ROOT) / "server" / "ai" / "replay_verdict.mjs"
    return script.is_file()


def call_retort(bundle, timeout=90, node="node", script=None, env=None):
    """Call the existing Retort path with a sealed prompt. Never fetches live research. Returns a verdict dict."""
    import subprocess
    script = Path(script or RETORT_SCRIPT)
    if not script.is_file():
        raise FileNotFoundError(f"replay_verdict.mjs not found at {script}")
    payload = json.dumps({"prompt": bundle["prompt"], "asof": bundle["asof"], "ticker": bundle["ticker"],
                          "masked": bundle.get("masked")})
    # Do not pass secrets in our own argv or stdout; the child reads env the same way the dashboard does.
    child_env = dict(env or os.environ)
    result = subprocess.run(
        [node, str(script)], input=payload.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout, cwd=str(script.parent), env=child_env)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Retort replay exited {result.returncode}: {err}")
    out = result.stdout.decode("utf-8", "replace").strip()
    # The script prints a single JSON object (and retort logging is disabled unless TRADING_AI_RETORT_LOG=1).
    try:
        return json.loads(out.splitlines()[-1])
    except ValueError as error:
        raise RuntimeError(f"Retort replay returned non-JSON: {out[:300]}") from error


def cache_key(bundle, model):
    return hashlib.sha256(f"{model}|{bundle.get('featureHash')}|{int(bool(bundle.get('masked')))}|{bundle.get('asof')}".encode()).hexdigest()


def cached_verdict(cache, bundle, model):
    key = cache_key(bundle, model)
    row = cache.execute("SELECT verdict FROM replay_cache WHERE cache_key=?", (key,)).fetchone()
    if not row:
        return None, key
    try:
        payload = row["verdict"] if isinstance(row, sqlite3.Row) or isinstance(row, dict) else row[0]
        return json.loads(payload), key
    except ValueError:
        return None, key


def store_cached(cache, key, bundle, model, verdict, true_ticker=None):
    cache.execute(
        "INSERT OR REPLACE INTO replay_cache (cache_key, asof, ticker, alias, masked, model, feature_hash, verdict, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (key, bundle["asof"], true_ticker, bundle.get("ticker"), int(bool(bundle.get("masked"))),
         model, bundle.get("featureHash"), json.dumps(verdict), _now_iso()))
    cache.commit()


def generate_verdict(bundle, model="heuristic", cache=None, true_ticker=None, timeout=90, verdict_fn=None):
    """Return (verdict, meta) with cache hits counted. verdict_fn, if given, is used instead of Retort."""
    model_name = "heuristic-pit" if model == "heuristic" and verdict_fn is None else (model or "retort")
    if cache is not None:
        hit, key = cached_verdict(cache, bundle, model_name)
        if hit is not None:
            return hit, {"cached": True, "model": model_name, "key": key}
    else:
        key = cache_key(bundle, model_name)
    if verdict_fn is not None:
        verdict = verdict_fn(bundle)
        used = getattr(verdict_fn, "name", "custom")
    elif model == "heuristic":
        verdict = heuristic_verdict(bundle)
        used = "heuristic-pit"
    else:
        try:
            verdict = call_retort(bundle, timeout=timeout)
            used = verdict.get("model") or "retort"
        except Exception as error:
            verdict = heuristic_verdict(bundle)
            verdict["fallbackReason"] = str(error)[:300]
            used = "heuristic-pit"
    if cache is not None:
        store_cached(cache, key, bundle, used, verdict, true_ticker=true_ticker)
    return verdict, {"cached": False, "model": used, "key": key}


# ------------------------------------------------------------------ sampling and run
def pick_sample(prices, cfg, tickers=None, calendar=None):
    """Causal sample of dates and tickers with enough history and a 60-day future window for labels."""
    calendar = calendar or session_calendar(prices)
    held = int(cfg.get("held_out_after_days", 60))
    min_hist = int(cfg.get("min_history", 80))
    if len(calendar) < min_hist + held + 5:
        usable = calendar[min_hist:-1] if len(calendar) > min_hist + 1 else calendar[min_hist:]
    else:
        usable = calendar[min_hist:-held]
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    n_dates = min(int(cfg.get("n_dates", 12)), len(usable))
    dates = sorted(rng.choice(usable, size=n_dates, replace=False).tolist()) if n_dates else []
    if tickers is None:
        tickers = load_sp500() or ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "JPM", "XOM"]
    tickers = [t for t in tickers if t not in {"SPY"}]
    n_tickers = min(int(cfg.get("n_tickers", 15)), len(tickers))
    chosen = sorted(rng.choice(tickers, size=n_tickers, replace=False).tolist()) if n_tickers else []
    return dates, chosen


def run_replay(prices=None, cfg=None, news_by_ticker=None, funds_by_ticker=None, names=None,
               verdict_fn=None, cache=None, score_db=None, prior_path=None, prices_path=None,
               ingest_calls=True, tickers=None):
    """Run the sealed replay, score it as a prior, and write analyst_replay_prior.json.

    news_by_ticker / funds_by_ticker are optional {ticker: [dated dicts]}; they are filtered to asof.
    verdict_fn is used in tests; otherwise cfg['model'] selects retort or heuristic.
    """
    cfg = load_config(overrides=cfg)
    close_prices = False
    if prices is None:
        prices = connect_prices(prices_path)
        close_prices = True
    cache_owned = False
    if cache is None:
        cache = connect_cache()
        cache_owned = True
    warnings = []
    try:
        calendar = session_calendar(prices)
        dates, tickers = pick_sample(prices, cfg, tickers=tickers, calendar=calendar)
        mask_frac = float(cfg.get("mask_fraction", 0.4))
        rng = np.random.default_rng(int(cfg.get("seed", 7)) + 13)
        mask_set = set()
        if tickers and mask_frac > 0:
            k = max(1, int(round(len(tickers) * mask_frac))) if mask_frac < 1 else len(tickers)
            mask_set = set(rng.choice(tickers, size=min(k, len(tickers)), replace=False).tolist())
        sector_map = load_sector_map()
        names = names or {}
        news_by_ticker = news_by_ticker or {}
        funds_by_ticker = funds_by_ticker or {}
        max_calls = int(cfg.get("max_calls", 80))
        model = cfg.get("model") or "retort"
        if verdict_fn is not None:
            model = "custom"
        new_calls = 0
        cached_calls = 0
        records = []  # (true_ticker, masked, asof, verdict, bundle)
        spy_cache = {}
        for asof in dates:
            if asof not in spy_cache:
                spy_cache[asof] = load_bars_pit(prices, "SPY", asof, cfg.get("lookback", 260))
            for ticker in tickers:
                if new_calls >= max_calls:
                    warnings.append(f"Cost cap reached ({max_calls} uncached model calls); remaining pairs skipped.")
                    break
                masked = ticker in mask_set
                bundle = build_bundle(
                    prices, ticker, asof, lookback=cfg.get("lookback", 260),
                    news=news_by_ticker.get(ticker), fundamentals=funds_by_ticker.get(ticker),
                    name=names.get(ticker), sector=sector_map.get(ticker),
                    masked=masked, spy_bars=spy_cache[asof])
                if not bundle.get("features") or bundle["features"].get("sessions", 0) < cfg.get("min_history", 80):
                    continue
                assert_point_in_time(bundle, asof, real_ticker=ticker if masked else None)
                verdict, meta = generate_verdict(
                    bundle, model="heuristic" if model in {"heuristic", "custom"} and verdict_fn is None else (
                        "heuristic" if model == "heuristic" else "retort"),
                    cache=cache, true_ticker=ticker, timeout=cfg.get("timeout_sec", 90),
                    verdict_fn=verdict_fn)
                if meta.get("cached"):
                    cached_calls += 1
                else:
                    new_calls += 1
                # Restore the real ticker for scoring / logging. The model never saw it when masked.
                verdict = dict(verdict)
                verdict["ticker"] = ticker
                verdict["masked"] = masked
                verdict["asof"] = asof
                records.append((ticker, masked, asof, verdict, bundle))
            if new_calls >= max_calls:
                break

        # Log prior calls (origin='prior') into the score DB so they stay separate from live.
        score = connect_score(score_db) if not isinstance(score_db, sqlite3.Connection) else score_db
        owned_score = not isinstance(score_db, sqlite3.Connection)
        try:
            for ticker, masked, asof, verdict, _bundle in records:
                source = verdict.get("source") or "prior"
                created = f"{asof}T00:00:00Z"
                payload = {**verdict, "masked": masked}
                if _num(verdict.get("flag") or (verdict.get("flag"))) or verdict.get("flag"):
                    log_call(score, ticker, asof, created, "FLAG", flag=verdict.get("flag"),
                             opportunity=verdict.get("opportunity"), stop=verdict.get("stop"),
                             ref_price=verdict.get("refPrice") or verdict.get("ref_price"),
                             source=source, origin="prior", model=verdict.get("model"), payload=payload)
                if _num(verdict.get("stop")) is not None:
                    log_call(score, ticker, asof, created, "STOP", flag=verdict.get("flag"),
                             stop=verdict.get("stop"),
                             ref_price=verdict.get("refPrice") or verdict.get("ref_price"),
                             source=source, origin="prior", model=verdict.get("model"), payload=payload)
                if verdict.get("opportunity") and str(verdict.get("opportunity")).upper() not in {"", "NONE"}:
                    log_call(score, ticker, asof, created, "OPPORTUNITY", flag=verdict.get("flag"),
                             opportunity=verdict.get("opportunity"),
                             ref_price=verdict.get("refPrice") or verdict.get("ref_price"),
                             source=source, origin="prior", model=verdict.get("model"), payload=payload)
            score.commit()
            prior_rows = [dict(r) for r in score.execute("SELECT * FROM calls WHERE origin='prior'")]
            # Score all prior, plus masked vs unmasked splits for the leakage check.
            universe = load_sp500()
            sectors = load_sector_map()
            prior_block = evaluate_calls(prior_rows, calendar, prices, sectors, universe,
                                         min_n=cfg.get("min_n", MIN_N), alpha=cfg.get("alpha", ALPHA),
                                         origin="prior", seed=cfg.get("seed", 7), cache_db=score)
            masked_rows, unmasked_rows = [], []
            for r in prior_rows:
                try:
                    payload = json.loads(r["payload"]) if r["payload"] else {}
                except ValueError:
                    payload = {}
                (masked_rows if payload.get("masked") else unmasked_rows).append(r)
            masked_block = evaluate_calls(masked_rows, calendar, prices, sectors, universe,
                                          min_n=max(10, MIN_N // 2), alpha=ALPHA, origin="prior_masked",
                                          seed=cfg.get("seed", 7), cache_db=score) if masked_rows else None
            unmasked_block = evaluate_calls(unmasked_rows, calendar, prices, sectors, universe,
                                            min_n=max(10, MIN_N // 2), alpha=ALPHA, origin="prior_unmasked",
                                            seed=cfg.get("seed", 7), cache_db=score) if unmasked_rows else None
            leakage = leakage_report(masked_block, unmasked_block)
            prior_block["gate"] = {
                "open": False, "redMaySkip": False, "yellowMayShrink": False, "bandMayDefer": False,
                "reason": "Replay/prior scores are informative only and never open the live gate.",
            }
            prior_block["leakage"] = leakage
            prior_block["sample"] = {
                "dates": dates, "tickers": tickers, "maskedTickers": sorted(mask_set),
                "nVerdicts": len(records), "newCalls": new_calls, "cachedCalls": cached_calls,
                "maxCalls": max_calls, "model": model, "config": {k: cfg[k] for k in DEFAULT_CONFIG},
            }
            doc = {
                "schema": "analyst_replay_prior/v1",
                "scoredAt": _now_iso(),
                "prior": prior_block,
                "leakage": leakage,
                "warnings": warnings,
            }
            path = Path(prior_path or PRIOR_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_atomically(path, json.dumps(_clean(doc), indent=2, default=str).encode("utf-8"))
            return _clean({**doc, "path": str(path)})
        finally:
            if owned_score:
                score.close()
    finally:
        if cache_owned:
            cache.close()
        if close_prices:
            prices.close()


def leakage_report(masked, unmasked):
    def ic(block, h="20"):
        if not block:
            return None
        return ((block.get("informationCoefficient") or {}).get(str(h)) or {}).get("ic")

    def hit(block, flag, h="20"):
        if not block:
            return None
        return (((block.get("flags") or {}).get(flag) or {}).get(str(h)) or {}).get("hitRate")

    mic, uic = ic(masked), ic(unmasked)
    warning = None
    if mic is not None and uic is not None and (uic - mic) >= 0.10:
        warning = (f"Unmasked 20-day information coefficient is materially higher than masked "
                   f"({uic:.3f} vs {mic:.3f}); possible ticker-recall leakage. "
                   f"The prior is not used for gating.")
    return _clean({
        "nMasked": (masked or {}).get("calls", {}).get("FLAG") if masked else 0,
        "nUnmasked": (unmasked or {}).get("calls", {}).get("FLAG") if unmasked else 0,
        "ic20Masked": mic, "ic20Unmasked": uic,
        "greenHit20Masked": hit(masked, "GREEN"), "greenHit20Unmasked": hit(unmasked, "GREEN"),
        "redHit20Masked": hit(masked, "RED"), "redHit20Unmasked": hit(unmasked, "RED"),
        "warning": warning,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--model", choices=["retort", "heuristic"])
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--n-dates", type=int)
    parser.add_argument("--n-tickers", type=int)
    parser.add_argument("--mask-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Force the PIT heuristic (no Retort calls).")
    args = parser.parse_args(argv)
    if args.command == "status":
        path = PRIOR_PATH
        if not path.is_file():
            print(json.dumps({"error": "No replay prior yet. Run: python analyst_replay.py run"}))
            return
        print(path.read_text(encoding="utf-8"))
        return
    overrides = {k: v for k, v in {
        "model": "heuristic" if args.dry_run else args.model,
        "max_calls": args.max_calls, "n_dates": args.n_dates, "n_tickers": args.n_tickers,
        "mask_fraction": args.mask_fraction, "seed": args.seed,
    }.items() if v is not None}
    doc = run_replay(cfg=overrides)
    prior = doc.get("prior") or {}
    print(json.dumps(_clean({
        "path": doc.get("path"),
        "sample": prior.get("sample"),
        "leakage": doc.get("leakage"),
        "ic20": (prior.get("informationCoefficient") or {}).get("20"),
        "green20": (prior.get("flags") or {}).get("GREEN", {}).get("20"),
        "red20": (prior.get("flags") or {}).get("RED", {}).get("20"),
        "gate": prior.get("gate"),
        "warnings": doc.get("warnings"),
    }), indent=2, default=str))


if __name__ == "__main__":
    main()
