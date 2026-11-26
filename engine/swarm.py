"""Swarm of simulated $10,000 paper accounts (shadow only; no broker is ever called).

Each account runs one variant of the shared strategy interface (strategies/*, portfolio_sim.simulate,
intraday.run_sessions) drawn from a logged, seeded design:

  * account 0  = the production configuration (CombinedStrategy defaults = what the real paper account trades)
  * account 1  = the TSMOM baseline (TimeSeriesMomentum defaults, 50/100-day)
  * accounts 2.. = a Latin-hypercube design over the swing and day-trade parameter space, so with only 50
    accounts every level of every parameter is still used about equally often (balanced one-way margins).

Every account carries two legs, mirroring the real setup (separate $10k swing and $10k day-trade accounts):
  swing   - daily bars, next-open fills, Schwab costs (SEC + TAF + spread + impact), whole shares, $100 minimum
            order, gross <= 100% of a $10k account (no margin), 30% per-name cap, 4% daily-loss circuit breaker,
            protective stops with gap fills; PDT: same-day round trips are counted and an account with more than
            3 in any 5 sessions is disqualified from selection; FIFO tax lots are tracked for every account
            (after-tax equity) and the tax-aware policy (defer / harvest / wash avoidance) is a design dimension.
  daytrade - opening-range breakout on cached Schwab 5-minute bars with the PDT 3-in-5 rule enforced.

Honesty protocol (``run``):
  1. The design (variants, seed, evaluation start, sealed holdout window) is frozen in design.json on the first
     run. The final ``HOLDOUT_FRACTION`` of the evaluation sessions is the sealed holdout.
  2. Variants are ranked and the champion is chosen ONLY from pre-holdout returns (select() never sees holdout
     rows; a unit test perturbs holdout data and checks the selection is unchanged). Walk-forward folds inside
     the pre-holdout period show how a "pick the best so far" rule would have done (stitched and re-simulated).
  3. The holdout is then evaluated once; every evaluation is appended to holdout_ledger.json so re-use of the
     holdout (e.g. after a redesign) is visible.
  4. Overfitting statistics: deflated Sharpe (trials = number of variants, and the eigenvalue effective number),
     probability of backtest overfitting by CSCV, White's Reality Check and Hansen's SPA (stationary bootstrap)
     against the TSMOM baseline and SPY, and per-variant one-sided bootstrap p-values with Holm and
     Benjamini-Hochberg corrections.

``forward`` (daily, after the paper run) advances every account through the sessions after the design end
(a deterministic shadow replay starting with $10,000 at the first post-design session; equivalent to advancing
one session per day because every strategy is causal) and appends new rows to forward/ledger.jsonl.

Nothing here is promoted automatically: the status file carries a recommendation only.

Usage (from engine/):
  python swarm.py run [--n 50] [--workers 8] [--redesign] [--no-orb]
  python swarm.py forward
  python swarm.py status
Outputs: Massive/Research/swarm/ (gitignored).
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically

SWARM_DIR = Path(os.environ.get("SWARM_OUT") or DATA_DIR / "Research" / "swarm")   # override for tests
CACHE_DIR = SWARM_DIR / "cache"
FORWARD_DIR = SWARM_DIR / "forward"
DESIGN_PATH = SWARM_DIR / "design.json"
STATUS_PATH = SWARM_DIR / "swarm_status.json"
LEDGER_PATH = SWARM_DIR / "holdout_ledger.json"
CODE_VERSION = "swarm-1"

DEFAULT_N = 50
DEFAULT_SEED = 20261125
INITIAL_EQUITY = 10_000.0
WARMUP = 110                 # same evaluation start as research.py (100-day lookback + buffer)
HOLDOUT_FRACTION = 0.25      # final quarter of the evaluation sessions is sealed
WF_TRAIN_MIN = 126           # walk-forward inside the pre-holdout period: expanding train window ...
WF_TEST_LEN = 42             # ... and two-month test windows
MIN_ORDER_VALUE = 100.0
DAILY_LOSS_LIMIT = 0.04
PER_NAME_CAP = 0.30
PDT_MAX, PDT_WINDOW = 3, 5
BOOTSTRAP = 2000
MEAN_BLOCK = 10
TRADING_DAYS = 252

# Levels for the Latin-hypercube design. Every value here is logged into design.json.
SWING_SPACE = {
    "overlay": [False, True],
    "regime": ["hmm", "fixed", "off"],
    "horizons": [[20, 60], [30, 90], [50, 100], [60, 120]],
    "target_vol": [0.08, 0.10, 0.12, 0.15],
    "max_positions": [6, 8, 10, 12],
    "rebalance": ["W-FRI", "D", "M"],
    "band": [0.01, 0.02, 0.04],
    "trail_atr": [None, 3.0, 5.0],
    "mr_lookback": [5, 10],
    "mr_entry_z": [-1.5, -2.0, -2.5],
    "mr_exit_z": [0.0, 0.5],
    "mr_max_hold": [5, 10],
    "mr_stop_atr": [2.0, 3.0, 4.0],
    "regime_threshold": [0.35, 0.45, 0.55],
    "tax_aware": [False, True],
}
ORB_SPACE = {
    "or_minutes": [15, 30],
    "target_r": [1.0, 2.0, None],
    "stop_frac": [0.5, 1.0],
    "min_rel_volume": [1.0, 1.5],
    "risk_per_trade": [0.0025, 0.005, 0.01],
    "max_positions": [1, 2],
    "entry_cutoff": [660, 690, 720],
    "trend_window": [20, 50],
}
PRODUCTION_SWING = {"family": "production"}
TSMOM_SWING = {"family": "tsmom_baseline"}


def log(message):
    print(f"{datetime.now():%H:%M:%S} swarm {message}", file=sys.stderr, flush=True)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value.date())
    return str(value)


def _finite(value):
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(_finite(payload), indent=2, default=_json_default, allow_nan=False).encode())


def stable_hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]


# ====================================================================== design
def latin_levels(space, n, seed):
    """n rows; each parameter's levels appear floor(n/L) or ceil(n/L) times (Latin hypercube margins)."""
    if n <= 0:
        return []
    keys = list(space)
    try:
        from scipy.stats import qmc
        try:
            sampler = qmc.LatinHypercube(d=len(keys), rng=np.random.default_rng(seed))
        except TypeError:
            sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
        u = sampler.random(n)
    except Exception:                                     # pragma: no cover - scipy is a declared dependency
        rng = np.random.default_rng(seed)
        u = np.column_stack([(rng.permutation(n) + rng.random(n)) / n for _ in keys])
    rows = []
    for r in range(n):
        rows.append({k: space[k][min(int(u[r, j] * len(space[k])), len(space[k]) - 1)] for j, k in enumerate(keys)})
    return rows


def normalize_swing(v, tax_available=True):
    """Drop parameters that do nothing for this variant so duplicates are visible and the config is explicit."""
    v = dict(v)
    if v.get("family"):
        return v
    v["family"] = "variant"
    if not v["overlay"]:
        for k in ("mr_lookback", "mr_entry_z", "mr_exit_z", "mr_max_hold", "mr_stop_atr", "regime_threshold"):
            v[k] = None
        if v["regime"] == "fixed":           # the fixed filter only gates the overlay: with no overlay it equals 'off'
            v["regime"] = "hmm"
    if v["regime"] != "hmm":
        v["regime_threshold"] = None
    if not tax_available:
        v["tax_aware"] = False
    return v


def make_design(n=DEFAULT_N, seed=DEFAULT_SEED, tax_available=True, orb_available=True):
    if n < 10:
        raise SystemExit("--n must be at least 10 (CSCV and the multiple-testing corrections need a cross-section)")
    sampled = n - 2
    swing = [normalize_swing(v, tax_available) for v in latin_levels(SWING_SPACE, sampled, seed)]
    orb = latin_levels(ORB_SPACE, sampled, seed + 1) if orb_available else [None] * sampled
    accounts = [{"id": 0, "label": "production", "swing": dict(PRODUCTION_SWING), "orb": {} if orb_available else None},
                {"id": 1, "label": "tsmom_baseline", "swing": dict(TSMOM_SWING), "orb": {} if orb_available else None}]
    for i, (s, o) in enumerate(zip(swing, orb)):
        accounts.append({"id": i + 2, "label": f"acct{i + 2:03d}", "swing": s, "orb": o})
    margins = {k: {json.dumps(lvl): sum(1 for v in swing if v.get(k) == lvl) for lvl in levels}
               for k, levels in SWING_SPACE.items()}
    return {"accounts": accounts, "n": n, "seed": seed, "method": "Latin hypercube (scipy.stats.qmc), 2 fixed baselines",
            "swingSpace": SWING_SPACE, "orbSpace": ORB_SPACE, "swingLevelCounts": margins,
            "taxAvailable": tax_available, "orbAvailable": orb_available}


# ====================================================================== data
def data_fingerprint():
    parts = []
    for name in ("prices.sqlite3", "prices.sqlite3-wal"):
        p = DATA_DIR / name
        if p.exists():
            st = p.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
    return stable_hash(parts)


def load_panel_cached(refresh=False):
    """load_panel() takes ~100 s; the swarm and its workers share one pickled copy keyed by the database stat."""
    from market_data import load_panel
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = data_fingerprint()
    path = CACHE_DIR / f"panel_{key}.pkl"
    if path.exists() and not refresh:
        with open(path, "rb") as fh:
            return pickle.load(fh), path
    log("loading price panel (cached for later runs)")
    panel = load_panel()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(panel, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    for old in CACHE_DIR.glob("panel_*.pkl"):
        if old != path:
            try:
                old.unlink()
            except OSError:
                pass
    return panel, path


def subset_panel(panel, tickers):
    from market_data import Panel
    cols = [t for t in tickers if t in panel.close.columns]
    pick = lambda f: f[cols]
    return Panel(pick(panel.open), pick(panel.high), pick(panel.low), pick(panel.close), pick(panel.volume),
                 pick(panel.dividends), panel.classes.reindex(cols),
                 None if panel.masked_returns is None else panel.masked_returns.reindex(columns=cols))


def load_orb_bundle(panel):
    """(bars5, small daily panel) or None when no Schwab bars are cached."""
    try:
        from contextlib import closing
        import intraday as it
        from schwab_data import connect_cache
        with closing(connect_cache()) as cache:
            bars = it.load_bars(cache, 5)
    except Exception as error:   # noqa: BLE001 - day-trade leg is optional
        log(f"day-trade bars unavailable: {error}")
        return None
    if bars is None or bars.empty:
        return None
    small = subset_panel(panel, sorted(set(bars["symbol"]) | {"SPY"}))
    return bars, small


def tax_available():
    try:
        import tax  # noqa: F401
        import tax_policy  # noqa: F401
        return True
    except Exception:
        return False


# ====================================================================== one account (worker side)
_WORKER = {}


def _worker_init(panel_path, orb_path):
    import warnings
    warnings.filterwarnings("ignore")
    with open(panel_path, "rb") as fh:
        _WORKER["panel"] = pickle.load(fh)
    _WORKER["orb"] = None
    if orb_path:
        with open(orb_path, "rb") as fh:
            _WORKER["orb"] = pickle.load(fh)
    _install_regime_cache()


def _install_regime_cache():
    """The HMM policy depends only on the panel and the regime settings, so variants sharing them share one fit."""
    import strategies.combined as sc
    if getattr(sc.regime_policy, "_swarm_cached", False):
        return
    original = sc.regime_policy
    memo = {}

    def cached(panel, model="hmm", overrides=None):
        key = (id(panel), len(panel.dates), model, json.dumps(overrides or {}, sort_keys=True, default=str))
        if key not in memo:
            memo[key] = original(panel, model=model, overrides=overrides)
        return memo[key]

    cached._swarm_cached = True
    sc.regime_policy = cached


class TrailingStop:
    """Causal ATR trailing stop for names held by the trend sleeve: 20-session highest close - k x ATR(14)."""

    def __init__(self, k):
        self.k = k

    def frame(self, panel, weights):
        from risk import atr
        cols = [c for c in weights.columns if (weights[c] > 0).any()]
        if not cols:
            return None
        close = panel.close[cols]
        level = close.rolling(20, min_periods=5).max() - self.k * atr(panel.high[cols], panel.low[cols], close)
        return level.where(weights[cols] > 0).reindex(columns=panel.close.columns)


def build_strategy(swing):
    from strategies import CombinedStrategy, RegimeMeanReversion, TimeSeriesMomentum
    family = swing.get("family")
    if family == "production":
        # The paper config as of the swarm design (hmm regime). Pinned, because CombinedStrategy's default
        # regime_model changed to "fixed" on 2026-11-25 and account 0 must keep meaning the same thing.
        return CombinedStrategy(regime_model="hmm")
    if family == "tsmom_baseline":
        return TimeSeriesMomentum()
    if family == "old_overlay_fixed":          # the real paper config before 2026-11-25
        return CombinedStrategy(regime_model="fixed")
    if family == "sleeve_meanrev":
        return RegimeMeanReversion()
    if family in ("growth_overlay", "sleeve_growth_mr"):
        from strategies.growth_meanrev import GrowthMeanReversion
        growth = GrowthMeanReversion(**dict(swing.get("growth") or {}))
        if family == "sleeve_growth_mr":
            return growth
        return CombinedStrategy(sleeves=[TimeSeriesMomentum(**dict(swing.get("tsmom") or {})), growth],
                                regime_model="fixed")
    if family in ("tsmom_exhaustion", "tsmom_stocks_exhaustion", "tsmom_stocks", "fade_exhaustion", "tsmom_plus_fade"):
        from exhaustion import build_strategy as build_exh
        return build_exh(swing)
    if family in ("tsmom_prob_filter", "prob_sized"):
        from prob_model import ProbStrategy
        return ProbStrategy(**dict(swing.get("prob") or {}), mode=family)
    mp = int(swing["max_positions"])
    tsmom = TimeSeriesMomentum(horizons=tuple(swing["horizons"]), target_vol=swing["target_vol"],
                               max_positions=max(3, mp - 2) if swing["overlay"] else mp,
                               rebalance=swing["rebalance"], per_name_cap=PER_NAME_CAP)
    sleeves = [tsmom]
    if swing["overlay"]:
        sleeves.append(RegimeMeanReversion(lookback=int(swing["mr_lookback"]), entry_z=swing["mr_entry_z"],
                                           exit_z=swing["mr_exit_z"], max_hold=int(swing["mr_max_hold"]),
                                           stop_atr=swing["mr_stop_atr"]))
    params = {"target_vol": swing["target_vol"], "max_positions": mp, "per_name_cap": PER_NAME_CAP,
              "regime_model": swing["regime"]}
    if swing.get("regime_threshold") is not None:
        params["regime_meanrev_threshold"] = swing["regime_threshold"]
    return CombinedStrategy(sleeves=sleeves, **params)


def tax_config(aware):
    from tax import TaxConfig
    if aware:
        return TaxConfig(enabled=True)
    # tracking only: lots, wash sales and after-tax equity, but no trade adjustments
    return TaxConfig(enabled=True, defer_near_lt=False, harvest_losses=False, avoid_wash_sales=False)


def pdt_stats(trades, dates):
    """Same-day round trips (bought and sold in one session) and the worst 5-session count."""
    if trades is None or not len(trades):
        return {"sameDayRoundTrips": 0, "maxIn5": 0, "breach": False}
    same = trades[trades["holding_sessions"] == 0]
    if not len(same):
        return {"sameDayRoundTrips": 0, "maxIn5": 0, "breach": False}
    counts = pd.Series(0, index=pd.DatetimeIndex(dates))
    for day in pd.to_datetime(same["exit_date"]):
        if day in counts.index:
            counts[day] += 1
    worst = int(counts.rolling(PDT_WINDOW, min_periods=1).sum().max())
    return {"sameDayRoundTrips": int(len(same)), "maxIn5": worst, "breach": worst > PDT_MAX}


def simulate_swing(panel, swing, start, use_tax=True, end=None):
    """Returns (SimResult, weights, stops) for one swing variant with full small-account realism."""
    from costs import CostModel
    from portfolio_sim import simulate
    from risk import CircuitBreaker
    strategy = build_strategy(swing)
    weights = strategy.target_weights(panel)
    try:
        if swing.get("dilution_filter", True):
            from dilution import apply_to_weights, load_flags, flag_lookup_from_payload, config_from_settings
            payload = load_flags()
            if payload:
                weights = apply_to_weights(weights, flag_lookup_from_payload(payload), config_from_settings(swing))
    except Exception:
        pass
    stops = strategy.stops(panel)
    trail = swing.get("trail_atr")
    if trail:
        extra = TrailingStop(float(trail)).frame(panel, getattr(strategy, "components", {}).get("tsmom", weights))
        if extra is not None:
            stops = extra if stops is None else stops.combine_first(extra)
    tcfg = tax_config(bool(swing.get("tax_aware"))) if use_tax and tax_available() else None
    result = simulate(panel, weights, stops, CostModel.for_broker("schwab"), INITIAL_EQUITY, "next_open",
                      start=start, end=end, breaker=CircuitBreaker(DAILY_LOSS_LIMIT),
                      min_trade_weight=float(swing.get("band") or 0.02),
                      sleeve_of=getattr(strategy, "sleeve_of", None), tax_config=tcfg,
                      whole_shares=True, min_order_value=MIN_ORDER_VALUE)
    return result, weights, stops


def _sparse(frame):
    if frame is None:
        return None
    frame = frame.astype("float32")
    keep = [c for c in frame.columns if np.isfinite(frame[c].to_numpy()).any() and (frame[c].fillna(0) != 0).any()]
    return frame[keep]


def run_swing_account(task):
    """Worker entry: one swing leg. Cached by (variant, data, code)."""
    account, start, cache_key = task["account"], task["start"], task["cacheKey"]
    path = CACHE_DIR / f"swing_{cache_key}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    panel = _WORKER.get("panel") if _WORKER else task["panel"]
    try:
        use_tax = account["swing"].get("use_tax", True)
        result, weights, stops = simulate_swing(panel, account["swing"], pd.Timestamp(start), use_tax=use_tax)
        dates = result.returns.index
        out = {"id": account["id"], "ok": True, "returns": result.returns.astype(float),
               "equity": result.equity.astype(float),
               "afterTaxEquity": None if result.after_tax_equity is None else result.after_tax_equity.astype(float),
               "turnover": float(result.turnover.mean() * TRADING_DAYS), "costs": float(result.costs.sum()),
               "gross": float(result.gross_exposure.mean()), "trades": int(len(result.trades)),
               "pdt": pdt_stats(result.trades, dates), "minCash": None,
               "weights": _sparse(weights), "stops": _sparse(stops), "seconds": round(time.time() - t0, 2)}
    except Exception as error:   # noqa: BLE001 - one bad variant must not kill the swarm
        out = {"id": account["id"], "ok": False, "error": f"{type(error).__name__}: {error}",
               "seconds": round(time.time() - t0, 2)}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return out


def orb_params(orb):
    import intraday as it
    return it.ORBParams(**(orb or {}))


def run_orb_account(task):
    account, cache_key = task["account"], task["cacheKey"]
    path = CACHE_DIR / f"orb_{cache_key}.pkl"
    if path.exists():
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    bundle = _WORKER.get("orb") if _WORKER else task.get("bundle")
    try:
        import intraday as it
        from costs import CostModel
        bars, small = bundle
        sessions = task.get("sessions")
        result, _ = it.backtest(bars, small, orb_params(account["orb"]), 5, INITIAL_EQUITY,
                                CostModel.for_broker("schwab"), sessions=sessions)
        daily = result.daily
        returns = pd.Series(daily["return"].to_numpy(dtype=float), index=pd.to_datetime(daily["session_date"])) \
            if len(daily) else pd.Series(dtype=float)
        out = {"id": account["id"], "ok": True, "returns": returns, "trades": int(len(result.trades)),
               "skipped": dict(result.skipped), "seconds": round(time.time() - t0, 2)}
    except Exception as error:   # noqa: BLE001
        out = {"id": account["id"], "ok": False, "error": f"{type(error).__name__}: {error}",
               "seconds": round(time.time() - t0, 2)}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return out


def run_tasks(func, tasks, workers, panel_path, orb_path):
    if workers <= 1 or len(tasks) <= 1:
        _worker_init(panel_path, orb_path)
        return [func(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(str(panel_path), orb_path)) as pool:
        for number, res in enumerate(pool.map(func, tasks, chunksize=1)):
            results.append(res)
            if (number + 1) % 10 == 0 or number + 1 == len(tasks):
                log(f"{func.__name__}: {number + 1}/{len(tasks)} done")
    return results


def default_workers():
    cpus = os.cpu_count() or 4
    return max(1, min(8, cpus - 4))   # each worker holds a ~150 MB panel plus strategy frames


# ====================================================================== statistics
def sharpe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def sharpe_cols(M):
    M = np.asarray(M, dtype=float)
    sd = M.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, M.mean(axis=0) / sd * math.sqrt(TRADING_DAYS), np.nan)


def perf(returns, after_tax_equity=None):
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return {"sessions": int(len(r))}
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    out = {"sessions": int(len(r)), "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
           "totalReturn": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else None,
           "vol": float(r.std(ddof=1) * math.sqrt(TRADING_DAYS)), "sharpe": sharpe(r),
           "maxDrawdown": float((eq / eq.cummax() - 1).min())}
    neg = r[r < 0]
    out["sortino"] = float(r.mean() / neg.std(ddof=1) * math.sqrt(TRADING_DAYS)) if len(neg) > 2 and neg.std() > 0 else None
    if after_tax_equity is not None:
        at = pd.Series(after_tax_equity).reindex(r.index).dropna()
        if len(at) > 2:
            ar = at.pct_change().dropna()
            growth = at.iloc[-1] / at.iloc[0]
            out["afterTaxSharpe"] = sharpe(ar)
            out["afterTaxCagr"] = float(growth ** (1 / years) - 1) if years > 0 and growth > 0 else None
    return out


def effective_trials(M):
    """Eigenvalue participation ratio of the return correlation matrix: (sum l)^2 / sum l^2."""
    M = np.asarray(M, dtype=float)
    if M.shape[1] < 2:
        return 1.0
    C = np.corrcoef(M, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    lam = np.clip(np.linalg.eigvalsh(C), 0, None)
    return float(lam.sum() ** 2 / (lam ** 2).sum()) if (lam ** 2).sum() > 0 else 1.0


def deflated_sharpe(x, trials, trial_sharpes):
    from quant_tools import sharpe_stats
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sd = float(np.nanstd(trial_sharpes, ddof=1)) if np.isfinite(trial_sharpes).sum() > 2 else None
    stats = sharpe_stats(x, TRADING_DAYS, 0.0, max(2, int(round(trials))), sd)
    return {"sharpe": stats["sharpeAnnual"], "psr": stats["probabilisticSharpe"],
            "trials": stats.get("deflatedSharpe", {}).get("trials"),
            "expectedMaxSharpe": stats.get("deflatedSharpe", {}).get("expectedMaxSharpeAnnual"),
            "deflatedSharpeProbability": stats.get("deflatedSharpe", {}).get("probability"),
            "trialSharpeStd": sd, "n": stats["n"], "skew": stats["skew"], "kurtosis": stats["kurtosis"]}


def pbo_cscv(M, n_blocks=12):
    """Bailey, Borwein, Lopez de Prado & Zhu (2015) CSCV. M: T x N returns. Returns PBO and diagnostics."""
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 2 or T < 2 * n_blocks:
        return {"pbo": None, "note": "not enough data"}
    n_blocks -= n_blocks % 2
    size = T // n_blocks
    M = M[T - size * n_blocks:]
    blocks = M.reshape(n_blocks, size, N)
    s1, s2, cnt = blocks.sum(axis=1), (blocks ** 2).sum(axis=1), size

    def sr(idx):
        n = cnt * len(idx)
        mean = s1[list(idx)].sum(axis=0) / n
        var = (s2[list(idx)].sum(axis=0) - n * mean ** 2) / (n - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(var > 0, mean / np.sqrt(var) * math.sqrt(TRADING_DAYS), -np.inf)

    logits, is_best, oos_best, oos_median = [], [], [], []
    all_idx = set(range(n_blocks))
    for combo in itertools.combinations(range(n_blocks), n_blocks // 2):
        is_sr, oos_sr = sr(combo), sr(tuple(sorted(all_idx - set(combo))))
        best = int(np.nanargmax(is_sr))
        rank = (oos_sr < oos_sr[best]).sum() + 0.5 * ((oos_sr == oos_sr[best]).sum() - 1) + 1
        omega = rank / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
        is_best.append(is_sr[best])
        oos_best.append(oos_sr[best])
        oos_median.append(float(np.nanmedian(oos_sr)))
    logits, is_best, oos_best = np.array(logits), np.array(is_best), np.array(oos_best)
    finite = np.isfinite(is_best) & np.isfinite(oos_best)
    slope = float(np.polyfit(is_best[finite], oos_best[finite], 1)[0]) if finite.sum() > 3 else None
    return {"pbo": float((logits <= 0).mean()), "splits": int(len(logits)), "blocks": n_blocks, "blockSessions": size,
            "variants": N, "medianLogit": float(np.median(logits)),
            "probOosLoss": float((oos_best < 0).mean()), "medianIsBestSharpe": float(np.nanmedian(is_best)),
            "medianOosOfIsBestSharpe": float(np.nanmedian(oos_best)), "degradationSlope": slope,
            "logitHistogram": np.histogram(logits, bins=10)[0].tolist()}


def stationary_indices(T, B, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    for t in range(1, T):
        jump = rng.random(B) < p
        idx[:, t] = np.where(jump, rng.integers(0, T, B), (idx[:, t - 1] + 1) % T)
    return idx


def spa_test(d, B=BOOTSTRAP, mean_block=MEAN_BLOCK, seed=7):
    """White (2000) Reality Check and Hansen (2005) SPA on d (T x K excess returns vs the benchmark).

    H0: no candidate beats the benchmark. Also returns one-sided studentized bootstrap p-values per candidate."""
    d = np.asarray(d, dtype=float)
    T, K = d.shape
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    boot = np.empty((B, K))
    for start in range(0, B, 250):
        b = min(250, B - start)
        idx = stationary_indices(T, b, mean_block, rng)
        boot[start:start + b] = d[idx].mean(axis=1)
    centered = boot - dbar
    omega = np.sqrt(T) * centered.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    rt = math.sqrt(T)
    # Reality Check (non-studentized)
    v = np.nanmax(rt * dbar)
    v_star = np.nanmax(rt * centered, axis=1)
    rc_p = float((v_star >= v).mean())
    # SPA (studentized), consistent / lower / upper recentering
    tstat = rt * dbar / omega
    t_spa = max(float(np.nanmax(tstat)), 0.0)
    thresh = -math.sqrt(2 * math.log(math.log(T))) if T > 15 else -1.0
    out = {}
    for name, g in (("consistent", np.where(tstat >= thresh, dbar, 0.0)), ("lower", np.maximum(dbar, 0.0)), ("upper", dbar)):
        z = rt * (boot - g) / omega
        stat = np.maximum(np.nanmax(z, axis=1), 0.0)
        out[name] = float((stat >= t_spa).mean())
    z_ind = rt * centered / omega
    p_ind = ((z_ind >= tstat[None, :]).sum(axis=0) + 1) / (B + 1)
    p_ind = np.where(np.isfinite(tstat), p_ind, 1.0)      # a zero-variance difference carries no evidence
    return {"realityCheckP": rc_p, "spaP": out["consistent"], "spaLowerP": out["lower"], "spaUpperP": out["upper"],
            "bestT": float(np.nanmax(tstat)), "sessions": T, "candidates": K, "bootstrap": B,
            "meanBlock": mean_block}, p_ind, tstat


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def bh(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, p[i] * m / (rank + 1))
        adj[i] = running
    return adj


# ====================================================================== selection (never sees holdout rows)
def select(pre_returns, eligible):
    """Rank variants on pre-holdout returns only. pre_returns: DataFrame (dates x account id)."""
    scores = {int(c): sharpe(pre_returns[c]) for c in pre_returns.columns if c in eligible}
    ranked = sorted((k for k, v in scores.items() if np.isfinite(v)), key=lambda k: (-scores[k], k))
    return ranked, scores


def walk_forward_select(pre_returns, eligible, train_min=WF_TRAIN_MIN, test_len=WF_TEST_LEN):
    """Folds inside the pre-holdout period: at each fold pick the best train Sharpe, use it on the next window."""
    n = len(pre_returns)
    folds = []
    for start in range(train_min, n, test_len):
        end = min(start + test_len, n)
        train = pre_returns.iloc[:start]
        test = pre_returns.iloc[start:end]
        ranked, scores = select(train, eligible)
        if not ranked:
            continue
        test_sr = {int(c): sharpe(test[c]) for c in test.columns if c in eligible}
        ids = [k for k in ranked if np.isfinite(test_sr.get(k, np.nan))]
        rank_corr = None
        if len(ids) > 3:
            rank_corr = float(pd.Series([scores[k] for k in ids]).rank().corr(pd.Series([test_sr[k] for k in ids]).rank()))
        folds.append({"trainStart": str(pre_returns.index[0].date()), "trainEnd": str(pre_returns.index[start - 1].date()),
                      "testStart": str(pre_returns.index[start].date()), "testEnd": str(pre_returns.index[end - 1].date()),
                      "chosen": ranked[0], "chosenTrainSharpe": scores[ranked[0]],
                      "chosenTestSharpe": test_sr.get(ranked[0]),
                      "medianTestSharpe": float(np.nanmedian(list(test_sr.values()))) if test_sr else None,
                      "trainTestRankCorr": rank_corr, "rows": (start, end)})
    return folds


# ====================================================================== orchestration
def load_design(n, seed, redesign, panel, orb_bundle):
    tax_ok, orb_ok = tax_available(), orb_bundle is not None
    if DESIGN_PATH.exists() and not redesign:
        design = json.loads(DESIGN_PATH.read_text())
        if design["n"] != n:
            raise SystemExit(f"design.json has n={design['n']}; rerun with --redesign to change it (the old holdout "
                             "view stays in holdout_ledger.json)")
        return design, False
    design = make_design(n, seed, tax_ok, orb_ok)
    dates = panel.dates
    eval_start = dates[WARMUP]
    eval_dates = dates[WARMUP:]
    holdout_start = eval_dates[int(len(eval_dates) * (1 - HOLDOUT_FRACTION))]
    design.update({"createdAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "codeVersion": CODE_VERSION,
                   "simStart": str(dates[WARMUP - 1].date()), "evalStart": str(eval_start.date()),
                   "holdoutStart": str(holdout_start.date()), "dataEnd": str(dates[-1].date()),
                   "holdoutFraction": HOLDOUT_FRACTION})
    if orb_ok:
        sessions = sorted(orb_bundle[0]["session_date"].unique())
        cut = int(len(sessions) * (1 - HOLDOUT_FRACTION))
        design.update({"orbFirst": sessions[0], "orbHoldoutStart": sessions[cut], "orbDataEnd": sessions[-1]})
    design["designHash"] = stable_hash({k: design[k] for k in ("accounts", "seed", "evalStart", "holdoutStart", "dataEnd")})
    if DESIGN_PATH.exists():
        old = json.loads(DESIGN_PATH.read_text())
        write_atomically(SWARM_DIR / f"design_{old.get('designHash', 'old')}.json", DESIGN_PATH.read_bytes())
    write_json(DESIGN_PATH, design)
    return design, True


def _bench_spy(panel, start, end):
    from costs import CostModel
    from market_data import MARKET
    from portfolio_sim import simulate
    from risk import CircuitBreaker
    weights = pd.DataFrame(0.0, index=panel.dates, columns=[MARKET])
    weights[MARKET] = 0.999
    small = subset_panel(panel, [MARKET])
    res = simulate(small, weights, None, CostModel.for_broker("schwab"), INITIAL_EQUITY, "next_open", start=start,
                   end=end, breaker=CircuitBreaker(DAILY_LOSS_LIMIT), whole_shares=True, min_order_value=MIN_ORDER_VALUE)
    return res.returns


def _stitched(panel, swing_results, folds, start, end):
    """Re-simulate the walk-forward choice per fold with the same realism, so switching costs are included."""
    from costs import CostModel
    from portfolio_sim import simulate
    from risk import CircuitBreaker
    if not folds:
        return None
    cols = sorted(set().union(*(set(r["weights"].columns) for r in swing_results.values() if r.get("ok"))))
    if not cols:
        return None
    small = subset_panel(panel, cols)
    dates = small.dates
    w = pd.DataFrame(0.0, index=dates, columns=small.close.columns)
    s = pd.DataFrame(np.nan, index=dates, columns=small.close.columns)
    for fold in folds:
        first, last = pd.Timestamp(fold["testStart"]), pd.Timestamp(fold["testEnd"])
        pos0, pos1 = dates.get_loc(first) - 1, dates.get_loc(last)
        rows = dates[pos0:pos1]          # decisions at these closes fill inside the test window
        chosen = swing_results[fold["chosen"]]
        w.loc[rows] = chosen["weights"].reindex(index=rows, columns=w.columns).fillna(0.0).to_numpy()
        if chosen.get("stops") is not None:
            s.loc[rows] = chosen["stops"].reindex(index=rows, columns=s.columns).to_numpy()
    first_test = pd.Timestamp(folds[0]["testStart"])
    res = simulate(small, w, s, CostModel.for_broker("schwab"), INITIAL_EQUITY, "next_open",
                   start=dates[dates.get_loc(first_test) - 1], end=end, breaker=CircuitBreaker(DAILY_LOSS_LIMIT),
                   whole_shares=True, min_order_value=MIN_ORDER_VALUE)
    return res.returns.loc[first_test:]


def _account_key(account, leg, fingerprint, extra=""):
    return stable_hash({"leg": leg, "cfg": account[leg], "data": fingerprint, "code": CODE_VERSION, "extra": extra})


def _leg_stats(returns, pre_mask, hold_mask, eligible, baseline_ids, bench_series, label):
    """Selection, DSR, CSCV, SPA and Holm/BH for one leg. returns: DataFrame dates x account id."""
    pre, hold = returns.loc[pre_mask], returns.loc[hold_mask]
    ranked, scores = select(pre, eligible)
    candidates = [c for c in returns.columns if c in eligible]
    pre_sr = np.array([scores.get(c, np.nan) for c in candidates])
    n_eff = effective_trials(pre[candidates].fillna(0).to_numpy())
    champion = ranked[0] if ranked else None
    out = {"leg": label, "eligible": len(candidates), "effectiveTrials": n_eff, "ranked": ranked,
           "preScores": scores}
    if champion is None:
        return out
    out["champion"] = champion
    out["deflatedSharpePre"] = deflated_sharpe(pre[champion].to_numpy(), len(candidates), pre_sr)
    out["deflatedSharpePreEffective"] = deflated_sharpe(pre[champion].to_numpy(), max(2, n_eff), pre_sr)
    out["pboPre"] = pbo_cscv(pre[candidates].fillna(0).to_numpy(), 12 if len(pre) >= 240 else 8)
    out["pboAllEval"] = pbo_cscv(returns[candidates].loc[pre_mask | hold_mask].fillna(0).to_numpy(),
                                 16 if len(returns) >= 320 else 8)
    from quant_tools import sharpe_stats
    hx = hold[champion].dropna().to_numpy()
    out["holdoutChampion"] = {"sharpe": sharpe(hx), "psrVsZero": sharpe_stats(hx, TRADING_DAYS)["probabilisticSharpe"]
                              if len(hx) > 10 else None}
    tests = {}
    for bname, bser in bench_series.items():
        if bser is None:
            continue
        comp = [c for c in candidates if c not in baseline_ids.get(bname, set())]
        for period, frame in (("holdout", hold), ("preHoldout", pre)):
            b = bser.reindex(frame.index).fillna(0.0).to_numpy()
            d = frame[comp].fillna(0.0).to_numpy() - b[:, None]
            if len(d) < 20:
                continue
            res, p_ind, tstat = spa_test(d, seed=11)
            res["holm"] = {str(c): float(v) for c, v in zip(comp, holm(p_ind))}
            res["bh"] = {str(c): float(v) for c, v in zip(comp, bh(p_ind))}
            res["rawP"] = {str(c): float(v) for c, v in zip(comp, p_ind)}
            res["significantHolm"] = [int(c) for c, v in zip(comp, holm(p_ind)) if v < 0.05]
            res["significantBH"] = [int(c) for c, v in zip(comp, bh(p_ind)) if v < 0.05]
            res["benchmarkSharpe"] = sharpe(b)
            # the pre-registered champion vs this benchmark on the holdout (single test, no correction needed)
            if period == "holdout" and champion in comp:
                j = comp.index(champion)
                res["championP"] = float(p_ind[j])
            tests[f"{bname}:{period}"] = res
    out["tests"] = tests
    return out


def run(n=DEFAULT_N, seed=DEFAULT_SEED, workers=None, redesign=False, with_orb=True):
    started = time.time()
    SWARM_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    panel, panel_path = load_panel_cached()
    t_panel = time.time() - started
    orb_bundle, orb_path = (load_orb_bundle(panel) if with_orb else None), None
    if orb_bundle is not None:
        orb_path = str(CACHE_DIR / "orb_bundle.pkl")
        with open(orb_path, "wb") as fh:
            pickle.dump(orb_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    design, new = load_design(n, seed, redesign, panel, orb_bundle)
    accounts = design["accounts"]
    fingerprint = data_fingerprint()
    sim_start, eval_start = pd.Timestamp(design["simStart"]), pd.Timestamp(design["evalStart"])
    holdout_start, data_end = pd.Timestamp(design["holdoutStart"]), pd.Timestamp(design["dataEnd"])
    workers = workers or default_workers()
    log(f"design {design['designHash']} ({'new' if new else 'reused'}): {len(accounts)} accounts, holdout from "
        f"{design['holdoutStart']}, {workers} workers")

    # ---- swing leg (sorted by regime key so a worker reuses its HMM fit)
    order = sorted(accounts, key=lambda a: (str(a["swing"].get("regime")), str(a["swing"].get("regime_threshold")), a["id"]))
    tasks = [{"account": a, "start": str(sim_start.date()), "cacheKey": _account_key(a, "swing", fingerprint,
              str(data_end.date()))} for a in order]
    t0 = time.time()
    swing_list = run_tasks(run_swing_account, tasks, workers, panel_path, None)
    t_swing = time.time() - t0
    swing = {r["id"]: r for r in swing_list}
    failed = {i: r["error"] for i, r in swing.items() if not r.get("ok")}
    ok_ids = sorted(i for i, r in swing.items() if r.get("ok"))
    rets = pd.DataFrame({i: swing[i]["returns"] for i in ok_ids}).loc[eval_start:data_end]
    pre_mask = rets.index < holdout_start
    hold_mask = rets.index >= holdout_start
    eligible = {i for i in ok_ids if not swing[i]["pdt"]["breach"]}
    spy = _bench_spy(panel, sim_start, data_end).reindex(rets.index).fillna(0.0)
    benches = {"tsmom_baseline": rets[1] if 1 in rets else None, "spy": spy}
    swing_stats = _leg_stats(rets, pre_mask, hold_mask, eligible, {"tsmom_baseline": {1}}, benches, "swing")
    folds = walk_forward_select(rets.loc[pre_mask], eligible)
    stitched = _stitched(panel, swing, folds, sim_start, holdout_start - pd.Timedelta(days=1))
    for f in folds:
        f.pop("rows", None)
    swing_stats["walkForward"] = {"folds": folds, "stitchedOos": perf(stitched) if stitched is not None else None,
                                  "distinctChoices": len({f["chosen"] for f in folds})}

    # ---- day-trade leg
    orb_stats, orb_rets, t_orb = None, None, 0.0
    if orb_bundle is not None and design.get("orbAvailable"):
        t0 = time.time()
        sessions = sorted(orb_bundle[0]["session_date"].unique())
        sessions = [s for s in sessions if s <= design["orbDataEnd"]]
        otasks = [{"account": a, "sessions": sessions, "cacheKey": _account_key(a, "orb", fingerprint,
                   design["orbDataEnd"])} for a in accounts]
        orb_list = run_tasks(run_orb_account, otasks, workers, panel_path, orb_path)
        t_orb = time.time() - t0
        orb = {r["id"]: r for r in orb_list}
        oids = sorted(i for i, r in orb.items() if r.get("ok"))
        failed.update({f"orb:{i}": r["error"] for i, r in orb.items() if not r.get("ok")})
        orb_rets = pd.DataFrame({i: orb[i]["returns"] for i in oids}).sort_index()
        oh = pd.Timestamp(design["orbHoldoutStart"])
        zero = pd.Series(0.0, index=orb_rets.index)
        # accounts 0 and 1 share the default ORB leg; keep one copy so the cross-section has no duplicate
        orb_stats = _leg_stats(orb_rets, orb_rets.index < oh, orb_rets.index >= oh, set(oids) - {1},
                               {"orb_default": {0, 1}}, {"orb_default": orb_rets[0] if 0 in orb_rets else None,
                                                         "cash": zero}, "daytrade")
        orb_stats["trades"] = {i: orb[i]["trades"] for i in oids}

    # ---- leaderboard
    rows = []
    pre, hold = rets.loc[pre_mask], rets.loc[hold_mask]
    wf_rows = pre.iloc[WF_TRAIN_MIN:] if len(pre) > WF_TRAIN_MIN else pre
    tb = swing_stats.get("tests", {}).get("tsmom_baseline:holdout", {})
    ts = swing_stats.get("tests", {}).get("spy:holdout", {})
    rank_of = {k: r + 1 for r, k in enumerate(swing_stats.get("ranked", []))}
    for a in accounts:
        i = a["id"]
        r = swing.get(i, {})
        row = {"id": i, "label": a["label"], "swing": a["swing"], "orb": a.get("orb"), "ok": bool(r.get("ok")),
               "error": r.get("error")}
        if r.get("ok"):
            at = r.get("afterTaxEquity")
            row.update({"selectionRank": rank_of.get(i), "eligible": i in eligible, "pdt": r["pdt"],
                        "pre": perf(pre[i], at), "wfOos": perf(wf_rows[i], at), "holdout": perf(hold[i], at),
                        "turnoverAnnual": r["turnover"], "avgGross": r["gross"], "trades": r["trades"],
                        "costDollars": r["costs"],
                        "holdoutVsTsmom": {"p": tb.get("rawP", {}).get(str(i)), "holm": tb.get("holm", {}).get(str(i)),
                                           "bh": tb.get("bh", {}).get(str(i))},
                        "holdoutVsSpy": {"p": ts.get("rawP", {}).get(str(i)), "holm": ts.get("holm", {}).get(str(i)),
                                         "bh": ts.get("bh", {}).get(str(i))}})
        if orb_rets is not None and i in orb_rets:
            oh = pd.Timestamp(design["orbHoldoutStart"])
            row["orbPre"] = perf(orb_rets[i].loc[orb_rets.index < oh])
            row["orbHoldout"] = perf(orb_rets[i].loc[orb_rets.index >= oh])
            row["orbTrades"] = orb_stats["trades"].get(i)
        rows.append(row)

    hold_sr = np.array([sharpe(hold[i]) for i in ok_ids if i in eligible])
    pre_sr = np.array([sharpe(pre[i]) for i in ok_ids if i in eligible])
    distribution = {
        "holdoutSharpe": _quantiles(hold_sr), "preSharpe": _quantiles(pre_sr),
        "holdoutSharpeHistogram": _hist(hold_sr), "preSharpeHistogram": _hist(pre_sr),
        "holdoutTotalReturn": _quantiles(np.array([(1 + hold[i]).prod() - 1 for i in ok_ids if i in eligible])),
        "shareBeatingTsmomHoldout": float(np.mean(hold_sr > sharpe(hold[1]))) if 1 in hold else None,
        "shareBeatingSpyHoldout": float(np.mean(hold_sr > sharpe(spy.loc[hold_mask]))),
        "preVsHoldoutRankCorr": float(pd.Series(pre_sr).rank().corr(pd.Series(hold_sr).rank())) if len(hold_sr) > 3 else None,
    }
    benchmarks = {"spy": {"pre": perf(spy.loc[pre_mask]), "holdout": perf(spy.loc[hold_mask])},
                  "tsmom_baseline": {"pre": perf(pre[1]), "holdout": perf(hold[1])} if 1 in rets else None,
                  "production": {"pre": perf(pre[0]), "holdout": perf(hold[0])} if 0 in rets else None}
    ledger = _record_holdout_view(design, swing_stats)
    verdict = _verdict(swing_stats, orb_stats, benchmarks)
    status = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "designHash": design["designHash"],
              "accounts": len(accounts), "initialEquity": INITIAL_EQUITY,
              "design": {k: design[k] for k in ("n", "seed", "method", "simStart", "evalStart", "holdoutStart",
                                                 "dataEnd", "holdoutFraction", "taxAvailable", "orbAvailable")},
              "orbWindow": {k: design.get(k) for k in ("orbFirst", "orbHoldoutStart", "orbDataEnd")},
              "realism": {"costs": "Schwab: $0 commission + SEC + FINRA TAF + half-spread + sqrt impact",
                          "wholeShares": True, "minOrderValue": MIN_ORDER_VALUE, "grossCap": 1.0,
                          "perNameCap": PER_NAME_CAP, "dailyLossBreaker": DAILY_LOSS_LIMIT,
                          "pdt": f"max {PDT_MAX} day trades / {PDT_WINDOW} sessions (enforced intraday; swing "
                                 "breaches disqualify)", "execution": "next open", "taxLots": tax_available()},
              "swing": _strip(swing_stats), "daytrade": _strip(orb_stats) if orb_stats else None,
              "benchmarks": benchmarks, "distribution": distribution, "leaderboard": rows,
              "holdoutViews": ledger["views"], "failed": failed, "verdict": verdict,
              "recommendation": verdict["recommendation"], "autoPromote": False,
              "runtime": {"panelSeconds": round(t_panel, 1), "swingSeconds": round(t_swing, 1),
                          "orbSeconds": round(t_orb, 1), "totalSeconds": round(time.time() - started, 1),
                          "workers": workers, "perAccountSeconds": _quantiles(np.array([r.get("seconds", 0) for r in swing_list]))}}
    warnings = list(getattr(panel, "warnings", None) or [])
    if warnings:
        status["dataWarnings"] = warnings
        status["usedLongHistory"] = bool(getattr(panel, "used_long_history", False))
    status["forward"] = _forward_summary()
    if STATUS_PATH.exists():                      # keep results written by follow-up studies
        try:
            previous = json.loads(STATUS_PATH.read_text())
            for key in ("growthStudy", "realPaperAccount"):
                if key in previous:
                    status[key] = previous[key]
        except ValueError:
            pass
    status["promptSummary"] = prompt_summary(status)
    write_json(STATUS_PATH, status)
    _frame_csv(rets, SWARM_DIR / "returns_swing.csv")
    if orb_rets is not None:
        _frame_csv(orb_rets, SWARM_DIR / "returns_daytrade.csv")
    log(f"done in {status['runtime']['totalSeconds']}s -> {STATUS_PATH}")
    return status


def _frame_csv(frame, path):
    write_atomically(path, frame.to_csv().encode())


def _quantiles(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    q = np.quantile(x, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    return dict(zip(["min", "p10", "p25", "median", "p75", "p90", "max"], q.tolist())) | {"mean": float(x.mean()), "n": int(len(x))}


def _hist(x, bins=12):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    counts, edges = np.histogram(x, bins=bins)
    return {"counts": counts.tolist(), "edges": edges.tolist()}


def _strip(stats):
    if not stats:
        return stats
    out = dict(stats)
    out.pop("preScores", None)
    tests = {}
    for key, t in (stats.get("tests") or {}).items():
        tests[key] = {k: v for k, v in t.items() if k not in ("rawP", "holm", "bh")}
    out["tests"] = tests
    return out


def _record_holdout_view(design, stats, study="swarm", design_hash=None):
    """Every holdout evaluation is appended. A distinct design or study on the same holdout window is another look."""
    ledger = json.loads(LEDGER_PATH.read_text()) if LEDGER_PATH.exists() else {"views": 0, "entries": []}
    design_hash = design_hash or design["designHash"]
    entry = {"at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "designHash": design_hash, "study": study,
             "holdoutStart": design["holdoutStart"], "dataEnd": design["dataEnd"], "champion": stats.get("champion")}
    same = [e for e in ledger["entries"] if e["designHash"] == design_hash]
    # Re-running the same frozen design reproduces the same numbers (no new information is used for selection),
    # so only distinct designs that touched the same holdout window count as extra looks.
    ledger["entries"].append(entry)
    ledger["views"] = len({e["designHash"] for e in ledger["entries"] if e["holdoutStart"] == design["holdoutStart"]})
    ledger["runsOfThisDesign"] = len(same) + 1
    write_json(LEDGER_PATH, ledger)
    return ledger


def _verdict(sw, orb, bench):
    lines, beats = [], False
    if sw and sw.get("champion") is not None:
        c = sw["champion"]
        t_tb = sw["tests"].get("tsmom_baseline:holdout", {})
        t_spy = sw["tests"].get("spy:holdout", {})
        holm_any = bool(t_tb.get("significantHolm")) or bool(t_spy.get("significantHolm"))
        spa_ok = (t_tb.get("spaP", 1) < 0.05) or (t_spy.get("spaP", 1) < 0.05)
        dsr = (sw.get("deflatedSharpePre") or {}).get("deflatedSharpeProbability")
        pbo = (sw.get("pboPre") or {}).get("pbo")
        beats = bool(holm_any and spa_ok and (dsr or 0) > 0.95 and (pbo is not None and pbo < 0.5)
                     and (t_tb.get("championP", 1) < 0.05))
        lines.append(f"swing champion acct {c}: holdout Sharpe {sw['holdoutChampion']['sharpe']:.2f}; "
                     f"SPA p vs TSMOM {t_tb.get('spaP')}, vs SPY {t_spy.get('spaP')}; DSR prob {dsr}; PBO {pbo}")
    if beats:
        rec = (f"Swing account {sw['champion']} passed every gate (DSR > 0.95, PBO < 0.5, SPA and Holm vs the "
               "benchmarks, pre-registered champion beats TSMOM on the holdout). Recommendation only: consider "
               "paper-trading its config in a second paper account; the real paper account is unchanged.")
    else:
        rec = ("Nothing beats the TSMOM baseline after multiple-testing correction. Keep the real paper account on "
               "its current config; let the forward shadow records accumulate before reconsidering.")
    return {"anyBeatsBaselineAfterCorrection": beats, "notes": lines, "recommendation": rec}


def prompt_summary(status):
    sw, dt = status.get("swing") or {}, status.get("daytrade") or {}
    b = status.get("benchmarks") or {}
    lb = {r["id"]: r for r in status.get("leaderboard", [])}
    out = [f"[SWARM] {status['accounts']} simulated $10k shadow accounts (no orders), design {status['designHash']}; "
           f"selection on pre-holdout data only, sealed holdout {status['design']['holdoutStart']}..{status['design']['dataEnd']}."]
    if sw.get("champion") is not None:
        c = lb.get(sw["champion"], {})
        h = c.get("holdout", {})
        out.append(f"Swing champion acct {sw['champion']} ({_short(c.get('swing'))}): pre Sharpe "
                   f"{_f(c.get('pre', {}).get('sharpe'))}, holdout Sharpe {_f(h.get('sharpe'))}, holdout return "
                   f"{_pct(h.get('totalReturn'))}, maxDD {_pct(h.get('maxDrawdown'))}.")
    for name in ("tsmom_baseline", "spy", "production"):
        if b.get(name):
            out.append(f"{name} holdout Sharpe {_f(b[name]['holdout'].get('sharpe'))}, return {_pct(b[name]['holdout'].get('totalReturn'))}.")
    dsr = sw.get("deflatedSharpePre") or {}
    out.append(f"Deflated Sharpe prob (pre, {dsr.get('trials')} trials) {_f(dsr.get('deflatedSharpeProbability'))}; "
               f"effective trials {_f(sw.get('effectiveTrials'))}; PBO (CSCV) {_f((sw.get('pboPre') or {}).get('pbo'))}.")
    for key in ("tsmom_baseline:holdout", "spy:holdout"):
        t = (sw.get("tests") or {}).get(key)
        if t:
            out.append(f"{key}: Reality Check p {_f(t.get('realityCheckP'))}, SPA p {_f(t.get('spaP'))}, Holm-significant "
                       f"{len(t.get('significantHolm', []))}, BH-significant {len(t.get('significantBH', []))}.")
    if dt.get("champion") is not None:
        out.append(f"Day-trade champion acct {dt['champion']}: holdout Sharpe {_f((dt.get('holdoutChampion') or {}).get('sharpe'))}; "
                   f"PBO {_f((dt.get('pboPre') or {}).get('pbo'))}; SPA p vs default ORB "
                   f"{_f(((dt.get('tests') or {}).get('orb_default:holdout') or {}).get('spaP'))}.")
    out.append(f"Verdict: {status['verdict']['recommendation']} Promotion is never automatic.")
    gs = status.get("growthStudy")
    if gs:
        out.append(f"Growth-adjusted mean reversion study: {gs.get('summary')}")
    ps = status.get("probStudy")
    if ps:
        out.append(f"Probability walk-forward study: {ps.get('summary')}")
    if status.get("realPaperAccount"):
        out.append(f"Real paper account: {status['realPaperAccount']}.")
    fw = status.get("forward") or {}
    if fw.get("sessions"):
        out.append(f"Forward shadow record: {fw['sessions']} sessions since {fw.get('start')}; leader acct {fw.get('leader')}.")
    return " ".join(out)


def _short(swing):
    if not swing:
        return "?"
    if swing.get("family") != "variant":
        return swing.get("family")
    s = f"TSMOM {swing['horizons']}, vol {swing['target_vol']}, max {swing['max_positions']}, {swing['rebalance']}, band {swing['band']}, regime {swing['regime']}"
    if swing.get("overlay"):
        s += f", overlay z{swing['mr_entry_z']}/{swing['mr_lookback']}d"
    if swing.get("trail_atr"):
        s += f", trail {swing['trail_atr']}ATR"
    if swing.get("tax_aware"):
        s += ", tax-aware"
    return s


def _f(x):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.2f}" if isinstance(x, float) else str(x)


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


# ====================================================================== forward (daily shadow mode)
def _forward_summary():
    path = FORWARD_DIR / "forward_status.json"
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text())
        return {k: doc.get(k) for k in ("start", "end", "sessions", "leader", "updatedAt", "status")}
    except Exception:
        return None


def forward(workers=None):
    """Advance every account through the sessions after the design end (shadow paper; never any orders)."""
    started = time.time()
    if not DESIGN_PATH.exists():
        log("no design yet; run `python swarm.py run` first")
        out = {"status": "no_design"}
        print(json.dumps(out))
        return out
    design = json.loads(DESIGN_PATH.read_text())
    FORWARD_DIR.mkdir(parents=True, exist_ok=True)
    panel, panel_path = load_panel_cached()
    design_end = pd.Timestamp(design["dataEnd"])
    new_dates = panel.dates[panel.dates > design_end]
    status = {"updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "designHash": design["designHash"],
              "start": str(new_dates[0].date()) if len(new_dates) else None, "end": str(panel.dates[-1].date()),
              "sessions": int(len(new_dates))}
    if not len(new_dates):
        status.update({"status": "awaiting", "message": f"no sessions after the design end {design['dataEnd']} yet"})
        try:
            from regime_allocation import attach_forward_shadows
            status["regimeAllocation"] = attach_forward_shadows(panel, design, new_dates)
        except Exception as error:  # never break the daily paper pipeline
            log(f"regime allocation forward shadows failed: {type(error).__name__}: {error}")
            status["regimeAllocation"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        write_json(FORWARD_DIR / "forward_status.json", status)
        print(json.dumps(status))
        return status
    workers = workers or default_workers()
    fingerprint = data_fingerprint()
    # forwardExtras: variants added after the design was frozen (e.g. growth-adjusted mean reversion); they do not
    # change the design hash or the holdout study, they only collect live shadow records.
    forward_accounts = design["accounts"] + list(design.get("forwardExtras", []))
    labels = {a["id"]: a.get("label") for a in forward_accounts}
    tasks = [{"account": a, "start": design["dataEnd"],
              "cacheKey": _account_key(a, "swing", fingerprint, "forward:" + design["dataEnd"])} for a in forward_accounts]
    results = run_tasks(run_swing_account, tasks, workers, panel_path, None)
    equity = pd.DataFrame({r["id"]: r["equity"] for r in results if r.get("ok")}).loc[new_dates[0]:]
    _frame_csv(equity, FORWARD_DIR / "forward_equity_swing.csv")
    revisions = _append_ledger(equity, "swing")
    # day-trade leg on sessions after the ORB design end
    orb_summary = None
    if design.get("orbAvailable"):
        bundle = load_orb_bundle(panel)
        if bundle is not None:
            sessions = [s for s in sorted(bundle[0]["session_date"].unique()) if s > design["orbDataEnd"]]
            if sessions:
                orb_path = str(CACHE_DIR / "orb_bundle.pkl")
                with open(orb_path, "wb") as fh:
                    pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
                otasks = [{"account": a, "sessions": sessions, "cacheKey": _account_key(a, "orb", fingerprint,
                           "forward:" + sessions[-1])} for a in design["accounts"]]
                orb_res = run_tasks(run_orb_account, otasks, workers, panel_path, orb_path)
                orb_eq = pd.DataFrame({r["id"]: INITIAL_EQUITY * (1 + r["returns"]).cumprod() for r in orb_res if r.get("ok")})
                _frame_csv(orb_eq, FORWARD_DIR / "forward_equity_daytrade.csv")
                revisions += _append_ledger(orb_eq, "daytrade")
                orb_summary = {"sessions": len(sessions), "leader": int(orb_eq.iloc[-1].idxmax()) if len(orb_eq) else None}
    rets = equity.pct_change().fillna(equity.iloc[0] / INITIAL_EQUITY - 1)
    last = equity.iloc[-1]
    table = [{"id": int(i), "label": labels.get(int(i)), "equity": float(last[i]), "totalReturn": float(last[i] / INITIAL_EQUITY - 1),
              "sharpe": sharpe(rets[i]) if len(rets) >= 20 else None} for i in equity.columns]
    table.sort(key=lambda r: -r["totalReturn"])
    status.update({"status": "tracking", "leader": table[0]["id"] if table else None, "accounts": table,
                   "daytrade": orb_summary, "revisions": revisions,
                   "note": "Shadow replay from $10,000 at the first post-design session; every strategy is causal, so "
                           "this equals advancing one session per day. Statistics are shown once 20+ sessions exist.",
                   "seconds": round(time.time() - started, 1)})
    if STATUS_PATH.exists() and len(rets) >= 20:
        sel = json.loads(STATUS_PATH.read_text()).get("swing", {}).get("ranked", [])
        fwd = {r["id"]: r["sharpe"] for r in table}
        ids = [i for i in sel if fwd.get(i) is not None]
        if len(ids) > 3:
            status["selectionVsForwardRankCorr"] = float(pd.Series(range(len(ids))).rank().corr(
                pd.Series([-fwd[i] for i in ids]).rank()))

    try:
        from regime_allocation import attach_forward_shadows
        status["regimeAllocation"] = attach_forward_shadows(panel, design, new_dates)
    except Exception as error:  # never break the daily paper pipeline
        log(f"regime allocation forward shadows failed: {type(error).__name__}: {error}")
        status["regimeAllocation"] = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    write_json(FORWARD_DIR / "forward_status.json", status)
    if STATUS_PATH.exists():
        doc = json.loads(STATUS_PATH.read_text())
        doc["forward"] = _forward_summary()
        doc["promptSummary"] = prompt_summary(doc)
        write_json(STATUS_PATH, doc)
    print(json.dumps({k: status.get(k) for k in ("status", "start", "end", "sessions", "leader", "seconds")}))
    return status


def _append_ledger(equity, leg):
    """Append-only daily rows; a recomputed earlier value that changed (data revision) is counted, not rewritten."""
    path = FORWARD_DIR / "ledger.jsonl"
    seen = {}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("leg") == leg:
                seen[(row["date"], row["id"])] = row["equity"]
    new_lines, revisions = [], 0
    for day, values in equity.iterrows():
        d = str(pd.Timestamp(day).date())
        for i, v in values.items():
            key = (d, int(i))
            if key in seen:
                if abs(seen[key] - float(v)) > 0.01:
                    revisions += 1
                continue
            new_lines.append(json.dumps({"leg": leg, "date": d, "id": int(i), "equity": round(float(v), 2)}))
    if new_lines:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")
    return revisions


# ====================================================================== CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "forward", "status"])
    parser.add_argument("--n", type=int, default=int(os.environ.get("SWARM_N", DEFAULT_N)))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--redesign", action="store_true", help="draw a new design (counts as another holdout look)")
    parser.add_argument("--no-orb", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "run":
        status = run(args.n, args.seed, args.workers, args.redesign, not args.no_orb)
        print(json.dumps({"verdict": status["verdict"], "runtime": status["runtime"]}, default=_json_default))
    elif args.command == "forward":
        try:
            forward(args.workers)
        except Exception as error:   # noqa: BLE001 - never break the daily paper pipeline
            log(f"forward failed: {type(error).__name__}: {error}")
            print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}))
            return 1
    else:
        print(STATUS_PATH.read_text() if STATUS_PATH.exists() else json.dumps({"status": "no_results"}))
    return 0


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("swarm"):
        sys.exit(main() or 0)
