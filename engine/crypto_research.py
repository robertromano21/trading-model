"""Crypto backtests: trend, cross-sectional momentum, mean reversion, volatility targeting and dollar / macro
feedback overlays, with the same honesty protocol as the equity swarm (selection on pre-holdout data only).

Market conventions
  * 24/7: one bar per UTC calendar day (Coinbase daily candles), annualization 365. Macro / ETF inputs are
    re-indexed onto the calendar by their public-availability date (crypto_data.RELEASE_RULES) and carried over
    weekends and holidays, so a Saturday signal only sees what was public by Saturday.
  * Timing: weights are decided on the close of day t (00:00 UTC) using data through t and filled at that close;
    they earn day t+1's return. Fees: Schwab Crypto 0.75% of notional per buy and per sell (no spread per
    Schwab) plus 5 bp slippage; $10 minimum trade; long only, no leverage, uninvested cash earns 0 (the Schwab
    Crypto account pays no interest).
  * Survivorship: the universe is today's liquid majors that still trade on Coinbase. Coins that died (LUNA, FTT,
    ...) are absent, which flatters cross-sectional and equal-weight results. BTC/ETH-only variants are least affected.

Protocol (``run``)
  * Design (variant grid) is hashed. Pre-holdout = EVAL_START .. 2026-07-08. Sealed holdout 2026-07-09 ..
    2026-11-24 (same window as every other study); after 2026-11-24 = forward shadow only.
  * Champion = best pre-holdout Sharpe. Walk-forward inside pre-holdout: expanding train (>= 2y), 180-day tests,
    30-day embargo between train end and test start (purges overlap with the longest lookbacks' signal decay).
  * Deflated Sharpe (raw and eigenvalue-effective trials), PBO by CSCV, White RC / Hansen SPA vs BTC buy-and-hold,
    per-variant bootstrap p-values with Holm and BH. Dollar question: paired tests of every dollar/macro overlay
    against its identical non-overlay base (SPA over all pairs + Holm/BH).
  * Lead/lag study on pre-holdout data: cross-correlations BTC vs UUP by session lag, predictive regressions with
    Newey-West errors, dollar-shock event study with placebo dates.
  * The holdout is scored once per design and every look is appended to Massive/Research/swarm/holdout_ledger.json.

  python crypto_research.py run [--no-holdout]
Outputs: Massive/Research/crypto/crypto_backtest_report.{json,md}, crypto_variant_returns.csv
"""

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from crypto_data import (ALL_COINS, CRYPTO_DIR, SCHWAB_SET, SCHWAB_SPOT, COINS, load_drivers, load_spot, write_json)

PPY = 365
EVAL_START = "2017-01-01"
HOLDOUT_START = "2026-07-09"
HOLDOUT_END = "2026-11-24"
FEE = 0.0075
SLIP = 0.0005
MIN_TRADE = 10.0
INITIAL = 10_000.0
WF_TRAIN_MIN = 730
WF_TEST = 180
WF_EMBARGO = 30
BOOT = 2000
MEAN_BLOCK = 10
CODE_VERSION = "crypto-1"

UNIVERSES = {"btc": ["BTC"], "btc_eth": ["BTC", "ETH"], "schwab": list(SCHWAB_SET), "all": list(ALL_COINS)}
REPORT_JSON = CRYPTO_DIR / "crypto_backtest_report.json"
REPORT_MD = CRYPTO_DIR / "crypto_backtest_report.md"
RETURNS_CSV = CRYPTO_DIR / "crypto_variant_returns.csv"


def log(msg):
    print(f"[crypto_research] {msg}", file=sys.stderr, flush=True)


def stable_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ====================================================================== signals (all causal: row t uses data <= t)
def _available(close, min_hist):
    return close.notna().cumsum().ge(min_hist) & close.notna()


def _normalize(mask_or_weights):
    w = mask_or_weights.astype(float)
    s = w.sum(axis=1)
    return w.div(s.where(s > 0), axis=0).fillna(0.0)


def _base_alloc(close, avail, weighting):
    if weighting == "invvol":
        vol = np.log(close).diff().rolling(60, min_periods=40).std()
        inv = (1.0 / vol).where(avail & vol.gt(0))
        return _normalize(inv.fillna(0.0))
    return _normalize(avail)


def base_weights(spec, close, drivers):
    coins = [c for c in UNIVERSES[spec.get("universe", "btc")] if c in close.columns]
    px = close[coins]
    fam = spec["base"]
    if fam == "hold":
        avail = _available(px, 30)
        fixed = spec.get("weights")
        if fixed:
            w = pd.DataFrame({c: np.where(avail[c], fixed.get(c, 0.0), 0.0) for c in coins}, index=px.index)
            s = w.sum(axis=1)
            return w.div(s.where(s > 0), axis=0).fillna(0.0)
        return _normalize(avail)
    if fam == "tsmom":
        lbs = spec["lookbacks"]
        avail = _available(px, max(lbs) + 1)
        sig = sum((px / px.shift(L) - 1).gt(0).astype(float) for L in lbs) / len(lbs)
        return _base_alloc(px, avail, spec.get("weighting", "ew")) * sig.where(avail, 0.0)
    if fam == "sma":
        n = spec["n"]
        avail = _available(px, n)
        sig = px.gt(px.rolling(n).mean()).astype(float)
        return _base_alloc(px, avail, spec.get("weighting", "ew")) * sig.where(avail, 0.0)
    if fam == "xsmom":
        L, k = spec["lookback"], spec["k"]
        avail = _available(px, max(L + 1, 60))
        score = (px / px.shift(L) - 1).where(avail)
        rank = score.where(score > 0).rank(axis=1, ascending=False, method="first")
        return rank.le(k).astype(float) / k
    if fam == "meanrev":
        avail = _available(px, 201)
        dip = (px / px.shift(spec["window"]) - 1).lt(-spec["thr"])
        trend = px.gt(px.rolling(200).mean())
        entry = (dip & trend & avail).astype(float)
        active = entry.rolling(spec["hold"], min_periods=1).max().gt(0)
        return _normalize(avail) * active.astype(float)
    if fam == "dollar_regime":
        avail = _available(px, 30)
        uup = drivers.get("UUP")
        if uup is None:
            return _normalize(avail) * 0.0
        chg = uup / uup.shift(spec["lookback"]) - 1
        on = chg.lt(0).astype(float)
        return _normalize(avail).mul(on, axis=0)
    raise ValueError(f"unknown family {fam}")


def overlay_multiplier(ov, drivers, index):
    """Exposure multiplier in [0, 1] from dollar / macro drivers (already lagged to availability)."""
    one = pd.Series(1.0, index=index)
    t, cut = ov["type"], ov.get("cut", 0.5)

    def col(name):
        return drivers[name] if name in drivers.columns else None

    if t in ("dollar_trend", "dtwex_trend"):
        s = col("UUP" if t == "dollar_trend" else "DTWEXBGS")
        if s is None:
            return one
        bad = (s / s.shift(ov["lookback"]) - 1).gt(0)
        return one.where(~bad, cut)
    if t == "dollar_shock":
        s = col("UUP")
        if s is None:
            return one
        r7 = s / s.shift(7) - 1
        z = r7 / r7.rolling(365, min_periods=180).std()
        trig = z.gt(ov["z"]).astype(float)
        hit = trig.rolling(ov["days"], min_periods=1).max().fillna(0)
        return 1 - hit * (1 - cut)
    if t == "real_yield":
        s = col("DFII10")
        if s is None:
            return one
        bad = (s - s.shift(ov["lookback"])).gt(ov.get("thr", 0.0))
        return one.where(~bad, cut)
    if t == "liquidity":
        s = col("NETLIQ")
        if s is None:
            return one
        bad = (s / s.shift(ov["lookback"]) - 1).lt(0)
        return one.where(~bad, cut)
    if t == "composite":
        u, r, n = col("UUP"), col("DFII10"), col("NETLIQ")
        bad = pd.Series(0.0, index=index)
        if u is not None:
            bad += (u / u.shift(90) - 1).gt(0).astype(float)
        if r is not None:
            bad += (r - r.shift(60)).gt(0).astype(float)
        if n is not None:
            bad += (n / n.shift(91) - 1).lt(0).astype(float)
        floor = ov.get("floor", 0.25)
        return 1 - bad / 3 * (1 - floor)
    if t == "nowcast":
        s = col(ov["column"])
        if s is None:
            return one
        x = s if ov.get("mode") == "level" else s - s.shift(ov.get("lookback", 30))
        bad = x.gt(0) if ov.get("rising_is_bad", True) else x.lt(0)
        return one.where(~(bad & x.notna()), cut)
    raise ValueError(f"unknown overlay {t}")


def target_weights(spec, close, drivers):
    W = base_weights(spec, close, drivers).reindex(columns=close.columns, fill_value=0.0).fillna(0.0)
    tv = spec.get("voltarget")
    if tv:
        R = close.pct_change(fill_method=None).fillna(0.0)
        port = (W.shift(1).fillna(0.0) * R).sum(axis=1)
        vol = port.rolling(30, min_periods=20).std() * math.sqrt(PPY)
        scale = (tv / vol.where(vol > 0)).clip(upper=1.0).fillna(0.0)
        W = W.mul(scale, axis=0)
    for ov in spec.get("overlays", []):
        W = W.mul(overlay_multiplier(ov, drivers, close.index).reindex(close.index).fillna(1.0), axis=0)
    return W.clip(lower=0.0)


# ====================================================================== simulator
def simulate(close, W, start, end=None, fee=FEE, slip=SLIP, band=0.05, rebalance="daily", initial=INITIAL):
    """Path-dependent long-only simulator with a rebalance band. Returns (equity Series, stats dict)."""
    idx = close.index
    s = idx.searchsorted(pd.Timestamp(start))
    e = len(idx) if end is None else idx.searchsorted(pd.Timestamp(end), side="right")
    P = close.to_numpy(dtype=float)
    R = np.nan_to_num(P[1:] / P[:-1] - 1.0, nan=0.0, posinf=0.0, neginf=0.0)
    R = np.vstack([np.zeros((1, P.shape[1])), R])
    Wn = np.nan_to_num(W.reindex(index=idx, columns=close.columns).to_numpy(dtype=float), nan=0.0)
    tradable = np.isfinite(P)
    c = fee + slip
    vals = np.zeros(P.shape[1])
    cash = initial
    eq = np.empty(e - s)
    traded, fees = 0.0, 0.0
    weekly = rebalance == "weekly"
    for j, t in enumerate(range(s, e)):
        if j:
            vals *= 1.0 + R[t]
        total = cash + vals.sum()
        tgt_w = np.where(tradable[t], Wn[t], 0.0)
        cur_w = vals / total if total > 0 else np.zeros_like(vals)
        diff = tgt_w - cur_w
        exits = (tgt_w <= 1e-9) & (vals > 0)
        rebal_day = (not weekly) or idx[t].weekday() == 0 or j == 0
        mask = exits | (rebal_day & (np.abs(diff) > band)) | ((j == 0) & (tgt_w > 0))
        if mask.any():
            dv = np.where(mask, tgt_w * total * (1 - c) - vals, 0.0)
            dv[exits] = -vals[exits]
            dv[np.abs(dv) < MIN_TRADE] = 0.0
            dv[exits] = -vals[exits]
            cost = np.abs(dv).sum() * c
            vals += dv
            cash -= dv.sum() + cost
            traded += np.abs(dv).sum()
            fees += cost
        eq[j] = cash + vals.sum()
    equity = pd.Series(eq, index=idx[s:e])
    years = max(len(equity) / PPY, 1e-9)
    return equity, {"turnoverPerYear": traded / max(equity.mean(), 1e-9) / years,
                    "costPerYear": fees / max(equity.mean(), 1e-9) / years}


def returns_from_equity(equity, initial=INITIAL):
    r = equity.pct_change()
    r.iloc[0] = equity.iloc[0] / initial - 1
    return r


def run_spec(spec, close, drivers, start=EVAL_START, end=None, fee=FEE):
    W = target_weights(spec, close, drivers)
    eq, st = simulate(close, W, start, end, fee=fee, band=spec.get("band", 0.05),
                      rebalance=spec.get("rebalance", "daily"))
    return returns_from_equity(eq), st, W


# ====================================================================== statistics
def sharpe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd * math.sqrt(PPY)) if sd > 0 else float("nan")


def perf(r):
    r = pd.Series(r).dropna()
    if len(r) < 2:
        return {"days": int(len(r))}
    eq = (1 + r).cumprod()
    yrs = len(r) / PPY
    return {"days": int(len(r)), "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
            "totalReturn": float(eq.iloc[-1] - 1), "cagr": float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0,
            "vol": float(r.std(ddof=1) * math.sqrt(PPY)), "sharpe": sharpe(r),
            "maxDrawdown": float((eq / eq.cummax() - 1).min())}


def stationary_indices(T, B, mean_block, rng):
    p = 1.0 / mean_block
    idx = np.empty((B, T), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    for t in range(1, T):
        jump = rng.random(B) < p
        idx[:, t] = np.where(jump, rng.integers(0, T, B), (idx[:, t - 1] + 1) % T)
    return idx


def spa_test(d, B=BOOT, mean_block=MEAN_BLOCK, seed=7, chunk=20):
    """White RC + Hansen SPA (consistent) on d (T x K) excess returns; per-candidate one-sided p-values."""
    d = np.asarray(d, dtype=float)
    T, K = d.shape
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    boot = np.empty((B, K))
    for b0 in range(0, B, chunk):
        b = min(chunk, B - b0)
        idx = stationary_indices(T, b, mean_block, rng)
        boot[b0:b0 + b] = d[idx].mean(axis=1)
    centered = boot - dbar
    omega = np.sqrt(T) * centered.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    rt = math.sqrt(T)
    rc_p = float((np.nanmax(rt * centered, axis=1) >= np.nanmax(rt * dbar)).mean())
    tstat = rt * dbar / omega
    t_spa = max(float(np.nanmax(tstat)), 0.0)
    thresh = -math.sqrt(2 * math.log(math.log(T))) if T > 15 else -1.0
    g = np.where(tstat >= thresh, dbar, 0.0)
    z = rt * (boot - g) / omega
    spa_p = float((np.maximum(np.nanmax(z, axis=1), 0.0) >= t_spa).mean())
    p_ind = ((rt * centered / omega >= tstat[None, :]).sum(axis=0) + 1) / (B + 1)
    p_ind = np.where(np.isfinite(tstat), p_ind, 1.0)
    return {"realityCheckP": rc_p, "spaP": spa_p, "bestT": float(np.nanmax(tstat)), "days": T,
            "candidates": K, "bootstrap": B}, p_ind


def deflated(x, trials, trial_sharpes):
    from quant_tools import sharpe_stats
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    finite = np.asarray(trial_sharpes, dtype=float)
    finite = finite[np.isfinite(finite)]
    sd = float(np.std(finite / math.sqrt(PPY), ddof=1) * math.sqrt(PPY)) if len(finite) > 2 else None
    st = sharpe_stats(x, PPY, 0.0, max(2, int(round(trials))), sd)
    ds = st.get("deflatedSharpe", {}) or {}
    return {"sharpe": st.get("sharpeAnnual"), "psr": st.get("probabilisticSharpe"), "trials": ds.get("trials"),
            "expectedMaxSharpe": ds.get("expectedMaxSharpeAnnual"), "dsrProbability": ds.get("probability")}


def pbo(M, n_blocks=12):
    import swarm as sw
    out = sw.pbo_cscv(M, n_blocks)
    k = math.sqrt(PPY / 252)
    for key in ("medianIsBestSharpe", "medianOosOfIsBestSharpe"):
        if out.get(key) is not None:
            out[key] *= k
    out.pop("logitHistogram", None)
    return out


def holm(p):
    import swarm as sw
    return sw.holm(p)


def bh(p):
    import swarm as sw
    return sw.bh(p)


def effective_trials(M):
    import swarm as sw
    return sw.effective_trials(M)


def nw_ols(y, x, lags):
    """OLS y = a + b x with Newey-West (Bartlett) standard errors. Returns (b, t, n, r2)."""
    m = np.isfinite(y) & np.isfinite(x)
    y, x = np.asarray(y)[m], np.asarray(x)[m]
    n = len(y)
    if n < 30 or np.std(x) == 0:
        return float("nan"), float("nan"), n, float("nan")
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return float(beta[1]), float(beta[1] / math.sqrt(V[1, 1])) if V[1, 1] > 0 else float("nan"), n, float(r2)


def p_from_t(t):
    from scipy import stats
    return float(2 * stats.norm.sf(abs(t))) if np.isfinite(t) else 1.0


# ====================================================================== design
def _key(spec):
    return stable_hash({k: v for k, v in spec.items() if k not in ("label",)})


def describe(spec):
    b = spec["base"]
    u = spec.get("universe", "btc")
    if b == "hold":
        w = spec.get("weights")
        s = "buy&hold BTC" if u == "btc" else f"buy&hold {u}" + (f" {w}" if w else " EW")
    elif b == "tsmom":
        s = f"TSMOM {u} {'/'.join(map(str, spec['lookbacks']))}d {spec.get('weighting', 'ew')}"
    elif b == "sma":
        s = f"SMA{spec['n']} trend {u}"
    elif b == "xsmom":
        s = f"x-sec momentum {u} top{spec['k']} {spec['lookback']}d"
    elif b == "meanrev":
        s = f"dip-buy {u} {spec['window']}d<-{int(spec['thr'] * 100)}% in uptrend, hold {spec['hold']}d"
    elif b == "dollar_regime":
        s = f"BTC only when dollar (UUP) {spec['lookback']}d trend is down"
    else:
        s = b
    if spec.get("voltarget"):
        s += f", vol target {int(spec['voltarget'] * 100)}%"
    for ov in spec.get("overlays", []):
        t = ov["type"]
        if t in ("dollar_trend", "dtwex_trend"):
            s += f", {'UUP' if t == 'dollar_trend' else 'broad-dollar'} {ov['lookback']}d up -> x{ov['cut']}"
        elif t == "dollar_shock":
            s += f", dollar shock z>{ov['z']} -> x{ov['cut']} for {ov['days']}d"
        elif t == "real_yield":
            s += f", real yield {ov['lookback']}d up -> x{ov['cut']}"
        elif t == "liquidity":
            s += f", net liquidity {ov['lookback']}d down -> x{ov['cut']}"
        elif t == "composite":
            s += ", macro composite (dollar+real yield+liquidity)"
        elif t == "nowcast":
            what = ov["column"].replace("NC_", "")
            direction = "above last print" if ov.get("mode") == "level" else f"{ov.get('lookback', 30)}d rising"
            if not ov.get("rising_is_bad", True):
                direction = "below last print" if ov.get("mode") == "level" else f"{ov.get('lookback', 30)}d falling"
            s += f", nowcast {what} {direction} -> x{ov['cut']}"
    s += f", {spec.get('rebalance', 'daily')} band {spec.get('band', 0.05)}"
    return s


DOLLAR_OVERLAYS = [
    {"type": "dollar_trend", "lookback": 30, "cut": 0.0}, {"type": "dollar_trend", "lookback": 30, "cut": 0.5},
    {"type": "dollar_trend", "lookback": 90, "cut": 0.0}, {"type": "dollar_trend", "lookback": 90, "cut": 0.5},
    {"type": "dollar_shock", "z": 1.5, "days": 10, "cut": 0.0}, {"type": "dollar_shock", "z": 2.0, "days": 10, "cut": 0.0},
    {"type": "dtwex_trend", "lookback": 30, "cut": 0.5}, {"type": "dtwex_trend", "lookback": 90, "cut": 0.5},
    {"type": "real_yield", "lookback": 30, "cut": 0.5}, {"type": "real_yield", "lookback": 90, "cut": 0.5},
    {"type": "liquidity", "lookback": 28, "cut": 0.5}, {"type": "liquidity", "lookback": 91, "cut": 0.5},
    {"type": "composite", "floor": 0.25},
]


def nowcast_overlay(col):
    """Higher inflation / unemployment = risk-off; weaker payrolls / GDP = risk-off. *_VS_LAST columns are
    surprises vs the last official print (sign rule); level columns use their 30-day change."""
    good_when_rising = any(k in col for k in ("PAYROLL", "GDP"))
    ov = {"type": "nowcast", "column": col, "cut": 0.5, "rising_is_bad": not good_when_rising}
    if col.endswith("_VS_LAST"):
        ov["mode"] = "level"
    else:
        ov["lookback"] = 30
    return ov


def make_design(nowcast_columns=()):
    v = []
    add = v.append
    add({"base": "hold", "universe": "btc", "band": 0.05})                                        # benchmark
    add({"base": "hold", "universe": "btc_eth", "weights": {"BTC": 0.6, "ETH": 0.4}, "band": 0.10})
    add({"base": "hold", "universe": "schwab", "band": 0.10})
    add({"base": "hold", "universe": "all", "band": 0.10})
    for u in ("btc", "btc_eth", "schwab", "all"):
        for lbs in ([20], [50], [100], [200], [20, 60, 120]):
            for wt in (["ew"] if u == "btc" else ["ew", "invvol"]):
                for reb, band in (("daily", 0.05), ("weekly", 0.10)):
                    add({"base": "tsmom", "universe": u, "lookbacks": lbs, "weighting": wt, "rebalance": reb,
                         "band": band})
    for u in ("btc", "btc_eth", "schwab"):
        for n in (50, 100, 200):
            add({"base": "sma", "universe": u, "n": n, "band": 0.05})
    for u in ("schwab", "all"):
        for L in (14, 30, 60, 90):
            for k in (1, 2, 3):
                add({"base": "xsmom", "universe": u, "lookback": L, "k": k, "rebalance": "weekly", "band": 0.10})
    for u in ("btc", "btc_eth", "schwab"):
        for w in (3, 7):
            for thr in (0.10, 0.20):
                for hold in (5, 15):
                    add({"base": "meanrev", "universe": u, "window": w, "thr": thr, "hold": hold, "band": 0.05})
    for u in ("btc", "schwab"):
        for tv in (0.3, 0.5, 0.8):
            add({"base": "tsmom", "universe": u, "lookbacks": [100], "weighting": "ew", "voltarget": tv, "band": 0.10})
        for tv in (0.4, 0.6):
            add({"base": "hold", "universe": u, "voltarget": tv, "band": 0.10})
    for L in (30, 60, 120):
        add({"base": "dollar_regime", "universe": "btc", "lookback": L, "band": 0.05})
    bases = [{"base": "hold", "universe": "btc", "band": 0.05},
             {"base": "hold", "universe": "schwab", "band": 0.10},
             {"base": "tsmom", "universe": "btc", "lookbacks": [100], "weighting": "ew", "rebalance": "daily", "band": 0.05},
             {"base": "tsmom", "universe": "schwab", "lookbacks": [100], "weighting": "ew", "rebalance": "daily", "band": 0.05},
             {"base": "sma", "universe": "btc", "n": 200, "band": 0.05},
             {"base": "xsmom", "universe": "schwab", "lookback": 30, "k": 2, "rebalance": "weekly", "band": 0.10}]
    overlays = list(DOLLAR_OVERLAYS) + [nowcast_overlay(c) for c in nowcast_columns]
    for b in bases:
        for ov in overlays:
            s = copy.deepcopy(b)
            s["overlays"] = [ov]
            s["pairBase"] = _key(b)
            add(s)
    for i, s in enumerate(v):
        s["id"] = i
        s["label"] = describe(s)
    return v


# ====================================================================== nowcast hook (macro_nowcast.py, read only)
def load_nowcast(calendar):
    """Point-in-time nowcast features from engine/macro_nowcast.py if it exists. Never writes its files.

    Accepts a module function returning a DataFrame (load_history / history / load_point_in_time /
    nowcast_history) or a CSV under Massive/Research/macro/ named *nowcast*.csv. If the frame has an
    'available'/'asof'/'published' column, rows are keyed by it; otherwise the row date + 1 day is used."""
    from crypto_data import SHARED_MACRO_DIR
    frame = None
    try:
        import macro_nowcast as mn   # noqa: F401
        for name in ("load_point_in_time", "load_history", "history", "nowcast_history", "load_nowcast_history"):
            fn = getattr(mn, name, None)
            if callable(fn):
                try:
                    frame = fn()
                    break
                except TypeError:
                    continue
    except ImportError:
        pass
    if frame is None and SHARED_MACRO_DIR.exists():
        for p in sorted(SHARED_MACRO_DIR.glob("*nowcast*.csv")):
            try:
                frame = pd.read_csv(p)
                break
            except Exception:   # noqa: BLE001
                continue
    if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(index=calendar), "macro_nowcast.py output not found"
    f = frame.copy()
    key = next((c for c in f.columns if str(c).lower() in ("available", "asof", "as_of", "published", "vintage")), None)
    if key is None:
        dcol = next((c for c in f.columns if str(c).lower() in ("date", "observation_date")), None)
        if dcol is not None:
            f["_avail"] = pd.to_datetime(f[dcol]) + pd.Timedelta(days=1)
        elif isinstance(f.index, pd.DatetimeIndex):
            f["_avail"] = f.index + pd.Timedelta(days=1)
        else:
            return pd.DataFrame(index=calendar), "nowcast frame has no date column"
    else:
        f["_avail"] = pd.to_datetime(f[key]) + pd.Timedelta(days=1)
    f["_avail"] = pd.to_datetime(f["_avail"]).dt.tz_localize(None).dt.normalize()
    num = f.select_dtypes(include=[np.number]).copy()
    num["_avail"] = f["_avail"]
    num = num.groupby("_avail").last().sort_index()
    num.columns = [c if str(c).startswith("NC_") else f"NC_{c}" for c in num.columns]
    out = num.reindex(num.index.union(calendar)).ffill().reindex(calendar)
    return out, f"{len(num.columns)} nowcast columns, point-in-time"


# ====================================================================== lead / lag study
def lead_lag(close, drivers, pre_end):
    out = {}
    btc = close["BTC"]
    from crypto_data import load_schwab_close
    uup = load_schwab_close("UUP")
    spy = load_schwab_close("SPY")
    if uup is not None:
        uup = uup.loc[:pre_end]
        sess = uup.index[uup.index >= pd.Timestamp("2016-01-01")]
        b = np.log(btc.reindex(sess)).diff()
        u = np.log(uup.reindex(sess)).diff()
        rows = []
        for k in range(-10, 11):
            # k > 0: dollar move k sessions BEFORE the BTC move (dollar leads)
            beta, t, n, _ = nw_ols(b.values, u.shift(k).values, 5)
            rows.append({"lag": k, "corr": float(b.corr(u.shift(k))), "beta": beta, "t": t, "n": n})
        p = np.array([p_from_t(r["t"]) for r in rows])
        for r, h, q in zip(rows, holm(p), bh(p)):
            r["pHolm"], r["pBH"] = float(h), float(q)
        out["dailyCrossCorr"] = rows
        wk_b = np.log(btc.resample("W-FRI").last()).diff().loc["2016":pre_end]
        wk_u = np.log(uup.resample("W-FRI").last()).diff().loc["2016":pre_end]
        out["weeklyCrossCorr"] = [{"lagWeeks": k, "corr": float(wk_b.corr(wk_u.shift(k)))} for k in range(-4, 5)]
        years = {}
        for y in range(2016, pd.Timestamp(pre_end).year + 1):
            m = (b.index.year == y)
            if m.sum() > 60:
                years[str(y)] = {"corrUUP": float(b[m].corr(u[m])),
                                 "corrSPY": float(b[m].corr(np.log(spy.reindex(sess)).diff()[m])) if spy is not None else None}
        out["corrByYear"] = years
        # dollar-shock event study (UUP 5-session return z > 2 vs trailing 252 sessions), first in 10 sessions
        r5 = uup / uup.shift(5) - 1
        z = r5 / r5.rolling(252, min_periods=126).std()
        events = {}
        for sign, name in ((1, "dollarSurge"), (-1, "dollarDrop")):
            hit = (sign * z) > 2
            dates, last = [], None
            for d in hit[hit].index:
                if d < pd.Timestamp("2016-01-01"):
                    continue
                if last is None or (uup.index.get_loc(d) - uup.index.get_loc(last)) > 10:
                    dates.append(d)
                last = d
            res = {"n": len(dates), "dates": [str(d.date()) for d in dates]}
            rng = np.random.default_rng(11)
            pool = btc.loc["2016-01-01":pre_end].index
            for h in (1, 5, 10, 20):
                fwd = np.log(btc.shift(-h) / btc)
                ev = fwd.reindex(dates).dropna()
                allf = fwd.loc[pool].dropna()
                if len(ev) < 3:
                    continue
                draws = np.array([allf.sample(len(ev), random_state=int(rng.integers(1e9))).mean() for _ in range(2000)])
                res[f"h{h}"] = {"meanLogRet": float(ev.mean()), "unconditional": float(allf.mean()),
                                "hitRate": float((ev > 0).mean()), "placeboPLower": float((draws <= ev.mean()).mean()),
                                "placeboPUpper": float((draws >= ev.mean()).mean())}
            events[name] = res
        out["dollarShockEvents"] = events
    # predictive regressions on the calendar with availability-lagged drivers
    feats = {}
    d = drivers.loc[:pre_end]
    if "UUP" in d:
        feats["UUP 30d change"] = d["UUP"] / d["UUP"].shift(30) - 1
        feats["UUP 90d change"] = d["UUP"] / d["UUP"].shift(90) - 1
        feats["UUP 7d change"] = d["UUP"] / d["UUP"].shift(7) - 1
    if "DTWEXBGS" in d:
        feats["Broad dollar 30d change (lagged H.10)"] = d["DTWEXBGS"] / d["DTWEXBGS"].shift(30) - 1
    if "DFII10" in d:
        feats["10y real yield 30d change"] = d["DFII10"] - d["DFII10"].shift(30)
    if "NETLIQ" in d:
        feats["Net liquidity 28d change"] = d["NETLIQ"] / d["NETLIQ"].shift(28) - 1
        feats["Net liquidity 91d change"] = d["NETLIQ"] / d["NETLIQ"].shift(91) - 1
    if "M2SL" in d:
        feats["M2 YoY (lagged release)"] = d["M2SL"] / d["M2SL"].shift(365) - 1
    if "VIXCLS" in d:
        feats["VIX level"] = d["VIXCLS"]
    for c in [c for c in d.columns if str(c).startswith("NC_")]:
        feats[f"Nowcast {c[3:]} level"] = d[c]
        feats[f"Nowcast {c[3:]} 30d change"] = d[c] - d[c].shift(30)
    preds = []
    for coin in ("BTC", "ETH"):
        px = close[coin].loc[:pre_end]
        for h in (7, 30):
            y = np.log(px.shift(-h) / px)
            y = y.loc["2016-01-01":]
            y = y[y.index <= pd.Timestamp(pre_end) - pd.Timedelta(days=h)]   # target must end before the holdout
            for name, x in feats.items():
                xs = x.reindex(y.index)
                zx = (xs - xs.mean()) / xs.std()
                b, t, n, r2 = nw_ols(y.values, zx.values, h)
                preds.append({"coin": coin, "horizonDays": h, "feature": name, "betaPerSd": b, "t": t, "n": n, "r2": r2,
                              "p": p_from_t(t)})
    if preds:
        p = np.array([r["p"] for r in preds])
        for r, h, q in zip(preds, holm(p), bh(p)):
            r["pHolm"], r["pBH"] = float(h), float(q)
    out["predictiveRegressions"] = preds
    return out


# ====================================================================== walk-forward
def walk_forward(pre, eligible):
    n = len(pre)
    folds, pieces = [], []
    for start in range(WF_TRAIN_MIN + WF_EMBARGO, n, WF_TEST):
        end = min(start + WF_TEST, n)
        train = pre.iloc[:start - WF_EMBARGO]
        scores = {c: sharpe(train[c]) for c in eligible}
        best = max((c for c in scores if np.isfinite(scores[c])), key=lambda c: (scores[c], -c))
        test = pre.iloc[start:end]
        pieces.append(test[best])
        folds.append({"trainEnd": str(train.index[-1].date()), "testStart": str(test.index[0].date()),
                      "testEnd": str(test.index[-1].date()), "chosen": int(best), "trainSharpe": scores[best],
                      "testSharpe": sharpe(test[best]), "btcTestSharpe": sharpe(test[0])})
    stitched = pd.concat(pieces) if pieces else pd.Series(dtype=float)
    return folds, stitched


def select(pre, eligible):
    scores = {int(c): sharpe(pre[c]) for c in eligible}
    ranked = sorted((k for k, v in scores.items() if np.isfinite(v)), key=lambda k: (-scores[k], k))
    return ranked, scores


# ====================================================================== run
def run(look_holdout=True, workers=None):
    t0 = time.time()
    close = load_spot(ALL_COINS)
    drivers = load_drivers(close.index)
    nc, nc_note = load_nowcast(close.index)
    nowcast_cols = []
    if not nc.empty and len(nc.columns):
        drivers = drivers.join(nc)
        nowcast_cols = [c for c in nc.columns if nc[c].loc[:HOLDOUT_START].notna().sum() > 365]
    design = make_design(nowcast_cols)
    design_hash = "crypto-" + stable_hash({"v": CODE_VERSION, "design": [{k: v for k, v in s.items() if k != "label"}
                                                                         for s in design],
                                           "fee": FEE, "slip": SLIP, "eval": EVAL_START, "holdout": HOLDOUT_START})
    log(f"{len(design)} variants, design {design_hash}; data {close.index[0].date()}..{close.index[-1].date()}")
    rets, extra = {}, {}
    for s in design:
        r, st, _ = run_spec(s, close, drivers)
        rets[s["id"]] = r
        extra[s["id"]] = st
    R = pd.DataFrame(rets)
    R.to_csv(RETURNS_CSV, index_label="date")
    pre = R.loc[:pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)]
    hold = R.loc[HOLDOUT_START:HOLDOUT_END]
    ids = list(R.columns)
    ranked, scores = select(pre, ids)
    champion = ranked[0]
    M = pre.to_numpy()
    trial_sr = np.array([scores[i] for i in ids])
    eff = effective_trials(M)
    dsr = deflated(pre[champion].values, len(ids), trial_sr)
    dsr_eff = deflated(pre[champion].values, eff, trial_sr)
    pbo_res = pbo(M, 12)
    bench = 0
    cands = [i for i in ids if i != bench]
    d = pre[cands].to_numpy() - pre[[bench]].to_numpy()
    spa, p_ind = spa_test(d)
    ph, pb = holm(p_ind), bh(p_ind)
    folds, stitched = walk_forward(pre, ids)
    wf = {"folds": folds, "stitched": perf(stitched),
          "btcSameDays": perf(pre[0].reindex(stitched.index)) if len(stitched) else None}
    # dollar / macro overlay pairs
    key_to_id = {_key({k: v for k, v in s.items() if k not in ("id", "label")}): s["id"] for s in design
                 if not s.get("overlays")}
    pairs = []
    for s in design:
        if s.get("overlays") and s.get("pairBase") in key_to_id:
            pairs.append((s["id"], key_to_id[s["pairBase"]], s["overlays"][0]["type"]))
    pair_rows, pair_spa = [], None
    if pairs:
        D = np.column_stack([pre[a].values - pre[b].values for a, b, _ in pairs])
        pair_spa, pp = spa_test(D, seed=13)
        ph2, pb2 = holm(pp), bh(pp)
        for (a, b, t), p, h, q in zip(pairs, pp, ph2, pb2):
            pa, pbase = perf(pre[a]), perf(pre[b])
            pair_rows.append({"id": int(a), "base": int(b), "type": t, "label": design[a]["label"],
                              "sharpe": pa["sharpe"], "baseSharpe": pbase["sharpe"],
                              "sharpeDelta": pa["sharpe"] - pbase["sharpe"],
                              "mdd": pa["maxDrawdown"], "baseMdd": pbase["maxDrawdown"],
                              "cagr": pa["cagr"], "baseCagr": pbase["cagr"],
                              "p": float(p), "pHolm": float(h), "pBH": float(q)})
    nowcast_block = None
    nc_idx = [k for k, (_, _, t) in enumerate(pairs) if t == "nowcast"]
    if nc_idx:
        Dn = np.column_stack([pre[pairs[k][0]].values - pre[pairs[k][1]].values for k in nc_idx])
        nspa, npv = spa_test(Dn, seed=17)
        nh, nb = holm(npv), bh(npv)
        rows_n = []
        for k, p, h, q in zip(nc_idx, npv, nh, nb):
            r = dict(pair_rows[k])
            r.update({"pWithinNowcast": float(p), "pHolmWithinNowcast": float(h), "pBHWithinNowcast": float(q)})
            rows_n.append(r)
        nowcast_block = {"spa": nspa, "pairs": rows_n,
                         "improvedSharpe": int(sum(r["sharpeDelta"] > 0 for r in rows_n)),
                         "improvedMdd": int(sum(r["mdd"] > r["baseMdd"] for r in rows_n)),
                         "significantHolm": [r["id"] for r in rows_n if r["pHolmWithinNowcast"] < 0.05],
                         "significantBH": [r["id"] for r in rows_n if r["pBHWithinNowcast"] < 0.10],
                         "sampleStart": str(drivers[nowcast_cols].apply(lambda c: c.first_valid_index()).min().date())}
    families = {}
    for r in pair_rows:
        f = families.setdefault(r["type"], {"n": 0, "improvedSharpe": 0, "improvedMdd": 0, "deltas": []})
        f["n"] += 1
        f["improvedSharpe"] += int(r["sharpeDelta"] > 0)
        f["improvedMdd"] += int(r["mdd"] > r["baseMdd"])
        f["deltas"].append(r["sharpeDelta"])
    for f in families.values():
        f["medianSharpeDelta"] = float(np.median(f.pop("deltas")))
    # subperiod robustness for the champion, BTC hold and the best dollar overlay
    def sub(i):
        return {name: sharpe(pre[i].loc[a:b]) for name, (a, b) in
                {"2017-2019": ("2017", "2019"), "2020-2022": ("2020", "2022"), "2023-2026H1": ("2023", HOLDOUT_START)}.items()}
    variants = []
    for i in ids:
        row = {"id": int(i), "label": design[i]["label"], "family": design[i]["base"],
               "overlay": design[i]["overlays"][0]["type"] if design[i].get("overlays") else None,
               "pre": perf(pre[i]), **extra[i], "rank": ranked.index(i) + 1 if i in ranked else None}
        if i != bench:
            j = cands.index(i)
            row.update({"pVsBtc": float(p_ind[j]), "pHolm": float(ph[j]), "pBH": float(pb[j])})
        variants.append(row)
    cost_sens = {}
    for i in {champion, 0, ranked[1] if len(ranked) > 1 else champion}:
        r_low, _, _ = run_spec(design[i], close, drivers, fee=0.0025)
        cost_sens[str(i)] = {"fee0.75%": perf(pre[i]), "fee0.25%": perf(r_low.loc[:pre.index[-1]])}
    ll = lead_lag(close, drivers, str((pd.Timestamp(HOLDOUT_START) - pd.Timedelta(days=1)).date()))
    report = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "designHash": design_hash,
              "codeVersion": CODE_VERSION, "variants": len(ids), "evalStart": EVAL_START,
              "preHoldoutEnd": str(pre.index[-1].date()), "holdout": [HOLDOUT_START, HOLDOUT_END],
              "dataEnd": str(close.index[-1].date()), "fees": {"perSide": FEE, "slippage": SLIP, "minTrade": MIN_TRADE},
              "universe": {c: COINS[c]["schwab"] for c in ALL_COINS},
              "coinStart": {c: str(close[c].first_valid_index().date()) for c in close.columns},
              "drivers": {c: str(drivers[c].first_valid_index().date()) if drivers[c].notna().any() else None
                          for c in drivers.columns},
              "nowcast": {"note": nc_note, "columns": nowcast_cols},
              "champion": {"id": int(champion), "label": design[champion]["label"], "pre": perf(pre[champion]),
                           "subperiods": sub(champion)},
              "btcHold": {"pre": perf(pre[0]), "subperiods": sub(0)},
              "ranked": [int(i) for i in ranked[:20]],
              "deflatedSharpe": dsr, "deflatedSharpeEffective": {**dsr_eff, "effectiveTrials": eff},
              "pbo": pbo_res, "spaVsBtc": {**spa, "significantHolm": [int(cands[j]) for j in np.where(ph < 0.05)[0]],
                                           "significantBH": [int(cands[j]) for j in np.where(pb < 0.10)[0]]},
              "walkForward": wf, "dollarPairs": {"spa": pair_spa, "families": families, "pairs": pair_rows},
              "nowcastPairs": nowcast_block,
              "costSensitivity": cost_sens, "leadLag": ll, "variantTable": variants,
              "design": design}
    if look_holdout and len(hold) > 20:
        hv = {int(i): perf(hold[i]) for i in ids}
        best_pair = max(pair_rows, key=lambda r: (r["sharpe"] - r["baseSharpe"])) if pair_rows else None
        report["holdoutLook"] = {
            "days": int(len(hold)), "champion": hv[champion], "btcHold": hv[0],
            "walkForwardLastPick": hv[folds[-1]["chosen"]] if folds else None,
            "bestPreDollarPair": ({"id": best_pair["id"], "overlay": hv[best_pair["id"]], "base": hv[best_pair["base"]]}
                                  if best_pair else None),
            "dollarPairsHoldout": {"improvedSharpe": int(sum(
                1 for a, b, _ in pairs if (hv[a].get("sharpe") or -9) > (hv[b].get("sharpe") or -9))), "pairs": len(pairs)},
            "allVariants": hv}
        import swarm as sw
        ledger = sw._record_holdout_view({"holdoutStart": HOLDOUT_START, "dataEnd": HOLDOUT_END},
                                         {"champion": int(champion)}, study="crypto", design_hash=design_hash)
        report["holdoutLedger"] = {"views": ledger.get("views"), "runsOfThisDesign": ledger.get("runsOfThisDesign"),
                                   "entries": len(ledger.get("entries", []))}
    if not report.get("holdoutLook") and REPORT_JSON.exists():
        try:
            prev = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
            look = prev.get("holdoutLook") or (prev.get("previousHoldoutLook") or {}).get("look")
            if look:
                report["previousHoldoutLook"] = {
                    "designHash": prev.get("designHash") if prev.get("holdoutLook") else
                    prev["previousHoldoutLook"].get("designHash"),
                    "note": "Holdout scored once on this earlier design; not re-scored for the current design.",
                    "look": look, "ledger": prev.get("holdoutLedger") or prev["previousHoldoutLook"].get("ledger")}
        except Exception:   # noqa: BLE001
            pass
    report["seconds"] = round(time.time() - t0, 1)
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    return report


# ====================================================================== report
def _p(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{d}f}%"


def _f(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def markdown(rep):
    L = []
    a = L.append
    a(f"# Crypto backtest report ({rep['generatedAt']} ET)\n")
    a(f"Design `{rep['designHash']}`, {rep['variants']} variants, evaluation {rep['evalStart']} to "
      f"{rep['preHoldoutEnd']} (pre-holdout); sealed holdout {rep['holdout'][0]} to {rep['holdout'][1]}. "
      f"Fees: Schwab Crypto 0.75% per side + 5 bp slippage, $10 minimum trade, long-only spot, cash earns 0. "
      f"Daily UTC bars, 365-day annualization; macro inputs lagged to their publication dates.\n")
    a("## Data\n")
    a("| Coin | Coinbase history from | Schwab |\n|---|---|---|")
    for c, s in rep["coinStart"].items():
        a(f"| {c} | {s} | {rep['universe'][c]} |")
    a("\nSchwab trades spot BTC and ETH today (SOL/AVAX/LINK announced). Its market-data API has no spot pairs; it "
      "serves the $BTC/$ETH indexes, CME futures and spot ETPs. Survivorship: dead coins are missing, which flatters "
      "equal-weight and cross-sectional results; BTC/ETH variants are least affected.\n")
    a("## Headline\n")
    b, c = rep["btcHold"]["pre"], rep["champion"]["pre"]
    a("| | Sharpe | CAGR | Max DD | Vol |\n|---|---|---|---|---|")
    a(f"| BTC buy & hold | {_f(b['sharpe'])} | {_p(b['cagr'])} | {_p(b['maxDrawdown'])} | {_p(b['vol'])} |")
    a(f"| Champion #{rep['champion']['id']}: {rep['champion']['label']} | {_f(c['sharpe'])} | {_p(c['cagr'])} | "
      f"{_p(c['maxDrawdown'])} | {_p(c['vol'])} |")
    wf = rep["walkForward"]
    if wf.get("stitched") and wf["stitched"].get("sharpe") is not None:
        s, bb = wf["stitched"], wf["btcSameDays"]
        a(f"| Walk-forward picks (stitched OOS {s['start']}..{s['end']}) | {_f(s['sharpe'])} | {_p(s['cagr'])} | "
          f"{_p(s['maxDrawdown'])} | {_p(s['vol'])} |")
        a(f"| BTC hold, same OOS days | {_f(bb['sharpe'])} | {_p(bb['cagr'])} | {_p(bb['maxDrawdown'])} | {_p(bb['vol'])} |")
    d, de, pb, sp = rep["deflatedSharpe"], rep["deflatedSharpeEffective"], rep["pbo"], rep["spaVsBtc"]
    a(f"\nDeflated Sharpe probability {_f(d.get('dsrProbability'), 3)} ({d.get('trials')} trials), "
      f"{_f(de.get('dsrProbability'), 3)} with {_f(de.get('effectiveTrials'), 1)} effective trials. "
      f"PBO {_f(pb.get('pbo'), 3)}. SPA vs BTC hold p = {_f(sp['spaP'], 3)} (Reality Check {_f(sp['realityCheckP'], 3)}); "
      f"Holm-significant: {len(sp['significantHolm'])}, BH (10%): {len(sp['significantBH'])}.\n")
    a("Champion Sharpe by subperiod: " + ", ".join(f"{k} {_f(v)}" for k, v in rep["champion"]["subperiods"].items()) +
      "; BTC hold: " + ", ".join(f"{k} {_f(v)}" for k, v in rep["btcHold"]["subperiods"].items()) + "\n")
    a("## Best pre-holdout variants by family\n")
    a("| id | Variant | Sharpe | CAGR | Max DD | Turnover/yr | Fee drag/yr | p vs BTC (Holm) |\n|---|---|---|---|---|---|---|---|")
    seen = set()
    by = sorted(rep["variantTable"], key=lambda r: -(r["pre"].get("sharpe") or -9))
    for r in by:
        fam = (r["family"], r["overlay"])
        if fam in seen:
            continue
        seen.add(fam)
        p = r["pre"]
        a(f"| {r['id']} | {r['label']} | {_f(p.get('sharpe'))} | {_p(p.get('cagr'))} | {_p(p.get('maxDrawdown'))} | "
          f"{_f(r['turnoverPerYear'], 1)}x | {_p(r['costPerYear'])} | {_f(r.get('pHolm'), 3)} |")
    a("\n## Does dollar / macro feedback help?\n")
    dp = rep["dollarPairs"]
    if dp.get("spa"):
        a(f"Each overlay is compared with the identical strategy without it (paired daily return differences, "
          f"pre-holdout). SPA over all {dp['spa']['candidates']} pairs: p = {_f(dp['spa']['spaP'], 3)} "
          f"(H0: no overlay improves its base).\n")
    a("| Overlay family | Pairs | Sharpe improved | Drawdown improved | Median Sharpe change |\n|---|---|---|---|---|")
    for k, f in dp["families"].items():
        a(f"| {k} | {f['n']} | {f['improvedSharpe']} | {f['improvedMdd']} | {_f(f['medianSharpeDelta'])} |")
    a("\nTop pairs:\n")
    a("| id | Variant | Sharpe | Base Sharpe | Max DD vs base | p | Holm | BH |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(dp["pairs"], key=lambda r: r["p"])[:12]:
        a(f"| {r['id']} | {r['label']} | {_f(r['sharpe'])} | {_f(r['baseSharpe'])} | {_p(r['mdd'])} vs {_p(r['baseMdd'])} | "
          f"{_f(r['p'], 3)} | {_f(r['pHolm'], 3)} | {_f(r['pBH'], 3)} |")
    ll = rep["leadLag"]
    if ll.get("dailyCrossCorr"):
        a("\n## Lead / lag: BTC vs the dollar (UUP), pre-holdout\n")
        a("Positive lag = dollar move that many sessions BEFORE the BTC move.\n")
        a("| Lag | Corr | NW t | Holm p |\n|---|---|---|---|")
        for r in ll["dailyCrossCorr"]:
            if abs(r["lag"]) <= 5:
                a(f"| {r['lag']} | {_f(r['corr'], 3)} | {_f(r['t'])} | {_f(r['pHolm'], 3)} |")
        a("\nCorrelation by year (daily, same session): " + ", ".join(
            f"{y} UUP {_f(v['corrUUP'])} / SPY {_f(v['corrSPY'])}" for y, v in ll["corrByYear"].items()) + "\n")
        for name, ev in ll.get("dollarShockEvents", {}).items():
            parts = [f"+{h[1:]}d {_p(v['meanLogRet'])} (uncond {_p(v['unconditional'])}, placebo p lower "
                     f"{_f(v['placeboPLower'], 3)})" for h, v in ev.items() if h.startswith("h")]
            a(f"- {name} (UUP 5-session z>2, n={ev['n']}): " + "; ".join(parts))
    pr = [r for r in ll.get("predictiveRegressions", []) if r.get("pBH", 1) < 0.10]
    a("\nPredictive regressions (future 7/30-day log return on a 1-sd move in each driver; Newey-West): "
      f"{len(ll.get('predictiveRegressions', []))} tests, {len(pr)} pass BH at 10%.\n")
    if pr:
        a("| Coin | Horizon | Feature | Beta per sd | t | BH p |\n|---|---|---|---|---|---|")
        for r in pr:
            a(f"| {r['coin']} | {r['horizonDays']}d | {r['feature']} | {_p(r['betaPerSd'], 2)} | {_f(r['t'])} | {_f(r['pBH'], 3)} |")
    a(f"\nNowcast features: {rep['nowcast']['note']}.\n")
    nb = rep.get("nowcastPairs")
    if nb:
        a("## Do macro nowcasts help? (engine/macro_nowcast.py, point-in-time)\n")
        a(f"{len(nb['pairs'])} nowcast overlays (12 nowcast columns x 6 base strategies; inflation / unemployment "
          f"rising or above the last print = halve exposure, payrolls / GDP falling or below = halve), each vs its "
          f"identical base, pre-holdout only. Nowcast values are keyed to the day they were known (+1 day). "
          f"SPA over the nowcast pairs: p = {_f(nb['spa']['spaP'], 3)} (Reality Check {_f(nb['spa']['realityCheckP'], 3)}). "
          f"Sharpe improved in {nb['improvedSharpe']} of {len(nb['pairs'])}, drawdown in {nb['improvedMdd']}. "
          f"Holm-significant: {len(nb['significantHolm'])}; BH (10%): {len(nb['significantBH'])}.\n")
        a("| id | Variant | Sharpe | Base Sharpe | Max DD vs base | p | Holm | BH |\n|---|---|---|---|---|---|---|---|")
        for r in sorted(nb["pairs"], key=lambda r: r["pWithinNowcast"])[:12]:
            a(f"| {r['id']} | {r['label']} | {_f(r['sharpe'])} | {_f(r['baseSharpe'])} | {_p(r['mdd'])} vs {_p(r['baseMdd'])} | "
              f"{_f(r['pWithinNowcast'], 3)} | {_f(r['pHolmWithinNowcast'], 3)} | {_f(r['pBHWithinNowcast'], 3)} |")
        a("")
    if rep.get("holdoutLook"):
        h = rep["holdoutLook"]
        a(f"## Sealed holdout ({h['days']} days, scored once; ledger views {rep['holdoutLedger']['views']})\n")
        a("| | Sharpe | Return | Max DD |\n|---|---|---|---|")
        a(f"| BTC hold | {_f(h['btcHold'].get('sharpe'))} | {_p(h['btcHold'].get('totalReturn'))} | {_p(h['btcHold'].get('maxDrawdown'))} |")
        a(f"| Champion | {_f(h['champion'].get('sharpe'))} | {_p(h['champion'].get('totalReturn'))} | {_p(h['champion'].get('maxDrawdown'))} |")
        if h.get("walkForwardLastPick"):
            w = h["walkForwardLastPick"]
            a(f"| Last walk-forward pick | {_f(w.get('sharpe'))} | {_p(w.get('totalReturn'))} | {_p(w.get('maxDrawdown'))} |")
        if h.get("bestPreDollarPair"):
            bp = h["bestPreDollarPair"]
            a(f"| Best dollar overlay #{bp['id']} | {_f(bp['overlay'].get('sharpe'))} | {_p(bp['overlay'].get('totalReturn'))} | {_p(bp['overlay'].get('maxDrawdown'))} |")
            a(f"| its base | {_f(bp['base'].get('sharpe'))} | {_p(bp['base'].get('totalReturn'))} | {_p(bp['base'].get('maxDrawdown'))} |")
        dh = h["dollarPairsHoldout"]
        a(f"\nDollar/macro overlays that beat their base on the holdout: {dh['improvedSharpe']} of {dh['pairs']}.\n")
    if rep.get("previousHoldoutLook") and not rep.get("holdoutLook"):
        pl = rep["previousHoldoutLook"]
        h = pl["look"]
        a(f"## Sealed holdout (from the earlier design {pl['designHash']}; not re-scored)\n")
        a("This run added nowcast variants and was run with --no-holdout, so the holdout was NOT looked at again. "
          "The numbers below are the single earlier look (base design, no nowcast variants).\n")
        a("| | Sharpe | Return | Max DD |\n|---|---|---|---|")
        for name, v in (("BTC hold", h.get("btcHold")), ("Champion", h.get("champion"))):
            if v:
                a(f"| {name} | {_f(v.get('sharpe'))} | {_p(v.get('totalReturn'))} | {_p(v.get('maxDrawdown'))} |")
        dh = h.get("dollarPairsHoldout")
        if dh:
            a(f"\nDollar/macro overlays that beat their base on that holdout look: {dh['improvedSharpe']} of {dh['pairs']}.\n")
    a("\n## Caveats\n")
    a("- Survivorship: today's majors only; dead coins (LUNA, FTT, ...) absent.\n- Coinbase prices are one venue; "
      "Schwab's crypto fills may differ. The 0.75% fee dominates for active rules; see turnover and fee drag.\n"
      "- SOL/AVAX/LINK are not yet tradable at Schwab (announced); XRP/LTC/BCH/DOGE/ADA/DOT are research breadth only.\n"
      "- Crypto Sharpe uses 365-day annualization; equity reports use 252.\n- Nothing here changes the live paper "
      "config (plain TSMOM). Crypto runs only as PAPER shadow accounts (crypto_paper.py, ids 200-219).\n")
    return "\n".join(L)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--no-holdout", action="store_true")
    args = parser.parse_args(argv)
    rep = run(not args.no_holdout)
    print(json.dumps({"designHash": rep["designHash"], "champion": rep["champion"]["id"],
                      "seconds": rep["seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
