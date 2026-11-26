"""Walk-forward probabilistic models for k-day direction and excess-vs-SPY.

Point-in-time features only (rows dated <= t). Labels are the k-session close-to-close return
starting at t, so a sample at t must be purged/embargoed from any training set whose end is
before t+k. Models: L2 logistic regression and an optional shallow stump ensemble (numpy only),
plus their simple average. Probabilities are calibrated with Platt scaling or isotonic
regression on an inner validation fold.

This module does not place orders or call brokers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from fetch_and_model import DATA_DIR
from indicators import tsmom_signal
from risk import TRADING_DAYS, apply_caps, blended_vol, hold_between_rebalances, keep_top, portfolio_vol_target
from universe import CORE_ETFS, eligible

MARKET = "SPY"
HORIZONS = (5, 20)
TSMOM_LOOKBACKS = (20, 50, 100)
RET_THRESHOLDS = (0.0,)
MODEL_DIR = DATA_DIR / "Research" / "prob_model"
LATEST_PATH = DATA_DIR / "Research" / "prob_latest.json"
STUDY_PATH = DATA_DIR / "Research" / "prob_walkforward.json"
FEATURE_NAMES = (
    "tsmom_20", "tsmom_50", "tsmom_100",
    "vol_20", "vol_60", "vol_of_vol",
    "growth_z", "mr_z",
    "regime_on", "spy_vol_pct",
    "dxy_ret", "dxy_ret20", "curve_slope", "hy_oas_chg20", "sofr_tbill_spread",
    "dilution_flag", "share_growth",
    "xs_tsmom100", "xs_vol60", "xs_growth_z",
    "ret_5", "ret_20",
)
SPY_FEATURE_NAMES = (
    "tsmom_20", "tsmom_50", "tsmom_100", "vol_20", "vol_60", "vol_of_vol",
    "regime_on", "spy_vol_pct",
    "dxy_ret", "dxy_ret20", "curve_slope", "hy_oas_chg20", "sofr_tbill_spread",
    "ret_5", "ret_20",
)
EPS = 1e-12


# ====================================================================== metrics

def _clip_p(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def brier_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return None
    return float(np.mean((p[mask] - y[mask]) ** 2))


def log_loss(y, p):
    y = np.asarray(y, dtype=float)
    p = _clip_p(p)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return None
    yy, pp = y[mask], p[mask]
    return float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))


def auc_binary(y, p):
    """Mann-Whitney AUC. None when a class is missing."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    # average ranks for ties
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[order[j + 1]] == p[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def _rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def information_coefficient(scores, forward_returns):
    """Spearman IC between predicted scores and continuous forward returns."""
    s = np.asarray(scores, dtype=float)
    r = np.asarray(forward_returns, dtype=float)
    mask = np.isfinite(s) & np.isfinite(r)
    if mask.sum() < 4:
        return None
    a, b = _rankdata(s[mask]), _rankdata(r[mask])
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom <= EPS:
        return None
    return float(np.dot(a, b) / denom)


def reliability_curve(y, p, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if sel.sum() == 0:
            continue
        rows.append({"binLow": float(lo), "binHigh": float(hi), "n": int(sel.sum()),
                     "meanPredicted": float(p[sel].mean()), "meanOutcome": float(y[sel].mean())})
    return rows


def classification_report(y, p, forward=None, n_bins=10):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    out = {"n": int(mask.sum()), "baseRate": float(y[mask].mean()) if mask.sum() else None,
           "meanPredicted": float(p[mask].mean()) if mask.sum() else None,
           "brier": brier_score(y, p), "logLoss": log_loss(y, p), "auc": auc_binary(y, p),
           "reliability": reliability_curve(y, p, n_bins)}
    if forward is not None:
        out["ic"] = information_coefficient(p, forward)
    return out


# ====================================================================== models

def _standardize_fit(X):
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        p = X.shape[1] if X.ndim == 2 else 0
        return np.zeros(p), np.ones(p)
    with np.errstate(all="ignore"):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    return mu, sd


def _standardize_apply(X, mu, sd):
    z = (X - mu) / sd
    return np.where(np.isfinite(z), z, 0.0)


def _logit_median(y):
    mu = float(np.clip(np.nanmean(y), 1e-3, 1.0 - 1e-3))
    return math.log(mu / (1.0 - mu))


def fit_logit_l2(X, y, lam=1.0):
    """L2-regularized logistic regression (intercept unpenalized). Coefficients on standardized X."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    if len(y) < 20 or y.min() == y.max():
        return {"kind": "logit", "intercept": _logit_median(y) if len(y) else 0.0,
                "coef": np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float),
                "lam": float(lam), "n": int(len(y)), "success": False}
    n, p = X.shape
    lam = max(float(lam), 0.0)

    def loss(theta):
        b, w = theta[0], theta[1:]
        z = np.clip(X @ w + b, -35.0, 35.0)
        ll = np.mean(-y * z + np.logaddexp(0.0, z))
        return float(ll + (lam / (2.0 * n)) * np.dot(w, w))

    def grad(theta):
        b, w = theta[0], theta[1:]
        phat = expit(np.clip(X @ w + b, -35.0, 35.0))
        err = phat - y
        return np.concatenate([[err.mean()], (X.T @ err) / n + (lam / n) * w])

    theta0 = np.zeros(p + 1, dtype=float)
    theta0[0] = _logit_median(y)
    res = minimize(loss, theta0, jac=grad, method="L-BFGS-B")
    return {"kind": "logit", "intercept": float(res.x[0]), "coef": np.asarray(res.x[1:], dtype=float),
            "lam": float(lam), "n": int(n), "success": bool(res.success)}


def predict_logit(model, X):
    X = np.asarray(X, dtype=float)
    z = X @ np.asarray(model["coef"], dtype=float) + float(model["intercept"])
    return expit(np.clip(z, -35.0, 35.0))


def fit_stumps(X, y, n_stumps=12, learning_rate=0.4, n_cuts=7, rng=None):
    """Shallow (depth-1) logit-boosted stumps. Numpy only; no sklearn."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[mask], y[mask]
    n, p = X.shape if X.ndim == 2 else (0, 0)
    intercept = _logit_median(y) if n else 0.0
    if n < 30 or p == 0 or y.min() == y.max():
        return {"kind": "stumps", "intercept": intercept, "stumps": [], "lr": float(learning_rate),
                "n": int(n), "success": False}
    score = np.full(n, intercept, dtype=float)
    cuts = np.quantile(X, np.linspace(0.15, 0.85, n_cuts), axis=0)
    stumps = []
    for _ in range(int(n_stumps)):
        phat = expit(np.clip(score, -35.0, 35.0))
        resid = y - phat
        wgt = np.clip(phat * (1.0 - phat), 1e-4, None)
        best = None
        for j in range(p):
            for cut in np.unique(cuts[:, j]):
                left = X[:, j] <= cut
                if left.sum() < 8 or (~left).sum() < 8:
                    continue
                # Newton leaf values and Friedman-style gain (2nd-order log-loss).
                def step(sel):
                    den = wgt[sel].sum()
                    return (resid[sel].sum() / den) if den > 1e-8 else 0.0
                def gain(sel):
                    den = wgt[sel].sum()
                    return (resid[sel].sum() ** 2 / den) if den > 1e-8 else 0.0
                a, b = step(left), step(~left)
                g = float(gain(left) + gain(~left))
                if best is None or g > best[0]:
                    best = (g, j, float(cut), float(a), float(b))
        if best is None:
            break
        _, j, cut, a, b = best
        lr = float(learning_rate)
        score = score + lr * np.where(X[:, j] <= cut, a, b)
        stumps.append({"j": int(j), "cut": cut, "left": a, "right": b})
    return {"kind": "stumps", "intercept": float(intercept), "stumps": stumps,
            "lr": float(learning_rate), "n": int(n), "success": bool(stumps)}


def predict_stumps(model, X):
    X = np.asarray(X, dtype=float)
    score = np.full(X.shape[0], float(model.get("intercept", 0.0)), dtype=float)
    lr = float(model.get("lr", 0.4))
    for st in model.get("stumps") or []:
        score = score + lr * np.where(X[:, int(st["j"])] <= float(st["cut"]), float(st["left"]), float(st["right"]))
    return expit(np.clip(score, -35.0, 35.0))


def predict_raw(model, X):
    kind = model.get("kind")
    if kind == "stumps":
        return predict_stumps(model, X)
    if kind == "ensemble":
        parts = [predict_raw(m, X) for m in model["members"]]
        return np.mean(np.vstack(parts), axis=0)
    return predict_logit(model, X)


# ====================================================================== calibration

def fit_platt(scores, y):
    """Platt scaling: logistic P(y=1 | score) with a, b on the raw probability's logit."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    z = np.log(_clip_p(s) / (1.0 - _clip_p(s))).reshape(-1, 1)
    if len(y) < 20 or y.min() == y.max():
        return {"method": "platt", "a": 1.0, "b": 0.0, "n": int(len(y))}
    fitted = fit_logit_l2(z, y, lam=0.1)
    return {"method": "platt", "a": float(fitted["coef"][0]) if len(fitted["coef"]) else 1.0,
            "b": float(fitted["intercept"]), "n": int(len(y))}


def apply_platt(cal, scores):
    s = _clip_p(scores)
    z = np.log(s / (1.0 - s))
    return expit(np.clip(float(cal.get("a", 1.0)) * z + float(cal.get("b", 0.0)), -35.0, 35.0))


def fit_isotonic(scores, y):
    """Pool-adjacent-violators isotonic regression of y on scores (non-decreasing)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    s, y = s[mask], y[mask]
    if len(y) < 8:
        return {"method": "isotonic", "knots": [0.0, 1.0], "values": [0.5, 0.5], "n": int(len(y))}
    order = np.argsort(s, kind="mergesort")
    s, y = s[order], y[order]
    # blocks: [sum, count, start, end]
    sums = y.astype(float).tolist()
    counts = [1] * len(y)
    i = 0
    blocks_s = s.tolist()
    while i < len(sums) - 1:
        if sums[i] / counts[i] > sums[i + 1] / counts[i + 1]:
            sums[i] += sums[i + 1]
            counts[i] += counts[i + 1]
            blocks_s[i] = blocks_s[i]  # representative = left edge; we'll store ranges later
            del sums[i + 1]
            del counts[i + 1]
            del blocks_s[i + 1]
            if i:
                i -= 1
            continue
        i += 1
    values = [sums[k] / counts[k] for k in range(len(sums))]
    # Map each block back to a score knot (block mean score, but we need original edges)
    # Rebuild knots from the ordered unique scores of each pooled block using counts.
    knots, vals = [], []
    idx = 0
    for c, v in zip(counts, values):
        block_scores = s[idx:idx + c]
        knots.append(float(block_scores[0]))
        vals.append(float(v))
        idx += c
    if knots[0] > 0:
        knots.insert(0, 0.0)
        vals.insert(0, vals[0])
    if knots[-1] < 1:
        knots.append(1.0)
        vals.append(vals[-1])
    return {"method": "isotonic", "knots": knots, "values": vals, "n": int(len(y))}


def apply_isotonic(cal, scores):
    knots = np.asarray(cal.get("knots") or [0.0, 1.0], dtype=float)
    values = np.asarray(cal.get("values") or [0.5, 0.5], dtype=float)
    return np.clip(np.interp(np.asarray(scores, dtype=float), knots, values), 0.0, 1.0)


def apply_calibrator(cal, scores):
    if not cal:
        return np.asarray(scores, dtype=float)
    if cal.get("method") == "isotonic":
        return apply_isotonic(cal, scores)
    return apply_platt(cal, scores)


def choose_calibrator(scores, y, method="auto"):
    """Fit Platt and isotonic; pick the lower-Brier one when method='auto'."""
    if method == "platt":
        return fit_platt(scores, y)
    if method == "isotonic":
        return fit_isotonic(scores, y)
    platt = fit_platt(scores, y)
    iso = fit_isotonic(scores, y)
    bp = brier_score(y, apply_platt(platt, scores))
    bi = brier_score(y, apply_isotonic(iso, scores))
    if bi is not None and bp is not None and bi < bp:
        iso["comparedBrier"] = {"platt": bp, "isotonic": bi}
        return iso
    platt["comparedBrier"] = {"platt": bp, "isotonic": bi}
    return platt


# ====================================================================== features / targets

def _tsmom_matrix(close, index, n):
    past = index.shift(n)
    ret = index / past - 1.0
    ma = close.rolling(n, min_periods=n).mean()
    part = 0.5 * (ret > 0).astype(float) + 0.5 * (close > ma).astype(float)
    return part.where(ret.notna() & ma.notna())


def _growth_z(panel):
    """Growth-path z-score when the sleeve is cheap to compute; otherwise NaN."""
    try:
        from strategies.growth_meanrev import GrowthMeanReversion, ew_slope
        close = panel.close.where(panel.close > 0)
        logp = np.log(close)
        strat = GrowthMeanReversion()
        g = ew_slope(logp, strat.params.get("trend_halflife", 60))
        z, _ = strat.zscore(logp, g)
        return z
    except Exception:  # noqa: BLE001 - feature is optional
        return pd.DataFrame(np.nan, index=panel.dates, columns=panel.close.columns)


def _flat_z(panel, lookback=10):
    close = panel.close
    mu = close.rolling(lookback, min_periods=lookback).mean()
    sd = close.rolling(lookback, min_periods=lookback).std()
    return (close - mu) / sd.where(sd > 0)


def _regime_frame(panel):
    try:
        from strategies.meanrev import RegimeMeanReversion
        return RegimeMeanReversion().regime(panel)
    except Exception:  # noqa: BLE001
        return pd.DataFrame({"on": 0.0, "spy_vol_pct": np.nan}, index=panel.dates)


def _macro_features(dates):
    try:
        from macro_research import load_features
        feats = load_features()
    except Exception:  # noqa: BLE001
        return pd.DataFrame(index=dates)
    if feats is None or feats.empty:
        return pd.DataFrame(index=dates)
    feats = feats.copy()
    feats.index = pd.to_datetime(feats.index)
    return feats.reindex(dates)


def _dilution_panels(dates, tickers):
    flag = pd.DataFrame(0.0, index=dates, columns=tickers)
    growth = pd.DataFrame(np.nan, index=dates, columns=tickers)
    try:
        import dilution as dil
    except Exception:  # noqa: BLE001
        return flag, growth
    for ticker in tickers:
        try:
            hist = dil.load_history(ticker)
        except Exception:  # noqa: BLE001
            continue
        if not hist:
            continue
        # Point-in-time: last known filing dated <= t.
        rows = []
        for item in hist if isinstance(hist, list) else hist.get("points") or hist.get("rows") or []:
            if not isinstance(item, dict):
                continue
            day = item.get("asof") or item.get("date") or item.get("filed") or item.get("periodEnd")
            if not day:
                continue
            rows.append((pd.Timestamp(day), item))
        if not rows:
            continue
        rows.sort(key=lambda r: r[0])
        series_flag, series_g = [], []
        j, cur_f, cur_g = -1, 0.0, np.nan
        days = list(dates)
        for day in days:
            while j + 1 < len(rows) and rows[j + 1][0] <= day:
                j += 1
                rec = rows[j][1]
                chg = rec.get("shareGrowth") or rec.get("yoy") or rec.get("change") or rec.get("dilutedYoy")
                try:
                    cur_g = float(chg) if chg is not None else cur_g
                except (TypeError, ValueError):
                    pass
                blocked = rec.get("blocksBuy") or rec.get("red") or rec.get("flag") or rec.get("blocked")
                if blocked is None and cur_g is not None and np.isfinite(cur_g):
                    blocked = abs(cur_g) >= 0.10
                cur_f = 1.0 if blocked else 0.0
            series_flag.append(cur_f)
            series_g.append(cur_g)
        flag[ticker] = series_flag
        growth[ticker] = series_g
    return flag, growth


def build_features(panel, tickers=None, include_macro=True, include_dilution=True):
    """Date x ticker feature frames (causal). Keys are FEATURE_NAMES (subset may be missing)."""
    cols = list(tickers) if tickers is not None else list(panel.close.columns)
    cols = [c for c in cols if c in panel.close.columns]
    close = panel.close[cols]
    rets = panel.total_return()[cols]
    index = (1 + rets.fillna(0.0)).cumprod().where(close.notna())
    frames = {}
    for n in TSMOM_LOOKBACKS:
        frames[f"tsmom_{n}"] = _tsmom_matrix(close, index, n)
    frames["vol_20"] = rets.rolling(20, min_periods=15).std() * math.sqrt(TRADING_DAYS)
    frames["vol_60"] = rets.rolling(60, min_periods=30).std() * math.sqrt(TRADING_DAYS)
    frames["vol_of_vol"] = frames["vol_20"].rolling(20, min_periods=15).std()
    frames["growth_z"] = _growth_z(panel).reindex(columns=cols)
    frames["mr_z"] = _flat_z(panel).reindex(columns=cols)
    regime = _regime_frame(panel)
    on = regime["on"] if "on" in regime.columns else pd.Series(0.0, index=panel.dates)
    if not isinstance(on, pd.Series):
        on = pd.Series(on, index=regime.index)
    frames["regime_on"] = pd.DataFrame({c: on.reindex(panel.dates).astype(float) for c in cols})
    vol_pct = regime["spy_vol_percentile"] if "spy_vol_percentile" in regime.columns else (
        regime["spy_vol_pct"] if "spy_vol_pct" in regime.columns else pd.Series(np.nan, index=panel.dates))
    if not isinstance(vol_pct, pd.Series):
        vol_pct = pd.Series(vol_pct, index=regime.index)
    frames["spy_vol_pct"] = pd.DataFrame({c: vol_pct.reindex(panel.dates).astype(float) for c in cols})
    if include_macro:
        macro = _macro_features(panel.dates)
        for name in ("dxy_ret", "dxy_ret20", "curve_slope", "hy_oas_chg20", "sofr_tbill_spread"):
            if name in macro.columns:
                frames[name] = pd.DataFrame({c: macro[name].astype(float) for c in cols})
    if include_dilution:
        flag, growth = _dilution_panels(panel.dates, cols)
        frames["dilution_flag"] = flag
        frames["share_growth"] = growth
    frames["ret_5"] = index / index.shift(5) - 1.0
    frames["ret_20"] = index / index.shift(20) - 1.0
    # Cross-sectional ranks use only same-day causal features (no future names or values).
    for src, dest in (("tsmom_100", "xs_tsmom100"), ("vol_60", "xs_vol60"), ("growth_z", "xs_growth_z")):
        if src in frames:
            frames[dest] = frames[src].rank(axis=1, pct=True)
    return frames


def stack_features(frames, tickers=None, names=None):
    """Long matrix (date, ticker) x features. Alignment is inner on available columns."""
    names = list(names or FEATURE_NAMES)
    present = [n for n in names if n in frames]
    if not present:
        return pd.DataFrame()
    sample = frames[present[0]]
    cols = [c for c in (tickers or sample.columns) if c in sample.columns]
    parts = []
    for name in present:
        frame = frames[name].reindex(index=sample.index, columns=cols)
        parts.append(frame.stack(future_stack=True).rename(name))
    out = pd.concat(parts, axis=1)
    out.index = out.index.set_names(["date", "ticker"])
    return out


def build_targets(panel, k=20, threshold=0.0, market=MARKET):
    """Forward k-session close-to-close total return, excess vs SPY, and binary labels.

    Label at t uses prices through t+k, so it is not known at decision time t.
    """
    k = int(k)
    rets = panel.total_return()
    fwd = (1.0 + rets.fillna(0.0)).rolling(k).apply(np.prod, raw=True) - 1.0
    # rolling().apply at t is the product of rets through t, i.e. return from t-k to t.
    # We want return from t to t+k, which is the same series shifted by -k.
    fwd = fwd.shift(-k)
    if market not in fwd.columns:
        excess = fwd * np.nan
        spy = pd.Series(np.nan, index=panel.dates)
    else:
        spy = fwd[market]
        excess = fwd.sub(spy, axis=0)
    y_excess = (excess > 0).astype(float).where(excess.notna())
    y_ret = (fwd > float(threshold)).astype(float).where(fwd.notna())
    y_spy = (spy > 0).astype(float).where(spy.notna())
    return {"forward": fwd, "excess": excess, "spyForward": spy,
            "yExcess": y_excess, "yRet": y_ret, "ySpy": y_spy, "k": k, "threshold": float(threshold)}


def target_key(kind, k, threshold=0.0):
    if kind == "spy":
        return f"spy_up_{int(k)}"
    if kind == "excess":
        return f"excess_{int(k)}"
    return f"ret_{int(k)}_{str(threshold).replace('.', 'p')}"


# ====================================================================== design matrices / purging

def assemble(features, targets, tickers, kind="excess", k=20, threshold=0.0):
    """Join stacked features with the chosen label and the continuous forward return."""
    k = int(k)
    if kind == "spy":
        y = targets["ySpy"].rename("y")
        fwd = targets["spyForward"].rename("forward")
        frame = features[features.index.get_level_values("ticker") == MARKET].copy()
        y = y.reindex(frame.index.get_level_values("date"))
        fwd = fwd.reindex(frame.index.get_level_values("date"))
        frame = frame.assign(y=y.to_numpy(), forward=fwd.to_numpy())
        return frame
    label = targets["yExcess"] if kind == "excess" else targets["yRet"]
    cont = targets["excess"] if kind == "excess" else targets["forward"]
    y = label[tickers].stack(future_stack=True).rename("y")
    f = cont[tickers].stack(future_stack=True).rename("forward")
    y.index = y.index.set_names(["date", "ticker"])
    f.index = f.index.set_names(["date", "ticker"])
    return features.join(y, how="inner").join(f, how="left")


def purge_mask(dates, train_end, k, embargo):
    """True for observation dates that may be used as training samples.

    A sample dated t has a label that uses prices through t+k, so we keep only
    t <= train_end - k. An embargo of E sessions before a test start T is applied
    by passing train_end = T - E sessions (the caller converts that to a timestamp).
    """
    dates = pd.DatetimeIndex(dates)
    train_end = pd.Timestamp(train_end)
    # number of sessions after each date until train_end
    loc = pd.Series(np.arange(len(dates)), index=dates)
    if train_end not in loc.index:
        earlier = dates[dates <= train_end]
        if not len(earlier):
            return pd.Series(False, index=dates)
        train_end = earlier[-1]
    end_i = int(loc.loc[train_end])
    keep = loc <= (end_i - int(k))
    return keep


def session_shift(dates, day, n):
    """The date n sessions before (n<0? after) ``day``. n>0 moves forward."""
    dates = pd.DatetimeIndex(dates)
    day = pd.Timestamp(day)
    if day not in dates:
        earlier = dates[dates <= day]
        if not len(earlier):
            return dates[0] if n >= 0 else dates[0]
        day = earlier[-1]
    i = int(dates.get_loc(day)) + int(n)
    i = max(0, min(len(dates) - 1, i))
    return dates[i]


# ====================================================================== pack / predict a frozen model

def pack_model(kind, feature_names, mu, sd, model, calibrator, target, extra=None):
    doc = {"kind": kind, "featureNames": list(feature_names), "mu": np.asarray(mu, dtype=float).tolist(),
           "sd": np.asarray(sd, dtype=float).tolist(), "model": _to_plain(model),
           "calibrator": _to_plain(calibrator), "target": dict(target)}
    if extra:
        doc.update(extra)
    return doc


def _to_plain(value):
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    return str(value)


def fit_pack(X, y, feature_names, kind="logit", lam=1.0, val_X=None, val_y=None, target=None,
             cal_method="auto"):
    mu, sd = _standardize_fit(X)
    Xs = _standardize_apply(X, mu, sd)
    if kind == "stumps":
        model = fit_stumps(Xs, y)
    elif kind == "ensemble":
        logit = fit_logit_l2(Xs, y, lam=lam)
        stumps = fit_stumps(Xs, y)
        model = {"kind": "ensemble", "members": [logit, stumps]}
    else:
        model = fit_logit_l2(Xs, y, lam=lam)
    raw_train = predict_raw(model, Xs)
    if val_X is not None and val_y is not None and len(val_y) >= 20:
        raw_val = predict_raw(model, _standardize_apply(val_X, mu, sd))
        calibrator = choose_calibrator(raw_val, val_y, method=cal_method)
    else:
        calibrator = choose_calibrator(raw_train, y, method=cal_method)
    return pack_model(kind, feature_names, mu, sd, model, calibrator, target or {})


def predict_pack(pack, X):
    names = pack["featureNames"]
    if isinstance(X, pd.DataFrame):
        X = X.reindex(columns=names).to_numpy(dtype=float)
    else:
        X = np.asarray(X, dtype=float)
    Xs = _standardize_apply(X, np.asarray(pack["mu"], dtype=float), np.asarray(pack["sd"], dtype=float))
    raw = predict_raw(pack["model"], Xs)
    return apply_calibrator(pack.get("calibrator"), raw)


# ====================================================================== trading weights

class ProbStrategy:
    """TSMOM filtered or probability-sized using a frozen walk-forward probability pack.

    Historically, stored out-of-sample probabilities are used (no in-sample leak). After the
    last stored date, the frozen pack is applied to point-in-time features.
    """

    name = "tsmom_prob"
    defaults = {"mode": "tsmom_prob_filter", "threshold": 0.60, "kelly_frac": 0.5,
                "target_vol": 0.10, "max_positions": 8, "per_name_cap": 0.30, "gross_cap": 1.0,
                "max_leverage": 1.0, "vol_window": 60, "rebalance": "W-FRI", "min_price": 5.0,
                "min_dollar_volume": 20e6, "universe": "etf", "target_name": "excess_20",
                "pack_path": None, "proba_path": None}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}
        self.last_diagnostics = {}
        self._pack = None
        self._proba = None

    def _load(self):
        if self._pack is None:
            path = Path(self.params["pack_path"] or (MODEL_DIR / "frozen.json"))
            if path.is_file():
                self._pack = json.loads(path.read_text(encoding="utf-8"))
        if self._proba is None:
            path = Path(self.params["proba_path"] or (MODEL_DIR / "oos_proba.csv"))
            if path.is_file():
                frame = pd.read_csv(path, parse_dates=["date"])
                self._proba = frame
        return self._pack, self._proba

    def _tsmom(self, panel):
        from strategies.tsmom import TimeSeriesMomentum
        p = self.params
        return TimeSeriesMomentum(universe=p["universe"], target_vol=p["target_vol"],
                                  max_positions=p["max_positions"], per_name_cap=p["per_name_cap"],
                                  gross_cap=p["gross_cap"], max_leverage=p["max_leverage"],
                                  vol_window=p["vol_window"], rebalance=p["rebalance"],
                                  min_price=p["min_price"], min_dollar_volume=p["min_dollar_volume"])

    def _probability_frame(self, panel, tickers):
        pack, stored = self._load()
        target = self.params["target_name"]
        out = pd.DataFrame(np.nan, index=panel.dates, columns=tickers)
        if stored is not None and "ticker" in stored.columns and target in stored.columns:
            sub = stored[stored["ticker"].isin(tickers)]
            if len(sub):
                pivot = sub.pivot_table(index="date", columns="ticker", values=target, aggfunc="last")
                pivot.index = pd.to_datetime(pivot.index)
                out = pivot.reindex(index=panel.dates, columns=tickers)
        last_stored = out.dropna(how="all").index.max() if out.notna().any().any() else None
        need = panel.dates if last_stored is None else panel.dates[panel.dates > last_stored]
        if pack is not None and len(need):
            # Apply the frozen pack only after the OOS tape ends (live / forward dates).
            frames = build_features(panel, tickers=tickers)
            stacked = stack_features(frames, tickers=tickers, names=pack.get("featureNames"))
            if len(stacked):
                stacked = stacked.loc[stacked.index.get_level_values("date").isin(need)]
                if len(stacked):
                    pred = predict_pack(pack, stacked)
                    pred = pd.Series(pred, index=stacked.index, name=target)
                    wide = pred.unstack("ticker")
                    out = out.combine_first(wide.reindex(index=panel.dates, columns=tickers))
        return out

    def target_weights(self, panel):
        p = self.params
        tsmom = self._tsmom(panel)
        base = tsmom.target_weights(panel)
        cols = [c for c in base.columns if c != MARKET or True]
        proba = self._probability_frame(panel, list(base.columns))
        thresh = float(p["threshold"])
        mode = p["mode"]
        if mode in ("tsmom_prob_filter", "tsmom_filter"):
            weights = base.where(proba.reindex_like(base) >= thresh, 0.0)
        else:
            edge = (2.0 * proba - 1.0).clip(lower=0.0)
            edge = edge.where(proba >= thresh, 0.0)
            rets = panel.total_return()[base.columns]
            vol = blended_vol(rets, p["vol_window"])
            raw = (float(p["kelly_frac"]) * edge / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            raw = keep_top(raw, proba.fillna(-np.inf), p["max_positions"])
            scaled = portfolio_vol_target(raw, rets, p["target_vol"], p["max_leverage"], window=p["vol_window"])
            weights = apply_caps(scaled, p["per_name_cap"], min(p["gross_cap"], p["max_leverage"]))
            weights = hold_between_rebalances(weights, p["rebalance"])
        self.last_diagnostics = {
            "mode": mode, "threshold": thresh, "target": p["target_name"],
            "nAbove": int((proba.iloc[-1] >= thresh).sum()) if len(proba) else 0,
            "active": int((weights.iloc[-1] > 0).sum()) if len(weights) else 0,
        }
        return weights.reindex(columns=panel.close.columns, fill_value=0.0)

    def stops(self, panel):
        return self._tsmom(panel).stops(panel)

    def target_weights_at(self, panel, asof):
        weights = self.target_weights(panel.upto(asof))
        return weights.iloc[-1] if len(weights) else pd.Series(dtype=float)

    def describe(self):
        return {"strategy": self.name, "params": {k: v for k, v in self.params.items()
                                                  if k not in ("pack_path", "proba_path") or v}}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_plain(payload), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def latest_snapshot(proba, packs, asof=None, metrics=None, caveats=None):
    """Compact per-ticker probabilities for the analyst and the ticker / paper pages."""
    if proba is None or not len(proba):
        return {"asof": None, "tickers": {}, "spy": {}, "note": "No probability tape yet."}
    frame = proba.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if asof is not None:
        frame = frame[frame["date"] <= pd.Timestamp(asof)]
    if not len(frame):
        return {"asof": None, "tickers": {}, "spy": {}, "note": "No probability rows at this date."}
    last = frame["date"].max()
    rowset = frame[frame["date"] == last]
    tickers = {}
    spy = {}
    for _, row in rowset.iterrows():
        ticker = str(row["ticker"])
        rec = {k: (None if not (isinstance(v, (int, float)) and np.isfinite(v)) else float(v))
               for k, v in row.items() if k not in ("date", "ticker")}
        rec["date"] = str(pd.Timestamp(last).date())
        if ticker == MARKET:
            spy = rec
        tickers[ticker] = rec
    return {"asof": str(pd.Timestamp(last).date()), "tickers": tickers, "spy": spy,
            "targets": [c for c in frame.columns if c not in ("date", "ticker")],
            "metrics": metrics or {}, "caveats": caveats or [],
            "frozen": {k: (p.get("target") if isinstance(p, dict) else None)
                       for k, p in (packs or {}).items()}}
