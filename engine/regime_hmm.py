"""Walk-forward hidden Markov market-regime model on volatility and breadth features.

The model allocates between the TSMOM sleeve, cash and bonds (IEF/TLT) and gates the mean-reversion
overlay. Parameters are estimated only on data through each date (rolling window, periodic refit).
Live and backtest allocations use the filtered probabilities P(s_t | x_1:t), never the smoothed
P(s_t | x_1:T). Smoothing is used only inside EM on that historical window.

regime_model='hmm' is the default at the combined-strategy level. regime_model='fixed' keeps the
original calm-and-ranging filter (SPY vol percentile + 50-day trend t-stat) and does not add a
bond overlay. Until the HMM has enough history, the allocator also falls back to that fixed filter.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically
from market_data import MARKET
from risk import TRADING_DAYS, realized_vol

STATE_NAMES = ("risk_on", "chop", "stress", "risk_off")

# Soft allocation mixed by filtered probabilities. Unallocated weight is cash.
POLICY = {
    "risk_on": {"tsmom": 1.00, "ief": 0.00, "tlt": 0.00, "meanrev": 0.0},
    "chop":    {"tsmom": 0.45, "ief": 0.20, "tlt": 0.00, "meanrev": 1.0},
    "stress":  {"tsmom": 0.55, "ief": 0.15, "tlt": 0.10, "meanrev": 0.0},
    "risk_off": {"tsmom": 0.10, "ief": 0.35, "tlt": 0.25, "meanrev": 0.0},
}

CREDIT_PAIRS = (("HYG", "IEF"), ("LQD", "IEF"), ("JNK", "IEF"))
BOND_TICKERS = ("IEF", "TLT")
# Keep duration and gold out of equity-breadth statistics.
BREADTH_EXCLUDE = {"IEF", "TLT", "GLD", "BND", "AGG", "LQD", "HYG", "JNK", "USHY", "IGSB", "SHY", "IEI"}

DEFAULTS = {
    "n_states": 3,
    "fit_window": 252,
    "min_fit": 80,
    "refit_every": 21,
    "em_iters": 25,
    "tol": 1e-4,
    "var_floor": 0.05,
    "seed": 7,
    "vol_span": 20,
    "breadth_short": 50,
    "breadth_long": 200,
    "ad_span": 10,
    "ret_span": 20,
    "z_window": 126,
    "z_min_periods": 30,
    "min_dollar_volume": 5e6,
    "meanrev_threshold": 0.45,
    "regime_vol_pct": 0.60,
    "regime_window": 252,
    "regime_min_history": 60,
    "trend_window": 50,
    "trend_t_max": 1.5,
    "include_macro": False,
}

STATE_PATH = DATA_DIR / "regime_state.json"
RESEARCH_STATE_PATH = DATA_DIR / "Research" / "regime_hmm_state.json"
# Optional dollar / curve / credit proxies (see macro_data.PROXY_DISCLAIMER). Off by default.
MACRO_HMM_FEATURES = ("dxy_ret20", "dxy_vol20", "curve_slope", "curve_level_chg20", "hy_oas", "sofr_tbill_spread")


def params_from(overrides=None):
    p = dict(DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if key in p and value is not None:
                p[key] = value
    n = int(p["n_states"])
    if n < 2 or n > 4:
        raise ValueError(f"n_states must be 2-4, got {n}")
    p["n_states"] = n
    return p


def _total_return_index(panel, ticker):
    if ticker not in panel.close.columns:
        return None
    ret = panel.total_return()[ticker]
    idx = (1 + ret.fillna(0.0)).cumprod()
    return idx.where(panel.close[ticker].notna())


def _trailing_return(index, span):
    return index / index.shift(span) - 1


def _liquid_columns(panel, min_dollar_volume):
    adv = (panel.close * panel.volume).rolling(20, min_periods=10).median()
    classes = panel.classes.reindex(panel.close.columns).fillna("other")
    keep = []
    for ticker in panel.close.columns:
        if ticker in BREADTH_EXCLUDE or ticker == MARKET:
            continue
        if classes.get(ticker) not in ("stock", "etf"):
            continue
        if float(adv[ticker].median()) >= min_dollar_volume:
            keep.append(ticker)
    return keep or [c for c in panel.close.columns if c not in BREADTH_EXCLUDE]


def build_features(panel, overrides=None):
    """Causal market-state features, one row per session. No lookahead."""
    p = params_from(overrides)
    features = pd.DataFrame(index=panel.dates)
    if MARKET not in panel.close.columns:
        return features

    spy_ret = panel.log_returns()[MARKET]
    features["spy_vol20"] = realized_vol(spy_ret, p["vol_span"], min_periods=max(10, p["vol_span"] - 5))
    spy = panel.close[MARKET]
    vol20 = realized_vol(spy_ret, 20, min_periods=15)
    features["spy_vol_percentile"] = vol20.rolling(p["regime_window"], min_periods=p["regime_min_history"]).apply(
        lambda window: float((window <= window[-1]).mean()), raw=True)
    features["spy_trend_t"] = np.log(spy / spy.shift(p["trend_window"])) / (
        spy_ret.rolling(p["trend_window"], min_periods=p["trend_window"] // 2).std() * math.sqrt(p["trend_window"]))

    cols = _liquid_columns(panel, p["min_dollar_volume"])
    close = panel.close[cols]
    returns = panel.total_return()[cols]

    def pct_above(window):
        mean = close.rolling(window, min_periods=window).mean()
        valid = close.notna() & mean.notna()
        counted = valid.sum(axis=1).replace(0, np.nan)
        return (close.gt(mean) & valid).sum(axis=1) / counted

    features["pct_above_50"] = pct_above(p["breadth_short"])
    features["pct_above_200"] = pct_above(p["breadth_long"])

    valid_ret = returns.notna()
    adv = (returns.gt(0) & valid_ret).sum(axis=1)
    dec = (returns.lt(0) & valid_ret).sum(axis=1)
    breadth = (adv - dec) / (adv + dec).replace(0, np.nan)
    features["advance_decline"] = breadth.rolling(p["ad_span"], min_periods=max(3, p["ad_span"] // 2)).mean()

    ief_idx = _total_return_index(panel, "IEF")
    tlt_idx = _total_return_index(panel, "TLT")
    if ief_idx is not None:
        features["bond_ret20"] = _trailing_return(ief_idx, p["ret_span"])
        features["ief_ret20"] = features["bond_ret20"]
    elif tlt_idx is not None:
        features["bond_ret20"] = _trailing_return(tlt_idx, p["ret_span"])
    if tlt_idx is not None:
        features["tlt_ret20"] = _trailing_return(tlt_idx, p["ret_span"])

    credit_name = None
    for risky, safe in CREDIT_PAIRS:
        risky_idx = _total_return_index(panel, risky)
        safe_idx = _total_return_index(panel, safe)
        if risky_idx is None or safe_idx is None:
            continue
        features["credit_excess20"] = _trailing_return(risky_idx, p["ret_span"]) - _trailing_return(safe_idx, p["ret_span"])
        credit_name = f"{risky}-{safe}"
        break
    features.attrs["creditProxy"] = credit_name
    features.attrs["breadthNames"] = len(cols)
    if p.get("include_macro"):
        try:
            from macro_data import load_macro
            macro = load_macro(lag=1)
        except Exception:
            macro = None
        if macro is not None and not macro.empty:
            aligned = macro.reindex(features.index).ffill()
            for col in MACRO_HMM_FEATURES:
                if col in aligned.columns and aligned[col].notna().any():
                    features[col] = aligned[col]
    return features


def fixed_meanrev_components(panel, overrides=None):
    """Original calm-and-ranging gate: low SPY vol percentile and a weak 50-day trend t-stat."""
    p = params_from(overrides)
    features = build_features(panel, p)
    if features.empty or "spy_vol_percentile" not in features:
        return pd.DataFrame({"on": False, "volPercentile": np.nan, "trendT": np.nan,
                             "calm": False, "ranging": False, "source": "fixed"}, index=panel.dates)
    pct = features["spy_vol_percentile"]
    trend_t = features["spy_trend_t"]
    calm = pct <= p["regime_vol_pct"]
    ranging = trend_t.abs() < p["trend_t_max"]
    return pd.DataFrame({"volPercentile": pct, "trendT": trend_t, "calm": calm, "ranging": ranging,
                         "on": (calm & ranging).fillna(False), "source": "fixed"})


def causal_zscore(frame, window, min_periods):
    mu = frame.rolling(window, min_periods=min_periods).mean()
    sd = frame.rolling(window, min_periods=min_periods).std(ddof=0)
    return (frame - mu) / sd.replace(0, np.nan)


HMM_FEATURE_ORDER = ("spy_vol20", "pct_above_50", "pct_above_200", "advance_decline",
                     "bond_ret20", "credit_excess20")


def feature_order(overrides=None):
    p = params_from(overrides)
    if p.get("include_macro"):
        return HMM_FEATURE_ORDER + MACRO_HMM_FEATURES
    return HMM_FEATURE_ORDER


def standardized_features(features, overrides=None):
    p = params_from(overrides)
    cols = [c for c in feature_order(p) if c in features.columns and features[c].notna().any()]
    raw = features[cols].copy()
    z = causal_zscore(raw, p["z_window"], p["z_min_periods"])
    return z.ffill()


# ---------------------------------------------------------------- numpy Gaussian HMM (diagonal covariance)
def _logsumexp(values, axis=-1, keepdims=False):
    values = np.asarray(values, dtype=float)
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    out = peak + np.log(np.exp(values - peak).sum(axis=axis, keepdims=True))
    if keepdims:
        return out
    return np.squeeze(out, axis=axis)


def log_emissions(X, mu, var):
    log_det = np.log(2.0 * np.pi * var).sum(axis=1)
    quad = ((X[:, None, :] - mu[None, :, :]) ** 2 / var[None, :, :]).sum(axis=2)
    return -0.5 * (log_det[None, :] + quad)


def forward_filter(X, theta):
    """Filtered P(s_t | x_1:t). Returns (T, K) probabilities and the last log-alpha row."""
    X = np.asarray(X, dtype=float)
    log_e = log_emissions(X, theta["mu"], theta["var"])
    log_pi = np.log(theta["pi"] + 1e-300)
    log_A = np.log(theta["A"] + 1e-300)
    t_count, k = log_e.shape
    log_alpha = np.empty((t_count, k))
    log_alpha[0] = log_pi + log_e[0]
    for t in range(1, t_count):
        log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_e[t]
    log_z = _logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_alpha - log_z), log_alpha[-1]


def forward_step(log_alpha, x, theta):
    log_A = np.log(theta["A"] + 1e-300)
    log_e = log_emissions(np.asarray(x, dtype=float)[None, :], theta["mu"], theta["var"])[0]
    new_log = _logsumexp(log_alpha[:, None] + log_A, axis=0) + log_e
    log_z = _logsumexp(new_log)
    return np.exp(new_log - log_z), new_log


def _kmeans(X, k, rng, iters=20):
    n = len(X)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
            else:
                centers[j] = X[rng.integers(0, n)]
    return labels, centers


def fit_gaussian_hmm(X, n_states, em_iters=25, tol=1e-4, var_floor=0.05, seed=7):
    """Baum-Welch (forward-backward only on this training window). Returns parameter dict."""
    X = np.asarray(X, dtype=float)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    t_count, dim = X.shape
    if t_count < n_states * 5:
        raise ValueError("not enough rows to fit the HMM")
    rng = np.random.default_rng(seed)
    labels, centers = _kmeans(X, n_states, rng)
    mu = centers.copy()
    var = np.zeros((n_states, dim))
    for j in range(n_states):
        members = X[labels == j]
        if len(members) < 2:
            var[j] = X.var(axis=0) + var_floor
        else:
            var[j] = members.var(axis=0)
    var = np.maximum(var, var_floor)
    sticky = 0.88
    A = np.full((n_states, n_states), (1 - sticky) / max(n_states - 1, 1))
    np.fill_diagonal(A, sticky)
    pi = np.ones(n_states) / n_states
    prev_ll = -np.inf
    for _ in range(em_iters):
        log_e = log_emissions(X, mu, var)
        log_pi = np.log(pi + 1e-300)
        log_A = np.log(A + 1e-300)
        log_alpha = np.empty((t_count, n_states))
        log_alpha[0] = log_pi + log_e[0]
        for t in range(1, t_count):
            log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + log_A, axis=0) + log_e[t]
        log_beta = np.zeros((t_count, n_states))
        for t in range(t_count - 2, -1, -1):
            log_beta[t] = _logsumexp(log_A + log_e[t + 1] + log_beta[t + 1], axis=1)
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        xi_sum = np.zeros((n_states, n_states))
        for t in range(t_count - 1):
            log_xi = log_alpha[t][:, None] + log_A + log_e[t + 1][None, :] + log_beta[t + 1][None, :]
            peak = np.max(log_xi)
            log_xi = log_xi - (peak + np.log(np.exp(log_xi - peak).sum()))
            xi_sum += np.exp(log_xi)
        pi = gamma[0] + 1e-3
        pi = pi / pi.sum()
        A = xi_sum + 0.25  # weak Dirichlet prior keeps transitions from collapsing
        A = A / A.sum(axis=1, keepdims=True)
        mass = gamma.sum(axis=0) + 1e-8
        mu = (gamma.T @ X) / mass[:, None]
        residual = X[:, None, :] - mu[None, :, :]
        var = (gamma[:, :, None] * residual ** 2).sum(axis=0) / mass[:, None]
        var = np.maximum(var, var_floor)
        ll = float(_logsumexp(log_alpha[-1]))
        if abs(ll - prev_ll) < tol and _ > 4:
            break
        prev_ll = ll
    return {"pi": pi, "A": A, "mu": mu, "var": var, "loglik": prev_ll, "n_states": n_states,
            "features": None, "n_obs": int(t_count)}


def label_states(theta, feature_names):
    """Name states from emission means only (no future returns). Higher score = more risk-on."""
    names = list(feature_names)
    mu = theta["mu"]
    index = {name: i for i, name in enumerate(names)}

    def take(row, name, weight=1.0, default=0.0):
        return weight * row[index[name]] if name in index else default

    scores, vols = [], []
    for row in mu:
        score = (take(row, "pct_above_50") + take(row, "pct_above_200") + take(row, "advance_decline")
                 + take(row, "credit_excess20") - take(row, "spy_vol20") - take(row, "bond_ret20")
                 - take(row, "dxy_ret20") - take(row, "hy_oas") - take(row, "curve_level_chg20")
                 - take(row, "sofr_tbill_spread"))
        scores.append(float(score))
        vols.append(float(take(row, "spy_vol20")))
    order = list(np.argsort(scores))  # low score first
    k = len(order)
    labels = [None] * k
    labels[order[0]] = "risk_off"
    labels[order[-1]] = "risk_on"
    if k == 3:
        labels[order[1]] = "chop"
    elif k == 4:
        mid = order[1:-1]
        mid_sorted = sorted(mid, key=lambda i: vols[i])
        labels[mid_sorted[0]] = "chop"
        labels[mid_sorted[1]] = "stress"
    return labels, scores


def _usable_columns(z_window, min_obs):
    cols = []
    for col in z_window.columns:
        series = z_window[col]
        if series.notna().sum() >= min_obs and np.isfinite(series.iloc[-1]):
            cols.append(col)
    return cols


def walk_forward_probs(z, overrides=None):
    """Filtered state probabilities at each date, fit only on history through that date.

    Returns (probs DataFrame, labels_by_row list[str]|None, fit_log list).
    """
    p = params_from(overrides)
    n_states = p["n_states"]
    dates = z.index
    empty = pd.DataFrame(0.0, index=dates, columns=[f"p{k}" for k in range(n_states)])
    labels_row = [None] * len(dates)
    columns_row = [None] * len(dates)
    thetas = [None] * len(dates)
    fit_log = []
    theta = None
    used_cols = None
    last_fit_at = None
    log_alpha = None

    for i in range(len(dates)):
        start = max(0, i - p["fit_window"] + 1)
        window = z.iloc[start:i + 1]
        min_obs = min(p["min_fit"], max(40, int(0.7 * len(window))))
        cols = _usable_columns(window, min_obs)
        # Need a stable core of at least 3 features.
        if len(cols) < 3:
            continue
        ready = window[cols].dropna()
        if len(ready) < p["min_fit"]:
            continue
        should_fit = theta is None or used_cols != cols or last_fit_at is None or (
            i - last_fit_at) >= p["refit_every"]
        if should_fit:
            try:
                theta = fit_gaussian_hmm(ready.to_numpy(), n_states, p["em_iters"], p["tol"],
                                        p["var_floor"], p["seed"] + 17 * len(cols))
                theta["features"] = list(cols)
                used_cols = cols
                last_fit_at = i
                filtered, log_alpha = forward_filter(ready.to_numpy(), theta)
                names, _ = label_states(theta, cols)
                fit_log.append({"date": str(dates[i].date()), "nObs": int(len(ready)),
                                "features": list(cols), "labels": names,
                                "loglik": float(theta.get("loglik", 0.0))})
            except ValueError:
                continue
        else:
            row = z.iloc[i][used_cols].to_numpy(dtype=float)
            if not np.isfinite(row).all():
                row = ready.iloc[-1][used_cols].to_numpy(dtype=float)
            _, log_alpha = forward_step(log_alpha, row, theta)
            filtered = None
        if theta is None:
            continue
        probs = filtered[-1] if filtered is not None else np.exp(log_alpha - _logsumexp(log_alpha))
        empty.iloc[i, :len(probs)] = probs
        names, _ = label_states(theta, theta["features"])
        labels_row[i] = names
        columns_row[i] = list(theta["features"])
        thetas[i] = theta
    return empty, labels_row, thetas, fit_log


def _blend(prob_row, names, key):
    total = 0.0
    for p, name in zip(prob_row, names):
        total += float(p) * POLICY[name][key]
    return total


def regime_policy(panel, model="hmm", overrides=None):
    """Date-indexed allocation policy. model is 'hmm', 'fixed', or 'off'."""
    p = params_from(overrides)
    features = build_features(panel, p)
    fixed = fixed_meanrev_components(panel, p)
    policy = pd.DataFrame({
        "tsmom_scale": 1.0, "ief_weight": 0.0, "tlt_weight": 0.0,
        "meanrev_score": fixed["on"].astype(float), "meanrev_on": fixed["on"].astype(bool),
        "state": np.nan, "state_name": "", "source": "fixed", "confidence": 0.0,
        "volPercentile": fixed["volPercentile"], "trendT": fixed["trendT"],
    }, index=panel.dates)
    for name in STATE_NAMES:
        policy[f"p_{name}"] = 0.0

    if model in (None, "off"):
        policy["source"] = "off"
        policy["meanrev_score"] = 1.0
        policy["meanrev_on"] = True
        policy.attrs["features"] = features
        policy.attrs["fitLog"] = []
        policy.attrs["creditProxy"] = features.attrs.get("creditProxy")
        return policy

    if model == "fixed":
        policy.attrs["features"] = features
        policy.attrs["fitLog"] = []
        policy.attrs["creditProxy"] = features.attrs.get("creditProxy")
        return policy

    z = standardized_features(features, p)
    probs, labels_row, thetas, fit_log = walk_forward_probs(z, p)
    ief_ok = "IEF" in panel.close.columns
    tlt_ok = "TLT" in panel.close.columns
    for i, day in enumerate(panel.dates):
        names = labels_row[i]
        if not names:
            continue
        row = probs.iloc[i].to_numpy()[:len(names)]
        theta = (thetas[i] if i < len(thetas) else None)
        feat_names = (theta or {}).get("features") or []
        _, scores = label_states(theta, feat_names) if theta is not None and feat_names else (names, [0.0] * len(names))
        score_range = float(max(scores) - min(scores)) if scores else 0.0
        # When states are barely separated (typical on a short one-regime sample), shrink toward
        # TSMOM=1 / no bond overlay / the fixed mean-reversion gate so we do not invent a crisis.
        confidence = float(np.clip((score_range - 0.4) / 1.6, 0.0, 1.0))
        tsmom = confidence * _blend(row, names, "tsmom") + (1.0 - confidence) * 1.0
        ief = confidence * _blend(row, names, "ief") if ief_ok else 0.0
        tlt = confidence * _blend(row, names, "tlt") if tlt_ok else 0.0
        hmm_score = _blend(row, names, "meanrev")
        if confidence < 0.35:
            meanrev_score = float(bool(fixed.at[day, "on"])) if day in fixed.index else 0.0
            meanrev_on = bool(meanrev_score)
        else:
            meanrev_score = hmm_score
            meanrev_on = bool(hmm_score >= p["meanrev_threshold"])
        policy.at[day, "source"] = "hmm"
        policy.at[day, "confidence"] = confidence
        policy.at[day, "tsmom_scale"] = tsmom
        policy.at[day, "ief_weight"] = ief
        policy.at[day, "tlt_weight"] = tlt
        policy.at[day, "meanrev_score"] = meanrev_score
        policy.at[day, "meanrev_on"] = meanrev_on
        winner = int(np.argmax(row))
        policy.at[day, "state"] = winner
        policy.at[day, "state_name"] = names[winner]
        for name, value in zip(names, row):
            policy.at[day, f"p_{name}"] = float(value)
    policy.attrs["features"] = features
    policy.attrs["fitLog"] = fit_log
    policy.attrs["creditProxy"] = features.attrs.get("creditProxy")
    policy.attrs["thetas"] = thetas
    policy.attrs["labels"] = labels_row
    return policy


def _clean_number(value):
    if value is None or (isinstance(value, (float, np.floating)) and not math.isfinite(value)):
        return None
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def latest_snapshot(policy, asof=None):
    """JSON-safe snapshot of the latest (or as-of) regime for the UI and the analyst."""
    if policy is None or policy.empty:
        return None
    asof = pd.Timestamp(asof) if asof is not None else policy.index[-1]
    if asof not in policy.index:
        prior = policy.index[policy.index <= asof]
        if not prior.size:
            return None
        asof = prior[-1]
    row = policy.loc[asof]
    features = policy.attrs.get("features")
    feat_row = features.loc[asof] if features is not None and asof in features.index else None
    probs = {name: _clean_number(row.get(f"p_{name}", 0.0)) for name in STATE_NAMES
             if f"p_{name}" in policy.columns and float(row.get(f"p_{name}", 0) or 0) > 1e-9}
    # Include zeros for names the current model actually uses.
    labels = None
    stored = policy.attrs.get("labels")
    if stored and asof in policy.index:
        idx = policy.index.get_loc(asof)
        labels = stored[idx] if idx < len(stored) else None
    if labels:
        probs = {name: _clean_number(row.get(f"p_{name}", 0.0)) for name in labels}
    snapshot = {
        "asOf": str(pd.Timestamp(asof).date()),
        "model": str(row.get("source") or "fixed"),
        "nStates": len(labels) if labels else int(sum(1 for v in probs.values() if v is not None)),
        "regime": row.get("state_name") or None,
        "state": _clean_number(row.get("state")),
        "probabilities": probs,
        "allocation": {
            "tsmomScale": _clean_number(row.get("tsmom_scale")),
            "ief": _clean_number(row.get("ief_weight")),
            "tlt": _clean_number(row.get("tlt_weight")),
            "meanrevOn": bool(row.get("meanrev_on")),
            "meanrevScore": _clean_number(row.get("meanrev_score")),
            "confidence": _clean_number(row.get("confidence")),
        },
        "features": {},
        "fixedFilter": {
            "on": bool(row.get("meanrev_on")) if row.get("source") == "fixed" else None,
            "volPercentile": _clean_number(row.get("volPercentile")),
            "trendT": _clean_number(row.get("trendT")),
        },
        "creditProxy": policy.attrs.get("creditProxy"),
        "fitLogTail": (policy.attrs.get("fitLog") or [])[-3:],
    }
    if row.get("source") == "fixed":
        snapshot["fixedFilter"]["on"] = bool(row.get("meanrev_on"))
    if feat_row is not None:
        for key, label in (("spy_vol20", "spyVol20"), ("spy_vol_percentile", "spyVolPercentile"),
                           ("spy_trend_t", "spyTrendT"), ("pct_above_50", "pctAbove50"),
                           ("pct_above_200", "pctAbove200"), ("advance_decline", "advanceDecline"),
                           ("ief_ret20", "iefRet20"), ("tlt_ret20", "tltRet20"),
                           ("credit_excess20", "creditExcess20")):
            if key in feat_row.index:
                snapshot["features"][label] = _clean_number(feat_row[key])
    thetas = policy.attrs.get("thetas")
    if thetas and asof in policy.index:
        theta = thetas[policy.index.get_loc(asof)]
        if theta is not None:
            names = theta.get("features") or []
            labeled, scores = label_states(theta, names)
            snapshot["stateMeans"] = {
                labeled[k]: {names[j]: _clean_number(theta["mu"][k, j]) for j in range(len(names))}
                for k in range(len(labeled))
            }
            snapshot["riskScores"] = {labeled[k]: _clean_number(scores[k]) for k in range(len(labeled))}
            snapshot["nStates"] = len(labeled)
    return snapshot


def write_regime_state(snapshot, path=None, also_research=True):
    if not snapshot:
        return None
    payload = json.dumps(snapshot, indent=2).encode()
    destinations = [Path(path) if path is not None else STATE_PATH]
    if also_research:
        destinations.append(RESEARCH_STATE_PATH)
    written = []
    for dest in destinations:
        dest.parent.mkdir(parents=True, exist_ok=True)
        write_atomically(dest, payload)
        written.append(str(dest))
    return written


def apply_bond_overlay(weights, policy, panel, gross_cap=1.0):
    """Add IEF/TLT overlay weights, shrinking other names if needed so gross stays within cap."""
    out = weights.copy()
    ief = policy["ief_weight"].reindex(out.index).fillna(0.0) if "IEF" in out.columns else 0.0
    tlt = policy["tlt_weight"].reindex(out.index).fillna(0.0) if "TLT" in out.columns else 0.0
    overlay = pd.Series(0.0, index=out.index)
    if "IEF" in out.columns:
        overlay = overlay + ief
    if "TLT" in out.columns:
        overlay = overlay + tlt
    sleeves = out.abs().sum(axis=1)
    room = (gross_cap - overlay).clip(lower=0.0)
    scale = (room / sleeves).where((sleeves > room) & (sleeves > 0), 1.0)
    out = out.mul(scale, axis=0)
    if "IEF" in out.columns and not isinstance(ief, float):
        out["IEF"] = out["IEF"].fillna(0.0) + ief
    if "TLT" in out.columns and not isinstance(tlt, float):
        out["TLT"] = out["TLT"].fillna(0.0) + tlt
    gross = out.abs().sum(axis=1)
    out = out.mul((gross_cap / gross).where(gross > gross_cap, 1.0), axis=0)
    return out


def scale_sleeve(weights, scale):
    scale = scale.reindex(weights.index).fillna(1.0)
    return weights.mul(scale, axis=0)
