"""Volatility estimates, volatility-targeted sizing, portfolio limits and tail-aware VaR.

Normal approximations are used for sizing (they fit 20-60 day returns well); tail risk is reported with
historical, bootstrap and Student-t estimates because daily returns are fat-tailed (measured excess kurtosis
about 10 for liquid US stocks in 2024-11..2026-11).
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

TRADING_DAYS = 252
NORMAL_Z = {0.90: 1.2815515655446004, 0.95: 1.6448536269514722, 0.975: 1.959963984540054,
            0.99: 2.3263478740408408, 0.995: 2.5758293035489004}


# ---------------------------------------------------------------- volatility estimates (all causal)
def realized_vol(returns, window=60, min_periods=None):
    """Annualized rolling standard deviation of daily returns through each date."""
    return returns.rolling(window, min_periods=min_periods or max(10, window // 2)).std() * math.sqrt(TRADING_DAYS)


def ewma_vol(returns, lam=0.94, min_periods=20):
    """Annualized RiskMetrics EWMA volatility through each date."""
    variance = (returns.fillna(0.0) ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=min_periods).mean()
    return np.sqrt(variance * TRADING_DAYS).where(returns.notna().cumsum() >= min_periods)


def blended_vol(returns, window=60, lam=0.94, floor=0.05):
    """Max of realized and EWMA vol (reacts quickly to rising risk, slowly to falling risk), floored."""
    return np.maximum(realized_vol(returns, window), ewma_vol(returns, lam)).clip(lower=floor)


def atr(high, low, close, period=14):
    previous = close.shift()
    true_range = np.maximum(high - low, np.maximum((high - previous).abs(), (low - previous).abs()))
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------- sizing and limits
def vol_target_weights(signal, vol, target_vol):
    """Weight each name so it contributes target_vol of annualized volatility at full signal."""
    return (signal * (target_vol / vol)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def keep_top(weights, score, max_positions):
    """Keep at most max_positions non-zero names per row, ranked by score (ties broken by ticker)."""
    if max_positions is None:
        return weights
    active = weights.ne(0)
    ranks = score.where(active).rank(axis=1, ascending=False, method="first")
    return weights.where(ranks <= max_positions, 0.0)


def apply_caps(weights, per_name_cap=0.25, gross_cap=1.0, long_only=True):
    capped = weights.clip(lower=0.0 if long_only else -per_name_cap, upper=per_name_cap)
    gross = capped.abs().sum(axis=1)
    scale = (gross_cap / gross).where(gross > gross_cap, 1.0)
    return capped.mul(scale, axis=0)


def portfolio_vol_target(weights, returns, target_vol=0.10, max_leverage=1.0, window=60, min_periods=40):
    """Scale each row so ex-ante portfolio vol (trailing covariance through that date) hits target_vol,
    never exceeding max_leverage gross. Rows with fewer than min_periods of history are left unscaled."""
    out = weights.copy()
    values = returns.to_numpy()
    columns = {name: i for i, name in enumerate(returns.columns)}
    for row_number, (day, row) in enumerate(weights.iterrows()):
        held = row[row.ne(0)]
        if held.empty:
            continue
        end = returns.index.get_indexer([day])[0]
        if end < 0:
            continue
        idx = [columns[name] for name in held.index]
        window_values = values[max(0, end - window + 1): end + 1][:, idx]
        valid = ~np.isnan(window_values).any(axis=1)
        if valid.sum() < min_periods:
            scale = 1.0
        else:
            cov = np.cov(window_values[valid], rowvar=False) * TRADING_DAYS
            cov = np.atleast_2d(cov)
            vol = math.sqrt(max(float(held.to_numpy() @ cov @ held.to_numpy()), 1e-12))
            scale = target_vol / vol
        gross = held.abs().sum()
        scale = min(scale, max_leverage / gross) if gross > 0 else 0.0
        out.loc[day, held.index] = held * scale
    return out


@lru_cache(maxsize=32)
def _nyse_sessions(start, end):
    import exchange_calendars

    calendar = exchange_calendars.get_calendar("XNYS", start=start, end=end)
    sessions = pd.DatetimeIndex(calendar.sessions)
    return sessions.tz_localize(None) if sessions.tz is not None else sessions


def rebalance_days(index, rule="W-FRI"):
    """Boolean Series: True on the last NYSE session of each week ('W-FRI') or month ('M').

    Uses the exchange calendar, not the data, so a truncated panel (paper trading on a Tuesday) agrees with a
    full-history backtest about which days rebalance."""
    index = pd.DatetimeIndex(index)
    if rule in (None, "D") or len(index) == 0:
        return pd.Series(True, index=index)
    sessions = _nyse_sessions(str((index[0] - pd.Timedelta(days=7)).date()),
                              str((index[-1] + pd.Timedelta(days=40)).date()))
    position = sessions.searchsorted(index)
    following = sessions[np.minimum(position + 1, len(sessions) - 1)]
    if rule.startswith("W"):
        flag = following.to_period("W-FRI") != index.to_period("W-FRI")
    else:
        flag = following.to_period("M") != index.to_period("M")
    return pd.Series(np.asarray(flag), index=index)


def hold_between_rebalances(weights, rule="W-FRI"):
    """Hold targets constant between rebalance sessions ('D' = daily)."""
    if rule in (None, "D"):
        return weights
    flags = rebalance_days(weights.index, rule)
    return weights.where(flags, np.nan).ffill().fillna(0.0)


@dataclass
class CircuitBreaker:
    """Block new buying for `cooldown` sessions after a daily portfolio loss worse than `daily_loss_limit`."""
    daily_loss_limit: float = 0.04
    cooldown: int = 1
    remaining: int = 0

    def update(self, daily_return):
        if daily_return is not None and daily_return <= -self.daily_loss_limit:
            self.remaining = self.cooldown
            return True
        return False

    def blocks_buying(self):
        return self.remaining > 0

    def tick(self):
        self.remaining = max(0, self.remaining - 1)


# ---------------------------------------------------------------- tail statistics
def moments(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return {"n": n, "mean": None, "std": None, "skew": None, "excessKurtosis": None}
    mean, std = x.mean(), x.std(ddof=1)
    z = (x - mean) / x.std(ddof=0) if x.std(ddof=0) > 0 else np.zeros(n)
    return {"n": n, "mean": float(mean), "std": float(std), "skew": float((z ** 3).mean()),
            "excessKurtosis": float((z ** 4).mean() - 3)}


def jarque_bera(values):
    m = moments(values)
    if m["skew"] is None:
        return None, None
    stat = m["n"] / 6 * (m["skew"] ** 2 + m["excessKurtosis"] ** 2 / 4)
    return float(stat), float(math.exp(-stat / 2))   # chi-square(2) survival function


_T_CACHE = {}


def student_t_quantile(alpha, df, draws=400_000):
    """Lower-tail quantile of a unit-variance Student-t (Monte Carlo, fixed seed; no SciPy needed)."""
    key = (round(alpha, 6), round(df, 2))
    if key not in _T_CACHE:
        sample = np.random.default_rng(12345).standard_t(df, draws)
        if df > 2:
            sample = sample / math.sqrt(df / (df - 2))
        _T_CACHE[key] = float(np.quantile(sample, alpha))
    return _T_CACHE[key]


def value_at_risk(returns, confidence=0.99, horizon=1, bootstrap=2000, block=5, seed=7):
    """Loss quantiles (positive numbers = loss as a fraction of equity) under four models."""
    x = np.asarray(returns, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return None
    alpha = 1 - confidence
    m = moments(x)
    mu, sigma = m["mean"] * horizon, m["std"] * math.sqrt(horizon)
    normal = -(mu - NORMAL_Z.get(confidence, 2.3263478740408408) * sigma)
    kurt = m["excessKurtosis"]
    df = 4 + 6 / kurt if kurt and kurt > 0 else 30.0
    student = -(mu + student_t_quantile(alpha, df) * sigma)
    if horizon == 1:
        historical = -float(np.quantile(x, alpha))
    else:
        sums = np.convolve(x, np.ones(horizon), mode="valid")
        historical = -float(np.quantile(sums, alpha))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max(1, len(x) - block), size=(bootstrap, math.ceil(horizon / block) if horizon > 1 else 1))
    if horizon == 1:
        boot = rng.choice(x, size=(bootstrap, 200))
        boot_var = -float(np.quantile(boot, alpha))
    else:
        paths = np.concatenate([x[s:s + block][None, :] for s in starts.ravel()], axis=0)
        paths = paths.reshape(bootstrap, -1)[:, :horizon].sum(axis=1)
        boot_var = -float(np.quantile(paths, alpha))
    return {"confidence": confidence, "horizonDays": horizon, "normal": normal, "studentT": student,
            "studentTdf": df, "historical": historical, "bootstrap": boot_var}
