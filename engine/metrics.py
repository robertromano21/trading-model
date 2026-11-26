"""Performance, trade and tail-risk metrics for simulator results (annualized with 252 sessions, risk-free 0)."""

import math

import numpy as np
import pandas as pd

from risk import TRADING_DAYS, moments, value_at_risk

HOLD_BUCKETS = [(1, 5, "1-5d"), (6, 20, "6-20d"), (21, 60, "21-60d"), (61, 10_000, "61d+")]


def _clean(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def drawdown(equity):
    return equity / equity.cummax() - 1


def performance(returns, equity=None, turnover=None, gross=None, costs=None):
    returns = returns.dropna()
    n = len(returns)
    if n < 2:
        return {"sessions": n}
    equity = (1 + returns).cumprod() if equity is None else equity.loc[returns.index]
    growth = float((1 + returns).prod())
    years = n / TRADING_DAYS
    cagr = growth ** (1 / years) - 1 if growth > 0 else -1.0
    vol = float(returns.std() * math.sqrt(TRADING_DAYS))
    downside = float(np.sqrt((np.minimum(returns, 0) ** 2).mean()) * math.sqrt(TRADING_DAYS))
    mdd = float(drawdown((1 + returns).cumprod()).min())
    sharpe = float(returns.mean() / returns.std() * math.sqrt(TRADING_DAYS)) if returns.std() > 0 else None
    out = {
        "start": str(returns.index[0].date()), "end": str(returns.index[-1].date()), "sessions": n,
        "totalReturn": growth - 1, "cagr": cagr, "volatility": vol, "sharpe": sharpe,
        "sharpeStdErr": math.sqrt((1 + 0.5 * (sharpe or 0) ** 2) / years) if years > 0 else None,
        "sortino": float(returns.mean() * TRADING_DAYS / downside) if downside > 0 else None,
        "maxDrawdown": mdd, "calmar": cagr / abs(mdd) if mdd < 0 else None,
        "bestDay": float(returns.max()), "worstDay": float(returns.min()),
        "dailyMoments": moments(returns),
    }
    if turnover is not None:
        out["turnoverPerYear"] = float(turnover.loc[returns.index].sum() / years)
    if gross is not None:
        g = gross.loc[returns.index]
        out["avgGrossExposure"] = float(g.mean())
        out["timeInMarket"] = float((g > 0.01).mean())
    if costs is not None:
        out["costBpsPerYear"] = float(costs.loc[returns.index].sum() / years * 1e4)
    var = {}
    for confidence in (0.95, 0.99):
        for horizon in (1, 5):
            v = value_at_risk(returns[returns.ne(0)] if (returns.ne(0)).sum() > 60 else returns, confidence, horizon)
            if v:
                var[f"{int(confidence * 100)}_{horizon}d"] = {k: _clean(x) for k, x in v.items()}
    out["valueAtRisk"] = var
    return {k: _clean(v) if not isinstance(v, dict) else v for k, v in out.items()}


def trade_stats(trades):
    if trades is None or trades.empty:
        return {"trades": 0}
    t = trades.copy()
    out = {"trades": int(len(t)), "winRate": float((t["pnl"] > 0).mean()),
           "avgHoldSessions": float(t["holding_sessions"].mean()),
           "avgReturn": float(t["return"].mean()), "exitReasons": t["exit_reason"].value_counts().to_dict()}
    buckets = {}
    for low, high, label in HOLD_BUCKETS:
        part = t[(t["holding_sessions"] >= low) & (t["holding_sessions"] <= high)]
        if len(part):
            buckets[label] = {"trades": int(len(part)), "winRate": float((part["pnl"] > 0).mean()),
                              "avgReturn": float(part["return"].mean())}
    out["byHoldingPeriod"] = buckets
    if "sleeve" in t and t["sleeve"].notna().any():
        out["bySleeve"] = {str(k): {"trades": int(len(g)), "winRate": float((g["pnl"] > 0).mean()),
                                    "pnl": float(g["pnl"].sum())} for k, g in t.groupby("sleeve")}
    return out


def summarize(result, start=None):
    """Metrics for a SimResult, optionally from `start` (e.g. after warm-up)."""
    returns = result.returns if start is None else result.returns.loc[pd.Timestamp(start):]
    trades = result.trades
    if start is not None and not trades.empty:
        trades = trades[pd.to_datetime(trades["entry_date"]) >= pd.Timestamp(start)]
    return {"performance": performance(returns, result.equity, result.turnover, result.gross_exposure, result.costs),
            "tradeStats": trade_stats(trades)}
