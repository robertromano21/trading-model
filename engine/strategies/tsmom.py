"""Volatility-targeted time-series momentum (trend following), long/flat.

Per horizon N (default 50 and 100 sessions): signal_N = 0.5 * [N-day total return > 0] + 0.5 * [close > N-day SMA].
The combined signal is the average over horizons (0, 0.25, 0.5, 0.75 or 1). Each active name is sized inversely to
its volatility (equal risk), the top ``max_positions`` by risk-adjusted momentum are kept, the book is scaled to a
portfolio volatility target with a leverage cap, and targets change only on rebalance sessions.
"""

import math

import numpy as np
import pandas as pd

from risk import apply_caps, blended_vol, hold_between_rebalances, keep_top, portfolio_vol_target
from universe import eligible

from .base import Strategy


class TimeSeriesMomentum(Strategy):
    name = "tsmom"
    defaults = {"horizons": (50, 100), "universe": "etf", "target_vol": 0.10, "max_positions": 8,
                "per_name_cap": 0.30, "gross_cap": 1.0, "max_leverage": 1.0, "vol_window": 60,
                "rebalance": "W-FRI", "min_price": 5.0, "min_dollar_volume": 20e6}

    def universe(self, panel):
        p = self.params
        kind = p["universe"]
        filters = {"min_price": p["min_price"]}
        if kind != "etf":
            filters["min_dollar_volume"] = p["min_dollar_volume"]
        return eligible(panel, kind, **filters)

    def _columns(self, mask):
        return list(mask.columns[mask.any()])

    def signals(self, panel, mask=None):
        mask = self.universe(panel) if mask is None else mask
        cols = self._columns(mask)
        close = panel.close[cols]
        index = (1 + panel.total_return()[cols].fillna(0)).cumprod().where(close.notna())
        parts, scores = [], []
        daily = panel.log_returns()[cols].rolling(60, min_periods=30).std()
        for horizon in self.params["horizons"]:
            horizon = int(horizon)
            past = index.shift(horizon)
            ret = index / past - 1
            ma = close.rolling(horizon, min_periods=horizon).mean()
            part = 0.5 * (ret > 0).astype(float) + 0.5 * (close > ma).astype(float)
            part = part.where(ret.notna() & ma.notna())
            parts.append(part)
            scores.append(np.log(index / past) / (daily * math.sqrt(horizon)))
        signal = sum(parts) / len(parts)
        score = sum(scores) / len(scores)
        signal = signal.where(mask[cols]).fillna(0.0)
        return signal, score.where(signal > 0)

    def target_weights(self, panel):
        p = self.params
        mask = self.universe(panel)
        cols = self._columns(mask)
        if not cols:
            return pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
        signal, score = self.signals(panel, mask)
        returns = panel.total_return()[cols]
        vol = blended_vol(returns, p["vol_window"])
        raw = (signal / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = keep_top(raw, score.fillna(-np.inf), p["max_positions"])
        scaled = portfolio_vol_target(raw, returns, p["target_vol"], p["max_leverage"], window=p["vol_window"])
        weights = apply_caps(scaled, p["per_name_cap"], min(p["gross_cap"], p["max_leverage"]))
        weights = hold_between_rebalances(weights, p["rebalance"])
        self.last_diagnostics = {"activeSignals": int((signal.iloc[-1] > 0).sum()) if len(signal) else 0}
        return weights.reindex(columns=panel.close.columns, fill_value=0.0)
