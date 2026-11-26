"""Regime-filtered short-horizon mean reversion overlay (long only, 5-10 day holds).

Entry: z = (close - SMA_L) / std_L below ``entry_z`` for a liquid stock that is above its ``trend_filter``-day SMA,
but only while the market regime allows new entries. The default gate is the original fixed filter:
  * SPY 20-day realized vol at or below the ``regime_vol_pct`` percentile of its trailing year, and
  * |SPY 50-day trend t-stat| below ``trend_t_max`` (log return / (daily vol * sqrt(50))).
The combined strategy can replace that gate with a walk-forward hidden Markov model (regime_filter='hmm'
or an external gate). Exit: z back above ``exit_z`` (the mean), ``max_hold`` sessions, or a ``stop_atr`` x ATR
protective stop. Positions are sized to equal volatility, at most ``max_positions`` at a time.
"""

import numpy as np
import pandas as pd

from regime_hmm import fixed_meanrev_components, regime_policy
from risk import apply_caps, atr, blended_vol
from universe import eligible

from .base import Strategy


class RegimeMeanReversion(Strategy):
    name = "meanrev"
    defaults = {"universe": "stocks", "lookback": 10, "entry_z": -2.0, "exit_z": 0.0, "max_hold": 10,
                "max_positions": 5, "name_target_vol": 0.04, "per_name_cap": 0.20, "gross_cap": 1.0,
                "regime_vol_pct": 0.60, "regime_window": 252, "regime_min_history": 60, "trend_window": 50,
                "trend_t_max": 1.5, "trend_filter": 100, "stop_atr": 3.0, "min_price": 5.0,
                "min_dollar_volume": 20e6, "regime_filter": "fixed"}

    def __init__(self, **params):
        super().__init__(**params)
        self._external_gate = None

    def set_regime_gate(self, gate):
        """Optional precomputed boolean Series (from the combined HMM) that replaces the entry gate."""
        self._external_gate = gate

    def universe(self, panel):
        p = self.params
        return eligible(panel, p["universe"], min_price=p["min_price"], min_dollar_volume=p["min_dollar_volume"])

    def regime(self, panel):
        p = self.params
        base = fixed_meanrev_components(panel, p)
        if self._external_gate is not None:
            gate = self._external_gate.reindex(panel.dates)
            if isinstance(gate, pd.DataFrame):
                gate = gate.iloc[:, 0]
            base["on"] = gate.fillna(False).astype(bool)
            base["source"] = "hmm"
            return base
        if p.get("regime_filter") == "hmm":
            policy = regime_policy(panel, model="hmm")
            base["on"] = policy["meanrev_on"].reindex(panel.dates).fillna(False).astype(bool)
            base["source"] = "hmm"
            return base
        return base

    def _states(self, panel):
        p = self.params
        mask = self.universe(panel)
        cols = list(mask.columns[mask.any()])
        regime = self.regime(panel)
        close = panel.close[cols]
        mean = close.rolling(p["lookback"], min_periods=p["lookback"]).mean()
        std = close.rolling(p["lookback"], min_periods=p["lookback"]).std(ddof=0)
        z = (close - mean) / std.replace(0, np.nan)
        trend_ok = close > close.rolling(p["trend_filter"], min_periods=p["trend_filter"]).mean()
        entry_ok = (mask[cols] & trend_ok & (z < p["entry_z"])).to_numpy()
        zv = z.to_numpy()
        closes = close.to_numpy()
        atr_values = atr(panel.high[cols], panel.low[cols], close).to_numpy()
        regime_on = regime["on"].to_numpy()
        active = np.zeros_like(zv)
        stops = np.full_like(zv, np.nan)
        open_positions = {}   # column -> (entry row, stop price)
        for row in range(len(zv)):
            for col in list(open_positions):
                entry_row, stop = open_positions[col]
                age = row - entry_row
                price = closes[row, col]
                if np.isnan(price):
                    active[row, col], stops[row, col] = 1.0, stop   # halted: keep, simulator handles staleness
                    continue
                if zv[row, col] >= p["exit_z"] or age >= p["max_hold"] or price <= stop:
                    del open_positions[col]
                else:
                    active[row, col], stops[row, col] = 1.0, stop
            free = p["max_positions"] - len(open_positions)
            if regime_on[row] and free > 0:
                candidates = np.flatnonzero(entry_ok[row])
                candidates = [c for c in candidates if c not in open_positions and np.isfinite(atr_values[row, c])]
                candidates.sort(key=lambda c: (zv[row, c], c))
                for col in candidates[:free]:
                    stop = closes[row, col] - p["stop_atr"] * atr_values[row, col]
                    open_positions[col] = (row, stop)
                    active[row, col], stops[row, col] = 1.0, stop
        frame = lambda values: pd.DataFrame(values, index=panel.dates, columns=cols)
        return frame(active), frame(stops), z, regime

    def signals(self, panel):
        active, _, _, _ = self._states(panel)
        return active

    def target_weights(self, panel):
        p = self.params
        active, stops, z, regime = self._states(panel)
        cols = list(active.columns)
        if not cols:
            self._stops = pd.DataFrame(index=panel.dates, columns=panel.close.columns, dtype=float)
            return pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
        vol = blended_vol(panel.total_return()[cols], 60)
        weights = (active * (p["name_target_vol"] / vol)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weights = apply_caps(weights, p["per_name_cap"], p["gross_cap"])
        self._stops = stops.reindex(columns=panel.close.columns)
        last = regime.iloc[-1] if len(regime) else None
        self.last_diagnostics = {
            "regimeOn": bool(last["on"]) if last is not None else False,
            "regimeSource": None if last is None else last.get("source", "fixed"),
            "marketVolPercentile": None if last is None or pd.isna(last["volPercentile"]) else float(last["volPercentile"]),
            "marketTrendT": None if last is None or pd.isna(last["trendT"]) else float(last["trendT"]),
            "openSignals": int(active.iloc[-1].sum()) if len(active) else 0,
        }
        return weights.reindex(columns=panel.close.columns, fill_value=0.0)

    def stops(self, panel):
        if not hasattr(self, "_stops"):
            self.target_weights(panel)
        return self._stops
