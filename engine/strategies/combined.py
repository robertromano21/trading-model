"""Equal-volatility (risk-parity across sleeves) combination of the trend core and the mean-reversion overlay.

Each sleeve's weights are rescaled so its trailing realized volatility equals target_vol / sqrt(n_sleeves); the
sum is then limited to ``max_positions`` names, per-name and gross caps, and the leverage cap.

A walk-forward hidden Markov regime model (see regime_hmm.py) then scales the TSMOM sleeve, gates the
mean-reversion overlay and can add an IEF/TLT overlay. Set regime_model='fixed' to keep the original
calm-and-ranging filter and skip the bond overlay; 'off' disables regime scaling entirely.
"""

import math

import numpy as np
import pandas as pd

from regime_hmm import (apply_bond_overlay, latest_snapshot, regime_policy, scale_sleeve,
                        write_regime_state)
from risk import TRADING_DAYS, apply_caps, keep_top

from .base import Strategy
from .meanrev import RegimeMeanReversion
from .tsmom import TimeSeriesMomentum


class CombinedStrategy(Strategy):
    name = "combined"
    defaults = {"target_vol": 0.10, "max_positions": 10, "per_name_cap": 0.30, "gross_cap": 1.0,
                "max_leverage": 1.0, "sleeve_vol_window": 60, "max_sleeve_scale": 3.0,
                "regime_model": "fixed", "regime_n_states": 3, "regime_fit_window": 252,
                "regime_min_fit": 80, "regime_refit_every": 21, "regime_meanrev_threshold": 0.45,
                "regime_write_state": False, "regime_include_macro": False, "include_bond_sleeve": False}

    def __init__(self, sleeves=None, **params):
        super().__init__(**params)
        self.sleeves = sleeves or [TimeSeriesMomentum(), RegimeMeanReversion()]
        if self.params.get("include_bond_sleeve") and not any(getattr(s, "name", None) == "bonds" for s in self.sleeves):
            from .bonds import FixedIncomeStrategy
            self.sleeves = list(self.sleeves) + [FixedIncomeStrategy()]

    def _regime_overrides(self):
        p = self.params
        return {"n_states": p["regime_n_states"], "fit_window": p["regime_fit_window"],
                "min_fit": p["regime_min_fit"], "refit_every": p["regime_refit_every"],
                "meanrev_threshold": p["regime_meanrev_threshold"],
                "include_macro": p.get("regime_include_macro", False)}

    def target_weights(self, panel):
        p = self.params
        model = p.get("regime_model", "fixed")
        policy = regime_policy(panel, model=model, overrides=self._regime_overrides())
        for sleeve in self.sleeves:
            setter = getattr(sleeve, "set_regime_gate", None)
            if setter is not None:
                setter(policy["meanrev_on"] if model == "hmm" else None)
        returns = panel.total_return().fillna(0.0)
        budget = p["target_vol"] / math.sqrt(len(self.sleeves))
        total = pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
        self.components, scales = {}, {}
        for sleeve in self.sleeves:
            weights = sleeve.target_weights(panel).reindex(columns=total.columns, fill_value=0.0)
            pnl = (weights.shift().fillna(0.0) * returns).sum(axis=1)
            exposure = weights.abs().sum(axis=1).shift().fillna(0.0)
            # Volatility per unit of exposure, measured only on days the sleeve was invested (causal).
            unit = (pnl / exposure.replace(0, np.nan)).rolling(p["sleeve_vol_window"], min_periods=20).std()
            unit = unit.ffill() * math.sqrt(TRADING_DAYS)
            sleeve_vol = unit * weights.abs().sum(axis=1)
            scale = (budget / sleeve_vol).replace([np.inf, -np.inf], np.nan).clip(upper=p["max_sleeve_scale"])
            scale = scale.fillna(1.0 / math.sqrt(len(self.sleeves)))
            scaled = weights.mul(scale, axis=0)
            self.components[sleeve.name] = scaled
            scales[sleeve.name] = scale
            total = total.add(scaled, fill_value=0.0)
        # Apply the HMM scale after sleeve vol targeting so the vol budget cannot undo a risk-off cut.
        if model == "hmm":
            if "tsmom" in self.components:
                total = total.sub(self.components["tsmom"], fill_value=0.0)
                self.components["tsmom"] = scale_sleeve(self.components["tsmom"], policy["tsmom_scale"])
                total = total.add(self.components["tsmom"], fill_value=0.0)
            bonds = pd.DataFrame(0.0, index=total.index, columns=total.columns)
            if "IEF" in bonds.columns:
                bonds["IEF"] = policy["ief_weight"].reindex(bonds.index).fillna(0.0)
            if "TLT" in bonds.columns:
                bonds["TLT"] = policy["tlt_weight"].reindex(bonds.index).fillna(0.0)
            self.components["bonds"] = bonds
            total = apply_bond_overlay(total, policy, panel, min(p["gross_cap"], p["max_leverage"]))
        total = keep_top(total, total.abs(), p["max_positions"])
        total = apply_caps(total, p["per_name_cap"], min(p["gross_cap"], p["max_leverage"]))
        self.last_diagnostics = {s.name: s.last_diagnostics for s in self.sleeves}
        self.last_diagnostics["sleeveScale"] = {name: float(series.iloc[-1]) for name, series in scales.items()
                                                if len(series)}
        snapshot = latest_snapshot(policy)
        self.last_diagnostics["regime"] = snapshot
        self._regime_policy = policy
        if p.get("regime_write_state") and snapshot:
            write_regime_state(snapshot)
        return total

    def stops(self, panel):
        frames = [s.stops(panel) for s in self.sleeves]
        frames = [f for f in frames if f is not None]
        if not frames:
            return None
        combined = frames[0]
        for frame in frames[1:]:
            combined = combined.combine_first(frame)
        return combined

    def sleeve_of(self, day, ticker):
        best, value = None, 0.0
        for name, frame in getattr(self, "components", {}).items():
            if ticker in frame.columns and day in frame.index and abs(frame.at[day, ticker]) > value:
                best, value = name, abs(frame.at[day, ticker])
        return best

    def describe(self):
        return {**super().describe(), "sleeves": [s.describe() for s in self.sleeves]}
