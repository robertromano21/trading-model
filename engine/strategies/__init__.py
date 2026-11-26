"""Strategy registry shared by research backtests and the paper trader."""

from .base import Strategy
from .bonds import FixedIncomeStrategy
from .combined import CombinedStrategy
from .meanrev import RegimeMeanReversion
from .tsmom import TimeSeriesMomentum
from .options_overlay import OptionsOverlay

REGISTRY = {"tsmom": TimeSeriesMomentum, "meanrev": RegimeMeanReversion, "combined": CombinedStrategy,
            "bonds": FixedIncomeStrategy}
# Config-gated overlays that run on top of a base strategy (see options_research.apply_overlay); off by default.
OVERLAYS = {"options": OptionsOverlay}


def build(config):
    """config: {"strategy": "combined", "params": {...}, "sleeves": {"tsmom": {...}, "meanrev": {...}, "bonds": {...}}}"""
    name = config.get("strategy", "combined")
    params = dict(config.get("params", {}))
    if name == "combined":
        sleeves = config.get("sleeves", {})
        objs = [TimeSeriesMomentum(**sleeves.get("tsmom", {})),
                RegimeMeanReversion(**sleeves.get("meanrev", {}))]
        if params.get("include_bond_sleeve") or "bonds" in sleeves:
            objs.append(FixedIncomeStrategy(**sleeves.get("bonds", {})))
        return CombinedStrategy(sleeves=objs, **params)
    return REGISTRY[name](**params)


__all__ = ["Strategy", "TimeSeriesMomentum", "RegimeMeanReversion", "CombinedStrategy",
           "FixedIncomeStrategy", "OptionsOverlay", "REGISTRY", "OVERLAYS", "build"]
