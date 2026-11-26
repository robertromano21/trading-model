"""Strategy interface. Every method must use only rows dated <= the row being computed (no lookahead).

Backtests call ``target_weights(panel)`` once for the whole history; the paper trader calls
``target_weights_at(panel, asof)``, which evaluates the same code on ``panel.upto(asof)``. A unit test checks that
both paths give identical weights."""

import pandas as pd


class Strategy:
    name = "base"
    defaults = {}

    def __init__(self, **params):
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameters {sorted(unknown)}")
        self.params = {**self.defaults, **params}
        self.last_diagnostics = {}

    # Eligible names per date (bool DataFrame).
    def universe(self, panel):
        raise NotImplementedError

    # Raw signal per date and name (DataFrame, 0 = flat).
    def signals(self, panel):
        raise NotImplementedError

    # Target portfolio weights (fraction of equity) decided at each date's close.
    def target_weights(self, panel):
        raise NotImplementedError

    # Optional protective stop prices valid for the next session (NaN = none).
    def stops(self, panel):
        return None

    def target_weights_at(self, panel, asof):
        weights = self.target_weights(panel.upto(asof))
        return weights.iloc[-1] if len(weights) else pd.Series(dtype=float)

    def describe(self):
        return {"strategy": self.name, "params": {k: list(v) if isinstance(v, tuple) else v for k, v in self.params.items()}}
