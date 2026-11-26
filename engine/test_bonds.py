"""Bond-ETF sleeve: causality, universe, and stance fallbacks. No network."""

import unittest
from datetime import date, timedelta

import numpy as np
import pandas as pd

from market_data import Panel
from strategies.bonds import BOND_ETFS, FixedIncomeStrategy
from strategies import REGISTRY, build


def _panel(n=200, names=None, seed=1):
    names = names or list(BOND_ETFS)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.004, size=(n, len(names))), axis=0))
    close = pd.DataFrame(close, index=idx, columns=names)
    # Add a simple SHY/TLT split so duration fallback has a signal later in the sample.
    if "SHY" in close.columns:
        close["SHY"] = 100 + np.linspace(0, 1, n)
    if "TLT" in close.columns:
        close["TLT"] = 100 + np.linspace(0, 8, n)
    volume = pd.DataFrame(1e6, index=idx, columns=names)
    dividends = pd.DataFrame(0.0, index=idx, columns=names)
    classes = pd.Series("etf", index=names)
    return Panel(close, close * 1.001, close * 0.999, close, volume, dividends, classes)


class BondSleeveTests(unittest.TestCase):
    def test_only_bond_etfs_get_weight(self):
        names = list(BOND_ETFS) + ["SPY", "AAPL"]
        panel = _panel(names=names)
        # Panel columns include stocks; strategy must ignore them.
        weights = FixedIncomeStrategy().target_weights(panel)
        self.assertIn("SPY", weights.columns)
        self.assertEqual(float(weights["SPY"].abs().sum()), 0.0)
        self.assertEqual(float(weights["AAPL"].abs().sum()), 0.0)
        self.assertGreater(weights[list(BOND_ETFS)].abs().sum(axis=1).iloc[-1], 0.0)

    def test_upto_matches_full_history_at_the_cut(self):
        panel = _panel(n=180)
        cut = panel.dates[120]
        strategy = FixedIncomeStrategy()
        full = strategy.target_weights(panel).loc[cut]
        clipped = FixedIncomeStrategy().target_weights(panel.upto(cut)).iloc[-1]
        aligned = pd.concat([full, clipped], axis=1).fillna(0.0)
        delta = (aligned.iloc[:, 0] - aligned.iloc[:, 1]).abs().max()
        self.assertLess(delta, 1e-10)

    def test_weights_are_nonnegative_and_capped(self):
        weights = FixedIncomeStrategy(per_name_cap=0.35, gross_cap=1.0).target_weights(_panel())
        late = weights.iloc[80:]
        self.assertTrue((late.min(axis=1) >= -1e-12).all())
        self.assertTrue((late.max(axis=1) <= 0.35 + 1e-9).all())
        self.assertTrue((late.abs().sum(axis=1) <= 1.0 + 1e-9).all())

    def test_registered_and_buildable(self):
        self.assertIn("bonds", REGISTRY)
        strategy = build({"strategy": "bonds", "params": {"target_vol": 0.05}})
        self.assertEqual(strategy.name, "bonds")
        self.assertEqual(strategy.params["target_vol"], 0.05)

    def test_combined_does_not_include_the_sleeve_by_default(self):
        strategy = build({"strategy": "combined", "params": {}, "sleeves": {}})
        names = [s.name for s in strategy.sleeves]
        self.assertNotIn("bonds", names)

    def test_combined_includes_the_sleeve_when_gated_on(self):
        strategy = build({"strategy": "combined", "params": {"include_bond_sleeve": True},
                          "sleeves": {"bonds": {"target_vol": 0.05}}})
        names = [s.name for s in strategy.sleeves]
        self.assertIn("bonds", names)


if __name__ == "__main__":
    unittest.main()
