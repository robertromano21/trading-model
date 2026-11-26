"""Offline tests for engine/macro_backtest.py: no look-ahead in positions, costs, SPA sanity."""
import unittest

import numpy as np
import pandas as pd

import macro_backtest as MB


class StrategyTests(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range("2000-01-03", periods=400)
        self.close = pd.Series(np.exp(np.linspace(0, 0.4, 400)), index=idx)

    def test_position_earns_next_day(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        pos = pd.Series([False, True, True, False], index=idx)
        eq = pd.Series([0.10, 0.20, 0.30, 0.40], index=idx)
        cash = pd.Series(0.0, index=idx)
        r = MB.strat_returns(pos, eq, cash, cost_bps=0)
        self.assertEqual(r.tolist(), [0.0, 0.0, 0.30, 0.40])   # signal at close d earns d+1

    def test_switch_costs(self):
        idx = pd.bdate_range("2020-01-01", periods=3)
        r = MB.strat_returns(pd.Series([True, True, True], index=idx), pd.Series(0.0, index=idx),
                             pd.Series(0.0, index=idx), cost_bps=5)
        self.assertAlmostEqual(r.iloc[1], -5e-4)
        self.assertAlmostEqual(r.iloc[2], 0.0)

    def test_tsmom_uses_only_past_prices(self):
        sig = MB.tsmom_signal(self.close)
        bumped = self.close.copy()
        bumped.iloc[300:] *= 0.1                               # crash after day 300
        sig2 = MB.tsmom_signal(bumped)
        self.assertTrue(sig.iloc[:300 + 21].equals(sig2.iloc[:300 + 21]))

    def test_candidates_are_tsmom_and_not_flag(self):
        flag = pd.Series(False, index=self.close.index)
        flag.iloc[350:] = True
        c = MB.candidates(self.close, {"x": flag})
        self.assertTrue((c["TSMOM_and_not_x"] <= c["TSMOM"]).all())
        self.assertFalse(c["TSMOM_and_not_x"].iloc[360])
        self.assertFalse(c["BH_and_not_x"].iloc[360])


class SpaTests(unittest.TestCase):
    def test_null_not_rejected_and_strong_edge_detected(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.01, size=(2000, 5))
        p = MB.spa_test(noise, B=200)[0]["spaP"]
        self.assertGreater(p, 0.05)
        edge = noise.copy()
        edge[:, 0] += 0.003
        p2 = MB.spa_test(edge, B=200)[0]["spaP"]
        self.assertLess(p2, 0.05)


if __name__ == "__main__":
    unittest.main()
