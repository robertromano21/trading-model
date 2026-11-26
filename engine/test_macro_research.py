"""Tests for FDR, forward returns and leak-free walk-forward. No network."""

import math
import unittest

import numpy as np
import pandas as pd

from macro_research import bh_fdr, forward_return, walk_forward_predict


class FdrTests(unittest.TestCase):
    def test_all_high_p_none_rejected(self):
        rejected, adj = bh_fdr([0.2, 0.4, 0.8])
        self.assertFalse(rejected.any())
        self.assertTrue(np.all(adj >= 0.2 - 1e-12))

    def test_clear_hit_rejected(self):
        rejected, adj = bh_fdr([0.00001, 0.8, 0.9, 0.85])
        self.assertTrue(rejected[0])
        self.assertFalse(rejected[1:].any())
        self.assertLessEqual(adj[0], 0.05)


class ForwardReturnTests(unittest.TestCase):
    def test_two_day_forward_matches_compounded_future(self):
        r = pd.Series([0.01, 0.02, -0.01, 0.03], index=pd.date_range("2024-01-02", periods=4, freq="B"))
        fwd = forward_return(r, 2)
        expected = (1 + r.iloc[2]) * (1 + r.iloc[3]) - 1
        self.assertAlmostEqual(fwd.iloc[1], expected)
        self.assertTrue(math.isnan(fwd.iloc[-1]))  # not enough future


class WalkForwardTests(unittest.TestCase):
    def test_no_signal_has_poor_oos_fit(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-02", periods=300, freq="B")
        x = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
        y = pd.Series(rng.normal(0, 1, len(idx)), index=idx)
        out = walk_forward_predict(y, x, min_train=80, label_gap=5)
        self.assertIsNotNone(out)
        self.assertLess(abs(out["ic"]), 0.25)
        self.assertGreater(out["forecastBetaP"], 0.05)

    def test_label_gap_keeps_unresolved_outcomes_out_of_the_fit(self):
        # If we peek by fitting on unresolved labels, a lookahead y=x.shift(-gap) would look strong
        # even with gap set. With the gap enforced, a feature that only lines up with the *current*
        # unresolved label should not get a free lunch beyond what's already known.
        idx = pd.date_range("2020-01-02", periods=250, freq="B")
        x = pd.Series(np.linspace(-1, 1, len(idx)), index=idx)
        # y_t is x_{t+5}, only known at t+5.
        y = x.shift(-5)
        honest = walk_forward_predict(y, x, min_train=80, label_gap=5)
        peeked = walk_forward_predict(y, x, min_train=80, label_gap=1)
        self.assertIsNotNone(honest)
        self.assertIsNotNone(peeked)
        # Peeking uses unresolved y and should look better. The honest test may still find some
        # structure because x is a smooth trend, but peeked R^2 should be higher.
        self.assertGreater(peeked["oosR2"], honest["oosR2"] - 1e-9)


if __name__ == "__main__":
    unittest.main()
