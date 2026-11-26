"""Offline tests for the dollar-shock detector and event study. No live network."""
from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

# Isolate writes before import.
_TMP = tempfile.TemporaryDirectory(prefix="bs_dollar_")
_DATA = Path(_TMP.name) / "Massive"
_DATA.mkdir(parents=True)
(_DATA / "Macro").mkdir()
(_DATA / "Research" / "black_swan").mkdir(parents=True)
os.environ["PRICES_DATA_DIR"] = str(_DATA)

import black_swan_dollar as d  # noqa: E402


def _gbm(n=400, seed=7, start="2018-01-02"):
    """Alternating +/- 40 bp days: 1d and 5d z-scores stay well below 2 until a jump is planted."""
    idx = pd.bdate_range(start, periods=n)
    rets = np.array([0.004, -0.004] * (n // 2 + 1), dtype=float)[:n]
    level = 100 * np.exp(np.cumsum(rets))
    return pd.Series(level, index=idx, name="dxy")




class FeatureAndDetectTests(unittest.TestCase):
    def test_planted_two_sigma_day_is_flagged_and_clustered(self):
        s = _gbm()
        # Plant a ~4-sigma up day and a second up day immediately after (should cluster).
        shock_i = 300
        s.iloc[shock_i] = s.iloc[shock_i - 1] * 1.03
        s.iloc[shock_i + 1] = s.iloc[shock_i] * 1.02
        # And a down shock later, past the refractory window.
        down_i = shock_i + 20
        s.iloc[down_i] = s.iloc[down_i - 1] * 0.97
        feat = d.features(s)
        self.assertTrue(bool(feat["shock"].iloc[shock_i]))
        self.assertEqual(feat["direction"].iloc[shock_i], "surge")
        self.assertEqual(feat["direction"].iloc[down_i], "weaken")
        events = d.cluster_events(feat)
        dates = [e["date"] for e in events]
        self.assertIn(s.index[shock_i], dates)
        self.assertNotIn(s.index[shock_i + 1], dates)  # clustered away
        self.assertIn(s.index[down_i], dates)

    def test_detect_latest_respects_asof_and_is_new(self):
        s = _gbm()
        shock_i = 300
        s.iloc[shock_i] = s.iloc[shock_i - 1] * 1.03
        with mock.patch.object(d, "load_dxy_level", return_value=s), \
             mock.patch.object(d, "load_uup_level", return_value=pd.Series(dtype=float)):
            before = d.detect_latest(asof=s.index[shock_i - 1])
            self.assertFalse(before["shock"])
            at = d.detect_latest(asof=s.index[shock_i])
            self.assertTrue(at["shock"])
            self.assertEqual(at["direction"], "surge")
            self.assertTrue(at["isNew"])
            self.assertFalse(at["tradingChange"])
            again = d.detect_latest(asof=s.index[shock_i], prev=at)
            self.assertTrue(again["shock"])
            self.assertFalse(again["isNew"])


class ForwardReturnTests(unittest.TestCase):
    def test_gap_and_horizons_are_from_the_event_close(self):
        idx = pd.bdate_range("2020-01-02", periods=30)
        close = pd.Series(np.linspace(100, 129, 30), index=idx)
        open_ = close.shift(1).fillna(100) * 1.01  # next open is 1% above previous close? wait
        # Construct known path: close t=100, open t+1=102 (gap 2%), close t+1=103, close t+5=110
        close[:] = 100
        close.iloc[10] = 100
        close.iloc[11] = 103
        close.iloc[12] = 104
        close.iloc[13] = 105
        close.iloc[14] = 107
        close.iloc[15] = 110
        close.iloc[16] = 111
        open_[:] = 100
        open_.iloc[11] = 102
        ohlc = pd.DataFrame({"open": open_, "high": close, "low": close, "close": close})
        rec = d._forward_row(ohlc, idx[10])
        self.assertAlmostEqual(rec["gap"], 0.02, places=8)
        self.assertAlmostEqual(rec["r1"], 0.03, places=8)
        self.assertAlmostEqual(rec["from_open_1"], 103 / 102 - 1, places=8)
        self.assertAlmostEqual(rec["r5"], 0.10, places=8)

    def test_pre_holdout_selection_does_not_need_holdout_events(self):
        # Pre-holdout events only should still produce a summary; holdout can be empty.
        idx = pd.bdate_range("2018-01-02", periods=40)
        close = pd.Series(100.0, index=idx)
        close.iloc[20:] = 101
        ohlc = pd.DataFrame({"open": close, "high": close, "low": close, "close": close})
        events = [{"date": idx[10], "direction": "surge", "z1": 2.5, "z5": 1.0}]
        out = d.summarize_events(events, {"SPY": ohlc}, ("SPY",), "pre-holdout")
        self.assertTrue(any(t["primary"] for t in out["tests"]))
        self.assertTrue(all(t["window"] == "pre-holdout" for t in out["tests"]))

    def test_multiple_testing_marks_a_real_effect_and_not_noise(self):
        # Strong planted mean should get a low p; BH/Holm helpers must return aligned lists.
        vals = [0.02] * 40
        st = d._ttest(vals)
        self.assertLess(st["p"], 0.01)
        mt = d._multiple_testing([st["p"], 0.8, 0.9])
        self.assertTrue(mt["bhReject"][0])
        self.assertFalse(mt["bhReject"][1])
        self.assertEqual(len(mt["bhAdj"]), 3)


class AttachAndHoldoutTests(unittest.TestCase):
    def test_attach_to_poll_does_not_change_playbook_weights(self):
        policy = {"scaleGross": 1.0, "freezeNewLongs": False, "allowVixy": False, "reason": "normal"}
        state = {"state": "NORMAL", "policy": dict(policy), "dollarShock": {}}
        reading = {
            "shock": True, "isNew": True, "direction": "surge", "tradingChange": False,
            "headline": "Dollar surge (test).", "updatedAt": "2026-11-25T12:00:00-0500",
            "live": {"source": "UUP", "asof": "2026-11-25", "z1": 2.4, "z5": 1.1, "direction": "surge"},
        }
        with mock.patch.object(d, "detect_latest", return_value=reading), \
             mock.patch.object(d, "emit_dollar_alert", return_value=False), \
             mock.patch.object(d, "log_dollar_chat", return_value=False):
            out = d.attach_to_poll(state, emit=False, root=_DATA / "Research" / "black_swan")
        self.assertTrue(out["shock"])
        self.assertEqual(state["policy"]["scaleGross"], 1.0)
        self.assertFalse(state["policy"]["freezeNewLongs"])
        self.assertTrue(state["policy"]["dollarShockIgnoredForTrading"])

    def test_holdout_ledger_records_a_look(self):
        root = Path(os.environ["PRICES_DATA_DIR"]) / "Research" / "black_swan"
        d.log_holdout("unit test pre", looked=False, extra={"n": 1}, root=root)
        d.log_holdout("unit test holdout look", looked=True, extra={"n": 2}, root=root)
        ledger = json.loads((root / "dollar_holdout_ledger.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(ledger), 2)
        self.assertFalse(ledger[-2]["lookedAtHoldout"])
        self.assertTrue(ledger[-1]["lookedAtHoldout"])
        self.assertEqual(ledger[-1]["holdoutStart"], "2023-01-03")
        self.assertEqual(ledger[-1]["tz"], "America/New_York")


class IntradayDescribeTests(unittest.TestCase):
    def test_describe_intraday_without_db_is_honest(self):
        with mock.patch.object(d, "_intraday_db", return_value=None):
            info = d.describe_intraday()
        self.assertFalse(info["available"])
        self.assertIn("intraday.sqlite3", info["reason"])


if __name__ == "__main__":
    unittest.main()
