"""Unit tests for the crypto data, research and PAPER shadow accounts. Offline: no network, no broker."""
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_TMP = tempfile.mkdtemp()
os.environ.setdefault("CRYPTO_OUT", str(Path(_TMP) / "crypto"))
os.environ.setdefault("SWARM_OUT", str(Path(_TMP) / "swarm"))

import crypto_data as cd       # noqa: E402
import crypto_paper as cp      # noqa: E402
import crypto_research as cr   # noqa: E402

ENGINE = Path(__file__).resolve().parent


def synthetic(days=900, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=days, freq="D")
    cols = {}
    for i, c in enumerate(cd.ALL_COINS):
        r = rng.normal(0.001, 0.04, days)
        px = 100 * np.exp(np.cumsum(r))
        s = pd.Series(px, index=idx)
        if i >= 2:
            s.iloc[: 100 * i] = np.nan     # later listings
        cols[c] = s
    close = pd.DataFrame(cols)
    uup = pd.Series(25 * np.exp(np.cumsum(rng.normal(0, 0.004, days))), index=idx)
    drivers = pd.DataFrame({"UUP": uup, "DTWEXBGS": uup * 4, "DFII10": 1 + np.cumsum(rng.normal(0, 0.02, days)),
                            "NETLIQ": 6000 + np.cumsum(rng.normal(0, 10, days))}, index=idx)
    return close, drivers


class SimulatorTests(unittest.TestCase):
    def test_buy_and_hold_pays_one_fee(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        close = pd.DataFrame({"BTC": np.linspace(100, 110, 10)}, index=idx)
        W = pd.DataFrame({"BTC": 1.0}, index=idx)
        eq, _ = cr.simulate(close, W, idx[0], band=0.05)
        c = cr.FEE + cr.SLIP
        self.assertAlmostEqual(eq.iloc[0], 10_000 - 10_000 * (1 - c) * c, delta=0.5)
        self.assertAlmostEqual(eq.iloc[-1] / eq.iloc[0], 1.10, delta=0.01)   # no further trades inside the band

    def test_exit_and_cash_earns_nothing(self):
        idx = pd.date_range("2020-01-01", periods=6, freq="D")
        close = pd.DataFrame({"BTC": [100, 100, 100, 50, 50, 200.0]}, index=idx)
        W = pd.DataFrame({"BTC": [1, 1, 0, 0, 0, 0.0]}, index=idx)
        eq, _ = cr.simulate(close, W, idx[0])
        self.assertAlmostEqual(eq.iloc[2], eq.iloc[-1], places=6)
        self.assertLess(eq.iloc[2], 10_000)

    def test_fee_constant_is_schwab(self):
        self.assertEqual(cr.FEE, 0.0075)


class CausalityTests(unittest.TestCase):
    def test_future_prices_do_not_change_past_weights(self):
        close, drivers = synthetic()
        cut = 600
        perturbed = close.copy()
        perturbed.iloc[cut:] *= 3.0
        dp = drivers.copy()
        dp.iloc[cut:] *= 2.0
        for spec in cr.make_design()[::7]:
            a = cr.target_weights(spec, close, drivers).iloc[:cut]
            b = cr.target_weights(spec, perturbed, dp).iloc[:cut]
            pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-12, obj=spec["label"])

    def test_release_lag_is_strict(self):
        cal = pd.date_range("2024-01-01", "2024-01-31", freq="D")
        s = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-03", "2024-01-10"]))   # Wednesdays
        out = cd.as_available(s, cd.RELEASE_RULES["DTWEXBGS"], cal)
        self.assertTrue(np.isnan(out.loc["2024-01-08"]))      # H.10 for Jan 3 is public Mon Jan 8 -> usable Jan 9
        self.assertEqual(out.loc["2024-01-09"], 1.0)
        self.assertEqual(out.loc["2024-01-15"], 1.0)
        self.assertEqual(out.loc["2024-01-16"], 2.0)
        m2 = cd.as_available(pd.Series([5.0], index=pd.to_datetime(["2024-01-01"])), cd.RELEASE_RULES["M2SL"], cal.append(
            pd.date_range("2024-02-01", "2024-03-31")))
        self.assertTrue(np.isnan(m2.loc["2024-02-27"]))
        self.assertEqual(m2.loc["2024-02-28"], 5.0)


class SelectionTests(unittest.TestCase):
    def test_selection_ignores_holdout_rows(self):
        rng = np.random.default_rng(3)
        idx = pd.date_range("2020-01-01", periods=400, freq="D")
        R = pd.DataFrame(rng.normal(0, 0.02, (400, 6)), index=idx, columns=range(6))
        pre = R.iloc[:300]
        a = cr.select(pre, list(R.columns))[0]
        R2 = R.copy()
        R2.iloc[300:] = rng.normal(0.05, 0.02, (100, 6))
        b = cr.select(R2.iloc[:300], list(R.columns))[0]
        self.assertEqual(a, b)

    def test_walk_forward_embargo(self):
        rng = np.random.default_rng(4)
        idx = pd.date_range("2017-01-01", periods=1400, freq="D")
        R = pd.DataFrame(rng.normal(0, 0.02, (1400, 4)), index=idx, columns=range(4))
        folds, stitched = cr.walk_forward(R, list(R.columns))
        self.assertTrue(folds)
        for f in folds:
            gap = (pd.Timestamp(f["testStart"]) - pd.Timestamp(f["trainEnd"])).days
            self.assertGreaterEqual(gap, cr.WF_EMBARGO)

    def test_design_pairs_have_bases(self):
        design = cr.make_design()
        keys = {cr._key({k: v for k, v in s.items() if k not in ("id", "label")}) for s in design if not s.get("overlays")}
        for s in design:
            if s.get("overlays"):
                self.assertIn(s["pairBase"], keys, s["label"])

    def test_nowcast_overlay_rules_and_pairs(self):
        idx = pd.date_range("2020-01-01", periods=60, freq="D")
        d = pd.DataFrame({"NC_CPI_MOM_VS_LAST": np.r_[np.full(30, -0.1), np.full(30, 0.2)],
                          "NC_PAYROLLS": np.linspace(300, 100, 60)}, index=idx)
        m1 = cr.overlay_multiplier(cr.nowcast_overlay("NC_CPI_MOM_VS_LAST"), d, idx)
        self.assertEqual(m1.iloc[10], 1.0)
        self.assertEqual(m1.iloc[40], 0.5)            # inflation nowcast above last print -> halve
        m2 = cr.overlay_multiplier(cr.nowcast_overlay("NC_PAYROLLS"), d, idx)
        self.assertEqual(m2.iloc[5], 1.0)             # no 30d history yet -> no signal
        self.assertEqual(m2.iloc[50], 0.5)            # payroll nowcast falling -> halve
        design = cr.make_design(["NC_CPI_MOM_VS_LAST", "NC_PAYROLLS"])
        nc = [s for s in design if s.get("overlays") and s["overlays"][0]["type"] == "nowcast"]
        self.assertEqual(len(nc), 12)

    def test_nowcast_loader_is_point_in_time(self):
        cal = pd.date_range("2024-01-01", periods=10, freq="D")
        frame = pd.DataFrame({"date": cal[:5], "asof": cal[:5], "NC_X": [1.0, 2, 3, 4, 5]})
        import types, sys as _sys
        fake = types.ModuleType("macro_nowcast")
        fake.load_point_in_time = lambda: frame
        _sys.modules["macro_nowcast"] = fake
        try:
            out, note = cr.load_nowcast(cal)
        finally:
            _sys.modules.pop("macro_nowcast", None)
        self.assertIn("NC_X", out.columns)
        self.assertTrue(np.isnan(out["NC_X"].iloc[0]))      # known at the close of day 0 -> usable day 1
        self.assertEqual(out["NC_X"].iloc[1], 1.0)

    def test_multiple_testing_helpers(self):
        p = np.array([0.01, 0.04, 0.03, 0.5])
        self.assertTrue((cr.holm(p) >= p).all())
        self.assertTrue((cr.bh(p) <= cr.holm(p) + 1e-12).all())


class PaperAccountTests(unittest.TestCase):
    def test_twenty_accounts_in_free_range(self):
        ids = [a["id"] for a in cp.accounts()]
        self.assertEqual(ids, list(range(200, 220)))
        existing = set(range(100, 106)) | set(range(110, 114)) | set(range(120, 126))
        self.assertFalse(existing & set(ids))
        labels = [a["label"] for a in cp.accounts()]
        self.assertTrue(any("baseline" in l for l in labels))
        self.assertGreaterEqual(sum(a["dollarAware"] for a in cp.accounts()), 6)

    def test_accounts_are_long_only_and_unlevered(self):
        close, drivers = synthetic()
        for a in cp.accounts():
            W = cr.target_weights(a["spec"], close, drivers)
            self.assertGreaterEqual(W.min().min(), 0.0, a["label"])
            self.assertLessEqual(W.sum(axis=1).max(), 1.0 + 1e-9, a["label"])

    def test_paper_accounts_start_at_10k(self):
        close, drivers = synthetic()
        start = close.index[700]
        for a in cp.accounts()[:5]:
            W = cr.target_weights(a["spec"], close, drivers)
            eq, _ = cr.simulate(close, W, start)
            self.assertLessEqual(eq.iloc[0], 10_000.0 + 1e-6)
            self.assertGreater(eq.iloc[0], 9_800.0)


class SafetyTests(unittest.TestCase):
    def test_no_order_endpoints_in_crypto_code(self):
        t = "trad" + "er"      # split so the repo-wide order-endpoint scan (test_daytrading) stays clean
        bad = re.compile("schwabapi\\.com/" + t + "|/" + t + "/v1|/orders|preview" + "Order|place_" + "order|place" + "Order",
                         re.I)
        for name in ("crypto_data.py", "crypto_research.py", "crypto_paper.py"):
            text = (ENGINE / name).read_text(encoding="utf-8")
            code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            code = re.sub(r'"""[\s\S]*?"""', "", code)
            self.assertIsNone(bad.search(code), name)

    def test_paper_module_never_imports_schwab(self):
        text = (ENGINE / "crypto_paper.py").read_text(encoding="utf-8")
        self.assertNotIn("import schwab", text)
        self.assertNotIn("from schwab", text)

    def test_schwab_client_refuses_trader_paths(self):
        import schwab_data
        with self.assertRaises(PermissionError):
            schwab_data._check_url("https://api.schwabapi.com/" + "trad" + "er/v1/accounts/x/orders")


if __name__ == "__main__":
    unittest.main()
