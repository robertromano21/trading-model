"""Unit tests for engine/attribution.py. Synthetic data only except an optional live-output check."""

import json
import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import attribution as att


def _dates(n=250, start="2025-01-02"):
    return pd.bdate_range(start, periods=n)


def _noise(index, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, scale, len(index)), index=index)


class TestHelpers(unittest.TestCase):
    def test_hac_lags_matches_quant_tools_rule(self):
        self.assertEqual(att.hac_lags(100), 4)
        self.assertEqual(att.hac_lags(391), max(1, int(4 * (391 / 100) ** (2 / 9))))
        self.assertGreaterEqual(att.hac_lags(10), 1)

    def test_returns_from_equity_drops_only_the_opening_nan(self):
        equity = pd.Series([100.0, 101.0, 101.0, 102.01], index=_dates(4))
        rets = att.returns_from_equity(equity)
        self.assertEqual(len(rets), 3)
        self.assertAlmostEqual(rets.iloc[0], 0.01)
        self.assertAlmostEqual(rets.iloc[1], 0.0)

    def test_finite_turns_nan_and_inf_into_none(self):
        out = att._finite({"a": float("nan"), "b": float("inf"), "c": 1.5, "d": [np.int64(3)]})
        self.assertIsNone(out["a"])
        self.assertIsNone(out["b"])
        self.assertEqual(out["c"], 1.5)
        self.assertEqual(out["d"], [3])


class TestRegression(unittest.TestCase):
    def test_ols_recovers_known_alpha_and_beta(self):
        idx = _dates(300)
        spy = _noise(idx, 0.01, seed=1)
        alpha, beta = 0.0004, 0.65
        y = alpha + beta * spy + _noise(idx, 0.002, seed=2)
        fit = att.ols_hac(y, spy.rename("SPY"))
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit["betas"]["SPY"], beta, places=1)
        self.assertAlmostEqual(fit["alphaDaily"], alpha, places=3)
        self.assertGreater(fit["r2"], 0.85)
        self.assertGreater(abs(fit["alphaT"]), 2.0)
        self.assertEqual(fit["hacLags"], att.hac_lags(fit["n"]))

    def test_capm_on_the_market_itself(self):
        idx = _dates(200)
        spy = _noise(idx, 0.01, seed=3) + 0.0005
        capm = att.capm_report(spy, spy)
        self.assertAlmostEqual(capm["beta"], 1.0, places=6)
        self.assertAlmostEqual(capm["alphaAnnual"] or 0.0, 0.0, places=6)
        self.assertAlmostEqual(capm["r2"], 1.0, places=6)
        self.assertAlmostEqual(capm["trackingError"], 0.0, places=6)

    def test_residualize_is_orthogonal_to_the_market(self):
        idx = _dates(200)
        spy = _noise(idx, 0.01, seed=4)
        xl = 0.9 * spy + _noise(idx, 0.004, seed=5)
        resid = att.residualize(xl, spy).dropna()
        self.assertLess(abs(resid.corr(spy.reindex(resid.index))), 0.08)


class TestBlend(unittest.TestCase):
    def test_beta_matched_blend_matches_clipped_beta(self):
        idx = _dates(250)
        spy = _noise(idx, 0.012, seed=6)
        ief = _noise(idx, 0.004, seed=7)
        beta = 0.6
        y = 0.0001 + beta * spy + 0.2 * ief + _noise(idx, 0.003, seed=8)
        out = att.beta_matched_blend(y, spy, ief, None)
        self.assertIsNotNone(out)
        w = out["simple"]["weights"]
        # Clip of estimated beta, remainder in IEF.
        self.assertAlmostEqual(w["SPY"] + w["IEF"] + w["cash"], 1.0, places=6)
        self.assertGreater(w["SPY"], 0.4)
        self.assertLess(w["SPY"], 0.85)
        self.assertAlmostEqual(w["IEF"], 1.0 - w["SPY"], places=6)
        # The blend itself should have a CAPM beta close to w_spy (IEF beta is small, not zero).
        blend = w["SPY"] * spy + w["IEF"] * ief
        capm_blend = att.capm_report(blend, spy)
        self.assertAlmostEqual(capm_blend["beta"], w["SPY"] + w["IEF"] * att.capm_report(ief, spy)["beta"], places=3)


class TestHoldingsAndBrinson(unittest.TestCase):
    def test_weights_forward_fill_from_fill_dates(self):
        dates = _dates(8, "2025-06-02")
        decisions = pd.DataFrame({
            "fill_date": ["2025-06-03", "2025-06-05", "2025-06-05"],
            "ticker": ["SPY", "SPY", "IEF"],
            "target_weight": [0.7, 0.4, 0.3],
            "status": ["filled", "filled", "filled"],
        })
        w = att.weights_from_decisions(decisions, dates)
        self.assertAlmostEqual(w.loc[dates[0], "SPY"], 0.0)
        self.assertAlmostEqual(w.loc[dates[1], "SPY"], 0.7)  # 2025-06-03
        self.assertAlmostEqual(w.loc[dates[2], "SPY"], 0.7)
        self.assertAlmostEqual(w.loc[dates[3], "SPY"], 0.4)  # 2025-06-05
        self.assertAlmostEqual(w.loc[dates[3], "IEF"], 0.3)
        self.assertAlmostEqual(w.loc[dates[-1], "IEF"], 0.3)

    def test_brinson_identity_holds(self):
        idx = _dates(80)
        spy = _noise(idx, 0.01, seed=9) + 0.0004
        xl = 0.8 * spy + _noise(idx, 0.008, seed=10) + 0.0002
        ief = _noise(idx, 0.003, seed=11)
        factors = pd.DataFrame({"SPY": spy, "XLK": xl, "IEF": ief})
        weights = pd.DataFrame({"SPY": 0.4, "XLK": 0.3, "IEF": 0.1}, index=idx)
        groups = att.assign_groups(["SPY", "XLK", "IEF"], factors)
        self.assertEqual(groups["SPY"], "US Market")
        self.assertEqual(groups["XLK"], "Technology")
        self.assertEqual(groups["IEF"], "Intermediate Treasuries")
        result = att.brinson(weights, factors, groups, factors, "spy")
        self.assertIsNotNone(result)
        self.assertLess(result["identityResidualMeanAbs"], 1e-12)
        parts = result["allocationSum"] + result["selectionSum"] + result["interactionSum"]
        self.assertAlmostEqual(parts, result["arithmeticExcessSum"], places=10)
        # With a constant 40% in SPY vs 100% benchmark, allocation to cash/bonds/tech should be nonzero.
        self.assertGreater(abs(result["allocationSum"]), 0.0)

    def test_assign_stock_to_highest_sector_correlation(self):
        idx = _dates(120)
        spy = _noise(idx, 0.01, seed=12)
        xlk = 0.7 * spy + _noise(idx, 0.01, seed=13)
        xle = 0.4 * spy + _noise(idx, 0.012, seed=14)
        stock = 0.9 * xlk + _noise(idx, 0.005, seed=15)
        factors = pd.DataFrame({"SPY": spy, "XLK": xlk, "XLE": xle})
        assets = factors.copy()
        assets["AAAA"] = stock
        groups = att.assign_groups(["AAAA"], factors, assets)
        self.assertEqual(groups["AAAA"], "Technology")


class TestVerdictAndTool(unittest.TestCase):
    def test_verdict_flags_a_pure_market_mix_as_mostly_market(self):
        capm = {"alphaAnnual": 0.01, "alphaT": 0.4, "alphaP": 0.69, "beta": 0.7, "betaT": 12.0, "r2": 0.82,
                "trackingError": 0.06, "informationRatio": 0.15}
        blend = {"simple": {"weights": {"SPY": 0.7, "IEF": 0.3, "cash": 0.0},
                            "strategy": {"sharpe": 1.2}, "blend": {"sharpe": 1.15}}}
        v = att.verdict("TSMOM core", capm, blend, None, "")
        self.assertEqual(v["label"], "mostly_market")
        self.assertTrue(v["isCheaperIndexFund"])

    def test_verdict_flags_significant_positive_alpha(self):
        capm = {"alphaAnnual": 0.08, "alphaT": 2.4, "alphaP": 0.016, "beta": 0.5, "betaT": 6.0, "r2": 0.4,
                "trackingError": 0.09, "informationRatio": 0.7}
        v = att.verdict("Skill sleeve", capm, None, None, "")
        self.assertEqual(v["label"], "significant_alpha")
        self.assertFalse(v["isCheaperIndexFund"])

    def test_tool_filters_series_and_handles_missing_file(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            payload = {
                "generatedAt": "2026-11-25 21:00",
                "methods": {},
                "data": {},
                "series": {
                    "backtest.core_tsmom": {"available": True, "label": "TSMOM"},
                    "walkforward.tsmom": {"available": True, "label": "TSMOM OOS"},
                    "daytrade.full_defaults": {"available": True, "label": "DT"},
                },
                "headlines": {"backtest.core_tsmom": "hello"},
                "paper": {"available": False, "reason": "none"},
                "caveats": [],
            }
            path = Path(tmp) / "attribution.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            out = att.tool_attribute_returns({"out_dir": tmp, "strategy": "tsmom", "source": "backtest"})
            self.assertIn("backtest.core_tsmom", out["series"])
            self.assertNotIn("walkforward.tsmom", out["series"])
            self.assertNotIn("daytrade.full_defaults", out["series"])

    def test_markdown_section_mentions_tsmom_and_caveats(self):
        attr = {
            "data": {"sampleNote": "short sample"},
            "series": {
                "backtest.core_tsmom": {
                    "available": True,
                    "label": "TSMOM core (full-period backtest)",
                    "period": {"sessions": 100},
                    "capm": {"beta": 0.6, "r2": 0.7, "alphaAnnual": 0.02, "alphaT": 0.8,
                             "trackingError": 0.07, "informationRatio": 0.2},
                    "performance": {"sharpe": 1.2},
                    "betaMatchedBlend": {"simple": {"blend": {"sharpe": 1.1}, "strategy": {"sharpe": 1.2},
                                                    "weights": {"SPY": 0.6, "IEF": 0.4},
                                                    "cagr": None}},
                    "verdict": {"label": "mostly_market", "summary": "Mostly a lower-beta mix.", "reasons": ["beta 0.60"]},
                    "sources": {"contributions": {"alphaAnnual": 0.02, "marketAnnual": 0.10,
                                                  "sectorTiltsAnnual": 0.0, "bondsAnnual": 0.01, "otherAnnual": 0.0}},
                }
            },
            "paper": {"available": False, "reason": "No paper equity history yet."},
            "caveats": ["short sample"],
        }
        md = att.markdown_section(attr)
        self.assertIn("## Return attribution", md)
        self.assertIn("TSMOM core", md)
        self.assertIn("mostly_market", md)
        self.assertIn("No paper equity history yet", md)


class TestPaperEmpty(unittest.TestCase):
    def test_empty_paper_database_is_a_clear_miss(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "paper.sqlite3"
            with sqlite3.connect(db) as con:
                con.execute("CREATE TABLE equity_history (account TEXT, date TEXT, equity REAL, daily_return REAL)")
                con.execute("CREATE TABLE fills (account TEXT, fill_date TEXT, ticker TEXT, shares REAL, price REAL, notional REAL)")
                con.execute("CREATE TABLE signal_state (account TEXT, asof TEXT, ticker TEXT, target_weight REAL, sleeve TEXT)")
            result, err = att.load_paper_returns(db)
            self.assertIsNone(err)
            self.assertFalse(result["available"])
            self.assertIn("No paper equity history", result["reason"])



class TestPanelAdapters(unittest.TestCase):
    def test_panel_report_maps_verdict_and_market_fields(self):
        row = {
            "name": "backtest.core_tsmom",
            "available": True,
            "period": {"start": "2025-05-07", "end": "2026-11-24", "sessions": 391},
            "performance": {"sharpe": 1.45, "cagr": 0.14},
            "capm": {"beta": 0.62, "betaT": 10.0, "r2": 0.71, "alphaAnnual": 0.02, "alphaT": 0.8,
                     "alphaP": 0.42, "trackingError": 0.07, "informationRatio": 0.25, "n": 391, "hacLags": 6},
            "sources": {"full": {"alphaAnnual": 0.015, "alphaT": 0.6, "r2": 0.8, "n": 391,
                                 "betas": {"SPY": 0.6, "XLK": 0.1, "IEF": 0.2},
                                 "betaT": {"SPY": 8.0, "XLK": 2.0, "IEF": 3.0}},
                        "contributions": {"alphaAnnual": 0.015}},
            "betaMatchedBlend": {"simple": {"weights": {"SPY": 0.62, "IEF": 0.38, "cash": 0.0},
                                            "strategy": {"sharpe": 1.45, "cagr": 0.14},
                                            "blend": {"sharpe": 1.3, "cagr": 0.12},
                                            "informationRatio": 0.2}},
            "brinson": {"vsSpy": {"sessions": 391, "allocationSum": 0.04, "selectionSum": -0.01,
                                  "interactionSum": 0.005, "byGroup": []}},
            "verdict": {"label": "mostly_market", "summary": "Mostly a lower-beta mix.", "reasons": ["R2 0.71"],
                        "isCheaperIndexFund": True},
        }
        report = att._to_panel_report(row)
        self.assertEqual(report["name"], "core_tsmom")
        self.assertEqual(report["verdict"]["label"], "cheap_index_fund")
        self.assertTrue(report["verdict"]["isCheaperIndexFund"])
        self.assertAlmostEqual(report["market"]["spyBeta"], 0.62)
        self.assertAlmostEqual(report["market"]["alphaAnnual"], 0.02)
        self.assertEqual(report["sectorLoadings"][0]["ticker"], "IEF")
        self.assertIn("annualized", report["brinson"])
        self.assertIsNotNone(report["brinson"]["annualized"]["allocation"])
        self.assertAlmostEqual(report["betaMatched"]["cashMix"]["sharpe"], 1.3)

    def test_daytrade_hook_skips_a_short_series_without_loading_prices(self):
        daily = pd.DataFrame({"session_date": pd.bdate_range("2026-01-02", periods=10),
                              "return": [0.0] * 10, "equity": [10000] * 10})
        report = att.attribute_daytrade_result(daily, name="daytrader")
        self.assertEqual(report["name"], "daytrader")
        self.assertIn("only 10 sessions", report["skipped"])


if __name__ == "__main__":
    unittest.main()
