"""Tests for analyst scoring, the live-only gate, and the sealed historical replay."""

import json
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np

ENGINE = Path(__file__).resolve().parent
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from analyst_score import (
    benjamini_hochberg, bootstrap_mean_p, brier_score, build_scorecard, decide_gate,
    evaluate_calls, forward_return, harvest_advisory_json, harvest_ai_research, harvest_memory,
    holm, live_effect_allowed, log_call, newey_west_t, permutation_mean_p, rescore,
    session_calendar, spearman_ic, stop_hit,
)
from analyst_replay import (
    PitLeak, assert_point_in_time, build_bundle, compute_features_pit, filter_pit,
    heuristic_verdict, leakage_report, run_replay,
)
from advisory import Verdict, adjust_entry


def _make_prices(path, tickers, start="2020-01-02", sessions=400, seed=1, drifts=None):
    """Write a daily_prices sqlite with a weekday calendar and optional per-ticker daily drifts."""
    dates = [d.strftime("%Y-%m-%d") for d in
             __import__("pandas").bdate_range(start, periods=sessions)]
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    db.execute("CREATE INDEX daily_prices_by_date ON daily_prices (date, ticker)")
    rng = np.random.default_rng(seed)
    drifts = drifts or {}
    for ticker in tickers:
        price = 100.0
        mu = drifts.get(ticker, 0.0003)
        for d in dates:
            ret = mu + rng.normal(0, 0.01)
            o = price
            price = max(1.0, price * (1 + ret))
            hi, lo = max(o, price) * 1.01, min(o, price) * 0.99
            db.execute("INSERT INTO daily_prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (ticker, d, o, hi, lo, price, 1_000_000))
    db.commit()
    return db, dates


def _calls(db, rows):
    for r in rows:
        log_call(db, **r)


class StatsTests(TestCase):
    def test_newey_west_detects_a_positive_mean(self):
        rng = np.random.default_rng(0)
        x = 0.02 + rng.normal(0, 0.01, 80)
        nw = newey_west_t(x, horizon=5)
        self.assertGreater(nw["t"], 0)
        self.assertLess(nw["p"], 0.01)

    def test_bootstrap_and_permutation_agree_on_a_clear_signal(self):
        x = np.full(40, 0.03)
        boot = bootstrap_mean_p(x, alternative="greater", n_boot=500, block=5, seed=1)
        perm = permutation_mean_p(x, alternative="greater", n_perm=500, seed=1)
        self.assertLess(boot["p"], 0.05)
        self.assertLess(perm["p"], 0.05)
        self.assertAlmostEqual(boot["mean"], 0.03, places=6)

    def test_permutation_is_not_significant_for_zero_mean_noise(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 0.02, 80)
        perm = permutation_mean_p(x, alternative="greater", n_perm=800, seed=2)
        self.assertGreater(perm["p"], 0.05)

    def test_holm_and_bh_control_a_known_family(self):
        # One tiny p, two large: Holm should reject only the first at alpha=0.05.
        family = [("a", 0.001), ("b", 0.20), ("c", 0.80)]
        h = holm(family, alpha=0.05)
        self.assertTrue(h["a"]["reject"])
        self.assertFalse(h["b"]["reject"])
        self.assertFalse(h["c"]["reject"])
        self.assertLessEqual(h["a"]["pAdj"], 0.05)
        bh = benjamini_hochberg(family, alpha=0.05)
        self.assertTrue(bh["a"]["reject"])
        self.assertFalse(bh["c"]["reject"])

    def test_brier_is_0_25_for_a_coin_flip(self):
        # Predicted 0.5, outcomes half 0 half 1.
        y = [0, 1, 0, 1]
        s = brier_score([0.5, 0.5, 0.5, 0.5], y)
        self.assertAlmostEqual(s["brier"], 0.25, places=6)

    def test_information_coefficient_recovers_rank_agreement(self):
        scores = [1, 1, 1, 0, 0, -1, -1, -1, 1, -1]
        returns = [0.1, 0.08, 0.05, 0.0, 0.01, -0.06, -0.09, -0.04, 0.07, -0.08]
        ic = spearman_ic(scores, returns, n_perm=400, seed=3)
        self.assertGreater(ic["ic"], 0.6)
        self.assertLess(ic["p"], 0.05)


class ForwardReturnAndStopTests(TestCase):
    def test_forward_return_and_stop_hit_on_synthetic_bars(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "prices.sqlite3"
            db, dates = _make_prices(path, ["AAA", "SPY"], sessions=80, seed=4, drifts={"AAA": 0.0, "SPY": 0.0})
            # Overwrite AAA so the path is deterministic: 10 days flat at 100, then a drop through 90.
            db.execute("DELETE FROM daily_prices WHERE ticker='AAA'")
            for i, day in enumerate(dates):
                close = 100.0 if i < 10 else (80.0 if i >= 12 else 100 - 10 * (i - 9))
                db.execute("INSERT INTO daily_prices VALUES ('AAA', ?, ?, ?, ?, ?, 1)",
                           (day, close, close + 1, close - 1, close))
            db.commit()
            calendar = dates
            closes = {r[0]: r[1] for r in db.execute("SELECT date, close FROM daily_prices WHERE ticker='AAA'")}
            fwd = forward_return(calendar, closes, dates[0], 5)
            self.assertEqual(fwd["start"], dates[0])
            self.assertEqual(fwd["end"], dates[5])
            self.assertAlmostEqual(fwd["ret"], 0.0, places=6)
            rows = [dict(r) for r in db.execute("SELECT date, open, high, low, close FROM daily_prices WHERE ticker='AAA'")]
            missed = stop_hit(calendar, rows, dates[0], stop=90, horizon=5)
            self.assertFalse(missed["hit"])
            hit = stop_hit(calendar, rows, dates[9], stop=90, horizon=20)
            self.assertTrue(hit["hit"])
            self.assertGreaterEqual(hit["daysToHit"], 1)
            db.close()


class HarvestAndScoreTests(TestCase):
    def test_harvest_json_research_and_memory(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            score = sqlite3.connect(Path(d) / "score.sqlite3")
            score.row_factory = sqlite3.Row
            from analyst_score import SCHEMA
            score.executescript(SCHEMA)
            js = d / "verdicts.json"
            js.write_text(json.dumps({
                "asof": "2024-06-03", "generatedAt": "2024-06-03T20:00:00Z",
                "verdicts": {"AAA": {"flag": "GREEN", "opportunity": "BREAKOUT", "stop": 90,
                                     "ref_price": 100, "source": "LLM", "asof": "2024-06-03"}}
            }))
            n, w = harvest_advisory_json(score, js)
            self.assertGreaterEqual(n, 3)  # FLAG + STOP + OPPORTUNITY
            self.assertEqual(w, [])
            research = sqlite3.connect(d / "ai_research.sqlite3")
            research.execute("""CREATE TABLE ai_verdicts (ticker, created_at, asof, source, model, flag, opportunity,
                                stop, ref_price, payload)""")
            research.execute("INSERT INTO ai_verdicts VALUES (?,?,?,?,?,?,?,?,?,?)",
                             ("BBB", "2024-06-04T10:00:00Z", "2024-06-04", "LLM", "gpt-test", "RED", "NONE",
                              80, 110, json.dumps({"ticker": "BBB", "flag": "RED"})))
            research.commit()
            research.close()
            n, w = harvest_ai_research(score, d / "ai_research.sqlite3")
            self.assertGreaterEqual(n, 2)
            mem = sqlite3.connect(d / "ai_memory.sqlite3")
            mem.execute("""CREATE TABLE memory_items (id INTEGER PRIMARY KEY, kind, ticker, ts, text, meta, hash, created_at,
                           embedded INTEGER, model_key)""")
            mem.execute("INSERT INTO memory_items VALUES (1,'VERDICT','CCC','2024-06-05T00:00:00Z',?,?,?, 't',0,NULL)",
                        ("Flagged YELLOW hold", json.dumps({"flag": "YELLOW", "refPrice": 50, "source": "LLM"}), "hash1"))
            mem.commit()
            mem.close()
            n, w = harvest_memory(score, d / "ai_memory.sqlite3")
            self.assertGreaterEqual(n, 1)
            flags = {r["ticker"] for r in score.execute("SELECT ticker FROM calls WHERE kind='FLAG'")}
            self.assertEqual(flags, {"AAA", "BBB", "CCC"})
            score.close()

    def test_evaluate_green_outperformance_and_red_underperformance(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "prices.sqlite3"
            # WIN rises, LOSE falls, SPY / UUU (universe stand-in) are flat-ish.
            db, dates = _make_prices(path, ["WIN", "LOSE", "SPY", "UUU"], sessions=200, seed=5,
                                     drifts={"WIN": 0.004, "LOSE": -0.004, "SPY": 0.0002, "UUU": 0.0002})
            # Decision dates with 60 days of future.
            asofs = dates[80:80 + 40]
            score = sqlite3.connect(Path(d) / "score.sqlite3")
            from analyst_score import SCHEMA
            score.executescript(SCHEMA)
            score.row_factory = sqlite3.Row
            for i, day in enumerate(asofs):
                log_call(score, "WIN", day, f"{day}T00:00:00Z", "FLAG", flag="GREEN", source="LLM", origin="live")
                log_call(score, "LOSE", day, f"{day}T00:00:00Z", "FLAG", flag="RED", source="LLM", origin="live")
            score.commit()
            rows = [dict(r) for r in score.execute("SELECT * FROM calls")]
            calendar = session_calendar(db, "SPY")
            block = evaluate_calls(rows, calendar, db, sector_map={}, universe_tickers=["UUU"],
                                   min_n=30, alpha=0.05, origin="live", seed=5)
            g20 = block["flags"]["GREEN"]["20"]
            r20 = block["flags"]["RED"]["20"]
            self.assertGreaterEqual(g20["n"], 30)
            self.assertGreater(g20["meanExcess"], 0)
            self.assertTrue(g20["beatsChance"])
            self.assertLess(r20["meanExcess"], 0)
            self.assertTrue(r20["beatsChance"])
            ic = block["informationCoefficient"]["20"]
            self.assertGreater(ic["ic"], 0.3)
            gate = decide_gate(block, min_n=30, alpha=0.05, horizon=20)
            self.assertTrue(gate["redMaySkip"])
            self.assertTrue(gate["yellowMayShrink"])
            self.assertTrue(gate["open"])
            db.close()
            score.close()

    def test_gate_stays_closed_when_the_sample_is_too_small(self):
        block = {"flags": {
            "GREEN": {"20": {"n": 4, "sufficient": False, "meanExcess": 0.2, "pMin": 0.001}},
            "RED": {"20": {"n": 3, "sufficient": False, "meanExcess": -0.2, "pMin": 0.001}},
            "YELLOW": {"20": {"n": 0, "sufficient": False}},
        }, "multipleTesting": {"holm": {}}}
        gate = decide_gate(block, min_n=30, alpha=0.05)
        self.assertFalse(gate["open"])
        self.assertFalse(gate["redMaySkip"])
        self.assertIn("Insufficient", gate["reason"])

    def test_prior_does_not_open_the_live_gate(self):
        # A strong prior plus an empty/tiny live record must leave the gate closed.
        strong = {"flags": {
            "GREEN": {"20": {"n": 80, "sufficient": True, "meanExcess": 0.05,
                             "permutation": {"p": 0.001}, "beatsChance": True}},
            "RED": {"20": {"n": 80, "sufficient": True, "meanExcess": -0.05,
                           "permutation": {"p": 0.001}, "beatsChance": True}},
            "YELLOW": {"20": {"n": 10, "sufficient": False}},
        }, "multipleTesting": {"holm": {
            "GREEN-20d-outperform": {"p": 0.001, "pAdj": 0.002, "reject": True},
            "RED-20d-underperform": {"p": 0.001, "pAdj": 0.002, "reject": True},
        }}}
        # The prior would pass decide_gate if it were (wrongly) used as live.
        self.assertTrue(decide_gate(strong)["open"])
        live = {"flags": {
            "GREEN": {"20": {"n": 1, "sufficient": False, "meanExcess": 0.4, "pMin": 0.01}},
            "RED": {"20": {"n": 1, "sufficient": False, "meanExcess": -0.4, "pMin": 0.01}},
            "YELLOW": {"20": {"n": 0}},
        }, "multipleTesting": {"holm": {}}}
        card = build_scorecard(live, prior=strong, asof="2026-11-25")
        self.assertFalse(card["live"]["gate"]["open"])
        self.assertFalse(card["prior"]["gate"]["open"])
        self.assertIn("informative starting point", card["prior"]["gate"]["reason"])
        self.assertFalse(live_effect_allowed(Verdict("AAA", "RED"), "advisory_red", card["live"]["gate"]))
        self.assertFalse(live_effect_allowed(Verdict("AAA", "YELLOW"), "advisory_yellow", card["live"]["gate"]))

    def test_live_gate_authorizes_only_the_earned_actions(self):
        open_gate = {"redMaySkip": True, "yellowMayShrink": False, "bandMayDefer": False}
        self.assertTrue(live_effect_allowed(Verdict("A", "RED"), "advisory_red", open_gate))
        self.assertFalse(live_effect_allowed(Verdict("A", "YELLOW"), "advisory_yellow", open_gate))
        self.assertFalse(live_effect_allowed(Verdict("A", "GREEN", entry_low=90, entry_high=95),
                                             "advisory_above_band", open_gate))
        # Raw adjust_entry still describes the rule; the caller decides whether to apply it.
        amount, note = adjust_entry(Verdict("A", "RED"), 0.2, 100)
        self.assertEqual((amount, note), (0.0, "advisory_red"))
        if not live_effect_allowed(Verdict("A", "RED"), note, {"redMaySkip": False}):
            amount = 0.2
        self.assertEqual(amount, 0.2)


class PitAndReplayTests(TestCase):
    def test_filter_and_assert_drop_future_rows(self):
        asof = "2024-06-03"
        news = [{"date": "2024-06-01", "headline": "old"}, {"date": "2024-06-10", "headline": "FUTURE LEAK"}]
        self.assertEqual([n["headline"] for n in filter_pit(news, asof)], ["old"])
        funds = [{"periodEnd": "2024-03-31", "revenue": 1}, {"periodEnd": "2024-09-30", "revenue": 99}]
        self.assertEqual(len(filter_pit(funds, asof)), 1)

    def test_bundle_is_sealed_and_masking_strips_identity(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "prices.sqlite3"
            db, dates = _make_prices(path, ["NVDA", "SPY"], sessions=200, seed=6)
            asof = dates[120]
            # Inject a future bar after asof — build_bundle must not include it.
            future = dates[150]
            self.assertGreater(future, asof)
            news = [{"date": dates[100], "headline": "past filing"},
                    {"date": future, "headline": "earnings beat tomorrow"}]
            bundle = build_bundle(db, "NVDA", asof, news=news, fundamentals=[{"periodEnd": future, "eps": 9}],
                                  name="A well-known chip company", sector="XLK", masked=True, salt="test")
            assert_point_in_time(bundle, asof, real_ticker="NVDA")
            self.assertTrue(bundle["masked"])
            self.assertIsNone(bundle["name"])
            self.assertIsNone(bundle["sector"])
            self.assertTrue(bundle["ticker"].startswith("NAME_"))
            self.assertNotIn("NVDA", json.dumps(bundle))
            self.assertNotIn("chip company", json.dumps(bundle).lower())
            self.assertTrue(all(b["date"] <= asof for b in bundle["bars"]))
            self.assertTrue(all(b["date"] <= asof for b in bundle["benchmarkBars"]))
            self.assertEqual(bundle["news"][0]["headline"], "past filing")
            self.assertEqual(len(bundle["news"]), 1)
            self.assertEqual(bundle["fundamentals"], [])
            # A leaky bundle must fail the assertion.
            leaky = dict(bundle)
            leaky["bars"] = list(bundle["bars"]) + [{"date": future, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}]
            with self.assertRaises(PitLeak):
                assert_point_in_time(leaky, asof, real_ticker="NVDA")
            named = dict(bundle)
            named["sector"] = "XLK"
            with self.assertRaises(PitLeak):
                assert_point_in_time(named, asof, real_ticker="NVDA")
            db.close()

    def test_heuristic_uses_only_bundle_features(self):
        bundle = {"asof": "2024-06-03", "ticker": "NAME_ABC", "masked": True,
                  "features": {"price": 100, "atr14": 2, "trend": "UP", "overextension": "OVERSOLD", "tsmom100": 1}}
        v = heuristic_verdict(bundle)
        self.assertEqual(v["flag"], "GREEN")
        self.assertEqual(v["opportunity"], "OVERSOLD_REVERSAL")
        down = heuristic_verdict({**bundle, "features": {**bundle["features"], "trend": "STRONG_DOWN", "tsmom100": -1,
                                                         "overextension": "OVEREXTENDED"}})
        self.assertEqual(down["flag"], "RED")

    def test_replay_writes_a_prior_that_cannot_open_the_gate(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            prices_path = d / "prices.sqlite3"
            db, dates = _make_prices(prices_path, ["AAA", "BBB", "CCC", "SPY"], sessions=250, seed=8,
                                     drifts={"AAA": 0.003, "BBB": -0.003, "CCC": 0.0004, "SPY": 0.0003})
            db.close()
            prices = sqlite3.connect(prices_path)
            prices.row_factory = sqlite3.Row
            cache = sqlite3.connect(Path(d) / "cache.sqlite3")
            cache.row_factory = sqlite3.Row
            from analyst_replay import CACHE_SCHEMA
            cache.executescript(CACHE_SCHEMA)
            score = sqlite3.connect(Path(d) / "score.sqlite3")
            from analyst_score import SCHEMA
            score.executescript(SCHEMA)
            score.row_factory = sqlite3.Row
            cfg = {"n_dates": 6, "n_tickers": 3, "mask_fraction": 0.5, "max_calls": 40, "min_history": 80,
                   "held_out_after_days": 60, "lookback": 200, "seed": 8, "model": "heuristic"}
            # First run: all uncached. Second run should be all cache hits (cost cap not consumed).
            doc = run_replay(prices=prices, cfg=cfg, cache=cache, score_db=score,
                             tickers=["AAA", "BBB", "CCC"],
                             prior_path=Path(d) / "prior.json", ingest_calls=True)
            self.assertTrue((d / "prior.json").is_file())
            self.assertFalse(doc["prior"]["gate"]["open"])
            self.assertIn("never open", doc["prior"]["gate"]["reason"])
            first_new = doc["prior"]["sample"]["newCalls"]
            self.assertGreater(first_new, 0)
            doc2 = run_replay(prices=prices, cfg=cfg, cache=cache, score_db=score,
                              tickers=["AAA", "BBB", "CCC"],
                              prior_path=Path(d) / "prior.json")
            self.assertEqual(doc2["prior"]["sample"]["newCalls"], 0)
            self.assertGreater(doc2["prior"]["sample"]["cachedCalls"], 0)
            # Leakage report shape is present even when the sample is small.
            self.assertIn("ic20Masked", doc["leakage"])
            # Build a live scorecard that includes this prior and confirm the live gate stays closed
            # because we have no live calls.
            live_block = {"flags": {"GREEN": {"20": {"n": 0, "sufficient": False}},
                                    "RED": {"20": {"n": 0, "sufficient": False}},
                                    "YELLOW": {"20": {"n": 0}}}, "multipleTesting": {"holm": {}}}
            card = build_scorecard(live_block, prior=doc["prior"], asof=dates[-1])
            self.assertFalse(card["live"]["gate"]["open"])
            self.assertFalse(live_effect_allowed(Verdict("AAA", "RED"), "advisory_red", card["live"]["gate"]))
            prices.close()
            cache.close()
            score.close()

    def test_leakage_warning_when_unmasked_ic_is_much_higher(self):
        def block(ic):
            return {"informationCoefficient": {"20": {"ic": ic}}, "flags": {}, "calls": {"FLAG": 20}}
        report = leakage_report(block(0.02), block(0.25))
        self.assertIsNotNone(report["warning"])
        self.assertIn("leakage", report["warning"].lower())
        quiet = leakage_report(block(0.10), block(0.12))
        self.assertIsNone(quiet["warning"])


class RescoreHookTests(TestCase):
    def test_rescore_on_empty_sources_writes_a_closed_gate_card(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            prices_path = d / "prices.sqlite3"
            db, dates = _make_prices(prices_path, ["SPY"], sessions=30, seed=9)
            db.close()
            card = rescore(asof=dates[-1], prices_path=prices_path, score_db=d / "score.sqlite3",
                           scorecard_path=d / "card.json", ingest_memory=False)
            self.assertTrue((d / "card.json").is_file())
            self.assertFalse(card["live"]["gate"]["open"])
            self.assertIn("Insufficient", card["live"]["gate"]["reason"])




class EnrichLiveScorecardTests(TestCase):
    def test_enrich_does_not_open_the_live_gate(self):
        from analyst_score import enrich_live_scorecard
        doc = {"schema": "analyst_scorecard/v1", "gate": {"RED": {"20": False}, "GREEN": {"20": False}, "YELLOW": {"20": False}},
               "gateOpen": [], "cells": [], "recent": [], "promptSummary": "No flag type has beaten chance yet."}
        # A strong prior on disk must not be copied into gate / gateOpen.
        with TemporaryDirectory(ignore_cleanup_errors=True) as d:
            prior = Path(d) / "prior.json"
            prior.write_text(json.dumps({
                "prior": {"flags": {"GREEN": {"20": {"n": 80, "beatsChance": True, "meanExcess": 0.1}}},
                          "informationCoefficient": {"20": {"ic": 0.4, "p": 0.001, "n": 80}}},
                "leakage": {"ic20Masked": 0.05, "ic20Unmasked": 0.06}
            }))
            out = enrich_live_scorecard(doc, prior_path=prior, store_path=Path(d) / "missing.sqlite3")
        self.assertEqual(out["gateOpen"], [])
        self.assertFalse(out["gate"]["RED"]["20"])
        self.assertIn("prior", out)
        self.assertFalse(out["prior"]["gate"]["open"])
        self.assertIn("information coefficient", out["promptSummary"].lower())
