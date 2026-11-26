"""Tests for engine/analyst_scoring.py (scorecard, chance tests, stops, ingestion) and the advisory gate."""
import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import TestCase, main

import numpy as np
import pandas as pd

import analyst_scoring as sc
from market_data import Panel

ETFS = ("SPY",) + sc.SECTOR_ETFS


def synthetic(n_stocks=120, n_days=320, seed=3, drifts=()):
    """Random-walk panel on real NYSE sessions. drifts: [(ticker, start_row, days, daily_drift)]."""
    closes = sc.session_closes("2025-01-01", "2026-12-31")
    dates = closes.index[:n_days]
    rng = np.random.default_rng(seed)
    tickers = list(ETFS) + [f"S{i:03d}" for i in range(n_stocks)]
    rets = pd.DataFrame(rng.normal(0.0004, 0.012, (n_days, len(tickers))), index=dates, columns=tickers)
    for t, start, days, drift in drifts:
        rets.iloc[start + 1:start + 1 + days, tickers.index(t)] += drift
    close = 100 * (1 + rets).cumprod()
    opening = close.shift().fillna(close)
    high, low = np.maximum(opening, close) * 1.005, np.minimum(opening, close) * 0.995
    volume = pd.DataFrame(1e6, index=dates, columns=tickers)
    classes = pd.Series({t: ("etf" if t in ETFS else "stock") for t in tickers}, dtype=object)
    zero = pd.DataFrame(0.0, index=dates, columns=tickers)
    return Panel(opening, high, low, close, volume, zero, classes, None), closes


def issued_intraday(closes, day):
    return (closes[day] - pd.Timedelta(hours=2)).isoformat().replace("+00:00", "Z")


class Helpers(TestCase):
    def test_wilson_and_holm(self):
        lo, hi = sc.wilson(15, 30)
        self.assertAlmostEqual(lo, 0.3315, places=3)
        self.assertAlmostEqual(hi, 0.6685, places=3)
        self.assertEqual(sc.wilson(0, 0), (None, None))
        self.assertEqual(sc.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_effective_sample_drops_overlap(self):
        rows = [{"ticker": "A", "anchor_row": r} for r in (10, 12, 31)] + [{"ticker": "B", "anchor_row": 12}]
        kept = sc.effective_sample(rows, 20)
        self.assertEqual(sorted((r["ticker"], r["anchor_row"]) for r in kept), [("A", 10), ("A", 31), ("B", 12)])

    def test_anchor_is_tradable_close(self):
        closes = sc.session_closes("2026-11-20", "2026-12-05")
        # intraday on Tue 2026-11-24 -> that day's close; after the close -> next session (Wed 11-25)
        self.assertEqual(str(sc.anchor_session("2026-11-24T18:00:00Z", None, closes).date()), "2026-11-24")
        self.assertEqual(str(sc.anchor_session("2026-11-24T22:30:00Z", None, closes).date()), "2026-11-25")
        # Thanksgiving 11-26 closed; 11-27 is an early close (18:00 UTC): 19:00 UTC on 11-27 -> Mon 11-30
        self.assertEqual(str(sc.anchor_session("2026-11-27T19:00:00Z", None, closes).date()), "2026-11-30")
        self.assertEqual(str(sc.anchor_session(None, "2026-11-26", closes).date()), "2026-11-27")

    def test_sic_mapping(self):
        self.assertEqual(sc.sic_to_etf(3674), "XLK")
        self.assertEqual(sc.sic_to_etf(2834), "XLV")
        self.assertEqual(sc.sic_to_etf(6022), "XLF")
        self.assertEqual(sc.sic_to_etf(1311), "XLE")
        self.assertEqual(sc.sector_for("GOOGL", {"GOOGL": 7370}, {}, {}), ("XLC", "known"))
        self.assertEqual(sc.sector_for("ZZZ", {}, {"ZZZ": "XLU"}, {}), ("XLU", "correlation"))
        self.assertEqual(sc.sector_for("XLK", {}, {}, {})[0], None)


class Scoring(TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)
        self.dbs = []

    def tearDown(self):
        for db in self.dbs:
            db.close()
        self.tmp.cleanup()

    def open(self):
        db = sc.connect(self.dir / "s.sqlite3")
        self.dbs.append(db)
        return db

    def card(self, panel, closes, db):
        return sc.build_scorecard(db, panel=panel, closes=closes, sector_overrides={}, sics={}, permutations=2000)

    def test_no_calls_means_gate_closed(self):
        panel, closes = synthetic(n_stocks=20, n_days=80)
        db = self.open()
        doc = self.card(panel, closes, db)
        self.assertEqual(doc["gateOpen"], [])
        self.assertIn("No flag has a completed", doc["promptSummary"])
        self.assertIn("do NOT shrink", doc["promptSummary"])

    def test_predictive_red_opens_gate_random_green_does_not(self):
        rng = np.random.default_rng(11)
        reds = [(f"S{i:03d}", 60 + 2 * i, 20, -0.01) for i in range(40)]
        panel, closes = synthetic(drifts=reds)
        db = self.open()
        for t, row, _, _ in reds:
            day = panel.dates[row]
            sc.log_call(db, ticker=t, flag="RED", issued_at=issued_intraday(closes, day), source="LLM",
                        stop=float(panel.close[t].iat[row]) * 0.95)
        for i in range(40):
            t, row = f"S{40 + i:03d}", int(rng.integers(30, 220))
            sc.log_call(db, ticker=t, flag="GREEN", issued_at=issued_intraday(closes, panel.dates[row]), source="LLM")
        # FALLBACK verdicts are logged but never scored for the gate
        sc.log_call(db, ticker="S100", flag="RED", issued_at=issued_intraday(closes, panel.dates[50]), source="FALLBACK")
        db.commit()
        doc = self.card(panel, closes, db)
        cell = {(c["flag"], c["horizon"], c["benchmark"]): c for c in doc["cells"]}
        red20 = cell[("RED", 20, "spy")]
        self.assertEqual(red20["n"], 40)
        self.assertLess(red20["meanExcess"], -0.1)
        self.assertGreater(red20["hitRate"], red20["baseRate"])
        self.assertTrue(red20["passes"], red20)
        self.assertTrue(doc["gate"]["RED"]["20"])
        self.assertFalse(doc["gate"]["GREEN"]["20"])
        self.assertFalse(any(doc["gate"]["YELLOW"].values()))
        self.assertEqual(doc["counts"]["scorable"], 80)
        # stops 5% below on names that fell ~18% are hit
        stop20 = [s for s in doc["stops"] if s["horizon"] == 20][0]
        self.assertEqual(stop20["n"], 40)
        self.assertGreaterEqual(stop20["hits"], 38)
        # sector benchmark is reported too
        self.assertGreater(cell[("RED", 20, "sector")]["n"], 0)
        self.assertIn("RED@20d", doc["promptSummary"])

    def test_small_sample_stays_closed_even_if_perfect(self):
        reds = [(f"S{i:03d}", 60 + 2 * i, 20, -0.02) for i in range(10)]
        panel, closes = synthetic(drifts=reds)
        db = self.open()
        for t, row, _, _ in reds:
            sc.log_call(db, ticker=t, flag="RED", issued_at=issued_intraday(closes, panel.dates[row]), source="LLM")
        db.commit()
        doc = self.card(panel, closes, db)
        red20 = [c for c in doc["cells"] if c["flag"] == "RED" and c["horizon"] == 20 and c["benchmark"] == "spy"][0]
        self.assertEqual(red20["n"], 10)
        self.assertFalse(red20["passes"])
        self.assertIn("< 30", red20["reason"])
        self.assertEqual(doc["gateOpen"], [])

    def test_pending_when_window_incomplete(self):
        panel, closes = synthetic(n_stocks=20, n_days=80)
        db = self.open()
        sc.log_call(db, ticker="S001", flag="GREEN", issued_at=issued_intraday(closes, panel.dates[70]), source="LLM")
        sc.log_call(db, ticker="S002", flag="GREEN", asof=str((panel.dates[-1] + pd.Timedelta(days=3)).date()), source="LLM")
        db.commit()
        doc = self.card(panel, closes, db)
        rows = db.execute("SELECT o.horizon_days, o.status FROM call_outcomes o JOIN analyst_calls c USING (call_id) "
                          "WHERE c.ticker='S001'").fetchall()
        self.assertEqual(dict((r[0], r[1]) for r in rows), {5: "scored", 20: "pending", 60: "pending"})
        self.assertEqual(doc["counts"]["outcomes"]["5"], {"scored": 1, "pending": 1})


class Ingest(TestCase):
    def test_backfill_from_ai_store_and_advisory_file_dedupes(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            ai = sqlite3.connect(d / "ai.sqlite3")
            ai.execute("""CREATE TABLE ai_verdicts (ticker TEXT, created_at TEXT, asof TEXT, source TEXT, model TEXT,
                flag TEXT, conviction TEXT, conviction_score REAL, action TEXT, horizon TEXT, trend TEXT, overextension TEXT,
                overextension_score REAL, fundamental_health TEXT, opportunity TEXT, size_hint TEXT, ref_price REAL,
                entry_low REAL, entry_high REAL, limit_price REAL, order_type TEXT, stop REAL, take_profit REAL, atr REAL,
                diversification TEXT, clamps TEXT, payload TEXT)""")
            ai.execute("CREATE TABLE ai_log (id INTEGER PRIMARY KEY, at TEXT, kind TEXT, model TEXT, status TEXT, input TEXT, output TEXT)")
            ai.execute("INSERT INTO ai_verdicts (ticker, created_at, asof, source, model, flag, conviction_score, ref_price, stop, horizon, payload) "
                       "VALUES ('NVDA','2026-11-24T15:00:00.000Z','2026-11-24','LLM','m','RED',0.8,150,140,'POSITION_20_60D','{}')")
            ai.execute("INSERT INTO ai_verdicts (ticker, created_at, asof, source, model, flag, payload) "
                       "VALUES ('AAPL','2026-11-24T15:00:00.000Z','2026-11-24','FALLBACK',NULL,'YELLOW','{}')")
            ai.execute("INSERT INTO ai_log VALUES (1,'2026-11-24T16:00:00.000Z','ask','m','ok',?,?)",
                       (json.dumps({"question": "q", "tickers": ["NVDA"]}),
                        json.dumps({"stance": "GREEN", "levels": [{"type": "stop", "price": 141}]})))
            ai.execute("INSERT INTO ai_log VALUES (2,'2026-11-24T16:00:00.000Z','ask','m','ok',?,?)",
                       (json.dumps({"question": "q", "tickers": ["NVDA", "AMD"]}), json.dumps({"stance": "RED"})))
            ai.commit()
            ai.close()
            adv = d / "adv.json"
            adv.write_text(json.dumps({"asof": "2026-11-24", "verdicts": {"NVDA": {
                "flag": "RED", "stop": 140, "source": "ai:m", "created_at": "2026-11-24T15:00:00.000Z"}}}))
            db = sc.connect(d / "s.sqlite3")
            added, warnings = sc.ingest_all(db, ai_research=d / "ai.sqlite3", advisory_json=adv, paper_db=d / "none.sqlite3")
            self.assertEqual((added, warnings), (3, []))
            rows = {r["call_id"]: dict(r) for r in db.execute("SELECT * FROM analyst_calls")}
            self.assertEqual(rows["verdict|NVDA|2026-11-24T15:00:00.000Z"]["stop"], 140)
            self.assertEqual(rows["verdict|AAPL|2026-11-24T15:00:00.000Z"]["scorable"], 0)
            self.assertEqual(rows["ask|NVDA|2026-11-24T16:00:00.000Z"]["stop"], 141)
            self.assertEqual(sc.ingest_all(db, ai_research=d / "ai.sqlite3", advisory_json=adv)[0], 0)   # idempotent
            db.close()


class Gate(TestCase):
    def write(self, path, gate, age_days=0):
        generated = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat().replace("+00:00", "Z")
        path.write_text(json.dumps({"schema": "analyst_scorecard/v1", "generatedAt": generated, "gate": gate,
                                    "cells": [], "params": {"minN": 30}}))
        sc._GATE_CACHE.clear()

    def test_fails_closed_and_opens_only_when_proven(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "card.json"
            self.assertFalse(sc.gate_open("RED", path=path)[0])                       # missing
            path.write_text("{not json")
            sc._GATE_CACHE.clear()
            self.assertFalse(sc.gate_open("RED", path=path)[0])                       # malformed
            self.write(path, {"RED": {"5": False, "20": True, "60": False}}, age_days=10)
            self.assertFalse(sc.gate_open("RED", path=path)[0])                       # stale
            self.write(path, {"RED": {"5": False, "20": True, "60": False}})
            self.assertTrue(sc.gate_open("RED", (20, 60), path=path)[0])
            self.assertFalse(sc.gate_open("RED", (5,), path=path)[0])                 # day trader horizon
            self.assertFalse(sc.gate_open("YELLOW", (20, 60), path=path)[0])

    def test_advisory_gated_adjust_entry(self):
        from advisory import Verdict, gated_adjust_entry
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            path = Path(d) / "card.json"
            old = os.environ.get("ANALYST_SCORECARD_PATH")
            os.environ["ANALYST_SCORECARD_PATH"] = str(path)
            try:
                sc._GATE_CACHE.clear()
                amount, note, ok, why = gated_adjust_entry(Verdict("A", "RED"), 0.2, 100)
                self.assertEqual((amount, note, ok), (0.0, "advisory_red", False))
                self.assertIn("log-only", why)
                self.assertEqual(gated_adjust_entry(Verdict("A", "GREEN"), 0.2, 100)[:3], (0.2, None, False))
                self.write(path, {"RED": {"20": True}, "YELLOW": {"20": False}})
                self.assertTrue(gated_adjust_entry(Verdict("A", "RED"), 0.2, 100)[2])
                self.assertFalse(gated_adjust_entry(Verdict("A", "YELLOW"), 0.2, 100)[2])
                self.assertFalse(gated_adjust_entry(Verdict("A", "RED"), 0.2, 100, horizons=(5,))[2])
            finally:
                if old is None:
                    os.environ.pop("ANALYST_SCORECARD_PATH", None)
                else:
                    os.environ["ANALYST_SCORECARD_PATH"] = old


if __name__ == "__main__":
    main()
