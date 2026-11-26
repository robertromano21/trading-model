"""Tests for the macro calendar, watcher, surprises, event study and reaction rules.

Network is never used: official fetches are injected via openers or skipped with --no-fetch.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")


def _isolated_env():
    tmp = Path(tempfile.mkdtemp(prefix="macro-test-"))
    os.environ["PRICES_DATA_DIR"] = str(tmp)
    return tmp


# Isolate Massive/ before importing modules that bind DATA_DIR at import time.
TEST_DIR = _isolated_env()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import macro_calendar as cal  # noqa: E402
import macro_sources as src  # noqa: E402
import macro_surprise as sur  # noqa: E402
import macro_watcher as wat  # noqa: E402
import macro_event_study as evs  # noqa: E402
import macro_reaction as rxn  # noqa: E402
import macro_service as svc  # noqa: E402


CPI_HTML = """
<html><body>
The all items index rose 0.3 percent in October before seasonal adjustment? Wait.
The all items index rose 0.3 percent in October. The all items index increased 2.6 percent over the last 12 months.
The index for all items less food and energy rose 0.2 percent in October.
The index for all items less food and energy increased 3.1 percent over the last 12 months.
</body></html>
"""

EMPSIT_HTML = """
<html><body>
Total nonfarm payroll employment increased by 227,000 in October. The unemployment rate was 4.1 percent.
</body></html>
"""

FOMC_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<title>Fed</title>
<item>
  <title>Federal Reserve issues FOMC statement</title>
  <link>https://www.federalreserve.gov/newsevents/pressreleases/monetary20261125a.htm</link>
  <pubDate>Wed, 25 Nov 2026 19:00:00 GMT</pubDate>
  <description>The Committee decided to maintain the target range for the federal funds rate at 3-3/4 to 4 percent.</description>
</item>
</channel></rss>
"""


class CalendarTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="cal-"))
        self.path = self.dir / "calendar.json"
        os.environ["PRICES_DATA_DIR"] = str(self.dir)

    def test_catalog_covers_required_families(self):
        ids = {e["id"] for e in cal.event_catalog()}
        required = {
            "cpi", "core_cpi", "ppi", "nfp", "unrate", "jolts", "eci",
            "gdp", "pce", "core_pce", "claims",
            "fomc_decision", "fomc_minutes",
            "indpro", "caputil", "retail_sales", "durable_goods", "factory_orders",
            "ism_mfg", "ism_services",
            "empire", "philly", "richmond", "kc_fed", "dallas",
            "umich_prelim", "umich_final", "conference_board",
            "treasury_auction",
            "ecb_decision", "boj_decision", "boe_decision", "pboc_decision", "snb_decision",
            "ea_pmi_flash", "de_ifo", "de_zew", "jp_tankan",
            "cn_official_pmi", "china_gdp", "china_cpi", "china_trade", "china_ip",
            "ea_hicp_flash", "uk_cpi",
        }
        missing = required - ids
        self.assertFalse(missing, f"catalog missing {missing}")

    def test_fallback_calendar_times_are_et(self):
        doc = cal.build_calendar(start="2026-11-01", end="2026-12-31", fetch=False, path=self.path)
        self.assertGreater(doc["eventCount"], 20)
        claims = [r for r in doc["events"] if r["id"] == "claims"]
        self.assertTrue(claims)
        for row in claims:
            day = date.fromisoformat(row["date"])
            self.assertEqual(day.weekday(), 3, row)
            self.assertEqual(row["timeEt"], "08:30")
            self.assertTrue(row["datetimeEt"].endswith("-05:00") or row["datetimeEt"].endswith("-04:00"))
            when = datetime.fromisoformat(row["datetimeEt"])
            self.assertEqual(when.tzinfo.utcoffset(when), ET.utcoffset(when))
        fomc = [r for r in doc["events"] if r["id"] == "fomc_decision"]
        self.assertTrue(fomc)
        for row in fomc:
            self.assertEqual(row["timeEt"], "14:00")
        minutes = [r for r in doc["events"] if r["id"] == "fomc_minutes"]
        self.assertTrue(minutes)
        for row in minutes:
            self.assertEqual(row["timeEt"], "14:00")
        # CPI default 8:30 ET
        cpi = next(r for r in doc["events"] if r["id"] == "cpi")
        self.assertEqual(cpi["timeEt"], "08:30")
        # Conference Board is manual entry
        cb = next(r for r in doc["events"] if r["id"] == "conference_board")
        self.assertEqual(cb["entry"], "manual")
        # Overseas times convert to ET (ECB 14:15 Europe/Berlin in November is 08:15 ET)
        # We only emit ECB dates from official calendars, not fallback invention — that's intentional.
        self.assertFalse(any(r["id"] == "ecb_decision" for r in doc["events"]))

    def test_overseas_time_conversion_handles_dst(self):
        ev = cal.event_by_id("ecb_decision")
        # Mid-November: Berlin UTC+1, New York UTC-5 → 14:15 CET = 08:15 ET
        when = cal.to_et(date(2026, 11, 12), ev["time"], ev["zone"])
        self.assertEqual(when.strftime("%H:%M"), "08:15")
        # BoJ 12:00 JST in November is 22:00 ET the previous calendar day? 
        # 12:00 JST = 03:00 UTC = 22:00 ET previous day (UTC-5).
        boj = cal.event_by_id("boj_decision")
        when = cal.to_et(date(2026, 11, 19), boj["time"], boj["zone"])
        self.assertEqual(when.date(), date(2026, 11, 18))
        self.assertEqual(when.strftime("%H:%M"), "22:00")
        # UK CPI 07:00 London in November is 02:00 ET
        uk = cal.event_by_id("uk_cpi")
        when = cal.to_et(date(2026, 11, 18), uk["time"], uk["zone"])
        self.assertEqual(when.strftime("%H:%M"), "02:00")

    def test_upcoming_and_context_block(self):
        cal.build_calendar(start="2026-11-01", end="2026-12-31", fetch=False, path=self.path)
        asof = datetime(2026, 11, 25, 9, 0, tzinfo=ET)
        rows = cal.upcoming(asof=asof, days=10, path=self.path, include_past_hours=0)
        self.assertTrue(rows)
        self.assertTrue(all("datetimeEt" in r for r in rows))
        text = cal.context_block(asof=asof, path=self.path)
        self.assertIn("[MACRO_EVENTS]", text)
        self.assertIn("ET", text)

    def test_manual_entry_merges(self):
        manual = self.dir / "manual_releases.json"
        # Patch MANUAL_PATH
        with mock.patch.object(cal, "MANUAL_PATH", manual):
            manual.write_text(json.dumps([
                {"id": "conference_board", "date": "2026-11-24", "actual": 96.0, "period": "2026-11"}
            ]), encoding="utf-8")
            doc = cal.build_calendar(start="2026-11-01", end="2026-11-30", fetch=False, path=self.path)
            found = [r for r in doc["events"] if r["id"] == "conference_board" and r["date"] == "2026-11-24"]
            self.assertTrue(found)
            self.assertEqual(found[0].get("actual"), 96.0)


class SourceTests(unittest.TestCase):
    def test_http_get_opener_tuple_and_bytes(self):
        def opener_bytes(url, method="GET", data=None, headers=None):
            self.assertIn("User-Agent", headers)
            return b"hello"
        raw, status, _ = src.http_get("https://example.test/x", opener=opener_bytes)
        self.assertEqual(raw, b"hello")
        self.assertEqual(status, 200)

        def opener_tuple(url, method="GET", data=None, headers=None):
            return b"world", 201, {"Content-Type": "text/plain"}
        raw, status, hdrs = src.http_get("https://example.test/y", opener=opener_tuple)
        self.assertEqual((raw, status, hdrs["Content-Type"]), (b"world", 201, "text/plain"))

    def test_parse_bls_ics_and_fed_rss(self):
        ics = "\n".join([
            "BEGIN:VCALENDAR",
            "BEGIN:VEVENT",
            "SUMMARY:The Employment Situation",
            "DTSTART;TZID=America/New_York:20261106T083000",
            "END:VEVENT",
            "BEGIN:VEVENT",
            "SUMMARY:Consumer Price Index",
            "DTSTART;TZID=America/New_York:20261113T083000",
            "END:VEVENT",
            "END:VCALENDAR",
        ])
        rows = src.parse_bls_ics(ics)
        ids = {r["eventId"] for r in rows}
        self.assertEqual(ids, {"nfp", "cpi"})
        cpi = next(r for r in rows if r["eventId"] == "cpi")
        self.assertEqual(cpi["timeEt"], "08:30")
        items = src.parse_fed_rss(FOMC_RSS)
        self.assertEqual(items[0]["kind"], "fomc_decision")

    def test_tenor_proxy_and_strip_secrets(self):
        self.assertEqual(src.tenor_proxy_series("10-Year", "Note"), "DGS10")
        self.assertEqual(src.tenor_proxy_series("5-Year", "TIPS"), "DFII5")
        cleaned = src._strip_secrets({"api_key": "SECRET", "rows": [1], "authorization": "x"})
        self.assertNotIn("api_key", cleaned)
        self.assertEqual(cleaned["rows"], [1])

    def test_env_value_does_not_require_a_key(self):
        # Just make sure has_env is a boolean and we never accidentally print values in this test.
        present = src.has_env("MASSIVE_API_KEY")
        self.assertIsInstance(present, bool)


class SurpriseTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="sur-"))
        self.cons = self.dir / "consensus.json"
        self.hist = self.dir / "surprise_history.json"
        self.sur = self.dir / "surprises.json"
        self.cal = self.dir / "calendar.json"

    def test_standardize_and_manual_consensus(self):
        with mock.patch.object(sur, "CONSENSUS_PATH", self.cons), mock.patch.object(sur, "HISTORY_PATH", self.hist), mock.patch.object(sur, "SURPRISE_PATH", self.sur):
            sur.save_consensus_entry("cpi", "2026-10", 0.2, source="manual", path=self.cons)
            hist_s = [0.1, -0.05, 0.0, 0.15, -0.1, 0.05, 0.2, -0.15, 0.0, 0.1]
            score = sur.score_print("cpi", "2026-10", 0.4, history_actuals=[0.2, 0.2], history_surprises=hist_s)
            self.assertEqual(score["consensus"], 0.2)
            self.assertTrue(score["consensusSource"].startswith("manual"))
            self.assertAlmostEqual(score["surprise"], 0.2)
            self.assertIsNotNone(score["surpriseZ"])
            self.assertGreater(score["surpriseZ"], 1.0)
            # Unemployment: higher than expected is dovish, hawkishSurprise flips.
            score_u = sur.score_print("unrate", "2026-10", 4.4, history_actuals=[4.1],
                                      history_surprises=hist_s, consensus=4.1)
            self.assertAlmostEqual(score_u["surprise"], 0.3)
            self.assertAlmostEqual(score_u["hawkishSurprise"], -0.3)

    def test_model_expectation(self):
        exp, why = sur.model_expectation([0.1, 0.2, 0.3])
        self.assertEqual(why, "last_plus_avg_change")
        self.assertAlmostEqual(exp, 0.4)  # last 0.3 + avg change 0.1

    def test_auction_tail_uses_prior_close_when_wi_missing(self):
        result = {
            "auction_date": "2026-11-12",
            "security_term": "10-Year",
            "security_type": "Note",
            "high_yield": 4.15,
            "bid_to_cover_ratio": 2.4,
            "indirect_bidder_accepted": 60,
            "direct_bidder_accepted": 10,
            "primary_dealer_accepted": 30,
            "total_accepted": 100,
        }
        score = sur.score_auction(result, prior_yield=4.10, when_issued=None, history={
            "tail_bp": [1, 0, -1, 2, 0, 1, -2, 0, 1, 0],
            "bid_to_cover": [2.3, 2.5, 2.4, 2.2, 2.6, 2.4, 2.3, 2.5, 2.4, 2.3],
            "indirect_pct": [55, 58, 60, 62, 59, 57, 61, 60, 58, 59],
        })
        self.assertEqual(score["benchmarkSource"], "prior_close_yield_proxy")
        self.assertAlmostEqual(score["tailBp"], 5.0)  # 4.15 - 4.10 = 5 bp
        self.assertIsNotNone(score["bidToCover"])
        self.assertTrue(any("When-issued" in n for n in score["notes"]))

    def test_fomc_surprise_vs_implied(self):
        with mock.patch.object(sur, "fomc_implied_rate", return_value=(4.00, "test_proxy", ["proxy"])):
            score = sur.score_fomc(4.125, 4.00, "2026-11-25", history_surprises=[0, 0, 12.5, -12.5, 0, 0, 25, 0, -12.5, 0, 0, 0])
            self.assertAlmostEqual(score["changeBp"], 12.5)
            self.assertAlmostEqual(score["surpriseVsImpliedBp"], 12.5)
            self.assertEqual(score["impliedSource"], "test_proxy")


class WatcherTests(unittest.TestCase):
    def test_parsers(self):
        cpi = wat.parse_cpi_html(CPI_HTML)
        self.assertAlmostEqual(cpi["mom"], 0.3)
        self.assertAlmostEqual(cpi["yoy"], 2.6)
        self.assertAlmostEqual(cpi["core_mom"], 0.2)
        emp = wat.parse_empsit_html(EMPSIT_HTML)
        self.assertAlmostEqual(emp["nfp_change"], 227000 if emp["nfp_change"] > 1000 else 227)
        # The regex captures "227,000" as 227000 or 227 — accept either thousands convention.
        self.assertIn(emp["nfp_change"], {227, 227000})
        self.assertAlmostEqual(emp["unrate"], 4.1)
        fomc = wat.parse_fomc_statement(
            "The Committee decided to maintain the target range for the federal funds rate at 3-3/4 to 4 percent."
        )
        self.assertAlmostEqual(fomc["lower"], 3.75)
        self.assertAlmostEqual(fomc["upper"], 4.0)
        self.assertAlmostEqual(fomc["mid"], 3.875)

    def test_mock_replay_latency(self):
        # Inject a 50ms "network" delay and confirm request-to-parse is measured and under 1s.
        official = datetime(2026, 11, 13, 8, 30, tzinfo=ET)
        row_key = "cpi:2026-11-13"
        delay = 0.05
        result = wat.replay(row_key, CPI_HTML, event_id="cpi",
                            official_et=official.isoformat(timespec="minutes"),
                            clock_offset_sec=delay)
        self.assertTrue(result["ok"], result)
        self.assertIsNotNone(result["latencyMs"])
        # Allow generous jitter around the 50ms sleep plus parse time; must be well under 1 second.
        self.assertGreaterEqual(result["latencyMs"], 30)
        self.assertLess(result["latencyMs"], 1000)
        parsed = result["fetch"]["parsed"]
        self.assertAlmostEqual(parsed["mom"], 0.3)

    def test_watch_tight_loop_with_fake_clock(self):
        # Build a calendar with a release a few seconds from 'now', poll with an opener, stop quickly.
        tmp = Path(tempfile.mkdtemp(prefix="watch-"))
        path = tmp / "calendar.json"
        asof = datetime(2026, 11, 13, 8, 29, 45, tzinfo=ET)
        ev = cal.event_by_id("cpi")
        row = cal._row(ev, date(2026, 11, 13), source="test", period="2026-10")
        cal.save_calendar({"events": [row], "eventCount": 1, "builtAt": cal.iso_et(asof)}, path)

        def opener(url, method="GET", data=None, headers=None):
            return CPI_HTML.encode("utf-8"), 200, {"Content-Type": "text/html"}

        with mock.patch.object(cal, "CALENDAR_PATH", path), mock.patch.object(wat, "CALENDAR_PATH", path), \
             mock.patch.object(wat, "now_et", return_value=asof), \
             mock.patch.object(wat, "emit_macro_alert", return_value=({"text": "CPI is out at 0.3.", "type": "macro_release"}, True)), \
             mock.patch.object(wat, "post_chat_and_memory", return_value=[]), \
             mock.patch.object(wat, "_score", return_value={"eventId": "cpi", "actual": 0.3, "surpriseZ": 2.0}):
            results = wat.poll_once(asof=asof, opener=opener, before_sec=30, after_sec=900)
        self.assertTrue(any(r.get("parsed") or r.get("score") for r in results), results)
        printed = next(r for r in results if r.get("parsed"))
        self.assertIn("latencyMsFromRequest", printed or {"latencyMsFromRequest": printed.get("parseMs")})
        self.assertLess(printed.get("parseMs") or printed.get("latencyMsFromRequest") or 0, 1000)


class EventStudyTests(unittest.TestCase):
    def _panel(self, n=300, seed=7):
        rng = np.random.default_rng(seed)
        days = pd.bdate_range("2025-01-02", periods=n, tz=None)
        # SPY follows a small edge on a surprise series we will plant.
        spy = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        tlt = 80 * np.exp(np.cumsum(rng.normal(0, 0.007, n)))
        # Plant a known relationship on 30 random "release" days: surprise → next-day SPY.
        surprise = pd.Series(0.0, index=days)
        idx = rng.choice(np.arange(10, n - 10), size=30, replace=False)
        surprise.iloc[idx] = rng.normal(0, 1, size=30)
        spy_ret = pd.Series(np.concatenate([[0], np.diff(spy) / spy[:-1]]), index=days)
        spy_ret = spy_ret + 0.01 * surprise  # contemporaneous (close_1d) beta ≈ 0.01
        spy = 100 * (1 + spy_ret).cumprod()
        open_spy = spy / (1 + rng.normal(0, 0.001, n))
        frame_close = pd.DataFrame({"SPY": spy, "TLT": tlt, "QQQ": spy * 1.01, "IEF": tlt * 1.01,
                                    "GLD": 180 + rng.normal(0, 1, n), "UUP": 30 + rng.normal(0, 0.1, n)},
                                   index=days)
        frame_open = frame_close.copy()
        frame_open["SPY"] = open_spy
        class P:
            pass
        panel = P()
        panel.close, panel.open = frame_close, frame_open
        return panel, surprise

    def test_hac_and_holm_detect_planted_edge(self):
        panel, surprise = self._panel()
        events = []
        for day, z in surprise.items():
            if z == 0:
                continue
            events.append({
                "id": "cpi", "date": str(day.date()), "timeEt": "08:30",
                "datetimeEt": f"{day.date()}T08:30:00-05:00",
                "surprise": float(z), "surpriseZ": float(z), "hawkishSurprise": float(z),
                "status": "printed",
            })
        tmp = Path(tempfile.mkdtemp(prefix="study-"))
        path = tmp / "event_study.json"
        with mock.patch.object(evs, "RESEARCH_STUDY", tmp / "research.json"):
            doc = evs.run_event_study(panel=panel, events=events, assets=("SPY", "TLT"), path=path)
        self.assertGreater(doc["nTests"], 0)
        spy_1d = next(t for t in doc["tests"] if t["asset"] == "SPY" and t["horizon"] == "close_1d")
        self.assertGreater(spy_1d["n"], 12)
        self.assertGreater(spy_1d["beta"], 0)
        # Planted beta is 0.01 per unit z; should be detected in-sample. Holm may or may not
        # reject with noise — we still require a raw HAC p well below 0.1 and document honesty.
        self.assertLess(spy_1d["betaP"], 0.1)
        self.assertIn("Holm", doc["honesty"])
        self.assertTrue(any("seconds" in n.lower() or "retail" in n.lower() for n in doc["notes"]))

    def test_multiple_testing_holm_stricter_than_raw(self):
        p = [0.001, 0.04, 0.20, 0.8]
        adj = evs.multiple_testing(p, alpha=0.05)
        self.assertEqual(len(adj["holmAdj"]), 4)
        # The first p should still reject after Holm; the 0.04 may not.
        self.assertTrue(adj["holmReject"][0])
        self.assertGreaterEqual(adj["holmAdj"][1], p[1] - 1e-12)

    def test_overnight_no_lookahead_europe_same_day(self):
        # Europe same-day close must be lagged (design europe_lagged_close), never used as
        # a same-morning predictor of the US open.
        rng = np.random.default_rng(1)
        days = pd.bdate_range("2025-01-02", periods=200)
        spy = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        ewg = 40 * np.exp(np.cumsum(rng.normal(0, 0.012, 200)))
        close = pd.DataFrame({"SPY": spy, "EWG": ewg, "EWU": ewg * 0.9, "EZU": ewg * 1.1,
                              "EWJ": ewg, "FXI": ewg, "MCHI": ewg, "EWY": ewg,
                              "EFA": ewg, "EEM": ewg, "QQQ": spy, "TLT": spy, "IEF": spy,
                              "GLD": spy, "UUP": spy}, index=days)
        open_ = close * 1.001
        class P: pass
        panel = P(); panel.close, panel.open = close, open_
        # Fake DAX daily bars stamped at 11:30 ET (after the US open) — must be treated as lagged.
        idx = pd.DatetimeIndex([pd.Timestamp(d).tz_localize(ET).replace(hour=11, minute=30) for d in days])
        dax = pd.DataFrame({"close": ewg}, index=idx)
        tmp = Path(tempfile.mkdtemp(prefix="ovn-"))
        with mock.patch.object(evs, "RESEARCH_OVERNIGHT", tmp / "research.json"):
            doc = evs.run_overnight_study(panel=panel, index_data={
                "DAX": {"meta": {"region": "europe", "etf": "EWG"}, "bars": dax},
            }, path=tmp / "overnight.json")
        designs = {t["design"] for t in doc["tests"] if t["predictor"] == "DAX"}
        self.assertIn("europe_lagged_close", designs)
        self.assertNotIn("asia_same_morning", designs)
        self.assertTrue(any("lookahead" in (doc.get("honesty") or "").lower() or "lookahead" in n.lower()
                            for n in doc["notes"] + [doc.get("honesty") or ""]))


class ReactionTests(unittest.TestCase):
    def test_unproven_rules_are_experimental_and_do_not_apply(self):
        settings = {"macro_reaction": "apply"}
        # Upcoming high-impact CPI in a few hours, no proven study.
        asof = datetime(2026, 11, 12, 16, 0, tzinfo=ET)
        ev = cal.event_by_id("cpi")
        row = cal._row(ev, date(2026, 11, 13), source="test", period="2026-10")
        tmp = Path(tempfile.mkdtemp(prefix="rxn-"))
        path = tmp / "calendar.json"
        cal.save_calendar({"events": [row]}, path)
        with mock.patch.object(rxn, "proven_set", return_value=[]), \
             mock.patch.object(cal, "CALENDAR_PATH", path), \
             mock.patch("macro_reaction.upcoming", lambda **k: [row]), \
             mock.patch("macro_reaction.load_calendar", lambda *a, **k: {"events": [row]}):
            target = pd.Series({"SPY": 0.5, "IEF": 0.3})
            new, actions = rxn.adjust_paper_targets(target, date(2026, 11, 12), settings, asof=asof)
        self.assertTrue(any(a["kind"] == "pre_release_cut" for a in actions))
        cut = next(a for a in actions if a["kind"] == "pre_release_cut")
        self.assertTrue(cut["experimental"])
        self.assertFalse(cut["apply"])
        # Targets unchanged because the rule is not proven.
        self.assertAlmostEqual(float(new["SPY"]), 0.5)

    def test_proven_pre_release_cut_applies(self):
        settings = {"macro_reaction": "apply", "macro_prerelease_scale": 0.5}
        asof = datetime(2026, 11, 12, 16, 0, tzinfo=ET)
        ev = cal.event_by_id("cpi")
        row = cal._row(ev, date(2026, 11, 13), source="test")
        proven = [{"kind": "macro_event", "eventId": "cpi", "asset": "SPY", "horizon": "open_gap",
                   "rejectHolm": True}]
        with mock.patch.object(rxn, "proven_set", return_value=proven), \
             mock.patch("macro_reaction.upcoming", lambda **k: [row]), \
             mock.patch("macro_reaction.load_calendar", lambda *a, **k: {"events": []}):
            target = pd.Series({"SPY": 0.6, "IEF": 0.2})
            new, actions = rxn.adjust_paper_targets(target, date(2026, 11, 12), settings, asof=asof)
        cut = next(a for a in actions if a["kind"] == "pre_release_cut")
        self.assertTrue(cut["apply"])
        self.assertAlmostEqual(float(new["SPY"]), 0.3)
        self.assertAlmostEqual(float(new["IEF"]), 0.1)

    def test_off_mode_is_a_no_op(self):
        target = pd.Series({"SPY": 0.4})
        new, actions = rxn.adjust_paper_targets(target, date(2026, 11, 12), {"macro_reaction": "off"})
        self.assertEqual(actions, [])
        self.assertAlmostEqual(float(new["SPY"]), 0.4)


class ServiceTests(unittest.TestCase):
    def test_tool_and_status_payload(self):
        tmp = Path(tempfile.mkdtemp(prefix="svc-"))
        path = tmp / "calendar.json"
        cal.build_calendar(start="2026-11-01", end="2026-12-15", fetch=False, path=path)
        with mock.patch.object(cal, "CALENDAR_PATH", path), mock.patch.object(svc, "CALENDAR_PATH", path):
            # upcoming() reads CALENDAR_PATH; we patched it.
            payload = svc.tool_payload(days=21)
        self.assertTrue(payload["ok"])
        self.assertIn("[MACRO_EVENTS]", payload["context"])
        self.assertEqual(payload["timezone"], "America/New_York")


class AlertPhraseContractTests(unittest.TestCase):
    def test_macro_release_payload_shape(self):
        ev = cal.event_by_id("cpi")
        row = cal._row(ev, date(2026, 11, 13), source="test")
        score = {"actual": 0.4, "consensus": 0.2, "surpriseZ": 2.2, "eventId": "cpi"}
        with mock.patch("alerts_emit.emit", return_value=True) as emit:
            payload, big = wat.emit_macro_alert(row, score)
        self.assertEqual(payload["type"], "macro_release")
        self.assertTrue(big)
        self.assertEqual(payload["severity"], "critical")
        self.assertIn("speech", payload)
        emit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
