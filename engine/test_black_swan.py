"""Offline tests for the black swan monitor. No live network is used."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")


def _isolated_env():
    tmp = Path(tempfile.mkdtemp(prefix="blackswan-test-"))
    os.environ["PRICES_DATA_DIR"] = str(tmp)
    os.environ.pop("TRADING_BLACKSWAN", None)
    return tmp


TEST_DIR = _isolated_env()
sys.path.insert(0, str(Path(__file__).resolve().parent))

import black_swan as bs  # noqa: E402
import black_swan_feeds as feeds  # noqa: E402
import black_swan_playbook as pb  # noqa: E402
import black_swan_study as study  # noqa: E402


RSS_BBC = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>BBC</title>
<item>
  <title>Russia launches missile strikes on Kyiv</title>
  <link>https://example.test/war</link>
  <pubDate>Tue, 24 Feb 2026 08:00:00 GMT</pubDate>
  <description>An invasion-scale barrage.</description>
</item>
<item>
  <title>Sports score: local team wins</title>
  <link>https://example.test/sports</link>
  <pubDate>Tue, 24 Feb 2026 09:00:00 GMT</pubDate>
</item>
</channel></rss>
"""

ATOM_FED = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Fed</title>
  <entry>
    <title>Federal Reserve issues FOMC statement</title>
    <link href="https://example.test/fed"/>
    <updated>2026-03-18T18:00:00Z</updated>
    <summary>The Committee decided to maintain the target range.</summary>
  </entry>
</feed>
"""

GDELT_ART = json.dumps({"articles": [
    {"title": "Sanctions imposed on Iranian banks", "url": "https://example.test/s", "seendate": "20260224", "tone": -6.2},
    {"title": "Weather turns milder", "url": "https://example.test/w", "seendate": "20260224", "tone": 1.1},
]})

# 30 quiet days then a 7-day spike (values are daily volume).
GDELT_TL = json.dumps({"timeline": [{"date": f"202601{i:02d}", "value": 10 + (i % 3)} for i in range(1, 31)]
                       + [{"date": f"202602{i:02d}", "value": 40 + i} for i in range(1, 8)]})


class FakeOpener:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, url, timeout=25, headers=None):
        self.calls.append(url)
        for key, body in self.mapping.items():
            if key in url:
                if isinstance(body, Exception):
                    raise body
                raw = body.encode("utf-8") if isinstance(body, str) else body
                return raw, "application/xml", "200"
        raise OSError(f"unexpected url {url}")


class FeedTests(unittest.TestCase):
    def test_rss_and_atom_parsers(self):
        items = feeds.parse_feed(RSS_BBC, "bbc_world", "BBC World")
        self.assertEqual(len(items), 2)
        self.assertIn("missile", items[0]["title"].lower())
        self.assertTrue(items[0]["published"].startswith("2026-02-24"))
        atom = feeds.parse_feed(ATOM_FED, "fed_press", "Federal Reserve", source="central_bank")
        self.assertEqual(len(atom), 1)
        self.assertIn("FOMC", atom[0]["title"])

    def test_gdelt_parsers_and_spike(self):
        arts = feeds.parse_gdelt_artlist(GDELT_ART)
        self.assertEqual(len(arts), 2)
        self.assertEqual(arts[0]["source"], "gdelt")
        tl = feeds.parse_gdelt_timeline(GDELT_TL)
        spike = feeds.gdelt_spike(tl, short=7, long=30)
        self.assertTrue(spike["ok"])
        self.assertTrue(spike["spike"])
        self.assertTrue(spike["shockSpike"])
        self.assertGreater(spike["z"], 3)

    def test_gpr_csv_and_features(self):
        lines = ["title row to skip", "DATE,GPRD,GPRD_ACT"]
        # 260 quiet days then an extreme print
        for i in range(260):
            day = datetime(2025, 1, 1) + __import__("datetime").timedelta(days=i)
            # skip a nice weekday series; dates can be any
            lines.append(f"{day.date().isoformat()},{(50 + i % 5)},1")
        lines.append("2025-09-18,400,10")
        rows = feeds._parse_gpr_csv("\n".join(lines))
        self.assertGreater(len(rows), 200)
        feat = feeds.gpr_features(rows)
        self.assertTrue(feat["ok"])
        self.assertTrue(feat["extreme"])
        self.assertGreater(feat["z"], 3)

    def test_collect_headlines_uses_injected_opener(self):
        gpr_csv = "date,gprd\n2025-01-01,50\n2025-01-02,51\n2025-09-18,80\n"
        opener = FakeOpener({
            "ArtList": GDELT_ART,
            "TimelineVol": GDELT_TL,
            "data_gpr_daily_recent.csv": gpr_csv,
            "feeds.bbci.co.uk": RSS_BBC,
            "theguardian.com": RSS_BBC,
            "feeds.npr.org": RSS_BBC,
            "aljazeera.com": RSS_BBC,
            "who.int": RSS_BBC,
            "federalreserve.gov": ATOM_FED,
            "ecb.europa.eu": ATOM_FED,
            "bankofengland.co.uk": ATOM_FED,
            "boj.or.jp": ATOM_FED,
            "ofac.treasury.gov": RSS_BBC,
        })
        bundle = feeds.collect_headlines(opener=opener, root=TEST_DIR / "Research" / "black_swan", use_cache=False)
        self.assertGreater(bundle["status"]["nHeadlines"], 0)
        self.assertTrue(any("missile" in (h.get("title") or "").lower() for h in bundle["headlines"]))
        self.assertTrue(bundle["gdeltSpike"].get("spike"))


class ClassifyTests(unittest.TestCase):
    def test_keyword_types_and_severity(self):
        war = bs.classify_headline({"title": "Russia invades Ukraine in a full-scale invasion"})
        self.assertEqual(war["event_type"], "war")
        self.assertEqual(war["direction"], "negative")
        self.assertGreaterEqual(war["severity"], 5)
        self.assertIn("Ukraine", war["affected_regions"])
        cease = bs.classify_headline({"title": "Ceasefire agreed in Gaza after peace deal talks"})
        self.assertEqual(cease["event_type"], "ceasefire")
        self.assertEqual(cease["direction"], "positive")
        self.assertGreaterEqual(cease["severity"], 4)
        tariff = bs.classify_headline({"title": "White House announces reciprocal tariffs on Chinese goods"})
        self.assertEqual(tariff["event_type"], "trade_war")
        self.assertEqual(tariff["direction"], "negative")
        mild = bs.classify_headline({"title": "Local council debates parking fees"})
        self.assertFalse(mild["relevant"])
        self.assertEqual(mild["event_type"], "other")
        hist = bs.classify_headline({"title": "WHO releases guidelines to help countries during the COVID-19 pandemic"})
        retro = bs.classify_headline({"title": "Russia has lost nearly 500,000 people in the Ukraine war",
                                      "summary": "Leaked data reveal how the full-scale invasion of Ukraine has changed mortality."})
        self.assertLessEqual(retro["severity"], 3)
        ofac = bs.classify_headline({"title": "Cyber-related and Transnational Criminal Organizations Designations; Cuba Designations"})
        self.assertTrue(ofac["severity"] < 4 or ofac.get("event_type") != "cyber")
        self.assertLess(ofac.get("severity") or 0, 4)


        self.assertLess(hist["severity"], 4)
        fresh = {"title": "Russia invades Ukraine", "published": "2026-11-25T12:00:00+00:00", "feed": "bbc_world"}
        stale = {"title": "WHO reports COVID-19 hospital strain as Russia invades Ukraine", "published": "2020-03-16T12:00:00+00:00", "feed": "who_news"}
        batch = bs.classify_headlines([fresh, stale])
        titles = [h["title"] for h in batch["relevant"]]
        self.assertIn(fresh["title"], titles)
        self.assertNotIn(stale["title"], titles)

    def test_batch_lift_from_gdelt_spike(self):
        headlines = [{"title": "Tensions rise after a naval skirmish"}]  # severity 3 war
        news = bs.classify_headlines(headlines, gdelt_spike={"spike": True, "shockSpike": True, "z": 4.0})
        self.assertGreaterEqual(news["newsSeverity"], 4)
        self.assertTrue(news["liftNotes"])

    def test_retort_schema_enums(self):
        enums = bs.CLASSIFY_SCHEMA["parameters"]["properties"]["event_type"]["enum"]
        for t in ("war", "sanctions", "peace_deal", "stimulus", "other"):
            self.assertIn(t, enums)
        self.assertEqual(bs.CLASSIFY_SCHEMA["parameters"]["properties"]["severity"]["enum"], [1, 2, 3, 4, 5])


class ConfirmAndStateTests(unittest.TestCase):
    def setUp(self):
        idx = pd.bdate_range("2024-01-02", periods=80)
        rng = np.random.default_rng(0)
        spy = 100 * np.cumprod(1 + rng.normal(0, 0.007, len(idx)))
        # Last day: a crash + VIX spike + gold/yen bid.
        spy[-1] = spy[-2] * 0.97
        close = pd.DataFrame({
            "SPY": spy,
            "GLD": 180 * np.cumprod(1 + rng.normal(0, 0.005, len(idx))),
            "TLT": 90 * np.cumprod(1 + rng.normal(0, 0.004, len(idx))),
            "USO": 70 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))),
            "HYG": 80 * np.cumprod(1 + rng.normal(0, 0.003, len(idx))),
            "LQD": 110 * np.cumprod(1 + rng.normal(0, 0.002, len(idx))),
        }, index=idx)
        close.loc[idx[-1], "GLD"] = close["GLD"].iloc[-2] * 1.03
        close.loc[idx[-1], "TLT"] = close["TLT"].iloc[-2] * 1.02
        close.loc[idx[-1], "HYG"] = close["HYG"].iloc[-2] * 0.985
        close.loc[idx[-1], "LQD"] = close["LQD"].iloc[-2] * 0.998
        self.close = close
        vix = pd.DataFrame({"vix": np.full(len(idx), 16.0), "vix3m": np.full(len(idx), 18.0)}, index=idx)
        vix.iloc[-1] = {"vix": 32.0, "vix3m": 24.0}
        vix["ratio"] = vix["vix"] / vix["vix3m"]
        vix["z"] = 3.5
        vix["pct"] = 0.99
        vix["peak20"] = vix["vix"].rolling(20, min_periods=1).max()
        self.vix = vix
        self.fx = {"USDJPY": pd.Series(150.0, index=idx)}
        self.fx["USDJPY"].iloc[-1] = 146.0

    def test_confirmation_agrees_with_a_crash(self):
        class P:
            pass
        panel = P()
        panel.close = self.close
        panel.open = self.close.shift(1)
        panel.open.iloc[-1] = self.close.iloc[-2] * 0.98  # gap down
        conf = bs.confirmation_score(inputs={
            "vix": self.vix, "panel": panel, "macro": None, "fx": self.fx,
            "asof": str(self.close.index[-1].date()), "sources": {},
        })
        self.assertGreater(conf["stress"], 0.5)
        self.assertIsNotNone(conf["spyGap"])
        self.assertLess(conf["spyGap"], 0)
        self.assertIn("vix_level", conf["parts"])

    def test_actionable_requires_agreement(self):
        news_strong = {"newsSeverity": 5, "direction": "negative", "eventType": "war"}
        news_weak = {"newsSeverity": 1, "direction": "unclear", "eventType": "other"}
        hot = {"stress": 0.75, "relief": 0.2, "notes": ["VIX 32"], "vixRollover": False}
        calm = {"stress": 0.10, "relief": 0.8, "notes": [], "vixRollover": False}
        th = dict(bs.DEFAULT_THRESHOLDS)
        self.assertEqual(bs.judge(news_strong, hot, th)["flag"], "ACTIONABLE")
        self.assertEqual(bs.judge(news_strong, calm, th)["flag"], "WATCH")  # news only
        self.assertEqual(bs.judge(news_weak, hot, th)["flag"], "WATCH")   # markets only
        self.assertEqual(bs.judge(news_weak, calm, th)["flag"], "CLEAR")

    def test_state_machine_hysteresis(self):
        th = dict(bs.DEFAULT_THRESHOLDS)
        th.update({"shockMinSessions": 1, "shockMaxSessions": 10, "recoverySessions": 3, "hysteresis": 0.12})
        news = {"newsSeverity": 5, "direction": "negative", "eventType": "war"}
        hot = {"stress": 0.80, "relief": 0.1, "vixRollover": False, "notes": []}
        cool = {"stress": 0.20, "relief": 0.7, "vixRollover": False, "notes": []}
        colder = {"stress": 0.05, "relief": 0.9, "vixRollover": False, "notes": []}
        s = bs.default_state()
        s = bs.step_state(s, bs.judge(news, hot, th), hot, news, th, asof="2026-01-02")
        self.assertEqual(s["state"], "SHOCK")
        self.assertTrue(s["transitioned"])
        self.assertTrue(s["policy"]["freezeNewLongs"])
        # A single slightly-cooler print must not flip immediately if still above shock - hysteresis.
        almost = {"stress": 0.50, "relief": 0.4, "vixRollover": False, "notes": []}  # 0.55-0.12=0.43 threshold
        # actually 0.50 > 0.43 so stay in SHOCK
        s2 = bs.step_state(s, bs.judge(news, almost, th), almost, news, th, asof="2026-01-03")
        self.assertEqual(s2["state"], "SHOCK")
        # Cool well below the hysteresis band -> RECOVERY
        s3 = bs.step_state(s2, bs.judge({"newsSeverity": 2, "direction": "negative", "eventType": "war"}, cool, th),
                           cool, {"newsSeverity": 2, "direction": "negative", "eventType": "war"}, th, asof="2026-01-06")
        self.assertEqual(s3["state"], "RECOVERY")
        self.assertTrue(s3["policy"]["reRisk"])
        # After recoverySessions of CLEAR/calm -> NORMAL
        s4 = s3
        for i, day in enumerate(("2026-01-07", "2026-01-08", "2026-01-09"), start=1):
            s4 = bs.step_state(s4, bs.judge({"newsSeverity": 0, "direction": "unclear", "eventType": "other"}, colder, th),
                               colder, {"newsSeverity": 0, "direction": "unclear", "eventType": "other"}, th, asof=day)
        self.assertEqual(s4["state"], "NORMAL")

    def test_positive_shock_goes_to_recovery_and_allows_svix(self):
        th = dict(bs.DEFAULT_THRESHOLDS)
        news = {"newsSeverity": 4, "direction": "positive", "eventType": "ceasefire"}
        conf = {"stress": 0.2, "relief": 0.7, "vixRollover": True, "notes": ["VIX rolling over from a spike"]}
        s = bs.step_state(bs.default_state(), bs.judge(news, conf, th), conf, news, th, asof="2026-02-01")
        self.assertEqual(s["state"], "RECOVERY")
        self.assertTrue(s["policy"]["allowSvix"])
        self.assertFalse(s["policy"]["freezeNewLongs"])


class PlaybookTests(unittest.TestCase):
    def test_log_does_not_change_weights_apply_does(self):
        target = pd.Series({"SPY": 0.6, "TLT": 0.1, "AAPL": 0.3})
        state = {"state": "SHOCK", "direction": "negative",
                 "policy": pb.policy_for("SHOCK", "negative")}
        logged, acts = pb.adjust_paper_targets(target, pd.Timestamp("2026-01-02"),
                                               {"black_swan": "log"}, state=state)
        self.assertTrue(np.allclose(logged.values, target.values))
        self.assertTrue(any(a["kind"] == "black_swan_scale_gross" and a["apply"] is False for a in acts))
        applied, acts2 = pb.adjust_paper_targets(target, pd.Timestamp("2026-01-02"),
                                                 {"black_swan": "apply"}, state=state)
        self.assertLess(float(applied[["SPY", "AAPL"]].abs().sum()), float(target[["SPY", "AAPL"]].abs().sum()))
        self.assertGreater(float(applied.get("TLT", 0)), float(target["TLT"]))
        self.assertTrue(any(a["apply"] for a in acts2))

    def test_freeze_new_long_only_in_apply_shock(self):
        state = {"policy": pb.policy_for("SHOCK", "negative")}
        self.assertTrue(pb.freeze_new_long("AAPL", 0.0, 0.2, {"black_swan": "apply"}, state=state))
        self.assertFalse(pb.freeze_new_long("AAPL", 0.0, 0.2, {"black_swan": "log"}, state=state))
        self.assertFalse(pb.freeze_new_long("VIXY", 0.0, 0.05, {"black_swan": "apply"}, state=state))
        self.assertFalse(pb.freeze_new_long("AAPL", 0.2, 0.1, {"black_swan": "apply"}, state=state))

    def test_daytrade_skip_in_apply_shock(self):
        c = pd.DataFrame({"symbol": ["AAPL"], "session_date": ["2026-01-02"]})
        out, acts = pb.adjust_daytrade_candidates(c, "2026-01-02", {"black_swan": "apply"},
                                                  state={"state": "SHOCK", "policy": pb.policy_for("SHOCK", "negative")})
        self.assertTrue(out["black_swan_skip"].all())
        self.assertTrue(acts[0]["apply"])


class StudyTests(unittest.TestCase):
    def test_forward_paths_and_playbook_walkforward(self):
        idx = pd.bdate_range("2019-01-02", periods=600)
        # A manufactured crash around a known date, then recovery.
        rets = np.full(len(idx), 0.0005)
        crash_day = pd.Timestamp("2020-03-16")
        # find loc
        loc = idx.get_loc(idx[idx >= crash_day][0])
        rets[loc] = -0.08
        rets[loc + 1] = -0.04
        rets[loc + 2:loc + 6] = -0.01
        px = 100 * np.cumprod(1 + rets)
        vix = np.full(len(idx), 15.0)
        vix[loc:loc + 8] = 45.0
        close = pd.DataFrame({"SPY": px, "VIX": vix, "GLD": px * 0 + 180, "TLT": px * 0 + 90,
                              "USO": px * 0 + 70}, index=idx)
        events = ({
            "id": "covid_climax", "name": "COVID", "eventDate": "2020-03-16",
            "firstSession": str(idx[loc].date()), "direction": "negative", "type": "pandemic", "note": "",
        },)
        rows = study.event_windows(close, events, assets=("SPY", "VIX"))
        spy = rows[0]["assets"]["SPY"]
        self.assertLess(spy["r1"], -0.05)
        self.assertLess(spy["maxDrawdown"], -0.05)
        cal, scores, quiet = study.calibrate_thresholds(close, events)
        self.assertTrue(cal["calibrated"])
        self.assertGreater(cal["shockStress"], 0)
        wf = study.walk_forward_playbook(close, events, thresholds=cal, hold=8)
        self.assertTrue(wf["ok"])
        self.assertIn("honesty", wf)
        self.assertGreaterEqual(wf["nConsidered"], 1)

    def test_study_offline_on_synthetic_writes(self):
        idx = pd.bdate_range("2018-01-02", periods=800)
        rng = np.random.default_rng(1)
        px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.009, len(idx)))
        close = pd.DataFrame({"SPY": px, "VIX": 18 + rng.normal(0, 1, len(idx)),
                              "GLD": 170 + rng.normal(0, 1, len(idx)),
                              "TLT": 95 + rng.normal(0, 0.5, len(idx)),
                              "USO": 60 + rng.normal(0, 1, len(idx)),
                              "XLF": px, "XLE": px, "XLK": px, "XLU": px}, index=idx)
        # Only events that fall inside this window
        ev = [e for e in study.EVENTS if pd.Timestamp(e["firstSession"]) >= idx[20]
              and pd.Timestamp(e["firstSession"]) <= idx[-30]]
        # Pin this test's temp dir at call time. Another test module may have
        # overwritten PRICES_DATA_DIR at import, and fetch_and_model.DATA_DIR is
        # frozen on first import (which can be the live Massive folder under discover).
        previous = os.environ.get("PRICES_DATA_DIR")
        os.environ["PRICES_DATA_DIR"] = str(TEST_DIR)
        try:
            doc = study.run_study(use_network=False, write=True, close=close,
                                  coverage={"notes": ["synthetic"], "columns": {}},
                                  events=tuple(ev) or study.EVENTS[:1])
            self.assertTrue((TEST_DIR / "Research" / "black_swan" / "study.json").is_file())
        finally:
            if previous is None:
                os.environ.pop("PRICES_DATA_DIR", None)
            else:
                os.environ["PRICES_DATA_DIR"] = previous
        self.assertIn("honesty", doc)
        self.assertGreaterEqual(doc["nEvents"], 1)


class PollAndKillSwitchTests(unittest.TestCase):
    def test_kill_switch(self):
        out = bs.poll(env={"TRADING_BLACKSWAN": "off"})
        self.assertTrue(out["skipped"])
        old = os.environ.get("TRADING_BLACKSWAN")
        os.environ["TRADING_BLACKSWAN"] = "off"
        try:
            self.assertEqual(bs.main(["poll"]), 0)
        finally:
            if old is None:
                os.environ.pop("TRADING_BLACKSWAN", None)
            else:
                os.environ["TRADING_BLACKSWAN"] = old

    def test_poll_with_fake_feeds_and_chat(self):
        # Quiet market data so we stay CLEAR/WATCH rather than needing sqlite prices.
        idx = pd.bdate_range("2026-01-02", periods=40)
        close = pd.DataFrame({"SPY": np.linspace(100, 102, len(idx))}, index=idx)

        class P:
            pass
        panel = P()
        panel.close = close
        panel.open = close.shift(1)
        vix = pd.DataFrame({"vix": 16.0, "vix3m": 18.0, "ratio": 16 / 18, "z": 0.1, "pct": 0.4, "peak20": 17.0}, index=idx)

        opener = FakeOpener({
            "ArtList": GDELT_ART,
            "TimelineVol": GDELT_TL,
            "data_gpr_daily_recent.csv": "date,gprd\n2026-01-01,50\n2026-02-01,55\n",
            "feeds.bbci.co.uk": RSS_BBC,
            "theguardian.com": RSS_BBC,
            "feeds.npr.org": RSS_BBC,
            "aljazeera.com": RSS_BBC,
            "who.int": RSS_BBC,
            "federalreserve.gov": ATOM_FED,
            "ecb.europa.eu": ATOM_FED,
            "bankofengland.co.uk": ATOM_FED,
            "boj.or.jp": ATOM_FED,
            "ofac.treasury.gov": RSS_BBC,
        })

        real_confirm_inputs = bs.load_confirm_inputs

        def fake_inputs(asof=None, **kwargs):
            return {"vix": vix, "panel": panel, "macro": None, "fx": {}, "asof": "2026-02-24", "sources": {}}

        bs.load_confirm_inputs = fake_inputs
        try:
            result = bs.poll(opener=opener, root=TEST_DIR / "Research" / "black_swan",
                             asof="2026-02-24", use_cache=False, emit=False, env={})
        finally:
            bs.load_confirm_inputs = real_confirm_inputs
        self.assertTrue(result["ok"])
        self.assertIn(result["state"], bs.STATES)
        st = bs.status(TEST_DIR / "Research" / "black_swan")
        self.assertEqual(st["state"]["state"], result["state"])


if __name__ == "__main__":
    unittest.main()
