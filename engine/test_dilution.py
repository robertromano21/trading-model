"""Dilution flags: split handling, thresholds, offering parsing (fixtures only), buy block, point-in-time."""

from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
FIXTURES = ROOT / "engine" / "test" / "fixtures" / "dilution"

import dilution as D


def share(end, val, filed):
    return {"end": end, "val": val, "filed": filed}


class ShareChangeTests(unittest.TestCase):
    def test_ten_percent_is_red_five_to_ten_is_warning(self):
        self.assertEqual(D.classify_growth(0.10), "DILUTION")
        self.assertEqual(D.classify_growth(0.12), "DILUTION")
        self.assertEqual(D.classify_growth(0.099), "DILUTION_WARNING")
        self.assertEqual(D.classify_growth(0.05), "DILUTION_WARNING")
        self.assertIsNone(D.classify_growth(0.049))
        self.assertIsNone(D.classify_growth(0.0))
        self.assertIsNone(D.classify_growth(-0.08))
        cfg = D.config_from_settings({"dilution_red_threshold": 0.15, "dilution_warn_threshold": 0.08})
        self.assertEqual(D.classify_growth(0.12, cfg), "DILUTION_WARNING")
        self.assertEqual(D.classify_growth(0.15, cfg), "DILUTION")

    def test_known_split_is_ignored(self):
        splits = [{"date": "2026-03-15", "from": 1, "to": 2}]
        # 200 after a 2-for-1 vs 100 a year earlier is not dilution.
        change = D.yoy_share_change(200, 100, splits, "2026-06-30", "2025-06-30")
        self.assertEqual(change, 0.0)

    def test_exact_2x_3x_4x_10x_without_split_record_is_ignored(self):
        for ratio in (2, 3, 4, 10):
            change = D.yoy_share_change(100 * ratio, 100, splits=None, current_end="2026-06-30", prior_end="2025-06-30")
            self.assertEqual(change, 0.0, msg=f"{ratio}x should look like a split")
        # 10% issuance on top of a 2-for-1, no split record, is 2.2x — not an exact split ratio.
        change = D.yoy_share_change(220, 100, None, "2026-06-30", "2025-06-30")
        self.assertAlmostEqual(change, 1.2, places=6)

    def test_split_plus_ten_percent_issuance_is_red(self):
        splits = [{"date": "2026-01-10", "from": 1, "to": 2}]
        change = D.yoy_share_change(220, 100, splits, "2026-06-30", "2025-06-30")
        self.assertAlmostEqual(change, 0.10, places=6)
        self.assertEqual(D.classify_growth(change), "DILUTION")

    def test_basic_shares_used_when_diluted_missing(self):
        basic = [
            share("2025-06-30", 100, "2025-08-01"),
            share("2026-06-30", 112, "2026-08-01"),
        ]
        growth = D.share_growth_asof([], splits=None, asof="2026-11-01", alt_series=basic)
        self.assertEqual(growth["source"], "basic")
        self.assertAlmostEqual(growth["shareChangeYoY"], 0.12, places=6)

    def test_split_in_window_prefers_basic_shares(self):
        # Weighted-average diluted shares only partly reflect a mid-year 2-for-1 (1.5x).
        diluted = [
            share("2025-06-30", 100, "2025-08-01"),
            share("2026-06-30", 150, "2026-08-01"),
        ]
        basic = [
            share("2025-06-30", 100, "2025-08-01"),
            share("2026-06-30", 200, "2026-08-01"),  # exact split on the instant
        ]
        splits = [{"date": "2026-01-15", "from": 1, "to": 2}]
        growth = D.share_growth_asof(diluted, splits, "2026-11-01", alt_series=basic)
        self.assertEqual(growth["source"], "basic")
        self.assertEqual(growth["shareChangeYoY"], 0.0)


class PointInTimeTests(unittest.TestCase):
    def test_uses_filing_date_not_period_end(self):
        # Period ended 2025-12-31, but the 10-K was filed 2026-02-15.
        series = [
            share("2024-12-31", 100, "2025-02-10"),
            share("2025-12-31", 130, "2026-02-15"),  # 30% jump — must not be visible on 2026-01-31
        ]
        early = D.share_growth_asof(series, asof="2026-01-31")
        self.assertIsNone(early)  # only one observation is filed by Jan 31
        later = D.share_growth_asof(series, asof="2026-02-15")
        self.assertIsNotNone(later)
        self.assertAlmostEqual(later["shareChangeYoY"], 0.30, places=6)
        self.assertEqual(later["current"]["filed"], "2026-02-15")

    def test_apply_to_weights_does_not_use_future_filings(self):
        # Flag becomes active on 2026-02-15 (filing date). A Jan 15 buy must go through.
        weights = pd.DataFrame(
            {"XYZ": [0.0, 0.2, 0.2, 0.2], "NEW": [0.0, 0.0, 0.2, 0.2]},
            index=pd.to_datetime(["2026-01-02", "2026-01-15", "2026-02-16", "2026-03-02"]),
        )
        payload = {
            "tickers": {
                "XYZ": {"asof": "2026-02-15", "flags": ["DILUTION"], "shareChangeYoY": 0.12},
                "NEW": {"asof": "2026-02-15", "flags": ["DILUTION"], "shareChangeYoY": 0.12},
            }
        }
        lookup = D.flag_lookup_from_payload(payload)
        out = D.apply_to_weights(weights, lookup)
        # Pre-flag entry on Jan 15 is allowed; after the filing we do not force a sale.
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-01-15"), "XYZ"]), 0.2)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-02-16"), "XYZ"]), 0.2)
        # Still-flat names cannot open a new position after the filing date.
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-01-15"), "NEW"]), 0.0)
        self.assertAlmostEqual(float(out.loc[pd.Timestamp("2026-02-16"), "NEW"]), 0.0)


class OfferingParseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((FIXTURES / "submissions_offering.json").read_text(encoding="utf-8"))
        cls.atm_text = (FIXTURES / "atm_supplement.txt").read_text(encoding="utf-8")

    def test_s3asr_424b5_and_8k_302_are_offerings_plain_8k_is_not(self):
        filings = D.parse_submissions(self.payload)
        forms = {f["form"]: f for f in filings if f["form"] != "8-K"}
        self.assertTrue(D.is_offering_filing(forms["S-3ASR"]))
        self.assertFalse(D.is_offering_filing({"form": "424B5", "items": "", "description": "424B5"}))
        self.assertTrue(D.is_offering_filing({"form": "424B5", "items": "", "description": "424B5 FORWARD PROSUPP"}))
        eight_ks = [f for f in filings if f["form"] == "8-K"]
        with_item = next(f for f in eight_ks if "3.02" in str(f["items"]))
        earnings = next(f for f in eight_ks if "2.02" in str(f["items"]))
        self.assertTrue(D.is_offering_filing(with_item))
        self.assertFalse(D.is_offering_filing(earnings))
        self.assertFalse(D.is_offering_filing({"form": "10-Q", "items": "", "description": "Quarterly report"}))
        self.assertFalse(D.is_offering_filing({"form": "S-8", "items": "", "description": "Employee plan"}))
        # Routine mega-cap debt / automatic shelves are not share dilution.
        self.assertFalse(D.is_offering_filing({"form": "424B2", "items": "", "description": "424B2"}))
        self.assertFalse(D.is_offering_filing({"form": "424B2", "items": "", "description": "Prospectus supplement for senior notes"}))
        self.assertFalse(D.is_offering_filing({"form": "S-3ASR", "items": "", "description": "S-3ASR"}))
        self.assertTrue(D.is_offering_filing({"form": "424B2", "items": "", "description": "Prospectus for common stock"}))
        self.assertTrue(D.is_offering_filing({"form": "S-3ASR", "items": "", "description": "Automatic shelf registration of common stock"}))

    def test_window_and_asof_are_point_in_time(self):
        # On 2026-11-25, the 2025 S-1 is outside the 30-day window; earnings 8-K is not an offering.
        hits = D.offerings_asof(self.payload, asof="2026-11-25", window_days=30)
        forms = sorted(h["form"] + ":" + h["filingDate"] for h in hits)
        self.assertIn("S-3ASR:2026-11-10", forms)
        self.assertIn("8-K:2026-11-05", forms)
        self.assertNotIn("8-K:2026-10-20", forms)  # earnings 8-K, and also > 30 days from? 2026-10-20 is 36 days before 11-25
        self.assertTrue(any(h["form"] == "424B5" for h in hits) or "424B5:2026-09-15" not in forms)
        # 424B5 on 2026-09-15 is well outside 30 days.
        self.assertFalse(any(h["form"] == "424B5" for h in hits))
        # Before the November filings, nothing in the 30-day window (S-1 is 2025).
        early = D.offerings_asof(self.payload, asof="2026-08-20", window_days=30)
        self.assertEqual(early, [])

    def test_atm_language_in_provided_text(self):
        filing = {"form": "424B5", "filingDate": "2026-11-20", "items": "", "description": "Prospectus supplement",
                  "accession": "0001", "primaryDocument": "atm.htm"}
        # Form already counts, but ATM language is recorded when text is supplied.
        hits = D.offerings_asof([filing], asof="2026-11-25", window_days=30,
                                document_texts={"0001": self.atm_text})
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0]["atm"])
        # An 8-K without 3.02 becomes an offering if the cheaply provided text mentions an ATM program.
        eightk = {"form": "8-K", "filingDate": "2026-11-20", "items": "1.01", "description": "Entry into agreement",
                  "accession": "0002", "primaryDocument": "d8k.htm"}
        self.assertFalse(D.is_offering_filing(eightk))
        self.assertTrue(D.is_offering_filing(eightk, self.atm_text))


class BuyBlockTests(unittest.TestCase):
    def test_new_buy_blocked_on_red_or_offering_not_on_warning_alone(self):
        red = D.evaluate("XYZ", "2026-11-25",
                         share_series=[share("2025-09-30", 100, "2025-11-01"), share("2026-09-30", 112, "2026-11-01")])
        self.assertIn("DILUTION", red["flags"])
        blocked, reason = D.blocks_new_buy(red, has_position=False)
        self.assertTrue(blocked)
        self.assertEqual(reason, "dilution")

        warn = D.evaluate("XYZ", "2026-11-25",
                          share_series=[share("2025-09-30", 100, "2025-11-01"), share("2026-09-30", 107, "2026-11-01")])
        self.assertEqual(warn["flags"], ["DILUTION_WARNING"])
        blocked, reason = D.blocks_new_buy(warn, has_position=False)
        self.assertFalse(blocked)
        self.assertIsNone(reason)

        offer = D.evaluate("XYZ", "2026-11-25",
                           share_series=[share("2025-09-30", 100, "2025-11-01"), share("2026-09-30", 101, "2026-11-01")],
                           filings={"filings": {"recent": {"form": ["S-3"], "filingDate": ["2026-11-10"],
                                                           "items": [""], "primaryDocDescription": ["shelf"],
                                                           "accessionNumber": ["1"], "primaryDocument": ["s.htm"],
                                                           "reportDate": ["2026-11-10"]}}})
        self.assertIn("DILUTION_OFFERING", offer["flags"])
        blocked, reason = D.blocks_new_buy(offer, has_position=False)
        self.assertTrue(blocked)
        self.assertEqual(reason, "dilution_offering")

    def test_existing_holding_logs_warning_and_does_not_force_sell(self):
        red = {"flags": ["DILUTION"], "shareChangeYoY": 0.12}
        blocked, reason = D.blocks_new_buy(red, has_position=True)
        self.assertFalse(blocked)
        self.assertEqual(reason, "dilution_warning")
        blocked, reason = D.blocks_new_buy(red, has_position=True, config={"dilution_exit": True})
        self.assertTrue(blocked)
        self.assertEqual(reason, "dilution_exit")

    def test_apply_to_weights_blocks_new_entries_allows_exits_caps_adds(self):
        # Day 0 is clean so HELD can be established; the flag starts on 2026-01-15.
        weights = pd.DataFrame(
            {"OK": [0.2, 0.2, 0.2, 0.0], "HOT": [0.0, 0.0, 0.2, 0.3], "HELD": [0.2, 0.2, 0.3, 0.1]},
            index=pd.to_datetime(["2026-01-02", "2026-01-16", "2026-02-02", "2026-03-02"]),
        )

        def lookup(day, ticker):
            asof = pd.Timestamp(day)
            if ticker in {"HOT", "HELD"} and asof >= pd.Timestamp("2026-01-15"):
                return {"flags": ["DILUTION"], "asof": "2026-01-15"}
            return {"flags": []}

        out = D.apply_to_weights(weights, lookup)
        # OK is never flagged.
        self.assertAlmostEqual(float(out["OK"].iloc[0]), 0.2)
        # HOT starts flat, so post-flag entries are blocked.
        self.assertAlmostEqual(float(out["HOT"].iloc[0]), 0.0)
        self.assertAlmostEqual(float(out["HOT"].iloc[1]), 0.0)
        self.assertAlmostEqual(float(out["HOT"].iloc[2]), 0.0)
        self.assertAlmostEqual(float(out["HOT"].iloc[3]), 0.0)
        # HELD was already 20% before the flag: the add to 30% is capped, the later cut to 10% is allowed.
        self.assertAlmostEqual(float(out["HELD"].iloc[0]), 0.2)
        self.assertAlmostEqual(float(out["HELD"].iloc[1]), 0.2)
        self.assertAlmostEqual(float(out["HELD"].iloc[2]), 0.2)
        self.assertAlmostEqual(float(out["HELD"].iloc[3]), 0.1)

    def test_verdict_clamp_red_and_weak(self):
        result = {"flags": ["DILUTION"], "shareChangeYoY": 0.12, "offerings": []}
        verdict, clamps = D.apply_to_verdict(
            {"flag": "GREEN", "action": "BUY", "fundamentalHealth": "STRONG",
             "entry": {"orderType": "LIMIT", "bandLow": 10, "bandHigh": 11, "limitPrice": 10.5},
             "risks": []},
            result,
        )
        self.assertEqual(verdict["flag"], "RED")
        self.assertEqual(verdict["action"], "WATCH")
        self.assertEqual(verdict["fundamentalHealth"], "WEAK")
        self.assertEqual(verdict["entry"]["orderType"], "NONE")
        self.assertTrue(any(r["kind"] == "DILUTION" and "12.0%" in r["detail"] for r in verdict["risks"]))
        self.assertTrue(any("RED" in c for c in clamps))


class NoNetworkScanTests(unittest.TestCase):
    def test_scan_uses_injected_fetch_and_never_needs_live_sec(self):
        facts = {
            "facts": {
                "us-gaap": {
                    "WeightedAverageNumberOfDilutedSharesOutstanding": {
                        "units": {"shares": [
                            {"form": "10-Q", "start": "2025-07-01", "end": "2025-09-30", "filed": "2025-11-01", "val": 100},
                            {"form": "10-Q", "start": "2026-07-01", "end": "2026-09-30", "filed": "2026-11-01", "val": 112},
                        ]}
                    }
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {
                        "units": {"shares": [
                            {"form": "10-Q", "end": "2025-09-30", "filed": "2025-11-01", "val": 100},
                            {"form": "10-Q", "end": "2026-09-30", "filed": "2026-11-01", "val": 112},
                        ]}
                    }
                },
            }
        }
        submissions = json.loads((FIXTURES / "submissions_offering.json").read_text(encoding="utf-8"))

        def fetch(url, headers=None):
            self.assertIn("User-Agent", headers or {})
            if "company_tickers" in url:
                return {"0": {"ticker": "DILU", "cik_str": 1234567}}
            if "companyfacts" in url:
                return facts
            if "submissions" in url:
                return submissions
            raise AssertionError("unexpected url " + url)

        # Isolate flags output so we do not touch the live Massive/Research file.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            old = os_env_swap(tmp)
            try:
                payload = D.scan(tickers=["DILU"], asof="2026-11-25", fetch_impl=fetch, emit=False,
                                 limiter=D.RateLimiter(0))
            finally:
                os_env_swap_restore(old)
        self.assertIn("DILU", payload["tickers"])
        flags = payload["tickers"]["DILU"]["flags"]
        self.assertIn("DILUTION", flags)
        self.assertIn("DILUTION_OFFERING", flags)
        self.assertAlmostEqual(payload["tickers"]["DILU"]["shareChangeYoY"], 0.12, places=6)


def os_env_swap(tmp):
    import os
    prev = os.environ.get("PRICES_DATA_DIR")
    os.environ["PRICES_DATA_DIR"] = tmp
    return prev


def os_env_swap_restore(prev):
    import os
    if prev is None:
        os.environ.pop("PRICES_DATA_DIR", None)
    else:
        os.environ["PRICES_DATA_DIR"] = prev


if __name__ == "__main__":
    unittest.main()
