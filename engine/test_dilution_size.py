"""Size-relative offering flag (dilution.py). Offline."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dilution as D  # noqa: E402

ASOF = "2026-11-20"
BASIC = [{"end": "2026-09-30", "filed": "2026-10-25", "val": 100_000_000}]


def filing(form="424B5", acc="0001-26-000001", desc="PROSPECTUS SUPPLEMENT", doc="d1.htm", day="2026-11-10"):
    return {"form": form, "filingDate": day, "accession": acc, "description": desc, "primaryDocument": doc, "items": ""}


def run(text, form="424B5", price=None, cfg=None):
    f = filing(form)
    return D.evaluate("TEST", asof=ASOF, basic_series=BASIC, filings=[f],
                      document_texts={f["accession"]: text}, price=price, config=cfg)


class ParseTests(unittest.TestCase):
    def test_shares(self):
        self.assertEqual(D.parse_offering_size("We are offering 3,000,000 shares of our common stock.")["shares"], 3_000_000)

    def test_dollars(self):
        out = D.parse_offering_size("having an aggregate offering price of up to $150 million from time to time")
        self.assertAlmostEqual(out["dollars"], 150e6)

    def test_per_share_price_ignored(self):
        self.assertIsNone(D.parse_offering_size("at a price of $12.50 per share")["dollars"])


class TierTests(unittest.TestCase):
    def test_small_does_not_block(self):
        r = run("We are offering 2,000,000 shares of common stock. Underwriters ...")
        self.assertTrue(r["offering"])                       # binary field kept
        self.assertIn("DILUTION_OFFERING", r["flags"])       # flag kept for compatibility
        self.assertEqual(r["offeringSeverity"], "LOW")
        self.assertAlmostEqual(r["offeringSizePct"], 0.02)
        self.assertFalse(r["blocksBuy"])
        blocked, reason = D.blocks_new_buy(r)
        self.assertFalse(blocked)
        self.assertEqual(reason, "dilution_offering_small")

    def test_medium_blocks(self):
        r = run("We are offering 10,000,000 shares of common stock.")
        self.assertEqual(r["offeringSeverity"], "MEDIUM")
        self.assertTrue(r["blocksBuy"])
        self.assertEqual(D.blocks_new_buy(r), (True, "dilution_offering"))

    def test_large_is_red(self):
        r = run("We are offering 25,000,000 shares of common stock.")
        self.assertEqual(r["offeringSeverity"], "HIGH")
        v, clamps = D.apply_to_verdict({"flag": "GREEN", "action": "BUY", "entry": {"orderType": "LIMIT", "bandLow": 1}}, r)
        self.assertEqual(v["flag"], "RED")
        self.assertEqual(v["action"], "WATCH")
        self.assertEqual(v["entry"]["orderType"], "NONE")
        self.assertEqual(D.risk_entry(r)["severity"], "HIGH")

    def test_small_verdict_no_clamp(self):
        r = run("We are offering 1,000,000 shares of common stock.")
        v, _ = D.apply_to_verdict({"flag": "GREEN", "action": "BUY"}, r)
        self.assertEqual((v["flag"], v["action"]), ("GREEN", "BUY"))
        self.assertEqual(D.risk_entry(r)["severity"], "LOW")

    def test_dollar_atm_uses_market_cap(self):
        text = "Equity distribution agreement ... at-the-market offering having an aggregate offering price of up to $50 million"
        r = run(text, price=10.0)                            # mcap $1bn -> 5%
        self.assertEqual(r["offeringType"], "ATM")
        self.assertAlmostEqual(r["offeringSizePct"], 0.05)
        self.assertEqual(r["offeringSeverity"], "MEDIUM")

    def test_unknown_size_stays_conservative(self):
        r = run("", form="424B4")
        self.assertEqual(r["offeringSeverity"], "UNKNOWN")
        self.assertTrue(r["blocksBuy"])

    def test_switch_off_restores_binary(self):
        r = run("We are offering 1,000,000 shares of common stock.", cfg={"offering_size_relative": False})
        self.assertTrue(r["blocksBuy"])
        self.assertEqual(D.blocks_new_buy(r), (True, "dilution_offering"))

    def test_underwritten_type(self):
        self.assertEqual(D.offering_type(filing("424B4")), "UNDERWRITTEN")
        self.assertEqual(D.offering_type(filing("8-K")), "PRIVATE_PLACEMENT")
        self.assertEqual(D.offering_type(filing("S-3")), "SHELF")

    def test_tier_edges(self):
        self.assertEqual(D.size_tier(0.0499), "SMALL")
        self.assertEqual(D.size_tier(0.05), "MEDIUM")
        self.assertEqual(D.size_tier(0.15), "MEDIUM")
        self.assertEqual(D.size_tier(0.1501), "LARGE")
        self.assertEqual(D.size_tier(None), "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
