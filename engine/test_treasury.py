"""Treasury pricing: clean/dirty, accrued, YTM, duration, convexity, spreads, paper book."""

import math
import unittest
from datetime import date, timedelta

import pandas as pd

from treasury import (
    GENERIC_TENORS, PAR, PaperTreasuryBook, accrued_interest, as_date, bill_price, bill_ytm,
    coupon_dates, dirty_price_from_ytm, duration_convexity, half_spread_points, issue_generic,
    mark_bond, ytm_from_dirty,
)


class ConventionTests(unittest.TestCase):
    def test_coupon_dates_are_semi_annual_and_end_on_maturity(self):
        sched = coupon_dates(date(2024, 1, 15), date(2026, 1, 15))
        self.assertEqual(sched[-1], date(2026, 1, 15))
        self.assertEqual(len(sched), 4)
        gaps = [(sched[i] - sched[i - 1]).days for i in range(1, len(sched))]
        self.assertTrue(all(180 <= g <= 185 for g in gaps))

    def test_accrued_is_zero_on_a_coupon_date(self):
        issue, maturity = date(2024, 1, 15), date(2026, 1, 15)
        sched = coupon_dates(issue, maturity)
        self.assertEqual(accrued_interest(6.0, date(2025, 1, 15), sched), 0.0)

    def test_accrued_halfway_is_about_half_a_coupon(self):
        issue, maturity = date(2024, 1, 15), date(2026, 1, 15)
        sched = coupon_dates(issue, maturity)
        mid = date(2025, 4, 15)  # ~3 months after Jan 15 coupon
        acc = accrued_interest(6.0, mid, sched)
        self.assertGreater(acc, 1.2)
        self.assertLess(acc, 1.8)  # half of 3.0 is 1.5


class PriceYieldTests(unittest.TestCase):
    def test_par_bond_just_after_issue_is_near_par_when_coupon_equals_yield(self):
        issue = date(2024, 1, 15)
        maturity = date(2026, 1, 15)
        sched = coupon_dates(issue, maturity)
        # Settlement one day after issue so the first coupon is slightly less than a full period away.
        settlement = date(2024, 1, 16)
        dirty = dirty_price_from_ytm(0.04, settlement, sched, 4.0)
        acc = accrued_interest(4.0, settlement, sched)
        clean = dirty - acc
        self.assertAlmostEqual(clean, PAR, delta=0.15)

    def test_ytm_roundtrip(self):
        issue, maturity = date(2024, 1, 15), date(2034, 1, 15)
        sched = coupon_dates(issue, maturity)
        settlement = date(2025, 3, 1)
        y = 0.045
        dirty = dirty_price_from_ytm(y, settlement, sched, 4.0)
        recovered = ytm_from_dirty(dirty, settlement, sched, 4.0, guess=0.03)
        self.assertAlmostEqual(recovered, y, places=6)

    def test_premium_bond_has_ytm_below_coupon(self):
        issue, maturity = date(2024, 1, 15), date(2034, 1, 15)
        sched = coupon_dates(issue, maturity)
        settlement = date(2024, 1, 16)
        dirty = dirty_price_from_ytm(0.03, settlement, sched, 5.0)
        self.assertGreater(dirty, PAR)
        y = ytm_from_dirty(dirty, settlement, sched, 5.0)
        self.assertAlmostEqual(y, 0.03, places=5)

    def test_clean_plus_accrued_equals_dirty(self):
        from treasury import BondSpec
        spec = BondSpec("TSY_10Y", date(2024, 1, 15), date(2034, 1, 15), 0.04, kind="note")
        mark = mark_bond(spec, date(2024, 7, 1), 0.042)
        self.assertAlmostEqual(mark["clean"] + mark["accrued"], mark["dirty"], places=8)
        self.assertLess(mark["bid"], mark["clean"])
        self.assertGreater(mark["ask"], mark["clean"])


class DurationConvexityTests(unittest.TestCase):
    def test_zero_coupon_macaulay_matches_remaining_life(self):
        issue, maturity = date(2024, 1, 15), date(2034, 1, 15)
        sched = [maturity]
        settlement = date(2024, 1, 15)
        years = (maturity - settlement).days / 365.2425
        mac, mod, conv = duration_convexity(0.04, settlement, sched, 0.0)
        self.assertAlmostEqual(mac, years, places=3)
        self.assertLess(mod, mac)
        self.assertGreater(conv, 0)

    def test_coupon_bond_duration_is_less_than_maturity(self):
        issue, maturity = date(2024, 1, 15), date(2034, 1, 15)
        sched = coupon_dates(issue, maturity)
        settlement = date(2024, 1, 16)
        mac, mod, conv = duration_convexity(0.04, settlement, sched, 4.0)
        self.assertLess(mac, 10.0)
        self.assertGreater(mac, 7.0)
        self.assertGreater(conv, mac)


class BillTests(unittest.TestCase):
    def test_discount_price_and_positive_ytm(self):
        settle, mat = date(2024, 1, 15), date(2024, 4, 15)
        price = bill_price(0.05, settle, mat)
        self.assertLess(price, PAR)
        self.assertGreater(price, 98.0)
        y = bill_ytm(price, settle, mat)
        self.assertGreater(y, 0.04)


class SpreadModelTests(unittest.TestCase):
    def test_longer_tenor_is_wider(self):
        self.assertLess(half_spread_points(2.0), half_spread_points(30.0))
        self.assertGreater(half_spread_points(30.0), 0)


class PaperBookTests(unittest.TestCase):
    def _curve(self):
        idx = pd.date_range("2024-01-16", periods=40, freq="B")
        # Flat 4% curve across tenors.
        data = {t: 4.0 for t in (0.25, 2.0, 5.0, 10.0, 30.0)}
        return pd.DataFrame(data, index=idx)

    def test_buy_marks_and_sell_round_trip_loses_the_spread(self):
        curve = self._curve()
        book = PaperTreasuryBook(cash=100_000)
        day = curve.index[0]
        book.buy("TSY_10Y", day, curve, quantity=10)
        marked = book.mark(day, curve)
        self.assertIn("TSY_10Y", marked["positions"])
        self.assertLess(marked["cash"], 100_000)
        equity_after_buy = marked["equity"]
        book.sell("TSY_10Y", day, curve, quantity=10)
        after = book.mark(day, curve)
        self.assertNotIn("TSY_10Y", after["positions"])
        # Buying at ask and selling at bid on the same day should lose money (the spread).
        self.assertLess(after["cash"], 100_000)
        self.assertLess(after["equity"], 100_000)
        self.assertAlmostEqual(equity_after_buy, marked["equity"])

    def test_unknown_generic_rejected(self):
        with self.assertRaises(ValueError):
            issue_generic("TSY_7Y", date(2024, 1, 16), 0.04)


if __name__ == "__main__":
    unittest.main()
