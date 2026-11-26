"""SBIT/BITI market-hours model: fills only in US sessions, stops only in hours, daily-reset NAV, no holdout scoring."""

import unittest

import numpy as np
import pandas as pd

import crypto_short as cs


def grid_and_masks(days=10, start="2026-01-05"):   # Monday
    grid = pd.date_range(pd.Timestamp(start), periods=24 * days, freq="h")
    sessions = pd.bdate_range(start, periods=days)
    return grid, cs.calendar_masks(grid, sessions), sessions


class MarketHoursTests(unittest.TestCase):
    def test_etf_fill_hours_are_us_session_only(self):
        grid, m, _ = grid_and_masks()
        et = grid.tz_localize("UTC").tz_convert(cs.ET)
        self.assertTrue(set(et[m["etf"]].hour) <= set(range(10, 17)))
        self.assertTrue(all(d < 5 for d in et[m["etf"]].weekday))
        self.assertTrue(set(et[m["reset"]].hour) == {16})
        sat = (et.weekday == 5)
        self.assertFalse(m["cme"][sat].any())
        self.assertTrue(m["perp"].all())

    def test_daily_reset_nav_is_minus_two_x_within_a_day(self):
        grid, m, _ = grid_and_masks(3)
        btc = pd.Series(100.0, index=grid)
        btc.iloc[30:] = 110.0   # +10% move
        nav = cs.synthetic_nav(btc, m["reset"], -2.0, 0.0)
        self.assertAlmostEqual(nav.iloc[29], 1.0)
        self.assertAlmostEqual(nav.iloc[31], 0.8, places=6)

    def test_signal_at_utc_close_fills_next_session_and_stop_waits_for_hours(self):
        grid, m, _ = grid_and_masks(8, "2026-01-09")   # Friday
        P = pd.DataFrame({"BTC": 100.0, "ETH": 10.0}, index=grid)
        P.iloc[60:, 0] = 130.0          # Sunday squeeze (+30%) while the ETF market is closed
        decide = np.array([24, 96])     # Fri UTC close (Sat 00:00 UTC) and Mon UTC close (Tue 00:00 UTC)
        WL = np.zeros((2, 1, 2))
        ES = np.ones((2, 1))
        nav = cs.synthetic_nav(P["BTC"], m["reset"], -1.0, 0.0)
        eq_etf, _ = cs.simulate(P, decide, WL, ES, [None], "etf", m, nav=nav, k=-1.0, start_i=0)
        eq_perp, _ = cs.simulate(P, decide, WL, ES, [None], "perp", m, k=-1.0, start_i=0)
        # the 24/7 short filled Saturday at 100 and ate the +30% Sunday squeeze; the ETF could only fill Monday at 130
        self.assertGreater(eq_etf[1, 0], 0.99 * cs.cr.INITIAL)
        self.assertLess(eq_perp[1, 0], 0.75 * cs.cr.INITIAL)

    def test_run_refuses_to_score_the_holdout(self):
        with self.assertRaises(SystemExit):
            cs.run(look_holdout=True)
        with self.assertRaises(SystemExit):
            cs.main(["run"])

    def test_design_and_shadow_accounts(self):
        d = cs.make_design()
        self.assertEqual(len(d), len(cs.LONGS) * len(cs.SHORT_RULES) * 4)
        self.assertEqual([a[0] for a in cs.PAPER_ACCOUNTS], [220, 221, 222, 223])
        self.assertTrue(all(a[1].startswith("SHADOW") for a in cs.PAPER_ACCOUNTS))
        self.assertEqual(cs.ETFS["SBIT"]["k"], -2.0)
        self.assertEqual(cs.ETFS["BITI"]["k"], -1.0)


if __name__ == "__main__":
    unittest.main()
