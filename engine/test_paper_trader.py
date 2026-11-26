"""Paper trader tests: restart safety, idempotence, dry-run rollback and reconciliation with the backtest."""

import tempfile
from contextlib import closing
from pathlib import Path
from unittest import TestCase

import pandas as pd

import paper_trader as pt
from test_strategy_framework import synthetic_panel

CONFIG = {"strategy": "combined", "params": {},
          "sleeves": {"tsmom": {"min_dollar_volume": 1e6},
                      "meanrev": {"min_dollar_volume": 1e6, "entry_z": -1.5, "regime_vol_pct": 0.9,
                                  "trend_t_max": 5, "trend_filter": 50}}}


def run(connection, panel, asof, account="t", dry=False):
    connection.execute("BEGIN IMMEDIATE")
    trader = pt.PaperTrader(connection, account, panel.upto(asof), CONFIG)
    result = trader.run()
    connection.execute("ROLLBACK" if dry else "COMMIT")
    return result


def ledger(connection, account="t"):
    fills = pd.read_sql("SELECT fill_date, ticker, shares, price FROM fills WHERE account=? ORDER BY fill_id",
                        connection, params=(account,))
    equity = pd.read_sql("SELECT date, equity FROM equity_history WHERE account=? ORDER BY date", connection,
                         params=(account,))
    return fills, equity


class PaperTraderTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()
        cls.dates = cls.panel.dates

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "state.sqlite3"

    def tearDown(self):
        self.directory.cleanup()

    def test_daily_runs_equal_one_catch_up_run_and_reruns_are_noops(self):
        start, end = self.dates[200], self.dates[-1]
        with closing(pt.connect(self.path)) as daily:
            run(daily, self.panel, start)
            for asof in self.dates[201:]:
                run(daily, self.panel, asof)
            run(daily, self.panel, end)                      # same date again: nothing new
            daily_fills, daily_equity = ledger(daily)
            orders = daily.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
            self.assertEqual(orders, daily.execute(
                "SELECT COUNT(DISTINCT account || decision_date || ticker || reason) FROM orders").fetchone()[0])
        other = Path(self.directory.name) / "other.sqlite3"
        with closing(pt.connect(other)) as catch_up:
            run(catch_up, self.panel, start)
            result = run(catch_up, self.panel, end)          # e.g. after the PC was off for months
            self.assertEqual(result["sessionsProcessed"], len(self.dates) - 201)
            fills, equity = ledger(catch_up)
        self.assertGreater(len(daily_fills), 0)
        pd.testing.assert_frame_equal(daily_fills, fills)
        pd.testing.assert_frame_equal(daily_equity, equity)

    def test_dry_run_saves_nothing(self):
        with closing(pt.connect(self.path)) as connection:
            run(connection, self.panel, self.dates[250], dry=True)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM accounts").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 0)

    def test_reconciles_with_the_backtest_simulator(self):
        with closing(pt.connect(self.path)) as connection:
            run(connection, self.panel, self.dates[180])
            run(connection, self.panel, self.dates[-1])
            row = connection.execute("SELECT equity_diff_pct, max_weight_diff FROM reconciliation "
                                     "ORDER BY date DESC LIMIT 1").fetchone()
            status = pt.status(connection, "t")
        self.assertLess(abs(row[0]), 0.5)                    # whole shares vs fractional shares only
        self.assertLess(row[1], 0.02)
        self.assertTrue(status["exists"])
        self.assertGreater(status["slippage"]["fills"], 0)
        decision_columns = set(pt.DECISION_COLUMNS)
        self.assertTrue(decision_columns.issubset(status["recentDecisions"][0].keys()))
