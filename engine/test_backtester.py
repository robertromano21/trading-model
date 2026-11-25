from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from backtester import Series, adjusted_signals, parse_config, run, simulate, universe
from price_history import open_history, save_session


def day_list(length):
    first = date(2025, 1, 1)
    return [(first + timedelta(days=offset)).isoformat() for offset in range(length)]


def config(dates, **overrides):
    return {"mode": "single", "tickers": ["AAA"], "strategy": "sma_crossover",
            "params": {"fast": 2, "slow": 3}, "startDate": dates[0], "endDate": dates[-1],
            "costs": {"initialCapital": 1000, "commissionPct": 1, "fixedFee": 1, "slippageBps": 100},
            **overrides}


class BacktesterTests(TestCase):
    def test_entries_and_exits_use_the_next_open_with_costs(self):
        dates = day_list(8)
        prices = [12, 11, 10, 13, 14, 12, 11, 18]
        opens = [12, 11, 10, 13, 20, 12, 11, 17]
        stat = SimpleNamespace(st_size=1, st_mtime_ns=1)
        with TemporaryDirectory() as directory, open_history(Path(directory) / "prices.sqlite3", writable=True) as db:
            for day, opening, close in zip(dates, opens, prices):
                save_session(db, day, [{"ticker": "AAA", "open": opening, "close": close,
                                        "low": min(opening, close), "high": max(opening, close), "volume": 100}], stat)
            result = run(db, config(dates))
            self.assertEqual(universe(db)["tickers"][0]["missingMarketSessions"], 0)
        trade, = result["trades"]
        self.assertEqual((trade["entryDate"], trade["exitDate"]), (dates[4], dates[7]))
        self.assertAlmostEqual(trade["entryPrice"], 20.2)
        self.assertAlmostEqual(trade["exitPrice"], 16.83)
        self.assertGreater(result["metrics"]["totalFeesPaid"], 2)
        self.assertAlmostEqual(result["metrics"]["finalEquity"], 1000 + trade["pnl"])

    def test_ex_date_does_not_create_a_signal_price_drop_or_double_count_income(self):
        dates = day_list(4)
        prices = np.array([10, 10, 9, 9], float)
        series = {"AAA": Series(np.arange(4), prices, prices, prices, np.zeros(4, dtype=bool))}
        dividends = {"AAA": {dates[2]: (1, 0)}}
        self.assertTrue(np.allclose(adjusted_signals(prices, np.array([0, 0, 1, 0])), 10))
        spec = parse_config(config(dates, costs={"initialCapital": 1000, "commissionPct": 0, "slippageBps": 0}))
        curve, trades, fees, income, _positions = simulate(
            spec, dates, 0, series, {1: [(1, "AAA", 10, 1)]}, dividends, "AAA",
        )
        self.assertEqual((trades, fees, income), ([], 0, 100))
        self.assertAlmostEqual(curve[-1]["equity"], 1000)
        self.assertAlmostEqual(curve[-1]["benchmark"], 1000)

    def test_portfolio_slots_allocate_equally_and_stop_gaps_use_the_open(self):
        dates = day_list(3)
        a = np.array([10, 10, 5], float)
        b = np.array([10, 10, 12], float)
        dates_idx = np.arange(3)
        flags = np.zeros(3, dtype=bool)
        series = {"AAA": Series(dates_idx, a, a, a, flags), "BBB": Series(dates_idx, b, b, b, flags)}
        spec = parse_config(config(dates, mode="portfolio", tickers=["AAA", "BBB"], costs={
            "initialCapital": 1000, "commissionPct": 0, "slippageBps": 0, "maxPositions": 2, "stopLossPct": 10,
        }))
        curve, trades, fees, income, positions = simulate(
            spec, dates, 0, series, {1: [(1, "AAA", 10, 1), (1, "BBB", 10, 1)]}, {}, "SPY",
        )
        trade, = trades
        self.assertEqual((trade["exitPrice"], trade["reason"]), (5, "Stop gap"))
        self.assertEqual((positions["BBB"].units, fees, income), (50, 0, 0))
        self.assertAlmostEqual(curve[-1]["equity"], 850)

    def test_dividends_require_a_covered_window(self):
        dates = day_list(2)
        stat = SimpleNamespace(st_size=1, st_mtime_ns=1)
        with TemporaryDirectory() as directory, open_history(Path(directory) / "prices.sqlite3", writable=True) as db:
            for day in dates:
                save_session(db, day, [{"ticker": "AAA", "open": 10, "close": 10, "low": 10, "high": 10,
                                        "volume": 100}], stat)
            with self.assertRaisesRegex(ValueError, "Dividend data does not cover"):
                run(db, config(dates, includeDividends=True))
