import logging
from datetime import date, timedelta
from math import log, pi, sin
from unittest import TestCase

from fetch_and_model import market_bars
from strategy_model import flat_base_signal, half_life, percentile, scan


class ModelTests(TestCase):
    def run_prices(self, prices, high_volume_day=None, **overrides):
        first = date(2025, 1, 1)
        sessions = []
        for index, close in enumerate(prices):
            sessions.append(((first + timedelta(days=index)).isoformat(), [{
                "ticker": "TEST", "open": close, "close": close, "high": close + 0.5, "low": close - 0.5,
                "volume": 300 if index == high_volume_day else 100,
            }]))
        options = {
            "date": sessions[-1][0], "baseline": "sma", "fastEma": 5, "trailing": "ma",
            "breakoutStop": "atr", "trendDays": 12, **overrides,
        }
        return scan(sessions, options, logging.getLogger(__name__))["rows"][0]

    def test_all_symbols_are_kept_from_massive_payload(self):
        records = [
            {"T": f"T{index}", "o": 10, "h": 12, "l": 9, "c": 11, "v": 1000}
            for index in range(12601)
        ]
        bars = market_bars(records, date(2025, 1, 1))
        self.assertEqual(len(bars), len(records))
        self.assertEqual(bars[-1]["ticker"], "T12600")

    def test_indicators_warm_up_per_symbol(self):
        prices = [100 + 0.2 * index for index in range(200)]
        short_history = self.run_prices(prices[:19])
        self.assertIsNone(short_history["sma20"])
        self.assertIsNone(short_history["adx14"])
        self.assertEqual(short_history["breakout"]["state"], "insufficient")
        ready = self.run_prices(prices)
        self.assertAlmostEqual(ready["sma20"], 137.9)
        self.assertAlmostEqual(ready["ema20"], 137.9)
        self.assertAlmostEqual(ready["sma200"], 119.9)
        self.assertAlmostEqual(ready["std20"], (0.2 ** 2 * 399 / 12) ** 0.5)
        self.assertAlmostEqual(ready["adx14"], 100)
        self.assertAlmostEqual(ready["atr14"], 1)
        self.assertEqual(ready["rsi14"], 100)
        self.assertEqual(ready["rvol"], 1)

    def test_breakouts_require_high_volume_on_both_sides_and_stop_on_gap(self):
        up = [100 + 0.2 * index for index in range(200)]
        down = [200 - 0.2 * index for index in range(200)]
        self.assertEqual(self.run_prices(up)["breakout"]["state"], "none")
        self.assertEqual(self.run_prices(down)["breakout"]["state"], "none")
        long = self.run_prices(up, high_volume_day=199)["breakout"]
        short = self.run_prices(down, high_volume_day=199)["breakout"]
        self.assertEqual((long["state"], long["side"]), ("entry", "long"))
        self.assertEqual((short["state"], short["side"]), ("entry", "short"))
        self.assertAlmostEqual(long["stop"], 137.8)
        self.assertAlmostEqual(short["stop"], 162.2)
        stopped = self.run_prices(up + [130], high_volume_day=199)["breakout"]
        self.assertEqual((stopped["state"], stopped["reason"], stopped["exitPrice"]),
                         ("exit", "Protective stop", 130))

    def test_breakout_time_stop_and_two_r_rule(self):
        up = [100 + 0.2 * index for index in range(200)]
        stale = self.run_prices(up + [139.8] * 10, high_volume_day=199, trendDays=10)["breakout"]
        self.assertEqual((stale["state"], stale["reason"]), ("exit", "Time stop: 2R was not reached"))
        reached = self.run_prices(up + [145] + [140] * 9, high_volume_day=199, trendDays=10)["breakout"]
        self.assertEqual(reached["state"], "open")

    def test_mean_reversion_entry_and_sma_target(self):
        prices = [100 + 0.3 * i + 4 * sin(2 * pi * i / 9) for i in range(197)] + [156, 154, 149]
        result = self.run_prices(prices)
        self.assertLess(result["halfLife60"], 10)
        self.assertEqual(result["pullback"]["state"], "entry")
        target = self.run_prices(prices + [165])["pullback"]
        self.assertEqual((target["state"], target["reason"]), ("exit", "SMA 20 profit target"))

    def test_crossover_starts_on_cross_not_merely_above_average(self):
        prices = [100 + 0.5 * index for index in range(196)] + [190] * 16
        self.assertEqual(self.run_prices(prices)["crossover"]["state"], "none")
        self.assertEqual(self.run_prices(prices + [240])["crossover"]["state"], "entry")

    def test_half_life_and_flat_base_thresholds(self):
        self.assertAlmostEqual(half_life([1 + 0.8 ** i for i in range(15)], 15), log(2) / 0.2)
        self.assertIsNone(half_life([1] * 60, 60))
        self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], 0.2), 1.8)
        self.assertEqual(flat_base_signal(0.39, True), "volatility_compression_squeeze")
        self.assertIsNone(flat_base_signal(0.39, False))
        self.assertIsNone(flat_base_signal(0.4, True))
        self.assertIsNone(flat_base_signal(2, True))
        self.assertEqual(flat_base_signal(2.01, False), "trend_emergence_decay")

    def test_breadth_uses_eligible_symbols_and_the_previous_full_session(self):
        first = date(2025, 1, 1)
        sessions = []
        for offset in range(202):
            market_date = (first + timedelta(days=offset)).isoformat()
            bars = []
            for ticker, close in [("UP", 100 + offset), ("DOWN", 500 - offset), ("NEW", 100)]:
                if (offset == 200 and ticker != "UP") or (ticker == "NEW" and offset < 201):
                    continue
                bars.append({"ticker": ticker, "open": close, "high": close + 0.5, "low": close - 0.5,
                             "close": close, "volume": 100})
            sessions.append((market_date, bars))
        full_dates = {market_date for index, (market_date, _) in enumerate(sessions) if index != 200}
        options = {"date": sessions[-1][0], "baseline": "sma", "fastEma": 5,
                   "trailing": "ma", "breakoutStop": "atr", "trendDays": 12}
        breadth = scan(sessions, options, logging.getLogger(__name__), full_dates)["breadth"]
        self.assertEqual(breadth["previousSession"], sessions[199][0])
        self.assertEqual((breadth["advancing"], breadth["declining"], breadth["unchanged"]), (1, 1, 0))
        for average in ("sma20", "sma50", "sma200"):
            self.assertEqual(breadth["averages"][average], {"above": 1, "eligible": 2})
