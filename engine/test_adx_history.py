from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from adx_history import FIELDS, compile_filters, condition_returns, condition_scenarios, forward_matches, indicator_history, replay
from adx_summary import summarize
from price_history import open_history, save_session


class ADXHistoryTests(TestCase):
    def test_breakdown_tests_each_condition_and_adx_state_on_its_own_sample(self):
        days = np.array([f"{day:02d}" for day in range(1, 11)])
        closes = np.array([100, 100, 100, 100, 100, 110, 120, 90, 115, 100], dtype=float)
        metrics = {
            "close": closes,
            "adx": np.array([10, 25, 10, 10, 10, 10, 10, 10, 10, 10], dtype=float),
            "ma20ZScore": np.array([-2, -2, -1, -1, -1, -2, -2, -2, -2, -2], dtype=float),
            "adxState": np.array([
                "Sharp Deterioration", "Sharp Deterioration", "Deteriorating", "Stable", "Stable",
                "Improving", "Improving", "Improving", "Improving", "Improving",
            ]),
        }
        rules = compile_filters({"columns": [
            ["ma20ZScore", {"join": "and", "conditions": [{"op": "lt", "value": "-1.75"}]}],
            ["adx", {"join": "and", "conditions": [{"op": "lt", "value": "20"}]}],
        ]})
        rows = condition_returns(days, metrics, days, condition_scenarios(rules))
        self.assertEqual([row["observations"] for row in rows], [2, 4, 1, 1, 2, 0, 0])
        self.assertAlmostEqual(rows[0]["mean"], 0.15)
        self.assertAlmostEqual(rows[1]["mean"], 0.0375)
        self.assertAlmostEqual(rows[2]["mean"], 0.1)
        self.assertAlmostEqual(rows[3]["mean"], -0.1)
        self.assertAlmostEqual(rows[4]["mean"], 0.075)
        self.assertIsNone(rows[-1]["mean"])

    def test_historical_indicators_match_summary_using_each_dates_prefix(self):
        days = np.array([(date(2025, 1, 1) + timedelta(days=index)).isoformat() for index in range(90)])
        angles = np.arange(len(days), dtype=float)
        closes = 100 + 0.1 * angles + 6 * np.sin(angles / 3)
        prices = np.column_stack([closes + 1, closes - 1, closes])
        fields = FIELDS - {"ticker", "sp500"}
        history = indicator_history(days, prices, fields)
        for index in range(len(days)):
            row = summarize("TEST", days[:index + 1], prices[:index + 1])
            row["maPosition"] = row["pct50"]
            row["gapChange"] = ((abs(row["pct50"]) - abs(row["priorPct50"])) * 100
                                if row["pct50Change"] is not None else None)
            for field in fields:
                with self.subTest(index=index, field=field):
                    actual, expected = history[field][index], row[field]
                    if isinstance(actual, (float, np.floating)) and np.isnan(actual):
                        self.assertIsNone(expected)
                    elif isinstance(expected, float):
                        self.assertAlmostEqual(actual, expected)
                    else:
                        self.assertEqual(actual, expected)

    def test_fixed_membership_missing_exit_and_cutoff_in_replay(self):
        stat = SimpleNamespace(st_size=1, st_mtime_ns=1)
        days = [(date(2025, 1, 6) + timedelta(days=index)).isoformat() for index in (0, 1, 2, 3, 4, 7, 8, 9, 10)]

        def bar(ticker, close):
            return {"ticker": ticker, "open": close, "close": close, "high": close + 1, "low": close - 1, "volume": 100}

        with TemporaryDirectory() as directory, open_history(Path(directory) / "prices.sqlite3", writable=True) as db:
            for index, day in enumerate(days):
                bars = [bar("OTHER", 20 + index)]
                if index != 5:
                    bars.append(bar("TEST", 100 + 2 * index))
                save_session(db, day, bars, stat)
            rules = compile_filters({"columns": [
                ["sp500", {"values": ["Yes"]}],
                ["close", {"join": "or", "conditions": [{"op": "lte", "value": "102"}, {"op": "gt", "value": "110"}]}],
            ]})
            result = replay(db, days[-2], rules, {"TEST", "OTHER"}, ["TEST"])
            member_only = replay(db, days[-2], rules, {"TEST"}, ["TEST", "OTHER"])
        self.assertEqual((result["observations"], result["tickers"], result["pending"], result["missing"]), (1, 1, 2, 1))
        self.assertEqual(set(result["tickerReturns"]), {"TEST"})
        self.assertEqual(result["tickerReturns"]["TEST"], {"observations": 1, "mean": result["mean"]})
        self.assertEqual(member_only["observations"], result["observations"])
        self.assertEqual(member_only["tickerReturns"]["OTHER"], {"observations": 0, "mean": None})
        self.assertEqual(result["recent"][0]["date"], days[1])
        self.assertEqual(result["recent"][0]["exitDate"], days[6])
        self.assertAlmostEqual(result["mean"], 112 / 102 - 1)
        self.assertEqual(result["positiveRate"], 1)

    def test_returns_skip_missing_fifth_session_without_sliding_exit(self):
        sessions = np.array(["01", "02", "03", "04", "05", "06", "07"])
        days = sessions[[0, 1, 2, 3, 4, 6]]
        starts, exits, returns, pending, missing = forward_matches(
            days, np.array([100, 100, 99, 98, 97, 103]), sessions, np.ones(len(days), dtype=bool))
        self.assertEqual(starts.tolist(), [1])
        self.assertEqual(exits.tolist(), ["07"])
        self.assertAlmostEqual(returns[0], 0.03)
        self.assertEqual((pending, missing), (4, 1))
