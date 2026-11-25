from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from adx_summary import ma20_gap_statistics, market_summary, summarize, worksheet_adx
from excel_adx import summary_values
from price_history import open_history, save_session


class ADXSummaryTests(TestCase):
    def test_ma20_gap_statistics_use_prior_gaps_on_the_current_side(self):
        gaps = np.array([np.nan, -0.10, -0.02, -0.06, 0.05, 0.12, 0.0, -0.08])
        expected_std = (0.0032 / 3) ** 0.5
        for direction, percentile in [(1, 1 / 3), (-1, 2 / 3)]:
            with self.subTest(direction=direction):
                stats = ma20_gap_statistics(gaps * direction)
                self.assertEqual(stats["ma20GapCount"], 3)
                self.assertAlmostEqual(stats["ma20GapMean"], -0.06 * direction)
                self.assertAlmostEqual(stats["ma20GapStd"], expected_std)
                self.assertAlmostEqual(stats["ma20GapPercentile"], percentile)
                self.assertAlmostEqual(stats["ma20ZScore"], -0.02 * direction / expected_std)

    def test_ma20_gap_statistics_handle_zero_variance_and_no_matching_history(self):
        stats = ma20_gap_statistics(np.array([-0.04, -0.04, 0.02, -0.04]))
        self.assertEqual(stats["ma20GapStd"], 0)
        self.assertEqual(stats["ma20GapPercentile"], 0.5)
        self.assertIsNone(stats["ma20ZScore"])
        for current in (0.0, 0.02, np.nan):
            with self.subTest(current=current):
                stats = ma20_gap_statistics(np.array([-0.02, -0.04, current]))
                self.assertEqual(stats["ma20GapCount"], 0)
                self.assertIsNone(stats["ma20ZScore"])

    def test_worksheet_seed_and_smoothing(self):
        series = np.arange(40, dtype=float)
        adx = worksheet_adx(series + 10, series + 8)
        seed = (1300 / 14 * 13 + 100) / 14
        self.assertTrue(np.isnan(adx[:25]).all())
        self.assertAlmostEqual(adx[25], seed)
        self.assertAlmostEqual(adx[26], (seed * 13 + 100) / 14)

    def test_market_fields_match_the_workbook_summary_rules(self):
        prices = [100 + index / 2 for index in range(56)]
        dates = [(date(2025, 1, 1) + timedelta(days=index)).isoformat() for index in range(len(prices))]
        row = summarize("TEST", dates, [(price + 1, price - 1, price) for price in prices])
        fields = ("adx", "priorAdx", "adxChange", "adxState", "close", "ma50", "pct50", "priorClose",
                  "priorMa50", "priorPct50", "pct50Change", "maStatus", "marketRegime")
        values = summary_values(prices[-55:], row["adx"], row["priorAdx"])
        for field, expected in zip(fields, values):
            if isinstance(expected, str):
                self.assertEqual(row[field], expected)
            else:
                self.assertAlmostEqual(row[field], expected)
        self.assertEqual(row["highDate"], dates[-1])
        self.assertEqual(row["lowDate"], dates[25])
        self.assertAlmostEqual(row["minPct50"], row["pct50"])

    def test_only_symbols_on_the_selected_session_are_included(self):
        stat = SimpleNamespace(st_size=1, st_mtime_ns=1)

        def bar(ticker, close):
            return {"ticker": ticker, "open": close, "close": close, "high": close + 1, "low": close - 1, "volume": 100}

        with TemporaryDirectory() as directory, open_history(Path(directory) / "prices.sqlite3", writable=True) as db:
            save_session(db, "2025-01-02", [bar("OLD", 10), bar("TEST", 20)], stat)
            save_session(db, "2025-01-03", [bar("TEST", 30)], stat)
            save_session(db, "2025-01-06", [bar("NEW", 99), bar("TEST", 60)], stat)
            result = market_summary(db, "2025-01-03")
        row, = result["rows"]
        self.assertEqual((row["ticker"], row["sessions"], row["close"]), ("TEST", 2, 30))
        self.assertIsNone(row["adx"])
        self.assertIsNone(row["ma50"])
        self.assertEqual(result["readyCount"], 0)
