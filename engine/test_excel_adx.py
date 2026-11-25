from unittest import TestCase

from excel_adx import adx_state, summary_values


class ExcelADXTests(TestCase):
    def test_macro_adx_thresholds(self):
        self.assertEqual(
            [adx_state(value) for value in (-5, -4, -2, 0, 2, 5, 6)],
            ["Sharp Deterioration", "Deteriorating", "Deteriorating", "Stable", "Improving", "Improving", "Accelerating"],
        )

    def test_summary_uses_the_separate_fifty_row_windows_and_regime(self):
        closes = [100] * 55
        closes[-6], closes[-1] = 80, 120
        output = summary_values(closes, 28, 20)
        self.assertEqual(output[:4], (28, 20, 8, "Accelerating"))
        self.assertEqual(output[4:10], (120, 100, 0.2, 80, 99.6, (80 - 99.6) / 99.6))
        self.assertAlmostEqual(output[10], output[6] - output[9])
        self.assertEqual(output[11:], ("Crossed Above", "Bullish breakout gaining momentum"))

    def test_summary_guards_missing_prices_and_excel_errors(self):
        with self.assertRaisesRegex(ValueError, "55 usable"):
            summary_values([10] * 54 + [None], 20, 14)
        with self.assertRaisesRegex(ValueError, "column AO"):
            summary_values([10] * 55, -2146826281, 14)

    def test_summary_counts_close_numbers_stored_as_text(self):
        values = summary_values(["100"] * 53 + [110, "120.25"], "30", 25)
        self.assertAlmostEqual(values[5], (48 * 100 + 110 + 120.25) / 50)
        self.assertEqual((values[0], values[2]), (30, 5))
