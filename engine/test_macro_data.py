"""Unit tests for the dollar proxy, FRED lag and curve interpolation. No network, no secrets."""

import math
import unittest
from datetime import date

import numpy as np
import pandas as pd

from macro_data import (
    ICE_CONSTANT, ICE_EXPONENTS, PUBLICATION_LAG, ice_dxy, interpolate_par_yield, load_macro,
)


class IceDxyTests(unittest.TestCase):
    def test_all_ones_equals_the_ice_constant(self):
        fx = pd.DataFrame({p: 1.0 for p in ICE_EXPONENTS}, index=pd.date_range("2024-01-02", periods=3, freq="B"))
        level = ice_dxy(fx)
        self.assertTrue(np.allclose(level, ICE_CONSTANT))

    def test_stronger_dollar_against_euro_raises_dxy(self):
        idx = pd.date_range("2024-01-02", periods=2, freq="B")
        base = {p: 1.1 if p in ("EURUSD", "GBPUSD") else 100.0 for p in ICE_EXPONENTS}
        a = pd.DataFrame(base, index=idx)
        b = a.copy()
        b.loc[idx[1], "EURUSD"] = 1.0  # fewer dollars per euro = stronger USD
        level_a = ice_dxy(a).iloc[1]
        level_b = ice_dxy(b).iloc[1]
        self.assertGreater(level_b, level_a)

    def test_missing_pair_raises(self):
        fx = pd.DataFrame({"EURUSD": [1.1]}, index=pd.date_range("2024-01-02", periods=1))
        with self.assertRaises(ValueError):
            ice_dxy(fx)

    def test_nonpositive_rates_are_nan(self):
        idx = pd.date_range("2024-01-02", periods=1)
        fx = pd.DataFrame({p: 1.1 for p in ICE_EXPONENTS}, index=idx)
        fx.loc[idx[0], "USDJPY"] = 0.0
        self.assertTrue(math.isnan(ice_dxy(fx).iloc[0]))

    def test_exponents_sum_to_ice_weights(self):
        # Absolute weights should match the published ICE basket (0.576 + ... = 1.0).
        self.assertAlmostEqual(sum(abs(v) for v in ICE_EXPONENTS.values()), 1.0, places=12)


class InterpolateCurveTests(unittest.TestCase):
    def test_linear_between_tenors(self):
        row = pd.Series({2.0: 4.0, 5.0: 4.3, 10.0: 4.6})
        self.assertAlmostEqual(interpolate_par_yield(row, 5.0), 4.3)
        self.assertAlmostEqual(interpolate_par_yield(row, 7.5), 4.45)
        self.assertAlmostEqual(interpolate_par_yield(row, 1.0), 4.0)   # clamp short
        self.assertAlmostEqual(interpolate_par_yield(row, 30.0), 4.6)  # clamp long

    def test_empty_is_nan(self):
        self.assertTrue(math.isnan(interpolate_par_yield(pd.Series(dtype=float), 10)))


class LagContractTests(unittest.TestCase):
    def test_publication_lag_default_is_one_session(self):
        self.assertEqual(PUBLICATION_LAG, 1)

    def test_load_macro_without_files_is_empty_not_an_error(self):
        frame = load_macro(start="1900-01-01", end="1900-01-02")
        self.assertIsInstance(frame, pd.DataFrame)



class FredCsvTests(unittest.TestCase):
    def test_parse_observation_date_header(self):
        from macro_data import parse_fred_csv
        raw = b"observation_date,DGS10\n2024-01-02,4.06\n2024-01-03,.\n2024-01-04,4.03\n"
        frame = parse_fred_csv(raw, ["DGS10"])
        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(float(frame.loc["2024-01-02", "DGS10"]), 4.06)
        self.assertTrue(math.isnan(float(frame.loc["2024-01-03", "DGS10"])))


    def test_parse_zip_of_csvs(self):
        import io, zipfile
        from macro_data import parse_fred_body
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as zf:
            zf.writestr('README.txt', 'ignore')
            zf.writestr('daily.csv', 'observation_date,DGS10\n2024-01-02,4.06\n')
            zf.writestr('daily,_close.csv', 'observation_date,HY\n2024-01-02,3.5\n')
        frame = parse_fred_body(buf.getvalue())
        self.assertAlmostEqual(float(frame.loc['2024-01-02', 'DGS10']), 4.06)
        self.assertAlmostEqual(float(frame.loc['2024-01-02', 'HY']), 3.5)

if __name__ == "__main__":
    unittest.main()
