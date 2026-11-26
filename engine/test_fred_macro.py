"""Offline tests for engine/fred_macro.py: availability lags, as-of (no look-ahead), ALFRED hybrid, chain, transforms."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import fred_macro as FM


class AvailabilityTests(unittest.TestCase):
    def test_monthly_lag_counts_from_period_end(self):
        a = FM.availability(pd.DatetimeIndex(["2020-01-01"]), "CPIAUCSL")
        self.assertEqual(a[0], pd.Timestamp("2020-01-31") + pd.Timedelta(days=FM.SERIES["CPIAUCSL"][1]))

    def test_quarterly_period_end(self):
        a = FM.availability(pd.DatetimeIndex(["2020-04-01"]), "GDPC1", lag_days=0)
        self.assertEqual(a[0], pd.Timestamp("2020-06-30"))

    def test_outside_forecasts_flagged(self):
        self.assertIn("GDPNOW", FM.OUTSIDE_FORECASTS)
        self.assertEqual(FM.SERIES["GDPNOW"][3], "outside_forecast")


class AsofTests(unittest.TestCase):
    def test_value_hidden_until_available(self):
        obs = pd.DataFrame({"value": [1.0, 2.0], "available": pd.to_datetime(["2020-02-10", "2020-03-10"])},
                           index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
        s = FM.asof_series(obs, pd.to_datetime(["2020-02-09", "2020-02-10", "2020-03-09", "2020-03-10"]))
        self.assertTrue(np.isnan(s.iloc[0]))
        self.assertEqual(list(s.iloc[1:]), [1.0, 1.0, 2.0])

    def test_late_revision_of_old_obs_does_not_replace_newer(self):
        obs = pd.DataFrame({"value": [5.0, 7.0], "available": pd.to_datetime(["2020-03-20", "2020-03-10"])},
                           index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
        s = FM.asof_series(obs, pd.to_datetime(["2020-03-25"]))
        self.assertEqual(s.iloc[0], 7.0)


class HybridVintageTests(unittest.TestCase):
    def test_stale_vintage_falls_back_to_lag_date_and_latest_value(self):
        idx = pd.date_range("2000-01-01", periods=4, freq="MS")
        latest = pd.Series([10.0, 11.0, 12.0, 13.0], index=idx, name="UNRATE")
        # ALFRED: first two rows only recorded in 2010 (stale), last two have real first-release dates
        vint = pd.DataFrame({"value": [9.0, 9.5, 11.9, 12.9],
                             "available": pd.to_datetime(["2010-01-01", "2010-01-01", "2000-04-07", "2000-05-05"])},
                            index=idx)
        with mock.patch.object(FM, "load_series", return_value=latest), \
                mock.patch.object(FM, "load_vintage", return_value=vint):
            obs = FM.observations("UNRATE")
        lag = FM.SERIES["UNRATE"][1]
        self.assertEqual(obs.loc["2000-01-01", "available"], pd.Timestamp("2000-01-31") + pd.Timedelta(days=lag))
        self.assertEqual(obs.loc["2000-01-01", "value"], 10.0)          # latest-vintage value, not the stale one
        self.assertFalse(bool(obs.loc["2000-01-01", "vintageDated"]))
        self.assertEqual(obs.loc["2000-03-01", "value"], 11.9)          # first release kept
        self.assertEqual(obs.loc["2000-03-01", "available"], pd.Timestamp("2000-04-07"))

    def test_vintage_never_before_period_end(self):
        idx = pd.DatetimeIndex(["2000-01-01"])
        vint = pd.DataFrame({"value": [1.0], "available": pd.to_datetime(["1999-12-01"])}, index=idx)
        with mock.patch.object(FM, "load_series", return_value=None), \
                mock.patch.object(FM, "load_vintage", return_value=vint):
            obs = FM.observations("UNRATE")
        self.assertGreaterEqual(obs["available"].iloc[0], pd.Timestamp("2000-01-31"))


class TransformChainTests(unittest.TestCase):
    def test_transforms(self):
        s = pd.Series([100.0, 101.0, 103.02])
        self.assertAlmostEqual(FM.transform(s, "pch", "M").iloc[1], 1.0)
        self.assertAlmostEqual(FM.transform(s, "chg", "M").iloc[2], 2.02)
        self.assertAlmostEqual(FM.transform(pd.Series([100.0, 101.0]), "pca", "Q").iloc[1], (1.01 ** 4 - 1) * 100)

    def test_chain_scales_old_to_new(self):
        old = pd.Series([50.0, 55.0, 60.0], index=pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]))
        new = pd.Series([120.0, 130.0], index=pd.to_datetime(["2000-01-03", "2000-01-04"]))
        c = FM.chain(old, new)
        self.assertAlmostEqual(c.loc["2000-01-01"], 100.0)
        self.assertEqual(c.loc["2000-01-04"], 130.0)

    def test_vintage_rows_keep_first_release(self):
        rows = [{"date": "2000-01-01", "value": "1.0", "realtime_start": "2000-02-04"},
                {"date": "2000-01-01", "value": "1.2", "realtime_start": "2000-03-03"},
                {"date": "2000-02-01", "value": ".", "realtime_start": "2000-03-03"}]
        f = FM.parse_api_rows(rows, "UNRATE", vintage=True)
        self.assertEqual(len(f), 1)
        self.assertEqual(f["value"].iloc[0], 1.0)


class SecretTests(unittest.TestCase):
    def test_http_error_never_echoes_query_string(self):
        from urllib.error import HTTPError
        secret = "abcdef0123456789abcdef0123456789"

        def boom(*_a, **_k):
            raise HTTPError("https://api.stlouisfed.org/x?api_key=" + secret, 400, "bad", {}, None)
        with mock.patch.object(FM, "urlopen", side_effect=boom), mock.patch.object(FM, "_polite"):
            with self.assertRaises(FM.FredError) as ctx:
                FM._get("https://api.stlouisfed.org/x?api_key=" + secret, safe_name="UNRATE")
        self.assertNotIn(secret, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
