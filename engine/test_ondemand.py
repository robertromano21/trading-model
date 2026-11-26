"""On-demand Schwab history (chart picker), the daily_long splice in chart charts, and the crypto chart payload.
No network: the Schwab client is faked and all paths point at a temp dir."""

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import chart_data
import schwab_daily as sd
import schwab_ondemand as od


class FakeClient:
    def __init__(self, candles=None, instruments=None):
        self.calls = []
        self.candles = candles or []
        self.instruments = instruments or []

    def get(self, path, **params):
        self.calls.append((path, params))
        assert path.startswith("/marketdata/v1/"), path
        if path.endswith("/instruments"):
            return {"instruments": self.instruments}
        return {"candles": self.candles, "empty": not self.candles}


class OnDemandTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.saved = (od.ONDEMAND_DIR, od.TRACKED_PATH, od.SEARCH_CACHE, od.STAMP_PATH, od.DATABASE)
        od.ONDEMAND_DIR = self.tmp / "ondemand"
        od.TRACKED_PATH = self.tmp / "ondemand_symbols.json"
        od.SEARCH_CACHE = od.ONDEMAND_DIR / "search_cache.json"
        od.STAMP_PATH = od.ONDEMAND_DIR / "updated.stamp"
        od.DATABASE = self.tmp / "prices.sqlite3"
        self.saved_connect = sd.connect_prices
        sd.connect_prices = lambda path=None, writable=False: self.saved_connect(od.DATABASE, writable=writable)

    def tearDown(self):
        od.ONDEMAND_DIR, od.TRACKED_PATH, od.SEARCH_CACHE, od.STAMP_PATH, od.DATABASE = self.saved
        sd.connect_prices = self.saved_connect

    def test_search_is_cached_and_read_only(self):
        client = FakeClient(instruments=[{"symbol": "BLOZF", "description": "CANNABIX", "exchange": "OTC Markets", "assetType": "EQUITY"},
                                         {"symbol": "BLOZX", "description": "OPT", "assetType": "OPTION"}])
        out = od.search("bloz", client=client)
        self.assertEqual([m["symbol"] for m in out["matches"]], ["BLOZF"])
        n = len(client.calls)
        again = od.search("BLOZ", client=client)
        self.assertTrue(again.get("cached"))
        self.assertEqual(len(client.calls), n)
        self.assertEqual(od.search("bad query!", client=client)["matches"], [])

    def test_fetch_stores_daily_long_tracks_symbol_and_cools_down(self):
        day = 86_400_000
        start = int(pd.Timestamp("2020-01-02 06:00", tz="America/New_York").timestamp() * 1000)
        candles = [{"datetime": start + i * day, "open": 1 + i, "high": 2 + i, "low": 0.5 + i, "close": 1.5 + i, "volume": 100}
                   for i in range(30)]
        client = FakeClient(candles=candles)
        out = od.fetch("BLOZF", client=client, intraday_days=0)
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["local"]["schwab"]["rows"], 30)
        self.assertIn("BLOZF", od.tracked())
        self.assertTrue(od.STAMP_PATH.is_file())
        again = od.fetch("BLOZF", client=client, intraday_days=0)
        self.assertEqual(again.get("skipped"), "fetched recently")
        self.assertIn("BLOZF", sd._load_json_tickers(od.TRACKED_PATH))

    def test_invalid_symbol_is_soft_failure(self):
        out = od.fetch("ZZZZQ", client=FakeClient(), intraday_days=0)
        self.assertEqual(out["status"], "invalid")
        ok, reason = od.should_fetch("ZZZZQ")
        self.assertFalse(ok)
        with self.assertRaises(ValueError):
            od.clean_symbol("bad symbol")

    def test_short_rule(self):
        self.assertTrue(od.is_short({"massive": {"rows": 2}, "schwab": {"rows": 0}}))
        self.assertFalse(od.is_short({"massive": {"rows": 2}, "schwab": {"rows": 3000}}))
        self.assertFalse(od.is_short({"massive": {"rows": 900}, "schwab": {"rows": 0}}))


class SpliceTests(unittest.TestCase):
    def test_massive_extended_with_scaled_schwab_history(self):
        con = sqlite3.connect(":memory:")
        con.row_factory = sqlite3.Row
        con.execute("CREATE TABLE daily_prices (ticker, date, open, high, low, close, volume)")
        con.executescript(sd.LONG_SCHEMA)
        dates = pd.bdate_range("2024-01-01", periods=10).strftime("%Y-%m-%d")
        for i, d in enumerate(dates):
            con.execute("INSERT INTO daily_long VALUES ('X', ?, ?, ?, ?, ?, 10, 'schwab', 'split', 'now')", (d, 2.0, 2.0, 2.0, 2.0))
            if i >= 8:
                con.execute("INSERT INTO daily_prices VALUES ('X', ?, 1, 1, 1, 1, 10)", (d,))
        frame, meta = chart_data.merged_daily(con, "X")
        self.assertEqual(len(frame), 10)
        self.assertEqual(meta["schwabRows"], 8)
        self.assertAlmostEqual(meta["scale"], 0.5)
        self.assertTrue(np.allclose(frame["close"].to_numpy(), 1.0))


if __name__ == "__main__":
    unittest.main()
