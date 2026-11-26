"""UI-support tests: chart indicators match the engine, chart payloads, decision chat feed, structured log redaction."""

import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from unittest import TestCase, mock

import numpy as np
import pandas as pd

import chart_data
import decision_feed
import indicators
import paper_trader as pt
import risk
import structured_log
from test_paper_trader import CONFIG, run
from test_strategy_framework import synthetic_panel


def frame_from_panel(panel, ticker):
    return pd.DataFrame({"open": panel.open[ticker], "high": panel.high[ticker], "low": panel.low[ticker],
                         "close": panel.close[ticker], "volume": panel.volume[ticker]})


class IndicatorTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = synthetic_panel()
        cls.frame = frame_from_panel(cls.panel, cls.panel.close.columns[0])

    def test_shared_definitions_match_engine(self):
        out = indicators.compute(self.frame)
        close = self.frame["close"]
        pd.testing.assert_series_equal(out["atr14"], risk.atr(self.frame["high"], self.frame["low"], close, 14),
                                       check_names=False)
        lookback = 20
        manual = (close - close.rolling(lookback).mean()) / close.rolling(lookback).std(ddof=0)
        np.testing.assert_allclose(out["z20"].dropna(), manual.dropna(), rtol=1e-9)
        self.assertTrue(((out["rsi14"].dropna() >= 0) & (out["rsi14"].dropna() <= 100)).all())
        self.assertTrue(out["tsmom_signal"].dropna().between(0, 1).all())
        upper, lower = out["donchian20_upper"], out["donchian20_lower"]
        self.assertAlmostEqual(upper.iloc[30], self.frame["high"].iloc[10:30].max())    # excludes the current bar
        self.assertAlmostEqual(lower.iloc[30], self.frame["low"].iloc[10:30].min())
        np.testing.assert_allclose(out["sma50"].iloc[-1], close.iloc[-50:].mean())

    def test_rsi_extremes_and_relative_strength(self):
        up = pd.Series(np.arange(1, 60, dtype=float))
        self.assertEqual(indicators.rsi(up).iloc[-1], 100.0)
        frame = self.frame.copy()
        out = indicators.compute(frame, market_close=frame["close"] / 2)
        np.testing.assert_allclose(out["rs_spy"].dropna(), 100.0)


class ChartPayloadTests(TestCase):
    def test_daily_payload_from_local_db(self):
        panel = synthetic_panel(sessions=260)
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "prices.sqlite3"
            with closing(sqlite3.connect(db)) as connection:
                connection.execute("CREATE TABLE daily_prices (ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL)")
                for ticker in [panel.close.columns[1], "SPY"]:
                    source = panel.close.columns[1] if ticker == "SPY" else ticker
                    frame = frame_from_panel(panel, source)
                    connection.executemany("INSERT INTO daily_prices VALUES (?,?,?,?,?,?,?)",
                                           [(ticker, d.strftime("%Y-%m-%d"), *row) for d, row in
                                            zip(frame.index, frame[["open", "high", "low", "close", "volume"]].to_numpy().tolist())])
                connection.commit()
            with mock.patch.object(chart_data, "DATABASE", db), \
                    mock.patch.object(chart_data, "PAPER_DB", Path(directory) / "none.sqlite3"), \
                    mock.patch.object(chart_data, "INTRADAY_DB", Path(directory) / "none2.sqlite3"):
                payload = chart_data.build(panel.close.columns[1], "1d")
                self.assertEqual(len(payload["bars"]), 260)
                self.assertIn("sma200", payload["indicators"])
                self.assertEqual(len(payload["indicators"]["sma200"]), 61)
                np.testing.assert_allclose([p["value"] for p in payload["indicators"]["rs_spy"]], 100.0)
                json.dumps(payload)                                  # serialisable (no NaN)
                missing = chart_data.build("ZZZZ", "1d")
                self.assertTrue(missing["warnings"])
                intraday = chart_data.build("SPY", "5m")
                self.assertIn("intraday", " ".join(intraday["warnings"]))


class DecisionFeedTests(TestCase):
    def test_paper_decisions_become_chat_messages_once(self):
        panel = synthetic_panel()
        with tempfile.TemporaryDirectory() as directory:
            paper_db, chat_db = Path(directory) / "paper.sqlite3", Path(directory) / "chat.sqlite3"
            with closing(pt.connect(paper_db)) as connection:
                for asof in panel.dates[200:240]:
                    run(connection, panel, asof)
                fills = connection.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
            self.assertGreater(fills, 0)
            first = decision_feed.ingest(paper_db, chat_db)
            self.assertGreater(first["added"], 0)
            self.assertEqual(decision_feed.ingest(paper_db, chat_db)["added"], 0)    # idempotent
            with closing(sqlite3.connect(chat_db)) as chat:
                chat.row_factory = sqlite3.Row
                rows = [dict(r) for r in chat.execute("SELECT * FROM chat_messages")]
            kinds = {r["kind"] for r in rows}
            self.assertIn("fill", kinds)
            self.assertIn("run", kinds)
            fill = next(r for r in rows if r["kind"] == "fill")
            self.assertTrue(fill["ticker"] and fill["qty"] and fill["price"])
            self.assertTrue(all(r["text"] for r in rows))
            decisions = [r for r in rows if r["kind"] in ("order", "skip", "stop", "exit", "signal")]
            self.assertTrue(decisions)
            self.assertTrue(all(r["rationale"] for r in decisions))


class StructuredLogTests(TestCase):
    def test_secrets_are_redacted(self):
        with mock.patch.dict("os.environ", {"SCHWAB_APP_SECRET": "verysecret123"}):
            text = structured_log.redact("secret verysecret123 and Bearer abcdefghijklmnop")
        self.assertNotIn("verysecret123", text)
        self.assertNotIn("abcdefghijklmnop", text)
        cleaned = structured_log._clean({"refresh_token": "x", "nested": {"accountNumber": "123"}, "ok": 1})
        self.assertEqual(cleaned["refresh_token"], "[redacted]")
        self.assertEqual(cleaned["nested"]["accountNumber"], "[redacted]")
        self.assertEqual(cleaned["ok"], 1)
