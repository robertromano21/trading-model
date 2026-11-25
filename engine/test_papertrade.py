from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from papertrade import trade_marks
from price_history import open_history, save_session


class PapertradeTests(TestCase):
    def test_holiday_week_hold_uses_the_scheduled_seventh_close(self):
        trade = {"ticker": "TEST", "entryDate": "2026-11-23"}
        stat = SimpleNamespace(st_size=1, st_mtime_ns=1)

        def bar(ticker, close):
            return {"ticker": ticker, "open": close, "high": close, "low": close, "close": close, "volume": 100}

        with TemporaryDirectory() as directory, open_history(Path(directory) / "prices.sqlite3", writable=True) as db:
            save_session(db, "2026-11-23", [bar("TEST", 40)], stat)
            save_session(db, "2026-11-26", [bar("TEST", 99)], stat)
            save_session(db, "2026-11-27", [bar("OTHER", 25)], stat)
            opened, = trade_marks(db, [trade])
            self.assertEqual(opened["exitDate"], "2026-12-03")
            self.assertEqual((opened["status"], opened["heldSessions"], opened["price"]), ("open", 3, 40))

            save_session(db, "2026-12-04", [bar("TEST", 60)], stat)
            waiting, = trade_marks(db, [trade])
            self.assertEqual((waiting["status"], waiting["heldSessions"], waiting["price"]), ("awaiting-exit", 7, 40))

            save_session(db, "2026-12-03", [bar("TEST", 44)], stat)
            closed, = trade_marks(db, [trade])
            self.assertEqual((closed["status"], closed["entryPrice"], closed["price"]), ("closed", 40, 44))
            self.assertEqual(closed["priceDate"], closed["exitDate"])
