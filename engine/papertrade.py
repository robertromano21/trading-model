"""Price fixed seven-session paper trades from the saved SQLite closes."""

import json
import logging
import sqlite3
import sys
from bisect import bisect_right
from datetime import date, timedelta

import exchange_calendars

from fetch_and_model import DATABASE
from price_history import open_history

HOLD_SESSIONS = 7


def trade_marks(connection, trades):
    entries = [date.fromisoformat(trade["entryDate"]) for trade in trades]
    calendar = exchange_calendars.get_calendar("XNYS", start=min(entries), end=max(entries) + timedelta(days=30))
    sessions = calendar.sessions.strftime("%Y-%m-%d").tolist()
    positions = {day: index for index, day in enumerate(sessions)}
    latest = connection.execute("SELECT MAX(date) FROM daily_prices").fetchone()[0]
    latest_index = bisect_right(sessions, latest) - 1 if latest else -1
    marks = []
    for trade in trades:
        ticker, entry = trade["ticker"], trade["entryDate"]
        if entry not in positions:
            raise ValueError(f"{entry} is not a scheduled trading session.")
        entry_index = positions[entry]
        exit_day = sessions[entry_index + HOLD_SESSIONS]
        held = min(HOLD_SESSIONS, max(0, latest_index - entry_index))
        opening = connection.execute(
            "SELECT close FROM daily_prices WHERE ticker = ? AND date = ? AND close > 0", (ticker, entry)).fetchone()
        holding_days = sessions[entry_index:entry_index + HOLD_SESSIONS + 1]
        placeholders = ",".join("?" for _ in holding_days)
        last = connection.execute(
            f"""SELECT date, close FROM daily_prices
                WHERE ticker = ? AND date IN ({placeholders}) AND close > 0
                ORDER BY date DESC LIMIT 1""", (ticker, *holding_days)).fetchone()
        mark_day, price = tuple(last) if last else (None, None)
        status = "closed" if mark_day == exit_day else "awaiting-exit" if held == HOLD_SESSIONS else "open"
        marks.append({
            "ticker": ticker, "entryDate": entry, "entryPrice": opening[0] if opening else None,
            "exitDate": exit_day, "heldSessions": held, "holdSessions": HOLD_SESSIONS,
            "price": price, "priceDate": mark_day, "status": status, "asOf": latest,
        })
    return marks


def main(request):
    if not isinstance(request, dict) or not isinstance(request.get("trades"), list) or not request["trades"]:
        raise ValueError("Choose paper trades to refresh.")
    trades = request["trades"]
    for trade in trades:
        if not isinstance(trade, dict) or not isinstance(trade.get("ticker"), str) or not trade["ticker"].strip():
            raise ValueError("A paper trade needs a ticker.")
        entry = trade.get("entryDate")
        if not isinstance(entry, str) or date.fromisoformat(entry).isoformat() != entry:
            raise ValueError("A paper trade needs a valid market date.")
    if not DATABASE.is_file():
        raise ValueError("Import market prices before starting a paper trade.")
    with open_history(DATABASE) as connection:
        connection.execute("BEGIN")
        return {"status": "complete", "marks": trade_marks(connection, trades)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        result = main(json.load(sys.stdin))
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    except (OSError, ValueError, sqlite3.Error) as error:
        logging.error("%s", error)
        print(json.dumps({"status": "failed", "message": str(error)}))
