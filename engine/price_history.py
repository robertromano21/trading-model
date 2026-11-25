"""SQLite storage for complete daily sessions and per-ticker queries."""

import sqlite3
from contextlib import contextmanager
from itertools import groupby

BAR_COLUMNS = "ticker, date, open, high, low, close, volume"
SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    PRIMARY KEY (ticker, date)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS daily_prices_by_date ON daily_prices (date, ticker);
CREATE TABLE IF NOT EXISTS market_sessions (
    date TEXT PRIMARY KEY,
    bar_count INTEGER NOT NULL,
    file_size INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL
) WITHOUT ROWID;
"""


@contextmanager
def open_history(path, *, writable=False):
    if writable:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
    else:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    try:
        connection.row_factory = sqlite3.Row
        if writable:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(SCHEMA)
        yield connection
    finally:
        connection.close()


def session_manifest(connection):
    return {row["date"]: (row["bar_count"], row["file_size"], row["modified_ns"])
            for row in connection.execute("SELECT date, bar_count, file_size, modified_ns FROM market_sessions")}


def save_session(connection, date, bars, stat):
    """Publish the bars and completion marker together, including empty market days."""
    values = ((bar["ticker"], date, bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"])
              for bar in bars)
    with connection:
        connection.execute("DELETE FROM daily_prices WHERE date = ?", (date,))
        connection.executemany(f"INSERT INTO daily_prices ({BAR_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)", values)
        connection.execute(
            """INSERT INTO market_sessions VALUES (?, ?, ?, ?)
               ON CONFLICT (date) DO UPDATE SET bar_count = excluded.bar_count,
                 file_size = excluded.file_size, modified_ns = excluded.modified_ns""",
            (date, len(bars), stat.st_size, stat.st_mtime_ns),
        )


def save_ticker_bars(connection, bars):
    """Fill dates without a grouped download. A later grouped session replaces those rows."""
    values = ((bar["ticker"], bar["date"], bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"], bar["date"])
              for bar in bars)
    before = connection.total_changes
    with connection:
        connection.executemany(
            f"""INSERT INTO daily_prices ({BAR_COLUMNS})
                SELECT ?, ?, ?, ?, ?, ?, ?
                WHERE NOT EXISTS (SELECT 1 FROM market_sessions WHERE date = ?)
                ON CONFLICT (ticker, date) DO NOTHING""",
            values,
        )
    return connection.total_changes - before


def complete_market_dates(connection):
    return {row["date"] for row in connection.execute("SELECT date FROM market_sessions WHERE bar_count > 0")}


def as_bar(row):
    bar = dict(row)
    opening, closing = bar["open"], bar["close"]
    bar["direction"] = "up" if closing > opening else "down" if closing < opening else "flat"
    return bar


def day_bars(connection, date):
    return [as_bar(row) for row in connection.execute(
        f"SELECT {BAR_COLUMNS} FROM daily_prices WHERE date = ? ORDER BY ticker", (date,),
    )]


def ticker_history(connection, ticker, through):
    return [as_bar(row) for row in connection.execute(
        f"""SELECT {BAR_COLUMNS} FROM daily_prices
            WHERE ticker = ? AND date <= ? ORDER BY date DESC LIMIT 50""",
        (ticker, through),
    )]


def model_history(connection, through):
    connection.execute("PRAGMA cache_size = -65536")
    rows = connection.execute(
        f"SELECT {BAR_COLUMNS} FROM daily_prices WHERE date <= ? ORDER BY date, ticker", (through,),
    )
    for market_date, group in groupby(rows, key=lambda row: row["date"]):
        yield market_date, [dict(row) for row in group]


def coverage(connection):
    row = connection.execute(
        """SELECT COUNT(*) AS sessions, COALESCE(SUM(bar_count), 0) AS bars,
                  MIN(date) AS firstDate, MAX(date) AS lastDate
           FROM market_sessions WHERE bar_count > 0""",
    ).fetchone()
    return dict(row)
