"""Load a date x ticker price panel from SQLite for research, backtests and paper trading."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fetch_and_model import DATABASE
from price_history import complete_market_dates, open_history
from universe import CORE_ETFS, classify

MARKET = "SPY"


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    dividends: pd.DataFrame
    classes: pd.Series = field(default_factory=lambda: pd.Series(dtype=object))
    masked_returns: pd.DataFrame | None = None   # True where a one-day return is ignored (status 'masked', e.g. spin-offs)
    warnings: list = field(default_factory=list)
    overlap: dict | None = None
    used_long_history: bool = False
    sources: dict | None = None

    @property
    def dates(self):
        return self.close.index

    def upto(self, asof):
        """Copy of the panel containing only rows dated <= asof (no lookahead)."""
        asof = pd.Timestamp(asof)
        cut = lambda frame: frame.loc[:asof]
        return Panel(cut(self.open), cut(self.high), cut(self.low), cut(self.close), cut(self.volume),
                     cut(self.dividends), self.classes,
                     None if self.masked_returns is None else cut(self.masked_returns),
                     warnings=list(self.warnings or []), overlap=self.overlap,
                     used_long_history=self.used_long_history, sources=self.sources)

    def total_return(self):
        """Close-to-close total return including cash dividends on the ex-date; masked days are NaN."""
        result = (self.close + self.dividends) / self.close.shift() - 1
        if self.masked_returns is not None:
            result = result.mask(self.masked_returns.reindex_like(result).fillna(False).astype(bool))
        return result

    def log_returns(self):
        return np.log1p(self.total_return())


def _table_exists(connection, name):
    return connection.execute("SELECT 1 FROM sqlite_schema WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _as_day(value):
    """Normalize a date or timestamp to a YYYY-MM-DD string."""
    if value is None:
        return None
    return pd.Timestamp(value).date().isoformat()


def load_panel(start=None, end=None, min_avg_dollar_volume=5e6, min_avg_price=2.0, tickers=None,
               exclude_open_anomalies=True, database=DATABASE, use_long_history=True):
    """Load complete sessions between start and end. Tickers are pre-filtered by average dollar volume and price
    over the loaded window (a loose screen; strategies apply causal daily liquidity filters themselves).

    When Massive/prices.sqlite3 also has a `daily_long` table (Schwab 20-year daily history), dates before
    (or after) the Massive grouped sessions are filled from that table. Overlapping dates prefer Massive.
    The panel carries `.warnings` (survivorship bias, adjustment notes, overlap mismatch rates).
    """
    start = _as_day(start)
    end = _as_day(end)
    with open_history(database) as connection:
        sessions = sorted(complete_market_dates(connection))
        long_available = use_long_history and _table_exists(connection, "daily_long")
        long_span = None
        if long_available:
            long_span = connection.execute(
                "SELECT MIN(date), MAX(date) FROM daily_long WHERE source='schwab'").fetchone()
        massive_start = sessions[0] if sessions else (long_span[0] if long_span else None)
        massive_end = sessions[-1] if sessions else (long_span[1] if long_span else None)
        long_first = long_span[0] if long_span else None
        long_last = long_span[1] if long_span else None
        if not start:
            start = min((day for day in (massive_start, long_first) if day), default=None)
        if not end:
            end = max((day for day in (massive_end, long_last) if day), default=None)
        if not start or not end:
            raise ValueError("No price sessions are available to load.")
        sessions = [day for day in sessions if start <= day <= end]
        if tickers is None:
            keep = {row[0] for row in connection.execute(
                """SELECT ticker FROM daily_prices WHERE date BETWEEN ? AND ?
                   GROUP BY ticker HAVING AVG(close * volume) >= ? AND AVG(close) >= ?""",
                (start, end, min_avg_dollar_volume, min_avg_price))}
            keep |= set(CORE_ETFS)
        else:
            keep = set(tickers) | {MARKET}
        flagged = set()
        masked = []
        if _table_exists(connection, "price_anomalies"):
            if exclude_open_anomalies:
                flagged = {row[0] for row in connection.execute(
                    """SELECT DISTINCT ticker FROM price_anomalies WHERE status IN ('open', 'persists')
                       AND kind IN ('jump_reversal','unadjusted_split','split_basis_mismatch')""")}
            masked = connection.execute(
                "SELECT ticker, date FROM price_anomalies WHERE status = 'masked'").fetchall()
        keep -= flagged - {MARKET}
        # A targeted analysis or intraday watch only needs its requested symbols plus SPY.
        ticker_filter, ticker_params = "", ()
        if tickers is not None and len(keep) <= 500:
            ticker_params = tuple(sorted(keep))
            ticker_filter = " AND ticker IN (" + ", ".join("?" for _ in ticker_params) + ")"
        frame = pd.read_sql_query(
            "SELECT ticker, date, open, high, low, close, volume FROM daily_prices WHERE date BETWEEN ? AND ?"
            + ticker_filter, connection, params=(start, end, *ticker_params))
        dividends = pd.DataFrame(columns=["ticker", "market_date", "cash"])
        if _table_exists(connection, "backtest_dividends"):
            dividends = pd.read_sql_query(
                "SELECT ticker, market_date, cash FROM backtest_dividends WHERE unsupported = 0 AND market_date BETWEEN ? AND ?"
                + ticker_filter, connection, params=(start, end, *ticker_params))
        long_rows = []
        if long_available and keep:
            from schwab_daily import load_long_frame
            long_rows = load_long_frame(connection, start=start, end=end, tickers=sorted(keep))
    frame = frame[frame.ticker.isin(keep)] if len(frame) else frame
    used_long = False
    meta = {}
    extra_dates = []
    if long_rows:
        from schwab_daily import data_warnings, merge_rows
        massive_rows = frame.to_dict("records") if len(frame) else []
        for row in massive_rows:
            row["date"] = _as_day(row["date"])
        for row in long_rows:
            row["date"] = _as_day(row["date"])
        merged, meta = merge_rows(massive_rows, long_rows, prefer="massive")
        frame = pd.DataFrame(merged)
        used_long = True
        session_set = set(sessions)
        extra_dates = sorted({row["date"] for row in long_rows
                              if start <= row["date"] <= end and row["date"] not in session_set})
    index = pd.DatetimeIndex(pd.to_datetime(sessions + extra_dates)).sort_values().unique()
    columns = sorted(frame.ticker.unique()) if len(frame) else []
    if len(frame):
        frame["date"] = pd.to_datetime(frame["date"])

    def pivot(name):
        if not len(frame):
            return pd.DataFrame(index=index, columns=columns, dtype=float)
        return frame.pivot(index="date", columns="ticker", values=name).reindex(index=index, columns=columns).astype(float)

    dividends = dividends[dividends.ticker.isin(columns)] if len(dividends) else dividends
    if len(dividends):
        dividends["market_date"] = pd.to_datetime(dividends["market_date"])
    cash = (dividends.pivot_table(index="market_date", columns="ticker", values="cash", aggfunc="sum")
            .reindex(index=index, columns=columns).fillna(0.0)) if len(dividends) else \
        pd.DataFrame(0.0, index=index, columns=columns)
    mask = pd.DataFrame(False, index=index, columns=columns)
    for ticker, day in masked:
        stamp = pd.Timestamp(day)
        if ticker in mask.columns and stamp in mask.index:
            mask.loc[stamp, ticker] = True
    warnings = []
    if used_long:
        from schwab_daily import data_warnings
        warnings = data_warnings(meta, used_long=True)
    return Panel(pivot("open"), pivot("high"), pivot("low"), pivot("close"), pivot("volume"), cash,
                 classify(columns), mask, warnings=warnings, overlap=(meta or {}).get("overlap"),
                 used_long_history=used_long, sources={"massive": True, "schwab_long": used_long})
