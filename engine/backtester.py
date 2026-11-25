"""Deterministic daily-bar backtests for a ticker or a long-only cash portfolio."""

import json
import logging
import math
import re
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from itertools import groupby
from uuid import uuid4

import numpy as np
import pandas as pd

from dividend_history import dividend_coverage, read_dividends
from fetch_and_model import DATABASE
from price_history import complete_market_dates, coverage, open_history

LOGGER = logging.getLogger(__name__)


def parameter(label, default, minimum, maximum, step=1):
    return {"label": label, "default": default, "min": minimum, "max": maximum,
            "step": step, "type": "integer" if step == 1 else "number"}


def signals_at_crossing(values, entry, exit_level, *, descending_entry=False):
    prior = values.shift()
    if descending_entry:
        return ((values < entry) & (prior >= entry), (values > exit_level) & (prior <= exit_level), -values)
    return ((values > entry) & (prior <= entry), (values < exit_level) & (prior >= exit_level), values)


def sma_crossover(close, params):
    fast = close.rolling(params["fast"]).mean()
    slow = close.rolling(params["slow"]).mean()
    return signals_at_crossing(fast.div(slow).sub(1), 0, 0)


def wilder(values, period):
    seeded = values.copy()
    if len(seeded) < period:
        return np.full(len(seeded), np.nan)
    seeded[period - 1] = seeded[:period].mean()
    seeded[:period - 1] = np.nan
    return pd.Series(seeded).ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def rsi_reversion(close, params):
    changes = np.diff(close.to_numpy())
    gains = wilder(changes.clip(min=0), params["period"])
    losses = wilder((-changes).clip(min=0), params["period"])
    total = gains + losses
    values = np.divide(gains * 100, total, out=np.full_like(total, 50.0), where=total != 0)
    oscillator = pd.Series(np.concatenate(([np.nan], values)))
    return signals_at_crossing(oscillator, params["oversold"], params["exitRsi"], descending_entry=True)


def zscore_breakout(close, params):
    window = close.rolling(params["lookback"])
    zscore = close.sub(window.mean()).div(window.std(ddof=0).replace(0, np.nan))
    return signals_at_crossing(zscore, params["entryZ"], params["exitZ"])


STRATEGIES = {
    "sma_crossover": ("SMA Crossover", sma_crossover, {
        "fast": parameter("Fast SMA (sessions)", 20, 2, 500),
        "slow": parameter("Slow SMA (sessions)", 50, 3, 500),
    }),
    "rsi_mean_reversion": ("RSI Mean Reversion", rsi_reversion, {
        "period": parameter("RSI period (sessions)", 14, 2, 250),
        "oversold": parameter("Oversold entry", 30, 1, 99, 0.1),
        "exitRsi": parameter("Exit RSI", 55, 1, 99, 0.1),
    }),
    "zscore_breakout": ("Z-Score Breakout", zscore_breakout, {
        "lookback": parameter("Z-score window (sessions)", 20, 3, 500),
        "entryZ": parameter("Entry Z-score", 2, -5, 5, 0.1),
        "exitZ": parameter("Exit Z-score", 0, -5, 5, 0.1),
    }),
}
COMMON = {
    "initialCapital": parameter("Starting capital ($)", 10000, 1, 1_000_000_000, 0.01),
    "commissionPct": parameter("Commission per side (%)", 0.1, 0, 25, 0.01),
    "fixedFee": parameter("Fixed fee per fill ($)", 0, 0, 100000, 0.01),
    "slippageBps": parameter("Slippage per side (bps)", 5, 0, 2500, 0.1),
    "stopLossPct": parameter("Stop loss (%)", 0, 0, 99.99, 0.01),
    "maxPositions": parameter("Maximum positions", 10, 1, 500),
}


def registry():
    return {"status": "complete", "strategies": [
        {"id": key, "name": name, "parameters": parameters}
        for key, (name, _strategy, parameters) in STRATEGIES.items()
    ], "settings": COMMON}


def numbers(values, definitions):
    if not isinstance(values, dict) or values.keys() - definitions.keys():
        raise ValueError("Unrecognized backtest parameters.")
    result = {}
    for key, field in definitions.items():
        value = values.get(key, field["default"])
        valid = type(value) in (int, float) and field["min"] <= value <= field["max"]
        if not valid or (field["type"] == "integer" and int(value) != value):
            raise ValueError(f"{field['label']} must be between {field['min']} and {field['max']}.")
        result[key] = int(value) if field["type"] == "integer" else value
    return result


def parse_config(config):
    if not isinstance(config, dict):
        raise ValueError("Provide a backtest configuration.")
    strategy = config.get("strategy")
    if not isinstance(strategy, str) or strategy not in STRATEGIES:
        raise ValueError("Choose an available strategy.")
    name, _function, definition = STRATEGIES[strategy]
    params = numbers(config.get("params", {}), definition)
    higher_first = {"rsi_mean_reversion": ("exitRsi", "oversold"), "zscore_breakout": ("entryZ", "exitZ"),
                    "sma_crossover": ("slow", "fast")}
    higher, lower = higher_first[strategy]
    if params[higher] <= params[lower]:
        raise ValueError(f"{definition[higher]['label']} must be greater than {definition[lower]['label']}.")
    mode = config.get("mode")
    tickers = config.get("tickers")
    if mode not in ("single", "portfolio") or not isinstance(tickers, list):
        raise ValueError("Choose single-ticker or portfolio mode and select tickers.")
    if (mode == "single" and len(tickers) != 1) or any(
        not isinstance(ticker, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,63}", ticker) for ticker in tickers
    ) or len(set(tickers)) != len(tickers):
        raise ValueError("Choose distinct, valid tickers. Single-ticker mode requires one symbol.")
    try:
        start, end = (date.fromisoformat(config[key]) for key in ("startDate", "endDate"))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Enter valid start and end dates.") from None
    if start >= end:
        raise ValueError("The backtest end date must be after its start date.")
    dividends = config.get("includeDividends", False)
    if type(dividends) is not bool:
        raise ValueError("Dividend inclusion must be true or false.")
    settings = numbers(config.get("costs", {}), COMMON)
    slots = 1 if mode == "single" else min(settings["maxPositions"], len(tickers) or settings["maxPositions"])
    return {"strategy": strategy, "strategyName": name, "params": params, "mode": mode, "tickers": tickers,
            "startDate": start.isoformat(), "endDate": end.isoformat(), "includeDividends": dividends,
            "costs": {**settings, "maxPositions": slots}}


def universe(connection):
    market = sorted(complete_market_dates(connection))
    cached = dividend_coverage(connection)
    distributions = {}
    if cached:
        distributions = {ticker: (events, missing) for ticker, events, missing in connection.execute(
            "SELECT ticker, COUNT(*), SUM(unsupported) FROM backtest_dividends GROUP BY ticker"
        )}
    rows = connection.execute(
        """SELECT p.ticker, MIN(p.date) AS firstDate, MAX(p.date) AS lastDate, COUNT(*) AS bars,
                  SUM(CASE WHEN s.bar_count > 0 THEN 1 ELSE 0 END) AS marketBars
           FROM daily_prices AS p LEFT JOIN market_sessions AS s ON s.date = p.date
           GROUP BY p.ticker ORDER BY p.ticker"""
    )
    tickers = []
    for row in rows:
        details = dict(row)
        expected = bisect_right(market, row["lastDate"]) - bisect_left(market, row["firstDate"])
        details["missingMarketSessions"] = expected - details.pop("marketBars")
        details["dividendDates"], details["unsupportedDividends"] = distributions.get(row["ticker"], (0, 0))
        tickers.append(details)
    return {"status": "complete", "prices": coverage(connection), "dividends": cached,
            "tickers": tickers}


def adjusted_signals(close, cash_dividends):
    return close * np.cumprod(1 + cash_dividends / close)


@dataclass
class Series:
    sessions: np.ndarray
    opening: np.ndarray
    low: np.ndarray
    close: np.ndarray
    exit: np.ndarray

    def index(self, session):
        index = int(self.sessions.searchsorted(session))
        return index if index < len(self.sessions) and self.sessions[index] == session else None


def price_rows(connection, through, tickers):
    # An empty list represents all tickers available on the saved test dates.
    columns = "ticker, date, open, low, close"
    if not tickers:
        return connection.execute(f"SELECT {columns} FROM daily_prices WHERE date <= ? ORDER BY ticker, date", (through,))

    def selected_rows():
        for ticker in sorted(set(tickers)):
            yield from connection.execute(
                f"SELECT {columns} FROM daily_prices WHERE ticker = ? AND date <= ? ORDER BY date", (ticker, through),
            )

    return selected_rows()


def load_prices(connection, config, calendar, start_index, dividend_data, benchmark):
    series, entries = {}, defaultdict(list)
    calendar_index = {day: index for index, day in enumerate(calendar)}
    selected = set(config["tickers"])
    through = config["endDate"]
    strategy = STRATEGIES[config["strategy"]][1]
    symbols = [*selected, benchmark] if selected else []
    for ticker, rows in groupby(price_rows(connection, through, symbols), key=lambda row: row[0]):
        bars = [(calendar_index[day], opening, low, close) for _, day, opening, low, close in rows
                if day in calendar_index and opening > 0 and low > 0 and close > 0]
        if not bars:
            continue
        data = np.asarray(bars, dtype=float)
        sessions = data[:, 0].astype(np.int32)
        opening, low, close = data[:, 1:].T
        cash = dividend_data.get(ticker, {})
        signal_close = adjusted_signals(close, np.array([cash.get(calendar[i], (0, 0))[0] for i in sessions])) if cash else close
        signal_entry, signal_exit, score = strategy(pd.Series(signal_close), config["params"])
        if not selected or ticker in selected:
            for index in np.flatnonzero(signal_entry.to_numpy()):
                next_session = int(sessions[index]) + 1
                if next_session <= start_index or index + 1 == len(sessions) or sessions[index + 1] != next_session:
                    continue
                entries[next_session].append((float(score.iloc[index]), ticker, float(opening[index + 1]), index + 1))
        series[ticker] = Series(sessions, opening, low, close, signal_exit.to_numpy())
    return series, entries


@dataclass
class Position:
    entry: int
    price: float
    units: float
    cost: float
    fee: float
    stop: float
    mark: float
    income: float = 0.0


def simulate(config, calendar, start_index, series, entries, dividends, benchmark):
    costs = config["costs"]
    cash = initial = equity = costs["initialCapital"]
    fee_rate, fixed_fee = costs["commissionPct"] / 100, costs["fixedFee"]
    slip, loss_pct = costs["slippageBps"] / 10000, costs["stopLossPct"] / 100
    slots = costs["maxPositions"]
    fees_paid = dividend_income = benchmark_units = benchmark_cash = 0.0
    benchmark_data = series.get(benchmark)
    benchmark_mark = None
    portfolio, trades, curve = {}, [], []

    def close_position(ticker, session, reference_price, reason):
        nonlocal cash, fees_paid
        position = portfolio.pop(ticker)
        price = reference_price * (1 - slip)
        gross = position.units * price
        fee = gross * fee_rate + fixed_fee
        fees_paid += fee
        cash += gross - fee
        pnl = gross - fee + position.income - position.cost
        trades.append({"ticker": ticker, "side": "LONG", "entryDate": calendar[position.entry],
                       "exitDate": calendar[session], "entryPrice": position.price, "exitPrice": price,
                       "units": position.units, "pnl": pnl, "pnlPct": 100 * pnl / position.cost,
                       "fees": position.fee + fee, "dividends": position.income, "reason": reason})

    for session in range(start_index, len(calendar)):
        day = calendar[session]
        at_open = {}
        for ticker, position in list(portfolio.items()):
            distribution, _ = dividends.get(ticker, {}).get(day, (0, 0))
            income = position.units * distribution
            cash += income
            dividend_income += income
            position.income += income
            position.stop = max(0, position.stop - distribution)
            data = series[ticker]
            index = data.index(session)
            if index is None:
                continue
            opening = data.opening[index]
            signal_exit = index > 0 and data.sessions[index - 1] >= position.entry and data.exit[index - 1]
            if position.stop and opening <= position.stop:
                close_position(ticker, session, opening, "Stop gap")
            elif signal_exit:
                close_position(ticker, session, opening, "Strategy signal")
            else:
                at_open[ticker] = index

        budget = max(0, equity) / slots
        for _score, ticker, opening, index in sorted(entries.get(session, ()), key=lambda row: (-row[0], row[1])):
            if len(portfolio) >= slots or cash <= fixed_fee:
                break
            if ticker in portfolio:
                continue
            allocation = min(cash, budget)
            if allocation <= fixed_fee:
                break
            price = opening * (1 + slip)
            units = (allocation - fixed_fee) / (price * (1 + fee_rate))
            fee = units * price * fee_rate + fixed_fee
            portfolio[ticker] = Position(session, price, units, allocation, fee,
                                         price * (1 - loss_pct) if loss_pct else 0, opening)
            cash -= allocation
            fees_paid += fee
            at_open[ticker] = index

        for ticker, index in at_open.items():
            position, data = portfolio[ticker], series[ticker]
            if position.stop and data.low[index] <= position.stop:
                close_position(ticker, session, min(position.stop, data.opening[index]), "Stop loss")
            else:
                position.mark = float(data.close[index])

        benchmark_value = None
        if benchmark_data:
            dividend, _ = dividends.get(benchmark, {}).get(day, (0, 0))
            benchmark_cash += benchmark_units * dividend
            index = benchmark_data.index(session)
            if index is not None:
                if not benchmark_units:
                    benchmark_units = initial / benchmark_data.opening[index]
                benchmark_mark = float(benchmark_data.close[index])
            if benchmark_mark is not None:
                benchmark_value = benchmark_units * benchmark_mark + benchmark_cash

        equity = cash + sum(item.units * item.mark for item in portfolio.values())
        curve.append({"date": day, "equity": float(equity), "benchmark": benchmark_value})

    return curve, trades, fees_paid, dividend_income, portfolio


def metrics(config, curve, trades, fees, income, positions):
    initial = config["costs"]["initialCapital"]
    equity = np.array([initial, *(row["equity"] for row in curve)])
    peak = np.maximum.accumulate(equity)
    previous = equity[:-1]
    returns = np.divide(np.diff(equity), previous, out=np.zeros_like(previous), where=previous > 0)
    annual_days = (date.fromisoformat(curve[-1]["date"]) - date.fromisoformat(curve[0]["date"])).days
    ratio = equity[-1] / initial
    annualized = -100
    if ratio > 0:
        exponent = math.log(ratio) * 365.25 / annual_days
        annualized = math.expm1(exponent) * 100 if exponent < math.log(sys.float_info.max / 100) else None
    deviation = returns.std(ddof=1)
    benchmark = curve[-1]["benchmark"]
    return {
        "initialCapital": initial, "finalEquity": float(equity[-1]), "totalReturnPct": float((ratio - 1) * 100),
        "cagrPct": annualized, "maxDrawdownPct": float(np.min(equity / peak - 1) * 100),
        "sharpeRatio": float(returns.mean() / deviation * math.sqrt(252)) if deviation > 0 else None,
        "winRatePct": sum(trade["pnl"] > 0 for trade in trades) / len(trades) * 100 if trades else None,
        "totalTrades": len(trades), "totalFeesPaid": fees, "dividendIncome": income, "openPositions": len(positions),
        "benchmarkReturnPct": float((benchmark / initial - 1) * 100) if benchmark is not None else None,
    }


def run(connection, config):
    config = parse_config(config)
    selected, end = config["tickers"], config["endDate"]
    benchmark = selected[0] if config["mode"] == "single" else "SPY"
    if config["mode"] == "single":
        dates = [row[0] for row in connection.execute(
            "SELECT date FROM daily_prices WHERE ticker = ? AND date <= ? ORDER BY date", (benchmark, end),
        )]
    else:
        dates = [day for day in sorted(complete_market_dates(connection)) if day <= end]
    index = bisect_left(dates, config["startDate"])
    if len(dates) - index < 2:
        raise ValueError("Choose a period with at least two saved trading sessions.")
    warnings = []
    dividends = {}
    if config["includeDividends"]:
        cached = dividend_coverage(connection)
        if not cached or cached["firstDate"] > dates[0] or cached["lastDate"] < dates[-1]:
            raise ValueError("Dividend data does not cover the price and warm-up window. Sync dividends or choose price-only returns.")
        dividends = read_dividends(connection, end, benchmark if config["mode"] == "single" else None)

    series, entries = load_prices(connection, config, dates, index, dividends, benchmark)
    missing = sorted(set(selected) - {ticker for ticker, data in series.items() if data.sessions[-1] >= index})
    if missing:
        raise ValueError("No usable prices in the test period for " + ", ".join(missing[:8]) + (", and others." if len(missing) > 8 else "."))
    symbol_count = len(selected) if selected else sum(data.sessions[-1] >= index for data in series.values())
    curve, trades, fees, income, positions = simulate(config, dates, index, series, entries, dividends, benchmark)
    unsupported = sum(
        amount for ticker, events in dividends.items() if ticker in series
        for day, (_cash, amount) in events.items() if dates[0] <= day <= dates[-1]
    )
    if unsupported:
        warnings.append(f"{unsupported} distribution records in the modeled data lacked a usable split-adjusted USD amount.")
    if positions:
        stale = sum(series[ticker].sessions[-1] != len(dates) - 1 for ticker in positions)
        if stale:
            warnings.append(f"{stale} open positions use the last available close because the final session has no bar.")
    if benchmark not in series or curve[-1]["benchmark"] is None:
        warnings.append(f"{benchmark} buy-and-hold prices are unavailable for the selected period.")
    return {
        "status": "complete", "runId": f"bt_{uuid4().hex}", "createdAt": datetime.now(timezone.utc).isoformat(),
        **config, "benchmark": benchmark, "symbolCount": int(symbol_count),
        "firstDate": curve[0]["date"], "lastDate": curve[-1]["date"],
        "metrics": metrics(config, curve, trades, fees, income, positions),
        "equityCurve": curve, "trades": trades, "warnings": warnings,
    }


def main(args):
    if args == ["--strategies"]:
        return registry()
    if not DATABASE.is_file():
        if args == ["--universe"]:
            return {"status": "complete", "prices": {}, "dividends": None, "tickers": []}
        raise ValueError("Import historical prices before running a backtest.")
    with open_history(DATABASE) as connection:
        connection.execute("BEGIN")
        if args == ["--universe"]:
            return universe(connection)
        return run(connection, json.load(sys.stdin))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(main(sys.argv[1:]), allow_nan=False, separators=(",", ":")))
    except (OSError, ValueError, sqlite3.Error) as error:
        LOGGER.error("%s", error)
        print(json.dumps({"status": "failed", "message": str(error),
                          "errorType": "validation" if isinstance(error, ValueError) else "runtime"}))
