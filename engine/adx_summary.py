"""Apply the ADX workbook summary to a full market session in SQLite."""

import json
import logging
import sqlite3
import sys
from datetime import date
from itertools import groupby

import numpy as np
import pandas as pd

from excel_adx import adx_state, market_regime, moving_average_status
from fetch_and_model import DATABASE
from price_history import complete_market_dates, open_history

LOGGER = logging.getLogger(__name__)


def worksheet_adx(high, low):
    """Port the ticker sheets' directional movement, DX, and column AO formulas."""
    output = np.full(len(high), np.nan)
    if len(high) < 26:
        return output
    high_move, low_move = np.diff(high), np.diff(low)
    abs_high, abs_low = np.abs(high_move), np.abs(low_move)
    mixed = ((high_move >= 0) != (low_move >= 0)) | ((high_move > 0) != (low_move > 0))
    is_up = np.where(mixed, abs_high > abs_low, (high_move > 0) & (low_move > 0))

    def smooth(movement):
        # AR4/AR5 seed rows 3:14. AG15/AH15 apply the first 14-period decay.
        seed = movement[:12].sum() / 14
        return pd.Series(np.r_[seed, movement[12:]]).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()[1:]

    up = smooth(np.where(is_up, abs_high, 0))
    down = smooth(np.where(is_up, 0, abs_low))
    # Both directional indices divide by the same smoothed true range, so it cancels in DX.
    total = up + down
    dx = np.divide(100 * np.abs(up - down), total, out=np.full(len(total), np.nan), where=total > 0)
    first_valid = np.flatnonzero(np.isfinite(dx))
    if not len(first_valid):
        return output
    seed_end = first_valid[0] + 12
    if seed_end >= len(dx):
        return output
    # AR6 sums 13 DX rows. AO27 combines that seed with the last DX.
    seed = (dx[seed_end-12:seed_end+1].sum() / 14 * 13 + dx[seed_end]) / 14
    output[seed_end + 13:] = pd.Series(np.r_[seed, dx[seed_end+1:]]).ewm(alpha=1 / 14, adjust=False).mean()
    return output


def numeric(value):
    return float(value) if np.isfinite(value) else None


def moving_average_deviations(closes, window):
    averages = closes.rolling(window).mean().to_numpy()
    deviations = np.divide(closes.to_numpy() - averages, averages,
                           out=np.full(len(closes), np.nan), where=averages > 0)
    return averages, deviations


def ma20_gap_statistics(deviations):
    current = numeric(deviations[-1])
    historical = deviations[:-1]
    baseline = historical[:0]
    if current:
        same_side = historical > 0 if current > 0 else historical < 0
        baseline = historical[same_side & np.isfinite(historical)]
    count = len(baseline)
    mean = deviation = percentile = z_score = None
    if count:
        mean = float(baseline.mean())
        deviation = float(baseline.std()) if np.ptp(baseline) > 0 else 0.0
        below = np.count_nonzero(baseline < current)
        tied = np.count_nonzero(baseline == current)
        percentile = float((below + tied / 2) / count)
        if deviation > 0:
            z_score = (current - mean) / deviation
    return {
        "ma20Gap": current, "ma20GapMean": mean, "ma20GapStd": deviation,
        "ma20GapPercentile": percentile, "ma20GapCount": count, "ma20ZScore": z_score,
    }


def history_extreme(values, dates, high):
    if not np.isfinite(values).any():
        return None, None
    index = np.nanargmax(values) if high else np.nanargmin(values)
    return float(values[index]), dates[index]


def summarize(ticker, dates, prices):
    high, low, close = np.asarray(prices, dtype=float).T
    history = worksheet_adx(high, low)
    closes = pd.Series(close)
    averages, deviations = moving_average_deviations(closes, 50)
    _, deviations20 = moving_average_deviations(closes, 20)
    prior = -6
    have_prior = len(close) >= abs(prior)
    current_adx, prior_adx = numeric(history[-1]), numeric(history[prior]) if have_prior else None
    change = current_adx - prior_adx if current_adx is not None and prior_adx is not None else None
    state = adx_state(change) if change is not None else None
    current_pct, prior_pct = numeric(deviations[-1]), numeric(deviations[prior]) if have_prior else None
    ma_status = moving_average_status(prior_pct, current_pct) if current_pct is not None and prior_pct is not None else None
    high_adx, high_date = history_extreme(history, dates, high=True)
    low_adx, low_date = history_extreme(history, dates, high=False)
    finite_deviations = deviations[np.isfinite(deviations)]
    return {
        "ticker": ticker, "sessions": len(dates), "highAdx": high_adx, "highDate": high_date,
        "lowAdx": low_adx, "lowDate": low_date, "adx": current_adx, "priorAdx": prior_adx,
        "adxChange": change, "adxState": state, "close": float(close[-1]), "ma50": numeric(averages[-1]),
        "pct50": current_pct, "priorClose": float(close[prior]) if have_prior else None,
        "priorMa50": numeric(averages[prior]) if have_prior else None, "priorPct50": prior_pct,
        "pct50Change": current_pct - prior_pct if ma_status is not None else None, "maStatus": ma_status,
        "marketRegime": market_regime(state, ma_status) if state is not None and ma_status is not None else None,
        "minPct50": float(finite_deviations.min()) if len(finite_deviations) else None,
        "maxPct50": float(finite_deviations.max()) if len(finite_deviations) else None,
        **ma20_gap_statistics(deviations20),
    }


def market_summary(connection, through):
    dates = complete_market_dates(connection)
    if through not in dates:
        raise ValueError("Choose a saved full-market session for the ADX Summary.")
    cursor = connection.execute(
        """SELECT ticker, date, high, low, close FROM daily_prices
           WHERE ticker IN (SELECT ticker FROM daily_prices WHERE date = ?) AND date <= ?
           ORDER BY ticker, date""", (through, through),
    )
    result = []
    for ticker, history in groupby(cursor, key=lambda row: row["ticker"]):
        ticker_dates, prices = [], []
        for _, day, high, low, close in history:
            ticker_dates.append(day)
            prices.append((high, low, close))
        result.append(summarize(ticker, ticker_dates, prices))
    return {"status": "complete", "date": through, "rows": result,
            "readyCount": sum(row["marketRegime"] is not None for row in result)}


def main(args):
    if args == ["--dates"]:
        if not DATABASE.is_file():
            return {"status": "complete", "dates": []}
        with open_history(DATABASE) as connection:
            return {"status": "complete", "dates": sorted(complete_market_dates(connection), reverse=True)}
    if len(args) != 1:
        raise ValueError("Pass a saved market date.")
    through = date.fromisoformat(args[0]).isoformat()
    if not DATABASE.is_file():
        raise ValueError("Import market history to build the ADX Summary.")
    with open_history(DATABASE) as connection:
        connection.execute("PRAGMA cache_size = -65536")
        connection.execute("BEGIN")
        return market_summary(connection, through)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(main(sys.argv[1:]), allow_nan=False, separators=(",", ":")))
    except (OSError, ValueError, sqlite3.Error) as error:
        LOGGER.error("%s", error)
        print(json.dumps({"status": "failed", "message": str(error)}))
