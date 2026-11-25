"""Replay ADX Summary filters against daily bars and measure forward returns."""

import heapq
import json
import logging
import operator
import re
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from adx_summary import moving_average_deviations, worksheet_adx
from excel_adx import adx_state, market_regime, moving_average_status
from fetch_and_model import DATABASE
from price_history import complete_market_dates, open_history

HORIZON = 5
DETAIL_LIMIT = 50
MEMBERSHIP = Path(__file__).resolve().parents[1] / "public" / "sp500.json"
CATEGORIES = {"sp500", "adxState", "maStatus", "marketRegime"}
TEXT = {"ticker", "highDate", "lowDate"}
FRACTIONS = {"ma20Gap", "ma20GapPercentile", "maPosition", "pct50", "priorPct50",
             "pct50Change", "minPct50", "maxPct50"}
ADX_FIELDS = {"adx", "priorAdx", "adxChange", "adxState", "marketRegime",
              "highAdx", "lowAdx", "highDate", "lowDate"}
MA50_FIELDS = {"ma50", "priorMa50", "pct50", "priorPct50", "pct50Change", "maPosition",
               "gapChange", "maStatus", "marketRegime", "minPct50", "maxPct50"}
MA20_FIELDS = {"ma20Gap", "ma20ZScore", "ma20GapPercentile"}
FIELDS = CATEGORIES | TEXT | ADX_FIELDS | MA50_FIELDS | MA20_FIELDS | {"close", "priorClose"}
OPERATORS = {"lt": operator.lt, "lte": operator.le, "gt": operator.gt,
             "gte": operator.ge, "eq": operator.eq, "contains": operator.contains}
ADX_STATES = ("Sharp Deterioration", "Deteriorating", "Stable", "Improving", "Accelerating")


def filter_matcher(field, rule):
    if field not in FIELDS or not isinstance(rule, dict):
        raise ValueError("Choose a supported ADX Summary filter.")
    if field in CATEGORIES:
        values = rule.get("values")
        if not isinstance(values, list) or not all(value is None or isinstance(value, str) for value in values):
            raise ValueError("Choose valid checkbox filter values.")
        return lambda actual: np.isin(actual, values)

    conditions = rule.get("conditions")
    join = rule.get("join")
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 2 or join not in ("and", "or"):
        raise ValueError("Choose valid column filter conditions.")
    comparisons = []
    for term in conditions:
        if not isinstance(term, dict) or not isinstance(term.get("op"), str) or term["op"] not in OPERATORS:
            raise ValueError("Choose a valid filter operator.")
        value = term.get("value")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Enter a value for each column filter.")
        operation = term["op"]
        if operation == "contains" and field != "ticker":
            raise ValueError("Contains is available for ticker filters.")
        target = value.upper() if field in TEXT else float(value)
        if field not in TEXT and not np.isfinite(target):
            raise ValueError("Enter a finite number for each numeric filter.")
        comparisons.append((OPERATORS[operation], target))

    if field in TEXT:
        combine = all if join == "and" else any
        return lambda actual: np.fromiter(
            (bool(value) and combine(compare(value.upper(), target) for compare, target in comparisons)
             for value in actual), bool, count=len(actual))

    combine = np.logical_and if join == "and" else np.logical_or

    def numeric_match(actual):
        values = actual * 100 if field in FRACTIONS else actual
        return np.isfinite(values) & combine.reduce([compare(values, target) for compare, target in comparisons])

    return numeric_match


def compile_filters(settings):
    if not isinstance(settings, dict) or not isinstance(settings.get("columns"), list):
        raise ValueError("Send the current ADX Summary filters.")
    ticker, state, ma = (settings.get(key, "") for key in ("ticker", "adxState", "maStatus"))
    exact = settings.get("exactTicker", False)
    if not all(isinstance(value, str) for value in (ticker, state, ma)) or not isinstance(exact, bool):
        raise ValueError("Choose valid ADX Summary toolbar filters.")
    rules = []
    for entry in settings["columns"]:
        if not isinstance(entry, list) or len(entry) != 2 or not isinstance(entry[0], str):
            raise ValueError("Choose valid ADX Summary columns.")
        rules.append(entry)
    if ticker.strip():
        rules.append(("ticker", {"join": "and", "conditions": [
            {"op": "eq" if exact else "contains", "value": ticker.strip()},
        ]}))
    if ma:
        rules.append(("maStatus", {"values": [ma]}))
    if state:
        field, value = ("marketRegime", None) if state == "missing" else ("adxState", state)
        rules.append((field, {"values": [value]}))
    return [(field, rule, filter_matcher(field, rule)) for field, rule in rules]


def shifted(values, periods):
    result = np.full(len(values), np.nan)
    result[periods:] = values[:-periods]
    return result


def running_extreme(values, dates, high):
    start, accumulate, better = (-np.inf, np.maximum, np.greater) if high else (np.inf, np.minimum, np.less)
    clean = np.where(np.isfinite(values), values, start)
    extreme = accumulate.accumulate(clean)
    is_new = better(clean, np.r_[start, extreme[:-1]])
    positions = np.maximum.accumulate(np.where(is_new, np.arange(len(values)), -1))
    return np.where(positions >= 0, extreme, np.nan), np.where(positions >= 0, dates[positions], None)


def ma20_history(deviations, fields):
    result = {"ma20Gap": deviations}
    for field in {"ma20ZScore", "ma20GapPercentile"} & fields:
        result[field] = np.full(len(deviations), np.nan)
    finite = np.isfinite(deviations)
    for same_side in (deviations < 0, deviations > 0):
        indices = np.flatnonzero(same_side & finite)
        if len(indices) < 2:
            continue
        gaps = pd.Series(deviations[indices])
        expanding = gaps.expanding(min_periods=2)
        if "ma20ZScore" in result:
            prior_std = expanding.std(ddof=0).shift().where(lambda values: values > 0)
            result["ma20ZScore"][indices] = ((gaps - expanding.mean().shift()) / prior_std).to_numpy()
        if "ma20GapPercentile" in result:
            ranks = expanding.rank(method="average").to_numpy()
            # The rank includes the current gap. Subtract its one-based rank contribution.
            result["ma20GapPercentile"][indices[1:]] = (ranks[1:] - 1) / np.arange(1, len(gaps))
    return result


def indicator_history(days, prices, fields):
    high, low, close = prices.T
    result = {"close": close}
    closes = pd.Series(close)
    if "priorClose" in fields:
        result["priorClose"] = shifted(close, 5)
    if fields & MA20_FIELDS:
        _, gap = moving_average_deviations(closes, 20)
        result.update(ma20_history(gap, fields))
    if fields & ADX_FIELDS:
        adx = worksheet_adx(high, low)
        prior_adx = shifted(adx, 5)
        change = adx - prior_adx
        result.update(adx=adx, priorAdx=prior_adx, adxChange=change)
        if fields & {"adxState", "marketRegime"}:
            result["adxState"] = np.array([adx_state(value) if np.isfinite(value) else None for value in change], dtype=object)
        if fields & {"highAdx", "highDate"}:
            result["highAdx"], result["highDate"] = running_extreme(adx, days, True)
        if fields & {"lowAdx", "lowDate"}:
            result["lowAdx"], result["lowDate"] = running_extreme(adx, days, False)
    if fields & MA50_FIELDS:
        average, gap = moving_average_deviations(closes, 50)
        prior_gap = shifted(gap, 5)
        result.update(ma50=average, priorMa50=shifted(average, 5), pct50=gap, priorPct50=prior_gap, maPosition=gap,
                      pct50Change=gap - prior_gap, gapChange=(np.abs(gap) - np.abs(prior_gap)) * 100)
        if fields & {"maStatus", "marketRegime"}:
            result["maStatus"] = np.array([
                moving_average_status(prior, value) if np.isfinite(prior) and np.isfinite(value) else None
                for prior, value in zip(prior_gap, gap)], dtype=object)
        if "minPct50" in fields:
            result["minPct50"], _ = running_extreme(gap, days, False)
        if "maxPct50" in fields:
            result["maxPct50"], _ = running_extreme(gap, days, True)
    if "marketRegime" in fields:
        result["marketRegime"] = np.array([
            market_regime(state, ma) if state is not None and ma is not None else None
            for state, ma in zip(result["adxState"], result["maStatus"])], dtype=object)
    return result


def forward_matches(days, close, sessions, matched):
    starts = np.flatnonzero(matched)
    targets = np.searchsorted(sessions, days[starts]) + HORIZON
    ready = targets < len(sessions)
    starts = starts[ready]
    exit_days = sessions[targets[ready]]
    exits = np.searchsorted(days, exit_days).clip(0, len(days) - 1)
    priced = (days[exits] == exit_days) & (close[starts] > 0) & (close[exits] > 0)
    returns = close[exits[priced]] / close[starts[priced]] - 1
    return starts[priced], exit_days[priced], returns, int(np.count_nonzero(~ready)), int(np.count_nonzero(~priced))


def return_stats(returns):
    return {"observations": len(returns), "mean": float(returns.mean()) if len(returns) else None}


def condition_scenarios(rules):
    scenarios = [[rule] for rule in rules]
    adx_rules = [rule for rule in rules if rule[0] == "adx"]
    if adx_rules:
        for state in ADX_STATES:
            condition = {"values": [state]}
            scenarios.append([*adx_rules, ("adxState", condition, filter_matcher("adxState", condition))])
    return scenarios


def condition_returns(days, metrics, sessions, scenarios):
    starts, _, returns, _, _ = forward_matches(
        days, metrics["close"], sessions, np.ones(len(days), dtype=bool))
    breakdown = []
    for scenario in scenarios:
        matched = np.ones(len(starts), dtype=bool)
        for field, _, matches in scenario:
            matched &= matches(metrics[field][starts])
        breakdown.append({
            "conditions": [[field, rule] for field, rule, _ in scenario],
            **return_stats(returns[matched]),
        })
    return breakdown


def replay(connection, through, rules, membership, tickers, *, breakdown=False):
    if through not in complete_market_dates(connection):
        raise ValueError("Choose a saved market date for the return analysis.")
    sessions = np.array([row[0] for row in connection.execute(
        "SELECT DISTINCT date FROM daily_prices WHERE date <= ? ORDER BY date", (through,))])
    static = {"ticker", "sp500"}
    dynamic_rules = [rule for rule in rules if rule[0] not in static]
    static_rules = [rule for rule in rules if rule[0] in static]
    scenarios = condition_scenarios(dynamic_rules) if breakdown else []
    fields = {field for field, _, _ in dynamic_rules}
    fields.update(field for scenario in scenarios for field, _, _ in scenario)
    samples, details = [], []
    by_condition = []
    ticker_returns = {ticker: {"observations": 0, "mean": None} for ticker in tickers}
    pending = missing = 0
    for ticker in ticker_returns:
        member = "Yes" if re.sub(r"[ /-]", ".", ticker.upper()) in membership else "No"
        values = {"ticker": ticker, "sp500": member}
        if not all(matches(np.asarray([values[field]]))[0] for field, _, matches in static_rules):
            continue
        bars = connection.execute(
            "SELECT date, high, low, close FROM daily_prices WHERE ticker = ? AND date <= ? ORDER BY date",
            (ticker, through)).fetchall()
        if not bars:
            continue
        days = np.array([bar[0] for bar in bars])
        prices = np.array([tuple(bar)[1:] for bar in bars], dtype=float)
        metrics = indicator_history(days, prices, fields)
        if scenarios:
            by_condition = condition_returns(days, metrics, sessions, scenarios)
        matched = np.ones(len(bars), dtype=bool)
        for field, _, matches in dynamic_rules:
            matched &= matches(metrics[field])
        starts, exits, returns, awaiting, unpriced = forward_matches(days, metrics["close"], sessions, matched)
        pending += awaiting
        missing += unpriced
        if len(returns):
            samples.append(returns)
            ticker_returns[ticker] = return_stats(returns)
            for start, end, value in zip(starts[-DETAIL_LIMIT:], exits[-DETAIL_LIMIT:], returns[-DETAIL_LIMIT:]):
                item = (str(days[start]), ticker, str(end), float(value))
                if len(details) < DETAIL_LIMIT:
                    heapq.heappush(details, item)
                elif item > details[0]:
                    heapq.heapreplace(details, item)
    returns = np.concatenate(samples) if samples else np.array([])
    return {
        "status": "complete", "firstDate": str(sessions[0]), "through": through,
        "observations": len(returns), "tickers": len(samples), "pending": pending, "missing": missing,
        "mean": float(returns.mean()) if len(returns) else None,
        "median": float(np.median(returns)) if len(returns) else None,
        "positiveRate": float(np.count_nonzero(returns > 0) / len(returns)) if len(returns) else None,
        "tickerReturns": ticker_returns,
        **({"byCondition": by_condition} if breakdown else {}),
        "recent": [{"date": day, "ticker": ticker, "exitDate": end, "return": value}
                   for day, ticker, end, value in sorted(details, reverse=True)],
    }


def main(request):
    if not isinstance(request, dict) or not isinstance(request.get("through"), str):
        raise ValueError("Choose a saved market date.")
    through = date.fromisoformat(request["through"]).isoformat()
    tickers = request.get("tickers")
    if not isinstance(tickers, list) or not tickers or not all(isinstance(ticker, str) and ticker for ticker in tickers):
        raise ValueError("Choose at least one ticker from the filtered table.")
    settings = request.get("filters")
    rules = compile_filters(settings)
    breakdown = settings.get("breakdown") is True
    if breakdown and len(tickers) != 1:
        raise ValueError("Choose one filtered stock for the condition breakdown.")
    membership, as_of = set(), None
    if any(field == "sp500" for field, _, _ in rules):
        snapshot = json.loads(MEMBERSHIP.read_text(encoding="utf-8"))
        membership = set(snapshot["tickers"])
        as_of = snapshot["asOf"]
    if not DATABASE.is_file():
        raise ValueError("Import SQLite price history first.")
    with open_history(DATABASE) as connection:
        connection.execute("PRAGMA cache_size = -65536")
        connection.execute("BEGIN")
        return {**replay(connection, through, rules, membership, tickers, breakdown=breakdown), "membershipAsOf": as_of}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        print(json.dumps(main(json.load(sys.stdin)), allow_nan=False, separators=(",", ":")))
    except (ValueError, OSError, sqlite3.Error) as error:
        logging.error("%s", error)
        print(json.dumps({"status": "failed", "message": str(error)}))
