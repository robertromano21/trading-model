"""Compare broad-dollar growth with asset turning points and U.S. trade volumes."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from math import isfinite

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

import crypto_intraday as intraday
import fred_macro as fred
from crypto_data import FRED_CRYPTO, MACRO_RAW, read_csv, write_json
from fetch_and_model import DATA_DIR, write_atomically
from macro_data import MacroError, download_fred
from macro_nowcast import _fit_ridge, _predict
from market_data import load_panel
from schwab_data import SchwabMarketData

RESEARCH = DATA_DIR / "Research"
REPORT = RESEARCH / "dollar_study.json"
MICRO = RESEARCH / "dollar_intraday_study.json"
TRADE = RESEARCH / "dollar_real_trade.csv"
ASSETS = {"SPY": "S&P 500", "QQQ": "Nasdaq-100", "IWM": "Russell 2000",
          "GLD": "Gold", "USO": "Oil", "DBC": "Broad commodities"}
LABELS = {"BTC": "Bitcoin", "WTI": "WTI spot crude", **ASSETS}
START = "2015-01-01"
DAY = pd.Timedelta(days=1)
TRADE_SERIES = {"IMPGSC1": "Real U.S. imports", "EXPGSC1": "Real U.S. exports"}


def number(value):
    return round(float(value), 6) if isfinite(value) else None


def date(stamp):
    return stamp.date().isoformat()


def year_growth(series):
    back = series.reindex(series.index - pd.DateOffset(years=1), method="ffill",
                          tolerance=7 * DAY).to_numpy()
    return (series / back - 1).dropna()


def turns(series, half_window):
    window = series.rolling(2 * half_window + 1, center=True)
    change = window.max() - window.min()
    return {
        "peak": series.index[(change > 0) & series.eq(window.max())].drop_duplicates(),
        "valley": series.index[(change > 0) & series.eq(window.min())].drop_duplicates(),
    }


def summary(values):
    data = pd.Series(values, dtype=float).dropna()
    return {"n": len(data), "median": number(data.median()), "up": int(data.gt(0).sum())}


def release(day):
    monday = day + (7 - day.weekday()) * DAY
    return CustomBusinessDay(calendar=USFederalHolidayCalendar()).rollforward(monday)


def asset_study(series, dollar, growth, daily=False):
    series = series.dropna()
    sample = series.loc[START:growth.index[-1]]
    if sample.empty:
        return None
    aligned = series.reindex(growth.index, method="ffill", tolerance=4 * DAY).dropna()
    same_month = aligned.pct_change(21, fill_method=None)
    lagged_month = dollar.pct_change(21, fill_method=None)
    same_year = year_growth(series).reindex(growth.index, method="ffill", tolerance=4 * DAY)
    percent_at_turn = {}
    for kind, days in turns(series, 30 if daily else 21).items():
        days = days[(days >= sample.index[0]) & (days <= sample.index[-1])]
        matched = growth.reindex(days, method="ffill", tolerance=4 * DAY)
        recent = matched.loc[growth.index[-1] - pd.DateOffset(years=1):]
        extreme_day = sample.idxmax() if kind == "peak" else sample.idxmin()
        percent_at_turn[kind] = {"all": summary(matched), "recent": summary(recent), "extreme": {
            "date": date(extreme_day), "price": number(sample.loc[extreme_day]),
            "dollarYoY": number(growth.asof(extreme_day))}}
    after_turn = {}
    for kind, days in turns(growth, 10).items():
        values = []
        for day in days[days >= pd.Timestamp(START)]:
            known = release(growth.index[growth.index.get_loc(day) + 10])
            i = series.index.searchsorted(known, side="right")
            if i >= len(series):
                continue
            j = series.index.searchsorted(series.index[i] + 14 * DAY)
            if j < len(series):
                values.append(series.iloc[j] / series.iloc[i] - 1)
        after_turn[kind] = summary(values)
    return {"start": date(sample.index[0]), "end": date(sample.index[-1]),
            "monthCorrelation": number(lagged_month.loc[START:].corr(same_month)),
            "recentMonthCorrelation": number(lagged_month.tail(252).corr(same_month)),
            "yearCorrelation": number(growth.loc[START:].corr(same_year)),
            "atPriceTurns": percent_at_turn, "afterDollarTurns": after_turn}


def trade_study(dollar, data):
    if data is None or data.empty:
        return []
    quarterly = dollar.resample("QS").mean().iloc[:-1]
    growth = quarterly.pct_change(4, fill_method=None)
    rows = []
    for symbol, label in TRADE_SERIES.items():
        actual = data[symbol].dropna().pct_change(4, fill_method=None).dropna()
        pair = pd.concat([growth.loc[START:].rename("usd"), actual.rename("volume")], axis=1).dropna()
        if pair.empty:
            continue
        lead = pd.concat([growth.loc[START:], actual.shift(-1)], axis=1).dropna()
        rows.append({"symbol": symbol, "label": label, "quarter": date(pair.index[-1]), "n": len(pair),
                     "dollarYoY": number(pair.usd.iloc[-1]), "volumeYoY": number(pair.volume.iloc[-1]),
                     "correlation": number(pair.usd.corr(pair.volume)),
                     "nextQuarterCorrelation": number(lead.iloc[:, 0].corr(lead.iloc[:, 1]))})
    return rows


def trade_dollar_outlook(dollar, data):
    if data is None or data.empty:
        return None
    frames = {}
    for symbol in TRADE_SERIES:
        obs = fred.observations(f"{symbol}:pca")
        if obs is None or obs.empty:
            values = ((data[symbol].dropna().pct_change(fill_method=None) + 1) ** 4 - 1).dropna() * 100
            obs = pd.DataFrame({"value": values, "available": fred.availability(values.index, symbol)})
        frames[symbol] = obs.sort_values("available")

    signals = pd.DataFrame({"quarter": dollar.pct_change(63, fill_method=None),
                            "year": year_growth(dollar)})
    for symbol, obs in frames.items():
        values = obs.set_index("available")["value"]
        values = values[~values.index.duplicated(keep="last")].sort_index()
        signals[symbol] = values.reindex(signals.index, method="ffill") / 100
    horizon = 63
    signals["return"] = dollar.shift(-horizon) / dollar - 1
    signals["known"] = pd.Series(dollar.index, index=dollar.index).shift(-horizon)
    # Sparse quarterly forecast origins limit overlap in the out-of-sample errors.
    quarterly = signals.groupby(signals.index.to_period("Q")).tail(1).iloc[:-1].dropna()
    base_columns = ["quarter", "year"]
    augmented = base_columns + list(TRADE_SERIES)
    errors = {"flat": [], "dollar": [], "trade": []}
    outcomes, estimates = [], None
    for stamp, row in pd.concat([quarterly, signals.tail(1)]).iterrows():
        if row[augmented].isna().any():
            continue
        train = quarterly[quarterly["known"] <= stamp]
        if len(train) < 24:
            continue
        estimates = [number(_predict(_fit_ridge(train[cols].to_numpy(), train["return"].to_numpy()),
                                     row[cols].to_numpy(dtype=float))[0]) for cols in (base_columns, augmented)]
        if pd.isna(row["return"]):
            break
        actual = float(row["return"])
        outcomes.append(actual)
        for model, forecast in zip(errors, (0.0, *estimates)):
            errors[model].append(forecast - actual)
    if not outcomes or estimates is None:
        return None
    rmse = {model: number(np.sqrt(np.mean(np.square(err)))) for model, err in errors.items()}
    actual = np.array(outcomes)
    hit = {model: number(np.mean(np.sign(actual + errors[model]) == np.sign(actual))) for model in ("dollar", "trade")}
    return {"asOf": date(signals.index[-1]), "horizonSessions": horizon, "n": len(outcomes),
            "rmse": rmse, "directionHitRate": hit,
            "baselineMove": estimates[0], "tradeMove": estimates[1],
            "beatsBaselines": rmse["trade"] < min(rmse["dollar"], rmse["flat"]),
            "vintages": all(fred.load_vintage(s, units="pca") is not None for s in TRADE_SERIES)}


def history(symbol, first, last):
    step = intraday.INTERVAL
    result = {}
    for start in range(first, last + 1, 280 * step):
        result.update(intraday.candle_range(symbol, start, min(last, start + 279 * step)))
    return result


def dollar_history(first, last):
    begin = datetime.fromtimestamp(first, timezone.utc) - timedelta(minutes=20)
    end = datetime.fromtimestamp(last + intraday.INTERVAL, timezone.utc)
    client = SchwabMarketData()
    result = {}
    while begin < end:
        stop = min(begin + timedelta(days=8), end)
        result.update(intraday.dollar_range(client, begin, stop))
        begin = stop
    return result


def micro_events(bars, moves):
    step = intraday.INTERVAL
    btc_close = pd.Series({stamp: bar[1] for stamp, bar in bars.items()}).reindex(moves.index)
    price_turns = turns(btc_close, 4)
    observations = {}
    for kind, dates in turns(moves, 4).items():
        reverse = "valley" if kind == "peak" else "peak"
        inverse_matches = sum(any(abs(t - d) <= 2 * step for t in price_turns[reverse]) for d in dates)
        same_matches = sum(any(abs(t - d) <= 2 * step for t in price_turns[kind]) for d in dates)
        raw, net = [], []
        for stamp in dates:
            entry = stamp + 5 * step  # after four more completed signal bars
            exit = entry + 3 * step
            if not all(t in bars for t in range(entry, exit + 1, step)):
                continue
            ratio = bars[exit][1] / bars[entry][0]
            raw.append(ratio - 1)
            net.append(ratio * (1 - intraday.SLIP) * (1 - intraday.FEE) /
                       ((1 + intraday.SLIP) * (1 + intraday.FEE)) - 1)
        observations[kind] = {"events": len(dates), "inverseMatches": inverse_matches, "sameMatches": same_matches,
                              "nextHour": summary(raw), "nextHourNetLong": summary(net)}
    return observations


def micro_study(days=40):
    step = intraday.INTERVAL
    last = int((time.time() - 60) // step) * step - step
    first = last - days * 86400 + step
    with ThreadPoolExecutor(max_workers=3) as pool:
        coins = {s: pool.submit(history, s, first, last) for s in ("BTC", "ETH")}
        dollar = pool.submit(dollar_history, first, last)
        dollar_bars = dollar.result()
        bars = {s: f.result() for s, f in coins.items()}
    moves = pd.Series({t: intraday.dollar_change(dollar_bars, t + step)
                       for t in range(first, last + 1, step)}, dtype=float)
    replay = {}
    for account, (coin, strategy) in intraday.ACCOUNTS.items():
        decisions, _ = intraday.advance(intraday.new_account(), bars[coin], last, strategy, dollar_bars, replay=True)
        replay[account] = dict(decisions)
    return {"start": intraday.iso(first), "end": intraday.iso(last + step), "uupMinutes": len(dollar_bars),
            "overlap": int(moves.notna().sum()), "events": {s: micro_events(b, moves) for s, b in bars.items()},
            "leaderboard": intraday.leaderboard(replay)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro", action="store_true", help="Refresh the 40-day Coinbase and Schwab intraday study.")
    parser.add_argument("--trade", action="store_true", help="Refresh BEA real imports and exports from FRED.")
    parser.add_argument("--refresh-trade", action="store_true", help="Refresh the trade cache if older than a week.")
    args = parser.parse_args()
    if args.micro:
        write_json(MICRO, micro_study())
    stale = not TRADE.exists() or time.time() - TRADE.stat().st_mtime > 7 * 86400
    trade_error = None
    if args.trade or (args.refresh_trade and stale):
        try:
            trade = download_fred(list(TRADE_SERIES), start="2013-01-01", timeout=15)
            if not trade.empty:
                TRADE.parent.mkdir(parents=True, exist_ok=True)
                write_atomically(TRADE, trade.rename_axis("date").to_csv().encode())
        except MacroError as error:
            trade_error = str(error)
    macro = read_csv(MACRO_RAW)
    dollar_candidates = [fred.load_series("DTWEXBGS"), macro["DTWEXBGS"].dropna() if macro is not None else None]
    dollar = max((s for s in dollar_candidates if s is not None and not s.empty), key=lambda s: s.index[-1])
    btc = read_csv(FRED_CRYPTO)
    growth = year_growth(dollar)
    move = {name: number(dollar.iloc[-1] / dollar.iloc[-1 - n] - 1)
            for name, n in (("week", 5), ("month", 21), ("quarter", 63))}
    panel = load_panel(start="2014-01-01", tickers=list(ASSETS), min_avg_dollar_volume=0, min_avg_price=0)
    prices = {"BTC": btc["CBBTCUSD"]} if btc is not None else {}
    prices.update({s: panel.close[s] for s in ASSETS if s in panel.close})
    oil = fred.load_series("DCOILWTICO")
    if oil is not None:
        prices["WTI"] = oil[oil > 0]
    assets = []
    for symbol, price in prices.items():
        result = asset_study(price, dollar, growth, daily=(symbol == "BTC"))
        if result:
            assets.append({"symbol": symbol, "label": LABELS[symbol], **result})
    trade_data = read_csv(TRADE)
    report = {"dollar": {"series": "DTWEXBGS", "asOf": date(dollar.index[-1]), "level": number(dollar.iloc[-1]),
                         **move, "year": number(growth.iloc[-1])},
              "assets": assets, "trade": trade_study(dollar, trade_data), "tradeError": trade_error,
              "tradeDollarOutlook": trade_dollar_outlook(dollar, trade_data)}
    if MICRO.exists():
        report["intraday"] = json.loads(MICRO.read_text(encoding="utf-8"))
    write_json(REPORT, report)
    print(f"Dollar study updated through {report['dollar']['asOf']} with {len(assets)} asset series.")


if __name__ == "__main__":
    main()
