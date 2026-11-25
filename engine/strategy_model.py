"""Calculate daily market indicators and replay the specified strategies at the close."""

from array import array
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from math import ceil, floor, isfinite, log, nan, sqrt

STRATEGIES = ("breakout", "pullback", "crossover")


class Smooth:
    def __init__(self, period, alpha=None):
        self.period = period
        self.alpha = 1 / period if alpha is None else alpha
        self.count = 0
        self.value = 0.0

    def add(self, sample):
        if self.count < self.period:
            self.count += 1
            self.value += sample / self.period
            if self.count < self.period:
                return None
        else:
            self.value += self.alpha * (sample - self.value)
        return self.value


class History:
    def __init__(self):
        self.days = array("I")
        self.opens, self.highs, self.lows, self.closes, self.volumes, self.atrs = (
            array("d") for _ in range(6)
        )
        self.atr = Smooth(14)

    def add(self, day, bar):
        high, low, close = bar["high"], bar["low"], bar["close"]
        atr = None
        if self.closes:
            previous = self.closes[-1]
            atr = self.atr.add(max(high - low, abs(high - previous), abs(low - previous)))
        self.days.append(day)
        self.opens.append(bar["open"])
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.volumes.append(bar["volume"])
        self.atrs.append(nan if atr is None else atr)
        return atr / close if atr is not None and close > 0 else None


def percentile(values, fraction):
    if not values:
        return None
    values.sort()
    rank = (len(values) - 1) * fraction
    lower, upper = floor(rank), ceil(rank)
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def half_life(deviations, window):
    if len(deviations) < window:
        return None
    values = deviations[-window:]
    lag = values[:-1]
    mean_lag = sum(lag) / len(lag)
    mean_change = (values[-1] - values[0]) / len(lag)
    centered = [sample - mean_lag for sample in lag]
    variance = sum(sample * sample for sample in centered)
    if variance <= 1e-18 * max(1, sum(sample * sample for sample in lag)):
        return None
    slope = sum(x * (after - before - mean_change)
                for x, before, after in zip(centered, lag, values[1:])) / variance
    return -log(2) / slope if -2 < slope < 0 else None


def rsi(gain, loss):
    if gain is None or loss is None:
        return None
    return 100 * (gain / (gain + loss)) if gain + loss else 50.0


@dataclass
class Position:
    direction: int
    entry: float
    entry_day: int
    stop: float
    initial_risk: float
    best: float
    expiry: int | None
    reached_2r: bool = False

    def display(self, state, day, dates, price, reason=None):
        return {
            "state": state, "side": "long" if self.direction == 1 else "short",
            "entryDate": dates[self.entry_day], "entryPrice": self.entry, "stop": self.stop,
            "initialRisk": self.initial_risk, "age": day - self.entry_day,
            "r": (price - self.entry) * self.direction / self.initial_risk, "reason": reason,
            "exitPrice": price if state == "exit" else None, "timeStopSessions": self.expiry,
        }


def update_position(position, strategy, day, prices, signals, options):
    opening, high, low, close, atr = prices
    baseline_cross, sma, crossover = signals
    side = position.direction
    if (low <= position.stop if side == 1 else high >= position.stop):
        fill = min(opening, position.stop) if side == 1 else max(opening, position.stop)
        return "Protective stop", fill

    progress = (close - position.entry) * side
    position.reached_2r |= progress >= 2 * position.initial_risk
    age = day - position.entry_day
    if strategy == "breakout":
        if options["trailing"] == "ma" and baseline_cross == -side:
            return "20-day average cross", close
        if age >= position.expiry and not position.reached_2r:
            return "Time stop: 2R was not reached", close
        if options["trailing"] == "atr":
            position.best = max(position.best, close) if side == 1 else min(position.best, close)
            candidate = position.best - side * 2 * atr
            position.stop = max(position.stop, candidate) if side == 1 else min(position.stop, candidate)
    elif strategy == "pullback":
        if close >= sma:
            return "SMA 20 profit target", close
        if age >= position.expiry:
            return "Half-life time stop", close
    elif crossover == -side:
        return "Fast EMA cross", close
    return None, close


def crossing(current, previous):
    if current is None or previous is None:
        return 0
    if current > 0 and previous <= 0:
        return 1
    if current < 0 and previous >= 0:
        return -1
    return 0


def flat_base_signal(ratio, contracting):
    if ratio is not None:
        if ratio < 0.4 and contracting:
            return "volatility_compression_squeeze"
        if ratio > 2.0:
            return "trend_emergence_decay"
    return None


def analyze(ticker, history, dates, cutoffs, options):
    fast_window = options["fastEma"]
    ema20, fast = Smooth(20, 2 / 21), Smooth(fast_window, 2 / (fast_window + 1))
    positive, negative = 0.0, 0.0
    adx = Smooth(14)
    gain14, loss14, gain2, loss2 = Smooth(14), Smooth(14), Smooth(2), Smooth(2)
    sma_sums = dict.fromkeys((20, 50, 200), 0.0)
    squares, volume_sum = 0.0, 0.0
    deviations = []
    positions = {}
    previous_baseline_gap = previous_fast_gap = previous_bandwidth = None
    highs, lows, closes, volumes = history.highs, history.lows, history.closes, history.volumes
    last_index = len(closes) - 1

    for index, close in enumerate(closes):
        latest = index == last_index
        day = history.days[index]
        high, low, volume = highs[index], lows[index], volumes[index]
        for period in sma_sums:
            sma_sums[period] += close - (closes[index - period] if index >= period else 0)
        squares += close * close - (closes[index - 20] ** 2 if index >= 20 else 0)
        volume_sum += volume - (volumes[index - 20] if index >= 20 else 0)

        sma = sma_sums[20] / 20 if index >= 19 else None
        sma50 = sma_sums[50] / 50 if index >= 49 else None
        sma200 = sma_sums[200] / 200 if index >= 199 else None
        exponential, fast_average = ema20.add(close), fast.add(close)
        baseline = exponential if options["baseline"] == "ema" else sma
        atr = history.atrs[index]
        atr = atr if isfinite(atr) else None
        natr = atr / close if atr is not None and close > 0 else None

        strength = rsi14 = rsi2 = None
        if index > 0:
            change = close - closes[index - 1]
            gain, loss = max(change, 0), max(-change, 0)
            rsi14 = rsi(gain14.add(gain), loss14.add(loss))
            rsi2 = rsi(gain2.add(gain), loss2.add(loss))
            up, down = high - highs[index - 1], lows[index - 1] - low
            decay = 13 / 14 if index >= 14 else 1
            positive = decay * positive + (up if up > max(0, down) else 0)
            negative = decay * negative + (down if down > max(0, up) else 0)
            if index >= 14:
                total = positive + negative
                strength = adx.add(100 * (abs(positive - negative) / total) if total else 0)

        standard_deviation = z = bandwidth = relative_volume = None
        if sma is not None:
            deviation = close - sma
            deviations.append(deviation)
            standard_deviation = sqrt(max(0, squares / 20 - sma * sma))
            z = deviation / standard_deviation if standard_deviation else None
            bandwidth = 4 * standard_deviation / sma if sma > 0 else None
            relative_volume = 20 * volume / volume_sum if volume_sum > 0 else None

        cutoff = cutoffs[day]
        flat_base = strength is not None and strength < 20 and natr is not None and cutoff is not None and natr <= cutoff
        pullback = (
            sma200 is not None and close > sma200 and (z is not None and z < -2 or close < sma * 0.97)
            and (rsi14 < 35 or rsi2 < 10) and relative_volume is not None and relative_volume < 1.2
        )
        tau60 = half_life(deviations, 60) if latest or flat_base or pullback else None
        tau15 = half_life(deviations, 15) if latest or flat_base else None
        ratio = tau15 / tau60 if tau15 is not None and tau60 is not None else None
        contracting = bandwidth is not None and previous_bandwidth is not None and bandwidth < previous_bandwidth
        regime_shift = flat_base_signal(ratio, contracting) if flat_base else None

        baseline_gap = close - baseline if baseline is not None else None
        fast_gap = fast_average - sma if fast_average is not None and sma is not None else None
        baseline_cross = crossing(baseline_gap, previous_baseline_gap)
        crossover = crossing(fast_gap, previous_fast_gap)
        ready = sma200 is not None and atr is not None
        entries = dict.fromkeys(STRATEGIES, 0)
        if ready:
            if relative_volume is not None and relative_volume > 1.5:
                if close > max(sma50, sma200) and baseline_gap > 0 and (
                    previous_baseline_gap > 0 or baseline_gap > atr
                ):
                    entries["breakout"] = 1
                elif close < min(sma50, sma200) and baseline_gap < 0 and (
                    previous_baseline_gap < 0 or baseline_gap < -atr
                ):
                    entries["breakout"] = -1
            if pullback and tau60 is not None and tau60 < 10 and regime_shift != "trend_emergence_decay":
                entries["pullback"] = 1
            if (close > sma200 and crossover == 1) or (close < sma200 and crossover == -1):
                entries["crossover"] = crossover

        states = {}
        for strategy, side in entries.items():
            position = positions.get(strategy)
            state = {"state": "none" if ready else "insufficient"}
            if position:
                prices = (history.opens[index], high, low, close, atr)
                reason, price = update_position(position, strategy, day, prices,
                                                (baseline_cross, sma, crossover), options)
                state = position.display("exit" if reason else "open", day, dates, price, reason)
                if reason:
                    del positions[strategy]
            elif side and atr > 0:
                expiry = {"breakout": options["trendDays"], "crossover": None}.get(strategy)
                if strategy == "pullback":
                    expiry = ceil(1.5 * tau60)
                multiplier = 1.5 if strategy == "pullback" else 2
                stop = close - side * multiplier * atr
                if strategy == "breakout" and options["breakoutStop"] == "swing":
                    lookback = max(0, index - 4)
                    stop = (min(lows[lookback:index + 1]) if side == 1
                            else max(highs[lookback:index + 1]))
                risk = (close - stop) * side
                if risk > 0:
                    position = Position(side, close, day, stop, risk, close, expiry)
                    positions[strategy] = position
                    state = position.display("entry", day, dates, close)
            if latest:
                states[strategy] = state

        if latest:
            regime = ("Range" if strength < 20 else "Trending" if strength > 25 else "Transition"
                      ) if strength is not None else None
            return {
                "ticker": ticker, "close": close, "bars": len(closes),
                "sma20": sma, "ema20": exponential, "sma50": sma50, "sma200": sma200, "fastEma": fast_average,
                "std20": standard_deviation, "zScore": z, "bandwidth": bandwidth, "bandwidthContracting": contracting,
                "upperBand": sma + 2 * standard_deviation if sma is not None else None,
                "lowerBand": sma - 2 * standard_deviation if sma is not None else None,
                "atr14": atr, "natr": natr, "natrCutoff": cutoff, "adx14": strength, "regime": regime,
                "rsi14": rsi14, "rsi2": rsi2, "rvol": relative_volume,
                "halfLife15": tau15, "halfLife60": tau60, "halfLifeRatio": ratio, "flatBase": flat_base,
                "regimeShift": regime_shift, **states,
            }
        previous_baseline_gap, previous_fast_gap, previous_bandwidth = baseline_gap, fast_gap, bandwidth


def market_breadth(rows, histories, dates, complete_dates):
    averages = {}
    for average in ("sma20", "sma50", "sma200"):
        eligible = [row for row in rows if row[average] is not None]
        above = sum(row["close"] > row[average] for row in eligible)
        averages[average] = {"above": above, "eligible": len(eligible)}

    current_is_complete = complete_dates is None or dates[-1] in complete_dates
    previous = next(
        (idx for idx in range(len(dates) - 2, -1, -1) if complete_dates is None or dates[idx] in complete_dates),
        None,
    ) if current_is_complete else None
    movement = {"previousSession": None, "advancing": 0, "declining": 0, "unchanged": 0}
    if previous is not None:
        movement["previousSession"] = dates[previous]
        for row in rows:
            history = histories[row["ticker"]]
            idx = bisect_left(history.days, previous)
            if idx == len(history.days) or history.days[idx] != previous:
                continue
            change = row["close"] - history.closes[idx]
            category = "advancing" if change > 0 else "declining" if change < 0 else "unchanged"
            movement[category] += 1
    return {"averages": averages, **movement}


def scan(session_bars, options, logger, complete_dates=None):
    dates, cutoffs = [], []
    histories = {}
    latest_tickers = []
    for date, bars in session_bars:
        if not bars:
            continue
        day = len(dates)
        dates.append(date)
        natr_values = []
        latest_tickers = []
        for bar in bars:
            ticker = bar["ticker"]
            history = histories.get(ticker)
            if history is None:
                history = histories[ticker] = History()
            natr = history.add(day, bar)
            if natr is not None:
                natr_values.append(natr)
            latest_tickers.append(ticker)
        is_complete = complete_dates is None or date in complete_dates
        cutoffs.append(percentile(natr_values, 0.2) if is_complete else None)

    if not dates or dates[-1] != options["date"]:
        raise ValueError(f"No market prices found for {options['date']}. Import that date before running the model.")
    logger.info("Model: calculating %s symbols across %s sessions", len(latest_tickers), len(dates))
    rows = [analyze(ticker, histories[ticker], dates, cutoffs, options) for ticker in latest_tickers]
    summary = Counter()
    for row in rows:
        summary["macroReady"] += row["sma200"] is not None
        summary["entries"] += sum(row[strategy]["state"] == "entry" for strategy in STRATEGIES)
        summary["exits"] += sum(row[strategy]["state"] == "exit" for strategy in STRATEGIES)
        summary["regimeShifts"] += row["regimeShift"] is not None
    return {
        "status": "complete", "date": dates[-1], "firstDate": dates[0], "sessionCount": len(dates),
        "symbolCount": len(rows), "options": options, "summary": dict(summary), "rows": rows,
        "breadth": market_breadth(rows, histories, dates, complete_dates),
    }
