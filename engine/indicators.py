"""Chart indicators for one ticker, built on the engine's own definitions so the UI matches the strategies.

Shared with the engine (imported, not re-implemented): ``risk.atr`` (Wilder ATR), ``risk.realized_vol``
(rolling std x sqrt(252)), the TSMOM signal definition (tsmom.TimeSeriesMomentum.defaults horizons) and the
mean-reversion z-score (meanrev: (close - SMA_L) / population std_L).
"""

import math

import numpy as np
import pandas as pd

from risk import TRADING_DAYS, atr, realized_vol
from strategies.meanrev import RegimeMeanReversion
from strategies.tsmom import TimeSeriesMomentum

MA_PERIODS = (20, 50, 100, 200)


def sma(series, period):
    return series.rolling(period, min_periods=period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def donchian(high, low, period=20):
    """Highest high / lowest low of the previous ``period`` bars (excludes the current bar, as a breakout level)."""
    upper = high.rolling(period, min_periods=period).max().shift()
    lower = low.rolling(period, min_periods=period).min().shift()
    return upper, lower, (upper + lower) / 2


def bollinger(close, period=20, k=2.0):
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return mid + k * std, mid, mid - k * std


def zscore(close, period):
    """Same definition as the mean-reversion overlay: population std over ``period`` bars."""
    mean = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    return (close - mean) / std.replace(0, np.nan)


def rsi(close, period=14):
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())


def tsmom_signal(close, horizons=None):
    """TSMOM signal per the strategy: mean over horizons of 0.5*[N-day return > 0] + 0.5*[close > N-day SMA]."""
    horizons = horizons or TimeSeriesMomentum.defaults["horizons"]
    parts = []
    for n in horizons:
        ret = close / close.shift(n) - 1
        above = close > sma(close, n)
        part = 0.5 * (ret > 0).astype(float) + 0.5 * above.astype(float)
        parts.append(part.where(ret.notna() & sma(close, n).notna()))
    return sum(parts) / len(parts)


def vwap(frame):
    """Session VWAP for intraday bars (frame needs session_date, high, low, close, volume)."""
    typical = (frame["high"] + frame["low"] + frame["close"]) / 3
    pv = (typical * frame["volume"]).groupby(frame["session_date"]).cumsum()
    vol = frame["volume"].groupby(frame["session_date"]).cumsum()
    return pv / vol.replace(0, np.nan)


def compute(frame, market_close=None, intraday=False, bars_per_session=1):
    """frame: DataFrame indexed by time with open/high/low/close/volume. Returns {name: Series}."""
    close, high, low = frame["close"], frame["high"], frame["low"]
    out = {}
    for n in MA_PERIODS:
        out[f"sma{n}"] = sma(close, n)
        out[f"ema{n}"] = ema(close, n)
    out["donchian20_upper"], out["donchian20_lower"], out["donchian20_mid"] = donchian(high, low, 20)
    out["donchian55_upper"], out["donchian55_lower"], _ = donchian(high, low, 55)
    out["bb20_upper"], out["bb20_mid"], out["bb20_lower"] = bollinger(close, 20, 2.0)
    out["z20"] = zscore(close, 20)
    mr_lookback = RegimeMeanReversion.defaults["lookback"]
    out[f"z{mr_lookback}_meanrev"] = zscore(close, mr_lookback)
    out["rsi14"] = rsi(close, 14)
    out["atr14"] = atr(high, low, close, 14)
    out["atr14_pct"] = out["atr14"] / close * 100
    log_ret = np.log(close / close.shift())
    scale = math.sqrt(bars_per_session)          # realized_vol annualizes per bar with sqrt(252); add bars/session
    out["rvol20"] = realized_vol(log_ret, 20, min_periods=15) * scale * 100
    out["rvol60"] = realized_vol(log_ret, 60) * scale * 100
    if market_close is not None:
        aligned = market_close.reindex(close.index).ffill()
        ratio = close / aligned
        first = ratio.dropna()
        if len(first):
            out["rs_spy"] = ratio / first.iloc[0] * 100
            out["rs_spy_sma50"] = sma(out["rs_spy"], 50)
    if intraday and "session_date" in frame:
        out["vwap"] = vwap(frame)
    if not intraday:
        out["tsmom_signal"] = tsmom_signal(close)
        try:
            import exhaustion as _exh
            raw = _exh.features_for_frame(frame)
            out["exhaustion"] = raw["score"]
            out["exhaustion_up"] = raw["score_up"]
            out["exhaustion_down"] = raw["score_down"]
            out["exhaustion_flag_up"] = raw["flag_up"]
            out["exhaustion_flag_down"] = raw["flag_down"]
            out["up_streak"] = raw["up_streak"].astype(float)
            out["down_streak"] = raw["down_streak"].astype(float)
            out["vol_fade"] = raw["vol_fade"]
            out["rsi2"] = raw["rsi2"]
        except Exception:
            pass
    return out


__all__ = ["sma", "ema", "donchian", "bollinger", "zscore", "rsi", "tsmom_signal", "vwap", "compute", "TRADING_DAYS"]
