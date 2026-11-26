"""Short-term momentum exhaustion: streaks, move size, volume fade, RSI and a TSMOM timing overlay.

Robert's idea (2026-11-25): a name that prints the same strong momentum four or five days in a row is near
a short-term peak and tends to reverse. This module measures those streaks and applies a causal overlay
on time-series momentum (delay entry, trim, re-enter after a pullback) plus an optional fade sleeve.

Every feature uses only data through date t. Forward returns for the event study start at t+1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from indicators import rsi as rsi_series
from risk import apply_caps, atr, blended_vol, keep_top, portfolio_vol_target
from strategies.base import Strategy
from strategies.tsmom import TimeSeriesMomentum

# Composite score weights (chart badge / default flag). Documented so the study can refer to them.
SCORE_WEIGHTS = {
    "streak4": 0.25, "streak5": 0.15, "move_1_5atr": 0.20, "move_2_5atr": 0.10,
    "vol_fade": 0.15, "rsi14": 0.15, "rsi2": 0.10, "stretch_atr": 0.10,
}
DEFAULT_FLAG = 0.50
HORIZONS = (1, 3, 5, 10)
STREAK_NS = (3, 4, 5, 6, 7)


def _consec(mask):
    """Consecutive True counts; resets to 0 on False. mask is a boolean ndarray (T, N)."""
    out = np.zeros(mask.shape, dtype=np.int16)
    if mask.size == 0:
        return out
    out[0] = mask[0]
    for i in range(1, len(mask)):
        prev = out[i - 1]
        out[i] = np.where(mask[i], prev + 1, 0)
    return out


def _run_sum(values, mask):
    """Cumulative sum of values within a current True-run; 0 when mask is False."""
    out = np.zeros(values.shape, dtype=float)
    if values.size == 0:
        return out
    out[0] = np.where(mask[0], values[0], 0.0)
    for i in range(1, len(values)):
        out[i] = np.where(mask[i], out[i - 1] + values[i], 0.0)
    return out


def _run_first(values, mask):
    """Value from the first day of the current True-run, carried forward."""
    out = np.full(values.shape, np.nan)
    if values.size == 0:
        return out
    out[0] = np.where(mask[0], values[0], np.nan)
    for i in range(1, len(values)):
        out[i] = np.where(mask[i], np.where(mask[i - 1], out[i - 1], values[i]), np.nan)
    return out


def rsi_frame(close, period):
    """Wilder RSI on a close DataFrame (same definition as indicators.rsi)."""
    return rsi_series(close, period)


def features(panel, columns=None):
    """Causal exhaustion features for many names. Returns a dict of DataFrames (dates x columns).

    Keys include up_streak, down_streak, up_move, down_move (total return over the streak),
    up_move_atr / down_move_atr (that move / ATR-14), up_move_z / down_move_z (move / (daily vol x sqrt(n))),
    vol_fade (1 if the last two days of the streak are quieter than the first two), vol_ratio,
    rsi2, rsi5, rsi14, dist_ma20, dist_ma20_atr, dist_ma10, above_sma100, tsmom, score (signed),
    flag_up, flag_down.
    """
    close = panel.close if columns is None else panel.close[columns]
    high = panel.high.reindex(columns=close.columns)
    low = panel.low.reindex(columns=close.columns)
    volume = panel.volume.reindex(columns=close.columns)
    cols, index = close.columns, close.index
    px = close.to_numpy(dtype=float)
    up = np.zeros(px.shape, dtype=bool)
    down = np.zeros(px.shape, dtype=bool)
    valid = np.isfinite(px)
    if len(px) > 1:
        up[1:] = valid[1:] & valid[:-1] & (px[1:] > px[:-1])
        down[1:] = valid[1:] & valid[:-1] & (px[1:] < px[:-1])
    up_streak = _consec(up)
    down_streak = _consec(down)
    log_ret = np.zeros(px.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        log_ret[1:] = np.where(valid[1:] & valid[:-1] & (px[:-1] > 0), np.log(px[1:] / px[:-1]), 0.0)
    up_log = _run_sum(log_ret, up)
    down_log = _run_sum(log_ret, down)
    up_move = np.expm1(up_log)
    down_move = np.expm1(down_log)          # negative or zero
    atr14 = atr(high, low, close, 14).to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        up_move_atr = np.where(atr14 > 0, (px * up_move) / atr14, np.nan)   # dollar move / ATR
        down_move_atr = np.where(atr14 > 0, (px * np.abs(down_move)) / atr14, np.nan)
    daily_vol = close.pct_change(fill_method=None).rolling(20, min_periods=10).std().to_numpy(dtype=float)
    n_up = np.maximum(up_streak, 1).astype(float)
    n_down = np.maximum(down_streak, 1).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        up_move_z = np.where(daily_vol > 0, up_move / (daily_vol * np.sqrt(n_up)), np.nan)
        down_move_z = np.where(daily_vol > 0, np.abs(down_move) / (daily_vol * np.sqrt(n_down)), np.nan)

    vol = volume.to_numpy(dtype=float)
    vol = np.where(np.isfinite(vol), vol, np.nan)
    first_up = _run_first(vol, up)
    first_down = _run_first(vol, down)
    prev_vol = np.vstack([np.full((1, vol.shape[1]), np.nan), vol[:-1]]) if len(vol) else vol
    last2_up = np.where(up_streak >= 2, (vol + prev_vol) / 2.0, vol)
    last2_down = np.where(down_streak >= 2, (vol + prev_vol) / 2.0, vol)
    # First two days: first-day volume carried, plus the second day's volume once the streak is >= 2.
    second_up = np.where((up_streak == 2), vol, np.nan)
    # carry second-day volume forward through the rest of the streak
    second_up_c = _run_first(np.where(up_streak >= 2, vol, np.nan), up_streak >= 2)
    first2_up = np.where(up_streak >= 2, (first_up + second_up_c) / 2.0, first_up)
    second_down_c = _run_first(np.where(down_streak >= 2, vol, np.nan), down_streak >= 2)
    first2_down = np.where(down_streak >= 2, (first_down + second_down_c) / 2.0, first_down)
    with np.errstate(invalid="ignore", divide="ignore"):
        vol_ratio_up = last2_up / first2_up
        vol_ratio_down = last2_down / first2_down
    vol_fade_up = (up_streak >= 3) & np.isfinite(vol_ratio_up) & (vol_ratio_up < 1.0)
    vol_fade_down = (down_streak >= 3) & np.isfinite(vol_ratio_down) & (vol_ratio_down < 1.0)
    # Combined fade flag for whichever streak is active.
    vol_fade = np.where(up_streak >= 3, vol_fade_up, np.where(down_streak >= 3, vol_fade_down, False))
    vol_ratio = np.where(up_streak >= 3, vol_ratio_up, np.where(down_streak >= 3, vol_ratio_down, np.nan))

    rsi2 = rsi_frame(close, 2)
    rsi5 = rsi_frame(close, 5)
    rsi14 = rsi_frame(close, 14)
    sma10 = close.rolling(10, min_periods=10).mean()
    sma20 = close.rolling(20, min_periods=20).mean()
    sma100 = close.rolling(100, min_periods=100).mean()
    dist_ma20 = close / sma20 - 1.0
    dist_ma10 = close / sma10 - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        dist_ma20_atr = (close - sma20) / atr(high, low, close, 14)
    above_sma100 = close > sma100
    tsmom = _tsmom_signal(close)

    score_up = _score(up_streak, up_move_atr, vol_fade_up, rsi14.to_numpy(dtype=float),
                      rsi2.to_numpy(dtype=float), dist_ma20_atr.to_numpy(dtype=float), side="up")
    score_down = _score(down_streak, down_move_atr, vol_fade_down, rsi14.to_numpy(dtype=float),
                        rsi2.to_numpy(dtype=float), dist_ma20_atr.to_numpy(dtype=float), side="down")
    score = score_up - score_down
    flag_up = score_up >= DEFAULT_FLAG
    flag_down = score_down >= DEFAULT_FLAG

    def frame(arr, dtype=float):
        return pd.DataFrame(arr, index=index, columns=cols, dtype=dtype)

    return {
        "up_streak": frame(up_streak, np.int16), "down_streak": frame(down_streak, np.int16),
        "up_move": frame(up_move), "down_move": frame(down_move),
        "up_move_atr": frame(up_move_atr), "down_move_atr": frame(down_move_atr),
        "up_move_z": frame(up_move_z), "down_move_z": frame(down_move_z),
        "vol_fade": frame(vol_fade.astype(float)), "vol_fade_up": frame(vol_fade_up.astype(float)),
        "vol_fade_down": frame(vol_fade_down.astype(float)), "vol_ratio": frame(vol_ratio),
        "rsi2": rsi2, "rsi5": rsi5, "rsi14": rsi14,
        "dist_ma20": dist_ma20, "dist_ma10": dist_ma10, "dist_ma20_atr": dist_ma20_atr,
        "above_sma100": above_sma100.astype(float), "tsmom": tsmom,
        "score_up": frame(score_up), "score_down": frame(score_down), "score": frame(score),
        "flag_up": frame(flag_up.astype(float)), "flag_down": frame(flag_down.astype(float)),
    }


def _tsmom_signal(close):
    from indicators import tsmom_signal
    # tsmom_signal is defined on a Series; apply column-wise would be slow. Recompute as the strategy does.
    parts = []
    for n in TimeSeriesMomentum.defaults["horizons"]:
        ret = close / close.shift(n) - 1
        ma = close.rolling(n, min_periods=n).mean()
        part = 0.5 * (ret > 0).astype(float) + 0.5 * (close > ma).astype(float)
        parts.append(part.where(ret.notna() & ma.notna()))
    return sum(parts) / len(parts)


def _score(streak, move_atr, fade, rsi14, rsi2, stretch, side="up"):
    s = np.zeros(streak.shape, dtype=float)
    s = s + SCORE_WEIGHTS["streak4"] * (streak >= 4)
    s = s + SCORE_WEIGHTS["streak5"] * (streak >= 5)
    s = s + SCORE_WEIGHTS["move_1_5atr"] * (np.nan_to_num(move_atr, nan=0) >= 1.5)
    s = s + SCORE_WEIGHTS["move_2_5atr"] * (np.nan_to_num(move_atr, nan=0) >= 2.5)
    s = s + SCORE_WEIGHTS["vol_fade"] * fade
    if side == "up":
        s = s + SCORE_WEIGHTS["rsi14"] * (np.nan_to_num(rsi14, nan=50) >= 70)
        s = s + SCORE_WEIGHTS["rsi2"] * (np.nan_to_num(rsi2, nan=50) >= 90)
        s = s + SCORE_WEIGHTS["stretch_atr"] * (np.nan_to_num(stretch, nan=0) >= 1.0)
    else:
        s = s + SCORE_WEIGHTS["rsi14"] * (np.nan_to_num(rsi14, nan=50) <= 30)
        s = s + SCORE_WEIGHTS["rsi2"] * (np.nan_to_num(rsi2, nan=50) <= 10)
        s = s + SCORE_WEIGHTS["stretch_atr"] * (np.nan_to_num(stretch, nan=0) <= -1.0)
    return np.clip(s, 0.0, 1.0)


def features_for_frame(frame):
    """Same features for a single-ticker OHLCV DataFrame (chart path)."""
    from market_data import Panel
    close = frame["close"].to_frame("X")
    high = frame["high"].to_frame("X")
    low = frame["low"].to_frame("X")
    volume = frame["volume"].to_frame("X")
    open_ = frame["open"].to_frame("X") if "open" in frame else close
    dividends = pd.DataFrame(0.0, index=frame.index, columns=["X"])
    panel = Panel(open_, high, low, close, volume, dividends, pd.Series({"X": "stock"}))
    raw = features(panel, columns=["X"])
    return {k: (v["X"] if isinstance(v, pd.DataFrame) else v) for k, v in raw.items()}


def badge(latest):
    """Compact ticker-page badge from a latest-values dict (keys as in features_for_frame)."""
    score = _num(latest.get("score") if "score" in latest else latest.get("exhaustion"))
    up = int(_num(latest.get("up_streak")) or 0)
    down = int(_num(latest.get("down_streak")) or 0)
    fade = _num(latest.get("vol_fade")) or 0
    flag_up = (_num(latest.get("flag_up") if "flag_up" in latest else latest.get("exhaustion_flag_up")) or 0) >= 0.5
    flag_down = (_num(latest.get("flag_down") if "flag_down" in latest else latest.get("exhaustion_flag_down")) or 0) >= 0.5
    if flag_up:
        label = f"Exhausted ↑ {up}d"
        if fade >= 0.5:
            label += " · vol fade"
        kind = "up"
    elif flag_down:
        label = f"Exhausted ↓ {down}d"
        if fade >= 0.5:
            label += " · vol fade"
        kind = "down"
    elif up >= 4 or down >= 4:
        side = "↑" if up >= down else "↓"
        n = up if up >= down else down
        label = f"Streak {side} {n}d"
        kind = "watch"
    else:
        label = "No exhaustion"
        kind = "none"
    return {"label": label, "kind": kind, "score": None if score is None else round(float(score), 3),
            "upStreak": up, "downStreak": down, "volumeFade": bool(fade >= 0.5),
            "flagUp": bool(flag_up), "flagDown": bool(flag_down)}


def _num(value):
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def exhausted_mask(feat, params, side="up"):
    """Boolean DataFrame: names that meet the overlay's exhaustion rule on this side."""
    p = params
    min_streak = int(p.get("min_streak", 4))
    min_move = float(p.get("min_move_atr", 0.0))
    need_fade = bool(p.get("require_vol_fade", False))
    rsi_rule = p.get("rsi_rule", "off")
    if side == "up":
        mask = feat["up_streak"] >= min_streak
        if min_move > 0:
            mask = mask & (feat["up_move_atr"] >= min_move)
        if need_fade:
            mask = mask & (feat["vol_fade_up"] >= 0.5)
        if rsi_rule == "rsi14_70":
            mask = mask & (feat["rsi14"] >= 70)
        elif rsi_rule == "rsi2_90":
            mask = mask & (feat["rsi2"] >= 90)
    else:
        mask = feat["down_streak"] >= min_streak
        if min_move > 0:
            mask = mask & (feat["down_move_atr"] >= min_move)
        if need_fade:
            mask = mask & (feat["vol_fade_down"] >= 0.5)
        if rsi_rule == "rsi14_70":
            mask = mask & (feat["rsi14"] <= 30)
        elif rsi_rule == "rsi2_90":
            mask = mask & (feat["rsi2"] <= 10)
    return mask.fillna(False)


def pullback_ready(feat, params, side="up"):
    """True when a delayed entry may fire: trend intact and a pullback confirmation prints."""
    rule = params.get("reentry_rule", "either")
    if side == "up":
        trend = (feat["tsmom"] > 0) | (feat["above_sma100"] >= 0.5)
        rsi_reset = (feat["rsi2"] < 50) | (feat["rsi14"] < 50)
        touch = feat["dist_ma20"] <= 0
        touch10 = feat["dist_ma10"] <= 0
    else:
        trend = (feat["tsmom"] < 0.5) | (feat["above_sma100"] < 0.5)
        rsi_reset = (feat["rsi2"] > 50) | (feat["rsi14"] > 50)
        touch = feat["dist_ma20"] >= 0
        touch10 = feat["dist_ma10"] >= 0
    if rule == "rsi_reset":
        confirm = rsi_reset
    elif rule == "touch_ma20":
        confirm = touch
    elif rule == "touch_ma10":
        confirm = touch10
    else:
        confirm = rsi_reset | touch | touch10
    return trend.fillna(False) & confirm.fillna(False)


def apply_overlay(base, feat, params):
    """Apply a causal don't-chase / trim / re-entry overlay to a long baseline weight frame."""
    mode = params.get("mode", "dont_chase")
    trim_frac = float(params.get("trim_frac", 0.5))
    trim_frac = min(1.0, max(0.0, trim_frac))
    exh = exhausted_mask(feat, params, "up")
    ready = pullback_ready(feat, params, "up")
    base = base.reindex(index=exh.index, columns=exh.columns).fillna(0.0)
    want = base > 0
    T, N = base.shape
    w = base.to_numpy(dtype=float)
    want_a = want.to_numpy()
    exh_a = exh.reindex(index=base.index, columns=base.columns).fillna(False).to_numpy()
    ready_a = ready.reindex(index=base.index, columns=base.columns).fillna(False).to_numpy()
    out = np.zeros((T, N), dtype=float)
    pending = np.zeros(N, dtype=bool)      # delayed a new entry, waiting to get in
    held = np.zeros(N, dtype=bool)
    last = np.zeros(N, dtype=float)
    for i in range(T):
        row = np.zeros(N, dtype=float)
        for j in range(N):
            target = w[i, j]
            if not want_a[i, j]:
                pending[j] = False
                held[j] = False
                last[j] = 0.0
                continue
            if not held[j] and not pending[j]:
                # fresh entry signal
                if exh_a[i, j] and mode in ("dont_chase", "reentry"):
                    pending[j] = True
                    continue
                if exh_a[i, j] and mode == "trim":
                    row[j] = target * (1.0 - trim_frac)
                    held[j] = row[j] > 0
                    last[j] = row[j]
                    continue
                row[j] = target
                held[j] = True
                last[j] = target
                continue
            if pending[j]:
                if mode == "reentry":
                    if (not exh_a[i, j]) and ready_a[i, j]:
                        row[j] = target
                        pending[j] = False
                        held[j] = True
                        last[j] = target
                    # else keep waiting
                else:
                    if not exh_a[i, j]:
                        row[j] = target
                        pending[j] = False
                        held[j] = True
                        last[j] = target
                continue
            # already held
            if mode == "trim" and exh_a[i, j]:
                row[j] = target * (1.0 - trim_frac)
                last[j] = row[j]
                held[j] = row[j] > 0
            elif mode == "reentry" and exh_a[i, j]:
                # hold the existing position (don't add); wait for a pullback to restore
                row[j] = last[j] if last[j] > 0 else target * (1.0 - trim_frac)
                if ready_a[i, j] and not exh_a[i, j]:
                    row[j] = target
                last[j] = row[j]
            else:
                row[j] = target
                last[j] = target
        out[i] = row
    return pd.DataFrame(out, index=base.index, columns=base.columns)


def fade_weights(panel, feat, params):
    """Short-term reversal sleeve: short exhausted-up names, long exhausted-down names, fixed hold."""
    p = params
    max_hold = int(p.get("max_hold", 5))
    max_positions = int(p.get("max_positions", 8))
    per_name = float(p.get("per_name_cap", 0.15))
    gross = float(p.get("gross_cap", 0.40))
    target_vol = float(p.get("target_vol", 0.06))
    both = bool(p.get("both_sides", True))
    up = exhausted_mask(feat, params, "up")
    down = exhausted_mask(feat, params, "down") if both else up * False
    cols = feat["up_streak"].columns
    index = feat["up_streak"].index
    up_a = up.reindex(index=index, columns=cols).fillna(False).to_numpy()
    down_a = down.reindex(index=index, columns=cols).fillna(False).to_numpy()
    score_up = feat["score_up"].reindex(index=index, columns=cols).fillna(0).to_numpy()
    score_down = feat["score_down"].reindex(index=index, columns=cols).fillna(0).to_numpy()
    T, N = up_a.shape
    pos = np.zeros((T, N), dtype=float)     # +1 long, -1 short
    hold = np.zeros(N, dtype=int)
    side = np.zeros(N, dtype=float)
    for i in range(T):
        for j in range(N):
            if hold[j] > 0:
                hold[j] -= 1
                if hold[j] == 0:
                    side[j] = 0.0
                pos[i, j] = side[j]
                continue
            if up_a[i, j]:
                side[j] = -1.0
                hold[j] = max_hold
                pos[i, j] = -1.0
            elif down_a[i, j]:
                side[j] = 1.0
                hold[j] = max_hold
                pos[i, j] = 1.0
    raw = pd.DataFrame(pos, index=index, columns=cols)
    # Size by inverse vol; rank by exhaustion score (abs).
    returns = panel.total_return().reindex(columns=cols)
    vol = blended_vol(returns, int(p.get("vol_window", 60)))
    sized = (raw / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rank = pd.DataFrame(np.where(pos < 0, score_up, np.where(pos > 0, score_down, 0.0)),
                        index=index, columns=cols)
    sized = keep_top(sized, rank, max_positions)
    active = [c for c in sized.columns if (sized[c].fillna(0) != 0).any()]
    if not active:
        return pd.DataFrame(0.0, index=index, columns=panel.close.columns)
    scaled = portfolio_vol_target(sized[active], returns.reindex(columns=active),
                                  target_vol, 1.0, window=int(p.get("vol_window", 60)))
    return apply_caps(scaled, per_name, gross, long_only=False).reindex(columns=panel.close.columns, fill_value=0.0)


# ====================================================================== strategies


class ExhaustionOverlay(Strategy):
    """TSMOM book with a don't-chase / trim / re-entry overlay. Long/flat; no shorts."""
    name = "tsmom_exhaustion"
    defaults = {
        "universe": "etf", "mode": "dont_chase", "min_streak": 4, "min_move_atr": 0.0,
        "require_vol_fade": False, "rsi_rule": "off", "trim_frac": 0.5,
        "reentry_rule": "either", "horizons": (50, 100), "target_vol": 0.10,
        "max_positions": 8, "per_name_cap": 0.30, "gross_cap": 1.0, "max_leverage": 1.0,
        "vol_window": 60, "rebalance": "W-FRI", "min_price": 5.0, "min_dollar_volume": 20e6,
    }

    def universe(self, panel):
        return TimeSeriesMomentum(**self._tsmom_params()).universe(panel)

    def _tsmom_params(self):
        p = self.params
        keys = ("horizons", "universe", "target_vol", "max_positions", "per_name_cap", "gross_cap",
                "max_leverage", "vol_window", "rebalance", "min_price", "min_dollar_volume")
        return {k: p[k] for k in keys}

    def target_weights(self, panel):
        tsmom = TimeSeriesMomentum(**self._tsmom_params())
        base = tsmom.target_weights(panel)
        cols = [c for c in base.columns if (base[c].abs() > 0).any()] or list(base.columns[:1])
        feat = features(panel, columns=cols)
        # Features may miss a column if we subset; align to base.
        for k, v in list(feat.items()):
            feat[k] = v.reindex(index=base.index, columns=base.columns)
        weights = apply_overlay(base, feat, self.params)
        self.last_diagnostics = {
            "mode": self.params["mode"],
            "exhaustedUp": int((exhausted_mask(feat, self.params, "up").iloc[-1]).sum()) if len(weights) else 0,
            "active": int((weights.iloc[-1] > 0).sum()) if len(weights) else 0,
            "baseActive": int((base.iloc[-1] > 0).sum()) if len(base) else 0,
        }
        self._base_weights = base
        return weights.reindex(columns=panel.close.columns, fill_value=0.0)

    def stops(self, panel):
        return TimeSeriesMomentum(**self._tsmom_params()).stops(panel)


class ExhaustionFade(Strategy):
    """Short-term fade of exhausted names (short up-streaks, long down-streaks). Separate from TSMOM."""
    name = "fade_exhaustion"
    defaults = {
        "min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True, "rsi_rule": "rsi14_70",
        "max_hold": 5, "max_positions": 8, "per_name_cap": 0.15, "gross_cap": 0.40,
        "target_vol": 0.06, "vol_window": 60, "both_sides": True, "universe": "stocks",
        "min_price": 5.0, "min_dollar_volume": 20e6,
    }

    def universe(self, panel):
        from universe import eligible
        kind = "stocks" if self.params["universe"] == "stocks" else "etf"
        return eligible(panel, kind, min_price=self.params["min_price"],
                        min_dollar_volume=self.params["min_dollar_volume"])

    def target_weights(self, panel):
        mask = self.universe(panel)
        cols = list(mask.columns[mask.any()])
        if not cols:
            return pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
        feat = features(panel, columns=cols)
        weights = fade_weights(panel, feat, self.params)
        self.last_diagnostics = {
            "longs": int((weights.iloc[-1] > 0).sum()) if len(weights) else 0,
            "shorts": int((weights.iloc[-1] < 0).sum()) if len(weights) else 0,
        }
        return weights.reindex(columns=panel.close.columns, fill_value=0.0)


def build_strategy(swing):
    """Swarm hook: construct an exhaustion family from a swing config dict."""
    family = swing.get("family")
    exh = dict(swing.get("exhaustion") or {})
    if family == "tsmom_exhaustion":
        return ExhaustionOverlay(**exh)
    if family == "tsmom_stocks_exhaustion":
        exh = {**exh, "universe": "stocks"}
        return ExhaustionOverlay(**exh)
    if family == "tsmom_stocks":
        return TimeSeriesMomentum(universe="stocks")
    if family == "fade_exhaustion":
        return ExhaustionFade(**exh)
    if family == "tsmom_plus_fade":
        from strategies.combined import CombinedStrategy
        overlay = ExhaustionOverlay(**dict(exh.get("overlay") or {}))
        fade = ExhaustionFade(**dict(exh.get("fade") or {}))
        return CombinedStrategy(sleeves=[overlay, fade], regime_model="off",
                                max_positions=int(exh.get("max_positions", 10)))
    raise ValueError(f"unknown exhaustion family {family!r}")


def entry_diagnostics(panel, base, overlay, start=None, end=None):
    """Compare delayed vs original entries: fill-price improvement (bps) and missed-run return.

    Fills are assumed at the next session's open, matching portfolio_sim's next_open execution.
    Positive improvement means the overlay bought cheaper than the baseline.
    Only names that actually appeared in either book are scanned.
    """
    idx = panel.dates
    if start is not None:
        start = pd.Timestamp(start)
    if end is not None:
        end = pd.Timestamp(end)
    in_window = np.ones(len(idx), dtype=bool)
    if start is not None:
        in_window &= np.asarray(idx >= start)
    if end is not None:
        in_window &= np.asarray(idx <= end)
    keep = [c for c in sorted(set(base.columns) | set(overlay.columns)) if c in panel.close.columns
            and (((c in base.columns) and (base[c].fillna(0) != 0).any())
                 or ((c in overlay.columns) and (overlay[c].fillna(0) != 0).any()))]
    empty = {"entryImprovementBps": {"n": 0}, "missedRunReturn": {"n": 0},
             "delayedEntries": 0, "skippedEntries": 0, "matchedEntries": 0}
    if not keep:
        return empty
    base = base.reindex(index=idx, columns=keep).fillna(0.0)
    overlay = overlay.reindex(index=idx, columns=keep).fillna(0.0)
    opens = panel.open.reindex(index=idx, columns=keep).to_numpy(dtype=float)
    b = base.to_numpy()
    o = overlay.to_numpy()
    improvements, missed, delayed, skipped, matched = [], [], 0, 0, 0
    T, N = b.shape
    for j in range(N):
        in_b = False
        in_o = False
        b_entry_i = None
        pending_from = None
        for i in range(T):
            wb, wo = b[i, j] > 0, o[i, j] > 0
            if not in_window[i]:
                in_b, in_o = wb, wo
                continue
            fill_i = i + 1
            if fill_i >= T:
                break
            fill_px = opens[fill_i, j]
            if wb and not in_b:
                b_entry_i = fill_i
                pending_from = fill_i
            if wo and not in_o:
                if pending_from is not None and b_entry_i is not None:
                    px0 = opens[b_entry_i, j]
                    if np.isfinite(fill_px) and np.isfinite(px0) and px0 > 0:
                        improvements.append((px0 / fill_px - 1.0) * 1e4)
                        delayed += 1
                        matched += 1
                    pending_from = None
                    b_entry_i = None
                else:
                    matched += 1
            if (not wb) and in_b:
                if pending_from is not None and not in_o:
                    px0 = opens[pending_from, j]
                    px1 = opens[min(fill_i, T - 1), j]
                    if np.isfinite(px0) and np.isfinite(px1) and px0 > 0:
                        missed.append(px1 / px0 - 1.0)
                    skipped += 1
                    pending_from = None
                    b_entry_i = None
                elif pending_from is not None and in_o:
                    pending_from = None
                    b_entry_i = None
            in_b, in_o = wb, wo
    def _summ(xs):
        xs = [x for x in xs if x is not None and math.isfinite(x)]
        if not xs:
            return {"n": 0}
        arr = np.array(xs, dtype=float)
        return {"n": int(len(arr)), "mean": float(arr.mean()), "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75))}
    return {"entryImprovementBps": _summ(improvements), "missedRunReturn": _summ(missed),
            "delayedEntries": delayed, "skippedEntries": skipped, "matchedEntries": matched}


def install(ops, tool_error):
    """Register the analyst `exhaustion` op on quant_tools.OPS."""
    def _op(args):
        return op_exhaustion(args, tool_error)
    _op.__doc__ = op_exhaustion.__doc__
    ops["exhaustion"] = _op


def op_exhaustion(args, tool_error=None):
    """Latest exhaustion features for tickers, plus the study summary if one has been written."""
    from pathlib import Path
    from fetch_and_model import DATA_DIR

    def fail(code, message):
        if tool_error:
            raise tool_error(code, message)
        raise ValueError(message)

    tickers = args.get("tickers") or args.get("ticker")
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [str(t).upper().strip() for t in (tickers or [])][:20]
    out = {}
    if tickers:
        from quant_tools import load_panel_for
        try:
            panel = load_panel_for(tickers)
        except Exception as error:
            fail("NO_DATA", str(error)[:400])
        feat = features(panel, columns=[c for c in tickers if c in panel.close.columns])
        latest = {}
        for t in tickers:
            if t not in feat["score"].columns:
                latest[t] = {"error": "no data"}
                continue
            row = {k: (None if not np.isfinite(float(feat[k][t].iloc[-1])) else float(feat[k][t].iloc[-1])
                       if feat[k][t].dtype != np.int16 else int(feat[k][t].iloc[-1]))
                   for k in ("score", "up_streak", "down_streak", "vol_fade", "rsi2", "rsi14",
                             "up_move_atr", "flag_up", "flag_down", "tsmom") if k in feat}
            row["asof"] = str(feat["score"].index[-1].date())
            row["badge"] = badge({**row, "score": row.get("score"), "up_streak": row.get("up_streak"),
                                  "down_streak": row.get("down_streak"), "vol_fade": row.get("vol_fade"),
                                  "flag_up": row.get("flag_up"), "flag_down": row.get("flag_down")})
            latest[t] = row
        out["latest"] = latest
    path = Path(args["path"]) if args.get("path") else DATA_DIR / "Research" / "swarm" / "exhaustion_study.json"
    if path.is_file():
        import json
        doc = json.loads(path.read_text(encoding="utf-8"))
        out["study"] = {"generatedAt": doc.get("generatedAt"), "verdict": doc.get("verdict"),
                        "eventStudyHighlights": doc.get("eventStudyHighlights"),
                        "walkForward": (doc.get("stats") or {}).get("walkForward"),
                        "overfitting": {k: (doc.get("stats") or {}).get(k)
                                        for k in ("deflatedSharpePre", "deflatedSharpeCumulative",
                                                  "pboPre", "tests")}}
    elif tickers:
        out["study"] = None
    else:
        fail("NO_DATA", "no tickers and no exhaustion_study.json yet")
    return out
