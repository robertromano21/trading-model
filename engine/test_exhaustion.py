"""Tests for engine/exhaustion.py: streaks, volume fade, no lookahead, and the TSMOM overlay state machine."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import exhaustion as ex
from market_data import Panel
from test_strategy_framework import synthetic_panel


def _panel_from_close(close, volume=None):
    close = close.astype(float)
    high = close * 1.005
    low = close * 0.995
    opening = close.shift().fillna(close)
    volume = volume if volume is not None else pd.DataFrame(1e6, index=close.index, columns=close.columns)
    dividends = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    classes = pd.Series({c: "stock" for c in close.columns})
    return Panel(opening, high, low, close, volume, dividends, classes,
                 pd.DataFrame(False, index=close.index, columns=close.columns))


def _close_with_streak(sessions=80, streak_start=50, streak_len=5, step=0.02, seed=1):
    """Mostly quiet path, then `streak_len` consecutive up days starting at streak_start."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2025-01-02", periods=sessions)
    # tiny noise that does not create accidental 4-day runs in the warmup
    noise = rng.choice([-0.001, 0.001, 0.0, 0.0], size=sessions)
    px = 50 * np.exp(np.cumsum(noise))
    for k in range(streak_len):
        px[streak_start + k] = px[streak_start + k - 1] * (1 + step)
    close = pd.DataFrame({"A": px}, index=idx)
    vol = np.full(sessions, 2e6, dtype=float)
    # fading volume over the streak
    for k in range(streak_len):
        vol[streak_start + k] = 2e6 * (1 - 0.15 * k)
    volume = pd.DataFrame({"A": vol}, index=idx)
    return _panel_from_close(close, volume), streak_start, streak_len


def test_up_and_down_streak_lengths():
    idx = pd.bdate_range("2025-01-02", periods=12)
    # 5 up days (indexes 1-5), a down day, 3 down days already started, then mix
    px = [100, 101, 102, 103, 104, 105, 104, 103, 102, 103, 104, 103]
    panel = _panel_from_close(pd.DataFrame({"A": px}, index=idx))
    feat = ex.features(panel)
    up = feat["up_streak"]["A"].to_numpy()
    down = feat["down_streak"]["A"].to_numpy()
    assert list(up) == [0, 1, 2, 3, 4, 5, 0, 0, 0, 1, 2, 0]
    assert list(down) == [0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 1]
    # 5-day cumulative up move is ~5%
    assert abs(feat["up_move"]["A"].iloc[5] - (105 / 100 - 1)) < 1e-12


def test_volume_fade_vs_rising():
    idx = pd.bdate_range("2025-01-02", periods=8)
    px = [100, 101, 102, 103, 104, 105, 106, 107]  # 7 up days from day 1
    fade = pd.DataFrame({"F": [1e6, 2e6, 1.8e6, 1.5e6, 1.2e6, 1.0e6, 0.8e6, 0.7e6]}, index=idx)
    rise = pd.DataFrame({"R": [1e6, 1e6, 1.2e6, 1.5e6, 1.8e6, 2.2e6, 2.5e6, 3e6]}, index=idx)
    close = pd.DataFrame({"F": px, "R": px}, index=idx)
    volume = pd.concat([fade, rise], axis=1)
    feat = ex.features(_panel_from_close(close, volume))
    # From day 3 onward the streak is >= 3, so fade is defined
    assert feat["vol_fade_up"]["F"].iloc[4] == 1.0
    assert feat["vol_fade_up"]["R"].iloc[4] == 0.0
    assert feat["vol_ratio"]["F"].iloc[4] < 1.0 < feat["vol_ratio"]["R"].iloc[4]


def test_rsi_matches_indicators():
    panel = synthetic_panel(sessions=80, stocks=3, seed=7)
    feat = ex.features(panel, columns=["SA"])
    from indicators import rsi
    expected = rsi(panel.close["SA"], 14)
    got = feat["rsi14"]["SA"]
    assert np.allclose(got.dropna().to_numpy(), expected.reindex(got.dropna().index).to_numpy(), equal_nan=True)


def test_features_have_no_lookahead():
    panel, mid, _ = _close_with_streak(sessions=70, streak_start=40, streak_len=5)
    cut = panel.dates[45]
    feat1 = ex.features(panel)
    shocked = _panel_from_close(panel.close.copy(), panel.volume.copy())
    shocked.close.loc[shocked.close.index > cut] *= 1.5
    shocked.high.loc[shocked.high.index > cut] *= 1.5
    shocked.low.loc[shocked.low.index > cut] *= 1.5
    feat2 = ex.features(shocked)
    for key in ("up_streak", "up_move", "vol_fade", "rsi14", "score", "flag_up"):
        a, b = feat1[key].loc[:cut], feat2[key].loc[:cut]
        assert np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True), key


def test_frame_features_match_panel():
    panel, _, _ = _close_with_streak()
    frame = pd.DataFrame({"open": panel.open["A"], "high": panel.high["A"], "low": panel.low["A"],
                          "close": panel.close["A"], "volume": panel.volume["A"]})
    one = ex.features_for_frame(frame)
    many = ex.features(panel, columns=["A"])
    for key in ("up_streak", "score", "rsi2", "flag_up"):
        assert np.allclose(one[key].to_numpy(dtype=float), many[key]["A"].to_numpy(dtype=float), equal_nan=True)


def test_dont_chase_delays_entry_until_streak_breaks():
    # Baseline wants to be long the whole window; a 5-day up streak should keep weight at 0,
    # then a down day (exhaustion clears) should allow entry.
    idx = pd.bdate_range("2025-01-02", periods=16)
    px = [100, 101, 102, 103, 104, 105, 106, 100, 101, 102, 103, 104, 105, 106, 107, 108]
    panel = _panel_from_close(pd.DataFrame({"A": px}, index=idx))
    feat = ex.features(panel)
    base = pd.DataFrame(0.25, index=idx, columns=["A"])
    base.iloc[:2] = 0.0          # first weights appear on the 3rd row (already 2 up days, not exhausted yet)
    # Force the "entry" to land on the 5th up day (index 5, streak=5)
    base.iloc[:] = 0.0
    base.iloc[5:] = 0.25
    delayed = ex.apply_overlay(base, feat, {"mode": "dont_chase", "min_streak": 4, "min_move_atr": 0.0,
                                            "require_vol_fade": False, "rsi_rule": "off"})
    assert delayed["A"].iloc[5] == 0.0          # exhausted, entry delayed
    assert delayed["A"].iloc[6] == 0.0          # still in the streak (6 up days)
    assert delayed["A"].iloc[7] == 0.25         # down day clears exhaustion, enter
    # A no-delay baseline would have been in from index 5
    plain = ex.apply_overlay(base, feat, {"mode": "dont_chase", "min_streak": 99})
    assert plain["A"].iloc[5] == 0.25


def test_trim_cuts_an_existing_position():
    idx = pd.bdate_range("2025-01-02", periods=12)
    px = [100, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # long up streak after index 2
    panel = _panel_from_close(pd.DataFrame({"A": px}, index=idx))
    feat = ex.features(panel)
    base = pd.DataFrame(0.40, index=idx, columns=["A"])
    trimmed = ex.apply_overlay(base, feat, {"mode": "trim", "min_streak": 4, "trim_frac": 0.5,
                                            "min_move_atr": 0.0, "require_vol_fade": False, "rsi_rule": "off"})
    # streak at index 5 is 4 (days 2-5? wait: day1 down 100->99, day2 up, ... index 5 is 5th bar = 104, up streak from idx 2)
    # idx 0: 100, idx 1: 99 down, idx 2: 100 up 1, ... idx 5: 104 up 4
    assert feat["up_streak"]["A"].iloc[5] == 4
    assert abs(trimmed["A"].iloc[5] - 0.20) < 1e-12
    assert abs(trimmed["A"].iloc[1] - 0.40) < 1e-12   # no up streak yet, full weight


def test_reentry_waits_for_pullback():
    idx = pd.bdate_range("2025-01-02", periods=20)
    # 5-day burst, then still above the 20d MA for a bit (20d MA isn't defined yet — use rsi_reset only)
    # Build a rising series so RSI stays high after the burst, then a sharp drop that resets RSI2.
    px = [100 + i * 0.1 for i in range(10)]
    burst = px[-1]
    for _ in range(5):
        burst *= 1.03
        px.append(burst)
    px.append(burst * 0.92)   # sharp pullback
    px.append(px[-1] * 0.99)
    px.extend([px[-1]] * (20 - len(px)))
    panel = _panel_from_close(pd.DataFrame({"A": px}, index=idx))
    feat = ex.features(panel)
    base = pd.DataFrame(0.0, index=idx, columns=["A"])
    # Entry signal starts on the 4th day of the burst (index 13)
    burst0 = 10
    base.iloc[burst0 + 3:] = 0.30
    out = ex.apply_overlay(base, feat, {"mode": "reentry", "min_streak": 4, "reentry_rule": "rsi_reset",
                                        "min_move_atr": 0.0, "require_vol_fade": False, "rsi_rule": "off"})
    # During the burst the position stays flat
    assert (out["A"].iloc[burst0 + 3:burst0 + 5] == 0.0).all()
    # After the pullback, RSI2 should reset and we may enter (trend may be thin without 100d MA;
    # pullback_ready uses tsmom OR sma100 — tsmom needs 50/100 days, so on this short panel trend is False).
    # For the unit test, poke trend on so we isolate the RSI-reset gate.
    feat = dict(feat)
    feat["tsmom"] = pd.DataFrame(1.0, index=idx, columns=["A"])
    feat["above_sma100"] = pd.DataFrame(1.0, index=idx, columns=["A"])
    out = ex.apply_overlay(base, feat, {"mode": "reentry", "min_streak": 4, "reentry_rule": "rsi_reset",
                                        "min_move_atr": 0.0, "require_vol_fade": False, "rsi_rule": "off"})
    assert out["A"].iloc[burst0 + 3] == 0.0
    assert out["A"].iloc[burst0 + 5] == 0.30 or out["A"].iloc[burst0 + 6] == 0.30


def test_overlay_no_lookahead():
    panel, mid, n = _close_with_streak(sessions=90, streak_start=50, streak_len=5)
    # A made-up baseline that turns on mid-sample
    base = pd.DataFrame(0.0, index=panel.dates, columns=["A"])
    base.iloc[40:] = 0.2
    feat = ex.features(panel)
    w1 = ex.apply_overlay(base, feat, {"mode": "dont_chase", "min_streak": 4})
    cut = panel.dates[55]
    shocked = _panel_from_close(panel.close.copy(), panel.volume.copy())
    shocked.close.loc[shocked.close.index > cut] *= 2.0
    feat2 = ex.features(shocked)
    w2 = ex.apply_overlay(base, feat2, {"mode": "dont_chase", "min_streak": 4})
    assert np.allclose(w1.loc[:cut].to_numpy(), w2.loc[:cut].to_numpy(), equal_nan=True)


def test_fade_goes_short_after_up_exhaustion():
    panel, start, n = _close_with_streak(sessions=60, streak_start=40, streak_len=5, step=0.03)
    # Stretch volume fade + RSI by making a strong 5-day run
    feat = ex.features(panel)
    params = {"min_streak": 4, "min_move_atr": 0.0, "require_vol_fade": False, "rsi_rule": "off",
              "max_hold": 3, "max_positions": 4, "per_name_cap": 0.2, "gross_cap": 0.5,
              "target_vol": 0.1, "both_sides": True, "vol_window": 20}
    w = ex.fade_weights(panel, feat, params)
    # On the 4th/5th up day the sleeve should be short (or starting to be)
    row = w["A"].iloc[start + 3]
    assert row < 0 or w["A"].iloc[start + 4] < 0
    # After max_hold the short is gone
    held = w["A"].iloc[start + 3:start + 3 + 6]
    assert (held.iloc[-1] == 0.0) or (held[held < 0].shape[0] <= 3)


def test_build_strategy_families():
    import swarm as sw
    o = sw.build_strategy({"family": "tsmom_exhaustion",
                           "exhaustion": {"mode": "dont_chase", "min_streak": 5}})
    assert isinstance(o, ex.ExhaustionOverlay) and o.params["min_streak"] == 5
    s = sw.build_strategy({"family": "tsmom_stocks_exhaustion", "exhaustion": {"mode": "trim"}})
    assert s.params["universe"] == "stocks"
    f = sw.build_strategy({"family": "fade_exhaustion", "exhaustion": {"max_hold": 3}})
    assert isinstance(f, ex.ExhaustionFade) and f.params["max_hold"] == 3
    b = sw.build_strategy({"family": "tsmom_stocks"})
    from strategies.tsmom import TimeSeriesMomentum
    assert isinstance(b, TimeSeriesMomentum) and b.params["universe"] == "stocks"


def test_badge_labels():
    up = ex.badge({"score": 0.7, "up_streak": 5, "down_streak": 0, "vol_fade": 1, "flag_up": 1, "flag_down": 0})
    assert up["kind"] == "up" and "5d" in up["label"] and "fade" in up["label"]
    none = ex.badge({"score": 0.0, "up_streak": 1, "down_streak": 0, "vol_fade": 0, "flag_up": 0, "flag_down": 0})
    assert none["kind"] == "none"
    watch = ex.badge({"score": 0.3, "up_streak": 4, "down_streak": 0, "vol_fade": 0, "flag_up": 0, "flag_down": 0})
    assert watch["kind"] == "watch"


def test_holdout_selection_ignores_holdout_rows():
    """Sanity: swarm.select (used by the study) is unchanged if we perturb the holdout block."""
    import swarm as sw
    idx = pd.bdate_range("2025-01-02", periods=40)
    rng = np.random.default_rng(0)
    pre = pd.DataFrame(rng.normal(0.0004, 0.01, (40, 3)), index=idx, columns=[1, 2, 3])
    ranked, _ = sw.select(pre.iloc[:30], {1, 2, 3})
    shocked = pre.copy()
    shocked.iloc[30:] = 9.0
    ranked2, _ = sw.select(shocked.iloc[:30], {1, 2, 3})
    assert ranked == ranked2


def test_entry_diagnostics_counts_a_delay():
    idx = pd.bdate_range("2025-01-02", periods=8)
    close = pd.DataFrame({"A": [100, 101, 102, 103, 104, 105, 106, 107]}, index=idx)
    panel = _panel_from_close(close)
    # Force opens = close so the math is easy
    panel.open.loc[:, "A"] = close["A"]
    base = pd.DataFrame(0.0, index=idx, columns=["A"])
    over = base.copy()
    base.iloc[2:] = 0.2          # decision day 3, fill next open (day 4 = 103)
    over.iloc[4:] = 0.2          # delayed two sessions, fill at day 6 = 105
    d = ex.entry_diagnostics(panel, base, over)
    assert d["delayedEntries"] >= 1
    # bought later at a higher price -> negative improvement
    if d["entryImprovementBps"]["n"]:
        assert d["entryImprovementBps"]["mean"] < 0


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
