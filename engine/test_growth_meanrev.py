"""Tests for growth-adjusted mean reversion (strategies/growth_meanrev.py): no lookahead, point-in-time
fundamentals, growth-path z-scores and the dip/exit state machine. No broker calls."""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from strategies import growth_meanrev as gm
from strategies.growth_meanrev import GrowthMeanReversion
from test_strategy_framework import synthetic_panel


def test_ew_slope_recovers_growth_rate():
    idx = pd.bdate_range("2025-01-01", periods=200)
    logp = pd.DataFrame({"A": 0.001 * np.arange(200) + 3.0, "B": -0.0005 * np.arange(200) + 2.0}, index=idx)
    s = gm.ew_slope(logp, 30)
    assert abs(s["A"].iloc[-1] - 0.001) < 1e-9 and abs(s["B"].iloc[-1] + 0.0005) < 1e-9
    assert s["A"].iloc[:19].isna().all()            # min_periods


def test_growth_path_removes_trend_bias():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2025-01-01", periods=400)
    logp = pd.DataFrame({"A": 0.002 * np.arange(400) + np.cumsum(rng.normal(0, 0.004, 400)) * 0.2 + 3}, index=idx)
    strat = GrowthMeanReversion()
    comps = {"trend": gm.ew_slope(logp, 60), "fund": logp * np.nan, "sector": logp * np.nan, "market": logp * np.nan}
    z_growth, _ = strat.zscore(logp, strat.blend(comps, (1, 0, 0, 0)))
    z_flat, _ = strat.zscore(logp, strat.blend(comps, (0, 0, 0, 0)))
    assert z_flat["A"].iloc[100:].mean() > 1.0          # a flat mean calls a steady uptrend "overextended"
    assert abs(z_growth["A"].iloc[100:].mean()) < 0.5 * abs(z_flat["A"].iloc[100:].mean())


def _weights(strategy_factory, panel, asof):
    return strategy_factory().target_weights(panel.upto(asof)).iloc[-1]


def test_no_lookahead_config_and_learned():
    panel = synthetic_panel(sessions=320, stocks=10, seed=4)
    for weights in ("config", "learned"):
        make = lambda: GrowthMeanReversion(regime_filter="off", entry_z=-1.0, weights=weights, learn_min_obs=120,
                                           learn_every=40, use_fundamentals=False)
        full = make().target_weights(panel)
        for asof in (panel.dates[180], panel.dates[251], panel.dates[-1]):
            part = _weights(make, panel, asof)
            assert np.allclose(full.loc[asof].to_numpy(), part.reindex(full.columns).to_numpy(), atol=1e-12), (weights, asof)


def test_future_data_does_not_change_past_z():
    panel = synthetic_panel(sessions=300, stocks=8, seed=9)
    cut = panel.dates[200]
    s1 = GrowthMeanReversion(regime_filter="off", use_fundamentals=False)
    cols = [c for c in panel.close.columns if c.startswith("S")]
    logp, comps = s1.components(panel, cols)
    z1, _ = s1.zscore(logp, s1.blend(comps, (0.5, 0, 0.3, 0.2)))
    shocked = synthetic_panel(sessions=300, stocks=8, seed=9)
    shocked.close.loc[shocked.close.index > cut] *= 3.0
    logp2, comps2 = s1.components(shocked, cols)
    z2, _ = s1.zscore(logp2, s1.blend(comps2, (0.5, 0, 0.3, 0.2)))
    assert np.allclose(z1.loc[:cut].to_numpy(), z2.loc[:cut].to_numpy(), equal_nan=True)


def test_sector_map_is_causal_and_sensible():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2025-01-01", periods=260)
    sec = pd.DataFrame(rng.normal(0, 0.01, (260, 3)), index=idx, columns=["XLK", "XLF", "XLE"])
    stocks = pd.DataFrame({"T1": sec["XLK"] + rng.normal(0, 0.003, 260), "F1": sec["XLF"] + rng.normal(0, 0.003, 260)}, index=idx)
    m = gm.causal_sector_map(stocks, sec, window=126, every=21, min_obs=60)
    assert m["T1"].iloc[-1] == "XLK" and m["F1"].iloc[-1] == "XLF"
    assert m.iloc[:60].isna().all().all()
    later = stocks.copy()
    later.loc[idx[150]:, "T1"] = sec.loc[idx[150]:, "XLE"]          # T1 changes sector later on
    m2 = gm.causal_sector_map(later, sec, window=126, every=21, min_obs=60)
    assert (m2["T1"].iloc[:151] == m["T1"].iloc[:151]).all() or m["T1"].iloc[:151].isna().equals(m2["T1"].iloc[:151].isna())
    assert m2["T1"].iloc[-1] == "XLE"


def test_fundamentals_are_point_in_time(tmp_path):
    db = Path(tmp_path) / "ai_research.sqlite3"
    quarters = [("2025-03-31", 100.0), ("2025-06-30", 105.0), ("2025-09-30", 110.0), ("2025-12-31", 115.0), ("2026-03-31", 130.0)]
    payload = {"quarterly": {"revenue": [{"end": e, "val": v} for e, v in quarters]}}
    from contextlib import closing
    with closing(sqlite3.connect(db)) as c, c:
        c.execute("CREATE TABLE source_cache (key TEXT, source TEXT, ticker TEXT, fetched_at TEXT, expires_at TEXT, payload TEXT)")
        c.execute("INSERT INTO source_cache VALUES ('sec:ZZZ','SEC_FINANCIALS','ZZZ','x','x',?)", (json.dumps(payload),))
    gm._FUND_MEMO.clear()
    dates = pd.bdate_range("2026-01-01", "2026-07-31")
    f = gm.load_fundamental_growth(dates, ["ZZZ", "OTHER"], lag_days=60, path=db)
    usable = pd.Timestamp("2026-03-31") + pd.Timedelta(days=60)
    assert f.loc[f.index < usable, "ZZZ"].isna().all()
    assert abs(f.loc[f.index >= usable, "ZZZ"].iloc[0] - math.log(1.3) / 252) < 1e-12
    assert f["OTHER"].isna().all()
    gm._FUND_MEMO.clear()


def test_dip_below_growth_path_enters_and_reverts():
    panel = synthetic_panel(sessions=320, stocks=6, seed=11)
    name = "SA"
    n = len(panel.dates)
    path = 50 * np.exp(0.0015 * np.arange(n) + np.random.default_rng(0).normal(0, 0.004, n))
    dip = 260
    path[dip:dip + 3] *= np.array([0.965, 0.96, 0.965])   # a dip that keeps the 100-day trend intact
    for frame in (panel.close, panel.open):
        frame[name] = path
    panel.high[name] = path * 1.005
    panel.low[name] = path * 0.995
    s = GrowthMeanReversion(regime_filter="off", entry_z=-2.0, max_hold=10, weights="config", w_trend=1.0, w_fund=0.0,
                            w_sector=0.0, w_market=0.0, use_fundamentals=False, stop_atr=20.0)
    active, stops, z, _ = s._states(panel)
    assert z[name].iloc[dip] < -2.0
    assert active[name].iloc[dip] == 1.0
    held = active[name].iloc[dip:dip + 12]
    assert held.sum() <= 10 and active[name].iloc[dip + 11] == 0.0      # reverted or max hold
    a = active[name].to_numpy()
    entries = [i for i in range(1, len(a)) if a[i] == 1.0 and a[i - 1] == 0.0]
    assert entries and all(z[name].iloc[i] < -2.0 for i in entries)    # every entry is a dip below the growth path


def test_swarm_families_build():
    import swarm as sw
    from strategies.growth_meanrev import GrowthMeanReversion as G
    s = sw.build_strategy({"family": "growth_overlay", "growth": {"weights": "learned", "entry_z": -1.5}})
    assert any(isinstance(x, G) for x in s.sleeves) and s.params["regime_model"] == "fixed"
    assert isinstance(sw.build_strategy({"family": "sleeve_growth_mr", "growth": {}}), G)
    assert sw.build_strategy({"family": "old_overlay_fixed"}).params["regime_model"] == "fixed"


if __name__ == "__main__":
    import tempfile
    import traceback
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames:
                    with tempfile.TemporaryDirectory() as td:
                        fn(Path(td))
                else:
                    fn()
                print("PASS", name)
            except Exception:
                failed += 1
                print("FAIL", name)
                traceback.print_exc()
    sys.exit(1 if failed else 0)
