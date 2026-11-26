"""Event study and walk-forward test of a short-term exhaustion overlay on TSMOM.

Event study (pre-holdout dates only) over liquid stocks and the core ETFs: consecutive-day streaks (N=3..7),
move size in ATR, volume fade, RSI(2/14) and distance above the 20-day average, split by whether the name
is in a longer-term uptrend. Forward 1/3/5/10-day returns and reversal rates are tested against same-date
universe base rates with date-clustered HAC standard errors and Holm/BH across the grid.

Timing overlay on TSMOM (ETF book, plus a stock-momentum book): don't-chase, trim, and re-entry after a
pullback, plus a separate fade sleeve. Parameter search uses a small Latin hypercube; every trial is counted.
Walk-forward is expanding, purged, after Schwab costs, whole shares and the $10k rules, versus plain TSMOM
and SPY. The champion is frozen on pre-holdout data; the sealed holdout is looked at once and logged.
Recommendation only: the live paper account is not changed.

  python exhaustion_study.py [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

import swarm as sw
import exhaustion as ex
from universe import CORE_ETFS, eligible

OUT = sw.SWARM_DIR / "exhaustion_study.json"
FORWARD_START_ID = 120
FORWARD_EXTRAS = 5
PURGE = 10                    # sessions dropped between train and test (max event horizon)
LHC_N = 16
LHC_SEED = 20261125
SPACE = {
    "mode": ["dont_chase", "trim", "reentry"],
    "min_streak": [4, 5],
    "min_move_atr": [0.0, 1.5, 2.5],
    "require_vol_fade": [False, True],
    "rsi_rule": ["off", "rsi14_70", "rsi2_90"],
    "trim_frac": [0.5, 1.0],
    "reentry_rule": ["rsi_reset", "touch_ma20", "either"],
}
FIXED = [
    {"id": 316, "label": "etf_dont_chase_5", "universe": "etf",
     "exh": {"mode": "dont_chase", "min_streak": 5, "min_move_atr": 0.0, "require_vol_fade": False,
             "rsi_rule": "off", "trim_frac": 0.5, "reentry_rule": "either"}},
    {"id": 317, "label": "etf_reentry_strict_4", "universe": "etf",
     "exh": {"mode": "reentry", "min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True,
             "rsi_rule": "rsi14_70", "trim_frac": 0.5, "reentry_rule": "either"}},
    {"id": 318, "label": "etf_trim_strict_4", "universe": "etf",
     "exh": {"mode": "trim", "min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True,
             "rsi_rule": "rsi14_70", "trim_frac": 0.5, "reentry_rule": "either"}},
]
STOCK_FIXED = [
    {"id": 320, "label": "stocks_tsmom", "family": "tsmom_stocks", "exh": None},
    {"id": 321, "label": "stocks_dont_chase_5", "universe": "stocks",
     "exh": {"mode": "dont_chase", "min_streak": 5, "min_move_atr": 0.0, "require_vol_fade": False,
             "rsi_rule": "off", "universe": "stocks"}},
    {"id": 322, "label": "stocks_reentry_strict_4", "universe": "stocks",
     "exh": {"mode": "reentry", "min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True,
             "rsi_rule": "rsi14_70", "reentry_rule": "either", "universe": "stocks"}},
    {"id": 323, "label": "stocks_trim_strict_4", "universe": "stocks",
     "exh": {"mode": "trim", "min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True,
             "rsi_rule": "rsi14_70", "trim_frac": 0.5, "universe": "stocks"}},
]
FADE = [
    {"id": 340, "label": "fade_stocks_default",
     "exh": {"min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True, "rsi_rule": "rsi14_70",
             "max_hold": 5, "universe": "stocks", "both_sides": True}},
    {"id": 341, "label": "fade_etf_hold3",
     "exh": {"min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True, "rsi_rule": "rsi14_70",
             "max_hold": 3, "universe": "etf", "both_sides": True}},
    {"id": 342, "label": "fade_stocks_streak5",
     "exh": {"min_streak": 5, "min_move_atr": 1.5, "require_vol_fade": True, "rsi_rule": "off",
             "max_hold": 5, "universe": "stocks", "both_sides": True}},
    {"id": 343, "label": "fade_stocks_up_only",
     "exh": {"min_streak": 4, "min_move_atr": 1.5, "require_vol_fade": True, "rsi_rule": "rsi14_70",
             "max_hold": 5, "universe": "stocks", "both_sides": False}},
]
REFERENCE = [
    {"id": 1, "label": "tsmom_baseline", "swing": {"family": "tsmom_baseline"}, "orb": None},
]


def _overlay_account(i, label, universe, exh):
    family = "tsmom_stocks_exhaustion" if universe == "stocks" else "tsmom_exhaustion"
    params = dict(exh)
    params["universe"] = universe
    return {"id": i, "label": label, "grid": params, "swing": {"family": family, "exhaustion": params}, "orb": None}


def variants():
    out = []
    for k, row in enumerate(sw.latin_levels(SPACE, LHC_N, LHC_SEED)):
        out.append(_overlay_account(300 + k, "lhc_" + "_".join(str(row[x]) for x in SPACE), "etf", row))
    for row in FIXED:
        out.append(_overlay_account(row["id"], row["label"], row["universe"], row["exh"]))
    for row in STOCK_FIXED:
        if row.get("family") == "tsmom_stocks":
            out.append({"id": row["id"], "label": row["label"], "grid": {"universe": "stocks"},
                        "swing": {"family": "tsmom_stocks"}, "orb": None})
        else:
            out.append(_overlay_account(row["id"], row["label"], "stocks", row["exh"]))
    for row in FADE:
        out.append({"id": row["id"], "label": row["label"], "grid": row["exh"],
                    "swing": {"family": "fade_exhaustion", "exhaustion": row["exh"], "use_tax": False},
                    "orb": None})
    return out


# ====================================================================== event study


def _hac_mean(values, lags):
    y = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = int(len(y))
    if n < 8:
        return {"nDates": n, "mean": None, "se": None, "t": None, "p": None, "pLess": None, "pGreater": None}
    lags = int(max(1, min(lags, n - 2)))
    res = sm.OLS(y, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    mean, se, t, p = float(res.params[0]), float(res.bse[0]), float(res.tvalues[0]), float(res.pvalues[0])
    # one-sided p-values from the HAC t, normal reference (HAC df is not well defined)
    p_less = float(norm.cdf(t))
    p_greater = float(norm.sf(t))
    return {"nDates": n, "mean": mean, "se": se, "t": t, "p": p, "pLess": p_less, "pGreater": p_greater}


def _cell(name, mask, side, **meta):
    return {"name": name, "mask": mask, "side": side, **meta}


def event_cells(feat, liquid):
    """Pre-registered cells. Main effects for N=3..7, one-way splits on N=4 and 5, a few interactions."""
    cells = []
    for n in ex.STREAK_NS:
        for side, streak in (("up", feat["up_streak"]), ("down", feat["down_streak"])):
            cells.append(_cell(f"{side}_exact_{n}", liquid & (streak == n), side, n=n, split="main"))
    for n in (4, 5):
        uptrend = (feat["above_sma100"] >= 0.5) | (feat["tsmom"] > 0)
        for side, streak, move, fade, rsi14, rsi2, stretch, trend in (
            ("up", feat["up_streak"], feat["up_move_atr"], feat["vol_fade_up"], feat["rsi14"], feat["rsi2"],
             feat["dist_ma20_atr"], uptrend),
            ("down", feat["down_streak"], feat["down_move_atr"], feat["vol_fade_down"], feat["rsi14"], feat["rsi2"],
             -feat["dist_ma20_atr"], uptrend),
        ):
            base = liquid & (streak == n)
            rsi_ext = (rsi14 >= 70) if side == "up" else (rsi14 <= 30)
            rsi2_ext = (rsi2 >= 90) if side == "up" else (rsi2 <= 10)
            cells.append(_cell(f"{side}_{n}_move>=1.5atr", base & (move >= 1.5), side, n=n, split="move"))
            cells.append(_cell(f"{side}_{n}_move>=2.5atr", base & (move >= 2.5), side, n=n, split="move"))
            cells.append(_cell(f"{side}_{n}_vol_fade", base & (fade >= 0.5), side, n=n, split="volume"))
            cells.append(_cell(f"{side}_{n}_vol_rising", base & (fade < 0.5), side, n=n, split="volume"))
            cells.append(_cell(f"{side}_{n}_rsi14_ext", base & rsi_ext, side, n=n, split="rsi14"))
            cells.append(_cell(f"{side}_{n}_rsi2_ext", base & rsi2_ext, side, n=n, split="rsi2"))
            cells.append(_cell(f"{side}_{n}_stretch>=1atr", base & (stretch >= 1.0), side, n=n, split="stretch"))
            cells.append(_cell(f"{side}_{n}_in_uptrend", base & trend.fillna(False), side, n=n, split="trend"))
            cells.append(_cell(f"{side}_{n}_not_uptrend", base & (~trend.fillna(False)), side, n=n, split="trend"))
            combo = base & (move >= 1.5) & (fade >= 0.5) & rsi_ext
            cells.append(_cell(f"{side}_{n}_large_fade_rsi", combo, side, n=n, split="interaction"))
            cells.append(_cell(f"{side}_{n}_large_fade_rsi_in_trend", combo & trend.fillna(False), side, n=n,
                               split="interaction"))
            cells.append(_cell(f"{side}_{n}_large_fade_rsi_out_trend", combo & (~trend.fillna(False)), side, n=n,
                               split="interaction"))
            cells.append(_cell(f"{side}_ge_{n}", liquid & (streak >= n), side, n=n, split="ge"))
    return cells


def run_event_study(panel, holdout_start, eval_start):
    stocks = eligible(panel, "stocks", min_price=5.0, min_dollar_volume=20e6)
    etfs = eligible(panel, "etf", min_price=5.0, min_dollar_volume=5e6)
    liquid = stocks.fillna(False) | etfs.fillna(False)
    cols = list(liquid.columns[liquid.any()])
    # Always include core ETFs that are in the panel
    for t in CORE_ETFS:
        if t in panel.close.columns and t not in cols:
            cols.append(t)
    sw.log(f"event study: {len(cols)} liquid names, {len(panel.dates)} sessions")
    feat = ex.features(panel, columns=cols)
    liquid = liquid.reindex(index=panel.dates, columns=cols).fillna(False)
    close = panel.close.reindex(columns=cols)
    opens = panel.open.reindex(columns=cols)
    # Use the full pre-holdout history (including Schwab daily_long when present).
    # eval_start is only a floor when the panel is short; 120 sessions of warmup come first.
    warmup = 120
    first = panel.dates[warmup] if len(panel.dates) > warmup else panel.dates[0]
    if pd.Timestamp(eval_start) < first:
        first = pd.Timestamp(eval_start)
    pre = (panel.dates >= first) & (panel.dates < pd.Timestamp(holdout_start))
    # Drop the last 10 pre-holdout days so 10-day forward returns do not spill into the holdout
    pre_idx = panel.dates[pre]
    if len(pre_idx) > PURGE:
        pre = pre & (panel.dates <= pre_idx[-PURGE - 1])
    pre_mask = pd.Series(pre, index=panel.dates)
    in_pre = liquid.copy()
    in_pre.loc[~pre_mask.reindex(in_pre.index).fillna(False)] = False
    fwd, fwd_open, univ, univ_open, univ_neg, univ_neg_open = {}, {}, {}, {}, {}, {}
    for h in ex.HORIZONS:
        # close_t -> close_{t+h}
        fc = close.shift(-h) / close - 1.0
        # next open -> open h sessions later (tradable)
        fo = opens.shift(-1 - h) / opens.shift(-1) - 1.0
        fwd[h], fwd_open[h] = fc, fo
        # daily universe means (equal-weight liquid names on pre-holdout dates)
        um = fc.where(in_pre).mean(axis=1)
        uo = fo.where(in_pre).mean(axis=1)
        un = (fc.where(in_pre) < 0).sum(axis=1) / in_pre.sum(axis=1).replace(0, np.nan)
        uno = (fo.where(in_pre) < 0).sum(axis=1) / in_pre.sum(axis=1).replace(0, np.nan)
        univ[h], univ_open[h], univ_neg[h], univ_neg_open[h] = um, uo, un, uno
    cells = event_cells({k: v.reindex(columns=cols) for k, v in feat.items()}, liquid)
    rows = []
    for cell in cells:
        mask = cell["mask"].reindex(index=panel.dates, columns=cols).fillna(False) & in_pre
        n_events = int(mask.to_numpy().sum())
        n_dates = int((mask.any(axis=1)).to_numpy().sum())
        rec = {k: cell[k] for k in cell if k != "mask"}
        rec["nEvents"] = n_events
        rec["nDates"] = n_dates
        rec["horizons"] = {}
        for h in ex.HORIZONS:
            rec["horizons"][str(h)] = _horizon_stats(mask, fwd[h], univ[h], univ_neg[h], cell["side"], h,
                                                     fwd_open[h], univ_open[h], univ_neg_open[h])
        rows.append(rec)
    tests = []
    for rec in rows:
        for h, st in rec["horizons"].items():
            if st.get("excess") and st["excess"].get("pLess") is not None:
                p_rev = st["excess"]["pLess"] if rec["side"] == "up" else st["excess"]["pGreater"]
                tests.append((rec["name"], int(h), "close_excess", p_rev, st["excess"].get("mean")))
            if st.get("openExcess") and st["openExcess"].get("pLess") is not None:
                p_rev = st["openExcess"]["pLess"] if rec["side"] == "up" else st["openExcess"]["pGreater"]
                tests.append((rec["name"], int(h), "open_excess", p_rev, st["openExcess"].get("mean")))
    pvals = [t[3] for t in tests]
    holm = sw.holm(pvals) if pvals else []
    bh = sw.bh(pvals) if pvals else []
    key = {(t[0], t[1], t[2]): (float(h), float(b), t[4]) for t, h, b in zip(tests, holm, bh)}
    for rec in rows:
        for h, st in rec["horizons"].items():
            for kind, slot in (("close_excess", "excess"), ("open_excess", "openExcess")):
                k = (rec["name"], int(h), kind)
                if k in key and st.get(slot):
                    st[slot]["holmP"] = key[k][0]
                    st[slot]["bhP"] = key[k][1]
    # Highlights: 5-day close-to-close excess, BH < 0.05, at least 30 dates
    highlights = []
    for rec in rows:
        st = rec["horizons"].get("5") or {}
        exs = st.get("excess") or {}
        if exs.get("bhP") is not None and exs["bhP"] < 0.05 and rec["nDates"] >= 30:
            sign_ok = (rec["side"] == "up" and (exs.get("mean") or 0) < 0) or (
                rec["side"] == "down" and (exs.get("mean") or 0) > 0)
            if sign_ok:
                highlights.append({"name": rec["name"], "side": rec["side"], "nEvents": rec["nEvents"],
                                   "nDates": rec["nDates"], "excess5d": exs.get("mean"),
                                   "reversalRate": (st.get("reversal") or {}).get("rate"),
                                   "baseRate": (st.get("reversal") or {}).get("baseRate"),
                                   "holmP": exs.get("holmP"), "bhP": exs.get("bhP")})
    highlights.sort(key=lambda r: (r.get("bhP") is None, r.get("bhP"), -abs(r.get("excess5d") or 0)))
    coverage = {
        "names": len(cols), "stocks": int(stocks.any().sum()), "etfs": int(etfs.any().sum()),
        "preStart": str(panel.dates[pre][0].date()) if pre.any() else None,
        "preEnd": str(panel.dates[pre][-1].date()) if pre.any() else None,
        "tests": len(tests),
        "note": ("Forward close-to-close returns start after the event close (no lookahead). Open-to-open "
                 "returns start at the next session's open, which is the tradable path. Tests are date-clustered "
                 "HAC on the daily cross-sectional mean excess versus same-date liquid-universe means. "
                 "Holm/BH are applied across every cell × horizon × close/open test. The last "
                 f"{PURGE} pre-holdout sessions are dropped so 10-day returns do not touch the holdout."),
    }
    # Drop bulky masks already removed; keep a compact table
    return {"coverage": coverage, "cells": rows, "highlights": highlights,
            "nSignificantBH_5d": len(highlights)}


def _horizon_stats(mask, fwd, univ, univ_neg, side, horizon, fwd_open, univ_open, univ_neg_open):
    ev = fwd.where(mask)
    ev_o = fwd_open.where(mask)
    daily = ev.mean(axis=1)
    daily_o = ev_o.mean(axis=1)
    excess = daily - univ
    excess_o = daily_o - univ_open
    # reversal rate: up-streak, P(fwd < 0); down-streak, P(fwd > 0)
    n_day = mask.sum(axis=1).replace(0, np.nan)
    if side == "up":
        rev = (ev < 0).sum(axis=1) / n_day
        rev_o = (ev_o < 0).sum(axis=1) / n_day
        base, base_o = univ_neg, univ_neg_open
    else:
        rev = (ev > 0).sum(axis=1) / n_day
        rev_o = (ev_o > 0).sum(axis=1) / n_day
        base, base_o = 1 - univ_neg, 1 - univ_neg_open
    n = int(mask.to_numpy().sum())
    raw_mean = float(ev.stack().mean()) if n else None
    out = {
        "nEvents": n,
        "raw": _hac_mean(daily, horizon),
        "excess": _hac_mean(excess, horizon),
        "openRaw": _hac_mean(daily_o, horizon),
        "openExcess": _hac_mean(excess_o, horizon),
        "reversal": _rate_vs_base(rev, base, horizon),
        "openReversal": _rate_vs_base(rev_o, base_o, horizon),
        "eventMean": raw_mean,
    }
    return out


def _rate_vs_base(rate, base, lags):
    diff = rate - base
    st = _hac_mean(diff, lags)
    r = pd.Series(rate).dropna()
    b = pd.Series(base).dropna()
    st["rate"] = float(r.mean()) if len(r) else None
    st["baseRate"] = float(b.reindex(r.index).mean()) if len(r) else None
    return st


# ====================================================================== walk-forward


def purged_walk_forward(pre_returns, eligible, train_min=None, test_len=None, purge=PURGE):
    train_min = train_min or sw.WF_TRAIN_MIN
    test_len = test_len or sw.WF_TEST_LEN
    n = len(pre_returns)
    folds = []
    start = train_min
    while start + purge < n:
        test0 = start + purge
        test1 = min(test0 + test_len, n)
        if test1 - test0 < 10:
            break
        train = pre_returns.iloc[:start]
        test = pre_returns.iloc[test0:test1]
        ranked, scores = sw.select(train, eligible)
        if not ranked:
            start = test1
            continue
        test_sr = {int(c): sw.sharpe(test[c]) for c in test.columns if c in eligible}
        ids = [k for k in ranked if np.isfinite(test_sr.get(k, np.nan))]
        rank_corr = None
        if len(ids) > 3:
            rank_corr = float(pd.Series([scores[k] for k in ids]).rank().corr(
                pd.Series([test_sr[k] for k in ids]).rank()))
        folds.append({"trainStart": str(pre_returns.index[0].date()),
                      "trainEnd": str(pre_returns.index[start - 1].date()),
                      "purgeSessions": purge,
                      "testStart": str(pre_returns.index[test0].date()),
                      "testEnd": str(pre_returns.index[test1 - 1].date()),
                      "chosen": ranked[0], "chosenTrainSharpe": scores[ranked[0]],
                      "chosenTestSharpe": test_sr.get(ranked[0]),
                      "medianTestSharpe": float(np.nanmedian(list(test_sr.values()))) if test_sr else None,
                      "trainTestRankCorr": rank_corr, "rows": (test0, test1)})
        start = test1
    return folds


# ====================================================================== driver


def _date_slice(panel, start, end):
    from market_data import Panel
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    cut = lambda f: f.loc[start:end]
    return Panel(cut(panel.open), cut(panel.high), cut(panel.low), cut(panel.close), cut(panel.volume),
                 cut(panel.dividends), panel.classes,
                 None if panel.masked_returns is None else cut(panel.masked_returns),
                 warnings=list(panel.warnings or []), overlap=getattr(panel, "overlap", None),
                 used_long_history=getattr(panel, "used_long_history", False),
                 sources=getattr(panel, "sources", None))


def _sim_panel(panel, sim_start, data_end, warmup=200):
    """Keep 200 sessions of warmup before simStart, cut after dataEnd. Avoid walking 20-year history."""
    dates = panel.dates
    i0 = int(dates.searchsorted(pd.Timestamp(sim_start)))
    i1 = int(dates.searchsorted(pd.Timestamp(data_end), side="right"))
    start = dates[max(0, i0 - warmup)]
    end = dates[min(len(dates), i1) - 1]
    if start == dates[0] and end == dates[-1]:
        return panel
    return _date_slice(panel, start, end)


def run(workers=None, out_path=None, mutate_swarm=True, load_cached=True):
    started = time.time()
    design = json.loads(sw.DESIGN_PATH.read_text())
    if load_cached:
        full_panel, _ = sw.load_panel_cached()
    else:
        # Do not touch the swarm's shared panel cache (a 20-year load would replace it).
        from market_data import load_panel
        sw.log("loading price panel directly (use_long_history, no shared cache write)")
        full_panel = load_panel()
    fingerprint = sw.data_fingerprint()
    sim_start, eval_start = pd.Timestamp(design["simStart"]), pd.Timestamp(design["evalStart"])
    holdout_start, data_end = pd.Timestamp(design["holdoutStart"]), pd.Timestamp(design["dataEnd"])
    sw.log("exhaustion event study (pre-holdout only, using full history if daily_long is present)")
    t_ev = time.time()
    event = run_event_study(full_panel, holdout_start, eval_start)
    event_seconds = time.time() - t_ev
    sw.log(f"event study done in {event_seconds:.1f}s; {event['coverage']['tests']} tests, "
           f"{event['nSignificantBH_5d']} BH-significant 5-day reversals")

    panel = _sim_panel(full_panel, sim_start, data_end)
    sw.log(f"sim panel {panel.dates[0].date()} to {panel.dates[-1].date()} ({len(panel.dates)} sessions)")
    import pickle, os
    panel_path = sw.CACHE_DIR / f"panel_exh_{fingerprint}_{panel.dates[0].date()}_{panel.dates[-1].date()}.pkl"
    if not panel_path.exists():
        sw.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = panel_path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(panel, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, panel_path)

    grid = variants()
    accounts = grid + REFERENCE
    tasks = [{"account": a, "start": str(sim_start.date()),
              "cacheKey": sw._account_key(a, "swing", fingerprint, str(data_end.date()) + ":exh2")}
             for a in accounts]
    workers = workers or sw.default_workers()
    sw.log(f"exhaustion sims: {len(grid)} variants + {len(REFERENCE)} references on {workers} workers")
    t0 = time.time()
    results = {r["id"]: r for r in sw.run_tasks(sw.run_swing_account, tasks, workers, panel_path, None)}
    sim_seconds = time.time() - t0
    failed = {i: r["error"] for i, r in results.items() if not r.get("ok")}
    ok = [i for i, r in results.items() if r.get("ok")]
    rets = pd.DataFrame({i: results[i]["returns"] for i in ok}).loc[eval_start:data_end]
    pre_mask, hold_mask = rets.index < holdout_start, rets.index >= holdout_start
    etf_ids = [a["id"] for a in grid if a["id"] in rets.columns and 300 <= a["id"] <= 318]
    stock_ids = [a["id"] for a in grid if a["id"] in rets.columns and 320 <= a["id"] <= 323]
    fade_ids = [a["id"] for a in grid if a["id"] in rets.columns and 340 <= a["id"] <= 349]
    eligible_etf = {i for i in etf_ids if not results[i]["pdt"]["breach"]}
    eligible_stock = {i for i in stock_ids if i != 320 and not results[i]["pdt"]["breach"]}
    eligible_fade = {i for i in fade_ids if not results[i]["pdt"]["breach"]}
    spy = sw._bench_spy(panel, sim_start, data_end).reindex(rets.index).fillna(0.0)
    tsmom = rets[1] if 1 in rets.columns else None
    stock_base = rets[320] if 320 in rets.columns else None
    benches_etf = {"tsmom_baseline": tsmom, "spy": spy}
    stats = sw._leg_stats(rets[etf_ids], pre_mask, hold_mask, eligible_etf, {}, benches_etf, "exhaustion_etf")
    folds = purged_walk_forward(rets[etf_ids].loc[pre_mask], eligible_etf)
    stitched = sw._stitched(panel, {i: results[i] for i in etf_ids}, folds, sim_start,
                            holdout_start - pd.Timedelta(1, unit="D"))
    for f in folds:
        f.pop("rows", None)
    stats["walkForward"] = {"folds": folds, "purgeSessions": PURGE,
                            "stitchedOos": sw.perf(stitched) if stitched is not None else None}
    stats_stock = (sw._leg_stats(rets[stock_ids], pre_mask, hold_mask, eligible_stock, {},
                                 {"tsmom_stocks": stock_base, "spy": spy, "tsmom_etf": tsmom},
                                 "exhaustion_stocks") if eligible_stock else {})
    stats_fade = (sw._leg_stats(rets[fade_ids], pre_mask, hold_mask, eligible_fade, {},
                                {"tsmom_baseline": tsmom, "spy": spy}, "fade") if eligible_fade else {})

    swarm_csv = sw.SWARM_DIR / "returns_swing.csv"
    prior_sr = []
    if swarm_csv.exists():
        prev = pd.read_csv(swarm_csv, index_col=0, parse_dates=True)
        prior_sr = [sw.sharpe(prev[c].loc[prev.index < holdout_start]) for c in prev.columns]
    pre = rets.loc[pre_mask]
    growth_path = sw.SWARM_DIR / "growth_mr_study.json"
    extra_prior = 0
    if growth_path.is_file():
        try:
            extra_prior = int((json.loads(growth_path.read_text()) or {}).get("variants") or 0)
        except Exception:
            extra_prior = 0
    etf_sr = [sw.sharpe(pre[i]) for i in etf_ids if i in eligible_etf]
    n_trials = len(etf_sr) + len(prior_sr) + extra_prior
    if stats.get("champion") is not None:
        stats["deflatedSharpeCumulative"] = sw.deflated_sharpe(pre[stats["champion"]].to_numpy(),
                                                               max(2, n_trials),
                                                               np.array(etf_sr + prior_sr))
        stats["cumulativeTrials"] = n_trials
        stats["priorSwarmTrials"] = len(prior_sr)
        stats["priorGrowthTrials"] = extra_prior

    hold = rets.loc[hold_mask]
    rows, rank = [], {k: n + 1 for n, k in enumerate(stats.get("ranked", []))}
    tests = stats.get("tests", {})
    by_account = {a["id"]: a for a in accounts}
    for a in accounts:
        i = a["id"]
        r = results.get(i, {})
        if not r.get("ok"):
            continue
        at = r.get("afterTaxEquity")
        group = ("etf_overlay" if 300 <= i <= 318 else "stocks" if 320 <= i <= 329
                 else "fade" if 340 <= i <= 349 else "reference")
        row = {"id": i, "label": a["label"], "group": group, "grid": a.get("grid"), "swing": a["swing"],
               "selectionRank": rank.get(i), "eligible": (i in eligible_etf or i in eligible_stock
                                                          or i in eligible_fade) if i != 1 else None,
               "pdt": r["pdt"], "trades": r["trades"], "turnoverAnnual": r["turnover"],
               "pre": sw.perf(pre[i], at), "holdout": sw.perf(hold[i], at)}
        if group == "etf_overlay":
            for b in ("tsmom_baseline", "spy"):
                t = tests.get(f"{b}:holdout", {})
                row[f"holdoutVs_{b}"] = {"p": t.get("rawP", {}).get(str(i)),
                                         "holm": t.get("holm", {}).get(str(i)),
                                         "bh": t.get("bh", {}).get(str(i))}
        if i != 1 and 1 in results and results[1].get("ok") and r.get("weights") is not None:
            try:
                row["entryDiagnostics"] = ex.entry_diagnostics(
                    panel, results[1]["weights"], r["weights"], start=eval_start, end=data_end)
            except Exception as error:  # noqa: BLE001
                row["entryDiagnostics"] = {"error": f"{type(error).__name__}: {error}"}
        rows.append(row)

    study_hash = sw.stable_hash({"space": SPACE, "fixed": FIXED, "stock": STOCK_FIXED, "fade": FADE,
                                 "lhc": LHC_N, "seed": LHC_SEED, "purge": PURGE, "code": "exhaustion-1"})
    ledger = sw._record_holdout_view(design, stats, study="exhaustion", design_hash="exhaustion-" + study_hash)
    verdict = _verdict(stats, stats_stock, stats_fade, event, rows, by_account)
    refs = {}
    by_id = {r["id"]: r for r in rows}
    if 1 in by_id:
        refs["tsmom_baseline"] = {"pre": by_id[1]["pre"], "holdout": by_id[1]["holdout"],
                                  "trades": by_id[1]["trades"], "turnoverAnnual": by_id[1]["turnoverAnnual"]}
    if 320 in by_id:
        refs["tsmom_stocks"] = {"pre": by_id[320]["pre"], "holdout": by_id[320]["holdout"],
                                "trades": by_id[320]["trades"]}
    refs["spy"] = {"pre": sw.perf(spy.loc[pre_mask]), "holdout": sw.perf(spy.loc[hold_mask])}
    extras = _add_forward_extras(design, stats, grid) if mutate_swarm else []
    # compact event cells for the json (keep full horizons)
    doc = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "studyHash": study_hash,
           "space": SPACE, "latinHypercube": {"n": LHC_N, "seed": LHC_SEED},
           "variants": {"etfOverlay": len(etf_ids), "stocks": len(stock_ids), "fade": len(fade_ids),
                        "totalTrialsCounted": len(etf_ids) + len(stock_ids) + len(fade_ids)},
           "window": {k: design[k] for k in ("evalStart", "holdoutStart", "dataEnd")},
           "eventStudy": event, "eventStudyHighlights": event["highlights"],
           "stats": sw._strip(stats), "statsStocks": sw._strip(stats_stock) if stats_stock else {},
           "statsFade": sw._strip(stats_fade) if stats_fade else {},
           "references": refs, "leaderboard": rows,
           "holdoutLedger": {"views": ledger["views"],
                             "note": "this study is an additional look at the swarm's sealed holdout"},
           "forwardExtras": extras, "verdict": verdict, "failed": failed,
           "runtime": {"eventSeconds": round(event_seconds, 1), "simSeconds": round(sim_seconds, 1),
                       "totalSeconds": round(time.time() - started, 1), "workers": workers},
           "caveats": [
               (f"Daily history used here is {full_panel.dates[0].date()} to {full_panel.dates[-1].date()} "
                f"({len(full_panel.dates)} sessions, {full_panel.close.shape[1]} tickers"
                f"{'; Schwab daily_long spliced in' if getattr(full_panel, 'used_long_history', False) else ''})."),
               "Survivorship bias: delisted names are missing from Schwab and Massive history, so stock-book "
               "results lean optimistic.",
               "Schwab pre-Massive returns are split-adjusted but not dividend-adjusted; overlapping Massive "
               "dates are split- and dividend-adjusted.",
               "TSMOM itself trades a small ETF book, so exhaustion (a single-stock effect) may barely show up "
               "there; the stock-momentum book is the fairer test of the idea.",
               "The fade sleeve is simulated without the tax lot ledger because short sales cannot be opened "
               "against existing long lots in that ledger.",
           ]}
    dest = Path(out_path) if out_path else OUT
    if not mutate_swarm:
        doc["swarmMutated"] = False
    sw.write_json(dest, doc)
    if mutate_swarm:
        _update_swarm_status(doc)
    sw.log(f"exhaustion study done in {doc['runtime']['totalSeconds']}s -> {dest}")
    return doc


def _verdict(stats, stats_stock, stats_fade, event, rows, by_account):
    c = stats.get("champion")
    t = stats.get("tests", {})
    tb_h, tb_p = t.get("tsmom_baseline:holdout", {}), t.get("tsmom_baseline:preHoldout", {})
    dsr = (stats.get("deflatedSharpeCumulative") or stats.get("deflatedSharpePre") or {}).get(
        "deflatedSharpeProbability")
    pbo = (stats.get("pboPre") or {}).get("pbo")
    gates = {"championBeatsTsmomOnHoldout(p<0.05)": (tb_h.get("championP") or 1) < 0.05,
             "spaVsTsmomPreHoldout(p<0.05)": (tb_p.get("spaP") or 1) < 0.05,
             "spaVsTsmomHoldout(p<0.05)": (tb_h.get("spaP") or 1) < 0.05,
             "deflatedSharpeCumulative>0.95": (dsr or 0) > 0.95,
             "pbo<0.5": pbo is not None and pbo < 0.5}
    passed = all(gates.values())
    n_sig = event.get("nSignificantBH_5d") or 0
    rec = (
        "Add the exhaustion timing overlay to the paper account (recommendation only; Robert decides)."
        if passed else
        "Do not change the paper account. The overlay does not beat plain trend-following after "
        "multiple-testing correction on this sample. It keeps running in the swarm's forward shadow "
        "mode so it can build a live record."
    )
    if n_sig == 0:
        rec += (" The event study also did not find a 5-day reversal that survives Benjamini-Hochberg "
                "correction with at least 30 event dates.")
    else:
        rec += (f" The event study did flag {n_sig} cell(s) with a BH-significant 5-day reversal; "
                "see eventStudyHighlights — that is evidence about the phenomenon, not a license to "
                "trade the best-looking overlay.")
    champ = next((r for r in rows if r["id"] == c), None)
    stock_c = stats_stock.get("champion")
    fade_c = stats_fade.get("champion")
    return {"champion": c, "championLabel": None if not champ else champ["label"],
            "stockChampion": stock_c, "fadeChampion": fade_c,
            "gates": gates, "passedAll": passed, "recommendation": rec,
            "eventStudySignificantBH_5d": n_sig}


def _add_forward_extras(design, stats, grid):
    by_id = {a["id"]: a for a in grid}
    picks = [i for i in (stats.get("ranked") or [])[:FORWARD_EXTRAS] if i in by_id]
    extras_new = []
    for n, i in enumerate(picks):
        a = by_id[i]
        extras_new.append({"id": FORWARD_START_ID + n, "label": f"exh_pre_rank{n + 1}:{a['label']}",
                           "swing": a["swing"], "orb": None})
    # Always keep a theoretically motivated default if it isn't already in the top ranks
    default = next((a for a in grid if a["id"] == 317), None)
    if default and default["id"] not in picks and len(extras_new) < FORWARD_EXTRAS + 1:
        extras_new.append({"id": FORWARD_START_ID + len(extras_new),
                           "label": "exh_default_reentry_strict:" + default["label"],
                           "swing": default["swing"], "orb": None})
    bak = sw.DESIGN_PATH.with_name(f"design.json.bak-{datetime.now():%Y%m%d-%H%M}")
    shutil.copy2(sw.DESIGN_PATH, bak)
    current = json.loads(sw.DESIGN_PATH.read_text())
    existing = list(current.get("forwardExtras") or [])
    # Drop previous exhaustion extras by label prefix; keep growth, probability, and anything else
    kept = [e for e in existing if not str(e.get("label", "")).startswith("exh_")]
    # Avoid id collisions with whatever another study may have added
    used = {int(e["id"]) for e in kept if e.get("id") is not None}
    next_id = max([FORWARD_START_ID - 1, *used]) + 1 if used else FORWARD_START_ID
    assigned = []
    for e in extras_new:
        eid = int(e["id"])
        if eid in used:
            eid = next_id
            next_id += 1
        e = dict(e)
        e["id"] = eid
        used.add(eid)
        assigned.append(e)
        next_id = max(next_id, eid + 1)
    current["forwardExtras"] = kept + assigned
    note = current.get("forwardExtrasNote") or ""
    add = ("Exhaustion overlay extras added 2026-11-25 (top pre-holdout ranks + a strict re-entry default). "
           "Shadow forward records only; not part of the frozen design hash.")
    current["forwardExtrasNote"] = (note + " " + add).strip() if "Exhaustion overlay extras" not in note else note
    sw.write_json(sw.DESIGN_PATH, current)
    return [{"id": e["id"], "label": e["label"]} for e in assigned]


def _update_swarm_status(doc):
    if not sw.STATUS_PATH.exists():
        return
    bak = sw.STATUS_PATH.with_name(f"swarm_status.json.bak-{datetime.now():%Y%m%d-%H%M}")
    shutil.copy2(sw.STATUS_PATH, bak)
    status = json.loads(sw.STATUS_PATH.read_text())
    st, refs, v = doc["stats"], doc["references"], doc["verdict"]
    champ = next((r for r in doc["leaderboard"] if r["id"] == st.get("champion")), {})
    t = st.get("tests", {})
    tsmom_h = (refs.get("tsmom_baseline") or {}).get("holdout") or {}
    spy_h = (refs.get("spy") or {}).get("holdout") or {}
    summary = (
        f"{doc['variants']['totalTrialsCounted']} variants; ETF champion {champ.get('label')} "
        f"pre Sharpe {sw._f(champ.get('pre', {}).get('sharpe'))}, holdout Sharpe "
        f"{sw._f(champ.get('holdout', {}).get('sharpe'))} vs TSMOM {sw._f(tsmom_h.get('sharpe'))}, "
        f"SPY {sw._f(spy_h.get('sharpe'))}; PBO {sw._f((st.get('pboPre') or {}).get('pbo'))}; "
        f"cumulative DSR prob {sw._f((st.get('deflatedSharpeCumulative') or {}).get('deflatedSharpeProbability'))}; "
        f"SPA p vs TSMOM holdout {sw._f(t.get('tsmom_baseline:holdout', {}).get('spaP'))}; "
        f"event-study BH-significant 5d reversals: {doc['eventStudy']['nSignificantBH_5d']}. "
        f"{v['recommendation']}"
    )
    status["exhaustionStudy"] = {"summary": summary, "passedAll": v["passedAll"],
                                 "generatedAt": doc["generatedAt"],
                                 "holdoutViews": doc["holdoutLedger"]["views"],
                                 "forwardExtras": doc["forwardExtras"]}
    status["holdoutViews"] = doc["holdoutLedger"]["views"]
    if hasattr(sw, "prompt_summary"):
        try:
            status["promptSummary"] = sw.prompt_summary(status)
        except Exception:
            pass
    sw.write_json(sw.STATUS_PATH, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--event-only", action="store_true", help="run the event study only (no portfolio sims)")
    parser.add_argument("--out", default=None, help="Write the study JSON here instead of exhaustion_study.json")
    parser.add_argument("--no-mutate-swarm", action="store_true",
                        help="Do not rewrite design.json forwardExtras or swarm status.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Load the panel with load_panel() and do not replace the swarm panel cache.")
    args = parser.parse_args(argv)
    if args.event_only:
        design = json.loads(sw.DESIGN_PATH.read_text())
        if args.no_cache:
            from market_data import load_panel
            panel = load_panel()
        else:
            panel, _ = sw.load_panel_cached()
        event = run_event_study(panel, design["holdoutStart"], design["evalStart"])
        print(json.dumps({"highlights": event["highlights"], "coverage": event["coverage"]},
                         default=sw._json_default))
        return 0
    doc = run(args.workers, out_path=args.out, mutate_swarm=not args.no_mutate_swarm,
              load_cached=not args.no_cache)
    print(json.dumps({"verdict": doc["verdict"], "runtime": doc["runtime"],
                      "highlights": doc["eventStudyHighlights"][:8]}, default=sw._json_default))
    return 0


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("exhaustion_study"):
        sys.exit(main() or 0)
