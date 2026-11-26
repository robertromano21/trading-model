"""Walk-forward training and after-cost evaluation of the probabilistic models.

Expanding train, purged and embargoed (embargo >= k sessions) predictions on the next unseen
window, refit every 21 sessions. A threshold and a trading rule are chosen on pre-holdout
out-of-sample data only. The frozen specification is evaluated once on the swarm's sealed
holdout and the look is logged in holdout_ledger.json.

The objective is after-cost (and after-tax, when the tax module is on) out-of-sample
risk-adjusted return under a drawdown constraint, not in-sample profit. Nothing here changes
the live paper account; winners can be added to swarm forwardExtras as shadow accounts.

  python research.py prob
  python prob_walkforward.py [--workers N] [--skip-holdout]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import prob_model as pm
import swarm as sw
from costs import CostModel
from fetch_and_model import DATA_DIR
from market_data import MARKET, load_panel
from portfolio_sim import simulate
from risk import CircuitBreaker, TRADING_DAYS
from strategies.tsmom import TimeSeriesMomentum
from universe import CORE_ETFS

REFIT_EVERY = 21
MIN_TRAIN = 126
INNER_VAL = 42
THRESHOLDS = (0.55, 0.60, 0.65)
KELLY_FRACS = (0.5,)
MODEL_KINDS = ("logit", "ensemble")
TARGET_SPECS = (("excess", 20, 0.0), ("excess", 5, 0.0), ("ret", 20, 0.0))
SIZINGS = ("tsmom_filter", "prob_sized")
DD_MULT = 1.25
DD_FLOOR = 0.20
FORWARD_EXTRAS = 4
INITIAL_EQUITY = 10_000.0
STUDY_CODE = "prob-wf-1"


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", file=sys.stderr, flush=True)


def _wanted_tickers(extra=None):
    names = list(CORE_ETFS) + ["NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "AVGO", "AMD"]
    if extra:
        names.extend(extra)
    try:
        import dilution as dil
        names.extend(dil.watchlist_tickers())
    except Exception:  # noqa: BLE001
        pass
    seen, out = set(), []
    for t in names:
        t = str(t).upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _universe(panel, extra=None):
    names = _wanted_tickers(extra)
    out = [t for t in names if t in panel.close.columns]
    if MARKET not in out and MARKET in panel.close.columns:
        out.insert(0, MARKET)
    return out


def _trade_universe(panel):
    return [t for t in CORE_ETFS if t in panel.close.columns]


def _load_panel():
    panel = load_panel(tickers=_wanted_tickers(), use_long_history=True)
    warnings = list(panel.warnings or [])
    if getattr(panel, "used_long_history", False):
        warnings.append("Schwab daily_long history is in use. Delisted names are still missing "
                        "(survivorship bias): results on the long sample lean optimistic.")
    else:
        warnings.append("Schwab daily_long table is not populated; the study uses the Massive "
                        "window only (about two years).")
    return panel, warnings


def walk_forward_probs(panel, features, tickers, kind, k, threshold=0.0, model_kind="logit",
                       min_train=MIN_TRAIN, refit_every=REFIT_EVERY, holdout_start=None,
                       cal_method="auto"):
    """Purged, embargoed expanding-window predictions. Returns a long frame and per-fold metrics.

    When holdout_start is set, folds whose test window meets or passes that date are not used
    to update the model or the threshold — the frozen pack from the last pre-holdout refit is
    applied instead (so the holdout evaluation does not refit on holdout rows).
    """
    k = int(k)
    embargo = k
    targets = pm.build_targets(panel, k=k, threshold=threshold)
    stacked = pm.assemble(features, targets, tickers, kind=kind, k=k, threshold=threshold)
    if stacked.empty:
        return pd.DataFrame(), [], None
    dates = panel.dates
    usable = stacked.dropna(subset=["y"])
    # First date with a complete feature row and enough history for the 100-day signal.
    feature_ok = usable.dropna(thresh=max(6, usable.shape[1] // 3))
    all_days = dates
    # sessions that have at least one usable sample
    sample_days = feature_ok.index.get_level_values("date").unique().sort_values()
    if len(sample_days) < min_train + refit_every:
        return pd.DataFrame(), [], None
    start_i = min_train
    packs_by_fold = []
    fold_rows = []
    pred_parts = []
    last_pack = None
    frozen_for_holdout = None
    holdout_start = pd.Timestamp(holdout_start) if holdout_start is not None else None
    fold_id = 0
    test_starts = list(range(start_i, len(sample_days), refit_every))
    feature_names = [c for c in feature_ok.columns if c not in ("y", "forward")]
    for ts in test_starts:
        test_day = sample_days[ts]
        test_end = sample_days[min(ts + refit_every, len(sample_days)) - 1]
        in_holdout = holdout_start is not None and test_day >= holdout_start
        if in_holdout:
            if frozen_for_holdout is None:
                # First holdout window: fit once on pre-holdout rows only, then freeze.
                train_end = pm.session_shift(all_days, holdout_start, -embargo)
                keep = pm.purge_mask(all_days, train_end, k, embargo)
                train = feature_ok[keep.reindex(feature_ok.index.get_level_values("date")).to_numpy()]
                if len(train) < 80:
                    continue
                train_days = train.index.get_level_values("date").unique().sort_values()
                if len(train_days) > INNER_VAL + 40:
                    val_cut = train_days[-INNER_VAL]
                    fit_keep = pm.purge_mask(all_days, pm.session_shift(all_days, val_cut, -embargo), k, embargo)
                    fit = train[fit_keep.reindex(train.index.get_level_values("date")).to_numpy()]
                    val = train[train.index.get_level_values("date") >= val_cut]
                else:
                    fit, val = train, train.iloc[0:0]
                X = fit[feature_names].to_numpy(dtype=float)
                y = fit["y"].to_numpy(dtype=float)
                val_X = val[feature_names].to_numpy(dtype=float) if len(val) else None
                val_y = val["y"].to_numpy(dtype=float) if len(val) else None
                frozen_for_holdout = pm.fit_pack(
                    X, y, feature_names, kind=model_kind, val_X=val_X, val_y=val_y,
                    target={"kind": kind, "k": k, "threshold": threshold,
                            "name": pm.target_key(kind, k, threshold)},
                    cal_method=cal_method)
            pack = frozen_for_holdout
        else:
            train_end = pm.session_shift(all_days, test_day, -embargo)
            keep = pm.purge_mask(all_days, train_end, k, embargo)
            train = feature_ok[keep.reindex(feature_ok.index.get_level_values("date")).to_numpy()]
            if len(train) < 80:
                continue
            train_days = train.index.get_level_values("date").unique().sort_values()
            if len(train_days) > INNER_VAL + 40:
                val_cut = train_days[-INNER_VAL]
                # Purge val from the fit set too (labels in the last INNER_VAL days stay out of the fit).
                fit = train[train.index.get_level_values("date") < val_cut]
                # also drop fit samples whose label window overlaps val
                fit_keep = pm.purge_mask(all_days, pm.session_shift(all_days, val_cut, -embargo), k, embargo)
                fit = train[fit_keep.reindex(train.index.get_level_values("date")).to_numpy()]
                val = train[train.index.get_level_values("date") >= val_cut]
            else:
                fit, val = train, train.iloc[0:0]
            if len(fit) < 80:
                if last_pack is None and frozen_for_holdout is None:
                    continue
                pack = frozen_for_holdout or last_pack
            else:
                X = fit[feature_names].to_numpy(dtype=float)
                y = fit["y"].to_numpy(dtype=float)
                val_X = val[feature_names].to_numpy(dtype=float) if len(val) else None
                val_y = val["y"].to_numpy(dtype=float) if len(val) else None
                pack = pm.fit_pack(X, y, feature_names, kind=model_kind, val_X=val_X, val_y=val_y,
                                   target={"kind": kind, "k": k, "threshold": threshold, "name": pm.target_key(kind, k, threshold)},
                                   cal_method=cal_method)
                last_pack = pack
            if holdout_start is None or test_end < holdout_start:
                frozen_for_holdout = pack
        # Predict on the unseen test window (no rows from train_end..test_day-1 leak into X, they
        # are simply not scored; that is the embargo gap).
        sl = feature_ok[(feature_ok.index.get_level_values("date") >= test_day)
                        & (feature_ok.index.get_level_values("date") <= test_end)]
        if not len(sl):
            continue
        p = pm.predict_pack(pack, sl[feature_names])
        part = sl[["y", "forward"]].copy()
        part["p"] = p
        part["fold"] = fold_id
        part["holdout"] = bool(in_holdout)
        pred_parts.append(part)
        report = pm.classification_report(part["y"].to_numpy(), part["p"].to_numpy(), part["forward"].to_numpy())
        report.update({"fold": fold_id, "testStart": str(pd.Timestamp(test_day).date()),
                       "testEnd": str(pd.Timestamp(test_end).date()), "holdout": bool(in_holdout),
                       "nTrain": pack["model"].get("n") if isinstance(pack.get("model"), dict) else None})
        fold_rows.append(report)
        packs_by_fold.append({"fold": fold_id, "pack": pack, "holdout": bool(in_holdout)})
        fold_id += 1
    if not pred_parts:
        return pd.DataFrame(), fold_rows, frozen_for_holdout or last_pack
    pred = pd.concat(pred_parts).sort_index()
    return pred, fold_rows, frozen_for_holdout or last_pack


def _pivot_p(pred):
    if pred is None or not len(pred):
        return pd.DataFrame()
    s = pred["p"].copy()
    return s.unstack("ticker")


def _simulate(panel, weights, start, end=None, use_tax=True):
    tcfg = None
    if use_tax and sw.tax_available():
        tcfg = sw.tax_config(False)  # track after-tax equity; no tax-aware trade changes
    result = simulate(panel, weights, None, CostModel.for_broker("schwab"), INITIAL_EQUITY, "next_open",
                      start=start, end=end, breaker=CircuitBreaker(), min_trade_weight=0.02,
                      whole_shares=True, min_order_value=getattr(sw, "MIN_ORDER_VALUE", 50.0),
                      tax_config=tcfg)
    return result


def tsmom_weights(panel):
    return TimeSeriesMomentum().target_weights(panel)


def spy_weights(panel):
    w = pd.DataFrame(0.0, index=panel.dates, columns=panel.close.columns)
    if MARKET in w.columns:
        w[MARKET] = 1.0
    return w


def filter_weights(base, proba, threshold):
    aligned = proba.reindex(index=base.index, columns=base.columns)
    return base.where(aligned >= float(threshold), 0.0)


def sized_weights(panel, proba, threshold, kelly_frac=0.5, target_vol=0.10, max_positions=8,
                  per_name_cap=0.30, vol_window=60, rebalance="W-FRI"):
    from risk import apply_caps, blended_vol, hold_between_rebalances, keep_top, portfolio_vol_target
    cols = [c for c in proba.columns if c in panel.close.columns]
    p = proba.reindex(index=panel.dates, columns=cols)
    edge = (2.0 * p - 1.0).clip(lower=0.0).where(p >= float(threshold), 0.0)
    rets = panel.total_return()[cols]
    vol = blended_vol(rets, vol_window)
    raw = (float(kelly_frac) * edge / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raw = keep_top(raw, p.fillna(-np.inf), max_positions)
    scaled = portfolio_vol_target(raw, rets, target_vol, 1.0, window=vol_window)
    weights = apply_caps(scaled, per_name_cap, 1.0)
    return hold_between_rebalances(weights, rebalance).reindex(columns=panel.close.columns, fill_value=0.0)


def _returns_of(result, start, end=None):
    r = result.returns
    r = r.loc[r.index >= pd.Timestamp(start)]
    if end is not None:
        r = r.loc[r.index <= pd.Timestamp(end)]
    return r


def _dd(returns):
    eq = (1.0 + returns.fillna(0.0)).cumprod()
    if not len(eq):
        return None
    return float((eq / eq.cummax() - 1.0).min())


def trial_label(spec):
    return (f"prob_{spec['model']}_{spec['target']}_p{spec['threshold']}_{spec['sizing']}")


def evaluate_trials(panel, tapes, tsmom_w, start, end, holdout_start):
    """Simulate every (model, target, threshold, sizing) trial. Returns a returns DataFrame and rows."""
    results = {}
    rows = []
    # References first so they always exist for SPA.
    refs = {
        1: ("tsmom_baseline", tsmom_w),
        2: ("spy", spy_weights(panel)),
    }
    next_id = 10
    trials = []
    for key, pred in tapes.items():
        proba = _pivot_p(pred)
        if proba.empty:
            continue
        model_kind, kind, k, thresh_ret = key
        target_name = pm.target_key(kind, k, thresh_ret)
        for pthr in THRESHOLDS:
            for sizing in SIZINGS:
                spec = {"id": next_id, "model": model_kind, "target": target_name,
                        "kind": kind, "k": k, "retThreshold": thresh_ret,
                        "threshold": pthr, "sizing": sizing, "kellyFrac": KELLY_FRACS[0]}
                trials.append((spec, proba))
                next_id += 1
    # Simulate references + trials
    all_items = list(refs.items()) + [(spec["id"], (trial_label(spec), spec, proba)) for spec, proba in trials]
    # rebuild a consistent structure
    sim_map = {}
    for i, (name, weights) in refs.items():
        result = _simulate(panel, weights, start, end)
        sim_map[i] = {"id": i, "label": name, "spec": {"family": name}, "result": result, "reference": True}
    for spec, proba in trials:
        if spec["sizing"] == "tsmom_filter":
            weights = filter_weights(tsmom_w, proba, spec["threshold"])
        else:
            weights = sized_weights(panel, proba, spec["threshold"], kelly_frac=spec["kellyFrac"])
        result = _simulate(panel, weights, start, end)
        sim_map[spec["id"]] = {"id": spec["id"], "label": trial_label(spec), "spec": spec,
                               "result": result, "reference": False}

    holdout_start = pd.Timestamp(holdout_start)
    rets = pd.DataFrame({i: rec["result"].returns for i, rec in sim_map.items()})
    rets = rets.loc[pd.Timestamp(start):pd.Timestamp(end) if end is not None else rets.index[-1]]
    pre_mask = rets.index < holdout_start
    hold_mask = rets.index >= holdout_start
    tsmom_pre = rets[1].loc[pre_mask] if 1 in rets else None
    tsmom_dd = _dd(tsmom_pre) if tsmom_pre is not None else -DD_FLOOR
    dd_cap = min(tsmom_dd * DD_MULT if tsmom_dd is not None else -DD_FLOOR, -DD_FLOOR) if (tsmom_dd or 0) < 0 else -DD_FLOOR
    # tsmom_dd is negative; cap is more negative * 1.25 or -0.20, whichever is larger in magnitude? 
    # Constraint: trial max DD must not be worse than max(|TSMOM DD| * 1.25, 0.20).
    tsmom_dd_mag = abs(tsmom_dd or 0.0)
    cap_mag = max(tsmom_dd_mag * DD_MULT, DD_FLOOR)

    for i, rec in sim_map.items():
        r = rets[i]
        pre, hold = r.loc[pre_mask], r.loc[hold_mask]
        at = rec["result"].after_tax_equity if hasattr(rec["result"], "after_tax_equity") else rec["result"].__dict__.get("afterTaxEquity")
        # SimResult field names
        after_tax = getattr(rec["result"], "after_tax_equity", None)
        if after_tax is None:
            after_tax = getattr(rec["result"], "afterTaxEquity", None)
        pre_dd = _dd(pre)
        row = {"id": i, "label": rec["label"], "spec": rec["spec"], "reference": rec["reference"],
               "pre": sw.perf(pre, after_tax), "holdout": sw.perf(hold, after_tax),
               "preMaxDrawdown": pre_dd, "holdoutMaxDrawdown": _dd(hold),
               "passesDd": (abs(pre_dd) <= cap_mag + 1e-12) if pre_dd is not None else False,
               "trades": int(getattr(rec["result"], "trade_count", None)
                             or (len(rec["result"].trades) if getattr(rec["result"], "trades", None) is not None else 0))}
        # after-tax Sharpe on pre-holdout when available
        if after_tax is not None and hasattr(after_tax, "index") and len(after_tax.index):
            try:
                at_ret = after_tax.pct_change().reindex(pre.index).fillna(0.0)
                row["preAfterTax"] = sw.perf(at_ret)
            except Exception:  # noqa: BLE001
                row["preAfterTax"] = None
        rows.append(row)
    return rets, rows, sim_map, cap_mag


def select_champion(rows):
    """Pick on pre-holdout after-cost Sharpe, among trials that pass the drawdown gate.

    References are never the champion of this study (the question is whether a probability
    rule beats TSMOM). If nobody passes the DD gate, fall back to the best Sharpe and mark it.
    """
    trials = [r for r in rows if not r.get("reference")]
    eligible = [r for r in trials if r.get("passesDd") and (r.get("pre") or {}).get("sharpe") is not None]
    pool = eligible or [r for r in trials if (r.get("pre") or {}).get("sharpe") is not None]
    if not pool:
        return None, False, []
    ranked = sorted(pool, key=lambda r: (r["pre"].get("sharpe") is not None, r["pre"].get("sharpe") or -999),
                    reverse=True)
    return ranked[0]["id"], bool(eligible), [r["id"] for r in ranked]


def _fold_table(folds):
    if not folds:
        return {"folds": [], "mean": {}}
    keys = ("brier", "logLoss", "auc", "ic", "n")
    mean = {}
    for k in keys:
        vals = [f[k] for f in folds if f.get(k) is not None]
        mean[k] = float(np.mean(vals)) if vals else None
    return {"folds": folds, "mean": mean, "nFolds": len(folds)}


def _add_forward_extras(design, champion_row, ranked_rows, by_id):
    """Append probability-rule shadows without dropping existing (e.g. growth-MR) extras."""
    extras = []
    seen_ids = {e["id"] for e in (design.get("forwardExtras") or [])}
    start_id = 110
    picks = []
    if champion_row is not None:
        picks.append(champion_row)
    for rid in ranked_rows:
        rec = by_id.get(rid)
        if rec is None or rec.get("reference"):
            continue
        if champion_row is not None and rec["id"] == champion_row["id"]:
            continue
        picks.append(rec)
        if len(picks) >= FORWARD_EXTRAS:
            break
    built = []
    for n, rec in enumerate(picks):
        spec = rec["spec"]
        i = start_id + n
        while i in seen_ids:
            i += 1
        seen_ids.add(i)
        mode = "tsmom_prob_filter" if spec.get("sizing") == "tsmom_filter" else "prob_sized"
        swing = {"family": mode, "prob": {
            "mode": mode, "threshold": spec.get("threshold"), "kelly_frac": spec.get("kellyFrac", 0.5),
            "target_name": spec.get("target"),
            "pack_path": str(pm.MODEL_DIR / "frozen.json"),
            "proba_path": str(pm.MODEL_DIR / "oos_proba.csv"),
        }}
        label = f"prob_pre_rank{n + 1}:{rec['label']}"
        built.append({"id": i, "label": label, "swing": swing, "orb": None})
    shutil.copy2(sw.DESIGN_PATH, sw.DESIGN_PATH.with_name(f"design.json.bak-{datetime.now():%Y%m%d-%H%M}"))
    current = json.loads(sw.DESIGN_PATH.read_text())
    # Keep existing non-prob extras (growth-MR etc.). Do not change designHash.
    kept = [e for e in (current.get("forwardExtras") or [])
            if not str(e.get("label", "")).startswith("prob_")]
    current["forwardExtras"] = kept + built
    note = current.get("forwardExtrasNote") or ""
    add = ("Added 2026-11-25 from the probability walk-forward study (top pre-holdout ranks). "
           "Shadow forward records only; not part of the frozen design hash.")
    current["forwardExtrasNote"] = (note + " " + add).strip() if add not in (note or "") else note
    sw.write_json(sw.DESIGN_PATH, current)
    return [{"id": e["id"], "label": e["label"]} for e in current["forwardExtras"]]


def _verdict(stats, champ_row, refs, cap_mag, passed_dd):
    t = stats.get("tests", {}) or {}
    tb_h = t.get("tsmom_baseline:holdout", {})
    tb_p = t.get("tsmom_baseline:preHoldout", {})
    dsr = (stats.get("deflatedSharpeCumulative") or stats.get("deflatedSharpePre") or {}).get("deflatedSharpeProbability")
    pbo = (stats.get("pboPre") or {}).get("pbo")
    gates = {"championBeatsTsmomOnHoldout(p<0.05)": (tb_h.get("championP") or 1) < 0.05,
             "spaVsTsmomPreHoldout(p<0.05)": (tb_p.get("spaP") or 1) < 0.05,
             "spaVsTsmomHoldout(p<0.05)": (tb_h.get("spaP") or 1) < 0.05,
             "deflatedSharpeCumulative>0.95": (dsr or 0) > 0.95,
             "pbo<0.5": pbo is not None and pbo < 0.5,
             "passedDrawdownGate": bool(passed_dd)}
    passed = all(gates.values())
    rec = ("Add the probability filter/sizing rule to the paper account (recommendation only; Robert decides)."
           if passed else
           "Do not change the paper account: the walk-forward probability rule does not beat plain TSMOM "
           "after costs and multiple-testing correction (or it fails the drawdown gate). It keeps running "
           "in the swarm's forward shadow mode to collect a live record.")
    return {"champion": None if champ_row is None else champ_row["id"],
            "championLabel": None if champ_row is None else champ_row["label"],
            "gates": gates, "passedAll": passed, "recommendation": rec,
            "drawdownCap": cap_mag,
            "references": {k: {"pre": v.get("pre"), "holdout": v.get("holdout")} for k, v in refs.items()}}


def _update_swarm_status(doc):
    if not sw.STATUS_PATH.exists():
        return
    status = json.loads(sw.STATUS_PATH.read_text())
    v, st, refs = doc["verdict"], doc["stats"], doc["references"]
    champ = next((r for r in doc["leaderboard"] if r["id"] == v.get("champion")), {})
    t = st.get("tests", {})
    summary = (
        f"{doc['nTrials']} probability trials; champion {champ.get('label')} pre Sharpe "
        f"{sw._f((champ.get('pre') or {}).get('sharpe'))}, holdout Sharpe "
        f"{sw._f((champ.get('holdout') or {}).get('sharpe'))} vs TSMOM "
        f"{sw._f(((refs.get('tsmom_baseline') or {}).get('holdout') or {}).get('sharpe'))}, SPY "
        f"{sw._f(((refs.get('spy') or {}).get('holdout') or {}).get('sharpe'))}; PBO "
        f"{sw._f((st.get('pboPre') or {}).get('pbo'))}; cumulative DSR prob "
        f"{sw._f((st.get('deflatedSharpeCumulative') or {}).get('deflatedSharpeProbability'))}; "
        f"SPA p vs TSMOM holdout {sw._f((t.get('tsmom_baseline:holdout') or {}).get('spaP'))}. "
        f"{v['recommendation']}"
    )
    status["probStudy"] = {"summary": summary, "passedAll": v["passedAll"],
                           "generatedAt": doc["generatedAt"],
                           "holdoutViews": doc["holdoutLedger"]["views"],
                           "forwardExtras": doc["forwardExtras"],
                           "calibration": doc.get("calibrationSummary")}
    status["holdoutViews"] = doc["holdoutLedger"]["views"]
    status["promptSummary"] = sw.prompt_summary(status)
    sw.write_json(sw.STATUS_PATH, status)


def _write_oos_tape(tapes, champion_spec, model_dir=None):
    model_dir = Path(model_dir) if model_dir else pm.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for (model_kind, kind, k, thr), pred in tapes.items():
        name = pm.target_key(kind, k, thr)
        col = pred["p"].rename(name).to_frame()
        col["model"] = model_kind
        frames.append(col.reset_index())
    if not frames:
        return None
    # Keep the champion model's probabilities (and, for display, the same model on every target
    # if available). Fall back to ensemble then logit.
    preferred = (champion_spec or {}).get("model") or "ensemble"
    long = pd.concat(frames, ignore_index=True)
    # Build one row per (date, ticker) with a column per target from the preferred model.
    pieces = []
    for (model_kind, kind, k, thr), pred in tapes.items():
        if model_kind != preferred and (long["model"] == preferred).any():
            continue
        name = pm.target_key(kind, k, thr)
        pieces.append(pred["p"].rename(name))
    if not pieces:
        return None
    out = pd.concat(pieces, axis=1)
    out = out.reset_index()
    path = model_dir / "oos_proba.csv"
    out.to_csv(path, index=False)
    return path


def _calibration_summary(fold_book):
    """Average pre-holdout Brier / AUC / IC by model and target."""
    rows = []
    for key, folds in fold_book.items():
        model_kind, kind, k, thr = key
        pre = [f for f in folds if not f.get("holdout")]
        if not pre:
            continue
        rows.append({"model": model_kind, "target": pm.target_key(kind, k, thr),
                     "nFolds": len(pre),
                     "brier": float(np.mean([f["brier"] for f in pre if f.get("brier") is not None] or [np.nan])),
                     "logLoss": float(np.mean([f["logLoss"] for f in pre if f.get("logLoss") is not None] or [np.nan])),
                     "auc": float(np.mean([f["auc"] for f in pre if f.get("auc") is not None] or [np.nan])),
                     "ic": float(np.mean([f["ic"] for f in pre if f.get("ic") is not None] or [np.nan])),
                     "reliability": pre[-1].get("reliability") if pre else []})
    return rows


def run(skip_holdout=False, min_train=MIN_TRAIN, out_dir=None, mutate_swarm=True):
    started = time.time()
    design = json.loads(sw.DESIGN_PATH.read_text())
    custom_dir = Path(out_dir) if out_dir else None
    model_dir = custom_dir if custom_dir is not None else pm.MODEL_DIR
    study_path = (custom_dir / "prob_walkforward.json") if custom_dir is not None else pm.STUDY_PATH
    latest_path = (custom_dir / "prob_latest.json") if custom_dir is not None else pm.LATEST_PATH
    frozen_path = model_dir / "frozen.json"
    if custom_dir is not None:
        custom_dir.mkdir(parents=True, exist_ok=True)
    panel, data_warnings = _load_panel()
    tickers = _universe(panel)
    trade = _trade_universe(panel)
    log(f"prob study: {len(panel.dates)} sessions, {len(tickers)} tickers "
        f"(trade on {len(trade)} CORE ETFs), long_history={getattr(panel, 'used_long_history', False)}")
    features = pm.stack_features(pm.build_features(panel, tickers=tickers), tickers=tickers)
    holdout_start = pd.Timestamp(design["holdoutStart"])
    data_end = pd.Timestamp(design["dataEnd"])
    eval_start = pd.Timestamp(design["evalStart"])
    sim_start = pd.Timestamp(design["simStart"])
    tapes, fold_book, packs = {}, {}, {}
    for model_kind in MODEL_KINDS:
        for kind, k, thr in TARGET_SPECS:
            key = (model_kind, kind, k, thr)
            log(f"walk-forward {model_kind} {pm.target_key(kind, k, thr)}")
            pred, folds, pack = walk_forward_probs(
                panel, features, tickers, kind, k, threshold=thr, model_kind=model_kind,
                min_train=min_train, holdout_start=None if skip_holdout else holdout_start)
            tapes[key] = pred
            fold_book[key] = folds
            packs[pm.target_key(kind, k, thr) + ":" + model_kind] = pack
    tsmom_w = tsmom_weights(panel)
    rets, rows, sim_map, cap_mag = evaluate_trials(panel, tapes, tsmom_w, sim_start, data_end, holdout_start)
    pre_mask = rets.index < holdout_start
    hold_mask = rets.index >= holdout_start
    trial_ids = [r["id"] for r in rows if not r.get("reference") and r["id"] in rets.columns]
    eligible = {r["id"] for r in rows if (not r.get("reference")) and r.get("passesDd")}
    spy = sw._bench_spy(panel, sim_start, data_end).reindex(rets.index).fillna(0.0)
    benches = {"tsmom_baseline": rets.get(1), "spy": spy}
    # Count every trial in DSR / PBO / SPA (honesty). Champion is chosen separately
    # with the drawdown gate, then championP is taken from that id's raw p-value.
    stats = sw._leg_stats(rets[trial_ids] if trial_ids else rets, pre_mask, hold_mask,
                          set(trial_ids), {}, benches, "prob_walkforward")
    champ_id, passed_dd, ranked_ids = select_champion(rows)
    if champ_id is not None:
        stats["champion"] = champ_id
        stats["ranked"] = ranked_ids
        for res in (stats.get("tests") or {}).values():
            if isinstance(res, dict) and "rawP" in res:
                res["championP"] = res["rawP"].get(str(champ_id))
    # Cumulative DSR: this study plus prior swarm trials on the same pre-holdout window.
    prior_sr = []
    swarm_csv = sw.SWARM_DIR / "returns_swing.csv"
    if swarm_csv.exists():
        prev = pd.read_csv(swarm_csv, index_col=0, parse_dates=True)
        prior_sr = [sw.sharpe(prev[c].loc[prev.index < holdout_start]) for c in prev.columns]
    pre = rets.loc[pre_mask]
    growth_csv = sw.SWARM_DIR / "growth_mr_study.json"
    n_prior_extra = 0
    if growth_csv.exists():
        try:
            gdoc = json.loads(growth_csv.read_text())
            n_prior_extra = int(gdoc.get("variants") or 0)
        except (OSError, ValueError):
            n_prior_extra = 0
    trial_sr = [sw.sharpe(pre[i]) for i in trial_ids if i in pre]
    if champ_id is not None and champ_id in pre:
        stats["deflatedSharpeCumulative"] = sw.deflated_sharpe(
            pre[champ_id].to_numpy(), len(trial_sr) + len(prior_sr) + n_prior_extra,
            np.array(trial_sr + prior_sr + [0.0] * n_prior_extra))
        stats["nTrialsCounted"] = len(trial_sr) + len(prior_sr) + n_prior_extra

    by_id = {r["id"]: r for r in rows}
    champ_row = by_id.get(champ_id)
    refs = {}
    for r in rows:
        if r.get("reference"):
            refs[r["label"]] = {"pre": r["pre"], "holdout": r["holdout"], "trades": r["trades"]}
    if "spy" not in refs:
        refs["spy"] = {"pre": sw.perf(spy.loc[pre_mask]), "holdout": sw.perf(spy.loc[hold_mask])}

    # Freeze the champion pack + OOS tape (pre-holdout rows are OOS; holdout rows were
    # predicted with the last pre-holdout pack).
    champ_spec = (champ_row or {}).get("spec") or {}
    tape_path = _write_oos_tape(tapes, champ_spec, model_dir=model_dir)
    frozen = None
    target_name = champ_spec.get("target")
    model_kind = champ_spec.get("model")
    if target_name and model_kind:
        frozen = packs.get(f"{target_name}:{model_kind}")
    if frozen is None and packs:
        frozen = next(iter(packs.values()))
    if frozen is not None:
        frozen = dict(frozen)
        frozen["chosen"] = champ_spec
        frozen["threshold"] = champ_spec.get("threshold")
        frozen["sizing"] = champ_spec.get("sizing")
        pm.write_json(frozen_path, frozen)

    study_hash = sw.stable_hash({"targets": TARGET_SPECS, "models": MODEL_KINDS, "thresholds": THRESHOLDS,
                                 "sizings": SIZINGS, "code": STUDY_CODE, "minTrain": min_train})
    if skip_holdout:
        ledger = {"views": None, "note": "holdout evaluation skipped"}
    else:
        ledger = sw._record_holdout_view(design, stats, study="prob_walkforward",
                                         design_hash="prob-" + study_hash)
    verdict = _verdict(stats, champ_row, refs, cap_mag, passed_dd)
    if mutate_swarm and champ_row is not None:
        extras = _add_forward_extras(design, champ_row, ranked_ids, by_id)
    elif mutate_swarm:
        extras = list((json.loads(sw.DESIGN_PATH.read_text()).get("forwardExtras") or []))
    else:
        extras = []

    cal = _calibration_summary(fold_book)
    # Latest snapshot for the UI / analyst tool (last OOS date of the preferred tape).
    proba_frame = None
    if tape_path and Path(tape_path).is_file():
        proba_frame = pd.read_csv(tape_path, parse_dates=["date"])
    caveats = list(data_warnings) + [
        "Labels are k-session close-to-close returns; the paper trader fills at the next open.",
        "Thresholds, model kind and sizing were chosen on pre-holdout out-of-sample data only.",
        "The sealed holdout is a single evaluation of the frozen specification.",
        "Schwab pre-Massive returns are split-adjusted but not dividend-adjusted; overlapping Massive "
        "dates are split- and dividend-adjusted. Delisted names are missing (survivorship bias).",
        "Recommendation only: the live paper account is not changed by this study.",
    ]
    latest = pm.latest_snapshot(proba_frame, packs, metrics={"calibration": cal, "champion": champ_spec},
                                caveats=caveats)
    latest["verdict"] = {"passedAll": verdict["passedAll"], "recommendation": verdict["recommendation"],
                         "championLabel": verdict.get("championLabel")}
    pm.write_json(latest_path, latest)

    fold_out = {f"{k[0]}:{pm.target_key(k[1], k[2], k[3])}": _fold_table(v) for k, v in fold_book.items()}
    doc = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "studyHash": study_hash,
           "grid": {"models": list(MODEL_KINDS), "targets": [pm.target_key(*t) for t in TARGET_SPECS],
                    "thresholds": list(THRESHOLDS), "sizings": list(SIZINGS)},
           "nTrials": len(trial_ids), "nTrialsCounted": stats.get("nTrialsCounted"),
           "window": {k: design[k] for k in ("evalStart", "holdoutStart", "dataEnd")},
           "data": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                    "sessions": len(panel.dates), "tickers": tickers, "tradeUniverse": trade,
                    "usedLongHistory": bool(getattr(panel, "used_long_history", False))},
           "stats": sw._strip(stats), "references": refs, "leaderboard": rows,
           "calibration": fold_out, "calibrationSummary": cal,
           "drawdownCap": cap_mag, "passedDrawdownGate": passed_dd,
           "holdoutLedger": {"views": ledger.get("views"),
                             "note": "this study is an additional look at the swarm's holdout"},
           "forwardExtras": extras, "verdict": verdict, "caveats": caveats,
           "latestPath": str(latest_path), "frozenPath": str(frozen_path),
           "swarmMutated": bool(mutate_swarm),
           "runtime": {"totalSeconds": round(time.time() - started, 1)}}
    sw.write_json(study_path, doc)
    if mutate_swarm:
        _update_swarm_status(doc)
    log(f"prob study done in {doc['runtime']['totalSeconds']}s -> {study_path}")
    return doc


def latest_for_tool(args=None):
    """Analyst-tool entry: latest calibrated probabilities, optionally for one ticker."""
    args = args or {}
    path = Path(args.get("path") or pm.LATEST_PATH)
    if not path.is_file():
        return {"available": False, "reason": "No probability study yet. Run python research.py prob."}
    doc = json.loads(path.read_text(encoding="utf-8"))
    ticker = (args.get("ticker") or args.get("symbol") or "").strip().upper()
    tickers = args.get("tickers") or ([] if not ticker else [ticker])
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = [str(t).upper() for t in tickers if t]
    if tickers:
        wanted = {t: (doc.get("tickers") or {}).get(t) for t in tickers}
        return {"asof": doc.get("asof"), "tickers": wanted, "spy": doc.get("spy"),
                "metrics": doc.get("metrics"), "verdict": doc.get("verdict"),
                "caveats": doc.get("caveats"), "available": True}
    return {**doc, "available": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-holdout", action="store_true",
                        help="Fit and score pre-holdout only (no holdout look, no ledger entry)")
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN)
    parser.add_argument("--out", default=None,
                        help="Write study artifacts into this directory instead of the live prob_model paths.")
    parser.add_argument("--no-mutate-swarm", action="store_true",
                        help="Do not rewrite design.json forwardExtras or swarm status.")
    args = parser.parse_args(argv)
    doc = run(skip_holdout=args.skip_holdout, min_train=args.min_train,
              out_dir=args.out, mutate_swarm=not args.no_mutate_swarm)
    print(json.dumps({"verdict": doc["verdict"], "runtime": doc["runtime"],
                      "nTrials": doc["nTrials"], "calibrationSummary": doc["calibrationSummary"]},
                     default=sw._json_default))
    return 0


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("prob_walkforward"):
        sys.exit(main() or 0)
