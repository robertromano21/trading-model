"""Walk-forward test of growth-adjusted mean reversion (strategies/growth_meanrev.py) as an overlay on TSMOM.

Compared after Schwab costs, whole shares and the $10k small-account rules (swarm.simulate_swing) against
  * plain TSMOM (the real paper config since 2026-11-25),
  * the old overlay (TSMOM + flat-mean reversion, regime fixed = the previous real config),
  * SPY buy-and-hold.
Selection uses pre-holdout rows only (swarm.select / walk_forward_select). The holdout window is the swarm's
sealed window; evaluating it here is logged in holdout_ledger.json as an additional look.
Statistics: deflated Sharpe (this study's trials and cumulative trials incl. the 50-account swarm), PBO (CSCV),
White's Reality Check / Hansen SPA vs each benchmark, Holm/BH. The top pre-holdout variants are added to the swarm's
forward shadow mode (design.json forwardExtras). Recommendation only; nothing is changed in the paper account.

  python growth_mr_study.py [--workers 8]
"""

import argparse
import itertools
import json
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import swarm as sw

WEIGHT_SETS = {"trend_only": (1.0, 0.0, 0.0, 0.0), "trend_sector": (0.6, 0.0, 0.3, 0.1),
               "blend": (0.5, 0.2, 0.2, 0.1), "market_heavy": (0.3, 0.1, 0.3, 0.3)}
GRID = {"weights": ["trend_only", "trend_sector", "blend", "market_heavy", "learned"],
        "entry_z": [-1.5, -2.0], "max_hold": [5, 10], "regime_filter": ["fixed", "off"]}
DEFAULT_GROWTH = {"weights": "blend", "entry_z": -2.0, "max_hold": 10, "regime_filter": "fixed"}
OUT = sw.SWARM_DIR / "growth_mr_study.json"
FORWARD_EXTRAS = 5


def growth_params(v):
    g = {"entry_z": v["entry_z"], "max_hold": v["max_hold"], "regime_filter": v["regime_filter"]}
    if v["weights"] == "learned":
        g["weights"] = "learned"
    else:
        wt, wf, ws, wm = WEIGHT_SETS[v["weights"]]
        g.update({"weights": "config", "w_trend": wt, "w_fund": wf, "w_sector": ws, "w_market": wm})
    return g


def variants():
    keys = list(GRID)
    out = []
    for k, values in enumerate(itertools.product(*(GRID[x] for x in keys))):
        v = dict(zip(keys, values))
        out.append({"id": 200 + k, "label": "growth_" + "_".join(str(x) for x in values), "grid": v,
                    "swing": {"family": "growth_overlay", "growth": growth_params(v)}, "orb": None})
    return out


REFERENCE = [
    {"id": 1, "label": "tsmom_baseline", "swing": {"family": "tsmom_baseline"}, "orb": None},
    {"id": 190, "label": "old_overlay_fixed", "swing": {"family": "old_overlay_fixed"}, "orb": None},
    {"id": 0, "label": "production_hmm", "swing": {"family": "production", "regime": "hmm"}, "orb": None},
    {"id": 191, "label": "sleeve_meanrev", "swing": {"family": "sleeve_meanrev"}, "orb": None},
    {"id": 192, "label": "sleeve_growth_mr_default", "swing": {"family": "sleeve_growth_mr",
                                                               "growth": growth_params(DEFAULT_GROWTH)}, "orb": None},
]


def run(workers=None):
    started = time.time()
    design = json.loads(sw.DESIGN_PATH.read_text())
    panel, panel_path = sw.load_panel_cached()
    fingerprint = sw.data_fingerprint()
    sim_start, eval_start = pd.Timestamp(design["simStart"]), pd.Timestamp(design["evalStart"])
    holdout_start, data_end = pd.Timestamp(design["holdoutStart"]), pd.Timestamp(design["dataEnd"])
    grid = variants()
    accounts = grid + REFERENCE
    tasks = [{"account": a, "start": str(sim_start.date()),
              "cacheKey": sw._account_key(a, "swing", fingerprint, str(data_end.date()))} for a in accounts]
    workers = workers or sw.default_workers()
    sw.log(f"growth study: {len(grid)} variants + {len(REFERENCE)} references on {workers} workers")
    t0 = time.time()
    results = {r["id"]: r for r in sw.run_tasks(sw.run_swing_account, tasks, workers, panel_path, None)}
    sim_seconds = time.time() - t0
    failed = {i: r["error"] for i, r in results.items() if not r.get("ok")}
    ok = [i for i, r in results.items() if r.get("ok")]
    rets = pd.DataFrame({i: results[i]["returns"] for i in ok}).loc[eval_start:data_end]
    pre_mask, hold_mask = rets.index < holdout_start, rets.index >= holdout_start
    gids = [a["id"] for a in grid if a["id"] in rets.columns]
    eligible = {i for i in gids if not results[i]["pdt"]["breach"]}
    spy = sw._bench_spy(panel, sim_start, data_end).reindex(rets.index).fillna(0.0)
    benches = {"tsmom_baseline": rets.get(1), "old_overlay": rets.get(190), "spy": spy}
    stats = sw._leg_stats(rets[gids], pre_mask, hold_mask, eligible, {}, benches, "growth_overlay")
    folds = sw.walk_forward_select(rets[gids].loc[pre_mask], eligible)
    stitched = sw._stitched(panel, {i: results[i] for i in gids}, folds, sim_start, holdout_start - pd.Timedelta(days=1))
    for f in folds:
        f.pop("rows", None)
    stats["walkForward"] = {"folds": folds, "stitchedOos": sw.perf(stitched) if stitched is not None else None}
    # cumulative multiple testing: this study plus the 49 swarm variants already tried on the same data
    swarm_csv = sw.SWARM_DIR / "returns_swing.csv"
    prior_sr = []
    if swarm_csv.exists():
        prev = pd.read_csv(swarm_csv, index_col=0, parse_dates=True)
        prior_sr = [sw.sharpe(prev[c].loc[prev.index < holdout_start]) for c in prev.columns]
    pre = rets.loc[pre_mask]
    growth_sr = [sw.sharpe(pre[i]) for i in gids if i in eligible]
    if stats.get("champion") is not None:
        stats["deflatedSharpeCumulative"] = sw.deflated_sharpe(pre[stats["champion"]].to_numpy(),
                                                               len(growth_sr) + len(prior_sr),
                                                               np.array(growth_sr + prior_sr))
    hold = rets.loc[hold_mask]
    rows = []
    rank = {k: n + 1 for n, k in enumerate(stats.get("ranked", []))}
    tests = stats.get("tests", {})
    for a in accounts:
        i = a["id"]
        r = results.get(i, {})
        if not r.get("ok"):
            continue
        at = r.get("afterTaxEquity")
        row = {"id": i, "label": a["label"], "grid": a.get("grid"), "swing": a["swing"], "selectionRank": rank.get(i),
               "eligible": i in eligible if i in gids else None, "pdt": r["pdt"], "trades": r["trades"],
               "turnoverAnnual": r["turnover"], "pre": sw.perf(pre[i], at), "holdout": sw.perf(hold[i], at)}
        for b in ("tsmom_baseline", "old_overlay", "spy"):
            t = tests.get(f"{b}:holdout", {})
            row[f"holdoutVs_{b}"] = {"p": t.get("rawP", {}).get(str(i)), "holm": t.get("holm", {}).get(str(i)),
                                     "bh": t.get("bh", {}).get(str(i))}
        rows.append(row)
    # factor effects on pre-holdout vs holdout Sharpe (descriptive)
    effects = {}
    for key in GRID:
        groups = {}
        for row in rows:
            if row.get("grid"):
                groups.setdefault(str(row["grid"][key]), []).append((row["pre"].get("sharpe"), row["holdout"].get("sharpe")))
        effects[key] = {lvl: {"n": len(v), "preSharpe": float(np.nanmean([x for x, _ in v if x is not None] or [np.nan])),
                              "holdoutSharpe": float(np.nanmean([y for _, y in v if y is not None] or [np.nan]))}
                        for lvl, v in groups.items()}
    study_hash = sw.stable_hash({"grid": GRID, "weights": WEIGHT_SETS, "code": "growth-mr-1"})
    ledger = sw._record_holdout_view(design, stats, study="growth_mr", design_hash="growth-" + study_hash)
    verdict = _verdict(stats)
    by_id = {r["id"]: r for r in rows}
    refs = {a["label"]: {"pre": by_id[a["id"]]["pre"], "holdout": by_id[a["id"]]["holdout"],
                         "trades": by_id[a["id"]]["trades"]} for a in REFERENCE if a["id"] in by_id}
    refs["spy"] = {"pre": sw.perf(spy.loc[pre_mask]), "holdout": sw.perf(spy.loc[hold_mask])}
    fund_cov = _fundamental_coverage(panel)
    extras = _add_forward_extras(design, stats, grid)
    doc = {"generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "studyHash": study_hash, "grid": GRID,
           "weightSets": WEIGHT_SETS, "variants": len(grid), "window": {k: design[k] for k in ("evalStart", "holdoutStart", "dataEnd")},
           "stats": sw._strip(stats), "references": refs, "leaderboard": rows, "factorEffects": effects,
           "holdoutLedger": {"views": ledger["views"], "note": "this study is an additional look at the swarm's holdout"},
           "fundamentalCoverage": fund_cov, "forwardExtras": extras, "verdict": verdict, "failed": failed,
           "runtime": {"simSeconds": round(sim_seconds, 1), "totalSeconds": round(time.time() - started, 1), "workers": workers}}
    sw.write_json(OUT, doc)
    _update_swarm_status(doc)
    sw.log(f"growth study done in {doc['runtime']['totalSeconds']}s -> {OUT}")
    return doc


def _fundamental_coverage(panel):
    from strategies.growth_meanrev import load_fundamental_growth
    f = load_fundamental_growth(panel.dates, list(panel.close.columns))
    have = [c for c in f.columns if f[c].notna().any()]
    return {"tickersWithPointInTimeGrowth": have, "analystEstimates": "none cached (component unused)",
            "note": "SEC quarterly revenue YoY, usable 60 days after period end; every other name renormalizes onto "
                    "trend / sector / market drift."}


def _verdict(stats):
    c = stats.get("champion")
    t = stats.get("tests", {})
    tb_h, ov_h = t.get("tsmom_baseline:holdout", {}), t.get("old_overlay:holdout", {})
    tb_p = t.get("tsmom_baseline:preHoldout", {})
    dsr = (stats.get("deflatedSharpeCumulative") or stats.get("deflatedSharpePre") or {}).get("deflatedSharpeProbability")
    pbo = (stats.get("pboPre") or {}).get("pbo")
    gates = {"championBeatsTsmomOnHoldout(p<0.05)": (tb_h.get("championP") or 1) < 0.05,
             "spaVsTsmomPreHoldout(p<0.05)": (tb_p.get("spaP") or 1) < 0.05,
             "spaVsTsmomHoldout(p<0.05)": (tb_h.get("spaP") or 1) < 0.05,
             "deflatedSharpeCumulative>0.95": (dsr or 0) > 0.95, "pbo<0.5": pbo is not None and pbo < 0.5}
    passed = all(gates.values())
    rec = ("Add the growth-adjusted overlay to the paper account (recommendation only; Robert decides)." if passed else
           "Do not add it to the paper account: it does not beat plain TSMOM after multiple-testing correction. "
           "It keeps running in the swarm's forward shadow mode to collect a live record.")
    return {"champion": c, "gates": gates, "passedAll": passed, "recommendation": rec,
            "beatsOldOverlayOnHoldout": (ov_h.get("championP") or 1) < 0.05}


def _add_forward_extras(design, stats, grid):
    by_id = {a["id"]: a for a in grid}
    picks = [i for i in stats.get("ranked", [])[:FORWARD_EXTRAS]]
    extras = [{"id": 100, "label": "growth_default", "swing": {"family": "growth_overlay", "growth": growth_params(DEFAULT_GROWTH)}, "orb": None}]
    for n, i in enumerate(picks):
        a = by_id[i]
        extras.append({"id": 101 + n, "label": f"growth_pre_rank{n + 1}:{a['label']}", "swing": a["swing"], "orb": None})
    shutil.copy2(sw.DESIGN_PATH, sw.DESIGN_PATH.with_name(f"design.json.bak-{datetime.now():%Y%m%d-%H%M}"))
    current = json.loads(sw.DESIGN_PATH.read_text())
    current["forwardExtras"] = extras
    current["forwardExtrasNote"] = ("Added 2026-11-25 from the growth-MR study (top pre-holdout ranks + default config). "
                                    "Shadow forward records only; not part of the frozen design hash.")
    sw.write_json(sw.DESIGN_PATH, current)
    return [{"id": e["id"], "label": e["label"]} for e in extras]


def _update_swarm_status(doc):
    if not sw.STATUS_PATH.exists():
        return
    status = json.loads(sw.STATUS_PATH.read_text())
    st, refs, v = doc["stats"], doc["references"], doc["verdict"]
    champ = next((r for r in doc["leaderboard"] if r["id"] == st.get("champion")), {})
    t = st.get("tests", {})
    summary = (f"{doc['variants']} variants; champion {champ.get('label')} pre Sharpe {sw._f(champ.get('pre', {}).get('sharpe'))}, "
               f"holdout Sharpe {sw._f(champ.get('holdout', {}).get('sharpe'))} vs TSMOM {sw._f(refs['tsmom_baseline']['holdout'].get('sharpe'))}, "
               f"old overlay {sw._f(refs['old_overlay_fixed']['holdout'].get('sharpe'))}, SPY {sw._f(refs['spy']['holdout'].get('sharpe'))}; "
               f"PBO {sw._f((st.get('pboPre') or {}).get('pbo'))}; cumulative DSR prob "
               f"{sw._f((st.get('deflatedSharpeCumulative') or {}).get('deflatedSharpeProbability'))}; SPA p vs TSMOM holdout "
               f"{sw._f(t.get('tsmom_baseline:holdout', {}).get('spaP'))}. {v['recommendation']}")
    status["growthStudy"] = {"summary": summary, "passedAll": v["passedAll"], "generatedAt": doc["generatedAt"],
                             "holdoutViews": doc["holdoutLedger"]["views"], "forwardExtras": doc["forwardExtras"]}
    status["realPaperAccount"] = "plain TSMOM baseline (= swarm acct 1) since 2026-11-25"
    status["holdoutViews"] = doc["holdoutLedger"]["views"]
    status["promptSummary"] = sw.prompt_summary(status)
    sw.write_json(sw.STATUS_PATH, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args(argv)
    doc = run(args.workers)
    print(json.dumps({"verdict": doc["verdict"], "runtime": doc["runtime"]}, default=sw._json_default))
    return 0


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("growth_mr_study"):
        sys.exit(main() or 0)
