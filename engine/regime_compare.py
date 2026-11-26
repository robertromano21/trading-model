"""Walk-forward after-cost comparison of the HMM regime allocator vs fixed filters and SPY.

The HMM is itself walk-forward (rolling refit, filtered probabilities only). Sleeve parameters stay at
defaults so the comparison isolates the regime model; it is not a joint parameter search.

Usage (from engine/):
  python regime_compare.py
"""

import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from costs import CostModel
from fetch_and_model import DATA_DIR, write_atomically
from market_data import MARKET, load_panel
from schwab_daily import merge_caveats
from metrics import performance, summarize
from portfolio_sim import simulate
from regime_hmm import latest_snapshot, regime_policy, write_regime_state
from research import WARMUP, benchmark_weights, run_strategy
from strategies import CombinedStrategy

OUT_DIR = DATA_DIR / "Research"


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", file=sys.stderr, flush=True)


def make_combined(model, n_states=3, include_macro=False):
    return CombinedStrategy(regime_model=model, regime_n_states=n_states, regime_write_state=False,
                            regime_include_macro=include_macro)


def _perf(result, start):
    return summarize(result, start)["performance"]


def _row(name, perf, extra=None):
    out = {"name": name, "start": perf.get("start"), "end": perf.get("end"),
           "sessions": perf.get("sessions"), "totalReturn": perf.get("totalReturn"),
           "cagr": perf.get("cagr"), "volatility": perf.get("volatility"), "sharpe": perf.get("sharpe"),
           "sharpeStdErr": perf.get("sharpeStdErr"), "maxDrawdown": perf.get("maxDrawdown"),
           "calmar": perf.get("calmar"), "sortino": perf.get("sortino"),
           "turnoverPerYear": perf.get("turnoverPerYear"), "avgGrossExposure": perf.get("avgGrossExposure"),
           "timeInMarket": perf.get("timeInMarket"), "costBpsPerYear": perf.get("costBpsPerYear")}
    if extra:
        out.update(extra)
    return out


def _fmt_pct(value, digits=1):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value, digits=2):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def _table(rows):
    headers = ["Model", "Start", "Sessions", "Total", "CAGR", "Vol", "Sharpe", "Sharpe SE",
               "Max DD", "Turnover/yr", "Avg gross", "Cost bps/yr"]
    lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
    for row in rows:
        lines.append(" | ".join([
            row["name"], str(row.get("start") or "—"), str(row.get("sessions") or "—"),
            _fmt_pct(row.get("totalReturn")), _fmt_pct(row.get("cagr")), _fmt_pct(row.get("volatility")),
            _fmt_num(row.get("sharpe")), _fmt_num(row.get("sharpeStdErr")),
            _fmt_pct(row.get("maxDrawdown")), _fmt_num(row.get("turnoverPerYear"), 1),
            _fmt_pct(row.get("avgGrossExposure"), 0), _fmt_num(row.get("costBpsPerYear"), 0),
        ]))
    return "\n".join(lines) + "\n"


def regime_occupancy(policy, start):
    hmm = policy.loc[pd.Timestamp(start):]
    hmm = hmm[hmm["source"] == "hmm"]
    if hmm.empty:
        return {"hmmSessions": 0}
    counts = hmm["state_name"].value_counts(normalize=True).to_dict()
    return {
        "hmmSessions": int(len(hmm)),
        "firstHmmDate": str(hmm.index[0].date()),
        "lastHmmDate": str(hmm.index[-1].date()),
        "shareByState": {k: float(v) for k, v in counts.items()},
        "avgAllocation": {
            "tsmomScale": float(hmm["tsmom_scale"].mean()),
            "ief": float(hmm["ief_weight"].mean()),
            "tlt": float(hmm["tlt_weight"].mean()),
            "meanrevOn": float(hmm["meanrev_on"].mean()),
        },
    }


def markdown(summary):
    common = summary["commonPeriod"]
    lines = [
        "# HMM regime allocator vs fixed filters and SPY",
        "",
        f"Generated {summary['generatedAt']} (America/New_York clock on the research machine). "
        f"Data {summary['data']['start']} to {summary['data']['end']} "
        f"({summary['data']['sessions']} sessions, {summary['data']['tickers']} tickers).",
        "",
        "Costs are the shared Schwab profile ($0 commission, SEC + FINRA TAF, spread and impact). "
        "The HMM is fit walk-forward on a rolling window and uses filtered — not smoothed — "
        "state probabilities. Sleeve parameters are defaults (not re-tuned) so the comparison "
        "isolates the regime model.",
        "",
        f"## Common evaluation window ({common['start']} to {common['end']})",
        "",
        _table(summary["rows"]["common"]),
        "",
        "### Also reported from the existing research warm-up and walk-forward starts",
        "",
    ]
    for label, key in (("After 110-session warm-up", "warmup"),
                       ("From the 252-session walk-forward start", "walkForwardStart")):
        if summary["rows"].get(key):
            lines += [f"#### {label}", "", _table(summary["rows"][key]), ""]
    occ = summary.get("occupancy") or {}
    lines += [
        "## How the HMM allocated in sample",
        "",
        f"HMM live from {occ.get('firstHmmDate')} to {occ.get('lastHmmDate')} "
        f"({occ.get('hmmSessions')} sessions).",
        "",
        "State occupancy (share of HMM sessions as the most likely state):",
        "",
    ]
    for name, share in (occ.get("shareByState") or {}).items():
        lines.append(f"- {name}: {share * 100:.1f}%")
    avg = occ.get("avgAllocation") or {}
    if avg:
        lines += [
            "",
            "Average blended allocation while the HMM was live:",
            "",
            f"- TSMOM scale: {avg['tsmomScale'] * 100:.1f}% of the unadjusted sleeve",
            f"- IEF overlay: {avg['ief'] * 100:.1f}%",
            f"- TLT overlay: {avg['tlt'] * 100:.1f}%",
            f"- Mean-reversion overlay on: {avg['meanrevOn'] * 100:.1f}% of sessions",
            "",
        ]
    if summary.get("latest"):
        snap = summary["latest"]
        lines += [
            f"## Latest snapshot ({snap.get('asOf')})",
            "",
            f"Most likely state: **{snap.get('regime')}** (model: {snap.get('model')}, "
            f"{snap.get('nStates')} states).",
            "",
            "Filtered probabilities: " + ", ".join(
                f"{k} {v * 100:.1f}%" for k, v in (snap.get('probabilities') or {}).items() if v is not None),
            "",
        ]
    lines += [
        "## Caveats",
        "",
        *[f"- {c}" for c in summary["caveats"]],
        "",
        summary["verdict"],
        "",
    ]
    return "\n".join(lines)


def verdict_text(common_rows):
    by = {r["name"]: r for r in common_rows}
    hmm = by.get("combined + HMM (3 states)") or by.get("combined + HMM")
    fixed = by.get("combined + fixed filters")
    spy = by.get("SPY buy and hold")
    if not hmm or not fixed:
        return "Not enough results to judge the HMM against the fixed filters."
    def _s(row):
        return row.get("sharpe")
    hmm_s, fix_s, spy_s = _s(hmm), _s(fixed), _s(spy)
    parts = []
    if hmm_s is not None and fix_s is not None:
        better = "higher" if hmm_s > fix_s else "lower"
        parts.append(
            f"On the common window the 3-state HMM Sharpe is {hmm_s:.2f} vs {fix_s:.2f} for the fixed "
            f"filters ({better} by {abs(hmm_s - fix_s):.2f}).")
        if hmm.get("cagr") is not None and fixed.get("cagr") is not None:
            parts.append(
                f"CAGR is {_fmt_pct(hmm['cagr'])} vs {_fmt_pct(fixed['cagr'])}; "
                f"max drawdown {_fmt_pct(hmm['maxDrawdown'])} vs {_fmt_pct(fixed['maxDrawdown'])}.")
    if spy_s is not None and hmm_s is not None:
        parts.append(f"SPY over the same window had a Sharpe of {spy_s:.2f} and a CAGR of {_fmt_pct(spy.get('cagr'))}.")
    se = hmm.get("sharpeStdErr") if hmm else None
    if se and hmm_s is not None and fix_s is not None and abs(hmm_s - fix_s) < 1.5 * se:
        parts.append(
            f"The Sharpe gap is small relative to the sampling error (HMM Sharpe SE ≈ {se:.2f}) "
            f"on only a couple of years of a single bull market, so this should not be treated as "
            f"evidence that the HMM is reliably better or worse.")
    if hmm and fixed and (hmm.get("sharpe") or 0) < (fixed.get("sharpe") or 0):
        parts.append(
            "The HMM did not improve risk-adjusted performance versus the fixed filters in this sample. "
            "That is a plausible outcome with only two years of data and no true crisis regime to learn from; "
            "keep the fixed filter as the config fallback and treat the HMM as experimental until a longer "
            "history (including bear markets and delisted names) is available.")
    elif hmm and fixed and (hmm.get("sharpe") or 0) > (fixed.get("sharpe") or 0):
        parts.append(
            "The HMM looks slightly better here, but two years of one market regime are not enough to "
            "trust a 2–4 state model out of sample. Re-check after more history before letting it "
            "override a working fixed rule in live size.")
    return " ".join(parts)


def _data_caveat(panel):
    start, end = panel.dates[0].date(), panel.dates[-1].date()
    sessions, tickers = len(panel.dates), panel.close.shape[1]
    long = bool(getattr(panel, "used_long_history", False))
    crises = start.isoformat() <= "2008-09-15"
    if long and crises:
        return (f"The panel covers {start} to {end} ({sessions} sessions, {tickers} tickers) with Schwab "
                "daily_long spliced under Massive. That includes the 2008 crisis and later shocks, but "
                "delisted names are still missing (survivorship bias) and pre-Massive Schwab returns omit "
                "dividends. Treat results as optimistic on the long sample.")
    if long:
        return (f"The panel covers {start} to {end} ({sessions} sessions, {tickers} tickers) with Schwab "
                "daily_long spliced in. Delisted names are missing and pre-Massive returns omit dividends.")
    return (f"The database holds {sessions} sessions ({start} to {end}, {tickers} tickers) and may cover "
            "only one broad regime. A hidden Markov model cannot be validated on crisis, recovery and "
            "quiet bull states it has never seen.")


CAVEATS = [
    # First caveat is replaced at runtime by _data_caveat(panel).
    "The database holds about two years (501 sessions, 2024-11-25 to 2026-11-24) and only one broad regime: a bull market. A hidden Markov model cannot be validated on crisis, recovery and quiet bull states it has never seen.",
    "The 200-day breadth feature, a causal z-score window and the minimum fit length mean the HMM starts well after the first session. Early dates fall back to the fixed filter; comparisons use a common window that starts when the HMM is live.",
    "Filtered probabilities at date t use parameters estimated only on data through t (rolling window, periodic refit). Smoothed probabilities are never used for allocation.",
    "With roughly 80–252 training days and 2–4 states, the emission estimates are noisy. A 4-state model is especially likely to overfit on this sample.",
    "HYG minus IEF is the credit proxy when both are in the panel; IEF/TLT are the bond overlay. Survivorship is still a concern for the stock-breadth statistics.",
    "Sleeve parameters are the system defaults (not the walk-forward grid). This isolates the regime model from a joint search that would overfit even faster.",
    "Costs are modeled, not live fills. Bond-overlay turnover can add spread and impact versus the fixed-filter book.",
]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="",
                        help="Append to output filenames, e.g. _20y writes regime_hmm_compare_20y.md")
    args = parser.parse_args(argv)
    suffix = args.suffix or ""
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading panel")
    panel = load_panel()
    log(f"panel {panel.dates[0].date()} to {panel.dates[-1].date()}  {len(panel.dates)} sessions, {panel.close.shape[1]} tickers")

    variants = [
        ("combined + fixed filters", "fixed", 3, False),
        ("combined + HMM (3 states)", "hmm", 3, False),
        ("combined + HMM (2 states)", "hmm", 2, False),
        ("combined + HMM (4 states)", "hmm", 4, False),
        ("combined + HMM (3 states, macro)", "hmm", 3, True),
    ]
    runs = {}
    policy_3 = None
    for name, model, n_states, include_macro in variants:
        log(f"strategy {name}")
        strategy = make_combined(model, n_states, include_macro=include_macro)
        weights, stops, result = run_strategy(panel, strategy, start=panel.dates[WARMUP - 1])
        runs[name] = {"strategy": strategy, "weights": weights, "stops": stops, "result": result}
        if name == "combined + HMM (3 states)":
            policy_3 = getattr(strategy, "_regime_policy", None)
            if policy_3 is None:
                policy_3 = regime_policy(panel, model="hmm", overrides={"n_states": 3})
            snap = latest_snapshot(policy_3)
            write_regime_state(snap)
            log(f"wrote regime snapshot as of {snap.get('asOf') if snap else '?'}")

    log("SPY benchmark")
    spy = simulate(panel, benchmark_weights(panel, "spy"), start=panel.dates[WARMUP - 1])
    runs["SPY buy and hold"] = {"result": spy}

    first_hmm = None
    if policy_3 is not None:
        live = policy_3.index[policy_3["source"] == "hmm"]
        if len(live):
            first_hmm = live[0]
    warmup_start = panel.dates[WARMUP]
    wf_start = panel.dates[min(252, len(panel.dates) - 2)]
    common_start = first_hmm if first_hmm is not None else warmup_start
    # Metrics should start on the first session the strategy is evaluated, which may be the next day after a decision.
    windows = {"common": common_start, "warmup": warmup_start, "walkForwardStart": wf_start}

    rows = {key: [] for key in windows}
    for name, payload in runs.items():
        result = payload["result"]
        for key, start in windows.items():
            start = max(start, result.returns.dropna().index[0]) if len(result.returns.dropna()) else start
            rows[key].append(_row(name, _perf(result, start)))

    occupancy = regime_occupancy(policy_3, common_start) if policy_3 is not None else {}
    latest = latest_snapshot(policy_3) if policy_3 is not None else None
    summary = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                 "sessions": len(panel.dates), "tickers": int(panel.close.shape[1])},
        "costModel": CostModel().as_dict(),
        "commonPeriod": {"start": str(pd.Timestamp(common_start).date()),
                         "end": str(panel.dates[-1].date())},
        "windows": {k: str(pd.Timestamp(v).date()) for k, v in windows.items()},
        "rows": rows,
        "occupancy": occupancy,
        "latest": latest,
        "caveats": merge_caveats([_data_caveat(panel), *CAVEATS[1:]], getattr(panel, "warnings", None)),
        "runSeconds": None,
    }
    summary["verdict"] = verdict_text(rows["common"])
    summary["runSeconds"] = round(time.time() - started, 1)

    curves = {name: payload["result"].equity.rename(name) for name, payload in runs.items()}
    equity_path = OUT_DIR / f"regime_hmm_equity{suffix}.csv"
    json_path = OUT_DIR / f"regime_hmm_compare{suffix}.json"
    md_path = OUT_DIR / f"regime_hmm_compare{suffix}.md"
    write_atomically(equity_path, pd.DataFrame(curves).to_csv().encode())
    write_atomically(json_path, json.dumps(summary, indent=2, default=str).encode())
    write_atomically(md_path, markdown(summary).encode())
    log(f"wrote {md_path} in {summary['runSeconds']}s")
    print()
    print(f"Common window {summary['commonPeriod']['start']} to {summary['commonPeriod']['end']}")
    print(_table(rows["common"]))
    print(summary["verdict"])
    return summary


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("regime_compare"):
        main()
