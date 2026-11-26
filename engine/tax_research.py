"""Compare pre-tax, after-tax (same trades) and tax-aware after-tax results.

Writes Massive/Research/tax_comparison.json and tax_comparison.md (gitignored).

Honesty: the local database is only about two years long, so few lots ever become
long-term, harvest/defer effects are limited, and tax drag is mostly short-term
gains plus dividends. These figures are descriptive, not a forecast. This is a
planning model, not tax advice.

Usage (from engine/):
  python tax_research.py
  python tax_research.py --family tsmom
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from costs import CostModel
from fetch_and_model import DATA_DIR, write_atomically
from market_data import load_panel
from metrics import summarize
from portfolio_sim import simulate
from tax import TaxConfig, summarize_tax_result

OUT_DIR = DATA_DIR / "Research"
DISCLAIMER = "Planning estimate only; not tax advice."
NOTE_2Y = (
    "Only about two years of daily history are available. Long-term holding-period "
    "effects, multi-year loss carryforwards and a full tax-aware edge are not identifiable "
    "in this window: many positions never reach the one-year anniversary, and tax drag is "
    "dominated by short-term gains and dividends."
)


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", file=sys.stderr, flush=True)


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)[:10]
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    return str(value)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        import numpy as np
        if isinstance(value, (np.floating,)):
            return None if not math.isfinite(float(value)) else float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
    except Exception:
        pass
    return value


def _run(panel, weights, stops, cost_model, tax_config, sleeve_of, initial_equity, start):
    return simulate(panel, weights, stops, cost_model, initial_equity, execution="next_open",
                    start=start, sleeve_of=sleeve_of, tax_config=tax_config)


def compare(panel, family="combined", initial_equity=10_000.0, out_dir=None, start=None):
    from research import WARMUP, make_strategy, run_strategy

    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy = make_strategy(family)
    log(f"building {family} weights")
    weights, stops, _ = run_strategy(panel, strategy)
    start = pd.Timestamp(start) if start is not None else panel.dates[min(WARMUP - 1, len(panel.dates) - 1)]
    cost_model = CostModel.for_broker("schwab")
    sleeve_of = getattr(strategy, "sleeve_of", None)
    common = dict(panel=panel, weights=weights, stops=stops, cost_model=cost_model,
                  sleeve_of=sleeve_of, initial_equity=initial_equity, start=start)

    log("pre-tax simulation")
    pre = _run(**common, tax_config=TaxConfig())
    log("after-tax accounting (same trades)")
    accounted = _run(**common, tax_config=TaxConfig.accounting())
    log("tax-aware policy (defer / harvest / wash avoidance)")
    aware = _run(**common, tax_config=TaxConfig.aware())

    pre_s = summarize(pre, start=start)
    acc_s = summarize_tax_result(accounted, start=start)
    aw_s = summarize_tax_result(aware, start=start)

    def pack(name, result, summary, policy):
        report = getattr(result, "tax_report", None) or {}
        last_pre = float(result.equity.iloc[-1]) if len(result.equity) else None
        after_eq = getattr(result, "after_tax_equity", None)
        last_at = float(after_eq.iloc[-1]) if after_eq is not None and len(after_eq) else last_pre
        return {
            "label": name,
            "policy": policy,
            "summary": summary,
            "finalEquityPreTax": last_pre,
            "finalEquityAfterTax": last_at,
            "taxBill": report.get("taxBill"),
            "realized": report.get("taxBill", {}).get("years") if report else None,
        }

    payload = {
        "generatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "disclaimer": DISCLAIMER,
        "note": NOTE_2Y,
        "family": family,
        "initialEquity": initial_equity,
        "data": {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                 "metricStart": str(pd.Timestamp(start).date()), "sessions": int(len(panel.dates)),
                 "tickers": int(panel.close.shape[1])},
        "rates": TaxConfig.accounting().as_dict(),
        "preTax": pack("pre-tax (no tax cash flows, no policy)", pre, pre_s, "off"),
        "afterTaxSameTrades": pack("same trades, taxes reserved / paid", accounted, acc_s, "account"),
        "taxAwareAfterTax": pack("tax-aware trades, taxes reserved / paid", aware, aw_s, "advise"),
    }
    pre_cagr = pre_s.get("performance", {}).get("cagr")
    acc_cagr = (acc_s.get("afterTax") or {}).get("afterTax", {}).get("cagr")
    aw_cagr = (aw_s.get("afterTax") or {}).get("afterTax", {}).get("cagr")
    payload["comparison"] = {
        "preTaxCagr": pre_cagr,
        "sameTradesAfterTaxCagr": acc_cagr,
        "taxAwareAfterTaxCagr": aw_cagr,
        "taxDragCagrSameTrades": None if pre_cagr is None or acc_cagr is None else pre_cagr - acc_cagr,
        "taxDragCagrTaxAware": None if pre_cagr is None or aw_cagr is None else pre_cagr - aw_cagr,
        "afterTaxCagrLiftFromPolicy": None if acc_cagr is None or aw_cagr is None else aw_cagr - acc_cagr,
        "preTaxSharpe": pre_s.get("performance", {}).get("sharpe"),
        "sameTradesAfterTaxSharpe": (acc_s.get("afterTax") or {}).get("afterTax", {}).get("sharpe"),
        "taxAwareAfterTaxSharpe": (aw_s.get("afterTax") or {}).get("afterTax", {}).get("sharpe"),
    }

    write_atomically(out_dir / "tax_comparison.json",
                     json.dumps(_clean(payload), indent=2, default=_json_default, allow_nan=False).encode())
    write_atomically(out_dir / "tax_comparison.md", markdown(payload).encode())
    if len(pre.equity):
        frame = pd.DataFrame({
            "preTaxEquity": pre.equity,
            "sameTradesAfterTaxEquity": getattr(accounted, "after_tax_equity", accounted.equity),
            "taxAwareAfterTaxEquity": getattr(aware, "after_tax_equity", aware.equity),
        })
        write_atomically(out_dir / "tax_comparison_equity.csv", frame.to_csv(index=True).encode())
    log(f"wrote {out_dir / 'tax_comparison.json'}")
    return payload


def _pct(value):
    if value is None:
        return "–"
    return f"{value * 100:.1f}%"


def _num(value):
    if value is None:
        return "–"
    return f"{value:.2f}"


def markdown(payload):
    c = payload["comparison"]
    lines = [
        "# Pre-tax vs tax-aware (US taxable account)",
        "",
        payload["disclaimer"],
        "",
        payload["note"],
        "",
        f"Strategy: `{payload['family']}` · starting equity ${payload['initialEquity']:,.0f} · "
        f"data {payload['data']['start']} to {payload['data']['end']} "
        f"({payload['data']['sessions']} sessions) · metrics from {payload['data']['metricStart']}.",
        "",
        "Default rates used for the estimate: federal short-term 32%, long-term 15%, NIIT 0%, state 0%. "
        "Lot method FIFO unless the tax-aware run identifies lots (HIFO / specific for discretionary harvests "
        "still uses the configured method; default FIFO).",
        "",
        "| Run | CAGR | Sharpe | Final equity (pre-tax mark) | Final equity (after tax) | Tax drag (CAGR) |",
        "|---|---|---|---|---|---|",
    ]

    def row(label, block, drag):
        perf = block["summary"].get("performance", {})
        after = (block["summary"].get("afterTax") or {}).get("afterTax") or {}
        cagr = after.get("cagr") if after else perf.get("cagr")
        sharpe = after.get("sharpe") if after else perf.get("sharpe")
        if label.startswith("pre-tax"):
            cagr, sharpe = perf.get("cagr"), perf.get("sharpe")
        lines.append(
            f"| {label} | {_pct(cagr)} | {_num(sharpe)} | "
            f"${block['finalEquityPreTax']:,.0f} | "
            f"${(block['finalEquityAfterTax'] or 0):,.0f} | {_pct(drag)} |"
        )

    row("pre-tax (no tax)", payload["preTax"], None)
    row("same trades, after tax", payload["afterTaxSameTrades"], c["taxDragCagrSameTrades"])
    row("tax-aware, after tax", payload["taxAwareAfterTax"], c["taxDragCagrTaxAware"])
    lines += [
        "",
        f"Policy lift (tax-aware after-tax CAGR minus same-trades after-tax CAGR): "
        f"{_pct(c['afterTaxCagrLiftFromPolicy'])}.",
        "",
        "## How to read this",
        "",
        "- **Pre-tax** is the existing engine: no tax cash flows and no tax-motivated trade changes.",
        "- **Same trades, after tax** keeps those trades and reserves (then pays at year-end) federal tax on "
        "realized short- and long-term gains, qualified dividends and ordinary dividends.",
        "- **Tax-aware** additionally defers weak trims of lots that are close to going long-term, harvests "
        "material losses into a correlated but not substantially identical ETF when one is in the universe, "
        "and avoids buying back a name (or an identical ETF) inside the 30-day wash window.",
        "- Stops and other risk exits are never deferred.",
        "",
        DISCLAIMER,
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", default="combined", choices=["combined", "tsmom", "meanrev"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--metric-start")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    started = time.time()
    log("loading panel")
    panel = load_panel(args.start, args.end)
    payload = compare(panel, args.family, args.initial_equity, args.out, start=args.metric_start)
    payload["runSeconds"] = round(time.time() - started, 1)
    print(json.dumps(_clean({"comparison": payload["comparison"], "data": payload["data"],
                             "disclaimer": payload["disclaimer"], "note": payload["note"],
                             "runSeconds": payload["runSeconds"]}), indent=2, default=_json_default))
    return payload


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("tax_research"):
        main()
