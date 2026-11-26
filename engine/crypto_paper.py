"""20 crypto-only $10,000 PAPER shadow accounts (ids 200-219). Simulation only: no broker is ever called.

SAFETY: this module never imports an order client and never calls Schwab. Prices come from Coinbase public candles
(crypto_data.py); the Schwab market-data client is only used by crypto_data.py for quotes / history. The live
paper config (plain TSMOM) is not touched; these accounts live under Massive/Research/crypto/paper/.

Each account runs one variant from crypto_research.py's design (same signal + simulator code, Schwab Crypto fee
0.75% per side + 5 bp slippage, long-only spot, cash earns 0). Forward mode is a deterministic replay from
$10,000 at FORWARD_START (the first day after the sealed holdout); every rule is causal, so this equals trading
one UTC day at a time. New daily rows are appended to paper/ledger.jsonl (append-only; a changed earlier value
from a data revision is counted, not rewritten). TRADING_CRYPTO_PAPER=off skips the daily step.

  python crypto_paper.py run [--fetch]     # optionally refresh data first, then advance all 20 accounts
  python crypto_paper.py status
"""

import argparse
import json
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from crypto_data import COINS, CRYPTO_DIR, load_drivers, load_spot, write_json
import crypto_research as cr

PAPER_DIR = CRYPTO_DIR / "paper"
STATUS_PATH = PAPER_DIR / "crypto_paper_status.json"
LEDGER_PATH = PAPER_DIR / "ledger.jsonl"
EQUITY_CSV = PAPER_DIR / "forward_equity.csv"
FORWARD_START = "2026-11-25"
INITIAL = cr.INITIAL
ID_RANGE = range(200, 220)

TSMOM_BTC = {"base": "tsmom", "universe": "btc", "lookbacks": [100], "weighting": "ew", "rebalance": "daily", "band": 0.05}
TSMOM_S5 = {"base": "tsmom", "universe": "schwab", "lookbacks": [100], "weighting": "ew", "rebalance": "daily", "band": 0.05}
XS_S5 = {"base": "xsmom", "universe": "schwab", "lookback": 30, "k": 2, "rebalance": "weekly", "band": 0.10}
HOLD_S5 = {"base": "hold", "universe": "schwab", "band": 0.10}
SMA200_BTC = {"base": "sma", "universe": "btc", "n": 200, "band": 0.05}


def _with(base, overlay):
    return {**base, "overlays": [overlay]}


ACCOUNTS = [
    (200, "BTC buy & hold (baseline)", {"base": "hold", "universe": "btc", "band": 0.05}),
    (201, "BTC/ETH 60/40 buy & hold", {"base": "hold", "universe": "btc_eth", "weights": {"BTC": 0.6, "ETH": 0.4}, "band": 0.10}),
    (202, "Schwab-5 equal-weight buy & hold", HOLD_S5),
    (203, "BTC trend: 100-day TSMOM", TSMOM_BTC),
    (204, "BTC trend: above 200-day SMA", SMA200_BTC),
    (205, "BTC/ETH TSMOM 20/60/120 inverse-vol, weekly (research champion)",
     {"base": "tsmom", "universe": "btc_eth", "lookbacks": [20, 60, 120], "weighting": "invvol", "rebalance": "weekly", "band": 0.10}),
    (206, "BTC/ETH above 50-day SMA", {"base": "sma", "universe": "btc_eth", "n": 50, "band": 0.05}),
    (207, "Schwab-5 TSMOM 100-day", TSMOM_S5),
    (208, "Schwab-5 cross-sectional momentum top-2 (30d)", XS_S5),
    (209, "BTC buy & hold, 40% vol target", {"base": "hold", "universe": "btc", "voltarget": 0.4, "band": 0.10}),
    (210, "Schwab-5 TSMOM 100-day, 50% vol target",
     {"base": "tsmom", "universe": "schwab", "lookbacks": [100], "weighting": "ew", "voltarget": 0.5, "band": 0.10}),
    (211, "BTC dip-buy (7d < -20% in uptrend, hold 15d)",
     {"base": "meanrev", "universe": "btc", "window": 7, "thr": 0.20, "hold": 15, "band": 0.05}),
    (212, "Dollar-aware: BTC TSMOM, halve when UUP 90d up", _with(TSMOM_BTC, {"type": "dollar_trend", "lookback": 90, "cut": 0.5})),
    (213, "Dollar-aware: BTC TSMOM, cash 10d after dollar shock",
     _with(TSMOM_BTC, {"type": "dollar_shock", "z": 2.0, "days": 10, "cut": 0.0})),
    (214, "Macro-aware: Schwab-5 hold, dollar+real-yield+liquidity composite", _with(HOLD_S5, {"type": "composite", "floor": 0.25})),
    (215, "Liquidity-aware: Schwab-5 hold, halve when net liquidity 91d down",
     _with(HOLD_S5, {"type": "liquidity", "lookback": 91, "cut": 0.5})),
    (216, "Dollar-aware: Schwab-5 x-sec momentum, halve when broad dollar 90d up",
     _with(XS_S5, {"type": "dtwex_trend", "lookback": 90, "cut": 0.5})),
    (217, "Real-yield-aware: BTC SMA200, halve when real yield 90d up",
     _with(SMA200_BTC, {"type": "real_yield", "lookback": 90, "cut": 0.5})),
    (218, "Dollar regime: BTC only when UUP 120d trend down", {"base": "dollar_regime", "universe": "btc", "lookback": 120, "band": 0.05}),
    (219, "Dollar-aware: BTC hold, halve when UUP 90d up",
     _with({"base": "hold", "universe": "btc", "band": 0.05}, {"type": "dollar_trend", "lookback": 90, "cut": 0.5})),
]


def accounts():
    out = []
    for i, label, spec in ACCOUNTS:
        coins = cr.UNIVERSES[spec.get("universe", "btc")]
        out.append({"id": i, "label": label, "spec": spec, "coins": coins,
                    "schwabTradableNow": all(COINS[c]["schwab"] == "spot" for c in coins),
                    "dollarAware": bool(spec.get("overlays")) or spec["base"] == "dollar_regime",
                    "detail": cr.describe(spec)})
    return out


def log(msg):
    print(f"[crypto_paper] {msg}", file=sys.stderr, flush=True)


def _append_ledger(equity):
    seen = {}
    if LEDGER_PATH.exists():
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            seen[(row["date"], row["id"])] = row["equity"]
    new, revisions = [], 0
    for day, values in equity.iterrows():
        d = str(pd.Timestamp(day).date())
        for i, v in values.items():
            key = (d, int(i))
            if key in seen:
                revisions += int(abs(seen[key] - float(v)) > 0.01)
                continue
            new.append(json.dumps({"date": d, "id": int(i), "equity": round(float(v), 2)}))
    if new:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "a", encoding="utf-8") as fh:
            fh.write("\n".join(new) + "\n")
    return revisions


def _research_summary():
    if not cr.REPORT_JSON.exists():
        return None
    try:
        rep = json.loads(cr.REPORT_JSON.read_text(encoding="utf-8"))
    except Exception:   # noqa: BLE001
        return None
    dp = rep.get("dollarPairs", {})
    return {"designHash": rep.get("designHash"), "generatedAt": rep.get("generatedAt"),
            "champion": rep.get("champion"), "btcHold": rep.get("btcHold"),
            "pbo": (rep.get("pbo") or {}).get("pbo"), "spaVsBtcP": (rep.get("spaVsBtc") or {}).get("spaP"),
            "dsr": (rep.get("deflatedSharpe") or {}).get("dsrProbability"),
            "walkForward": (rep.get("walkForward") or {}).get("stitched"),
            "dollarPairsSpaP": (dp.get("spa") or {}).get("spaP"), "dollarFamilies": dp.get("families"),
            "holdoutLook": {k: v for k, v in (rep.get("holdoutLook") or {}).items() if k != "allVariants"} or None}


def _with_market(status):
    """Keep the latest market snapshot (crypto_snapshot.py) in the status file served at /api/crypto/status."""
    snap_path = CRYPTO_DIR / "crypto_snapshot.json"
    try:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        status["market"] = {k: snap.get(k) for k in ("asOf", "coins", "macro", "nowcast", "schwab", "verdict",
                                                     "universe", "short", "context")}
    except Exception:   # noqa: BLE001 - snapshot is optional
        pass
    return status


def _short_accounts():
    """SBIT shadow experiments 220-223 (crypto_short.py: market-hours ETF model). Paper only; never breaks the run."""
    try:
        import crypto_short
        rows = crypto_short.paper(write=True).get("accounts") or []
        for r in rows:
            r.setdefault("detail", "SBIT (-2x inverse BTC ETF) fills only in US market hours; shadow experiment")
            r.setdefault("coins", ["BTC", "ETH"])
            r.setdefault("preHoldoutBacktest", None)
        return rows
    except Exception as error:   # noqa: BLE001
        log(f"short shadow accounts skipped: {type(error).__name__}: {error}")
        return []


def run(fetch=False):
    started = time.time()
    if fetch:
        try:
            import crypto_data
            crypto_data.update()
        except Exception as error:   # noqa: BLE001 - never break the daily pipeline on a fetch problem
            log(f"fetch failed: {type(error).__name__}: {error}")
    close = load_spot()
    drivers = load_drivers(close.index)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    accts = accounts()
    status = {"updatedAt": datetime.now().strftime("%Y-%m-%d %H:%M"), "forwardStart": FORWARD_START,
              "dataEnd": str(close.index[-1].date()), "initialEquity": INITIAL,
              "fees": {"perSide": cr.FEE, "slippage": cr.SLIP},
              "note": ("PAPER shadow accounts only: simulated fills at the UTC daily close on Coinbase prices with "
                       "Schwab Crypto fees. No orders or previews are ever sent. Schwab trades BTC and ETH today; "
                       "SOL/AVAX/LINK are announced, so 'Schwab-5' accounts are not fully tradable yet.")}
    research = None
    if cr.RETURNS_CSV.exists():
        research = pd.read_csv(cr.RETURNS_CSV, parse_dates=["date"]).set_index("date")
    # map account specs to research variant ids using the design saved with the returns (ids shift when
    # nowcast variants are added), falling back to the base design
    saved = None
    if cr.REPORT_JSON.exists():
        try:
            saved = json.loads(cr.REPORT_JSON.read_text(encoding="utf-8")).get("design")
        except Exception:   # noqa: BLE001
            saved = None
    design = {cr._key({k: v for k, v in s.items() if k not in ("id", "label", "pairBase")}): s["id"]
              for s in (saved or cr.make_design())} if research is not None else {}
    if close.index[-1] < pd.Timestamp(FORWARD_START):
        status.update({"status": "awaiting", "accounts": [{k: a[k] for k in ("id", "label", "detail")} for a in accts]})
        write_json(STATUS_PATH, _with_market(status))
        return status
    equities, table = {}, []
    for a in accts:
        W = cr.target_weights(a["spec"], close, drivers)
        eq, st = cr.simulate(close, W, FORWARD_START, fee=cr.FEE, slip=cr.SLIP, band=a["spec"].get("band", 0.05),
                             rebalance=a["spec"].get("rebalance", "daily"))
        equities[a["id"]] = eq
        last_w = W.iloc[-1]
        rid = design.get(cr._key({k: v for k, v in a["spec"].items()}))
        pre = None
        if research is not None and rid is not None and str(rid) in research.columns:
            r = research[str(rid)].loc[:pd.Timestamp(cr.HOLDOUT_START) - pd.Timedelta(days=1)]
            pre = cr.perf(r)
        rets = cr.returns_from_equity(eq, INITIAL)
        table.append({"id": a["id"], "label": a["label"], "detail": a["detail"], "coins": a["coins"],
                      "schwabTradableNow": a["schwabTradableNow"], "dollarAware": a["dollarAware"],
                      "researchVariant": rid, "equity": float(eq.iloc[-1]),
                      "totalReturn": float(eq.iloc[-1] / INITIAL - 1),
                      "maxDrawdown": float((eq / eq.cummax() - 1).min()),
                      "sharpe": cr.sharpe(rets) if len(rets) >= 20 else None,
                      "targetWeights": {c: round(float(w), 4) for c, w in last_w.items() if w > 1e-4},
                      "cash": round(float(max(0.0, 1 - last_w.sum())), 4), "feesPaidPctPerYear": st["costPerYear"],
                      "preHoldoutBacktest": pre})
    table.extend(_short_accounts())
    eq_frame = pd.DataFrame(equities)
    eq_frame.to_csv(EQUITY_CSV, index_label="date")
    revisions = _append_ledger(eq_frame)
    table.sort(key=lambda r: -r["totalReturn"])
    btc = next(r for r in table if r["id"] == 200)
    status.update({"status": "tracking", "days": int(len(eq_frame)), "leader": table[0]["id"], "accounts": table,
                   "btcBaselineReturn": btc["totalReturn"], "revisions": revisions, "research": _research_summary(),
                   "seconds": round(time.time() - started, 1)})
    write_json(STATUS_PATH, _with_market(status))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--fetch", action="store_true", help="refresh Coinbase/FRED/Schwab market data first")
    args = parser.parse_args(argv)
    if args.command == "run":
        try:
            st = run(args.fetch)
        except Exception as error:   # noqa: BLE001 - never break the daily paper pipeline
            log(f"run failed: {type(error).__name__}: {error}")
            print(json.dumps({"status": "error", "error": f"{type(error).__name__}: {error}"}))
            return 1
        print(json.dumps({k: st.get(k) for k in ("status", "forwardStart", "dataEnd", "days", "leader", "seconds")}))
    else:
        print(STATUS_PATH.read_text(encoding="utf-8") if STATUS_PATH.exists() else json.dumps({"status": "no_results"}))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
