"""Daily paper runner for the volatility sleeve (primary rule) and the config-gated options overlay.

No broker calls of any kind: prices come from the local market-data panel and FRED VIX files, option prices are
MODELED. Every entry, exit (with its reason) and pending order is written to the decision chat
(Massive/chat_log.sqlite3, account "options-paper"), de-duplicated by source_key.

The book is rebuilt each run by replaying the same causal state machine from the paper start date (decisions at the
close, fills at the next open), so the paper account and the backtest can never drift apart.

Config: Massive/PaperOptions/config.json (created on first run)
  {"volSleeve": {"enabled": true, "start": "<first run date>", "equity": 10000, "params": {}},
   "overlay":   {"enabled": false, "start": null, "fractional": false, "params": {"mode": "bearish_puts", ...}}}
State:  Massive/PaperOptions/vol_sleeve_paper.json, overlay_paper.json

  python options_paper.py run [--refresh-vix] [--no-chat]
  python options_paper.py status
"""

import json
import math
import sys
from datetime import datetime

import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically

DIR = DATA_DIR / "PaperOptions"
CONFIG_PATH = DIR / "config.json"
SLEEVE_PATH = DIR / "vol_sleeve_paper.json"
OVERLAY_PATH = DIR / "overlay_paper.json"
ACCOUNT = "options-paper"
DEFAULT_CONFIG = {
    "volSleeve": {"enabled": True, "start": None, "equity": 10_000.0, "params": {}},
    "overlay": {"enabled": False, "start": None, "fractional": False,
                "params": {"mode": "bearish_puts", "trigger": "top", "top_k": 2, "structure": "put"}},
}
EXIT_RATIONALE = {
    "calm_low_vol": "VIXY entry: VIX in its low percentile, VIX below the level cap and the VIX/VIX3M curve in steep contango (calm).",
    "post_spike_fade": "SVIX entry: a volatility spike happened recently, contango is back and VIX has fallen from its 20-day peak.",
    "take_profit_spike": "VIXY take-profit: VIX spiked (z-score or 252-day percentile high).",
    "backwardation_rollback": "VIXY exit: the term structure flipped to backwardation and has rolled back (spike fading).",
    "max_hold": "Time exit: the maximum holding period was reached.",
    "decay_stop": "VIXY decay stop: VIXY fell the stop percentage below entry (contango roll decay).",
    "normalized": "SVIX take-profit: VIX percentile is back to normal and contango restored.",
    "regime_flip": "SVIX exit: volatility regime flipped (backwardation again or VIX z-score spike).",
    "macro_event": "SVIX exit before a major macro event (FOMC/CPI/NFP calendar).",
    "trailing_stop": "SVIX trailing stop from the highest close since entry.",
    "hard_stop": "SVIX hard stop hit intraday.",
    "hard_stop_gap": "SVIX opened through the hard stop: filled at the open.",
}


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_PATH.exists():
        saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        for k, v in saved.items():
            cfg[k] = {**cfg.get(k, {}), **v} if isinstance(v, dict) else v
    return cfg


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(obj, indent=1, default=str).encode())


def _chat_rows_sleeve(trades, pending, asof):
    rows = []
    for t in trades:
        base = f"optpaper:vol:{t['ticker']}:{t['entryDate']}"
        rows.append((base + ":in", dict(kind="entry", ticker=t["ticker"], side="buy", qty=t["shares"], price=t["entry"],
                    decision_date=t["entryDate"], text=f"VOL SLEEVE BUY {t['shares']:g} {t['ticker']} @ ${t['entry']:,.2f} "
                    f"({t['entryReason']}) on {t['entryDate']}.", rationale=EXIT_RATIONALE.get(t["entryReason"], t["entryReason"]),
                    data=t)))
        if t.get("exitDate"):
            kind = "stop" if "stop" in t["exitReason"] else "exit"
            rows.append((base + ":out", dict(kind=kind, ticker=t["ticker"], side="sell", qty=t["shares"], price=t["exit"],
                        decision_date=t["exitDate"], text=f"VOL SLEEVE EXIT {t['ticker']} @ ${t['exit']:,.2f} on {t['exitDate']} "
                        f"— reason: {t['exitReason']}. P&L ${t['pnl']:,.2f}.",
                        rationale=EXIT_RATIONALE.get(t["exitReason"], t["exitReason"]), data=t)))
    for o in pending:
        rows.append((f"optpaper:vol:pending:{asof}:{o['ticker']}:{o['action']}", dict(
            kind="order", ticker=o["ticker"], side=o["action"], decision_date=asof,
            text=f"VOL SLEEVE ORDER for next open: {o['action'].upper()} {o['ticker']} — reason: {o['reason']} "
                 f"(VIX {o['detail'].get('vix')}, VIX/VIX3M {o['detail'].get('ratio')}, pct {o['detail'].get('pct')}).",
            rationale=EXIT_RATIONALE.get(o["reason"], o["reason"]), data=o)))
    return rows


def _chat_rows_overlay(res, asof):
    rows = []
    labels = {"open": "OPEN", "close": "EXIT", "short": "SHORT", "cover": "COVER", "early_assignment_risk": "EARLY-ASSIGNMENT RISK",
              "pin_risk": "PIN RISK", "short_dividend_owed": "DIVIDEND OWED"}
    for e in res.get("events", []):
        ev = e.get("event")
        if ev not in labels:
            continue
        who = e.get("strategy") or e.get("ticker") or ""
        pnl = e.get("pnl")
        text = (f"OPTIONS OVERLAY {labels[ev]} {who} {e.get('underlying') or ''} on {e['date']}".replace("  ", " ")
                + (f" — reason: {e['reason']}" if e.get("reason") else "")
                + (f". P&L ${pnl:,.2f}" if isinstance(pnl, (int, float)) else "") + " (MODELED option prices).")
        rows.append((f"optpaper:ovl:{e['date']}:{ev}:{e.get('id') or e.get('ticker')}", dict(
            kind="exit" if ev in ("close", "cover") else "entry" if ev in ("open", "short") else "risk",
            ticker=e.get("underlying") or e.get("ticker"), decision_date=e["date"], text=text,
            rationale="Options overlay (config-gated paper, modeled prices).", data=e)))
    for t in res.get("trades", []):
        if "ticker" in t and "strategy" not in t:          # inverse-ETF sleeve trades
            rows.append((f"optpaper:ovl:etf:{t['ticker']}:{t['entryDate']}:out", dict(
                kind="exit", ticker=t["ticker"], decision_date=t["exitDate"],
                text=f"OPTIONS OVERLAY EXIT {t['ticker']} @ ${t['exit']:,.2f} on {t['exitDate']} — reason: {t['exitReason']}. "
                     f"P&L ${t['pnl']:,.2f}.", rationale="Inverse-ETF hedge exit.", data=t)))
    return rows


def write_chat(rows):
    from decision_feed import _insert, connect_chat
    chat = connect_chat()
    added = 0
    with chat:
        for key, m in rows:
            added += _insert(chat, key, account=ACCOUNT, **m)
    chat.close()
    return added


def run_sleeve(panel, cfg, asof):
    import vix_data
    import vol_sleeve
    vs = cfg["volSleeve"]
    vixf = vix_data.features(vix_data.load())
    st = {}
    rets, eqs, trades = vol_sleeve.simulate(panel, vixf, vs.get("params") or {}, start=vs["start"], equity0=float(vs["equity"]),
                                            state_out=st)
    vix_last = vixf.dropna(subset=["vix"]).iloc[-1]
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "asof": asof, "start": vs["start"],
           "equity": float(eqs.iloc[-1]) if len(eqs) else float(vs["equity"]), "cash": st.get("cash"),
           "positions": st.get("sleeve", {}).get("pos", {}), "hardStops": st.get("hardStops"), "pending": st.get("pending", []),
           "trades": trades, "vix": {"date": str(vixf.dropna(subset=["vix"]).index[-1].date()),
                                     **{k: (None if pd.isna(v) else float(v)) for k, v in vix_last.items()}},
           "vixStale": str(vixf.dropna(subset=["vix"]).index[-1].date()) < asof,
           "params": st.get("sleeve", {}).get("params"),
           "note": "Paper only. Real ETF prices; decisions at the close, fills at the next open; Schwab cost model."}
    save_json(SLEEVE_PATH, out)
    return out


def run_overlay(panel, cfg, asof):
    import options_research as orx
    from research import run_strategy
    from strategies import TimeSeriesMomentum
    import top_detector
    import vix_data
    oc = cfg["overlay"]
    start = pd.Timestamp(oc["start"])
    _, _, base = run_strategy(panel, TimeSeriesMomentum(), start=start, initial_equity=10_000.0, whole_shares=True, min_order_value=100.0)
    signals = top_detector.build_signals(panel)
    res = orx.apply_overlay(panel, base, signals, dict(oc.get("params") or {}), start, fractional=bool(oc.get("fractional")),
                            vix=vix_data.load()["vix"])
    out = {"generated": datetime.now().isoformat(timespec="seconds"), "asof": asof, "start": oc["start"],
           "params": oc.get("params"), "fractional": bool(oc.get("fractional")), "pricing": "MODELED " + res["pricing"],
           "overlayPnl": float(res["pnl"].iloc[-1]) if len(res["pnl"]) else 0.0, "trades": res["trades"],
           "events": res["events"][-200:], "skips": res["skips"][-50:],
           "note": "Shadow overlay on the core TSMOM $10k reference book. Option prices are MODELED, not quotes."}
    save_json(OVERLAY_PATH, out)
    return out, res


def run(refresh_vix=False, chat=True):
    from market_data import load_panel
    import vix_data
    if refresh_vix:
        try:
            vix_data.refresh()
        except Exception as error:  # noqa: BLE001 - stale VIX is reported, never fatal
            print(f"VIX refresh failed: {error}", file=sys.stderr)
    panel = load_panel()
    asof = str(panel.dates[-1].date())
    cfg = load_config()
    changed = False
    for k in ("volSleeve", "overlay"):
        if cfg[k].get("enabled") and not cfg[k].get("start"):
            cfg[k]["start"] = asof          # start today: no back-filled paper history
            changed = True
    if changed or not CONFIG_PATH.exists():
        save_json(CONFIG_PATH, cfg)
    report, rows = {"asof": asof}, []
    if cfg["volSleeve"].get("enabled"):
        s = run_sleeve(panel, cfg, asof)
        rows += _chat_rows_sleeve(s["trades"], s["pending"], asof)
        report["volSleeve"] = {"equity": s["equity"], "positions": s["positions"], "pending": s["pending"], "vix": s["vix"],
                               "vixStale": s["vixStale"]}
    if cfg["overlay"].get("enabled"):
        o, res = run_overlay(panel, cfg, asof)
        rows += _chat_rows_overlay(res, asof)
        report["overlay"] = {"pnl": o["overlayPnl"], "openTrades": len([t for t in o["trades"] if not (t.get("exitDate") or t.get("closed"))])}
    if chat and rows:
        report["chatAdded"] = write_chat(rows)
    return report


def status():
    out = {"config": load_config()}
    for name, path in (("volSleeve", SLEEVE_PATH), ("overlay", OVERLAY_PATH)):
        out[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "run"
    if cmd == "run":
        print(json.dumps(run(refresh_vix="--refresh-vix" in argv, chat="--no-chat" not in argv), indent=1, default=str))
    elif cmd == "status":
        print(json.dumps(status(), indent=1, default=str))
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
