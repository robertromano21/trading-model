"""Turn every paper-trader decision into an AI-chat entry (Massive/chat_log.sqlite3, gitignored).

Sources (read-only): paper_trading.sqlite3 tables runs, decision_log (swing: signals, orders, fills, skips, blocks,
stops, exits, write-offs, advisory effects), dt_runs, dt_trades (day-trade entries/exits incl. stop/target/close),
dt_daily (signals skipped for PDT / slots / circuit breaker), plus optional intraday "watch" JSON (provisional
signals for today's unfinished session). Each entry records ticker, qty, price, a plain-English rationale and the
AI advisory verdict in force for that ticker at decision time (advisory_verdicts table / advisory_verdicts.json).
Idempotent: every source row has a unique key, so the feed can run after every job.

  python decision_feed.py                    # ingest new decisions
  python decision_feed.py --watch-json FILE  # add provisional intraday signals (daytrader run --dry-run --include-today)
"""

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from fetch_and_model import DATA_DIR

PAPER_DB = DATA_DIR / "paper_trading.sqlite3"
CHAT_DB = DATA_DIR / "chat_log.sqlite3"
ADVISORY_JSON = DATA_DIR / "Research" / "advisory_verdicts.json"
SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, role TEXT NOT NULL, kind TEXT NOT NULL,
  account TEXT, ticker TEXT, side TEXT, qty REAL, price REAL, decision_date TEXT, text TEXT NOT NULL,
  rationale TEXT, verdict TEXT, data TEXT, source_key TEXT UNIQUE, parent_id INTEGER);
CREATE INDEX IF NOT EXISTS chat_by_ticker ON chat_messages (ticker, id);
CREATE INDEX IF NOT EXISTS chat_by_kind ON chat_messages (kind, id);
"""

RATIONALE = {
    "entry": "Strategy signal turned on. Size = volatility-targeted (risk-parity) weight under the per-name, gross and position caps.",
    "exit": "Strategy signal turned off, so the target weight went to 0.",
    "rebalance": "Weight drifted from its target by more than the 2% minimum trade weight.",
    "stop": "ATR protective stop hit; re-entry is blocked until the strategy's target resets to 0.",
    "stop_gap": "Opened through the ATR stop (gap): filled at the open; re-entry blocked until the target resets.",
    "stale_exit": "No price for more than 5 sessions (halt/delisting): written off at the last close.",
    "below_one_share": "The target position is worth less than one whole share in the $10k account: skipped.",
    "below_min_order": "Order value under the $100 minimum order size: skipped.",
    "risk_limit": "Blocked by the risk engine (max gross, max name weight or max positions).",
    "circuit_breaker": "Blocked: the daily-loss circuit breaker is active, no new risk until it resets.",
    "corporate_action": "Corporate-action day masked: booked as a cash adjustment, not a trade.",
    "advisory_red": "AI verdict RED: new entry/increase blocked (advisory apply mode). The risk engine stays final.",
    "advisory_below_band": "Price below the AI entry band: entry deferred.",
    "advisory_above_band": "Price above the AI entry band (overextended): entry deferred.",
    "advisory_yellow": "AI verdict YELLOW: order size scaled by the yellow factor (0.5).",
    "orb_entry": "Opening-range breakout: price broke the opening-range high with the trend/volatility filters on.",
    "orb_stop": "Day trade stopped out at the opening-range stop.",
    "orb_target": "Day trade hit its R-multiple profit target.",
    "orb_close": "Day trade closed before the bell (flat by the close).",
    "orb_breaker": "Day trade closed by the daily-loss circuit breaker.",
}
DT_REASON = {"stop": "orb_stop", "target": "orb_target", "close": "orb_close", "breaker": "orb_breaker"}


def connect_chat(path=CHAT_DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    db.row_factory = sqlite3.Row
    return db


def _ro(path):
    if not Path(path).is_file():
        return None
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    db.row_factory = sqlite3.Row
    return db


def _has(db, name):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


class Verdicts:
    """AI advisory verdict in force for a ticker on a date (latest with asof <= date)."""

    def __init__(self, paper):
        self.rows = {}
        if paper is not None and _has(paper, "advisory_verdicts"):
            cols = [r[1] for r in paper.execute("PRAGMA table_info(advisory_verdicts)")]
            for r in paper.execute("SELECT * FROM advisory_verdicts"):
                d = dict(zip(cols, r))
                if d.get("ticker"):
                    self.rows.setdefault(d["ticker"].upper(), []).append(d)
        try:
            doc = json.loads(ADVISORY_JSON.read_text(encoding="utf-8"))
            for t, v in (doc.get("verdicts") or {}).items():
                self.rows.setdefault(t.upper(), []).append({"ticker": t.upper(), **v, "asof": v.get("asof") or doc.get("asof")})
        except (OSError, ValueError):
            pass

    def at(self, ticker, date):
        options = [v for v in self.rows.get((ticker or "").upper(), []) if not date or str(v.get("asof") or "") <= str(date)]
        if not options:
            return None
        v = max(options, key=lambda x: str(x.get("asof") or ""))
        keep = ("flag", "conviction", "entry_low", "entry_high", "stop", "overextension", "action", "size_scale", "asof", "source")
        return {k: v.get(k) for k in keep if v.get(k) is not None}


def _verdict_text(v):
    if not v:
        return "No AI verdict in force (no change to the risk engine's decision)."
    band = f", entry band {v.get('entry_low')}–{v.get('entry_high')}" if v.get("entry_low") is not None else ""
    stop = f", stop {v.get('stop')}" if v.get("stop") is not None else ""
    return f"AI verdict {v.get('flag')} (conviction {v.get('conviction')}{band}{stop}, as of {v.get('asof')}, {v.get('source')})."


def _money(x):
    return "–" if x is None else f"${x:,.2f}"


def _insert(chat, key, **m):
    m.setdefault("at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    m.setdefault("role", "trader")
    for field in ("verdict", "data"):
        if isinstance(m.get(field), (dict, list)):
            m[field] = json.dumps(m[field], default=str)
    cols = ["source_key", *m.keys()]
    cur = chat.execute(f"INSERT OR IGNORE INTO chat_messages ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                       [key, *m.values()])
    return cur.rowcount


def ingest(paper_path=PAPER_DB, chat_path=CHAT_DB):
    paper = _ro(paper_path)
    if paper is None:
        return {"added": 0, "note": "no paper_trading.sqlite3 yet"}
    chat = connect_chat(chat_path)
    verdicts = Verdicts(paper)
    added = 0
    with closing(paper), closing(chat), paper, chat:
        if _has(paper, "runs"):
            for r in paper.execute("SELECT * FROM runs ORDER BY run_id"):
                diag = json.loads(r["diagnostics"] or "{}")
                strat = diag.get("strategy", {})
                mr = strat.get("meanrev", {})
                regime = ""
                if mr:
                    regime = (f" Mean-reversion regime {'ON' if mr.get('regimeOn') else 'OFF'} (SPY vol percentile "
                              f"{(mr.get('marketVolPercentile') or 0) * 100:.0f}%, trend t {mr.get('marketTrendT', 0):.2f}).")
                hmm = strat.get("regime") or {}
                if hmm:
                    probs = ", ".join(f"{k} {v:.0%}" for k, v in (hmm.get("probabilities") or {}).items() if v is not None)
                    alloc = hmm.get("allocation") or {}
                    regime += (f" HMM regime {hmm.get('regime') or hmm.get('model')} ({probs or 'n/a'})."
                               f" TSMOM scale {(alloc.get('tsmomScale') or 0):.0%}, IEF {(alloc.get('ief') or 0):.0%}, "
                               f"TLT {(alloc.get('tlt') or 0):.0%}, mean-rev {'ON' if alloc.get('meanrevOn') else 'OFF'}.")
                tsm = strat.get("tsmom", {})
                text = (f"Swing paper run #{r['run_id']} ({r['account']}) as of {r['asof']}: {r['message'] or r['status']}. "
                        f"Equity {_money(r['equity'])}, cash {_money(r['cash'])}, gross {(r['gross'] or 0) * 100:.0f}%."
                        f"{regime}{' TSMOM active signals: ' + str(tsm.get('activeSignals')) + '.' if tsm else ''}")
                adv = diag.get("advisory") or {}
                rationale = (f"Advisory mode {adv.get('mode', 'log')}, {adv.get('verdicts', 0)} verdict(s) loaded. "
                             f"Reconciliation vs shadow backtest: {json.dumps(diag.get('reconciliation', {}))}")
                added += _insert(chat, f"run:{r['account']}:{r['run_id']}", at=r["finished_at"] or r["started_at"], kind="run",
                                 account=r["account"], decision_date=r["asof"], text=text, rationale=rationale, data=diag)
        if _has(paper, "decision_log"):
            for r in paper.execute("SELECT rowid AS rid, * FROM decision_log ORDER BY rowid"):
                reason, status, action = r["reason"] or "", r["status"] or "", r["action"] or ""
                kind = ("fill" if status == "filled" and reason not in ("stop", "stop_gap", "stale_exit") else
                        "stop" if reason.startswith("stop") else "exit" if reason in ("exit", "stale_exit") and status != "pending" else
                        "skip" if status == "skipped" else "blocked" if status == "blocked" else
                        "advisory" if status == "advisory_applied" else "order" if status == "pending" else status or "decision")
                qty = abs(r["shares"] or 0)
                price = r["fill_price"] if r["fill_price"] is not None else r["ref_price"]
                verb = {"fill": "FILLED", "stop": "STOP", "exit": "EXIT", "skip": "SKIPPED", "blocked": "BLOCKED",
                        "advisory": "ADVISORY", "order": "ORDER"}.get(kind, kind.upper())
                weights = ""
                if r["target_weight"] is not None:
                    weights = f" target {r['target_weight'] * 100:.1f}% (was {(r['current_weight'] or 0) * 100:.1f}%)"
                v = verdicts.at(r["ticker"], r["decision_date"])
                cost_note = f", cost {r['cost_bps']:.1f} bps" if r["cost_bps"] is not None else ""
                text = (f"{verb} {action.upper()} {qty:g} {r['ticker']} @ {_money(price)}"
                        f"{' on ' + str(r['fill_date']) if r['fill_date'] else ''} [{r['sleeve'] or '?'} sleeve, {reason}]{weights}"
                        f"{cost_note}.")
                added += _insert(chat, f"decision:{r['account']}:{r['rid']}", kind=kind, account=r["account"], ticker=r["ticker"],
                                 side=action, qty=qty, price=price, decision_date=r["decision_date"], text=text,
                                 rationale=f"{RATIONALE.get(reason, reason)} {_verdict_text(v)}", verdict=v,
                                 data={k: r[k] for k in r.keys() if k != "rid"})
        if _has(paper, "dt_runs"):
            for r in paper.execute("SELECT * FROM dt_runs ORDER BY run_id"):
                if r["mode"] != "run":
                    continue
                added += _insert(chat, f"dtrun:{r['account']}:{r['run_id']}", at=r["started_at"], kind="run", account=r["account"],
                                 text=f"Day-trading paper run #{r['run_id']} ({r['account']}): {r['sessions']} session(s) processed. {r['message'] or ''}".strip())
        if _has(paper, "dt_trades"):
            for t in paper.execute("SELECT * FROM dt_trades ORDER BY session_date, entry_minute"):
                v = verdicts.at(t["symbol"], t["session_date"])
                hm = lambda m: f"{int(m) // 60:02d}:{int(m) % 60:02d}" if m is not None else "?"
                side = "BUY" if (t["direction"] or 1) > 0 else "SHORT"
                base = f"dt:{t['account']}:{t['session_date']}:{t['symbol']}:{t['entry_minute']}"
                added += _insert(chat, base + ":in", kind="entry", account=t["account"], ticker=t["symbol"], side=side.lower(),
                                 qty=t["shares"], price=t["entry_price"], decision_date=t["session_date"],
                                 text=f"DAY TRADE {side} {t['shares']:g} {t['symbol']} @ {_money(t['entry_price'])} at {hm(t['entry_minute'])} ET "
                                      f"({t['session_date']}), stop {_money(t['stop'])}.",
                                 rationale=f"{RATIONALE['orb_entry']} {_verdict_text(v)}", verdict=v, data=dict(t))
                reason = DT_REASON.get(t["reason"], f"orb_{t['reason']}")
                added += _insert(chat, base + ":out", kind="stop" if t["reason"] == "stop" else "exit", account=t["account"],
                                 ticker=t["symbol"], side="sell" if side == "BUY" else "cover", qty=t["shares"], price=t["exit_price"],
                                 decision_date=t["session_date"],
                                 text=f"DAY TRADE EXIT {t['symbol']} @ {_money(t['exit_price'])} at {hm(t['exit_minute'])} ET ({t['reason']}): "
                                      f"P&L {_money(t['pnl'])} ({(t['r_multiple'] or 0):+.2f}R), costs {_money(t['cost'])}.",
                                 rationale=RATIONALE.get(reason, reason), verdict=v, data=dict(t))
        if _has(paper, "dt_daily"):
            for d in paper.execute("SELECT * FROM dt_daily ORDER BY session_date"):
                skipped = json.loads(d["skipped"] or "{}") if isinstance(d["skipped"], str) else {}
                parts = [f"{n} for {k.upper() if k == 'pdt' else k}" for k, n in skipped.items() if n]
                if not parts:
                    continue
                why = {"pdt": "pattern-day-trader limit (3 day trades per 5 sessions under $25k)", "slots": "max open positions",
                       "breaker": "daily-loss circuit breaker", "size": "position too small"}
                added += _insert(chat, f"dtskip:{d['account']}:{d['session_date']}", kind="skip", account=d["account"],
                                 decision_date=d["session_date"],
                                 text=f"Day-trading {d['session_date']}: skipped signals — {', '.join(parts)}.",
                                 rationale="; ".join(f"{k}: {why.get(k, k)}" for k, n in skipped.items() if n), data=skipped)
    chat.close()
    return {"added": added}


def ingest_watch(path, chat_path=CHAT_DB):
    """Provisional intraday signals from `daytrader.py run --dry-run --include-today` (nothing booked)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    chat = connect_chat(chat_path)
    added = 0
    with chat:
        for t in doc.get("tradesThisRun") or []:
            key = f"watch:{t.get('session_date')}:{t.get('symbol')}:{t.get('entry_minute')}:{t.get('reason')}"
            m = t.get("entry_minute")
            added += _insert(chat, key, kind="watch", account=doc.get("account"), ticker=t.get("symbol"), side="buy",
                             qty=t.get("shares"), price=t.get("entry_price"), decision_date=t.get("session_date"),
                             text=(f"INTRADAY SIGNAL (provisional) {t.get('symbol')}: entry {_money(t.get('entry_price'))} at "
                                   f"{int(m) // 60:02d}:{int(m) % 60:02d} ET, stop {_money(t.get('stop'))}, so far {t.get('reason')} "
                                   f"{_money(t.get('exit_price'))} (P&L {_money(t.get('pnl'))})." if m is not None else
                                   f"INTRADAY SIGNAL (provisional) {t.get('symbol')}"),
                             rationale=RATIONALE["orb_entry"] + " Provisional: the session is booked after the close.", data=t)
    chat.close()
    return {"added": added}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch-json")
    args = parser.parse_args(argv)
    result = ingest_watch(args.watch_json) if args.watch_json else ingest()
    json.dump(result, sys.stdout)


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("decision_feed"):
        main()
