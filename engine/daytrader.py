"""Day-trading research and paper trading (opening-range breakout, flat by close). No orders are ever sent.

  python daytrader.py research                 # backtest + walk-forward on cached Schwab bars -> Massive/Research
  python daytrader.py fetch                    # refresh Schwab 1- and 5-minute bars + quotes for the targets
  python daytrader.py run [--fetch]            # paper-trade every cached session not yet processed
  python daytrader.py run --dry-run            # same, rolled back
  python daytrader.py status

The paper ledger lives in Massive/paper_trading.sqlite3 (dt_* tables, separate $10,000 'daytrade' account).
Sessions are replayed from cached bars after the close with exactly the backtest code (intraday.run_sessions),
carrying equity and the PDT day-trade history between runs; re-running is a no-op, missed days are caught up.
"""

import argparse
import itertools
import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime
from pathlib import Path

import pandas as pd

import intraday as it
from costs import CostModel
from fetch_and_model import DATA_DIR, write_atomically
from market_data import load_panel
from schwab_data import CACHE_DB, DEFAULT_SYMBOLS, SchwabAuthError, connect_cache

RESEARCH = DATA_DIR / "Research"
STATE_DB = DATA_DIR / "paper_trading.sqlite3"
STATUS_FILE = RESEARCH / "daytrade_status.json"
GRID = {"or_minutes": [15, 30], "target_r": [1.0, 2.0, None], "stop_frac": [0.5, 1.0]}

SCHEMA = """
CREATE TABLE IF NOT EXISTS dt_accounts (name TEXT PRIMARY KEY, params TEXT NOT NULL, initial_equity REAL NOT NULL,
  equity REAL NOT NULL, frequency INTEGER NOT NULL, start_session TEXT, last_session TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS dt_trades (account TEXT NOT NULL, session_date TEXT NOT NULL, symbol TEXT NOT NULL,
  direction INTEGER, entry_minute INTEGER, exit_minute INTEGER, shares REAL, entry_price REAL, exit_price REAL,
  stop REAL, reason TEXT, pnl REAL, fees REAL, cost REAL, r_multiple REAL, run_id INTEGER,
  PRIMARY KEY (account, session_date, symbol, entry_minute));
CREATE TABLE IF NOT EXISTS dt_daily (account TEXT NOT NULL, session_date TEXT NOT NULL, start_equity REAL, pnl REAL,
  equity REAL, trades INTEGER, skipped TEXT, PRIMARY KEY (account, session_date));
CREATE TABLE IF NOT EXISTS dt_runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT, started_at TEXT,
  mode TEXT, sessions INTEGER, message TEXT);
CREATE TABLE IF NOT EXISTS dt_decision_log (account TEXT NOT NULL, run_id INTEGER, source TEXT NOT NULL,
  decision_date TEXT, fill_date TEXT, ticker TEXT, sleeve TEXT, action TEXT, reason TEXT, target_weight REAL,
  current_weight REAL, ref_price REAL, shares REAL, fill_price REAL, notional REAL, cost_bps REAL,
  slippage_bps REAL, status TEXT);
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _json(obj):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and v != v:
            return None
        if hasattr(v, "item"):
            return clean(v.item())
        return v
    return json.dumps(clean(obj), indent=2, default=str)


# ------------------------------------------------------------------ research
def research(frequency=5, initial_equity=10_000.0, train_fraction=0.65):
    with closing(connect_cache()) as cache:
        bars5 = it.load_bars(cache, frequency)
        bars1 = it.load_bars(cache, 1)
    if bars5.empty:
        raise SystemExit("No cached Schwab bars. Run: python daytrader.py fetch")
    panel = load_panel(tickers=sorted(set(bars5["symbol"]) | {"SPY"}))
    sessions = sorted(bars5["session_date"].unique())
    split = int(len(sessions) * train_fraction)
    train, test = sessions[:split], sessions[split:]
    cost = CostModel.for_broker("schwab")
    trials = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        params = it.ORBParams(**dict(zip(keys, values)))
        tr, _ = it.backtest(bars5, panel, params, frequency, initial_equity, cost, sessions=train)
        te, _ = it.backtest(bars5, panel, params, frequency, initial_equity, cost, sessions=test)
        str_, ste = it.summarize(tr, frequency), it.summarize(te, frequency)
        trials.append({**dict(zip(keys, values)), "trainSharpe": str_["performance"].get("sharpe"),
                       "trainReturn": str_["performance"].get("totalReturn"), "trainTrades": str_["trades"],
                       "testSharpe": ste["performance"].get("sharpe"), "testReturn": ste["performance"].get("totalReturn"),
                       "testTrades": ste["trades"]})
    frame = pd.DataFrame(trials)
    best = frame.sort_values("trainSharpe", ascending=False, na_position="last").iloc[0]
    chosen = it.ORBParams(**{k: (None if pd.isna(best[k]) else (int(best[k]) if k == "or_minutes" else float(best[k])))
                             for k in keys})
    oos, _ = it.backtest(bars5, panel, chosen, frequency, initial_equity, cost, sessions=test)
    defaults = it.ORBParams()
    full, _ = it.backtest(bars5, panel, defaults, frequency, initial_equity, cost)
    full_no_pdt, _ = it.backtest(bars5, panel, it.ORBParams(pdt_max_day_trades=None), frequency, initial_equity, cost)
    one_min = None
    if not bars1.empty:
        r1, _ = it.backtest(bars1, panel, chosen, 1, initial_equity, cost)
        one_min = it.summarize(r1, 1)
    rank_corr = frame[["trainSharpe", "testSharpe"]].rank().corr().iloc[0, 1]
    summary = {
        "generatedAt": now(), "strategy": "opening-range breakout, long-only, flat by close",
        "data": {"source": "Schwab pricehistory (regular session)", "fiveMinuteSessions": len(sessions),
                 "first": sessions[0], "last": sessions[-1], "symbols": sorted(bars5["symbol"].unique()),
                 "oneMinuteSessions": int(bars1["session_date"].nunique()) if not bars1.empty else 0},
        "costModel": cost.as_dict(), "initialEquity": initial_equity,
        "fullPeriodDefaults": {"params": defaults.as_dict(), **it.summarize(full, frequency)},
        "fullPeriodDefaultsNoPdt": it.summarize(full_no_pdt, frequency),
        "walkForward": {"trainSessions": [train[0], train[-1], len(train)], "testSessions": [test[0], test[-1], len(test)],
                        "trials": len(frame), "chosen": chosen.as_dict(), "outOfSample": it.summarize(oos, frequency),
                        "trainTestRankCorrelation": None if pd.isna(rank_corr) else float(rank_corr)},
        "oneMinuteCheckChosenParams": one_min,
        "caveats": [
            f"Only {len(sessions)} sessions of 5-minute and ~{summary_sessions(bars1)} of 1-minute history are "
            "available from Schwab; one or two hundred trades cannot separate a small edge from noise.",
            "Fills are simulated at the next bar open with half-spread, impact and extra slippage; real fills, "
            "especially stop orders in fast markets, can be worse.",
            "The PDT rule (3 day trades per 5 sessions under $25k) caps trading to ~0.6 trades per day.",
            "Universe is 30 hand-picked liquid symbols (today's large caps) - survivorship bias.",
        ],
    }
    RESEARCH.mkdir(parents=True, exist_ok=True)
    write_atomically(RESEARCH / "daytrade_summary.json", _json(summary).encode())
    write_atomically(RESEARCH / "daytrade_trials.csv", frame.to_csv(index=False).encode())
    write_atomically(RESEARCH / "daytrade_trades_defaults.csv", full.trades.to_csv(index=False).encode())
    if full.daily is not None and not full.daily.empty:
        daily = full.daily.set_index("session_date")[["equity", "return", "pnl", "trades"]]
        daily.index.name = "session_date"
        write_atomically(RESEARCH / "daytrade_defaults_equity.csv", daily.to_csv().encode())
        try:
            from attribution import attribute_daytrade_result
            summary["attribution"] = attribute_daytrade_result(full.daily, name="daytrader")
            oos_daily = getattr(oos, "daily", None)
            if oos_daily is not None and not oos_daily.empty:
                summary["attributionOos"] = attribute_daytrade_result(oos_daily, name="daytrader_oos")
            write_atomically(RESEARCH / "daytrade_summary.json", _json(summary).encode())
        except Exception as error:
            summary["attribution"] = {"name": "daytrader", "error": str(error)[:300]}
            write_atomically(RESEARCH / "daytrade_summary.json", _json(summary).encode())
    return summary


def summary_sessions(bars):
    return int(bars["session_date"].nunique()) if not bars.empty else 0


# ------------------------------------------------------------------ fetch (Schwab market data only)
def swing_symbols():
    """Current swing paper-trading targets and positions (so the AI layer and monitor get fresh Schwab data)."""
    try:
        with closing(sqlite3.connect(STATE_DB)) as db:
            rows = db.execute("SELECT ticker FROM positions UNION SELECT ticker FROM signal_state WHERE asof = "
                              "(SELECT MAX(asof) FROM signal_state)").fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error:
        return []


def fetch(symbols=None):
    from schwab_data import SchwabMarketData, fetch_history, store_quotes
    symbols = sorted(set(symbols or DEFAULT_SYMBOLS) | set(swing_symbols()))
    client = SchwabMarketData()
    out = {"token": client.token_status()}
    with closing(connect_cache()) as cache:
        out["oneMinute"] = fetch_history(symbols, 10, 10, client, cache, log=lambda m: None, frequency=1)
        out["fiveMinute"] = fetch_history(symbols, 10, 10, client, cache, log=lambda m: None, frequency=5)
        rows = store_quotes(cache, client.quotes(list(symbols)))
        out["quotes"] = len(rows)
    return out


# ------------------------------------------------------------------ paper trading
def apply_advisory(db, candidates, mode, asof):
    """Optional advisory verdicts (advisory.py): RED / out-of-band candidates are dropped and YELLOW ones get half
    size in 'apply' mode; 'log' only records. Missing or failed verdicts change nothing."""
    if mode == "off" or candidates is None or not len(candidates):
        return []
    from advisory import gated_adjust_entry, load_verdicts
    verdicts, warnings = load_verdicts(asof, connection=db)
    notes, keep, scales = list(warnings), [], []
    for row in candidates.itertuples():
        # Analyst-scorecard gate: only flag types proven at the 5-day horizon may drop or shrink a day trade.
        allowed, note, gate_ok, why = gated_adjust_entry(verdicts.get(row.symbol), 1.0, row.raw_entry, horizons=(5,))
        apply_now = mode == "apply" and gate_ok
        if note:
            notes.append(f"{row.session_date} {row.symbol}: {note} ({mode}"
                         f"{', ungated: ' + str(why) if mode == 'apply' and not gate_ok else ''})")
        keep.append(allowed > 0 or not apply_now)
        scales.append(allowed if apply_now else 1.0)
    candidates["risk_scale"] = scales
    candidates.drop(candidates.index[[not k for k in keep]], inplace=True)
    return notes


def run(account="daytrade", dry_run=False, frequency=5, initial_equity=10_000.0, params=None, advisory_mode="log",
        include_today=False):
    with closing(sqlite3.connect(STATE_DB, timeout=30, isolation_level=None)) as db, closing(connect_cache()) as cache:
        db.row_factory = sqlite3.Row
        db.executescript(SCHEMA)
        db.execute("BEGIN IMMEDIATE")
        try:
            row = db.execute("SELECT * FROM dt_accounts WHERE name=?", (account,)).fetchone()
            bars = it.load_bars(cache, frequency)
            if bars.empty:
                raise SystemExit("No cached Schwab bars. Run: python daytrader.py fetch")
            all_sessions = sorted(bars["session_date"].unique())
            if row is None:
                p = params or it.ORBParams()
                # Start with the latest cached session so the first run trades "today".
                db.execute("INSERT INTO dt_accounts VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                           (account, json.dumps(p.as_dict()), initial_equity, initial_equity, frequency,
                            all_sessions[-1], now()))
                row = db.execute("SELECT * FROM dt_accounts WHERE name=?", (account,)).fetchone()
            p = it.ORBParams(**json.loads(row["params"]))
            frequency = row["frequency"]
            pending = [d for d in all_sessions if d >= row["start_session"] and (row["last_session"] is None or d > row["last_session"])]
            if not include_today and pending and pending[-1] == _eastern_today() and not _session_closed():
                pending = pending[:-1]     # never book today's session before the close (16:05 ET); see --include-today
            run_id = db.execute("INSERT INTO dt_runs (account, started_at, mode, sessions, message) VALUES (?, ?, ?, ?, ?)",
                                (account, now(), "dry-run" if dry_run else "run", len(pending), None)).lastrowid
            result, advisory_notes, macro_notes = None, [], []
            if pending:
                panel = load_panel(tickers=sorted(set(bars["symbol"]) | {"SPY"}))
                history = [r[0] for r in db.execute(
                    "SELECT session_date FROM dt_trades WHERE account=? ORDER BY session_date", (account,))]
                cost = CostModel.for_broker("schwab")
                window = bars[bars["session_date"] >= min(all_sessions[max(0, all_sessions.index(pending[0]) - 15)], pending[0])]
                symbols = sorted(window["symbol"].unique())
                context = it.daily_context(panel, symbols, p)
                last_daily = panel.dates[-1].strftime("%Y-%m-%d")
                later = sorted(d for d in window["session_date"].unique() if d > last_daily)
                if later:
                    context = pd.concat([context, it.next_session_context(panel, symbols, p, later[0])], ignore_index=True)
                ranges = it.opening_ranges(window, p, frequency)
                candidates = it.candidate_trades(window, ranges, context, p, frequency, cost)
                if len(candidates):
                    candidates = candidates[candidates["session_date"].isin(pending)].copy()
                advisory_notes = apply_advisory(db, candidates, advisory_mode, pending[-1])
                dilution_notes = []
                try:
                    from dilution import is_buy_blocked
                    if candidates is not None and len(candidates):
                        keep = []
                        for row in candidates.itertuples():
                            blocked, reason, _ = is_buy_blocked(row.symbol, asof=row.session_date, has_position=False)
                            if blocked:
                                dilution_notes.append(f"{row.session_date} {row.symbol}: blocked ({reason})")
                                keep.append(False)
                            else:
                                keep.append(True)
                        if not all(keep):
                            candidates = candidates.loc[[c for c, k in zip(candidates.index, keep) if k]].copy()
                except Exception as error:  # noqa: BLE001 — never fail the day-trade run
                    dilution_notes = [f"error: {error}"]
                macro_notes = []
                try:
                    from macro_reaction import adjust_daytrade_candidates
                    macro_settings = {"macro_reaction": "off"}
                    if isinstance(params, dict):
                        macro_settings.update({k: params[k] for k in params if str(k).startswith("macro_")})
                    candidates, macro_notes = adjust_daytrade_candidates(candidates, pending[-1], macro_settings)
                    if getattr(candidates, "columns", None) is not None and "macro_skip" in candidates.columns:
                        candidates = candidates[candidates["macro_skip"] != True]
                except Exception as error:  # noqa: BLE001 — never fail the day-trade run
                    macro_notes = [{"error": str(error)}]
                try:
                    from black_swan_playbook import adjust_daytrade_candidates as adjust_bs_day
                    bs_settings = {"black_swan": "log"}
                    if isinstance(params, dict) and params.get("black_swan"):
                        bs_settings["black_swan"] = params["black_swan"]
                    candidates, bs_notes = adjust_bs_day(candidates, pending[-1], bs_settings)
                    if getattr(candidates, "columns", None) is not None and "black_swan_skip" in candidates.columns:
                        candidates = candidates[candidates["black_swan_skip"] != True]
                    macro_notes = list(macro_notes) + list(bs_notes or [])
                except Exception as error:  # noqa: BLE001 — never fail the day-trade run
                    macro_notes = list(macro_notes) + [{"error": f"black_swan: {error}"}]
                result = it.run_sessions(candidates, pending, p, row["equity"], cost, pdt_history=history)
                for t in result.trades.itertuples(index=False):
                    db.execute("INSERT OR IGNORE INTO dt_trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (account, t.session_date, t.symbol, t.direction, t.entry_minute, t.exit_minute, t.shares,
                                t.entry_price, t.exit_price, t.stop, t.reason, t.pnl, t.fees, t.cost, t.r_multiple, run_id))
                decisions = it.decisions_from_trades(result.trades)
                for d in decisions.itertuples(index=False):
                    db.execute("INSERT INTO dt_decision_log VALUES (?, ?, 'paper', " + ", ".join("?" * len(decisions.columns)) + ")",
                               (account, run_id, *[None if (isinstance(v, float) and v != v) else v for v in d]))
                for d in result.daily.itertuples(index=False):
                    db.execute("INSERT OR REPLACE INTO dt_daily VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (account, d.session_date, d.start_equity, d.pnl, d.equity, d.trades,
                                json.dumps(result.skipped)))
                equity = float(result.daily["equity"].iloc[-1])
                db.execute("UPDATE dt_accounts SET equity=?, last_session=? WHERE name=?", (equity, pending[-1], account))
            status = status_payload(db, cache, account)
            status["dryRun"] = dry_run
            status["lastRun"] = {"runId": run_id, "sessionsProcessed": len(pending),
                                 "skipped": result.skipped if result else {},
                                 "advisory": {"mode": advisory_mode, "notes": advisory_notes if pending else []},
                                 "macro": macro_notes if pending else []}
            db.execute("ROLLBACK" if dry_run else "COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    try:
        from attribution import attribute_daytrade_result
        daily = status.get("daily")
        if daily:
            import pandas as pd
            status["attribution"] = attribute_daytrade_result(pd.DataFrame(daily), name="daytrader_paper")
    except Exception as error:
        status["attribution"] = {"name": "daytrader_paper", "error": str(error)[:300]}
    write_atomically(STATUS_FILE.with_name("daytrade_status_dryrun.json") if dry_run else STATUS_FILE, _json(status).encode())
    return status


def status_payload(db, cache, account="daytrade"):
    row = db.execute("SELECT * FROM dt_accounts WHERE name=?", (account,)).fetchone()
    q = lambda sql, *a: [dict(r) for r in db.execute(sql, a)]
    out = {"account": account, "generatedAt": now(), "exists": row is not None}
    if row is not None:
        out.update({"params": json.loads(row["params"]), "initialEquity": row["initial_equity"], "equity": row["equity"],
                    "frequency": row["frequency"], "startSession": row["start_session"], "lastSession": row["last_session"],
                    "trades": q("SELECT * FROM dt_trades WHERE account=? ORDER BY session_date DESC, entry_minute DESC LIMIT 50", account),
                    "daily": q("SELECT * FROM dt_daily WHERE account=? ORDER BY session_date DESC LIMIT 60", account)})
        recent = [r["session_date"] for r in out["daily"][:4]]
        out["pdt"] = {"limit": out["params"].get("pdt_max_day_trades"), "windowSessions": out["params"].get("pdt_window"),
                      "usedInLast4Sessions": sum(1 for t in out["trades"] if t["session_date"] in recent),
                      "appliesBelowEquity": it.PDT_EQUITY}
    quotes = pd.read_sql_query("SELECT * FROM intraday_quotes WHERE fetched_at = (SELECT MAX(fetched_at) FROM intraday_quotes)", cache)
    out["latestQuotes"] = quotes.to_dict("records")
    return out


def _eastern_now():
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def _eastern_today():
    return _eastern_now().date().isoformat()


def _session_closed():
    stamp = _eastern_now()
    return stamp.hour * 60 + stamp.minute >= 16 * 60 + 5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["research", "fetch", "run", "status"])
    parser.add_argument("--account", default="daytrade")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fetch", action="store_true", help="Refresh Schwab bars/quotes before a paper run.")
    parser.add_argument("--initial-equity", type=float, default=10_000.0)
    parser.add_argument("--no-pdt", action="store_true", help="New account without the PDT limit (>= $25k or cash).")
    parser.add_argument("--include-today", action="store_true",
                        help="Also simulate today's unfinished session (use with --dry-run for an intraday watch).")
    parser.add_argument("--advisory", choices=["off", "log", "apply"], default="log",
                        help="Advisory verdicts: off, log (default; record only) or apply (can only reduce risk).")
    args = parser.parse_args(argv)
    try:
        if args.command == "research":
            result = research(initial_equity=args.initial_equity)
        elif args.command == "fetch":
            result = fetch()
        elif args.command == "run":
            if args.fetch:
                try:
                    print(_json({"fetch": fetch()}), file=sys.stderr)
                except SchwabAuthError as error:
                    print(_json({"fetch": "skipped", "reason": str(error)}), file=sys.stderr)
            params = it.ORBParams(pdt_max_day_trades=None) if args.no_pdt else None
            result = run(args.account, args.dry_run, initial_equity=args.initial_equity, params=params,
                         advisory_mode=args.advisory, include_today=args.include_today)
            result = {k: result.get(k) for k in ("account", "equity", "lastSession", "lastRun", "pdt", "dryRun")} | {
                "tradesThisRun": [t for t in result.get("trades", []) if t.get("run_id") == result["lastRun"]["runId"]]}
        else:
            with closing(sqlite3.connect(STATE_DB)) as db, closing(connect_cache()) as cache:
                db.row_factory = sqlite3.Row
                db.executescript(SCHEMA)
                result = status_payload(db, cache, args.account)
    except SchwabAuthError as error:
        result = {"status": "auth_required", "message": str(error)}
    print(_json(result))


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("daytrader"):
        main()
