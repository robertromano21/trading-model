"""Paper trader for the TSMOM + mean-reversion strategies. No broker is ever called.

It reuses the research signal code (strategies/*) and the research cost model, keeps its own ledger in SQLite
(Massive/paper_trading.sqlite3, gitignored) and is restart-safe: every run replays the sessions since the last
processed one in order (fill yesterday's orders at today's open, stops, dividends, stale-position handling,
mark to market, today's decisions), so a missed day is caught up and running twice for the same date is a no-op.

  python paper_trader.py run                 # process through the latest complete session in the price database
  python paper_trader.py run --dry-run       # same, but roll back; prints the orders it would place
  python paper_trader.py run --execution close   # market-on-close fills instead of next open
  python paper_trader.py status              # print the status JSON

After each run it re-simulates the same strategy from the account start with the backtest simulator (the "shadow"
backtest) and records the equity / position differences in the reconciliation table. Status for the dashboard is
written to Massive/Research/paper_status.json.
"""

import argparse
import json
import math
import sqlite3
import sys
from contextlib import closing
from datetime import datetime

import numpy as np
import pandas as pd

from costs import CostModel
from fetch_and_model import DATA_DIR, write_atomically
from market_data import load_panel
from portfolio_sim import DECISION_COLUMNS, simulate
from risk import CircuitBreaker
from regime_hmm import write_regime_state
from strategies import build
from universe import dollar_volume

STATE_DB = DATA_DIR / "paper_trading.sqlite3"
STATUS_FILE = DATA_DIR / "Research" / "paper_status.json"
# 2026-11-25 (Robert approved): plain TSMOM baseline, no mean-reversion overlay, no regime scaling (fixed).
# Previous: {"strategy": "combined", "params": {"regime_model": "fixed"}, "sleeves": {"tsmom": {}, "meanrev": {}}}
DEFAULT_CONFIG = {"strategy": "tsmom", "params": {}}
# Starting balance defaults to $10,000 (override with --initial-equity when creating an account). Small-account
# effects are modelled explicitly: whole shares (a name whose target is worth less than one share is skipped),
# a minimum order value, and an optional fixed per-order fee (--fixed-fee) on top of the bps cost model.
DEFAULTS = {"initial_equity": 10_000.0, "execution": "next_open", "whole_shares": True, "min_trade_weight": 0.02,
            "min_order_value": 100.0, "broker": "schwab", "fixed_fee": None,
            # Optional advisory verdicts (advisory.py): 'off', 'log' (record only, default) or 'apply' (may only
            # shrink or defer new entries; the risk engine stays final; missing verdicts change nothing).
            "advisory_mode": "log", "advisory_yellow_scale": 0.5, "advisory_max_age_days": 3,
            # Dilution filter: block new buys when share-count growth is >= 10% YoY or a recent offering is flagged.
            "dilution_filter": True, "dilution_exit": False, "dilution_red_threshold": 0.10,
            "dilution_warn_threshold": 0.05, "dilution_offering_days": 30,
            "stale_sessions": 5, "daily_loss_limit": 0.04, "max_gross": 1.0, "max_name_weight": 0.30,
            "max_positions": 10,
            # US taxable-account lot tracking / after-tax metrics / tax-aware policy (off = pre-tax path).
            "tax_aware": False,
            # Macro release reactions (macro_reaction.py): off (default), log, or apply.
            # apply only uses event-study-proven rules unless macro_allow_experimental is on.
            "macro_reaction": "off", "macro_allow_experimental": False,
            # Black swan playbook (black_swan_playbook.py): log (default) or apply (paper only).
            "black_swan": "log"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
  name TEXT PRIMARY KEY, config TEXT NOT NULL, settings TEXT NOT NULL, initial_equity REAL NOT NULL,
  cash REAL NOT NULL, start_date TEXT NOT NULL, last_processed TEXT, breaker_remaining INTEGER NOT NULL DEFAULT 0,
  blocked TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, asof TEXT NOT NULL, mode TEXT NOT NULL,
  started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, sessions_processed INTEGER, equity REAL,
  cash REAL, gross REAL, message TEXT, diagnostics TEXT);
CREATE TABLE IF NOT EXISTS signal_state (
  account TEXT NOT NULL, asof TEXT NOT NULL, ticker TEXT NOT NULL, target_weight REAL NOT NULL, sleeve TEXT,
  stop REAL, PRIMARY KEY (account, asof, ticker));
CREATE TABLE IF NOT EXISTS orders (
  order_id INTEGER PRIMARY KEY AUTOINCREMENT, account TEXT NOT NULL, run_id INTEGER, decision_date TEXT NOT NULL,
  ticker TEXT NOT NULL, side TEXT NOT NULL, shares REAL NOT NULL, ref_price REAL, target_weight REAL,
  current_weight REAL, sleeve TEXT, reason TEXT, execution TEXT NOT NULL, status TEXT NOT NULL,
  status_date TEXT, created_at TEXT NOT NULL, UNIQUE (account, decision_date, ticker, reason));
CREATE TABLE IF NOT EXISTS fills (
  fill_id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, account TEXT NOT NULL, fill_date TEXT NOT NULL,
  ticker TEXT NOT NULL, shares REAL NOT NULL, price REAL NOT NULL, notional REAL NOT NULL, cost_bps REAL NOT NULL,
  effective_price REAL NOT NULL, slippage_bps REAL);
CREATE TABLE IF NOT EXISTS positions (
  account TEXT NOT NULL, ticker TEXT NOT NULL, shares REAL NOT NULL, avg_cost REAL NOT NULL, entry_date TEXT NOT NULL,
  last_price REAL, last_price_date TEXT, stale_sessions INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (account, ticker));
CREATE TABLE IF NOT EXISTS decision_log (
  account TEXT NOT NULL, run_id INTEGER, source TEXT NOT NULL, decision_date TEXT, fill_date TEXT, ticker TEXT,
  sleeve TEXT, action TEXT, reason TEXT, target_weight REAL, current_weight REAL, ref_price REAL, shares REAL,
  fill_price REAL, notional REAL, cost_bps REAL, slippage_bps REAL, status TEXT);
CREATE TABLE IF NOT EXISTS equity_history (
  account TEXT NOT NULL, date TEXT NOT NULL, equity REAL NOT NULL, cash REAL NOT NULL, gross REAL NOT NULL,
  daily_return REAL, PRIMARY KEY (account, date));
CREATE TABLE IF NOT EXISTS reconciliation (
  account TEXT NOT NULL, date TEXT NOT NULL, paper_equity REAL, shadow_equity REAL, equity_diff_pct REAL,
  max_weight_diff REAL, details TEXT, PRIMARY KEY (account, date));
"""


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def connect(path=STATE_DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(SCHEMA)
    from tax import ensure_schema
    ensure_schema(connection)
    return connection


class PaperTrader:
    def __init__(self, connection, account, panel, config=None, settings=None, cost_model=None):
        self.db = connection
        self.account = account
        self.panel = panel
        self.cost_model = cost_model
        row = self.db.execute("SELECT * FROM accounts WHERE name=?", (account,)).fetchone()
        if row is None:
            self.config = config or DEFAULT_CONFIG
            self.settings = {**DEFAULTS, **(settings or {})}
            self.new_account = True
        else:
            self.config = json.loads(row["config"])
            self.settings = {**DEFAULTS, **json.loads(row["settings"])}
            runtime = {k: v for k, v in (settings or {}).items()
                       if k.startswith("advisory_") or k.startswith("macro_") or k.startswith("dilution_")
                       or k in {"tax_aware", "tax", "black_swan"}}
            self.settings.update(runtime)            # advisory / tax-aware / macro reaction can be changed per run
            self.new_account = False
        if cost_model is None:
            overrides = {} if self.settings.get("fixed_fee") is None else {"fixed_fee": float(self.settings["fixed_fee"])}
            self.cost_model = CostModel.for_broker(self.settings.get("broker", "schwab"), **overrides)
        self.strategy = build(self.config)
        from tax import LotLedger, tax_config_from_settings
        self.tax_config = tax_config_from_settings(self.settings)
        self.tax_ledger = None
        if self.tax_config.enabled:
            if self.new_account:
                self.tax_ledger = LotLedger(self.tax_config, self.account)
            else:
                self.tax_ledger = LotLedger.load_sqlite(self.db, self.account, self.tax_config)
        self.dates = list(panel.dates)
        self.row = {d: i for i, d in enumerate(self.dates)}
        self.adv = dollar_volume(panel)
        self.daily_vol = panel.total_return().rolling(20, min_periods=10).std()
        self.last_close = panel.close.ffill()
        self.log = []

    # ---------------------------------------------------------------- helpers
    def _price(self, frame, day, ticker):
        if ticker not in frame.columns:
            return float("nan")
        value = frame.at[day, ticker]
        return float(value) if pd.notna(value) else float("nan")

    def _positions(self):
        return {r["ticker"]: dict(r) for r in self.db.execute("SELECT * FROM positions WHERE account=?", (self.account,))}

    def _cash(self):
        return float(self.db.execute("SELECT cash FROM accounts WHERE name=?", (self.account,)).fetchone()[0])

    def _set_cash(self, cash):
        self.db.execute("UPDATE accounts SET cash=? WHERE name=?", (cash, self.account))

    def _equity(self, day):
        total = self._cash()
        for ticker, p in self._positions().items():
            price = self._price(self.last_close, day, ticker)
            total += p["shares"] * (price if math.isfinite(price) else p["last_price"] or p["avg_cost"])
        return total

    def _decision(self, run_id, **values):
        row = {c: values.get(c) for c in DECISION_COLUMNS}
        for key in ("decision_date", "fill_date"):
            if isinstance(row[key], pd.Timestamp):
                row[key] = str(row[key].date())
        self.db.execute(
            f"INSERT INTO decision_log (account, run_id, source, {', '.join(DECISION_COLUMNS)}) VALUES (?, ?, 'paper', "
            + ", ".join("?" * len(DECISION_COLUMNS)) + ")", (self.account, run_id, *[row[c] for c in DECISION_COLUMNS]))
        self.log.append(row)

    def _trade(self, run_id, day, ticker, shares, price, reason, order=None, ref_price=None, decision_date=None,
               sleeve=None, target_weight=None, current_weight=None, apply_costs=True, cost_day=None):
        """Book a fill of `shares` (signed) at `price` on `day`. Costs use liquidity known at `cost_day`
        (default: the previous session, i.e. the decision close for next-open fills)."""
        ref_day = cost_day if cost_day is not None else self.dates[max(self.row[day] - 1, 0)]
        notional = abs(shares) * price
        bps = self.cost_model.cost_bps(notional, self._price(self.adv, ref_day, ticker),
                                       self._price(self.daily_vol, ref_day, ticker),
                                       "buy" if shares > 0 else "sell", price) if apply_costs else 0.0
        effective = price * (1 + math.copysign(bps, shares) / 1e4)
        cash = self._cash() - shares * price - notional * bps / 1e4
        self._set_cash(cash)
        positions = self._positions()
        current = positions.get(ticker)
        after = (current["shares"] if current else 0.0) + shares
        if abs(after) < 1e-9:
            self.db.execute("DELETE FROM positions WHERE account=? AND ticker=?", (self.account, ticker))
        elif current is None:
            self.db.execute("""INSERT INTO positions (account, ticker, shares, avg_cost, entry_date, last_price,
                               last_price_date) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (self.account, ticker, after, effective, str(day.date()), price, str(day.date())))
        else:
            avg = current["avg_cost"]
            if shares > 0:
                avg = (current["shares"] * avg + shares * effective) / after
            self.db.execute("UPDATE positions SET shares=?, avg_cost=? WHERE account=? AND ticker=?",
                            (after, avg, self.account, ticker))
        slippage = None
        if ref_price and math.isfinite(ref_price):
            slippage = (effective / ref_price - 1) * 1e4 * (1 if shares > 0 else -1)
        self.db.execute("""INSERT INTO fills (order_id, account, fill_date, ticker, shares, price, notional, cost_bps,
                           effective_price, slippage_bps) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (order, self.account, str(day.date()), ticker, shares, price, notional, bps, effective, slippage))
        self._decision(run_id, decision_date=decision_date or day, fill_date=day, ticker=ticker, sleeve=sleeve,
                       action="buy" if shares > 0 else "sell", reason=reason, target_weight=target_weight,
                       current_weight=current_weight, ref_price=ref_price, shares=shares, fill_price=effective,
                       notional=notional, cost_bps=bps, slippage_bps=slippage, status="filled")
        if self.tax_ledger is not None:
            fee = notional * bps / 1e4
            if shares > 0:
                self.tax_ledger.buy(ticker, shares, price, day, fee)
            else:
                self.tax_ledger.sell(ticker, -shares, price, day, fee)
        return notional

    # ---------------------------------------------------------------- one session
    def process_session(self, run_id, day, weights, stops, breaker):
        s = self.settings
        i = self.row[day]
        date = str(day.date())
        # 1. Fill pending next-open orders decided before today.
        pending = self.db.execute("""SELECT * FROM orders WHERE account=? AND status='pending' AND decision_date < ?
                                     ORDER BY order_id""", (self.account, date)).fetchall()
        for order in pending:
            ticker = order["ticker"]
            price = self._price(self.panel.open, day, ticker)
            if not math.isfinite(price) or price <= 0:
                age = i - self.row.get(pd.Timestamp(order["decision_date"]), i)
                if age > s["stale_sessions"]:
                    self.db.execute("UPDATE orders SET status='cancelled', status_date=? WHERE order_id=?",
                                    (date, order["order_id"]))
                continue
            shares = order["shares"] if order["side"] == "buy" else -order["shares"]
            if order["side"] == "sell":
                held = self._positions().get(ticker, {}).get("shares", 0.0)
                shares = -min(order["shares"], held)
                if shares == 0:
                    self.db.execute("UPDATE orders SET status='cancelled', status_date=? WHERE order_id=?",
                                    (date, order["order_id"]))
                    continue
            self._trade(run_id, day, ticker, shares, price, order["reason"], order["order_id"], order["ref_price"],
                        order["decision_date"], order["sleeve"], order["target_weight"], order["current_weight"])
            self.db.execute("UPDATE orders SET status='filled', status_date=? WHERE order_id=?", (date, order["order_id"]))
        breaker.tick()
        # 2. Protective stops from the previous decision.
        if stops is not None and i > 0:
            previous = self.dates[i - 1]
            for ticker, p in self._positions().items():
                level = self._price(stops, previous, ticker) if ticker in stops.columns else float("nan")
                low, opening = self._price(self.panel.low, day, ticker), self._price(self.panel.open, day, ticker)
                if math.isfinite(level) and math.isfinite(low) and low <= level:
                    price = opening if opening <= level else level
                    self._trade(run_id, day, ticker, -p["shares"], price, "stop", ref_price=level,
                                decision_date=previous, target_weight=0.0)
                    self._blocked.add(ticker)
        # 3. Dividends, prices, stale / delisted positions.
        masks = self.panel.masked_returns
        for ticker, p in self._positions().items():
            cash_div = self._price(self.panel.dividends, day, ticker) if ticker in self.panel.dividends.columns else 0.0
            if cash_div and math.isfinite(cash_div):
                self._set_cash(self._cash() + p["shares"] * cash_div)
                if self.tax_ledger is not None:
                    self.tax_ledger.dividend(ticker, cash_div, day, p["shares"])
            if masks is not None and ticker in masks.columns and bool(masks.at[day, ticker]) and i > 0:
                before, after = self._price(self.last_close, self.dates[i - 1], ticker), self._price(self.panel.close, day, ticker)
                if math.isfinite(before) and math.isfinite(after):
                    adjustment = p["shares"] * (before - after)   # spin-off etc.: zero return for the holder
                    self._set_cash(self._cash() + adjustment)
                    self._decision(run_id, decision_date=day, fill_date=day, ticker=ticker, action="adjust",
                                   reason="corporate_action", ref_price=before, shares=0.0, notional=adjustment,
                                   cost_bps=0.0, status="booked")
            close = self._price(self.panel.close, day, ticker)
            if math.isfinite(close):
                self.db.execute("""UPDATE positions SET last_price=?, last_price_date=?, stale_sessions=0
                                   WHERE account=? AND ticker=?""", (close, date, self.account, ticker))
            else:
                stale = p["stale_sessions"] + 1
                self.db.execute("UPDATE positions SET stale_sessions=? WHERE account=? AND ticker=?",
                                (stale, self.account, ticker))
                if stale > s["stale_sessions"]:
                    last = p["last_price"] or p["avg_cost"]
                    self._trade(run_id, day, ticker, -p["shares"], last, "stale_exit", ref_price=last,
                                target_weight=0.0, apply_costs=False)
                    self._blocked.add(ticker)
        if self.tax_ledger is not None:
            paid = float(self.tax_ledger.maybe_settle(day, self._cash()) or 0.0)
            if paid:
                self._set_cash(self._cash() - paid)
                self._decision(run_id, decision_date=day, fill_date=day, ticker="CASH", action="adjust",
                               reason="tax_payment", notional=paid, status="booked")
        # 4. Mark to market and circuit breaker.
        equity = self._equity(day)
        previous_row = self.db.execute("""SELECT equity FROM equity_history WHERE account=? AND date<?
                                          ORDER BY date DESC LIMIT 1""", (self.account, date)).fetchone()
        daily = equity / previous_row[0] - 1 if previous_row else 0.0
        breaker.update(daily)
        gross = sum(abs(p["shares"]) * (self._price(self.last_close, day, t) if math.isfinite(self._price(self.last_close, day, t)) else 0)
                    for t, p in self._positions().items()) / equity if equity else 0.0
        self.db.execute("INSERT OR REPLACE INTO equity_history VALUES (?, ?, ?, ?, ?, ?)",
                        (self.account, date, equity, self._cash(), gross, daily))
        # 5. Today's decisions -> orders (or immediate MOC fills).
        target = weights.loc[day] if day in weights.index else pd.Series(dtype=float)
        target = target[target.ne(0)]
        for ticker in list(self._blocked):
            if target.get(ticker, 0) == 0:
                self._blocked.discard(ticker)
        self.db.execute("DELETE FROM signal_state WHERE account=? AND asof=?", (self.account, date))
        for ticker, weight in target.items():
            stop = self._price(stops, day, ticker) if stops is not None and ticker in stops.columns else float("nan")
            self.db.execute("INSERT INTO signal_state VALUES (?, ?, ?, ?, ?, ?)",
                            (self.account, date, ticker, float(weight), self._sleeve(day, ticker),
                             stop if math.isfinite(stop) else None))
        if self.tax_ledger is not None and self.tax_config.enabled:
            from tax_policy import adjust_targets
            names = sorted(set(self._positions()) | set(target.index))
            prices = {t: self._price(self.last_close, day, t) for t in names}
            prices = {t: p for t, p in prices.items() if math.isfinite(p) and p > 0}
            # harvest substitutes must be visible
            for t in list(self.panel.close.columns):
                px = self._price(self.last_close, day, t)
                if math.isfinite(px) and px > 0:
                    prices.setdefault(t, px)
            current = {}
            if equity:
                for t, p in self._positions().items():
                    if t in prices:
                        current[t] = p["shares"] * prices[t] / equity
            targets_map = {t: float(target.get(t, 0.0)) for t in set(names) | set(target.index)}
            adjusted, actions = adjust_targets(
                self.tax_ledger, targets_map, prices, day, equity, self.tax_config,
                current=current, forced_exits=set(self._blocked),
                available=set(self.panel.close.columns))
            target = pd.Series(adjusted, dtype=float)
            for act in actions:
                self._decision(run_id, decision_date=day, ticker=act.get("ticker"),
                               action="adjust", reason=act.get("kind"),
                               target_weight=act.get("toWeight"), current_weight=act.get("fromWeight"),
                               status="tax_adjusted")
        if str(self.settings.get("macro_reaction") or "off").lower() not in {"off", "", "none"}:
            from macro_reaction import adjust_paper_targets
            target, macro_actions = adjust_paper_targets(target, day, self.settings, equity=equity)
            for act in macro_actions:
                self._decision(run_id, decision_date=day, ticker=act.get("ticker"),
                               action="adjust", reason=act.get("kind") or "macro",
                               status="macro_adjusted" if act.get("apply") else "macro_logged")
        if str(self.settings.get("black_swan") or "log").lower() not in {"off", "", "none"}:
            from black_swan_playbook import adjust_paper_targets as adjust_black_swan_targets
            target, swan_actions = adjust_black_swan_targets(target, day, self.settings, equity=equity)
            for act in swan_actions:
                self._decision(run_id, decision_date=day, ticker=act.get("ticker"),
                               action="adjust", reason=act.get("kind") or "black_swan",
                               status="black_swan_adjusted" if act.get("apply") else "black_swan_logged")
        self._make_orders(run_id, day, target, equity, breaker)
        if self.tax_ledger is not None:
            self.tax_ledger.save_sqlite(self.db, self.account)
        return equity, gross, daily

    def _sleeve(self, day, ticker):
        finder = getattr(self.strategy, "sleeve_of", None)
        return finder(day, ticker) if finder else self.strategy.name

    def _make_orders(self, run_id, day, target, equity, breaker):
        s = self.settings
        date = str(day.date())
        positions = self._positions()
        names = sorted(set(positions) | set(target.index))
        planned = []
        for ticker in names:
            weight = 0.0 if ticker in self._blocked else float(target.get(ticker, 0.0))
            price = self._price(self.last_close, day, ticker)
            if not math.isfinite(price) or price <= 0:
                continue
            current = positions.get(ticker, {}).get("shares", 0.0)
            current_weight = current * price / equity if equity else 0.0
            desired = weight * equity / price
            if s["whole_shares"]:
                desired = math.floor(desired)
                if weight > 0 and desired == 0 and current == 0:
                    self._decision(run_id, decision_date=day, ticker=ticker, sleeve=self._sleeve(day, ticker),
                                   action="buy", reason="below_one_share", target_weight=weight, current_weight=0.0,
                                   ref_price=price, shares=0.0, status="skipped")
                    continue
            delta = desired - current
            if abs(delta) < 1e-9 or (abs(weight - current_weight) < s["min_trade_weight"] and not (weight == 0 and current)):
                continue
            if abs(delta) * price < s.get("min_order_value", 0.0) and not (weight == 0 and current):
                if weight and not current:
                    self._decision(run_id, decision_date=day, ticker=ticker, sleeve=self._sleeve(day, ticker),
                                   action="buy", reason="below_min_order", target_weight=weight,
                                   current_weight=current_weight, ref_price=price, shares=delta, status="skipped")
                continue
            planned.append((ticker, weight, current_weight, price, delta))
        # Independent hard guard on the strategy's intended book (the strategy already applies these caps).
        target_gross = float(target.abs().sum())
        book_breach = target_gross > s["max_gross"] + 0.01 or int(target.ne(0).sum()) > s["max_positions"]
        for ticker, weight, current_weight, price, delta in planned:
            reason = "exit" if weight == 0 else ("entry" if current_weight == 0 else "rebalance")
            status = "pending"
            verdict = self._verdicts.get(ticker) if delta > 0 else None
            if verdict is not None:
                # Analyst-scorecard gate (analyst_scoring.py): even in 'apply' mode a verdict only changes the order
                # when its flag type has beaten chance at the swing horizons; otherwise it is logged ('ungated').
                from advisory import gated_adjust_entry
                increase = max(weight - current_weight, 0.0)
                allowed, note, gate_ok, _gate_why = gated_adjust_entry(
                    verdict, increase, price, s.get("advisory_yellow_scale", 0.5),
                    tuple(s.get("advisory_gate_horizons", (20, 60))))
                apply_now = s.get("advisory_mode") == "apply" and gate_ok
                if note:
                    self._decision(run_id, decision_date=day, ticker=ticker, sleeve=self._sleeve(day, ticker),
                                   action="buy", reason=note, target_weight=weight, current_weight=current_weight,
                                   ref_price=price, shares=delta,
                                   status="advisory_applied" if apply_now else
                                   "advisory_ungated" if s.get("advisory_mode") == "apply" else "advisory_logged")
                    if apply_now:
                        weight = current_weight + allowed
                        desired = weight * equity / price
                        desired = math.floor(desired) if s["whole_shares"] else desired
                        delta = desired - positions.get(ticker, {}).get("shares", 0.0)
                        if delta < 1e-9 or delta * price < s.get("min_order_value", 0.0):
                            continue
            if delta > 0 and s.get("dilution_filter", True):
                try:
                    from dilution import is_buy_blocked
                    has_pos = current_weight > 1e-12 or current > 0
                    blocked_dil, dil_reason, _ = is_buy_blocked(
                        ticker, asof=day, has_position=has_pos, config=s)
                    if blocked_dil or (dil_reason == "dilution_warning" and delta > 0):
                        status, reason = "blocked", dil_reason or "dilution"
                    elif dil_reason == "dilution_warning":
                        self._decision(run_id, decision_date=day, ticker=ticker, sleeve=self._sleeve(day, ticker),
                                       action="hold", reason="dilution_warning", target_weight=weight,
                                       current_weight=current_weight, ref_price=price, shares=0.0, status="logged")
                except Exception:
                    pass
            if delta > 0:
                try:
                    from black_swan_playbook import freeze_new_long
                    if freeze_new_long(ticker, current_weight, weight, s):
                        status, reason = "blocked", "black_swan_freeze"
                except Exception:
                    pass
            if delta > 0 and breaker.blocks_buying():
                status, reason = "blocked", "circuit_breaker"
            elif delta > 0 and (weight > s["max_name_weight"] + 1e-6 or book_breach):
                status, reason = "blocked", "risk_limit"
            side = "buy" if delta > 0 else "sell"
            cursor = self.db.execute(
                """INSERT OR IGNORE INTO orders (account, run_id, decision_date, ticker, side, shares, ref_price,
                   target_weight, current_weight, sleeve, reason, execution, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (self.account, run_id, date, ticker, side, abs(delta), price, weight, current_weight,
                 self._sleeve(day, ticker), reason, s["execution"], status, now()))
            if cursor.rowcount == 0:
                continue
            order_id = cursor.lastrowid
            if status == "blocked":
                self._decision(run_id, decision_date=day, ticker=ticker, sleeve=self._sleeve(day, ticker), action=side,
                               reason=reason, target_weight=weight, current_weight=current_weight, ref_price=price,
                               shares=delta, status="blocked")
            elif s["execution"] == "close":
                close = self._price(self.panel.close, day, ticker)
                if math.isfinite(close):
                    self._trade(run_id, day, ticker, delta, close, reason, order_id, price, day,
                                self._sleeve(day, ticker), weight, current_weight, cost_day=day)
                    self.db.execute("UPDATE orders SET status='filled', status_date=? WHERE order_id=?", (date, order_id))
            else:
                self._decision(run_id, decision_date=day, ticker=ticker, sleeve=self._sleeve(day, ticker), action=side,
                               reason=reason, target_weight=weight, current_weight=current_weight, ref_price=price,
                               shares=delta, status="pending")

    def _prepare_tax_ledger(self):
        if not self.tax_config.enabled:
            return
        from tax import LotLedger
        if self.tax_ledger is None:
            self.tax_ledger = LotLedger(self.tax_config, self.account) if self.new_account else \
                LotLedger.load_sqlite(self.db, self.account, self.tax_config)
        if not self.tax_ledger.open_lots():
            self.tax_ledger.seed_from_positions(self._positions().values(), self.settings.get("start") or None)
        self.db.execute("UPDATE accounts SET settings=? WHERE name=?",
                        (json.dumps(self.settings), self.account))

    # ---------------------------------------------------------------- full run
    def run(self, asof=None, mode="run"):
        asof = pd.Timestamp(asof) if asof else self.dates[-1]
        if asof not in self.row:
            raise ValueError(f"{asof.date()} is not a complete session in the price database.")
        if self.new_account:
            self.db.execute("""INSERT INTO accounts (name, config, settings, initial_equity, cash, start_date,
                               created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (self.account, json.dumps(self.config), json.dumps(self.settings),
                             self.settings["initial_equity"], self.settings["initial_equity"], str(asof.date()), now()))
        account = self.db.execute("SELECT * FROM accounts WHERE name=?", (self.account,)).fetchone()
        self._prepare_tax_ledger()
        last = pd.Timestamp(account["last_processed"]) if account["last_processed"] else None
        sessions = [d for d in self.dates if (last is None or d > last) and d <= asof]
        if last is None:
            sessions = [asof]
        run_id = self.db.execute("INSERT INTO runs (account, asof, mode, started_at, status) VALUES (?, ?, ?, ?, 'running')",
                                 (self.account, str(asof.date()), mode, now())).lastrowid
        weights = self.strategy.target_weights(self.panel)
        stops = self.strategy.stops(self.panel)
        breaker = CircuitBreaker(self.settings["daily_loss_limit"], 1, account["breaker_remaining"])
        self._blocked = set(json.loads(account["blocked"] or "[]"))   # stopped-out names awaiting a flat signal
        self._verdicts, self._advisory_warnings = {}, []
        if self.settings.get("advisory_mode", "off") != "off":
            from advisory import load_verdicts
            self._verdicts, self._advisory_warnings = load_verdicts(
                asof, self.settings.get("advisory_max_age_days", 3), connection=self.db)
        equity = gross = None
        for day in sessions:
            equity, gross, _ = self.process_session(run_id, day, weights, stops, breaker)
        if sessions:
            self.db.execute("UPDATE accounts SET last_processed=?, breaker_remaining=?, blocked=? WHERE name=?",
                            (str(sessions[-1].date()), breaker.remaining, json.dumps(sorted(self._blocked)),
                             self.account))
        else:
            equity = self._equity(asof)
        reconciliation = self.reconcile(asof, weights, stops)
        diagnostics = {"strategy": self.strategy.last_diagnostics, "reconciliation": reconciliation,
                       "advisory": {"mode": self.settings.get("advisory_mode"), "verdicts": len(self._verdicts),
                                    "warnings": self._advisory_warnings}}
        snap = (self.strategy.last_diagnostics or {}).get("regime")
        # Only update the shared snapshot from the live paper ledger, never from unit-test temp DBs.
        if snap:
            row = self.db.execute("PRAGMA database_list").fetchone()
            live = STATE_DB.__class__(row[2] or "").resolve() if row and row[2] else None
            if live == STATE_DB.resolve():
                write_regime_state(snap)
        message = f"processed {len(sessions)} session(s)" + (" (catch-up)" if len(sessions) > 1 else "")
        if mode != "dry-run":
            try:
                from analyst_scoring import run as rescore_analyst
                result = rescore_analyst(memory=True)
                diagnostics["analystScore"] = {k: result.get(k) for k in
                                               ("status", "calls", "scorable", "gateOpen", "seconds", "summary")
                                               if k in result}
            except Exception as error:
                diagnostics["analystScore"] = {"status": "error", "error": str(error)}
        self.db.execute("""UPDATE runs SET finished_at=?, status='ok', sessions_processed=?, equity=?, cash=?, gross=?,
                           message=?, diagnostics=? WHERE run_id=?""",
                        (now(), len(sessions), equity, self._cash(), gross, message,
                         json.dumps(diagnostics, default=str), run_id))
        return {"runId": run_id, "asof": str(asof.date()), "sessionsProcessed": len(sessions), "equity": equity,
                "message": message}

    def reconcile(self, asof, weights, stops):
        """Compare the ledger with a fresh backtest of the same weights from the account start (fractional shares)."""
        account = self.db.execute("SELECT * FROM accounts WHERE name=?", (self.account,)).fetchone()
        start = pd.Timestamp(account["start_date"])
        s = self.settings
        # Same starting balance and costs as the ledger, but fractional shares and no minimum order: the difference
        # measures the small-account (rounding / minimum-order) effects plus any data revisions or missed runs.
        shadow = simulate(self.panel, weights, stops, self.cost_model, account["initial_equity"], s["execution"],
                          start=start, end=asof, stale_sessions=s["stale_sessions"],
                          breaker=CircuitBreaker(s["daily_loss_limit"]), min_trade_weight=s["min_trade_weight"],
                          tax_config=self.tax_config if self.tax_config.enabled else None)
        paper_equity = self._equity(asof)
        shadow_equity = float(shadow.equity.iloc[-1])
        paper_weights = {t: p["shares"] * (self._price(self.last_close, asof, t)) / paper_equity
                         for t, p in self._positions().items() if p["shares"]}
        shadow_weights = shadow.holdings.iloc[-1][shadow.holdings.iloc[-1].ne(0)].to_dict() if len(shadow.holdings.columns) else {}
        names = set(paper_weights) | set(shadow_weights)
        diffs = {t: paper_weights.get(t, 0.0) - shadow_weights.get(t, 0.0) for t in names}
        max_diff = max((abs(v) for v in diffs.values()), default=0.0)
        details = {"paperWeights": paper_weights, "shadowWeights": shadow_weights,
                   "largestDiffs": dict(sorted(diffs.items(), key=lambda kv: -abs(kv[1]))[:10])}
        diff_pct = (paper_equity / shadow_equity - 1) * 100 if shadow_equity else None
        self.db.execute("INSERT OR REPLACE INTO reconciliation VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (self.account, str(asof.date()), paper_equity, shadow_equity, diff_pct, max_diff,
                         json.dumps(details)))
        return {"date": str(asof.date()), "paperEquity": paper_equity, "shadowEquity": shadow_equity,
                "equityDiffPct": diff_pct, "maxWeightDiff": max_diff}


# -------------------------------------------------------------------- status / CLI
def status(connection, account="default"):
    def rows(sql, *params):
        return [dict(r) for r in connection.execute(sql, params)]
    acct = connection.execute("SELECT * FROM accounts WHERE name=?", (account,)).fetchone()
    if acct is None:
        return {"account": account, "exists": False, "generatedAt": now()}
    equity = rows("SELECT date, equity, cash, gross, daily_return FROM equity_history WHERE account=? ORDER BY date", account)
    fills = rows("SELECT slippage_bps, cost_bps FROM fills WHERE account=? AND slippage_bps IS NOT NULL", account)
    return {
        "account": account, "exists": True, "generatedAt": now(),
        "config": json.loads(acct["config"]), "settings": json.loads(acct["settings"]),
        "initialEquity": acct["initial_equity"], "cash": acct["cash"], "startDate": acct["start_date"],
        "lastProcessed": acct["last_processed"], "circuitBreakerRemaining": acct["breaker_remaining"],
        "equityHistory": equity[-260:],
        "positions": rows("SELECT ticker, shares, avg_cost, entry_date, last_price, last_price_date, stale_sessions "
                          "FROM positions WHERE account=? ORDER BY ticker", account),
        "openOrders": rows("SELECT order_id, decision_date, ticker, side, shares, ref_price, target_weight, "
                           "current_weight, sleeve, reason, execution, status FROM orders WHERE account=? "
                           "AND (status='pending' OR (status='blocked' AND decision_date=?)) ORDER BY order_id",
                           account, acct["last_processed"]),
        "recentDecisions": rows("SELECT * FROM decision_log WHERE account=? ORDER BY rowid DESC LIMIT 50", account),
        "targets": rows("SELECT ticker, target_weight, sleeve, stop FROM signal_state WHERE account=? AND asof="
                        "(SELECT MAX(asof) FROM signal_state WHERE account=?) ORDER BY target_weight DESC",
                        account, account),
        "slippage": {"fills": len(fills),
                     "avgSlippageBps": float(np.mean([f["slippage_bps"] for f in fills])) if fills else None,
                     "avgCostBps": float(np.mean([f["cost_bps"] for f in fills])) if fills else None},
        "reconciliation": rows("SELECT date, paper_equity, shadow_equity, equity_diff_pct, max_weight_diff "
                               "FROM reconciliation WHERE account=? ORDER BY date DESC LIMIT 30", account),
        "runs": rows("SELECT run_id, asof, mode, started_at, finished_at, status, sessions_processed, equity, message "
                     "FROM runs WHERE account=? ORDER BY run_id DESC LIMIT 20", account),
        "tax": _tax_status(connection, account, dict(acct), rows(
            "SELECT ticker, shares, avg_cost, entry_date, last_price FROM positions WHERE account=?", account)),
    }


def _tax_status(connection, account, acct, positions):
    """Lot snapshot for the UI. Never raises into status(); missing tables just look empty."""
    from tax import LotLedger, TaxConfig, tax_config_from_settings
    settings = {}
    try:
        settings = json.loads(acct["settings"]) if acct and acct["settings"] else {}
    except (TypeError, json.JSONDecodeError, KeyError):
        settings = {}
    cfg = tax_config_from_settings(settings)
    empty = {"enabled": cfg.enabled, "openLots": [], "harvestCandidates": [],
             "disclaimer": "Planning estimate only; not tax advice."}
    try:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='tax_lots'").fetchone():
            return empty
        ledger = LotLedger.load_sqlite(connection, account, cfg)
        if cfg.enabled and not ledger.open_lots() and positions:
            ledger.seed_from_positions(positions, acct.get("last_processed") if acct else None)
        prices = {p["ticker"]: p["last_price"] for p in positions if p.get("last_price")}
        snap = ledger.snapshot(prices, acct.get("last_processed") if acct else None)
        snap["enabled"] = cfg.enabled
        return snap
    except Exception as error:
        empty["error"] = str(error)[:300]
        return empty


def _clean_json(value):
    if isinstance(value, dict):
        return {k: _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_status(payload, path=STATUS_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(_clean_json(payload), indent=2, default=str).encode())


def schwab_quotes_for(snapshot):
    """Cache Schwab quotes (intraday_quotes table in Massive/intraday.sqlite3) for targets and positions."""
    symbols = sorted({t["ticker"] for t in snapshot.get("targets", [])} | {p["ticker"] for p in snapshot.get("positions", [])}
                     | {o["ticker"] for o in snapshot.get("openOrders", [])})
    if not symbols:
        return {"status": "no symbols"}
    try:
        from schwab_data import SchwabAuthError, SchwabMarketData, connect_cache, store_quotes
        with closing(connect_cache()) as cache:
            rows = store_quotes(cache, SchwabMarketData().quotes(symbols))
        return {"status": "ok", "quotes": [dict(zip(("symbol", "fetchedAt", "last", "bid", "ask"), r[:5])) for r in rows]}
    except Exception as error:          # market data is informational; never fail the paper run over it
        return {"status": "unavailable", "message": str(error)[:300]}


def sync_prices():
    """Same options the dashboard uses on launch: sessions through yesterday (US/Eastern), plus today after 9 PM ET."""
    from zoneinfo import ZoneInfo

    from history_backfill import backfill
    from local_env import load_local_env

    load_local_env()
    eastern = datetime.now(ZoneInfo("America/New_York"))
    options = {"date": (eastern - pd.Timedelta(days=1)).date().isoformat(),
               "mode": "massive" if __import__("os").environ.get("MASSIVE_API_KEY", "").strip() else "local",
               "startup": True}
    if options["mode"] == "massive" and eastern.hour >= 21:
        options["today"] = eastern.date().isoformat()
    result = backfill(options)
    return {k: result.get(k) for k in ("status", "lastSaved", "downloaded", "sessions", "errors", "message")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--account", default="default")
    parser.add_argument("--asof")
    parser.add_argument("--dry-run", action="store_true", help="Process and print, then roll back (nothing saved).")
    parser.add_argument("--execution", choices=["next_open", "close"])
    parser.add_argument("--initial-equity", type=float, help="Starting balance for a new account (default $10,000).")
    parser.add_argument("--broker", choices=sorted(__import__("costs").BROKER_PROFILES),
                        help="Commission profile for a new account (default schwab: $0 commission + SEC/TAF on sells).")
    parser.add_argument("--fixed-fee", type=float, help="Override the profile's dollar fee per order (new account).")
    parser.add_argument("--fractional", action="store_true", help="New account trades fractional shares.")
    parser.add_argument("--advisory", choices=["off", "log", "apply"],
                        help="Advisory verdict mode for this run (default: account setting, 'log').")
    parser.add_argument("--tax-aware", action="store_true",
                        help="Enable US taxable-account lot tracking and tax-aware trade policy (persisted).")
    parser.add_argument("--macro-reaction", choices=["off", "log", "apply"],
                        help="Macro release reactions for this run (default: off). apply uses only proven rules.")
    parser.add_argument("--black-swan", choices=["off", "log", "apply"],
                        help="Black swan playbook for this run (default: log). apply changes paper targets only.")
    parser.add_argument("--schwab-quotes", action="store_true",
                        help="After the run, cache Schwab quotes for current targets/positions (market data only).")
    parser.add_argument("--config", help="JSON strategy config for a new account.")
    parser.add_argument("--state-db", default=str(STATE_DB))
    parser.add_argument("--no-status-file", action="store_true")
    parser.add_argument("--sync-prices", action="store_true",
                        help="First download missing sessions like the dashboard's launch sync (needs MASSIVE_API_KEY).")
    args = parser.parse_args(argv)
    from pathlib import Path
    with closing(connect(Path(args.state_db))) as connection:
        if args.command == "status":
            print(json.dumps(_clean_json(status(connection, args.account)), indent=2, default=str))
            return
        if args.sync_prices and not args.dry_run:
            print(json.dumps({"priceSync": sync_prices()}, default=str), file=sys.stderr, flush=True)
        settings = {k: v for k, v in {"execution": args.execution, "initial_equity": args.initial_equity,
                                        "fixed_fee": args.fixed_fee, "broker": args.broker}.items() if v is not None}
        if args.fractional:
            settings["whole_shares"] = False
        if args.advisory:
            settings["advisory_mode"] = args.advisory
        if args.tax_aware:
            settings["tax_aware"] = True
        if getattr(args, "macro_reaction", None):
            settings["macro_reaction"] = args.macro_reaction
        if getattr(args, "black_swan", None):
            settings["black_swan"] = args.black_swan
        config = json.loads(args.config) if args.config else None
        print(f"{datetime.now():%H:%M:%S} loading prices", file=sys.stderr, flush=True)
        panel = load_panel(end=args.asof)
        connection.execute("BEGIN IMMEDIATE")
        try:
            trader = PaperTrader(connection, args.account, panel, config, settings)
            result = trader.run(args.asof, "dry-run" if args.dry_run else "run")
            snapshot = status(connection, args.account)
            snapshot["dryRun"] = bool(args.dry_run)
            snapshot["lastRun"] = result
            if args.dry_run:
                connection.execute("ROLLBACK")
            else:
                connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        if args.schwab_quotes and not args.dry_run:
            snapshot["schwabQuotes"] = schwab_quotes_for(snapshot)
        try:
            from attribution import attribute_paper_account
            snapshot["attribution"] = attribute_paper_account(None if args.dry_run else connection, args.account)
        except Exception as error:
            snapshot["attribution"] = {"name": "paper_account", "error": str(error)[:300]}
        if not args.no_status_file:
            write_status(snapshot, STATUS_FILE.with_name("paper_status_dryrun.json") if args.dry_run else STATUS_FILE)
        print(json.dumps(_clean_json({"result": result, "openOrders": snapshot["openOrders"],
                                      "targets": snapshot["targets"],
                                      "reconciliation": snapshot["reconciliation"][:1]}), indent=2, default=str))


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("paper_trader"):
        main()
