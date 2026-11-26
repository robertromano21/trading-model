"""Event-driven daily portfolio simulator shared by research backtests and the paper trader's shadow backtest.

Timing (default ``execution='next_open'``): targets decided from data through the close of day t are filled at the
open of day t+1, so overnight gaps hit the fill. ``execution='close'`` fills at the same close (a market-on-close
approximation; slightly optimistic because the signal also uses that close).

Every order pays CostModel costs (commission + half-spread by dollar volume + square-root impact by volatility and
size). Protective stops gap: if the open is through the stop the fill is the open, otherwise the stop price when
the day's low touches it. Cash dividends are credited on the ex-date. A position whose ticker stops printing for
more than ``stale_sessions`` sessions (halt/delisting) is written off at its last close and logged. A daily loss
beyond the circuit breaker limit blocks new buying for the cooldown period.

``whole_shares=True`` floors every target to whole shares and ``min_order_value`` skips orders smaller than that
many dollars (full exits always go through); both default off, which keeps the original fractional path.

Optional ``tax_config`` (see tax.TaxConfig): when ``enabled`` is false (the default) this function is
bit-identical to the pre-tax path. When enabled, a lot ledger tracks FIFO/HIFO lots, wash sales and
dividends, reserves tax, and may adjust non-forced trades (defer / harvest / wash avoidance). This is a planning model, not tax advice.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from costs import CostModel
from risk import CircuitBreaker
from universe import dollar_volume

# One schema for backtest and paper-trading decisions so they can be compared row by row.
DECISION_COLUMNS = ["decision_date", "fill_date", "ticker", "sleeve", "action", "reason", "target_weight",
                    "current_weight", "ref_price", "shares", "fill_price", "notional", "cost_bps",
                    "slippage_bps", "status"]
TRADE_COLUMNS = ["ticker", "sleeve", "entry_date", "exit_date", "entry_price", "exit_price", "max_shares",
                 "holding_sessions", "pnl", "return", "costs", "exit_reason"]


@dataclass
class SimResult:
    equity: pd.Series
    returns: pd.Series
    gross_exposure: pd.Series
    turnover: pd.Series
    costs: pd.Series
    holdings: pd.DataFrame
    decisions: pd.DataFrame
    trades: pd.DataFrame
    final_positions: dict = field(default_factory=dict)
    final_cash: float = 0.0
    after_tax_equity: object = None
    tax_paid_series: object = None
    tax_ledger: object = None
    tax_report: dict | None = None


def simulate(panel, weights, stops=None, cost_model=None, initial_equity=100_000.0, execution="next_open",
             start=None, end=None, stale_sessions=5, breaker=None, min_trade_weight=0.02, sleeve_of=None,
             stop_cooldown=True, tax_config=None, tax_ledger=None, whole_shares=False, min_order_value=0.0):
    cost_model = cost_model or CostModel()
    breaker = breaker or CircuitBreaker()
    ledger = None
    if tax_config is not None and getattr(tax_config, "enabled", False):
        from tax import LotLedger
        ledger = tax_ledger if tax_ledger is not None else LotLedger(tax_config)
    out_after, out_tax_paid = [], []
    dates = panel.dates
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    columns = list(weights.columns)
    col = {t: i for i, t in enumerate(columns)}
    frame = lambda f: f.reindex(index=panel.dates, columns=columns)
    opens, highs, lows, closes = (frame(panel.open).to_numpy(), frame(panel.high).to_numpy(),
                                  frame(panel.low).to_numpy(), frame(panel.close).to_numpy())
    divs = frame(panel.dividends).fillna(0.0).to_numpy()
    masked = (frame(panel.masked_returns).fillna(False).astype(bool).to_numpy()
              if panel.masked_returns is not None else np.zeros(opens.shape, dtype=bool))
    last_close = frame(panel.close).ffill().to_numpy()
    adv = frame(dollar_volume(panel)).to_numpy()
    daily_vol = frame(panel.total_return().rolling(20, min_periods=10).std()).to_numpy()
    target = weights.reindex(index=panel.dates).fillna(0.0).to_numpy()
    stop_values = stops.reindex(index=panel.dates, columns=columns).to_numpy() if stops is not None else None
    row_of = {d: i for i, d in enumerate(panel.dates)}

    cash, shares = float(initial_equity), {}
    stale = {}
    open_trades = {}
    blocked = set()
    equity_prev = float(initial_equity)
    out_equity, out_returns, out_gross, out_turnover, out_costs, out_holdings = [], [], [], [], [], []
    decisions, trades = [], []

    def sleeve(day, ticker):
        return sleeve_of(day, ticker) if sleeve_of else None

    def trade_cost(notional, i, j, side="buy", price=None):
        return cost_model.cost_bps(notional, adv[i, j], daily_vol[i, j], side, price)

    def book(ticker, delta, price, cost_value, day, reason, decision_day):
        nonlocal cash
        j = col[ticker]
        before = shares.get(ticker, 0.0)
        after = before + delta
        cash -= delta * price + cost_value
        if ledger is not None and abs(delta) > 1e-12:
            if delta > 0:
                ledger.buy(ticker, delta, price, day, cost_value)
            else:
                ledger.sell(ticker, -delta, price, day, cost_value)
        record = open_trades.get(ticker)
        if before == 0 and delta > 0:
            record = open_trades[ticker] = {"ticker": ticker, "sleeve": sleeve(decision_day, ticker),
                                            "entry_date": day, "cost_basis": 0.0, "proceeds": 0.0,
                                            "costs": 0.0, "max_shares": 0.0, "bought": 0.0}
        if record is not None:
            record["costs"] += cost_value
            if delta > 0:
                record["cost_basis"] += delta * price
                record["bought"] += delta
            else:
                record["proceeds"] += -delta * price
            record["max_shares"] = max(record["max_shares"], after)
        if abs(after) < 1e-9:
            shares.pop(ticker, None)
            if record is not None:
                open_trades.pop(ticker, None)
                pnl = record["proceeds"] + record.get("dividends", 0.0) - record["cost_basis"] - record["costs"]
                entry_price = record["cost_basis"] / record["bought"] if record["bought"] else float("nan")
                held = row_of[day] - row_of[record["entry_date"]]
                trades.append({"ticker": ticker, "sleeve": record["sleeve"], "entry_date": record["entry_date"],
                               "exit_date": day, "entry_price": entry_price, "exit_price": price,
                               "max_shares": record["max_shares"], "holding_sessions": held, "pnl": pnl,
                               "return": pnl / record["cost_basis"] if record["cost_basis"] else float("nan"),
                               "costs": record["costs"], "exit_reason": reason})
        else:
            shares[ticker] = after

    def execute_orders(decision_row, fill_row, price_matrix, day):
        """Move holdings toward target[decision_row] using fills at price_matrix[fill_row]."""
        decision_day = panel.dates[decision_row]
        equity_ref = cash + sum(n * last_close[decision_row, col[t]] for t, n in shares.items())
        wanted = target[decision_row]
        if ledger is not None:
            from tax_policy import adjust_targets
            wanted = np.array(wanted, dtype=float, copy=True)
            prices = {t: float(last_close[decision_row, col[t]]) for t in columns
                      if np.isfinite(last_close[decision_row, col[t]]) and last_close[decision_row, col[t]] > 0}
            targets_map = {columns[j]: float(wanted[j]) for j in range(len(columns))}
            current_map = {}
            if equity_ref:
                for t, n in shares.items():
                    px = prices.get(t)
                    if px:
                        current_map[t] = n * px / equity_ref
            adjusted, actions = adjust_targets(
                ledger, targets_map, prices, day, equity_ref, tax_config,
                current=current_map, available=set(columns))
            wanted[:] = 0.0
            for t, w in adjusted.items():
                if t in col:
                    wanted[col[t]] = w
            for act in actions:
                decisions.append(_decision(
                    decision_day, None, act.get("ticker"), sleeve(decision_day, act.get("ticker")),
                    "adjust", act.get("kind"), act.get("toWeight"), act.get("fromWeight"),
                    prices.get(act.get("ticker")), None, None, 0, None, None, "tax_adjusted"))
        traded = cost_total = 0.0
        names = set(shares) | {columns[j] for j in np.flatnonzero(wanted)}
        for ticker in sorted(names):
            j = col[ticker]
            w = float(wanted[j])
            if w == 0:
                blocked.discard(ticker)
            if ticker in blocked:
                w = 0.0
            ref = last_close[decision_row, j]
            if not np.isfinite(ref) or ref <= 0:
                continue
            current = shares.get(ticker, 0.0)
            current_weight = current * ref / equity_ref if equity_ref else 0.0
            if abs(w - current_weight) < min_trade_weight and not (w == 0 and current != 0):
                continue
            desired = w * equity_ref / ref
            if whole_shares:          # small-account realism (paper trader / swarm): no fractional shares
                desired = float(math.floor(desired + 1e-9))
            delta = desired - current
            if abs(delta) < 1e-12 or (min_order_value and abs(delta) * ref < min_order_value
                                      and not (w == 0 and current != 0)):
                continue
            if delta > 0 and breaker.blocks_buying():
                decisions.append(_decision(decision_day, None, ticker, sleeve(decision_day, ticker), "buy",
                                           "circuit_breaker", w, current_weight, ref, delta, None, 0, None, None,
                                           "blocked"))
                continue
            price = price_matrix[fill_row, j]
            if not np.isfinite(price) or price <= 0:
                decisions.append(_decision(decision_day, None, ticker, sleeve(decision_day, ticker),
                                           "buy" if delta > 0 else "sell", "no_print", w, current_weight, ref,
                                           delta, None, 0, None, None, "unfilled"))
                continue
            notional = abs(delta) * price
            bps = trade_cost(notional, decision_row, j, "buy" if delta > 0 else "sell", price)
            cost_value = notional * bps / 1e4
            reason = "exit" if w == 0 else ("entry" if current == 0 else "rebalance")
            book(ticker, delta, price, cost_value, day, reason, decision_day)
            effective = price * (1 + math.copysign(bps, delta) / 1e4)
            slippage = (effective / ref - 1) * 1e4 * (1 if delta > 0 else -1)
            decisions.append(_decision(decision_day, day, ticker, sleeve(decision_day, ticker),
                                       "buy" if delta > 0 else "sell", reason, w, current_weight, ref, delta,
                                       effective, notional, bps, slippage, "filled"))
            traded += notional
            cost_total += cost_value
        return traded, cost_total

    first = row_of[dates[0]] if len(dates) else 0
    for day in dates:
        i = row_of[day]
        traded = cost_total = 0.0
        # 1. Opening auction: yesterday's decisions.
        if execution == "next_open" and i > first:
            t, c = execute_orders(i - 1, i, opens, day)
            traded += t
            cost_total += c
        breaker.tick()
        # 2. Intraday protective stops (valid from the previous decision).
        if stop_values is not None and i > first:
            for ticker in sorted(shares):
                j = col[ticker]
                level = stop_values[i - 1, j]
                if not np.isfinite(level) or shares[ticker] <= 0 or not np.isfinite(lows[i, j]):
                    continue
                if opens[i, j] <= level or lows[i, j] <= level:
                    price = opens[i, j] if opens[i, j] <= level else level
                    notional = shares[ticker] * price
                    bps = trade_cost(notional, i - 1, j, "sell", price)
                    cost_value = notional * bps / 1e4
                    qty = shares[ticker]
                    current_weight = qty * last_close[i - 1, j] / equity_prev if equity_prev else 0
                    book(ticker, -qty, price, cost_value, day, "stop_gap" if price == opens[i, j] else "stop", panel.dates[i - 1])
                    decisions.append(_decision(panel.dates[i - 1], day, ticker, sleeve(panel.dates[i - 1], ticker),
                                               "sell", "stop", 0.0, current_weight, level, -qty,
                                               price * (1 - bps / 1e4), notional, bps,
                                               (1 - price * (1 - bps / 1e4) / level) * 1e4, "filled"))
                    traded += notional
                    cost_total += cost_value
                    if stop_cooldown:
                        blocked.add(ticker)
        # 3. Dividends (ex-date), masked corporate-action days and stale / delisted positions.
        for ticker in list(shares):
            j = col[ticker]
            if masked[i, j] and i > 0 and np.isfinite(closes[i, j]) and np.isfinite(last_close[i - 1, j]):
                # Spin-off or similar: the price drop is not a loss to the holder (the distributed shares are not in
                # the data), so the day is booked as a zero return for this name.
                adjustment = shares[ticker] * (last_close[i - 1, j] - closes[i, j])
                cash += adjustment
                if ticker in open_trades:
                    open_trades[ticker]["dividends"] = open_trades[ticker].get("dividends", 0.0) + adjustment
                decisions.append(_decision(day, day, ticker, sleeve(day, ticker), "adjust", "corporate_action", None,
                                           None, last_close[i - 1, j], 0.0, None, adjustment, 0.0, None, "booked"))
            if divs[i, j]:
                cash += shares[ticker] * divs[i, j]
                if ticker in open_trades:
                    open_trades[ticker]["dividends"] = open_trades[ticker].get("dividends", 0.0) + shares[ticker] * divs[i, j]
                if ledger is not None:
                    ledger.dividend(ticker, float(divs[i, j]), day, shares[ticker])
            if np.isfinite(closes[i, j]):
                stale[ticker] = 0
            else:
                stale[ticker] = stale.get(ticker, 0) + 1
                if stale[ticker] > stale_sessions:
                    price = last_close[i, j]
                    qty = shares[ticker]
                    book(ticker, -qty, price, 0.0, day, "stale_exit", day)
                    decisions.append(_decision(day, day, ticker, sleeve(day, ticker), "sell", "stale_exit", 0.0,
                                               None, price, -qty, price, qty * price, 0.0, 0.0, "written_off"))
                    blocked.add(ticker)
        paid_tax = 0.0
        if ledger is not None:
            paid_tax = float(ledger.maybe_settle(day, cash) or 0.0)
            if paid_tax:
                cash -= paid_tax
                decisions.append(_decision(day, day, "CASH", None, "adjust", "tax_payment", None, None,
                                           None, 0.0, None, paid_tax, 0.0, None, "booked"))
        # 4. Market-on-close execution of today's decisions.
        if execution == "close":
            t, c = execute_orders(i, i, closes, day)
            traded += t
            cost_total += c
        equity = cash + sum(n * last_close[i, col[t]] for t, n in shares.items())
        daily = equity / equity_prev - 1 if equity_prev else 0.0
        breaker.update(daily)
        gross = sum(abs(n) * last_close[i, col[t]] for t, n in shares.items()) / equity if equity else 0.0
        out_equity.append(equity)
        out_returns.append(daily)
        out_gross.append(gross)
        out_turnover.append(traded / equity_prev if equity_prev else 0.0)
        out_costs.append(cost_total / equity_prev if equity_prev else 0.0)
        out_holdings.append({t: n * last_close[i, col[t]] / equity for t, n in shares.items()} if equity else {})
        if ledger is not None:
            out_after.append(equity - ledger.unpaid_tax)
            out_tax_paid.append(paid_tax)
        equity_prev = equity
    index = pd.DatetimeIndex(dates)
    holdings = pd.DataFrame(out_holdings, index=index).fillna(0.0) if out_holdings else pd.DataFrame(index=index)
    result = SimResult(pd.Series(out_equity, index=index), pd.Series(out_returns, index=index),
                       pd.Series(out_gross, index=index), pd.Series(out_turnover, index=index),
                       pd.Series(out_costs, index=index), holdings,
                       pd.DataFrame(decisions, columns=DECISION_COLUMNS), pd.DataFrame(trades, columns=TRADE_COLUMNS),
                       dict(shares), cash)
    if ledger is not None:
        result.after_tax_equity = pd.Series(out_after, index=index)
        result.tax_paid_series = pd.Series(out_tax_paid, index=index)
        result.tax_ledger = ledger
        last_prices = {}
        if len(dates):
            last_i = row_of[dates[-1]]
            last_prices = {t: float(last_close[last_i, col[t]]) for t in columns
                           if np.isfinite(last_close[last_i, col[t]])}
        result.tax_report = ledger.snapshot(last_prices, dates[-1] if len(dates) else None)
    return result


def _decision(decision_date, fill_date, ticker, sleeve, action, reason, target_weight, current_weight, ref_price,
              shares, fill_price, notional, cost_bps, slippage_bps, status):
    return dict(zip(DECISION_COLUMNS, [decision_date, fill_date, ticker, sleeve, action, reason, target_weight,
                                       current_weight, ref_price, shares, fill_price, notional, cost_bps,
                                       slippage_bps, status]))
