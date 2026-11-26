"""Day-trading component: regime-filtered opening-range breakout (ORB), flat by the close.

Signal (long only by default, optional shorts):
  * Opening range = the first ``or_minutes`` of the regular session (09:30 ET).
  * Daily context, computed from the Massive daily database through the PREVIOUS session (no lookahead):
      - symbol trend: previous close above its 20-day SMA (longs) / below (shorts),
      - market regime: SPY 20-day realized vol at or below the ``max_vol_pct`` percentile of its trailing year
        (skip the most turbulent tape), same estimator as the swing overlay (risk.realized_vol),
      - liquidity: 20-day median dollar volume >= ``min_dollar_volume`` and price > $5 (universe filters).
  * "In play": opening-range volume / average opening-range volume of the prior 10 sessions >= ``min_rel_volume``.
  * Entry: first bar after the opening range that CLOSES above the range high (below the low for shorts) and
    before ``entry_cutoff``; fill at the NEXT bar's open plus half-spread and slippage.
  * Exit: stop at the other side of the range (or ``stop_frac`` of it), a ``target_r`` x risk target, or the
    close of the last bar of the day (flat by close). A gap through the stop fills at the bar open.
Risk: risk ``risk_per_trade`` of equity per trade, notional <= ``max_notional_frac`` x equity (no leverage),
``max_positions`` at once, a daily loss circuit breaker, and the FINRA pattern-day-trader limit (at most
``pdt_max_day_trades`` day trades in any 5 rolling sessions while equity < $25,000).
Costs: Schwab profile ($0 commission) + half-spread by dollar volume + square-root impact + extra intraday
slippage (more for stop fills) + SEC/TAF on sells (costs.CostModel).
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from costs import CostModel
from metrics import performance
from portfolio_sim import DECISION_COLUMNS
from risk import realized_vol

OPEN_MINUTE = 570          # 09:30 ET
PDT_EQUITY = 25_000.0


@dataclass
class ORBParams:
    or_minutes: int = 30
    entry_cutoff: int = 690            # 11:30 ET, minutes after midnight
    target_r: float | None = 2.0       # None = hold to the close unless stopped
    stop_frac: float = 1.0             # 1.0 = stop at the far side of the range; 0.5 = range midpoint
    allow_short: bool = False
    min_rel_volume: float = 1.0
    trend_window: int = 20
    max_vol_pct: float = 0.90
    min_dollar_volume: float = 20e6
    min_price: float = 5.0
    risk_per_trade: float = 0.005
    max_notional_frac: float = 1.0
    max_positions: int = 1
    daily_loss_limit: float = 0.02
    pdt_max_day_trades: int | None = 3  # None disables the PDT limit (e.g. equity >= $25k or a cash account)
    pdt_window: int = 5
    slippage_bps: float = 2.0
    stop_slippage_bps: float = 5.0

    def as_dict(self):
        return dict(self.__dict__)


# ------------------------------------------------------------------ data
def load_bars(cache, frequency=5, symbols=None, start=None, end=None):
    query = ("SELECT symbol, session_date, minute, open, high, low, close, volume FROM intraday_bars "
             "WHERE frequency = ?")
    params = [frequency]
    if symbols:
        query += f" AND symbol IN ({','.join('?' * len(symbols))})"
        params += list(symbols)
    if start:
        query += " AND session_date >= ?"
        params.append(start)
    if end:
        query += " AND session_date <= ?"
        params.append(end)
    bars = pd.read_sql_query(query + " ORDER BY symbol, session_date, minute", cache, params=params)
    return bars


def daily_context(panel, symbols, params):
    """Per (session, symbol) flags known BEFORE the session opens (everything shifted by one daily row)."""
    p = params
    cols = [s for s in symbols if s in panel.close.columns]
    close = panel.close[cols]
    sma = close.rolling(p.trend_window, min_periods=p.trend_window).mean()
    adv = (panel.close[cols] * panel.volume[cols]).rolling(20, min_periods=15).median()
    spy = panel.log_returns()["SPY"]
    vol20 = realized_vol(spy, 20, min_periods=15)
    pct = vol20.rolling(252, min_periods=60).apply(lambda w: float((w <= w[-1]).mean()), raw=True)
    daily_vol = panel.total_return()[cols].rolling(20, min_periods=10).std()
    frame = {
        "prev_close": close, "trend_up": close > sma, "trend_down": close < sma, "adv": adv, "daily_vol": daily_vol,
        "liquid": (adv >= p.min_dollar_volume) & (close > p.min_price),
    }
    # Shift by one row: the context for session D uses daily data through the session before D.
    out = {k: v.shift(1) for k, v in frame.items()}
    regime_ok = (pct <= p.max_vol_pct).shift(1)
    rows = []
    for day in panel.dates:
        for symbol in cols:
            rows.append((day.strftime("%Y-%m-%d"), symbol, *(out[k].at[day, symbol] for k in out),
                         regime_ok.get(day, np.nan)))
    context = pd.DataFrame(rows, columns=["session_date", "symbol", *out, "regime_ok"])
    return context


def next_session_context(panel, symbols, params, session_date):
    """Context for a session after the last daily row (e.g. today, before Massive has today's bar)."""
    p = params
    cols = [s for s in symbols if s in panel.close.columns]
    close = panel.close[cols]
    sma = close.rolling(p.trend_window, min_periods=p.trend_window).mean().iloc[-1]
    adv = (panel.close[cols] * panel.volume[cols]).rolling(20, min_periods=15).median().iloc[-1]
    spy = panel.log_returns()["SPY"]
    vol20 = realized_vol(spy, 20, min_periods=15)
    window = vol20.iloc[-252:]
    pct = float((window <= window.iloc[-1]).mean())
    last = close.iloc[-1]
    return pd.DataFrame({"session_date": session_date, "symbol": cols, "prev_close": last.values,
                         "trend_up": (last > sma).values, "trend_down": (last < sma).values, "adv": adv.values,
                         "daily_vol": panel.total_return()[cols].rolling(20, min_periods=10).std().iloc[-1].values,
                         "liquid": ((adv >= p.min_dollar_volume) & (last > p.min_price)).values,
                         "regime_ok": pct <= p.max_vol_pct})


# ------------------------------------------------------------------ signals
def opening_ranges(bars, params, frequency):
    p = params
    n_or = p.or_minutes // frequency
    rows = []
    for (symbol, day), g in bars.groupby(["symbol", "session_date"], sort=True):
        first = g[g["minute"] < OPEN_MINUTE + p.or_minutes]
        if len(first) < n_or or first["minute"].iloc[0] != OPEN_MINUTE:
            continue
        rows.append((symbol, day, first["high"].max(), first["low"].min(), first["volume"].sum()))
    ranges = pd.DataFrame(rows, columns=["symbol", "session_date", "or_high", "or_low", "or_volume"])
    ranges = ranges.sort_values(["symbol", "session_date"])
    prior = ranges.groupby("symbol")["or_volume"].transform(lambda v: v.shift(1).rolling(10, min_periods=5).mean())
    ranges["rel_volume"] = ranges["or_volume"] / prior
    return ranges


def candidate_trades(bars, ranges, context, params, frequency, cost_model, preview_session=None):
    """Every qualifying breakout with its full path to exit (selection under limits happens later)."""
    p = params
    merged = ranges.merge(context, on=["symbol", "session_date"], how="inner")
    merged = merged[merged["liquid"].fillna(False).astype(bool) & merged["regime_ok"].fillna(False).astype(bool)
                    & (merged["rel_volume"] >= p.min_rel_volume)]
    groups = {key: g for key, g in bars.groupby(["symbol", "session_date"], sort=False)}
    trades = []
    for row in merged.itertuples(index=False):
        g = groups.get((row.symbol, row.session_date))
        if g is None:
            continue
        minutes, o, h, l, c = (g[k].to_numpy() for k in ("minute", "open", "high", "low", "close"))
        after = np.flatnonzero((minutes >= OPEN_MINUTE + p.or_minutes) & (minutes < p.entry_cutoff))
        directions = []
        if bool(row.trend_up):
            directions.append(1)
        if p.allow_short and bool(row.trend_down):
            directions.append(-1)
        for direction in directions:
            level = row.or_high if direction > 0 else row.or_low
            hits = [i for i in after if (c[i] > level if direction > 0 else c[i] < level)]
            if not hits or hits[0] + 1 >= len(minutes):
                continue
            signal = hits[0]
            entry_i = signal + 1
            width = row.or_high - row.or_low
            stop = (row.or_high - p.stop_frac * width) if direction > 0 else (row.or_low + p.stop_frac * width)
            raw_entry = o[entry_i]
            if (direction > 0 and raw_entry <= stop) or (direction < 0 and raw_entry >= stop):
                continue                                   # gapped through the stop before we got in
            risk = abs(raw_entry - stop)
            target = raw_entry + direction * p.target_r * risk if p.target_r else None
            exit_i, exit_raw = len(minutes) - 1, c[-1]
            reason = "open" if row.session_date == preview_session else "close"
            for i in range(entry_i, len(minutes)):
                if direction > 0:
                    if l[i] <= stop:
                        exit_i, exit_raw, reason = i, min(o[i], stop), "stop"
                        break
                    if target is not None and h[i] >= target:
                        exit_i, exit_raw, reason = i, max(o[i], target), "target"
                        break
                else:
                    if h[i] >= stop:
                        exit_i, exit_raw, reason = i, max(o[i], stop), "stop"
                        break
                    if target is not None and l[i] <= target:
                        exit_i, exit_raw, reason = i, min(o[i], target), "target"
                        break
            trades.append({"symbol": row.symbol, "session_date": row.session_date, "direction": direction,
                           "signal_minute": int(minutes[signal]), "entry_minute": int(minutes[entry_i]),
                           "exit_minute": int(minutes[exit_i]) + frequency, "raw_entry": float(raw_entry),
                           "raw_exit": float(exit_raw), "stop": float(stop), "risk": float(risk),
                           "reason": reason, "rel_volume": float(row.rel_volume), "adv": row.adv,
                           "daily_vol": row.daily_vol})
    return pd.DataFrame(trades)


# ------------------------------------------------------------------ selection under limits + P&L
@dataclass
class DayTradeResult:
    trades: pd.DataFrame
    daily: pd.DataFrame
    skipped: dict = field(default_factory=dict)


def run_sessions(candidates, sessions, params, initial_equity=10_000.0, cost_model=None, pdt_history=None,
                 prior_sessions=()):
    """Walk sessions in order, taking candidates chronologically subject to slots, PDT and the loss breaker."""
    p = params
    cost_model = cost_model or CostModel.for_broker("schwab")
    equity = float(initial_equity)
    day_trade_log = list(pdt_history or [])          # session dates of earlier day trades (one entry per trade)
    calendar = list(prior_sessions) + list(sessions)
    taken, daily_rows = [], []
    skipped = {"pdt": 0, "slots": 0, "breaker": 0, "size": 0}
    by_day = {d: g.sort_values(["entry_minute", "rel_volume"], ascending=[True, False])
              for d, g in candidates.groupby("session_date")} if len(candidates) else {}
    for index, day in enumerate(sessions):
        start_equity = equity
        offset = len(prior_sessions) + index
        recent_sessions = set(calendar[max(0, offset - p.pdt_window + 1):offset])
        used = sum(1 for d in day_trade_log if d in recent_sessions)
        open_until = []                                     # (exit minute, notional) of open positions
        realized = []                                       # (exit_minute, pnl)
        for t in (by_day.get(day, pd.DataFrame()).itertuples(index=False)):
            open_until = [(m, n) for m, n in open_until if m > t.entry_minute]
            if len(open_until) >= p.max_positions:
                skipped["slots"] += 1
                continue
            closed_pnl = sum(pnl for m, pnl in realized if m <= t.entry_minute)
            if closed_pnl <= -p.daily_loss_limit * start_equity:
                skipped["breaker"] += 1
                continue
            limit_applies = p.pdt_max_day_trades is not None and start_equity < PDT_EQUITY
            if limit_applies and used >= p.pdt_max_day_trades:
                skipped["pdt"] += 1
                continue
            d = t.direction
            half = cost_model.half_spread_bps(t.adv)
            scale = getattr(t, "risk_scale", 1.0)            # advisory hook may shrink (never enlarge) size
            scale = 1.0 if scale is None or scale != scale else min(max(scale, 0.0), 1.0)
            buying_power = p.max_notional_frac * start_equity - sum(n for _, n in open_until)   # no leverage
            shares = math.floor(scale * min(p.risk_per_trade * start_equity / t.risk,
                                            max(buying_power, 0.0) / t.raw_entry))
            if shares < 1:
                skipped["size"] += 1
                continue
            notional = shares * t.raw_entry
            impact = cost_model.impact_bps(notional, t.adv, t.daily_vol)
            entry = t.raw_entry * (1 + d * (half + impact + p.slippage_bps) / 1e4)
            exit_slip = p.stop_slippage_bps if t.reason == "stop" else p.slippage_bps
            exit_price = t.raw_exit * (1 - d * (half + impact + exit_slip) / 1e4)
            fees = cost_model.regulatory_fee(shares * (exit_price if d > 0 else entry), "sell",
                                             exit_price if d > 0 else entry)
            fees += 2 * cost_model.fixed_fee + cost_model.commission_bps / 1e4 * shares * (entry + exit_price)
            pnl = d * shares * (exit_price - entry) - fees
            gross = d * shares * (t.raw_exit - t.raw_entry)
            taken.append({**t._asdict(), "shares": shares, "entry_price": entry, "exit_price": exit_price,
                          "fees": fees, "pnl": pnl, "gross_pnl": gross, "cost": gross - pnl,
                          "r_multiple": pnl / (shares * t.risk) if t.risk else None,
                          "return_on_equity": pnl / start_equity})
            open_until.append((t.exit_minute, notional))
            realized.append((t.exit_minute, pnl))
            used += 1
            day_trade_log.append(day)
        day_pnl = sum(pnl for _, pnl in realized)
        equity += day_pnl
        daily_rows.append({"session_date": day, "start_equity": start_equity, "pnl": day_pnl, "equity": equity,
                           "trades": len(realized), "return": day_pnl / start_equity if start_equity else 0.0})
    return DayTradeResult(pd.DataFrame(taken), pd.DataFrame(daily_rows), skipped)


def summarize(result, frequency):
    daily = result.daily
    if daily.empty:
        return {"sessions": 0}
    returns = pd.Series(daily["return"].to_numpy(), index=pd.to_datetime(daily["session_date"]))
    perf = performance(returns)
    trades = result.trades
    out = {"sessions": int(len(daily)), "first": daily["session_date"].iloc[0], "last": daily["session_date"].iloc[-1],
           "barMinutes": frequency, "performance": perf, "skipped": result.skipped,
           "trades": int(len(trades)), "daysWithTrade": int((daily["trades"] > 0).sum())}
    if len(trades):
        held = (trades["exit_minute"] - trades["entry_minute"]).clip(lower=0)
        out.update({"winRate": float((trades["pnl"] > 0).mean()), "avgR": float(trades["r_multiple"].mean()),
                    "avgPnl": float(trades["pnl"].mean()), "totalPnl": float(trades["pnl"].sum()),
                    "totalCosts": float(trades["cost"].sum()),
                    "costPerTradeBps": float((trades["cost"] / (trades["shares"] * trades["raw_entry"])).mean() * 1e4),
                    "avgHoldMinutes": float(held.mean()),
                    "exitReasons": trades["reason"].value_counts().to_dict(),
                    "timeInMarketPct": float(held.sum() / (len(daily) * 390) * 100)})
    return out


def backtest(bars, panel, params, frequency, initial_equity=10_000.0, cost_model=None, sessions=None):
    cost_model = cost_model or CostModel.for_broker("schwab")
    symbols = sorted(bars["symbol"].unique())
    context = daily_context(panel, symbols, params)
    last_daily = panel.dates[-1].strftime("%Y-%m-%d")
    later = sorted(d for d in bars["session_date"].unique() if d > last_daily)
    if later:   # e.g. today: the daily database ends yesterday, which is exactly the context needed
        context = pd.concat([context, next_session_context(panel, symbols, params, later[0])], ignore_index=True)
    ranges = opening_ranges(bars, params, frequency)
    candidates = candidate_trades(bars, ranges, context, params, frequency, cost_model)
    sessions = sessions or sorted(bars["session_date"].unique())
    if len(candidates):
        candidates = candidates[candidates["session_date"].isin(sessions)]
    return run_sessions(candidates, sessions, params, initial_equity, cost_model), candidates


def decisions_from_trades(trades):
    """Map day trades onto the shared decision-log schema (two rows per trade: entry and exit)."""
    rows = []
    for t in trades.itertuples(index=False):
        for action, price, minute, reason in (("buy" if t.direction > 0 else "sell_short", t.entry_price,
                                               t.entry_minute, "orb_entry"),
                                              ("sell" if t.direction > 0 else "buy_to_cover", t.exit_price,
                                               t.exit_minute, f"orb_{t.reason}")):
            rows.append(dict(zip(DECISION_COLUMNS, [t.session_date, f"{t.session_date} {minute // 60:02d}:{minute % 60:02d}",
                                                    t.symbol, "daytrade_orb", action, reason, None, None,
                                                    t.raw_entry, t.shares if action in ("buy", "buy_to_cover") else -t.shares,
                                                    price, t.shares * price, None, None, "filled"])))
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)
