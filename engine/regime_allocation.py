"""Volatility- and overextension-aware sleeve allocation.

Compares how much of a paper book should sit in the plain TSMOM risk-on sleeve versus a defensive
sleeve (T-bills, Treasuries, gold) and an inverse/hedge sleeve (short-equity ETFs, long vol, and a
modeled protective put) as realized and implied vol rise and fall, and as the market gets stretched.

Signals are point-in-time (data through the close of t only). Target weights decided at t are applied
to returns of t+1, with transition smoothing, hysteresis and the shared Schwab cost model on sleeve
turnover. Walk-forward selection uses pre-holdout data only; any sealed-holdout evaluation is logged
as a new look in Massive/Research/swarm/holdout_ledger.json. This module never changes the live
paper-account config and never places orders.

  python regime_allocation.py research   # or: python research.py regime-allocation
  python regime_allocation.py state      # current state + recommended weights
  python regime_allocation.py forward    # shadow swarm-forward variants (no orders)
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from costs import CostModel
from fetch_and_model import DATA_DIR, DATABASE, write_atomically
from market_data import MARKET, Panel, load_panel
from metrics import performance
from risk import TRADING_DAYS
from universe import CORE_ETFS

OUT_DIR = DATA_DIR / "Research" / "regime_allocation"
WARMUP = 110
HOLDOUT_FRACTION = 0.25
WF_TRAIN_MIN = 126
WF_TEST_LEN = 42
MAX_STEP = 0.08
SMOOTH_LAMBDA = 0.45
BUCKET_CONFIRM = 2
INITIAL_EQUITY = 10_000.0

DEFENSIVE_CANDIDATES = ("BIL", "SGOV", "IEF", "TLT", "GLD")
DEFENSIVE_WEIGHTS = {"BIL": 0.45, "SGOV": 0.45, "IEF": 0.25, "TLT": 0.10, "GLD": 0.20}
HEDGE_ETFS = ("SH", "PSQ", "VIXY")
HEDGE_WEIGHTS = {"SH": 0.60, "PSQ": 0.00, "VIXY": 0.20, "put_overlay": 0.20}
CREDIT_ETFS = ("HYG", "LQD", "JNK", "IEF")
EQUITY_BREADTH = ("SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
                  "XLI", "XLB", "XLU", "XLRE", "XLC", "EFA", "EEM")

STATIC_MIXES = {
    "static_100_0_0": (1.00, 0.00, 0.00),
    "static_90_10_0": (0.90, 0.10, 0.00),
    "static_80_20_0": (0.80, 0.20, 0.00),
    "static_70_20_10": (0.70, 0.20, 0.10),
    "static_60_30_10": (0.60, 0.30, 0.10),
    "static_50_30_20": (0.50, 0.30, 0.20),
}

BUCKET_LEVELS = ("low", "normal", "elevated", "high", "crisis")
BUCKET_TABLES = {
    "bucket_moderate": {
        ("low", "falling"): (1.00, 0.00, 0.00), ("low", "rising"): (0.90, 0.10, 0.00),
        ("normal", "falling"): (0.90, 0.10, 0.00), ("normal", "rising"): (0.80, 0.15, 0.05),
        ("elevated", "falling"): (0.70, 0.25, 0.05), ("elevated", "rising"): (0.50, 0.30, 0.20),
        ("high", "falling"): (0.40, 0.40, 0.20), ("high", "rising"): (0.20, 0.40, 0.40),
        ("crisis", "falling"): (0.15, 0.50, 0.35), ("crisis", "rising"): (0.00, 0.50, 0.50),
    },
    "bucket_conservative": {
        ("low", "falling"): (0.85, 0.15, 0.00), ("low", "rising"): (0.75, 0.20, 0.05),
        ("normal", "falling"): (0.75, 0.20, 0.05), ("normal", "rising"): (0.60, 0.25, 0.15),
        ("elevated", "falling"): (0.50, 0.30, 0.20), ("elevated", "rising"): (0.30, 0.40, 0.30),
        ("high", "falling"): (0.20, 0.45, 0.35), ("high", "rising"): (0.05, 0.45, 0.50),
        ("crisis", "falling"): (0.00, 0.50, 0.50), ("crisis", "rising"): (0.00, 0.40, 0.60),
    },
    "bucket_aggressive": {
        ("low", "falling"): (1.00, 0.00, 0.00), ("low", "rising"): (1.00, 0.00, 0.00),
        ("normal", "falling"): (1.00, 0.00, 0.00), ("normal", "rising"): (0.90, 0.10, 0.00),
        ("elevated", "falling"): (0.85, 0.15, 0.00), ("elevated", "rising"): (0.70, 0.20, 0.10),
        ("high", "falling"): (0.55, 0.30, 0.15), ("high", "rising"): (0.35, 0.35, 0.30),
        ("crisis", "falling"): (0.25, 0.40, 0.35), ("crisis", "rising"): (0.10, 0.40, 0.50),
    },
}

KNOWN_SPIKE_WINDOWS = (
    ("2024-08-01", "2024-08-16", "Aug 2024 vol spike"),
    ("2025-04-01", "2025-04-30", "April 2025 vol spike"),
)

SCHWAB_TABLE_CANDIDATES = (
    "daily_long",
    "schwab_daily_prices", "schwab_prices", "long_daily_prices",
    "daily_prices_schwab", "schwab_history", "schwab_long_prices",
    "schwab_daily_long",
)


def log(message):
    print(f"{datetime.now():%H:%M:%S} {message}", file=sys.stderr, flush=True)


def _finite(value, default=None):
    if value is None:
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _json_default(value):
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)[:19]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, json.dumps(payload, indent=1, default=_json_default).encode())


def rolling_percentile(series, window=252, min_periods=60):
    """Causal percentile of today's value in the trailing window (includes today)."""
    return series.rolling(window, min_periods=min_periods).apply(
        lambda w: float((w <= w[-1]).mean()) if len(w) else np.nan, raw=True)


def logistic(x):
    x = np.clip(np.asarray(x, dtype=float), -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def rsi_wilder(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    ad = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _table_exists(connection, name):
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
    return row is not None

def detect_long_history(database=DATABASE):
    """Return table metadata if a Schwab/long daily table is present, else None."""
    path = Path(database)
    if not path.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        for name in SCHWAB_TABLE_CANDIDATES:
            if not _table_exists(con, name):
                continue
            cols = {r[1] for r in con.execute(f"PRAGMA table_info({name})")}
            date_col = next((c for c in ("date", "session", "market_date") if c in cols), None)
            tick_col = next((c for c in ("ticker", "symbol") if c in cols), None)
            if not date_col or not tick_col:
                continue
            mn, mx, n, k = con.execute(
                f"SELECT MIN({date_col}), MAX({date_col}), COUNT(*), COUNT(DISTINCT {tick_col}) FROM {name}"
            ).fetchone()
            if mn and str(mn) < "2024-11-25" and (n or 0) > 0:
                return {"table": name, "start": str(mn)[:10], "end": str(mx)[:10],
                        "rows": int(n), "tickers": int(k), "survivorshipBias": True}
    finally:
        con.close()
    return None


def needed_tickers():
    names = set(CORE_ETFS) | set(DEFENSIVE_CANDIDATES) | set(HEDGE_ETFS) | set(CREDIT_ETFS)
    names.add(MARKET)
    names.update(EQUITY_BREADTH)
    return sorted(names)


def load_study_panel(start=None, end=None, database=DATABASE):
    """Load the Massive daily panel, spliced with Schwab daily_long when present.

    Overlapping dates prefer Massive (split- and dividend-adjusted). Pre-Massive Schwab
    returns are split-adjusted only. The live paper config is never changed here.
    """
    long = detect_long_history(database)
    tickers = needed_tickers()
    notes = []
    panel = load_panel(start, end, tickers=tickers, database=database, use_long_history=True)
    if long and getattr(panel, "used_long_history", False):
        notes.append(
            f"Using long history table {long['table']} {long['start']}..{long['end']} "
            f"({long['tickers']} tickers), spliced with Massive daily_prices "
            "(Massive preferred on overlap). Survivorship bias: delisted names are typically missing "
            "from a broker price-history API, so backtests on this series lean optimistic."
        )
        notes.append(
            "Schwab pre-Massive returns are split-adjusted but not dividend-adjusted; "
            "Massive overlapping dates are split- and dividend-adjusted."
        )
    elif long:
        notes.append(
            f"Long history table {long['table']} was detected but not used by load_panel; "
            "falling back to the Massive window."
        )
    else:
        notes.append(
            "Price history is Massive daily_prices only (about two years). A Schwab 20-year backfill "
            "was not present at run time, so earlier crises are not in the sample."
        )
    if getattr(panel, "warnings", None):
        notes.extend(panel.warnings)
    present = [t for t in tickers if t in panel.close.columns and panel.close[t].notna().any()]
    missing = [t for t in tickers if t not in present]
    return panel, {"longHistory": long, "present": present, "missing": missing, "notes": notes,
                   "start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                   "sessions": int(len(panel.dates))}


def _load_named_table(database, table, start, end, tickers):
    path = Path(database)
    try:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        cols = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        date_col = next((c for c in ("date", "session", "market_date") if c in cols), "date")
        tick_col = next((c for c in ("ticker", "symbol") if c in cols), "ticker")
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(cols):
            con.close()
            return None
        q = (f"SELECT {tick_col} AS ticker, {date_col} AS date, open, high, low, close, volume "
             f"FROM {table} WHERE {date_col} BETWEEN ? AND ?")
        frame = pd.read_sql_query(q, con, params=(start, end))
        dividends = pd.DataFrame(columns=["ticker", "market_date", "cash"])
        if _table_exists(con, "backtest_dividends"):
            dividends = pd.read_sql_query(
                "SELECT ticker, market_date, cash FROM backtest_dividends "
                "WHERE unsupported = 0 AND market_date BETWEEN ? AND ?",
                con, params=(start, end))
        con.close()
    except (sqlite3.Error, pd.errors.DatabaseError, ValueError):
        return None
    if frame.empty:
        return None
    frame["date"] = pd.to_datetime(frame["date"])
    keep = set(tickers) | {MARKET}
    frame = frame[frame.ticker.isin(keep)]
    index = pd.DatetimeIndex(sorted(frame["date"].unique()))
    columns = sorted(frame.ticker.unique())

    def pivot(name):
        return (frame.pivot(index="date", columns="ticker", values=name)
                .reindex(index=index, columns=columns).astype(float))

    dividends = dividends[dividends.ticker.isin(columns)] if len(dividends) else dividends
    if len(dividends):
        dividends["market_date"] = pd.to_datetime(dividends["market_date"])
        cash = (dividends.pivot_table(index="market_date", columns="ticker", values="cash", aggfunc="sum")
                .reindex(index=index, columns=columns).fillna(0.0))
    else:
        cash = pd.DataFrame(0.0, index=index, columns=columns)
    from universe import classify
    return Panel(pivot("open"), pivot("high"), pivot("low"), pivot("close"), pivot("volume"),
                 cash, classify(columns), pd.DataFrame(False, index=index, columns=columns))


def load_vix_frame(dates):
    try:
        import vix_data
        frame = vix_data.load()
    except (SystemExit, FileNotFoundError, OSError, ValueError):
        return pd.DataFrame(index=pd.DatetimeIndex(dates))
    return frame.reindex(pd.DatetimeIndex(dates)).ffill()


def load_macro_series(dates):
    """Lagged FRED rates and credit (1-session publication lag) plus same-day ETF credit proxies."""
    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    try:
        from macro_data import load_macro
        macro = load_macro(lag=1)
        if macro is not None and len(macro):
            macro = macro.reindex(out.index).ffill()
            for col in ("dgs10", "dgs3mo", "hy_oas", "ig_oas", "hy_ig_oas"):
                if col in macro.columns:
                    out[col] = macro[col]
    except Exception:
        pass
    return out


def load_black_swan_state(dates):
    """Optional GB-17 state; returns (crisis series, module name) or (None, None).

    A historical daily series is built from Massive/Research/black_swan/events.jsonl when
    present. The current load_state() snapshot flags only the last session if the live
    machine is in SHOCK (or a negative WATCH). Missing history is treated as 0, not as
    an all-clear over the full sample.
    """
    index = pd.DatetimeIndex(dates)
    series = pd.Series(False, index=index)
    used = None
    try:
        import black_swan
        used = "black_swan"
        path = Path(getattr(black_swan, "EVENTS_PATH", DATA_DIR / "Research" / "black_swan" / "events.jsonl"))
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                day = row.get("date") or row.get("asof") or row.get("session") or row.get("day")
                state = str(row.get("state") or row.get("flag") or row.get("status") or "")
                direction = str(row.get("direction") or row.get("agreedDirection") or "")
                shock = state.upper() in {"SHOCK", "ACTIONABLE"} and direction.lower() in {"", "negative", "mixed"}
                if not shock:
                    continue
                if not day:
                    continue
                ts = pd.Timestamp(str(day)[:10])
                if ts in series.index:
                    series.loc[ts] = True
        try:
            st = black_swan.load_state()
        except Exception:
            st = None
        if isinstance(st, dict):
            state = str(st.get("state") or "")
            direction = str(st.get("direction") or "")
            if state.upper() in {"SHOCK"} or (state.upper() == "WATCH" and direction.lower() == "negative"):
                if len(series):
                    series.iloc[-1] = True
    except ImportError:
        used = None
        for name in ("blackswan", "swan_monitor", "event_monitor"):
            try:
                mod = importlib.import_module(name)
            except ImportError:
                continue
            for attr in ("state_series", "crisis_flag", "threat_series", "load_state"):
                fn = getattr(mod, attr, None)
                if not callable(fn):
                    continue
                got = None
                try:
                    got = fn(index)
                except TypeError:
                    try:
                        got = fn()
                    except Exception:
                        continue
                except Exception:
                    continue
                if got is None:
                    continue
                if isinstance(got, pd.DataFrame):
                    col = next((c for c in ("crisis", "flag", "threat", "active") if c in got.columns), got.columns[0])
                    got = got[col]
                series = pd.Series(got).reindex(index).fillna(False).astype(bool)
                return series, name
        return (None, None)
    if used and (series.any() or used):
        return series.astype(bool), used
    return None, None


def available_defensive(panel):
    """T-bill slot is BIL if present else SGOV; other names keep their weights; then renormalize."""
    present = [t for t in DEFENSIVE_CANDIDATES
               if t in panel.close.columns and panel.close[t].notna().sum() >= 20]
    if not present:
        return {}, present
    weights = {}
    if "BIL" in present:
        weights["BIL"] = DEFENSIVE_WEIGHTS["BIL"]
    elif "SGOV" in present:
        weights["SGOV"] = DEFENSIVE_WEIGHTS["SGOV"]
    for t in ("IEF", "TLT", "GLD"):
        if t in present:
            weights[t] = DEFENSIVE_WEIGHTS[t]
    total = sum(weights.values())
    if total <= 0:
        return {}, present
    return {k: v / total for k, v in weights.items()}, present


def available_hedge(panel, include_put=True):
    present = [t for t in HEDGE_ETFS if t in panel.close.columns and panel.close[t].notna().sum() >= 20]
    weights = {t: HEDGE_WEIGHTS[t] for t in present if HEDGE_WEIGHTS.get(t, 0) > 0}
    if include_put:
        weights["put_overlay"] = HEDGE_WEIGHTS["put_overlay"]
    total = sum(weights.values())
    if total <= 0:
        return {}, present, include_put
    return {k: v / total for k, v in weights.items()}, present, include_put


def _mix_returns(total_returns, weights):
    """Close-to-close mixed return; missing constituents are dropped and weights renormalized per day."""
    cols = [c for c in weights if c in total_returns.columns]
    if not cols:
        return pd.Series(0.0, index=total_returns.index)
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)
    block = total_returns[cols]
    avail = block.notna().astype(float).mul(w, axis=1)
    denom = avail.sum(axis=1).replace(0.0, np.nan)
    return block.fillna(0.0).mul(w, axis=1).sum(axis=1) / denom


def modeled_put_overlay_returns(panel, vix, rate=0.04, otm=0.05, dte=30, roll_every=21, target_premium=0.02):
    """Daily returns of a sleeve that keeps about `target_premium` of NAV in a rolling 5% OTM SPY put.

    Prices are MODELED (Black-Scholes with VIX as IV, plus the modeled bid/ask and Schwab option fees).
    There are no historical option quotes in the database, so decay and roll costs are honest about the
    model but are not a backtest of live fills.
    """
    try:
        from options_pricing import (CONTRACT_MULTIPLIER, OptionFees, OptionSpreadModel,
                                     bsm_greeks, bsm_price)
    except ImportError:
        return None, "options_pricing module not importable"
    if MARKET not in panel.close.columns:
        return None, "SPY missing; cannot model a put overlay"
    close = panel.close[MARKET].astype(float)
    index = close.index
    vix_level = None
    if vix is not None and len(vix):
        col = "vix" if "vix" in vix.columns else vix.columns[0]
        vix_level = vix[col].reindex(index).ffill()
    fees, spreads = OptionFees(), OptionSpreadModel()
    if MARKET in panel.dividends.columns:
        div = panel.dividends[MARKET].rolling(252, min_periods=60).sum() / close.replace(0, np.nan)
    else:
        div = 0.0
    out = pd.Series(0.0, index=index)
    nav = 1.0
    position = None
    last_roll = -roll_every
    notes = "MODELED protective put (BSM + VIX IV + bid/ask + Schwab option fees); not quoted fills."

    def iv_at(i):
        if vix_level is not None and pd.notna(vix_level.iloc[i]) and vix_level.iloc[i] > 0:
            return float(vix_level.iloc[i]) / 100.0
        r = close.iloc[max(0, i - 21):i + 1].pct_change().std()
        return float(r * math.sqrt(TRADING_DAYS)) if r and math.isfinite(r) and r > 0 else 0.20

    def qy(i):
        if isinstance(div, pd.Series) and pd.notna(div.iloc[i]):
            return max(0.0, float(div.iloc[i]))
        return 0.012

    for i, _day in enumerate(index):
        spot = float(close.iloc[i])
        if not math.isfinite(spot) or spot <= 0:
            out.iloc[i] = 0.0
            continue
        vol = max(0.08, iv_at(i))
        yld = qy(i)
        pnl = 0.0
        if position is not None:
            remain = position["expiry_i"] - i
            t = max(remain, 0) / 365.0
            mid = bsm_price(spot, position["strike"], t, vol, rate, yld, "put")
            g = bsm_greeks(spot, position["strike"], t, vol, rate, yld, "put")
            bid, _ask = spreads.quote(mid, abs(g["delta"]), max(remain, 0), is_etf=True)
            marked = bid
            pnl += position["contracts"] * CONTRACT_MULTIPLIER * (marked - position["mark"])
            position["mark"] = marked
            if remain <= 0 or (i - position["open_i"]) >= roll_every:
                pnl -= fees.fee(position["contracts"], marked, "sell")
                position = None
                last_roll = i
        if position is None and (i - last_roll) >= roll_every and i + 5 < len(index):
            strike = round(spot * (1.0 - otm))
            t = dte / 365.0
            mid = bsm_price(spot, strike, t, vol, rate, yld, "put")
            g = bsm_greeks(spot, strike, t, vol, rate, yld, "put")
            _bid, ask = spreads.quote(mid, abs(g["delta"]), dte, is_etf=True)
            if ask > 0.05:
                budget = nav * target_premium
                contracts = max(1, int(budget / (ask * CONTRACT_MULTIPLIER)))
                cost = contracts * CONTRACT_MULTIPLIER * ask + fees.fee(contracts, ask, "buy")
                if cost <= nav * target_premium * 3 and cost > 0:
                    pnl -= cost
                    position = {"strike": strike, "expiry_i": i + dte, "contracts": contracts,
                                "mark": ask, "open_i": i}
                    last_roll = i
        nav = max(1e-9, nav + pnl)
        prev = nav - pnl
        out.iloc[i] = (pnl / prev) if prev > 0 else 0.0
    return out, notes


def tsmom_sleeve_returns(panel, start=None, cost_model=None):
    """Simulate the default plain TSMOM book (next-open fills, Schwab costs)."""
    from portfolio_sim import simulate
    from strategies import TimeSeriesMomentum
    strat = TimeSeriesMomentum()
    weights = strat.target_weights(panel)
    result = simulate(panel, weights, cost_model=cost_model or CostModel(),
                      initial_equity=100_000.0, execution="next_open", start=start)
    return result.returns.rename("risk_on"), result


def build_sleeves(panel, vix=None, cost_model=None, include_put=True):
    """Date-aligned close-to-close returns for risk-on, defensive and hedge sleeves."""
    cost_model = cost_model or CostModel()
    total = panel.total_return()
    missing, notes = [], []
    risk, tsmom_result = tsmom_sleeve_returns(panel, cost_model=cost_model)
    def_w, def_present = available_defensive(panel)
    if def_w:
        defensive = _mix_returns(total, def_w).rename("defensive")
        notes.append("Defensive mix (renormalized over present names): "
                     + str({k: round(v, 3) for k, v in def_w.items()}))
    else:
        defensive = pd.Series(0.0, index=panel.dates, name="defensive")
        missing.append("defensive ETFs (BIL/SGOV/IEF/TLT/GLD)")
        notes.append("No defensive ETFs in the panel; defensive sleeve earns 0.")
    hedge_w, hedge_present, _put_on = available_hedge(panel, include_put=include_put)
    parts = {}
    if "put_overlay" in hedge_w:
        put, put_note = modeled_put_overlay_returns(panel, vix)
        if put is None:
            notes.append(f"Put overlay omitted: {put_note}")
            missing.append("modeled put overlay")
            hedge_w.pop("put_overlay", None)
            total_w = sum(hedge_w.values())
            if total_w:
                hedge_w = {k: v / total_w for k, v in hedge_w.items()}
        else:
            parts["put_overlay"] = put
            notes.append(put_note)
    for t in hedge_present:
        if t in hedge_w:
            parts[t] = total[t]
    if parts:
        w = {k: hedge_w[k] for k in parts if k in hedge_w}
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()} if s else w
        hedge = pd.Series(0.0, index=panel.dates, name="hedge")
        for k, series in parts.items():
            hedge = hedge.add(series.reindex(panel.dates).fillna(0.0) * w.get(k, 0.0), fill_value=0.0)
        notes.append("Hedge mix (renormalized): " + str({k: round(v, 3) for k, v in w.items()}))
    else:
        hedge = pd.Series(0.0, index=panel.dates, name="hedge")
        missing.append("hedge ETFs (SH/PSQ/VIXY) and put overlay")
        notes.append("No hedge instruments available; hedge sleeve earns 0.")
    spy = total[MARKET].rename("spy") if MARKET in total.columns else pd.Series(index=panel.dates, dtype=float)
    frames = pd.concat([risk.reindex(panel.dates), defensive, hedge, spy], axis=1)
    frames.columns = ["risk_on", "defensive", "hedge", "spy"]
    info = {"defensiveWeights": def_w, "defensivePresent": def_present,
            "hedgeWeights": hedge_w, "hedgePresent": hedge_present,
            "putOverlay": bool("put_overlay" in (hedge_w or {})),
            "missing": missing, "notes": notes, "tsmomGross": None}
    if tsmom_result is not None and tsmom_result.gross_exposure is not None:
        info["tsmomGross"] = tsmom_result.gross_exposure.reindex(panel.dates)
    return frames, info, tsmom_result

def build_signals(panel, vix=None, macro=None, black_swan=None):
    """Point-in-time state features indexed by date (no lookahead)."""
    close = panel.close
    if MARKET not in close.columns:
        raise ValueError("SPY is required for allocation signals")
    spy = close[MARKET]
    ret = panel.total_return()[MARKET]
    out = pd.DataFrame(index=panel.dates)
    out["spy_ret"] = ret
    out["spy_close"] = spy
    out["rv5"] = ret.rolling(5, min_periods=5).std() * math.sqrt(TRADING_DAYS)
    out["rv21"] = ret.rolling(21, min_periods=15).std() * math.sqrt(TRADING_DAYS)
    out["rv63"] = ret.rolling(63, min_periods=40).std() * math.sqrt(TRADING_DAYS)
    out["rv21_pct"] = rolling_percentile(out["rv21"], 252, 60)
    out["vol_of_vol"] = out["rv21"].rolling(21, min_periods=10).std()
    ratio = out["rv5"] / out["rv21"].replace(0.0, np.nan)
    out["vol_ratio_5_21"] = ratio
    out["vol_direction"] = np.where(ratio >= 1.10, "rising", np.where(ratio <= 0.90, "falling", "flat"))
    if vix is not None and "vix" in vix.columns:
        out["vix"] = vix["vix"].reindex(out.index).ffill()
        out["vix3m"] = vix["vix3m"].reindex(out.index).ffill() if "vix3m" in vix.columns else np.nan
        out["vix_ratio"] = out["vix"] / out["vix3m"].replace(0.0, np.nan)
        out["vix_pct"] = rolling_percentile(out["vix"], 252, 60)
    elif "VIXY" in close.columns:
        # VIXY is a decaying short-term vol product; use a relative-to-its-mean proxy, never the raw price as VIX.
        vx = close["VIXY"]
        out["vix_proxy"] = vx / vx.rolling(60, min_periods=30).mean()
        out["vix_pct"] = rolling_percentile(out["vix_proxy"], 252, 60)
        out["vix_ratio"] = np.nan
        out["vix"] = np.nan
        out["vix_source"] = "VIXY_relative"
    else:
        out["vix"] = np.nan
        out["vix_pct"] = out["rv21_pct"]
        out["vix_ratio"] = np.nan
    out["ma50"] = spy.rolling(50, min_periods=50).mean()
    out["ma200"] = spy.rolling(200, min_periods=150).mean()
    out["above_ma50"] = (spy > out["ma50"]).astype(float)
    out["above_ma200"] = (spy > out["ma200"]).astype(float)
    out["trend"] = np.where(out["above_ma200"] == 1, "up",
                            np.where(out["ma200"].notna(), "down", "unknown"))
    out["drawdown"] = spy / spy.cummax() - 1.0
    if macro is not None and "hy_oas" in macro.columns and macro["hy_oas"].notna().any():
        out["hy_oas"] = macro["hy_oas"]
        out["credit_pct"] = rolling_percentile(out["hy_oas"], 252, 60)
        out["credit_source"] = "fred_hy_oas_lag1"
    elif all(t in close.columns for t in ("HYG", "IEF")):
        rel = close["HYG"] / close["IEF"]
        out["hyg_ief"] = rel
        out["credit_pct"] = 1.0 - rolling_percentile(rel, 252, 60)
        out["credit_source"] = "HYG_IEF_ratio"
    else:
        out["credit_pct"] = np.nan
        out["credit_source"] = "missing"
    if macro is not None and "dgs10" in macro.columns:
        out["dgs10"] = macro["dgs10"]
    if macro is not None and "dgs3mo" in macro.columns:
        out["dgs3mo"] = macro["dgs3mo"]
    try:
        from regime_hmm import fixed_meanrev_components
        fixed = fixed_meanrev_components(panel)
        out["fixed_on"] = fixed["on"].astype(float).reindex(out.index)
        if "volPercentile" in fixed.columns:
            out["fixed_vol_pct"] = fixed["volPercentile"].reindex(out.index)
    except Exception:
        out["fixed_on"] = np.nan
    if black_swan is not None:
        out["black_swan"] = pd.Series(black_swan, index=out.index).astype(float)
    else:
        out["black_swan"] = 0.0
    oe = overextension_components(panel, out, vix=vix, macro=macro)
    for col in oe.columns:
        out[col] = oe[col]
    out["vol_bucket"] = vol_bucket(out)
    return out


def overextension_components(panel, signals, vix=None, macro=None):
    """0-1 components and a composite overextension score. All causal."""
    close = panel.close
    spy = close[MARKET]
    out = pd.DataFrame(index=signals.index)
    daily = panel.total_return()[MARKET].rolling(21, min_periods=15).std()
    parts_z = []
    for span, col in ((50, "ma50"), (200, "ma200")):
        ma = signals[col] if col in signals.columns else spy.rolling(span, min_periods=min(span, 150)).mean()
        scale = (daily * math.sqrt(span)).replace(0.0, np.nan)
        z = (spy / ma - 1.0) / scale
        out[f"stretch_z_{span}"] = z
        parts_z.append(z)
    stretch_z = pd.concat(parts_z, axis=1).mean(axis=1) if parts_z else pd.Series(np.nan, index=out.index)
    out["stretch_z"] = stretch_z
    out["oe_stretch"] = logistic((stretch_z - 1.0) / 0.75)
    out["rsi14"] = rsi_wilder(spy, 14)
    out["oe_rsi"] = ((out["rsi14"] - 55.0) / 25.0).clip(0.0, 1.0)
    high = panel.high[MARKET] if MARKET in panel.high.columns else spy
    low = panel.low[MARKET] if MARKET in panel.low.columns else spy
    upper = high.rolling(20, min_periods=15).max()
    lower = low.rolling(20, min_periods=15).min()
    width = (upper - lower).replace(0.0, np.nan)
    mid = (upper + lower) / 2.0
    pos = (spy - mid) / width
    out["channel_pos"] = pos
    out["oe_channel"] = ((pos + 0.05) / 0.55).clip(0.0, 1.0)
    names = [t for t in EQUITY_BREADTH if t in close.columns]
    if len(names) >= 5:
        ma50 = close[names].rolling(50, min_periods=50).mean()
        breadth = (close[names] > ma50).mean(axis=1)
        out["breadth"] = breadth
        out["oe_breadth"] = ((breadth - 0.55) / 0.35).clip(0.0, 1.0)
        breadth_chg = breadth - breadth.shift(20)
        spy_chg = spy / spy.shift(20) - 1.0
        div = np.where((spy_chg > 0) & (breadth_chg < 0), (-breadth_chg / 0.25).clip(0.0, 1.0), 0.0)
        out["oe_breadth_div"] = div
        out["breadth_chg20"] = breadth_chg
    else:
        out["breadth"] = np.nan
        out["oe_breadth"] = np.nan
        out["oe_breadth_div"] = np.nan
    vix_pct = signals.get("vix_pct")
    vix_ratio = signals.get("vix_ratio")
    if vix_pct is not None:
        low = (1.0 - vix_pct).clip(0.0, 1.0)
        if vix_ratio is not None and pd.Series(vix_ratio).notna().any():
            steep = ((1.00 - vix_ratio) / 0.20).clip(0.0, 1.0)
            both = ((vix_pct <= 0.30) & (vix_ratio <= 0.95)).astype(float)
            out["oe_complacency"] = (0.6 * low + 0.4 * steep) * (0.4 + 0.6 * both)
        else:
            out["oe_complacency"] = low * 0.5
    else:
        out["oe_complacency"] = np.nan
    # Valuation: trailing SPY dividend yield vs lagged 10Y. Earnings yield is not in the local data.
    if MARKET in panel.dividends.columns:
        trail = panel.dividends[MARKET].rolling(252, min_periods=60).sum()
        div_yield = trail / spy.replace(0.0, np.nan)
        out["div_yield"] = div_yield
        if "dgs10" in signals.columns and signals["dgs10"].notna().any():
            gap = signals["dgs10"] / 100.0 - div_yield
            out["yield_gap"] = gap
            out["oe_value"] = rolling_percentile(gap, 252, 60)
            out["value_source"] = "spy_div_yield_vs_dgs10"
        else:
            out["oe_value"] = np.nan
            out["value_source"] = "no_10y"
    else:
        out["oe_value"] = np.nan
        out["value_source"] = "no_dividends"
    out["earnings_yield_available"] = False
    if "credit_pct" in signals.columns and signals["credit_pct"].notna().any():
        tight = (1.0 - signals["credit_pct"]).clip(0.0, 1.0)
        out["oe_credit"] = np.where(signals["credit_pct"] <= 0.35, tight, tight * 0.25)
    else:
        out["oe_credit"] = np.nan
    component_cols = ["oe_stretch", "oe_rsi", "oe_channel", "oe_breadth", "oe_breadth_div",
                      "oe_complacency", "oe_value", "oe_credit"]
    available = [c for c in component_cols if c in out.columns and out[c].notna().any()]
    out["overextension"] = out[available].mean(axis=1) if available else np.nan
    out["oe_components_used"] = len(available)
    return out


def vol_bucket(signals):
    """low / normal / elevated / high / crisis from realized-vol percentile, VIX and drawdown."""
    pct = signals["rv21_pct"]
    vix = signals["vix"] if "vix" in signals.columns else pd.Series(np.nan, index=signals.index)
    dd = signals["drawdown"] if "drawdown" in signals.columns else pd.Series(0.0, index=signals.index)
    swan = signals["black_swan"] if "black_swan" in signals.columns else pd.Series(0.0, index=signals.index)
    labels = []
    for i in signals.index:
        p = pct.at[i] if i in pct.index else np.nan
        v = vix.at[i] if i in vix.index else np.nan
        d = dd.at[i] if i in dd.index else 0.0
        crisis = ((pd.notna(v) and v >= 35.0) or (pd.notna(d) and d <= -0.15)
                  or (swan.at[i] if i in swan.index else 0) >= 1)
        if crisis or (pd.notna(p) and p >= 0.97 and pd.notna(d) and d <= -0.10):
            labels.append("crisis")
        elif (pd.notna(v) and v >= 28.0) or (pd.notna(p) and p >= 0.85):
            labels.append("high")
        elif (pd.notna(v) and v >= 20.0) or (pd.notna(p) and p >= 0.60):
            labels.append("elevated")
        elif pd.notna(p) and p >= 0.25:
            labels.append("normal")
        elif pd.notna(p) or pd.notna(v):
            labels.append("low")
        else:
            labels.append("unknown")
    return pd.Series(labels, index=signals.index, dtype=object)


def confirm_series(values, hold=BUCKET_CONFIRM):
    """Hysteresis: a new label must persist `hold` sessions before the published state switches."""
    out, current, pending, count = [], None, None, 0
    for v in values:
        if current is None:
            current, pending, count = v, v, 1
        elif v == current:
            pending, count = v, 0
        elif v == pending:
            count += 1
            if count >= hold:
                current, count = v, 0
        else:
            pending, count = v, 1
        out.append(current)
    return pd.Series(out, index=getattr(values, "index", None), dtype=object)


def washout_flag(signals):
    rsi = signals["rsi14"] if "rsi14" in signals.columns else pd.Series(np.nan, index=signals.index)
    dd = signals["drawdown"]
    direction = signals["vol_direction"]
    pct = signals["rv21_pct"]
    washed = (rsi <= 30) | ((dd <= -0.08) & direction.eq("falling") & (pct >= 0.70))
    return washed.fillna(False)

def _clip_weights(risk, defensive, hedge):
    risk = max(0.0, float(risk))
    defensive = max(0.0, float(defensive))
    hedge = max(0.0, float(hedge))
    s = risk + defensive + hedge
    if s <= 0:
        return (1.0, 0.0, 0.0)
    return (risk / s, defensive / s, hedge / s)


def piecewise_unit(x, lo=0.35, hi=0.80):
    """0 below lo, 1 at/above hi, linear in between."""
    if not math.isfinite(x):
        return 0.0
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def scheme_static(signals, mix=(1.0, 0.0, 0.0), **_k):
    w = _clip_weights(*mix)
    return pd.DataFrame([w] * len(signals), index=signals.index, columns=["risk_on", "defensive", "hedge"])


def scheme_bucket(signals, table_name="bucket_moderate", **_k):
    table = BUCKET_TABLES[table_name]
    bucket = confirm_series(signals["vol_bucket"])
    rows = []
    for i in signals.index:
        level = bucket.at[i]
        direction = signals.at[i, "vol_direction"]
        if direction == "flat":
            direction = "falling"
        if level not in BUCKET_LEVELS:
            level = "normal"
        rows.append(table.get((level, direction), table.get((level, "falling"), (1.0, 0.0, 0.0))))
    return pd.DataFrame(rows, index=signals.index, columns=["risk_on", "defensive", "hedge"])


def scheme_vol_target(signals, sleeve_returns, target_vol=0.10, **_k):
    """Scale the risk-on sleeve toward a trailing-vol target; leftover goes to defensive."""
    rv = sleeve_returns["risk_on"].rolling(21, min_periods=15).std() * math.sqrt(TRADING_DAYS)
    spy_rv = signals["rv21"]
    use = pd.concat([rv, spy_rv], axis=1).mean(axis=1)
    scale = (target_vol / use.replace(0.0, np.nan)).clip(0.0, 1.0).fillna(1.0)
    if "vol_bucket" in signals.columns:
        scale = scale.where(~signals["vol_bucket"].isin(["high", "crisis"]), scale.clip(upper=0.40))
        scale = scale.where(signals["vol_bucket"].ne("crisis"), scale.clip(upper=0.15))
    if "black_swan" in signals.columns:
        scale = scale.where(signals["black_swan"].fillna(0) < 1, scale.clip(upper=0.15))
    return pd.DataFrame({"risk_on": scale, "defensive": 1.0 - scale,
                         "hedge": pd.Series(0.0, index=signals.index)})


def scheme_logistic(signals, vol_coef=1.4, oe_coef=1.1, trend_coef=0.9, **_k):
    """Continuous shift from risk-on toward defensive/hedge from a small logistic score."""
    pct = signals["rv21_pct"].fillna(0.5)
    rising = signals["vol_direction"].map({"rising": 1.0, "flat": 0.3, "falling": 0.0}).fillna(0.3)
    oe = signals["overextension"].fillna(0.0) if "overextension" in signals.columns else 0.0
    if not isinstance(oe, pd.Series):
        oe = pd.Series(oe, index=signals.index)
    dd = (-signals["drawdown"].clip(upper=0.0)).fillna(0.0)
    trend_down = signals["above_ma200"].eq(0).astype(float) if "above_ma200" in signals.columns else 0.0
    swan = signals["black_swan"].fillna(0.0)
    z = (vol_coef * (2 * pct - 1) + vol_coef * 0.8 * (rising - 0.3)
         + oe_coef * (2 * oe - 0.6) + 1.2 * (dd / 0.10).clip(0, 2)
         + trend_coef * trend_down + 1.5 * swan)
    off = pd.Series(logistic(z), index=signals.index)
    hedge_share = (0.25 + 0.45 * rising + 0.30 * oe).clip(0.10, 0.70)
    hedge = (off * hedge_share).clip(0.0, 0.50)
    defensive = (off - hedge).clip(0.0, 0.70)
    risk = (1.0 - defensive - hedge).clip(0.0, 1.0)
    frame = pd.DataFrame({"risk_on": risk, "defensive": defensive, "hedge": hedge})
    return frame.div(frame.sum(axis=1), axis=0)


def scheme_trend_broken(signals, ma="ma200", hedge_frac=0.20, def_frac=0.30, **_k):
    """100% risk-on while the chosen MA trend is intact; shift only after a break."""
    col = "above_ma200" if ma in ("ma200", "200") else "above_ma50"
    on = signals[col].fillna(1.0) if col in signals.columns else pd.Series(1.0, index=signals.index)
    rows = []
    for i in signals.index:
        if on.at[i] >= 1.0:
            rows.append((1.0, 0.0, 0.0))
        else:
            rows.append(_clip_weights(1.0 - def_frac - hedge_frac, def_frac, hedge_frac))
    return pd.DataFrame(rows, index=signals.index, columns=["risk_on", "defensive", "hedge"])


def scheme_overextension(signals, cap=0.25, def_cap=0.20, lo=0.35, hi=0.80,
                         enter=0.45, exit=0.30, shape="logistic", **_k):
    """Scheme (f): hedge/defensive weight scales up continuously with overextension, even before vol rises.

    Monotone mapping (piecewise-linear or logistic) from the overextension score to a hedge weight,
    capped at `cap`. Hysteresis: the hedge turns on only after `enter` and stays on until the score
    falls through `exit` or a confirmed washout (RSI <= 30, or an 8%+ drawdown with vol falling from
    a high percentile). Release is then smoothed by the shared transition limiter.
    """
    oe = signals["overextension"].fillna(0.0) if "overextension" in signals.columns else pd.Series(0.0, index=signals.index)
    washed = washout_flag(signals)
    swan = signals["black_swan"].fillna(0.0) if "black_swan" in signals.columns else pd.Series(0.0, index=signals.index)
    rows, on, locked = [], False, False
    for i in signals.index:
        x = float(oe.at[i]) if pd.notna(oe.at[i]) else 0.0
        if shape == "linear":
            unit = piecewise_unit(x, lo, hi)
        else:
            mid = 0.5 * (lo + hi)
            width = max(0.08, 0.35 * (hi - lo))
            unit = float(logistic((x - mid) / width))
        # Washout releases the hedge and locks new entries until the stretch actually
        # unwinds (oe <= exit). A live black-swan flag can keep or re-open the hedge.
        if bool(washed.at[i]) and not (swan.at[i] >= 1):
            on, locked = False, True
        elif locked:
            if x <= exit:
                locked = False
        if swan.at[i] >= 1:
            on, locked = True, False
        elif not on and not locked:
            on = x >= enter
        elif on and x <= exit:
            on = False
        if on:
            hedge = max(cap * unit, 0.25 * cap)
            defensive = def_cap * unit
        else:
            hedge = 0.0
            defensive = 0.0
        if signals.at[i, "vol_bucket"] == "crisis":
            hedge = max(hedge, cap)
            defensive = max(defensive, def_cap)
        risk = max(0.0, 1.0 - hedge - defensive)
        rows.append(_clip_weights(risk, defensive, hedge))
    return pd.DataFrame(rows, index=signals.index, columns=["risk_on", "defensive", "hedge"])


SCHEME_BUILDERS = {
    "static": scheme_static,
    "bucket": scheme_bucket,
    "vol_target": scheme_vol_target,
    "logistic": scheme_logistic,
    "trend_broken": scheme_trend_broken,
    "overextension": scheme_overextension,
}


def smooth_weights(raw, max_step=MAX_STEP, lam=SMOOTH_LAMBDA):
    """EWMA toward the target, then cap the per-day step to limit whipsaw."""
    if raw.empty:
        return raw
    out = raw.astype(float).copy()
    prev = list(_clip_weights(*out.iloc[0].tolist()))
    out.iloc[0] = prev
    for i in range(1, len(out)):
        target = out.iloc[i].tolist()
        blended = [lam * t + (1.0 - lam) * p for t, p in zip(target, prev)]
        stepped = [p + max(-max_step, min(max_step, b - p)) for b, p in zip(blended, prev)]
        stepped = list(_clip_weights(*stepped))
        out.iloc[i] = stepped
        prev = stepped
    return out


def apply_weights(sleeve_returns, weights, cost_model=None, adv=None, daily_vol=None,
                  lag=1, smooth=True):
    """Apply lagged (and optionally smoothed) sleeve weights to next-day sleeve returns, minus costs.

    weights on date t are decided from data through t and therefore multiply returns of t+lag.
    """
    cost_model = cost_model or CostModel()
    w = smooth_weights(weights) if smooth else weights.astype(float).copy()
    w = w.reindex(sleeve_returns.index).ffill().fillna(0.0)
    applied = w.shift(lag) if lag else w
    applied = applied.fillna(0.0)
    if lag:
        applied.iloc[:lag] = np.array([1.0, 0.0, 0.0])
    cols = ["risk_on", "defensive", "hedge"]
    gross = (applied[cols] * sleeve_returns[cols]).sum(axis=1)
    delta = applied[cols].diff().abs().fillna(0.0)
    proxies = {"risk_on": "SPY", "defensive": "BIL", "hedge": "SH"}
    cost = pd.Series(0.0, index=applied.index)
    for sleeve, proxy in proxies.items():
        a = adv[proxy] if adv is not None and proxy in getattr(adv, "columns", []) else None
        v = daily_vol[proxy] if daily_vol is not None and proxy in getattr(daily_vol, "columns", []) else None
        for i, day in enumerate(applied.index):
            dw = float(delta.at[day, sleeve]) if sleeve in delta.columns else 0.0
            if dw <= 1e-12:
                continue
            adv_i = float(a.at[day]) if a is not None and pd.notna(a.at[day]) else cost_model.default_adv
            vol_i = float(v.at[day]) if v is not None and pd.notna(v.at[day]) else cost_model.default_daily_vol
            bps = cost_model.cost_bps(max(dw, 1e-6) * INITIAL_EQUITY, adv_i, vol_i, side="buy", price=100.0)
            cost.iat[i] += dw * bps / 1e4
    net = gross - cost
    exposure = applied["risk_on"] - applied["hedge"]
    return {
        "returns": net.rename("return"),
        "gross_returns": gross.rename("gross"),
        "costs": cost.rename("cost"),
        "weights": w,
        "applied": applied,
        "turnover": (delta.sum(axis=1) / 2.0).rename("turnover"),
        "exposure": exposure.rename("exposure"),
    }

def capture_ratio(returns, bench, side="down"):
    aligned = pd.concat([returns, bench], axis=1, join="inner").dropna()
    if aligned.empty:
        return None
    r, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    mask = b < 0 if side == "down" else b > 0
    if mask.sum() < 5 or abs(b[mask].sum()) < 1e-12:
        return None
    return float(r[mask].sum() / b[mask].sum())


def worst_month(returns):
    if returns is None or returns.empty:
        return None
    m = (1.0 + returns).resample("ME").prod() - 1.0
    if m.empty:
        return None
    day = m.idxmin()
    return {"month": str(day)[:7], "return": float(m.min())}


def time_to_recover(returns):
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    if dd.min() >= 0 or not math.isfinite(float(dd.min())):
        return {"sessions": 0, "recovered": True}
    trough = dd.idxmin()
    prior_peak = peak.loc[:trough].iloc[-1]
    after = equity.loc[trough:]
    recovered = after[after >= prior_peak]
    if recovered.empty:
        return {"sessions": int(len(after) - 1), "recovered": False, "trough": str(trough.date())}
    return {"sessions": int(len(equity.loc[trough:recovered.index[0]]) - 1),
            "recovered": True, "trough": str(trough.date()),
            "recoveredOn": str(recovered.index[0].date())}


def spike_windows(signals, extra=KNOWN_SPIKE_WINDOWS):
    windows = list(extra)
    pct = signals["rv21_pct"]
    vix_pct = signals["vix_pct"] if "vix_pct" in signals.columns else pct
    hot = ((pct >= 0.90) | (vix_pct >= 0.90)).fillna(False)
    start = None
    for i, day in enumerate(signals.index):
        if hot.iat[i] and start is None:
            start = day
        elif not hot.iat[i] and start is not None:
            end = signals.index[i - 1]
            if (end - start).days >= 2:
                windows.append((str(start.date()), str(end.date()), "auto vol spike"))
            start = None
    if start is not None:
        windows.append((str(start.date()), str(signals.index[-1].date()), "auto vol spike"))
    cleaned, seen = [], set()
    for a, b, label in windows:
        if (a, b) in seen:
            continue
        seen.add((a, b))
        cleaned.append((a, b, label))
    return cleaned


def window_performance(returns, bench, start, end):
    sl = returns.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
    if len(sl) < 2:
        return None
    b = bench.reindex(sl.index)
    eq = (1.0 + sl).cumprod()
    return {
        "start": str(sl.index[0].date()), "end": str(sl.index[-1].date()), "sessions": int(len(sl)),
        "totalReturn": float((1.0 + sl).prod() - 1.0),
        "maxDrawdown": float((eq / eq.cummax() - 1.0).min()),
        "benchReturn": float((1.0 + b.fillna(0)).prod() - 1.0) if b.notna().any() else None,
    }


def meltup_vs_drawdown(scheme_returns, baseline, signals, bench=None):
    """Cost of hedging early (drag in melt-ups) versus drawdown saved versus the TSMOM baseline."""
    aligned = pd.concat([scheme_returns.rename("s"), baseline.rename("b")], axis=1, join="inner").dropna()
    if aligned.empty:
        return {}
    s, b = aligned["s"], aligned["b"]
    spy20 = signals["spy_ret"].rolling(20, min_periods=10).sum() if "spy_ret" in signals.columns else None
    if spy20 is None and bench is not None:
        spy20 = bench.rolling(20, min_periods=10).sum()
    vol_ok = ~signals["vol_bucket"].isin(["high", "crisis"])
    oe = signals["overextension"] if "overextension" in signals.columns else pd.Series(0.5, index=signals.index)
    melt = pd.Series(False, index=signals.index)
    if spy20 is not None:
        melt = (spy20 >= 0.05) & vol_ok.reindex(signals.index).fillna(True) & (oe.fillna(0) >= 0.35)
    melt = melt.reindex(aligned.index).fillna(False)
    drag = float((s - b)[melt].sum()) if melt.any() else None
    melt_s = float((1.0 + s[melt]).prod() - 1.0) if melt.any() else None
    melt_b = float((1.0 + b[melt]).prod() - 1.0) if melt.any() else None
    eq_s = (1.0 + s).cumprod()
    eq_b = (1.0 + b).cumprod()
    dd_s = float((eq_s / eq_s.cummax() - 1.0).min())
    dd_b = float((eq_b / eq_b.cummax() - 1.0).min())
    return {
        "meltUpSessions": int(melt.sum()),
        "meltUpDragSum": drag,
        "meltUpSchemeReturn": melt_s,
        "meltUpBaselineReturn": melt_b,
        "meltUpRelative": (None if melt_s is None or melt_b is None else melt_s - melt_b),
        "schemeMaxDrawdown": dd_s,
        "baselineMaxDrawdown": dd_b,
        "drawdownSaved": dd_s - dd_b,
        "drawdownSavedAbs": float(abs(dd_b) - abs(dd_s)),
        "note": ("Melt-ups are 20-session SPY runs of +5% or more while the vol bucket is not high/crisis "
                 "and overextension is at least 0.35. Drag is scheme minus baseline return over those days. "
                 "drawdownSavedAbs is how much shallower the scheme max drawdown was than plain TSMOM."),
    }


def summarize_returns(returns, bench, applied=None, turnover=None, costs=None, signals=None, label=None):
    perf = performance(returns, turnover=turnover, costs=costs,
                       gross=applied["risk_on"] if applied is not None and "risk_on" in applied.columns else None)
    extra = {
        "label": label,
        "sortino": perf.get("sortino"),
        "calmar": perf.get("calmar"),
        "downsideCapture": capture_ratio(returns, bench, "down"),
        "upsideCapture": capture_ratio(returns, bench, "up"),
        "worstMonth": worst_month(returns),
        "timeToRecover": time_to_recover(returns),
        "avgMarketExposure": (float(applied["risk_on"].mean() - applied["hedge"].mean())
                              if applied is not None else None),
        "avgRiskOn": float(applied["risk_on"].mean()) if applied is not None else None,
        "avgDefensive": float(applied["defensive"].mean()) if applied is not None else None,
        "avgHedge": float(applied["hedge"].mean()) if applied is not None else None,
    }
    if signals is not None and len(returns) >= 2:
        extra["spikeWindows"] = {}
        for a, b, lab in spike_windows(signals):
            wp = window_performance(returns, bench, a, b)
            if wp:
                extra["spikeWindows"][f"{lab} {a}..{b}"] = wp
    perf.update({k: v for k, v in extra.items() if v is not None or k in ("worstMonth", "timeToRecover")})
    return perf


def candidate_list():
    """Small, pre-declared set of schemes and parameterizations (reported in full for DSR / PBO)."""
    out = []
    for name, mix in STATIC_MIXES.items():
        out.append({"id": name, "scheme": "static", "params": {"mix": mix}})
    for name in BUCKET_TABLES:
        out.append({"id": name, "scheme": "bucket", "params": {"table_name": name}})
    for tv in (0.08, 0.10, 0.12):
        out.append({"id": f"vol_target_{int(tv * 100)}", "scheme": "vol_target",
                    "params": {"target_vol": tv}})
    for vol_c, oe_c in ((1.0, 0.8), (1.4, 1.1), (1.8, 1.4)):
        out.append({"id": f"logistic_v{vol_c}_o{oe_c}", "scheme": "logistic",
                    "params": {"vol_coef": vol_c, "oe_coef": oe_c, "trend_coef": 0.9}})
    for ma, h, d in (("ma200", 0.20, 0.30), ("ma200", 0.30, 0.30), ("ma50", 0.20, 0.20)):
        out.append({"id": f"trend_broken_{ma}_{int(h * 100)}_{int(d * 100)}",
                    "scheme": "trend_broken", "params": {"ma": ma, "hedge_frac": h, "def_frac": d}})
    for cap, shape in ((0.15, "logistic"), (0.25, "logistic"), (0.35, "logistic"),
                       (0.25, "linear"), (0.35, "linear")):
        out.append({"id": f"overextension_c{int(cap * 100)}_{shape}",
                    "scheme": "overextension",
                    "params": {"cap": cap, "def_cap": min(0.25, cap), "lo": 0.35, "hi": 0.80,
                               "enter": 0.45, "exit": 0.30, "shape": shape}})
    return out


def latin_or_grid_candidates():
    """Return the pre-declared grid (already small). A LHS hook can expand this later."""
    return candidate_list()


def weights_for(candidate, signals, sleeve_returns):
    builder = SCHEME_BUILDERS[candidate["scheme"]]
    raw = builder(signals, sleeve_returns=sleeve_returns, **candidate.get("params", {}))
    return smooth_weights(raw)


def simulate_candidate(candidate, signals, sleeve_returns, cost_model, adv, daily_vol, start=None, end=None):
    weights = weights_for(candidate, signals, sleeve_returns)
    path = apply_weights(sleeve_returns, weights, cost_model=cost_model, adv=adv, daily_vol=daily_vol)
    returns = path["returns"]
    applied, turnover, costs = path["applied"], path["turnover"], path["costs"]
    if start is not None:
        returns = returns.loc[pd.Timestamp(start):]
    if end is not None:
        returns = returns.loc[:pd.Timestamp(end)]
    applied = applied.reindex(returns.index)
    turnover = turnover.reindex(returns.index)
    costs = costs.reindex(returns.index)
    return path, returns, applied, turnover, costs


def holdout_bounds(dates, eval_start, holdout_start=None, data_end=None, fraction=HOLDOUT_FRACTION):
    dates = pd.DatetimeIndex(dates)
    dates = dates[dates >= pd.Timestamp(eval_start)]
    if data_end is not None:
        dates = dates[dates <= pd.Timestamp(data_end)]
    if holdout_start is not None:
        hs = pd.Timestamp(holdout_start)
    else:
        cut = max(1, int(round(len(dates) * (1.0 - fraction))))
        hs = dates[cut] if cut < len(dates) else dates[-1]
    pre = dates[dates < hs]
    hold = dates[dates >= hs]
    return pre, hold, hs


def bootstrap_ci(returns, fn, n=1000, seed=7, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(returns.dropna(), dtype=float)
    if len(x) < 30:
        return {"lo": None, "hi": None, "n": int(len(x))}
    stats = []
    for _ in range(n):
        samp = rng.choice(x, size=len(x), replace=True)
        try:
            stats.append(fn(samp))
        except Exception:
            continue
    if not stats:
        return {"lo": None, "hi": None, "n": int(len(x))}
    lo, hi = np.quantile(stats, [alpha / 2, 1.0 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "n": int(len(x)), "samples": int(len(stats))}


def _sharpe_arr(x):
    x = np.asarray(x, dtype=float)
    if x.std() <= 0:
        return 0.0
    return float(x.mean() / x.std() * math.sqrt(TRADING_DAYS))


def _cagr_arr(x):
    x = np.asarray(x, dtype=float)
    growth = float(np.prod(1.0 + x))
    years = len(x) / TRADING_DAYS
    if growth <= 0 or years <= 0:
        return -1.0
    return growth ** (1.0 / years) - 1.0

def _adv_and_vol(panel):
    adv = (panel.close * panel.volume).rolling(20, min_periods=5).median()
    daily_vol = panel.total_return().rolling(21, min_periods=10).std()
    return adv, daily_vol


def _align_holdout_with_swarm():
    path = DATA_DIR / "Research" / "swarm" / "design.json"
    if not path.exists():
        return None
    try:
        design = json.loads(path.read_text())
        return {k: design.get(k) for k in ("evalStart", "holdoutStart", "dataEnd", "designHash", "simStart")}
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def evaluate_candidates(panel, signals, sleeves, cost_model=None, eval_start=None, holdout_start=None, data_end=None):
    cost_model = cost_model or CostModel()
    adv, daily_vol = _adv_and_vol(panel)
    dates = panel.dates
    eval_start = pd.Timestamp(eval_start or dates[min(WARMUP, len(dates) - 2)])
    pre, hold, hs = holdout_bounds(dates, eval_start, holdout_start, data_end)
    candidates = latin_or_grid_candidates()
    spy = sleeves["spy"].reindex(dates)
    rows, series = [], {}
    for cand in candidates:
        _path, returns, applied, turnover, costs = simulate_candidate(
            cand, signals, sleeves, cost_model, adv, daily_vol)
        pre_r = returns.reindex(pre).dropna()
        hold_r = returns.reindex(hold).dropna()
        pre_stats = summarize_returns(pre_r, spy.reindex(pre_r.index), applied.reindex(pre_r.index),
                                      turnover.reindex(pre_r.index), costs.reindex(pre_r.index),
                                      signals.reindex(pre_r.index), label=cand["id"])
        hold_stats = (summarize_returns(hold_r, spy.reindex(hold_r.index), applied.reindex(hold_r.index),
                                        turnover.reindex(hold_r.index), costs.reindex(hold_r.index),
                                        signals.reindex(hold_r.index), label=cand["id"])
                      if len(hold_r) >= 5 else {})
        rows.append({
            "id": cand["id"], "scheme": cand["scheme"], "params": cand["params"],
            "pre": pre_stats, "holdout": hold_stats,
            "preSharpe": pre_stats.get("sharpe"), "preCagr": pre_stats.get("cagr"),
            "preMaxDrawdown": pre_stats.get("maxDrawdown"), "preCalmar": pre_stats.get("calmar"),
            "preDownsideCapture": pre_stats.get("downsideCapture"),
            "holdoutSharpe": hold_stats.get("sharpe"), "holdoutCagr": hold_stats.get("cagr"),
            "holdoutMaxDrawdown": hold_stats.get("maxDrawdown"),
        })
        series[cand["id"]] = returns
    pre_frame = pd.DataFrame({r["id"]: series[r["id"]].reindex(pre) for r in rows})

    def _key(r):
        sh = r["preSharpe"]
        dd = r["preMaxDrawdown"]
        cal = r["preCalmar"]
        return (sh is not None, sh if sh is not None else -999, -(dd if dd is not None else 0),
                cal if cal is not None else -999)

    ranked = sorted(rows, key=_key, reverse=True)
    champion = ranked[0]["id"] if ranked else None
    return {
        "candidates": rows, "rankedIds": [r["id"] for r in ranked], "champion": champion,
        "preDates": (str(pre[0].date()), str(pre[-1].date()), int(len(pre))) if len(pre) else None,
        "holdoutDates": (str(hold[0].date()), str(hold[-1].date()), int(len(hold))) if len(hold) else None,
        "holdoutStart": str(hs.date()),
        "evalStart": str(eval_start.date()),
        "series": series, "preFrame": pre_frame, "spy": spy, "baselineId": "static_100_0_0",
    }


def overfitting_stats(pre_frame, champion, baseline_id="static_100_0_0"):
    out = {}
    try:
        from swarm import deflated_sharpe, effective_trials, pbo_cscv, spa_test
    except Exception as exc:
        return {"error": f"swarm stats unavailable: {type(exc).__name__}: {exc}"}
    M = pre_frame.dropna(how="all")
    if M.shape[1] < 2 or M.shape[0] < 40:
        return {"error": "not enough pre-holdout observations for DSR/PBO/SPA"}
    sharpes = M.apply(lambda s: _sharpe_arr(s.dropna().to_numpy()), axis=0)
    trials = effective_trials(M) if M.shape[1] >= 3 else float(M.shape[1])
    if champion in M.columns:
        out["deflatedSharpePre"] = deflated_sharpe(M[champion].dropna(), trials, sharpes.dropna().to_numpy())
    try:
        out["pboPre"] = pbo_cscv(np.asarray(M.dropna(how="any"), dtype=float))
    except Exception as exc:
        out["pboPre"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["effectiveTrials"] = _finite(trials)
    out["trialCount"] = int(M.shape[1])
    if baseline_id in M.columns:
        excess = M.subtract(M[baseline_id], axis=0)
        if baseline_id in excess.columns:
            excess = excess.drop(columns=[baseline_id])
        excess = excess.dropna(how="any")
        if excess.shape[0] >= 40 and excess.shape[1] >= 1:
            stats, p_ind, tstat = spa_test(excess.to_numpy())
            out["spaVsTsmomPre"] = dict(stats)
            if champion in excess.columns:
                idx = list(excess.columns).index(champion)
                out["spaVsTsmomPre"]["championP"] = float(p_ind[idx])
                out["spaVsTsmomPre"]["championT"] = float(tstat[idx])
    return out


def _state_weight_table(signals, weights, min_count=3):
    """Average applied weights by confirmed vol state."""
    bucket = confirm_series(signals["vol_bucket"])
    direction = signals["vol_direction"].replace({"flat": "falling"})
    oe = signals["overextension"] if "overextension" in signals.columns else pd.Series(np.nan, index=signals.index)
    frame = weights.join(pd.DataFrame({"bucket": bucket, "direction": direction}), how="inner")
    rows = []
    for (b, d), g in frame.groupby(["bucket", "direction"], observed=True):
        if len(g) < min_count:
            continue
        rows.append({
            "volBucket": str(b), "volDirection": str(d), "sessions": int(len(g)),
            "riskOn": float(g["risk_on"].mean()), "defensive": float(g["defensive"].mean()),
            "hedge": float(g["hedge"].mean()),
            "overextensionMean": float(oe.reindex(g.index).mean()) if oe.reindex(g.index).notna().any() else None,
        })
    rows.sort(key=lambda r: (BUCKET_LEVELS.index(r["volBucket"]) if r["volBucket"] in BUCKET_LEVELS else 99,
                             0 if r["volDirection"] == "falling" else 1))
    return rows


def current_state_payload(signals, weights=None, recommendation=None):
    if signals is None or signals.empty:
        return {"available": False}
    row = signals.iloc[-1]
    day = signals.index[-1]
    w = None
    if weights is not None and len(weights):
        loc = weights.loc[:day]
        if len(loc):
            last = loc.iloc[-1]
            w = {"riskOn": float(last["risk_on"]), "defensive": float(last["defensive"]),
                 "hedge": float(last["hedge"])}

    def g(*names):
        for n in names:
            if n in signals.columns and pd.notna(row[n]):
                val = row[n]
                if isinstance(val, (np.floating, float)) and math.isfinite(float(val)):
                    return float(val)
                if isinstance(val, str):
                    return val
                if isinstance(val, (np.integer, int)):
                    return int(val)
        return None

    components = {k: g(k) for k in ("oe_stretch", "oe_rsi", "oe_channel", "oe_breadth", "oe_breadth_div",
                                    "oe_complacency", "oe_value", "oe_credit", "stretch_z", "rsi14",
                                    "channel_pos", "breadth")}
    return {
        "available": True,
        "asOf": str(day.date()),
        "volBucket": g("vol_bucket"),
        "volDirection": g("vol_direction"),
        "realizedVol21": g("rv21"),
        "realizedVolPercentile": g("rv21_pct"),
        "volOfVol": g("vol_of_vol"),
        "vix": g("vix"),
        "vix3m": g("vix3m"),
        "vixRatio": g("vix_ratio"),
        "vixPercentile": g("vix_pct"),
        "trend": g("trend"),
        "aboveMa50": g("above_ma50"),
        "aboveMa200": g("above_ma200"),
        "drawdown": g("drawdown"),
        "creditPercentile": g("credit_pct"),
        "creditSource": row["credit_source"] if "credit_source" in signals.columns else None,
        "blackSwan": bool(g("black_swan") or 0),
        "overextension": g("overextension"),
        "overextensionComponents": components,
        "washout": bool(washout_flag(signals).iloc[-1]) if len(signals) else False,
        "valueSource": (row["value_source"] if "value_source" in signals.columns else None),
        "earningsYieldAvailable": False,
        "weights": w,
        "recommendation": recommendation,
    }


def recommend(study, baseline_id="static_100_0_0"):
    """Stay static (plain TSMOM) unless a challenger beats it robustly after multiple-testing correction."""
    champ_id = study["champion"]
    by_id = {r["id"]: r for r in study["candidates"]}
    base = by_id.get(baseline_id, {})
    champ = by_id.get(champ_id, {})
    of = study.get("overfitting") or {}
    dsr = None
    if isinstance(of.get("deflatedSharpePre"), dict):
        dsr = (of["deflatedSharpePre"].get("deflatedSharpeProbability")
               or of["deflatedSharpePre"].get("probability"))
        if dsr is None:
            for k, v in of["deflatedSharpePre"].items():
                if "prob" in k.lower() or k in ("dsr", "psr"):
                    dsr = v
                    break
    pbo = of["pboPre"].get("pbo") if isinstance(of.get("pboPre"), dict) else None
    spa = of.get("spaVsTsmomPre")
    spa_p = None
    if isinstance(spa, dict):
        spa_p = spa.get("p") or spa.get("spaP") or spa.get("pvalue")
    hold_s, hold_b = champ.get("holdoutSharpe"), base.get("holdoutSharpe")
    pre_s, pre_b = champ.get("preSharpe"), base.get("preSharpe")
    dd_s, dd_b = champ.get("preMaxDrawdown"), base.get("preMaxDrawdown")
    better_dd = (dd_s is not None and dd_b is not None and abs(dd_s) < abs(dd_b) - 1e-6)
    better_sharpe = (pre_s is not None and pre_b is not None and pre_s > pre_b + 0.05)
    gates = {
        "betterPreSharpe": better_sharpe,
        "betterPreDrawdown": better_dd,
        "deflatedSharpeProb": _finite(dsr),
        "pbo": _finite(pbo),
        "spaPVsTsmom": _finite(spa_p),
        "holdoutSharpeVsTsmom": (None if hold_s is None or hold_b is None else hold_s - hold_b),
    }
    robust = bool(better_sharpe and better_dd and (dsr is None or dsr >= 0.95)
                  and (pbo is None or pbo < 0.5) and (spa_p is None or spa_p < 0.05))
    if champ_id == baseline_id or not better_sharpe:
        decision = "stay_static"
        text = ("Stay with 100% plain TSMOM (static 100/0/0). No alternative sleeve mix improved pre-holdout "
                "Sharpe enough to justify leaving the baseline, after costs and with the short sample in mind.")
    elif not robust:
        decision = "stay_static_with_interest"
        text = (f"Keep the live paper account on plain TSMOM. The pre-holdout leader is {champ_id} "
                f"(Sharpe {None if pre_s is None else round(pre_s, 2)}"
                f", max DD {None if dd_s is None else round(100 * dd_s, 1)}% vs TSMOM "
                f"{None if dd_b is None else round(100 * dd_b, 1)}%), but it does not clear the robustness "
                "gates (deflated Sharpe, PBO, SPA vs TSMOM and a drawdown test) on this short sample. "
                "Treat the state-weight table as research, not a live change.")
    else:
        decision = "recommend_challenger"
        text = (f"Recommendation only (the live paper account is unchanged): {champ_id} beat plain TSMOM on "
                "pre-holdout Sharpe and drawdown and cleared the multiple-testing gates. Consider a shadow "
                "paper sleeve with those state weights before any live-config change.")
    return {"decision": decision, "champion": champ_id, "baseline": baseline_id, "robust": robust,
            "gates": gates, "text": text, "livePaperUnchanged": True}


def _log_holdout(design_hash, holdout_start, data_end, champion, extra=None):
    payload_design = {"holdoutStart": holdout_start, "dataEnd": data_end, "designHash": design_hash}
    stats = {"champion": champion}
    if extra:
        stats.update(extra)
    try:
        from swarm import _record_holdout_view
        return _record_holdout_view(payload_design, stats, study="regime_allocation", design_hash=design_hash)
    except Exception as exc:
        path = DATA_DIR / "Research" / "swarm" / "holdout_ledger.json"
        try:
            ledger = json.loads(path.read_text()) if path.exists() else {"views": 0, "entries": []}
            entry = {"at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "designHash": design_hash,
                     "study": "regime_allocation", "holdoutStart": holdout_start, "dataEnd": data_end,
                     "champion": champion, "note": f"fallback logger ({type(exc).__name__})"}
            ledger["entries"].append(entry)
            ledger["views"] = len({e["designHash"] for e in ledger["entries"]
                                   if e.get("holdoutStart") == holdout_start})
            write_json(path, ledger)
            return ledger
        except Exception as exc2:
            return {"error": f"{type(exc).__name__}: {exc}; fallback {type(exc2).__name__}: {exc2}"}

def _fmt(x, d=2):
    x = _finite(x)
    return "–" if x is None else f"{x:.{d}f}"


def _pct(x, d=1):
    x = _finite(x)
    return "–" if x is None else f"{100 * x:.{d}f}%"


def _markdown(summary):
    rec = summary.get("recommendation") or {}
    data = summary.get("data") or {}
    lines = [
        "# Regime allocation study",
        "",
        f"Generated {summary.get('generatedAt')}. Live paper account is **unchanged** (recommendation only).",
        "",
        "## Data",
        f"- Sessions: {data.get('start')} .. {data.get('end')} ({data.get('sessions')} sessions).",
        f"- Long history: {data.get('longHistory') or 'not present'}.",
        f"- Missing instruments: {', '.join(data.get('missing') or []) or 'none of the requested sleeve names'}.",
    ]
    for n in (data.get("notes") or []):
        lines.append(f"- {n}")
    lines += ["", "## Recommendation", rec.get("text") or "",
              f"- Decision: `{rec.get('decision')}` · champion `{rec.get('champion')}` · robust `{rec.get('robust')}`.",
              "", "## Pre-holdout leaders"]
    lines.append("| id | scheme | Sharpe | CAGR | Max DD | Calmar | Downside cap. | Holdout Sharpe | Holdout max DD |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in (summary.get("results") or [])[:12]:
        lines.append(
            f"| {row['id']} | {row['scheme']} | {_fmt(row.get('preSharpe'))} | {_pct(row.get('preCagr'))} | "
            f"{_pct(row.get('preMaxDrawdown'))} | {_fmt(row.get('preCalmar'))} | {_fmt(row.get('preDownsideCapture'))} | "
            f"{_fmt(row.get('holdoutSharpe'))} | {_pct(row.get('holdoutMaxDrawdown'))} |"
        )
    lines += ["", "## State-to-weight table (champion, average weights in each confirmed vol state)",
              "| Vol bucket | Direction | Sessions | Risk-on | Defensive | Hedge |",
              "|---|---|---:|---:|---:|---:|"]
    for r in summary.get("stateWeightTable") or []:
        lines.append(
            f"| {r['volBucket']} | {r['volDirection']} | {r['sessions']} | {_pct(r['riskOn'], 1)} | "
            f"{_pct(r['defensive'], 1)} | {_pct(r['hedge'], 1)} |"
        )
    oe = summary.get("overextensionCost") or {}
    if oe:
        lines += ["", "## Overextension hedge: melt-up drag vs drawdown saved",
                  f"- Melt-up sessions: {oe.get('meltUpSessions')}. Relative return vs TSMOM: {_pct(oe.get('meltUpRelative'))}.",
                  f"- Drawdown saved (absolute): {_pct(oe.get('drawdownSavedAbs'))} "
                  f"(scheme max DD {_pct(oe.get('schemeMaxDrawdown'))} vs TSMOM {_pct(oe.get('baselineMaxDrawdown'))}).",
                  f"- {oe.get('note', '')}"]
    of = summary.get("overfitting") or {}
    lines += ["", "## Multiple-testing / robustness",
              f"```json\n{json.dumps(of, indent=1, default=_json_default)}\n```",
              "", "## Honesty",
              "- Pre-holdout selection uses only dates before the sealed holdout. A short eval window has wide "
              "error bars; a 'win' on a two-year bull tape is not a 20-year proof. A longer spliced panel still "
              "has survivorship bias and omits dividends on pre-Massive Schwab returns.",
              "- The protective-put sleeve is MODELED (Black-Scholes on VIX, modeled bid/ask, Schwab option fees), "
              "not a backtest of historical option quotes. VIXY and inverse ETFs have structural decay.",
              "- Earnings yield versus the 10-year was not available locally; the value component uses trailing "
              "SPY dividend yield versus lagged FRED 10Y when that series is present.",
              "- Selection used pre-holdout data only. The holdout numbers are one logged look, not a tuning input."]
    return "\n".join(lines) + "\n"


def research(start=None, end=None, out_dir=None, log_holdout=True, panel=None, eval_start=None):
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    t0 = time.time()
    log("loading panel and features")
    if panel is None:
        panel, data_info = load_study_panel(start, end)
    else:
        data_info = {"start": str(panel.dates[0].date()), "end": str(panel.dates[-1].date()),
                     "sessions": int(len(panel.dates)), "notes": ["caller-supplied panel"],
                     "missing": [], "longHistory": detect_long_history(), "present": list(panel.close.columns)}
    vix = load_vix_frame(panel.dates)
    macro = load_macro_series(panel.dates)
    swan, swan_name = load_black_swan_state(panel.dates)
    if swan_name:
        data_info.setdefault("notes", []).append(f"Black-swan state from {swan_name}.")
    else:
        data_info.setdefault("notes", []).append("GB-17 black-swan module was not present; that flag is 0.")
    if "vix" not in vix.columns or not vix["vix"].notna().any():
        data_info.setdefault("notes", []).append(
            "VIX/VIX3M file missing or empty; falling back to a VIXY-relative or realized-vol proxy.")
    elif vix["vix"].notna().any():
        data_info.setdefault("notes", []).append(
            f"VIX/VIX3M from Massive/Options/vix_daily.csv "
            f"({str(vix['vix'].dropna().index[0].date())}..{str(vix['vix'].dropna().index[-1].date())}).")
    log("building sleeves (TSMOM + defensive + hedge)")
    sleeves, sleeve_info, _tsmom = build_sleeves(panel, vix=vix)
    data_info["missing"] = list(dict.fromkeys((data_info.get("missing") or []) + (sleeve_info.get("missing") or [])))
    data_info.setdefault("notes", []).extend(sleeve_info.get("notes") or [])
    data_info["sleeves"] = {k: sleeve_info[k] for k in ("defensiveWeights", "hedgeWeights", "putOverlay")
                            if k in sleeve_info}
    log("building point-in-time signals")
    signals = build_signals(panel, vix=vix, macro=macro, black_swan=swan)
    swarm_align = _align_holdout_with_swarm()
    holdout_start = (swarm_align or {}).get("holdoutStart")
    data_end = (swarm_align or {}).get("dataEnd") or str(panel.dates[-1].date())
    if eval_start:
        eval_start = str(pd.Timestamp(eval_start).date())
        data_info.setdefault("notes", []).append(
            f"Eval start overridden to {eval_start} so the longer panel can be used for pre-holdout "
            f"selection. Sealed holdout stays {holdout_start} .. {data_end}.")
    else:
        eval_start = (swarm_align or {}).get("evalStart")
        if swarm_align:
            data_info.setdefault("notes", []).append(
                f"Holdout window aligned with the swarm design ({holdout_start} .. {data_end}, eval from {eval_start}).")
    log("evaluating allocation candidates on pre-holdout data")
    study = evaluate_candidates(panel, signals, sleeves, eval_start=eval_start,
                                holdout_start=holdout_start, data_end=data_end)
    study["overfitting"] = overfitting_stats(study["preFrame"], study["champion"])
    rec = recommend(study)
    by_id = {c["id"]: c for c in latin_or_grid_candidates()}
    champ = by_id[study["champion"]]
    cost_model = CostModel()
    adv, daily_vol = _adv_and_vol(panel)
    path, ret, applied, turnover, costs = simulate_candidate(champ, signals, sleeves, cost_model, adv, daily_vol)
    base = by_id["static_100_0_0"]
    _bpath, base_ret, base_applied, _, _ = simulate_candidate(base, signals, sleeves, cost_model, adv, daily_vol)
    oe_candidate = next((c for c in latin_or_grid_candidates()
                         if c["scheme"] == "overextension" and c["params"].get("cap") == 0.25
                         and c["params"].get("shape") == "logistic"), None)
    oe_cost, oe_ret = {}, None
    if oe_candidate is not None:
        _p, oe_ret, _, _, _ = simulate_candidate(oe_candidate, signals, sleeves, cost_model, adv, daily_vol)
        eval_idx = ret.loc[pd.Timestamp(study["evalStart"]):].index
        oe_cost = meltup_vs_drawdown(oe_ret.reindex(eval_idx), base_ret.reindex(eval_idx),
                                     signals.reindex(eval_idx), bench=sleeves["spy"].reindex(eval_idx))
        if study["champion"].startswith("overextension"):
            oe_cost["usedAsChampion"] = True
    current = current_state_payload(signals, applied, recommendation=rec)
    table = _state_weight_table(signals.loc[pd.Timestamp(study["evalStart"]):],
                                applied.loc[pd.Timestamp(study["evalStart"]):])
    pre_a = study["preDates"][0] if study["preDates"] else study["evalStart"]
    pre_b = study["holdoutStart"]
    champ_pre = ret.loc[pd.Timestamp(pre_a):pd.Timestamp(pre_b) - pd.Timedelta(days=1)].dropna()
    base_pre = base_ret.reindex(champ_pre.index).dropna()
    ci = {
        "championSharpe": bootstrap_ci(champ_pre, _sharpe_arr),
        "championCagr": bootstrap_ci(champ_pre, _cagr_arr),
        "tsmomSharpe": bootstrap_ci(base_pre, _sharpe_arr),
        "tsmomCagr": bootstrap_ci(base_pre, _cagr_arr),
    }
    design_hash = "alloc-" + hashlib.sha256(
        json.dumps({"candidates": [c["id"] for c in latin_or_grid_candidates()],
                    "eval": study["evalStart"], "holdout": study["holdoutStart"],
                    "dataEnd": data_end, "code": "regime-allocation-1"},
                   sort_keys=True).encode()
    ).hexdigest()[:16]
    ledger = None
    if log_holdout and study["holdoutDates"]:
        ledger = _log_holdout(design_hash, study["holdoutStart"], data_end, study["champion"],
                              extra={"decision": rec["decision"], "robust": rec["robust"]})
    results_rows = []
    for row in study["candidates"]:
        slim = {k: row[k] for k in row if k not in ("pre", "holdout")}
        slim["pre"] = {k: row["pre"].get(k) for k in ("sharpe", "cagr", "sortino", "maxDrawdown", "calmar",
                                                      "downsideCapture", "upsideCapture", "avgMarketExposure",
                                                      "turnoverPerYear", "worstMonth", "timeToRecover",
                                                      "totalReturn", "volatility") if k in (row.get("pre") or {})}
        if row.get("pre") and row["pre"].get("spikeWindows"):
            slim["pre"]["spikeWindows"] = row["pre"]["spikeWindows"]
        slim["holdout"] = {k: (row.get("holdout") or {}).get(k)
                           for k in ("sharpe", "cagr", "maxDrawdown", "calmar", "downsideCapture", "upsideCapture")}
        results_rows.append(slim)
    order = {i: n for n, i in enumerate(study["rankedIds"])}
    results_rows.sort(key=lambda r: order.get(r["id"], 999))
    summary = {
        "generatedAt": started.strftime("%Y-%m-%d %H:%M"),
        "codeVersion": "regime-allocation-1",
        "designHash": design_hash,
        "data": data_info,
        "evalStart": study["evalStart"],
        "holdoutStart": study["holdoutStart"],
        "holdoutDates": study["holdoutDates"],
        "preDates": study["preDates"],
        "recommendation": rec,
        "results": results_rows,
        "stateWeightTable": table,
        "currentState": current,
        "overfitting": study["overfitting"],
        "bootstrapCI": ci,
        "overextensionCost": oe_cost,
        "holdoutLedger": ({"views": (ledger or {}).get("views"),
                           "logged": bool(ledger and "error" not in (ledger or {}))}
                          if log_holdout else {"logged": False}),
        "livePaperUnchanged": True,
        "runSeconds": round(time.time() - t0, 1),
    }
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "current_state.json", current)
    eq = pd.DataFrame({
        "tsmom": (1.0 + base_ret.fillna(0)).cumprod(),
        "champion": (1.0 + ret.fillna(0)).cumprod(),
        "spy": (1.0 + sleeves["spy"].reindex(ret.index).fillna(0)).cumprod(),
    })
    if oe_ret is not None:
        eq["overextension_c25_logistic"] = (1.0 + oe_ret.reindex(eq.index).fillna(0)).cumprod()
    eq.to_csv(out_dir / "equity_curves.csv")
    applied.to_csv(out_dir / "champion_weights.csv")
    trials = pd.DataFrame([{k: r[k] for k in ("id", "scheme", "preSharpe", "preCagr", "preMaxDrawdown",
                                              "preCalmar", "preDownsideCapture", "holdoutSharpe",
                                              "holdoutCagr", "holdoutMaxDrawdown")} for r in study["candidates"]])
    trials.to_csv(out_dir / "trials.csv", index=False)
    write_atomically(out_dir / "report.md", _markdown(summary).encode())
    study["preFrame"].to_csv(out_dir / "pre_holdout_returns.csv")
    log(f"wrote {out_dir} in {summary['runSeconds']}s; decision={rec['decision']} champion={rec['champion']}")
    return summary


def latest_state(panel=None):
    if panel is None:
        panel, _info = load_study_panel()
    vix = load_vix_frame(panel.dates)
    macro = load_macro_series(panel.dates)
    swan, _name = load_black_swan_state(panel.dates)
    sleeves, _info, _ = build_sleeves(panel, vix=vix)
    signals = build_signals(panel, vix=vix, macro=macro, black_swan=swan)
    summary_path = OUT_DIR / "summary.json"
    rec = None
    champ = next(c for c in latin_or_grid_candidates() if c["id"] == "static_100_0_0")
    if summary_path.exists():
        try:
            doc = json.loads(summary_path.read_text())
            rec = doc.get("recommendation")
            wanted = (rec or {}).get("champion") or "static_100_0_0"
            champ = next((c for c in latin_or_grid_candidates() if c["id"] == wanted), champ)
        except (OSError, ValueError, json.JSONDecodeError):
            rec = None
    weights = weights_for(champ, signals, sleeves)
    cost_model = CostModel()
    adv, daily_vol = _adv_and_vol(panel)
    path = apply_weights(sleeves, weights, cost_model=cost_model, adv=adv, daily_vol=daily_vol)
    payload = current_state_payload(signals, path["applied"], recommendation=rec)
    write_json(OUT_DIR / "current_state.json", payload)
    return payload


SHADOW_VARIANTS = (
    "static_100_0_0",
    "static_70_20_10",
    "bucket_moderate",
    "vol_target_10",
    "trend_broken_ma200_20_30",
    "overextension_c25_logistic",
    "logistic_v1.4_o1.1",
)


def attach_forward_shadows(panel, design, new_dates=None):
    """Shadow $10k paths for a few allocation variants after the swarm design end. No orders, no design change."""
    out_dir = DATA_DIR / "Research" / "swarm" / "forward"
    out_dir.mkdir(parents=True, exist_ok=True)
    vix = load_vix_frame(panel.dates)
    macro = load_macro_series(panel.dates)
    swan, _ = load_black_swan_state(panel.dates)
    sleeves, sleeve_info, _ = build_sleeves(panel, vix=vix)
    signals = build_signals(panel, vix=vix, macro=macro, black_swan=swan)
    cost_model = CostModel()
    adv, daily_vol = _adv_and_vol(panel)
    design_end = pd.Timestamp(design["dataEnd"])
    if new_dates is None:
        new_dates = panel.dates[panel.dates > design_end]
    if not len(new_dates):
        return {"status": "awaiting", "message": f"no sessions after {design['dataEnd']}"}
    ids = list(SHADOW_VARIANTS)
    summary_path = OUT_DIR / "summary.json"
    rec = None
    if summary_path.exists():
        try:
            rec = json.loads(summary_path.read_text()).get("recommendation")
            if rec and rec.get("champion") and rec["champion"] not in ids:
                ids.append(rec["champion"])
        except (OSError, ValueError, json.JSONDecodeError):
            rec = None
    by_id = {c["id"]: c for c in latin_or_grid_candidates()}
    equity = {}
    start = new_dates[0]
    for cid in ids:
        cand = by_id.get(cid)
        if cand is None:
            continue
        path, returns, applied, _t, _c = simulate_candidate(cand, signals, sleeves, cost_model, adv, daily_vol)
        r = returns.loc[start:].fillna(0.0)
        if r.empty:
            continue
        equity[cid] = INITIAL_EQUITY * (1.0 + r).cumprod()
    if not equity:
        return {"status": "empty", "message": "no shadow equity rows"}
    frame = pd.DataFrame(equity)
    frame.to_csv(out_dir / "regime_allocation_forward_equity.csv")
    ledger_path = out_dir / "regime_allocation_forward.jsonl"
    seen = set()
    if ledger_path.exists():
        for line in ledger_path.read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            seen.add((row.get("date"), row.get("id")))
    new_lines = []
    for day, values in frame.iterrows():
        d = str(pd.Timestamp(day).date())
        for cid, v in values.items():
            if (d, cid) in seen:
                continue
            new_lines.append(json.dumps({"leg": "regime_allocation", "date": d, "id": cid,
                                         "equity": round(float(v), 2)}))
    if new_lines:
        with open(ledger_path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(new_lines) + "\n")
    last = frame.iloc[-1]
    table = [{"id": cid, "equity": float(last[cid]), "totalReturn": float(last[cid] / INITIAL_EQUITY - 1.0)}
             for cid in frame.columns]
    table.sort(key=lambda r: -r["totalReturn"])
    payload = {
        "status": "tracking",
        "start": str(new_dates[0].date()),
        "end": str(frame.index[-1].date()),
        "sessions": int(len(frame)),
        "accounts": table,
        "leader": table[0]["id"] if table else None,
        "recommendation": rec,
        "sleeveNotes": sleeve_info.get("notes"),
        "note": "Shadow $10k allocation variants from the first post-design session. Not part of the frozen swarm design. No orders.",
        "livePaperUnchanged": True,
    }
    write_json(out_dir / "regime_allocation_forward.json", payload)
    return payload


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["research", "state", "forward"])
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--no-holdout-log", action="store_true")
    parser.add_argument("--eval-start", default=None,
                        help="Override swarm evalStart for pre-holdout selection (holdout dates stay sealed).")
    args = parser.parse_args(argv)
    if args.command == "research":
        from structured_log import log_run
        with log_run("regime_allocation"):
            summary = research(start=args.start, end=args.end, out_dir=args.out,
                               log_holdout=not args.no_holdout_log, eval_start=args.eval_start)
            rec = summary.get("recommendation") or {}
            print(json.dumps({"decision": rec.get("decision"), "champion": rec.get("champion"),
                              "robust": rec.get("robust"), "text": rec.get("text"),
                              "holdoutStart": summary.get("holdoutStart"),
                              "runSeconds": summary.get("runSeconds"),
                              "missing": (summary.get("data") or {}).get("missing")},
                             indent=1, default=_json_default))
        return 0
    if args.command == "state":
        print(json.dumps(latest_state(), indent=1, default=_json_default))
        return 0
    design_path = DATA_DIR / "Research" / "swarm" / "design.json"
    if not design_path.exists():
        print(json.dumps({"status": "no_design"}))
        return 0
    design = json.loads(design_path.read_text())
    panel, _ = load_study_panel()
    payload = attach_forward_shadows(panel, design)
    print(json.dumps(payload, indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
