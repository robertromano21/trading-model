"""Chart payload for one ticker: OHLCV bars, engine-consistent indicators, paper-trade markers and data flags.

Read-only. Daily bars come from Massive/prices.sqlite3 (full local history); intraday bars from the Schwab cache
Massive/intraday.sqlite3 (table intraday_bars, 1- or 5-minute, regular session). Times: daily = "YYYY-MM-DD";
intraday = US/Eastern wall-clock time encoded as UTC epoch seconds (so the chart axis reads in ET).

  python chart_data.py NVDA --tf 1d
  python chart_data.py NVDA --tf 5m --sessions 10
"""

import argparse
import json
import math
import sqlite3
import sys
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_and_model import DATABASE, DATA_DIR
import indicators

INTRADAY_DB = DATA_DIR / "intraday.sqlite3"
PAPER_DB = DATA_DIR / "paper_trading.sqlite3"
MARKET = "SPY"
EXCLUDING_KINDS = ("jump_reversal", "unadjusted_split", "split_basis_mismatch")


def _ro(path):
    if not Path(path).is_file():
        return None
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _has_table(connection, name):
    return connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def _clean(value):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, 6)
    if isinstance(value, (np.floating,)):
        return _clean(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _series_points(times, series):
    values = series.to_numpy(dtype=float)
    return [{"time": t, "value": round(float(v), 6)} for t, v in zip(times, values) if math.isfinite(v)]


def daily_frame(connection, ticker, start=None):
    query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE ticker = ?"
    params = [ticker]
    if start:
        query += " AND date >= ?"
        params.append(start)
    frame = pd.read_sql_query(query + " ORDER BY date", connection, params=params)
    return frame.set_index("date")


def long_frame(connection, ticker, start=None):
    """Schwab daily_long bars (split-adjusted; schwab_daily.py backfill or the chart picker's on-demand fetch)."""
    if connection is None or not _has_table(connection, "daily_long"):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    query = "SELECT date, open, high, low, close, volume FROM daily_long WHERE ticker = ? AND source = 'schwab'"
    params = [ticker]
    if start:
        query += " AND date >= ?"
        params.append(start)
    return pd.read_sql_query(query + " ORDER BY date", connection, params=params).set_index("date")


def merged_daily(connection, ticker, start=None):
    """Massive daily_prices (preferred) extended with Schwab daily_long before/after it. The Schwab leg is scaled to
    Massive on the first overlapping day (Massive is split+dividend adjusted, Schwab split-only). Returns (frame, meta)."""
    mass = daily_frame(connection, ticker, start)
    long = long_frame(connection, ticker, start)
    if long.empty:
        return mass, None
    if mass.empty:
        return long, {"schwabRows": len(long), "scale": None}
    first, last = mass.index[0], mass.index[-1]
    factor = None
    if first in long.index and float(long.loc[first, "close"]):
        f = float(mass.loc[first, "close"]) / float(long.loc[first, "close"])
        factor = f if 0.5 <= f <= 2.0 else None
    earlier = long[long.index < first].copy()
    if factor and len(earlier):
        earlier[["open", "high", "low", "close"]] = earlier[["open", "high", "low", "close"]] * factor
    later = long[long.index > last]
    frame = pd.concat([earlier, mass[["open", "high", "low", "close", "volume"]], later])
    return frame, {"schwabRows": int(len(earlier) + len(later)), "scale": factor}


def intraday_frame(connection, symbol, frequency, sessions):
    dates = [r[0] for r in connection.execute(
        "SELECT DISTINCT session_date FROM intraday_bars WHERE symbol=? AND frequency=? ORDER BY session_date DESC LIMIT ?",
        (symbol, frequency, sessions))]
    if not dates:
        return pd.DataFrame(columns=["session_date", "minute", "open", "high", "low", "close", "volume", "ts"])
    frame = pd.read_sql_query(
        "SELECT ts, session_date, minute, open, high, low, close, volume FROM intraday_bars "
        "WHERE symbol=? AND frequency=? AND session_date >= ? ORDER BY ts", connection, params=(symbol, frequency, min(dates)))
    frame["time"] = [int(datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()) + int(m) * 60
                     for d, m in zip(frame["session_date"], frame["minute"])]
    return frame.set_index("time")


def data_flags(connection, ticker):
    if connection is None or not _has_table(connection, "price_anomalies"):
        return {"anomalies": [], "excluded": False, "masked": []}
    rows = [dict(r) for r in connection.execute(
        "SELECT date, kind, ratio, status FROM price_anomalies WHERE ticker=? ORDER BY date", (ticker,))]
    excluded = any(r["status"] in ("open", "persists") and r["kind"] in EXCLUDING_KINDS for r in rows)
    repairs = []
    if _has_table(connection, "ticker_repairs"):
        repairs = [dict(r) for r in connection.execute(
            "SELECT repaired_at, max_change FROM ticker_repairs WHERE ticker=? ORDER BY repaired_at", (ticker,))]
    return {"anomalies": rows[-20:], "anomalyCount": len(rows), "excluded": excluded, "repairs": repairs,
            "masked": [r["date"] for r in rows if r["status"] == "masked"]}


def paper_markers(ticker, intraday):
    """Swing paper fills and open orders (daily charts) or day-trading entries/exits (intraday charts)."""
    connection = _ro(PAPER_DB)
    out = {"fills": [], "orders": [], "positions": [], "dayTrades": [], "stops": []}
    if connection is None:
        return out
    with closing(connection):
        if _has_table(connection, "fills"):
            out["fills"] = [dict(r) for r in connection.execute(
                "SELECT account, fill_date, shares, price, effective_price, cost_bps FROM fills WHERE ticker=? "
                "ORDER BY fill_date", (ticker,))]
        if _has_table(connection, "orders"):
            out["orders"] = [dict(r) for r in connection.execute(
                "SELECT account, decision_date, side, shares, ref_price, reason, status FROM orders WHERE ticker=? "
                "AND status IN ('pending','open') ORDER BY decision_date", (ticker,))]
        if _has_table(connection, "positions"):
            out["positions"] = [dict(r) for r in connection.execute(
                "SELECT account, shares, avg_cost, entry_date, last_price, last_price_date FROM positions WHERE ticker=?",
                (ticker,))]
        if _has_table(connection, "signal_state"):
            out["stops"] = [dict(r) for r in connection.execute(
                "SELECT account, asof, target_weight, sleeve, stop FROM signal_state WHERE ticker=? AND asof = "
                "(SELECT MAX(asof) FROM signal_state s2 WHERE s2.account = signal_state.account)", (ticker,))]
        if _has_table(connection, "dt_trades"):
            out["dayTrades"] = [dict(r) for r in connection.execute(
                "SELECT account, session_date, direction, entry_minute, exit_minute, shares, entry_price, exit_price, "
                "stop, reason, pnl FROM dt_trades WHERE symbol=? ORDER BY session_date DESC LIMIT 200", (ticker,))]
    return out


def _build(ticker, tf, sessions, start, prices):
    ticker = ticker.upper().strip()
    intraday = tf in ("1m", "5m")
    payload = {"ticker": ticker, "tf": tf, "generatedAt": datetime.now().isoformat(timespec="seconds"),
               "bars": [], "volume": [], "indicators": {}, "warnings": []}
    if intraday:
        frequency = 1 if tf == "1m" else 5
        sessions = sessions or (5 if frequency == 1 else 20)
        cache = _ro(INTRADAY_DB)
        if cache is None or not _has_table(cache, "intraday_bars"):
            payload["warnings"].append("No Schwab intraday cache yet (Massive/intraday.sqlite3). Run schwab_data.py fetch.")
            return payload
        # load extra history for indicator warm-up (200-bar averages), then trim to the requested sessions
        frame = intraday_frame(cache, ticker, frequency, sessions + 3)
        market = intraday_frame(cache, MARKET, frequency, sessions + 3) if ticker != MARKET else None
        payload["source"] = f"Schwab {frequency}-minute bars (local cache)"
        bars_per_session = 390 // frequency
        last_fetch = cache.execute("SELECT MAX(fetched_at) FROM intraday_fetches WHERE symbol=? AND frequency=?",
                                   (ticker, frequency)).fetchone()[0] if _has_table(cache, "intraday_fetches") else None
        payload["lastFetch"] = last_fetch
        cache.close()
    else:
        if prices is None:
            payload["warnings"].append("Local price database Massive/prices.sqlite3 not found.")
            return payload
        frame, long_meta = merged_daily(prices, ticker, start)
        market = merged_daily(prices, MARKET, start)[0]["close"] if ticker != MARKET else None
        payload["source"] = "Local daily database (Massive/prices.sqlite3)"
        if long_meta and long_meta["schwabRows"]:
            payload["source"] += f" + {long_meta['schwabRows']} Schwab daily_long bars (split-adjusted" + \
                (", scaled to Massive at the splice)" if long_meta.get("scale") else ")")
        payload["history"] = {"bars": int(len(frame)), "schwab": bool(long_meta and long_meta["schwabRows"])}
        bars_per_session = 1
    if frame.empty:
        payload["warnings"].append(f"No {tf} bars for {ticker} in the local data.")
        payload["dataFlags"] = data_flags(prices, ticker)
        return payload
    market_close = market["close"] if intraday and market is not None and not market.empty else market
    values = indicators.compute(frame, market_close=market_close, intraday=intraday, bars_per_session=bars_per_session)
    if intraday:
        keep_dates = sorted(frame["session_date"].unique())[-sessions:]
        keep = frame["session_date"].isin(keep_dates).to_numpy()
    else:
        keep = np.ones(len(frame), dtype=bool)
    shown = frame[keep]
    times = list(shown.index)
    payload["bars"] = [{"time": t, "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
                       for t, o, h, l, c in zip(times, shown["open"], shown["high"], shown["low"], shown["close"])]
    payload["volume"] = [{"time": t, "value": float(v), "up": bool(c >= o)}
                         for t, v, o, c in zip(times, shown["volume"], shown["open"], shown["close"])]
    payload["indicators"] = {name: _series_points(times, series[keep]) for name, series in values.items()}
    if not intraday:
        dxy_path = DATA_DIR / "Macro" / "dxy_proxy.csv"
        if dxy_path.is_file() and times:
            try:
                dxy = pd.read_csv(dxy_path, parse_dates=[0], index_col=0)["dxy"].astype(float)
                aligned = dxy.reindex(pd.DatetimeIndex(pd.to_datetime(times))).ffill()
                payload["indicators"]["dxy_proxy"] = _series_points(times, aligned)
            except Exception:
                pass
    last = shown.iloc[-1]
    latest = {name: _clean(float(series[keep].iloc[-1])) for name, series in values.items() if len(series[keep])}
    payload["latest"] = {"time": times[-1], "close": float(last["close"]), **latest}
    if not intraday:
        try:
            import exhaustion as _exh
            payload["exhaustion"] = _exh.badge({
                "score": latest.get("exhaustion"),
                "up_streak": latest.get("up_streak"),
                "down_streak": latest.get("down_streak"),
                "vol_fade": latest.get("vol_fade"),
                "flag_up": latest.get("exhaustion_flag_up"),
                "flag_down": latest.get("exhaustion_flag_down"),
            })
        except Exception:
            pass
    payload["first"], payload["last"] = times[0], times[-1]
    if intraday:
        payload["sessions"] = keep_dates
        payload["firstSession"], payload["lastSession"] = keep_dates[0], keep_dates[-1]
    payload["dataFlags"] = data_flags(prices, ticker)
    payload["paper"] = paper_markers(ticker, intraday)
    return payload


def build(ticker, tf="1d", sessions=None, start=None):
    prices = _ro(DATABASE)
    try:
        return _build(ticker, tf, sessions, start, prices)
    finally:
        if prices is not None:
            prices.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ticker")
    parser.add_argument("--tf", choices=["1d", "5m", "1m"], default="1d")
    parser.add_argument("--sessions", type=int, help="intraday sessions to return (default 5 for 1m, 20 for 5m)")
    parser.add_argument("--start", help="first daily date (default: full local history)")
    args = parser.parse_args(argv)
    if not args.ticker.replace(".", "").replace("-", "").isalnum() or len(args.ticker) > 10:
        raise SystemExit("Invalid ticker")
    sessions = max(1, min(args.sessions, 60)) if args.sessions else None
    json.dump(build(args.ticker, args.tf, sessions, args.start), sys.stdout, separators=(",", ":"), default=_clean)


if __name__ == "__main__":
    main()
