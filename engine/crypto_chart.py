"""Chart payload for one crypto coin, in the same shape as chart_data.py so public/ticker.js can draw it.

Read-only market data. Daily bars come from the stored Coinbase public candles (Massive/Research/crypto/spot/<SYM>.csv,
UTC days, 24/7); --refresh tops them up from Coinbase. Hourly bars come from Coinbase hourly candles cached in
Massive/Research/crypto/hourly/<SYM>.csv (fetched incrementally, ~42 days kept). Hourly times are US/Eastern wall-clock
time encoded as UTC epoch seconds, like the stock intraday charts, so the axis reads in ET.

  python crypto_chart.py BTC --tf 1d
  python crypto_chart.py ETH --tf 1h --refresh
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import crypto_data as cd
import indicators

HOURLY_DIR = cd.CRYPTO_DIR / "hourly"
HOURLY_KEEP_DAYS = 42
NAMES = {"BTC": "Bitcoin", "ETH": "Ether (Ethereum)", "SOL": "Solana", "AVAX": "Avalanche", "LINK": "Chainlink",
         "XRP": "XRP", "DOGE": "Dogecoin", "ADA": "Cardano", "DOT": "Polkadot", "LTC": "Litecoin", "BCH": "Bitcoin Cash"}


def _clean(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else round(value, 8)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _points(times, series):
    values = series.to_numpy(dtype=float)
    return [{"time": t, "value": round(float(v), 8)} for t, v in zip(times, values) if np.isfinite(v)]


def daily_frame(coin):
    path = cd.SPOT_DIR / f"{coin}.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["date"] = frame["date"].astype(str).str.slice(0, 10)
    return frame.drop_duplicates("date", keep="last").set_index("date").sort_index()


def _fetch_hourly(coin, start, end):
    """Coinbase hourly candles [start, end) in 300-bar pages. Rows: [time, low, high, open, close, volume]."""
    rows = []
    cursor = start
    while cursor < end:
        stop = min(end, cursor + timedelta(hours=300))
        url = (cd.COINBASE.format(product=cd.COINS[coin]["product"]) + "?granularity=3600"
               + f"&start={cursor.strftime('%Y-%m-%dT%H:%M:%SZ')}&end={stop.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        data = cd._get_json(url)
        if isinstance(data, list):
            rows.extend(data)
        cursor = stop
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    frame = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"])
    return frame[["ts", "open", "high", "low", "close", "volume"]].astype(float).astype({"ts": "int64"})


def hourly_raw(coin, refresh):
    """Cached Coinbase hourly candles (UTC epoch ts), topped up when older than 5 minutes or on --refresh."""
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    path = HOURLY_DIR / f"{coin}.csv"
    cached = pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    stale = cached.empty or (time.time() - path.stat().st_mtime > 300)
    warning = None
    if refresh or stale:
        start = now - timedelta(days=HOURLY_KEEP_DAYS)
        if not cached.empty:
            start = max(start, datetime.fromtimestamp(int(cached["ts"].max()), timezone.utc) - timedelta(hours=2))
        try:
            fresh = _fetch_hourly(coin, start, now)
            cached = pd.concat([cached, fresh]).drop_duplicates("ts", keep="last")
            cutoff = (now - timedelta(days=HOURLY_KEEP_DAYS)).timestamp()
            cached = cached[cached["ts"] >= cutoff].sort_values("ts")
            tmp = path.with_suffix(".tmp")
            cached.to_csv(tmp, index=False)
            tmp.replace(path)
        except Exception as error:  # keep the cache
            warning = f"Coinbase hourly refresh failed ({type(error).__name__}); showing cached bars."
    return cached.sort_values("ts").reset_index(drop=True), warning


def hourly_frame(coin, refresh):
    raw, warning = hourly_raw(coin, refresh)
    if raw.empty:
        return pd.DataFrame(), warning
    et = pd.to_datetime(raw["ts"].astype("int64"), unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    raw = raw.assign(time=((et - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64").to_numpy())
    return raw.set_index("time")[["open", "high", "low", "close", "volume"]].astype(float), warning


def partial_today(coin, daily, refresh):
    """Today's still-open UTC daily bar built from hourly candles after the last stored daily close (or None)."""
    raw, warning = hourly_raw(coin, refresh)
    if raw.empty or daily.empty:
        return None, warning
    day = pd.to_datetime(raw["ts"].astype("int64"), unit="s", utc=True).dt.strftime("%Y-%m-%d")
    rows = raw[day > str(daily.index[-1])]
    if rows.empty:
        return None, warning
    out = {}
    for d, g in rows.groupby(day[rows.index]):
        out[d] = {"open": float(g["open"].iloc[0]), "high": float(g["high"].max()), "low": float(g["low"].min()),
                  "close": float(g["close"].iloc[-1]), "volume": float(g["volume"].sum())}
    return pd.DataFrame.from_dict(out, orient="index"), warning


def build(coin, tf="1d", refresh=False):
    coin = coin.upper().strip()
    payload = {"ticker": coin, "asset": "crypto", "name": NAMES.get(coin, coin), "tf": tf,
               "generatedAt": datetime.now().isoformat(timespec="seconds"), "bars": [], "volume": [], "indicators": {},
               "warnings": [], "paper": {}, "dataFlags": {"anomalies": [], "excluded": False, "masked": []}}
    if coin not in cd.ALL_COINS:
        payload["warnings"].append(f"{coin} is not in the local crypto universe ({', '.join(cd.ALL_COINS)}).")
        return payload
    if tf == "1h":
        frame, warning = hourly_frame(coin, refresh)
        if warning:
            payload["warnings"].append(warning)
        payload["source"] = "Coinbase hourly candles (public market data; times ET)"
        bars_per_session = 24
    else:
        if refresh:
            try:
                cd.update_spot(coins=[coin])
            except Exception as error:
                payload["warnings"].append(f"Coinbase daily refresh failed ({type(error).__name__}); showing stored bars.")
        frame = daily_frame(coin)
        payload["source"] = "Coinbase daily candles (UTC days, trades 24/7; Massive/Research/crypto/spot)"
        try:
            today, warning = partial_today(coin, frame, refresh)
            if today is not None and len(today):
                frame = pd.concat([frame[["open", "high", "low", "close", "volume"]], today])
                payload["partialFrom"] = str(today.index[0])
                payload["warnings"].append(f"Bars from {today.index[0]} are still open (built from Coinbase hourly candles).")
            elif warning:
                payload["warnings"].append(warning)
        except Exception as error:
            payload["warnings"].append(f"Could not add today's partial bar ({type(error).__name__}).")
        bars_per_session = 1
    if frame.empty:
        payload["warnings"].append(f"No {tf} candles stored for {coin}. Run `python crypto_data.py update`.")
        return payload
    # crypto trades 365 days a year; realized vol in indicators annualizes with sqrt(252) per bar, rescale to 365
    values = indicators.compute(frame, market_close=None, intraday=tf == "1h", bars_per_session=bars_per_session)
    adj = float(np.sqrt(365 / 252))
    for name in ("rvol20", "rvol60"):
        if name in values:
            values[name] = values[name] * adj
    times = list(frame.index)
    payload["bars"] = [{"time": t, "open": float(o), "high": float(h), "low": float(l), "close": float(c)}
                       for t, o, h, l, c in zip(times, frame["open"], frame["high"], frame["low"], frame["close"])]
    payload["volume"] = [{"time": t, "value": float(v), "up": bool(c >= o)}
                         for t, v, o, c in zip(times, frame["volume"], frame["open"], frame["close"])]
    payload["indicators"] = {name: _points(times, series) for name, series in values.items()}
    latest = {name: _clean(series.iloc[-1]) for name, series in values.items() if len(series)}
    payload["latest"] = {"time": times[-1], "close": float(frame["close"].iloc[-1]), **latest}
    if tf != "1h":
        try:
            import exhaustion as _exh
            payload["exhaustion"] = _exh.badge({"score": latest.get("exhaustion"), "up_streak": latest.get("up_streak"),
                                                "down_streak": latest.get("down_streak"), "vol_fade": latest.get("vol_fade"),
                                                "flag_up": latest.get("exhaustion_flag_up"),
                                                "flag_down": latest.get("exhaustion_flag_down")})
        except Exception:
            pass
    payload["first"], payload["last"] = times[0], times[-1]
    payload["schwab"] = cd.COINS[coin]["schwab"]   # spot = Schwab Crypto trades it; announced; no
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coin")
    parser.add_argument("--tf", choices=["1d", "1h"], default="1d")
    parser.add_argument("--refresh", action="store_true", help="top up from Coinbase before building")
    args = parser.parse_args(argv)
    if not args.coin.isalnum() or len(args.coin) > 6:
        raise SystemExit("Invalid coin")
    json.dump(build(args.coin, args.tf, args.refresh), sys.stdout, separators=(",", ":"), default=_clean)


if __name__ == "__main__":
    main()
