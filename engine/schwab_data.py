"""Read-only Schwab market-data client (quotes and price history) with a local intraday cache.

SAFETY: this module only talks to Schwab's market-data API (https://api.schwabapi.com/marketdata/v1/...) and, to
refresh an expired access token, the OAuth token endpoint. Every request URL is checked against that allow-list;
trading and account endpoints (/trader/v1/...: accounts, orders, transactions) are refused in code.

Auth: uses the schwab-py style token file Robert created (default <project>/schwab_token.json, override with
SCHWAB_TOKEN_PATH). The access token lasts 30 minutes; refreshing it needs SCHWAB_APP_KEY and SCHWAB_APP_SECRET in
the environment or .env.local. Secrets and tokens are never printed or logged.

Cache: Massive/intraday.sqlite3 (gitignored), table intraday_bars.

  python schwab_data.py status                      # token state (no secrets)
  python schwab_data.py quotes SPY QQQ AAPL
  python schwab_data.py fetch --days 48 [SYMBOLS]   # 1-minute bars, regular session, cached
"""

import argparse
import base64
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from fetch_and_model import DATA_DIR, write_atomically
from local_env import PROJECT_ROOT, load_local_env

API = "https://api.schwabapi.com"
ALLOWED_PREFIXES = ("/marketdata/v1/", "/v1/oauth/token")
FORBIDDEN_PREFIXES = ("/trader/",)
CACHE_DB = DATA_DIR / "intraday.sqlite3"
DEFAULT_SYMBOLS = ("SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AMD",
                   "AVGO", "NFLX", "JPM", "XLF", "XLK", "XLE", "TLT", "GLD", "PLTR", "MU", "SMH", "COIN", "UBER",
                   "LLY", "COST", "ORCL", "BAC", "XOM")


class SchwabAuthError(RuntimeError):
    pass


def _check_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "api.schwabapi.com":
        raise PermissionError(f"Refusing non-Schwab URL {parsed.netloc}")
    if parsed.path.startswith(FORBIDDEN_PREFIXES) or not parsed.path.startswith(ALLOWED_PREFIXES):
        raise PermissionError(f"Refusing {parsed.path}: only market-data endpoints are allowed.")


class SchwabMarketData:
    def __init__(self, token_path=None, min_interval=0.55):
        load_local_env()
        self.token_path = Path(token_path or os.environ.get("SCHWAB_TOKEN_PATH") or PROJECT_ROOT / "schwab_token.json")
        self.min_interval = min_interval
        self._last = 0.0

    # ------------------------------------------------------------------ auth (no secrets printed)
    def _read_token(self):
        if not self.token_path.is_file():
            raise SchwabAuthError(f"No Schwab token file at {self.token_path}. " + INSTRUCTIONS)
        return json.loads(self.token_path.read_text(encoding="utf-8"))

    def token_status(self):
        try:
            data = self._read_token()
        except SchwabAuthError as error:
            return {"usable": False, "message": str(error)}
        token = data.get("token", data)
        expires = token.get("expires_at", 0)
        created = data.get("creation_timestamp", 0)
        can_refresh = bool(os.environ.get("SCHWAB_APP_KEY") and os.environ.get("SCHWAB_APP_SECRET"))
        return {"tokenFile": str(self.token_path), "accessTokenExpires": _local(expires),
                "accessTokenValid": expires > time.time() + 60, "created": _local(created),
                "refreshTokenExpiresAbout": _local(created + 7 * 86400) if created else None,
                "canRefresh": can_refresh, "scope": token.get("scope")}

    def _refresh(self, data):
        key, secret = os.environ.get("SCHWAB_APP_KEY", "").strip(), os.environ.get("SCHWAB_APP_SECRET", "").strip()
        if not key or not secret:
            raise SchwabAuthError("The Schwab access token has expired and SCHWAB_APP_KEY / SCHWAB_APP_SECRET are not "
                                  "set, so it cannot be refreshed. " + INSTRUCTIONS)
        token = data["token"]
        url = f"{API}/v1/oauth/token"
        _check_url(url)
        body = urlencode({"grant_type": "refresh_token", "refresh_token": token["refresh_token"]}).encode()
        basic = base64.b64encode(f"{key}:{secret}".encode()).decode()
        request = Request(url, data=body, method="POST", headers={
            "Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urlopen(request, timeout=30) as response:
                fresh = json.loads(response.read())
        except HTTPError as error:
            raise SchwabAuthError(f"Schwab token refresh failed (HTTP {error.code}); the refresh token may have "
                                  "expired (7 days). " + INSTRUCTIONS) from None
        fresh.setdefault("refresh_token", token["refresh_token"])
        fresh["expires_at"] = int(time.time()) + int(fresh.get("expires_in", 1800))
        data["token"] = {**token, **fresh}
        write_atomically(self.token_path, json.dumps(data).encode())
        return data["token"]["access_token"]

    def _access_token(self):
        data = self._read_token()
        token = data.get("token", {})
        if token.get("expires_at", 0) > time.time() + 60:
            return token["access_token"]
        return self._refresh(data)

    # ------------------------------------------------------------------ requests
    def get(self, path, **params):
        url = f"{API}{path}?{urlencode({k: v for k, v in params.items() if v is not None})}"
        _check_url(url)
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(4):
            self._last = time.monotonic()
            request = Request(url, headers={"Authorization": f"Bearer {self._access_token()}",
                                            "Accept": "application/json"})
            try:
                with urlopen(request, timeout=60) as response:
                    return json.loads(response.read())
            except HTTPError as error:
                if error.code == 429 and attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                if error.code == 401:
                    raise SchwabAuthError("Schwab rejected the access token (HTTP 401). " + INSTRUCTIONS) from None
                raise RuntimeError(f"Schwab {path} failed with HTTP {error.code}") from None

    def quotes(self, symbols):
        return self.get("/marketdata/v1/quotes", symbols=",".join(symbols), fields="quote,reference")

    def price_history(self, symbol, start, end, frequency=1, extended=False):
        """Minute bars between datetimes start and end (Schwab keeps roughly the last 48 days of 1-minute data)."""
        return self.get("/marketdata/v1/pricehistory", symbol=symbol, periodType="day", frequencyType="minute",
                        frequency=frequency, startDate=int(start.timestamp() * 1000),
                        endDate=int(end.timestamp() * 1000), needExtendedHoursData=str(extended).lower())


INSTRUCTIONS = ("To fix: in PowerShell run  $env:SCHWAB_APP_KEY='<app key>'; $env:SCHWAB_APP_SECRET='<secret>'  (or put "
                "SCHWAB_APP_KEY=... and SCHWAB_APP_SECRET=... in .env.local, which is gitignored), then re-create the "
                "token with schwab-py:  python -c \"from schwab import auth; auth.easy_client(<key>, <secret>, "
                "'https://127.0.0.1:8182', 'schwab_token.json')\"  from the project folder. Refresh tokens last 7 days.")


def _local(epoch):
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S") if epoch else None


# ---------------------------------------------------------------------- cache
CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS intraday_bars (
  symbol TEXT NOT NULL, ts INTEGER NOT NULL, session_date TEXT NOT NULL, minute INTEGER NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL,
  frequency INTEGER NOT NULL DEFAULT 1, source TEXT NOT NULL DEFAULT 'schwab',
  PRIMARY KEY (symbol, frequency, ts)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS intraday_bars_date ON intraday_bars (session_date, symbol);
CREATE TABLE IF NOT EXISTS intraday_quotes (
  symbol TEXT NOT NULL, fetched_at TEXT NOT NULL, last REAL, bid REAL, ask REAL, bid_size REAL, ask_size REAL,
  volume REAL, quote_time INTEGER, PRIMARY KEY (symbol, fetched_at));
CREATE TABLE IF NOT EXISTS intraday_fetches (
  symbol TEXT NOT NULL, frequency INTEGER NOT NULL, start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL,
  bars INTEGER NOT NULL, fetched_at TEXT NOT NULL);
"""


def connect_cache(path=CACHE_DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(CACHE_SCHEMA)
    return connection


def _eastern(ms):
    from zoneinfo import ZoneInfo
    return datetime.fromtimestamp(ms / 1000, ZoneInfo("America/New_York"))


def store_candles(connection, symbol, candles, frequency=1, regular_only=True):
    rows = []
    for c in candles:
        stamp = _eastern(c["datetime"])
        minute = stamp.hour * 60 + stamp.minute
        if regular_only and not (570 <= minute < 960):      # 09:30 <= bar start < 16:00 ET
            continue
        rows.append((symbol, int(c["datetime"]), stamp.date().isoformat(), minute, float(c["open"]), float(c["high"]),
                     float(c["low"]), float(c["close"]), float(c.get("volume") or 0), frequency))
    with connection:
        connection.executemany("""INSERT OR REPLACE INTO intraday_bars (symbol, ts, session_date, minute, open, high, low,
                                  close, volume, frequency) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
    return len(rows)


def fetch_history(symbols, days=48, chunk_days=10, client=None, cache=None, log=print, frequency=1):
    client = client or SchwabMarketData()
    own = cache is None
    cache = cache or connect_cache()
    end = datetime.now(timezone.utc)
    summary = {}
    try:
        for symbol in symbols:
            total = 0
            stop = end
            while stop > end - timedelta(days=days):
                start = max(stop - timedelta(days=chunk_days), end - timedelta(days=days))
                payload = client.price_history(symbol, start, stop, frequency=frequency)
                candles = payload.get("candles", []) if isinstance(payload, dict) else []
                stored = store_candles(cache, symbol, candles, frequency)
                with cache:
                    cache.execute("INSERT INTO intraday_fetches VALUES (?, ?, ?, ?, ?, ?)",
                                  (symbol, frequency, int(start.timestamp() * 1000), int(stop.timestamp() * 1000), stored,
                                   datetime.now().isoformat(timespec="seconds")))
                total += stored
                stop = start
                if not candles:
                    break                       # nothing older available
            first, last, sessions = cache.execute(
                "SELECT MIN(session_date), MAX(session_date), COUNT(DISTINCT session_date) FROM intraday_bars "
                "WHERE symbol=? AND frequency=?", (symbol, frequency)).fetchone()
            summary[symbol] = {"stored": total, "first": first, "last": last, "sessions": sessions}
            log(f"{symbol}: {summary[symbol]}")
    finally:
        if own:
            cache.close()
    return summary


def store_quotes(connection, payload):
    stamp = datetime.now().isoformat(timespec="seconds")
    rows = []
    for symbol, item in (payload or {}).items():
        q = item.get("quote", {}) if isinstance(item, dict) else {}
        rows.append((symbol, stamp, q.get("lastPrice"), q.get("bidPrice"), q.get("askPrice"), q.get("bidSize"),
                     q.get("askSize"), q.get("totalVolume"), q.get("quoteTime")))
    with connection:
        connection.executemany("INSERT OR REPLACE INTO intraday_quotes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "quotes", "fetch"])
    parser.add_argument("symbols", nargs="*")
    parser.add_argument("--days", type=int, default=48)
    parser.add_argument("--frequency", type=int, default=1, choices=[1, 5, 10, 15, 30],
                        help="Minutes per bar. Schwab keeps ~48 days of 1-minute and ~9 months of 5-minute bars.")
    parser.add_argument("--chunk-days", type=int, default=10)
    args = parser.parse_args(argv)
    client = SchwabMarketData()
    try:
        if args.command == "status":
            print(json.dumps(client.token_status(), indent=2))
        elif args.command == "quotes":
            payload = client.quotes(args.symbols or DEFAULT_SYMBOLS[:10])
            with connect_cache() as cache:
                rows = store_quotes(cache, payload)
            print(json.dumps([dict(zip(("symbol", "fetchedAt", "last", "bid", "ask"), r[:5])) for r in rows], indent=2))
        else:
            summary = fetch_history(args.symbols or DEFAULT_SYMBOLS, args.days, args.chunk_days,
                                    log=lambda m: print(m, file=sys.stderr, flush=True), frequency=args.frequency)
            print(json.dumps(summary, indent=2))
    except SchwabAuthError as error:
        print(json.dumps({"status": "auth_required", "message": str(error)}, indent=2))
        sys.exit(2)


if __name__ == "__main__":
    from structured_log import log_run

    with log_run("schwab_data"):
        main()
