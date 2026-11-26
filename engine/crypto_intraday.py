"""Continuous BTC and ETH intraday paper accounts using Coinbase candles."""

import json
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from math import isfinite
from urllib.parse import urlencode

from crypto_data import COINBASE, COINS, CRYPTO_DIR, _get_json, write_json
from crypto_research import FEE, SLIP
from schwab_data import SchwabMarketData

STRATEGIES = {
    "trend": ("Moving-average momentum", 224),
    "breakout": ("Price breakout", 226),
    "matched": ("Momentum during dollar coverage", 228),
    "dollar": ("Momentum with a flat or falling dollar", 230),
}
ACCOUNTS = {first + i: (symbol, strategy) for strategy, (_, first) in STRATEGIES.items()
            for i, symbol in enumerate(("BTC", "ETH"))}
INTERVAL = 15 * 60
HISTORY = 192
FAST, SLOW = 8, 24
INITIAL, ALLOCATION, ENTRY_SPREAD = 10_000.0, 0.25, 0.005
PAPER_DIR = CRYPTO_DIR / "paper"


def iso(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def candle_range(symbol, first, latest):
    query = urlencode({"granularity": INTERVAL, "start": iso(first), "end": iso(latest + INTERVAL)})
    url = COINBASE.format(product=COINS[symbol]["product"])
    bars = {}
    for stamp, _, _, opening, close, _ in _get_json(f"{url}?{query}", timeout=6):
        if first <= stamp <= latest and stamp % INTERVAL == 0 and all(isfinite(p) and p > 0 for p in (opening, close)):
            bars[stamp] = (opening, close)
    return bars


def candles(symbol, latest):
    return candle_range(symbol, latest - (HISTORY - 1) * INTERVAL, latest)


def dollar_range(client, start, end):
    payload = client.price_history("UUP", start, end, frequency=1, extended=True)
    return {int(r["datetime"]) // 1000: r["close"] for r in payload.get("candles", [])
            if isfinite(r["close"]) and r["close"] > 0 and r["datetime"] // 1000 + 60 <= end.timestamp()}


def dollar_candles(latest):
    end = datetime.fromtimestamp(latest + INTERVAL, timezone.utc)
    start = end - timedelta(seconds=HISTORY * INTERVAL, minutes=20)
    return dollar_range(SchwabMarketData(), start, end)


def dollar_change(bars, close_time):
    for delay in range(3):
        last = close_time - (delay + 1) * 60
        prior = last - 15 * 60
        if last in bars and prior in bars:
            return bars[last] / bars[prior] - 1
    return None


def new_account():
    return dict(cash=INITIAL, equity=INITIAL, qty=0.0, last=None, pending=None, price=None, decision=None)


def advance(state, bars, latest, strategy="trend", dollar_bars=None, replay=False):
    """Replay completed, contiguous bars; a close signal can fill only on the following open."""
    if latest not in bars:
        return [], "Waiting for the latest completed Coinbase candle."
    last = state["last"]
    if last is not None and latest <= last:
        return [], None

    first = latest if last is None else last + INTERVAL
    start = latest
    while start - INTERVAL in bars:
        start -= INTERVAL
    if replay:
        first = start + (SLOW - 1) * INTERVAL
    if first > latest:
        return [], "Waiting for contiguous candles to calculate both averages."
    gap = last is not None and first - (SLOW - 1) * INTERVAL < start
    if gap:
        # Resume at the newest close. An order from before the gap has no observable next open.
        first = latest
    if first - (SLOW - 1) * INTERVAL < start:
        return [], "Waiting for contiguous candles to calculate both averages."
    if gap:
        state["pending"] = None

    decisions = []
    for stamp in range(first, latest + 1, INTERVAL):
        opening, close = bars[stamp]
        fill = None
        side = state["pending"]
        if side:
            price = opening * (1 + SLIP if side == "BUY" else 1 - SLIP)
            if side == "BUY":
                budget = state["cash"] * ALLOCATION
                qty = budget / (price * (1 + FEE))
                fee = qty * price * FEE
                state["cash"] -= budget
                state["qty"] = qty
            else:
                qty = state["qty"]
                fee = qty * price * FEE
                state["cash"] += qty * price - fee
                state["qty"] = 0.0
            fill = {"side": side, "at": iso(stamp), "price": price, "qty": qty, "fee": fee}

        closes = [bars[t][1] for t in range(stamp - (SLOW - 1) * INTERVAL, stamp + 1, INTERVAL)]
        spread = (sum(closes[-FAST:]) / FAST) / (sum(closes) / SLOW) - 1
        move = dollar_change(dollar_bars or {}, stamp + INTERVAL)
        if strategy == "breakout":
            exit_signal = close <= sum(closes[-FAST:]) / FAST
            entry_signal = close > max(closes[-13:-1])
            exit_reason = "The close reached or fell below its 8-bar average."
            entry_reason = "The close exceeded the previous 12 closes."
            hold_reason = "The price breakout strategy found no exit or new entry."
        else:
            exit_signal, entry_signal = spread <= 0, spread >= ENTRY_SPREAD
            exit_reason = "The 8-bar average fell to or below the 24-bar average."
            entry_reason = "The 8-bar average is at least 0.5% above the 24-bar average."
            hold_reason = ("The 8-bar average still leads the 24-bar average." if state["qty"] else
                           "The 8-bar average is below the entry threshold.")
        if state["qty"] and exit_signal:
            signal = "SELL"
            reason = exit_reason
        elif not state["qty"] and entry_signal:
            if strategy in ("matched", "dollar") and move is None:
                signal, reason = "HOLD", "The entry signal needs recent one-minute UUP prices."
            elif strategy == "dollar" and move > 0:
                signal, reason = "HOLD", "UUP rose over the last 15 minutes, so the dollar filter blocked the entry."
            else:
                signal, reason = "BUY", entry_reason
        else:
            signal, reason = "HOLD", hold_reason
        if gap:
            reason = f"Resumed from the newest candle after a data gap. {reason}"
        state.update(last=stamp, pending=signal if signal != "HOLD" else None,
                     equity=state["cash"] + state["qty"] * close, price=close)
        decision = {"at": iso(stamp + INTERVAL), "signal": signal, "reason": reason, "spread": spread,
                    "dollarReturn": move, "equity": state["equity"], "fill": fill}
        state["decision"] = decision
        decisions.append((stamp, decision))
    return decisions, None


def leaderboard(events):
    aligned = set.intersection(*(set(events[i]) for i in ACCOUNTS))
    stamps = sorted(aligned)
    if not stamps:
        return {"bars": 0, "rows": [], "dollarBars": 0, "dollarDelta": None}
    first = stamps[0]
    rows = []
    for strategy, (label, account) in STRATEGIES.items():
        pair = (events[account], events[account + 1])
        peak, drawdown, fills = 1.0, 0.0, 0
        for stamp in stamps:
            value = sum(book[stamp]["equity"] / book[first]["equity"] for book in pair) / 2
            peak = max(peak, value)
            drawdown = min(drawdown, value / peak - 1)
            if stamp > first:
                fills += sum(book[stamp]["fill"] is not None for book in pair)
        rows.append({"strategy": strategy, "label": label, "return": value - 1, "maxDrawdown": drawdown, "fills": fills})
    returns = {row["strategy"]: row["return"] for row in rows}
    dollar_bars = sum(events[228][t].get("dollarReturn") is not None for t in stamps)
    return {"bars": len(stamps), "start": iso(first + INTERVAL), "end": iso(stamps[-1] + INTERVAL),
            "dollarBars": dollar_bars, "dollarDelta": returns["dollar"] - returns["matched"] if dollar_bars else None,
            "rows": sorted(rows, key=lambda row: -row["return"])}


def run(directory=PAPER_DIR, now=None, load=candles, load_dollar=dollar_candles):
    now = time.time() if now is None else now
    latest = int((now - 60) // INTERVAL) * INTERVAL - INTERVAL
    feeds = {}
    for symbol in ("BTC", "ETH"):
        try:
            feeds[symbol] = (load(symbol, latest), None)
        except (RuntimeError, OSError, ValueError, TypeError) as error:
            feeds[symbol] = ({}, str(error))
    dollar_error = None
    try:
        dollar_bars = load_dollar(latest)
    except (RuntimeError, OSError, ValueError, TypeError) as error:
        dollar_bars, dollar_error = {}, str(error).split(" To fix:")[0]

    replay = {}
    for account, (symbol, strategy) in ACCOUNTS.items():
        decisions, _ = advance(new_account(), feeds[symbol][0], latest, strategy, dollar_bars, replay=True)
        replay[account] = dict(decisions)

    directory.mkdir(parents=True, exist_ok=True)
    accounts, n_new, n_fills = [], 0, 0
    with closing(sqlite3.connect(directory / "crypto_intraday.sqlite3", timeout=30)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE IF NOT EXISTS accounts (id INTEGER PRIMARY KEY, state TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS decisions (
            account INTEGER, candle INTEGER, payload TEXT NOT NULL, PRIMARY KEY (account, candle))""")
        db.execute("""CREATE INDEX IF NOT EXISTS crypto_fills ON decisions (candle DESC, account)
                      WHERE json_extract(payload, '$.fill') IS NOT NULL""")
        for account, (symbol, strategy) in ACCOUNTS.items():
            row = db.execute("SELECT state FROM accounts WHERE id = ?", (account,)).fetchone()
            state = json.loads(row[0]) if row else new_account()
            bars, error = feeds[symbol]
            decisions, message = advance(state, bars, latest, strategy, dollar_bars) if not error else ([], error)
            if decisions:
                db.executemany("INSERT INTO decisions VALUES (?, ?, ?)",
                               [(account, stamp, json.dumps(decision)) for stamp, decision in decisions])
                db.execute("INSERT OR REPLACE INTO accounts VALUES (?, ?)", (account, json.dumps(state)))
            n_new += len(decisions)
            n_fills += sum(d["fill"] is not None for _, d in decisions)
            accounts.append({"id": account, "symbol": symbol, "strategy": STRATEGIES[strategy][0], **state, "message": message,
                             "totalReturn": state["equity"] / INITIAL - 1})
        recent = db.execute("""SELECT account, payload FROM decisions
                               WHERE json_extract(payload, '$.fill') IS NOT NULL
                               ORDER BY candle DESC, account LIMIT 20""").fetchall()
        history = {account: {} for account in ACCOUNTS}
        for account, candle, payload in db.execute("SELECT account, candle, payload FROM decisions WHERE candle >= ?",
                                                   (latest - 30 * 86400,)):
            decision = json.loads(payload)
            if "equity" in decision:
                history[account][candle] = decision
        status = {"status": "waiting" if all(a["message"] for a in accounts) else
                  "partial" if any(a["message"] for a in accounts) else "tracking",
                  "updatedAt": iso(now), "intervalMinutes": INTERVAL // 60, "initialEquity": INITIAL,
                  "fees": {"perSide": FEE, "slippage": SLIP}, "accounts": accounts,
                  "recentFills": [{"account": i, "symbol": ACCOUNTS[i][0], **json.loads(payload)["fill"]}
                                  for i, payload in recent],
                  "dollar": {"symbol": "UUP", "lastBar": iso(max(dollar_bars) + 60) if dollar_bars else None,
                             "change": dollar_change(dollar_bars, latest + INTERVAL), "error": dollar_error},
                  "leaderboard": leaderboard(history), "replayLeaderboard": leaderboard(replay)}
    # Serialize the status export as well so a late concurrent runner cannot publish an older ledger.
    with closing(sqlite3.connect(directory / "crypto_intraday.sqlite3", timeout=30)) as db, db:
        db.execute("BEGIN IMMEDIATE")
        for account in accounts:
            saved = db.execute("SELECT state FROM accounts WHERE id = ?", (account["id"],)).fetchone()
            if saved and (account["last"] is None or json.loads(saved[0])["last"] > account["last"]):
                return {**status, "message": "A concurrent run advanced the intraday ledger."}
        path = directory / "crypto_intraday_status.json"
        if path.exists() and json.loads(path.read_text(encoding="utf8"))["updatedAt"] > status["updatedAt"]:
            return {**status, "message": "A newer intraday status was already published."}
        write_json(path, status)
    return {**status, "message": f"Crypto intraday: {n_new} new decision(s), {n_fills} paper fill(s)."}


if __name__ == "__main__":
    print(json.dumps(run()))
