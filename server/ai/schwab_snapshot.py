"""Read-only Schwab quote + fundamentals snapshot for the AI analyst (server/ai/sources.js).

Reuses engine/schwab_data.py's SchwabMarketData, whose URL allow-list only permits /marketdata/v1/ and the OAuth
token refresh; account and order endpoints (/trader/...) are refused in code. Prints one JSON object; never prints
tokens or secrets. Run from the engine folder:  python ../server/ai/schwab_snapshot.py NVDA AAPL
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())


def main(symbols):
    try:
        from schwab_data import SchwabAuthError, SchwabMarketData
    except Exception as error:  # engine not importable
        print(json.dumps({"status": "error", "message": f"schwab_data unavailable: {error}"}))
        return
    try:
        client = SchwabMarketData()
        data = client.get("/marketdata/v1/quotes", symbols=",".join(symbols), fields="quote,fundamental,reference,regular")
    except SchwabAuthError as error:
        print(json.dumps({"status": "auth_required", "message": str(error)[:400]}))
        return
    except Exception as error:
        print(json.dumps({"status": "error", "message": str(error)[:400]}))
        return
    quotes = {}
    for symbol, item in (data or {}).items():
        if not isinstance(item, dict):
            continue
        q, f, r, reg = (item.get(k, {}) or {} for k in ("quote", "fundamental", "reference", "regular"))
        quotes[symbol] = {
            "last": q.get("lastPrice"), "bid": q.get("bidPrice"), "ask": q.get("askPrice"), "mark": q.get("mark"),
            "open": q.get("openPrice"), "high": q.get("highPrice"), "low": q.get("lowPrice"),
            "prevClose": q.get("closePrice"), "changePct": q.get("netPercentChange"), "volume": q.get("totalVolume"),
            "quoteTimeMs": q.get("quoteTime"), "high52": q.get("52WeekHigh"), "low52": q.get("52WeekLow"),
            "regularClose": reg.get("regularMarketLastPrice"), "regularChangePct": reg.get("regularMarketPercentChange"),
            "regularTradeTimeMs": reg.get("regularMarketTradeTime"),
            "peRatio": f.get("peRatio"), "eps": f.get("eps"), "divYield": f.get("divYield"),
            "divExDate": f.get("divExDate"), "lastEarningsDate": f.get("lastEarningsDate"),
            "avg10DaysVolume": f.get("avg10DaysVolume"), "avg1YearVolume": f.get("avg1YearVolume"),
            "sharesOutstanding": f.get("sharesOutstanding"), "description": r.get("description"),
            "exchange": r.get("exchangeName"), "shortable": r.get("isShortable"), "hardToBorrow": r.get("isHardToBorrow"),
        }
    print(json.dumps({"status": "ok", "quotes": quotes}))


if __name__ == "__main__":
    main([s.upper() for s in sys.argv[1:]] or ["SPY"])
