"""Paper-only ledger for generic individual Treasuries marked to the FRED curve.

Never talks to Schwab. Bond ETFs already trade as equities in the existing paper trader.
Ledger path: Massive/Macro/treasury_paper.json (gitignored).

Usage (from engine/):
  python paper_treasuries.py status
  python paper_treasuries.py price TSY_10Y
  python paper_treasuries.py buy TSY_10Y --quantity 10
  python paper_treasuries.py sell TSY_10Y
  python paper_treasuries.py mark
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from fetch_and_model import DATA_DIR, write_atomically
from macro_data import load_treasury_curve
from treasury import GENERIC_TENORS, PaperTreasuryBook, as_date, issue_generic, mark_from_curve

LEDGER = DATA_DIR / "Macro" / "treasury_paper.json"


def _curve():
    curve = load_treasury_curve(lag=1)
    if curve is None or curve.empty:
        raise SystemExit("No FRED Treasury curve on disk. Run: python macro_data.py update")
    return curve


def _book_from(payload):
    book = PaperTreasuryBook(cash=float(payload.get("cash", 100_000.0)))
    for ticker, pos in (payload.get("positions") or {}).items():
        spec = issue_generic(ticker, pos["issue"], pos["coupon"], face=pos.get("face", 1000.0))
        spec.maturity = as_date(pos["maturity"])
        spec.issue = as_date(pos["issue"])
        spec.kind = pos.get("kind", spec.kind)
        book.positions[ticker] = {"spec": spec, "quantity": pos["quantity"], "cost_dirty": pos.get("cost_dirty", 0.0)}
    book.history = list(payload.get("history") or [])
    return book


def load_ledger():
    if not LEDGER.is_file():
        return PaperTreasuryBook()
    return _book_from(json.loads(LEDGER.read_text(encoding="utf-8")))


def save_ledger(book, marked=None):
    payload = {
        "cash": book.cash,
        "positions": {
            t: {"issue": p["spec"].issue.isoformat(), "maturity": p["spec"].maturity.isoformat(),
                "coupon": p["spec"].coupon, "kind": p["spec"].kind, "face": p["spec"].face,
                "quantity": p["quantity"], "cost_dirty": p["cost_dirty"]}
            for t, p in book.positions.items()
        },
        "history": book.history,
        "lastMark": marked,
    }
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(LEDGER, json.dumps(payload, indent=2, default=str).encode())
    return payload


def price_one(ticker, settlement=None):
    if ticker not in GENERIC_TENORS:
        raise SystemExit(f"unknown {ticker}; one of {sorted(GENERIC_TENORS)}")
    curve = _curve()
    settlement = as_date(settlement or curve.index[-1])
    row = curve.loc[:pd.Timestamp(settlement)]
    if row.empty:
        raise SystemExit("no curve on or before this date")
    from macro_data import interpolate_par_yield
    ytm = interpolate_par_yield(row.iloc[-1], GENERIC_TENORS[ticker]) / 100.0
    spec = issue_generic(ticker, settlement, ytm)
    return mark_from_curve(spec, settlement, row.iloc[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["status", "price", "buy", "sell", "mark"], nargs="?", default="status")
    parser.add_argument("ticker", nargs="?", help="TSY_3M, TSY_2Y, TSY_5Y, TSY_10Y or TSY_30Y")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--settlement", help="YYYY-MM-DD (default: last lagged curve date)")
    parser.add_argument("--cash", type=float, default=None, help="set starting cash when creating the ledger")
    args = parser.parse_args(argv)
    if args.command == "price":
        if not args.ticker:
            raise SystemExit("price needs a ticker")
        print(json.dumps(price_one(args.ticker, args.settlement), indent=2, default=str))
        return 0
    book = load_ledger()
    if args.cash is not None and not book.positions and args.command == "status":
        book.cash = args.cash
        save_ledger(book)
    curve = _curve()
    settlement = as_date(args.settlement or curve.index[-1])
    if args.command == "buy":
        if not args.ticker:
            raise SystemExit("buy needs a ticker")
        book.buy(args.ticker, settlement, curve, quantity=args.quantity)
        marked = book.mark(settlement, curve)
        save_ledger(book, marked)
        print(json.dumps(marked, indent=2, default=str))
        return 0
    if args.command == "sell":
        if not args.ticker:
            raise SystemExit("sell needs a ticker")
        book.sell(args.ticker, settlement, curve, quantity=args.quantity)
        marked = book.mark(settlement, curve)
        save_ledger(book, marked)
        print(json.dumps(marked, indent=2, default=str))
        return 0
    marked = book.mark(settlement, curve)
    if args.command == "mark":
        save_ledger(book, marked)
    print(json.dumps({**marked, "history": book.history[-10:]}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
