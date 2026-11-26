"""Repair one stored market session: move a mis-dated local JSON aside, re-download it from Massive, reload SQLite.

Usage (dry run first, then --apply):
    python engine/repair_session.py --date 2026-11-06
    python engine/repair_session.py --date 2026-11-06 --apply
"""

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

from fetch_and_model import (
    DATA_DIR, DATABASE, HISTORY_JSON, check_session_date, download, market_bars, read_results, stamped_dates,
    write_atomically,
)
from local_env import massive_key
from price_history import open_history, save_session

LOGGER = logging.getLogger(__name__)


def local_copies(day):
    name = f"prices_{day}.json"
    return [path for path in (DATA_DIR / name, DATA_DIR / "Processed" / name, HISTORY_JSON / name) if path.is_file()]


def inspect(day):
    report = []
    for path in local_copies(day):
        records = read_results(path.read_bytes())
        stamped = sorted(d.isoformat() for d in stamped_dates(records))
        report.append({"path": str(path), "records": len(records), "stampedDates": stamped,
                       "matches": stamped in ([day.isoformat()], [])})
    return report


def repair(day, apply=False, include_otc=False):
    found = inspect(day)
    result = {"date": day.isoformat(), "localFiles": found, "applied": False}
    if not apply:
        return result
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    moved = []
    for item in found:
        if not item["matches"]:
            source = Path(item["path"])
            target = source.with_name(f"{source.name}.misdated-{stamp}")
            source.replace(target)
            moved.append(str(target))
    raw = download(day, massive_key(), include_otc=include_otc)
    records = read_results(raw, require_adjusted=True)
    check_session_date(records, day)
    bars = market_bars(records, day)
    if not bars:
        raise ValueError(f"Massive returned no bars for {day}.")
    path = HISTORY_JSON / f"prices_{day}.json"
    write_atomically(path, raw)
    with open_history(DATABASE, writable=True) as connection:
        save_session(connection, day.isoformat(), bars, path.stat())
        spy = connection.execute("SELECT open, close FROM daily_prices WHERE ticker='SPY' AND date=?",
                                 (day.isoformat(),)).fetchone()
    result.update(applied=True, movedAside=moved, savedFile=str(path), bars=len(bars),
                  spy=dict(spy) if spy else None)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--include-otc", action="store_true")
    args = parser.parse_args()
    print(json.dumps(repair(date.fromisoformat(args.date), args.apply, args.include_otc), indent=1))
