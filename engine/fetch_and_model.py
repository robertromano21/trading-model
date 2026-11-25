"""Load Massive daily bars and optionally append them to the existing ADX workbook."""

import json
import logging
import math
import os
import re
import sqlite3
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from price_history import (
    complete_market_dates, coverage, day_bars, model_history, open_history, save_session, save_ticker_bars,
    session_manifest, ticker_history,
)

ADX_TICKERS = ("NVDA", "ARM", "TSLA", "BROS", "CHWY", "FLY", "MRNA", "STLD", "HUT", "PLTR")
FIELDS = {"open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
MARKET_HEADERS = ("Date", "Ticker", "Open", "High", "Low", "Close", "Volume")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("PRICES_DATA_DIR") or PROJECT_ROOT / "Massive")
DATABASE = DATA_DIR / "prices.sqlite3"
HISTORY_JSON = DATA_DIR / "History" / "JSON"
WORKBOOK = Path(os.environ.get("ADX_WORKBOOK_PATH") or PROJECT_ROOT / "ADX worksheet 2.0.xlsm")
MASSIVE_AGGS_URL = "https://api.massive.com/v2/aggs"
LOGGER = logging.getLogger(__name__)


class MassiveError(RuntimeError):
    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def write_atomically(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            temporary.write(contents)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)


def read_results(raw, *, require_adjusted=False):
    document = json.loads(raw)
    if not isinstance(document, dict) or document.get("status") not in (None, "OK", "DELAYED"):
        raise ValueError("The JSON does not contain a successful Massive response.")
    results = document.get("results")
    if results is None and document.get("resultsCount") == 0:
        return []
    if not isinstance(results, list) or any(not isinstance(record, dict) for record in results):
        raise ValueError("The JSON does not contain a Massive results array.")
    if results and require_adjusted and document.get("adjusted") is not True:
        raise ValueError("Massive did not mark the returned stock prices as split-adjusted.")
    if "resultsCount" in document and document["resultsCount"] != len(results):
        raise ValueError("The Massive results count does not match the JSON records.")
    return results


def download_json(url, api_key):
    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as error:
        detail = {
            401: " Check MASSIVE_API_KEY.",
            403: " Check API key permissions and plan access for this date.",
            429: " The request exceeded the API rate limit.",
        }.get(error.code, "")
        retry_after = error.headers.get("Retry-After", "") if error.headers else ""
        error.close()
        raise MassiveError(f"Massive returned HTTP {error.code}.{detail}", error.code,
                           int(retry_after) if retry_after.isdecimal() else None) from None
    except (URLError, TimeoutError) as error:
        raise MassiveError(f"Could not reach Massive: {error.reason if isinstance(error, URLError) else error}") from None


def download(market_date, api_key, *, include_otc=False):
    url = f"{MASSIVE_AGGS_URL}/grouped/locale/us/market/stocks/{market_date}?adjusted=true"
    if include_otc:
        url += "&include_otc=true"
    return download_json(url, api_key)


def legacy_market_date(records):
    timestamp = next((record.get("t") for record in records if type(record.get("t")) is int), None)
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).date() if timestamp is not None else None


def dated_files(directory):
    dates = set()
    for path in directory.glob("prices_????-??-??.json"):
        if path.is_file():
            try:
                dates.add(date.fromisoformat(path.stem.removeprefix("prices_")))
            except ValueError:
                LOGGER.warning("Ignoring unrecognized JSON filename: %s", path.name)
    return dates


def available_dates(include_archive):
    dates = dated_files(DATA_DIR)
    if include_archive:
        dates |= dated_files(DATA_DIR / "Processed")
        dates |= dated_files(HISTORY_JSON)
    legacy = DATA_DIR / "prices.json"
    if legacy.is_file():
        try:
            market_date = legacy_market_date(read_results(legacy.read_bytes()))
        except (OSError, ValueError, OverflowError) as error:
            market_date = None
            LOGGER.warning("Could not read prices.json: %s", error)
        if market_date is None:
            LOGGER.warning("Omitting prices.json because its market date could not be determined.")
        elif include_archive or not (DATA_DIR / "Processed" / f"prices_{market_date}.json").is_file():
            dates.add(market_date)
    return sorted(dates)


def read_local(market_date):
    filename = f"prices_{market_date}.json"
    for path in (DATA_DIR / filename, DATA_DIR / "Processed" / filename, HISTORY_JSON / filename):
        try:
            return path, read_results(path.read_bytes())
        except FileNotFoundError:
            continue

    legacy = DATA_DIR / "prices.json"
    if legacy.is_file():
        records = read_results(legacy.read_bytes())
        if legacy_market_date(records) == market_date:
            return legacy, records
    return None, []


def dated_source(market_date):
    name = f"prices_{market_date}.json"
    return next((path for path in (DATA_DIR / name, DATA_DIR / "Processed" / name, HISTORY_JSON / name)
                 if path.is_file()), None)


def market_bars(records, market_date):
    bars = []
    seen = set()
    for record in records:
        ticker = record.get("T")
        if not isinstance(ticker, str) or not ticker.strip() or ticker in seen:
            raise ValueError("The JSON contains a missing or duplicate ticker.")
        seen.add(ticker)
        values = {field: record.get(key) for field, key in FIELDS.items()}
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
            raise ValueError(f"{ticker} has missing or invalid OHLCV values.")
        opening, closing = values["open"], values["close"]
        direction = "up" if closing > opening else "down" if closing < opening else "flat"
        bars.append({"ticker": ticker, "date": market_date.isoformat(), **values, "direction": direction})
    return bars


def as_market_date(value, epoch):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, (float, int)) and 0 <= value < 2957000:
        return epoch + timedelta(days=int(value))
    if isinstance(value, str):
        for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y"):
            try:
                return datetime.strptime(value.strip().split(" ")[0], pattern).date()
            except ValueError:
                continue
    return None


def open_latest_sheet(workbook):
    from pywintypes import com_error

    try:
        sheet = workbook.Worksheets("Latest Prices")
    except com_error:
        sheet = workbook.Worksheets.Add(After=workbook.Worksheets(workbook.Worksheets.Count))
        sheet.Name = "Latest Prices"
        sheet.Range("A1:G1").Value2 = (MARKET_HEADERS,)
        sheet.Range("A1:G1").Font.Bold = True
        sheet.Columns("A:B").ColumnWidth = 14
    if tuple(sheet.Range("A1:G1").Value2[0]) != MARKET_HEADERS:
        raise RuntimeError('The worksheet "Latest Prices" has different headers.')

    last = max(sheet.Cells(sheet.Rows.Count, column).End(-4162).Row for column in range(1, 8))
    return sheet, last


def write_latest_snapshot(sheet, previous_last, bars, market_date, epoch):
    serial = (market_date - epoch).days
    rows = tuple((serial, bar["ticker"], *(bar[field] for field in FIELDS)) for bar in bars)
    last = len(rows) + 1
    if last > sheet.Rows.Count:
        raise RuntimeError('The worksheet "Latest Prices" is full.')
    sheet.Range(f"B2:B{last}").NumberFormat = "@"
    sheet.Range(f"A2:G{last}").Value2 = rows
    sheet.Range(f"A2:A{last}").NumberFormat = "yyyy-mm-dd"
    if previous_last > last:
        sheet.Range(f"A{last + 1}:G{previous_last}").ClearContents()


def append_adx_sheets(workbook, bars, market_date, epoch):
    from pywintypes import com_error
    from excel_adx import extend_adx_formulas

    appended = 0
    bars_by_ticker = {bar["ticker"]: bar for bar in bars}
    for ticker in ADX_TICKERS:
        bar = bars_by_ticker.get(ticker)
        if bar is None:
            continue
        try:
            sheet = workbook.Worksheets(ticker.lower())
        except com_error:
            LOGGER.warning("Workbook has no %s ADX sheet; its data is on the market sheet.", ticker.lower())
            continue

        last = max(sheet.Cells(sheet.Rows.Count, column).End(-4162).Row for column in range(1, 7))
        if last == 2:
            known = {as_market_date(sheet.Cells(2, 1).Value2, epoch)}
        elif last > 2:
            known = {as_market_date(row[0], epoch) for row in sheet.Range(f"A2:A{last}").Value2}
        else:
            known = set()

        newest = max((value for value in known if value is not None), default=date.min)
        if market_date <= newest:
            extend_adx_formulas(sheet, sheet.Cells(sheet.Rows.Count, 1).End(-4162).Row)
            continue
        row = last + 1
        sheet.Range(f"A{row}:F{row}").Value2 = (((market_date - epoch).days, *(bar[field] for field in FIELDS)),)
        sheet.Cells(row, 1).NumberFormat = "yyyy-mm-dd"
        extend_adx_formulas(sheet, row)
        appended += 1
    return appended


def update_workbook(bars, market_date):
    from pywintypes import com_error
    from win32com.client import DispatchEx
    from excel_adx import refresh_adx_summary

    if not WORKBOOK.is_file():
        raise RuntimeError(f"Workbook not found: {WORKBOOK}")
    excel = None
    workbook = None
    try:
        excel = DispatchEx("Excel.Application")
        excel.Visible = False
        excel.DisplayAlerts = False
        excel.EnableEvents = False
        excel.AutomationSecurity = 3  # msoAutomationSecurityForceDisable
        workbook = excel.Workbooks.Open(str(WORKBOOK.resolve()), 0, False, None, "", "", True)
        if workbook.ReadOnly:
            raise RuntimeError("Excel opened the workbook read-only. Close it in Excel and retry.")

        epoch = date(1904, 1, 1) if workbook.Date1904 else date(1899, 12, 30)
        sheet, last = open_latest_sheet(workbook)
        previous = as_market_date(sheet.Cells(2, 1).Value2, epoch)
        if previous is not None and previous > market_date:
            return {"status": "newer", "date": previous.isoformat(), "snapshot": 0, "adxAppended": 0}
        write_latest_snapshot(sheet, last, bars, market_date, epoch)
        adx_appended = append_adx_sheets(workbook, bars, market_date, epoch)
        excel.Calculate()
        adx_summary = refresh_adx_summary(workbook)
        workbook.Save()
        return {"status": "saved", "date": market_date.isoformat(), "snapshot": len(bars),
                "adxAppended": adx_appended, "adxSummary": adx_summary}
    except com_error as error:
        details = error.excepinfo
        raise RuntimeError(details[2].strip() if details and details[2] else "Excel could not update the workbook.") from None
    finally:
        try:
            if workbook is not None:
                workbook.Close(False)
        finally:
            if excel is not None:
                excel.Quit()


def archive_session(path, market_date):
    archive = DATA_DIR / "Processed" / f"prices_{market_date}.json"
    if path == archive:
        return "already archived"
    if path.parent == HISTORY_JSON:
        return "kept in history"
    if path.name == "prices.json":
        write_atomically(archive, path.read_bytes())
        return "copied to archive"
    archive.parent.mkdir(parents=True, exist_ok=True)
    path.replace(archive)
    return "archived"


def run(options):
    with open_history(DATABASE, writable=True) as history:
        return import_prices(options, history)


def import_prices(options, history):
    source = options["source"]
    if source == "pending":
        dates = available_dates(include_archive=False)
    elif source in ("local", "massive"):
        start = date.fromisoformat(options["startDate"])
        end = date.fromisoformat(options["endDate"])
        count = (end - start).days + 1
        if not 1 <= count <= 366:
            raise ValueError("Choose an ordered date range of up to 366 days.")
        dates = [start + timedelta(days=offset) for offset in range(count)]
        if count > 1:
            dates = [market_date for market_date in dates if market_date.weekday() < 5]
    else:
        raise ValueError("Choose existing JSON, pending JSON, or a Massive download.")
    api_key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if source == "massive" and not api_key:
        raise ValueError("Set MASSIVE_API_KEY before requesting downloads.")

    bars_count, sessions, to_archive = 0, [], []
    excel = {"status": "skipped"}
    next_download = time.monotonic()
    for market_date in dates:
        session = {"date": market_date.isoformat(), "status": "ready", "count": 0}
        sessions.append(session)
        try:
            if source == "massive":
                time.sleep(max(0, next_download - time.monotonic()))
                next_download = time.monotonic() + 2
                raw = download(market_date, api_key)
                records = read_results(raw, require_adjusted=True)
                daily_bars = market_bars(records, market_date)
                path = DATA_DIR / f"prices_{market_date}.json"
                write_atomically(path, raw)
            else:
                path, records = read_local(market_date)
                daily_bars = market_bars(records, market_date)

            if path is None:
                session.update(status="missing", message="No matching JSON file.")
                LOGGER.warning("%s: no matching JSON file", market_date)
                continue
            save_session(history, market_date.isoformat(), daily_bars, path.stat())
            session.update(file=path.name, count=len(daily_bars))
            if not daily_bars:
                session.update(status="empty", message="No stock prices in the response.")
            else:
                bars_count += len(daily_bars)
                to_archive.append((path, session))
            LOGGER.info("%s: %s loaded with %s stock prices", market_date, path.name, len(daily_bars))
        except (OSError, ValueError, RuntimeError, OverflowError, sqlite3.Error) as error:
            session.update(status="error", message=str(error))
            LOGGER.error("%s: %s", market_date, error)

    if options["updateWorkbook"]:
        excel["status"] = "no-data"
        if bars_count:
            try:
                latest = coverage(history)["lastDate"]
                excel = update_workbook(day_bars(history, latest), date.fromisoformat(latest))
                LOGGER.info("Excel: %s for %s; %s snapshot rows, %s ADX rows appended",
                            excel["status"], excel["date"], excel["snapshot"], excel["adxAppended"])
            except (OSError, ValueError, RuntimeError, ImportError, sqlite3.Error) as error:
                excel.update(status="failed", message=str(error))
                LOGGER.error("Excel: %s", error)
            else:
                for path, session in to_archive:
                    try:
                        action = archive_session(path, session["date"])
                        session["archive"] = action
                        LOGGER.info("%s: %s", session["date"], action)
                    except OSError as error:
                        session.update(status="error", message=f"SQLite saved; archive failed: {error}")
                        LOGGER.error("%s: %s", session["date"], session["message"])

    has_issues = excel["status"] == "failed" or any(session["status"] in ("missing", "error") for session in sessions)
    status = ("partial" if bars_count else "failed") if has_issues else "complete"
    result = {"status": status, "barsCount": bars_count, "sessions": sessions, "excel": excel}
    if source == "pending" and not dates:
        result["message"] = "No pending JSON files found."
    return result


def read_session(market_date):
    parsed_date = date.fromisoformat(market_date)
    if DATABASE.is_file():
        with open_history(DATABASE) as history:
            if history.execute("SELECT 1 FROM market_sessions WHERE date = ?", (market_date,)).fetchone():
                return {"status": "complete", "date": market_date, "bars": day_bars(history, market_date)}
    path, records = read_local(parsed_date)
    if path is None:
        return {"status": "missing", "bars": []}
    return {"status": "complete", "date": market_date, "bars": market_bars(records, parsed_date)}


def index_local_files(history, as_of):
    saved = session_manifest(history)
    for market_date in available_dates(include_archive=True):
        if market_date > as_of:
            break
        try:
            path = dated_source(market_date)
            day = market_date.isoformat()
            if path is None:
                path, records = read_local(market_date)
                if path is None:
                    continue
                stat = path.stat()
            else:
                stat = path.stat()
                if day in saved and saved[day][1:] == (stat.st_size, stat.st_mtime_ns):
                    continue
                records = read_results(path.read_bytes())
            save_session(history, day, market_bars(records, market_date), stat)
        except (OSError, ValueError, OverflowError) as error:
            if market_date == as_of:
                raise
            LOGGER.warning("Model skipped %s: %s", market_date, error)


def run_strategies(options):
    from strategy_model import scan

    as_of = date.fromisoformat(options["date"])
    with open_history(DATABASE, writable=True) as history:
        index_local_files(history, as_of)
        history.execute("BEGIN")
        return scan(model_history(history, options["date"]), options, LOGGER, complete_market_dates(history))


def import_ticker_range(ticker, start, end, api_key):
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,31}", ticker):
        raise ValueError("Use a valid stock ticker.")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last:
        raise ValueError("Choose an ordered date range.")
    if not api_key:
        raise ValueError("Set MASSIVE_API_KEY before downloading ticker history.")
    url = f"{MASSIVE_AGGS_URL}/ticker/{ticker}/range/1/day/{start}/{end}?adjusted=true&sort=asc&limit=50000"
    raw = download_json(url, api_key)
    document = json.loads(raw)
    records = read_results(raw, require_adjusted=True)
    if document.get("ticker") != ticker or document.get("next_url"):
        raise ValueError("The response did not contain the complete adjusted ticker range.")

    daily = {}
    for record in records:
        timestamp = record.get("t")
        if type(timestamp) is not int:
            raise ValueError("A stock price is missing its date.")
        market_date = datetime.fromtimestamp(timestamp / 1000, timezone.utc).date()
        if market_date in daily or not first <= market_date <= last:
            raise ValueError("The ticker range contains a duplicate or out-of-range date.")
        daily[market_date] = market_bars([{**record, "T": ticker}], market_date)[0]

    path = DATA_DIR / "History" / "TickerRanges" / f"{ticker}_{start}_{end}.json"
    write_atomically(path, raw)
    with open_history(DATABASE, writable=True) as history:
        inserted = save_ticker_bars(history, daily.values())
    return {"status": "complete", "ticker": ticker, "received": len(daily), "inserted": inserted,
            "firstDate": min(daily).isoformat() if daily else None, "lastDate": max(daily).isoformat() if daily else None}


def query_ticker(ticker, through):
    if not DATABASE.is_file():
        return {"status": "complete", "ticker": ticker, "bars": []}
    with open_history(DATABASE) as history:
        return {"status": "complete", "ticker": ticker, "bars": ticker_history(history, ticker, through)}


def history_status():
    if not DATABASE.is_file():
        return {"status": "complete", "sessions": 0, "bars": 0, "firstDate": None, "lastDate": None}
    with open_history(DATABASE) as history:
        return {"status": "complete", **coverage(history)}


def saved_market_dates():
    if not DATABASE.is_file():
        return {"status": "complete", "dates": []}
    with open_history(DATABASE) as history:
        return {"status": "complete", "dates": sorted(complete_market_dates(history), reverse=True)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--read-date":
            result = read_session(sys.argv[2])
        elif len(sys.argv) == 2 and sys.argv[1] == "--strategies":
            result = run_strategies(json.load(sys.stdin))
        elif len(sys.argv) == 4 and sys.argv[1] == "--ticker-history":
            result = query_ticker(sys.argv[2], sys.argv[3])
        elif len(sys.argv) == 5 and sys.argv[1] == "--ticker-range":
            result = import_ticker_range(*sys.argv[2:5], os.environ.get("MASSIVE_API_KEY", "").strip())
        elif len(sys.argv) == 2 and sys.argv[1] == "--history-status":
            result = history_status()
        elif len(sys.argv) == 2 and sys.argv[1] == "--market-dates":
            result = saved_market_dates()
        else:
            result = run(json.load(sys.stdin))
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    except (ValueError, KeyError, TypeError, OSError, OverflowError, sqlite3.Error, MassiveError) as error:
        LOGGER.error("%s", error)
        print(json.dumps({"status": "failed", "message": str(error), "barsCount": 0, "sessions": []}))
