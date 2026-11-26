"""Fetchers for official calendars, first-release vintages, FX/dollar, auctions and overseas indices.

Network access is optional. Callers may inject an `opener` for tests (no live I/O). Secrets
(BLS / FRED / Massive keys) are read via local_env and are never logged or printed.

If the GB-12 macro worker has landed `macro_data.py`, its keyless FRED loader, DXY proxy and
Macro/ directory layout are reused.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from io import StringIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as XET
from zoneinfo import ZoneInfo

from fetch_and_model import DATA_DIR, write_atomically
from local_env import load_local_env

ET = ZoneInfo("America/New_York")
USER_AGENT = "Trading-Model-main/macro-calendar (personal research dashboard; +local)"
MACRO_DIR = DATA_DIR / "Macro"
OVERSEAS_DIR = MACRO_DIR / "overseas"
AUCTION_DIR = MACRO_DIR / "auctions"
FRED_DIR = MACRO_DIR / "fred"
ALFRED_DIR = MACRO_DIR / "alfred"

# Prefer the GB-12 user-agent and directory if that module is importable.
try:
    import macro_data as _GB12
    USER_AGENT = getattr(_GB12, "USER_AGENT", USER_AGENT)
    MACRO_DIR = getattr(_GB12, "MACRO_DIR", MACRO_DIR)
    FRED_DIR = getattr(_GB12, "FRED_DIR", FRED_DIR)
except Exception:  # noqa: BLE001
    _GB12 = None

FREDGRAPH = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALFREDGRAPH = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred"
BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_ICS = "https://www.bls.gov/schedule/news_release/bls.ics"
FED_MONETARY_RSS = "https://www.federalreserve.gov/feeds/press_monetary.xml"
FED_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
FISCAL_UPCOMING = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/upcoming_auctions"
FISCAL_AUCTIONS = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
CLEVELAND_NOWCAST = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"

# Yahoo symbols for cash equity indices (delayed; used for overnight features, not execution).
OVERSEAS_INDICES = {
    "Nikkei": {"yahoo": "^N225", "etf": "EWJ", "region": "asia", "closeEtApprox": "02:00"},
    "Hang Seng": {"yahoo": "^HSI", "etf": "FXI", "region": "asia", "closeEtApprox": "04:00"},
    "Shanghai Composite": {"yahoo": "000001.SS", "etf": "MCHI", "region": "asia", "closeEtApprox": "03:00"},
    "KOSPI": {"yahoo": "^KS11", "etf": "EWY", "region": "asia", "closeEtApprox": "02:30"},
    "DAX": {"yahoo": "^GDAXI", "etf": "EWG", "region": "europe", "closeEtApprox": "11:30"},
    "FTSE 100": {"yahoo": "^FTSE", "etf": "EWU", "region": "europe", "closeEtApprox": "11:30"},
    "CAC 40": {"yahoo": "^FCHI", "etf": "EZU", "region": "europe", "closeEtApprox": "11:30"},
    "Euro Stoxx 50": {"yahoo": "^STOXX50E", "etf": "FEZ", "region": "europe", "closeEtApprox": "11:30"},
}
OVERSEAS_ETFS = ("EWJ", "FXI", "MCHI", "EWY", "EWG", "EWU", "EZU", "EFA", "EEM")

# Map auction tenor keywords to a FRED par-yield series (prior-close WI proxy).
AUCTION_YIELD_PROXY = {
    "4-week": "DGS1MO", "8-week": "DGS1MO", "13-week": "DGS3MO", "17-week": "DGS3MO",
    "26-week": "DGS6MO", "52-week": "DGS1",
    "2-year": "DGS2", "3-year": "DGS3", "5-year": "DGS5", "7-year": "DGS7",
    "10-year": "DGS10", "20-year": "DGS20", "30-year": "DGS30",
    "tips 5": "DFII5", "tips 10": "DFII10", "tips 30": "DFII30",
}

# Sleep between outbound calls when building a calendar (be polite).
MIN_GAP_SEC = 0.4


class FetchError(RuntimeError):
    pass


def env_value(name, default=""):
    """Read a local setting. Never log or print the value."""
    load_local_env()
    import os
    return (os.environ.get(name) or default).strip()


def has_env(name):
    return bool(env_value(name))


def _strip_query(url):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def http_get(url, opener=None, timeout=30, headers=None, method="GET", data=None, accept=None):
    """GET/POST bytes. `opener` is a callable(url, method, data, headers) -> bytes for tests."""
    hdrs = {"User-Agent": USER_AGENT, "Accept": accept or "*/*"}
    if headers:
        hdrs.update(headers)
    if opener is not None:
        result = opener(url, method=method, data=data, headers=hdrs)
        if isinstance(result, tuple):
            if len(result) == 3:
                return result
            if len(result) == 2:
                return result[0], result[1], {}
        return result, 200, {}
    request = Request(url, data=data, headers=hdrs, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read(), getattr(response, "status", 200), dict(response.headers)
    except HTTPError as error:
        body = error.read() if hasattr(error, "read") else b""
        code = error.code
        try:
            error.close()
        except Exception:
            pass
        raise FetchError(f"HTTP {code} for {_strip_query(url)}") from None
    except (URLError, TimeoutError, OSError) as error:
        reason = getattr(error, "reason", error)
        raise FetchError(f"Could not reach {_strip_query(url)}: {reason}") from None


def http_text(url, **kwargs):
    raw, status, headers = http_get(url, **kwargs)
    return raw.decode("utf-8-sig", errors="replace"), status, headers


def _rate_sleep(state, gap=MIN_GAP_SEC):
    last = state.get("last")
    if last is not None:
        wait = gap - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    state["last"] = time.monotonic()


# ---- FRED / ALFRED (reuse GB-12 when present) ---------------------------------

def download_fred(series_ids, start=None, end=None, opener=None, timeout=60):
    """Keyless FRED graph CSV. Reuses macro_data.download_fred when no custom opener is set."""
    if opener is None and _GB12 is not None and hasattr(_GB12, "download_fred"):
        return _GB12.download_fred(series_ids, start=start, end=end, timeout=timeout)
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    params = {"id": ",".join(series_ids)}
    if start:
        params["cosd"] = str(start)
    if end:
        params["coed"] = str(end)
    url = FREDGRAPH + "?" + urlencode(params)
    raw, _, _ = http_get(url, opener=opener, timeout=timeout, accept="text/csv")
    return parse_fred_csv(raw, series_ids)


def parse_fred_csv(raw, series_ids=None):
    if _GB12 is not None and hasattr(_GB12, "parse_fred_csv") and series_ids is not None:
        try:
            return _GB12.parse_fred_csv(raw, series_ids)
        except Exception:
            pass
    import numpy as np
    import pandas as pd
    text = raw.decode("utf-8-sig") if isinstance(raw, (bytes, bytearray)) else raw
    frame = pd.read_csv(StringIO(text))
    if frame.empty:
        raise FetchError("FRED CSV was empty.")
    date_col = next((c for c in frame.columns if c.upper() == "DATE"), frame.columns[0])
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).set_index(date_col).sort_index()
    frame.index.name = "date"
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col].replace(".", np.nan), errors="coerce")
    return frame


def alfred_vintage(series_id, vintage_date, opener=None, start=None):
    """One ALFRED vintage via the keyless alfredgraph.csv (values as they stood on vintage_date)."""
    params = {"id": series_id, "vintage_date": str(vintage_date)[:10]}
    if start:
        params["cosd"] = str(start)
    url = ALFREDGRAPH + "?" + urlencode(params)
    raw, _, _ = http_get(url, opener=opener, accept="text/csv")
    frame = parse_fred_csv(raw, [series_id])
    col = series_id if series_id in frame.columns else frame.columns[0]
    return frame[col].rename(series_id)


def first_release_value(series_id, observation_date, vintage_date, opener=None):
    """Value of observation_date as first published on vintage_date (ALFRED)."""
    series = alfred_vintage(series_id, vintage_date, opener=opener,
                            start=str(observation_date)[:10])
    import pandas as pd
    ts = pd.Timestamp(observation_date)
    if ts in series.index and pd.notna(series.loc[ts]):
        return float(series.loc[ts])
    # Monthly series are often dated to the 1st of the reference month.
    month = series[series.index.to_period("M") == ts.to_period("M")]
    if len(month) and pd.notna(month.iloc[-1]):
        return float(month.iloc[-1])
    return None


def fred_api(path, params, opener=None):
    """Optional official FRED API (needs FRED_API_KEY in .env.local)."""
    key = env_value("FRED_API_KEY")
    if not key:
        raise FetchError("FRED_API_KEY is not set.")
    query = dict(params or {})
    query["api_key"] = key
    query.setdefault("file_type", "json")
    url = f"{FRED_API}/{path}?{urlencode(query)}"
    # Rebuild the URL without the key if we ever raise; http_get already strips query on errors.
    raw, status, _ = http_get(url, opener=opener, accept="application/json")
    return json.loads(raw.decode("utf-8"))


# ---- BLS ---------------------------------------------------------------------

def bls_series(series_ids, start_year, end_year, opener=None):
    """BLS public API v2. Uses BLS_API_KEY from local_env when present (never printed)."""
    if isinstance(series_ids, str):
        series_ids = [series_ids]
    payload = {"seriesid": list(series_ids), "startyear": str(start_year), "endyear": str(end_year),
               "annualaverage": False, "calculations": True}
    key = env_value("BLS_API_KEY")
    if key:
        payload["registrationkey"] = key
    data = json.dumps(payload).encode("utf-8")
    raw, status, _ = http_get(
        BLS_API, opener=opener, method="POST", data=data,
        headers={"Content-Type": "application/json"}, accept="application/json",
    )
    doc = json.loads(raw.decode("utf-8"))
    if str(doc.get("status", "")).lower() not in {"request succeeded", "succeeded", ""} and not doc.get("Results"):
        raise FetchError(f"BLS API status: {doc.get('status') or doc.get('message') or 'unknown'}")
    return doc


def parse_bls_ics(text):
    """Parse the public BLS iCalendar into (event_hint, date, time_et) rows."""
    rows = []
    blocks = re.split(r"\nBEGIN:VEVENT", text)
    for block in blocks[1:]:
        summary = _ics_field(block, "SUMMARY") or ""
        dtstart = _ics_field(block, "DTSTART") or ""
        if not dtstart:
            continue
        day, clock = _ics_datetime(dtstart)
        if day is None:
            continue
        hint = _bls_hint(summary)
        rows.append({"summary": summary.strip(), "date": day.isoformat(),
                     "timeEt": clock or "08:30", "eventId": hint})
    return rows


def _ics_field(block, name):
    match = re.search(rf"^{name}(?:;[^:]*)?:(.+)$", block, re.MULTILINE)
    if not match:
        return None
    value = match.group(1).strip()
    value = value.replace("\\n", " ").replace("\\,", ",").replace("\\;", ";")
    return value


def _ics_datetime(value):
    value = value.strip()
    # DTSTART;TZID=America/New_York:20261107T083000 or 20261107 or 20261107T133000Z
    if "T" in value:
        raw, zulu = value, value.endswith("Z")
        raw = raw.rstrip("Z")
        try:
            dt = datetime.strptime(raw, "%Y%m%dT%H%M%S")
        except ValueError:
            try:
                dt = datetime.strptime(raw, "%Y%m%dT%H%M")
            except ValueError:
                return None, None
        if zulu:
            dt = dt.replace(tzinfo=ZoneInfo("UTC")).astimezone(ET)
        else:
            dt = dt.replace(tzinfo=ET)
        return dt.date(), dt.strftime("%H:%M")
    try:
        return datetime.strptime(value[:8], "%Y%m%d").date(), None
    except ValueError:
        return None, None


def _bls_hint(summary):
    s = summary.lower()
    if "consumer price" in s or re.search(r"\bcpi\b", s):
        return "cpi"
    if "producer price" in s or re.search(r"\bppi\b", s):
        return "ppi"
    if "employment situation" in s or "employment sit" in s:
        return "nfp"
    if "job openings" in s or "jolts" in s:
        return "jolts"
    if "employment cost" in s:
        return "eci"
    return None


# ---- Fed RSS / calendar ------------------------------------------------------

def parse_fed_rss(text):
    """Return press items from the Fed monetary-policy RSS feed."""
    items = []
    try:
        root = XET.fromstring(text)
    except XET.ParseError as error:
        raise FetchError(f"Fed RSS was not valid XML: {error}") from None
    for item in root.findall("./channel/item") + root.findall(".//{*}item"):
        title = (item.findtext("title") or item.findtext("{*}title") or "").strip()
        link = (item.findtext("link") or item.findtext("{*}link") or "").strip()
        pub = (item.findtext("pubDate") or item.findtext("{*}pubDate") or "").strip()
        desc = (item.findtext("description") or item.findtext("{*}description") or "").strip()
        when = None
        if pub:
            try:
                when = parsedate_to_datetime(pub).astimezone(ET)
            except Exception:
                when = None
        kind = None
        low = title.lower() + " " + desc.lower()
        if "federal reserve issues fomc statement" in low or "fomc statement" in low:
            kind = "fomc_decision"
        elif "minutes of the federal open market committee" in low or "fomc minutes" in low:
            kind = "fomc_minutes"
        items.append({"title": title, "link": link, "datetimeEt": when.isoformat(timespec="minutes") if when else None,
                      "kind": kind, "text": _strip_tags(desc)})
    return items


def _strip_tags(html):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html or "")).strip()


def parse_fed_calendar_html(text):
    """Best-effort scrape of announcement dates from the FOMC calendars page."""
    days = []
    # The page lists month names and days; look for ISO-like or "Month DD, YYYY" near "Statement".
    for match in re.finditer(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:\s*[–-]\s*(\d{1,2}))?,?\s+(20\d{2})",
        text,
    ):
        month, d1, d2, year = match.group(1), int(match.group(2)), match.group(3), int(match.group(4))
        # Decision is on the last day of a two-day meeting.
        day_n = int(d2) if d2 else d1
        try:
            days.append(date(year, _month_num(month), day_n).isoformat())
        except ValueError:
            continue
    # Unique, sorted.
    return sorted(set(days))


def _month_num(name):
    months = ["january", "february", "march", "april", "may", "june",
              "july", "august", "september", "october", "november", "december"]
    return months.index(name.lower()) + 1


# ---- FiscalData auctions -----------------------------------------------------

def fiscal_get(url, params=None, opener=None, pages=8):
    """Paginated FiscalData JSON (no API key)."""
    rows, page = [], 1
    while page <= pages:
        query = dict(params or {})
        query.setdefault("page[size]", 100)
        query["page[number]"] = page
        body, _, _ = http_get(url + "?" + urlencode(query, safe="[],:"), opener=opener,
                              accept="application/json")
        doc = json.loads(body.decode("utf-8"))
        data = doc.get("data") or []
        rows.extend(data)
        total = int(((doc.get("meta") or {}).get("total-pages") or 1))
        if page >= total or not data:
            break
        page += 1
        if opener is None:
            time.sleep(MIN_GAP_SEC)
    return rows


def upcoming_auctions(opener=None, start=None, end=None):
    params = {"sort": "auction_date"}
    if start or end:
        lo = start or "2024-01-01"
        hi = end or "2030-12-31"
        params["filter"] = f"auction_date:gte:{lo},auction_date:lte:{hi}"
    return fiscal_get(FISCAL_UPCOMING, params, opener=opener)


def auction_results(opener=None, start=None, end=None):
    params = {"sort": "-auction_date"}
    if start or end:
        lo = start or "2024-01-01"
        hi = end or date.today().isoformat()
        params["filter"] = f"auction_date:gte:{lo},auction_date:lte:{hi}"
    # Request the fields we score. FiscalData ignores unknown fields rather than failing.
    params["fields"] = ",".join([
        "auction_date", "issue_date", "maturity_date", "security_type", "security_term",
        "cusip", "high_yield", "high_discount_rate", "high_investment_rate",
        "bid_to_cover_ratio", "indirect_bidder_accepted", "direct_bidder_accepted",
        "primary_dealer_accepted", "total_accepted", "total_tendered", "offering_amt",
        "allocation_percentage", "auction_format",
    ])
    return fiscal_get(FISCAL_AUCTIONS, params, opener=opener)


def tenor_proxy_series(term, security_type=""):
    blob = f"{term} {security_type}".lower()
    if "tips" in blob or "inflation" in blob:
        if "5-year" in blob or "5 year" in blob:
            return AUCTION_YIELD_PROXY["tips 5"]
        if "30-year" in blob or "30 year" in blob:
            return AUCTION_YIELD_PROXY["tips 30"]
        return AUCTION_YIELD_PROXY["tips 10"]
    for key, series in AUCTION_YIELD_PROXY.items():
        if key.startswith("tips"):
            continue
        if key.replace("-", " ") in blob.replace("-", " "):
            return series
    return None


# ---- Overseas indices (Yahoo, delayed) ---------------------------------------

def yahoo_chart(symbol, range_="1y", interval="1d", opener=None):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}&events=div%7Csplit"
    raw, _, _ = http_get(url, opener=opener, accept="application/json")
    doc = json.loads(raw.decode("utf-8"))
    result = ((doc.get("chart") or {}).get("result") or [None])[0]
    if not result:
        raise FetchError(f"Yahoo chart for {symbol} was empty.")
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    import pandas as pd
    idx = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("America/New_York")
    frame = pd.DataFrame({
        "open": quote.get("open"), "high": quote.get("high"), "low": quote.get("low"),
        "close": quote.get("close"), "volume": quote.get("volume"),
    }, index=idx)
    frame.index.name = "datetimeEt"
    return frame.dropna(how="all")


def load_overseas_indices(opener=None, range_="1y", cache=True):
    """Daily (and, when interval allows, open) bars for Asia/Europe cash indices."""
    import pandas as pd
    out = {}
    notes = []
    OVERSEAS_DIR.mkdir(parents=True, exist_ok=True)
    for name, meta in OVERSEAS_INDICES.items():
        path = OVERSEAS_DIR / f"{meta['etf']}_{name.replace(' ', '_').lower()}.csv"
        try:
            frame = yahoo_chart(meta["yahoo"], range_=range_, opener=opener)
            if cache:
                frame.to_csv(path)
            out[name] = {"meta": meta, "bars": frame, "path": str(path)}
        except Exception as error:  # noqa: BLE001
            notes.append(f"{name}: {error}")
            if cache and path.is_file():
                try:
                    cached = pd.read_csv(path, parse_dates=[0], index_col=0)
                    out[name] = {"meta": meta, "bars": cached, "path": str(path), "cached": True}
                except Exception:
                    pass
    return out, notes


# ---- Massive / Polygon economic calendar (optional, fail soft) ----------------

def try_massive_economics(opener=None, start=None, end=None):
    """Best-effort Massive/Polygon economics endpoints. Returns None if the plan has no such feed."""
    key = env_value("MASSIVE_API_KEY") or env_value("POLYGON_API_KEY")
    if not key:
        return None, "no Massive/Polygon key configured"
    # Candidate paths; Massive may 404 any of these depending on the plan. Never log the key.
    urls = [
        "https://api.massive.com/benzinga/v1/economics",
        "https://api.polygon.io/benzinga/v1/economics",
        "https://api.massive.com/v1/economics",
    ]
    errors = []
    params = {}
    if start:
        params["date.gte"] = str(start)
    if end:
        params["date.lte"] = str(end)
    for base in urls:
        url = base + ("?" + urlencode(params) if params else "")
        try:
            raw, status, _ = http_get(
                url, opener=opener,
                headers={"Authorization": f"Bearer {key}"},
                accept="application/json",
            )
            doc = json.loads(raw.decode("utf-8"))
            return doc, _strip_query(base)
        except Exception as error:  # noqa: BLE001
            errors.append(f"{_strip_query(base)}: {error}")
    return None, "; ".join(errors)


def try_cleveland_nowcast(opener=None):
    """Best-effort fetch of the Cleveland Fed inflation nowcast page / data. Fail soft."""
    try:
        text, status, _ = http_text(CLEVELAND_NOWCAST, opener=opener)
    except Exception as error:  # noqa: BLE001
        return None, str(error)
    # Look for a "CPI nowcast" style percentage on the page. This is a fallback, not a structured API.
    match = re.search(r"(?:CPI|nowcast)[^%]{0,80}?([+-]?\d+\.\d+)\s*%", text, re.IGNORECASE)
    if not match:
        return {"source": "cleveland_fed", "rawFound": False}, "no numeric nowcast parsed"
    return {"source": "cleveland_fed", "cpiNowcastPct": float(match.group(1)), "rawFound": True}, None


# ---- Official schedule merge --------------------------------------------------

BLS_ICS_TO_EVENTS = {
    "cpi": ("cpi", "core_cpi"),
    "ppi": ("ppi",),
    "nfp": ("nfp", "unrate"),
    "jolts": ("jolts",),
    "eci": ("eci",),
}


def fetch_official_schedules(start, end, opener=None):
    """Pull official calendars and return (rows, notes, source_names).

    Rows use the same shape as macro_calendar._row (datetimeEt, id, key, ...).
    """
    from macro_calendar import event_by_id, to_et, iso_et, _row, attach_manual

    start = start if isinstance(start, date) else date.fromisoformat(str(start))
    end = end if isinstance(end, date) else date.fromisoformat(str(end))
    rows, notes, used = [], [], []
    state = {}

    def grab(url, **kwargs):
        _rate_sleep(state)
        return http_text(url, opener=opener, **kwargs)

    # BLS ICS
    try:
        text, _, _ = grab(BLS_ICS, accept="text/calendar")
        ics_rows = parse_bls_ics(text)
        n = 0
        for item in ics_rows:
            day = date.fromisoformat(item["date"])
            if not (start <= day <= end) or not item.get("eventId"):
                continue
            for event_id in BLS_ICS_TO_EVENTS.get(item["eventId"], ()):
                event = event_by_id(event_id)
                if not event:
                    continue
                # Official ICS time wins over the default, still expressed in ET.
                event = dict(event)
                event["time"] = item["timeEt"] or event["time"]
                event["zone"] = "America/New_York"
                rows.append(_row(event, day, source="bls_ics", period=day.isoformat()[:7]))
                n += 1
        used.append("bls_ics")
        notes.append(f"BLS ICS: {n} dated releases in window.")
    except Exception as error:  # noqa: BLE001
        notes.append(f"BLS ICS unavailable: {error}")

    # Fed calendar HTML + published dates already in the fallback.
    try:
        text, _, _ = grab(FED_CALENDAR, accept="text/html")
        days = parse_fed_calendar_html(text)
        decision = event_by_id("fomc_decision")
        minutes = event_by_id("fomc_minutes")
        n = 0
        for raw in days:
            day = date.fromisoformat(raw)
            if start <= day <= end:
                rows.append(_row(decision, day, source="fed_calendar_html", period=raw))
                mday = day + timedelta(days=21)
                if start <= mday <= end:
                    rows.append(_row(minutes, mday, source="fed_calendar_html_plus_21d",
                                     period=raw, extra={"decisionDate": raw}))
                n += 1
        used.append("fed_calendar_html")
        notes.append(f"Fed calendar HTML: {n} meeting dates parsed (minutes default +21d until the official minutes date is confirmed).")
    except Exception as error:  # noqa: BLE001
        notes.append(f"Fed calendar HTML unavailable: {error}")

    # Treasury upcoming auctions
    try:
        auctions = upcoming_auctions(opener=opener, start=start, end=end)
        event = event_by_id("treasury_auction")
        n = 0
        for item in auctions:
            raw = (item.get("auction_date") or item.get("record_date") or "")[:10]
            if not raw:
                continue
            day = date.fromisoformat(raw)
            if not (start <= day <= end):
                continue
            term = item.get("security_term") or item.get("security_type") or "Treasury"
            typ = item.get("security_type") or ""
            clock = item.get("announcement_time") or item.get("auction_time") or event["time"]
            # Auction times from FiscalData, if present, are usually ET already.
            ev = dict(event)
            ev["time"] = str(clock)[:5] if str(clock)[:5].count(":") == 1 else event["time"]
            ev["zone"] = "America/New_York"
            rows.append(_row(
                ev, day, source="fiscaldata_upcoming",
                period=raw,
                extra={
                    "securityTerm": term,
                    "securityType": typ,
                    "cusip": item.get("cusip"),
                    "key": f"treasury_auction:{day.isoformat()}:{term}:{item.get('cusip') or typ}",
                    "name": f"Treasury auction: {term} {typ}".strip(),
                },
            ))
            n += 1
        used.append("fiscaldata_upcoming")
        notes.append(f"FiscalData upcoming auctions: {n} in window.")
    except Exception as error:  # noqa: BLE001
        notes.append(f"FiscalData upcoming auctions unavailable: {error}")

    # Optional Massive economics calendar (consensus dates / forecasts).
    try:
        doc, info = try_massive_economics(opener=opener, start=start, end=end)
        if doc:
            used.append(f"massive_economics:{info}")
            notes.append("Massive/Polygon economics endpoint responded; consensus will be merged in macro_surprise.")
            MACRO_DIR.mkdir(parents=True, exist_ok=True)
            write_atomically(MACRO_DIR / "massive_economics.json",
                             json.dumps({"fetchedAt": datetime.now(ET).isoformat(timespec="seconds"),
                                         "endpoint": info, "payload": _strip_secrets(doc)},
                                        default=str).encode("utf-8"))
        else:
            notes.append(f"Massive/Polygon economics not used ({info}).")
    except Exception as error:  # noqa: BLE001
        notes.append(f"Massive economics skipped: {error}")

    return rows, notes, used


def _strip_secrets(obj):
    """Drop common key-shaped fields before writing a cache file."""
    if isinstance(obj, dict):
        drop = {"apiKey", "api_key", "authorization", "Authorization", "registrationkey", "registrationKey"}
        return {k: _strip_secrets(v) for k, v in obj.items() if k not in drop}
    if isinstance(obj, list):
        return [_strip_secrets(v) for v in obj]
    return obj


def gb12_available():
    return _GB12 is not None


def load_dxy_proxy(lag=0):
    """Reuse the GB-12 DXY proxy when present; otherwise UUP from the price DB."""
    if _GB12 is not None and hasattr(_GB12, "load_dxy"):
        try:
            return _GB12.load_dxy(lag=lag), "macro_data.load_dxy"
        except Exception:
            pass
    try:
        from market_data import load_panel
        panel = load_panel(tickers=["UUP", "SPY"])
        close = panel.close["UUP"].dropna()
        if lag:
            close = close.shift(lag)
        return close.rename("dxy"), "uup_close"
    except Exception as error:  # noqa: BLE001
        return None, f"dxy unavailable: {error}"


def load_curve_yield(series_id, asof):
    """Prior-close par yield from GB-12's Treasury curve cache, or a FRED pull."""
    asof = str(asof)[:10]
    if _GB12 is not None and hasattr(_GB12, "load_treasury_curve"):
        try:
            curve = _GB12.load_treasury_curve(lag=0, asof=asof)
            # curve is a DataFrame indexed by date, columns like dgs10.
            col = series_id.lower() if hasattr(curve, "columns") else series_id
            if hasattr(curve, "columns"):
                for candidate in (series_id, series_id.lower(), col):
                    if candidate in curve.columns:
                        s = curve[candidate].dropna()
                        s = s[s.index <= _GB12_asof(asof)] if len(s) else s
                        if len(s):
                            return float(s.iloc[-1]), candidate, "macro_data.load_treasury_curve"
        except Exception:
            pass
    try:
        frame = download_fred(series_id, end=asof)
        col = series_id if series_id in frame.columns else frame.columns[0]
        s = frame[col].dropna()
        if len(s):
            return float(s.iloc[-1]), series_id, "fredgraph"
    except Exception:
        pass
    return None, series_id, "unavailable"


def _GB12_asof(asof):
    import pandas as pd
    return pd.Timestamp(asof)
