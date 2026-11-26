"""Public detection feeds for the geopolitical / black swan monitor.

Sources (free/public only; polite User-Agent; cached; rate-limited):
  * GDELT 2.0 DOC API — recent English articles and a 7-day vs 30-day volume
    timeline for conflict, protest and terror themes
  * Caldara–Iacoviello Geopolitical Risk (GPR) daily index from matteoiacoviello.com
  * Public RSS headline feeds that publish RSS (BBC, Guardian, NPR, Al Jazeera, WHO)
  * OFAC recent-actions page / RSS
  * Central-bank press RSS (Fed, ECB, Bank of England, Bank of Japan)

Never places orders. Network is injected via an `opener` so tests run offline.
Rate limits and terms are respected: one request at a time, a pause between
hosts, 15-minute caches, and no scraping behind logins or paywalls.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fetch_and_model import DATA_DIR, write_atomically

UA = "trading-model/1.0 (personal research; black-swan monitor; +local)"
TIMEOUT = 25
PAUSE_SEC = 1.25
CACHE_DIR = DATA_DIR / "Research" / "black_swan"
CACHE_TTL_SEC = 15 * 60

# GDELT 2.0 DOC API (https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = (
    '(war OR invasion OR missile OR sanctions OR "ceasefire" OR "peace deal" '
    "OR coup OR default OR pandemic OR earthquake OR hurricane OR cyclone "
    "OR cyberattack OR ransomware OR protest OR strike OR tariff OR embargo "
    "OR terrorism OR hostage) sourcelang:english"
)

# Official GPR daily file lives as .xls on the author's site; we try CSV mirrors
# first so we do not need xlrd. All of these are the same public series.
GPR_URLS = (
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.csv",
    "https://www.policyuncertainty.com/media/data_gpr_daily_recent.csv",
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
    "https://www.policyuncertainty.com/media/data_gpr_daily_recent.xls",
)

RSS_FEEDS = (
    {"id": "bbc_world", "name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"id": "guardian_world", "name": "The Guardian World", "url": "https://www.theguardian.com/world/rss"},
    {"id": "npr_world", "name": "NPR World", "url": "https://feeds.npr.org/1004/rss.xml"},
    {"id": "aljazeera", "name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml"},
    {"id": "who_news", "name": "WHO news", "url": "https://www.who.int/rss-feeds/news-english.xml"},
)

OFAC_URLS = (
    "https://ofac.treasury.gov/recent-actions/rss.xml",
    "https://ofac.treasury.gov/recent-actions",
)

CENTRAL_BANK_FEEDS = (
    {"id": "fed_press", "name": "Federal Reserve", "url": "https://www.federalreserve.gov/feeds/press_all.xml"},
    {"id": "fed_monetary", "name": "Federal Reserve monetary", "url": "https://www.federalreserve.gov/feeds/press_monetary.xml"},
    {"id": "ecb_press", "name": "ECB", "url": "https://www.ecb.europa.eu/rss/press.html"},
    {"id": "boe_news", "name": "Bank of England", "url": "https://www.bankofengland.co.uk/rss/news"},
    {"id": "boj_news", "name": "Bank of Japan", "url": "https://www.boj.or.jp/en/rss/whatsnew.xml"},
)


def _now():
    return datetime.now(timezone.utc)


def iso(dt=None):
    dt = dt or _now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def cache_dir(root=None):
    path = Path(root) if root else CACHE_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_opener(url, timeout=TIMEOUT, headers=None):
    hdrs = {"User-Agent": UA, "Accept": "application/json, application/xml, text/xml, text/csv, text/html;q=0.9, */*;q=0.8"}
    if headers:
        hdrs.update(headers)
    req = Request(url, headers=hdrs)
    with urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("Content-Type", ""), str(resp.status if hasattr(resp, "status") else 200)


def _decode(body):
    if body is None:
        return ""
    if isinstance(body, str):
        return body
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return body.decode(enc)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def fetch_url(url, opener=None, pause=PAUSE_SEC, timeout=TIMEOUT):
    opener = opener or default_opener
    if pause:
        time.sleep(pause)
    body, ctype, status = opener(url, timeout=timeout)
    return _decode(body), ctype or "", str(status)


def _cache_get(path, ttl=CACHE_TTL_SEC):
    path = Path(path)
    if not path.is_file():
        return None
    age = time.time() - path.stat().st_mtime
    if ttl and age > ttl:
        return None
    try:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return path.read_text(encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _cache_put(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, (dict, list)):
        data = json.dumps(payload, indent=2, default=str).encode("utf-8")
    else:
        data = str(payload).encode("utf-8")
    write_atomically(path, data)


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------
def _gdelt_url(mode, timespan, extra=None):
    from urllib.parse import urlencode
    q = {"query": GDELT_QUERY, "mode": mode, "format": "json", "timespan": timespan}
    if extra:
        q.update(extra)
    return GDELT_DOC + "?" + urlencode(q)


def parse_gdelt_artlist(payload):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return []
    arts = payload.get("articles") or payload.get("Articles") or []
    out = []
    for a in arts:
        title = (a.get("title") or a.get("Title") or "").strip()
        if not title:
            continue
        out.append({
            "source": "gdelt",
            "feed": "gdelt_doc",
            "title": title,
            "url": a.get("url") or a.get("urlMobile") or a.get("URL") or "",
            "seendate": a.get("seendate") or a.get("seenDate") or "",
            "domain": a.get("domain") or a.get("sourceCountry") or "",
            "language": a.get("language") or "English",
            "tone": _num(a.get("tone") or a.get("Tone")),
        })
    return out


def parse_gdelt_timeline(payload):
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return []
    series = payload.get("timeline") or payload.get("data") or []
    # DOC TimelineVol: [{date, value}] or [{series, data: [...]}]
    if series and isinstance(series, list) and isinstance(series[0], dict) and "data" in series[0]:
        series = series[0]["data"]
    rows = []
    for row in series or []:
        if not isinstance(row, dict):
            continue
        date = str(row.get("date") or row.get("datetime") or row.get("bin") or "")[:14]
        val = _num(row.get("value") or row.get("count") or row.get("volume"))
        if date and val is not None:
            rows.append({"date": date, "value": val})
    return rows


def gdelt_spike(timeline, short=7, long=30, z_watch=2.0, z_shock=3.0):
    """Causal spike: last `short` bins vs the preceding `long` bins (not including the short window)."""
    vals = [r["value"] for r in timeline if r.get("value") is not None]
    if len(vals) < max(short + 5, 12):
        return {"ok": False, "reason": "not enough GDELT timeline history", "z": None, "spike": False}
    short_mean = sum(vals[-short:]) / float(short)
    base = vals[-(short + long):-short] if len(vals) >= short + long else vals[:-short]
    if len(base) < 5:
        return {"ok": False, "reason": "not enough baseline", "z": None, "spike": False}
    mu = sum(base) / len(base)
    var = sum((x - mu) ** 2 for x in base) / max(len(base) - 1, 1)
    sd = var ** 0.5
    z = (short_mean - mu) / sd if sd > 1e-9 else 0.0
    return {
        "ok": True,
        "shortMean": short_mean,
        "baselineMean": mu,
        "baselineSd": sd,
        "z": z,
        "spike": z >= z_watch,
        "shockSpike": z >= z_shock,
        "last": vals[-1],
        "n": len(vals),
    }


def fetch_gdelt(opener=None, root=None, use_cache=True):
    root = cache_dir(root)
    cached = _cache_get(root / "gdelt_cache.json") if use_cache else None
    if cached and cached.get("articles") is not None:
        return cached
    status = {"source": "gdelt", "ok": False, "error": None}
    articles, timeline, raw_errors = [], [], []
    try:
        body, _, _ = fetch_url(_gdelt_url("ArtList", "1d", {"maxrecords": 75, "sort": "datedesc"}), opener=opener)
        articles = parse_gdelt_artlist(body)
    except (URLError, HTTPError, TimeoutError, OSError, ValueError) as exc:
        raw_errors.append(f"artlist: {exc}")
    try:
        body, _, _ = fetch_url(_gdelt_url("TimelineVol", "1month"), opener=opener)
        timeline = parse_gdelt_timeline(body)
    except (URLError, HTTPError, TimeoutError, OSError, ValueError) as exc:
        raw_errors.append(f"timeline: {exc}")
    spike = gdelt_spike(timeline) if timeline else {"ok": False, "reason": "no timeline", "z": None, "spike": False}
    status.update({
        "ok": bool(articles or timeline),
        "error": "; ".join(raw_errors) if raw_errors else None,
        "nArticles": len(articles),
        "nTimeline": len(timeline),
        "spike": spike,
        "fetchedAt": iso(),
    })
    doc = {"articles": articles, "timeline": timeline, "spike": spike, "status": status, "fetchedAt": iso()}
    _cache_put(root / "gdelt_cache.json", doc)
    return doc


# ---------------------------------------------------------------------------
# GPR
# ---------------------------------------------------------------------------
def _parse_gpr_csv(text):
    sample = text.lstrip()
    if not sample or sample[0] in "<{" or "html" in sample[:40].lower():
        return []
    # Some exports start with a title row before the header.
    lines = sample.splitlines()
    header_i = 0
    for i, line in enumerate(lines[:8]):
        low = line.lower()
        if "date" in low and ("gpr" in low or "daily" in low):
            header_i = i
            break
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_i:])))
    rows = []
    for row in reader:
        keys = {k.lower().strip(): k for k in row if k}
        date_k = keys.get("date") or keys.get("day") or next((k for k in keys if "date" in k), None)
        # Prefer the daily GPR index (not the actual-events-only or threat-only splits unless that's all we have).
        gpr_k = None
        for cand in ("gprd", "gpr", "gpr_daily", "gprdaily", "daily_gpr"):
            if cand in keys:
                gpr_k = keys[cand]
                break
        if gpr_k is None:
            for k, orig in keys.items():
                if k.startswith("gpr") and "act" not in k and "threat" not in k:
                    gpr_k = orig
                    break
        if not date_k or not gpr_k:
            continue
        raw_date = str(row.get(date_k) or "").strip()
        day = _parse_date(raw_date)
        val = _num(row.get(gpr_k))
        if day and val is not None:
            rows.append({"date": day, "gpr": val})
    rows.sort(key=lambda r: r["date"])
    return rows


def _parse_date(raw):
    raw = str(raw).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw[:10] if fmt == "%Y-%m-%d" else raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Excel serial dates sometimes appear in exports.
    try:
        serial = float(raw)
        if 20000 < serial < 80000:
            from datetime import date, timedelta
            day = date(1899, 12, 30) + timedelta(days=int(serial))
            return day.isoformat()
    except ValueError:
        return None
    return None


def gpr_features(rows, z_window=252):
    if not rows:
        return {"ok": False, "reason": "no GPR rows"}
    vals = [r["gpr"] for r in rows]
    last = rows[-1]
    window = vals[-z_window:] if len(vals) >= 20 else vals
    mu = sum(window) / len(window)
    var = sum((x - mu) ** 2 for x in window) / max(len(window) - 1, 1)
    sd = var ** 0.5
    z = (last["gpr"] - mu) / sd if sd > 1e-9 else 0.0
    # percentile of last vs window
    pct = sum(1 for x in window if x <= last["gpr"]) / len(window)
    chg5 = last["gpr"] - vals[-6] if len(vals) > 5 else None
    return {
        "ok": True,
        "date": last["date"],
        "gpr": last["gpr"],
        "z": z,
        "percentile": pct,
        "mean": mu,
        "sd": sd,
        "chg5": chg5,
        "n": len(rows),
        "elevated": z >= 2.0 or pct >= 0.95,
        "extreme": z >= 3.0 or pct >= 0.99,
    }


def fetch_gpr(opener=None, root=None, use_cache=True):
    root = cache_dir(root)
    csv_path = root / "gpr_daily.csv"
    cached = _cache_get(csv_path, ttl=6 * 3600) if use_cache else None
    rows, used, errors = [], None, []
    if cached:
        rows = _parse_gpr_csv(cached)
        used = "cache"
    if not rows:
        for url in GPR_URLS:
            try:
                body, ctype, _ = fetch_url(url, opener=opener)
            except (URLError, HTTPError, TimeoutError, OSError, ValueError) as exc:
                errors.append(f"{url.split('/')[-1]}: {exc}")
                continue
            parsed = _parse_gpr_csv(body)
            if parsed:
                rows, used = parsed, url
                _cache_put(csv_path, "date,gpr\n" + "\n".join(f"{r['date']},{r['gpr']}" for r in rows) + "\n")
                break
            errors.append(f"{url.split('/')[-1]}: not a parseable CSV (content-type {ctype})")
    feat = gpr_features(rows) if rows else {"ok": False, "reason": "could not parse any GPR source"}
    return {
        "rows": rows[-10:],
        "n": len(rows),
        "features": feat,
        "source": used,
        "errors": errors,
        "fetchedAt": iso(),
        "ok": bool(rows),
    }


# ---------------------------------------------------------------------------
# RSS / OFAC / central banks
# ---------------------------------------------------------------------------
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def _child(el, *tags):
    """Find the first child by local tag or explicit Clark notation. Never bool()-test Elements."""
    atom = "{http://www.w3.org/2005/Atom}"
    for tag in tags:
        for cand in (tag, atom + tag if not tag.startswith("{") else tag):
            node = el.find(cand)
            if node is not None:
                return node
    return None


def _text(el):
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _item_date(el):
    for tag in ("pubDate", "published", "updated", "date"):
        node = _child(el, tag)
        raw = _text(node)
        if not raw:
            continue
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return iso(dt)
        except (TypeError, ValueError, OverflowError):
            pass
        day = _parse_date(raw[:10])
        if day:
            return day + "T00:00:00+00:00"
    return None


def parse_feed(text, feed_id, feed_name, source="rss"):
    text = (text or "").strip()
    if not text:
        return []
    # OFAC page sometimes returns HTML; pull obvious headlines.
    if text.lstrip().lower().startswith("<html") or "<!doctype html" in text[:200].lower():
        return _parse_ofac_html(text, feed_id, feed_name)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    items = []
    # RSS 2.0
    for item in root.findall(".//item"):
        title = _text(_child(item, "title"))
        if not title:
            continue
        link = _text(_child(item, "link")) or _text(_child(item, "guid"))
        summary = _text(_child(item, "description", "summary"))
        items.append({
            "source": source, "feed": feed_id, "feedName": feed_name,
            "title": re.sub(r"\s+", " ", title), "url": link,
            "summary": re.sub(r"<[^>]+>", " ", summary or "")[:400].strip(),
            "published": _item_date(item),
        })
    # Atom
    if not items:
        seen = set()
        entries = list(root.findall(".//{http://www.w3.org/2005/Atom}entry")) + list(root.findall(".//entry"))
        for item in entries:
            if id(item) in seen:
                continue
            seen.add(id(item))
            title = _text(_child(item, "title"))
            if not title:
                continue
            link_el = _child(item, "link")
            href = ""
            if link_el is not None:
                href = link_el.get("href") or _text(link_el)
            summary = _text(_child(item, "summary", "content"))
            items.append({
                "source": source, "feed": feed_id, "feedName": feed_name,
                "title": re.sub(r"\s+", " ", title), "url": href,
                "summary": re.sub(r"<[^>]+>", " ", summary or "")[:400].strip(),
                "published": _item_date(item),
            })
    return items


def _parse_ofac_html(text, feed_id, feed_name):
    items = []
    # Recent-actions index: links that look like dated actions.
    for href, title in re.findall(r'href="([^"]+)"[^>]*>([^<]{8,180})</a>', text, re.IGNORECASE):
        clean = re.sub(r"\s+", " ", title).strip()
        if not clean or "skip to" in clean.lower() or len(clean) < 12:
            continue
        if re.search(r"(sign up|e-mail updates|email alerts|contact ofac|appeal an ofac|dealing with|skip to|javascript)",
                     clean, re.IGNORECASE):
            continue
        if re.fullmatch(r"Sanctions List Updates", clean, re.IGNORECASE):
            continue
        if re.search(r"list of country[- ]related sanctions|country[- ]related sanctions\b", clean, re.IGNORECASE):
            continue
        if not re.search(r"(sanction|ofac|designat|general license|determination|ukraine|iran|russia|terror| SDN)",
                         clean, re.IGNORECASE):
            # Keep dated bulletin-style headlines even without those words.
            if not re.search(r"20\d{2}", clean):
                continue
        # Prefer dated action posts; skip bare section titles.
        if len(clean) < 24 and not re.search(r"20\d{2}", clean):
            continue
        if not href.startswith("http"):
            href = "https://ofac.treasury.gov" + (href if href.startswith("/") else "/" + href)
        items.append({
            "source": "ofac", "feed": feed_id, "feedName": feed_name,
            "title": clean, "url": href, "summary": "", "published": None,
        })
        if len(items) >= 25:
            break
    return items


def fetch_feed_list(feeds, opener=None, root=None, use_cache=True, source="rss"):
    root = cache_dir(root)
    cache_path = root / f"feeds_{source}.json"
    cached = _cache_get(cache_path) if use_cache else None
    if cached and cached.get("items") is not None:
        return cached
    items, statuses = [], []
    for feed in feeds:
        st = {"id": feed["id"], "name": feed["name"], "ok": False, "error": None, "n": 0, "url": feed["url"]}
        try:
            body, _, _ = fetch_url(feed["url"], opener=opener)
            parsed = parse_feed(body, feed["id"], feed["name"], source=source)
            items.extend(parsed)
            st["ok"] = True
            st["n"] = len(parsed)
        except (URLError, HTTPError, TimeoutError, OSError, ValueError, ET.ParseError) as exc:
            st["error"] = str(exc)
        statuses.append(st)
    doc = {"items": items, "feeds": statuses, "fetchedAt": iso(), "ok": any(s["ok"] for s in statuses)}
    _cache_put(cache_path, doc)
    return doc


def fetch_rss(opener=None, root=None, use_cache=True):
    return fetch_feed_list(RSS_FEEDS, opener=opener, root=root, use_cache=use_cache, source="rss")


def fetch_central_banks(opener=None, root=None, use_cache=True):
    return fetch_feed_list(CENTRAL_BANK_FEEDS, opener=opener, root=root, use_cache=use_cache, source="central_bank")


def fetch_ofac(opener=None, root=None, use_cache=True):
    root = cache_dir(root)
    cache_path = root / "feeds_ofac.json"
    cached = _cache_get(cache_path) if use_cache else None
    if cached and cached.get("items") is not None:
        return cached
    items, errors, used = [], [], None
    for url in OFAC_URLS:
        try:
            body, _, _ = fetch_url(url, opener=opener)
            parsed = parse_feed(body, "ofac", "OFAC recent actions", source="ofac")
            if parsed:
                items, used = parsed, url
                break
            errors.append(f"{url}: no items parsed")
        except (URLError, HTTPError, TimeoutError, OSError, ValueError, ET.ParseError) as exc:
            errors.append(f"{url}: {exc}")
    doc = {"items": items, "source": used, "errors": errors, "fetchedAt": iso(), "ok": bool(items)}
    _cache_put(cache_path, doc)
    return doc


def collect_headlines(opener=None, root=None, use_cache=True):
    """Pull every feed and return a unified headline list plus per-source status."""
    gdelt = fetch_gdelt(opener=opener, root=root, use_cache=use_cache)
    gpr = fetch_gpr(opener=opener, root=root, use_cache=use_cache)
    rss = fetch_rss(opener=opener, root=root, use_cache=use_cache)
    banks = fetch_central_banks(opener=opener, root=root, use_cache=use_cache)
    ofac = fetch_ofac(opener=opener, root=root, use_cache=use_cache)
    headlines = []
    for art in gdelt.get("articles") or []:
        headlines.append({**art, "summary": art.get("summary") or ""})
    for bundle in (rss, banks, ofac):
        headlines.extend(bundle.get("items") or [])
    # Dedup by title (casefold).
    seen, uniq = set(), []
    for h in headlines:
        key = re.sub(r"\s+", " ", (h.get("title") or "").strip().casefold())
        if not key or key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    status = {
        "gdelt": (gdelt.get("status") or {"ok": gdelt.get("ok")}),
        "gpr": {"ok": gpr.get("ok"), "source": gpr.get("source"), "errors": gpr.get("errors"), "features": gpr.get("features")},
        "rss": rss.get("feeds"),
        "centralBanks": banks.get("feeds"),
        "ofac": {"ok": ofac.get("ok"), "n": len(ofac.get("items") or []), "errors": ofac.get("errors"), "source": ofac.get("source")},
        "nHeadlines": len(uniq),
        "fetchedAt": iso(),
    }
    return {
        "headlines": uniq,
        "gdeltSpike": gdelt.get("spike") or {},
        "gpr": gpr.get("features") or {},
        "status": status,
    }


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
