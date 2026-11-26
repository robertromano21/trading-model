"""Surprise scoring: actual vs consensus, standardized by history.







Consensus, in order of preference:



  1. A manual entry in Massive/Macro/consensus.json (or the calendar UI).



  2. A Massive/Polygon economics forecast, when the plan has that endpoint.



  3. The Cleveland Fed inflation nowcast (CPI / core PCE), when it can be parsed.



  4. A model expectation: last official print plus the trailing change (a simple nowcast).







FOMC surprise is the announced mid-target minus the prior mid-target, and minus the



implied rate from fed-funds futures (Massive) or a prior-day short-rate proxy (DGS1MO / DFF).







Treasury auction surprise is the tail (high yield minus when-issued, or the prior close of



the matching FRED par yield if WI is not free), plus bid-to-cover and takedown vs history.



"""



from __future__ import annotations







import json



import math



from datetime import date, datetime, timedelta



from pathlib import Path



from zoneinfo import ZoneInfo







from fetch_and_model import DATA_DIR, write_atomically



from macro_calendar import CALENDAR_PATH, CONSENSUS_PATH, event_by_id, load_calendar, save_calendar







ET = ZoneInfo("America/New_York")



MACRO_DIR = DATA_DIR / "Macro"



SURPRISE_PATH = MACRO_DIR / "surprises.json"



HISTORY_PATH = MACRO_DIR / "surprise_history.json"







# Minimum history before we treat a z-score as standardized (otherwise z is null, raw surprise kept).



MIN_HISTORY = 8











def _read(path, default):



    path = Path(path)



    if not path.is_file():



        return default



    return json.loads(path.read_text(encoding="utf-8"))











def _write(path, doc):



    write_atomically(Path(path), json.dumps(doc, indent=2, default=str).encode("utf-8"))











def load_consensus(path=None):



    path = Path(path or CONSENSUS_PATH)



    doc = _read(path, default={"entries": []})



    if isinstance(doc, list):



        doc = {"entries": doc}



    doc.setdefault("entries", [])



    return doc











def save_consensus_entry(event_id, period, value, source="manual", field="headline", notes=None, path=None):


    path = Path(path or CONSENSUS_PATH)



    """Add or replace a manual (or other) consensus value. Never used for live orders."""



    doc = load_consensus(path)



    key = f"{event_id}:{period}:{field}"



    entries = [e for e in doc["entries"] if f"{e.get('eventId')}:{e.get('period')}:{e.get('field') or 'headline'}" != key]



    entries.append({



        "eventId": event_id, "period": str(period), "field": field, "value": _num(value),



        "source": source, "notes": notes,



        "enteredAt": datetime.now(ET).isoformat(timespec="seconds"),



        "key": key,



    })



    doc["entries"] = entries



    doc["updatedAt"] = datetime.now(ET).isoformat(timespec="seconds")



    _write(path, doc)



    return entries[-1]











def _num(value):



    if value is None or value == "":



        return None



    if isinstance(value, (int, float)) and math.isfinite(value):



        return float(value)



    try:



        return float(str(value).replace(",", "").replace("%", "").strip())



    except (TypeError, ValueError):



        return None











def lookup_consensus(event_id, period, field="headline", path=None):

    path = Path(path or CONSENSUS_PATH)



    doc = load_consensus(path)



    wanted = []



    for e in doc["entries"]:



        if e.get("eventId") != event_id:



            continue



        if period and str(e.get("period")) not in {str(period), str(period)[:7], str(period)[:10]}:



            # Allow a date-key to match a month period and vice versa.



            if str(e.get("period"))[:7] != str(period)[:7]:



                continue



        if (e.get("field") or "headline") != field and field == "headline" and e.get("field") not in {None, "headline", "value"}:



            continue



        wanted.append(e)



    if not wanted:



        return None



    wanted.sort(key=lambda e: e.get("enteredAt") or "")



    return wanted[-1]











def model_expectation(history, method="last_plus_avg_change", window=6):



    """history: list of official prints, oldest first (numbers)."""



    vals = [v for v in history if v is not None and math.isfinite(v)]



    if not vals:



        return None, "no_history"



    if method == "last" or len(vals) == 1:



        return float(vals[-1]), "last_print"



    changes = [vals[i] - vals[i - 1] for i in range(1, len(vals))]



    avg = sum(changes[-window:]) / max(len(changes[-window:]), 1)



    return float(vals[-1] + avg), "last_plus_avg_change"











def stdev(values):



    vals = [float(v) for v in values if v is not None and math.isfinite(v)]



    if len(vals) < 2:



        return None



    mean = sum(vals) / len(vals)



    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)



    return math.sqrt(var) if var > 0 else 0.0











def standardize(surprise, history_surprises, min_n=MIN_HISTORY):



    hist = [s for s in history_surprises if s is not None and math.isfinite(s)]



    sigma = stdev(hist)



    if sigma is None or sigma == 0 or len(hist) < min_n:



        return None, sigma, len(hist)



    return surprise / sigma, sigma, len(hist)











def resolve_expectation(event_id, period, history=None, field="headline", extras=None):



    """Pick the best available expected value and label its source."""



    extras = extras or {}



    manual = lookup_consensus(event_id, period, field=field)



    if manual and manual.get("value") is not None:



        return float(manual["value"]), f"manual:{manual.get('source') or 'manual'}"



    if extras.get("massive") is not None:



        return float(extras["massive"]), "massive_economics"



    if extras.get("nowcast") is not None and event_id in {"cpi", "core_cpi", "pce", "core_pce"}:



        return float(extras["nowcast"]), extras.get("nowcastSource") or "cleveland_nowcast"



    if extras.get("implied") is not None and event_id == "fomc_decision":



        return float(extras["implied"]), extras.get("impliedSource") or "implied_path"



    model, why = model_expectation(history or [])



    if model is not None:



        return model, why



    return None, "unavailable"











def score_print(event_id, period, actual, history_actuals=None, history_surprises=None,



                field="headline", extras=None, consensus=None):



    """Return a surprise record. actual / consensus are in native units (%, count, yield, ...)."""



    event = event_by_id(event_id) or {"id": event_id, "hawkish_if": "above", "name": event_id}



    actual = _num(actual)



    if consensus is not None:



        expected, source = _num(consensus), "provided"



    else:



        expected, source = resolve_expectation(event_id, period, history=history_actuals,



                                               field=field, extras=extras)



    surprise = None if actual is None or expected is None else actual - expected



    # Direction: for hawkish_if=below (e.g. unemployment, claims), a higher print is dovish,



    # so the signed "hawkish surprise" flips.



    hawkish = event.get("hawkish_if") or "above"



    signed = surprise



    if signed is not None:



        if hawkish == "below":



            signed = -signed



        elif hawkish in {"none", "mixed", "auction"}:



            pass



    z, sigma, n = (None, None, len(history_surprises or []))



    if surprise is not None:



        z, sigma, n = standardize(signed if hawkish in {"above", "below", "tighter"} else surprise,



                                  history_surprises or [])



    return {



        "eventId": event_id,



        "name": event.get("name"),



        "period": period,



        "field": field,



        "actual": actual,



        "consensus": expected,



        "consensusSource": source,



        "surprise": surprise,



        "hawkishSurprise": signed,



        "hawkishIf": hawkish,



        "surpriseZ": z,



        "sigma": sigma,



        "historyN": n,



        "scoredAt": datetime.now(ET).isoformat(timespec="seconds"),



    }











# ---- FOMC implied path -------------------------------------------------------







def fomc_implied_rate(asof, opener=None):



    """Best-effort implied funds rate (percent) ahead of the decision.







    Preference: Massive/Polygon fed-funds futures mid. Fallback: prior-day DFF or DGS1MO



    from FRED (a coarse proxy, labelled as such — not a true FedWatch probability).



    """



    notes = []



    # Massive futures symbols vary by plan (FF, ZQ, or a rates snapshot). Fail soft.



    try:



        from macro_sources import env_value, http_get, _strip_query



        key = env_value("MASSIVE_API_KEY")



        if key:



            # A generic snapshot; if it 404s we just fall through.



            url = "https://api.massive.com/v3/snapshot?ticker.any_of=FF,ZQ"



            raw, _, _ = http_get(url, opener=opener, headers={"Authorization": f"Bearer {key}"},



                                 accept="application/json")



            doc = json.loads(raw.decode("utf-8"))



            # We do not assume a schema; look for a mid/last that looks like a rate.



            last = _dig_rate(doc)



            if last is not None:



                return last, "massive_ff_futures", notes



            notes.append("Massive snapshot had no usable fed-funds future mid.")



        else:



            notes.append("No Massive key; skipping fed-funds futures.")



    except Exception as error:  # noqa: BLE001



        notes.append(f"fed-funds futures unavailable: {error}")



    try:



        from macro_sources import download_fred



        for series, label in (("DFF", "fred_dff"), ("DGS1MO", "fred_dgs1mo")):



            frame = download_fred(series, end=str(asof)[:10], opener=opener)



            col = series if series in frame.columns else frame.columns[0]



            s = frame[col].dropna()



            if len(s):



                notes.append(f"Using {label} as a coarse implied-rate proxy, not futures-implied probabilities.")



                return float(s.iloc[-1]), label, notes



    except Exception as error:  # noqa: BLE001



        notes.append(f"short-rate proxy unavailable: {error}")



    return None, "unavailable", notes











def _dig_rate(doc):



    """Walk a JSON-ish object for a plausible interest-rate mid (0–20)."""



    stack = [doc]



    while stack:



        cur = stack.pop()



        if isinstance(cur, dict):



            for k, v in cur.items():



                if k.lower() in {"mid", "last", "close", "price", "fair", "impliedrate"} and isinstance(v, (int, float)):



                    # FF futures are sometimes quoted as 100 - rate.



                    if 80 <= v <= 100:



                        return 100.0 - float(v)



                    if 0 <= v <= 20:



                        return float(v)



                else:



                    stack.append(v)



        elif isinstance(cur, list):



            stack.extend(cur[:20])



    return None











def score_fomc(actual_mid, prior_mid, asof, opener=None, history_surprises=None):



    change_bp = None if actual_mid is None or prior_mid is None else (actual_mid - prior_mid) * 100



    implied, implied_source, notes = fomc_implied_rate(asof, opener=opener)



    vs_implied = None if actual_mid is None or implied is None else (actual_mid - implied) * 100



    # Standardize the implied surprise (basis points) when we have history.



    z, sigma, n = (None, None, 0)



    if vs_implied is not None:



        z, sigma, n = standardize(vs_implied, history_surprises or [])



    elif change_bp is not None:



        z, sigma, n = standardize(change_bp, history_surprises or [])



    return {



        "eventId": "fomc_decision",



        "actualMid": _num(actual_mid),



        "priorMid": _num(prior_mid),



        "changeBp": change_bp,



        "impliedRate": implied,



        "impliedSource": implied_source,



        "surpriseVsImpliedBp": vs_implied,



        "surpriseZ": z,



        "sigma": sigma,



        "historyN": n,



        "notes": notes,



        "scoredAt": datetime.now(ET).isoformat(timespec="seconds"),



    }











# ---- Treasury auction scoring ------------------------------------------------







def _pct(part, whole):



    part, whole = _num(part), _num(whole)



    if part is None or not whole:



        return None



    # FiscalData sometimes already stores percents (0–100), sometimes notional dollars.



    if 0 <= part <= 100 and whole > 100:



        # part is notional, whole is notional



        return 100.0 * part / whole



    if 0 <= part <= 1 and whole <= 1.0001:



        return 100.0 * part



    if 0 <= part <= 100 and whole <= 100:



        return part



    return 100.0 * part / whole











def score_auction(result, prior_yield=None, when_issued=None, history=None):



    """Tail, bid-to-cover and takedown surprises for one auction result dict."""



    high = _num(result.get("high_yield") or result.get("high_investment_rate") or result.get("high_discount_rate"))



    wi = _num(when_issued if when_issued is not None else result.get("when_issued") or result.get("wi_yield"))



    proxy = _num(prior_yield)



    benchmark, bench_src = (wi, "when_issued") if wi is not None else (proxy, "prior_close_yield_proxy")



    tail_bp = None if high is None or benchmark is None else (high - benchmark) * 100



    btc = _num(result.get("bid_to_cover_ratio") or result.get("bid_to_cover"))



    total = result.get("total_accepted") or result.get("offering_amt")



    indirect = _pct(result.get("indirect_bidder_accepted"), total)



    direct = _pct(result.get("direct_bidder_accepted"), total)



    dealer = _pct(result.get("primary_dealer_accepted"), total)



    history = history or {}







    def vs(name, value):



        if value is None:



            return None, None, 0



        hist = history.get(name) or []



        z, sigma, n = standardize(value - (sum(hist) / len(hist) if hist else 0), 



                                  [h - (sum(hist) / len(hist)) for h in hist] if len(hist) >= MIN_HISTORY else [])



        # If we lack a surprise history, still report the raw value vs the mean when possible.



        mean = (sum(hist) / len(hist)) if hist else None



        raw = None if mean is None else value - mean



        if z is None and raw is not None and len(hist) >= MIN_HISTORY:



            z, sigma, n = standardize(raw, [x - mean for x in hist])



        return raw, z, n if hist else 0







    btc_s, btc_z, btc_n = vs("bid_to_cover", btc)



    tail_s, tail_z, tail_n = vs("tail_bp", tail_bp)



    ind_s, ind_z, _ = vs("indirect_pct", indirect)



    return {



        "eventId": "treasury_auction",



        "auctionDate": result.get("auction_date"),



        "securityTerm": result.get("security_term") or result.get("securityTerm"),



        "securityType": result.get("security_type") or result.get("securityType"),



        "highYield": high,



        "whenIssued": wi,



        "priorCloseYield": proxy,



        "benchmark": benchmark,



        "benchmarkSource": bench_src if benchmark is not None else "unavailable",



        "tailBp": tail_bp,



        "tailSurprise": tail_s,



        "tailZ": tail_z,



        "bidToCover": btc,



        "bidToCoverSurprise": btc_s,



        "bidToCoverZ": btc_z,



        "indirectPct": indirect,



        "directPct": direct,



        "dealerPct": dealer,



        "indirectSurprise": ind_s,



        "indirectZ": ind_z,



        "historyN": max(btc_n, tail_n),



        "notes": ([] if wi is not None else



                  ["When-issued yield was not in the free FiscalData feed; used the prior close of the matching FRED par yield as a proxy."



                   if proxy is not None else



                   "No when-issued or prior-close yield proxy was available; tail is missing."]),



        "scoredAt": datetime.now(ET).isoformat(timespec="seconds"),



    }











# ---- Persist onto the calendar + history -------------------------------------







def record_surprise(score, calendar_key=None, path=None, calendar_path=None):
    path = Path(path or SURPRISE_PATH)
    calendar_path = Path(calendar_path or CALENDAR_PATH)



    """Append to surprises.json and, if calendar_key is set, stamp the calendar row."""



    doc = _read(path, default={"items": []})



    items = doc.get("items") or []



    items.append(dict(score, calendarKey=calendar_key))



    doc["items"] = items[-5000:]



    doc["updatedAt"] = datetime.now(ET).isoformat(timespec="seconds")



    _write(path, doc)



    if calendar_key:



        cal = load_calendar(calendar_path)



        for row in cal.get("events") or []:



            if row.get("key") == calendar_key:



                row["actual"] = score.get("actual", score.get("actualMid", score.get("highYield")))



                row["consensus"] = score.get("consensus", score.get("impliedRate", score.get("benchmark")))



                row["consensusSource"] = score.get("consensusSource", score.get("impliedSource", score.get("benchmarkSource")))



                row["surprise"] = score.get("surprise", score.get("surpriseVsImpliedBp", score.get("tailBp")))



                row["surpriseZ"] = score.get("surpriseZ", score.get("tailZ"))



                row["status"] = "printed"



                row["printedAtEt"] = score.get("scoredAt")



        save_calendar(cal, calendar_path)



    # history used for later z-scores



    hist = _read(HISTORY_PATH, default={})



    event_id = score.get("eventId")



    bucket = hist.setdefault(event_id, [])



    value = score.get("hawkishSurprise", score.get("surprise", score.get("surpriseVsImpliedBp", score.get("tailBp"))))



    if value is not None:



        bucket.append({"period": score.get("period") or score.get("auctionDate"), "surprise": value,



                       "at": score.get("scoredAt")})



        hist[event_id] = bucket[-240:]



        _write(HISTORY_PATH, hist)



    return score











def history_of(event_id, path=None):
    path = Path(path or HISTORY_PATH)



    doc = _read(path, default={})



    return [x.get("surprise") for x in (doc.get(event_id) or []) if x.get("surprise") is not None]











def latest_surprises(n=12, path=None):
    path = Path(path or SURPRISE_PATH)



    items = _read(path, default={"items": []}).get("items") or []



    return list(reversed(items[-n:]))



