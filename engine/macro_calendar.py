"""Economic calendar: official release dates and times in America/New_York (ET).







Covers US labor, inflation, activity, sentiment, FOMC, Treasury auctions, and the major



overseas releases and central-bank decisions. The on-disk cache lives under



Massive/Macro/ (gitignored). Official schedules are fetched when possible; otherwise a



deterministic fallback (weekday rules, published FOMC dates, or manual entries) is used



and labelled as such.







  python macro_calendar.py build



  python macro_calendar.py upcoming



  python macro_calendar.py show --days 14



"""



from __future__ import annotations







import argparse



import json



import sys



from datetime import date, datetime, timedelta, time



from pathlib import Path



from zoneinfo import ZoneInfo







from fetch_and_model import DATA_DIR, write_atomically







ET = ZoneInfo("America/New_York")



UTC = ZoneInfo("Utc") if False else ZoneInfo("UTC")







MACRO_DIR = DATA_DIR / "Macro"



CALENDAR_PATH = MACRO_DIR / "calendar.json"



CONSENSUS_PATH = MACRO_DIR / "consensus.json"



MANUAL_PATH = MACRO_DIR / "manual_releases.json"







# Typical local release times. These are converted to ET on the release date so



# US/Europe DST mismatches are handled correctly.



#



# entry: official | official_or_manual | manual



# hawkish_if: above (hotter-than-expected is hawkish) | below | tighter (policy)



#             | mixed | auction | none







EVENTS = [



    # ---- BLS ----



    dict(id="cpi", name="CPI (all items)", agency="BLS", country="US", category="inflation",



         zone="America/New_York", time="08:30", importance="high", hawkish_if="above",



         source="bls", entry="official", fred="CPIAUCSL", bls="CUSR0000SA0",



         schedule="bls_cpi", fields=("value", "mom", "yoy")),



    dict(id="core_cpi", name="Core CPI (ex food and energy)", agency="BLS", country="US",



         category="inflation", zone="America/New_York", time="08:30", importance="high",



         hawkish_if="above", source="bls", entry="official", fred="CPILFESL",



         bls="CUSR0000SA0L1E", schedule="bls_cpi", fields=("value", "mom", "yoy")),



    dict(id="ppi", name="PPI (final demand)", agency="BLS", country="US", category="inflation",



         zone="America/New_York", time="08:30", importance="high", hawkish_if="above",



         source="bls", entry="official", fred="PPIFIS", bls="WPSFD4", schedule="bls_ppi",



         fields=("value", "mom", "yoy")),



    dict(id="nfp", name="Nonfarm payrolls", agency="BLS", country="US", category="labor",



         zone="America/New_York", time="08:30", importance="high", hawkish_if="above",



         source="bls", entry="official", fred="PAYEMS", bls="CES0000000001",



         schedule="bls_empsit", fields=("value", "change")),



    dict(id="unrate", name="Unemployment rate", agency="BLS", country="US", category="labor",



         zone="America/New_York", time="08:30", importance="high", hawkish_if="below",



         source="bls", entry="official", fred="UNRATE", bls="LNS14000000",



         schedule="bls_empsit", fields=("value",)),



    dict(id="jolts", name="JOLTS job openings", agency="BLS", country="US", category="labor",



         zone="America/New_York", time="10:00", importance="medium", hawkish_if="above",



         source="bls", entry="official", fred="JTSJOL", bls="JTS000000000000000JOL",



         schedule="bls_jolts", fields=("value",)),



    dict(id="eci", name="Employment Cost Index", agency="BLS", country="US", category="labor",



         zone="America/New_York", time="08:30", importance="high", hawkish_if="above",



         source="bls", entry="official", fred="ECIALLCIV", bls="CIS1010000000000I",



         schedule="bls_eci", fields=("value", "qoq")),



    # ---- BEA ----



    dict(id="gdp", name="GDP (advance / second / third)", agency="BEA", country="US",



         category="activity", zone="America/New_York", time="08:30", importance="high",



         hawkish_if="above", source="bea", entry="official", fred="GDP", schedule="bea_gdp",



         fields=("value", "qoq_saar")),



    dict(id="pce", name="PCE price index", agency="BEA", country="US", category="inflation",



         zone="America/New_York", time="08:30", importance="high", hawkish_if="above",



         source="bea", entry="official", fred="PCEPI", schedule="bea_pce",



         fields=("value", "mom", "yoy")),



    dict(id="core_pce", name="Core PCE price index", agency="BEA", country="US",



         category="inflation", zone="America/New_York", time="08:30", importance="high",



         hawkish_if="above", source="bea", entry="official", fred="PCEPILFE",



         schedule="bea_pce", fields=("value", "mom", "yoy")),



    # ---- DOL ----



    dict(id="claims", name="Weekly initial jobless claims", agency="DOL", country="US",



         category="labor", zone="America/New_York", time="08:30", importance="high",



         hawkish_if="below", source="dol", entry="official", fred="ICSA",



         schedule="dol_claims", fields=("value",), notes="Usually Thursday 8:30 ET; delayed after federal holidays."),



    # ---- Fed policy and G.17 ----



    dict(id="fomc_decision", name="FOMC rate decision and statement", agency="Federal Reserve",



         country="US", category="policy", zone="America/New_York", time="14:00", importance="high",



         hawkish_if="tighter", source="fed_rss", entry="official", fred="DFEDTARU",



         schedule="fomc", fields=("upper", "lower", "mid", "change_bp"),



         notes="Statement at 2:00 ET; press conference typically 2:30 ET. Surprise vs the prior target and vs the implied path from fed-funds futures or the prior-day curve."),



    dict(id="fomc_minutes", name="FOMC minutes", agency="Federal Reserve", country="US",



         category="policy", zone="America/New_York", time="14:00", importance="medium",



         hawkish_if="tighter", source="fed_rss", entry="official", schedule="fomc_minutes",



         fields=("released",), notes="Usually 2:00 ET, three weeks after the decision."),



    dict(id="indpro", name="Industrial production", agency="Federal Reserve", country="US",



         category="activity", zone="America/New_York", time="09:15", importance="medium",



         hawkish_if="above", source="fred", entry="official", fred="INDPRO",



         schedule="fed_g17", fields=("value", "mom")),



    dict(id="caputil", name="Capacity utilization", agency="Federal Reserve", country="US",



         category="activity", zone="America/New_York", time="09:15", importance="medium",



         hawkish_if="above", source="fred", entry="official", fred="TCU",



         schedule="fed_g17", fields=("value",)),



    # ---- Census ----



    dict(id="retail_sales", name="Advance retail sales", agency="Census", country="US",



         category="activity", zone="America/New_York", time="08:30", importance="high",



         hawkish_if="above", source="census", entry="official_or_manual", fred="RSAFS",



         schedule="census_retail", fields=("value", "mom")),



    dict(id="durable_goods", name="Durable goods orders", agency="Census", country="US",



         category="activity", zone="America/New_York", time="08:30", importance="medium",



         hawkish_if="above", source="census", entry="official_or_manual", fred="DGORDER",



         schedule="census_durable", fields=("value", "mom", "ex_transport")),



    dict(id="factory_orders", name="Factory orders", agency="Census", country="US",



         category="activity", zone="America/New_York", time="10:00", importance="medium",



         hawkish_if="above", source="census", entry="official_or_manual", fred="AMTMNO",



         schedule="census_factory", fields=("value", "mom")),



    # ---- ISM (headline often paid; FRED NAPM lags) ----



    dict(id="ism_mfg", name="ISM Manufacturing PMI", agency="ISM", country="US",



         category="activity", zone="America/New_York", time="10:00", importance="high",



         hawkish_if="above", source="ism", entry="official_or_manual", fred="NAPM",



         schedule="ism_mfg", fields=("value",),



         notes="Headline is often paywalled. Use a manual entry or a lagged FRED print (NAPM). First business day, 10:00 ET."),



    dict(id="ism_services", name="ISM Services PMI", agency="ISM", country="US",



         category="activity", zone="America/New_York", time="10:00", importance="high",



         hawkish_if="above", source="ism", entry="official_or_manual",



         schedule="ism_services", fields=("value",),



         notes="Headline often paywalled. Third business day, 10:00 ET. Manual entry if no free feed."),



    dict(id="sp_global_flash_us", name="S&P Global US flash PMI (fallback)", agency="S&P Global",



         country="US", category="activity", zone="America/New_York", time="09:45",



         importance="medium", hawkish_if="above", source="spglobal", entry="official_or_manual",



         schedule="sp_flash_us", fields=("mfg", "services"),



         notes="Used only as a free-headline fallback when ISM is unavailable; otherwise mark manual."),



    # ---- Regional Fed surveys ----



    dict(id="empire", name="Empire State Manufacturing Survey", agency="NY Fed", country="US",



         category="activity", zone="America/New_York", time="08:30", importance="medium",



         hawkish_if="above", source="fred", entry="official", fred="GACDISA066MSFRBNY",



         schedule="nyfed_empire", fields=("value",)),



    dict(id="philly", name="Philly Fed Business Outlook", agency="Philadelphia Fed", country="US",



         category="activity", zone="America/New_York", time="08:30", importance="medium",



         hawkish_if="above", source="fred", entry="official", fred="GACDFSA066MSFRBPHI",



         schedule="philly_fed", fields=("value",)),



    dict(id="richmond", name="Richmond Fed manufacturing", agency="Richmond Fed", country="US",



         category="activity", zone="America/New_York", time="10:00", importance="low",



         hawkish_if="above", source="fred", entry="official", fred="RICHMANIND",



         schedule="richmond_fed", fields=("value",)),



    dict(id="kc_fed", name="Kansas City Fed manufacturing", agency="Kansas City Fed", country="US",



         category="activity", zone="America/New_York", time="11:00", importance="low",



         hawkish_if="above", source="fred", entry="official", fred="KCMANUMANF",



         schedule="kc_fed", fields=("value",)),



    dict(id="dallas", name="Dallas Fed manufacturing", agency="Dallas Fed", country="US",



         category="activity", zone="America/New_York", time="10:30", importance="low",



         hawkish_if="above", source="fred", entry="official", fred="GACDFSA066MSFRBDAL",



         schedule="dallas_fed", fields=("value",)),



    # ---- Sentiment ----



    dict(id="umich_prelim", name="University of Michigan consumer sentiment (preliminary)",



         agency="University of Michigan", country="US", category="sentiment",



         zone="America/New_York", time="10:00", importance="medium", hawkish_if="above",



         source="umich", entry="official_or_manual", fred="UMCSENT", schedule="umich_prelim",



         fields=("sentiment", "inflex_1y", "inflex_5y"),



         notes="FRED UMCSENT lags the official print and usually carries the final (not preliminary) figure. Prefer a manual or official headline, and record both sentiment and inflation expectations."),



    dict(id="umich_final", name="University of Michigan consumer sentiment (final)",



         agency="University of Michigan", country="US", category="sentiment",



         zone="America/New_York", time="10:00", importance="medium", hawkish_if="above",



         source="umich", entry="official_or_manual", fred="UMCSENT", schedule="umich_final",



         fields=("sentiment", "inflex_1y", "inflex_5y"),



         notes="FRED UMCSENT lags; final typically end-of-month Friday 10:00 ET."),



    dict(id="conference_board", name="Conference Board Consumer Confidence",



         agency="Conference Board", country="US", category="sentiment",



         zone="America/New_York", time="10:00", importance="medium", hawkish_if="above",



         source="conference_board", entry="manual", schedule="conference_board",



         fields=("value",),



         notes="Headline is usually paywalled. Enter the number manually. Typically last Tuesday of the month, 10:00 ET."),



    # ---- Treasury auctions (rows are generated from FiscalData) ----



    dict(id="treasury_auction", name="Treasury auction", agency="Treasury", country="US",



         category="auction", zone="America/New_York", time="13:00", importance="medium",



         hawkish_if="auction", source="fiscaldata", entry="official", schedule="fiscal_auctions",



         fields=("high_yield", "tail_bp", "bid_to_cover", "indirect_pct", "direct_pct", "dealer_pct"),



         notes="Bills, 2/5/7/10/20/30Y notes and bonds, and TIPS. Tail = high yield minus when-issued (or the prior close of the matching FRED par yield if WI is not free)."),



    # ---- Overseas central banks ----



    dict(id="ecb_decision", name="ECB rate decision", agency="ECB", country="EA",



         category="policy", zone="Europe/Berlin", time="14:15", importance="high",



         hawkish_if="tighter", source="ecb", entry="official_or_manual", schedule="ecb",



         fields=("deposit_rate", "main_refi", "change_bp"),



         notes="Statement typically 14:15 Central European time; press conference 14:45 CET. Times are converted to ET on the day."),



    dict(id="boj_decision", name="Bank of Japan rate decision", agency="BoJ", country="JP",



         category="policy", zone="Asia/Tokyo", time="12:00", importance="high",



         hawkish_if="tighter", source="boj", entry="official_or_manual", schedule="boj",



         fields=("rate", "change_bp"),



         notes="BoJ timing varies with the meeting close; 12:00 JST is a placeholder until the official notice is fetched. Converted to ET (often the prior evening)."),



    dict(id="boe_decision", name="Bank of England rate decision", agency="BoE", country="GB",



         category="policy", zone="Europe/London", time="12:00", importance="high",



         hawkish_if="tighter", source="boe", entry="official_or_manual", schedule="boe",



         fields=("rate", "change_bp")),



    dict(id="pboc_decision", name="PBOC policy rate / LPR", agency="PBOC", country="CN",



         category="policy", zone="Asia/Shanghai", time="09:15", importance="high",



         hawkish_if="tighter", source="pboc", entry="official_or_manual", schedule="pboc",



         fields=("lpr_1y", "lpr_5y", "mlf"),



         notes="Irregular. Official notices preferred; otherwise mark a manual entry. Beijing morning = prior evening ET."),



    dict(id="snb_decision", name="Swiss National Bank rate decision", agency="SNB", country="CH",



         category="policy", zone="Europe/Zurich", time="09:30", importance="medium",



         hawkish_if="tighter", source="snb", entry="official_or_manual", schedule="snb",



         fields=("rate", "change_bp")),



    # ---- Overseas PMIs / surveys ----



    dict(id="ea_pmi_flash", name="Euro area flash PMI (S&P Global)", agency="S&P Global",



         country="EA", category="activity", zone="Europe/Berlin", time="10:00", importance="high",



         hawkish_if="above", source="spglobal", entry="official_or_manual", schedule="ea_pmi_flash",



         fields=("mfg", "services", "composite"),



         notes="Headline may be paywalled; mark manual if no free feed."),



    dict(id="de_ifo", name="German Ifo business climate", agency="Ifo", country="DE",



         category="sentiment", zone="Europe/Berlin", time="10:00", importance="high",



         hawkish_if="above", source="ifo", entry="official_or_manual", fred="BSCICP03DEQ665S",



         schedule="ifo", fields=("climate", "current", "expectations")),



    dict(id="de_zew", name="ZEW economic sentiment (Germany)", agency="ZEW", country="DE",



         category="sentiment", zone="Europe/Berlin", time="11:00", importance="medium",



         hawkish_if="above", source="zew", entry="official_or_manual", fred="ZEW",



         schedule="zew", fields=("value",)),



    dict(id="jp_tankan", name="BoJ Tankan (large manufacturers)", agency="BoJ", country="JP",



         category="sentiment", zone="Asia/Tokyo", time="08:50", importance="high",



         hawkish_if="above", source="boj", entry="official_or_manual", fred="JPNAMTANK",



         schedule="tankan", fields=("diffusion",)),



    dict(id="cn_official_pmi", name="China official NBS manufacturing PMI", agency="NBS",



         country="CN", category="activity", zone="Asia/Shanghai", time="09:30", importance="high",



         hawkish_if="above", source="nbs", entry="official_or_manual", schedule="cn_official_pmi",



         fields=("value",)),



    dict(id="cn_caixin_pmi", name="Caixin China manufacturing PMI", agency="Caixin / S&P Global",



         country="CN", category="activity", zone="Asia/Shanghai", time="09:45", importance="medium",



         hawkish_if="above", source="spglobal", entry="official_or_manual", schedule="cn_caixin_pmi",



         fields=("value",)),



    dict(id="uk_pmi_flash", name="UK flash PMI (S&P Global)", agency="S&P Global", country="GB",



         category="activity", zone="Europe/London", time="09:30", importance="medium",



         hawkish_if="above", source="spglobal", entry="official_or_manual", schedule="uk_pmi_flash",



         fields=("mfg", "services")),



    dict(id="jp_pmi_flash", name="Japan flash PMI (S&P Global)", agency="S&P Global", country="JP",



         category="activity", zone="Asia/Tokyo", time="09:30", importance="medium",



         hawkish_if="above", source="spglobal", entry="official_or_manual", schedule="jp_pmi_flash",



         fields=("mfg", "services")),



    # ---- Overseas activity / prices ----



    dict(id="china_gdp", name="China GDP", agency="NBS", country="CN", category="activity",



         zone="Asia/Shanghai", time="10:00", importance="high", hawkish_if="above",



         source="nbs", entry="official_or_manual", schedule="china_gdp",



         fields=("yoy", "qoq")),



    dict(id="china_cpi", name="China CPI", agency="NBS", country="CN", category="inflation",



         zone="Asia/Shanghai", time="09:30", importance="high", hawkish_if="above",



         source="nbs", entry="official_or_manual", fred="CPALTT01CNM659N", schedule="china_cpi",



         fields=("yoy",)),



    dict(id="china_trade", name="China trade balance / exports", agency="Customs", country="CN",



         category="activity", zone="Asia/Shanghai", time="10:00", importance="medium",



         hawkish_if="above", source="customs_cn", entry="official_or_manual",



         schedule="china_trade", fields=("exports_yoy", "imports_yoy", "balance")),



    dict(id="china_ip", name="China industrial production", agency="NBS", country="CN",



         category="activity", zone="Asia/Shanghai", time="10:00", importance="medium",



         hawkish_if="above", source="nbs", entry="official_or_manual", schedule="china_ip",



         fields=("yoy",)),



    dict(id="ea_hicp_flash", name="Euro area HICP flash", agency="Eurostat", country="EA",



         category="inflation", zone="Europe/Berlin", time="11:00", importance="high",



         hawkish_if="above", source="eurostat", entry="official_or_manual", fred="CP0000EZ19M086NEST",



         schedule="ea_hicp", fields=("yoy", "core_yoy")),



    dict(id="uk_cpi", name="UK CPI", agency="ONS", country="GB", category="inflation",



         zone="Europe/London", time="07:00", importance="high", hawkish_if="above",



         source="ons", entry="official_or_manual", fred="GBRCPIALLMINMEI", schedule="uk_cpi",



         fields=("yoy", "core_yoy")),



]







# Published FOMC announcement days (2:00 ET). Minutes default to three weeks later,



# unless an official calendar overrides them. Verify against the Fed calendar on build.



FOMC_DECISIONS = (



    # 2024



    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",



    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",



    # 2025



    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",



    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",



    # 2026 (Federal Reserve published calendar)



    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",



    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",



    # 2027 (Federal Reserve published calendar; confirm on fetch)



    "2027-01-27", "2027-03-17", "2027-05-05", "2027-06-16",



    "2027-07-28", "2027-09-15", "2027-11-03", "2027-12-15",



)







# Mid-month-ish fallback day used only when an official schedule cannot be fetched.



FALLBACK_DAY = {



    "bls_cpi": 13, "bls_ppi": 11, "bls_jolts": 4, "bls_eci": None,



    "bea_pce": 28, "bea_gdp": 27, "fed_g17": 16, "census_retail": 15,



    "census_durable": 26, "census_factory": 4, "nyfed_empire": 15,



    "philly_fed": 16, "richmond_fed": 21, "kc_fed": 23, "dallas_fed": 25,



    "umich_prelim": 14, "umich_final": 28, "ifo": 25, "zew": 18,



    "china_cpi": 10, "china_trade": 8, "china_ip": 15, "ea_hicp": 30,



    "uk_cpi": 17, "tankan": None, "china_gdp": None,



}







AUCTION_TERMS = ("4-week", "8-week", "13-week", "17-week", "26-week", "52-week",



                 "2-year", "3-year", "5-year", "7-year", "10-year", "20-year", "30-year",



                 "5-year TIPS", "10-year TIPS", "30-year TIPS")











def event_catalog():



    """Return a copy of the static event definitions."""



    return [dict(e) for e in EVENTS]











def event_by_id(event_id):



    for e in EVENTS:



        if e["id"] == event_id:



            return dict(e)



    return None











def _parse_hm(value):



    hour, minute = str(value).split(":")



    return int(hour), int(minute)











def to_et(day, clock, zone_name):



    """Combine a calendar date and a local clock time, then convert to ET."""



    if isinstance(day, datetime):



        day = day.date()



    hour, minute = _parse_hm(clock)



    zone = ZoneInfo(zone_name)



    local = datetime.combine(day, time(hour, minute), tzinfo=zone)



    return local.astimezone(ET)











def iso_et(dt):



    if dt.tzinfo is None:



        dt = dt.replace(tzinfo=ET)



    return dt.astimezone(ET).isoformat(timespec="minutes")











def nyse_calendar():



    try:



        import exchange_calendars as xcals



        return xcals.get_calendar("XNYS")



    except Exception:



        return None











def is_us_session_day(day, calendar=None):



    calendar = calendar or nyse_calendar()



    if calendar is None:



        return day.weekday() < 5



    try:



        return bool(calendar.is_session(pd_timestamp(day)))



    except Exception:



        return day.weekday() < 5











def pd_timestamp(day):



    import pandas as pd



    return pd.Timestamp(day)











def next_business_day(day, n=1, calendar=None):



    calendar = calendar or nyse_calendar()



    current = day



    stepped = 0



    # Walk forward (or backward if n < 0) skipping weekends / NYSE holidays.



    step = 1 if n >= 0 else -1



    target = abs(n)



    while stepped < target:



        current += timedelta(days=step)



        if is_us_session_day(current, calendar):



            stepped += 1



    return current











def first_weekday(year, month, weekday, calendar=None, nth=1, business=False):



    """weekday: Mon=0 ... Sun=6. If business, skip NYSE holidays after landing on the weekday."""



    day = date(year, month, 1)



    while day.weekday() != weekday:



        day += timedelta(days=1)



    day += timedelta(weeks=nth - 1)



    if business:



        while not is_us_session_day(day, calendar):



            day += timedelta(days=1)



    return day











def nth_business_day(year, month, n, calendar=None):



    day = date(year, month, 1)



    count = 0



    while True:



        if is_us_session_day(day, calendar):



            count += 1



            if count == n:



                return day



        day += timedelta(days=1)



        if day.month != month:



            return date(year, month, 1)











def last_weekday(year, month, weekday, calendar=None, business=False):



    if month == 12:



        day = date(year + 1, 1, 1) - timedelta(days=1)



    else:



        day = date(year, month + 1, 1) - timedelta(days=1)



    while day.weekday() != weekday:



        day -= timedelta(days=1)



    if business:



        while not is_us_session_day(day, calendar):



            day -= timedelta(days=1)



    return day











def months_between(start, end):



    year, month = start.year, start.month



    while date(year, month, 1) <= end:



        yield year, month



        if month == 12:



            year, month = year + 1, 1



        else:



            month += 1











def _row(event, day, source="fallback", period=None, extra=None):



    when = to_et(day, event["time"], event["zone"])



    payload = {



        "id": event["id"],



        "name": event["name"],



        "agency": event["agency"],



        "country": event["country"],



        "category": event["category"],



        "importance": event["importance"],



        "hawkishIf": event["hawkish_if"],



        "source": event["source"],



        "entry": event["entry"],



        "fred": event.get("fred"),



        "bls": event.get("bls"),



        "schedule": event.get("schedule"),



        "fields": list(event.get("fields") or ()),



        "notes": event.get("notes"),



        "date": day.isoformat(),



        "timeEt": when.strftime("%H:%M"),



        "datetimeEt": iso_et(when),



        "zone": event["zone"],



        "localTime": event["time"],



        "scheduleSource": source,



        "period": period,



        "key": f"{event['id']}:{day.isoformat()}",



    }



    if extra:



        payload.update(extra)



    return payload











def _fallback_day(year, month, schedule, calendar=None):



    if schedule == "bls_empsit":



        return first_weekday(year, month, 4, calendar, nth=1, business=True)  # first Friday



    if schedule == "ism_mfg" or schedule == "sp_flash_us":



        return nth_business_day(year, month, 1, calendar)



    if schedule == "ism_services":



        return nth_business_day(year, month, 3, calendar)



    if schedule == "conference_board":



        return last_weekday(year, month, 1, calendar, business=True)  # last Tuesday



    if schedule == "umich_prelim":



        # Mid-month Friday approximation



        day = date(year, month, min(14, 28))



        while day.weekday() != 4:



            day += timedelta(days=1)



            if day.month != month:



                day -= timedelta(days=1)



                break



        return day



    if schedule == "umich_final":



        day = last_weekday(year, month, 4, calendar, business=True)



        return day



    if schedule == "ea_pmi_flash" or schedule == "uk_pmi_flash" or schedule == "jp_pmi_flash":



        return date(year, month, 23)



    if schedule == "cn_official_pmi":



        # Last day of the month in Beijing, often the last calendar day.



        if month == 12:



            return date(year, 12, 31)



        return date(year, month + 1, 1) - timedelta(days=1)



    if schedule == "cn_caixin_pmi":



        return date(year, month, 1)



    day_num = FALLBACK_DAY.get(schedule)



    if not day_num:



        return None



    try:



        day = date(year, month, day_num)



    except ValueError:



        day = date(year, month, 28)



    # Shift off weekends (not always a NYSE session — overseas releases can fall on US holidays).



    if day.weekday() >= 5:



        day = next_business_day(day, 1, calendar) if schedule.startswith(("bls", "bea", "fed", "census", "nyfed", "philly")) else day - timedelta(days=day.weekday() - 4)



    return day











def generate_fallback(start, end, calendar=None):



    """Deterministic schedule used when official calendars cannot be fetched."""



    calendar = calendar or nyse_calendar()



    start = start if isinstance(start, date) else date.fromisoformat(str(start))



    end = end if isinstance(end, date) else date.fromisoformat(str(end))



    rows = []



    # Weekly claims: Thursday 8:30 ET, skip NYSE holidays (approximation for federal holidays).



    claims = event_by_id("claims")



    day = start



    while day <= end:



        if day.weekday() == 3 and is_us_session_day(day, calendar):



            week = day.isocalendar()



            rows.append(_row(claims, day, source="fallback_thursday",



                             period=f"{week[0]}-W{week[1]:02d}"))



        day += timedelta(days=1)



    # FOMC



    decision = event_by_id("fomc_decision")



    minutes = event_by_id("fomc_minutes")



    for raw in FOMC_DECISIONS:



        dday = date.fromisoformat(raw)



        if start <= dday <= end:



            rows.append(_row(decision, dday, source="published_fomc_calendar", period=raw))



        mday = dday + timedelta(days=21)



        # Minutes are usually a Wednesday three weeks later; if that is a holiday, the



        # official calendar (when fetched) wins. Fallback keeps the +21 calendar day.



        if start <= mday <= end:



            rows.append(_row(minutes, mday, source="decision_plus_21d", period=raw,



                             extra={"decisionDate": raw}))



    # Monthly / quarterly events



    skip = {"claims", "fomc_decision", "fomc_minutes", "treasury_auction"}



    for event in EVENTS:



        if event["id"] in skip:



            continue



        sched = event.get("schedule")



        for year, month in months_between(start, end):



            if sched == "bls_eci" and month not in (1, 4, 7, 10):



                continue



            if sched == "tankan" and month not in (4, 7, 10, 1):



                continue



            if sched == "china_gdp" and month not in (1, 4, 7, 10):



                continue



            if sched in {"ecb", "boj", "boe", "pboc", "snb"}:



                # Policy meetings are irregular; do not invent dates in the fallback.



                continue



            day = _fallback_day(year, month, sched, calendar)



            if day is None or not (start <= day <= end):



                continue



            period = f"{year}-{month:02d}"



            rows.append(_row(event, day, source="fallback_rule", period=period))



    return rows











def _merge(existing, incoming, prefer="incoming"):



    by_key = {r["key"]: r for r in existing}



    for row in incoming:



        prev = by_key.get(row["key"])



        if prev is None:



            by_key[row["key"]] = row



            continue



        if prefer == "incoming" or (prev.get("scheduleSource", "").startswith("fallback")



                                    and not str(row.get("scheduleSource", "")).startswith("fallback")):



            # Keep any already-recorded print / consensus when refreshing the schedule.



            for field in ("actual", "consensus", "surprise", "surpriseZ", "printedAtEt", "status"):



                if prev.get(field) is not None and row.get(field) is None:



                    row[field] = prev[field]



            by_key[row["key"]] = row



    rows = list(by_key.values())



    rows.sort(key=lambda r: (r.get("datetimeEt") or "", r.get("id") or ""))



    return rows











def attach_manual(rows, path=None):



    path = Path(path or MANUAL_PATH)



    extras = _read_json(path, default=[])



    if not extras:



        return rows



    normalized = []



    for item in extras:



        event = event_by_id(item.get("id") or item.get("eventId") or "")



        if not event:



            continue



        day = date.fromisoformat(str(item["date"])[:10])



        row = _row(event, day, source="manual", period=item.get("period"),



                   extra={k: item[k] for k in item if k not in {"id", "eventId", "date"}})



        normalized.append(row)



    return _merge(rows, normalized, prefer="incoming")











def _read_json(path, default=None):



    path = Path(path)



    if not path.is_file():



        return default



    return json.loads(path.read_text(encoding="utf-8"))











def load_calendar(path=None):



    path = Path(path or CALENDAR_PATH)



    doc = _read_json(path, default=None)



    if not doc:



        return {"events": [], "builtAt": None, "start": None, "end": None, "sources": []}



    return doc











def save_calendar(doc, path=None):
    path = Path(path or CALENDAR_PATH)



    write_atomically(path, json.dumps(doc, indent=2, default=str).encode("utf-8"))



    return path











def upcoming(asof=None, days=7, path=None, include_past_hours=1):

    path = Path(path or CALENDAR_PATH)



    """Releases from (asof - include_past_hours) through asof + days, in ET."""



    asof = asof or datetime.now(ET)



    if asof.tzinfo is None:



        asof = asof.replace(tzinfo=ET)



    asof = asof.astimezone(ET)



    start = asof - timedelta(hours=include_past_hours)



    end = asof + timedelta(days=days)



    rows = (load_calendar(path).get("events") or [])



    out = []



    for row in rows:



        when = datetime.fromisoformat(row["datetimeEt"])



        if when.tzinfo is None:



            when = when.replace(tzinfo=ET)



        if start <= when <= end:



            out.append(row)



    return out











def due_window(asof=None, before_sec=30, after_sec=15 * 60, path=None):
    path = Path(path or CALENDAR_PATH)



    """Events whose official time is in [asof - after, asof + before] — the watcher window."""



    asof = asof or datetime.now(ET)



    if asof.tzinfo is None:



        asof = asof.replace(tzinfo=ET)



    asof = asof.astimezone(ET)



    lo = asof - timedelta(seconds=after_sec)



    hi = asof + timedelta(seconds=before_sec)



    rows = []



    for row in (load_calendar(path).get("events") or []):



        when = datetime.fromisoformat(row["datetimeEt"])



        if when.tzinfo is None:



            when = when.replace(tzinfo=ET)



        if lo <= when.astimezone(ET) <= hi and row.get("status") not in {"printed", "skipped"}:



            rows.append(row)



    return rows











def build_calendar(start=None, end=None, fetch=True, path=None):
    path = Path(path or CALENDAR_PATH)



    """Build or refresh the cached calendar. Network fetches are optional and never required."""



    today = datetime.now(ET).date()



    start = date.fromisoformat(str(start)) if start else date(today.year - 2, 1, 1)



    end = date.fromisoformat(str(end)) if end else date(today.year + 1, 12, 31)



    sources = ["fallback_rules", "published_fomc_calendar"]



    rows = generate_fallback(start, end)



    fetch_notes = []



    if fetch:



        try:



            from macro_sources import fetch_official_schedules



            official, notes, used = fetch_official_schedules(start, end)



            if official:



                rows = _merge(rows, official, prefer="incoming")



            sources.extend(used)



            fetch_notes.extend(notes)



        except Exception as error:  # noqa: BLE001 — a failed fetch must not block the fallback calendar



            fetch_notes.append(f"official schedule fetch failed: {error}")



    rows = attach_manual(rows)



    previous = load_calendar(path).get("events") or []



    if previous:



        rows = _merge(previous, rows, prefer="incoming")



        # Drop rows far outside the requested window to keep the cache bounded, but



        # keep anything already printed or that has a consensus attached.



        lo, hi = start - timedelta(days=14), end + timedelta(days=14)



        kept = []



        for row in rows:



            day = date.fromisoformat(row["date"])



            if lo <= day <= hi or row.get("actual") is not None or row.get("consensus") is not None:



                kept.append(row)



        rows = kept



    doc = {



        "builtAt": iso_et(datetime.now(ET)),



        "timezone": "America/New_York",



        "start": start.isoformat(),



        "end": end.isoformat(),



        "sources": sources,



        "notes": fetch_notes,



        "eventCount": len(rows),



        "catalog": [{"id": e["id"], "name": e["name"], "agency": e["agency"], "country": e["country"],



                     "category": e["category"], "importance": e["importance"], "entry": e["entry"],



                     "time": e["time"], "zone": e["zone"], "notes": e.get("notes")} for e in EVENTS],



        "events": rows,



    }



    save_calendar(doc, path)



    return doc











def context_block(asof=None, upcoming_days=7, last_n=8, path=None):
    path = Path(path or CALENDAR_PATH)



    """Plain-text [MACRO_EVENTS] block for the analyst prompt."""



    asof = asof or datetime.now(ET)



    coming = upcoming(asof=asof, days=upcoming_days, path=path, include_past_hours=0)



    doc = load_calendar(path)



    past = []



    for row in reversed(doc.get("events") or []):



        if row.get("actual") is None and row.get("status") != "printed":



            continue



        past.append(row)



        if len(past) >= last_n:



            break



    lines = ["[MACRO_EVENTS]"]



    if coming:



        lines.append("Upcoming releases (times in ET):")



        for row in coming[:16]:



            cons = row.get("consensus")



            cons_s = f", consensus {cons}" if cons is not None else ""



            entry = " (manual entry)" if row.get("entry") == "manual" else ""



            lines.append(f"  {row['datetimeEt']} {row['name']}{cons_s}{entry}")



    else:



        lines.append("No upcoming releases in the next week (calendar may need a refresh).")



    if past:



        lines.append("Latest prints (surprise is standardized vs history; positive = hotter/hawkish when hawkishIf=above):")



        for row in past:



            z = row.get("surpriseZ")



            z_s = f", z={z:+.2f}" if isinstance(z, (int, float)) else ""



            act = row.get("actual")



            cons = row.get("consensus")



            lines.append(f"  {row.get('datetimeEt') or row.get('date')} {row['name']}: actual {act}, consensus {cons}{z_s}")



    notes = [



        "Retail systems usually cannot trade the first tick; most of the liquid move often happens within seconds of the print.",



        "Spreads widen around high-impact releases. FOMC surprise is vs the prior target and vs the implied path, not just the headline change.",



        "UMCSENT on FRED lags the official Michigan print. Conference Board and some PMI headlines may be manual entries.",



    ]



    lines.extend(notes)



    return "\n".join(lines)











def main(argv=None):



    parser = argparse.ArgumentParser(description="Build or inspect the economic calendar (times in ET).")



    sub = parser.add_subparsers(dest="cmd", required=True)



    b = sub.add_parser("build")



    b.add_argument("--start")



    b.add_argument("--end")



    b.add_argument("--no-fetch", action="store_true")



    u = sub.add_parser("upcoming")



    u.add_argument("--days", type=int, default=7)



    sub.add_parser("show")



    sub.add_parser("catalog")



    args = parser.parse_args(argv)



    if args.cmd == "catalog":



        print(json.dumps([{"id": e["id"], "name": e["name"], "agency": e["agency"],



                           "country": e["country"], "time": e["time"], "zone": e["zone"]}



                          for e in EVENTS], indent=2))



        return 0



    if args.cmd == "build":



        doc = build_calendar(start=args.start, end=args.end, fetch=not args.no_fetch)



        print(json.dumps({"ok": True, "path": str(CALENDAR_PATH), "eventCount": doc["eventCount"],



                          "start": doc["start"], "end": doc["end"], "sources": doc["sources"],



                          "notes": doc["notes"]}, indent=2))



        return 0



    if args.cmd == "upcoming":



        rows = upcoming(days=args.days)



        print(json.dumps({"count": len(rows), "events": rows}, indent=2, default=str))



        return 0



    doc = load_calendar()



    print(json.dumps({"builtAt": doc.get("builtAt"), "eventCount": doc.get("eventCount"),



                      "start": doc.get("start"), "end": doc.get("end")}, indent=2))



    return 0











if __name__ == "__main__":



    sys.exit(main())



