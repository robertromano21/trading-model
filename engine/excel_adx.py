"""Refresh the workbook's ADX summary from its own worksheet calculations."""

import logging
import math

LOGGER = logging.getLogger(__name__)
REGIMES = {
    "Accelerating": {
        "Above": "Momentum accelerating above trend",
        "Crossed Above": "Bullish breakout gaining momentum",
        "Crossed Below": "Bearish breakdown accelerating",
        "Below": "Downtrend accelerating",
    },
    "Improving": {
        "Above": "Momentum improving above trend",
        "Crossed Above": "Potential trend recovery",
    },
    "Stable": {"Above": "Stable uptrend"},
    "Deteriorating": {
        "Above": "Uptrend losing momentum",
        "Crossed Below": "Trend position deteriorating rapidly",
    },
}
DEFAULT_REGIMES = {
    "Improving": "Momentum improving below trend",
    "Stable": "Stable downtrend",
    "Deteriorating": "Downtrend strengthening",
    "Sharp Deterioration": "Severe trend deterioration",
}


def workbook_number(value):
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    return value if type(value) in (int, float) and math.isfinite(value) else None


def adx_state(change):
    if change <= -5:
        return "Sharp Deterioration"
    if change <= -2:
        return "Deteriorating"
    if change < 2:
        return "Stable"
    if change <= 5:
        return "Improving"
    return "Accelerating"


def moving_average_status(prior, current):
    if prior < 0 < current:
        return "Crossed Above"
    if prior > 0 > current:
        return "Crossed Below"
    return "Above" if current > 0 else "Below"


def market_regime(state, ma_status):
    return REGIMES.get(state, {}).get(ma_status, DEFAULT_REGIMES.get(state, ""))


def summary_values(closes, current_adx, prior_adx):
    """Return the UpdateADXStates output in Summary G through S order."""
    closes = [workbook_number(value) for value in closes]
    if len(closes) != 55 or any(price is None or price <= 0 for price in closes):
        raise ValueError("55 usable closing prices are required.")
    current_adx, prior_adx = workbook_number(current_adx), workbook_number(prior_adx)
    if any(value is None or not 0 <= value <= 100 for value in (current_adx, prior_adx)):
        raise ValueError("Current and five-session-prior ADX values are required in column AO.")

    current_close, prior_close = closes[-1], closes[-6]
    current_ma = sum(closes[5:]) / 50
    prior_ma = sum(closes[:-5]) / 50
    current_pct = (current_close - current_ma) / current_ma
    prior_pct = (prior_close - prior_ma) / prior_ma
    ma_status = moving_average_status(prior_pct, current_pct)
    change = current_adx - prior_adx
    state = adx_state(change)
    regime = market_regime(state, ma_status)
    return (
        current_adx, prior_adx, change, state, current_close, current_ma, current_pct,
        prior_close, prior_ma, prior_pct, current_pct - prior_pct, ma_status, regime,
    )


def extend_adx_formulas(sheet, row):
    """Continue the ticker sheet's formulas when its prefilled rows run out."""
    if row <= 2:
        return
    adx = sheet.Range(f"AO{row}")
    if adx.HasFormula or adx.Value2 not in (None, "") or not sheet.Range(f"AO{row - 1}").HasFormula:
        return
    source = sheet.Range(f"G{row - 1}:AO{row - 1}").FormulaR1C1[0]
    target = sheet.Range(f"G{row}:AO{row}").FormulaR1C1[0]
    for column, (formula, value) in enumerate(zip(source, target), start=7):
        if value in (None, "") and isinstance(formula, str) and formula.startswith("="):
            sheet.Cells(row, column).FormulaR1C1 = formula


def refresh_adx_summary(workbook):
    from pywintypes import com_error

    try:
        summary = workbook.Worksheets("Summary")
    except com_error:
        LOGGER.info("ADX Summary refresh skipped: workbook has no Summary sheet.")
        return {"updated": 0, "skipped": 0}

    headers = summary.Range("G1:S1").Value2[0]
    if (headers[0], headers[1], headers[-2], headers[-1]) != ("Current ADX", "ADX 5D Ago", "50 MA Status", "Market Regime"):
        LOGGER.warning("ADX Summary refresh skipped: expected headers are missing from G:S.")
        return {"updated": 0, "skipped": 0}

    updated, skipped = 0, 0
    last = summary.Cells(summary.Rows.Count, 1).End(-4162).Row
    for row in range(2, last + 1):
        ticker = str(summary.Cells(row, 1).Value2 or "").strip()
        if not ticker:
            break
        try:
            sheet = workbook.Worksheets(ticker)
        except com_error:
            LOGGER.warning("ADX Summary skipped %s: ticker sheet unavailable.", ticker)
            skipped += 1
            continue
        last_date = sheet.Cells(sheet.Rows.Count, 1).End(-4162).Row
        if last_date < 56:
            LOGGER.warning("ADX Summary skipped %s: fewer than 55 closing-price rows.", ticker)
            skipped += 1
            continue
        try:
            closes = [item[0] for item in sheet.Range(f"E{last_date - 54}:E{last_date}").Value2]
            values = summary_values(closes, sheet.Range(f"AO{last_date}").Value2, sheet.Range(f"AO{last_date - 5}").Value2)
        except ValueError as error:
            LOGGER.warning("ADX Summary skipped %s: %s", ticker, error)
            skipped += 1
            continue
        summary.Range(f"G{row}:S{row}").Value2 = (values,)
        updated += 1
    LOGGER.info("ADX Summary refreshed: %s rows updated, %s skipped.", updated, skipped)
    return {"updated": updated, "skipped": skipped}
