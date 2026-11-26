// Share-dilution flags for the analyst (split-adjusted YoY share growth + offering filings).
// Pure helpers plus a flags-file reader. Point-in-time comparisons use filing dates, not period ends.
// Mirrors engine/dilution.py so Node verdicts and the Python paper trader agree.

export const DEFAULTS = Object.freeze({
  redThreshold: 0.10,
  warnThreshold: 0.05,
  offeringDays: 30,
  dilutionExit: false,
  dilutionFilter: true,
  splitRatioTolerance: 0.005,
  // Size-relative offering flag (mirrors engine/dilution.py): <5% LOW (no block), 5-15% MEDIUM, >15% HIGH (RED).
  offeringSizeRelative: true,
  offeringSmallPct: 0.05,
  offeringLargePct: 0.15,
});

export const OFFERING_FORMS = Object.freeze([
  "S-1", "S-1/A", "S-3", "S-3/A", "S-3ASR",
  "F-1", "F-1/A", "F-3", "F-3/A", "F-3ASR",
  "424B", "424B1", "424B2", "424B3", "424B4", "424B5",
]);
export const EIGHT_K_DILUTION_ITEMS = Object.freeze(["3.02"]);
export const ATM_RE = /at[-\s]?the[-\s]?market|\batm (offering|program|facility|issuance)\b|equity distribution agreement|sales agreement|controlled equity offering/i;
const SPLIT_RATIOS = [
  2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100,
  1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 8, 1 / 10, 1 / 15, 1 / 20, 1 / 25, 1 / 30, 1 / 40, 1 / 50, 1 / 100,
  1.5, 2 / 3, 3 / 2, 5 / 4, 4 / 3,
];

export function configFrom(settings = {}) {
  const num = (v, d) => { const n = Number(v); return Number.isFinite(n) ? n : d; };
  const on = (v, d) => (v == null ? d : typeof v === "string" ? /^(1|true|yes|on)$/i.test(v) : Boolean(v));
  const red = num(settings.redThreshold ?? settings.dilution_red_threshold ?? settings.red_threshold, DEFAULTS.redThreshold);
  const warn = num(settings.warnThreshold ?? settings.dilution_warn_threshold ?? settings.warn_threshold, DEFAULTS.warnThreshold);
  return {
    redThreshold: red,
    warnThreshold: Math.min(warn, red),
    offeringDays: Math.max(1, num(settings.offeringDays ?? settings.dilution_offering_days ?? settings.offering_days, DEFAULTS.offeringDays)),
    dilutionExit: on(settings.dilutionExit ?? settings.dilution_exit, DEFAULTS.dilutionExit),
    dilutionFilter: on(settings.dilutionFilter ?? settings.dilution_filter, DEFAULTS.dilutionFilter),
    splitRatioTolerance: num(settings.splitRatioTolerance, DEFAULTS.splitRatioTolerance),
    offeringSizeRelative: on(settings.offeringSizeRelative ?? settings.dilution_offering_size_relative ?? settings.offering_size_relative, DEFAULTS.offeringSizeRelative),
    offeringSmallPct: num(settings.offeringSmallPct ?? settings.dilution_offering_small_pct ?? settings.offering_small_pct, DEFAULTS.offeringSmallPct),
    offeringLargePct: num(settings.offeringLargePct ?? settings.dilution_offering_large_pct ?? settings.offering_large_pct, DEFAULTS.offeringLargePct),
  };
}

// Severity from offering hits already sized by the Python scan (sizeTier on each hit). Unsized -> UNKNOWN.
export function offeringSeverity(offerings) {
  if (!offerings?.length) return null;
  const tiers = offerings.map((o) => o?.sizeTier || "UNKNOWN");
  if (tiers.includes("LARGE")) return "HIGH";
  if (tiers.includes("MEDIUM")) return "MEDIUM";
  if (tiers.includes("UNKNOWN")) return "UNKNOWN";
  return "LOW";
}

export function asDate(value) {
  if (value == null || value === "") return null;
  if (value instanceof Date && !Number.isNaN(value.getTime())) return value.toISOString().slice(0, 10);
  const text = String(value).trim().replace(/\//g, "-").slice(0, 10);
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null;
}

function dayMs(iso) { const d = asDate(iso); return d ? Date.parse(d + "T00:00:00Z") : NaN; }
function daysBetween(a, b) { return Math.round((dayMs(b) - dayMs(a)) / 86400000); }

export function splitFactor(splits, start, end) {
  const lo = asDate(start), hi = asDate(end);
  if (!lo || !hi || !splits?.length) return 1;
  let factor = 1;
  for (const item of splits) {
    const date = asDate(item.date || item.execution_date || item.executionDate || item[0]);
    const frm = item.from ?? item.split_from ?? item[1];
    const to = item.to ?? item.split_to ?? item[2];
    if (!date || !(frm > 0) || !(to > 0)) continue;
    if (date > lo && date <= hi) factor *= Number(to) / Number(frm);
  }
  return factor;
}

export function looksLikeSplit(ratio, tolerance = DEFAULTS.splitRatioTolerance) {
  const r = Number(ratio);
  if (!Number.isFinite(r) || r <= 0) return null;
  for (const canonical of SPLIT_RATIOS) {
    if (Math.abs(r - canonical) <= Math.max(tolerance, Math.abs(canonical) * tolerance)) return canonical;
  }
  return null;
}

export function yoyShareChange(currentVal, priorVal, splits, currentEnd, priorEnd) {
  const current = Number(currentVal), prior = Number(priorVal);
  if (!(prior > 0) || !(current > 0)) return null;
  const raw = current / prior;
  const factor = splitFactor(splits, priorEnd, currentEnd);
  const adjusted = factor > 0 ? raw / factor : raw;
  if (looksLikeSplit(adjusted)) return 0;
  return adjusted - 1;
}

export function pointInTime(rows, asof) {
  const cutoff = asDate(asof);
  if (!cutoff) return [];
  return (rows || []).map((row) => {
    const filed = asDate(row.filed || row.filedAt || row.filed_date);
    const val = Number(row.val ?? row.value);
    const end = asDate(row.end || row.periodEnd || row.period_end);
    if (!filed || filed > cutoff || !(val > 0)) return null;
    return { ...row, val, filed, end: end || row.end };
  }).filter(Boolean).sort((a, b) => (a.filed + a.end).localeCompare(b.filed + b.end));
}

function closestPrior(rows, latestEnd, loDays = 300, hiDays = 430) {
  const latest = asDate(latestEnd);
  if (!latest) return null;
  let best = null, bestDist = null;
  for (const row of rows) {
    const end = asDate(row.end);
    if (!end || end >= latest) continue;
    const dist = daysBetween(end, latest);
    if (dist >= loDays && dist <= hiDays && (bestDist == null || Math.abs(dist - 365) < Math.abs(bestDist - 365))) {
      best = row; bestDist = dist;
    }
  }
  return best;
}

export function shareGrowthAsof(series, splits, asof, altSeries) {
  const pair = (rows, source) => {
    if (!rows || rows.length < 2) return null;
    const latest = rows.at(-1);
    const prior = closestPrior(rows, latest.end);
    if (!prior) return null;
    const change = yoyShareChange(latest.val, prior.val, splits, latest.end, prior.end);
    if (change == null) return null;
    const factor = splitFactor(splits, prior.end, latest.end);
    return {
      source, shareChangeYoY: change, shareChangeRawYoY: prior.val ? latest.val / prior.val - 1 : null,
      splitFactor: factor, splitInWindow: Math.abs(factor - 1) > 1e-9,
      current: { end: latest.end, filed: latest.filed, val: latest.val },
      prior: { end: prior.end, filed: prior.filed, val: prior.val },
    };
  };
  const main = pair(pointInTime(series, asof), "diluted");
  const alt = pair(pointInTime(altSeries, asof), "basic");
  if (main?.splitInWindow && alt) return alt;
  return main || alt;
}

export function classifyGrowth(change, settings) {
  const cfg = configFrom(settings);
  if (change == null || !Number.isFinite(change)) return null;
  if (change >= cfg.redThreshold) return "DILUTION";
  if (change >= cfg.warnThreshold) return "DILUTION_WARNING";
  return null;
}

export function parseSubmissions(payload) {
  if (!payload) return [];
  if (Array.isArray(payload)) return payload.map(normalizeFiling).filter(Boolean);
  const recent = payload.filings?.recent || payload.recent || payload.filings;
  if (Array.isArray(recent)) return recent.map(normalizeFiling).filter(Boolean);
  if (recent && typeof recent === "object" && Array.isArray(recent.form)) {
    const n = recent.form.length;
    const keys = Object.keys(recent);
    const out = [];
    for (let i = 0; i < n; i++) {
      const row = {};
      for (const k of keys) row[k] = Array.isArray(recent[k]) ? recent[k][i] : null;
      const norm = normalizeFiling(row);
      if (norm) out.push(norm);
    }
    return out.sort((a, b) => b.filingDate.localeCompare(a.filingDate));
  }
  return [];
}

function normalizeFiling(row) {
  if (!row || typeof row !== "object") return null;
  const form = String(row.form || row.formType || "").toUpperCase().replace(/\s+/g, "");
  const filingDate = asDate(row.filingDate || row.filed || row.acceptanceDateTime);
  if (!form || !filingDate) return null;
  return {
    form, filingDate,
    reportDate: asDate(row.reportDate) || null,
    accession: row.accessionNumber || row.accession || "",
    items: row.items || row.item || "",
    description: row.primaryDocDescription || row.description || "",
    primaryDocument: row.primaryDocument || "",
  };
}

const DEBT_RE = /\b(senior notes?|debt securities|notes due|medium[-\s]term notes?|commercial paper|debentures?|bonds? payable)\b/i;
const EQUITY_RE = /\b(common stock|ordinary shares|class [a-z] common|preferred stock|depositary shares|at[-\s]?the[-\s]?market|\batm\b|equity distribution|warrants? to purchase|forward sale|forward prosupp)\b/i;

export function isOfferingFiling(filing, text) {
  const form = String(filing?.form || filing || "").toUpperCase().replace(/\s+/g, "");
  const items = filing?.items || "";
  const desc = filing?.description || "";
  const blob = [desc, text || ""].filter(Boolean).join(" ");
  if (form === "8-K" || form === "8-K/A") {
    const tokens = String(items).split(/[;,]/).map((s) => s.trim()).filter(Boolean);
    if (tokens.includes("3.02")) return true;
    if (/unregistered sales of equity|3\.02/i.test(`${items} ${blob}`)) return true;
    if (ATM_RE.test(blob) || ATM_RE.test(String(items)) || EQUITY_RE.test(blob)) return true;
    return false;
  }
  if (["S-1", "S-1/A", "F-1", "F-1/A"].includes(form)) return true;
  if (text && (ATM_RE.test(text) || EQUITY_RE.test(text)) && !DEBT_RE.test(text)) return true;
  if (["S-3ASR", "F-3ASR", "424B2", "424B5"].includes(form)) {
    return Boolean(ATM_RE.test(blob) || EQUITY_RE.test(blob)) && !DEBT_RE.test(blob);
  }
  if (["S-3", "S-3/A", "F-3", "F-3/A", "424B", "424B1", "424B3", "424B4"].includes(form)) {
    if (DEBT_RE.test(blob)) return false;
    return true;
  }
  if (ATM_RE.test(blob)) return true;
  return false;
}

export function offeringsAsof(filings, asof, settings, documentTexts = {}) {
  const cfg = configFrom(settings);
  const cutoff = asDate(asof);
  if (!cutoff) return [];
  const startMs = dayMs(cutoff) - cfg.offeringDays * 86400000;
  const rows = parseSubmissions(filings);
  // Carry the Python scan's size annotations (sizeTier / sizePct / offeringType) through normalization.
  const rawRows = Array.isArray(filings) ? filings : Array.isArray(filings?.filings?.recent) ? filings.filings.recent : [];
  const sizing = new Map(rawRows.filter((r) => r && (r.accession || r.accessionNumber)).map((r) => [r.accession || r.accessionNumber, r]));
  const SIZE_KEYS = ["sizeTier", "sizePct", "offeringType", "sizeShares", "sizeDollars"];
  const hits = [];
  for (const row of rows) {
    const filed = asDate(row.filingDate);
    if (!filed || filed > cutoff || dayMs(filed) < startMs) continue;
    const text = documentTexts[row.accession] || documentTexts[row.primaryDocument];
    if (isOfferingFiling(row, text)) {
      const src = sizing.get(row.accession) || {};
      const extra = Object.fromEntries(SIZE_KEYS.filter((k) => src[k] !== undefined && src[k] !== null).map((k) => [k, src[k]]));
      hits.push({ ...row, atm: Boolean((text && ATM_RE.test(text)) || ATM_RE.test(row.description || "")), ...extra });
    }
  }
  return hits;
}

export function evaluate({ ticker, asof, shareSeries, basicSeries, splits, filings, documentTexts, settings } = {}) {
  const cfg = configFrom(settings);
  const day = asDate(asof) || new Date().toISOString().slice(0, 10);
  const growth = shareGrowthAsof(shareSeries, splits, day, basicSeries);
  const change = growth?.shareChangeYoY ?? null;
  const flags = [];
  const kind = classifyGrowth(change, cfg);
  if (kind) flags.push(kind);
  const offerings = filings != null ? offeringsAsof(filings, day, cfg, documentTexts) : [];
  if (offerings.length) flags.push("DILUTION_OFFERING");
  const severity = offeringSeverity(offerings);
  const sized = offerings.filter((o) => Number.isFinite(o?.sizePct));
  const offerBlocks = flags.includes("DILUTION_OFFERING") && !(cfg.offeringSizeRelative && severity === "LOW");
  return {
    ticker: String(ticker || "").toUpperCase(), asof: day, flags,
    shareChangeYoY: change, growth, offerings,
    red: flags.includes("DILUTION"),
    warning: flags.includes("DILUTION_WARNING"),
    offering: flags.includes("DILUTION_OFFERING"),
    offeringSeverity: severity,
    offeringSizePct: sized.length ? Math.max(...sized.map((o) => o.sizePct)) : null,
    blocksBuy: flags.includes("DILUTION") || offerBlocks,
    config: cfg,
  };
}

export function riskEntry(result) {
  if (!result?.flags?.length) return null;
  const change = result.shareChangeYoY;
  const pct = change == null || !Number.isFinite(change) ? "n/a" : `${(change * 100).toFixed(1)}%`;
  const parts = [];
  if (result.flags.includes("DILUTION")) parts.push(`split-adjusted diluted (or basic) share count is up ${pct} year over year`);
  else if (result.flags.includes("DILUTION_WARNING")) parts.push(`share count is up ${pct} year over year (warning, below the 10% red threshold)`);
  if (result.flags.includes("DILUTION_OFFERING") && result.offerings?.[0]) {
    const o = result.offerings[0];
    const size = Number.isFinite(result.offeringSizePct) ? `, about ${(result.offeringSizePct * 100).toFixed(1)}% of shares outstanding` : "";
    parts.push(`recent ${o.form} filed ${o.filingDate}${o.atm ? " (ATM language)" : ""}${o.offeringType ? ` [${o.offeringType}]` : ""}${size}${result.offeringSeverity ? `, offering severity ${result.offeringSeverity}` : ""}`);
  }
  const sev = result.offeringSeverity;
  const sizeRel = result.config?.offeringSizeRelative ?? true;
  let severity = "MEDIUM";
  if (result.flags.includes("DILUTION") || sev === "HIGH") severity = "HIGH";
  else if (result.flags.includes("DILUTION_OFFERING") && sev === "LOW" && sizeRel) severity = "LOW";
  else if (result.flags.includes("DILUTION_OFFERING")) severity = sev ? "MEDIUM" : "HIGH";
  return {
    kind: "DILUTION",
    severity,
    detail: parts.join("; ") + ".",
  };
}

const HEALTH_RANK = ["UNKNOWN", "STRONG", "ADEQUATE", "WEAK", "DISTRESSED"];
const atLeast = (current, floor) => {
  const a = HEALTH_RANK.indexOf(current), b = HEALTH_RANK.indexOf(floor);
  if (a < 0) return floor;
  return a < b ? floor : current;
};

export function applyDilutionToVerdict(verdict, result) {
  const v = { ...(verdict || {}) };
  const clamps = [...(v.clamps || [])];
  if (!result) return { verdict: v, clamps };
  const flags = result.flags || [];
  const risk = riskEntry(result);
  const risks = [...(v.risks || [])];
  if (risk && !risks.some((r) => r?.kind === "DILUTION")) {
    risks.push(risk);
    v.risks = risks;
    clamps.push("dilution risk entry added");
  }
  const sev = result.offeringSeverity;
  const sizeRel = result.config?.offeringSizeRelative ?? true;
  if (!flags.includes("DILUTION") && flags.includes("DILUTION_OFFERING") && sizeRel && sev === "HIGH") {
    if (v.flag !== "RED") { clamps.push(`flag ${v.flag} -> RED (large offering)`); v.flag = "RED"; }
    if (["BUY", "ADD"].includes(v.action)) { clamps.push(`action ${v.action} -> WATCH (large offering)`); v.action = "WATCH"; }
    const entry = { ...(v.entry || {}) };
    if (entry.bandLow != null || entry.bandHigh != null || (entry.orderType && entry.orderType !== "NONE")) {
      v.entry = { ...entry, orderType: "NONE", bandLow: null, bandHigh: null, limitPrice: null };
      clamps.push("entry band removed (large offering)");
    }
  } else if (!flags.includes("DILUTION") && flags.includes("DILUTION_OFFERING") && sizeRel && sev === "LOW") {
    clamps.push("small offering (<5% of shares): risk noted, no clamp");
  } else if (flags.includes("DILUTION")) {
    if (v.flag !== "RED") { clamps.push(`flag ${v.flag} -> RED (dilution ${result.shareChangeYoY})`); v.flag = "RED"; }
    if (!["WEAK", "DISTRESSED"].includes(v.fundamentalHealth)) {
      clamps.push(`fundamentalHealth ${v.fundamentalHealth} -> WEAK (dilution)`);
      v.fundamentalHealth = atLeast(v.fundamentalHealth || "UNKNOWN", "WEAK");
    }
    if (["BUY", "ADD"].includes(v.action)) { clamps.push(`action ${v.action} -> WATCH (dilution red flag)`); v.action = "WATCH"; }
    const entry = { ...(v.entry || {}) };
    if (entry.bandLow != null || entry.bandHigh != null || (entry.orderType && entry.orderType !== "NONE")) {
      v.entry = { ...entry, orderType: "NONE", bandLow: null, bandHigh: null, limitPrice: null };
      clamps.push("entry band removed (dilution red flag)");
    }
  } else if (flags.includes("DILUTION_OFFERING")) {
    if (v.flag === "GREEN" || !v.flag) { clamps.push(`flag ${v.flag} -> YELLOW (dilution offering)`); v.flag = "YELLOW"; }
    if (["BUY", "ADD"].includes(v.action)) { clamps.push(`action ${v.action} -> WATCH (dilution offering)`); v.action = "WATCH"; }
  }
  if (clamps.length) v.clamps = [...new Set([...(v.clamps || []), ...clamps])];
  return { verdict: v, clamps };
}

export function annotateFinancials(financials, { splits, filings, flagsFile, asof, settings } = {}) {
  const derived = { ...(financials?.derived || {}) };
  const quarterly = financials?.quarterly || {};
  const history = financials?.history || {};
  const diluted = history.dilutedShares || quarterly.dilutedShares || [];
  const basic = history.basicShares || quarterly.basicShares || [];
  const snap = flagsFile?.tickers?.[financials?.ticker] || null;
  const day = asDate(asof) || derived.lastQuarterEnd || new Date().toISOString().slice(0, 10);
  const result = evaluate({
    ticker: financials?.ticker, asof: day, shareSeries: diluted, basicSeries: basic,
    splits, filings: filings ?? (snap ? { filings: { recent: snap.offerings || [] } } : undefined),
    settings,
  });
  // If the Python scan already has a fresher offering flag, keep it.
  if (snap?.flags?.includes("DILUTION_OFFERING") && !result.flags.includes("DILUTION_OFFERING")) {
    result.flags.push("DILUTION_OFFERING");
    result.offering = true;
    result.offerings = snap.offerings || result.offerings;
    result.offeringSeverity = snap.offeringSeverity ?? offeringSeverity(result.offerings);
    result.offeringSizePct = snap.offeringSizePct ?? result.offeringSizePct ?? null;
    result.blocksBuy = result.flags.includes("DILUTION")
      || !(result.config?.offeringSizeRelative && result.offeringSeverity === "LOW");
  }
  derived.dilutedShareChangeYoY = result.shareChangeYoY;
  derived.dilutedShareChangeRawYoY = result.growth?.shareChangeRawYoY ?? derived.dilutedShareChangeRawYoY ?? null;
  derived.dilutionSource = result.growth?.source || null;
  derived.dilutionSplitAdjusted = true;
  derived.dilutionFlag = result.red ? "RED" : result.warning ? "WARNING" : result.offering ? "OFFERING" : null;
  derived.dilutionFlags = result.flags;
  derived.dilutionOffering = result.offerings?.[0] || null;
  derived.dilutionOfferingSeverity = result.offeringSeverity ?? null;
  derived.dilutionOfferingSizePct = result.offeringSizePct ?? null;
  derived.basicShareChangeYoY = result.growth?.source === "basic" ? result.shareChangeYoY : derived.basicShareChangeYoY ?? null;
  return { derived, dilution: result };
}

export function extractShareSeries(facts) {
  const flow = conceptFacts(facts, "us-gaap", [
    "WeightedAverageNumberOfDilutedSharesOutstanding",
    "WeightedAverageNumberOfDilutedSharesOutstandingBasicAndDiluted",
    "WeightedAverageNumberOfShareOutstandingBasicAndDiluted",
  ], "flow");
  const quarterly = flow.filter((r) => r.period === "Q");
  for (const fy of flow.filter((r) => r.period === "FY")) {
    if (!quarterly.some((q) => q.end && fy.end && Math.abs(daysBetween(q.end, fy.end)) < 10)) quarterly.push(fy);
  }
  quarterly.sort((a, b) => a.end.localeCompare(b.end));
  let basic = conceptFacts(facts, "dei", ["EntityCommonStockSharesOutstanding"], "instant");
  if (!basic.length) {
    basic = conceptFacts(facts, "us-gaap", ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"], "instant");
  }
  return { dilutedShares: quarterly, basicShares: basic };
}

function conceptFacts(facts, namespace, names, kind) {
  const ns = facts?.facts?.[namespace] || {};
  for (const name of names) {
    const units = ns[name]?.units || {};
    let rows = units.shares || units.pure || units.USD || [];
    if (!rows.length) {
      for (const values of Object.values(units)) { if (values?.length) { rows = values; break; } }
    }
    const byEnd = new Map();
    for (const f of rows) {
      if (!/^10-[QK]/.test(f.form || "")) continue;
      const end = asDate(f.end), filed = asDate(f.filed), val = Number(f.val);
      if (!end || !filed || !Number.isFinite(val)) continue;
      if (kind === "flow") {
        const start = asDate(f.start);
        if (!start) continue;
        const len = daysBetween(start, end);
        const period = len >= 80 && len <= 100 ? "Q" : len >= 350 && len <= 380 ? "FY" : null;
        if (!period) continue;
        const key = period + end;
        const prev = byEnd.get(key);
        if (!prev || filed > prev.filed) byEnd.set(key, { period, end, val, filed, concept: name });
      } else {
        const prev = byEnd.get(end);
        if (!prev || filed > prev.filed) byEnd.set(end, { end, val, filed, concept: name });
      }
    }
    if (byEnd.size) return [...byEnd.values()].sort((a, b) => a.end.localeCompare(b.end) || a.filed.localeCompare(b.filed));
  }
  return [];
}

export function readFlagsFile(fsRead, path) {
  try {
    return JSON.parse(fsRead(path, "utf8"));
  } catch {
    return null;
  }
}
