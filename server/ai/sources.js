// External data for the analyst, all cached locally with timestamps (store.cached):
//  * Massive (formerly Polygon): news & press releases (/v2/reference/news), ticker details (/v3/reference/tickers),
//    dividends, splits, ticker search. Massive financials are NOT on the current plan (403), so:
//  * SEC EDGAR XBRL company facts (free): revenue, margins, cash flow, debt, cash, equity, dilution, buybacks.
//  * Schwab market data (read-only, via engine/schwab_data.py's allow-listed client): live quote + fundamentals.
// The Massive key comes from MASSIVE_API_KEY (.env.local) and is never logged; error text is scrubbed.
import { spawn } from "node:child_process";
import { extractShareSeries, shareGrowthAsof } from "./dilution.js";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const MIN = 60_000, HOUR = 60 * MIN, DAY = 24 * HOUR;
const PRESS_WIRES = /globe ?newswire|pr ?newswire|business ?wire|accesswire|newsfile|investor relations/i;
const scrub = (s) => String(s || "").replace(/apiKey=[^&\s]+/gi, "apiKey=***");

async function getJson(url, { fetchImpl = globalThis.fetch, headers = {}, timeoutMs = 20000 } = {}) {
  for (let attempt = 1; attempt <= 3; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetchImpl(url, { headers, signal: controller.signal });
      if (response.status === 429 && attempt < 3) { await new Promise((r) => setTimeout(r, 1500 * attempt)); continue; }
      if (!response.ok) throw Object.assign(new Error(`HTTP ${response.status}`), { status: response.status });
      return await response.json();
    } catch (error) {
      if (attempt === 3 || error.status) throw new Error(scrub(error.name === "AbortError" ? "timed out" : error.message));
    } finally { clearTimeout(timer); }
  }
}

export function createSources({ store, env = process.env, fetchImpl = globalThis.fetch, root, python }) {
  const massiveKey = () => (env.MASSIVE_API_KEY || "").trim();
  const massive = (path, params = {}) => {
    if (!massiveKey()) throw new Error("MASSIVE_API_KEY is not set");
    const query = new URLSearchParams({ ...params, apiKey: massiveKey() });
    return getJson(`https://api.massive.com${path}?${query}`, { fetchImpl });
  };
  const secAgent = () => env.SEC_USER_AGENT || "Trading-Model-main local research (personal use)";

  async function news(ticker, { days = 30, limit = 40, refresh } = {}) {
    const since = new Date(Date.now() - days * DAY).toISOString().slice(0, 10);
    return store.cached(`news:${ticker}:${days}`, "MASSIVE_NEWS", ticker, 30 * MIN, async () => {
      const data = await massive("/v2/reference/news", { ticker, "published_utc.gte": since, order: "desc",
        sort: "published_utc", limit: String(limit) });
      return (data.results || []).map((n) => {
        const insight = (n.insights || []).find((i) => i.ticker === ticker);
        return { published: n.published_utc, publisher: n.publisher?.name || "", title: n.title || "",
          description: n.description || "", url: n.article_url || "", tickers: (n.tickers || []).slice(0, 8),
          pressRelease: PRESS_WIRES.test(n.publisher?.name || "") || /press release/i.test(n.title || ""),
          sentiment: insight?.sentiment || null, sentimentReason: insight?.sentiment_reasoning || null };
      });
    }, { refresh });
  }

  async function details(ticker, { refresh } = {}) {
    return store.cached(`details:${ticker}`, "TICKER_DETAILS", ticker, 7 * DAY, async () => {
      const r = (await massive(`/v3/reference/tickers/${encodeURIComponent(ticker)}`)).results || {};
      return { ticker: r.ticker, name: r.name, type: r.type, exchange: r.primary_exchange, active: r.active,
        cik: r.cik, sic: r.sic_code, industry: r.sic_description, marketCap: r.market_cap, listDate: r.list_date,
        sharesOutstanding: r.share_class_shares_outstanding ?? r.weighted_shares_outstanding,
        employees: r.total_employees, description: r.description };
    }, { refresh });
  }

  async function dividends(ticker, { refresh } = {}) {
    return store.cached(`dividends:${ticker}`, "DIVIDENDS", ticker, DAY, async () =>
      ((await massive("/v3/reference/dividends", { ticker, limit: "8", order: "desc", sort: "ex_dividend_date" })).results || [])
        .map((d) => ({ exDate: d.ex_dividend_date, payDate: d.pay_date, cash: d.cash_amount, frequency: d.frequency, type: d.dividend_type })),
    { refresh });
  }

  async function splits(ticker, { refresh } = {}) {
    return store.cached(`splits:${ticker}`, "SPLITS", ticker, DAY, async () =>
      ((await massive("/v3/reference/splits", { ticker, limit: "5", order: "desc", sort: "execution_date" })).results || [])
        .map((s) => ({ date: s.execution_date, from: s.split_from, to: s.split_to })), { refresh });
  }

  async function search(name) {
    const key = `search:${name.toLowerCase()}`;
    return store.cached(key, "TICKER_DETAILS", null, 30 * DAY, async () =>
      ((await massive("/v3/reference/tickers", { search: name, active: "true", market: "stocks", limit: "8" })).results || [])
        .map((r) => ({ ticker: r.ticker, name: r.name, type: r.type, exchange: r.primary_exchange })));
  }

  async function secFinancials(ticker, cik, { refresh } = {}) {
    if (!cik) {
      const map = await store.cached("sec:tickers", "SEC_FINANCIALS", null, 7 * DAY, async () => {
        const data = await getJson("https://www.sec.gov/files/company_tickers.json", { fetchImpl, headers: { "User-Agent": secAgent() } });
        return Object.fromEntries(Object.values(data).map((r) => [r.ticker, String(r.cik_str)]));
      });
      cik = map.value[ticker] || map.value[ticker.replace(".", "-")];
      if (!cik) throw new Error("no SEC CIK for " + ticker + " (ETF, fund or foreign filer)");
    }
    const padded = String(cik).padStart(10, "0");
    return store.cached(`sec:${ticker}`, "SEC_FINANCIALS", ticker, DAY, async () => {
      const facts = await getJson(`https://data.sec.gov/api/xbrl/companyfacts/CIK${padded}.json`,
        { fetchImpl, headers: { "User-Agent": secAgent() }, timeoutMs: 45000 });
      return summarizeFacts(facts);
    }, { refresh });
  }

  // Schwab: quote + fundamentals through the engine's read-only client (market-data URLs only; /trader/ refused).
  function schwabQuotes(tickers, { refresh } = {}) {
    const key = `schwab:${[...tickers].sort().join(",")}`;
    return store.cached(key, "SCHWAB_QUOTE", tickers.length === 1 ? tickers[0] : null, MIN, () => runSchwab(tickers), { refresh });
  }
  function runSchwab(tickers) {
    const helper = join(fileURLToPath(new URL(".", import.meta.url)), "schwab_snapshot.py");
    const venv = join(root, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
    const exe = existsSync(venv) ? venv : python;
    return new Promise((resolve, reject) => {
      const child = spawn(exe, ["-u", helper, ...tickers], { cwd: join(root, "engine"), windowsHide: true,
        env: { ...process.env, PYTHONUTF8: "1" } });
      let out = "", err = "";
      const timer = setTimeout(() => { child.kill(); reject(new Error("Schwab snapshot timed out")); }, 30000);
      child.stdout.on("data", (d) => { out += d; });
      child.stderr.on("data", (d) => { err += d; });
      child.on("error", (e) => { clearTimeout(timer); reject(new Error("Could not start Python: " + e.message)); });
      child.on("close", () => {
        clearTimeout(timer);
        let parsed; try { parsed = JSON.parse(out); } catch { return reject(new Error("Schwab snapshot failed: " + scrub(err).slice(-300))); }
        if (parsed.status !== "ok") return reject(new Error(parsed.message || "Schwab unavailable"));
        resolve(parsed.quotes);
      });
    });
  }

  return { news, details, dividends, splits, search, secFinancials, schwabQuotes, hasMassive: () => Boolean(massiveKey()) };
}

// ---------------------------------------------------------------- SEC XBRL -> compact financial summary
const CONCEPTS = {
  revenue: ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
  grossProfit: ["GrossProfit"],
  operatingIncome: ["OperatingIncomeLoss"],
  netIncome: ["NetIncomeLoss"],
  cfo: ["NetCashProvidedByUsedInOperatingActivities"],
  capex: ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
  buybacks: ["PaymentsForRepurchaseOfCommonStock"],
  dividendsPaid: ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
  interestExpense: ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
  dilutedShares: ["WeightedAverageNumberOfDilutedSharesOutstanding"],
};
const INSTANTS = {
  cash: ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
  shortInvestments: ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
  longTermDebt: ["LongTermDebtNoncurrent", "LongTermDebt"],
  currentDebt: ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
  equity: ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
  assets: ["Assets"],
  liabilities: ["Liabilities"],
};
const days = (a, b) => (Date.parse(b) - Date.parse(a)) / DAY;

function series(gaap, names, kind) {
  for (const name of names) {
    const units = gaap[name]?.units;
    const facts = units?.USD || units?.shares || units?.["USD/shares"];
    if (!facts?.length) continue;
    const byEnd = new Map();
    for (const f of facts) {
      if (!/^10-[QK]/.test(f.form || "")) continue;
      if (kind === "flow") {
        if (!f.start) continue;
        const len = days(f.start, f.end);
        const period = len >= 80 && len <= 100 ? "Q" : len >= 350 && len <= 380 ? "FY" : null;
        if (!period) continue;
        const k = period + f.end;
        if (!byEnd.has(k) || f.filed > byEnd.get(k).filed) byEnd.set(k, { period, end: f.end, val: f.val, filed: f.filed, concept: name });
      } else {
        if (!byEnd.has(f.end) || f.filed > byEnd.get(f.end).filed) byEnd.set(f.end, { end: f.end, val: f.val, filed: f.filed, concept: name });
      }
    }
    const rows = [...byEnd.values()].sort((a, b) => a.end.localeCompare(b.end));
    if (rows.length && rows.at(-1).end >= new Date(Date.now() - 400 * DAY).toISOString().slice(0, 10)) return rows;
  }
  return [];
}

// Quarterly values, deriving Q4 = FY - (Q1+Q2+Q3) when only the annual figure was reported.
function quarters(rows) {
  const q = rows.filter((r) => r.period === "Q").map((r) => ({ end: r.end, val: r.val, filed: r.filed }));
  for (const fy of rows.filter((r) => r.period === "FY")) {
    if (q.some((x) => Math.abs(days(x.end, fy.end)) < 10)) continue;
    const inside = q.filter((x) => days(x.end, fy.end) > 20 && days(x.end, fy.end) < 300);
    if (inside.length === 3) q.push({ end: fy.end, val: fy.val - inside.reduce((s, x) => s + x.val, 0), derived: true, filed: fy.filed });
  }
  return q.sort((a, b) => a.end.localeCompare(b.end));
}

export function summarizeFacts(facts) {
  const gaap = facts?.facts?.["us-gaap"] || {};
  const out = { entity: facts?.entityName || null, quarterly: {}, latest: {}, derived: {} };
  for (const [key, names] of Object.entries(CONCEPTS)) {
    const q = quarters(series(gaap, names, "flow")).slice(-8);
    if (q.length) out.quarterly[key] = q.map((x) => ({ end: x.end, val: x.val, ...(x.filed ? { filed: x.filed } : {}), ...(x.derived ? { derived: true } : {}) }));
  }
  for (const [key, names] of Object.entries(INSTANTS)) {
    const s = series(gaap, names, "instant");
    if (s.length) out.latest[key] = { end: s.at(-1).end, val: s.at(-1).val };
  }
  const ttm = (key) => { const q = out.quarterly[key]; return q && q.length >= 4 ? q.slice(-4).reduce((s, x) => s + x.val, 0) : null; };
  const priorTtm = (key) => { const q = out.quarterly[key]; return q && q.length >= 8 ? q.slice(-8, -4).reduce((s, x) => s + x.val, 0) : null; };
  const rev = out.quarterly.revenue || [];
  const d = out.derived;
  d.lastQuarterEnd = rev.at(-1)?.end || null;
  d.revenueTTM = ttm("revenue");
  d.revenueGrowthTTM = d.revenueTTM && priorTtm("revenue") ? d.revenueTTM / priorTtm("revenue") - 1 : null;
  d.revenueGrowthLastQuarterYoY = rev.length >= 5 && rev.at(-5).val ? rev.at(-1).val / rev.at(-5).val - 1 : null;
  d.revenueGrowthLastQuarterQoQ = rev.length >= 2 && rev.at(-2).val ? rev.at(-1).val / rev.at(-2).val - 1 : null;
  for (const [name, key] of [["grossMarginTTM", "grossProfit"], ["operatingMarginTTM", "operatingIncome"], ["netMarginTTM", "netIncome"]])
    d[name] = d.revenueTTM && ttm(key) != null ? ttm(key) / d.revenueTTM : null;
  d.cfoTTM = ttm("cfo");
  d.capexTTM = ttm("capex");
  d.fcfTTM = d.cfoTTM != null ? d.cfoTTM - (d.capexTTM || 0) : null;
  d.fcfMarginTTM = d.fcfTTM != null && d.revenueTTM ? d.fcfTTM / d.revenueTTM : null;
  d.buybacksTTM = ttm("buybacks");
  d.dividendsPaidTTM = ttm("dividendsPaid");
  const L = out.latest;
  const debt = (L.longTermDebt?.val || 0) + (L.currentDebt?.val || 0);
  const cash = (L.cash?.val || 0) + (L.shortInvestments?.val || 0);
  d.totalDebt = L.longTermDebt || L.currentDebt ? debt : null;
  d.cashAndInvestments = L.cash ? cash : null;
  d.netDebt = d.totalDebt != null && d.cashAndInvestments != null ? debt - cash : null;
  d.debtToEquity = d.totalDebt != null && L.equity?.val > 0 ? debt / L.equity.val : null;
  const opTTM = ttm("operatingIncome"), intTTM = ttm("interestExpense");
  d.interestCoverage = opTTM != null && intTTM ? opTTM / intTTM : null;
  d.netDebtToOperatingIncome = d.netDebt != null && opTTM > 0 ? d.netDebt / opTTM : null;
  const shares = extractShareSeries(facts);
  out.history = { dilutedShares: shares.dilutedShares, basicShares: shares.basicShares };
  const asof = new Date().toISOString().slice(0, 10);
  const growth = shareGrowthAsof(shares.dilutedShares, null, asof, shares.basicShares);
  d.dilutedShareChangeYoY = growth?.shareChangeYoY ?? null;
  d.dilutedShareChangeRawYoY = growth?.shareChangeRawYoY ?? null;
  d.dilutionSource = growth?.source || null;
  d.dilutionSplitAdjusted = true;
  if (d.dilutedShareChangeYoY == null) {
    const sh = out.quarterly.dilutedShares || [];
    d.dilutedShareChangeYoY = sh.length >= 5 && sh.at(-5).val ? sh.at(-1).val / sh.at(-5).val - 1 : null;
  }
  return out;
}
