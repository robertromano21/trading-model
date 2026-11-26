// Compact, prioritized prompt context. The INPUT is budgeted (TRADING_AI_CONTEXT_CHARS, default ~60k chars ≈ 15k
// tokens) by choosing the most relevant pieces; the model's OUTPUT is never limited.
import { readFileSync, statSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { cryptoBlock, isCryptoQuestion } from "./crypto.js";
const massiveDir = (dataDir) => dataDir || process.env.PRICES_DATA_DIR || fileURLToPath(new URL("../../Massive", import.meta.url));
const readJsonSafe = (path) => { try { return JSON.parse(readFileSync(path, "utf8")); } catch { return null; } };
const pct = (x, d = 1) => (x == null ? "n/a" : (x * 100).toFixed(d) + "%");
const num = (x, d = 2) => (x == null ? "n/a" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d }));
const big = (x) => {
  if (x == null) return "n/a";
  const a = Math.abs(x);
  return (x < 0 ? "-" : "") + (a >= 1e12 ? (a / 1e12).toFixed(2) + "T" : a >= 1e9 ? (a / 1e9).toFixed(2) + "B" : a >= 1e6 ? (a / 1e6).toFixed(1) + "M" : a.toFixed(0));
};

export function priceBlock(t, f) {
  if (!f) return `[PRICE_HISTORY] ${t}: no local price history.`;
  return [`[PRICE_HISTORY] ${t} — last local bar ${f.lastBarDate} (${f.sessions} sessions). Price used: ${num(f.price)} (${f.priceSource}, as of ${f.priceAsOf}).`,
    `Returns: 1d ${pct(f.ret1)}, 5d ${pct(f.ret5)}, 20d ${pct(f.ret20)}, 60d ${pct(f.ret60)}, 100d ${pct(f.ret100)}, 250d ${pct(f.ret250)}.`,
    `Moving averages: SMA20 ${num(f.sma20)} (${pct(f.distSma20)}), SMA50 ${num(f.sma50)} (${pct(f.distSma50)}), SMA100 ${num(f.sma100)} (${pct(f.distSma100)}), SMA200 ${num(f.sma200)} (${pct(f.distSma200)}).`,
    `Stretch: z-score vs 20d ${num(f.zPrice20)}, (price-SMA50)/ATR ${num(f.atrDistance50)}, RSI2 ${num(f.rsi2, 1)}, RSI14 ${num(f.rsi14, 1)} -> code classification ${f.overextension}; trend classification ${f.trend}.`,
    `Volatility: ATR14 ${num(f.atr14)} (${pct(f.atrPct, 2)} of price), realized vol 20d ${pct(f.vol20)}, 60d ${pct(f.vol60)}; daily skew ${num(f.dailySkew252)}, excess kurtosis ${num(f.dailyExKurt252)} (fat tails if > 0).`,
    `Range: 52w high ${num(f.high252)} (${pct(f.fromHigh252)} from it), 52w low ${num(f.low252)}; max drawdown 1y ${pct(f.maxDrawdown252)}. Donchian20 ${f.donchian20 ? num(f.donchian20.low) + "-" + num(f.donchian20.high) : "n/a"}, Donchian55 ${f.donchian55 ? num(f.donchian55.low) + "-" + num(f.donchian55.high) : "n/a"}.`,
    `Momentum signs: TSMOM50 ${f.tsmom50}, TSMOM100 ${f.tsmom100}; relative strength vs SPY 60d ${pct(f.rs60VsSpy)}, 120d ${pct(f.rs120VsSpy)}; beta ${num(f.beta120)}, corr ${num(f.corrSpy120)}.`,
    `Liquidity / gap risk: 20d avg dollar volume ${big(f.dollarVolume20)}, last volume vs 20d avg ${num(f.volumeRatio)}x; overnight gaps > 3 sd in last year: ${f.bigGaps252}, worst gap ${pct(f.worstGap252)}.`].join("\n");
}

export function quoteBlock(t, q, fetchedAt) {
  if (!q) return null;
  const when = q.quoteTimeMs ? new Date(q.quoteTimeMs).toISOString() : fetchedAt;
  return [`[SCHWAB_QUOTE] ${t} at ${when}: last ${num(q.last)} (bid ${num(q.bid)} / ask ${num(q.ask)}), day ${num(q.low)}-${num(q.high)}, prev close ${num(q.prevClose)}, change ${num(q.changePct)}%, volume ${big(q.volume)}; regular-session close ${num(q.regularClose)} (${num(q.regularChangePct)}%).`,
    `[SCHWAB_FUNDAMENTAL] ${t}: P/E ${num(q.peRatio)}, EPS ${num(q.eps)}, dividend yield ${num(q.divYield)}%, next ex-div ${q.divExDate?.slice(0, 10) || "n/a"}, last earnings ${q.lastEarningsDate?.slice(0, 10) || "n/a"}, avg volume 10d ${big(q.avg10DaysVolume)} vs 1y ${big(q.avg1YearVolume)}, shares ${big(q.sharesOutstanding)}.`].join("\n");
}

export function financialBlock(t, fin, fetchedAt) {
  if (!fin) return null;
  const d = fin.derived || {}, q = fin.quarterly || {};
  const rev = (q.revenue || []).slice(-5).map((x) => `${x.end}: ${big(x.val)}`).join("; ");
  const ni = (q.netIncome || []).slice(-5).map((x) => `${x.end}: ${big(x.val)}`).join("; ");
  return [`[SEC_FINANCIALS] ${t} (${fin.entity || ""}) from SEC XBRL filings, fetched ${fetchedAt}; latest quarter ended ${d.lastQuarterEnd || "n/a"}.`,
    `Revenue by quarter: ${rev || "n/a"}. TTM revenue ${big(d.revenueTTM)}, TTM growth ${pct(d.revenueGrowthTTM)}, last quarter YoY ${pct(d.revenueGrowthLastQuarterYoY)}, QoQ ${pct(d.revenueGrowthLastQuarterQoQ)}.`,
    `Net income by quarter: ${ni || "n/a"}. TTM margins: gross ${pct(d.grossMarginTTM)}, operating ${pct(d.operatingMarginTTM)}, net ${pct(d.netMarginTTM)}.`,
    `Cash flow TTM: operating ${big(d.cfoTTM)}, capex ${big(d.capexTTM)}, free cash flow ${big(d.fcfTTM)} (${pct(d.fcfMarginTTM)} of revenue); buybacks ${big(d.buybacksTTM)}, dividends paid ${big(d.dividendsPaidTTM)}.`,
    `Balance sheet: total debt ${big(d.totalDebt)}, cash+short-term investments ${big(d.cashAndInvestments)}, net debt ${big(d.netDebt)}, debt/equity ${num(d.debtToEquity)}, net debt/operating income ${num(d.netDebtToOperatingIncome)}, interest coverage ${num(d.interestCoverage)}x; diluted share count change YoY ${pct(d.dilutedShareChangeYoY)}` +
    (d.dilutionFlag ? ` — DILUTION FLAG ${d.dilutionFlag}` + (d.dilutionOffering ? ` (offering ${d.dilutionOffering.form} filed ${d.dilutionOffering.filingDate})` : "") + ". 10%+ split-adjusted growth is a standalone RED flag; an active offering also blocks new buys." : ".")].join("\n");
}

export function detailsBlock(t, d) {
  if (!d) return null;
  return `[TICKER_DETAILS] ${t}: ${d.name || ""}; ${d.type || ""} on ${d.exchange || ""}; industry ${d.industry || "n/a"}; market cap ${big(d.marketCap)}; employees ${num(d.employees, 0)}.` +
    (d.description ? ` Business: ${String(d.description).slice(0, 600)}` : "");
}

export function newsBlock(t, items, fetchedAt, maxItems) {
  if (!items?.length) return `[MASSIVE_NEWS] ${t}: no articles in the window (fetched ${fetchedAt}).`;
  // Press releases first, then most recent; descriptions are trimmed in the INPUT only.
  const ranked = [...items].sort((a, b) => (b.pressRelease - a.pressRelease) || b.published.localeCompare(a.published)).slice(0, maxItems)
    .sort((a, b) => b.published.localeCompare(a.published));
  return [`[MASSIVE_NEWS] ${t}: ${ranked.length} of ${items.length} articles (fetched ${fetchedAt}); PR = company press release.`,
    ...ranked.map((n) => `- ${n.published.slice(0, 10)} ${n.pressRelease ? "PR " : ""}${n.publisher}: ${n.title}` +
      (n.description ? ` — ${n.description.slice(0, 280)}` : "") + (n.sentiment ? ` [sentiment ${n.sentiment}${n.sentimentReason ? ": " + n.sentimentReason.slice(0, 160) : ""}]` : ""))].join("\n");
}

export function eventsBlock(t, divs, splits) {
  const parts = [];
  if (divs?.length) parts.push(`dividends: ${divs.slice(0, 4).map((d) => `${d.exDate} $${d.cash}`).join(", ")}`);
  if (splits?.length) parts.push(`splits: ${splits.map((s) => `${s.date} ${s.from}:${s.to}`).join(", ")}`);
  return parts.length ? `[DIVIDENDS/SPLITS] ${t}: ${parts.join("; ")}.` : null;
}

export function bookBlock(book, div) {
  if (!book?.available) return "[PAPER_BOOK] No paper-trading ledger yet (Massive/paper_trading.sqlite3 missing).";
  const pos = book.positions.map((p) => `${p.ticker} ${num(p.shares, 0)} sh @${num(p.avg_cost)} (${pct(p.weight)})`).join(", ") || "none";
  return `[PAPER_BOOK] account ${book.account}: equity ${num(book.equity)}, cash ${num(book.cash)}; positions: ${pos}.` +
    (div ? ` Diversification check for this name: ${div.status}; sector peers held ${div.sectorPeers.join(", ") || "none"} (weight ${pct(div.sectorWeight)}); most correlated holding ${div.mostCorrelated ? div.mostCorrelated.ticker + " corr " + div.mostCorrelated.corr : "n/a"}. Limits: sector ${pct(div.limits.maxSectorWeight)}, pair corr ${div.limits.maxCorr}.` : "");
}

export function regimeBlock(r) {
  if (!r) return null;
  const lines = [];
  if (r.asOf || r.spyPrice != null) {
    lines.push(`[MARKET_REGIME] as of ${r.asOf}: SPY ${num(r.spyPrice)}, ${r.spyAbove200 ? "above" : "below"} its 200d average (${pct(r.spyDistSma200)}), 20d return ${pct(r.spyRet20)}, 20d vol ${pct(r.spyVol20)} (1y percentile ${pct(r.spyVolPercentile1y, 0)}) -> ${r.regime}.`);
  }
  const hmm = r.hmm;
  if (hmm) {
    const probs = Object.entries(hmm.probabilities || {}).map(([k, v]) => `${k} ${pct(v, 0)}`).join(", ");
    const a = hmm.allocation || {};
    const f = hmm.features || {};
    lines.push(`[HMM_REGIME] as of ${hmm.asOf}: most likely ${hmm.regime || "n/a"} (${hmm.model}, ${hmm.nStates} states); filtered probabilities ${probs || "n/a"}. Allocation: TSMOM scale ${pct(a.tsmomScale)}, IEF ${pct(a.ief)}, TLT ${pct(a.tlt)}, mean-reversion overlay ${a.meanrevOn ? "ON" : "OFF"} (score ${num(a.meanrevScore)}). Features: SPY 20d vol ${pct(f.spyVol20)} (percentile ${pct(f.spyVolPercentile)}), ${pct(f.pctAbove50)} of the universe above its 50d average, ${pct(f.pctAbove200)} above its 200d, advance/decline ${num(f.advanceDecline)}, credit proxy ${hmm.creditProxy || "n/a"} ${pct(f.creditExcess20)}.`);
  }
  return lines.length ? lines.join("\n") : null;
}

// [ANALYST_SCORECARD] The analyst's own scored track record (engine/analyst_scoring.py writes promptSummary into
// Massive/Research/analyst_scorecard.json after each scoring run) so verdicts stay calibrated. Cached by mtime.
let scorecardCache = { path: null, mtime: 0, text: null };
export function scorecardBlock(path = process.env.ANALYST_SCORECARD_PATH ||
  join(process.env.PRICES_DATA_DIR || fileURLToPath(new URL("../../Massive", import.meta.url)), "Research", "analyst_scorecard.json")) {
  try {
    const mtime = statSync(path).mtimeMs;
    if (scorecardCache.path !== path || scorecardCache.mtime !== mtime) {
      scorecardCache = { path, mtime, text: JSON.parse(readFileSync(path, "utf8")).promptSummary || null };
    }
    return scorecardCache.text;
  } catch {
    return null;
  }
}

// [SWARM] Leaderboard of the simulated $10k shadow accounts (engine/swarm.py writes promptSummary into
// Massive/Research/swarm/swarm_status.json): champion vs TSMOM / SPY on the sealed holdout, PBO, deflated Sharpe, SPA.
let swarmCache = { path: null, mtime: 0, text: null };
let macroCache = { path: null, mtime: 0, text: null };
export function macroBlock(path = process.env.MACRO_SNAPSHOT_PATH ||
  join(process.env.PRICES_DATA_DIR || fileURLToPath(new URL("../../Massive", import.meta.url)), "Research", "macro_snapshot.json")) {
  try {
    const outlookPath = join(dirname(path), "dxy_forecast_outlook.json");
    const mtime = statSync(path).mtimeMs + (existsSync(outlookPath) ? statSync(outlookPath).mtimeMs : 0);
    if (macroCache.path !== path || macroCache.mtime !== mtime) {
      const snap = JSON.parse(readFileSync(path, "utf8"));
      if (!snap?.available) macroCache = { path, mtime, text: null };
      else {
        const dx = snap.dxy || {}, rates = snap.rates || {}, credit = snap.credit || {};
        const trend = dx.trend || "n/a";
        const pctn = (x, d=1) => (x == null ? "n/a" : (Number(x) * 100).toFixed(d) + "%");
        const pp = (x, d=2) => (x == null ? "n/a" : Number(x).toFixed(d) + "pp");
        let text = `[MACRO] as of ${snap.asOf}: DXY-style proxy ${num(dx.level)} (${trend}, 20d ${pctn(dx.ret20)}, 20d vol ${pctn(dx.vol20)}); `
          + `Treasury 10Y ${num(rates.dgs10)}%, 2Y ${num(rates.dgs2)}%, slope 10Y-2Y ${pp(rates.curveSlope10y2y)}, 10Y 20d change ${pp(rates.curveLevelChg20)}; `
          + `SOFR ${num(rates.sofr)}%, SOFR-3M Treasury ${pp(rates.sofrTbillSpread)} (funding proxy, not a swap spread); `
          + `HY OAS ${num(credit.hyOas)}% (20d ${pp(credit.hyOasChg20)}), IG OAS ${num(credit.igOas)}%, BBB ${num(credit.bbbOas)}%, CCC ${num(credit.cccOas)}%. `
          + (snap.proxyDisclaimer || snap.note || "Real CDS/CDX and ICE swap quotes are proprietary; these are free-data proxies.");
        let outlook = snap.dxyOutlook;
        if (!outlook && existsSync(outlookPath)) {
          try { outlook = JSON.parse(readFileSync(outlookPath, "utf8")); } catch { outlook = null; }
        }
        if (outlook && (outlook.expectedMove != null || outlook.pStronger != null || outlook.direction)) {
          const band = Array.isArray(outlook.ci80) ? ` 80% band [${outlook.ci80.map((x) => pctn(x, 2)).join(", ")}]` : "";
          const alert = outlook.alert?.flag ? ` ALERT: ${outlook.alert.reason}.` : "";
          text += ` Dollar outlook (recommend only, trading default off): as of ${outlook.asOf || "n/a"}, 20-session expected move ${pctn(outlook.expectedMove, 2)}${band}, P(stronger) ${pctn(outlook.pStronger, 1)}, direction ${outlook.direction || "n/a"}, confidence ${outlook.confidence || "n/a"}, gates ${outlook.gatesPassed ? "passed" : "not passed"}.${alert}`;
        }
        macroCache = { path, mtime, text };
      }
    }
    return macroCache.text;
  } catch {
    return null;
  }
}

export function swarmBlock(path = process.env.SWARM_STATUS_PATH ||
  join(process.env.PRICES_DATA_DIR || fileURLToPath(new URL("../../Massive", import.meta.url)), "Research", "swarm", "swarm_status.json")) {
  try {
    const mtime = statSync(path).mtimeMs;
    if (swarmCache.path !== path || swarmCache.mtime !== mtime) {
      swarmCache = { path, mtime, text: JSON.parse(readFileSync(path, "utf8")).promptSummary || null };
    }
    return swarmCache.text;
  } catch {
    return null;
  }
}

let macroEventsCache = { path: null, mtime: 0, text: null };
export function macroEventsBlock(path = process.env.MACRO_CALENDAR_PATH ||
  join(process.env.PRICES_DATA_DIR || fileURLToPath(new URL("../../Massive", import.meta.url)), "Macro", "calendar.json")) {
  try {
    const mtime = statSync(path).mtimeMs;
    if (macroEventsCache.path !== path || macroEventsCache.mtime !== mtime) {
      const doc = JSON.parse(readFileSync(path, "utf8"));
      const now = Date.now();
      const upcoming = (doc.events || []).filter((e) => {
        const t = e.datetimeEt ? Date.parse(e.datetimeEt) : NaN;
        return Number.isFinite(t) && t >= now && t <= now + 7 * 86400 * 1000;
      }).slice(0, 16);
      const printed = (doc.events || []).filter((e) => e.actual != null || e.status === "printed").slice(-8).reverse();
      const lines = ["[MACRO_EVENTS] Upcoming releases (times in ET):"];
      if (upcoming.length) {
        for (const e of upcoming) {
          lines.push(`  ${e.datetimeEt} ${e.name}${e.consensus != null ? ", consensus " + e.consensus : ""}${e.entry === "manual" ? " (manual entry)" : ""}`);
        }
      } else {
        lines.push("  No upcoming releases in the next week (calendar may need a refresh).");
      }
      if (printed.length) {
        lines.push("Latest prints (z is standardized vs history; positive is hawkish when hawkishIf=above):");
        for (const e of printed) {
          lines.push(`  ${e.datetimeEt || e.date} ${e.name}: actual ${e.actual}, consensus ${e.consensus}, z=${e.surpriseZ ?? "n/a"}`);
        }
      }
      lines.push("Retail systems usually cannot trade the first tick; most of the liquid move often happens within seconds of the print. Spreads widen around high-impact releases.");
      macroEventsCache = { path, mtime, text: lines.join("\n") };
    }
    return macroEventsCache.text;
  } catch {
    return null;
  }
}

function fitBlocks(blocks, budget) {
  const out = [];
  let used = 0;
  for (const b of blocks) {
    if (!b) continue;
    const sep = out.length ? 2 : 0;
    if (used + sep + b.length <= budget) { out.push(b); used += sep + b.length; continue; }
    const room = budget - used - sep;
    if (room > 200) { out.push(b.slice(0, room - 1) + "…"); }
    break;
  }
  return out.join("\n\n");
}

export function allocationBlock(dataDir) {
  const s = readJsonSafe(join(massiveDir(dataDir), "Research", "regime_allocation", "current_state.json"));
  if (!s || !(s.available || s.recommendation)) return null;
  const rec = s.recommendation || {}, g = rec.gates || {};
  const n = (x, d = 2) => (x == null || x === "" ? "n/a" : Number(x).toLocaleString("en-US", { maximumFractionDigits: d }));
  return `[REGIME_ALLOCATION] as of ${s.asOf || "n/a"}: vol ${s.volBucket || "n/a"} ${s.volDirection || ""}, overextension score ${n(s.overextension, 3)}.`
    + ` Champion ${rec.champion || "n/a"} vs ${rec.baseline || "TSMOM"}; robust ${rec.robust ? "yes" : "no"}`
    + ` (deflated Sharpe p ${n(g.deflatedSharpeProb, 3)}, PBO ${n(g.pbo, 2)}, SPA vs TSMOM ${n(g.spaPVsTsmom, 2)}).`
    + ` Decision: ${rec.decision || "n/a"}. ${rec.text || ""}`
    + (rec.livePaperUnchanged ? " Live paper account is unchanged." : "");
}

export function blackSwanContextBlock(dataDir) {
  const dir = join(massiveDir(dataDir), "Research", "black_swan");
  const state = readJsonSafe(join(dir, "state.json"));
  if (!state) return null;
  const study = readJsonSafe(join(dir, "study.json")) || {};
  const wf = study.walkForward || {};
  const honesty = wf.honesty || study.honesty || "";
  return `[BLACK_SWAN] ${state.state || "UNKNOWN"} (${state.direction || "n/a"}). ${state.headline || ""}`
    + (state.policy ? ` Playbook: gross x${state.policy.scaleGross ?? 1}, freezeLongs=${!!state.policy.freezeNewLongs}; alerts only unless gates pass.` : "")
    + (honesty ? ` Event-study: ${String(honesty).slice(0, 300)}` : "");
}

export function dxyOutlookBlock(dataDir) {
  const o = readJsonSafe(join(massiveDir(dataDir), "Research", "dxy_forecast_outlook.json"));
  if (!o || (o.expectedMove == null && o.pStronger == null && !o.direction && !o.recommendation)) return null;
  const pctn = (x, d = 2) => (x == null ? "n/a" : (Number(x) * 100).toFixed(d) + "%");
  const band = Array.isArray(o.ci80) ? ` 80% band [${o.ci80.map((x) => pctn(x, 2)).join(", ")}]` : "";
  return `[DXY_FORECAST] as of ${o.asOf || "n/a"}: 20-session expected move ${pctn(o.expectedMove, 2)}${band}, P(stronger) ${pctn(o.pStronger, 1)}, direction ${o.direction || "n/a"}, confidence ${o.confidence || "n/a"}, gates ${o.gatesPassed ? "passed" : "not passed"}, trading default ${o.tradingDefault || "off"}.`
    + (o.recommendation ? ` ${o.recommendation}` : "")
    + (o.alert?.flag ? ` ALERT: ${o.alert.reason}.` : "");
}

/** [MACRO_NOWCAST] our raw-data nowcasts (engine/macro_nowcast.py). Outside estimates are display-only. */
export function nowcastBlock(dataDir, maxChars = 900) {
  const L = readJsonSafe(join(massiveDir(dataDir), "Research", "macro", "nowcast_latest.json"));
  if (!L || !Array.isArray(L.nowcasts) || !L.nowcasts.length) return null;
  const d = (t) => (t === "PAYROLLS" ? 0 : t === "GDP_SAAR" || t === "UNRATE" ? 2 : 2);
  const f = (x, t) => (x == null ? "n/a" : Number(x).toFixed(d(t)));
  const parts = L.nowcasts.filter((n) => n.nowcast != null).map((n) => {
    const out = Object.entries(n.outside || {}).map(([k, v]) => `${k} ${f(v?.value, n.target)}`).join(", ");
    return `${n.target} ${n.period}: ${f(n.nowcast, n.target)} (80% ${n.band80 ? n.band80.map((x) => f(x, n.target)).join("..") : "n/a"}; last ${f(n.lastPrint, n.target)}; next ${n.nextRelease || n.calendarNextRelease || n.estimatedRelease}${out ? `; outside ${out}` : ""})`;
  });
  const text = `[MACRO_NOWCAST] Our nowcasts as of ${L.asof} from raw point-in-time data (outside forecasts shown for comparison only, never inputs). `
    + `Regime: ${L.regime?.label || "n/a"}. ${parts.join("; ")}. Research only; the paper accounts are unchanged.`;
  return text.length <= maxChars ? text : text.slice(0, maxChars - 1) + "…";
}

const RESEARCH_TOPICS = [
  { id: "regime_allocation", re: /\b(over-?extension|overextended|allocation|hedge|state[- ]dependent|stretch|vol(?:atility)?[- ]target)\b/i, load: allocationBlock },
  { id: "black_swan", re: /\b(black[- ]swan|geopolit|shock|war|sanction|invasion|crisis)\b/i, load: blackSwanContextBlock },
  { id: "macro_nowcast", re: /\b(nowcast|economy|economic|recession|gdp|cpi|inflation|pce|payrolls?|jobs report|unemployment|macro)\b/i, load: nowcastBlock },
  { id: "dxy_forecast", re: /\b(dollar|dxy|currency|fx|\busd\b)\b/i, load: dxyOutlookBlock },
  { id: "crypto", re: /\b(crypto\w*|bitcoin|btc|ether(?:eum)?|eth|solana|altcoins?)\b/i, load: (dir) => cryptoBlock(dir) },
  { id: "swarm", re: /\b(swarm|champion|shadow account|pbo|holdout|walk-?forward)\b/i, load: (dir) => swarmBlock(join(massiveDir(dir), "Research", "swarm", "swarm_status.json")) },
];

/** Local study summaries for a general question. Used by the no-key fallback and by RESEARCH with no ticker. */
export function researchFallbackText(question, { dataDir, budgetChars = 3500 } = {}) {
  const q = String(question || "");
  const loaded = RESEARCH_TOPICS.map((topic) => ({ ...topic, text: topic.load(dataDir) })).filter((t) => t.text);
  const matched = loaded.filter((t) => t.re.test(q));
  const chosen = matched.length ? matched : loaded;
  if (!chosen.length) return "";
  const header = matched.length ? "Local research summaries that match the question:" : "Local research summaries on file (no close keyword match):";
  return fitBlocks([header, ...chosen.map((t) => t.text)], budgetChars);
}

/** research: output of advisor.research(); returns the context string within the character budget. */
export function buildContext(researchList, { budgetChars = Number(process.env.TRADING_AI_CONTEXT_CHARS || 60000), regime, book, dataDir, question = "" } = {}) {
  const list = Array.isArray(researchList) ? researchList : [];
  const root = massiveDir(dataDir);
  const reserved = Math.min(8000, Math.max(1200, Math.floor(budgetChars * 0.35)));
  const globals = fitBlocks([
    isCryptoQuestion(question) ? cryptoBlock(dataDir) : null,   // crypto questions: market + paper accounts first
    regimeBlock(regime),
    allocationBlock(dataDir),
    blackSwanContextBlock(dataDir),
    macroBlock(join(root, "Research", "macro_snapshot.json")),
    nowcastBlock(dataDir, 700),
    macroEventsBlock(join(root, "Macro", "calendar.json")),
    scorecardBlock(join(root, "Research", "analyst_scorecard.json")),
    swarmBlock(join(root, "Research", "swarm", "swarm_status.json")),
    list.length ? null : bookBlock(book),
  ].filter(Boolean), reserved);
  const remain = Math.max(800, budgetChars - (globals ? globals.length + 2 : 0));
  const perTicker = Math.max(800, Math.floor(remain / Math.max(1, list.length)));
  const blocks = globals ? [globals] : [];
  for (const r of list) {
    const fixed = [priceBlock(r.ticker, r.features), quoteBlock(r.ticker, r.quote?.value, r.quote?.fetchedAt),
      financialBlock(r.ticker, r.financials?.value, r.financials?.fetchedAt), detailsBlock(r.ticker, r.details?.value),
      eventsBlock(r.ticker, r.dividends?.value, r.splits?.value), bookBlock(book, r.diversification),
      r.gaps.length ? `[DATA_GAPS] ${r.ticker}: ${r.gaps.join("; ")}` : null].filter(Boolean).join("\n");
    let maxNews = 25, news;
    do { news = newsBlock(r.ticker, r.news?.value, r.news?.fetchedAt, maxNews); maxNews -= 3; }
    while (fixed.length + news.length > perTicker && maxNews > 0);
    const chunk = (fixed + "\n" + news).slice(0, perTicker);
    blocks.push(chunk);
  }
  const text = fitBlocks(blocks, budgetChars);
  return text.length <= budgetChars ? text : text.slice(0, budgetChars);
}
