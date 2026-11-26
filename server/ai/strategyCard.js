// The analyst's MASTER system prompt, generated from the engine code and measured research so it stays in sync:
// engine/quant_tools.py strategy_card reads strategy classes (docstrings + defaults), ORBParams, paper-trader
// defaults, risk/cost settings, research.py grids and Massive/Research/*.json results. The card is cached in
// Massive/ai_strategy_card.json and rebuilt whenever any of those source files is newer (or after 12 h).
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync, renameSync } from "node:fs";
import { join } from "node:path";

export const CARD_MAX_AGE_MS = 12 * 3600 * 1000;

export function cardSources(root, dataDir) {
  const engine = join(root, "engine");
  const files = ["intraday.py", "daytrader.py", "paper_trader.py", "risk.py", "costs.py", "universe.py", "research.py",
    "quant_tools.py", "metrics.py"].map((f) => join(engine, f));
  try { for (const f of readdirSync(join(engine, "strategies"))) if (f.endsWith(".py")) files.push(join(engine, "strategies", f)); } catch { /* none */ }
  for (const f of ["research_summary.json", "daytrade_summary.json", "rule_studies.json"]) files.push(join(dataDir, "Research", f));
  return files;
}
const newestMtime = (files) => Math.max(0, ...files.map((f) => { try { return statSync(f).mtimeMs; } catch { return 0; } }));

export function createStrategyCard({ root, dataDir, quant, cacheFile = join(dataDir, "ai_strategy_card.json"), log = () => {} }) {
  let memo = null;
  function readCache() {
    if (memo) return memo;
    try { memo = JSON.parse(readFileSync(cacheFile, "utf8")); } catch { memo = null; }
    return memo;
  }
  function stale(entry) {
    if (!entry?.card) return true;
    const built = Date.parse(entry.builtAt || 0);
    return Date.now() - built > CARD_MAX_AGE_MS || newestMtime(cardSources(root, dataDir)) > built;
  }
  let pending = null;
  async function get({ refresh = false } = {}) {
    const cached = readCache();
    if (!refresh && !stale(cached)) return cached.card;
    if (!quant) return cached?.card || null;
    pending ||= (async () => {
      const res = await quant.run("strategy_card", {}, { timeoutMs: 60000 });
      if (!res.ok) { log("warn", "AI strategy card refresh failed; using the cached card", { error: res.error }); return cached?.card || null; }
      memo = { builtAt: new Date().toISOString(), card: res.result };
      try { writeFileSync(cacheFile + ".tmp", JSON.stringify(memo)); renameSync(cacheFile + ".tmp", cacheFile); } catch { /* memory only */ }
      return memo.card;
    })().finally(() => { pending = null; });
    return pending;
  }
  return { get, cached: () => readCache()?.card || null, cacheFile };
}

// ---------------------------------------------------------------- rendering
const isNum = (x) => typeof x === "number" && Number.isFinite(x);
const pct = (x, d = 1) => (isNum(x) ? `${(x * 100).toFixed(d)}%` : "n/a");
const num = (x, d = 2) => (isNum(x) ? x.toFixed(d) : "n/a");
const kv = (obj) => Object.entries(obj || {}).map(([k, v]) => `${k}=${Array.isArray(v) ? `[${v.join(",")}]` : v === null ? "None" : typeof v === "object" ? JSON.stringify(v) : v}`).join(", ");
const indent = (text, pad = "  ") => String(text || "").split("\n").map((l) => pad + l).join("\n");

function perfLine(name, p) {
  if (!p) return `- ${name}: n/a`;
  return `- ${name}: CAGR ${pct(p.cagr)}, vol ${pct(p.volatility)}, Sharpe ${num(p.sharpe)}${isNum(p.sharpeStdErr) ? ` (±${num(p.sharpeStdErr)} SE)` : ""}, ` +
    `max DD ${pct(p.maxDrawdown)}${isNum(p.turnoverPerYear) ? `, turnover ${num(p.turnoverPerYear, 1)}/yr` : ""}` +
    `${isNum(p.costBpsPerYear) ? `, costs ${num(p.costBpsPerYear, 0)} bps/yr` : ""}${isNum(p.trades) ? `, ${p.trades} trades` : ""}` +
    `${isNum(p.winRate) ? `, win ${pct(p.winRate)}` : ""}`;
}

export function renderStrategies(card) {
  const s = card.strategies || {}, r = card.research || {}, d = card.daytrade || {}, paper = card.paper || {};
  const out = [];
  out.push("## STRATEGIES (rules and parameters read from engine/strategies, engine/intraday.py, engine/paper_trader.py)");
  if (s.tsmom) out.push(`### Core: volatility-targeted 50/100-day time-series momentum (tsmom) — also the ETF trend sleeve`,
    indent(s.tsmom.doc), `  Defaults: ${kv(s.tsmom.defaults)}`, `  Universe (CORE_ETFS): ${(s.tsmom.universe || []).join(", ")}`);
  if (s.meanrev) out.push(`### Overlay: regime-filtered 5-10 day mean reversion (meanrev)`, indent(s.meanrev.doc), `  Defaults: ${kv(s.meanrev.defaults)}`);
  if (s.combined) out.push(`### Combined book (what the swing paper trader runs)`, indent(s.combined.doc), `  Defaults: ${kv(s.combined.defaults)}`,
    `  Paper config: ${JSON.stringify(s.combined.paperConfig)}`);
  if (paper.defaults) out.push(`### Swing paper trader (${paper.doc || "no broker"})`, `  Defaults: ${kv(paper.defaults)}`,
    `  Fills at the next open (or MOC), whole shares, min order $${paper.defaults.min_order_value}; advisory verdicts may only shrink/defer entries (RED blocks, YELLOW x${paper.defaults.advisory_yellow_scale}, out-of-band deferred).`);
  if (d.params) out.push(`### Day trading: regime-filtered opening-range breakout (separate $10,000 'daytrade' paper account)`, indent(d.doc),
    `  Parameters (ORBParams): ${kv(d.params)}`, `  PDT applies below $${d.pdtEquity}; research grid ${JSON.stringify(d.grid)}.`);
  if (r.grids) out.push(`### Research protocol`, `  Warm-up ${r.warmupSessions} sessions; walk-forward ${kv(r.walkForwardDefaults)} (expanding window, best training Sharpe per fold, stitched OOS re-simulated with costs).`,
    `  Grids: ${JSON.stringify(r.grids)}. Horizons studied: ${(r.horizons || []).join(", ")} days.`);
  const cr = card.candidateRules;
  if (cr?.definitions) {
    out.push(`### Candidate strategies NOT implemented in the engine (never traded by the paper trader)`,
      `  Cross-sectional momentum, the SPY 200-day trend filter, the index RSI(2) dip-buy, Donchian breakouts and pairs/stat-arb exist only as`,
      `  quick research studies (RULE_STUDY tool; close-to-close, flat ${3} bps cost, in-sample, not walk-forward validated). Exact study rules:`);
    for (const [k, def] of Object.entries(cr.definitions)) {
      const st = cr.studies?.[k];
      const res = st && !st.error ? ` Measured ${st.window?.join("..")} on ${(st.tickers || []).length > 3 ? "CORE_ETFS" : (st.tickers || []).join("/")}: CAGR ${pct(st.cagr)}, Sharpe ${num(st.sharpe)}, max DD ${pct(st.maxDrawdown)}, in market ${pct(st.timeInMarket, 0)}.` : "";
      out.push(`  - ${k}: ${def}${res}`);
    }
    out.push("  Treat these as ideas to test (run RULE_STUDY / STATS), not as the account's strategies.");
  }
  return out.join("\n");
}

export function renderResults(card) {
  const m = card.measured, dm = card.daytradeMeasured, out = [];
  if (m) {
    out.push(`## MEASURED RESULTS (Massive/Research, generated ${m.generatedAt}; data ${m.data?.start}..${m.data?.end}, ${m.data?.sessions} sessions, ${m.data?.tickers} tickers)`);
    out.push(`Full-period backtests from ${m.evaluationStart} (in-sample defaults, Schwab cost model):`);
    for (const [k, p] of Object.entries(m.backtests || {})) out.push(perfLine(k, p));
    out.push("Walk-forward, out-of-sample only:");
    for (const [k, w] of Object.entries(m.walkForward || {})) out.push(perfLine(`${k} (OOS from ${w.oosStart}, ${w.trials} trials x ${w.folds} folds, train/test rank corr ${JSON.stringify(w.rankCorrByFold)})`, w));
    const mh = m.multiHorizon?.byHorizon;
    if (mh) {
      out.push(`Multi-horizon trend (signal horizon -> own-period Sharpe / common-period Sharpe & CAGR from ${m.multiHorizon.commonPeriodStart}):`);
      out.push("  " + Object.entries(mh).map(([h, v]) => `${h}d: ${num(v.ownPeriod?.sharpe)} / ${num(v.commonPeriod?.sharpe)}, ${pct(v.commonPeriod?.cagr)} (${v.independentSignalPeriods} indep.)`).join("; "));
    }
    if (m.caveats?.length) out.push("Caveats:", ...m.caveats.map((c) => `- ${c}`));
  }
  if (dm) {
    const f = dm.fullPeriodDefaults || {}, n = dm.fullPeriodNoPdt || {}, o = dm.walkForward?.outOfSample || {};
    out.push(`Day trading (ORB) on ${dm.data?.fiveMinuteSessions} five-minute sessions ${dm.data?.first}..${dm.data?.last}, ${(dm.symbols || []).length} symbols:`,
      `- defaults: return ${pct(f.totalReturn)}, Sharpe ${num(f.sharpe)}, max DD ${pct(f.maxDrawdown)}, ${f.trades} trades, win ${pct(f.winRate)}, avg R ${num(f.avgR)}; skipped ${JSON.stringify(f.skipped)}`,
      `- without PDT: return ${pct(n.totalReturn)}, Sharpe ${num(n.sharpe)}, ${n.trades} trades`,
      `- walk-forward OOS ${o.first}..${o.last}: return ${pct(o.totalReturn)}, Sharpe ${num(o.sharpe)}, ${o.trades} trades (chosen ${kv(dm.walkForward?.chosen && { or_minutes: dm.walkForward.chosen.or_minutes, target_r: dm.walkForward.chosen.target_r, stop_frac: dm.walkForward.chosen.stop_frac })})`);
    if (dm.caveats?.length) out.push(...dm.caveats.map((c) => `- ${c}`));
  }
  return out.join("\n");
}

export function renderRisk(card) {
  const r = card.risk || {}, c = card.costs?.schwab || {}, p = card.paper?.defaults || {}, d = card.daytrade?.params || {}, out = [];
  out.push("## RISK RULES (engine/risk.py, paper_trader.py, intraday.py, costs.py)",
    indent(r.doc),
    `- Volatility: blended realized/EWMA vol ${kv(r.blendedVol)}; portfolio vol target ${kv(r.portfolioVolTarget)}.`,
    `- Caps: swing book max ${p.max_positions} positions, ${pct(p.max_name_weight, 0)} per name, gross ${p.max_gross}x (no leverage); tsmom per-name cap ${card.strategies?.tsmom?.defaults?.per_name_cap}, meanrev ${card.strategies?.meanrev?.defaults?.per_name_cap}.`,
    `- Stops: meanrev ${card.strategies?.meanrev?.defaults?.stop_atr} x ATR protective stop, max hold ${card.strategies?.meanrev?.defaults?.max_hold} sessions; tsmom exits on signal loss at rebalances; ORB stop at the range (stop_frac ${d.stop_frac}), target ${d.target_r}R, flat by the close.`,
    `- Circuit breakers: swing — block new buying for ${r.circuitBreaker?.cooldown} session after a daily loss worse than ${pct(p.daily_loss_limit ?? r.circuitBreaker?.daily_loss_limit, 0)}; day trading — stop for the day after ${pct(d.daily_loss_limit, 0)}; stale data ${p.stale_sessions} sessions.`,
    `- Day-trade sizing: risk ${pct(d.risk_per_trade, 2)} of equity per trade, notional <= ${d.max_notional_frac}x equity, ${d.max_positions} position at a time, PDT max ${d.pdt_max_day_trades} day trades per ${d.pdt_window} sessions under $25k.`,
    `- Dilution: split-adjusted diluted (or basic) share-count growth of 10%+ YoY is a standalone RED flag; 5-10% is a warning. A recent S-1/S-3/F-1/F-3, 424B1-5 or 8-K item 3.02 / ATM filing (DILUTION_OFFERING, default 30-day window) also blocks new paper buys. Existing holdings are not sold unless dilution_exit=true (default false). Comparisons use filing dates, not period ends, and ignore stock-split jumps.`,
    `- Tail risk: VaR reported as normal, Student-t, historical and block-bootstrap (${kv(r.valueAtRisk)}); daily returns are fat-tailed, so quote t/historical numbers for 1-10 day risk.`);
  const bt = card.measured?.backtests || {};
  const vars = Object.entries(bt).filter(([, v]) => v.var99_1d).map(([k, v]) => `${k} 99% 1d VaR normal ${pct(v.var99_1d.normal, 2)} vs t ${pct(v.var99_1d.studentT, 2)} vs hist ${pct(v.var99_1d.historical, 2)} (skew ${num(v.dailySkew)}, ex.kurt ${num(v.dailyExcessKurtosis)})`);
  if (vars.length) out.push("- Measured: " + vars.join("; ") + ".");
  out.push(`## COSTS (Schwab profile)`, `- ${card.costs?.doc?.split("\n")[0] || ""}`,
    `- Commission ${c.commission_bps} bps and $${c.fixed_fee} per order on US-listed stocks/ETFs; half-spread clip(1 + ${c.spread_coef}/sqrt(ADV$M), ${c.min_half_spread_bps}, ${c.max_half_spread_bps}) bps; impact ${c.impact_coef} x daily vol x sqrt(order/ADV); SEC fee $${c.sec_fee_per_million}/$1M sold; FINRA TAF $${c.taf_per_share}/share (max $${c.taf_max}).`);
  return out.join("\n");
}

export function renderNormality(card) {
  const n = card.measured?.normality;
  if (!n?.table) return "";
  const rows = n.table.map((t) => `  ${t.horizonDays}d: skew ${num(t.standardized?.skew)}, ex.kurt ${num(t.standardized?.excessKurtosis)}, ` +
    `>2σ ${num(t.standardized?.beyond2SigmaPct)}% (normal 4.55), >3σ ${num(t.standardized?.beyond3SigmaPct)}% (0.27), 1% quantile ${num(t.standardized?.q01)} (-2.33), ` +
    `indep. periods ${t.independentPeriods}; SPY skew ${num(t.spyOverlapping?.skew)}, ex.kurt ${num(t.spyOverlapping?.excessKurtosis)}`);
  return ["## NORMALITY BY HORIZON (vol-standardized pooled returns; JB p = 0 at every horizon)", ...rows,
    "  Reading: 1-10 day returns are strongly fat-tailed with left skew (gaps through stops happen); 20-day returns are closest to normal;",
    "  50-200 day pooled numbers rest on 2-8 independent periods (right-skewed winners) and are descriptive only.",
    "  Use normal approximations only for sizing at 20-60 day horizons; use Student-t / historical / bootstrap for tail risk."].join("\n");
}

export function renderOptions(card) {
  const o = card.options;
  if (!o || o.error) return "";
  const out = ["## OPTIONS, DOWNSIDE AND VOLATILITY SLEEVE (engine/options_*.py, vol_sleeve.py, top_detector.py; PAPER unless Robert confirms a live ticket)"];
  if (o.volSleeveRules) out.push("### Primary vol-sleeve rule (paper account 'options-paper')", indent(o.volSleeveRules));
  const v = o.volSleeve;
  if (v?.oos) out.push(`  Walk-forward OOS ${v.oos.oosStart || ""}: return ${pct(v.oos.totalReturn)}, Sharpe ${num(v.oos.sharpe)}, max DD ${pct(v.oos.maxDrawdown)}, ${v.oos.trades ?? "?"} trades; p(mean>0) ${num(v.pOosMeanReturnGt0)}, p(full exits beat time-only) ${num(v.pFullExitsBeatTimeOnly)}.`,
    `  ${v.honesty || ""}`);
  const t = o.topDetector;
  if (t) out.push("### Top / adverse-regime detector", `  ${t.verdict || ""}`, `  Base rates ${kv(t.baseRates)}; latest ${kv(t.latest)}.`);
  if (o.overlays?.results?.length) {
    out.push("### Options overlays (walk-forward, MODELED option prices, $10k account vs fractional reference)");
    for (const r of o.overlays.results) out.push(`  - ${r.mode} [${r.sizing}]: overlay Sharpe ${num(r.overlay?.sharpe)} vs base ${num(r.base?.sharpe)}, p ${num(r.pOverlayBeatsBase)}, Holm ${num(r.holmP)}`);
    if (o.overlays.honesty) out.push(`  ${o.overlays.honesty}`);
  }
  if (o.optionsLimits) out.push(`### Paper options risk limits: ${kv(o.optionsLimits)}`);
  out.push("### Rules for the analyst",
    "- Defined-risk bearish expressions first (long puts, bear put spreads, SH/PSQ/SDS sized by max loss). Never suggest naked short calls; undefined-risk shorts and short stock are off by default.",
    "- Option prices in backtests are MODELED (Black-Scholes on VIX-scaled vol); say so. A $10k account cannot hold most 100-share covered calls or cash-secured puts on index ETFs.",
    "- Live options/short tickets exist only as Robert's manual ticket -> Schwab preview -> typed CONFIRM flow (LIMIT/NET_DEBIT/NET_CREDIT only, $25k cap). You cannot preview or place orders.");
  return out.join("\n");
}

export function renderMaster(card) {
  if (!card) return "";
  return [`# TRADING MODEL REFERENCE (generated ${card.generatedAt} from engine code + research outputs)`,
    renderStrategies(card), renderResults(card), renderRisk(card), renderNormality(card), renderOptions(card)].filter(Boolean).join("\n\n");
}
