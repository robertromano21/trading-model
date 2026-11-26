// Return-attribution panel for /paper2. Reads research_summary.json (via /api/research/summary)
// and, when present, paper_status.json attribution. No secrets, no order endpoints.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 1) => (x == null || !Number.isFinite(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x == null || !Number.isFinite(Number(x)) ? "–" : Number(x).toFixed(d));
const tone = (x) => (x == null || !Number.isFinite(Number(x)) ? "" : Number(x) > 0 ? "pos" : Number(x) < 0 ? "neg" : "");

const PREFERRED = ["walkforward.tsmom", "backtest.core_tsmom", "backtest.combined", "walkforward.combined",
  "paper.default", "daytrade.full_defaults", "backtest.benchmark_spy"];

function capmOf(report) {
  return report?.capm || report?.market || {};
}
function verdictOf(report) {
  return report?.verdict || {};
}
function pick(strategies) {
  for (const name of PREFERRED) if (strategies[name] && (strategies[name].available !== false) && !strategies[name].skipped) return name;
  return Object.keys(strategies).find((n) => strategies[n].available !== false && !strategies[n].skipped) || Object.keys(strategies)[0];
}
function verdictClass(label) {
  if (label === "significant_alpha" || label === "has_skill") return "pos";
  if (label === "significant_negative_alpha" || label === "underperforms_beta_mix" || label === "mostly_market" || label === "cheap_index_fund") return "neg";
  return "";
}
function sectorTilts(report) {
  const betas = report?.sources?.full?.betas || report?.sectorLoadings;
  if (Array.isArray(betas)) {
    return betas.slice(0, 6).map((x) => `${x.ticker} ${num(x.loading)} (t=${num(x.tStat)})`).join(", ") || "–";
  }
  if (!betas || typeof betas !== "object") return "–";
  const tstats = report?.sources?.full?.betaT || {};
  return Object.entries(betas)
    .filter(([k]) => k !== "SPY" && k !== "IEF" && k !== "TLT")
    .sort((a, b) => Math.abs(b[1] || 0) - Math.abs(a[1] || 0))
    .slice(0, 6)
    .map(([k, v]) => `${k} ${num(v)} (t=${num(tstats[k])})`)
    .join(", ") || "–";
}
function row(name, r) {
  if (r.skipped || r.available === false) return [esc(name), "–", "–", "–", "–", "–", "–", esc(r.skipped || r.reason || "")];
  const m = capmOf(r), v = verdictOf(r);
  const sectorAlpha = r.sources?.full?.alphaAnnual ?? r.sector?.alphaAnnual;
  return [esc(r.label || name), num(m.beta ?? m.spyBeta), num(m.r2),
    `<span class="${tone(m.alphaAnnual)}">${pct(m.alphaAnnual)}</span>`,
    `<span class="${tone(m.alphaT)}">${num(m.alphaT)}</span>`, num(m.informationRatio ?? m.appraisalRatio),
    sectorAlpha == null ? "–" : `<span class="${tone(sectorAlpha)}">${pct(sectorAlpha)}</span>`,
    esc(v.label || "")];
}
function blendLine(report) {
  const simple = report?.betaMatchedBlend?.simple || report?.betaMatched;
  if (!simple) return "";
  const strat = simple.strategy || {};
  const mix = simple.blend || simple.cashMix || {};
  if (strat.sharpe == null && mix.sharpe == null) return "";
  return `<p class="muted">Strategy Sharpe ${num(strat.sharpe)} / CAGR ${pct(strat.cagr)} vs frictionless β-matched SPY+IEF Sharpe ${num(mix.sharpe)} / CAGR ${pct(mix.cagr)}.</p>`;
}
function brinsonLine(report) {
  const br = report?.brinson?.vsSpy || report?.brinson;
  const ann = br?.annualized || br?.cumulative;
  if (!ann) return "";
  return `<p class="muted">Brinson-Fachler: allocation ${pct(ann.allocation)}, selection ${pct(ann.selection)}, interaction ${pct(ann.interaction)}.</p>`;
}

function render(host, payload, paper) {
  if (!host) return;
  const strategies = { ...(payload?.series || payload?.strategies || {}) };
  if (paper && paper.available !== false && !paper.skipped && !paper.error) {
    strategies["paper.default"] = paper;
  }
  if (!Object.keys(strategies).length) {
    host.innerHTML = `<p class="muted">No attribution yet. Run the “Research: return attribution” job (or <code>python attribution.py</code> from engine/).</p>`;
    return;
  }
  const selected = pick(strategies);
  const report = strategies[selected] || {};
  const m = capmOf(report), v = verdictOf(report);
  const cards = (report.skipped || report.available === false) ? `<p class="muted">${esc(report.skipped || report.reason || "")}</p>` : `
    <div class="cards">
      <div class="card"><div class="label">Focus</div><div class="value small">${esc(report.label || selected)}</div></div>
      <div class="card"><div class="label">SPY beta</div><div class="value">${num(m.beta ?? m.spyBeta)}</div></div>
      <div class="card"><div class="label">Market R²</div><div class="value">${num(m.r2)}</div></div>
      <div class="card"><div class="label">Alpha (ann.)</div><div class="value ${tone(m.alphaAnnual)}">${pct(m.alphaAnnual)}</div></div>
      <div class="card"><div class="label">Alpha t (HAC)</div><div class="value ${tone(m.alphaT)}">${num(m.alphaT)}</div></div>
      <div class="card"><div class="label">Information ratio</div><div class="value ${tone(m.informationRatio ?? m.appraisalRatio)}">${num(m.informationRatio ?? m.appraisalRatio)}</div></div>
    </div>
    <p class="${verdictClass(v.label)}">${esc(v.summary || v.text || "")}</p>
    <p class="muted">Largest sector/other tilts (incremental to SPY): ${esc(sectorTilts(report))}</p>
    ${blendLine(report)}
    ${brinsonLine(report)}
  `;
  const names = Object.keys(strategies);
  host.innerHTML = `
    <p class="muted">Market and sector-ETF regression of daily returns with HAC/Newey-West t-stats. Generated ${esc(payload?.generatedAt || "")}.</p>
    ${cards}
    <table id="attr-table"><thead><tr>
      <th>Strategy</th><th>SPY β</th><th>R²</th><th>α ann.</th><th>α t</th><th>IR</th><th>Sources α</th><th>Verdict</th>
    </tr></thead><tbody>
      ${names.map((n) => `<tr>${row(n, strategies[n]).map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}
    </tbody></table>`;
}

async function loadAttribution() {
  const host = $("attribution-panel");
  if (!host) return;
  try {
    const [summary, paperStatus] = await Promise.all([
      fetch("/api/research/summary", { cache: "no-store" }).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
      fetch("/api/paper2/status", { cache: "no-store" }).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
    ]);
    render(host, summary.attribution, paperStatus.attribution);
  } catch (error) {
    host.innerHTML = `<p class="muted">${esc(error.message)}</p>`;
  }
}

loadAttribution();
export { loadAttribution, render };
