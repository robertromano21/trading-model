// Swarm leaderboard for /paper2: 50 simulated $10k shadow accounts (engine/swarm.py; no orders, ever).
// Shows the top variants by out-of-sample risk-adjusted return, the overfitting statistics and the distribution
// across all accounts. Promoting a variant to the real paper account is a recommendation only.
if (!document.querySelector('link[href^="/swarm-panel.css"]')) {  // external stylesheet: the dashboard CSP blocks inline styles
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = "/swarm-panel.css";
  document.head.appendChild(link);
}
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = (x, d = 2) => (x == null || !Number.isFinite(Number(x)) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x == null || !Number.isFinite(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);
const cls = (x) => (x == null ? "muted" : Number(x) > 0 ? "pos" : Number(x) < 0 ? "neg" : "muted");
let sortKey = "wfOos";

function describe(s) {
  if (!s) return "?";
  if (s.family === "production") return "production config (real paper account)";
  if (s.family === "tsmom_baseline") return "TSMOM baseline 50/100";
  const parts = [`TSMOM ${s.horizons?.join("/")}`, `vol ${s.target_vol}`, `max ${s.max_positions}`, s.rebalance,
    `band ${s.band}`, `regime ${s.regime}`];
  if (s.overlay) parts.push(`MR z${s.mr_entry_z}/${s.mr_lookback}d hold ${s.mr_max_hold}`);
  if (s.trail_atr) parts.push(`trail ${s.trail_atr}ATR`);
  if (s.tax_aware) parts.push("tax-aware");
  return parts.join(", ");
}

function hist(h, label) {
  if (!h || !h.counts?.length) return "";
  const max = Math.max(...h.counts, 1);
  const bars = h.counts.map((c, i) => {
    const lo = h.edges[i], hi = h.edges[i + 1];
    return `<div class="sw-bin" title="${num(lo)} to ${num(hi)}: ${c}" data-w="${100 / h.counts.length}">
      <div class="sw-bar ${(lo + hi) / 2 >= 0 ? "pos" : "neg"}" data-h="${Math.round(60 * c / max)}"></div></div>`;
  }).join("");
  return `<div class="sw-hist"><div class="muted">${esc(label)} (${num(h.edges[0])} … ${num(h.edges[h.edges.length - 1])})</div>
    <div class="sw-hist-bars">${bars}</div></div>`;
}

function testRow(name, t) {
  if (!t) return "";
  return `<tr><td>${esc(name)}</td><td>${num(t.benchmarkSharpe)}</td><td>${num(t.realityCheckP, 3)}</td><td>${num(t.spaP, 3)}</td>
    <td>${t.significantHolm?.length ?? 0}</td><td>${t.significantBH?.length ?? 0}</td><td>${num(t.championP, 3)}</td></tr>`;
}

function value(row, key) {
  if (key === "rank") return -(row.selectionRank ?? 1e9);
  if (key === "holdout") return row.holdout?.sharpe ?? -1e9;
  return row.wfOos?.sharpe ?? -1e9;
}

function render(doc) {
  const box = $("swarm-panel");
  const sw = doc.swing || {}, dt = doc.daytrade || {}, b = doc.benchmarks || {}, dist = doc.distribution || {};
  const champ = (doc.leaderboard || []).find((r) => r.id === sw.champion);
  const verdict = doc.verdict || {};
  const rows = (doc.leaderboard || []).filter((r) => r.ok).sort((a, c) => value(c, sortKey) - value(a, sortKey)).slice(0, 15);
  const dsr = sw.deflatedSharpePre || {}, pbo = sw.pboPre || {};
  const t = sw.tests || {}, dtt = dt.tests || {};
  const fw = doc.forward || {};
  box.innerHTML = `
    <p class="${verdict.anyBeatsBaselineAfterCorrection ? "pos" : "warn"}"><b>${verdict.anyBeatsBaselineAfterCorrection ? "A variant passed every gate" : "Nothing beats the baseline after correction"}</b> —
      ${esc(doc.recommendation)} <span class="muted">(recommendation only; never applied automatically)</span></p>
    <p class="muted">${doc.accounts} accounts × $${Number(doc.initialEquity || 10000).toLocaleString()} · design ${esc(doc.designHash)} ·
      selection on ${esc(doc.design?.evalStart)} → ${esc(doc.design?.holdoutStart)} only · sealed holdout ${esc(doc.design?.holdoutStart)} → ${esc(doc.design?.dataEnd)}
      · holdout looks: ${doc.holdoutViews ?? "–"} · generated ${esc(doc.generatedAt)} · ${num(doc.runtime?.totalSeconds, 0)} s on ${doc.runtime?.workers} workers</p>
    ${doc.realPaperAccount ? `<p class="muted">Real paper account: ${esc(typeof doc.realPaperAccount === "string" ? doc.realPaperAccount : (doc.realPaperAccount.summary || JSON.stringify(doc.realPaperAccount)))}. Account 0 is the earlier production config (combined, hmm regime).</p>` : ""}
    ${doc.growthStudy ? `<p class="${doc.growthStudy.passedAll ? "pos" : "warn"}"><b>Growth-adjusted mean reversion study</b> — ${esc(doc.growthStudy.summary)}
      <span class="muted">(${(doc.growthStudy.forwardExtras || []).length} variants in forward shadow mode)</span></p>` : ""}
    <div class="cards">
      <div class="card"><div class="muted">Champion (picked before the holdout)</div><b>acct ${esc(sw.champion)}</b>
        <div>holdout Sharpe <span class="${cls(champ?.holdout?.sharpe)}">${num(champ?.holdout?.sharpe)}</span>, return ${pct(champ?.holdout?.totalReturn)}</div></div>
      <div class="card"><div class="muted">TSMOM baseline (holdout)</div><b>${num(b.tsmom_baseline?.holdout?.sharpe)}</b> Sharpe, ${pct(b.tsmom_baseline?.holdout?.totalReturn)}</div>
      <div class="card"><div class="muted">SPY (holdout)</div><b>${num(b.spy?.holdout?.sharpe)}</b> Sharpe, ${pct(b.spy?.holdout?.totalReturn)}</div>
      <div class="card"><div class="muted">Deflated Sharpe prob.</div><b>${num(dsr.deflatedSharpeProbability)}</b>
        <div class="muted">${dsr.trials} trials, effective ${num(sw.effectiveTrials, 1)}; E[max SR] ${num(dsr.expectedMaxSharpe)}</div></div>
      <div class="card"><div class="muted">PBO (CSCV, ${pbo.splits ?? "–"} splits)</div><b class="${pbo.pbo > 0.5 ? "neg" : "pos"}">${pct(pbo.pbo, 0)}</b>
        <div class="muted">P(OOS loss) ${pct(pbo.probOosLoss, 0)}, slope ${num(pbo.degradationSlope)}</div></div>
    </div>
    <h3>Multiple-testing tests (swing)</h3>
    <table><tr><th>benchmark : period</th><th>bench Sharpe</th><th>White RC p</th><th>Hansen SPA p</th><th>Holm sig.</th><th>BH sig.</th><th>champion p</th></tr>
      ${testRow("TSMOM baseline : holdout", t["tsmom_baseline:holdout"])}${testRow("SPY : holdout", t["spy:holdout"])}
      ${testRow("TSMOM baseline : pre-holdout", t["tsmom_baseline:preHoldout"])}${testRow("SPY : pre-holdout", t["spy:preHoldout"])}</table>
    <h3>Leaderboard (top 15)
      <select id="swarm-sort">
        <option value="wfOos" ${sortKey === "wfOos" ? "selected" : ""}>by walk-forward out-of-sample Sharpe</option>
        <option value="rank" ${sortKey === "rank" ? "selected" : ""}>by selection rank (pre-holdout)</option>
        <option value="holdout" ${sortKey === "holdout" ? "selected" : ""}>by holdout Sharpe (look only — choosing on this is overfitting)</option>
      </select></h3>
    <table><tr><th>acct</th><th>rank</th><th>config</th><th>WF OOS Sharpe</th><th>holdout Sharpe</th><th>holdout return</th><th>holdout maxDD</th>
      <th>after-tax Sharpe</th><th>turnover/yr</th><th>p vs TSMOM (Holm)</th><th>p vs SPY (Holm)</th><th>day-trade holdout Sharpe</th></tr>
      ${rows.map((r) => `<tr${r.id === sw.champion ? ' class="sw-champ"' : ""}><td>${r.id}${r.eligible ? "" : ' <span class="neg" title="PDT breach: disqualified">PDT</span>'}</td>
        <td>${r.selectionRank ?? "–"}</td><td class="sw-cfg">${esc(describe(r.swing))}</td>
        <td class="${cls(r.wfOos?.sharpe)}">${num(r.wfOos?.sharpe)}</td><td class="${cls(r.holdout?.sharpe)}">${num(r.holdout?.sharpe)}</td>
        <td class="${cls(r.holdout?.totalReturn)}">${pct(r.holdout?.totalReturn)}</td><td>${pct(r.holdout?.maxDrawdown)}</td>
        <td>${num(r.holdout?.afterTaxSharpe)}</td><td>${num(r.turnoverAnnual, 1)}×</td>
        <td>${num(r.holdoutVsTsmom?.p, 3)} (${num(r.holdoutVsTsmom?.holm, 2)})</td><td>${num(r.holdoutVsSpy?.p, 3)} (${num(r.holdoutVsSpy?.holm, 2)})</td>
        <td class="${cls(r.orbHoldout?.sharpe)}">${num(r.orbHoldout?.sharpe)}</td></tr>`).join("")}</table>
    <h3>Distribution across all accounts</h3>
    <p class="muted">Holdout Sharpe median ${num(dist.holdoutSharpe?.median)} (p10 ${num(dist.holdoutSharpe?.p10)}, p90 ${num(dist.holdoutSharpe?.p90)}) ·
      beat TSMOM on the holdout: ${pct(dist.shareBeatingTsmomHoldout, 0)} · beat SPY: ${pct(dist.shareBeatingSpyHoldout, 0)} ·
      pre vs holdout rank correlation ${num(dist.preVsHoldoutRankCorr)}</p>
    ${hist(dist.preSharpeHistogram, "Pre-holdout Sharpe")}${hist(dist.holdoutSharpeHistogram, "Holdout Sharpe")}
    <h3>Day-trade leg (ORB, PDT enforced)</h3>
    <p class="muted">Champion acct ${esc(dt.champion)} · holdout Sharpe ${num(dt.holdoutChampion?.sharpe)} · PBO ${pct(dt.pboPre?.pbo, 0)} ·
      SPA p vs default ORB ${num(dtt["orb_default:holdout"]?.spaP, 3)} · vs cash ${num(dtt["cash:holdout"]?.spaP, 3)}</p>
    <h3>Forward shadow record</h3>
    <p class="muted">${fw.sessions ? `${fw.sessions} session(s) since ${esc(fw.start)}; leader acct ${esc(fw.leader)} (updated ${esc(fw.updatedAt)})`
      : "Starts with the first session after the design end; advanced after each daily paper run (autopilot / run-paper-daily.ps1)."}</p>`;
  // Histogram sizes are data: set them through the DOM (allowed by the CSP) instead of style="" attributes.
  box.querySelectorAll(".sw-bin[data-w]").forEach((el) => { el.style.width = el.dataset.w + "%"; });
  box.querySelectorAll(".sw-bar[data-h]").forEach((el) => { el.style.height = el.dataset.h + "px"; });
  $("swarm-sort")?.addEventListener("change", (e) => { sortKey = e.target.value; render(doc); });
}

async function renderSwarm() {
  const box = $("swarm-panel");
  if (!box) return;
  try {
    const r = await fetch("/api/swarm/status", { cache: "no-store" });
    const doc = await r.json().catch(() => ({}));
    if (r.status === 404) {
      box.innerHTML = `<p class="muted">No swarm results yet. Run <code>python engine/swarm.py run</code> (or the “Swarm: 50 simulated accounts” job).</p>`;
      return;
    }
    if (!r.ok) throw new Error(doc.error || r.statusText);
    render(doc);
  } catch (error) {
    box.innerHTML = `<p class="neg">Swarm panel failed: ${esc(error.message)}</p>`;
    console.error("swarm panel", error);
  }
}

renderSwarm();
setInterval(renderSwarm, 5 * 60 * 1000);
