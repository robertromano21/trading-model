// Return-attribution panel for /ops.html and /paper2.html. Reads /api/research/attribution.
const $ = (id, root = document) => root.getElementById(id) || document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 2) => (x == null || !Number.isFinite(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x == null || !Number.isFinite(Number(x)) ? "–" : Number(x).toFixed(d));

function tableHtml(headers, rows) {
  if (!rows || !rows.length) return "<p class=\"muted\">None.</p>";
  return `<table><thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${
    rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

const ORDER = [
  "backtest.core_tsmom", "walkforward.tsmom",
  "backtest.combined", "walkforward.combined",
  "backtest.overlay_meanrev", "walkforward.meanrev",
  "daytrade.full_defaults", "paper.default",
  "backtest.benchmark_spy", "backtest.benchmark_equal_weight_etfs",
];

function verdictClass(row) {
  const tag = row?.verdict?.label;
  if (tag === "significant_alpha") return "ok";
  if (tag === "significant_negative_alpha" || tag === "mostly_market") return tag === "mostly_market" ? "" : "bad";
  return "muted";
}

function seriesRows(data) {
  const keys = [...ORDER.filter((k) => data.series?.[k]), ...Object.keys(data.series || {}).filter((k) => !ORDER.includes(k))];
  return keys.map((key) => {
    const row = data.series[key];
    if (!row?.available) {
      return [esc(row?.label || key), "–", "–", "–", "–", "–", "–", "–", "–", esc(row?.reason || "n/a")];
    }
    const capm = row.capm || {};
    const blend = row.betaMatchedBlend?.simple || {};
    return [
      esc(row.label || key),
      String(row.period?.sessions ?? "–"),
      num(capm.beta),
      num(capm.r2),
      pct(capm.alphaAnnual),
      num(capm.alphaT),
      pct(capm.trackingError, 1),
      num(capm.informationRatio),
      `${num((blend.blend || {}).sharpe)} / ${num((row.performance || {}).sharpe)}`,
      esc(row.verdict?.label || ""),
    ];
  });
}

function headline(data) {
  const main = data.series?.["backtest.core_tsmom"] || data.series?.["walkforward.tsmom"];
  const text = main?.verdict?.summary || data.headlines?.["backtest.core_tsmom"] || "No TSMOM attribution yet.";
  const cls = verdictClass(main);
  return `<p class="${cls}"><strong>Is TSMOM just a cheaper index fund?</strong> ${esc(text)}</p>`;
}

function contributions(row) {
  const c = row?.sources?.contributions;
  if (!c) return "";
  const v = c.varianceShare || {};
  return `<p class="small">Annualized arithmetic contributions: alpha ${pct(c.alphaAnnual)}, market ${pct(c.marketAnnual)}, `
    + `sector tilts ${pct(c.sectorTiltsAnnual)}, bonds ${pct(c.bondsAnnual)}, other ${pct(c.otherAnnual)}. `
    + `Variance shares — market ${pct(v.market, 0)}, bonds ${pct(v.bondsIncremental, 0)}, `
    + `sector/other ${pct(v.sectorAndOtherIncremental, 0)}, residual ${pct(v.residual, 0)}.</p>`;
}

function factorTable(row) {
  const full = row?.sources?.full || row?.spyBonds;
  if (!full?.betas) return "";
  const names = Object.keys(full.betas);
  const rows = names.map((f) => [
    esc(f), num(full.betas[f]), num((full.betaT || {})[f]),
    (full.betaP || {})[f] != null ? num(full.betaP[f], 3) : "–",
  ]);
  rows.unshift(["alpha (annualized)", pct(full.alphaAnnual), num(full.alphaT), full.alphaP != null ? num(full.alphaP, 3) : "–"]);
  return `<h3 class="small">Factor betas — ${esc(full.label || "multi-factor")} (HAC t-stats, n=${full.n ?? "–"})</h3>`
    + tableHtml(["Factor", "Beta / value", "t (HAC)", "p"], rows);
}

function brinsonTable(row) {
  const br = row?.brinson?.vsSpy;
  if (!br?.byGroup?.length) return "";
  const rows = br.byGroup.slice(0, 12).map((g) => [
    esc(g.group), pct(g.avgPortfolioWeight, 1), pct(g.avgBenchmarkWeight, 1),
    num(g.allocation, 3), num(g.selection, 3), num(g.interaction, 3), num(g.total, 3),
  ]);
  return `<h3 class="small">Brinson vs SPY (${esc(br.start)} to ${esc(br.end)}, ${br.sessions} sessions; `
    + `compounded ${pct(br.portfolioReturn)} vs ${pct(br.benchmarkReturn)})</h3>`
    + tableHtml(["Group", "Avg w (port)", "Avg w (SPY)", "Allocation", "Selection", "Interaction", "Total"], rows)
    + `<p class="small muted">Sums: allocation ${num(br.allocationSum, 3)}, selection ${num(br.selectionSum, 3)}, `
    + `interaction ${num(br.interactionSum, 3)}; identity residual ${num(br.identityResidualMeanAbs, 6)} (mean abs. daily).</p>`;
}

function blendLine(row) {
  const b = row?.betaMatchedBlend?.simple;
  if (!b) return "";
  const w = b.weights || {};
  const wtxt = Object.entries(w).filter(([, v]) => v != null && Math.abs(v) > 0.005)
    .map(([k, v]) => `${k} ${pct(v, 0)}`).join(", ");
  return `<p class="small">Beta-matched blend (${esc(wtxt)}): blend CAGR ${pct(b.blend?.cagr)}, Sharpe ${num(b.blend?.sharpe)}, `
    + `max DD ${pct(b.blend?.maxDrawdown)} vs strategy CAGR ${pct(b.strategy?.cagr)}, Sharpe ${num(b.strategy?.sharpe)}, `
    + `max DD ${pct(b.strategy?.maxDrawdown)}. IR vs blend ${num(b.informationRatio)}, `
    + `alpha vs blend ${pct(b.alphaVsBlendAnnual)} (t=${num(b.alphaVsBlendT)}).</p>`;
}

function rollingLine(row) {
  const r = row?.rollingCapm || {};
  const parts = Object.keys(r).sort((a, b) => Number(a) - Number(b)).map((w) => {
    const block = r[w] || {};
    const last = block.latestWindow || {};
    const beta = block.beta || {};
    return `${w}d window: beta latest ${num(last.beta ?? beta.latest)} (mean ${num(beta.mean)}, range ${num(beta.min)}–${num(beta.max)}), `
      + `alpha ${pct(last.alphaAnnual)} (t=${num(last.alphaT)})`;
  });
  return parts.length ? `<p class="small muted">Rolling CAPM — ${esc(parts.join("; "))}.</p>` : "";
}

function detail(data, key) {
  const row = data.series?.[key];
  if (!row?.available) return "";
  const reasons = (row.verdict?.reasons || []).map((r) => `<li>${esc(r)}</li>`).join("");
  return `<article class="attr-series">
    <h3>${esc(row.label || key)}</h3>
    <p>${esc(row.verdict?.summary || "")}</p>
    ${reasons ? `<ul class="small">${reasons}</ul>` : ""}
    ${contributions(row)}${blendLine(row)}${rollingLine(row)}${factorTable(row)}${brinsonTable(row)}
  </article>`;
}

export async function loadAttribution() {
  const note = $("attr-note");
  const verdict = $("attr-verdict");
  const summary = $("attr-summary");
  const detailBox = $("attr-detail");
  const caveats = $("attr-caveats");
  if (!note && !summary) return;
  try {
    const response = await fetch("/api/research/attribution", { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || response.statusText);
    if (note) {
      note.textContent = `Generated ${data.generatedAt || "?"} · factors ${data.data?.factorStart || "?"} to ${data.data?.factorEnd || "?"} (${data.data?.factorSessions ?? "?"} sessions).`;
    }
    if (verdict) verdict.innerHTML = headline(data);
    if (summary) {
      summary.innerHTML = tableHtml(
        ["Series", "Sessions", "Beta vs SPY", "R²", "Alpha %/yr", "Alpha t (HAC)", "TE vs SPY", "IR vs SPY", "Sharpe blend / strat", "Verdict"],
        seriesRows(data),
      );
    }
    if (detailBox) {
      const prefer = ["backtest.core_tsmom", "walkforward.tsmom", "backtest.combined", "walkforward.combined",
        "backtest.overlay_meanrev", "daytrade.full_defaults"];
      detailBox.innerHTML = prefer.map((k) => detail(data, k)).join("");
    }
    if (caveats) {
      const extras = [];
      if (data.paper && data.paper.available === false) extras.push(data.paper.reason);
      const list = [...(data.caveats || []), ...extras].filter(Boolean);
      caveats.innerHTML = list.map((c) => `<li>${esc(c)}</li>`).join("");
    }
  } catch (error) {
    if (note) note.textContent = error.message;
    if (verdict) verdict.textContent = "";
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => loadAttribution());
} else {
  loadAttribution();
}
