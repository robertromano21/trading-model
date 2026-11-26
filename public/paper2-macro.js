// Macro card on the /paper2 risk & regime panel. Reads /api/macro/snapshot (local FRED/DXY proxy).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x == null || Number.isNaN(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);
const pp = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? "–" : `${Number(x).toFixed(d)} pp`);

function trendLabel(t) {
  if (t === "stronger") return "stronger dollar";
  if (t === "weaker") return "weaker dollar";
  if (t === "sideways") return "sideways";
  return "–";
}

async function renderMacro() {
  const box = $("macro-state");
  if (!box) return;
  try {
    const r = await fetch("/api/macro/snapshot", { cache: "no-store" });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || r.statusText);
    if (!d.available) {
      box.innerHTML = `<div class="card"><div class="label">Macro (dollar, rates, credit)</div>
        <div class="muted small">${esc(d.reason || "Macro data not built yet. Run python macro_data.py update.")}</div></div>`;
      return;
    }
    const dx = d.dxy || {}, rates = d.rates || {}, credit = d.credit || {}, uup = d.uup || {}, ol = d.dxyOutlook || {};
    const band = Array.isArray(ol.ci80) ? `80% band ${pct(ol.ci80[0])} to ${pct(ol.ci80[1])}` : "";
    const alert = ol.alert && ol.alert.flag
      ? `<div class="muted small p2m-alert"><strong>Alert:</strong> ${esc(ol.alert.reason || "outlook changed")}</div>`
      : "";
    const outlookCard = (ol.expectedMove != null || ol.pStronger != null || ol.direction)
      ? `<div class="card"><div class="label">Dollar outlook (recommend only)</div>
          <div class="value">${esc(ol.direction || "–")} <span class="muted small">${esc(ol.confidence || "")} confidence</span></div>
          <div class="muted small">20d expected ${pct(ol.expectedMove)} · ${esc(band)} · P(stronger) ${pct(ol.pStronger, 0)}
            · gates ${ol.gatesPassed ? "passed" : "not passed"} · trading default off
            ${ol.asOf ? `<br>as of ${esc(ol.asOf)}` : ""}</div>
          ${alert}</div>`
      : "";
    box.innerHTML = `
      <div class="cards">
        <div class="card"><div class="label">Dollar (DXY proxy)</div>
          <div class="value">${num(dx.level, 2)} <span class="muted small">${esc(trendLabel(dx.trend))}</span></div>
          <div class="muted small">20d ${pct(dx.ret20)} · 5d ${pct(dx.ret5)} · 20d vol ${num(dx.vol20 != null ? dx.vol20 * 100 : null, 1)}%
            ${uup.available ? `<br>vs UUP corr ${num(uup.corr, 2)}` : ""}
            ${dx.source ? ` · source ${esc(dx.source)}` : ""}</div></div>
        ${outlookCard}
        <div class="card"><div class="label">Treasury curve / SOFR</div>
          <div class="value">${num(rates.dgs10, 2)}% <span class="muted small">10Y</span></div>
          <div class="muted small">2Y ${num(rates.dgs2, 2)}% · 3M ${num(rates.dgs3mo, 2)}% · slope 10Y–2Y ${pp(rates.curveSlope10y2y)}
            · curve 20d ${pp(rates.curveLevelChg20)}<br>SOFR ${num(rates.sofr, 2)}% · SOFR–3M ${pp(rates.sofrTbillSpread)} (funding proxy, not a swap spread)</div></div>
        <div class="card"><div class="label">Credit (OAS proxy, not CDS)</div>
          <div class="value">${credit.hyOas != null ? num(credit.hyOas, 2) + "%" : "–"} <span class="muted small">HY OAS</span></div>
          <div class="muted small">IG ${num(credit.igOas, 2)}% · BBB ${num(credit.bbbOas, 2)}% · CCC ${num(credit.cccOas, 2)}%
            · HY–IG ${pp(credit.hyIgOas)} · HY 20d ${pp(credit.hyOasChg20)}</div></div>
      </div>
      <p class="muted small">As of ${esc(d.asOf || "–")}. Decision-time features lag published values by ${esc(d.publicationLagSessions ?? 1)} session(s).
        ${esc(d.proxyDisclaimer || d.note || "")}</p>`;
  } catch (error) {
    box.innerHTML = `<p class="muted">Macro snapshot unavailable: ${esc(error.message)}</p>`;
  }
}
renderMacro();
setInterval(renderMacro, 60_000);
