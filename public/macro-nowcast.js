// Our macro nowcasts (GDPNow-style, raw data only) with outside estimates shown beside them for comparison.
// Mounts into #nowcast-panel (macro.html and the paper page). Read-only: no orders, no keys.
if (!document.querySelector('link[href^="/macro-nowcast.css"]')) {  // external stylesheet: the dashboard CSP blocks inline styles
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = "/macro-nowcast.css";
  document.head.appendChild(link);
}
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d));
const sign = (x, d = 2) => (x == null ? "—" : (x > 0 ? "+" : "") + Number(x).toFixed(d));
const DIGITS = { PAYROLLS: 0, GDP_SAAR: 2, IMPORTS_SAAR: 2, EXPORTS_SAAR: 2, UNRATE: 2 };
const UNITS = { PAYROLLS: "k jobs", GDP_SAAR: "% SAAR", IMPORTS_SAAR: "% SAAR", EXPORTS_SAAR: "% SAAR", UNRATE: "%" };

async function getJson(url, opts) {
  const r = await fetch(url, { cache: "no-store", ...opts });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
  return body;
}

let Chart = null;
function loadChart() {
  if (window.Chart) return Promise.resolve(window.Chart);
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = "/assets/chart.umd.js";
    s.onload = () => (window.Chart ? resolve(window.Chart) : reject(new Error("Chart.js unavailable")));
    s.onerror = () => reject(new Error("Chart.js unavailable"));
    document.head.appendChild(s);
  });
}

function outsideCell(n) {
  const o = n.outside || {};
  const keys = Object.keys(o);
  if (!keys.length) return '<span class="muted">—</span>';
  return keys.map((k) => `<span title="${esc(o[k].source || k)}${o[k].asof ? ` · as of ${esc(o[k].asof)}` : ""}">${esc(k)} ${fmt(o[k].value, DIGITS[n.target] ?? 3)}</span>`).join("<br>");
}

function table(latest) {
  const rows = (latest.nowcasts || []).map((n) => {
    if (n.error) return `<tr><td>${esc(n.target)}</td><td colspan="8" class="bad">${esc(n.error)}</td></tr>`;
    const d = DIGITS[n.target] ?? 3;
    const drivers = (n.drivers || []).slice(0, 3).map((x) => `${esc(x.label)} (${sign(x.contribution, d === 0 ? 0 : 3)})`).join("; ");
    const surprise = n.impliedSurpriseVsLastPrint;
    return `<tr>
      <td><b>${esc(n.label || n.target)}</b><div class="muted small">${esc(n.period)}</div></td>
      <td>${esc(n.nextRelease || n.calendarNextRelease || n.estimatedRelease)}<div class="muted small">${esc(n.nextReleaseSource || "")}</div></td>
      <td><b>${fmt(n.nowcast, d)}</b></td>
      <td class="muted">${n.band80 ? `${fmt(n.band80[0], d)} … ${fmt(n.band80[1], d)}` : "—"}</td>
      <td>${fmt(n.lastPrint, d)}</td>
      <td class="${surprise > 0 ? "good" : surprise < 0 ? "bad" : ""}">${sign(surprise, d)}</td>
      <td>${outsideCell(n)}</td>
      <td>${n.oursMinusOutside == null ? "—" : `${sign(n.oursMinusOutside, d)}<div class="muted small">vs ${esc(n.oursMinusOutsideRef)}</div>`}</td>
      <td class="small">${drivers || "—"}</td></tr>`;
  }).join("");
  return `<table class="nowcast-table"><thead><tr><th>Release</th><th>Next release</th><th>Our nowcast</th><th>80% band</th><th>Last print</th>
    <th>Implied surprise vs last</th><th>Outside (display only)</th><th>Ours − outside</th><th>Top drivers</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function accuracyTable(acc) {
  if (!acc?.byTarget) return "";
  const rows = Object.entries(acc.byTarget).map(([t, a]) => `<tr><td>${esc(t)}</td><td>${a?.n ?? "—"}</td>
    <td>${fmt(a?.rmseModel, 3)}</td><td>${fmt(a?.rmseLastPrint, 3)}</td><td>${fmt(a?.rmseRatio, 2)}</td><td>${fmt(a?.clarkWestP, 3)}</td><td>${a?.directionHitRate == null ? "—" : Math.round(a.directionHitRate * 100) + "%"}</td></tr>`).join("");
  const out = Object.entries(acc.outside || {}).filter(([, v]) => v && typeof v === "object" && v.n).map(([k, v]) =>
    `<tr><td>${esc(k)} vs ${esc(v.vs || "")}</td><td>${v.n}</td><td>${fmt(v.rmseOurs, 3)}</td><td>${fmt(v.rmseTheirs, 3)}</td><td>${fmt(v.dmP, 3)}</td><td>${v.dmT == null ? "—" : v.dmT > 0 ? "ours" : "theirs"}${v.dmP != null && v.dmP < 0.05 ? "" : " (not significant)"}</td></tr>`).join("");
  return `<details><summary>Track record (real-time walk-forward, before the sealed holdout, pandemic months excluded)</summary>
    <table class="small"><thead><tr><th>Target</th><th>n</th><th>RMSE ours</th><th>RMSE last-print</th><th>Ratio</th><th>Clark-West p</th><th>Direction hit</th></tr></thead><tbody>${rows}</tbody></table>
    ${out ? `<h4>Ours vs outside forecasts (compared after the fact; never inputs)</h4><table class="small"><thead><tr><th>Comparison</th><th>n</th><th>RMSE ours</th><th>RMSE outside</th><th>DM p</th><th>Better</th></tr></thead><tbody>${out}</tbody></table>` : ""}
    <p class="muted small">Point-in-time: ${esc(acc.pointInTime || "")}. Generated ${esc(acc.generatedAt || "")}.</p></details>`;
}

const charts = {};
async function drawEvolution(root, target) {
  const hist = await getJson(`/api/macro/nowcast/history?target=${encodeURIComponent(target)}`);
  const C = Chart || (Chart = await loadChart());
  const periods = hist.backfilled || [];
  const labels = periods.map((p) => p.period.slice(0, 7));
  const finals = periods.map((p) => (p.path?.length ? p.path[p.path.length - 1].nowcast : null));
  const actual = periods.map((p) => p.actual);
  charts.track?.destroy();
  charts.track = new C($("#nc-track", root), { type: "line", data: { labels, datasets: [
    { label: "Our final nowcast", data: finals, borderColor: "#4ea1ff", tension: 0.2 },
    { label: "Official first release", data: actual, borderColor: "#f2c14e", borderDash: [5, 4], tension: 0.2 },
  ] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "#ccc" } } } } });
  // current period: live publishes (history.jsonl) on top of the backfilled weekly path
  const live = (hist.live || []);
  const cur = live.length ? live[live.length - 1].period : null;
  const pts = live.filter((r) => r.period === cur).map((r) => ({ x: r.asof, y: r.nowcast, lo: r.band80?.[0], hi: r.band80?.[1], out: r.outside }));
  const outsideNames = [...new Set(pts.flatMap((p) => Object.keys(p.out || {})))];
  charts.live?.destroy();
  charts.live = new C($("#nc-live", root), { type: "line", data: { labels: pts.map((p) => p.x), datasets: [
    { label: `Our nowcast for ${cur || "current period"}`, data: pts.map((p) => p.y), borderColor: "#4ea1ff" },
    { label: "80% band low", data: pts.map((p) => p.lo), borderColor: "rgba(78,161,255,.3)", pointRadius: 0 },
    { label: "80% band high", data: pts.map((p) => p.hi), borderColor: "rgba(78,161,255,.3)", pointRadius: 0, fill: "-1", backgroundColor: "rgba(78,161,255,.08)" },
    ...outsideNames.map((k, i) => ({ label: `${k} (display only)`, data: pts.map((p) => p.out?.[k] ?? null), borderColor: ["#f2c14e", "#b98cff", "#7bd389"][i % 3], borderDash: [4, 3] })),
  ] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "#ccc" } } } } });
}

export async function mountNowcast(root) {
  if (!root) return;
  root.innerHTML = '<p class="muted">Loading our nowcasts…</p>';
  let data;
  try { data = await getJson("/api/macro/nowcast"); } catch (e) {
    root.innerHTML = `<p class="muted">Nowcast route unavailable (${esc(e.message)}). Restart the dashboard once to load it.</p>`;
    return;
  }
  if (!data.available) {
    root.innerHTML = `<p class="muted">No nowcast yet. ${data.fredKey?.set ? "" : "Add a FRED API key on the Research &amp; data page, then "}press refresh.</p><button type="button" class="nc-refresh">Refresh now</button>`;
  } else {
    const L = data.latest, r = L.regime || {};
    const targets = [...new Set((L.nowcasts || []).map((n) => n.target))];
    root.innerHTML = `
      <p><b>Regime:</b> ${esc(r.label || "n/a")} <span class="muted small">(curve 10y-3m ${fmt(r.values?.curve10y3m)}, Sahm ${fmt(r.values?.sahm)}, NFCI ${fmt(r.values?.nfci)}, core CPI 3m ann. ${fmt(r.values?.coreCpi3mAnnualized)}%)</span></p>
      <p class="muted small">As of ${esc(L.asof)} · generated ${esc(L.generatedAt)} (${data.ageMinutes ?? "?"} min ago) · ${esc(L.pointInTime || "")}${data.running ? " · <b>refresh running…</b>" : ""}
        <button type="button" class="nc-refresh"${data.running ? " disabled" : ""}>Refresh now</button></p>
      ${table(L)}
      <p class="muted small">${esc(L.inputsPolicy || "")} ${esc(L.outsideNote || "")} Units: ${targets.map((t) => `${t} ${UNITS[t] || "% m/m"}`).join(", ")}.</p>
      ${accuracyTable(data.accuracy)}
      <div class="bar nc-bar"><label>Evolution: <select class="nc-target">${targets.map((t) => `<option>${esc(t)}</option>`).join("")}</select></label></div>
      <div class="nc-charts">
        <div class="nc-chart"><canvas id="nc-track"></canvas></div><div class="nc-chart"><canvas id="nc-live"></canvas></div></div>`;
    const sel = $(".nc-target", root);
    sel?.addEventListener("change", () => drawEvolution(root, sel.value).catch(() => {}));
    if (targets.length) drawEvolution(root, targets[0]).catch((e) => { $("#nc-track", root)?.insertAdjacentText("afterend", e.message); });
  }
  $(".nc-refresh", root)?.addEventListener("click", async (ev) => {
    ev.target.disabled = true;
    try { await getJson("/api/macro/nowcast/refresh", { method: "POST" }); ev.target.textContent = "Refresh started (3–5 min)…"; setTimeout(() => mountNowcast(root), 240000); }
    catch (e) { ev.target.textContent = e.message; }
  });
}

const auto = document.getElementById("nowcast-panel");
if (auto) mountNowcast(auto);
