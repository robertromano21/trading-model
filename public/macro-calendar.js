// Calendar card + release ticker for /paper2 (and any page that includes a #macro mount).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function fmtZ(z) {
  if (z == null || Number.isNaN(Number(z))) return "–";
  const x = Number(z);
  return `${x >= 0 ? "+" : ""}${x.toFixed(2)}`;
}
function fmtNum(v) {
  if (v == null || v === "") return "–";
  const x = Number(v);
  if (Number.isFinite(x)) return Math.abs(x) >= 100 ? x.toLocaleString("en-US", { maximumFractionDigits: 1 }) : String(Number(x.toFixed(3)));
  return esc(v);
}
function whenEt(iso) {
  if (!iso) return "–";
  // Stored as an ISO string already in America/New_York; display the clock and date without re-converting.
  const m = String(iso).match(/^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})/);
  if (m) return `${m[1]} ${m[2]} ET`;
  return esc(iso);
}
function chipClass(row) {
  const z = Number(row.surpriseZ);
  if (!Number.isFinite(z)) return row.actual == null ? "pending" : "";
  if (z >= 1) return "hawkish";
  if (z <= -1) return "dovish";
  return "";
}

async function getJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  const doc = await r.json().catch(() => ({}));
  if (r.status === 404) return { missing: true, ...doc };
  if (!r.ok) throw new Error(doc.error || r.statusText);
  return doc;
}

function renderTicker(surprises, upcoming) {
  const box = $("macro-ticker");
  if (!box) return;
  const chips = [];
  for (const s of (surprises || []).slice(0, 8)) {
    const name = s.name || s.eventId || "release";
    chips.push(`<span class="macro-chip ${chipClass(s)}" title="${esc(s.consensusSource || "")}">
      <span>${esc(name)}</span>
      <span>${fmtNum(s.actual ?? s.actualMid ?? s.highYield)}</span>
      <span class="z">z ${fmtZ(s.surpriseZ ?? s.tailZ)}</span>
    </span>`);
  }
  for (const u of (upcoming || []).slice(0, 4)) {
    chips.push(`<span class="macro-chip pending">
      <span>${esc(u.name)}</span>
      <span>${esc(whenEt(u.datetimeEt))}</span>
    </span>`);
  }
  box.innerHTML = chips.join("") || `<span class="muted">No recent prints or upcoming releases in the cache yet.</span>`;
}

function renderTable(events) {
  const box = $("macro-calendar");
  if (!box) return;
  const rows = (events || []).map((e) => {
    const hi = e.importance === "high" ? "hi" : "";
    const manual = e.entry === "manual" ? "manual" : "";
    return `<tr class="${hi}">
      <td>${esc(whenEt(e.datetimeEt))}</td>
      <td class="${manual}">${esc(e.name)}${e.entry === "manual" ? " <span class='muted'>(manual)</span>" : ""}</td>
      <td>${esc(e.agency || "")}</td>
      <td>${fmtNum(e.consensus)}</td>
      <td>${fmtNum(e.actual)}</td>
      <td class="${chipClass(e)}">${fmtZ(e.surpriseZ)}</td>
    </tr>`;
  }).join("");
  box.innerHTML = `
    <table class="macro-cal">
      <thead><tr><th>When (ET)</th><th>Release</th><th>Source</th><th>Consensus</th><th>Actual</th><th>Surprise z</th></tr></thead>
      <tbody>${rows || `<tr><td colspan="6" class="muted">No rows in the selected window.</td></tr>`}</tbody>
    </table>`;
}

function renderStudy(doc, target, empty) {
  const box = $(target);
  if (!box) return;
  if (!doc || doc.missing) {
    box.innerHTML = `<p class="muted">${esc(empty)}</p>`;
    return;
  }
  const proven = doc.nProvenHolm ?? doc.nProvenOosHolm ?? 0;
  box.innerHTML = `
    <p>Tests: ${esc(doc.nTests ?? "–")} · proven after Holm (and OOS where applicable): <b>${esc(proven)}</b>
      · as of ${esc(doc.generatedAt || "–")}</p>
    <p class="macro-honesty muted">${esc(doc.honesty || "")}</p>`;
}

async function submitConsensus(ev) {
  ev.preventDefault();
  const status = $("macro-consensus-status");
  const id = $("macro-cons-id")?.value?.trim();
  const period = $("macro-cons-period")?.value?.trim();
  const value = Number($("macro-cons-value")?.value);
  if (!id || !period || !Number.isFinite(value)) {
    if (status) status.textContent = "Need an event id, period and a number.";
    return;
  }
  if (status) status.textContent = "Saving…";
  try {
    const r = await fetch("/api/macro/consensus", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id, period, value, source: "manual" }),
    });
    const doc = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(doc.error || r.statusText);
    if (status) status.textContent = "Saved. Refreshing…";
    await renderMacro();
  } catch (error) {
    if (status) status.textContent = error.message;
  }
}

async function renderMacro() {
  const root = $("macro") || $("macro-panel");
  if (!root) return;
  const err = $("macro-error");
  try {
    const [status, upcoming, surprises, evStudy, ovStudy] = await Promise.all([
      getJson("/api/macro/status").catch(() => ({ missing: true })),
      getJson("/api/macro/upcoming?days=14").catch(() => ({ events: [] })),
      getJson("/api/macro/surprises").catch(() => ({ items: [] })),
      getJson("/api/macro/event-study").catch(() => ({ missing: true })),
      getJson("/api/macro/overnight").catch(() => ({ missing: true })),
    ]);
    if (err) { err.hidden = true; err.textContent = ""; }
    const items = surprises.items || surprises.latestSurprises || status.latestSurprises || [];
    renderTicker(items, upcoming.events || status.upcoming || []);
    renderTable(upcoming.events || []);
    const watch = status.watcher || {};
    const w = $("macro-watcher");
    if (w) {
      w.innerHTML = `Watcher: ${esc(watch.state || "not started")} · polls ${esc(watch.polls ?? "–")} · prints ${esc(watch.prints ?? "–")}
        · updated ${esc(watch.updatedAt || "–")} ET window.`;
    }
    renderStudy(evStudy, "macro-event-study", "No event study yet. It runs from python macro_service.py study (or the daily job).");
    renderStudy(ovStudy, "macro-overnight-study", "No overnight overseas study yet.");
  } catch (error) {
    if (err) { err.hidden = false; err.textContent = error.message; }
  }
}

function boot() {
  if (!$("macro") && !$("macro-panel")) return;
  $("macro-cons-form")?.addEventListener("submit", submitConsensus);
  $("macro-refresh")?.addEventListener("click", async () => {
    const s = $("macro-consensus-status");
    if (s) s.textContent = "Refreshing calendar…";
    try {
      const r = await fetch("/api/macro/refresh", { method: "POST" });
      const doc = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(doc.error || r.statusText);
      if (s) s.textContent = `Calendar rebuilt (${doc.eventCount ?? "?"} events).`;
      await renderMacro();
    } catch (error) {
      if (s) s.textContent = error.message;
    }
  });
  renderMacro();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
else boot();
