// Analyst track-record panel (AI page): how the analyst's flags and stops did 5/20/60 trading days later, vs SPY,
// vs the sector ETF and vs a date-matched random stock, and which flag types (if any) may change position sizes.
// Data: GET /api/analyst-scorecard (engine/analyst_scoring.py). Self-mounting after #verdicts-panel.
(() => {
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 1, sign = false) => (x == null ? "–" : (sign && x > 0 ? "+" : "") + (x * 100).toFixed(d) + "%");
  const p = (x) => (x == null ? "–" : x < 0.001 ? "<0.001" : x.toFixed(3));

  function mount() {
    if (document.getElementById("analyst-scorecard")) return document.getElementById("analyst-scorecard");
    const css = document.createElement("link");   // external stylesheet: the dashboard CSP blocks inline styles
    css.rel = "stylesheet";
    css.href = "/analyst-scorecard.css";
    document.head.appendChild(css);
    const section = document.createElement("section");
    section.id = "analyst-scorecard";
    section.innerHTML = `<h2>Analyst track record <button type="button" id="scorecard-refresh">Refresh</button>
      <span id="scorecard-status" class="muted" role="status"></span></h2><div id="scorecard-body" class="muted">Loading…</div>`;
    const anchor = document.getElementById("verdicts-panel");
    if (anchor) anchor.after(section); else (document.querySelector("main") || document.body).appendChild(section);
    section.querySelector("#scorecard-refresh").addEventListener("click", refresh);
    return section;
  }

  const signed = (x) => `<span class="${x == null ? "" : x >= 0 ? "pos" : "neg"}">${pct(x, 1, true)}</span>`;

  function cellsTable(cells, benchmark) {
    const rows = cells.filter((c) => c.benchmark === benchmark);
    return `<table><thead><tr><th>Flag</th><th class="num">Horizon</th><th class="num">n (indep.)</th><th class="num">Hit rate</th>
      <th class="num">95% CI</th><th class="num">Chance</th><th class="num">Mean excess</th><th class="num">95% CI</th>
      <th class="num">p (perm.)</th>${benchmark === "spy" ? '<th class="num">p (Holm)</th><th>Gate</th>' : ""}</tr></thead><tbody>${rows.map((c) => `
      <tr class="${c.passes ? "pass" : ""}"><td><span class="badge ${esc(c.flag)}">${esc(c.flag)}</span></td><td class="num">${c.horizon}d</td>
      <td class="num">${c.n}${c.nRaw > c.n ? ` <span class="muted">(${c.nRaw})</span>` : ""}</td>
      <td class="num">${c.n ? pct(c.hitRate, 0) : "–"}</td><td class="num">${c.n ? pct(c.wilson?.[0], 0) + "–" + pct(c.wilson?.[1], 0) : "–"}</td>
      <td class="num">${c.n ? pct(c.baseRate, 0) : "–"}</td><td class="num">${c.n ? signed(c.meanExcess) : "–"}</td>
      <td class="num">${c.n ? pct(c.ciMean?.[0], 1, true) + " / " + pct(c.ciMean?.[1], 1, true) : "–"}</td><td class="num">${c.n ? p(c.pMean) : "–"}</td>
      ${benchmark === "spy" ? `<td class="num">${p(c.pHolm)}</td><td>${c.passes ? "<b>ACTIVE</b>" : `<span class="muted">${esc(c.reason || "log-only")}</span>`}</td>` : ""}</tr>`).join("")}</tbody></table>`;
  }


  function extrasBlock(doc) {
    const ic = (doc.informationCoefficient || {})["20"] || {};
    const brier = (doc.calibration && doc.calibration.brierByHorizon || {})["20"] || {};
    const nw = (doc.neweyWest || {})["20"] || {};
    if (ic.ic == null && brier.brier == null) return "";
    const nwLine = ["GREEN", "RED"].map((f) => {
      const s = nw[f] || {};
      return s.t == null ? "" : `${f} Newey-West t=${Number(s.t).toFixed(2)} (p=${p(s.pOneSided ?? s.p)})`;
    }).filter(Boolean).join(" · ");
    return `<p>20-day information coefficient (GREEN=+1, YELLOW=0, RED=−1 vs SPY excess): <strong>${ic.ic == null ? "–" : Number(ic.ic).toFixed(3)}</strong>
      (permutation p=${p(ic.p)}, n=${ic.n ?? 0}). Brier score: ${brier.brier == null ? "–" : Number(brier.brier).toFixed(3)}
      (0.25 is a coin-flip; lower is better). ${esc(nwLine)}</p>
      ${doc.statsNote ? `<p class="muted">${esc(doc.statsNote)}</p>` : ""}`;
  }

  function priorBlock(doc) {
    const prior = doc.prior, leak = doc.leakage || (prior && prior.leakage);
    if (!prior) return `<p class="muted">No sealed historical-replay prior yet. Run <code>python analyst_replay.py run --dry-run</code> (PIT heuristic) or <code>--model retort</code>. The prior never opens the live gate.</p>`;
    const ic = (prior.informationCoefficient || {})["20"] || {};
    const g = (prior.gate && prior.gate.reason) || "Replay/prior scores are informative only and never open the live gate.";
    return `<details><summary>Replay prior (starting reputation — does not open the gate)</summary>
      <div class="gate closed"><b>Prior cannot open the gate.</b> ${esc(g)}</div>
      <p>20-day IC: ${ic.ic == null ? "–" : Number(ic.ic).toFixed(3)} (p=${p(ic.p)}, n=${ic.n ?? 0}).
        Sample: ${esc(JSON.stringify(prior.sample ? {dates: (prior.sample.dates||[]).length, tickers: (prior.sample.tickers||[]).length, verdicts: prior.sample.nVerdicts, cached: prior.sample.cachedCalls, newCalls: prior.sample.newCalls} : {}))}.</p>
      ${leak && leak.warning ? `<p class="neg">${esc(leak.warning)}</p>` : (leak ? `<p class="muted">Masked vs unmasked 20-day IC: ${leak.ic20Masked ?? "–"} vs ${leak.ic20Unmasked ?? "–"}.</p>` : "")}
    </details>`;
  }

  function render(doc) {
    const body = document.getElementById("scorecard-body");
    const open = doc.gateOpen || [];
    const counts = doc.counts || {};
    const out = counts.outcomes || {};
    const stops = (doc.stops || []).filter((s) => s.n);
    const recent = doc.recent || [];
    const outcome = (r, h) => {
      const o = r.outcomes?.[h] || {};
      if (o.status === "scored") return signed(o.excess_spy) + (o.stop_hit ? ' <span class="muted">stop</span>' : "");
      return `<span class="muted">${esc(o.status || "–")}</span>`;
    };
    body.className = "";
    body.innerHTML = `
      <div class="gate ${open.length ? "open" : "closed"}">${open.length
        ? `<b>Active:</b> ${open.map(esc).join(", ")} — in <code>apply</code> mode these verdicts may shrink or skip new entries (never enlarge).`
        : `<b>No flag type has beaten chance yet.</b> Verdicts are logged and shown but do not change any position. A flag type becomes active only with ≥${doc.params?.minN ?? 30} independent calls and a Holm-adjusted p &lt; ${doc.params?.alpha ?? 0.05} vs a date-matched random liquid stock.`}</div>
      <p class="muted">Scorecard ${esc(doc.generatedAt ? new Date(doc.generatedAt).toLocaleString() : "–")} · prices through ${esc(doc.dataThrough)} · ${counts.calls ?? 0} flag(s) logged, ${counts.scorable ?? 0} LLM (scored) ·
        windows done 5d/20d/60d: ${out["5"]?.scored ?? 0}/${out["20"]?.scored ?? 0}/${out["60"]?.scored ?? 0}, pending ${out["5"]?.pending ?? 0}/${out["20"]?.pending ?? 0}/${out["60"]?.pending ?? 0}</p>
      <h3>Excess return vs SPY (decides the gate)</h3>${cellsTable(doc.cells || [], "spy")}
      <details><summary>Excess return vs sector ETF</summary>${cellsTable(doc.cells || [], "sector")}</details>
      <details><summary>Stops (${stops.length ? stops.map((s) => `${s.horizon}d ${s.hits}/${s.n}`).join(", ") : "none scored yet"})</summary>
        <table><thead><tr><th class="num">Horizon</th><th class="num">Calls with stop</th><th class="num">Hit</th><th class="num">95% CI</th>
        <th class="num">Return after stop</th><th class="num">Stops that cost money</th><th class="num">Median days</th></tr></thead><tbody>${(doc.stops || []).map((s) => `
        <tr><td class="num">${s.horizon}d</td><td class="num">${s.n}</td><td class="num">${s.n ? pct(s.hitRate, 0) : "–"}</td>
        <td class="num">${s.n ? pct(s.wilson?.[0], 0) + "–" + pct(s.wilson?.[1], 0) : "–"}</td><td class="num">${signed(s.meanReturnAfterStop)}</td>
        <td class="num">${s.stopsThatCostMoney ?? 0}</td><td class="num">${s.medianDaysToStop ?? "–"}</td></tr>`).join("")}</tbody></table></details>
      ${extrasBlock(doc)}
      ${priorBlock(doc)}
      <details ${recent.length ? "open" : ""}><summary>Logged calls (${recent.length})</summary>
        <table><thead><tr><th>Ticker</th><th>Flag</th><th>Issued</th><th>Source</th><th class="num">Price</th><th class="num">Stop</th>
        <th class="num">5d vs SPY</th><th class="num">20d</th><th class="num">60d</th></tr></thead><tbody>${recent.map((r) => `
        <tr><td><a href="/ticker.html?symbol=${encodeURIComponent(r.ticker)}">${esc(r.ticker)}</a></td><td><span class="badge ${esc(r.flag)}">${esc(r.flag)}</span></td>
        <td>${esc(r.issued_at ? new Date(r.issued_at).toLocaleString() : r.asof)}</td><td>${esc(r.source)}${r.scorable ? "" : ' <span class="muted">(not scored)</span>'}</td>
        <td class="num">${r.ref_price ?? "–"}</td><td class="num">${r.stop ?? "–"}</td><td class="num">${outcome(r, "5")}</td><td class="num">${outcome(r, "20")}</td><td class="num">${outcome(r, "60")}</td></tr>`).join("")}</tbody></table></details>
      <p class="hint">Hit = GREEN beat SPY, YELLOW/RED lagged it. "Chance" is the share of liquid stocks that would have hit on the same dates. Calls anchor at the first close after they were issued; repeated calls on a ticker inside one window count once. FALLBACK (non-LLM) verdicts are logged but never scored.</p>`;
  }

  async function load() {
    const status = document.getElementById("scorecard-status");
    try {
      const r = await fetch("/api/analyst-scorecard?limit=100");
      const doc = await r.json();
      if (!r.ok) { document.getElementById("scorecard-body").textContent = doc.error || "No scorecard yet."; return; }
      render(doc);
      status.textContent = doc.refreshing ? "refreshing…" : "";
    } catch (error) {
      document.getElementById("scorecard-body").textContent = "Could not load the scorecard: " + error.message;
    }
  }

  async function refresh() {
    const button = document.getElementById("scorecard-refresh");
    const status = document.getElementById("scorecard-status");
    button.disabled = true;
    status.textContent = "scoring… (loads the price panel when a window has completed; can take a minute or two)";
    try {
      const r = await fetch("/api/analyst-scorecard/refresh", { method: "POST" });
      const result = await r.json();
      status.textContent = result.status === "ok" ? `done: ${result.calls} call(s), ${result.ingested} new` : `failed: ${result.message || "see server log"}`;
    } catch (error) {
      status.textContent = "failed: " + error.message;
    } finally {
      button.disabled = false;
      load();
    }
  }

  const start = () => {
    const params = new URLSearchParams(location.search);
    const pageTicker = (params.get("symbol") || params.get("ticker") || "").toUpperCase();
    if (document.getElementById("verdicts-panel")) { mount(); load(); }
    const host = document.getElementById("ticker-scorecard-host");
    if (host && pageTicker) {
      if (!document.getElementById("analyst-scorecard")) {
        if (!document.querySelector('link[href^="/analyst-scorecard.css"]')) {  // external stylesheet: the dashboard CSP blocks inline styles
          const link = document.createElement("link");
          link.rel = "stylesheet";
          link.href = "/analyst-scorecard.css";
          document.head.appendChild(link);
        }
        const section = document.createElement("div");
        section.id = "analyst-scorecard";
        section.innerHTML = `<div id="scorecard-body" class="muted">Loading…</div>`;
        host.appendChild(section);
      }
      (async () => {
        try {
          const r = await fetch("/api/analyst-scorecard?ticker=" + encodeURIComponent(pageTicker) + "&limit=15");
          const doc = await r.json();
          if (!r.ok) { document.getElementById("scorecard-body").textContent = doc.error || "No scorecard yet."; return; }
          render(doc);
        } catch (error) {
          document.getElementById("scorecard-body").textContent = "Could not load the scorecard: " + error.message;
        }
      })();
    }
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
