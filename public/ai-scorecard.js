// Analyst scorecard panel used by the AI page and the ticker page.
// Renders Massive/Research/analyst_scorecard.json honestly, including "sample too small" and a closed gate.
(function (root) {
  "use strict";
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x, d = 1) => (x == null || !Number.isFinite(Number(x)) ? "–" : ((Number(x) >= 0 ? "+" : "") + (Number(x) * 100).toFixed(d) + "%"));
  const num = (x, d = 3) => (x == null || !Number.isFinite(Number(x)) ? "–" : Number(x).toFixed(d));
  const yn = (v) => (v ? "yes" : "no");

  function flagRow(flag, block) {
    const h = block || {};
    const boot = h.bootstrap?.p, perm = h.permutation?.p, nw = h.neweyWest?.pOneSided ?? h.neweyWest?.p;
    return `<tr>
      <td><span class="badge ${esc(flag)}">${esc(flag)}</span></td>
      <td>${h.n ?? 0}${h.sufficient ? "" : " <span class='muted'>(too few)</span>"}</td>
      <td>${pct(h.meanExcess)}</td>
      <td>${pct(h.meanExcessSector)}</td>
      <td>${h.hitRate == null ? "–" : (h.hitRate * 100).toFixed(0) + "%"}</td>
      <td>${num(perm, 3)}</td>
      <td>${num(boot, 3)}</td>
      <td>${num(nw, 3)}</td>
      <td>${h.beatsChance ? "beats chance" : "no"}</td>
    </tr>`;
  }

  function horizonsTable(flags, horizon) {
    const head = `<thead><tr><th>Flag</th><th>n</th><th>Mean excess vs universe</th><th>vs sector</th><th>Hit rate</th><th>Perm. p</th><th>Boot. p</th><th>NW p</th><th>Beats chance?</th></tr></thead>`;
    const body = ["GREEN", "YELLOW", "RED"].map((f) => flagRow(f, (flags?.[f] || {})[String(horizon)])).join("");
    return `<table class="score-table">${head}<tbody>${body}</tbody></table>`;
  }

  function gateBox(gate, kind) {
    const open = kind === "prior" ? false : Boolean(gate?.open);
    const label = kind === "prior" ? "Prior (replay) — never opens the gate" : (open ? "Live gate OPEN" : "Live gate CLOSED");
    return `<div class="gate ${open ? "open" : "closed"}"><strong>${esc(label)}</strong>
      <p>${esc(gate?.reason || (kind === "prior"
        ? "Replay scores are an informative starting point only."
        : "No live scorecard yet. Verdicts are logged and have no effect."))}</p>
      ${kind === "live" ? `<p class="muted">RED may skip: ${yn(gate?.redMaySkip)} · YELLOW may shrink: ${yn(gate?.yellowMayShrink)} · Entry-band deferral: ${yn(gate?.bandMayDefer)}</p>` : ""}
    </div>`;
  }

  function renderScorecard(card, { ticker = null } = {}) {
    if (!card) {
      return `<p class="muted">No scorecard yet. It is built automatically after a paper-trader run, or by <code>python analyst_score.py rescore</code>.</p>`;
    }
    const live = card.live || {}, prior = card.prior, settings = card.settings || {};
    const ic = (live.informationCoefficient || {})["20"] || {};
    const brier = (live.calibration?.brierByHorizon || {})["20"] || {};
    const stops = (live.stops || {})["20"] || {};
    const leak = (prior && (prior.leakage || card.leakage)) || null;
    const warnings = (card.warnings || []).map((w) => `<li>${esc(w)}</li>`).join("");
    let tickerBlock = "";
    if (ticker) {
      const rec = (live.tickers || {})[ticker] || (prior?.tickers || {})[ticker];
      if (!rec) {
        tickerBlock = `<p class="muted">No scored calls for ${esc(ticker)} yet.</p>`;
      } else {
        const lines = (rec.calls || []).filter((c) => c.horizon === 20 || c.kind === "STOP").slice(0, 15)
          .map((c) => `<tr><td>${esc(c.asof)}</td><td>${esc(c.kind)}${c.flag ? " " + esc(c.flag) : ""}</td>
            <td>${c.status === "pending" ? "pending (horizon not elapsed)" : pct(c.ret)}</td>
            <td>${c.excessUniverse == null ? "–" : pct(c.excessUniverse)}</td>
            <td>${c.hit == null ? "–" : (c.hit ? "stop hit" : "not hit")}</td></tr>`).join("");
        tickerBlock = `<h3>${esc(ticker)} track (20-day outcomes)</h3>
          <table class="score-table"><thead><tr><th>As of</th><th>Call</th><th>Forward return</th><th>Excess vs universe</th><th>Stop</th></tr></thead>
          <tbody>${lines || "<tr><td colspan='5' class='muted'>No completed 20-day outcomes yet.</td></tr>"}</tbody></table>`;
      }
    }
    return `
      <p class="muted">Scored ${esc(card.scoredAt || "")} through ${esc(card.asof || "")}.
        Gate uses live out-of-sample calls only (min n=${settings.minN ?? 30}, α=${settings.alpha ?? 0.05}, horizon ${settings.gateHorizon ?? 20}d).
        Replay prior does not authorize shrinkage.</p>
      ${gateBox(live.gate, "live")}
      <h3>Live flags at 20 trading days</h3>
      ${horizonsTable(live.flags, 20)}
      <p>20-day information coefficient (GREEN=+1, YELLOW=0, RED=−1 vs universe excess):
        <strong>${num(ic.ic, 3)}</strong> (permutation p=${num(ic.p, 3)}, n=${ic.n ?? 0}).
        Brier score: ${num(brier.brier, 3)} (0.25 is a coin-flip; lower is better).
        Stops hit within 20d: ${stops.hits ?? 0}/${stops.n ?? 0}${stops.hitRate == null ? "" : " (" + (stops.hitRate * 100).toFixed(0) + "%)"}.</p>
      <details><summary>5-day and 60-day live tables</summary>
        <h4>5 days</h4>${horizonsTable(live.flags, 5)}
        <h4>60 days</h4>${horizonsTable(live.flags, 60)}
      </details>
      ${prior ? `<h3>Replay prior (starting reputation)</h3>
        ${gateBox(prior.gate, "prior")}
        ${horizonsTable(prior.flags, 20)}
        <p>20-day IC: ${num((prior.informationCoefficient || {})["20"]?.ic, 3)}.
          ${leak?.warning ? `<strong class="neg">${esc(leak.warning)}</strong>` : "Masked vs unmasked leakage check: " +
            `IC masked ${num(leak?.ic20Masked, 3)}, unmasked ${num(leak?.ic20Unmasked, 3)}.`}</p>` : "<p class='muted'>No sealed replay prior yet. Run <code>python analyst_replay.py run --dry-run</code> (heuristic) or <code>--model retort</code>.</p>"}
      ${tickerBlock}
      ${warnings ? `<details><summary>Warnings</summary><ul>${warnings}</ul></details>` : ""}
      <p class="muted">${esc(card.statsNote || "")}</p>`;
  }

  async function loadScorecard() {
    const r = await fetch("/api/ai/scorecard", { cache: "no-store" });
    if (r.status === 404) return null;
    const v = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(v.error || "Could not load the scorecard.");
    return v;
  }

  async function mount(el, opts = {}) {
    if (!el) return;
    el.innerHTML = "<p class='muted'>Loading scorecard…</p>";
    try {
      const card = await loadScorecard();
      el.innerHTML = renderScorecard(card, opts);
    } catch (e) {
      el.innerHTML = `<p class="muted">${esc(e.message)}</p>`;
    }
  }

  root.AnalystScorecard = { mount, renderScorecard, loadScorecard };
})(window);
