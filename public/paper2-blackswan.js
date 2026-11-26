// Black swan panel on /paper2.html#black-swan. Reads /api/black-swan (local files only).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x == null || Number.isNaN(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);

function feedLine(feeds) {
  if (!feeds || typeof feeds !== "object") return "Feeds have not been polled yet.";
  const bits = [];
  if (feeds.gdelt) bits.push(`GDELT ${feeds.gdelt.ok ? "ok" : "down"}`);
  if (feeds.gpr) bits.push(`GPR ${feeds.gpr.ok ? "ok" : "down"}`);
  if (Array.isArray(feeds.rss)) {
    const ok = feeds.rss.filter((f) => f.ok).length;
    bits.push(`RSS ${ok}/${feeds.rss.length}`);
  }
  if (Array.isArray(feeds.centralBanks)) {
    const ok = feeds.centralBanks.filter((f) => f.ok).length;
    bits.push(`central banks ${ok}/${feeds.centralBanks.length}`);
  }
  if (feeds.ofac) bits.push(`OFAC ${feeds.ofac.ok ? "ok" : "down"}`);
  return bits.join(" · ") || "Feeds have not been polled yet.";
}

function eventTable(study) {
  const rows = (study?.summary && Object.entries(study.summary)) || [];
  if (!rows.length) return `<p class="muted">Event study has not been run yet.</p>`;
  const cell = (h, key) => {
    const rec = (h || {})[key];
    return rec ? pct(rec.median) : "–";
  };
  const body = rows.map(([asset, c]) => {
    const h = c.horizons || {};
    const recov = c.medianRecoverSessions != null ? Number(c.medianRecoverSessions).toFixed(0) : "–";
    return `<tr><td>${esc(asset)}</td><td>${c.n ?? "–"}</td><td>${cell(h, "r1")}</td><td>${cell(h, "r5")}</td>
      <td>${cell(h, "r20")}</td><td>${cell(h, "r60")}</td><td>${pct(c.medianMaxDrawdown)}</td><td>${esc(recov)}</td></tr>`;
  }).join("");
  return `<table class="blackswan-table"><thead><tr>
    <th>Asset</th><th>n</th><th>1d med</th><th>5d med</th><th>20d med</th><th>60d med</th>
    <th>median max DD</th><th>median recover (sessions)</th>
  </tr></thead><tbody>${body}</tbody></table>`;
}

async function renderBlackSwan() {
  const box = $("black-swan-panel");
  if (!box) return;
  try {
    const r = await fetch("/api/black-swan", { cache: "no-store" });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || r.statusText);
    const st = d.state || {};
    const conf = st.confirm || {};
    const news = st.news || {};
    const policy = st.policy || {};
    const study = d.study || {};
    const wf = study.walkForward || {};
    const dollar = d.dollarShock || st.dollarShock || {};
    const dollarLive = dollar.live || {};
    const dollarStudy = d.dollarStudy || {};
    const stateClass = `bs-state bs-state-${String(st.state || "NORMAL").toLowerCase()}`;
    const honesty = (study.honesty || []).slice(0, 4).map((h) => `<li>${esc(h)}</li>`).join("");
    const events = (d.recentEvents || []).slice(0, 8).map((e) =>
      `<li><strong>${esc(e.state)}</strong> ${esc(e.asof || e.at || "")}: ${esc(e.headline || e.eventType || "")}</li>`).join("");
    box.innerHTML = `
      <div class="cards">
        <div class="card"><div class="label">State</div>
          <div class="value"><span class="${stateClass}">${esc(st.state || "NORMAL")}</span>
            <span class="muted small">${esc(st.direction || "")}</span></div>
          <div class="muted small">${esc(st.headline || "")}</div></div>
        <div class="card"><div class="label">News / confirmation</div>
          <div class="value">sev ${esc(news.newsSeverity ?? "–")}/5 · stress ${num(conf.stress, 2)}</div>
          <div class="muted small">flag ${esc(st.lastFlag || "–")} · VIX ${num(conf.vix, 1)}
            ${conf.vixRollover ? " · VIX rolling over" : ""} · SPY gap ${pct(conf.spyGap, 2)}
            · method ${esc(news.method || "–")}</div></div>
        <div class="card"><div class="label">Dollar shock (alert only)</div>
          <div class="value">${dollar.shock ? `SHOCK · ${esc(dollarLive.direction || dollar.direction || "")}` : "no dollar shock"}</div>
          <div class="muted small">${esc(dollar.headline || "DXY / UUP 2-sigma detector has not run yet.")}
            ${dollarLive.asof ? ` · ${esc(dollarLive.source || "")} as of ${esc(dollarLive.asof)} · 1d z=${num(dollarLive.z1, 2)} · 5d z=${num(dollarLive.z5, 2)}` : ""}
            · no book change</div></div>
        <div class="card"><div class="label">Playbook (paper)</div>
          <div class="value">${policy.freezeNewLongs ? "freeze new longs" : "no long freeze"}
            · gross ×${num(policy.scaleGross, 2)}</div>
          <div class="muted small">${esc(policy.reason || "–")}
            ${policy.allowVixy ? " · VIXY allowed" : ""} ${policy.allowSvix ? " · SVIX allowed" : ""}
            ${policy.allowBonds ? " · bonds allowed" : ""} ${policy.allowShort ? " · short sleeve allowed" : ""}</div></div>
      </div>
      <p class="muted small">${d.disabled ? "Polling is off (TRADING_BLACKSWAN=off). " : ""}
        Feeds: ${esc(feedLine(d.feeds))}. Updated ${esc(st.updatedAt || "–")}.</p>
      <h3>Recent transitions</h3>
      ${events ? `<ul class="blackswan-events">${events}</ul>` : `<p class="muted">No actionable flags logged yet.</p>`}
      <h3>Event study (median across curated shocks)</h3>
      ${eventTable(study)}
      ${wf && wf.ok ? `<p class="muted small">Playbook vs 12-1 TSMOM on SPY/proxy:
        baseline CAGR ${pct(wf.baseline?.cagr)} / max DD ${pct(wf.baseline?.maxDrawdown)};
        playbook CAGR ${pct(wf.playbook?.cagr)} / max DD ${pct(wf.playbook?.maxDrawdown)}.
        Applied to ${esc(wf.nApplied)} of ${esc(wf.nConsidered)} events.
        Mean event-window excess ${pct(wf.eventWindowExcessMean)} (SE ${pct(wf.eventWindowExcessSE)}).
        ${esc(wf.honesty || "")}</p>` : ""}
      ${dollarStudy && dollarStudy.verdict ? `<h3>Dollar shock study</h3>
        <p class="muted small">${esc((dollarStudy.verdict && dollarStudy.verdict.answer) || "")}
          Holdout from ${esc(dollarStudy.holdoutStart || "2023-01-03")}.
          Events: ${esc((dollarStudy.events && dollarStudy.events.n) ?? "–")}
          (surge ${(dollarStudy.events && dollarStudy.events.nSurge) ?? "–"},
          weaken ${(dollarStudy.events && dollarStudy.events.nWeaken) ?? "–"}).
          ${esc((dollarStudy.verdict && dollarStudy.verdict.intraday) || "")}</p>` : ""}
      ${honesty ? `<h3>Limits</h3><ul class="muted small">${honesty}</ul>` : ""}`;
  } catch (error) {
    box.innerHTML = `<p class="muted">Black swan monitor unavailable: ${esc(error.message)}</p>`;
  }
}

if (!document.querySelector('link[href^="/paper2-blackswan.css"]')) {  // external stylesheet: the dashboard CSP blocks inline styles
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = "/paper2-blackswan.css";
  document.head.appendChild(link);
}
renderBlackSwan();
setInterval(renderBlackSwan, 60_000);
