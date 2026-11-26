// Drift panel for /paper2: realized paper (or later live) metrics vs walk-forward 5–95 / 1–99 bands and CUSUM.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function fmt(metric, value) {
  if (value == null || Number.isNaN(Number(value))) return "–";
  const x = Number(value);
  if (metric === "slippage_bps") return `${x.toFixed(1)} bps`;
  if (metric === "position_count") return x.toFixed(0);
  if (metric === "daytrade_fill_rate" && Math.abs(x) > 1.5) return `${x.toFixed(2)} / session`;
  if (["signal_agreement", "daily_return", "roll20_return", "roll20_vol", "drawdown", "turnover", "gross_exposure", "daytrade_fill_rate"].includes(metric)) {
    return `${(x * 100).toFixed(2)}%`;
  }
  return x.toFixed(4);
}

function flagClass(flag, cusum) {
  if (flag === "severe" || cusum) return "neg";
  if (flag === "warning") return "warn";
  if (flag === "ok") return "pos";
  return "muted";
}

function statusLabel(status) {
  return { ok: "In range", warning: "Warning", severe: "Serious drift", awaiting: "Awaiting paper data" }[status] || (status || "–");
}

async function renderDrift() {
  const box = $("drift-panel");
  if (!box) return;
  try {
    const r = await fetch("/api/paper2/drift", { cache: "no-store" });
    const doc = await r.json().catch(() => ({}));
    if (r.status === 404) {
      box.innerHTML = `<p class="muted">No drift report yet. It is written after the daily paper run (autopilot or <code>run-paper-daily.ps1</code>).</p>`;
      return;
    }
    if (!r.ok) throw new Error(doc.error || r.statusText);
    const status = doc.status || "awaiting";
    const learn = doc.learning || {};
    const expected = doc.expected || {};
    const rows = Object.entries(doc.metrics || {}).map(([name, m]) => {
      const band = m.band || {};
      const alarm = m.cusumAlarm || m.seqAlarm;
      const flag = alarm && m.flag !== "severe" ? `${m.flag || "ok"} + persistent` : (m.flag || "–");
      return `<tr class="${flagClass(m.flag, alarm)}-row">
        <td>${esc(m.label || name)}</td>
        <td class="${flagClass(m.flag, alarm)}">${esc(fmt(name, m.value))}</td>
        <td>${esc(fmt(name, band.p50))}</td>
        <td>${esc(fmt(name, band.p5))} to ${esc(fmt(name, band.p95))}</td>
        <td>${esc(fmt(name, band.p1))} to ${esc(fmt(name, band.p99))}</td>
        <td class="${flagClass(m.flag, alarm)}">${esc(flag)}</td>
        <td>${alarm ? `<b class="neg">CUSUM ${Number(m.cusumPlus || 0).toFixed(1)} / ${Number(m.cusumMinus || 0).toFixed(1)}</b>` : `${Number(m.cusumPlus || 0).toFixed(1)} / ${Number(m.cusumMinus || 0).toFixed(1)}`}</td>
      </tr>`;
    }).join("");
    const notes = (doc.notes || []).map((n) => `<li>${esc(n)}</li>`).join("");
    box.innerHTML = `
      <div class="cards">
        <div class="card">
          <div class="label">Status</div>
          <div class="value ${flagClass(status === "severe" ? "severe" : status === "warning" ? "warning" : status === "ok" ? "ok" : null)}">${esc(statusLabel(status))}</div>
          <div class="muted small">as of ${esc(doc.asof || "–")} · ${esc(doc.source || "paper")} / ${esc(doc.account || "default")}</div>
        </div>
        <div class="card">
          <div class="label">Paper sessions scored</div>
          <div class="value">${esc(doc.paperSessions ?? 0)}</div>
          <div class="muted small">${esc(learn.metricsInsideP95 ?? 0)} / ${esc(learn.metricsCompared ?? 0)} metrics inside the 5–95% band</div>
        </div>
        <div class="card">
          <div class="label">Walk-forward reference</div>
          <div class="value small">${esc(expected.start || "–")} to ${esc(expected.end || "–")}</div>
          <div class="muted small">${esc(expected.sessions ?? "–")} OOS sessions${expected.daytradeTrades != null ? `, ${esc(expected.daytradeTrades)} day-trade fills in research` : ""}</div>
        </div>
        <div class="card">
          <div class="label">Persistent drift</div>
          <div class="value small">${(learn.persistentMetrics || []).length ? `<b class="neg">${esc((learn.persistentMetrics || []).join(", "))}</b>` : `<span class="pos">none</span>`}</div>
          <div class="muted small">CUSUM h=${esc(learn.cusumThreshold ?? 5)} · sequential |z̄√n| &gt; ${esc(learn.sequentialThreshold ?? 2.58)}</div>
        </div>
      </div>
      <table class="drift-table">
        <thead><tr>
          <th>Metric</th><th>Realized</th><th>Expected mid</th><th>5–95% band</th><th>1–99% band</th><th>Flag</th><th>CUSUM + / −</th>
        </tr></thead>
        <tbody>${rows || `<tr><td colspan="7" class="muted">No metrics yet.</td></tr>`}</tbody>
      </table>
      <p class="muted small">${esc(learn.note || "")}</p>
      ${notes ? `<ul class="muted small drift-notes">${notes}</ul>` : ""}
    `;
  } catch (error) {
    box.innerHTML = `<p class="error">Drift panel unavailable: ${esc(error.message)}</p>`;
  }
}

renderDrift();
setInterval(renderDrift, 60_000);
