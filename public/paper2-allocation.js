// Regime-allocation panel on /paper2. Reads /api/regime-allocation (research only; paper account unchanged).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x == null || Number.isNaN(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);

function cls(x) {
  if (x == null || Number.isNaN(Number(x))) return "";
  return Number(x) >= 0 ? "pos" : "neg";
}

function recClass(decision) {
  if (decision === "recommend_challenger") return "pos";
  if (decision === "stay_static_with_interest") return "warn";
  return "";
}

async function renderAllocation() {
  const box = $("allocation-panel");
  if (!box) return;
  try {
    const r = await fetch("/api/regime-allocation", { cache: "no-store" });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.error || r.statusText);
    const rec = d.recommendation || {};
    const st = d.currentState || {};
    const w = st.weights || rec.weights || {};
    const oe = st.overextensionComponents || {};
    const table = d.stateWeightTable || [];
    const currentKey = `${st.volBucket || ""}|${st.volDirection === "flat" ? "falling" : (st.volDirection || "")}`;
    const leaders = (d.results || []).slice(0, 8);
    const cost = d.overextensionCost || {};
    const data = d.data || {};
    const gates = rec.gates || {};
    box.innerHTML = `
      <p class="${recClass(rec.decision)}">${esc(rec.text || "No recommendation yet.")}</p>
      <p class="muted small">Live paper account is unchanged. As of ${esc(st.asOf || d.generatedAt || "–")}.
        Champion <code>${esc(rec.champion || "–")}</code> · decision <code>${esc(rec.decision || "–")}</code>
        · robust ${rec.robust ? "yes" : "no"}.</p>
      <div class="cards">
        <div class="card"><div class="label">Current state</div>
          <div class="value">${esc(st.volBucket || "–")} <span class="muted small">${esc(st.volDirection || "")}</span></div>
          <div class="muted small">VIX ${num(st.vix, 1)} (pct ${pct(st.vixPercentile, 0)}) · term ${num(st.vixRatio, 2)}
            · RV21 ${pct(st.realizedVol21, 1)} (pct ${pct(st.realizedVolPercentile, 0)})
            · trend ${esc(st.trend || "–")} · DD ${pct(st.drawdown, 1)}</div></div>
        <div class="card"><div class="label">Overextension</div>
          <div class="value">${num(st.overextension, 2)}</div>
          <div class="muted small">stretch z ${num(oe.stretch_z, 2)} · RSI ${num(oe.rsi14, 0)}
            · channel ${num(oe.channel_pos, 2)} · breadth ${pct(oe.breadth, 0)}
            · complacency ${num(oe.oe_complacency, 2)} · value ${num(oe.oe_value, 2)}
            · credit tight ${num(oe.oe_credit, 2)}
            ${st.washout ? " · washout yes" : ""}
            ${st.blackSwan ? " · black swan flag" : ""}</div></div>
        <div class="card"><div class="label">Recommended weights now</div>
          <div class="value small">risk ${pct(w.riskOn, 0)} · def ${pct(w.defensive, 0)} · hedge ${pct(w.hedge, 0)}</div>
          <div class="muted small">Research path only. ${esc(st.valueSource ? `Value: ${st.valueSource}` : "Earnings yield not in local data.")}</div></div>
      </div>
      <h3>State-to-weight table</h3>
      <p class="muted small">Average champion weights in each confirmed vol state (hysteresis already applied). The current row is highlighted.</p>
      <table class="alloc-table"><thead><tr>
        <th>Vol bucket</th><th>Direction</th><th>Sessions</th><th>Risk-on</th><th>Defensive</th><th>Hedge</th><th>OE mean</th>
      </tr></thead><tbody>
        ${table.map((row) => {
          const key = `${row.volBucket}|${row.volDirection}`;
          const on = key === currentKey ? " class=\"alloc-current\"" : "";
          return `<tr${on}><td>${esc(row.volBucket)}</td><td>${esc(row.volDirection)}</td><td>${row.sessions}</td>
            <td>${pct(row.riskOn, 0)}</td><td>${pct(row.defensive, 0)}</td><td>${pct(row.hedge, 0)}</td>
            <td>${num(row.overextensionMean, 2)}</td></tr>`;
        }).join("") || "<tr><td colspan=\"7\">No state table yet.</td></tr>"}
      </tbody></table>
      <h3>Pre-holdout leaders vs plain TSMOM</h3>
      <table><thead><tr>
        <th>Scheme</th><th>Sharpe</th><th>CAGR</th><th>Max DD</th><th>Calmar</th><th>Down cap.</th><th>Holdout Sharpe</th><th>Holdout DD</th>
      </tr></thead><tbody>
        ${leaders.map((row) => `<tr>
          <td>${esc(row.id)}${row.id === rec.champion ? " ★" : ""}${row.id === "static_100_0_0" ? " (TSMOM)" : ""}</td>
          <td class="${cls(row.preSharpe)}">${num(row.preSharpe)}</td>
          <td class="${cls(row.preCagr)}">${pct(row.preCagr)}</td>
          <td class="${cls(row.preMaxDrawdown)}">${pct(row.preMaxDrawdown)}</td>
          <td>${num(row.preCalmar)}</td>
          <td>${num(row.preDownsideCapture)}</td>
          <td class="${cls(row.holdoutSharpe)}">${num(row.holdoutSharpe)}</td>
          <td class="${cls(row.holdoutMaxDrawdown)}">${pct(row.holdoutMaxDrawdown)}</td>
        </tr>`).join("")}
      </tbody></table>
      <h3>Overextension hedge: melt-up drag vs drawdown saved</h3>
      <p class="muted small">${esc(cost.note || "Scheme (f) scales the hedge up with overextension even before vol rises.")}
        Melt-up sessions ${esc(cost.meltUpSessions ?? "–")} · relative return vs TSMOM
        <span class="${cls(cost.meltUpRelative)}">${pct(cost.meltUpRelative)}</span>
        · drawdown saved <span class="${cls(cost.drawdownSavedAbs)}">${pct(cost.drawdownSavedAbs)}</span>
        (scheme DD ${pct(cost.schemeMaxDrawdown)} vs TSMOM ${pct(cost.baselineMaxDrawdown)}).</p>
      <p class="muted small">Robustness: DSR ${num(gates.deflatedSharpeProb, 2)} · PBO ${num(gates.pbo, 2)}
        · SPA p vs TSMOM ${num(gates.spaPVsTsmom, 3)}
        · holdout Sharpe vs TSMOM ${num(gates.holdoutSharpeVsTsmom, 2)}.
        Holdout ${esc(d.holdoutStart || "–")} (one logged look; selection used pre-holdout data only).
        ${esc((data.notes || []).slice(0, 3).join(" "))}
        Missing: ${esc((data.missing || []).join(", ") || "none")}.</p>`;
  } catch (error) {
    box.innerHTML = `<p class="muted">Regime allocation study unavailable: ${esc(error.message)}</p>`;
  }
}
renderAllocation();
setInterval(renderAllocation, 60_000);
