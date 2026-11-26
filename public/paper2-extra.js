// Extra paper-trading panels: risk & regime state, circuit breaker, PDT, fills and day-trading daily log.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (x) => (x == null ? "–" : `$${Number(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
async function json(url) {
  const r = await fetch(url, { cache: "no-store" });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}
function tableHtml(headers, rows) {
  return `<thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
}
const tick = (t) => `<a href="/ticker.html?symbol=${encodeURIComponent(t)}">${esc(t)}</a>`;
async function render() {
  const box = $("risk-state");
  try {
    const [tables, swing, day] = await Promise.all([json("/api/paper2/tables?account=default&limit=200"),
      json("/api/paper2/status").catch(() => null), json("/api/daytrade/status").catch(() => null)]);
    const run = (tables.runs ?? []).find((r) => r.account === "default" && r.mode !== "dry-run") ?? tables.runs?.[0];
    const diag = run?.diagnostics ? JSON.parse(run.diagnostics) : {};
    const mr = diag.strategy?.meanrev ?? {}, ts = diag.strategy?.tsmom ?? {}, sleeves = diag.strategy?.sleeveScale ?? {}, hmm = diag.strategy?.regime ?? {};
    const hmmAlloc = hmm.allocation ?? {};
    const hmmProbs = Object.entries(hmm.probabilities || {}).map(([k, v]) => `${esc(k.replaceAll("_", " "))} ${(Number(v) * 100).toFixed(0)}%`).join(" · ") || "–";
    const acct = (tables.accounts ?? []).find((a) => a.name === "default");
    const settings = swing?.settings ?? (acct?.settings ? JSON.parse(acct.settings) : {});
    const blocked = acct?.blocked ? JSON.parse(acct.blocked) : [];
    const lastDay = (tables.dtDaily ?? []).find((d) => d.account === "daytrade");
    const dtSkip = lastDay?.skipped ? JSON.parse(lastDay.skipped) : {};
    const pdt = day?.pdt ?? {};
    const flag = (on, okText, badText) => (on ? `<b class="neg">${badText}</b>` : `<span class="pos">${okText}</span>`);
    box.innerHTML = `
      <div class="cards">
        <div class="card"><div class="label">Market regime (HMM)</div><div class="value">${hmm.regime ? esc(String(hmm.regime).replaceAll("_", " ")) : (hmm.model === "fixed" ? "fixed filter" : "–")}</div>
          <div class="muted small">${hmmProbs}<br>TSMOM ${hmmAlloc.tsmomScale == null ? "–" : pct(hmmAlloc.tsmomScale, 0)} · IEF ${hmmAlloc.ief == null ? "–" : pct(hmmAlloc.ief, 0)} · TLT ${hmmAlloc.tlt == null ? "–" : pct(hmmAlloc.tlt, 0)} · mean-rev ${hmmAlloc.meanrevOn ? "ON" : hmmAlloc.meanrevOn === false ? "OFF" : "–"}</div></div>
        <div class="card"><div class="label">Mean-reversion regime</div><div class="value">${mr.regimeOn == null ? "–" : mr.regimeOn ? "ON" : "OFF"}</div>
          <div class="muted small">SPY vol pct ${mr.marketVolPercentile != null ? (mr.marketVolPercentile * 100).toFixed(0) + "%" : "–"} (≤60% needed), trend t ${mr.marketTrendT?.toFixed?.(2) ?? "–"} (|t|<1.5)${mr.regimeSource ? ` · gate ${esc(mr.regimeSource)}` : ""}</div></div>
        <div class="card"><div class="label">TSMOM active signals</div><div class="value">${esc(ts.activeSignals ?? "–")}</div>
          <div class="muted small">Sleeve scales: ${Object.entries(sleeves).map(([k, v]) => `${esc(k)} ${Number(v).toFixed(2)}`).join(", ") || "–"}</div></div>
        <div class="card"><div class="label">Swing circuit breaker</div><div class="value">${flag((swing?.circuitBreakerRemaining ?? acct?.breaker_remaining ?? 0) > 0, "Clear", `ACTIVE (${swing?.circuitBreakerRemaining ?? acct?.breaker_remaining} sessions)`)}</div>
          <div class="muted small">Daily loss limit ${pct(settings.daily_loss_limit, 0)}</div></div>
        <div class="card"><div class="label">Risk caps</div><div class="value small">gross ≤ ${pct(settings.max_gross, 0)}, name ≤ ${pct(settings.max_name_weight, 0)}, ≤ ${esc(settings.max_positions)} names</div>
          <div class="muted small">min trade ${pct(settings.min_trade_weight, 0)}, min order ${money(settings.min_order_value)}, whole shares ${settings.whole_shares ? "yes" : "no"}, advisory ${esc(settings.advisory_mode)}</div></div>
        <div class="card"><div class="label">Stop-blocked re-entries</div><div class="value small">${blocked.length ? blocked.map(tick).join(", ") : "none"}</div></div>
        <div class="card"><div class="label">Day-trading PDT</div><div class="value">${esc(pdt.usedInLast4Sessions ?? "–")} / ${esc(pdt.limit ?? 3)}</div>
          <div class="muted small">day trades in the last 4 sessions; limit ${esc(pdt.limit)} per ${esc(pdt.windowSessions)} sessions below ${money(pdt.appliesBelowEquity)}</div></div>
        <div class="card"><div class="label">Day-trading breaker / skips</div><div class="value small">${lastDay ? `${esc(lastDay.session_date)}: ${Object.entries(dtSkip).filter(([, n]) => n).map(([k, n]) => `${esc(k)} ${n}`).join(", ") || "no skips"}` : "no sessions yet"}</div>
          <div class="muted small">${dtSkip.breaker ? '<b class="neg">breaker tripped</b>' : "breaker not tripped"}</div></div>
      </div>
      <p class="muted small">From the latest swing run ${esc(run?.asof ?? "–")} (${esc(run?.mode ?? "")}). Both accounts are separate $10k paper accounts.</p>`;
    $("fills").innerHTML = tableHtml(["Date", "Ticker", "Shares", "Price", "Effective", "Notional", "Cost bps", "Slippage bps"],
      (tables.fills ?? []).map((f) => [esc(f.fill_date), tick(f.ticker), esc(f.shares), money(f.price), money(f.effective_price), money(f.notional),
        Number(f.cost_bps ?? 0).toFixed(1), f.slippage_bps == null ? "–" : Number(f.slippage_bps).toFixed(1)]));
    $("dt-daily").innerHTML = tableHtml(["Session", "Start equity", "P&L", "Equity", "Trades", "Skipped"],
      (tables.dtDaily ?? []).filter((d) => d.account === "daytrade").slice(0, 30).map((d) => [esc(d.session_date), money(d.start_equity), money(d.pnl),
        money(d.equity), esc(d.trades), esc(d.skipped)]));
  } catch (error) {
    box.textContent = `Risk state unavailable: ${error.message}`;
  }
}
render();
setInterval(render, 60_000);
