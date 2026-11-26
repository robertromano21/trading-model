// Tax-lot panel for the paper account: unrealized ST/LT by lot, harvest candidates, estimated bill.
// Planning estimate only; not tax advice.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const money = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? "–" : Number(x).toLocaleString(undefined, { style: "currency", currency: "USD", minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (x, d = 1) => (x == null || Number.isNaN(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x == null || Number.isNaN(Number(x)) ? "–" : Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const cls = (x) => (x == null ? "" : Number(x) < 0 ? "neg" : Number(x) > 0 ? "pos" : "");
const tick = (t) => `<a href="/ticker.html?symbol=${encodeURIComponent(t)}">${esc(t)}</a>`;

function tableHtml(headers, rows) {
  if (!rows.length) {
    return `<thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody><tr><td class="muted" colspan="${headers.length}">None</td></tr></tbody>`;
  }
  return `<thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

async function loadTax() {
  const root = $("tax-panel");
  if (!root) return;
  const err = $("tax-error");
  try {
    const response = await fetch(`/api/paper2/status${$("dry")?.checked ? "?dry=1" : ""}`, { cache: "no-store" });
    const status = await response.json();
    if (!response.ok) throw new Error(status.error || response.statusText);
    const tax = status.tax;
    const settings = status.settings || {};
    const enabled = Boolean(tax?.enabled ?? settings.tax_aware);
    $("tax-enabled").textContent = enabled ? "On" : "Off (pre-tax behavior)";
    $("tax-enabled").className = "value " + (enabled ? "pos" : "muted");
    if (!tax) {
      $("tax-bill-cards").innerHTML = `<p class="muted">No tax ledger yet. Enable <code>tax_aware</code> on the paper account and run the paper trader to start tracking lots.</p>`;
      return;
    }
    const bill = tax.taxBill || {};
    const rates = bill.rates || {};
    $("tax-bill-cards").innerHTML = `
      <div class="cards">
        <div class="card"><div class="label">Unrealized short-term</div><div class="value ${cls(tax.unrealizedShortTerm)}">${money(tax.unrealizedShortTerm)}</div></div>
        <div class="card"><div class="label">Unrealized long-term</div><div class="value ${cls(tax.unrealizedLongTerm)}">${money(tax.unrealizedLongTerm)}</div></div>
        <div class="card"><div class="label">Unpaid realized tax</div><div class="value">${money(bill.unpaidRealizedTax)}</div>
          <div class="muted small">Paid to date ${money(bill.taxPaidToDate)}</div></div>
        <div class="card"><div class="label">Est. bill if liquidated</div><div class="value">${money(bill.estimatedBillIfLiquidated)}</div>
          <div class="muted small">of which unrealized ${money(bill.liquidationTax)}</div></div>
        <div class="card"><div class="label">Assumed rates</div><div class="value small">ST ${pct(rates.shortTerm, 1)} · LT ${pct(rates.longTerm, 1)} · QD ${pct(rates.qualifiedDividend, 1)}</div>
          <div class="muted small">Defaults 32% / 15% federal; state and NIIT 0 unless set.</div></div>
      </div>`;
    const lots = tax.openLots || [];
    $("tax-lots").innerHTML = tableHtml(
      ["Lot", "Ticker", "Shares", "Acquired", "Cost / sh", "Price", "Unrealized", "Character", "Days to LT"],
      lots.map((r) => [esc(r.lotId), tick(r.ticker), num(r.shares, 2), esc(r.holdingPeriodStart || r.acquired),
        money(r.costPerShare), money(r.price), `<span class="${cls(r.unrealized)}">${money(r.unrealized)}</span>`,
        esc(r.character), r.character === "long-term" ? "—" : esc(r.daysToLongTerm)])
    );
    const harvest = tax.harvestCandidates || [];
    $("tax-harvest").innerHTML = tableHtml(
      ["Ticker", "Shares", "Unrealized loss", "Loss %", "Substitute", "Wash window"],
      harvest.map((r) => [tick(r.ticker), num(r.shares, 2), `<span class="neg">${money(r.unrealized)}</span>`,
        pct(r.lossPct, 1), r.substitute ? tick(r.substitute) : "<span class='muted'>none in universe</span>",
        r.washWindow ? "<span class='neg'>yes</span>" : "no"])
    );
    const years = Object.values(bill.years || {});
    $("tax-years").innerHTML = tableHtml(
      ["Year", "ST gain", "LT gain", "Qualified div", "Ordinary div", "Wash disallowed", "Tax accrued", "Tax paid"],
      years.map((y) => [esc(y.year), `<span class="${cls(y.st_gain)}">${money(y.st_gain)}</span>`,
        `<span class="${cls(y.lt_gain)}">${money(y.lt_gain)}</span>`, money(y.qualified_div), money(y.ordinary_div),
        money(y.disallowed_wash), money(y.tax_accrued), money(y.tax_paid)])
    );
    $("tax-disclaimer").textContent = tax.disclaimer || "Planning estimate only; not tax advice.";
    if (err) err.hidden = true;
  } catch (error) {
    if (err) {
      err.textContent = `Tax panel unavailable: ${error.message}`;
      err.hidden = false;
    }
  }
}

loadTax();
setInterval(loadTax, 60_000);
document.getElementById("dry")?.addEventListener("change", loadTax);
