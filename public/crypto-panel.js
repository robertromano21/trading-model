// Crypto PAPER shadow accounts for /paper2 (engine/crypto_paper.py, ids 200-219). Simulation only: no orders,
// no previews, ever. Prices are Coinbase daily closes (UTC) with Schwab Crypto fees (0.75% per side).
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = (x, d = 2) => (x == null || !Number.isFinite(Number(x)) ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x == null || !Number.isFinite(Number(x)) ? "–" : `${(Number(x) * 100).toFixed(d)}%`);
const money = (x) => (x == null || !Number.isFinite(Number(x)) ? "–" : `$${Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 })}`);
const cls = (x) => (x == null ? "muted" : Number(x) > 0 ? "pos" : Number(x) < 0 ? "neg" : "muted");

function research(r) {
  if (!r) return `<p class="muted">No crypto backtest yet. Run <code>python engine/crypto_research.py run</code>.</p>`;
  const c = r.champion || {}, b = r.btcHold || {}, w = r.walkForward || {};
  const fam = Object.entries(r.dollarFamilies || {}).map(([k, f]) =>
    `<tr><td>${esc(k)}</td><td>${f.n}</td><td>${f.improvedSharpe}</td><td>${f.improvedMdd}</td><td class="${cls(f.medianSharpeDelta)}">${num(f.medianSharpeDelta)}</td></tr>`).join("");
  return `<p>Backtest ${esc(r.designHash)} (${esc(r.generatedAt)} ET, pre-holdout 2017 to 2026-07-08): BTC hold Sharpe
    ${num(b.pre?.sharpe)} (max DD ${pct(b.pre?.maxDrawdown)}); best variant #${esc(c.id)} ${esc(c.label)} Sharpe ${num(c.pre?.sharpe)}
    (max DD ${pct(c.pre?.maxDrawdown)}); walk-forward picks Sharpe ${num(w.sharpe)}. PBO ${num(r.pbo, 2)}, SPA vs BTC p ${num(r.spaVsBtcP, 3)},
    deflated Sharpe ${num(r.dsr, 3)}.</p>
    <details><summary>Dollar / macro overlays vs the same strategy without them (SPA p ${num(r.dollarPairsSpaP, 3)})</summary>
    <table><tr><th>Overlay</th><th>Pairs</th><th>Sharpe better</th><th>Drawdown better</th><th>Median Sharpe change</th></tr>${fam}</table></details>
    <p class="muted">Full report: Massive/Research/crypto/crypto_backtest_report.md</p>`;
}

function render(doc) {
  const rows = (doc.accounts || []).map((a) => `<tr>
      <td>${a.id}</td><td title="${esc(a.detail)}">${esc(a.label)}${a.dollarAware ? ' <span class="muted">($)</span>' : ""}</td>
      <td>${a.schwabTradableNow ? "yes" : '<span class="muted">partly (SOL/AVAX/LINK announced)</span>'}</td>
      <td>${money(a.equity)}</td><td class="${cls(a.totalReturn)}">${pct(a.totalReturn, 2)}</td><td>${pct(a.maxDrawdown)}</td>
      <td>${esc(Object.entries(a.targetWeights || {}).map(([k, v]) => `${k} ${pct(v, 0)}`).join(", ") || "cash")}</td>
      <td>${num(a.preHoldoutBacktest?.sharpe)}</td><td>${pct(a.preHoldoutBacktest?.maxDrawdown)}</td></tr>`).join("");
  $("crypto-panel").innerHTML = `
    <p>${esc(doc.note || "")}</p>
    <p>Forward record from ${esc(doc.forwardStart)} through ${esc(doc.dataEnd)} (${doc.days ?? 0} UTC day(s)); updated ${esc(doc.updatedAt)} ET.
      BTC baseline ${pct(doc.btcBaselineReturn, 2)}. ($) = dollar / macro aware.</p>
    <table><tr><th>Acct</th><th>Strategy</th><th>Tradable at Schwab now</th><th>Equity</th><th>Return</th><th>Max DD</th>
      <th>Current target</th><th>Backtest Sharpe</th><th>Backtest max DD</th></tr>${rows}</table>
    ${research(doc.research)}`;
}

async function renderCrypto() {
  const box = $("crypto-panel");
  if (!box) return;
  try {
    const r = await fetch("/api/crypto/status", { cache: "no-store" });
    if (r.status === 404) {
      box.innerHTML = `<p class="muted">No crypto paper accounts yet. Run <code>python engine/crypto_paper.py run --fetch</code>.</p>`;
      return;
    }
    render(await r.json());
  } catch (error) {
    box.innerHTML = `<p class="neg">Crypto panel failed: ${esc(error.message)}</p>`;
    console.error("crypto panel", error);
  }
}

renderCrypto();
setInterval(renderCrypto, 5 * 60 * 1000);
