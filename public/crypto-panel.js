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

const dateTime = (value) => value ? new Date(value).toLocaleString() : "—";

function ranked(board) {
  if (!board || board.bars < 2) return "<p>Waiting for shared trading data to rank the strategies.</p>";
  const rows = board.rows.map((row) => `<tr><td>${esc(row.label)}</td>
      <td class="${cls(row.return)}">${pct(row.return, 2)}</td><td>${pct(row.maxDrawdown, 2)}</td><td>${row.fills}</td></tr>`).join("");
  const delta = board.dollarDelta == null ? "Waiting for overlapping dollar prices." :
    `Dollar filter vs. matched momentum ${(100 * board.dollarDelta).toFixed(2)} percentage points across ${board.dollarBars} bars with dollar coverage.`;
  return `<p>${board.bars} shared bars, ${esc(dateTime(board.start))} through ${esc(dateTime(board.end))}. ${delta}</p>
    <table><tr><th>Strategy</th><th>Net return</th><th>Max drawdown</th><th>Fills</th></tr>${rows}</table>`;
}

async function renderIntraday() {
  const box = $("crypto-intraday");
  if (!box) return;
  try {
    const r = await fetch("/api/crypto/intraday", { cache: "no-store" });
    if (r.status === 404) {
      box.textContent = "Waiting for the first crypto intraday check.";
      return;
    }
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const doc = await r.json();
    const opened = new Set([...box.querySelectorAll("details[open]")].map((el) => el.id));
    const rows = doc.accounts.map((a) => `<tr><td>${a.id}</td><td>${esc(a.symbol)}</td><td>${esc(a.strategy)}</td><td>${money(a.equity)}</td>
      <td class="${cls(a.totalReturn)}">${pct(a.totalReturn, 2)}</td><td>${num(a.qty, 5)}</td>
      <td>${esc(a.message || a.decision?.signal || "Waiting")}</td><td>${esc(a.decision?.reason)}</td>
      <td>${esc(dateTime(a.decision?.at))}</td></tr>`).join("");
    const fills = doc.recentFills.map((f) => `<tr><td>${esc(dateTime(f.at))}</td><td>${f.account}</td>
      <td>${esc(f.symbol)}</td><td>${esc(f.side)}</td><td>${num(f.qty, 5)}</td><td>${num(f.price)}</td>
      <td>${num(f.fee)}</td></tr>`).join("");
    box.innerHTML = `<p>Checked ${esc(dateTime(doc.updatedAt))}. Decisions use completed ${doc.intervalMinutes}-minute Coinbase bars.
      Paper fills include ${pct(doc.fees.perSide, 2)} fees per side and ${pct(doc.fees.slippage, 2)} slippage.</p>
      <h4>Intraday strategy leaderboard</h4>
      <p>Equal BTC and ETH weighting over shared bars, ranked after costs. The dollar comparison uses one-minute UUP prices when available.${doc.dollar?.lastBar ?
        ` Last UUP bar ${esc(dateTime(doc.dollar.lastBar))}.` : ""}</p>
      ${ranked(doc.leaderboard)}
      <details id="crypto-replay" ${opened.has("crypto-replay") ? "open" : ""}><summary>Recent historical replay</summary>
        <p>A short replay of recent prices. Results are preliminary.</p>${ranked(doc.replayLeaderboard)}</details>
      <details id="crypto-accounts" ${opened.has("crypto-accounts") ? "open" : ""}><summary>Accounts and latest decisions</summary>
        <table><tr><th>Acct</th><th>Coin</th><th>Strategy</th><th>Equity</th><th>Return</th><th>Position</th><th>Decision</th>
        <th>Reason</th><th>Bar close</th></tr>${rows}</table></details>
      ${fills ? `<details id="crypto-fills" ${opened.has("crypto-fills") ? "open" : ""}><summary>Recent paper fills</summary><table><tr><th>Time</th><th>Acct</th><th>Coin</th>
        <th>Side</th><th>Quantity</th><th>Price</th><th>Fee</th></tr>${fills}</table></details>` : "<p>No intraday fills yet.</p>"}`;
  } catch (error) {
    box.textContent = `Crypto intraday status unavailable (${error.message}).`;
  }
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
renderIntraday();
setInterval(renderIntraday, 60_000);
