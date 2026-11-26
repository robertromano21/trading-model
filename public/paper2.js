const $ = (id) => document.getElementById(id);
const pct = (x, d = 2) => (x === null || x === undefined ? "–" : `${(x * 100).toFixed(d)}%`);
const num = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const money = (x) => (x === null || x === undefined ? "–" : `$${num(x, 2)}`);

function table(element, columns, rows) {
  element.replaceChildren();
  const head = element.createTHead().insertRow();
  for (const [label] of columns) {
    const th = document.createElement("th");
    th.textContent = label;
    head.appendChild(th);
  }
  const body = element.createTBody();
  if (!rows || rows.length === 0) {
    const cell = body.insertRow().insertCell();
    cell.colSpan = columns.length;
    cell.textContent = "None";
    cell.className = "muted";
    return;
  }
  for (const row of rows) {
    const tr = body.insertRow();
    for (const [, format] of columns) {
      const cell = tr.insertCell();
      cell.textContent = format(row);
    }
  }
}

function card(label, value) {
  const div = document.createElement("div");
  div.className = "card";
  const l = document.createElement("div");
  l.className = "label";
  l.textContent = label;
  const v = document.createElement("div");
  v.className = "value";
  v.textContent = value;
  div.append(l, v);
  return div;
}

function drawEquity(history) {
  const canvas = $("equity");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (!history || history.length < 2) return;
  const values = history.map((row) => row.equity);
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  context.strokeStyle = "#6cb6ff";
  context.lineWidth = 2;
  context.beginPath();
  values.forEach((value, index) => {
    const x = 10 + (index / (values.length - 1)) * (canvas.width - 20);
    const y = canvas.height - 15 - ((value - min) / span) * (canvas.height - 30);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();
  context.fillStyle = "#9aa4ad";
  context.font = "12px system-ui";
  context.fillText(`${history[0].date}  ${money(values[0])}`, 12, 14);
  context.fillText(`${history.at(-1).date}  ${money(values.at(-1))}`, canvas.width - 220, 14);
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || response.statusText);
  return body;
}

async function loadStatus() {
  $("error").hidden = true;
  try {
    const status = await getJson(`/api/paper2/status${$("dry").checked ? "?dry=1" : ""}`);
    $("updated").textContent = `Status written ${status.generatedAt}${status.dryRun ? " (dry run, not saved)" : ""}`;
    const last = status.equityHistory?.at(-1);
    const recon = status.reconciliation?.[0];
    $("cards").replaceChildren(
      card("Account", status.account),
      card("Equity", money(last?.equity ?? status.initialEquity)),
      card("Return since start", last ? pct(last.equity / status.initialEquity - 1) : "–"),
      card("Cash", money(status.cash)),
      card("Gross exposure", last ? pct(last.gross, 1) : "–"),
      card("Last processed", status.lastProcessed ?? "–"),
      card("Avg slippage (bps)", num(status.slippage?.avgSlippageBps, 1)),
      card("Paper vs shadow", recon ? `${num(recon.equity_diff_pct, 3)}%` : "–"),
      card("Circuit breaker", status.circuitBreakerRemaining ? "ACTIVE" : "off"),
    );
    drawEquity(status.equityHistory);
    table($("positions"), [["Ticker", (r) => r.ticker], ["Shares", (r) => num(r.shares, 0)], ["Avg cost", (r) => money(r.avg_cost)],
      ["Last", (r) => money(r.last_price)], ["Last date", (r) => r.last_price_date], ["Entry", (r) => r.entry_date],
      ["Stale sessions", (r) => r.stale_sessions]], status.positions);
    table($("orders"), [["Decided", (r) => r.decision_date], ["Ticker", (r) => r.ticker], ["Side", (r) => r.side],
      ["Shares", (r) => num(r.shares, 0)], ["Ref price", (r) => money(r.ref_price)], ["Target wt", (r) => pct(r.target_weight, 1)],
      ["Current wt", (r) => pct(r.current_weight, 1)], ["Sleeve", (r) => r.sleeve ?? ""], ["Reason", (r) => r.reason],
      ["Execution", (r) => r.execution], ["Status", (r) => r.status]], status.openOrders);
    table($("targets"), [["Ticker", (r) => r.ticker], ["Target weight", (r) => pct(r.target_weight, 1)],
      ["Sleeve", (r) => r.sleeve ?? ""], ["Stop", (r) => (r.stop ? money(r.stop) : "")]], status.targets);
    table($("recon"), [["Date", (r) => r.date], ["Paper equity", (r) => money(r.paper_equity)], ["Shadow equity", (r) => money(r.shadow_equity)],
      ["Diff %", (r) => num(r.equity_diff_pct, 3)], ["Max weight diff", (r) => pct(r.max_weight_diff, 2)]], status.reconciliation);
    table($("decisions"), [["Decided", (r) => r.decision_date ?? ""], ["Filled", (r) => r.fill_date ?? ""], ["Ticker", (r) => r.ticker],
      ["Action", (r) => r.action], ["Reason", (r) => r.reason], ["Shares", (r) => num(r.shares, 0)], ["Fill", (r) => (r.fill_price ? money(r.fill_price) : "")],
      ["Cost bps", (r) => num(r.cost_bps, 1)], ["Slippage bps", (r) => num(r.slippage_bps, 1)], ["Status", (r) => r.status]], status.recentDecisions);
    table($("runs"), [["Run", (r) => r.run_id], ["As of", (r) => r.asof], ["Mode", (r) => r.mode], ["Finished", (r) => r.finished_at ?? ""],
      ["Status", (r) => r.status], ["Sessions", (r) => r.sessions_processed], ["Equity", (r) => money(r.equity)], ["Message", (r) => r.message ?? ""]], status.runs);
  } catch (error) {
    $("error").textContent = error.message;
    $("error").hidden = false;
  }
}

async function loadResearch() {
  try {
    const summary = await getJson("/api/research/summary");
    const rows = Object.entries(summary.backtests?.results ?? {}).map(([name, r]) => ({ name, ...r.performance, trades: r.tradeStats?.trades }));
    for (const [name, w] of Object.entries(summary.walkForward ?? {})) rows.push({ name: `walk-forward OOS: ${name}`, ...w.outOfSample.performance, trades: w.outOfSample.tradeStats?.trades });
    table($("research"), [["Strategy", (r) => r.name], ["Start", (r) => r.start], ["CAGR", (r) => pct(r.cagr, 1)], ["Vol", (r) => pct(r.volatility, 1)],
      ["Sharpe", (r) => num(r.sharpe)], ["Max DD", (r) => pct(r.maxDrawdown, 1)], ["Turnover/yr", (r) => num(r.turnoverPerYear, 1)],
      ["Avg gross", (r) => num(r.avgGrossExposure)], ["Trades", (r) => r.trades ?? ""]], rows);
    $("research-note").textContent = `Generated ${summary.generatedAt} on ${summary.data.start} to ${summary.data.end}. Full report: Massive/Research/research_report.md`;
  } catch (error) {
    $("research-note").textContent = error.message;
  }
}

const hhmm = (m) => (m === null || m === undefined ? "" : `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`);

async function loadStockWatch() {
  const note = $("dt-watch");
  try {
    const status = await getJson("/api/autopilot");
    const watch = status.last?.daytrade_watch;
    const active = status.active?.some((task) => task.name === "daytrade_watch");
    const rows = watch?.session === status.market?.date ? watch.swingTargets ?? [] : [];
    const latest = watch?.finishedAt ? `Last check ${new Date(watch.finishedAt).toLocaleString()}. ${watch.summary || ""}` : "";
    const mode = !status.enabled ? "Autopilot is off." : active ? "Checking stock targets now." :
      "Stock targets are checked during NYSE sessions.";
    note.textContent = [mode, latest].filter(Boolean).join(" ");
    $("dt-swing-watch").hidden = !rows.length;
    table($("dt-swing-targets"), [["Account", (r) => r.account], ["Ticker", (r) => r.ticker], ["Decision", (r) => r.signal],
      ["Target", (r) => r.target_weight == null ? "" : pct(r.target_weight)], ["Latest bar", (r) => money(r.price)],
      ["Stop", (r) => money(r.stop)]], rows);
  } catch (error) {
    note.textContent = `Stock watch unavailable (${error.message}).`;
  }
}

async function loadDayTrading() {
  try {
    const status = await getJson(`/api/daytrade/status${$("dry").checked ? "?dry=1" : ""}`);
    $("dt-cards").replaceChildren(
      card("Equity", money(status.equity)),
      card("Return since start", status.initialEquity ? pct(status.equity / status.initialEquity - 1) : "–"),
      card("Last session", status.lastSession ?? "–"),
      card("PDT used (last 4 sessions)", status.pdt ? `${status.pdt.usedInLast4Sessions} / ${status.pdt.limit ?? "∞"}` : "–"),
    );
    table($("dt-trades"), [["Session", (r) => r.session_date], ["Symbol", (r) => r.symbol], ["In", (r) => hhmm(r.entry_minute)],
      ["Out", (r) => hhmm(r.exit_minute)], ["Shares", (r) => num(r.shares, 0)], ["Entry", (r) => money(r.entry_price)],
      ["Exit", (r) => money(r.exit_price)], ["Reason", (r) => r.reason], ["P&L", (r) => money(r.pnl)], ["R", (r) => num(r.r_multiple)]], status.trades);
    table($("dt-quotes"), [["Symbol", (r) => r.symbol], ["Fetched", (r) => r.fetched_at], ["Last", (r) => money(r.last)],
      ["Bid", (r) => money(r.bid)], ["Ask", (r) => money(r.ask)]], status.latestQuotes);
  } catch (error) {
    $("dt-cards").replaceChildren(card("Day trading", error.message));
  }
  try {
    const research = await getJson("/api/daytrade/research");
    const rows = [["Full period (defaults)", research.fullPeriodDefaults], ["Full period, no PDT limit", research.fullPeriodDefaultsNoPdt],
      ["Walk-forward OOS", research.walkForward.outOfSample], ["1-minute check", research.oneMinuteCheckChosenParams]]
      .filter(([, r]) => r).map(([name, r]) => ({ name, ...r, ...r.performance }));
    table($("dt-research"), [["Run", (r) => r.name], ["Sessions", (r) => r.sessions], ["Trades", (r) => r.trades],
      ["Win rate", (r) => pct(r.winRate, 0)], ["Avg R", (r) => num(r.avgR)], ["Total return", (r) => pct(r.totalReturn)],
      ["Sharpe", (r) => num(r.sharpe)], ["Max DD", (r) => pct(r.maxDrawdown)]], rows);
    $("dt-note").textContent = `Schwab ${research.data.barMinutes ?? 5}-minute bars ${research.data.first} to ${research.data.last} (${research.data.fiveMinuteSessions} sessions). Generated ${research.generatedAt}.`;
  } catch (error) {
    $("dt-note").textContent = error.message;
  }
}

$("dry").addEventListener("change", loadDayTrading);
loadDayTrading();
loadStockWatch();
$("dry").addEventListener("change", loadStatus);
loadStatus();
loadResearch();
setInterval(loadStatus, 60_000);
setInterval(loadStockWatch, 60_000);
