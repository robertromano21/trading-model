// Manual Schwab trading panel: Buy/Sell buttons on tickers -> ticket -> preview -> explicit Confirm.
// Nothing here runs automatically; every order needs a click on Confirm after a fresh preview.
const $ = (id) => document.getElementById(id);
const money = (x) => (x === null || x === undefined ? "–" : `$${Number(x).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
let lastPreview = null;
let submitting = false;
const TICKET_HTML = `
  <form method="dialog" class="ticket">
    <h3>Order ticket (real Schwab order)</h3>
    <label>Symbol <input id="t-symbol" autocomplete="off" required></label>
    <label>Side <select id="t-side"><option>BUY</option><option>SELL</option></select></label>
    <label>Amount <input id="t-amount" type="number" min="0" step="any" required>
      <select id="t-mode"><option value="shares">shares</option><option value="dollars">dollars</option></select></label>
    <label>Type <select id="t-type"><option>MARKET</option><option>LIMIT</option></select></label>
    <label id="t-limit-row" hidden>Limit price <input id="t-limit" type="number" min="0" step="0.01"></label>
    <label>Time in force <select id="t-tif"><option>DAY</option><option value="GOOD_TILL_CANCEL">GTC</option></select></label>
    <label>Account <select id="t-account"></select></label>
    <div id="preview-box" hidden><pre id="preview-text"></pre></div>
    <p id="t-error" class="error" hidden></p>
    <div class="buttons">
      <button type="button" id="t-preview">Preview</button>
      <button type="button" id="t-confirm" class="danger" disabled>Confirm &amp; send to Schwab</button>
      <button type="button" id="t-cancel">Cancel</button>
    </div>
  </form>`;
if (!$("ticket")) {
  const dialog = document.createElement("dialog");
  dialog.id = "ticket";
  dialog.innerHTML = TICKET_HTML;
  document.body.appendChild(dialog);
} else if (!$("t-error")) {
  const p = Object.assign(document.createElement("p"), { id: "t-error", className: "error", hidden: true });
  $("preview-box").after(p);
}

async function call(url, body) {
  const response = await fetch(url, body === undefined ? { cache: "no-store" } : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function showError(message) {
  for (const id of ["live-error", "t-error"]) {
    if (!$(id)) continue;
    $(id).textContent = message;
    $(id).hidden = !message;
  }
}

async function refreshStatus() {
  if (!$("live-state")) return;
  try {
    const status = await call("/api/live/status");
    $("live-state").textContent = status.liveTradingEnabled ? "LIVE TRADING ON" : "Live trading OFF";
    $("live-state").className = status.liveTradingEnabled ? "neg" : "pos";
    $("live-env").textContent = status.envAllowed ? "allowed by LIVE_TRADING_ALLOWED=1" : "blocked (start the dashboard with LIVE_TRADING_ALLOWED=1)";
    $("live-toggle").textContent = status.liveTradingEnabled ? "Turn live trading OFF" : "Turn live trading ON";
    $("live-toggle").dataset.enabled = String(status.liveTradingEnabled);
    $("live-cap").textContent = money(status.maxOrderUsd);
  } catch (error) {
    $("live-state").textContent = `Unavailable: ${error.message}`;
  }
}

async function loadAccounts() {
  try {
    const { accounts } = await call("/api/live/accounts");
    $("t-account").replaceChildren(...accounts.map((a) => Object.assign(document.createElement("option"), { value: a.hash, textContent: a.label })));
  } catch (error) {
    showError(`Accounts unavailable: ${error.message}`);
  }
}

async function loadJournal() {
  if (!$("live-journal")) return;
  try {
    const { journal } = await call("/api/live/orders");
    const body = $("live-journal");
    body.replaceChildren();
    for (const e of journal.slice(0, 25)) {
      const row = body.insertRow();
      for (const value of [e.at, e.event, e.symbol ?? "", e.side ?? "", e.quantity ?? "", e.orderId ?? "", e.status ?? e.message ?? ""]) {
        row.insertCell().textContent = String(value);
      }
    }
  } catch { /* journal is informational */ }
}

function openTicket(symbol, side) {
  showError("");
  $("t-symbol").value = symbol || "";
  $("t-side").value = side || "BUY";
  $("preview-box").hidden = true;
  $("t-confirm").disabled = true;
  lastPreview = null;
  if (!$("t-account").options.length) loadAccounts();
  $("ticket").showModal();
}

function ticketValues() {
  const amountMode = $("t-mode").value;
  return {
    symbol: $("t-symbol").value, side: $("t-side").value, orderType: $("t-type").value,
    limitPrice: $("t-type").value === "LIMIT" ? $("t-limit").value : undefined, duration: $("t-tif").value,
    accountHash: $("t-account").value,
    quantity: amountMode === "shares" ? $("t-amount").value : undefined,
    dollars: amountMode === "dollars" ? $("t-amount").value : undefined,
  };
}

async function preview() {
  showError("");
  $("t-confirm").disabled = true;
  try {
    lastPreview = await call("/api/live/preview", ticketValues());
    const p = lastPreview;
    $("preview-text").textContent = [
      `${p.ticket.side} ${p.ticket.quantity} ${p.ticket.symbol} ${p.ticket.orderType}${p.ticket.limitPrice ? ` @ ${money(p.ticket.limitPrice)}` : ""} ${p.ticket.duration}, account ${p.ticket.account}`,
      `Quote: last ${money(p.quote.last)}  bid ${money(p.quote.bid)}  ask ${money(p.quote.ask)}`,
      `Estimated value ${money(p.estimatedNotional)}; commission $0.00; regulatory fees ${money(p.estimatedFees.regulatory)}`,
      `Schwab preview: ${p.schwabPreview?.unavailable ? `unavailable (${p.schwabPreview.unavailable})` : "accepted"}`,
      p.liveTradingEnabled ? "Confirm within 2 minutes to send this order to Schwab." : "Live trading is OFF: this order cannot be sent.",
    ].join("\n");
    $("preview-box").hidden = false;
    $("t-confirm").disabled = !p.liveTradingEnabled;
    setTimeout(() => { if (lastPreview === p) { $("t-confirm").disabled = true; $("preview-text").textContent += "\nPreview expired."; } }, 115_000);
  } catch (error) {
    showError(error.message);
  }
}

async function confirmOrder() {
  if (!lastPreview || submitting) return;
  submitting = true;
  $("t-confirm").disabled = true;
  const p = lastPreview;
  lastPreview = null;
  try {
    const result = await call("/api/live/confirm", { previewId: p.previewId, idempotencyKey: crypto.randomUUID() });
    $("preview-text").textContent += `\nSubmitted to Schwab. Order id ${result.orderId ?? "(pending)"}.`;
    setTimeout(async () => {
      try {
        const status = await call("/api/live/order-status", { previewId: p.previewId });
        $("preview-text").textContent += `\nStatus: ${status.status}, filled ${status.filledQuantity ?? 0}.`;
      } catch (error) { showError(error.message); }
      loadJournal();
    }, 3000);
  } catch (error) {
    showError(error.message);
  } finally {
    submitting = false;
    loadJournal();
  }
}

function watchTables() {
  for (const id of ["positions", "targets", "orders", "dt-quotes", "dt-trades"]) {
    const table = $(id);
    if (!table) continue;
    const symbolIndex = id === "orders" || id === "dt-trades" ? 1 : 0;
    const observer = new MutationObserver(() => {
      for (const row of table.tBodies[0]?.rows ?? []) {
        if (row.dataset.trade || row.cells.length < 2) continue;
        const symbol = row.cells[symbolIndex].textContent.trim();
        if (!/^[A-Z][A-Z0-9.\-]{0,9}$/.test(symbol)) continue;
        row.dataset.trade = "1";
        const cell = row.insertCell();
        const chart = Object.assign(document.createElement("a"), { href: `/ticker.html?symbol=${encodeURIComponent(symbol)}`,
          textContent: "Chart", className: "trade chart" });
        cell.appendChild(chart);
        for (const side of ["BUY", "SELL"]) {
          const button = Object.assign(document.createElement("button"), { type: "button", textContent: side === "BUY" ? "Buy" : "Sell",
            className: side === "BUY" ? "trade buy" : "trade sell" });
          button.addEventListener("click", () => openTicket(symbol, side));
          cell.appendChild(button);
        }
      }
    });
    observer.observe(table, { childList: true, subtree: true });
  }
}

$("live-toggle")?.addEventListener("click", async () => {
  showError("");
  const turnOn = $("live-toggle").dataset.enabled !== "true";
  try {
    const confirmText = turnOn ? prompt("Type ENABLE LIVE TRADING to allow real Schwab orders from this dashboard:") : undefined;
    if (turnOn && confirmText === null) return;
    await call("/api/live/toggle", { enabled: turnOn, confirmText });
  } catch (error) {
    showError(error.message);
  }
  refreshStatus();
});
$("live-new")?.addEventListener("click", () => openTicket("", "BUY"));
$("t-type").addEventListener("change", () => { $("t-limit-row").hidden = $("t-type").value !== "LIMIT"; });
$("t-preview").addEventListener("click", preview);
$("t-confirm").addEventListener("click", confirmOrder);
$("t-cancel").addEventListener("click", () => { lastPreview = null; $("ticket").close(); });
watchTables();
refreshStatus();
loadJournal();
setInterval(refreshStatus, 30_000);
window.LiveTrading = { openTicket, refreshStatus };
window.dispatchEvent(new Event("live-trading-ready"));
