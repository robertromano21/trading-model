import { listPaperTrades, savePaperTrades, updatePaperTradeMarks } from "./db.js";

const amount = 10;
const pageSize = 50;
const markBatchSize = 4000;
const number = new Intl.NumberFormat();
const currency = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" });
const price = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 6 });
const shares = new Intl.NumberFormat(undefined, { maximumFractionDigits: 8 });
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });
const tradeKey = (ticker, entryDate) => JSON.stringify([ticker, entryDate]);
const statuses = { open: "Open", closed: "Closed", "awaiting-exit": "Awaiting exit price" };

export function createPapertrade(api) {
  const node = id => document.getElementById(id);
  const status = node("papertrade-status");
  const adxStatus = node("adx-papertrade-status");
  const refreshButton = node("papertrade-refresh");
  const tradesTable = node("papertrade-table");
  const empty = node("papertrade-empty");
  const pagination = node("papertrade-pagination");
  const previous = node("papertrade-previous");
  const next = node("papertrade-next");
  const pending = new Set();
  let trades;
  let refreshing;
  let onChange;
  let page = 0;

  function render() {
    const positions = [...trades.values()].sort((a, b) =>
      b.entryDate.localeCompare(a.entryDate) || a.ticker.localeCompare(b.ticker));
    page = Math.min(page, Math.max(0, Math.ceil(positions.length / pageSize) - 1));
    const start = page * pageSize;
    const visible = positions.slice(start, start + pageSize);
    node("papertrade-rows").replaceChildren(...visible.map(trade => {
      const { ticker, entryDate, entryPrice, mark } = trade;
      const quantity = trade.amount / entryPrice;
      const quote = mark.price ?? entryPrice;
      const change = quote / entryPrice - 1;
      const row = document.createElement("tr");
      for (const value of [
        ticker, statuses[mark.status], entryDate, price.format(entryPrice), currency.format(trade.amount),
        shares.format(quantity), `${mark.heldSessions}/${mark.holdSessions}`, mark.exitDate,
        mark.priceDate ?? entryDate, price.format(quote), currency.format(trade.amount * change),
        percent.format(change),
      ]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      row.firstChild.className = "ticker";
      return row;
    }));
    tradesTable.hidden = !positions.length;
    empty.hidden = !!positions.length;
    pagination.hidden = positions.length <= pageSize;
    previous.disabled = page === 0;
    next.disabled = start + visible.length >= positions.length;
    node("papertrade-page").textContent = `${number.format(start + 1)}–${number.format(start + visible.length)} of ${number.format(positions.length)} positions`;
  }

  async function loadTrades() {
    const saved = await listPaperTrades();
    trades = new Map(saved.map(trade => [tradeKey(trade.ticker, trade.entryDate), trade]));
    render();
    onChange?.();
  }

  async function readMarks(requested) {
    const marks = [];
    for (let start = 0; start < requested.length; start += markBatchSize) {
      const result = await api("/api/papertrade/marks", { trades: requested.slice(start, start + markBatchSize) });
      marks.push(...result.marks);
    }
    return marks;
  }

  function refresh() {
    if (refreshing) return refreshing;
    refreshButton.disabled = true;
    status.textContent = "Checking saved prices…";
    refreshing = (async () => {
      try {
        await loadTrades();
        const openTrades = [...trades.values()].filter(trade => trade.mark.status !== "closed")
          .map(({ ticker, entryDate }) => ({ ticker, entryDate }));
        if (openTrades.length) {
          await updatePaperTradeMarks(await readMarks(openTrades));
          await loadTrades();
        }
        status.textContent = "";
      } catch (error) {
        status.textContent = error.message;
      } finally {
        refreshButton.disabled = false;
        refreshing = undefined;
      }
    })();
    return refreshing;
  }

  function selected(ticker, entryDate) {
    const key = tradeKey(ticker, entryDate);
    return Boolean(trades?.has(key) || pending.has(key));
  }

  async function track(tickers, entryDate) {
    if (!trades || !entryDate) return;
    const requested = tickers.filter(ticker => !selected(ticker, entryDate)).map(ticker => ({ ticker, entryDate }));
    if (!requested.length) return;
    const keys = requested.map(({ ticker }) => tradeKey(ticker, entryDate));
    for (const key of keys) pending.add(key);
    adxStatus.textContent = requested.length === 1 ? `Saving ${requested[0].ticker} paper trade…`
      : `Saving ${number.format(requested.length)} paper trades…`;
    onChange?.();
    try {
      const marks = await readMarks(requested);
      const positions = marks.filter(mark => mark.entryPrice > 0)
        .map(mark => ({ ticker: mark.ticker, entryDate, amount, entryPrice: mark.entryPrice, mark }));
      const added = await savePaperTrades(positions);
      await loadTrades();
      const missing = requested.length - positions.length;
      adxStatus.textContent = requested.length === 1 && !missing
        ? `Tracking ${requested[0].ticker} for ${entryDate} in Papertrade.`
        : `Added ${number.format(added)} paper trades for ${entryDate}.` +
          (missing ? ` ${number.format(missing)} skipped due to missing entry prices.` : "");
    } catch (error) {
      adxStatus.textContent = error.message;
    } finally {
      for (const key of keys) pending.delete(key);
      onChange?.();
    }
  }

  refreshButton.addEventListener("click", refresh);
  previous.addEventListener("click", () => { page--; render(); });
  next.addEventListener("click", () => { page++; render(); });
  void loadTrades().catch(error => { adxStatus.textContent = error.message; });

  return {
    refresh,
    onChange(callback) { onChange = callback; },
    selectAllControl() {
      const label = document.createElement("label");
      label.className = "adx-papertrade-select-all";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.className = "papertrade-checkbox";
      input.setAttribute("aria-label", "Select all filtered paper trades");
      input.disabled = true;
      let tickers = [];
      let entryDate;
      label.append(input, "Select all");
      input.addEventListener("change", () => { if (input.checked) void track(tickers, entryDate); });
      return {
        label,
        update(rows, date) {
          tickers = rows.filter(row => row.close > 0).map(row => row.ticker);
          entryDate = date;
          const remaining = tickers.filter(ticker => !selected(ticker, entryDate)).length;
          input.checked = tickers.length > 0 && remaining === 0;
          input.indeterminate = remaining > 0 && remaining < tickers.length;
          input.disabled = !trades || remaining === 0;
          label.title = `Add $10 paper trades for ${number.format(remaining)} filtered stocks across all pages`;
        },
      };
    },
    checkbox(ticker, entryDate, close) {
      const input = document.createElement("input");
      input.type = "checkbox";
      input.className = "papertrade-checkbox";
      input.setAttribute("aria-label", `Paper trade ${ticker}`);
      input.checked = selected(ticker, entryDate);
      input.disabled = !trades || input.checked || !(close > 0);
      input.title = input.checked ? `Tracking ${ticker} from ${entryDate}`
        : `Track a $10 paper trade from the ${entryDate} close`;
      input.addEventListener("change", () => { if (input.checked) void track([ticker], entryDate); });
      return input;
    },
  };
}
