import { describeAdxCondition } from "./adx-column-filters.js";

export function createAdxForwardReturns(api, readRequest, number, percent, renderTable, columnLabels) {
  const node = id => document.getElementById(id);
  const run = node("adx-run-forward");
  const status = node("adx-forward-status");
  const panel = node("adx-forward-results");
  const stale = node("adx-forward-stale");
  const detail = node("adx-forward-detail");
  const dialog = node("adx-condition-dialog");
  const conditionStatus = node("adx-condition-status");
  const conditionTable = node("adx-condition-table");
  let running = false;
  let lastRequest;
  let tickerReturns;
  const breakdowns = new Map();
  const formatReturn = value => value == null ? "—" : percent.format(value);

  function sync() {
    const current = readRequest();
    run.disabled = running || !current;
    const isCurrent = lastRequest === JSON.stringify(current);
    stale.hidden = !lastRequest || isCurrent;
    return isCurrent ? tickerReturns : null;
  }

  function show(result) {
    node("adx-forward-average").textContent = formatReturn(result.mean);
    node("adx-forward-median").textContent = formatReturn(result.median);
    node("adx-forward-positive").textContent = formatReturn(result.positiveRate);
    node("adx-forward-count").textContent = number.format(result.observations);
    node("adx-forward-coverage").textContent = [
      `${result.firstDate} through ${result.through}`,
      `${number.format(tickerReturns.size)} filtered stocks · ${number.format(result.tickers)} with returns`,
      result.membershipAsOf && `S&P 500 list dated ${result.membershipAsOf}`,
    ].filter(Boolean).join(" · ");
    node("adx-forward-skipped").textContent = `${number.format(result.pending)} without five later sessions by the cutoff · ${number.format(result.missing)} with missing prices`;
    node("adx-forward-rows").replaceChildren(...result.recent.map(match => {
      const row = document.createElement("tr");
      for (const value of [match.ticker, match.date, match.exitDate, formatReturn(match.return)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      return row;
    }));
    detail.hidden = !result.recent.length;
    panel.hidden = false;
  }

  function showBreakdown(ticker, result) {
    const rows = [
      { label: "All active filters", ...result.tickerReturns[ticker] },
      ...result.byCondition.map(item => ({
        ...item,
        label: item.conditions.map(([field, rule]) => describeAdxCondition(field, columnLabels.get(field), rule)).join(" and "),
      })),
    ];
    node("adx-condition-rows").replaceChildren(...rows.map(item => {
      const row = document.createElement("tr");
      const label = document.createElement("th");
      label.scope = "row";
      label.textContent = item.label;
      row.append(label);
      for (const value of [formatReturn(item.mean), number.format(item.observations)]) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      return row;
    }));
    conditionStatus.textContent = "";
    conditionTable.hidden = false;
  }

  async function openBreakdown(ticker) {
    const request = readRequest();
    if (running || lastRequest !== JSON.stringify(request) || !tickerReturns?.has(ticker)) return;
    node("adx-condition-title").textContent = `${ticker} 5-day returns by condition`;
    conditionTable.hidden = true;
    conditionStatus.textContent = "";
    dialog.showModal();
    if (breakdowns.has(ticker)) {
      showBreakdown(ticker, breakdowns.get(ticker));
      return;
    }
    running = true;
    sync();
    conditionStatus.textContent = "Checking historical matches…";
    try {
      const result = await api("/api/adx-summary/forward-returns", {
        through: request.through,
        filters: { ...request.filters, breakdown: true },
        tickers: [ticker],
      });
      breakdowns.set(ticker, result);
      if (dialog.open) showBreakdown(ticker, result);
    } catch (error) {
      conditionStatus.textContent = error.message;
    } finally {
      running = false;
      sync();
    }
  }

  node("adx-condition-close").addEventListener("click", () => dialog.close());
  run.addEventListener("click", async () => {
    const request = readRequest();
    if (running || !request) return;
    running = true;
    status.textContent = "Checking historical matches…";
    sync();
    try {
      const result = await api("/api/adx-summary/forward-returns", request);
      tickerReturns = new Map(Object.entries(result.tickerReturns));
      lastRequest = JSON.stringify(request);
      breakdowns.clear();
      show(result);
      status.textContent = "";
    } catch (error) {
      status.textContent = error.status === 404 ? "Restart the Prices server to run this analysis." : error.message;
    } finally {
      running = false;
      renderTable();
    }
  });
  return { sync, openBreakdown };
}
