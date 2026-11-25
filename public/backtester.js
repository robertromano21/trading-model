import { listBacktestRuns, readBacktestRun, saveBacktestRun, setBacktestPin } from "./db.js";

const node = (id) => document.getElementById(id);
const number = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });
const currency = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD" });
const fillPrice = new Intl.NumberFormat(undefined, { style: "currency", currency: "USD", maximumFractionDigits: 6 });
const ratio = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });
const PAGE_SIZE = 25;
const COLORS = ["#80e7bd", "#e8b675", "#95b9ff", "#e8a1cd", "#aacd75", "#bdadfc", "#f99780"];
const METRICS = [
  ["finalEquity", "Final equity", currency], ["totalReturnPct", "Total return", percent],
  ["benchmarkReturnPct", "Buy & hold return", percent], ["cagrPct", "Annualized return", percent],
  ["maxDrawdownPct", "Max drawdown", percent], ["sharpeRatio", "Sharpe", ratio],
  ["winRatePct", "Win rate", percent], ["totalTrades", "Closed trades", number],
  ["totalFeesPaid", "Fees paid", currency], ["dividendIncome", "Dividend income", currency],
];
let chartPromise;

function chartLibrary() {
  if (!chartPromise) {
    chartPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "/assets/chart.umd.js";
      script.onload = () => window.Chart ? resolve(window.Chart) : reject(new Error("Chart.js is unavailable."));
      script.onerror = () => reject(new Error("The chart library could not be loaded."));
      document.head.append(script);
    }).catch((error) => {
      chartPromise = undefined;
      throw error;
    });
  }
  return chartPromise;
}

function tableCell(text, tag = "td") {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}

function display(value, formatter) {
  if (!Number.isFinite(value)) return "—";
  return formatter.format(formatter === percent ? value / 100 : value);
}

function inputFields(container, definitions, prefix) {
  container.replaceChildren(...Object.entries(definitions).map(([key, field]) => {
    const group = document.createElement("div");
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.id = `bt-${prefix}-${key}`;
    input.name = key;
    input.type = "number";
    input.value = field.default;
    input.min = field.min;
    input.max = field.max;
    input.step = field.type === "integer" ? "1" : "any";
    input.inputMode = field.type === "integer" ? "numeric" : "decimal";
    input.required = true;
    label.htmlFor = input.id;
    label.textContent = field.label;
    group.append(label, input);
    return group;
  }));
}

function fieldValues(container) {
  return Object.fromEntries([...container.querySelectorAll("input:not(:disabled)")]
    .map((input) => [input.name, input.valueAsNumber]));
}

function runName(run) {
  return run.label || run.strategyName;
}

function universeName(run) {
  if (!run.tickers.length) return `All saved tickers (${number.format(run.symbolCount)})`;
  return run.tickers.length <= 4 ? run.tickers.join(", ") :
    `${run.tickers.slice(0, 3).join(", ")} +${run.tickers.length - 3}`;
}

function dataset(label, values, color, dashed = false) {
  return { label, data: values, borderColor: color, backgroundColor: color, borderWidth: 2, pointRadius: 0,
    pointHoverRadius: 3, spanGaps: true, borderDash: dashed ? [5, 4] : [] };
}

function chartOptions(formatValue) {
  const axes = { ticks: { color: "#a9bac6", maxTicksLimit: 7 }, grid: { color: "#2b3b47" } };
  return {
    responsive: true, maintainAspectRatio: false, animation: false,
    interaction: { mode: "index", intersect: false },
    scales: { x: { ...axes, ticks: { ...axes.ticks, maxRotation: 0 } },
      y: { ...axes, ticks: { ...axes.ticks, callback: formatValue } } },
    plugins: { legend: { labels: { color: "#edf4f7", boxWidth: 14 } },
      tooltip: { callbacks: { label: (item) => `${item.dataset.label} · ${formatValue(item.parsed.y)}` } } },
  };
}

export function createBacktester(api) {
  const form = node("bt-form");
  const mode = node("bt-mode");
  const all = node("bt-all");
  const search = node("bt-search");
  const picker = node("bt-ticker");
  const starting = node("bt-start");
  const ending = node("bt-end");
  const strategySelect = node("bt-strategy");
  const parameterFields = node("bt-parameters");
  const costFields = node("bt-costs");
  const dividends = node("bt-dividends");
  const status = node("bt-status");
  const selected = new Set();
  const chartVersions = new Map();
  const charts = new Map();
  let registry;
  let universe;
  let coverage = new Map();
  let pick = "NVDA";
  let busy = false;
  let syncing = false;
  let currentRun;
  let tradePage = 0;
  let saved = [];
  let comparisonVersion = 0;
  let opening;
  let savedLoaded = false;
  let coverageError = "";
  let strategyError = "";

  function notify(message) {
    status.textContent = message;
  }

  function errorMessage(error) {
    notify(error.message);
    console.error("Backtester", error.message);
  }

  function chosenTickers() {
    if (mode.value === "single") return picker.value ? [picker.value] : [];
    return all.checked ? [] : [...selected];
  }

  function renderPicker() {
    const query = search.value.trim().toUpperCase();
    const rows = universe?.tickers ?? [];
    const matches = rows.filter(({ ticker }) => ticker.includes(query));
    const options = matches.slice(0, 50);
    if (query) {
      const exact = coverage.get(query);
      if (exact && !options.includes(exact)) options.unshift(exact);
    } else if (coverage.has(pick) && !options.includes(coverage.get(pick))) {
      options.unshift(coverage.get(pick));
    }
    picker.replaceChildren(...(options.length ? options.map(({ ticker }) => new Option(ticker, ticker)) :
      [new Option(universe ? "No matching tickers" : coverageError ? "Tickers unavailable" : "Loading tickers…", "")]));
    picker.value = options.some(({ ticker }) => ticker === pick) ? pick : options[0]?.ticker ?? "";
    pick = picker.value;
    refreshForm();
  }

  function refreshForm() {
    const portfolio = mode.value === "portfolio";
    const wholeMarket = portfolio && all.checked;
    node("bt-all-field").hidden = !portfolio;
    node("bt-picker").hidden = wholeMarket;
    picker.disabled = !universe;
    node("bt-add").hidden = !portfolio;
    node("bt-add").disabled = !picker.value || selected.has(picker.value);
    node("bt-selected").hidden = !portfolio || wholeMarket || !selected.size;
    const maxPositions = node("bt-cost-maxPositions");
    if (maxPositions) {
      maxPositions.disabled = !portfolio;
      maxPositions.parentElement.hidden = !portfolio;
    }
    const chips = [...selected].map((ticker) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.textContent = `${ticker} ×`;
      chip.setAttribute("aria-label", `Remove ${ticker}`);
      chip.addEventListener("click", () => { selected.delete(ticker); refreshForm(); });
      return chip;
    });
    node("bt-selected").replaceChildren(...chips);
    const chosen = chosenTickers().map((ticker) => coverage.get(ticker)).filter(Boolean);
    const savedPrices = universe?.prices;
    let description = coverageError || (universe ? "Select a saved ticker." : "Reading SQLite coverage…");
    if (wholeMarket) {
      description = `${number.format(universe?.tickers.length ?? 0)} saved tickers · ${number.format(savedPrices?.sessions ?? 0)} market sessions`;
    } else if (chosen.length === 1) {
      const item = chosen[0];
      description = `${item.ticker} · ${number.format(item.bars)} sessions · ${item.firstDate} to ${item.lastDate}`;
      if (item.missingMarketSessions) description += ` · ${number.format(item.missingMarketSessions)} gaps`;
    } else if (chosen.length) {
      const fewest = Math.min(...chosen.map((item) => item.bars));
      const most = Math.max(...chosen.map((item) => item.bars));
      const gaps = chosen.filter((item) => item.missingMarketSessions > 0).length;
      description = `${number.format(chosen.length)} tickers · ${number.format(fewest)}–${number.format(most)} sessions per ticker`;
      if (gaps) description += ` · ${number.format(gaps)} with saved-session gaps`;
    }
    node("bt-coverage").textContent = description;

    const cached = universe?.dividends;
    const priceFirst = portfolio ? savedPrices?.firstDate : chosen[0]?.firstDate;
    // The engine uses saved pre-start history for the strategy warm-up.
    const priceLast = [ending.value, portfolio ? savedPrices?.lastDate : chosen[0]?.lastDate].filter(Boolean).sort()[0];
    const dividendsReady = cached && priceFirst && priceLast && cached.firstDate <= priceFirst && cached.lastDate >= priceLast;
    dividends.disabled = !dividendsReady;
    const selectedDistributions = (wholeMarket ? universe?.tickers ?? [] : chosen)
      .reduce((total, entry) => total + entry.unsupportedDividends, 0);
    node("bt-dividend-status").textContent = !universe ? "" : !cached ? "Price returns · dividend history unavailable."
      : dividendsReady ? `Dividend coverage · ${cached.firstDate} to ${cached.lastDate}`
          + (selectedDistributions ? ` · ${number.format(selectedDistributions)} records lack a usable USD amount` : "")
      : `Dividend cache ${cached.firstDate} to ${cached.lastDate} · selected prices or warm-up extend beyond coverage.`;
    node("bt-sync").hidden = !universe?.dividendSyncAvailable;
    node("bt-sync").disabled = syncing || busy;
    node("bt-sync").textContent = syncing ? "Syncing dividends…" : "Sync dividends";
    node("bt-submit").disabled = !registry || !universe || busy || syncing ||
      (!chosen.length && !wholeMarket) || !starting.value || !ending.value;
    node("bt-submit").textContent = busy ? "Running…" : "Run backtest";
    strategySelect.disabled = !registry;
    node("bt-strategy-status").textContent = strategyError;
    node("bt-retry").hidden = !strategyError && !coverageError;
  }

  async function refreshUniverse() {
    universe = await api("/api/universe");
    coverage = new Map(universe.tickers.map((entry) => [entry.ticker, entry]));
    if (!starting.value) starting.value = universe.prices.firstDate ?? "";
    if (!ending.value) ending.value = universe.prices.lastDate ?? "";
    renderPicker();
  }

  function renderParameters() {
    const choice = registry?.strategies.find(({ id }) => id === strategySelect.value);
    if (choice) inputFields(parameterFields, choice.parameters, "param");
  }

  async function renderChart(id, labels, datasets, formatter) {
    const version = (chartVersions.get(id) || 0) + 1;
    chartVersions.set(id, version);
    try {
      const Chart = await chartLibrary();
      if (version !== chartVersions.get(id)) return;
      charts.get(id)?.destroy();
      charts.set(id, new Chart(node(id), { type: "line", data: { labels, datasets }, options: chartOptions(formatter) }));
    } catch (error) {
      node("bt-chart-status").hidden = false;
      node("bt-chart-status").textContent = error.message;
      console.error("Backtest chart", error.message);
    }
  }

  function renderTrades() {
    const trades = currentRun.trades;
    const start = tradePage * PAGE_SIZE;
    const page = trades.slice(start, start + PAGE_SIZE);
    node("bt-trades").replaceChildren(...page.map((trade) => {
      const row = document.createElement("tr");
      row.append(...[trade.ticker, trade.entryDate, fillPrice.format(trade.entryPrice), trade.exitDate,
        fillPrice.format(trade.exitPrice), currency.format(trade.pnl), display(trade.pnlPct, percent), trade.reason].map(
        (value) => tableCell(value),
      ));
      return row;
    }));
    node("bt-trade-summary").textContent = `${number.format(trades.length)} closed · ${number.format(currentRun.metrics.openPositions)} open at the end`;
    node("bt-trade-prev").disabled = tradePage <= 0;
    node("bt-trade-next").disabled = start + PAGE_SIZE >= trades.length;
    node("bt-trade-page").textContent = trades.length ? `${start + 1}–${start + page.length} of ${number.format(trades.length)}` : "";
  }

  function renderRun(run) {
    currentRun = run;
    tradePage = 0;
    const basis = run.includeDividends ? "Cash dividends included" : "Split-adjusted price return";
    node("bt-result-summary").textContent = `${runName(run)} · ${universeName(run)} · ${run.firstDate} to ${run.lastDate} · ${basis}`;
    node("bt-result").hidden = false;
    node("bt-chart-status").hidden = true;
    node("bt-metrics").replaceChildren(...METRICS.map(([key, label, formatter]) => {
      const group = document.createElement("div");
      const term = document.createElement("dt");
      const value = document.createElement("dd");
      term.textContent = label;
      value.textContent = display(run.metrics[key], formatter);
      group.append(term, value);
      return group;
    }));
    const notices = [...run.warnings];
    if (run.notes) notices.push(run.notes);
    node("bt-warnings").replaceChildren(...notices.map((notice) => tableCell(notice, "li")));
    node("bt-warnings").hidden = !notices.length;
    renderTrades();
    const curves = [dataset("Strategy equity", run.equityCurve.map(({ equity }) => equity), COLORS[0])];
    if (run.equityCurve.some(({ benchmark }) => benchmark !== null)) {
      curves.push(dataset(`${run.benchmark} buy & hold`, run.equityCurve.map(({ benchmark }) => benchmark), COLORS[1], true));
    }
    void renderChart("bt-equity", run.equityCurve.map(({ date }) => date), curves, (value) => currency.format(value));
  }

  async function renderComparison() {
    const version = ++comparisonVersion;
    const pins = saved.filter(({ pinned }) => pinned);
    node("bt-comparison").hidden = !pins.length;
    if (!pins.length) return;
    const runs = (await Promise.all(pins.map(({ runId }) => readBacktestRun(runId)))).filter(Boolean);
    if (version !== comparisonVersion) return;
    const labels = [...new Set(runs.flatMap(({ equityCurve }) => equityCurve.map(({ date }) => date)))].sort();
    const lines = runs.map((run, i) => {
      const byDate = new Map(run.equityCurve.map(({ date, equity }) => [date, equity / run.metrics.initialCapital - 1]));
      return dataset(runName(run), labels.map((date) => byDate.get(date) ?? null), COLORS[i % COLORS.length]);
    });
    void renderChart("bt-overlay", labels, lines, (value) => percent.format(value));

    const table = node("bt-compare-table");
    table.replaceChildren();
    const head = table.createTHead().insertRow();
    head.append(tableCell("Metric", "th"), ...runs.map((run) => tableCell(runName(run), "th")));
    const body = table.createTBody();
    for (const [label, valueFor] of [
      ["Strategy", (run) => run.strategyName],
      ["Universe", universeName],
      ["Return basis", (run) => run.includeDividends ? "Cash dividends included" : "Price only"],
      ["Execution costs", (run) => `${run.costs.commissionPct}% + ${currency.format(run.costs.fixedFee)} + ${run.costs.slippageBps} bps`],
      ...METRICS.map(([key, label, formatter]) => [label, (run) => display(run.metrics[key], formatter)]),
    ]) {
      const row = body.insertRow();
      const labelCell = tableCell(label, "th");
      labelCell.scope = "row";
      row.append(labelCell, ...runs.map((run) => tableCell(valueFor(run))));
    }
  }

  async function renderSaved() {
    saved = await listBacktestRuns();
    node("bt-saved-empty").hidden = Boolean(saved.length);
    node("bt-saved").replaceChildren(...saved.map((run) => {
      const row = document.createElement("tr");
      const compare = document.createElement("input");
      compare.type = "checkbox";
      compare.checked = Boolean(run.pinned);
      compare.setAttribute("aria-label", `Compare ${runName(run)}`);
      compare.addEventListener("change", async () => {
        try {
          run.pinned = compare.checked;
          await setBacktestPin(run.runId, run.pinned);
          console.log("Backtest comparison selection", run.runId, run.pinned);
          await renderComparison();
        } catch (error) {
          errorMessage(error);
        }
      });
      const open = document.createElement("button");
      open.type = "button";
      open.textContent = "Open";
      open.setAttribute("aria-label", `Open ${runName(run)}`);
      open.addEventListener("click", async () => {
        try {
          const report = await readBacktestRun(run.runId);
          if (!report) throw new Error("That saved backtest is unavailable.");
          renderRun(report);
          node("bt-result-title").scrollIntoView();
          console.log("Opened saved backtest", report.runId);
        } catch (error) {
          errorMessage(error);
        }
      });
      const pinCell = tableCell("");
      pinCell.append(compare);
      const openCell = tableCell("");
      openCell.append(open);
      const name = tableCell(runName(run));
      name.title = run.notes || run.strategyName;
      row.append(pinCell, name, tableCell(universeName(run)), tableCell(`${run.firstDate} – ${run.lastDate}`),
        tableCell(display(run.metrics.totalReturnPct, percent)), openCell);
      return row;
    }));
    await renderComparison();
  }

  search.addEventListener("input", renderPicker);
  picker.addEventListener("change", () => { pick = picker.value; refreshForm(); });
  mode.addEventListener("change", refreshForm);
  all.addEventListener("change", refreshForm);
  starting.addEventListener("change", refreshForm);
  ending.addEventListener("change", refreshForm);
  strategySelect.addEventListener("change", renderParameters);
  node("bt-add").addEventListener("click", () => { if (picker.value) selected.add(picker.value); refreshForm(); });
  node("bt-trade-prev").addEventListener("click", () => { tradePage--; renderTrades(); });
  node("bt-trade-next").addEventListener("click", () => { tradePage++; renderTrades(); });
  node("bt-sync").addEventListener("click", async () => {
    syncing = true;
    refreshForm();
    notify("Syncing dividend history…");
    try {
      const result = await api("/api/backtest/dividends", {});
      await refreshUniverse();
      dividends.checked = !dividends.disabled;
      notify(`Saved ${number.format(result.events)} distribution records.`);
      console.log("Synced backtest dividends", result.firstDate, result.lastDate);
    } catch (error) {
      errorMessage(error);
    } finally {
      syncing = false;
      refreshForm();
    }
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const options = {
      mode: mode.value, tickers: chosenTickers(), strategy: strategySelect.value,
      startDate: starting.value, endDate: ending.value, includeDividends: !dividends.disabled && dividends.checked,
      params: fieldValues(parameterFields), costs: fieldValues(costFields),
    };
    const label = node("bt-label").value.trim();
    const notes = node("bt-notes").value.trim();
    busy = true;
    refreshForm();
    notify("Running backtest. Full-market runs may take a few minutes.");
    console.log("Backtest started", options.strategy, options.mode, options.tickers.length);
    try {
      const run = { ...await api("/api/backtest", options), label, notes };
      renderRun(run);
      try {
        await saveBacktestRun(run);
        await renderSaved();
        notify("Backtest complete and saved in this browser.");
        console.log("Backtest saved", run.runId, run.metrics.totalTrades);
      } catch (error) {
        notify(`Backtest complete. Browser storage error · ${error.message}`);
        console.error("Could not save backtest", error.message);
      }
    } catch (error) {
      errorMessage(error);
    } finally {
      busy = false;
      refreshForm();
    }
  });

  function setupMessage(error, subject) {
    return error.status === 404
      ? "The running dashboard needs a restart to load Backtester. Restart it in PowerShell, then retry here."
      : `Could not load ${subject}. ${error.message}`;
  }

  async function loadRegistry() {
    const data = await api("/api/strategies");
    strategySelect.replaceChildren(...data.strategies.map((item) => new Option(item.name, item.id)));
    inputFields(costFields, data.settings, "cost");
    registry = data;
    renderParameters();
    refreshForm();
  }

  async function open() {
    if (opening) return opening;
    opening = (async () => {
      const firstCoverage = !universe;
      strategyError = coverageError = "";
      refreshForm();
      const [catalog, prices, storage] = await Promise.allSettled([
        registry ? undefined : loadRegistry(), refreshUniverse(), savedLoaded ? undefined : renderSaved(),
      ]);
      if (catalog.status === "rejected") {
        strategyError = setupMessage(catalog.reason, "strategies");
        strategySelect.replaceChildren(new Option("Strategies unavailable", ""));
        console.error("Backtester strategy setup", catalog.reason.message);
      }
      if (prices.status === "rejected") {
        coverageError = setupMessage(prices.reason, "ticker coverage");
        console.error("Backtester ticker setup", prices.reason.message);
      }
      if (storage.status === "fulfilled") savedLoaded = true;
      else console.error("Backtester saved runs", storage.reason.message);
      renderPicker();
      if (prices.status === "fulfilled" && firstCoverage) dividends.checked = !dividends.disabled;
      if (!busy && !syncing) {
        notify(strategyError || coverageError || (storage.status === "rejected"
          ? "Ready. Saved runs could not be loaded from this browser." : "Ready"));
      }
    })();
    try {
      await opening;
    } finally {
      opening = undefined;
    }
  }

  node("bt-retry").addEventListener("click", open);
  return open;
}
