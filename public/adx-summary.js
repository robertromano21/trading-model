import { createAdxColumnFilters } from "./adx-column-filters.js";
import { createAdxFilterPresets } from "./adx-filter-presets.js";
import { createAdxForwardReturns } from "./adx-forward-returns.js";

const number = new Intl.NumberFormat();
const indicator = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 });
const price = new Intl.NumberFormat(undefined, { maximumFractionDigits: 4, minimumFractionDigits: 2 });
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 2 });
const ordinal = new Intl.PluralRules("en", { type: "ordinal" });
const ordinalSuffix = { one: "st", two: "nd", few: "rd", other: "th" };
const forwardReturnField = "avg5DayReturn";
const papertradeField = "paperTrade";
const columns = [
  ["ticker", "Ticker"], [papertradeField, "Paper trade"], ["sp500", "S&P 500"],
  ["adx", "Current ADX", indicator], ["priorAdx", "ADX 5D Ago", indicator],
  ["adxChange", "ADX Change", indicator], ["adxState", "ADX State"],
  ["ma20ZScore", "20-day MA Z-score", indicator], ["ma20Gap", "20-day MA Position", describePosition],
  ["ma20GapPercentile", "20-day MA Percentile", describePercentile],
  [forwardReturnField, "Avg. 5-day Return", percent],
  ["maPosition", "50-day MA Position", describePosition], ["gapChange", "50-day MA Gap (5D)", describeGap],
  ["close", "Last Close", price],
  ["ma50", "Current 50 MA", price], ["pct50", "% vs. 50 MA", percent], ["priorClose", "Close 5D Ago", price],
  ["priorMa50", "50 MA 5D Ago", price], ["priorPct50", "% vs. 50 MA 5D Ago", percent],
  ["pct50Change", "Change vs. 50 MA (pp)", (value) => indicator.format(value * 100)],
  ["maStatus", "50 MA Status"], ["marketRegime", "Workbook Market Regime"],
  ["highAdx", "High ADX", indicator], ["highDate", "High Date"], ["lowAdx", "Low ADX", indicator],
  ["lowDate", "Low Date"], ["minPct50", "Historical Min % vs. 50 MA", percent],
  ["maxPct50", "Historical Max % vs. 50 MA", percent],
];
const compactFields = new Set([
  "ticker", papertradeField, "sp500", "adx", "adxChange", "adxState", "ma20ZScore", "ma20Gap", "ma20GapPercentile",
  forwardReturnField, "maPosition", "gapChange", "close",
]);
const compactColumns = columns.filter(([field]) => compactFields.has(field));
const pageSize = 50;

function describePosition(distance) {
  if (distance === 0) return "At average";
  return `${distance > 0 ? "Above" : "Below"} by ${percent.format(Math.abs(distance))}`;
}

function describeGap(change) {
  if (Math.abs(change) < 0.005) return "Unchanged";
  return `${change < 0 ? "Narrowing" : "Widening"} ${indicator.format(Math.abs(change))} pp`;
}

function describePercentile(value) {
  const rank = Math.round(value * 100);
  return `${rank}${ordinalSuffix[ordinal.select(rank)]} percentile`;
}

function describeZScore(row) {
  if (row.ma20Gap == null) return "Z-score requires a 20-session moving average.";
  if (row.ma20Gap === 0) return "The close is at the 20-day moving average.";
  const side = row.ma20Gap < 0 ? "below" : "above";
  const details = [`Current gap ${percent.format(row.ma20Gap)}`];
  if (!row.ma20GapCount) {
    details.push(`No prior sessions ${side} the 20-day MA.`);
  } else {
    details.push(
      `Prior sessions ${side} the 20-day MA: ${number.format(row.ma20GapCount)}`,
      `Mean gap ${percent.format(row.ma20GapMean)}`,
      `Standard deviation ${percent.format(row.ma20GapStd)}`,
      `Historical rank ${describePercentile(row.ma20GapPercentile)} (${percent.format(row.ma20GapPercentile)})`,
    );
    if (row.ma20ZScore == null) details.push("Z-score requires variation in the prior gaps.");
  }
  return details.join("\n");
}

function display(value, formatter) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof formatter === "function") return formatter(value);
  return formatter ? formatter.format(value) : value;
}

function indexTicker(ticker) {
  return ticker.trim().toUpperCase().replace(/[ /-]/g, ".");
}

export function createAdxSummary(api, papertrade) {
  const node = (id) => document.getElementById(id);
  const dateSelect = node("adx-date");
  const tickerSearch = node("adx-ticker");
  const stateFilter = node("adx-state");
  const maFilter = node("adx-ma");
  const refresh = node("adx-refresh");
  const columnsToggle = node("adx-columns-toggle");
  const zScoreHelp = node("adx-zscore-help");
  const status = node("adx-status");
  const empty = node("adx-empty");
  const membershipDate = node("adx-sp500-date");
  let report;
  let latest;
  let page = 0;
  let exactTicker = false;
  let showAllColumns = false;
  let sortBy = "ticker";
  let direction = 1;
  let version = 0;
  let loadingDates;
  let filteredTickers = [];
  const columnFilters = createAdxColumnFilters(updateResults);
  const selectAllPapertrades = papertrade.selectAllControl();
  const forwardReturns = createAdxForwardReturns(api, () => report && filteredTickers.length ? {
    through: report.date,
    filters: readFilterSettings(),
    tickers: filteredTickers,
    reportVersion: version,
  } : null, number, percent, render, new Map(columns.map(([field, label]) => [field, label])));

  const headings = columns.map(([field, label]) => {
    const header = document.createElement("th");
    header.scope = "col";
    header.hidden = !compactFields.has(field);
    if (field === papertradeField) {
      header.className = "adx-papertrade-cell";
      header.textContent = label;
      header.append(selectAllPapertrades.label);
      return header;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "adx-sort-button";
    button.textContent = label;
    if (field === "ma20GapPercentile") {
      button.title = "Ranks the signed gap against earlier gaps on the same side of the 20-day average.";
    } else if (field === forwardReturnField) {
      button.title = "Average five-session close-to-close return. Click a match count to see returns by condition.";
    }
    button.addEventListener("click", () => {
      direction = sortBy === field ? -direction : 1;
      sortBy = field;
      updateResults();
    });
    const controls = document.createElement("span");
    controls.className = "adx-heading-controls";
    controls.append(button);
    if (field === "ma20ZScore") {
      const help = document.createElement("button");
      help.type = "button";
      help.className = "adx-help-button";
      help.textContent = "?";
      help.setAttribute("aria-label", "Explain 20-day MA Z-score");
      help.setAttribute("aria-controls", zScoreHelp.id);
      help.setAttribute("aria-expanded", "false");
      help.addEventListener("click", () => {
        zScoreHelp.hidden = !zScoreHelp.hidden;
        help.setAttribute("aria-expanded", String(!zScoreHelp.hidden));
      });
      controls.append(help);
    }
    if (field !== forwardReturnField) controls.append(columnFilters.createButton(field, label));
    header.append(controls);
    return header;
  });
  node("adx-columns").replaceChildren(...headings);

  function readFilterSettings() {
    return {
      ticker: tickerSearch.value,
      exactTicker,
      adxState: stateFilter.value,
      maStatus: maFilter.value,
      columns: columnFilters.snapshot(),
    };
  }

  createAdxFilterPresets(
    readFilterSettings,
    (settings) => {
      tickerSearch.value = settings.ticker;
      exactTicker = settings.exactTicker;
      stateFilter.value = settings.adxState;
      maFilter.value = settings.maStatus;
      columnFilters.restore(settings.columns);
    },
  );

  function render() {
    const query = tickerSearch.value.trim().toUpperCase();
    const state = stateFilter.value;
    const ma = maFilter.value;
    const exactMatch = exactTicker && query !== "";
    const visibleColumns = showAllColumns ? columns : compactColumns;
    const predicates = columnFilters.predicates();
    const rows = (report?.rows ?? []).filter((row) => {
      const ticker = row.ticker.toUpperCase();
      return (exactMatch ? ticker === query : ticker.includes(query))
        && (!state || (state === "missing" ? row.marketRegime === null : state === row.adxState))
        && (!ma || ma === row.maStatus)
        && predicates.every(matches => matches(row));
    });
    filteredTickers = rows.map(row => row.ticker);
    selectAllPapertrades.update(rows, report?.date);
    const historical = forwardReturns.sync();
    const valueFor = (row, field) => field === forwardReturnField ? historical?.get(row.ticker)?.mean : row[field];
    rows.sort((a, b) => {
      const first = valueFor(a, sortBy), second = valueFor(b, sortBy);
      if (first === second) return a.ticker.localeCompare(b.ticker);
      if (first == null) return 1;
      if (second == null) return -1;
      return direction * (typeof first === "number" ? first - second : first.localeCompare(second));
    });
    headings.forEach((header, index) => {
      const [field] = columns[index];
      header.hidden = !showAllColumns && !compactFields.has(field);
      if (field === sortBy) header.setAttribute("aria-sort", direction > 0 ? "ascending" : "descending");
      else header.removeAttribute("aria-sort");
    });
    const start = page * pageSize;
    const visible = rows.slice(start, start + pageSize);
    node("adx-rows").replaceChildren(...visible.map((row) => {
      const entry = document.createElement("tr");
      entry.append(...visibleColumns.map(([field, _label, formatter]) => {
        const cell = document.createElement("td");
        cell.dataset.field = field;
        if (field === papertradeField) {
          cell.className = "adx-papertrade-cell";
          cell.append(papertrade.checkbox(row.ticker, report.date, row.close));
          return cell;
        }
        cell.textContent = display(valueFor(row, field), formatter);
        if (field === "ticker") {
          cell.className = "ticker";
          cell.title = `${number.format(row.sessions)} saved sessions`;
        } else if (field === "ma20ZScore") {
          cell.title = describeZScore(row);
        } else if (field === "ma20GapPercentile" && row[field] != null) {
          cell.title = percent.format(row[field]);
        } else if (field === forwardReturnField) {
          const result = historical?.get(row.ticker);
          if (result) {
            const count = document.createElement("button");
            count.type = "button";
            count.className = "adx-return-count";
            count.textContent = `${number.format(result.observations)} ${result.observations === 1 ? "match" : "matches"}`;
            count.title = `See ${row.ticker} returns by condition`;
            count.setAttribute("aria-label", `${count.title}, ${count.textContent}`);
            count.setAttribute("aria-haspopup", "dialog");
            count.addEventListener("click", () => forwardReturns.openBreakdown(row.ticker));
            cell.append(count);
          }
        }
        return cell;
      }));
      return entry;
    }));
    empty.hidden = !!visible.length;
    if (report) {
      empty.textContent = exactMatch
        ? `No exact match for ${query} with the current filters.`
        : "No tickers match these filters.";
    }
    node("adx-previous").disabled = page === 0;
    node("adx-next").disabled = start + pageSize >= rows.length;
    node("adx-page").textContent = rows.length
      ? `${number.format(start + 1)}–${number.format(start + visible.length)} of ${number.format(rows.length)} tickers`
      : "";
    if (report) {
      status.textContent = `${report.date} · ${number.format(report.rows.length)} tickers · ${number.format(report.readyCount)} with a market regime`;
    }
    node("adx-scroll").scrollTop = 0;
  }

  function updateResults() {
    page = 0;
    render();
  }

  function message(error) {
    return error.status === 404 ? "Restart the dashboard to load ADX Summary, then choose Refresh." : error.message;
  }

  async function loadMembership() {
    try {
      const snapshot = await api("/sp500.json");
      if (!Array.isArray(snapshot.tickers) || !snapshot.tickers.length || !snapshot.asOf) {
        throw new Error("The S&P 500 snapshot is empty or invalid.");
      }
      return { asOf: snapshot.asOf, tickers: new Set(snapshot.tickers.map(indexTicker)) };
    } catch (error) {
      console.warn("S&P 500 membership", error.message);
      return null;
    }
  }

  async function loadReport() {
    const date = dateSelect.value;
    if (!date) return;
    columnFilters.close();
    const request = ++version;
    report = undefined;
    empty.textContent = "Calculating the market summary…";
    status.textContent = `Reading saved prices through ${date}…`;
    refresh.disabled = true;
    updateResults();
    try {
      const [result, membership] = await Promise.all([
        api(`/api/adx-summary?date=${encodeURIComponent(date)}`),
        loadMembership(),
      ]);
      if (request !== version) return;
      report = {
        ...result,
        rows: result.rows.map((row) => ({
          ...row,
          sp500: membership ? (membership.tickers.has(indexTicker(row.ticker)) ? "Yes" : "No") : null,
          maPosition: row.pct50,
          gapChange: Number.isFinite(row.pct50) && Number.isFinite(row.priorPct50)
            ? (Math.abs(row.pct50) - Math.abs(row.priorPct50)) * 100
            : null,
        })),
      };
      membershipDate.textContent = membership ? `holdings on ${membership.asOf}.` : "holdings unavailable.";
      columnFilters.setRows(report.rows);
      updateResults();
    } catch (error) {
      if (request !== version) return;
      status.textContent = message(error);
      empty.textContent = "ADX Summary could not be loaded.";
      console.error("ADX Summary", error.message);
    } finally {
      if (request === version) refresh.disabled = false;
    }
  }

  async function loadDates() {
    if (loadingDates) return loadingDates;
    const selected = dateSelect.value;
    refresh.disabled = true;
    loadingDates = (async () => {
      try {
        const { dates } = await api("/api/adx-summary/dates");
        const keepDate = selected && selected !== latest && dates.includes(selected);
        latest = dates[0];
        dateSelect.replaceChildren(...(dates.length ? dates.map((date) => new Option(date, date))
          : [new Option("No saved market sessions", "")]));
        dateSelect.disabled = !latest;
        if (latest) {
          dateSelect.value = keepDate ? selected : latest;
          await loadReport();
        } else {
          status.textContent = "Import market prices to build the ADX Summary.";
          empty.textContent = "No saved market sessions.";
        }
      } catch (error) {
        status.textContent = message(error);
        empty.textContent = "ADX Summary could not be loaded.";
        console.error("ADX Summary dates", error.message);
      } finally {
        refresh.disabled = false;
      }
    })();
    try {
      await loadingDates;
    } finally {
      loadingDates = undefined;
    }
  }

  dateSelect.addEventListener("change", loadReport);
  tickerSearch.addEventListener("input", () => {
    exactTicker = false;
    updateResults();
  });
  tickerSearch.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.isComposing) return;
    event.preventDefault();
    exactTicker = true;
    updateResults();
  });
  stateFilter.addEventListener("change", updateResults);
  maFilter.addEventListener("change", updateResults);
  refresh.addEventListener("click", loadDates);
  columnsToggle.addEventListener("click", () => {
    columnFilters.close();
    showAllColumns = !showAllColumns;
    columnsToggle.setAttribute("aria-expanded", String(showAllColumns));
    columnsToggle.textContent = showAllColumns ? "Show fewer columns" : "Show all columns";
    node("adx-table").classList.toggle("adx-table-compact", !showAllColumns);
    node("adx-scroll").scrollLeft = 0;
    if (!showAllColumns && !compactFields.has(sortBy)) {
      sortBy = "ticker";
      direction = 1;
      updateResults();
    } else {
      render();
    }
  });
  node("adx-previous").addEventListener("click", () => { page--; render(); });
  node("adx-next").addEventListener("click", () => { page++; render(); });
  papertrade.onChange(render);
  return () => latest ? undefined : loadDates();
}
