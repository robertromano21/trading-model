import { cacheSession, readLogs, readModel, readSession, saveLog, saveModel, sessionDates } from "./db.js";
import { createBacktester } from "./backtester.js";
import { createAdxSummary } from "./adx-summary.js";
import { createPapertrade } from "./papertrade.js";

const byId = (id) => document.getElementById(id);
const marketSection = byId("prices");
const historySection = byId("history");
const modelSection = byId("model");
const adxSummarySection = byId("adx-summary");
const papertradeSection = byId("adx-positions");
const backtestSection = byId("backtest");
const logSection = byId("logs");
const connection = byId("connection");
const form = byId("run-form");
const source = byId("source");
const startDate = byId("start-date");
const endDate = byId("end-date");
const updateWorkbook = byId("update-workbook");
const runButton = byId("run-button");
const runState = byId("run-state");
const runSummary = byId("run-summary");
const cacheSummary = byId("cache-summary");
const runIssues = byId("run-issues");
const runFiles = byId("run-files");
const priceSyncStatus = byId("price-sync-status");
const sessionSelect = byId("session-date");
const tickerSearch = byId("ticker-search");
const historyForm = byId("history-form");
const historySource = byId("history-source");
const historyDate = byId("history-date");
const historyButton = byId("history-submit");
const modelForm = byId("model-form");
const modelDate = byId("model-date");
const modelButton = byId("model-submit");
const modelSearch = byId("model-search");
const modelFilter = byId("model-filter");
const logSource = byId("log-source");
const logDate = byId("log-date");
const logLevel = byId("log-level");
const logSearch = byId("log-search");
const mirror = byId("mirror-errors");
const number = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, { style: "percent", maximumFractionDigits: 1 });
const price = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 8 });
const clock = new Intl.DateTimeFormat(undefined, { dateStyle: "short", timeStyle: "medium" });
const PAGE_SIZE = 50;
const strategyNames = { breakout: "Trend-Following Breakout", pullback: "Mean-Reversion Pullback", crossover: "Dual MA Crossover" };
const scannerNames = {
  volatility_compression_squeeze: "Volatility compression",
  trend_emergence_decay: "Trend emergence",
};
const metricFields = [
  ["bars", "Saved sessions"], ["close", "Close"], ["sma20", "SMA 20"], ["ema20", "EMA 20"], ["fastEma", "Fast EMA"],
  ["sma50", "SMA 50"], ["sma200", "SMA 200"], ["std20", "Standard deviation 20"], ["zScore", "Z-score"],
  ["upperBand", "Upper Bollinger band"], ["lowerBand", "Lower Bollinger band"], ["bandwidth", "Band width"],
  ["bandwidthContracting", "Band width contracting"], ["adx14", "ADX 14"], ["regime", "ADX regime"], ["atr14", "ATR 14"],
  ["natr", "Normalized ATR"], ["natrCutoff", "20th-percentile NATR"], ["rsi14", "RSI 14"], ["rsi2", "RSI 2"],
  ["rvol", "Relative volume"], ["halfLife15", "Half-life 15 (sessions)"], ["halfLife60", "Half-life 60 (sessions)"],
  ["halfLifeRatio", "Half-life ratio"], ["flatBase", "Flat-base prefilter"],
];

let serverConfig;
let socket;
let job;
let submitting = false;
let dates = [];
let sessionRows = [];
let marketPage = 0;
let logPage = 0;
let savedLogs = [];
let filteredLogs = [];
let refreshLogTimer;
let logRequest = 0;
let syncQueue = Promise.resolve();
let modelPage = 0;
let model;
let modelRequest;
let detailSymbol;

function preference(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch {
    // The current tab continues without saved preferences.
  }
}

async function api(path, payload) {
  const options = payload === undefined ? undefined : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
  const response = await fetch(path, options);
  if (response.status === 204) return null;
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body.error || "The server could not complete the request.");
    error.status = response.status;
    error.job = body.job;
    throw error;
  }
  return body;
}

function storageError(error) {
  const notice = byId("storage-error");
  notice.textContent = `Browser storage unavailable: ${error.message}`;
  notice.hidden = false;
}

function formatLogArgument(value) {
  if (value instanceof Error) return value.stack || value.message;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return Object.prototype.toString.call(value);
  }
}

function recordLog(level, args) {
  const entry = {
    level,
    timestamp: new Date().toISOString(),
    message: args.map(formatLogArgument).join(" ").slice(0, 4000),
  };
  saveLog(entry).then(() => {
    if (!logSection.hidden && logSource.value === "browser") {
      clearTimeout(refreshLogTimer);
      refreshLogTimer = setTimeout(refreshLogs, 150);
    }
  }).catch(storageError);
  if (level === "error" && mirror.checked) {
    api("/api/browser-logs", entry).catch(() => {
      // The entry is already in the browser log if the server is offline.
    });
  }
}

for (const level of ["log", "warn", "error"]) {
  const original = console[level].bind(console);
  console[level] = (...args) => {
    original(...args);
    recordLog(level, args);
  };
}
window.addEventListener("error", (event) => recordLog("error", [event.error || event.message || "Resource failed to load."]));
window.addEventListener("unhandledrejection", (event) => recordLog("error", [event.reason]));
mirror.checked = preference("prices-mirror-errors") === "true";
mirror.addEventListener("change", () => preference("prices-mirror-errors", String(mirror.checked)));

function cell(text, className) {
  const td = document.createElement("td");
  td.textContent = text;
  if (className) td.className = className;
  return td;
}

function paginate(prefix, count, page) {
  const lastPage = Math.max(0, Math.ceil(count / PAGE_SIZE) - 1);
  const current = Math.min(page, lastPage);
  const first = current * PAGE_SIZE;
  byId(`${prefix}-page`).textContent = count
    ? `${number.format(first + 1)}–${number.format(Math.min(first + PAGE_SIZE, count))} of ${number.format(count)}`
    : "";
  byId(`${prefix}-prev`).disabled = current === 0;
  byId(`${prefix}-next`).disabled = current === lastPage;
  return current;
}

function renderMarket() {
  const query = tickerSearch.value.trim().toUpperCase();
  const filtered = query ? sessionRows.filter((bar) => bar.ticker.toUpperCase().includes(query)) : sessionRows;
  marketPage = paginate("market", filtered.length, marketPage);
  const first = marketPage * PAGE_SIZE;
  const rows = filtered.slice(first, first + PAGE_SIZE).map((bar) => {
    const row = document.createElement("tr");
    row.append(cell(bar.ticker, "ticker"), cell(price.format(bar.open)), cell(price.format(bar.high)),
      cell(price.format(bar.low)), cell(price.format(bar.close)), cell(number.format(bar.volume)),
      cell(bar.direction[0].toUpperCase() + bar.direction.slice(1), bar.direction));
    return row;
  });
  byId("market-rows").replaceChildren(...rows);
  const empty = byId("market-empty");
  empty.textContent = sessionRows.length ? "No symbols match your search." : "Import a session to view its prices.";
  empty.hidden = rows.length > 0;
  const count = sessionSelect.disabled ? 0 : sessionSelect.options.length;
  byId("market-summary").textContent = count
    ? `${number.format(sessionRows.length)} symbols in this session · ${number.format(count)} saved sessions`
    : "No saved sessions";
}

let serverDates = [];
let marketDatesRequest = 0;

function setDates(selected = sessionSelect.value) {
  const available = [...new Set([...dates, ...serverDates])].sort().reverse();
  const options = available.map((date) => new Option(date, date));
  sessionSelect.replaceChildren(...(options.length ? options : [new Option("No sessions", "")]));
  sessionSelect.disabled = available.length === 0;
  sessionSelect.value = available.includes(selected) ? selected : available[0] ?? "";
}

async function showSession() {
  const date = sessionSelect.value;
  let bars;
  if (serverDates.includes(date)) {
    try {
      ({ bars } = await api(`/api/market-data/${date}`));
    } catch (error) {
      if (!dates.includes(date)) throw error;
    }
  }
  if (!bars) bars = date ? await readSession(date) : [];
  if (date !== sessionSelect.value) return;
  sessionRows = bars.sort((a, b) => a.ticker.localeCompare(b.ticker));
  marketPage = 0;
  renderMarket();
}

async function refreshMarketDates() {
  const request = ++marketDatesRequest;
  const { dates: saved } = await api("/api/market-data/dates");
  if (request !== marketDatesRequest) return;
  const previousLatest = serverDates[0] ?? serverConfig?.latestLocalDate;
  const followedLatest = !sessionSelect.value || sessionSelect.value === sessionSelect.options[0]?.value;
  serverDates = saved;
  setDates(followedLatest ? "" : sessionSelect.value);
  const latest = saved[0];
  if (latest && (!startDate.value || (startDate.value === previousLatest && !endDate.value))) {
    startDate.value = latest;
    endDate.min = latest;
  }
  if (latest && (!modelDate.value || modelDate.value === previousLatest)) modelDate.value = latest;
  await showSession();
}

sessionSelect.addEventListener("change", () => showSession().catch(storageError));
tickerSearch.addEventListener("input", () => { marketPage = 0; renderMarket(); });
byId("market-prev").addEventListener("click", () => { marketPage -= 1; renderMarket(); });
byId("market-next").addEventListener("click", () => { marketPage += 1; renderMarket(); });

function renderLogs() {
  const query = logSearch.value.trim().toLowerCase();
  const local = logSource.value === "browser";
  filteredLogs = local ? savedLogs.filter((entry) => (!logLevel.value || logLevel.value === entry.level) &&
    entry.message.toLowerCase().includes(query)) : savedLogs;
  logPage = paginate("log", filteredLogs.length, logPage);
  const first = logPage * PAGE_SIZE;
  const rows = filteredLogs.slice(first, first + PAGE_SIZE).map((entry) => {
    const row = document.createElement("tr");
    const { timestamp, level, message, id, ...metadata } = entry;
    const text = cell(message);
    if (!local && Object.keys(metadata).length) {
      const details = document.createElement("details");
      const summary = document.createElement("summary");
      const content = document.createElement("pre");
      summary.textContent = "Details";
      content.textContent = JSON.stringify(metadata, null, 2);
      details.append(summary, content);
      text.append(details);
    }
    row.append(cell(clock.format(new Date(timestamp))), cell(level, level), text);
    return row;
  });
  byId("log-rows").replaceChildren(...rows);
  byId("log-empty").hidden = rows.length > 0;
  byId("export-logs").disabled = rows.length === 0;
}

async function refreshLogs() {
  const request = ++logRequest;
  try {
    const stream = logSource.value;
    if (stream === "browser") {
      const entries = await readLogs();
      if (request !== logRequest) return;
      savedLogs = entries;
      byId("log-summary").textContent = "The latest 5,000 dashboard console messages are kept in this browser.";
    } else {
      const params = new URLSearchParams({ stream, level: logLevel.value, q: logSearch.value.trim() });
      if (logDate.value) params.set("date", logDate.value);
      const data = await api(`/api/logs?${params}`);
      if (request !== logRequest) return;
      const selected = logDate.value;
      logDate.replaceChildren(new Option("Latest", ""), ...data.dates.map((date) => new Option(date, date)));
      logDate.value = data.dates.includes(selected) ? selected : "";
      savedLogs = data.entries;
      byId("log-summary").textContent = data.matches > data.entries.length
        ? `Showing the latest ${number.format(data.entries.length)} of ${number.format(data.matches)} matching entries for ${data.date} UTC.`
        : data.date ? `${number.format(data.matches)} matching entries for ${data.date} UTC.` : "No log files yet.";
    }
    byId("log-error").hidden = true;
    renderLogs();
  } catch (error) {
    if (request !== logRequest) return;
    const notice = byId("log-error");
    notice.textContent = `Could not read logs: ${error.message}`;
    notice.hidden = false;
  }
}

logSource.addEventListener("change", () => {
  const local = logSource.value === "browser";
  byId("log-date-field").hidden = local;
  const info = byId("log-info");
  info.textContent = local ? "Log" : "Info";
  info.value = local ? "log" : "info";
  logDate.value = "";
  logLevel.value = "";
  logPage = 0;
  savedLogs = [];
  renderLogs();
  void refreshLogs();
});
logDate.addEventListener("change", () => { logPage = 0; void refreshLogs(); });
logLevel.addEventListener("change", () => {
  logPage = 0;
  if (logSource.value === "browser") renderLogs();
  else void refreshLogs();
});
logSearch.addEventListener("input", () => {
  logPage = 0;
  if (logSource.value === "browser") {
    renderLogs();
  } else {
    clearTimeout(refreshLogTimer);
    refreshLogTimer = setTimeout(refreshLogs, 200);
  }
});
byId("refresh-logs").addEventListener("click", refreshLogs);
byId("log-prev").addEventListener("click", () => { logPage -= 1; renderLogs(); });
byId("log-next").addEventListener("click", () => { logPage += 1; renderLogs(); });
byId("export-logs").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(filteredLogs, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${logSource.value}-logs-${logDate.value || new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

function showSection() {
  if (location.hash === "#papertrade") {
    location.replace("/paper2.html");
    return;
  }
  const sections = [marketSection, historySection, adxSummarySection, papertradeSection, modelSection, backtestSection, logSection];
  const current = sections.find((section) => `#${section.id}` === location.hash) || marketSection;
  const activeTab = current === papertradeSection ? adxSummarySection : current;
  for (const section of sections) {
    section.hidden = section !== current;
    const tab = byId(`${section.id}-tab`);
    if (section === activeTab) tab?.setAttribute("aria-current", "page");
    else tab?.removeAttribute("aria-current");
  }
  if (current === logSection) void refreshLogs();
  if (current === historySection) void refreshHistory();
  if (current === adxSummarySection) void openAdxSummary();
  if (current === papertradeSection) void papertrade.refresh();
  if (current === backtestSection) void openBacktester();
}

const openBacktester = createBacktester(api);
const papertrade = createPapertrade(api);
const openAdxSummary = createAdxSummary(api, papertrade);
window.addEventListener("hashchange", showSection);
showSection();

function refreshControls() {
  const running = job?.status === "running" || submitting;
  const needsKey = source.value === "massive" && !serverConfig?.massiveAvailable;
  const usesDates = source.value !== "pending";
  runButton.disabled = !serverConfig || running || needsKey;
  runButton.textContent = running && job?.kind === "import" ? "Importing…" : "Import prices";
  modelButton.disabled = !serverConfig || running;
  modelButton.textContent = running && job?.kind === "strategies" ? "Running…" : "Run model";
  const downloadsHistory = historySource.value === "massive";
  const missingHistoryKey = downloadsHistory && !serverConfig?.massiveAvailable;
  historyDate.max = downloadsHistory ? serverConfig?.latestHistoryDownload ?? "" : "";
  if (historyDate.max && historyDate.value > historyDate.max) historyDate.value = historyDate.max;
  historyButton.disabled = !serverConfig || running || missingHistoryKey;
  historyButton.textContent = running && job?.kind === "history" ? "Building history…" : "Build history";
  byId("history-source-detail").textContent = missingHistoryKey
    ? "Add a Massive API key (Research & data > Data keys) to download missing dates."
    : downloadsHistory ? "Downloads only dates missing from local storage. Requires three years of plan access." : "";
  byId("date-fields").hidden = !usesDates;
  startDate.disabled = !usesDates;
  endDate.disabled = !usesDates;
  byId("source-detail").textContent = source.value !== "massive" ? serverConfig?.dataDir ?? "" :
    needsKey ? "Add a Massive API key (Research & data > Data keys) to enable downloads." : "Download adjusted daily bars.";
  updateWorkbook.disabled = !serverConfig?.workbookExists;
  byId("workbook-detail").textContent = serverConfig?.workbookExists ? serverConfig.workbook :
    serverConfig ? "Excel workbook unavailable." : "";
}

function renderJob() {
  refreshControls();
  if (!job) return;
  runIssues.replaceChildren();
  runIssues.hidden = true;
  runFiles.hidden = true;
  if (job.status === "running") {
    runState.textContent = "Import in progress";
    runSummary.textContent = job.source === "pending" ? "Reading pending JSON files." :
      job.startDate === job.endDate ? job.startDate : `${job.startDate} – ${job.endDate}`;
    return;
  }
  const result = job.result;
  runState.textContent = { complete: "Import complete", partial: "Import completed with issues", failed: "Import failed" }[job.status];
  const count = `${number.format(result.barsCount)} prices across ${result.sessions.filter((day) => day.count > 0).length} sessions.`;
  const excel = result.excel;
  const adxSummary = excel?.adxSummary;
  const summaryText = adxSummary && (adxSummary.updated || adxSummary.skipped)
    ? ` ADX Summary updated ${number.format(adxSummary.updated)} tickers` +
      (adxSummary.skipped ? ` and skipped ${number.format(adxSummary.skipped)}. See the server log for details.` : ".")
    : "";
  const excelSummary = excel?.status === "saved"
    ? ` Excel saved ${number.format(excel.snapshot)} symbols for ${excel.date} and added ${number.format(excel.adxAppended)} ADX rows.${summaryText}`
    : excel?.status === "newer" ? ` Excel already has a newer snapshot for ${excel.date}.` : "";
  runSummary.textContent = result.message ?? count + excelSummary;
  const issues = result.sessions.filter((day) => day.message).map((day) => `${day.date} · ${day.message}`);
  if (excel?.message) issues.unshift(`Excel · ${excel.message}`);
  const visible = issues.slice(0, 6);
  if (issues.length > visible.length) visible.push(`${issues.length - visible.length} more. See the server log.`);
  runIssues.replaceChildren(...visible.map((issue) => {
    const item = document.createElement("li");
    item.textContent = issue;
    return item;
  }));
  runIssues.hidden = visible.length === 0;
  const files = result.sessions.filter((day) => day.file);
  byId("run-file-list").replaceChildren(...files.map((day) => {
    const item = document.createElement("li");
    item.textContent = [day.date, day.file, `${number.format(day.count)} prices`, day.archive].filter(Boolean).join(" · ");
    return item;
  }));
  runFiles.hidden = files.length === 0;
}

async function storeJobSessions(completed) {
  const sessions = completed.result.sessions.filter((day) => day.count > 0);
  let cached = 0;
  let latest;
  for (const day of sessions) {
    try {
      const { bars } = await api(`/api/market-data/${day.date}`);
      await cacheSession(day.date, bars);
      cached += 1;
      latest = day.date;
      if (!dates.includes(day.date)) dates = [...dates, day.date].sort().reverse();
      const selected = sessionSelect.value;
      setDates(selected);
      if (!selected || selected === day.date) await showSession();
    } catch (error) {
      console.error(`Could not cache ${day.date}.`, error);
    }
    if (job?.id === completed.id) {
      cacheSummary.textContent = `${cached} of ${sessions.length} sessions cached in this browser.`;
    }
  }
  if (cached === sessions.length) {
    preference("prices-cached-job", completed.id);
  }
  if (latest && job?.id === completed.id) {
    setDates(latest);
    await showSession().catch(storageError);
  }
}

function acceptJob(incoming) {
  if (!incoming || (job && incoming.startedAt < job.startedAt)) return;
  if (job?.id === incoming.id) {
    const progress = incoming.kind === "history" && incoming.status === "running" &&
      job.status === "running" && (incoming.progress?.checked ?? 0) > (job.progress?.checked ?? 0);
    if (!progress && (incoming.status === job.status || incoming.status === "running")) return;
  }
  job = incoming;
  if (incoming.status !== "running" && ["import", "history"].includes(incoming.kind) && !papertradeSection.hidden) {
    void papertrade.refresh();
  }
  if (incoming.kind === "history") {
    refreshControls();
    renderBackfill();
    if (incoming.startup) renderPriceSync(incoming);
    if (incoming.status !== "running") void refreshMarketDates().catch(storageError);
    return;
  }
  if (incoming.kind === "strategies") {
    refreshControls();
    void acceptModelJob(incoming);
    return;
  }
  cacheSummary.textContent = "";
  renderJob();
  if (incoming.status !== "running" && !historySection.hidden) void refreshHistory();
  if (incoming.status !== "running" && incoming.result.barsCount &&
      (preference("prices-cached-job") !== incoming.id ||
       incoming.result.sessions.some((day) => day.count > 0 && !dates.includes(day.date)))) {
    syncQueue = syncQueue.then(() => storeJobSessions(incoming));
  }
}

source.addEventListener("change", refreshControls);
startDate.addEventListener("change", () => { endDate.min = startDate.value; });
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitting = true;
  refreshControls();
  runState.textContent = "Starting import";
  runSummary.textContent = "";
  cacheSummary.textContent = "";
  runIssues.hidden = true;
  runFiles.hidden = true;
  try {
    const { job: started } = await api("/api/run-model", {
      source: source.value,
      startDate: startDate.value,
      endDate: endDate.value || startDate.value,
      updateWorkbook: updateWorkbook.checked && !updateWorkbook.disabled,
    });
    acceptJob(started);
  } catch (error) {
    if (error.job) acceptJob(error.job);
    else {
      runState.textContent = "Could not start import";
      runSummary.textContent = error.message;
    }
    console.warn(error);
  } finally {
    submitting = false;
    refreshControls();
  }
});

async function refreshServer() {
  try {
    serverConfig = await api("/api/status");
    if (!startDate.value && serverConfig.latestLocalDate) {
      startDate.value = serverConfig.latestLocalDate;
      endDate.min = startDate.value;
    }
    if (!modelDate.value && serverConfig.latestLocalDate) modelDate.value = serverConfig.latestLocalDate;
    if (!historyDate.value) historyDate.value = serverConfig.latestLocalDate || serverConfig.latestHistoryDownload;
    acceptJob(serverConfig.job);
    void refreshMarketDates().catch(storageError);
  } catch {
    serverConfig = undefined;
  }
  refreshControls();
}

async function refreshHistory() {
  const notice = byId("history-coverage");
  try {
    const coverage = await api("/api/history/status");
    notice.textContent = coverage.sessions
      ? `${number.format(coverage.bars)} prices across ${number.format(coverage.sessions)} saved sessions · ${coverage.firstDate} to ${coverage.lastDate}`
      : "Build history to store daily prices in local SQLite.";
  } catch (error) {
    notice.textContent = error.message;
  }
}

function renderPriceSync(sync) {
  if (sync.status === "running") {
    const { checked, total } = sync.progress ?? {};
    priceSyncStatus.textContent = total
      ? `Updating market prices… ${number.format(checked)} of ${number.format(total)} trading sessions checked.`
      : "Checking for the latest market prices…";
    return;
  }
  const { lastSaved, downloaded, indexed, missing, errors, message, issues } = sync.result;
  const updated = (downloaded ?? 0) + (indexed ?? 0);
  const latest = lastSaved ? `Latest saved session: ${lastSaved}.` : "Price history is unavailable.";
  const pending = (missing ?? 0) + (errors ?? 0);
  priceSyncStatus.textContent = [
    updated ? `Updated ${number.format(updated)} market ${updated === 1 ? "session" : "sessions"}.` : "",
    latest,
    pending ? `${number.format(pending)} ${pending === 1 ? "session needs" : "sessions need"} another check.` : "",
    message || issues?.[0],
  ].filter(Boolean).join(" ");
}

function renderBackfill() {
  const state = byId("history-state");
  const status = byId("history-status");
  const meter = byId("history-progress");
  const issues = byId("history-issues");
  issues.hidden = true;
  meter.hidden = job.status !== "running";
  if (job.status === "running") {
    state.textContent = "Building history";
    if (job.progress) {
      const { checked, total, date, sessions, downloaded } = job.progress;
      meter.max = total;
      meter.value = checked;
      const unit = job.startup ? "trading sessions" : "weekdays";
      status.textContent = `${number.format(checked)} of ${number.format(total)} ${unit} checked. ${date} · ${number.format(sessions)} saved sessions, ${number.format(downloaded)} downloaded.`;
    } else {
      meter.removeAttribute("value");
      status.textContent = `Checking saved dates through ${job.today ?? job.date}.`;
    }
    return;
  }
  const result = job.result;
  state.textContent = { complete: "History complete", partial: "History is partial", failed: "History could not be completed" }[job.status];
  status.textContent = Number.isInteger(result.sessions)
    ? [`${number.format(result.sessions)} saved sessions and ${number.format(result.bars)} prices. ${number.format(result.missing)} ${job.startup ? "trading sessions" : "weekdays"} are unverified.`,
      result.empty ? `${number.format(result.empty)} confirmed dates without trading data.` : "", result.message].filter(Boolean).join(" ")
    : result.message;
  issues.replaceChildren(...(result.issues ?? []).map((message) => {
    const item = document.createElement("li");
    item.textContent = message;
    return item;
  }));
  issues.hidden = !issues.childElementCount;
  void refreshHistory();
}

historySource.addEventListener("change", refreshControls);
historyForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitting = true;
  refreshControls();
  byId("history-state").textContent = "Starting history build";
  byId("history-status").textContent = "";
  byId("history-issues").hidden = true;
  try {
    const { job: started } = await api("/api/backfill-history", { date: historyDate.value, mode: historySource.value });
    acceptJob(started);
  } catch (error) {
    if (error.job) acceptJob(error.job);
    byId("history-state").textContent = "Could not start history";
    byId("history-status").textContent = error.message;
    console.warn(error);
  } finally {
    submitting = false;
    refreshControls();
  }
});

byId("ticker-history-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("ticker-history-submit");
  const rows = byId("ticker-history-rows");
  const notice = byId("ticker-history-status");
  const empty = byId("ticker-history-empty");
  const query = new URLSearchParams({ ticker: byId("history-ticker").value.trim() });
  const through = byId("ticker-history-date").value;
  if (through) query.set("through", through);
  button.disabled = true;
  notice.textContent = "Reading SQLite history…";
  rows.replaceChildren();
  empty.hidden = true;
  try {
    const { ticker, bars } = await api(`/api/history/ticker?${query}`);
    notice.textContent = `${ticker} · ${number.format(bars.length)} saved ${bars.length === 1 ? "session" : "sessions"}`;
    rows.replaceChildren(...bars.map((bar) => {
      const row = document.createElement("tr");
      row.append(cell(bar.date), cell(price.format(bar.open)), cell(price.format(bar.high)),
        cell(price.format(bar.low)), cell(price.format(bar.close)), cell(number.format(bar.volume)));
      return row;
    }));
    empty.textContent = "No saved prices for this ticker and date.";
    empty.hidden = bars.length > 0;
  } catch (error) {
    notice.textContent = error.message;
    console.warn(error);
  } finally {
    button.disabled = false;
  }
});

function signalLabel(signal) {
  if (signal.state === "insufficient") return "Needs history";
  if (signal.state === "none") return "—";
  return `${{ entry: "Enter", exit: "Exit", open: "Open" }[signal.state]} ${signal.side}`;
}

function modelValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return typeof value === "number" ? price.format(value) : value;
}

function renderModelDetail() {
  const detail = byId("model-detail");
  const row = model?.rows.find((symbol) => symbol.ticker === detailSymbol);
  detail.hidden = !row;
  if (!row) return;
  byId("model-detail-title").textContent = `${row.ticker} · ${model.date}`;
  byId("model-metrics").replaceChildren(...metricFields.map(([key, label]) => {
    const item = document.createElement("div");
    const term = document.createElement("dt");
    const value = document.createElement("dd");
    term.textContent = label;
    value.textContent = key === "bars" ? number.format(row.bars) : modelValue(row[key]);
    item.append(term, value);
    return item;
  }));
  const risk = byId("model-risk").valueAsNumber;
  const positions = Object.entries(strategyNames).map(([key, name]) => {
    const signal = row[key];
    const item = document.createElement("li");
    const parts = [name];
    if (["entry", "exit", "open"].includes(signal.state)) {
      parts.push(signalLabel(signal), `entry ${modelValue(signal.entryPrice)} on ${signal.entryDate}`,
        `stop ${modelValue(signal.stop)}`, `initial risk ${modelValue(signal.initialRisk)} per share`,
        `${modelValue(signal.r)}R after ${signal.age} sessions`);
      if (signal.timeStopSessions) parts.push(`time stop at ${signal.timeStopSessions} sessions`);
      if (signal.state === "exit") parts.push(`${signal.reason} at ${modelValue(signal.exitPrice)}`);
      else if (Number.isFinite(risk) && risk > 0) parts.push(`${number.format(Math.floor(risk / signal.initialRisk))} shares at this risk budget`);
    } else {
      parts.push(signal.state === "insufficient" ? "Needs more saved history" : "No position or entry signal");
    }
    item.textContent = parts.join(" · ");
    return item;
  });
  if (row.regimeShift) {
    const item = document.createElement("li");
    item.textContent = row.regimeShift === "trend_emergence_decay"
      ? "Trend emergence · Mean-reversion entries paused for this session."
      : "Volatility compression · Flat-base half-life ratio below 0.4 with contracting band width.";
    positions.push(item);
  }
  byId("model-positions").replaceChildren(...positions);
}

function renderModel() {
  const query = modelSearch.value.trim().toUpperCase();
  const filter = modelFilter.value;
  const rows = (model?.rows ?? []).filter((row) => {
    if (!row.ticker.toUpperCase().includes(query)) return false;
    if (filter === "shift") return Boolean(row.regimeShift);
    if (filter === "insufficient") return row.sma200 === null;
    return !filter || Object.keys(strategyNames).some((key) => row[key].state === filter);
  });
  modelPage = paginate("model", rows.length, modelPage);
  const first = modelPage * PAGE_SIZE;
  byId("model-rows").replaceChildren(...rows.slice(first, first + PAGE_SIZE).map((row) => {
    const tr = document.createElement("tr");
    const symbol = document.createElement("button");
    symbol.type = "button";
    symbol.className = "symbol-button";
    symbol.textContent = row.ticker;
    symbol.setAttribute("aria-label", `Show ${row.ticker} model details`);
    symbol.addEventListener("click", () => {
      detailSymbol = row.ticker;
      renderModelDetail();
      byId("model-detail-title").focus();
    });
    const ticker = cell("");
    ticker.append(symbol);
    const strategies = Object.keys(strategyNames).map((key) => {
      const signal = row[key];
      return cell(signalLabel(signal), signal.state === "entry" ? "up" : signal.state === "exit" ? "warn" : "");
    });
    const scanner = scannerNames[row.regimeShift] ?? (row.flatBase ? "Flat base" : row.bars < 79 ? "Needs history" : "—");
    tr.append(ticker, cell(modelValue(row.close)), cell(modelValue(row.adx14)), ...strategies,
      cell(scanner, row.regimeShift ? "warn" : ""));
    return tr;
  }));
  const empty = byId("model-empty");
  empty.hidden = rows.length > 0;
  empty.textContent = model ? "No symbols match these filters." : "No model results to display.";
  if (model) {
    const stats = model.summary;
    byId("model-summary").textContent = [
      model.date, `${number.format(model.symbolCount)} symbols`,
      `${number.format(stats.macroReady ?? 0)} with SMA 200`, `${number.format(stats.entries ?? 0)} entries`,
      `${number.format(stats.exits ?? 0)} exits`, `${number.format(stats.regimeShifts ?? 0)} regime shifts`,
    ].join(" · ");
  }
  renderModelDetail();
}

function showModel(result) {
  model = result;
  model.rows.sort((a, b) => a.ticker.localeCompare(b.ticker));
  modelPage = 0;
  const { sessionCount, firstDate, date, options } = model;
  byId("model-state").textContent = "Model complete";
  byId("model-status").textContent = `${number.format(sessionCount)} market sessions from ${firstDate} through ${date}.`;
  if (!modelDate.value) {
    modelDate.value = date;
    byId("model-baseline").value = options.baseline;
    byId("model-fast").value = String(options.fastEma);
    byId("model-trailing").value = options.trailing;
    byId("model-stop").value = options.breakoutStop;
    byId("model-time").value = options.trendDays;
  }
  renderBreadth(model.breadth);
  renderModel();
}

function renderBreadth(breadth) {
  const section = byId("model-breadth");
  section.hidden = !breadth;
  if (!breadth) return;
  const metrics = Object.entries(breadth.averages).map(([key, { above, eligible }]) => [
    `Above SMA ${key.slice(3)}`,
    eligible ? percent.format(above / eligible) : "—",
    eligible ? `${number.format(above)} of ${number.format(eligible)} eligible` : "No eligible symbols",
  ]);
  metrics.push([
    "Advancing / declining",
    breadth.previousSession ? `${number.format(breadth.advancing)} / ${number.format(breadth.declining)}` : "—",
    breadth.previousSession ? `${number.format(breadth.unchanged)} unchanged` : "No prior full-market session",
  ]);
  byId("breadth-metrics").replaceChildren(...metrics.map(([label, value, detail]) => {
    const container = document.createElement("div");
    const term = document.createElement("dt");
    const count = document.createElement("dd");
    const explanation = document.createElement("small");
    term.textContent = label;
    count.textContent = value;
    explanation.textContent = detail;
    count.append(explanation);
    container.append(term, count);
    return container;
  }));
  byId("breadth-session").textContent = breadth.previousSession
    ? `Close compared with the full-market session on ${breadth.previousSession}.` : "";
}

async function acceptModelJob(incoming) {
  const state = byId("model-state");
  const status = byId("model-status");
  if (incoming.status === "running") {
    modelRequest = incoming.id;
    state.textContent = "Model running";
    status.textContent = `Reading market history through ${incoming.date}.`;
    return;
  }
  if (incoming.status !== "complete") {
    state.textContent = "Model failed";
    status.textContent = incoming.result.message || "See the server logs for details.";
    return;
  }
  if (model?.id === incoming.id) return;
  modelRequest = incoming.id;
  state.textContent = "Loading signals";
  status.textContent = "";
  try {
    const { rows } = await api(`/api/strategy-results/${incoming.id}`);
    if (modelRequest !== incoming.id) return;
    showModel({ id: incoming.id, ...incoming.result, rows });
    await saveModel(model);
  } catch (error) {
    if (modelRequest !== incoming.id) return;
    if (model?.id === incoming.id) {
      status.textContent += " The browser could not save this scan.";
      storageError(error);
    } else {
      state.textContent = "Could not load results";
      status.textContent = error.message;
    }
  }
}

modelSearch.addEventListener("input", () => { modelPage = 0; renderModel(); });
modelFilter.addEventListener("change", () => { modelPage = 0; renderModel(); });
byId("model-prev").addEventListener("click", () => { modelPage -= 1; renderModel(); });
byId("model-next").addEventListener("click", () => { modelPage += 1; renderModel(); });
byId("model-risk").addEventListener("input", renderModelDetail);
byId("close-model-detail").addEventListener("click", () => {
  detailSymbol = null;
  renderModelDetail();
  modelSearch.focus();
});

modelForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitting = true;
  refreshControls();
  byId("model-state").textContent = "Starting model";
  byId("model-status").textContent = "";
  try {
    const { job: started } = await api("/api/run-strategies", {
      date: modelDate.value,
      baseline: byId("model-baseline").value,
      fastEma: Number(byId("model-fast").value),
      trailing: byId("model-trailing").value,
      breakoutStop: byId("model-stop").value,
      trendDays: Number(byId("model-time").value),
    });
    acceptJob(started);
  } catch (error) {
    if (error.job) acceptJob(error.job);
    byId("model-state").textContent = "Could not start model";
    byId("model-status").textContent = error.message;
    console.warn(error);
  } finally {
    submitting = false;
    refreshControls();
  }
});

function connect() {
  socket = new WebSocket(`ws://${location.host}/live`);
  socket.addEventListener("open", () => {
    connection.textContent = "Local server online";
    connection.dataset.state = "online";
    void refreshServer();
    if (!papertradeSection.hidden) void papertrade.refresh();
  });
  socket.addEventListener("message", (event) => acceptJob(JSON.parse(event.data).job));
  socket.addEventListener("close", () => {
    connection.textContent = "Reconnecting";
    connection.dataset.state = "offline";
    serverConfig = undefined;
    marketDatesRequest += 1;
    serverDates = [];
    setDates();
    void showSession().catch(storageError);
    refreshControls();
    setTimeout(connect, 3000);
  });
}

async function initialize() {
  try {
    dates = await sessionDates();
    setDates();
    await showSession();
    const cached = await readModel();
    if (cached) showModel(cached);
  } catch (error) {
    storageError(error);
  }
  connect();
}

void initialize();
