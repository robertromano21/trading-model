// Per-ticker view: chart (Schwab intraday / local daily) with engine-computed indicators, live quote updates,
// paper-trade markers, AI verdict levels, fundamentals/news (AI layer), and the manual Buy/Sell ticket (live.js).
const LWC = window.LightweightCharts;
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const symbol = (params.get("symbol") || "SPY").toUpperCase().replace(/[^A-Z0-9.\-]/g, "").slice(0, 10) || "SPY";
// ?asset=crypto: the coin (Coinbase candles via /api/crypto/chart), not a same-symbol stock/ETF. Chosen in the nav picker.
const isCrypto = params.get("asset") === "crypto";
if (isCrypto) document.body.classList.add("crypto-mode");
let cryptoInfo = null;
const store = (key, fallback) => { try { return JSON.parse(localStorage.getItem(`ticker.${key}`)) ?? fallback; } catch { return fallback; } };
const save = (key, value) => localStorage.setItem(`ticker.${key}`, JSON.stringify(value));
const fmt = (x, d = 2) => (x == null || !Number.isFinite(Number(x)) ? "–" : Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const OVERLAYS = [
  ["sma20", "SMA 20", "#f2c14e", ["sma20"]], ["sma50", "SMA 50", "#f78154", ["sma50"]], ["sma100", "SMA 100", "#b4436c", ["sma100"]],
  ["sma200", "SMA 200", "#5f0f40", ["sma200"]], ["ema20", "EMA 20", "#8bd3dd", ["ema20"]], ["ema50", "EMA 50", "#4d9de0", ["ema50"]],
  ["ema100", "EMA 100", "#3e60c1", ["ema100"]], ["ema200", "EMA 200", "#7768ae", ["ema200"]],
  ["don20", "Donchian 20", "#9aa5b1", ["donchian20_upper", "donchian20_lower"]], ["don55", "Donchian 55", "#6c7a89", ["donchian55_upper", "donchian55_lower"]],
  ["bb20", "Bollinger 20/2", "#c3a6ff", ["bb20_upper", "bb20_mid", "bb20_lower"]], ["vwap", "VWAP (intraday)", "#ffffff", ["vwap"]],
  ["trades", "Paper trades/fills", "#6fd08c", []], ["ai", "AI entry band / stop", "#ffd166", []], ["levels", "Paper cost / stop", "#ff8a80", []],
  ["exhMarks", "Exhaustion events", "#ff8a80", []],
];
const PANES = [
  ["volume", "Volume", null], ["rsi", "RSI 14", [["rsi14", "#f2c14e"]]], ["atr", "ATR 14 (% of price)", [["atr14_pct", "#f78154"]]],
  ["rvol", "Realized vol 20/60 (ann. %)", [["rvol20", "#4d9de0"], ["rvol60", "#7768ae"]]],
  ["z", "Z-score 20 / mean-rev 10", [["z20", "#8bd3dd"], ["z10_meanrev", "#ff8a80"]]],
  ["rs", "Relative strength vs SPY", [["rs_spy", "#6fd08c"], ["rs_spy_sma50", "#9aa5b1"]]],
  ["tsmom", "TSMOM signal (daily)", [["tsmom_signal", "#f2c14e"]]],
  ["dxy", "DXY proxy (dollar index, not ICE futures)", [["dxy_proxy", "#7aa2f7"]]],
  ["exh", "Exhaustion score (daily)", [["exhaustion", "#ff8a80"], ["exhaustion_up", "#f2c14e"], ["exhaustion_down", "#6fd08c"]]],
];
const RANGES = { "1d": [["63", "3 months"], ["126", "6 months"], ["252", "1 year"], ["all", "All history"]],
  "5m": [["1", "1 session"], ["2", "2 sessions"], ["5", "5 sessions"], ["10", "10 sessions"], ["20", "20 sessions"], ["40", "40 sessions"]],
  "1m": [["1", "1 session"], ["2", "2 sessions"], ["5", "5 sessions"], ["10", "10 sessions"], ["20", "20 sessions"]],
  "1h": [["24", "1 day"], ["72", "3 days"], ["168", "1 week"], ["336", "2 weeks"], ["720", "30 days"], ["all", "All cached (~6 weeks)"]] };
const CRYPTO_DAILY = [["90", "3 months"], ["180", "6 months"], ["365", "1 year"], ["730", "2 years"], ["all", "All history"]];
const rangesFor = (t) => (isCrypto && t === "1d" ? CRYPTO_DAILY : RANGES[t]);
const barRanges = (t) => t === "1d" || t === "1h";   // range = number of bars (client-side), not sessions to load
const TF_KEY = isCrypto ? "crypto.tf" : "tf";
const rangeKey = (t) => `${isCrypto ? "crypto." : ""}range.${t}`;
const defaultRange = (t) => (t === "1d" ? (isCrypto ? "365" : "252") : t === "1h" ? "168" : "5");

let tf = store(TF_KEY, "1d");
if (isCrypto ? !["1d", "1h"].includes(tf) : !["1d", "5m", "1m"].includes(tf)) tf = "1d";
let overlays = store("overlays", { sma50: true, sma200: true, bb20: false, trades: true, ai: true, levels: true, vwap: true, exhMarks: true });
let panes = store("panes", { volume: true, rsi: true, z: false, rs: false, atr: false, rvol: false, tsmom: false, dxy: false, exh: false });
if (overlays.exhMarks === undefined) overlays.exhMarks = true;
let range = store(rangeKey(tf), defaultRange(tf));
let data = null, chart = null, candles = null, volumeSeries = null, markerApi = null;
let verdict = null, lastQuote = null, schwab = null, pollTimer = null, barTimer = null;

// ------------------------------------------------------------------ controls
function buildControls() {
  $("symbol").textContent = isCrypto ? `${symbol} · crypto` : symbol;
  document.title = `${symbol}${isCrypto ? " (crypto)" : ""} · Trading Model`;
  for (const b of $("tf").querySelectorAll("button")) {
    b.hidden = isCrypto ? !["1d", "1h"].includes(b.dataset.tf) : b.dataset.tf === "1h";
    b.classList.toggle("on", b.dataset.tf === tf);
    b.onclick = () => { tf = b.dataset.tf; save(TF_KEY, tf); range = store(rangeKey(tf), defaultRange(tf)); buildRange(); buildControls(); load(); };
  }
  buildRange();
  $("overlays").replaceChildren(...OVERLAYS.map(([key, label, color]) => toggle(key, label, color, overlays, "overlays")));
  $("panes").replaceChildren(...PANES.map(([key, label, series]) => toggle(key, label, series?.[0]?.[1] ?? "#9aa5b1", panes, "panes")));
}
function buildRange() {
  $("range").replaceChildren(...rangesFor(tf).map(([v, l]) => Object.assign(document.createElement("option"), { value: v, textContent: l, selected: v === range })));
  $("range").onchange = () => { range = $("range").value; save(rangeKey(tf), range); barRanges(tf) ? render() : load(); };
}
function toggle(key, label, color, state, name) {
  const wrap = document.createElement("label");
  const box = Object.assign(document.createElement("input"), { type: "checkbox", checked: Boolean(state[key]) });
  box.onchange = () => { state[key] = box.checked; save(name, state); render(); };
  const sw = Object.assign(document.createElement("span"), { className: "swatch" });
  sw.style.background = color;
  wrap.append(box, sw, label);
  return wrap;
}

// ------------------------------------------------------------------ data
async function getJson(url, body) {
  const response = await fetch(url, body === undefined ? { cache: "no-store" } :
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}
async function load(refresh = false) {
  $("chart-note").textContent = "Loading…";
  try {
    if (isCrypto) data = await getJson(`/api/crypto/chart/${encodeURIComponent(symbol)}?tf=${tf}${refresh ? "&refresh=1" : ""}`);
    else {
      const sessions = tf === "1d" ? "" : `&sessions=${range}`;
      data = await getJson(`/api/chart/${encodeURIComponent(symbol)}?tf=${tf}${sessions}${refresh ? "&refresh=1" : ""}`);
      schwab = data.schwab;
    }
    render();
    renderSide();
  } catch (error) {
    $("chart-note").textContent = `Chart unavailable: ${error.message}`;
  }
}

// Missing or short local history (e.g. an OTC name with 2 days of Massive bars): fetch it once from Schwab market data
// (engine/schwab_ondemand.py via /api/symbols/fetch), show "Loading from Schwab…", then redraw. Also when picked with ?fetch=1.
let historyChecked = false;
async function maybeFetchHistory() {
  if (isCrypto || historyChecked || tf !== "1d") return;
  historyChecked = true;
  const bars = data?.bars?.length ?? 0;
  const short = params.get("fetch") === "1" || bars === 0 || (!data?.history?.schwab && bars < 750);
  if (!short) return;
  const url = `/api/symbols/fetch/${encodeURIComponent(symbol)}`;
  try {
    let st = await getJson(url, {});
    const note = (text) => { $("chart-note").textContent = text; };
    for (let i = 0; i < 60 && st.status === "running"; i += 1) {
      note(`Loading full history from Schwab… (${bars} local bars so far)`);
      await new Promise((r) => setTimeout(r, 2500));
      st = await getJson(url);
    }
    if (st.status === "complete") {
      const before = bars;
      await load();
      const after = data?.bars?.length ?? 0;
      if (after > before) note(`${$("chart-note").textContent} Loaded ${after - before} more daily bars from Schwab.`);
    } else if (["invalid", "empty"].includes(st.status)) note(`Schwab has no daily history for ${symbol}. ${st.message ?? ""}`);
    else if (st.status === "error") note(`Schwab history fetch failed: ${st.message ?? "unknown error"}. Showing local data.`);
  } catch (error) {
    $("chart-note").textContent += ` (Schwab history fetch unavailable: ${error.message})`;
  }
}

// ------------------------------------------------------------------ chart
function snapTime(t) {
  // map a time onto the last bar at or before it (markers must sit on a bar)
  let lo = 0, hi = data.bars.length - 1, best = null;
  const key = (x) => (typeof x === "string" ? x : Number(x));
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (key(data.bars[mid].time) <= key(t)) { best = data.bars[mid].time; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}
function dayStart(date) { return Date.UTC(+date.slice(0, 4), +date.slice(5, 7) - 1, +date.slice(8, 10)) / 1000; }
function intradayTime(date, minute) {
  const f = tf === "1m" ? 1 : 5;
  return dayStart(date) + Math.floor(minute / f) * f * 60;
}

function render() {
  if (!data) return;
  const el = $("chart");
  if (chart) { chart.remove(); chart = null; }
  const notes = [...(data.warnings ?? []), ...(data.notes ?? [])];
  if (!data.bars?.length) { $("chart-note").textContent = notes.join(" ") || "No data."; return; }
  chart = LWC.createChart(el, {
    autoSize: true,
    // attribution logo off (it injects an inline <style> the CSP blocks); the credit link is in the page footer instead
    layout: { background: { type: "solid", color: "#0f1419" }, textColor: "#c9d4dc", panes: { separatorColor: "#26323d" },
      attributionLogo: false },
    grid: { vertLines: { color: "#18222b" }, horzLines: { color: "#18222b" } },
    timeScale: { timeVisible: tf !== "1d", secondsVisible: false, borderColor: "#26323d" },
    rightPriceScale: { borderColor: "#26323d" },
    crosshair: { mode: LWC.CrosshairMode.Normal },
  });
  candles = chart.addSeries(LWC.CandlestickSeries, { upColor: "#26a69a", downColor: "#ef5350", borderVisible: false,
    wickUpColor: "#26a69a", wickDownColor: "#ef5350" }, 0);
  candles.setData(data.bars);
  const line = (pane, color, values, opts = {}) => {
    const s = chart.addSeries(LWC.LineSeries, { color, lineWidth: 1, priceLineVisible: false, lastValueVisible: false,
      crosshairMarkerVisible: false, ...opts }, pane);
    s.setData(values ?? []);
    return s;
  };
  for (const [key, , color, names] of OVERLAYS) {
    if (!overlays[key] || !names.length) continue;
    names.forEach((n, i) => data.indicators[n] && line(0, color, data.indicators[n], { lineStyle: i === 1 && key.startsWith("bb") ? 2 : 0 }));
  }
  let pane = 0;
  if (panes.volume) {
    volumeSeries = chart.addSeries(LWC.HistogramSeries, { priceFormat: { type: "volume" }, priceScaleId: "vol", lastValueVisible: false, priceLineVisible: false }, 0);
    volumeSeries.setData(data.volume.map((v) => ({ time: v.time, value: v.value, color: v.up ? "rgba(38,166,154,.45)" : "rgba(239,83,80,.45)" })));
    chart.priceScale("vol", 0).applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    candles.priceScale().applyOptions({ scaleMargins: { top: 0.05, bottom: 0.22 } });
  } else volumeSeries = null;
  for (const [key, , series] of PANES) {
    if (!series || !panes[key]) continue;
    const present = series.filter(([n]) => data.indicators[n]?.length);
    if (!present.length) continue;
    pane += 1;
    const created = present.map(([n, color], i) => line(pane, color, data.indicators[n], { lastValueVisible: i === 0, title: n }));
    const guide = (price, color = "#4a5864") => created[0].createPriceLine({ price, color, lineWidth: 1, lineStyle: LWC.LineStyle.Dotted, axisLabelVisible: false });
    if (key === "rsi") { guide(70); guide(30); }
    if (key === "z") { guide(2); guide(0); guide(-2, "#ff8a80"); }
    if (key === "tsmom") { guide(0.5); }
    if (key === "rs") { guide(100); }
  }
  const paneList = chart.panes();
  paneList.forEach((p, i) => p.setStretchFactor(i === 0 ? 3 : 1));
  markerApi = LWC.createSeriesMarkers(candles, buildMarkers());
  priceLines();
  // visible range
  if (barRanges(tf) && range !== "all") {
    const n = data.bars.length;
    chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, n - Number(range)), to: n + 3 });
  } else chart.timeScale().fitContent();
  if (!lastQuote && data.latest?.close != null) {   // no live quote (Schwab off / closed): show the last local close
    $("last").textContent = fmt(data.latest.close);
    const back = isCrypto && tf === "1h" ? 24 : 1;   // crypto hourly: change over 24 h
    const prev = data.bars.length > back ? data.bars[data.bars.length - 1 - back].close : null;
    if (prev) {
      const ch = data.latest.close - prev;
      $("change").textContent = `${ch >= 0 ? "+" : ""}${fmt(ch)} (${ch >= 0 ? "+" : ""}${fmt((ch / prev) * 100)}%)`;
      $("change").className = ch >= 0 ? "pos" : "neg";
    }
    $("quote-meta").textContent = isCrypto
      ? `latest Coinbase ${tf === "1h" ? "hourly candle (24 h change)" : "daily candle (UTC day, still open)"} · ${data.name || symbol}`
      : `last close in the local data (${data.last})`;
  }
  const flags = data.dataFlags ?? {};
  $("chart-note").textContent = [
    `${data.source}. ${data.bars.length} bars ${tf === "1h" ? "(hourly)" : `${data.first} → ${data.last}`}${tf !== "1d" ? " (times ET)" : ""}.`,
    flags.excluded ? "This ticker is EXCLUDED from strategies (unresolved price anomaly)." : "",
    ...notes,
  ].filter(Boolean).join(" ");
  if (lastQuote) applyQuote(lastQuote, false);
}

function buildMarkers() {
  const out = [];
  const paper = data.paper ?? {};
  if (overlays.trades) {
    if (tf === "1d") {
      for (const f of paper.fills ?? []) {
        const t = snapTime(f.fill_date);
        if (!t) continue;
        const buy = f.shares > 0;
        out.push({ time: t, position: buy ? "belowBar" : "aboveBar", color: buy ? "#6fd08c" : "#ff8a80",
          shape: buy ? "arrowUp" : "arrowDown", text: `${f.account === "default" ? "" : f.account + " "}${buy ? "B" : "S"} ${Math.abs(f.shares)} @ ${fmt(f.price)}` });
      }
      for (const o of paper.orders ?? []) {
        const t = snapTime(o.decision_date);
        if (t) out.push({ time: t, position: o.side === "buy" ? "belowBar" : "aboveBar", color: "#ffd166", shape: "circle",
          text: `pending ${o.side} ${o.shares}` });
      }
    } else {
      const shown = new Set(data.sessions ?? []);
      for (const d of paper.dayTrades ?? []) {
        if (!shown.has(d.session_date)) continue;
        const tIn = snapTime(intradayTime(d.session_date, d.entry_minute));
        const tOut = d.exit_minute != null ? snapTime(intradayTime(d.session_date, d.exit_minute)) : null;
        if (tIn) out.push({ time: tIn, position: "belowBar", color: "#6fd08c", shape: "arrowUp", text: `DT in ${d.shares} @ ${fmt(d.entry_price)}` });
        if (tOut) out.push({ time: tOut, position: "aboveBar", color: d.pnl >= 0 ? "#6fd08c" : "#ff8a80", shape: "arrowDown",
          text: `out ${d.reason} ${d.pnl >= 0 ? "+" : ""}${fmt(d.pnl)}` });
      }
    }
  }
  if (tf === "1d" && overlays.exhMarks) {
    const streakUp = Object.fromEntries((data.indicators?.up_streak ?? []).map((p) => [p.time, p.value]));
    const streakDn = Object.fromEntries((data.indicators?.down_streak ?? []).map((p) => [p.time, p.value]));
    for (const p of data.indicators?.exhaustion_flag_up ?? []) {
      if (p.value >= 0.5) out.push({ time: p.time, position: "aboveBar", color: "#ff8a80", shape: "arrowDown",
        text: `exh ↑ ${Math.round(streakUp[p.time] || 0)}d` });
    }
    for (const p of data.indicators?.exhaustion_flag_down ?? []) {
      if (p.value >= 0.5) out.push({ time: p.time, position: "belowBar", color: "#6fd08c", shape: "arrowUp",
        text: `exh ↓ ${Math.round(streakDn[p.time] || 0)}d` });
    }
  }
  const key = (x) => (typeof x === "string" ? x : Number(x));
  return out.sort((a, b) => (key(a.time) < key(b.time) ? -1 : key(a.time) > key(b.time) ? 1 : 0));
}

function priceLines() {
  const add = (price, color, title, style = LWC.LineStyle.Dashed) => price != null && Number.isFinite(Number(price)) &&
    candles.createPriceLine({ price: Number(price), color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title });
  if (overlays.ai && verdict) {
    const e = verdict.entry ?? {};
    add(e.bandLow ?? verdict.entry_low, "#ffd166", "AI entry low");
    add(e.bandHigh ?? verdict.entry_high, "#ffd166", "AI entry high");
    add(e.limitPrice ?? verdict.limit_price, "#ffe9a8", "AI limit", LWC.LineStyle.Dotted);
    add(verdict.stop?.price ?? verdict.stop, "#ff6b6b", "AI stop");
    add(verdict.takeProfit?.price ?? verdict.take_profit, "#6fd08c", "AI target", LWC.LineStyle.Dotted);
  }
  if (overlays.levels && data.paper) {
    for (const p of data.paper.positions ?? []) add(p.avg_cost, "#9cc9ff", `${p.account} cost`, LWC.LineStyle.Dotted);
    for (const s of data.paper.stops ?? []) add(s.stop, "#ff8a80", `${s.account} stop`);
  }
}

// ------------------------------------------------------------------ live quotes
function applyQuote(q, updateChart = true) {
  $("last").textContent = fmt(q.last);
  const ch = q.change, pct = q.changePct;
  $("change").textContent = ch == null ? "" : `${ch >= 0 ? "+" : ""}${fmt(ch)} (${pct >= 0 ? "+" : ""}${fmt(pct)}%)`;
  $("change").className = ch >= 0 ? "pos" : "neg";
  const when = q.tradeTime ? new Date(q.tradeTime).toLocaleTimeString("en-US", { timeZone: "America/New_York" }) + " ET" : "";
  $("quote-meta").textContent = `bid ${fmt(q.bid)} × ask ${fmt(q.ask)} · vol ${fmt(q.volume, 0)} · ${when} · Schwab${q.ageSeconds ? ` (${q.ageSeconds}s old)` : ""}`;
  if (q.description && !$("name").textContent) $("name").textContent = q.description;
  if (!updateChart || !candles || !data?.bars?.length || q.last == null) return;
  const market = lastQuote?.market;
  if (!market?.weekday) return;
  const bars = data.bars;
  const last = bars[bars.length - 1];
  let time;
  if (tf === "1d") {
    if (!(market.regular || market.extended)) return;
    time = market.date;
  } else {
    if (!market.regular) return;
    time = intradayTime(market.date, market.minute);
  }
  const lastKey = typeof last.time === "string" ? last.time : Number(last.time);
  if (time === lastKey) {
    last.high = Math.max(last.high, q.last); last.low = Math.min(last.low, q.last); last.close = q.last;
    candles.update(last);
  } else if (time > lastKey) {
    const bar = tf === "1d"
      ? { time, open: q.open ?? q.last, high: Math.max(q.high ?? q.last, q.last), low: Math.min(q.low ?? q.last, q.last), close: q.last }
      : { time, open: q.last, high: q.last, low: q.last, close: q.last };
    bars.push(bar);
    candles.update(bar);
  }
}
async function pollCrypto() {
  // Crypto trades 24/7: reload the cached Coinbase chart every minute (server tops candles up every ~5 min).
  clearTimeout(pollTimer);
  if ($("live").checked) {
    const range0 = chart?.timeScale().getVisibleLogicalRange();
    await load();
    if (range0) chart?.timeScale().setVisibleLogicalRange(range0);
    $("quote-live-state").textContent = "Live: Coinbase candles (24/7), checking every 60s";
  } else $("quote-live-state").textContent = "Live: paused";
  pollTimer = setTimeout(pollCrypto, 60_000);
}
async function pollQuote() {
  if (isCrypto) return pollCrypto();
  clearTimeout(pollTimer);
  let delay = 60_000;
  if ($("live").checked) {
    try {
      const r = await getJson(`/api/market/quotes?symbols=${encodeURIComponent(symbol)}`);
      const q = r.quotes?.[symbol];
      if (q) { lastQuote = { ...q, market: r.market }; applyQuote(lastQuote); }
      schwab = r.schwab;
      const active = Boolean(r.market?.regular || r.market?.extended);
      delay = r.schwab?.usable && active ? 5_000 : 60_000;
      $("quote-live-state").textContent = !r.schwab?.usable ? "Live: off (Schwab unavailable)"
        : active ? `Live: polling every 5s (${r.market?.regular ? "regular" : "extended"} hours)` : "Live: market closed (checking every 60s)";
      banner();
    } catch (error) {
      $("quote-live-state").textContent = `Live: ${error.message}`;
    }
  } else $("quote-live-state").textContent = "Live: paused";
  pollTimer = setTimeout(pollQuote, delay);
}
function banner() {
  const el = $("data-banner");
  if (isCrypto) { el.hidden = true; return; }
  if (schwab && !schwab.usable) {
    const through = data?.last ? ` Showing local data through ${data.last}${tf !== "1d" ? " (cached Schwab bars)" : ""}.` : "";
    el.textContent = `Live Schwab data unavailable: ${schwab.reason}${through} Use “Click here to reauthorize” above.`;
    el.hidden = false;
  } else el.hidden = true;
}
function scheduleBarRefresh() {
  clearInterval(barTimer);
  barTimer = setInterval(() => {
    if (!isCrypto && $("live").checked && tf !== "1d" && schwab?.usable && lastQuote?.market?.regular) {
      const range0 = chart?.timeScale().getVisibleLogicalRange();
      load(true).then(() => range0 && chart?.timeScale().setVisibleLogicalRange(range0));
    }
  }, 60_000);
}

// ------------------------------------------------------------------ side panels
function renderSide() {
  const L = data.latest ?? {};
  const rows = [["Close", fmt(L.close)], ["SMA 20 / 50", `${fmt(L.sma20)} / ${fmt(L.sma50)}`], ["SMA 100 / 200", `${fmt(L.sma100)} / ${fmt(L.sma200)}`],
    ["EMA 20 / 50", `${fmt(L.ema20)} / ${fmt(L.ema50)}`], ["Donchian 20", `${fmt(L.donchian20_lower)} – ${fmt(L.donchian20_upper)}`],
    ["Bollinger 20", `${fmt(L.bb20_lower)} – ${fmt(L.bb20_upper)}`], ["Z-score 20", fmt(L.z20)], ["Mean-rev z (10)", fmt(L.z10_meanrev)],
    ["RSI 14", fmt(L.rsi14, 1)], ["ATR 14", `${fmt(L.atr14)} (${fmt(L.atr14_pct)}%)`], ["Realized vol 20 / 60", `${fmt(L.rvol20, 1)}% / ${fmt(L.rvol60, 1)}%`],
    ...(isCrypto ? [] : [["RS vs SPY", fmt(L.rs_spy, 1)]])];
  if (tf === "1d") {
    rows.push(["TSMOM signal (0–1)", fmt(L.tsmom_signal)]);
    rows.push(["Exhaustion score (−1..1)", fmt(L.exhaustion, 2)]);
    rows.push(["Up / down streak (days)", `${fmt(L.up_streak, 0)} / ${fmt(L.down_streak, 0)}`]);
    rows.push(["RSI 2 / volume fade", `${fmt(L.rsi2, 1)} / ${L.vol_fade == null ? "–" : (Number(L.vol_fade) >= 0.5 ? "yes" : "no")}`]);
  }
  if (tf !== "1d") rows.push(["VWAP", fmt(L.vwap)]);
  $("ind").replaceChildren(...rows.map(([k, v]) => { const tr = document.createElement("tr"); tr.innerHTML = `<td>${esc(k)}</td><td>${esc(v)}</td>`; return tr; }));
  const badge = data.exhaustion, el = $("exh-badge");
  if (el) {
    if (tf === "1d" && badge && badge.kind && badge.kind !== "none") {
      el.hidden = false;
      el.className = `exh-badge ${badge.kind}`;
      el.textContent = badge.label;
    } else {
      el.hidden = true;
      el.textContent = "";
    }
  }
  if (isCrypto) { renderCryptoSide(); return; }
  const p = data.paper ?? {};
  const pos = (p.positions ?? []).map((x) => `${x.account}: ${x.shares} sh @ ${fmt(x.avg_cost)} since ${x.entry_date}`);
  const st = (p.stops ?? []).map((x) => `${x.account} target ${fmt((x.target_weight ?? 0) * 100, 1)}% (${x.sleeve ?? "?"})${x.stop ? `, stop ${fmt(x.stop)}` : ""}`);
  const fills = (p.fills ?? []).slice(-5).reverse().map((f) => `${f.fill_date} ${f.account} ${f.shares > 0 ? "bought" : "sold"} ${Math.abs(f.shares)} @ ${fmt(f.price)}`);
  const dts = (p.dayTrades ?? []).slice(0, 5).map((d) => `${d.session_date} DT ${d.shares} @ ${fmt(d.entry_price)} → ${fmt(d.exit_price)} (${d.reason}, ${fmt(d.pnl)})`);
  $("paper").innerHTML = [...pos, ...st, ...fills, ...dts].map(esc).join("<br>") || "No paper positions, targets or fills for this ticker.";
  const f = data.dataFlags ?? {};
  $("flags").innerHTML = [
    f.excluded ? "<b class='neg'>Excluded from strategies</b> (open/persisting price anomaly)." : "Included (no unresolved anomaly).",
    f.anomalyCount ? `${f.anomalyCount} anomaly record(s); latest: ${(f.anomalies ?? []).slice(-3).map((a) => `${esc(a.date)} ${esc(a.kind)} (${esc(a.status)})`).join(", ")}` : "",
    f.repairs?.length ? `Repaired ${f.repairs.length}× (last ${esc(f.repairs.at(-1).repaired_at)})` : "",
    f.masked?.length ? `Masked return days: ${f.masked.map(esc).join(", ")}` : "",
  ].filter(Boolean).join("<br>");
}

// ------------------------------------------------------------------ crypto side panels (/api/crypto/status)
function pctTxt(x) { return x == null || !Number.isFinite(Number(x)) ? "–" : `${x >= 0 ? "+" : ""}${fmt(x * 100, 1)}%`; }
function renderCryptoSide() {
  const accounts = (cryptoInfo?.accounts ?? []).filter((a) => Number(a.targetWeights?.[symbol] ?? a.weights?.[symbol] ?? 0) > 0);
  $("paper").innerHTML = accounts.length
    ? accounts.map((a) => esc(`#${a.id} ${a.label ?? ""}: target ${fmt((a.targetWeights?.[symbol] ?? a.weights?.[symbol]) * 100, 0)}% ${symbol}, equity $${fmt(a.equity, 0)} (${pctTxt(a.totalReturn)})`)).join("<br>")
      + "<br><span class='small'>Crypto accounts 200+ are paper simulations only.</span>"
    : `No crypto paper account (ids 200+) currently targets ${esc(symbol)}.`;
  $("flags").innerHTML = [`${esc(data?.source ?? "Coinbase candles")}.`, "Prices are Coinbase public candles (UTC days, 24/7); not exchange-consolidated.",
    data?.partialFrom ? `Bars from ${esc(data.partialFrom)} are still open.` : ""].filter(Boolean).join("<br>");
}
async function loadCrypto() {
  $("ai-flag").disabled = true;
  $("ask").placeholder = `Ask the AI about ${symbol} (crypto)…`;
  $("verdict").textContent = "Crypto: no stock-style AI verdict. Ask below — the analyst sees the crypto snapshot, the crypto paper accounts and the backtest verdict (CRYPTO_STATUS).";
  $("news").replaceChildren(Object.assign(document.createElement("li"), { textContent: "No crypto news feed; ask the analyst." }));
  try {
    const st = await getJson("/api/crypto/status");
    const m = st.market ?? {};
    const coin = (m.coins ?? []).find((c) => c.coin === symbol) ?? null;
    cryptoInfo = { accounts: st.accounts ?? [], coin, market: m };
    $("name").textContent = `${data?.name || symbol} · crypto · Coinbase ${symbol}-USD`;
    const kv = coin ? [["Price (snapshot)", `$${fmt(coin.price, coin.price < 10 ? 4 : 2)}`], ["As of", m.asOf ?? "–"],
      ["24h / 7d / 30d", `${pctTxt(coin.chg24h)} / ${pctTxt(coin.chg7d)} / ${pctTxt(coin.chg30d)}`],
      ["90d / 1y", `${pctTxt(coin.chg90d)} / ${pctTxt(coin.chg365d)}`],
      ["Vol 30d / 90d (ann.)", `${pctTxt(coin.vol30d)} / ${pctTxt(coin.vol90d)}`], ["Trend", coin.trend ?? "–"],
      ["vs 50d / 200d", `${pctTxt(coin.vsSma50)} / ${pctTxt(coin.vsSma200)}`], ["Drawdown from 1y high", pctTxt(coin.drawdownFrom1yHigh)],
      ["Schwab Crypto", coin.schwab ?? data?.schwab ?? "–"], ["Source", coin.source ?? "–"]] : [["Snapshot", "no row for this coin yet"]];
    $("fundamentals").classList.remove("muted");
    $("fundamentals").innerHTML = `<dl class="kv">${kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      <p class="small muted">Dollar/liquidity and nowcast context are in the analyst's crypto summary.</p>`;
    if (data) renderCryptoSide();
  } catch (error) {
    $("fundamentals").textContent = `Crypto snapshot unavailable: ${error.message}`;
  }
}

let verdictModel = null, verdictRunning = false;

function verdictMessage(message) {
  $("verdict-status").textContent = message;
  $("verdict-status").hidden = !message;
}

function updateVerdictControls() {
  $("verdict-refresh").hidden = !verdictModel?.ready;
  $("verdict-refresh").disabled = verdictRunning;
  $("ai-flag").disabled = verdictRunning || verdictModel?.ready === false;
}

async function loadVerdict() {
  let storedFallback = false;
  try {
    const [{ verdicts, modelVerdicts }, models] = await Promise.all([
      getJson(`/api/ai/verdicts?tickers=${encodeURIComponent(symbol)}&includeModel=1`),
      getJson("/api/ai/models").catch(() => null),
    ]);
    verdictModel = models?.current ?? null;
    const latest = verdicts?.[0] ?? null;
    verdict = latest?.source === "FALLBACK" ? modelVerdicts?.[0] ?? latest : latest;
    storedFallback = latest?.source === "FALLBACK" && verdict?.source === "LLM";
  } catch (error) {
    if (!verdict) $("verdict").textContent = "No saved AI verdict was loaded.";
    verdictMessage(`AI verdict unavailable: ${error.message}`);
    return;
  }
  updateVerdictControls();
  $("verdict-refresh").textContent = verdict?.source === "LLM" ? "Refresh AI verdict" : "Generate AI verdict";
  const needsKey = verdict?.source === "FALLBACK" && /MODEL_NO_KEY/.test(verdict.summary || "");
  if (storedFallback) verdictMessage("Showing the most recent model verdict. The later attempt produced a fallback.");
  else if (needsKey && verdictModel?.ready) verdictMessage(`${verdictModel.name} is ready. Generate a new verdict to replace this saved fallback.`);
  else if (verdict?.source === "FALLBACK" && verdictModel?.ready) verdictMessage(`${verdictModel.name} is ready. You can retry the verdict.`);
  else if (verdictModel?.ready === false) verdictMessage("Set up a model on the AI analyst page to generate a verdict.");
  else verdictMessage("");
  if (!verdict) { $("verdict").textContent = "No saved AI verdict for this ticker."; if (data) render(); return; }
  const v = verdict, e = v.entry ?? {};
  const summary = needsKey && verdictModel?.ready ? "This fallback was saved when the model had no active API key." : v.summary;
  $("verdict").classList.remove("muted");
  $("verdict").innerHTML = `<span class="flag ${esc(v.flag)}">${esc(v.flag)}</span> ${esc(v.action ?? "")} · conviction ${esc(v.conviction ?? "?")}
    <dl class="kv">
      <dt>Entry band</dt><dd>${fmt(e.bandLow)} – ${fmt(e.bandHigh)}${e.limitPrice ? ` (limit ${fmt(e.limitPrice)})` : ""}</dd>
      <dt>Stop</dt><dd>${fmt(v.stop?.price)}${v.stop?.atrMultiple ? ` (${fmt(v.stop.atrMultiple, 1)} ATR)` : ""}</dd>
      <dt>Target</dt><dd>${fmt(v.takeProfit?.price)}</dd>
      <dt>Overextension</dt><dd>${esc(v.overextension ?? "–")}</dd>
      <dt>Trend</dt><dd>${esc(v.trend ?? "–")}</dd><dt>Size hint</dt><dd>${esc(v.sizeHint ?? "–")}</dd>
      <dt>Fundamentals</dt><dd>${esc(v.fundamentalHealth ?? "–")}</dd><dt>Source</dt><dd>${esc(v.source ?? "–")}${v.model ? ` · ${esc(v.model)}` : ""} · ${esc(v.asof ?? v.createdAt ?? "")}</dd>
    </dl><p class="small">${esc(summary ?? "")}</p>
    <p class="small muted">Advisory only: the paper trader's risk engine stays final; a verdict can only reduce or defer an entry.</p>`;
  if (data) render();
}

async function aiFlag() {
  if (verdictRunning) return;
  verdictRunning = true;
  updateVerdictControls();
  verdictMessage("Generating an AI verdict…");
  try {
    const job = await getJson("/api/ai/flag", { tickers: [symbol] });
    for (let i = 0; i < 120; i += 1) {
      await new Promise((r) => setTimeout(r, 2000));
      const j = await getJson(`/api/ai/flag/${job.id}`);
      if (j.status === "failed") throw new Error(j.error);
      if (j.status === "complete") { await loadVerdict(); return; }
    }
    verdictMessage("The verdict is still running. Reload the chart to check for the result.");
  } catch (error) {
    verdictMessage(`AI verdict failed: ${error.message}`);
  } finally {
    verdictRunning = false;
    updateVerdictControls();
  }
}

async function ask(event) {
  event.preventDefault();
  const question = $("ask").value.trim();
  if (!question) return;
  $("answer").textContent = "Thinking…";
  try {
    const r = await getJson("/api/ai/ask", isCrypto ? { question: `${symbol} crypto (${data?.name || symbol}): ${question}`, tickers: [] }
      : { question, tickers: [symbol] });
    $("answer").textContent = [r.explanation ?? r.answer ?? r.text ?? "", ...(r.keyPoints ?? []).map((k) => `• ${k}`),
      r.source === "FALLBACK" ? "\n(Fallback answer: the AI model was unavailable.)" : ""].join("\n");
  } catch (error) {
    $("answer").textContent = `AI unavailable: ${error.message}`;
  }
}

async function loadResearch() {
  try {
    const r = await getJson(`/api/ai/research/${encodeURIComponent(symbol)}`);
    const d = r.details?.value ?? {};
    if (d.name) $("name").textContent = `${d.name}${d.exchange ? ` · ${d.exchange}` : ""}${d.industry ? ` · ${d.industry}` : ""}`;
    const fin = r.financials?.value ?? null;
    const kv = [["Type", d.type], ["Market cap", d.marketCap ? `$${fmt(d.marketCap / 1e9, 1)}B` : null], ["Shares out", d.sharesOutstanding ? fmt(d.sharesOutstanding, 0) : null],
      ["Employees", d.employees ? fmt(d.employees, 0) : null], ["Listed", d.listDate]];
    if (fin && typeof fin === "object") {
      // SEC facts: show scalars and {end, val} facts one level deep (e.g. derived.revenueTTM, latest.cash); skip raw series
      const label = (k) => k.replace(/([a-z])([A-Z])/g, "$1 $2").replace(/^./, (c) => c.toUpperCase());
      const show = (k, v) => {
        const value = v && typeof v === "object" && !Array.isArray(v) ? v.val ?? v.value : v;
        if (value == null || typeof value === "object") return;
        const money = typeof value === "number" && Math.abs(value) > 1e6 && !/growth|ratio|pct|margin/i.test(k);
        const pct = typeof value === "number" && /growth|margin|change|dilution/i.test(k) && Math.abs(value) < 10;
        kv.push([label(k), money ? `$${fmt(value / 1e6, 1)}M` : pct ? `${fmt(value * 100, 1)}%` : typeof value === "number" ? fmt(value) : value]);
      };
      for (const [k, v] of Object.entries(fin)) {
        if (v == null || Array.isArray(v) || k === "entity") continue;
        if (typeof v === "object" && v.val === undefined && v.value === undefined) {
          for (const [k2, v2] of Object.entries(v)) if (!Array.isArray(v2)) show(k2, v2);
        } else show(k, v);
      }
    }
    const divs = (r.dividends?.value ?? []).slice(0, 4).map((x) => `${x.exDate} $${x.cash}`).join(", ");
    if (divs) kv.push(["Dividends", divs]);
    const feat = r.features ?? {};
    const dilFlag = fin?.derived?.dilutionFlag;
    const dilChange = fin?.derived?.dilutedShareChangeYoY;
    const dilPct = dilChange != null && Number.isFinite(dilChange) ? `${(dilChange * 100).toFixed(1)}%` : null;
    const offer = fin?.derived?.dilutionOffering;
    const dilBanner = dilFlag ? `<p class="${dilFlag === "RED" || dilFlag === "OFFERING" ? "" : "small"}">${dilFlag === "RED" ? "RED FLAG" : dilFlag === "WARNING" ? "Warning" : "Offering flag"}: ${dilPct ? `split-adjusted share count is up ${dilPct} year over year` : "a recent equity offering was filed"}${offer ? ` (${esc(offer.form)} on ${esc(offer.filingDate)})` : ""}. New paper buys are blocked while this is active.</p>` : "";
    $("fundamentals").classList.remove("muted");
    $("fundamentals").innerHTML = `${dilBanner}<dl class="kv">${kv.filter(([, v]) => v != null && v !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
      ${d.description ? `<p class="small">${esc(d.description)}</p>` : ""}
      ${r.gaps?.length ? `<p class="small muted">Gaps: ${r.gaps.map(esc).join("; ")}</p>` : ""}
      ${feat.priceSource ? `<p class="small muted">AI price features from ${esc(feat.priceSource)}.</p>` : ""}`;
    const news = r.news?.value ?? [];
    $("news").replaceChildren(...(news.length ? news.slice(0, 25) : [null]).map((n) => {
      const li = document.createElement("li");
      if (!n) { li.textContent = "No recent news (needs MASSIVE_API_KEY)."; return li; }
      const safeUrl = /^https?:\/\//.test(n.url) ? n.url : null;
      li.innerHTML = `${safeUrl ? `<a href="${esc(safeUrl)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>` : esc(n.title)}
        <div class="meta">${esc(n.published?.slice(0, 16).replace("T", " "))} UTC · ${esc(n.publisher)}${n.pressRelease ? " · press release" : ""}${n.sentiment ? ` · ${esc(n.sentiment)}` : ""}</div>`;
      return li;
    }));
  } catch (error) {
    $("fundamentals").textContent = `Fundamentals unavailable: ${error.message}`;
  }
}

// ------------------------------------------------------------------ wiring
const openTicket = (side) => window.LiveTrading ? window.LiveTrading.openTicket(symbol, side)
  : alert("The order ticket is still loading; try again.");
if (!isCrypto) {   // crypto: paper only, no order ticket, no stock verdict
  $("buy").onclick = () => openTicket("BUY");
  $("sell").onclick = () => openTicket("SELL");
  $("ai-flag").onclick = aiFlag;
  $("verdict-refresh").onclick = aiFlag;
}
$("ask-form").addEventListener("submit", ask);
$("live").onchange = pollQuote;
window.addEventListener("schwab-status", (e) => { schwab = { ...e.detail }; banner(); });
buildControls();
load().then(() => { pollQuote(); scheduleBarRefresh(); maybeFetchHistory(); });
if (isCrypto) loadCrypto();
else { loadVerdict(); loadResearch(); }
