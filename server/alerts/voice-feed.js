// Decision voice feed: turns what the paper traders and research jobs write (decision chat log, crypto paper status,
// swarm forward status, macro nowcast, market regime) into alerts on the existing /api/alerts stream, so the browser
// can show and speak them per category. Read-only: it never places, previews or changes anything.
// TM_VOICE_FEED=off disables it. History present at startup is never replayed (only changes after start are announced).
import { existsSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { accountGroup } from "./categories.js";

export const GROUP_LABEL = Object.freeze({
  trade_main: "Main paper",
  trade_swarm: "Swarm shadow",
  trade_crypto: "Crypto paper",
  trade_options: "Options paper",
});

// Chat kinds that are announced. Black swan, drift and macro entries are skipped because their engines
// already emit their own alerts; skips/watches/config/follow-ups are too chatty to speak.
const TRADE_KINDS = new Set(["order", "fill", "entry", "exit", "stop", "risk"]);

export function spokenPrice(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n <= 0) return "";
  if (n >= 1000) return `${Math.round(n).toLocaleString("en-US")} dollars`;
  const dollars = Math.floor(n), cents = Math.round((n - dollars) * 100);
  return cents ? `${dollars} dollars ${cents}` : `${dollars} dollars`;
}

function spokenQty(q) {
  const n = Number(q);
  if (!Number.isFinite(n) || n === 0) return "";
  return Math.abs(n) >= 1 ? String(Math.round(Math.abs(n) * 1000) / 1000) : String(Math.round(Math.abs(n) * 10000) / 10000);
}

// Pull side/qty/ticker/price out of a chat entry, falling back to the "ORDER BUY 10 XLV @ $183.68" text form.
export function tradeParts(msg = {}) {
  const text = String(msg.text || "");
  const m = text.match(/\b(BUY|SELL|SHORT|COVER|ADJUST)\s+([\d.]+)\s+([A-Z][A-Z0-9.\-/]{0,11})\s*@\s*\$?([\d,.]+)?/i);
  const side = String(msg.side || (m && m[1]) || "").toLowerCase();
  const qty = msg.qty ?? (m ? Number(m[2]) : null);
  const ticker = String(msg.ticker || (m && m[3]) || "").toUpperCase();
  const price = msg.price ?? (m && m[4] ? Number(String(m[4]).replace(/,/g, "")) : null);
  return { side, qty, ticker, price };
}

const VERB = {
  order: { buy: "order to buy", sell: "order to sell", short: "order to short", cover: "order to cover" },
  fill: { buy: "bought", sell: "sold", short: "shorted", cover: "covered" },
  entry: { buy: "opened a long in", sell: "opened a short in", short: "opened a short in" },
  exit: { buy: "closed", sell: "closed", short: "covered", cover: "covered" },
};

export function spokenTrade(msg = {}) {
  const group = accountGroup(msg.account);
  const who = GROUP_LABEL[group] || "Paper";
  const { side, qty, ticker, price } = tradeParts(msg);
  const q = spokenQty(qty), px = spokenPrice(price);
  const kind = String(msg.kind || "");
  if (kind === "stop") return `${who}: stopped out of ${ticker || "a position"}${px ? ` at ${px}` : ""}.`;
  if (kind === "risk") return `${who}: risk change${ticker ? ` on ${ticker}` : ""}. ${clip(firstSentence(msg.text), 120)}`.trim();
  const verb = (VERB[kind] && VERB[kind][side]) || (kind === "exit" ? "closed" : kind === "entry" ? "opened" : side || kind);
  const amount = q && kind !== "exit" ? `${q} ` : "";
  return `${who}: ${verb} ${amount}${ticker || ""}${px ? ` at ${px}` : ""}.`.replace(/\s+/g, " ").replace(" .", ".");
}

function firstSentence(text) {
  const s = String(text || "").replace(/\[[^\]]*\]/g, "").replace(/\s+/g, " ").trim();
  const m = s.match(/^(.{20,}?[.!?])\s/);
  return m ? m[1] : s;
}
function clip(s, n) { s = String(s || ""); return s.length <= n ? s : s.slice(0, n - 1).replace(/\s+\S*$/, "") + "."; }

function isBreakerRun(msg) {
  return msg.kind === "run" && /circuit|breaker|halt|kill switch/i.test(`${msg.text || ""} ${msg.rationale || ""}`);
}

// New chat rows -> alert payloads. Bursts (more than 3 trades from one account group in one poll) collapse
// into one summary so the voice doesn't read six orders in a row; the visual text still lists them all.
export function chatToAlerts(messages = []) {
  const out = [], groups = new Map();
  for (const msg of messages) {
    if (!msg || msg.role === "user" || String(msg.kind || "").startsWith("followup")) continue;
    if (isBreakerRun(msg)) {
      out.push({ type: "circuit_breaker", severity: "critical", text: clip(firstSentence(msg.text), 300),
        meta: { account: msg.account || "default", chatId: msg.id, kind: msg.kind }, href: "/paper2.html" });
      continue;
    }
    if (!TRADE_KINDS.has(msg.kind)) continue;
    const g = accountGroup(msg.account);
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(msg);
  }
  for (const [group, list] of groups) {
    if (list.length > 3) {
      const tickers = [...new Set(list.map((m) => tradeParts(m).ticker).filter(Boolean))];
      const named = tickers.slice(0, 4).join(", ") + (tickers.length > 4 ? ` and ${tickers.length - 4} more` : "");
      const kinds = [...new Set(list.map((m) => m.kind))];
      const noun = kinds.length === 1 ? ({ order: "orders", fill: "fills", entry: "entries", exit: "exits", stop: "stops", risk: "risk changes" })[kinds[0]] : "trade decisions";
      out.push({
        type: list.some((m) => m.kind === "stop") ? "stop_hit" : "decision",
        title: `${GROUP_LABEL[group]}: ${list.length} ${noun}`,
        text: list.map((m) => String(m.text || "").replace(/\s+/g, " ").slice(0, 160)).join(" | ").slice(0, 1000),
        speech: `${GROUP_LABEL[group]} logged ${list.length} ${noun}${named ? `: ${named}` : ""}.`,
        meta: { account: list[0].account || "default", category: group, chatIds: list.map((m) => m.id), burst: true },
        href: "/paper2.html",
      });
      continue;
    }
    for (const msg of list) {
      const parts = tradeParts(msg);
      out.push({
        type: msg.kind === "stop" ? "stop_hit" : msg.kind === "fill" ? "fill" : "decision",
        ticker: parts.ticker,
        title: `${GROUP_LABEL[group]}: ${msg.kind}`,
        text: String(msg.text || "").replace(/\s+/g, " ").slice(0, 600),
        speech: spokenTrade(msg),
        meta: { account: msg.account || "default", category: group, chatId: msg.id, kind: msg.kind,
          side: parts.side, qty: parts.qty, price: parts.price },
        href: "/paper2.html",
      });
    }
  }
  return out;
}

function pct(x, digits = 1) {
  const n = Number(x);
  if (!Number.isFinite(n)) return "n/a";
  const v = (n * 100).toFixed(digits);
  return `${n >= 0 ? "plus " : "minus "}${Math.abs(Number(v))} percent`;
}
function weightsText(w = {}) {
  const parts = Object.entries(w || {}).filter(([, v]) => Number(v) > 0.0005)
    .sort((a, b) => b[1] - a[1]).map(([k, v]) => `${Math.round(Number(v) * 100)}% ${k}`);
  return parts.length ? parts.join(", ") : "all cash";
}

// Crypto paper status -> alerts: position (target weight) changes per account, plus a daily standings line.
export function cryptoStatusToAlerts(prev, cur) {
  if (!cur || !Array.isArray(cur.accounts)) return [];
  if (!prev || prev.updatedAt === cur.updatedAt) return [];
  const before = new Map((prev.accounts || []).map((a) => [a.id, a]));
  const changed = [];
  for (const a of cur.accounts) {
    const p = before.get(a.id);
    if (!p) continue;
    if (JSON.stringify(roundW(p.targetWeights)) !== JSON.stringify(roundW(a.targetWeights))) changed.push(a);
  }
  const out = [];
  if (changed.length) {
    const lines = changed.map((a) => `Account ${a.id} (${a.label}): ${weightsText(before.get(a.id).targetWeights)} -> ${weightsText(a.targetWeights)}`);
    const first = changed[0];
    out.push({
      type: "decision", title: `Crypto paper: ${changed.length} position change${changed.length > 1 ? "s" : ""}`,
      text: lines.join(" | ").slice(0, 1000),
      speech: changed.length === 1
        ? `Crypto paper account ${first.id} moved to ${weightsText(first.targetWeights)}.`
        : `Crypto paper: ${changed.length} accounts changed positions. Account ${first.id} is now ${weightsText(first.targetWeights)}.`,
      meta: { category: "trade_crypto", account: `crypto-${first.id}`, accounts: changed.map((a) => a.id) },
      href: "/paper2.html",
    });
  }
  if (cur.leader != null && (prev.days !== cur.days || prev.leader !== cur.leader)) {
    const lead = cur.accounts.find((a) => a.id === cur.leader);
    out.push({
      type: "decision", severity: "info", title: "Crypto paper standings",
      text: `Day ${cur.days ?? "?"}: leader account ${cur.leader}${lead ? ` (${lead.label}) ${pct(lead.totalReturn, 2)}` : ""}; BTC buy-and-hold ${pct(cur.btcBaselineReturn, 2)}.`,
      speech: `Crypto paper, day ${cur.days ?? "unknown"}. Leader is account ${cur.leader}${lead ? `, ${pct(lead.totalReturn)}` : ""}. Bitcoin buy and hold is ${pct(cur.btcBaselineReturn)}.`,
      meta: { category: "trade_crypto", account: `crypto-${cur.leader}`, standings: true },
      href: "/paper2.html",
    });
  }
  return out;
}
function roundW(w) {
  return Object.fromEntries(Object.entries(w || {}).filter(([, v]) => Math.abs(Number(v)) > 0.005)
    .map(([k, v]) => [k, Math.round(Number(v) * 100)]).sort());
}

export function swarmStatusToAlerts(prev, cur) {
  const a = prev?.forward, b = cur?.forward;
  if (!b || !a || a.updatedAt === b.updatedAt) return [];
  const leaderChanged = a.leader !== b.leader;
  return [{
    type: "decision", severity: "info", title: "Swarm shadow update",
    text: `Swarm shadow accounts updated through ${b.end}: ${b.sessions} forward session(s), leader account ${b.leader}${leaderChanged ? ` (was ${a.leader})` : ""}.`,
    speech: `Swarm shadow accounts updated. ${b.sessions} forward sessions, ${leaderChanged ? `new leader account ${b.leader}` : `leader is still account ${b.leader}`}.`,
    meta: { category: "trade_swarm", account: "swarm", leader: b.leader },
    href: "/paper2.html",
  }];
}

function spokenNum(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "unknown";
  return `${n < 0 ? "minus " : ""}${Math.abs(Math.round(n * 100) / 100)}`;
}

// Nowcast -> alerts when an estimate for the same target/period moves by at least 0.75 of its historical
// error, or when the implied surprise versus the last print first exceeds 2x that error.
export function nowcastToAlerts(prev, cur) {
  if (!prev || !cur || prev.generatedAt === cur.generatedAt) return [];
  const before = new Map((prev.nowcasts || []).map((n) => [`${n.target}|${n.period}`, n]));
  const out = [];
  for (const n of cur.nowcasts || []) {
    const p = before.get(`${n.target}|${n.period}`);
    const err = Number(n.rmseHistory);
    if (!p || !Number.isFinite(err) || err <= 0) continue;
    const move = Number(n.nowcast) - Number(p.nowcast);
    const surprise = Number(n.impliedSurpriseVsLastPrint);
    const bigMove = Number.isFinite(move) && Math.abs(move) >= 0.75 * err;
    const bigSurprise = Number.isFinite(surprise) && Math.abs(surprise) >= 2 * err &&
      !(Math.abs(Number(p.impliedSurpriseVsLastPrint)) >= 2 * err);
    if (!bigMove && !bigSurprise) continue;
    const label = String(n.label || n.target).replace(/\s*m\/m\s*%?/i, " month over month").replace(/%/g, " percent");
    out.push({
      type: "macro_nowcast", severity: bigSurprise ? "warning" : "info",
      title: `Nowcast: ${n.label || n.target} (${n.period})`,
      text: `${n.label || n.target} ${n.period}: nowcast ${n.nowcast} (was ${p.nowcast}); last print ${n.lastPrint}; implied surprise ${n.impliedSurpriseVsLastPrint}.`,
      speech: `My nowcast for ${label} is now ${spokenNum(n.nowcast)}, ${move >= 0 ? "up" : "down"} from ${spokenNum(p.nowcast)}.` +
        (bigSurprise ? ` That's a big gap versus the last print of ${spokenNum(n.lastPrint)}.` : ""),
      meta: { category: "macro", target: n.target, period: n.period, nowcast: n.nowcast, previous: p.nowcast, big: bigSurprise },
      href: "/paper2.html",
    });
  }
  return out;
}

export function regimeLabel(r) {
  if (!r) return null;
  const spy = r.regime ? String(r.regime) : null;
  const h = r.hmm && (r.hmm.label || r.hmm.state || r.hmm.regime);
  return spy || (typeof h === "string" ? h : null);
}

export function regimeToAlert(prevLabel, curLabel) {
  if (!prevLabel || !curLabel || prevLabel === curLabel) return null;
  const words = (s) => String(s).replace(/_/g, " ").toLowerCase();
  return {
    type: "regime", severity: /down/i.test(curLabel) ? "warning" : "info", title: "Market regime change",
    text: `Market regime changed from ${prevLabel} to ${curLabel}.`,
    speech: `Heads up: the market regime changed from ${words(prevLabel)} to ${words(curLabel)}.`,
    meta: { category: "regime", from: prevLabel, to: curLabel },
    href: "/paper2.html",
  };
}

function readJsonIfChanged(path, cache) {
  try {
    if (!existsSync(path)) return { changed: false, value: cache.value };
    const m = statSync(path).mtimeMs;
    if (m === cache.mtime) return { changed: false, value: cache.value };
    const value = JSON.parse(readFileSync(path, "utf8"));
    cache.mtime = m;
    const prev = cache.value;
    cache.value = value;
    return { changed: true, prev, value };
  } catch {
    return { changed: false, value: cache.value }; // half-written file: try next poll
  }
}

export function createVoiceFeed({ emit, listChat, getRegime, dataDir, intervalMs = 20_000, log = () => {}, env = process.env } = {}) {
  const enabled = String(env.TM_VOICE_FEED || "").toLowerCase() !== "off";
  const files = {
    crypto: { path: join(dataDir || ".", "Research", "crypto", "paper", "crypto_paper_status.json"), toAlerts: cryptoStatusToAlerts },
    swarm: { path: join(dataDir || ".", "Research", "swarm", "swarm_status.json"), toAlerts: swarmStatusToAlerts },
    nowcast: { path: join(dataDir || ".", "Research", "macro", "nowcast_latest.json"), toAlerts: nowcastToAlerts },
  };
  const caches = Object.fromEntries(Object.keys(files).map((k) => [k, { mtime: null, value: null }]));
  let lastChatId = null, lastRegime = null, timer = null, busy = false, primed = false, lastRegimeCheck = 0;

  async function send(list, source) {
    for (const a of list) {
      try { await emit(a); } catch (error) { log("warn", "voice feed emit failed", { source, error: error.message }); }
    }
  }

  async function poll() {
    if (busy) return;
    busy = true;
    try {
      // decision chat log
      if (typeof listChat === "function") {
        try {
          if (lastChatId == null) {
            const { messages } = await listChat({ limit: 1 });
            lastChatId = messages?.length ? Number(messages[messages.length - 1].id) : 0;
          } else {
            const { messages } = await listChat({ afterId: lastChatId, limit: 200 });
            if (messages?.length) {
              lastChatId = Math.max(lastChatId, ...messages.map((m) => Number(m.id) || 0));
              await send(chatToAlerts(messages), "chat");
            }
          }
        } catch (error) { log("warn", "voice feed chat poll failed", { error: error.message }); }
      }
      // status files (first read only primes the cache)
      for (const [name, f] of Object.entries(files)) {
        const r = readJsonIfChanged(f.path, caches[name]);
        if (r.changed && primed && r.prev) await send(f.toAlerts(r.prev, r.value), name);
      }
      // regime
      if (typeof getRegime === "function" && Date.now() - lastRegimeCheck >= 120_000) {
        lastRegimeCheck = Date.now();
        try {
          const label = regimeLabel(await getRegime());
          const alert = regimeToAlert(lastRegime, label);
          if (label) lastRegime = label;
          if (alert && primed) await send([alert], "regime");
        } catch (error) { log("warn", "voice feed regime check failed", { error: error.message }); }
      }
      primed = true;
    } finally {
      busy = false;
    }
  }

  function start() {
    if (!enabled || timer) return api;
    poll().catch(() => {});
    timer = setInterval(() => poll().catch(() => {}), Math.max(2000, intervalMs));
    timer.unref?.();
    log("info", "decision voice feed started", { intervalMs });
    return api;
  }
  function stop() { if (timer) clearInterval(timer); timer = null; }
  const api = { start, stop, poll, get enabled() { return enabled; }, _state: () => ({ lastChatId, lastRegime, primed }) };
  return api;
}
