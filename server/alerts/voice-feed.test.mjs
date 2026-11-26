// Decision voice feed, alert categories, prefs and the analyst dock helpers. No real orders, no dashboard restart.
// Run from server/:  node --test alerts/voice-feed.test.mjs
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, utimesSync } from "node:fs";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { accountGroup, categorize, CATEGORY_IDS } from "./categories.js";
import { chatToAlerts, cryptoStatusToAlerts, createVoiceFeed, nowcastToAlerts, regimeToAlert, spokenTrade, swarmStatusToAlerts, tradeParts } from "./voice-feed.js";
import { cleanPrefs, emitAlert } from "./index.js";
import { normalizeType } from "./phrase.js";

const here = dirname(fileURLToPath(import.meta.url));
const publicDir = join(here, "..", "..", "public");
const require = createRequire(import.meta.url);
const VS = require(join(publicDir, "voice-summary.js"));

test("account groups and categories", () => {
  assert.equal(accountGroup("default"), "trade_main");
  assert.equal(accountGroup("daytrade"), "trade_main");
  assert.equal(accountGroup("options-paper"), "trade_options");
  assert.equal(accountGroup("crypto-203"), "trade_crypto");
  assert.equal(accountGroup("203"), "trade_crypto");
  assert.equal(accountGroup("swarm-101"), "trade_swarm");
  assert.equal(accountGroup(112), "trade_swarm");
  assert.equal(categorize({ type: "black_swan", meta: { eventType: "dollar_shock" } }), "dollar_shock");
  assert.equal(categorize({ type: "black_swan", meta: { state: "SHOCK" } }), "black_swan");
  assert.equal(categorize({ type: "fill", meta: { account: "options-paper" } }), "trade_options");
  assert.equal(categorize({ type: "macro_nowcast" }), "macro");
  assert.equal(categorize({ type: "regime" }), "regime");
  assert.equal(categorize({ type: "schwab_auth" }), "system");
  assert.equal(categorize({ type: "decision", meta: { category: "analyst" } }), "analyst");
  assert.ok(CATEGORY_IDS.length >= 10);
  assert.equal(normalizeType("regime_change"), "regime");
  assert.equal(normalizeType("nowcast"), "macro_nowcast");
});

test("emitted alerts carry a category (dollar shock alias included)", async () => {
  const a = await emitAlert({ type: "dollar_shock", text: "Dollar up 2 sigma in 3 days." });
  assert.equal(a.ok, true);
  assert.equal(a.alert.category, "dollar_shock");
  const b = await emitAlert({ type: "fill", ticker: "BTC", account: "crypto-203", meta: { side: "buy", qty: 0.01, price: 90000 } });
  assert.equal(b.alert.category, "trade_crypto");
});

test("trade text parsing and spoken phrasing", () => {
  const m = { kind: "order", account: "default", text: "ORDER BUY 10 XLV @ $183.68 [tsmom sleeve, entry] target 18.7% (was 0.0%)." };
  assert.deepEqual(tradeParts(m), { side: "buy", qty: 10, ticker: "XLV", price: 183.68 });
  const said = spokenTrade(m);
  assert.match(said, /^Main paper: order to buy 10 XLV at 183 dollars 68\.$/);
  assert.match(spokenTrade({ kind: "stop", account: "options-paper", ticker: "SPY", price: 501.2 }), /Options paper: stopped out of SPY/);
  assert.doesNotMatch(said, /\[|\$/);
});

test("chat rows -> alerts: singles, bursts, skipped kinds, breaker", () => {
  const singles = chatToAlerts([
    { id: 1, role: "trader", kind: "fill", account: "default", ticker: "IEF", side: "buy", qty: 30, price: 92.06, text: "FILL BUY 30 IEF @ $92.06" },
    { id: 2, role: "trader", kind: "skip", account: "default", text: "skipped" },
    { id: 3, role: "trader", kind: "black_swan", text: "Watch" },
    { id: 4, role: "user", kind: "followup_q", text: "why?" },
  ]);
  assert.equal(singles.length, 1);
  assert.equal(singles[0].type, "fill");
  assert.equal(singles[0].meta.category, "trade_main");
  const burst = chatToAlerts([1, 2, 3, 4, 5, 6].map((i) => ({ id: i, role: "trader", kind: "order", account: "default", text: `ORDER BUY ${i} T${i} @ $10.00` })));
  assert.equal(burst.length, 1);
  assert.match(burst[0].speech, /Main paper logged 6 orders: T1, T2, T3, T4 and 2 more\./);
  const breaker = chatToAlerts([{ id: 9, role: "trader", kind: "run", account: "daytrade", text: "Circuit breaker tripped: daily loss limit." }]);
  assert.equal(breaker[0].type, "circuit_breaker");
  const quietRun = chatToAlerts([{ id: 10, role: "trader", kind: "run", account: "default", text: "Swing paper run #2: processed 0 session(s)." }]);
  assert.equal(quietRun.length, 0);
});

test("crypto status: position changes and standings", () => {
  const prev = { updatedAt: "a", days: 1, leader: 203, btcBaselineReturn: -0.01, accounts: [{ id: 203, label: "BTC trend", targetWeights: {}, totalReturn: 0 }, { id: 207, label: "x", targetWeights: { BTC: 0.5 }, totalReturn: 0 }] };
  const cur = { updatedAt: "b", days: 2, leader: 207, btcBaselineReturn: 0.02, accounts: [{ id: 203, label: "BTC trend", targetWeights: { BTC: 1 }, totalReturn: 0.01 }, { id: 207, label: "x", targetWeights: { BTC: 0.5 }, totalReturn: 0.03 }] };
  const out = cryptoStatusToAlerts(prev, cur);
  assert.equal(out.length, 2);
  assert.equal(out[0].meta.category, "trade_crypto");
  assert.match(out[0].speech, /account 203 moved to 100% BTC/);
  assert.match(out[1].speech, /Leader is account 207/);
  assert.deepEqual(cryptoStatusToAlerts(prev, prev), []);
  assert.deepEqual(cryptoStatusToAlerts(null, cur), []);
});

test("swarm, nowcast and regime alerts", () => {
  const sw = swarmStatusToAlerts({ forward: { updatedAt: "1", leader: 20, sessions: 1 } }, { forward: { updatedAt: "2", leader: 7, sessions: 2, end: "2026-11-26" } });
  assert.equal(sw[0].meta.category, "trade_swarm");
  assert.match(sw[0].speech, /new leader account 7/);
  const p = { generatedAt: "1", nowcasts: [{ target: "CPI_MOM", label: "CPI (headline) m/m %", period: "2026-11", nowcast: 0.2, rmseHistory: 0.18, lastPrint: 0.35, impliedSurpriseVsLastPrint: -0.15 }] };
  const c = { generatedAt: "2", nowcasts: [{ target: "CPI_MOM", label: "CPI (headline) m/m %", period: "2026-11", nowcast: -0.1, rmseHistory: 0.18, lastPrint: 0.35, impliedSurpriseVsLastPrint: -0.45 }] };
  const nc = nowcastToAlerts(p, c);
  assert.equal(nc.length, 1);
  assert.equal(nc[0].type, "macro_nowcast");
  assert.equal(nc[0].meta.big, true);
  assert.match(nc[0].speech, /down from 0\.2/);
  const small = nowcastToAlerts(p, { generatedAt: "3", nowcasts: [{ ...p.nowcasts[0], nowcast: 0.25 }] });
  assert.equal(small.length, 0);
  assert.equal(regimeToAlert(null, "UPTREND"), null);
  assert.equal(regimeToAlert("UPTREND", "UPTREND"), null);
  const r = regimeToAlert("UPTREND", "DOWNTREND_HIGH_VOL");
  assert.equal(r.type, "regime");
  assert.equal(r.severity, "warning");
  assert.match(r.speech, /downtrend high vol/);
});

test("voice feed never replays history and announces only new events", async () => {
  const dataDir = mkdtempSync(join(tmpdir(), "tm-feed-"));
  try {
    const cryptoPath = join(dataDir, "Research", "crypto", "paper");
    mkdirSync(cryptoPath, { recursive: true });
    const file = join(cryptoPath, "crypto_paper_status.json");
    writeFileSync(file, JSON.stringify({ updatedAt: "a", days: 1, leader: 203, accounts: [{ id: 203, label: "BTC", targetWeights: {} }] }));
    let rows = [{ id: 5, role: "trader", kind: "order", account: "default", text: "ORDER BUY 1 SPY @ $500.00" }];
    const listChat = async ({ afterId = 0 } = {}) => ({ messages: rows.filter((m) => m.id > afterId).slice(-200) });
    let regime = "UPTREND";
    const sent = [];
    const feed = createVoiceFeed({ emit: async (a) => sent.push(a), listChat, getRegime: () => ({ regime }), dataDir, env: {} });
    await feed.poll();
    assert.equal(sent.length, 0, "startup state is a baseline, not news");
    rows = rows.concat([{ id: 6, role: "trader", kind: "fill", account: "options-paper", ticker: "SPY", side: "buy", qty: 1, price: 4.2, text: "FILL" }]);
    writeFileSync(file, JSON.stringify({ updatedAt: "b", days: 2, leader: 203, accounts: [{ id: 203, label: "BTC", targetWeights: { BTC: 1 } }] }));
    const t = new Date(Date.now() + 5000); utimesSync(file, t, t);
    await feed.poll();
    assert.ok(sent.some((a) => a.meta.category === "trade_options" && a.type === "fill"));
    assert.ok(sent.some((a) => a.meta.category === "trade_crypto"));
    const off = createVoiceFeed({ emit: async () => {}, listChat, dataDir, env: { TM_VOICE_FEED: "off" } });
    assert.equal(off.enabled, false);
  } finally {
    rmSync(dataDir, { recursive: true, force: true });
  }
});

test("prefs are whitelisted", () => {
  const c = cleanPrefs({ masterMute: true, volume: 9, engine: "browser", speakCats: { analyst: false, bogus: true }, evil: "<script>", quietStart: "22:00", quietEnd: "x" });
  assert.deepEqual(c, { masterMute: true, volume: 1.5, engine: "browser", speakCats: { analyst: false }, quietStart: "22:00" });
  assert.equal(cleanPrefs(null), null);
  assert.equal(cleanPrefs([1]), null);
});

test("analyst replies become short spoken summaries, not tables or long text", () => {
  const long = "## Outlook\n**Nvidia** is in a downtrend, 24% below its 50-day average. Cash is strong at $26.3 billion. " +
    "Momentum is weak. The flag is yellow. I'd wait for a better entry near the 200-day line. More detail follows here. " +
    "\n| col | col |\n|---|---|\n| 1 | 2 |\n- bullet one\n- bullet two\nSee https://example.com for more.";
  const s = VS.summarizeReply({ kind: "answer", answer: long, stance: "BEARISH", conviction: "MEDIUM" });
  assert.ok(s.length <= VS.MAX, String(s.length));
  assert.match(s, /^bearish, medium conviction\./);
  assert.match(s, /24 percent/);
  assert.doesNotMatch(s, /\||##|\*\*|https?:/);
  assert.equal(VS.summarizeReply({ kind: "help", text: "commands..." }), "");
  assert.match(VS.summarizeReply({ kind: "decision", decisions: [{ ticker: "IEF", usable: true, flag: "GREEN", action: "BUY" }] }), /IEF green buy/);
  assert.match(VS.summarizeReply({ kind: "decision", decisions: [{ ticker: "IEF", usable: false }] }), /No decision changes/);
});

test("dock and alert scripts are wired in and never touch order or preview endpoints", () => {
  const dock = readFileSync(join(publicDir, "analyst-dock.js"), "utf8");
  const alerts = readFileSync(join(publicDir, "alerts", "alerts.js"), "utf8");
  const nav = readFileSync(join(publicDir, "nav.js"), "utf8");
  const feed = readFileSync(join(here, "voice-feed.js"), "utf8");
  assert.match(nav, /\/analyst-dock\.js/);
  assert.match(dock, /\/api\/ai\/"\s*\+\s*path/);
  assert.match(dock, /api\("command"/);
  assert.match(dock, /localStorage/);
  for (const src of [dock, alerts, feed]) {
    assert.doesNotMatch(src, /\/orders|previewOrder|order\/preview|\/api\/live\/(order|preview)|\/api\/options\/live/i);
  }
  // voice defaults on, with per-category switches and a gesture unlock hint
  assert.match(alerts, /masterMute: false/);
  assert.match(alerts, /Click anywhere to enable voice/);
  for (const id of CATEGORY_IDS) assert.ok(alerts.includes(`"${id}"`), `alerts.js lists category ${id}`);
});
