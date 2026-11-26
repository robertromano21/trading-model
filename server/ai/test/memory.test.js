// Analyst memory tests with a deterministic mock embedder (no model download). Real MiniLM: memory.smoke.test.js.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { createMemory, memoryHash } from "../memory/memory.js";
import { passages, lexical } from "../memory/lexical.js";
import { createAdvisor } from "../advisor.js";
import { openStore } from "../store.js";
import { createModelService } from "../models.js";

const ROOT = fileURLToPath(new URL("../../../", import.meta.url));
const VENV_PY = join(ROOT, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
const PY = process.env.TRADING_PYTHON || (existsSync(VENV_PY) ? VENV_PY : process.platform === "win32" ? "python" : "python3");
const tmp = () => mkdtempSync(join(tmpdir(), "tm-mem-"));
// Hashed bag-of-words "embedding": shared words -> high cosine. Deterministic and fast.
function mockEmbed(texts) {
  return texts.map((t) => {
    const v = new Array(384).fill(0);
    for (const w of String(t).toLowerCase().match(/[a-z]{3,}/g) || []) {
      let h = 0; for (const c of w) h = (h * 31 + c.charCodeAt(0)) >>> 0;
      v[h % 384] = 1;
    }
    v[383] += 0.01;
    return v;
  });
}
const DAY = 86400000;
const NOW = Date.parse("2026-11-25T20:00:00Z");
const mem = (opts = {}) => createMemory({ dataDir: tmp(), embed: mockEmbed, now: () => NOW, autoEmbed: false, ...opts });

test("ingest validates, dedupes, and matches the Python hash", async () => {
  const m = mem();
  const a = m.ingest({ kind: "note", ticker: "nvda", text: "Watch  the data center guide", ts: "2026-11-20T15:00:00Z" });
  const b = m.ingest({ kind: "NOTE", ticker: "NVDA", text: "Watch the data center guide" });
  assert.equal(a.duplicate, false); assert.equal(b.duplicate, true); assert.equal(a.id, b.id);
  assert.throws(() => m.ingest({ kind: "BOGUS", text: "x" }), /Unknown memory kind/);
  assert.throws(() => m.ingest({ kind: "NOTE", text: "  " }), /empty/);
  const py = execFileSync(PY, ["-c", "import sys; sys.path.insert(0, sys.argv[1]); from memory_ingest import memory_hash; print(memory_hash('NOTE','NVDA','Watch  the data center guide'))",
    fileURLToPath(new URL("..", import.meta.url))]).toString().trim();
  assert.equal(py, memoryHash("NOTE", "NVDA", "Watch  the data center guide"));
});

test("embedPending + semantic recall ranks relevant items, filters tickers, shows outcomes", async () => {
  const m = mem();
  m.ingest({ kind: "VERDICT", ticker: "NVDA", ts: NOW - 7 * DAY, text: "Flagged YELLOW, action WATCH: data center demand strong but stock extended.", meta: { refPrice: 160 } });
  m.ingest({ kind: "NEWS", ticker: "NVDA", ts: NOW - 2 * DAY, text: "Nvidia data center revenue beats estimates" });
  m.ingest({ kind: "NEWS", ticker: "XOM", ts: NOW - 1 * DAY, text: "Oil prices fall on supply glut" });
  m.ingest({ kind: "NOTE", ticker: null, ts: NOW - 30 * DAY, text: "Avoid adding to semis before CPI." });
  assert.equal((await m.embedPending()).embedded, 4);
  const r = await m.recall({ query: "NVDA data center demand outlook", tickers: ["NVDA"], prices: { NVDA: 155 } });
  assert.equal(r.method, "semantic");
  assert.match(r.text, /NVDA track record \(latest first\):\n- \d{4}-\d\d-\d\d 160\.00 -> 155\.00 \(-3\.1%\): Flagged YELLOW, action WATCH: data center demand strong but stock extended\./);
  assert.ok(!r.text.includes("Oil prices"), "other tickers are filtered out");
  assert.ok(r.items.length >= 2);
});

test("fixed budget: prompt size does not grow with history; near-duplicates suppressed", async () => {
  const m = mem();
  for (let i = 0; i < 400; i++) m.ingest({ kind: "NEWS", ticker: "NVDA", ts: NOW - i * 3600000, text: `Nvidia story number ${i} about data center GPUs and AI demand ${"detail ".repeat(40)}` });
  for (let i = 0; i < 5; i++) m.ingest({ kind: "QA", ticker: "NVDA", ts: NOW - i * 60000, text: `Q: What about Nvidia ${i}?\nA: ${"long answer ".repeat(300)}` });
  await m.embedPending(10000);
  const r = await m.recall({ query: "Nvidia data center demand", tickers: ["NVDA"], budgetChars: 3000 });
  assert.ok(r.chars <= 3000 + 300, String(r.chars));
  const big = await m.recall({ query: "Nvidia data center demand", tickers: ["NVDA"] });
  assert.ok(big.chars <= 6000 + 300, String(big.chars));
  const stories = (r.text.match(/Nvidia story number/g) || []).length;
  assert.equal(stories, 1, "400 near-identical stories collapse to one line (cosine > 0.92 dedupe)");
  assert.match(r.text, /Most recent exchanges/);
});

test("lexical fallback when the model is missing, slow, or broken; recall never throws", async () => {
  const missing = createMemory({ dataDir: tmp(), modelDir: join(tmp(), "none"), autoEmbed: false, now: () => NOW });
  missing.ingest({ kind: "NOTE", ticker: "AMD", text: "AMD margin guidance looked weak" });
  const a = await missing.recall({ query: "AMD margin guidance", tickers: ["AMD"] });
  assert.equal(a.method, "lexical"); assert.match(a.text, /margin guidance/);
  const slow = mem({ embed: (t) => new Promise((r) => setTimeout(() => r(mockEmbed(t)), 300)), recallTimeoutMs: 50 });
  slow.ingest({ kind: "NOTE", ticker: "AMD", text: "AMD margin guidance looked weak" });
  await slow.embedPending();
  const b = await slow.recall({ query: "AMD margin", tickers: ["AMD"] });
  assert.equal(b.method, "lexical");
  const broken = mem({ embed: () => { throw new Error("boom"); } });
  broken.ingest({ kind: "NOTE", ticker: "AMD", text: "AMD note" });
  await assert.rejects(broken.embedPending());
  const c = await broken.recall({ query: "AMD", tickers: ["AMD"] });
  assert.ok(typeof c.text === "string");
});

test("rows written by Python are embedded on the next pass", async () => {
  const dir = tmp();
  execFileSync(PY, ["-c", `import sys; sys.path.insert(0, sys.argv[1]); from memory_ingest import ingest; print(ingest('DECISION','BUY 12 NVDA @ 151.20 tsmom sleeve', ticker='NVDA', ts='2026-11-24T21:30:00-05:00', meta={'refPrice':151.2,'dedupeKey':'order:1'}, db_path=sys.argv[2])); print(ingest('DECISION','dup', ticker='NVDA', meta={'dedupeKey':'order:1'}, db_path=sys.argv[2]))`,
    fileURLToPath(new URL("..", import.meta.url)), join(dir, "ai_memory.sqlite3")]);
  const m = createMemory({ dataDir: dir, embed: mockEmbed, autoEmbed: false, now: () => NOW });
  assert.equal(m.stats().pending, 1);
  await m.embedPending();
  const r = await m.recall({ query: "NVDA buy decision tsmom", tickers: ["NVDA"], prices: { NVDA: 155 } });
  assert.match(r.text, /NVDA track record[\s\S]*- 2026-11-2\d DECISION 151\.20 -> 155\.00 \(\+2\.5%\): BUY 12 NVDA/);
});

test("track record: latest 15 verdicts per ticker, compact, inside a fixed total budget with room for the rest", async () => {
  const { TRACK_RECORD_VERDICTS, BUDGET_CHARS } = await import("../memory/memory.js");
  assert.equal(TRACK_RECORD_VERDICTS, 15); assert.equal(BUDGET_CHARS, 10000);
  const m = mem();
  const verdict = (t, i) => m.ingest({ kind: "VERDICT", ticker: t, ts: NOW - i * DAY,
    text: `Flagged YELLOW (MEDIUM), action WATCH, POSITION_20_60D; trend UP, stretch EXTENDED; entry band 150-155, stop 142.8. Verdict ${i}: ${"strong data center demand but valuation stretched ".repeat(4)} Second sentence.`,
    meta: { refPrice: 150 + i, flag: "YELLOW", action: "WATCH" } });
  for (let i = 0; i < 40; i++) { verdict("NVDA", i); verdict("AMD", i); }
  for (let i = 0; i < 6; i++) m.ingest({ kind: "DECISION", ticker: "NVDA", ts: NOW - i * DAY - 3600000, text: `BUY ${i} NVDA`, meta: { refPrice: 140, action: "BUY" } });
  for (let i = 0; i < 300; i++) m.ingest({ kind: "NEWS", ticker: "NVDA", ts: NOW - i * 3600000, text: `Nvidia headline ${i} ${["gpu", "cloud", "tariff", "china", "margin", "supply"][i % 6]} ${"words ".repeat(30)}` });
  for (let i = 0; i < 5; i++) m.ingest({ kind: "QA", ticker: "NVDA", ts: NOW - i * 60000, text: `Q: Nvidia ${i}?\nA: ${"long answer ".repeat(300)}` });
  await m.embedPending(100000);
  const r = await m.recall({ query: "Nvidia gpu cloud demand", tickers: ["NVDA"], prices: { NVDA: 155 } });
  const track = r.text.split("NVDA track record (latest first):\n")[1].split(/\n(?!- )/)[0].split("\n");
  const verdictLines = track.filter((l) => /YELLOW\/WATCH/.test(l));
  assert.equal(verdictLines.length, 15, "15 most recent verdicts, newest first");
  assert.match(verdictLines[0], /^- \d{4}-\d\d-\d\d YELLOW\/WATCH 150\.00 -> 155\.00 \(\+3\.3%\): Verdict 0: strong/);
  assert.ok(verdictLines.every((l) => l.length <= 160), "compact one-line entries");
  assert.equal(track.filter((l) => /DECISION BUY/.test(l)).length, 3);
  assert.match(r.text, /Older relevant memory/); assert.match(r.text, /Most recent exchanges/);
  assert.ok(r.chars <= BUDGET_CHARS + 300, String(r.chars));
  // Many tickers share the same pinned budget: the block still does not grow.
  const two = await m.recall({ query: "chips", tickers: ["NVDA", "AMD"], prices: { NVDA: 155, AMD: 150 } });
  assert.ok(two.chars <= BUDGET_CHARS + 300, String(two.chars));
  assert.match(two.text, /AMD track record/);
});

test("passages and lexical helpers", () => {
  const p = passages("Sentence one. ".repeat(100));
  assert.ok(p.every((x) => x.length <= 420));
  const s = lexical("margin guidance", [{ id: 1, text: "margin guidance cut" }, { id: 2, text: "unrelated" }]);
  assert.ok(s.get(1) > s.get(2));
});

test("advisor uses memory: prior verdict outcome in the verdict prompt, QA remembered, decision carries memory", async () => {
  const dir = tmp(), store = openStore(dir);
  const memory = createMemory({ dataDir: dir, embed: mockEmbed, autoEmbed: false });
  const bars = []; let c = 150;
  for (let i = 0; i < 260; i++) { const p = c; c *= 1 + 0.001 * Math.sin(i / 5); bars.push({ date: new Date(Date.now() - (260 - i) * DAY).toISOString().slice(0, 10), open: p, high: Math.max(p, c) * 1.01, low: Math.min(p, c) * 0.99, close: c, volume: 1e6 }); }
  const prices = { bars: () => bars, has: () => true };
  const sources = { details: async () => ({ value: { type: "CS", industry: "SEMICONDUCTORS" } }),
    news: async () => ({ fetchedAt: "now", value: [{ published: new Date().toISOString(), publisher: "GlobeNewswire", title: "NVIDIA announces record data center revenue", description: "", url: "u1", pressRelease: true }] }),
    dividends: async () => ({ value: [] }), splits: async () => ({ value: [] }), schwabQuotes: async () => { throw new Error("no"); },
    secFinancials: async () => ({ value: { derived: { lastQuarterEnd: "2026-10-26", revenueTTM: 3e11 }, quarterly: {}, latest: {} } }) };
  memory.ingest({ kind: "VERDICT", ticker: "NVDA", ts: new Date(Date.now() - 7 * DAY).toISOString(), text: "Flagged YELLOW at the highs; data center demand strong.", meta: { refPrice: 160 } });
  await memory.embedPending();
  const models = createModelService({ root: dir, env: { OPENAI_API_KEY: "sk-test-0123456789abcdefghijk" }, settingsPath: join(dir, "s.json") });
  const prompts = [];
  const fetchImpl = async (_u, init) => {
    const body = JSON.parse(init.body); prompts.push(body.messages.map((m) => m.content).join("\n"));
    const name = body.tool_choice.function.name;
    const args = name === "analystAnswer"
      ? { answer: "Still YELLOW.", stance: "YELLOW", conviction: "MEDIUM", keyPoints: [], levels: [], citations: [], followUps: [] }
      : { ticker: "NVDA", flag: "YELLOW", conviction: "MEDIUM", fundamentalHealth: "STRONG", trend: "SIDEWAYS", overextension: "NEUTRAL", opportunity: "NONE",
          action: "WATCH", horizon: "POSITION_20_60D", summary: "ok", entry: { orderType: "NONE", limitPrice: null, bandLow: null, bandHigh: null, rationale: "" },
          stop: { price: null, atrMultiple: null, rationale: "" }, takeProfit: { price: null, rationale: "" }, sizeHint: "NONE",
          diversification: { status: "OK", note: "" }, catalysts: [], risks: [], dataGaps: [], citations: [] };
    return { ok: true, status: 200, text: async () => JSON.stringify({ choices: [{ finish_reason: "tool_calls", message: { tool_calls: [{ function: { name, arguments: JSON.stringify(args) } }] } }] }) };
  };
  const advisor = createAdvisor({ dataDir: dir, models, store, sources, prices, memory, env: { ADVISORY_VERDICTS_PATH: join(dir, "a.json") }, fetchImpl, sleep: async () => {} });
  const v = await advisor.verdict("NVDA");
  assert.equal(v.source, "LLM");
  assert.match(prompts[0], /\[MEMORY\]/);
  assert.match(prompts[0], /NVDA track record \(latest first\):\n- \d{4}-\d\d-\d\d 160\.00 -> \d+\.\d\d \([+-]\d+\.\d%\): Flagged YELLOW at the highs; data center demand strong\./);
  await advisor.ask("What do you think of Nvidia today?", ["NVDA"]);
  await memory.embedPending();
  const kinds = memory.stats().byKind;
  assert.ok(kinds.QA === 1 && kinds.VERDICT === 2 && kinds.NEWS >= 1 && kinds.FUNDAMENTALS === 1, JSON.stringify(kinds));
  const d = (await advisor.decision(["NVDA"]))[0];
  assert.match(d.memory, /NVDA track record[\s\S]*YELLOW\/WATCH/);
});
