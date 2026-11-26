// Crypto awareness: question routing, [CRYPTO] context block within the budget, CRYPTO_STATUS tool (fixtures, no network).
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import * as E from "../enums.js";
import { isCryptoQuestion, stripCoinTickers, cryptoBlock, cryptoStatus, startCryptoRefresher } from "../crypto.js";
import { fallbackIntent, classify } from "../command.js";
import { buildContext } from "../context.js";
import { planCall, createTools, TOOLS } from "../tools.js";

const call = (tool, extra = {}) => ({ tool, method: "DEFAULT", family: "NONE", tickers: [], start: "", end: "", horizonDays: null, options: [], query: "", purpose: "", ...extra });

function fixture(context = "[CRYPTO] as of 2026-11-26 07:57 ET. BTC $60,805.40 (24h +3.6%, 7d -3.5%, 30d -24.0%, downtrend).") {
  const dir = mkdtempSync(join(tmpdir(), "tm-crypto-"));
  mkdirSync(join(dir, "Research", "crypto", "paper"), { recursive: true });
  writeFileSync(join(dir, "Research", "crypto", "crypto_snapshot.json"), JSON.stringify({
    asOf: "2026-11-26 07:57", context, coins: [{ coin: "BTC", price: 60805.4, chg24h: 0.036, trend: "downtrend" }],
    macro: { uup30d: 0.023 }, verdict: { btcHoldSharpe: 0.98 } }));
  writeFileSync(join(dir, "Research", "crypto", "paper", "crypto_paper_status.json"), JSON.stringify({
    forwardStart: "2026-11-25", days: 1, accounts: [{ id: 210, label: "BTC/ETH TSMOM", equity: 9986, totalReturn: -0.0014, targetWeights: { BTC: 0.5 } }] }));
  return dir;
}

test("crypto questions are detected; ordinary English is not", () => {
  for (const q of ["how is crypto doing", "Bitcoin today?", "btc", "what about ETH and SOL", "how are the crypto paper accounts", "ethereum vs solana", "XRP?"])
    assert.equal(isCryptoQuestion(q), true, q);
  for (const q of ["link my account", "what is the solution", "What do you think of Nvidia today?", "dot plot from the Fed", ""])
    assert.equal(isCryptoQuestion(q), false, q);
  assert.deepEqual(stripCoinTickers(["BTC", "IBIT", "COIN", "SOL"], "bitcoin vs IBIT"), ["IBIT", "COIN"]);
  assert.deepEqual(stripCoinTickers(["SOL"], "Is SOL a buy"), []);
  assert.deepEqual(stripCoinTickers(["NVDA"], "What about NVDA"), ["NVDA"]);
});

test("crypto routes to GENERAL (not ticker, MARKET or HELP)", async () => {
  for (const q of ["how is crypto doing", "How is BTC doing", "how is the crypto market", "bitcoin", "ETH?", "crypto help me understand the dollar link"]) {
    const r = fallbackIntent(q);
    assert.equal(r.intent, "GENERAL", q);
    assert.deepEqual(r.tickers, [], q);
  }
  assert.deepEqual(fallbackIntent("compare IBIT and bitcoin").tickers, ["IBIT"]);
  const c = await classify("how is crypto doing", { prices: { has: () => true } });
  assert.equal(c.intent, "GENERAL");
  assert.deepEqual(c.tickers, []);
});

test("[CRYPTO] block joins the context for crypto questions and stays within the budget", () => {
  const dir = fixture("[CRYPTO] " + "x".repeat(20000));
  const block = cryptoBlock(dir);
  assert.ok(block.length <= 1800);
  const ctx = buildContext([], { budgetChars: 9000, book: { available: false }, dataDir: dir, question: "how is crypto doing" });
  assert.ok(ctx.includes("[CRYPTO]"));
  assert.ok(ctx.length <= 9500, String(ctx.length));
  const other = buildContext([], { budgetChars: 9000, book: { available: false }, dataDir: dir, question: "is overextension a good model" });
  assert.ok(!other.includes("[CRYPTO]"));
  assert.equal(cryptoBlock(mkdtempSync(join(tmpdir(), "tm-crypto-empty-"))), null);
});

test("CRYPTO_STATUS tool returns prices, paper accounts and verdict (fresh snapshot: no refresh)", async () => {
  assert.ok(E.TOOL.includes("CRYPTO_STATUS"));
  assert.ok(TOOLS.CRYPTO_STATUS);
  const plan = planCall(call("CRYPTO_STATUS"));
  assert.equal(plan.kind, "node");
  assert.equal(plan.op, "cryptoStatus");
  const dir = fixture();
  const tools = createTools({ root: dir, dataDir: dir, quant: null, store: null });
  const out = await tools.execute(call("CRYPTO_STATUS"));
  assert.equal(out.ok, true, JSON.stringify(out.error));
  assert.equal(out.result.coins[0].coin, "BTC");
  assert.equal(out.result.paper.accounts[0].id, 210);
  assert.equal(out.result.refreshed, null);
  assert.match(out.result.context, /^\[CRYPTO\]/);
  const empty = await cryptoStatus({ root: dir, dataDir: mkdtempSync(join(tmpdir(), "tm-crypto-none-")) });
  assert.equal(empty.available, false);
});

test("refresher can be disabled and never references order endpoints", async () => {
  const stop = startCryptoRefresher({ root: "nowhere", env: { TRADING_CRYPTO_SNAPSHOT: "off" } });
  assert.equal(typeof stop, "function");
  stop();
  const { readFileSync } = await import("node:fs");
  const src = readFileSync(new URL("../crypto.js", import.meta.url), "utf8");
  for (const needle of ["/trader/", "/api/live/", "preview" + "Order", "place" + "Order"]) assert.ok(!src.includes(needle), needle);
});
