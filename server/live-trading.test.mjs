// Mocked tests for the manual live-trading path. No network: fetch is replaced; no real order is ever sent.
// Run: node --test live-trading.test.mjs
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createLiveTrading } from "./live-trading.js";

function setup({ allowed = true, held = 10, clock = { t: 1_800_000_000_000 } } = {}) {
  const root = mkdtempSync(join(tmpdir(), "live-"));
  writeFileSync(join(root, "schwab_token.json"), JSON.stringify({ token: { access_token: "fake", refresh_token: "r",
    expires_at: Math.floor(clock.t / 1000) + 1800 } }));
  const calls = [];
  const fetchImpl = async (url, options = {}) => {
    const path = new URL(url).pathname;
    calls.push({ path, method: options.method ?? "GET", body: options.body });
    const reply = (status, body, headers = {}) => ({ ok: status < 300, status, text: async () => (body ? JSON.stringify(body) : ""),
      json: async () => body, headers: { get: (k) => headers[k.toLowerCase()] ?? null } });
    if (path.endsWith("/accountNumbers")) return reply(200, [{ accountNumber: "12345678", hashValue: "HASH1" }]);
    if (path.startsWith("/marketdata/v1/quotes")) return reply(200, { AAPL: { quote: { lastPrice: 100, bidPrice: 99.9, askPrice: 100.1 } } });
    if (path.endsWith("/previewOrder")) return reply(200, { orderStrategy: { orderBalance: { orderValue: 500 } } });
    if (path === "/trader/v1/accounts/HASH1" ) return reply(200, { securitiesAccount: { positions: [{ instrument: { symbol: "AAPL" }, longQuantity: held }] } });
    if (path.endsWith("/orders") && options.method === "POST") return reply(201, null, { location: `${url}/987654` });
    if (path.includes("/orders/987654")) return reply(200, { status: "FILLED", filledQuantity: 5, remainingQuantity: 0,
      orderActivityCollection: [{ executionLegs: [{ quantity: 5, price: 100.05, time: "t" }] }] });
    return reply(404, { message: "unexpected" });
  };
  const env = { LIVE_TRADING_ALLOWED: allowed ? "1" : "0" };
  const live = createLiveTrading({ root, dataDir: join(root, "Massive"), fetchImpl, env, now: () => clock.t, log: {} });
  return { live, calls, root, clock };
}
const ticket = { symbol: "aapl", side: "BUY", quantity: 5, orderType: "LIMIT", limitPrice: 100.5, duration: "DAY", accountHash: "HASH1" };
const orderPosts = (calls) => calls.filter((c) => c.path.endsWith("/orders") && c.method === "POST");

test("live trading starts OFF and confirm is refused while OFF", async () => {
  const { live, calls } = setup();
  assert.equal(live.settings().enabled, false);
  const p = await live.preview(ticket);
  assert.equal(p.liveTradingEnabled, false);
  assert.equal(p.ticket.account, "…5678");
  await assert.rejects(live.confirm({ previewId: p.previewId, idempotencyKey: "key-00001" }), /OFF/);
  assert.equal(orderPosts(calls).length, 0);
});

test("environment switch is required to enable", async () => {
  const { live } = setup({ allowed: false });
  assert.throws(() => live.setEnabled(true, "ENABLE LIVE TRADING"), /LIVE_TRADING_ALLOWED/);
  const ok = setup();
  assert.throws(() => ok.live.setEnabled(true, "yes"), /Type ENABLE/);
});

test("preview -> confirm sends exactly the previewed order once (idempotent)", async () => {
  const { live, calls } = setup();
  live.setEnabled(true, "ENABLE LIVE TRADING");
  const p = await live.preview(ticket);
  assert.deepEqual(p.order.orderLegCollection[0], { instruction: "BUY", quantity: 5, instrument: { symbol: "AAPL", assetType: "EQUITY" } });
  assert.equal(p.order.price, "100.50");
  const first = await live.confirm({ previewId: p.previewId, idempotencyKey: "key-00002" });
  assert.equal(first.orderId, "987654");
  const again = await live.confirm({ previewId: p.previewId, idempotencyKey: "key-00002" });
  assert.equal(again.duplicate, true);
  await assert.rejects(live.confirm({ previewId: p.previewId, idempotencyKey: "key-00003" }), /already submitted/);
  assert.equal(orderPosts(calls).length, 1);
  assert.deepEqual(JSON.parse(orderPosts(calls)[0].body), p.order);
  const status = await live.orderStatus(p.previewId);
  assert.equal(status.status, "FILLED");
});

test("concurrent double click submits once", async () => {
  const { live, calls } = setup();
  live.setEnabled(true, "ENABLE LIVE TRADING");
  const p = await live.preview(ticket);
  const results = await Promise.allSettled([live.confirm({ previewId: p.previewId, idempotencyKey: "key-aaaa1" }),
    live.confirm({ previewId: p.previewId, idempotencyKey: "key-aaaa2" })]);
  assert.equal(results.filter((r) => r.status === "fulfilled").length, 1);
  assert.equal(orderPosts(calls).length, 1);
});

test("expired previews, oversells, bad tickets and unknown accounts are refused", async () => {
  const { live, calls, clock } = setup({ held: 3 });
  live.setEnabled(true, "ENABLE LIVE TRADING");
  const p = await live.preview(ticket);
  clock.t += 121_000;
  await assert.rejects(live.confirm({ previewId: p.previewId, idempotencyKey: "key-00004" }), /expired/);
  await assert.rejects(live.preview({ ...ticket, side: "SELL", quantity: 5 }), /hold 3/);
  await assert.rejects(live.preview({ ...ticket, quantity: 1.5 }), /whole number/);
  await assert.rejects(live.preview({ ...ticket, orderType: "MARKET", duration: "GOOD_TILL_CANCEL" }), /DAY/);
  await assert.rejects(live.preview({ ...ticket, accountHash: "OTHER" }), /Unknown account/);
  await assert.rejects(live.preview({ ...ticket, quantity: undefined, dollars: 50 }), /less than one share/);
  assert.equal(orderPosts(calls).length, 0);
});

test("dollar amount is converted to whole shares and everything is journaled without secrets", async () => {
  const { live, root } = setup();
  const p = await live.preview({ ...ticket, quantity: undefined, dollars: 1000, orderType: "MARKET" });
  assert.equal(p.ticket.quantity, 9);                      // floor(1000 / ask 100.1)
  const text = readFileSync(join(root, "Massive", "LiveTrading", "orders.jsonl"), "utf8");
  assert.match(text, /"event":"preview"/);
  assert.doesNotMatch(text, /fake|12345678/);              // no token, no full account number
});
