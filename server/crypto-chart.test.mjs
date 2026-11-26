// Chart picker lookup (pure local: fixed coin list + local price DB, no AI) and the crypto chart route wiring.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { cleanQuery, cryptoMatches, lookupSymbols, mountCryptoChart, CRYPTO_UNIVERSE } from "./crypto-chart.js";

const stocks = (known) => ({ get: (s) => (known[s] ? { sessions: 502, last: "2026-11-25" } : null) });

test("BTC with a same-symbol ETF gives two matches (crypto first); unique symbols give one", () => {
  const both = lookupSymbols("btc", stocks({ BTC: 1, NVDA: 1 }));
  assert.equal(both.matches.length, 2);
  assert.equal(both.matches[0].kind, "crypto");
  assert.match(both.matches[0].href, /symbol=BTC&asset=crypto$/);
  assert.equal(both.matches[1].kind, "stock");
  assert.equal(both.matches[1].href, "/ticker.html?symbol=BTC");
  assert.deepEqual(lookupSymbols("NVDA", stocks({ NVDA: 1 })).matches.map((m) => m.kind), ["stock"]);
  assert.deepEqual(lookupSymbols("DOGE", stocks({})).matches.map((m) => m.kind), ["crypto"]);
  assert.deepEqual(lookupSymbols("bitcoin", stocks({ BTC: 1 })).matches.map((m) => m.symbol), ["BTC"]);
  assert.deepEqual(lookupSymbols("$ETH", stocks({ ETH: 1 })).matches.map((m) => m.kind), ["crypto", "stock"]);
  assert.deepEqual(lookupSymbols("ZZZZ", stocks({})).matches.map((m) => m.detail), ["Stock / ETF (not in the local database)"]);
  assert.equal(lookupSymbols("btc-usd", stocks({ BTC: 1 })).matches.length, 1);
  assert.equal(cleanQuery("  $sol "), "SOL");
  assert.equal(cryptoMatches("solana")[0].symbol, "SOL");
  assert.equal(CRYPTO_UNIVERSE.length, 11);
});

test("lookup and chart code make no AI/LLM or order calls", () => {
  for (const f of ["./crypto-chart.js", "../public/nav.js"]) {
    const src = readFileSync(new URL(f, import.meta.url), "utf8");
    for (const needle of ["/api/ai/", "openai", "x.ai", "complete(", "/trader/", "preview" + "Order", "place" + "Order"]) {
      if (f.endsWith("nav.js") && needle === "/api/ai/") continue;   // nav links to the AI page, not the picker
      assert.ok(!src.includes(needle), `${f} contains ${needle}`);
    }
  }
  const nav = readFileSync(new URL("../public/nav.js", import.meta.url), "utf8");
  const picker = nav.slice(nav.indexOf("// Chart picker"), nav.indexOf("nav.appendChild(form)"));
  assert.ok(picker.includes("/api/symbols/lookup"));
  assert.ok(!/\/api\/ai\//.test(picker));
});

test("routes mount: lookup answers locally; unknown coin is rejected", async () => {
  const routes = {};
  const app = { get: (path, fn) => { routes[path] = fn; }, post: () => {} };
  mountCryptoChart(app, { root: "nowhere", dataDir: "nowhere" });
  const res = () => { const r = { code: 200, headers: {} }; r.status = (c) => { r.code = c; return r; }; r.set = (k, v) => { r.headers[k] = v; return r; }; r.json = (b) => { r.body = b; return r; }; return r; };
  const a = res(); routes["/api/symbols/lookup"]({ query: { q: "btc" } }, a);
  assert.equal(a.body.matches[0].kind, "crypto");
  const b = res(); routes["/api/symbols/lookup"]({ query: { q: "" } }, b);
  assert.equal(b.code, 400);
  const c = res(); await routes["/api/crypto/chart/:coin"]({ params: { coin: "NVDA" }, query: {} }, c);
  assert.equal(c.code, 400);
});

test("Schwab instrument hits join the dropdown only for symbols not in the local data", async () => {
  const { withRemote } = await import("./crypto-chart.js");
  const local = lookupSymbols("BLOZ", stocks({}));
  assert.equal(local.matches[0].detail, "Stock / ETF (not in the local database)");
  const merged = withRemote(local, { matches: [{ symbol: "BLOZF", description: "CANNABIX TECHNOLOGIE", exchange: "OTC Markets", assetType: "EQUITY" }] });
  assert.deepEqual(merged.matches.map((m) => [m.kind, m.symbol]), [["schwab", "BLOZF"]]);
  assert.match(merged.matches[0].href, /symbol=BLOZF&fetch=1$/);
  const none = withRemote(local, { matches: [], error: "SchwabAuthError" });
  assert.equal(none.matches.length, 1);
  assert.equal(none.remoteError, "SchwabAuthError");
  const routes = {};
  mountCryptoChart({ get: (p, fn) => { routes["GET " + p] = fn; }, post: (p, fn) => { routes["POST " + p] = fn; } }, { root: "nowhere", dataDir: "nowhere" });
  const r = { code: 200 }; r.status = (c) => { r.code = c; return r; }; r.json = (b) => { r.body = b; return r; };
  routes["POST /api/symbols/fetch/:ticker"]({ params: { ticker: "bad symbol!" } }, r);
  assert.equal(r.code, 400);
  const g = { code: 200 }; g.status = (c) => { g.code = c; return g; }; g.json = (b) => { g.body = b; return g; };
  routes["GET /api/symbols/fetch/:ticker"]({ params: { ticker: "BLOZF" } }, g);
  assert.equal(g.body.status, "idle");
});
