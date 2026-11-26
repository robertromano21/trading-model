// Real MiniLM smoke test (skipped when models/memory is not installed). Uses the all-MiniLM-L6-v2 ONNX model.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { createEmbedder } from "../memory/embedder.js";
import { createMemory } from "../memory/memory.js";

const modelDir = process.env.TRADING_AI_MEMORY_MODEL_DIR || fileURLToPath(new URL("../../../models/memory", import.meta.url));
const installed = existsSync(join(modelDir, "manifest.json"));

test("real MiniLM embeddings: semantic neighbours and recall", { skip: !installed && "models/memory not installed" }, async () => {
  const e = createEmbedder({ modelDir });
  const started = Date.now();
  const [a, b, c] = await e.vectorize(["Nvidia quarterly results beat expectations on data center demand",
    "NVDA earnings topped estimates thanks to AI chip sales", "Crude oil prices fell on an OPEC supply increase"]);
  const dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0);
  assert.equal(a.length, 384);
  assert.ok(Math.abs(Math.hypot(...a) - 1) < 1e-3);
  assert.ok(dot(a, b) > dot(a, c) + 0.15, `${dot(a, b)} vs ${dot(a, c)}`);
  console.log(`MiniLM ok in ${Date.now() - started} ms: sim(nvda,nvda)=${dot(a, b).toFixed(3)} sim(nvda,oil)=${dot(a, c).toFixed(3)}`);
  const m = createMemory({ dataDir: mkdtempSync(join(tmpdir(), "tm-smoke-")), embedder: e, autoEmbed: false });
  m.ingest({ kind: "NOTE", ticker: "NVDA", text: "Robert worries about hyperscaler capex cuts hurting GPU orders." });
  m.ingest({ kind: "NOTE", ticker: "NVDA", text: "Dividend is tiny; not an income holding." });
  await m.embedPending();
  const r = await m.recall({ query: "Will cloud spending slowdowns hit Nvidia's chip demand?", tickers: ["NVDA"] });
  assert.equal(r.method, "semantic");
  assert.ok(r.text.indexOf("hyperscaler") >= 0);
  await e.close();
});
