// Sealed historical replay: read a point-in-time prompt from stdin and return one structured verdict.
// This path never loads the price database, never fetches news or fundamentals, and never places an order.
// The Python caller (engine/analyst_replay.py) is responsible for building a PIT-only bundle and for
// asserting that no future rows or (when masked) identity fields are present.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { converse } from "./runtime.js";
import { ANALYST_RULES, VERDICT_TASK } from "./prompts.js";
import { VERDICT } from "./schemas.js";
import { createModelService } from "./models.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const dataDir = process.env.PRICES_DATA_DIR || join(root, "Massive");
const body = readFileSync(0, "utf8");
const bundle = JSON.parse(body);
const prompt = bundle.prompt;
if (!prompt || typeof prompt !== "string") {
  process.stderr.write("replay_verdict: stdin must be JSON with a 'prompt' string (PIT context only).\n");
  process.exit(2);
}

const models = createModelService({ root, env: process.env, settingsPath: join(dataDir, "ai_settings.json") });
const current = models.current();
if (!current?.ready) {
  process.stderr.write("replay_verdict: no ready non-reasoning model (API key missing). " +
    "The Python caller should fall back to the PIT heuristic.\n");
  process.exit(3);
}

const result = await converse(($) => {
  $.system`${ANALYST_RULES}

You are scoring a historical, point-in-time snapshot. Use only the facts in the user message.
There is no data after the stated as-of date. If the name is anonymized, do not guess its identity.`;
  $.user`${prompt}`;
  $.user`${VERDICT_TASK} As-of date (US Eastern session close): ${bundle.asof || "unknown"}. Label: ${bundle.ticker || "unknown"}.`;
}, {
  models, modelId: bundle.modelId, name: "replayVerdict",
  description: "Structured red/yellow/green verdict for a sealed historical snapshot.",
  parameters: VERDICT, temperature: 0.2,
});

const value = result.value || {};
const out = {
  ticker: bundle.ticker, asof: bundle.asof, source: "LLM",
  flag: value.flag, action: value.action, opportunity: value.opportunity,
  conviction: value.conviction, trend: value.trend, overextension: value.overextension,
  stop: value.stop?.price ?? value.stop ?? null,
  entryLow: value.entry?.bandLow ?? null, entryHigh: value.entry?.bandHigh ?? null,
  summary: value.summary ?? null, model: result.modelId || current.id,
  masked: Boolean(bundle.masked),
};
process.stdout.write(JSON.stringify(out) + "\n");
