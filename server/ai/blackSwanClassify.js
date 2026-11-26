#!/usr/bin/env node
// Optional Retort forced-function classifier for black-swan headlines.
// Reads one JSON object on stdin: { title, summary }.
// Uses the existing AI transport when a provider key is already available.
// Never prints secrets. Exits 2 and {ok:false} when no key is configured.
import { stdin, stdout } from "node:process";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const EVENT_TYPES = ["war", "sanctions", "terror", "pandemic", "cyber", "disaster", "coup",
  "sovereign_default", "financial_contagion", "trade_war", "peace_deal", "ceasefire", "stimulus", "other"];
const DIRECTIONS = ["negative", "positive", "mixed", "unclear"];

export const CLASSIFY_SCHEMA = {
  name: "classifyBlackSwanHeadline",
  description: "Classify a news headline as a geopolitical or financial black-swan event.",
  parameters: {
    type: "object",
    additionalProperties: false,
    properties: {
      event_type: { type: "string", enum: EVENT_TYPES },
      direction: { type: "string", enum: DIRECTIONS },
      severity: { type: "integer", enum: [1, 2, 3, 4, 5] },
      affected_regions: { type: "array", items: { type: "string" } },
      affected_assets: { type: "array", items: { type: "string" } },
      affected_sectors: { type: "array", items: { type: "string" } },
      rationale: { type: "string" },
    },
    required: ["event_type", "direction", "severity", "affected_regions",
      "affected_assets", "affected_sectors", "rationale"],
  },
};

async function readStdin() {
  const chunks = [];
  for await (const c of stdin) chunks.push(c);
  const raw = Buffer.concat(chunks).toString("utf8").trim();
  return raw ? JSON.parse(raw) : {};
}

function hasKey(env = process.env) {
  return Boolean(env.OPENAI_API_KEY || env.XAI_API_KEY || env.GROQ_API_KEY);
}

export async function classifyHeadline(headline, { env = process.env } = {}) {
  if (!hasKey(env)) {
    return { ok: false, error: "NO_KEY", message: "No AI provider key is configured; use the keyword fallback." };
  }
  // Lazy-load the existing transport so this file can be imported in tests without a key.
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
  let chat;
  try {
    const transport = await import("./transport.js");
    const models = await import("./models.js");
    const service = models.createModelService({ root, env });
    const profile = service.getActive?.() || service.active?.() || service.profile || null;
    const apiKey = service.keyFor?.(profile?.provider) || service.getKey?.(profile?.provider);
    if (!profile || !apiKey) {
      return { ok: false, error: "NO_KEY", message: "No active AI model key." };
    }
    const messages = [
      { role: "system", content: "You classify geopolitical and financial black-swan headlines. Use only the forced function. Severity 5 is reserved for wars of invasion, nuclear use, major sovereign defaults, systemic bank failures or pandemics; 1 is a weak mention. Do not invent facts that are not in the headline." },
      { role: "user", content: `Headline: ${headline.title || ""}\nSummary: ${headline.summary || ""}` },
    ];
    const result = await transport.chatCompletions({
      profile, apiKey, messages,
      tools: [{ type: "function", function: CLASSIFY_SCHEMA }],
      tool_choice: { type: "function", function: { name: CLASSIFY_SCHEMA.name } },
      timeoutMs: 20_000,
    });
    chat = result;
  } catch (error) {
    return { ok: false, error: "MODEL_ERROR", message: String(error.message || error).slice(0, 300) };
  }
  try {
    const call = chat?.choices?.[0]?.message?.tool_calls?.[0]
      || chat?.tool_calls?.[0]
      || chat?.functionCall
      || chat;
    const raw = call?.function?.arguments || call?.arguments || call?.classification;
    const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
    if (!parsed || !EVENT_TYPES.includes(parsed.event_type)) {
      return { ok: false, error: "BAD_RESULT", message: "Model did not return a valid classification." };
    }
    return { ok: true, classification: parsed };
  } catch (error) {
    return { ok: false, error: "BAD_RESULT", message: String(error.message || error).slice(0, 300) };
  }
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("blackSwanClassify.js")) {
  const headline = await readStdin();
  const out = await classifyHeadline(headline);
  stdout.write(JSON.stringify(out) + "\n");
  process.exit(out.ok ? 0 : 2);
}
