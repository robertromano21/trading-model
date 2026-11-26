// Retort conversation runtime for the analyst. Prompts are written Retort-style ($.system`...`, $.user`...`)
// in the usual Retort style; the generation step is routed through our structured transport
// (strict enum schema, validation, retries, timeouts) instead of retort-js's built-in OpenAI client.
import { createRequire } from "node:module";
import { complete } from "./transport.js";

const require = createRequire(import.meta.url);
const Retort = require("retort-js");
// retort-js prints every message (whole prompt contexts) to stdout; keep that off unless debugging.
const logModule = require("retort-js/dist/log-message");
const originalLog = logModule.logMessage;
logModule.logMessage = (message) => { if (process.env.TRADING_AI_RETORT_LOG === "1") originalLog(message); };

async function plainMessages($) {
  return Promise.all($.messages.map(async (m) => {
    const resolved = await (m.promise || m);
    return { role: resolved.role, content: resolved.content };
  }));
}

/**
 * converse(build, generation): build(conversation) writes prompts with Retort template tags, then the structured
 * generation runs through transport.complete. The validated answer is appended as the assistant turn.
 */
export async function converse(build, generation) {
  return Retort.run(Retort.retort(async ($) => {
    await build($);
    const messages = await plainMessages($);
    const result = await complete({ ...generation, messages });
    try { $.assistant(JSON.stringify(result.value)); } catch { /* transcript only */ }
    return result;
  }), null, { shouldSaveToLog: false });
}
export { plainMessages };
