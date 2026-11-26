// Structured model calls for non-reasoning models (OpenAI-compatible chat completions: OpenAI, xAI, Groq).
// Model transport, generation policy and failure handling:
//  * the answer is forced through ONE function call whose parameters are the strict Retort schema (enums enforced)
//  * no max_tokens / max_completion_tokens is ever sent (no output cap); finish_reason "length" is rejected
//    instead of accepting a truncated answer
//  * reasoning_effort "none" for models whose catalog row says reasoningNone
//  * refusals, empty answers, invalid JSON and schema mismatches are typed failures
//  * provider retries (408/429/5xx) and validation retries share a budget of 1 attempt + 3 retries, with backoff
//  * every attempt has a deadline (AbortController); API keys are never logged or included in errors
import { contract, decodeStrictOptionals, matches, mismatch } from "./retortSchema.js";
import { BASE_URL } from "./models.js";

export const RETRIES = 3;
export const ATTEMPTS = RETRIES + 1;

export const failure = (code, message, details = {}) => Object.assign(new Error(message), { code, ...details });

export function providerRetry(error) {
  if (error?.recoverable === false || /AUTH|REFUS|CANCEL|NO_KEY/.test(error?.code || "")) return false;
  const status = error?.status;
  if (/quota|billing|credit|spending.limit/i.test(String(error?.providerCode || "") + " " + (error?.message || ""))) return false;
  return status === 408 || status === 429 || (status >= 500 && status <= 599) || error?.code === "MODEL_TIMEOUT" ||
    error?.code === "MODEL_NETWORK";
}
export function validationRetry(error) {
  return ["MODEL_SCHEMA", "MODEL_INVALID_JSON", "MODEL_EMPTY", "MODEL_NO_STRUCTURED", "MODEL_OUTPUT_LIMIT"].includes(error?.code);
}
const backoff = (attempt, sleep) => sleep(Math.min(4000, 500 * 2 ** (attempt - 1)));

export function requestBody({ profile, messages, name, description, parameters, temperature = 0.2 }) {
  const { wireSchema } = contract(parameters);
  const body = {
    model: profile.apiModel,
    messages,
    temperature,
    tools: [{ type: "function", function: { name, description, parameters: wireSchema, strict: true } }],
    tool_choice: { type: "function", function: { name } },
    parallel_tool_calls: false,
  };
  if (profile.reasoningNone) body.reasoning_effort = "none";
  if (profile.reasoningEffort) body.reasoning_effort = profile.reasoningEffort;
  // Deliberately uncapped: no max_tokens / max_completion_tokens / max_output_tokens.
  return body;
}

function verifyChoice(choice) {
  if (!choice) throw failure("MODEL_EMPTY", "The model returned no answer.");
  if (choice.finish_reason === "length")
    throw failure("MODEL_OUTPUT_LIMIT", "The model reached the provider output limit. No partial answer was accepted.");
  if (choice.message?.refusal || choice.finish_reason === "content_filter")
    throw failure("MODEL_REFUSED", "The model declined to answer.", { recoverable: false });
}

async function completeOnce({ profile, apiKey, body, name, answerSchema, fetchImpl, timeoutMs }) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetchImpl(BASE_URL[profile.provider]() + "/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: "Bearer " + apiKey },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (error) {
    if (controller.signal.aborted) throw failure("MODEL_TIMEOUT", `The model did not answer within ${Math.round(timeoutMs / 1000)} s.`);
    throw failure("MODEL_NETWORK", "Could not reach the model provider: " + (error?.cause?.code || error?.message || "network error"));
  } finally { clearTimeout(timer); }
  let raw;
  const text = await response.text().catch(() => "");
  try { raw = JSON.parse(text); } catch { raw = null; }
  if (!response.ok) {
    const providerCode = raw?.error?.code || raw?.error?.type || "";
    const message = String(raw?.error?.message || `HTTP ${response.status}`).replace(/(sk|xai|gsk)-[A-Za-z0-9_-]{8,}/g, "[key]");
    throw failure(response.status === 401 || response.status === 403 ? "MODEL_AUTH" : "MODEL_HTTP",
      `Model provider error ${response.status}: ${message}`, { status: response.status, providerCode,
        recoverable: response.status === 401 || response.status === 403 ? false : undefined });
  }
  if (!raw) throw failure("MODEL_INVALID_JSON", "The provider returned a non-JSON response.", { status: 502 });
  const choice = raw.choices?.[0];
  verifyChoice(choice);
  const call = choice.message?.tool_calls?.[0];
  if (!call) throw failure("MODEL_NO_STRUCTURED", "The model returned text instead of the requested structured answer.");
  if (choice.message.tool_calls.length > 1) throw failure("MODEL_NO_STRUCTURED", "The model returned more than one structured answer.");
  if (call.function?.name !== name) throw failure("MODEL_NO_STRUCTURED", "The model returned an unexpected answer function.");
  const content = String(call.function?.arguments ?? "");
  if (!content.trim()) throw failure("MODEL_EMPTY", "The model returned an empty structured answer.");
  let parsed;
  try { parsed = JSON.parse(content); } catch { throw failure("MODEL_INVALID_JSON", "The model returned invalid structured JSON."); }
  decodeStrictOptionals(parsed, answerSchema);
  if (!matches(parsed, answerSchema))
    throw failure("MODEL_SCHEMA", "The model answer does not match the schema: " + mismatch(parsed, answerSchema), { rejectedAnswer: parsed });
  return { value: parsed, usage: raw.usage || null, model: raw.model || profile.apiModel };
}

/**
 * Run one structured call with retries. Resolves { value, usage, model, attempts } or throws the last typed failure.
 * Callers wrap this in a deterministic fallback, so a failure never changes trading.
 */
export async function complete({ models, modelId, messages, name, description, parameters, temperature,
  fetchImpl = globalThis.fetch, timeoutMs, sleep = (ms) => new Promise((r) => setTimeout(r, ms)), log = () => {} }) {
  const profile = modelId ? models.profile(modelId) : models.current();
  const apiKey = models.apiKey(profile.provider);
  if (!apiKey) throw failure("MODEL_NO_KEY", `${profile.name} needs an ${({ openai: "OpenAI", xai: "xAI", groq: "Groq" })[profile.provider]} API key. Enter it in the AI analyst page (Model > key).`, { recoverable: false });
  const { answerSchema } = contract(parameters);
  const body = requestBody({ profile, messages, name, description, parameters, temperature });
  const deadline = timeoutMs ?? Number(process.env.TRADING_AI_TIMEOUT_MS || 120000);
  let lastError;
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    try {
      const result = await completeOnce({ profile, apiKey, body, name, answerSchema, fetchImpl, timeoutMs: deadline });
      return { ...result, attempts: attempt, modelId: profile.id };
    } catch (error) {
      lastError = error;
      error.attempts = attempt;
      const retry = attempt < ATTEMPTS && (providerRetry(error) || validationRetry(error));
      log({ status: retry ? "retrying" : "failed", attempt, code: error.code, message: error.message, model: profile.id });
      if (!retry) break;
      await backoff(attempt, sleep);
    }
  }
  throw lastError;
}
