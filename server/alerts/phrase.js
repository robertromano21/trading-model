// First-person, short phrasing for the AI analyst. Callers may supply `speech`.
// If the AI layer later exports a summarize/phrase helper we use it; otherwise templates.
export const ALERT_TYPES = Object.freeze({
  decision: { label: "Paper decision", severity: "info", speakDefault: true },
  fill: { label: "Fill", severity: "success", speakDefault: true },
  stop_hit: { label: "Stop hit", severity: "critical", speakDefault: true },
  circuit_breaker: { label: "Circuit breaker", severity: "critical", speakDefault: true },
  flag_change: { label: "AI flag change", severity: "warning", speakDefault: true },
  overextension: { label: "Overextension", severity: "warning", speakDefault: true },
  price_threshold: { label: "Price threshold", severity: "info", speakDefault: true },
  schwab_auth: { label: "Schwab authorization", severity: "warning", speakDefault: true },
  error: { label: "Error", severity: "critical", speakDefault: true },
  drift: { label: "Backtest drift", severity: "warning", speakDefault: true },
  macro_release: { label: "Macro release", severity: "info", speakDefault: true },
  dilution: { label: "Dilution flag", severity: "warning", speakDefault: true },
  black_swan: { label: "Black swan", severity: "warning", speakDefault: true },
  regime: { label: "Regime change", severity: "warning", speakDefault: true },
  macro_nowcast: { label: "Macro nowcast", severity: "info", speakDefault: true },
  analyst: { label: "AI analyst", severity: "info", speakDefault: true },
  system: { label: "System", severity: "info", speakDefault: true },
});

export const SEVERITIES = Object.freeze(["info", "success", "warning", "critical"]);

const FLAG_WORD = { GREEN: "green", YELLOW: "yellow", RED: "red" };

export function clipSpeech(text, max = 240) {
  const s = String(text || "").replace(/\s+/g, " ").trim();
  if (s.length <= max) return s;
  const cut = s.slice(0, max - 1);
  const at = Math.max(cut.lastIndexOf(". "), cut.lastIndexOf(", "), cut.lastIndexOf(" "));
  return (at > 80 ? cut.slice(0, at) : cut).trim() + ".";
}

export function normalizeType(type) {
  const key = String(type || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
  const aliases = {
    paper_trade: "decision",
    paper_decision: "decision",
    trade_decision: "decision",
    filled: "fill",
    order_fill: "fill",
    stop: "stop_hit",
    stop_loss: "stop_hit",
    breaker: "circuit_breaker",
    halt: "circuit_breaker",
    flag: "flag_change",
    ai_flag: "flag_change",
    threshold: "price_threshold",
    price: "price_threshold",
    schwab: "schwab_auth",
    auth: "schwab_auth",
    schwab_auth_needed: "schwab_auth",
    backtest_drift: "drift",
    drift_alert: "drift",
    macro: "macro_release",
    economic_release: "macro_release",
    macro_event: "macro_release",
    dilution_flag: "dilution",
    share_dilution: "dilution",
    offering: "dilution",
    blackswan: "black_swan",
    geopolitical: "black_swan",
    geo_event: "black_swan",
    dollar_shock: "black_swan",
    dollar_surge: "black_swan",
    regime_change: "regime",
    market_regime: "regime",
    nowcast: "macro_nowcast",
    nowcast_surprise: "macro_nowcast",
    macro_surprise: "macro_nowcast",
    analyst_reply: "analyst",
    ai_reply: "analyst",
    server: "system",
    lifecycle: "system",
  };
  const resolved = aliases[key] || key;
  return Object.hasOwn(ALERT_TYPES, resolved) ? resolved : null;
}

function flagWord(alert) {
  const raw = String(alert?.meta?.flag || alert?.flag || "").toUpperCase();
  return FLAG_WORD[raw] || (raw ? raw.toLowerCase() : "updated");
}

export function templateSpeech(alert) {
  const t = alert.ticker ? String(alert.ticker).toUpperCase() : "";
  const meta = alert.meta && typeof alert.meta === "object" ? alert.meta : {};
  const named = t ? `${t}: ` : "";
  switch (alert.type) {
    case "fill": {
      const side = meta.side ? String(meta.side).toLowerCase() : "order";
      const qty = meta.qty != null ? ` for ${meta.qty}` : "";
      const px = meta.price != null ? ` at ${meta.price}` : "";
      return `${named}the ${side} filled${qty}${px}.`;
    }
    case "decision":
      return clipSpeech(alert.text) || `${named}I have a new trading decision.`;
    case "stop_hit": {
      const px = meta.price != null ? ` at ${meta.price}` : "";
      return `${named}hit the stop${px}. I'm out of the position.`;
    }
    case "circuit_breaker":
      return clipSpeech(`I've halted trading. ${alert.text || "The daily loss limit was reached."}`);
    case "flag_change":
      return clipSpeech(`${named}is now flagged ${flagWord(alert)}. ${alert.text || ""}`.trim());
    case "overextension": {
      const pct = meta.pct != null ? `, ${meta.pct} from its average` : "";
      return `${named}looks overextended${pct}. I'd wait for a better entry.`;
    }
    case "price_threshold":
      return `${named}crossed ${meta.threshold || "a price threshold"}.`;
    case "schwab_auth":
      return "Schwab authorization is needed before I can get live quotes.";
    case "error":
      return clipSpeech(`Something went wrong. ${alert.text || "Please check the logs."}`);
    case "drift": {
      const asof = meta.asof ? ` as of ${meta.asof}` : "";
      if ((alert.severity || "") === "critical") {
        return clipSpeech(`I've detected a serious drift versus the backtest${asof}. ${alert.text || "Paper results no longer match the backtest's expected range."} I'd treat them with caution until this is explained.`);
      }
      return clipSpeech(`Paper trading is outside the backtest's usual range${asof}. ${alert.text || "It's a warning, not a halt, but I'll keep watching."}`);
    }
    case "macro_release": {
      const name = meta.name || meta.eventId || "A macro release";
      return clipSpeech(alert.text) || clipSpeech(`${name} is out. ${meta.surpriseZ != null ? "The surprise z-score is " + meta.surpriseZ + "." : ""}`.trim());
    }
    case "dilution": {
      const pct = meta.shareChangeYoY != null ? `, ${(Number(meta.shareChangeYoY) * 100).toFixed(1)} percent year over year` : "";
      return clipSpeech(`${named}is flagged for dilution${pct}. ${alert.text || "I'm blocking new buys while this is active."}`.trim());
    }
    case "black_swan": {
      if (meta.eventType === "dollar_shock") {
        return clipSpeech(alert.text) || clipSpeech("Dollar shock alert. This is a watch only — no paper trades were changed.");
      }
      const st = meta.state || "updated";
      return clipSpeech(alert.text) || clipSpeech(`Black swan state is ${st}. ${alert.text || "I'm watching markets and the news together before acting."}`);
    }
    case "regime": {
      const from = meta.from ? ` from ${String(meta.from).replace(/_/g, " ").toLowerCase()}` : "";
      const to = meta.to ? String(meta.to).replace(/_/g, " ").toLowerCase() : "a new state";
      return clipSpeech(alert.text) || `The market regime changed${from} to ${to}.`;
    }
    case "macro_nowcast":
      return clipSpeech(alert.text) || "My macro nowcast moved.";
    case "analyst":
      return clipSpeech(alert.text) || "I have an answer for you.";
    case "system":
      return clipSpeech(alert.text) || "System notice.";
    default:
      return clipSpeech(alert.text) || "New alert.";
  }
}

let phraseHelper = null;
let phraseHelperTried = false;

async function loadPhraseHelper() {
  if (phraseHelperTried) return phraseHelper;
  phraseHelperTried = true;
  try {
    const mod = await import("../ai/phrase.js");
    if (typeof mod.phraseAlert === "function") phraseHelper = mod.phraseAlert;
    else if (typeof mod.summarizeAlert === "function") phraseHelper = mod.summarizeAlert;
    else if (typeof mod.summarize === "function") phraseHelper = mod.summarize;
  } catch {
    try {
      const mod = await import("../ai/sources.js");
      if (typeof mod.summarizeAlert === "function") phraseHelper = mod.summarizeAlert;
    } catch { /* AI layer has no alert summarizer yet; templates are enough. */ }
  }
  return phraseHelper;
}

export async function phraseAlert(alert) {
  const provided = clipSpeech(alert.speech);
  if (provided) return provided;
  try {
    const helper = await loadPhraseHelper();
    if (helper) {
      const spoken = clipSpeech(await helper(alert));
      if (spoken) return spoken;
    }
  } catch { /* never fail emit because phrasing failed */ }
  return clipSpeech(templateSpeech(alert));
}
