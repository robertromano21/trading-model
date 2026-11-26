// Alert categories for per-category mute (voice + visual). Every alert gets exactly one category.
// The browser settings panel (public/alerts/alerts.js) shows one row per category in this order.
export const CATEGORIES = Object.freeze([
  { id: "analyst", label: "AI analyst replies" },
  { id: "trade_main", label: "Trades: main paper (swing + day trading)" },
  { id: "trade_swarm", label: "Trades: swarm shadow accounts" },
  { id: "trade_crypto", label: "Trades: crypto paper accounts" },
  { id: "trade_options", label: "Trades: options paper" },
  { id: "signals", label: "AI flags, overextension, dilution, drift" },
  { id: "black_swan", label: "Black swan" },
  { id: "dollar_shock", label: "Dollar shock" },
  { id: "regime", label: "Market regime changes" },
  { id: "macro", label: "Macro releases and nowcast surprises" },
  { id: "system", label: "System and server (Schwab auth, errors)" },
]);

export const CATEGORY_IDS = Object.freeze(CATEGORIES.map((c) => c.id));

// Account naming: swarm shadow ids 100-125 (and "swarm-*"), crypto paper ids 200-219 (and "crypto-*"),
// options paper "options-paper"; everything else (default, daytrade) is the main paper account.
export function accountGroup(account) {
  const a = String(account ?? "").trim().toLowerCase();
  if (!a) return "trade_main";
  if (/crypto|btc|coin/.test(a)) return "trade_crypto";
  if (/option/.test(a)) return "trade_options";
  if (/swarm|shadow/.test(a)) return "trade_swarm";
  const n = Number(a.replace(/^(acct|account)[-_ ]?/, ""));
  if (Number.isInteger(n)) {
    if (n >= 200 && n <= 299) return "trade_crypto";
    if (n >= 100 && n <= 199) return "trade_swarm";
  }
  return "trade_main";
}

const TYPE_CATEGORY = {
  analyst: "analyst",
  flag_change: "signals",
  overextension: "signals",
  price_threshold: "signals",
  dilution: "signals",
  drift: "signals",
  regime: "regime",
  macro_release: "macro",
  macro_nowcast: "macro",
  schwab_auth: "system",
  error: "system",
  system: "system",
};

export function categorize(alert = {}) {
  const meta = alert.meta && typeof alert.meta === "object" ? alert.meta : {};
  const explicit = String(meta.category || alert.category || "").toLowerCase();
  if (CATEGORY_IDS.includes(explicit)) return explicit;
  if (alert.type === "black_swan") {
    const ev = String(meta.eventType || meta.kind || "").toLowerCase();
    return ev.includes("dollar") ? "dollar_shock" : "black_swan";
  }
  if (TYPE_CATEGORY[alert.type]) return TYPE_CATEGORY[alert.type];
  // decision, fill, stop_hit, circuit_breaker: by account group
  return accountGroup(meta.account ?? alert.account);
}
