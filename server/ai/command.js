// Command prompt: LLM-first intent classification with a deterministic fallback (structured
// enum intent, temperature 0, short timeout, safe NONE default). Slash commands bypass the model.
import { INTENT } from "./schemas.js";
import { INTENT_SYSTEM } from "./prompts.js";
import { INTENT as INTENTS } from "./enums.js";
import { complete } from "./transport.js";
import { isCryptoQuestion, stripCoinTickers } from "./crypto.js";

export const NAMES = Object.freeze({
  NVIDIA: "NVDA", APPLE: "AAPL", MICROSOFT: "MSFT", AMAZON: "AMZN", GOOGLE: "GOOGL", ALPHABET: "GOOGL", META: "META",
  FACEBOOK: "META", TESLA: "TSLA", BROADCOM: "AVGO", NETFLIX: "NFLX", AMD: "AMD", "ADVANCED MICRO": "AMD", INTEL: "INTC",
  PALANTIR: "PLTR", MICRON: "MU", ORACLE: "ORCL", SALESFORCE: "CRM", ADOBE: "ADBE", JPMORGAN: "JPM", "JP MORGAN": "JPM",
  "BANK OF AMERICA": "BAC", BERKSHIRE: "BRK.B", "ELI LILLY": "LLY", LILLY: "LLY", COSTCO: "COST", WALMART: "WMT",
  EXXON: "XOM", CHEVRON: "CVX", COINBASE: "COIN", UBER: "UBER", DISNEY: "DIS", BOEING: "BA", QUALCOMM: "QCOM",
  "S&P": "SPY", "S&P 500": "SPY", NASDAQ: "QQQ", "RUSSELL": "IWM", "TAIWAN SEMI": "TSM", TSMC: "TSM", ARM: "ARM",
  SUPERMICRO: "SMCI", "SUPER MICRO": "SMCI", VISA: "V", MASTERCARD: "MA", "UNITEDHEALTH": "UNH", PFIZER: "PFE",
});
const STOP = new Set(["I", "A", "AN", "THE", "AND", "OR", "OF", "TO", "IN", "ON", "IS", "IT", "AM", "AI", "ME", "MY", "DO",
  "BE", "AT", "BY", "FOR", "WHAT", "HOW", "WHY", "YOU", "ARE", "BUY", "SELL", "HOLD", "TODAY", "NOW", "THINK", "ABOUT",
  "OK", "US", "VS", "ETF", "PE", "EPS", "FCF", "ATR", "RSI", "SMA", "CEO", "IPO", "Q", "GO", "SO", "UP", "ALL", "ANY",
  "CAN", "SHOULD", "WILL", "STOCK", "STOCKS", "FLAG", "FLAGS", "RED", "GREEN", "YELLOW", "HELP", "TELL", "GIVE", "WITH",
  "WE", "NEED", "MORE", "DATA", "GOOD", "MODEL", "IF", "NOT", "BUT", "THIS", "THAT", "THAN", "THEN", "WHEN", "FROM",
  "HAS", "HAD", "WAS", "WERE", "BEEN", "HAVE", "OUR", "YOUR", "THEY", "THEM", "ITS", "AS", "NO", "YES", "GET", "GOT",
  "NEW", "OLD", "BEST", "JUST", "ALSO", "ONLY", "EVEN", "VERY", "MUCH", "MANY", "SOME", "SUCH", "INTO", "OVER", "UNDER",
  "AFTER", "DOES", "DID", "MAY", "MUST", "LIKE", "WANT", "MAKE", "MADE", "TAKE", "USED", "USE", "THESE", "THOSE", "WHICH",
  "WHILE", "WHERE", "THERE", "HERE", "EACH", "BOTH", "MOST", "LESS", "SAME", "OTHER", "FIRST", "LAST", "NEXT", "BACK",
  "WELL", "STILL", "BEING", "DOING", "GOING", "KNOW", "SAID", "SAY", "SAYS", "TOLD", "ASK", "ASKS", "BASED", "USING",
  "THEIR", "COULD", "WOULD", "MIGHT", "SHALL", "ONTO", "UPON", "PER", "VIA", "OFF", "OUT", "WHO", "WHOM", "ONCE", "EVERY",
  "NEVER", "YET", "NOR", "TOO", "LOT", "BIT", "BIG", "BAD", "FEW", "OWN", "SURE", "TRUE", "FULL", "PLUS", "DOWN", "LEFT",
  "RIGHT", "AGAIN", "QUITE", "SINCE", "UNTIL", "ABOVE", "BELOW", "AHEAD", "ALONG", "AMONG"]);

/** Help is only for empty input, /help, or a bare "help" / "?". General questions must not fall through to the help text. */
export function isHelpRequest(text) {
  const t = String(text || "").trim();
  return !t || /^\/help\b/i.test(t) || /^(help|\?)$/i.test(t);
}

export function conversationContext(history) {
  if (!Array.isArray(history)) return undefined;
  const turns = history.slice(-8).filter((turn) => ["user", "assistant"].includes(turn?.role) && typeof turn.content === "string")
    .map(({ role, content }) => ({ role, content: content.trim().slice(0, 1200) })).filter((turn) => turn.content);
  return turns.length ? "Prior turns in this chat (context for the latest request only; never repeat or approve earlier actions):\n" + JSON.stringify(turns) : "";
}

export function fallbackIntent(text, has = () => true) {
  const upper = ` ${text.toUpperCase()} `;
  const tickers = [];
  for (const [name, t] of Object.entries(NAMES)) if (upper.includes(name)) tickers.push(t);
  for (const m of text.matchAll(/\$?\b([A-Z]{1,5}(?:\.[A-Z])?)\b/g)) {
    const t = m[1];
    if ((m[0].startsWith("$") || !STOP.has(t)) && has(t)) tickers.push(t);
  }
  const unique = stripCoinTickers([...new Set(tickers)], text);   // BTC/ETH/SOL in a crypto question are coins, not stocks
  const l = text.toLowerCase();
  let intent = "NONE";
  if (/^\s*(help|\?)\s*$/.test(l)) intent = "HELP";
  else if (/\b(flag|screen|red ?flag|green ?flag)\b/.test(l)) intent = "FLAG_TICKERS";
  else if (isCryptoQuestion(text) && !unique.length) intent = "GENERAL";   // crypto -> analyst with [CRYPTO] + CRYPTO_STATUS
  else if (/\b(should (i|we|the (model|trader)) (buy|sell|add|trim)|entry|stop|position size|decision)\b/.test(l) && unique.length) intent = "DECISION";
  else if (/\b(portfolio|book|holdings|positions|diversif)/.test(l)) intent = "PORTFOLIO";
  else if (unique.length > 1 && /\b(vs\.?|versus|compare|or)\b/.test(l)) intent = "COMPARE";
  else if (unique.length) intent = "ASK_STOCK";
  else if (/\b(market|spy|regime|s&p)\b/.test(l)) intent = "MARKET";
  else if (text.trim()) intent = "GENERAL";
  return { intent, tickers: unique, question: text.trim(), source: "fallback" };
}

export function parseSlash(text) {
  const m = text.trim().match(/^\/(\w+)\s*(.*)$/s);
  if (!m) return null;
  const args = m[2].split(/[\s,]+/).filter(Boolean).map((a) => a.toUpperCase());
  const map = { ask: "ASK_STOCK", flag: "FLAG_TICKERS", decide: "DECISION", decision: "DECISION", data: "DATA",
    portfolio: "PORTFOLIO", market: "MARKET", help: "HELP", compare: "COMPARE", verdicts: "VERDICTS",
    note: "NOTE", memory: "MEMORY", recall: "MEMORY", tool: "TOOL", run: "TOOL" };
  const intent = map[m[1].toLowerCase()] || "HELP";
  if (intent === "TOOL") return { intent, tickers: [], question: text.trim(), source: "slash" };
  if (intent === "NOTE" || intent === "MEMORY") return { intent, tickers: args.filter((a) => /^[A-Z]{1,5}(\.[A-Z])?$/.test(a)), question: text.trim(), source: "slash" };
  if (intent === "ASK_STOCK" || intent === "COMPARE") {
    const words = m[2].trim().split(/\s+/);
    return { intent, tickers: [words[0].toUpperCase()], question: m[2].trim(), source: "slash" };
  }
  return { intent, tickers: args, question: text.trim(), source: "slash" };
}

export async function classify(text, { models, prices, sources, fetchImpl, sleep, conversation } = {}) {
  text = String(text || "").trim();
  if (!text) return { intent: "NONE", tickers: [], question: "", source: "empty" };
  const slash = parseSlash(text);
  if (slash) return slash;
  const has = (t) => (prices ? prices.has(t) : true);
  const fallback = fallbackIntent(text, has);
  let result = fallback;
  if (models?.current().ready) {
    try {
      const r = await complete({ models, name: "commandIntent", description: "Classify a trading-model command.",
        parameters: INTENT, temperature: 0, timeoutMs: 15000, fetchImpl, sleep,
        messages: [{ role: "system", content: INTENT_SYSTEM }, { role: "user", content:
          (conversation ? conversation + "\n\n" : "") + "Latest user input: " + JSON.stringify(text) }] });
      const intent = INTENTS.includes(r.value.intent) ? r.value.intent : "NONE";
      const tickers = [...new Set(r.value.tickers.map((t) => String(t).toUpperCase().replace(/^\$/, "").trim()).filter((t) => /^[A-Z]{1,5}(\.[A-Z])?$/.test(t)))];
      result = { intent: intent === "NONE" ? fallback.intent : intent, tickers: tickers.length ? tickers : fallback.tickers,
        question: r.value.question || text, source: "model" };
    } catch { result = fallback; }
  }
  result.tickers = stripCoinTickers(result.tickers, text);
  if (isCryptoQuestion(text) && !result.tickers.length && ["ASK_STOCK", "COMPARE", "MARKET", "PORTFOLIO", "DECISION", "HELP", "NONE"].includes(result.intent)) {
    result = { ...result, intent: "GENERAL" };
  }
  // Validate tickers against local history; unknown names go through Massive's ticker search.
  const checked = [];
  for (const t of result.tickers) {
    if (STOP.has(t)) continue;
    if (has(t) || has(t.replace(".", ""))) { checked.push(t); continue; }
    try {
      const hits = (await sources.search(t)).value;
      if (hits?.[0]?.ticker) checked.push(hits[0].ticker);
    } catch { /* drop unknown */ }
  }
  return { ...result, tickers: [...new Set(checked)] };
}

export const HELP = `Ask anything in plain English, e.g. "What do you think of Nvidia today?", "Compare AMD and AVGO", or "Is overextension a good model?".
Commands:
  /ask NVDA <question>    informed answer using live Schwab quote, local history, SEC financials, Massive news
  /flag [TICKERS]         red/yellow/green verdicts (default: paper-trader targets) -> advisory_verdicts.json
  /decide TICKERS         what the paper trader's advisory hook would do (entry band, stop, size scale)
  /data TICKER            raw research bundle (no model call)
  /verdicts               latest stored verdicts
  /note NVDA <text>       save a note to the analyst's memory (ticker optional)
  /memory NVDA <query>    show what the analyst remembers (MiniLM recall, fixed budget)
  /tool TOOL[/METHOD] TICKERS key=value   run a computation directly, e.g. /tool STATS/NORMALITY NVDA horizons=[1,5,20],
                          /tool RUN_BACKTEST/TSMOM max_positions=5, /tool RISK NVDA horizon=5, /tool VOL_FORECAST NVDA
                          (tools: RUN_BACKTEST WALK_FORWARD DAYTRADE_BACKTEST RULE_STUDY PAPER_STATUS STATS ECONOMETRICS
                          VOL_FORECAST INDICATORS RISK SIZING MONTE_CARLO PORTFOLIO_OPT MEMORY_SEARCH RESEARCH)
  /market                 market regime
  /help                   this text`;
