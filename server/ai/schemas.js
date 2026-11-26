// Retort parameter definitions (retort-js schema format: { type, enum, description, nullable }).
// No maxLength / character limits anywhere: the analyst's text fields are never truncated.
import * as E from "./enums.js";

const text = (description) => ({ type: String, description });
const pick = (values, description) => ({ type: String, enum: [...values], description });
const price = (description) => ({ type: Number, nullable: true, description });

const CITATION = { type: {
  source: pick(E.SOURCE, "Which supplied data block this point comes from."),
  asOf: text("Date or timestamp of that data exactly as given in the context (YYYY-MM-DD or ISO time)."),
  detail: text("The specific figure or headline being cited."),
} };

export const VERDICT = {
  ticker: text("The ticker being judged, exactly as given."),
  flag: pick(E.FLAG, "GREEN = data supports owning/adding now; YELLOW = mixed or wait; RED = material red flag, do not add."),
  conviction: pick(E.CONVICTION, "How strongly the supplied evidence supports the flag."),
  fundamentalHealth: pick(E.FUNDAMENTAL_HEALTH, "Revenue trend, margins, free cash flow, debt and cash. UNKNOWN if financials are missing."),
  trend: pick(E.TREND, "Price trend over 50-200 sessions from the supplied features."),
  overextension: pick(E.OVEREXTENSION, "Stretch versus moving averages / ATR / RSI from the supplied features."),
  opportunity: pick(E.OPPORTUNITY, "Buying-opportunity pattern present now, or NONE."),
  action: pick(E.ACTION, "Suggested action for a long-only paper account."),
  horizon: pick(E.HORIZON, "Holding period the suggestion applies to."),
  summary: text("Plain-English rationale grounded in the supplied numbers and dated news. Any length."),
  entry: { type: {
    orderType: pick(E.ORDER_TYPE, "How to work an entry; NONE if no entry is advised."),
    limitPrice: price("Suggested limit price near a trading threshold (support, moving average, ATR band); null if none."),
    bandLow: price("Lowest price at which a new entry still makes sense; null if no entry."),
    bandHigh: price("Highest price at which a new entry still makes sense (above it the name is too stretched); null if no entry."),
    rationale: text("Which thresholds the band is built from."),
  } },
  stop: { type: {
    price: price("Protective stop for a long position; null only if no position is advised."),
    atrMultiple: price("Stop distance in ATR(14) units."),
    rationale: text("Why this level (ATR, swing low, moving average)."),
  } },
  takeProfit: { type: {
    price: price("Level to trim or take profit; null if none."),
    rationale: text("Why this level."),
  } },
  sizeHint: pick(E.SIZE_HINT, "Fraction of the risk engine's allowed size to use. The risk engine may still reduce it."),
  diversification: { type: {
    status: pick(E.DIVERSIFICATION, "Effect on the current paper book."),
    note: text("Sector / correlation reasoning."),
  } },
  catalysts: { type: [{ type: {
    kind: pick(E.CATALYST_KIND, "Catalyst type."),
    impact: pick(E.IMPACT, "Likely price impact."),
    date: text("Date of the item as given (YYYY-MM-DD) or 'upcoming'."),
    headline: text("What happened or is scheduled."),
  } }], description: "Dated catalysts from news / press releases / earnings, newest first. Empty if none." },
  risks: { type: [{ type: {
    kind: pick(E.RISK_KIND, "Risk type."),
    severity: pick(E.SEVERITY, "Severity."),
    detail: text("The specific risk and the figure behind it."),
  } }], description: "Material risks; empty if none." },
  dataGaps: { type: [String], description: "Data that was missing or stale and could change the view." },
  citations: { type: [CITATION], description: "The data points the verdict relies on." },
};

export const ANSWER = {
  answer: text("Full answer to the user's question in Markdown. Cite figures with their dates. Any length; be thorough."),
  stance: pick(E.FLAG, "Overall stance on the main ticker: GREEN, YELLOW or RED."),
  conviction: pick(E.CONVICTION, "Confidence in that stance."),
  keyPoints: { type: [String], description: "Short bullet takeaways." },
  levels: { type: [{ type: {
    label: text("e.g. 'support (50-day MA)', 'stop', 'entry band high'."),
    price: { type: Number, description: "Price level." },
  } }], description: "Key price levels mentioned in the answer." },
  citations: { type: [CITATION], description: "Data points used." },
  followUps: { type: [String], description: "Useful follow-up questions or commands." },
};

export const INTENT = {
  intent: pick(E.INTENT, "ASK_STOCK: question about one stock; COMPARE: two or more stocks; FLAG_TICKERS: run red/green flags; DECISION: what the paper trader should do; PORTFOLIO: about the paper book; MARKET: broad market; GENERAL: a non-ticker question about the model, research, regime, hedges or similar; HELP: how to use the prompt; NONE if empty or unclear."),
  tickers: { type: [String], description: "US ticker symbols the user refers to (resolve company names, e.g. Nvidia -> NVDA). Empty if none." },
  question: text("The user's question restated clearly."),
};

// One step of the tool loop. Every field is required (strict schema); unused fields take DEFAULT / NONE / "" / [].
export const TOOL_REQUEST = { type: {
  tool: pick(E.TOOL, "Which server-side tool to run."),
  method: pick(E.TOOL_METHOD, "Sub-method for STATS / ECONOMETRICS / VOL_FORECAST / MONTE_CARLO / PORTFOLIO_OPT / RULE_STUDY; DEFAULT otherwise."),
  family: pick(E.STRATEGY_FAMILY, "Strategy family for RUN_BACKTEST / WALK_FORWARD; NONE otherwise."),
  tickers: { type: [String], description: "Ticker symbols the tool works on (may be empty)." },
  start: text("Start date YYYY-MM-DD, or empty for all local history."),
  end: text("End date YYYY-MM-DD, or empty for the latest data."),
  horizonDays: { type: Number, nullable: true, description: "Return horizon / VaR horizon / forecast steps / Monte Carlo days; null for the default." },
  options: { type: [{ type: {
    name: text("Option or strategy parameter name exactly as listed in the tool catalog (e.g. max_positions, entry_z, window, weights)."),
    value: text("Value as text or JSON (numbers, lists like [50,100], objects like {\"NVDA\":0.5})."),
  } }], description: "Tool options and strategy parameter overrides; empty for defaults." },
  query: text("Search text for MEMORY_SEARCH; empty otherwise."),
  purpose: text("One line: what this call should establish for the answer."),
} };

export const TOOL_STEP = {
  decision: pick(E.TOOL_STEP, "CALL_TOOLS to run 1-4 tools now, ANSWER when the evidence is sufficient (or tools cannot help)."),
  reasoning: text("Short plan: what is still unknown and why these tools answer it."),
  calls: { type: [TOOL_REQUEST], description: "1-4 tool calls when decision is CALL_TOOLS; empty when ANSWER." },
};
