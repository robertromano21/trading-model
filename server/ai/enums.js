// Retort enums for the AI analyst layer. Every model answer is constrained to these choices by the
// provider's strict function schema AND re-checked locally (retortSchema.matches) before it is used.
// Intent pattern: INTENTS enum + safe "none" default.
import { ACTION_IDS } from "./actionRegistry.js";

const freeze = (values) => Object.freeze([...values]);

export const FLAG = freeze(["GREEN", "YELLOW", "RED"]);
export const CONVICTION = freeze(["LOW", "MEDIUM", "HIGH"]);
export const TREND = freeze(["STRONG_UP", "UP", "SIDEWAYS", "DOWN", "STRONG_DOWN"]);
export const OVEREXTENSION = freeze(["DEEPLY_OVERSOLD", "OVERSOLD", "NEUTRAL", "EXTENDED", "OVEREXTENDED"]);
export const FUNDAMENTAL_HEALTH = freeze(["STRONG", "ADEQUATE", "WEAK", "DISTRESSED", "UNKNOWN"]);
export const ACTION = freeze(["BUY", "ADD", "HOLD", "TRIM", "EXIT", "AVOID", "WATCH"]);
export const HORIZON = freeze(["INTRADAY", "SWING_1_10D", "POSITION_20_60D", "TREND_60_200D"]);
export const ORDER_TYPE = freeze(["LIMIT", "MARKET_ON_OPEN", "MARKET_ON_CLOSE", "NONE"]);
export const SIZE_HINT = freeze(["NONE", "QUARTER", "HALF", "FULL"]);
export const OPPORTUNITY = freeze(["NONE", "PULLBACK_IN_UPTREND", "OVERSOLD_REVERSAL", "BREAKOUT",
  "POST_EARNINGS_DRIFT", "VALUE_DISLOCATION"]);
export const CATALYST_KIND = freeze(["EARNINGS", "GUIDANCE", "PRODUCT", "REGULATORY", "LEGAL", "M_AND_A",
  "CAPITAL_RETURN", "DEBT_FINANCING", "MANAGEMENT", "ANALYST", "MACRO", "OTHER"]);
export const IMPACT = freeze(["POSITIVE", "NEGATIVE", "MIXED", "NEUTRAL"]);
export const RISK_KIND = freeze(["VALUATION", "LEVERAGE", "LIQUIDITY", "DILUTION", "CONCENTRATION", "VOLATILITY",
  "GAP_RISK", "EARNINGS_EVENT", "LEGAL_REGULATORY", "DEMAND", "MARGIN", "DATA_QUALITY", "OTHER"]);
export const SEVERITY = freeze(["LOW", "MEDIUM", "HIGH"]);
export const DIVERSIFICATION = freeze(["OK", "SECTOR_HEAVY", "CORRELATED_WITH_BOOK", "WOULD_BREACH_CAP"]);
export const SOURCE = freeze(["PRICE_HISTORY", "SCHWAB_QUOTE", "SCHWAB_FUNDAMENTAL", "MASSIVE_NEWS", "SEC_FINANCIALS",
  "DIVIDENDS", "SPLITS", "TICKER_DETAILS", "PAPER_BOOK", "MARKET_REGIME", "TOOL_RESULT", "MEMORY", "STRATEGY_REFERENCE"]);
export const INTENT = freeze(["ASK_STOCK", "COMPARE", "FLAG_TICKERS", "DECISION", "PORTFOLIO", "MARKET", "GENERAL", "HELP", "NONE"]);
// Tool loop (analystToolStep): the model picks tools from these enums; tools.js maps them to fixed server operations.
export const TOOL_STEP = freeze(["CALL_TOOLS", "ANSWER"]);
export const TOOL = freeze(["RUN_BACKTEST", "WALK_FORWARD", "DAYTRADE_BACKTEST", "RULE_STUDY", "PAPER_STATUS", "STATS",
  "ECONOMETRICS", "VOL_FORECAST", "INDICATORS", "RISK", "SIZING", "MONTE_CARLO", "PORTFOLIO_OPT", "MEMORY_SEARCH", "RESEARCH", "MACRO", "MACRO_CALENDAR", "OPTIONS", "ATTRIBUTION", "PROB_MODEL", "BLACK_SWAN", "MACRO_NOWCAST", "CRYPTO_STATUS"]);
export const TOOL_METHOD = freeze(["DEFAULT",
  "DESCRIBE", "NORMALITY", "BOOTSTRAP", "T_TEST", "CORRELATION", "SHARPE_INFERENCE",                      // STATS
  "OLS", "ROLLING_OLS", "FACTOR_BETAS", "ADF_KPSS", "ENGLE_GRANGER", "JOHANSEN", "ARIMA", "HALF_LIFE_HURST", // ECONOMETRICS
  "GARCH", "EWMA",                                                                                         // VOL_FORECAST
  "VAR_CVAR",                                                                                              // RISK
  "BOOTSTRAP_PATHS", "NORMAL_PATHS", "STUDENT_T_PATHS",                                                    // MONTE_CARLO
  "RISK_PARITY", "MIN_VARIANCE", "MAX_SHARPE", "MEAN_VARIANCE", "EQUAL_WEIGHT",                            // PORTFOLIO_OPT
  "SPY_200D_FILTER", "RSI2_DIP", "DONCHIAN_BREAKOUT", "XS_MOMENTUM_12_1", "XS_MOMENTUM_6_1", "PAIRS_ZSCORE", // RULE_STUDY
  "SNAPSHOT", "LEADLAG", "PREDICT", "BETAS", "OUTLOOK", // MACRO
  "CHAIN", "GREEKS", "PAYOFF", "POP", "EXPECTED_MOVE", "TOP_RISK", "VOL_REGIME"]); // OPTIONS
export const STRATEGY_FAMILY = freeze(["NONE", "TSMOM", "MEANREV", "COMBINED"]);
export const VERDICT_SOURCE = freeze(["LLM", "FALLBACK"]);

// Numeric mappings used by the paper-trader contract (engine/advisory.py expects conviction 0..1).
export const CONVICTION_SCORE = Object.freeze({ LOW: 0.35, MEDIUM: 0.6, HIGH: 0.85 });
export const SIZE_SCALE = Object.freeze({ NONE: 0, QUARTER: 0.25, HALF: 0.5, FULL: 1 });

// Action adjudication (server/ai/actions.js): one closed enum of everything the analyst may do on the user's
// request (dashboard routes + read-only math, from actionRegistry.js) plus three meta answers. Human-only steps
// (real-money orders, the live switch, keys/secrets, Schwab sign-in) are deliberately NOT in it.
export const APP_ACTION = freeze([...ACTION_IDS, "ASK_ANALYST", "CLARIFY", "MANUAL_STEP"]);
export const ACTION_CONFIDENCE = freeze(["HIGH", "MEDIUM", "LOW"]);
export const MANUAL_TOPIC = freeze(["NONE", "LIVE_ORDER", "LIVE_TOGGLE", "API_KEYS", "SCHWAB_SIGNIN", "OTHER"]);
