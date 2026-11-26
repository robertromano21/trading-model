# AI analyst (server/ai)

Advisory research layer for the trading model, built on the Retort (retort-js) prompt framework. It answers questions
("What do you think of Nvidia today?"), red/yellow/green-flags targets from filings, press releases, news and price
data, and hands the paper trader a small verdict file. The deterministic risk engine stays final: verdicts can only
reduce or defer risk, and a failed model call never changes trading.

## Pieces
| File | Role |
|---|---|
| enums.js | Retort enums: FLAG, CONVICTION, TREND, OVEREXTENSION, ACTION, ORDER_TYPE, SIZE_HINT, OPPORTUNITY, CATALYST_KIND, RISK_KIND, DIVERSIFICATION, INTENT ... |
| schemas.js | Retort parameter schemas VERDICT / ANSWER / INTENT. No maxLength anywhere. |
| retortSchema.js | Retort -> strict JSON schema, enum $defs compaction, local `matches`/`mismatch` validation |
| models.js | Model catalog (GPT-6 Luna default, GPT-6 Sol, GPT-5.6, GPT-4.1, Grok 4.20 non-reasoning, Groq) and provider-key storage |
| transport.js | Forced function-call structured output, reasoning off, no output cap, length/refusal/empty/schema failures, 1+3 attempts with backoff, per-call deadline |
| runtime.js | Retort conversation ($.system / $.user template prompts) routed through transport |
| command.js | Command prompt: slash commands + LLM-first intent with deterministic fallback |
| sources.js, schwab_snapshot.py | Massive news/press releases, ticker details, dividends, splits; SEC XBRL financials; Schwab quote + fundamentals (read-only client) |
| features.js, context.js | Price features from Massive/prices.sqlite3, regime, paper-book diversification; budgeted prompt context |
| advisor.js | research -> verdict/answer -> code clamps -> store -> advisory_verdicts.json; decision API |
| routes.js, cli.js | /api/ai/* routes and the command-line entry |

## Keys
Open http://localhost:3003/ai.html -> **API keys…**, choose OpenAI / xAI / Groq and paste the key. It is written to
`keys/<provider>.key` (gitignored), loads automatically on the next start, is never logged and never sent back to the
browser. An environment key (OPENAI_API_KEY / XAI_API_KEY / GROQ_API_KEY, e.g. in .env.local) keeps priority.
Optional: TRADING_AI_KEY_DIR (other key folder), TRADING_AI_MODEL, TRADING_AI_TARGETS, TRADING_AI_TIMEOUT_MS,
TRADING_AI_CONTEXT_CHARS, TRADING_AI_MAX_AGE_DAYS, SEC_USER_AGENT ("Name email" recommended by the SEC).
Data keys already used by the model: MASSIVE_API_KEY; Schwab token file + SCHWAB_APP_KEY / SCHWAB_APP_SECRET for refresh.

## Paper-trader contract (read by engine/advisory.py)
`Massive/Research/advisory_verdicts.json` (override ADVISORY_VERDICTS_PATH), rewritten atomically after each flag run:
```json
{"schema": "advisory_verdicts/v1", "generatedAt": "2026-11-25T23:59:00Z", "asof": "2026-11-25",
 "verdicts": {"NVDA": {"flag": "GREEN|YELLOW|RED", "conviction": 0.35|0.6|0.85, "entry_low": 146.1, "entry_high": 151.9,
   "stop": 139.2, "overextension": -0.42, "asof": "2026-11-25", "source": "ai:gpt-6-luna",
   "action": "BUY", "size_hint": "HALF", "size_scale": 0.5, "limit_price": 149.5, "order_type": "LIMIT",
   "take_profit": 170, "horizon": "POSITION_20_60D", "trend": "UP", "overextension_class": "NEUTRAL",
   "opportunity": "PULLBACK_IN_UPTREND", "fundamental_health": "STRONG", "diversification": "OK",
   "ref_price": 151.4, "atr": 5.9, "created_at": "..."}}}
```
* `overextension` (number) = z-score of price vs its 20-day mean; `overextension_class` is the enum.
* Only model (LLM) verdicts younger than TRADING_AI_MAX_AGE_DAYS (3) are exported. FALLBACK verdicts are kept in
  `Massive/ai_research.sqlite3` (table ai_verdicts) for display but never exported, so a model failure = no change.
* engine/advisory.py applies: RED -> no new entry/increase; outside [entry_low, entry_high] -> entry deferred;
  YELLOW -> size x yellow_scale; GREEN -> unchanged (never enlarged). Extra fields are informational.
* Code clamps every model verdict before storage: stop must be 0.5-6 ATR below price; band inside [p-4ATR, p+1.5ATR]
  and ordered; RED/AVOID/EXIT -> no band; RED+BUY -> WATCH; diversification cap -> WATCH and size <= HALF;
  overextension enum replaced by the code's if off by 2+ steps; GREEN -> YELLOW if price data is > 7 days old.

## Use
* UI: http://localhost:3003/ai.html (command prompt, model + key picker, verdict table, Flag targets).
* API: POST /api/ai/command {text}; POST /api/ai/ask {question, tickers?}; POST /api/ai/flag {tickers?} + GET /api/ai/flag/:id;
  GET /api/ai/verdicts; GET|POST /api/ai/decision; GET /api/ai/research/:ticker; GET|POST /api/ai/models; POST /api/ai/provider-key.
* CLI (from the project folder): `node server/ai/cli.js flag NVDA AMD`, `node server/ai/cli.js ask "What do you think of Nvidia today?"`,
  `node server/ai/cli.js decide NVDA`, `node server/ai/cli.js research NVDA`, `node server/ai/cli.js models`.
* Tests: `cd server && node --test ai/test/ai.test.js` (mocked model and data; no network).

## Analyst memory (MiniLM, local)

Semantic recall (embedding worker + history recall + lexical fallback).
Model: `sentence-transformers/all-MiniLM-L6-v2` quantized ONNX (384-d), run by `onnxruntime-node` in a worker thread,
files in `models/memory/` (gitignored; `node server/ai/memory/install-model.js` downloads + checksums them).
Store: `Massive/ai_memory.sqlite3` (gitignored) — `memory_items` + `memory_chunks` (420-char passages, float32 vectors).

Kinds: `QA, CHAT, VERDICT, DECISION, NEWS, FUNDAMENTALS, NOTE, EVENT` (recency half-lives 30/21/45/60/7/120/365/30 days).

Prompt block (`[MEMORY]`) has a FIXED total budget of 10,000 chars (~2,500 tokens; `BUDGET_CHARS`):
per-ticker track record — latest `TRACK_RECORD_VERDICTS` = 15 verdicts + `TRACK_RECORD_DECISIONS` = 3 decisions per
ticker as compact lines (`- 2026-11-18 YELLOW/WATCH 160.00 -> 155.00 (-3.1%): one-line reason`, ~140 chars), ≤4,000
chars split evenly across tickers (env overrides `TRADING_AI_TRACK_RECORD_VERDICTS` / `_DECISIONS`) + last 3 Q&A (≤2,000 chars) +
older items ranked by cosine relevance + 0.15·recency + ticker boost, near-duplicates (cos > 0.92) dropped.
Other items with `meta.refPrice` render as `-> then 160.00, now 155.00 (-3.1%)`; older relevant items always keep ≥4,000 chars. Semantic search times out after 2.5 s
and falls back to lexical; recall never throws.

### Writing to memory (for the paper trader / chat log)

JS (same process as the server — shares the singleton):
```js
import { getMemory } from "./ai/memory/memory.js";
const memory = getMemory({ dataDir: "<root>/Massive" });
memory.ingest({ kind: "DECISION", ticker: "NVDA", text: "BUY 12 @ 151.20 (TSMOM sleeve, ATR stop 142.8)",
  ts: new Date().toISOString(), meta: { refPrice: 151.2, dedupeKey: "order:<id>" } }); // -> { id, duplicate }
```
Synchronous, deduped on `kind|ticker|dedupeKey-or-text`; embedding happens in the background.

HTTP: `POST /api/ai/memory/ingest` with one item or `{ "items": [ ... ] }` (≤200).
Python (engine side, no Node needed): 
```python
from server.ai.memory_ingest import ingest   # or sys.path-insert server/ai
ingest("DECISION", "BUY 12 NVDA @ 151.20", ticker="NVDA", meta={"refPrice": 151.2, "dedupeKey": "order:1"},
       db_path="Massive/ai_memory.sqlite3")
```
Python rows are stored unembedded; Node embeds them within 60 s (or `POST /api/ai/memory/reindex`).

Read: `GET /api/ai/memory/status`, `GET /api/ai/memory/search?q=&tickers=NVDA&kinds=VERDICT,DECISION&since=&budget=`,
`GET /api/ai/memory/items?ticker=&kind=&limit=`. Chat: `/note NVDA <text>`, `/memory NVDA <query>`.
CLI: `node server/ai/cli.js memory NVDA earnings`, `memory-status`, `reindex`.

Automatic ingestion: research → NEWS (≤25, dedupe by url) + FUNDAMENTALS (per SEC quarter);
LLM verdicts → VERDICT; LLM answers → QA. FALLBACK outputs are never stored.

## Master system prompt (generated, stays in sync)
`advisor.system()` = grounding rules (`prompts.js`) + **TRADING MODEL REFERENCE** rendered by `strategyCard.js` from
`engine/quant_tools.py strategy_card`. That card is read from the code itself: the strategy classes (docstrings and default
parameters), `ORBParams`, paper-trader defaults, risk/circuit-breaker and Schwab cost settings, research grids, plus the
measured results in `Massive/Research/{research_summary,daytrade_summary,rule_studies}.json`. It is cached in
`Massive/ai_strategy_card.json` and rebuilt when any of those files changes (or after 12 h). Sections: STRATEGIES (the live
TSMOM / mean-reversion / combined sleeves and ORB day trading, then candidate rules that are NOT in the engine (SPY 200d
filter, RSI(2) dip, Donchian breakout, 12-1/6-1 cross-sectional momentum, pairs z-score) with their quick-study results),
MEASURED RESULTS (in-sample + walk-forward OOS), RISK RULES, COSTS, NORMALITY BY HORIZON.
`node server/ai/cli.js prompt [--refresh]`, `GET /api/ai/prompt[?refresh=1]`, `POST /api/ai/strategy-card/refresh`.
Regenerate the candidate-rule studies with `.venv\Scripts\python engine\quant_tools.py rule-studies`.

## Tools (computations the analyst can run)
`advisor.ask` runs a bounded tool loop before answering: the model returns `analystToolStep` (Retort enums
`TOOL_STEP` CALL_TOOLS/ANSWER, `TOOL`, `TOOL_METHOD`, `STRATEGY_FAMILY`; options as name/value text, no length limits),
the server runs up to 4 calls per step (duplicates dropped), feeds compact summaries back as `[TOOL_RESULTS]`, and after
at most `TRADING_AI_TOOL_STEPS` (default 4) steps or `TRADING_AI_TOOL_BUDGET_MS` (default 20 min) produces the normal
`analystAnswer`. A planner failure or missing key just skips tools; the deterministic risk engine is unaffected.
Tools: RUN_BACKTEST, WALK_FORWARD, DAYTRADE_BACKTEST, RULE_STUDY, PAPER_STATUS, STATS (describe / normality /
bootstrap / t-test / correlation / Sharpe CI + deflated Sharpe), ECONOMETRICS (OLS+HAC, rolling OLS, factor betas,
ADF/KPSS, Engle-Granger, Johansen, ARIMA, half-life/Hurst), VOL_FORECAST (GARCH(1,1), EWMA), INDICATORS, RISK
(VaR/CVaR normal/t/historical), SIZING (Kelly, vol target), MONTE_CARLO, PORTFOLIO_OPT (risk parity, min variance, max
Sharpe, mean-variance; Ledoit-Wolf), MEMORY_SEARCH, RESEARCH. Python tools run `engine/quant_tools.py` (JSON in/out, a
fixed list of operations, no code execution) with the `.venv` interpreter; results are cached 6 h in `source_cache`
and logged in full to `ai_log` (kind `tool`; loop steps are kind `tool-step`).
Direct use: `/tool STATS/NORMALITY NVDA horizons=[1,5,20]` in the chat, `node server/ai/cli.js tool RISK NVDA horizon=5`,
`GET /api/ai/tools`, `POST /api/ai/tools/run {"tool":"RUN_BACKTEST","family":"TSMOM","options":{"max_positions":5}}`.
Python deps: `scipy`, `statsmodels`, `arch` (engine/requirements.txt).
