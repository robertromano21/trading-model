# Testing checklist

A short guide for trying the new trading-model features on this PC. Times below are America/New_York (ET).

## Start the dashboard

In PowerShell, from the project folder (the one with `start-server.ps1`):

```powershell
.\start-server.ps1
```

Note the backslash (`.\`, not `.start-server.ps1`). The script opens http://localhost:3003 (the default port is now 3003, so it no longer collides with other apps on 3000), or reuses the dashboard already running there.

If you see `An older Prices Dashboard is running on port 3003`, stop the old one with Ctrl+C in its PowerShell window, then run `.\start-server.ps1` again. That restart is what loads the latest analyst, auto-shutdown, and paper-page changes.

## Schwab

The nav bar has the Schwab chip and a connect / reauthorize dialog (app key and secret are saved to `.env.local` on this PC; they are never sent back to the browser).

- Access tokens last about 30 minutes and refresh on their own if the app key and secret are saved.
- The refresh token lasts about 7 days. The current one is good until about **2026-12-02 22:53 ET**. Reauthorize before then or quotes, backfill retries, and the live ticket panel will stop working.

## 20-year backfill

This run has finished (status `partial` only because of a few tickers).

- **4,722** tickers, **14,503,137** daily rows in `prices.sqlite3` table `daily_long`
- Range: **2006-11-24** to **2026-11-25**
- 2,937 already had history and were skipped; 268 came back empty
- 6 HTTP 400 errors to retry later if you want: ITGR, CBZ, RKLZ, BIPC, LXP, UCFI

Schwab daily history is **split-adjusted, not dividend-adjusted**. Massive grouped bars (about the last two years) are split- and dividend-adjusted and are preferred on overlapping dates.

**Survivorship bias:** only symbols that are still listed today are in this file, so long backtests will look a bit more optimistic than a point-in-time universe. The ops page (`/ops.html`) shows this status and has a **Backfill 20-year daily history...** button if you want to retry the six failures. The backfill is resumable, runs as its own process, and survives a dashboard restart.

## API keys (optional, but needed for full answers)

- **AI analyst:** on `/ai.html`, click **API keys...** and paste an OpenAI, xAI, or Groq key. It is saved under `keys/<provider>.key` (gitignored) and is never sent back to the browser. An `OPENAI_API_KEY`, `XAI_API_KEY`, or `GROQ_API_KEY` already in `.env.local` takes priority. Without a key you still get data and tool output, but not a model-written answer.
- **`SEC_USER_AGENT`** in `.env.local` (for example `Your Name email@domain`): recommended so SEC share-count / dilution requests use your contact, per the SEC's fair-access rule.
- **FRED API key:** on `/ops.html#keys` (also inside the **Set up / reauthorize Schwab…** dialog and on `/macro.html`), paste the free key from https://fredaccount.stlouisfed.org/apikeys and click **Save key**. The server checks it with FRED first, saves it to `.env.local` as `FRED_API_KEY`, and only ever shows **set / not set**; **Test saved key** re-checks it server-side. All code reads `FRED_API_KEY`. With it, the macro import uses ALFRED first-release dates; without it, conservative publication lags.
  - Check: the status says **set**; reload the page and confirm the key box is empty (the key is never sent back); a wrong key shows "FRED rejected the key" and is not saved.

## Page-by-page click list

Alert toasts (decisions, fills, Schwab auth, black swan, macro releases, and so on) can appear on any of these pages.

1. **History / ADX** (`/`, `index.html`) — the usual session, ticker, and workbook tools.
2. **Charts** (`/ticker.html?symbol=NVDA`) — chart and indicators; **exhaustion badge** next to the symbol; exhaustion markers and score on the chart; **Walk-forward probabilities** card (advisory only); fundamentals, news, and the Buy/Sell ticket.
3. **Paper trading** (`/paper2.html`)
   - Risk and regime, plus the **macro / dollar (DXY) outlook** card (recommend only)
   - **Regime allocation (vol and overextension)**
   - **Geopolitical and black swan monitor** (alerts and log only by default)
   - **Economic calendar and releases** (times in ET)
   - **Macro nowcasts (ours, raw data only)** — next CPI / core CPI / core PCE / jobs / GDP prints with 80% bands, last print, implied surprise, drivers, outside estimates beside ours (display only) and ours-minus-outside
   - Equity, positions, orders, targets, fills, paper-vs-shadow reconciliation
   - Swarm: simulated $10k shadow accounts (no orders)
   - Day-trading paper book
   - **Live trading (Schwab, manual only)** — ticket, then Schwab preview, then type to Confirm. Off until you turn it on.
   - **Options, downside and volatility sleeve (paper)** — vol sleeve on, overlay account off
   - Probability model, attribution, tax lots, research summary
4. **Research and data** (`/ops.html`) — jobs, data health, Schwab status, 20-year backfill button and coverage, **Data keys** (FRED), logs (also `/ops.html#logs`). New jobs: **Macro: refresh FRED data + publish our nowcasts**, **Macro: nowcast walk-forward accuracy**, **Macro: regime filters vs plain TSMOM**, **Schwab: refresh long index history**.
5. **Macro nowcasts** (`/macro.html`, linked from Research and data) — forecast table, regime line, track record vs the last print and vs Cleveland Fed / GDPNow, and two charts (our final nowcast vs the official print for the last 18 releases; the current period's nowcast path with its band and outside estimates). Click **Refresh now**: it starts a detached refresh (3–5 minutes) and the table updates after. While the dashboard is up it also refreshes itself when the latest forecast is more than 6 hours old (`MACRO_NOWCAST_AUTO=0` turns that off). API: `GET /api/macro/nowcast`, `GET /api/macro/nowcast/history?target=CPI_MOM`.
6. **AI analyst** (`/ai.html`)
   - Ask a general question with no ticker, for example: `Is overextension a good model or do we need more data?`
   - Then a stock question, for example: `/ask NVDA What do you think today?` or plain `What do you think of Nvidia today?`
   - Ask `What do our nowcasts say about next month's CPI?` (uses the `[MACRO_NOWCAST]` block) or `/tool MACRO_NOWCAST target=CPI_MOM`.
   - Optional tools: `/tool STATS/NORMALITY NVDA horizons=[1,5,20]`, `/tool RUN_BACKTEST/TSMOM max_positions=5`, `/tool RISK NVDA horizon=5`, `/tool VOL_FORECAST NVDA`
   - `/help` lists the rest (`/flag`, `/decide`, `/data`, `/market`, ...). Every verdict is advisory; the analyst cannot place an order.

## Fresh clone: keys and voice models

Nothing secret is in the repository. On a new machine:

0. Install **Node.js 22.13 or newer** (24 LTS recommended; the server uses the built-in `node:sqlite`) and **Python 3.10+**, then run `.\start-server.ps1`. It creates `.venv`, installs `engine/requirements.txt` and the npm packages, and opens http://localhost:3003. Market data, research output and paper state all live in the gitignored `Massive/` folder, which starts empty.

1. Copy `.env.example` to `.env.local` (gitignored), or leave it empty and enter keys in the app.
2. **Massive** (market data) and **FRED** (macro): Research & data > **Data keys**, or the Schwab setup dialog on any page. Keys are masked, saved to `.env.local`, and only "set / not set" is ever shown.
3. **Schwab**: the setup dialog (Schwab chip in the top bar). The Schwab callback stays `https://127.0.0.1:8182`, which is separate from the dashboard port.
4. **AI analyst**: `/ai.html` > API keys (saved under the gitignored `keys/`), or `OPENAI_API_KEY` / `XAI_API_KEY` / `GROQ_API_KEY` in `.env.local`.
5. **Analyst memory model** (optional): `node server/ai/memory/install-model.js` downloads the small MiniLM model into the gitignored `models/memory/`.
6. **SID voice models**: download `en_US-ljspeech-high` and `en_US-ryan-high` (`.onnx` + `.onnx.json`) from rhasspy/piper-voices into `models/piper/`. See `server/voice/README.md`. Without them, the browser voice speaks instead.

## Command console (analyst chat + decision log) and voice alerts

After updating, hard-refresh each open dashboard tab once (Ctrl+F5). Later changes load by themselves because nav.js adds `?v=<version>` to the scripts it loads.

1. On every page, a black **Command log** panel with a white **Type a command** box sits in the lower right. If it's missing, open the browser console (F12). There should be no "Content Security Policy" error naming `analyst-dock.js` or `alerts.js`; the styles now come from `/analyst-dock.css` and `/alerts/alerts.css`.
2. Type "What do you think of Nvidia today?" and press Enter (or click **Command**). Shift+Enter adds a new line. Your line shows as `> ...` in white, then a gold **ANALYST** reply. Follow-up buttons copy their question into the box.
3. **History:** press Up/Down in the box to step through earlier commands (the last 50, kept across pages).
4. **One log, colors by type:** **MAIN** paper trades (green), **SWARM** (blue), **CRYPTO** (orange), **OPTIONS** (violet), **SIGNAL** flags/drift (teal), **SWAN** black swan (red), **DOLLAR** shock (orange-red), **REGIME** (amber), **MACRO** (khaki), **SYSTEM** (grey), errors (red background). Every line has a time (ET; hover for the full date). Recent alerts are filled in from the server when a page opens.
5. **Filters:** **All / Analyst / Trades / Signals / Risk / Macro / System** only change what the log shows; they don't mute anything.
6. **Autoscroll:** the log follows new lines. Scroll up and a **▼ new** button appears instead; click it to jump back down.
7. **Collapse:** **▾** in the log header, **Log ▾** under the box, or **Esc** hides the log and leaves only the prompt. **_** or a second **Esc** shrinks everything to a **Command ▴** tab, and a yellow dot on the tab means something new arrived. **Alt+Shift+A** brings the console back and focuses the box. The mode, filter and font stay as you left them across pages. You can drag the log's bottom-right corner to resize it. **Aa** switches the log between pixel and plain text. **Clear** (or `/clear`) empties the log.
8. **Pick-one replies:** when a symbol could mean two things (e.g. `BTC`, a coin or a stock), the reply lists numbered buttons. Click one, or press **1**/**2** with the box empty, and that choice is sent; the buttons then grey out. Reply types the console doesn't know show as plain text.
9. **Voice is on by default.** Browsers block sound until you click or press a key, so **Click to enable voice** shows in the box's top bar until you do. Anything waiting then plays. Alerts more than 10 minutes old are dropped from that queue, not spoken. **Voice on/off** in the log header is the master switch.
10. The analyst speaks a short summary of each reply (or the question, for a pick-one reply), not the tables or the full text. The subtitle bar shows what is being said, just above the console.
11. Trade decisions are spoken and logged as they land in the decision log: main paper orders, fills, stops and exits, plus options paper. Bursts are collapsed into one line, such as "Main paper logged 6 orders: ...". Also spoken and logged: crypto paper position changes and daily standings, swarm shadow updates, black swan and dollar-shock alerts, market regime changes, and meaningful macro nowcast moves. Nothing already in the logs when the dashboard starts is replayed.
12. Click **Mute…** in the log header. Check the master **Voice** switch, **Visual alerts**, voice engine (**SID chip** or **Browser**), volume, rate and quiet hours. Then check the **Speak** and **Show** columns for each category: analyst replies, main paper, swarm, crypto, options, flags/drift, black swan, dollar shock, regime, macro, and system. Use **Mute all speech** and **Unmute all** too. Settings persist per browser, and a copy is kept in `logs/voice/alert-prefs.json`. The old floating **Alerts** button is hidden while the console is on the page.
13. With two tabs open, each alert should be spoken only once (by the visible tab). The pop-up card and the log line can show in both.
14. To turn the server-side decision feed off, start the dashboard with `TM_VOICE_FEED=off`. Pop-ups from the engines still work.

## AI analyst actions (adjudicated prompts, Allow / Cancel, experiments, learning feed)

How it works: every console prompt (except the existing /slash commands) goes through one strict-JSON model call that
picks ONE action from a closed enum (`APP_ACTION` in server/ai/enums.js = every dashboard route action + every math
tool in server/ai/actionRegistry.js + ASK_ANALYST / CLARIFY / MANUAL_STEP), with typed args. The server re-checks the
enum and each argument (server/ai/actions.js `legalize`). Plain questions fall through to the normal analyst answer.
With no AI key (or a model error) a small rule-based fallback handles the headline prompts below.
Routes run over loopback through the exact handler the page uses, tagged `x-tm-origin: analyst_user_request`.

### Allow / Cancel prompt (all state-changing actions)
- [ ] `paper buy 10 SPY` shows an **Allow this change?** box: title + action id, what it affects ("PAPER BUY 10 SPY in experiment account 1000 at the latest price (no broker)"), the arguments, **Allow** / **Cancel**, and "expires h:mm".
- [ ] Nothing changes until you click **Allow** (check Research/experiments/accounts.json before and after).
- [ ] With an empty prompt, **Enter** = Allow, **Esc** = Cancel. A double-tapped Enter right after sending does not Allow (700 ms guard).
- [ ] **Cancel** -> "Cancelled. Nothing was changed."; the box shows "Cancelled — nothing changed".
- [ ] Leave one for 5 minutes -> it flips to "Expired — nothing changed"; Allow on an old token (e.g. via another tab) -> "That request expired".
- [ ] Reload the page with an open prompt -> it is still there (or expired); it never runs by itself.
- [ ] A token can be used once (second Allow -> expired). Read-only actions and math never show the box.

### Human-only (refused with a pointer to the UI, never called)
- [ ] `buy 10 SPY in my Schwab account` -> "Live Schwab orders are a manual step: …"
- [ ] `turn live trading on` -> manual step (Paper trading > Live trading switch).
- [ ] `set my FRED key to …` / `what is my xAI API key` -> manual step; no key is read or written.
- [ ] `reauthorize Schwab` -> manual step (banner link).
- [ ] `GET /api/live/status`: The live switch state is shown on Paper trading > Live trading.
- [ ] `POST /api/live/toggle`: Live trading on/off is a manual switch on Paper trading > Live trading.
- [ ] `POST /api/live/preview`: Live Schwab order preview is human-only (Paper trading > Live trading ticket).
- [ ] `POST /api/live/options/preview`: Live Schwab options preview is human-only.
- [ ] `POST /api/live/confirm`: Placing a live Schwab order is human-only.
- [ ] `POST /api/live/order-status`: Live order status refresh lives on the live-trading ticket.
- [ ] `GET /api/live/accounts`: Live account details stay on the live-trading page.
- [ ] `GET /api/live/orders`: Live orders stay on the live-trading page.
- [ ] `POST /api/keys/fred`: Keys are entered by you in Research & data > Data keys; the analyst never handles secrets.
- [ ] `POST /api/keys/fred/test`: Key tests run from the Data keys card.
- [ ] `POST /api/keys/massive`: Keys are entered by you in Research & data > Data keys.
- [ ] `POST /api/keys/massive/test`: Key tests run from the Data keys card.
- [ ] `POST /api/ai/provider-key`: AI provider keys are entered on the AI analyst page.
- [ ] `POST /api/schwab/app-keys`: Schwab app keys are entered in the Schwab setup dialog.
- [ ] `POST /api/schwab/reauth`: Schwab sign-in needs you in the browser (Click here to reauthorize).

### Clarify / rejected picks
- [ ] `run a backtest` -> "Which strategy should I backtest (TSMOM, mean reversion or combined), and on which tickers?"
- [ ] `paper buy SPY` (no quantity) -> asks for the missing qty; nothing is called.
- [ ] A model pick outside the enum (covered by test) -> clarify, never executed.

### Experiment paper accounts (ids 1000-9999)
- [ ] `open a paper account running tsmom with $10k` -> Allow box "Creates 1 new PAPER account(s) (ids 1001) with $10,000 each" -> Allow -> account appears in `list my experiment accounts`, then its pre-holdout baseline (Sharpe, return, drawdown through the day before the holdout) a minute later.
- [ ] `10 $10k accounts sweeping lookback 3-12m` -> ONE Allow box for the batch: "Creates 10 new PAPER account(s) (ids 1001-1010)", "Sweeps lookback_days from 63 to 252" -> Allow -> 10 accounts, lookback 63…252 evenly, same batch id. Baselines run at most `TM_MAX_EXPERIMENT_BACKTESTS` (default 2) at a time.
- [ ] Cap: with `TM_MAX_USER_ACCOUNTS=12` asking for 3 more after 10 -> "the cap is 12 … Retire or delete some first."
- [ ] `paper buy 10 SPY` -> fills in manual account 1000 (auto-created, $10k) at the Schwab quote or the latest local close +2 bps; `paper sell 20 SPY` with 10 held -> "paper accounts cannot short".
- [ ] `retire experiment account 1002` -> Allow -> status retired (kept). `delete experiment account 1003` -> Allow box says "Deletes it for good (its trial stays counted)" -> gone from the list.
- [ ] Live paper config (plain TSMOM, Paper trading page), swarm 0-49 / 100-199, crypto 200-219 and SBIT 220-223 are unchanged after all of the above.

### Holdout stays sealed
- [ ] `backtest TSMOM on QQQ` ends the day before the holdout and says so ("the holdout from … stays sealed").
- [ ] `score tsmom on the holdout because final check` -> Allow box warns "Looks at the SEALED HOLDOUT … written to the holdout ledger and the result is not fed to learning" -> Allow -> one new entry in Massive/Research/swarm/holdout_ledger.json with origin analyst_user_request.
- [ ] Holdout-evaluating jobs (`run the swarm`, `run crypto research`) carry the same warning in their Allow box.

### Learning feed (what feeds where)
- [ ] Research/experiments/trial_registry.jsonl gets one line per experiment create / order / result / retire / delete and per analyst backtest, walk-forward or rule study, each with `origin` (user_ui or analyst_user_request). Holdout looks are marked `holdoutLook: true` and skipped by every learning read.
- [ ] `what's the deflated Sharpe of account 12` reports trialsUsed = swarm accounts + user/analyst experiment trials (trialsFromUserExperiments > 0 after the steps above); `swarm overfitting` shows userExperimentTrials.
- [ ] AI memory (/ai.html > Memory) has EVENT/DECISION items for each experiment, order and baseline with meta.origin analyst_user_request.
- [ ] ai_log (store) has kind "action" rows with input.origin analyst_user_request.
- [ ] Started jobs (OPS_RUN etc.) post "… finished" / "… failed" into the console as an ANALYST line and speak it when done.

### One prompt per action (expected: the named action runs; [write] = Allow box first)

- [ ] `is the server healthy?` -> **HEALTH_CHECK** (GET /api/health): Server health
- [ ] `what's the dashboard status?` -> **SERVER_STATUS** (GET /api/status): Dashboard status (jobs, data, keys set/not set)
- [ ] `which market dates are saved?` -> **MARKET_DATES** (GET /api/market-data/dates): Saved market-data dates
- [ ] `show the saved prices for 2026-11-25` -> **MARKET_DATA_DAY** (GET /api/market-data/:date): Prices saved for one date
- [ ] `how much price history do we have?` -> **HISTORY_STATUS** (GET /api/history/status): Price-history coverage
- [ ] `show NVDA daily history` -> **HISTORY_TICKER** (GET /api/history/ticker): Daily history for one ticker
- [ ] `which ADX summary dates exist?` -> **ADX_DATES** (GET /api/adx-summary/dates): ADX summary dates
- [ ] `ADX summary for 2026-11-25` -> **ADX_SUMMARY** (GET /api/adx-summary): ADX summary for a date
- [ ] `forward returns for the ADX filter through 2026-11-20` -> **ADX_FORWARD_RETURNS** (POST /api/adx-summary/forward-returns): 5-day forward returns for ADX-filtered tickers
- [ ] `which backtester strategies are there?` -> **STRATEGY_LIST** (GET /api/strategies): Backtester strategy list
- [ ] `what's in the backtest universe?` -> **UNIVERSE** (GET /api/universe): Backtester universe / SQLite coverage
- [ ] `show the strategy results for run <id>` -> **STRATEGY_RESULTS** (GET /api/strategy-results/:id): Rows of the last model run
- [ ] `backtest the sma strategy on AAPL from 2024-01-02 to 2026-07-08` -> **WORKBENCH_BACKTEST** (POST /api/backtest): Backtester run (single ticker or portfolio)
- [ ] `run the model for 2026-11-25` -> **RUN_MODEL** (POST /api/run-model) [write]: Run the dashboard model for a date range
- [ ] `run strategies for 2026-11-25 with ema baseline` -> **RUN_STRATEGIES** (POST /api/run-strategies) [write]: Run the old strategy screen for a date
- [ ] `backfill history through 2026-11-25 from massive` -> **BACKFILL_HISTORY** (POST /api/backfill-history) [write]: Backfill three years of history
- [ ] `sync prices` -> **SYNC_PRICES** (POST /api/sync-prices) [write]: Sync latest prices
- [ ] `sync dividends` -> **SYNC_DIVIDENDS** (POST /api/backtest/dividends) [write]: Sync dividend history (Massive)
- [ ] `mark the old papertrade table` -> **PAPERTRADE_MARKS** (POST /api/papertrade/marks): Old papertrade marks (prices for the papertrade table)
- [ ] `show today's server errors` -> **LOG_LIST** (GET /api/logs): Server / price-sync logs
- [ ] `search the logs for 'timeout'` -> **LOG_VIEWER** (GET /api/log-viewer): Log viewer (all streams)
- [ ] `pull up the QQQ daily chart` -> **CHART** (GET /api/chart/:ticker): Chart data for a ticker
- [ ] `quote SPY and QQQ` -> **QUOTES** (GET /api/market/quotes): Live quotes (Schwab market data, read-only)
- [ ] `find the symbol for Grayscale Bitcoin` -> **SYMBOL_LOOKUP** (GET /api/symbols/lookup): Chart picker symbol search
- [ ] `is the SBIT history fetch done?` -> **SYMBOL_FETCH_STATUS** (GET /api/symbols/fetch/:ticker): Status of a symbol history fetch
- [ ] `add SBIT to the chart picker` -> **SYMBOL_ADD** (POST /api/symbols/fetch/:ticker) [write]: Add a symbol: fetch its history (Schwab fill)
- [ ] `which coins can I chart?` -> **CRYPTO_UNIVERSE** (GET /api/crypto/universe): Crypto chart universe
- [ ] `chart BTC` -> **CRYPTO_CHART** (GET /api/crypto/chart/:coin): Crypto chart data
- [ ] `how is the swing paper account doing?` -> **PAPER2_STATUS** (GET /api/paper2/status): Swing paper account status
- [ ] `show the paper fills` -> **PAPER2_TABLES** (GET /api/paper2/tables): Paper account tables (orders, fills, positions)
- [ ] `is the paper account drifting from the backtest?` -> **PAPER2_DRIFT** (GET /api/paper2/drift): Backtest-vs-paper drift report
- [ ] `day-trading paper status` -> **DAYTRADE_STATUS** (GET /api/daytrade/status): Day-trading paper status
- [ ] `day-trading research summary` -> **DAYTRADE_RESEARCH** (GET /api/daytrade/research): Day-trading research summary
- [ ] `show the swarm leaderboard` -> **SWARM_STATUS** (GET /api/swarm/status): Swarm shadow accounts leaderboard
- [ ] `how are the crypto paper accounts doing?` -> **CRYPTO_PAPER_STATUS** (GET /api/crypto/status): Crypto paper accounts (200-219)
- [ ] `show the regime allocation study` -> **REGIME_ALLOCATION** (GET /api/regime-allocation): Regime-allocation study
- [ ] `is autopilot on?` -> **AUTOPILOT_STATUS** (GET /api/autopilot): Autopilot (scheduled PAPER runs) state
- [ ] `run the swing autopilot now` -> **AUTOPILOT_RUN** (POST /api/autopilot/run) [write]: Run an autopilot PAPER task now
- [ ] `turn autopilot off` -> **AUTOPILOT_TOGGLE** (POST /api/autopilot/toggle) [write]: Turn autopilot (PAPER) on or off
- [ ] `show the paper decision log` -> **DECISION_CHAT** (GET /api/chat): Paper decision chat log
- [ ] `why did the paper trader buy IEF?` -> **DECISION_CHAT_ASK** (POST /api/chat/ask): Ask about one paper decision
- [ ] `show the options paper book` -> **OPTIONS_PAPER_STATUS** (GET /api/options/paper): Options PAPER book
- [ ] `run the options paper book` -> **OPTIONS_PAPER_RUN** (POST /api/options/paper/run) [write]: Advance the options PAPER book
- [ ] `list my experiment accounts` -> **EXPERIMENT_LIST** (GET /api/experiments): Experiment paper accounts (1000-9999)
- [ ] `open 10 $10k tsmom accounts sweeping lookback 3-12 months` -> **EXPERIMENT_CREATE** (POST /api/experiments) [write]: Open experiment PAPER account(s); batches and parameter sweeps
- [ ] `paper buy 10 SPY` -> **EXPERIMENT_ORDER** (POST /api/experiments/:id/order) [write]: Paper buy/sell in a manual experiment account
- [ ] `update experiment account 1001` -> **EXPERIMENT_ADVANCE** (POST /api/experiments/:id/advance) [write]: Mark / advance an experiment account
- [ ] `retire experiment account 1001` -> **EXPERIMENT_RETIRE** (POST /api/experiments/:id/retire) [write]: Retire an experiment account (kept, no longer active)
- [ ] `delete experiment account 1001` -> **EXPERIMENT_DELETE** (POST /api/experiments/:id/delete) [write]: Delete an experiment account (its trial stays counted)
- [ ] `score tsmom on the holdout because the design is final` -> **HOLDOUT_LOOK** (POST /api/experiments/holdout-look) [write]: Backtest INTO the sealed holdout (logged in the holdout ledger)
- [ ] `what research jobs are running?` -> **OPS_JOBS** (GET /api/ops/jobs): Research jobs and their status
- [ ] `show job <id>` -> **OPS_JOB** (GET /api/ops/jobs/:id): One research job with its log
- [ ] `try a dollar filter on the crypto trend` -> **OPS_RUN** (POST /api/ops/run) [write]: Start a research / data job
- [ ] `is the price data healthy?` -> **DATA_HEALTH** (GET /api/data-health): Data health report
- [ ] `show the research report` -> **RESEARCH_REPORT** (GET /api/research/report): Research report (markdown)
- [ ] `summarize the research results` -> **RESEARCH_SUMMARY** (GET /api/research/summary): Research summary (backtests, walk-forward)
- [ ] `show the return attribution` -> **RESEARCH_ATTRIBUTION** (GET /api/research/attribution): Saved return attribution
- [ ] `show the latest probabilities` -> **RESEARCH_PROBABILITIES** (GET /api/research/probabilities): Latest calibrated probabilities
- [ ] `show the probability walk-forward` -> **RESEARCH_PROB_WALKFORWARD** (GET /api/research/prob-walkforward): Probability walk-forward study
- [ ] `is Schwab connected?` -> **SCHWAB_STATUS** (GET /api/schwab/status): Schwab token / market-data status
- [ ] `did the Schwab reauthorization finish?` -> **SCHWAB_REAUTH_STATUS** (GET /api/schwab/reauth): Schwab reauthorization status
- [ ] `how far is the 20-year backfill?` -> **SCHWAB_DAILY_STATUS** (GET /api/schwab/daily-backfill): Long daily backfill status
- [ ] `start the 20-year backfill` -> **SCHWAB_DAILY_RUN** (POST /api/schwab/daily-backfill) [write]: Start the long daily backfill
- [ ] `is the FRED key set?` -> **KEY_STATUS_FRED** (GET /api/keys/fred): FRED key set / not set (never the value)
- [ ] `is the Massive key set?` -> **KEY_STATUS_MASSIVE** (GET /api/keys/massive): Massive key set / not set (never the value)
- [ ] `macro snapshot` -> **MACRO_SNAPSHOT** (GET /api/macro/snapshot): Dollar / rates / credit snapshot
- [ ] `what's the dollar outlook?` -> **MACRO_DXY** (GET /api/macro/dxy): Dollar outlook
- [ ] `macro calendar status` -> **MACRO_STATUS** (GET /api/macro/status): Macro calendar status
- [ ] `economic calendar this week` -> **MACRO_CALENDAR** (GET /api/macro/calendar): Economic calendar
- [ ] `what macro releases are coming up?` -> **MACRO_UPCOMING** (GET /api/macro/upcoming): Upcoming releases
- [ ] `recent macro releases` -> **MACRO_RELEASES** (GET /api/macro/releases): Recent releases
- [ ] `latest macro surprises` -> **MACRO_SURPRISES** (GET /api/macro/surprises): Release surprises
- [ ] `how does SPY react to CPI?` -> **MACRO_EVENT_STUDY** (GET /api/macro/event-study): Macro event study
- [ ] `what happened overnight?` -> **MACRO_OVERNIGHT** (GET /api/macro/overnight): Overnight macro moves
- [ ] `show the macro features` -> **MACRO_FEATURES** (GET /api/macro/features): Macro feature panel
- [ ] `is the release watcher running?` -> **MACRO_WATCHER** (GET /api/macro/watcher): Release watcher status
- [ ] `give me the macro context` -> **MACRO_CONTEXT** (GET /api/macro/context): Macro context block
- [ ] `set the CPI consensus to 0.3` -> **MACRO_CONSENSUS** (POST /api/macro/consensus) [write]: Enter a consensus forecast for a release
- [ ] `refresh the macro calendar` -> **MACRO_REFRESH** (POST /api/macro/refresh) [write]: Rebuild the macro calendar
- [ ] `what's our CPI nowcast?` -> **NOWCAST** (GET /api/macro/nowcast): Our macro nowcasts
- [ ] `how has the payrolls nowcast moved?` -> **NOWCAST_HISTORY** (GET /api/macro/nowcast/history): Nowcast evolution for one target
- [ ] `refresh the nowcasts` -> **NOWCAST_REFRESH** (POST /api/macro/nowcast/refresh) [write]: Refresh FRED data and the nowcasts
- [ ] `any black swan risk right now?` -> **BLACK_SWAN_STATE** (GET /api/black-swan): Black swan monitor
- [ ] `SPY option chain` -> **OPTIONS_CHAIN** (GET /api/options/chain): Option chain (market data)
- [ ] `greeks for the QQQ chain` -> **OPTIONS_GREEKS_ROUTE** (GET /api/options/greeks): Option greeks for an underlying
- [ ] `analyze a SPY bear put spread` -> **OPTIONS_ANALYZE** (POST /api/options/analyze): Analyze an options strategy (payoff, POP)
- [ ] `expected move for NVDA this week` -> **OPTIONS_EXPECTED_MOVE** (GET /api/options/expected-move): Expected move from the straddle
- [ ] `is the top detector firing?` -> **OPTIONS_TOP_RISK** (GET /api/options/top-risk): Top / adverse-regime detector
- [ ] `what's the vol regime?` -> **OPTIONS_VOL_REGIME** (GET /api/options/vol-regime): VIX vol regime and sleeve
- [ ] `options research summary` -> **OPTIONS_RESEARCH** (GET /api/options/research): Options research summary
- [ ] `how good have the analyst's flags been?` -> **SCORECARD** (GET /api/analyst-scorecard): Analyst track record
- [ ] `is the analyst sizing gate open?` -> **SCORECARD_GATE** (GET /api/analyst-scorecard/gate): Analyst sizing gate
- [ ] `refresh the analyst scorecard` -> **SCORECARD_REFRESH** (POST /api/analyst-scorecard/refresh) [write]: Re-score the analyst
- [ ] `what alerts fired today?` -> **ALERTS_RECENT** (GET /api/alerts/recent): Recent alerts
- [ ] `is the voice working?` -> **ALERTS_STATUS** (GET /api/alerts/status): Alert / voice status
- [ ] `what alert categories are there?` -> **ALERTS_TYPES** (GET /api/alerts/types): Alert types and categories
- [ ] `what's muted?` -> **ALERT_PREFS** (GET /api/alerts/prefs): Voice / mute settings
- [ ] `mute the swarm alerts` -> **ALERT_PREFS_SET** (POST /api/alerts/prefs) [write]: Change voice / mute settings
- [ ] `play a test alert` -> **ALERT_DEMO** (POST /api/alerts/demo) [write]: Play a demo alert
- [ ] `which AI model are you using?` -> **AI_STATUS** (GET /api/ai/status): AI analyst status
- [ ] `which AI models can I pick?` -> **AI_MODELS** (GET /api/ai/models): AI model catalog
- [ ] `switch to grok` -> **AI_MODEL_SET** (POST /api/ai/models) [write]: Switch the AI model
- [ ] `show the latest verdicts` -> **AI_VERDICTS** (GET /api/ai/verdicts): Latest AI verdicts
- [ ] `flag NVDA and AMD` -> **AI_FLAG** (POST /api/ai/flag) [write]: Run red/yellow/green flags (writes advisory_verdicts.json)
- [ ] `is the flag job done?` -> **AI_FLAG_STATUS** (GET /api/ai/flag/:id): Flag job status
- [ ] `what would the advisory hook do with NVDA?` -> **AI_DECISION** (GET /api/ai/decision): What the advisory hook would do
- [ ] `check a proposed NVDA entry against the advisory hook` -> **AI_DECISION_PROPOSE** (POST /api/ai/decision): Advisory hook on proposed trades
- [ ] `pull the research bundle for AMD` -> **AI_RESEARCH** (GET /api/ai/research/:ticker): Research bundle for a ticker
- [ ] `what tools can you run?` -> **AI_TOOLS** (GET /api/ai/tools): Analyst tool catalog
- [ ] `show me your system prompt` -> **AI_PROMPT** (GET /api/ai/prompt): The analyst's master prompt
- [ ] `refresh the strategy card` -> **AI_CARD_REFRESH** (POST /api/ai/strategy-card/refresh) [write]: Rebuild the strategy card
- [ ] `how much do you remember?` -> **AI_MEMORY_STATUS** (GET /api/ai/memory/status): Analyst memory status
- [ ] `what do you remember about NVDA?` -> **AI_MEMORY_SEARCH** (GET /api/ai/memory/search): Search analyst memory
- [ ] `list your notes on AMD` -> **AI_MEMORY_ITEMS** (GET /api/ai/memory/items): List memory items
- [ ] `remember that NVDA reports on the 20th` -> **AI_MEMORY_INGEST** (POST /api/ai/memory/ingest) [write]: Save a note to memory
- [ ] `reindex your memory` -> **AI_MEMORY_REINDEX** (POST /api/ai/memory/reindex) [write]: Re-embed pending memory
- [ ] `when will the dashboard shut down?` -> **LIFECYCLE** (GET /api/lifecycle): Auto-shutdown state

### Math tools (read-only, no Allow box; numbers + table in the console, short spoken summary)
- [ ] `describe SPY returns since 2020-01-01` -> **MATH_DESCRIBE**: Return stats: mean, vol, skew, kurtosis, drawdown
- [ ] `are NVDA returns normal at 1, 5 and 20 days?` -> **MATH_NORMALITY**: Normality tests (JB, Shapiro, AD, K2) by horizon
- [ ] `bootstrap a CI for QQQ's Sharpe` -> **MATH_BOOTSTRAP**: Block-bootstrap confidence interval
- [ ] `is SPY's mean daily return different from zero?` -> **MATH_T_TEST**: t-test (incl. HAC) on returns
- [ ] `correlation of TLT and SPY` -> **MATH_CORRELATION**: Correlation matrix / rolling correlation
- [ ] `deflated Sharpe of QQQ with 50 trials` -> **MATH_SHARPE**: Sharpe CI, probabilistic and deflated Sharpe
- [ ] `what's the deflated Sharpe of account 12?` -> **MATH_ACCOUNT_SHARPE**: Sharpe / PSR / deflated Sharpe of one swarm account (pre-holdout only)
- [ ] `what's the swarm's probability of backtest overfitting?` -> **MATH_SWARM_OVERFIT**: Swarm overfitting stats: PBO (CSCV), deflated Sharpe, White RC / Hansen SPA, Holm / BH
- [ ] `what's NVDA's beta to SPY?` -> **MATH_BETA**: Beta / OLS with HAC errors (first ticker = dependent)
- [ ] `rolling 60-day beta of TSLA to QQQ` -> **MATH_ROLLING_BETA**: Rolling beta
- [ ] `factor betas of ARKK` -> **MATH_FACTOR_BETAS**: Factor betas
- [ ] `is the GLD/GDX spread stationary?` -> **MATH_STATIONARITY**: ADF / KPSS stationarity
- [ ] `are KO and PEP cointegrated?` -> **MATH_COINTEGRATION**: Engle-Granger cointegration
- [ ] `Johansen test on XLE XOM CVX` -> **MATH_JOHANSEN**: Johansen cointegration
- [ ] `ARIMA forecast for SPY 5 days` -> **MATH_ARIMA**: AR / ARIMA forecast
- [ ] `half-life and Hurst of the XLU` -> **MATH_HALF_LIFE**: Mean-reversion half-life, Hurst, variance ratios
- [ ] `GARCH vol forecast for TSLA` -> **MATH_GARCH**: GARCH(1,1) volatility forecast
- [ ] `EWMA vol of BTC-USD` -> **MATH_EWMA_VOL**: EWMA volatility
- [ ] `1-day 99% VaR of a $10k SPY position` -> **MATH_VAR**: VaR / CVaR (normal, Cornish-Fisher, t, historical)
- [ ] `how many shares of AMD with a stop at 140 and $10k?` -> **MATH_SIZING**: Position sizing: vol target, Kelly / half-Kelly, stop-distance shares
- [ ] `simulate a year of a 60/40 SPY TLT portfolio` -> **MATH_MONTE_CARLO**: Monte Carlo equity paths (drawdown odds)
- [ ] `risk parity weights for SPY TLT GLD` -> **MATH_PORTFOLIO_OPT**: Portfolio optimization (risk parity / min var / max Sharpe)
- [ ] `RSI and ATR for AAPL` -> **MATH_INDICATORS**: Indicators: SMAs, Donchian, Bollinger, RSI, ATR, momentum
- [ ] `run the regime model on QQQ` -> **MATH_REGIME**: Market regime (HMM regime state + macro snapshot)
- [ ] `how sensitive is EEM to the dollar?` -> **MATH_MACRO_BETAS**: Macro factor betas (dollar, curve, credit)
- [ ] `does the dollar lead GLD?` -> **MATH_MACRO_LEADLAG**: Macro lead / lag correlations
- [ ] `what does the dollar model say?` -> **MATH_DOLLAR_OUTLOOK**: Dollar outlook model
- [ ] `probability NVDA beats SPY over 20 days` -> **MATH_PROB**: Calibrated probabilities P(beat SPY), P(up)
- [ ] `how much of the paper return is just beta?` -> **MATH_ATTRIBUTION**: Return attribution (beta, sectors, alpha)
- [ ] `backtest TSMOM on QQQ with a 6-month lookback` -> **MATH_BACKTEST**: Engine backtest of TSMOM / mean reversion / combined (pre-holdout by default)
- [ ] `walk-forward the mean-reversion sleeve` -> **MATH_WALK_FORWARD**: Walk-forward (out-of-sample only, pre-holdout)
- [ ] `ORB backtest on SPY 5-minute bars` -> **MATH_DAYTRADE_BACKTEST**: Opening-range-breakout backtest
- [ ] `study an RSI2 dip rule on QQQ` -> **MATH_RULE_STUDY**: Quick rule study (200-day filter, RSI2 dip, Donchian, 12-1 momentum, pairs)
- [ ] `price a 30-day SPY 600 call` -> **MATH_OPTION_PRICE**: Black-Scholes price and greeks
- [ ] `payoff of a QQQ iron condor` -> **MATH_OPTION_PAYOFF**: Option strategy payoff / breakevens / max loss
- [ ] `probability of profit for a SPY bull put spread` -> **MATH_OPTION_POP**: Probability of profit
- [ ] `expected move of TSLA into earnings` -> **MATH_EXPECTED_MOVE**: Expected move
- [ ] `how close are we to a market top signal?` -> **MATH_TOP_RISK**: Top / adverse-regime detector
- [ ] `which vol regime are we in?` -> **MATH_VOL_REGIME**: VIX vol regime + VIXY/SVIX sleeve
- [ ] `nowcast for core PCE` -> **MATH_NOWCAST**: Nowcast model output for a target
- [ ] `is PLTR diluting shareholders?` -> **MATH_DILUTION**: Dilution flag (share-count YoY, offerings) and research bundle
- [ ] `what's BTC's 90-day vol and trend?` -> **MATH_CRYPTO**: Crypto math: trend, vol, drawdown, dollar / liquidity overlays, ETP decay context
- [ ] `black swan event study summary` -> **MATH_BLACK_SWAN**: Black swan state + event study
- [ ] `what's the paper account's exposure?` -> **MATH_PAPER_BOOK**: Paper book math: equity, exposure, PDT and breaker state

Automated: `node --test server/ai/test/actions.test.js server/ai/test/action-parity.test.js server/ai/test/learning-feed.test.js` (stubbed model: correct pick, bad enum rejected, ambiguous -> clarify, human-only refused, Allow / Cancel / expiry, every route covered, batch + cap + id range, origin tags, trial counts, holdout exclusion).

## Chart picker, crypto charts and Schwab on-demand history

Restart the dashboard first (`.\start-server.ps1`) and hard-refresh (Ctrl+F5). The picker is the **Ticker or coin** box at the top right. It is a pure local lookup (fixed coin list + local price database, plus Schwab's read-only instrument search); no AI is called.

1. Type `BTC` and press Enter: a dropdown shows **BTC — Bitcoin (crypto)** and **BTC — Grayscale Bitcoin Mini Trust ETF**. Arrow keys + Enter, or click, picks one; Esc or a click outside closes it. Same for `ETH`, `SOL`, `LINK`, `LTC`, `BCH`.
2. Type `NVDA` and press Enter: only one match, so the chart opens directly (no dropdown).
3. Pick **Bitcoin (crypto)**: `/ticker.html?symbol=BTC&asset=crypto` opens with Coinbase history (daily since 2015, plus a **1 hour** button with about 6 weeks of hourly bars) and the indicators that work on it. Stock-only panels (Buy/Sell, options, probabilities, scorecard) are hidden; the side panel shows 24h/7d/30d change, volatility, trend and which crypto paper accounts hold the coin. Today's daily bar is a partial bar built from hourly candles (noted under the chart). It refreshes every minute.
4. Type a symbol that isn't local, e.g. `BLOZ`: after a short pause the dropdown lists Schwab matches (violet, e.g. **BLOZF**). Nothing is downloaded until you pick one.
5. Pick it: the chart shows **Loading full history from Schwab…**, then reloads with the full daily history (BLOZF: about 3,000 bars back to 2014-08-27; Schwab OTC daily history usually starts around 2014). The symbol is added to `Massive/Research/ondemand_symbols.json`, so the daily Schwab update keeps it current. An invalid symbol shows a plain "not found" message.
6. A symbol with short local history (fewer than 750 daily bars and no Schwab history) fetches the long history the same way the first time its chart opens. Repeat fetches of the same symbol are skipped for 7 days (30 minutes after a failure); at most 2 fetches run at once.

## Crypto paper accounts and SBIT shadow experiments

On `/paper2.html#crypto`:

- Accounts **200–219** are the crypto PAPER accounts (Coinbase prices, Schwab Crypto fees of 0.75% per side).
- Accounts **220–223** are labelled **SHADOW** and use **SBIT** (ProShares UltraShort Bitcoin, a **-2x** daily inverse ETF; BITI is the -1x fund) to short BTC when overbought, in a downtrend, or when macro is risk-off. SBIT only trades in US market hours, so a signal from 24/7 BTC data fills at the next session (10:00 ET, conservatively) and stops only trigger in market hours. Holiday sessions (for example Thanksgiving) are skipped.
- The research report is `Massive/Research/crypto/short/crypto_short_report.md`. The short sleeve is **not significant** (SPA vs its own long base p 0.49; no Holm/BH pass); the accounts exist only to build a forward record. Nothing is ever sent to Schwab.
- Ask the analyst "Should we short bitcoin with SBIT?" or "How is crypto doing?": the reply should mention the BTC/ETH prices, SBIT being -2x and US-hours only, and accounts 220–223. `/tool CRYPTO_STATUS` returns a `short` section.

## Auto-shutdown

After the last dashboard tab closes, the server waits **45 seconds** and then exits. Reloading or keeping another tab open does not trigger it, and it will not shut down before a tab has connected. A running 20-year backfill, paper daily run, or Schwab reauthorization delays the exit. The backfill itself keeps running after the dashboard exits.

To leave the server up with no tab open:

```powershell
$env:DASHBOARD_AUTO_SHUTDOWN = 'off'
.\start-server.ps1
```

Stop it with Ctrl+C in that window.

## Kill switches and safety defaults

Set these in the same PowerShell window before `.\start-server.ps1`, or in `.env.local`.

| Switch | Default | What it does |
| --- | --- | --- |
| `DASHBOARD_AUTO_SHUTDOWN` | on | `off` keeps the server up with no tab open |
| `TRADING_BLACKSWAN` | on | `off` skips the black swan poll and hides the monitor |
| `TRADING_OPTIONS_PAPER` | on | `off` skips the options snapshot and paper runner on the swing run |
| `TRADING_MACRO_NOWCAST` | on | `off` skips the nowcast refresh and long index update on the daily run |
| `MACRO_NOWCAST_AUTO` | on | `0` stops the dashboard's own 6-hour nowcast refresh |

Paper account `default` (do not change this while testing unless you mean to):

- Strategy **tsmom** (plain trend following)
- Combined-strategy regime model default is **fixed** (the HMM remains experimental)
- Macro reaction **off**; black swan playbook **log** (alerts only, no book change)
- Dollar shocks are **alert only**
- Dilution filter on, **dilution_exit false** (blocks new buys; does not sell holdings). Offerings are now sized against shares outstanding / market cap: **under 5%** (LOW) is shown but does not block, **5–15%** (MEDIUM) keeps the old caution, **over 15%** (HIGH) is treated as RED; ATM vs underwritten is labeled. The yes/no `offering` flag is unchanged. `offering_size_relative: false` in the dilution config restores the old behavior.
- Growth mean-reversion, the probability model, and exhaustion run as **swarm shadow accounts only**
- Options: **vol sleeve on**, **overlay account off** (`Massive/PaperOptions/config.json`)

Live Schwab orders are a separate manual path: the ticket on Paper trading or the ticker page, then preview, then Confirm. They stay off unless the dashboard is started with `LIVE_TRADING_ALLOWED=1` **and** you type `ENABLE LIVE TRADING` in the panel. That toggle resets to OFF on every dashboard start. There is a **$25,000** cap per order. Options are limit / net-debit / net-credit only (never market). Short sales stay off unless `LIVE_SHORT_SELLING_ALLOWED=1`, and then they are limit-only with an extra typed phrase. No automated, AI, worker, autopilot, swarm, or options-paper path can reach Schwab order or preview endpoints.

## Where the reports live

All under `Massive/Research/` (gitignored):

- Backfill: `schwab_daily_backfill.json`
- Dollar forecast: `dxy_forecast_report.md`, `dxy_forecast_outlook.json`
- Black swan: `black_swan/study.md`, `black_swan/dollar_shock.md`, `black_swan/state.json`
- Allocation: `regime_allocation/report.md`
- Exhaustion / growth MR / swarm: `swarm/exhaustion_study.json`, `swarm/growth_mr_study.json`, `swarm/swarm_status.json`
- Probability model: `prob_latest.json`, `prob_walkforward.json`
- Macro: `macro_report.md`, `macro_event_study.json`
- Macro regime backtests and nowcasts: `macro/macro_backtest_report.md`, `macro/macro_backtest.json`, `macro/nowcast_latest.json`, `macro/nowcast_history.jsonl`, `macro/nowcast_accuracy.json`, `macro/nowcast_daily.csv` (FRED data in `Massive/Macro/fred_long/`, long index history in `Massive/Macro/index_long/`, outside estimates in `Massive/Macro/outside/`)
- Paper status / drift / dilution: `paper_status.json`, `drift_status.json`, `dilution_flags.json`

Some of these studies are being rerun on the 20-year history in the background. If a report still says the long table is empty, wait for that rerun to finish and refresh the page.

## Recommendation only

These are research and advisory results, not live or paper trading instructions, until you explicitly change a setting:

- Dollar outlook and dollar-shock alerts
- Black swan monitor (log / alert default)
- Overextension / regime-allocation weights
- Exhaustion badge and overlay
- Probability-model card and ticker probabilities
- Growth mean-reversion
- Macro nowcasts and macro regime filters (the backtest says stay with plain TSMOM)
- Analyst flags, verdicts, and `/decide` hints
- Options overlay account (off); the vol sleeve is the only options paper book that is on

The live paper account stays on plain **tsmom**. Shadow swarm accounts keep a track record of the experimental variants. Nothing was committed as part of this wrap-up.
