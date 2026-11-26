# Trading Model

A local research dashboard for US stocks, macro data and crypto. It covers daily price history and ADX tools, charts,
walk-forward strategy research, simulated (paper) trading accounts and an optional AI analyst. Everything runs on your own
Windows PC at http://localhost:3003. Paper accounts are simulations; no automated, AI or background code can place a
broker order.

## Fresh-clone setup (Windows)

1. Install **Node.js 22.13 or newer** (24 LTS recommended; the server uses the built-in `node:sqlite`) and
   **Python 3.10 or newer** (from python.org, not the Microsoft Store stub). Excel is only needed for the ADX workbook
   update.
2. Clone the repository and open PowerShell in its folder.
3. Start the dashboard:

   ```powershell
   .\start-server.ps1
   ```

   Note the backslash. The first run creates `.venv`, installs `engine/requirements.txt`, runs `npm ci` in `server/`,
   and opens http://localhost:3003 (use `-Port` to pick another port, `-NoBrowser` to skip the browser). If PowerShell
   refuses to run scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
4. Enter your keys in the app. They are saved on your PC only (gitignored) and are never shown again in the browser:
   - **Massive** (market data) and **FRED** (macro data): Research & data (`/ops.html#keys`) > **Data keys**.
   - **Schwab** (optional, read-only market data): the Schwab chip in the top bar > set up / reauthorize. Create an app
     at developer.schwab.com with the callback `https://127.0.0.1:8182` (separate from the dashboard port).
   - **AI analyst** (optional): `/ai.html` > **API keys...** (OpenAI, xAI or Groq), saved under `keys/`.

   Or copy `.env.example` to `.env.local` and fill it in there. Never commit `.env.local`.
5. Optional models (large, gitignored under `models/`):
   - SID voice: download `en_US-ljspeech-high` and `en_US-ryan-high` (`.onnx` + `.onnx.json`) from
     [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) into `models/piper/`, then check with
     `node voice/install-voices.mjs` from `server/`. See `server/voice/README.md`. Without them the browser's voice is used.
   - Analyst memory: `node server/ai/memory/install-model.js`.

All market data, research output and paper-trading state live in the gitignored `Massive/` folder, which starts empty on
a fresh clone. With a Massive key, each launch downloads the latest daily sessions. The 20-year Schwab daily backfill is
optional; start it from Research & data.

The daily paper run is `.\run-paper-daily.ps1`. Run it yourself, or schedule it with the `schtasks` command in its header
(it is never registered automatically).

`TESTING-CHECKLIST.md` has a page-by-page click list, the kill switches and where each report is written.

## AI analyst actions

The command console (lower right on every page) takes plain-English requests. Each prompt is matched by one strict-JSON
model call to a single action from a closed list (server/ai/actionRegistry.js: every dashboard action plus the read-only
math tools); the server re-validates the action and its arguments and asks a clarifying question when a request is ambiguous.

- Reads and math (quotes, backtests, deflated Sharpe, regime model, option pricing, ...) run at once.
- Anything that changes state (experiment paper accounts, paper orders, research jobs, settings) shows an **Allow / Cancel**
  prompt with the action, its arguments and the accounts it affects. Enter = Allow, Esc = Cancel; it expires after
  5 minutes and is never allowed automatically.
- Experiment paper accounts use ids 1000-9999 and never touch the live paper config. Limits: TM_MAX_USER_ACCOUNTS
  (active accounts, default 200) and TM_MAX_EXPERIMENT_BACKTESTS (concurrent baseline backtests, default 2).
- Backtests stop before the sealed holdout. Every analyst experiment is counted in
  Massive/Research/experiments/trial_registry.jsonl (local user data, not committed) so multiple-testing statistics include it.
- Live orders, the live-trading switch, API keys and Schwab sign-in are manual steps; the analyst only points you to the page.
## Tests

```powershell
cd server; $env:TRADING_CRYPTO_SNAPSHOT='off'; node --test (Get-ChildItem -Recurse -Include *.test.mjs,*.test.js -File | ? FullName -notmatch 'node_modules' | % FullName); cd ..
cd engine; $env:PYTHONPATH=(Get-Location).Path; ..\.venv\Scripts\python.exe -m unittest discover -p "test_*.py"; cd ..
```

## Safety

- The Schwab client is market data only. The only order path is the manual live ticket (preview, then typed Confirm). It is
  off unless the dashboard is started with `LIVE_TRADING_ALLOWED=1` and you enable it in the panel.
- The research uses a sealed holdout window. Every look at it is logged in `Massive/Research/swarm/holdout_ledger.json`.
- This is research software, not investment advice.
