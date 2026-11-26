<#
.SYNOPSIS
Daily paper-trading run for the TSMOM + mean-reversion strategy (local simulation; no broker is called).
.DESCRIPTION
1. Downloads any missing daily sessions from Massive the same way the dashboard does at launch
   (MASSIVE_API_KEY from the environment or .env.local; skipped with -NoSync).
2. Runs engine\paper_trader.py: fills yesterday's orders at today's open, applies stops, dividends and
   stale-position rules, marks to market, records today's decisions/orders, and reconciles with the backtest.
3. Runs engine\daytrader.py (opening-range breakout day trades, flat by close, PDT-limited $10k account) on
   today's Schwab bars; Schwab is used for market data only (quotes, price history), never for orders.
4. Runs engine\drift_monitor.py: compares realized paper behavior to walk-forward expected ranges and writes
   Massive\Research\drift_status.json (shown on /paper2.html#drift).
5. Writes Massive\Research\paper_status.json / daytrade_status.json (shown at http://localhost:3003/paper2.html) and appends to
   Massive\Research\paper_daily.log.
6. Runs engine\swarm.py forward: advances the 50 simulated $10k shadow accounts (no orders) through the new
   session(s) so they build live out-of-sample records (Massive\Research\swarm\forward). Never fails the run.
7. Refreshes our macro nowcasts (engine\macro_nowcast.py refresh; TRADING_MACRO_NOWCAST=off skips) and the long
   Schwab index history (engine\index_long.py). Never fails the run.
Safe to re-run: a session that was already processed is skipped; missed days are caught up in order.

To schedule it on weekdays at 9:30 PM (NOT registered automatically; run once yourself if you want it):
  schtasks /Create /TN "TradingModel\PaperDaily" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 21:30 /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File \"<full path to this repo>\run-paper-daily.ps1\""
Remove it with:  schtasks /Delete /TN "TradingModel\PaperDaily" /F
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$NoSync,
    [string]$Account = 'default',
    [ValidateSet('next_open', 'close')]
    [string]$Execution,
    [switch]$NoDayTrade,
    [ValidateSet('off', 'log', 'apply')]
    [string]$Advisory = 'log'
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { $python = 'python' }
$researchDirectory = Join-Path $projectRoot 'Massive\Research'
New-Item -ItemType Directory -Force -Path $researchDirectory | Out-Null
$log = Join-Path $researchDirectory 'paper_daily.log'

$arguments = @('-u', (Join-Path $projectRoot 'engine\paper_trader.py'), 'run', '--account', $Account)
if (-not $NoSync) { $arguments += '--sync-prices' }
if ($DryRun) { $arguments += '--dry-run' }
if ($Execution) { $arguments += @('--execution', $Execution) }
$arguments += @('--advisory', $Advisory)
if (-not $NoSync -and -not $DryRun) { $arguments += '--schwab-quotes' }

$env:PYTHONUTF8 = '1'
"==== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') paper run (dry=$DryRun sync=$(-not $NoSync)) ====" | Add-Content -Path $log
Push-Location (Join-Path $projectRoot 'engine')
try {
    # Python logs progress on stderr; keep it as text instead of PowerShell error records.
    $ErrorActionPreference = 'Continue'
    if (-not $DryRun) {
        # Analyst scorecard: score matured AI flags/stops and refresh the advisory gate first (never blocks the run).
        & $python -u (Join-Path $projectRoot 'engine\analyst_scoring.py') run 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    }
    # Dilution / offering scan: refresh flags used to block new buys. Never fails the booked paper run.
    & $python -u (Join-Path $projectRoot 'engine\dilution.py') scan 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "==== dilution scan exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    & $python @arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    $code = $LASTEXITCODE
    if (-not $NoDayTrade) {
        # Day trader: refresh Schwab bars/quotes (market data only) and paper-trade today's session after the close.
        $dayArguments = @('-u', (Join-Path $projectRoot 'engine\daytrader.py'), 'run', '--advisory', $Advisory)
        if (-not $NoSync) { $dayArguments += '--fetch' }
        if ($DryRun) { $dayArguments += '--dry-run' }
        & $python @dayArguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0 -and $code -eq 0) { $code = $LASTEXITCODE }
    }
    # Drift monitor: compare today's paper (swing + day-trade) behavior to walk-forward expected ranges.
    # A failure here must not fail the booked paper run; it is logged and the exit code is left as-is.
    $driftArguments = @('-u', (Join-Path $projectRoot 'engine\drift_monitor.py'), 'run')
    & $python @driftArguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "==== drift monitor exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    # Swarm forward mode: advance the simulated $10k shadow accounts (no orders). A failure never fails the paper run.
    if (-not $DryRun -and $env:TRADING_SWARM -ne 'off') {
        & $python -u (Join-Path $projectRoot 'engine\swarm.py') forward 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { "==== swarm forward exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    }
    # Crypto PAPER shadow accounts (ids 200-219): refresh Coinbase/FRED/Schwab market data, then replay all 20 (no orders).
    if (-not $DryRun -and $env:TRADING_CRYPTO_PAPER -ne 'off') {
        $cryptoArguments = @('-u', (Join-Path $projectRoot 'engine\crypto_paper.py'), 'run')
        if (-not $NoSync) { $cryptoArguments += '--fetch' }
        & $python @cryptoArguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { "==== crypto paper exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    }
    # Macro / dollar / rates / credit update: never fails the booked paper run.
    $macroArguments = @('-u', (Join-Path $projectRoot 'engine\macro_data.py'), 'update')
    & $python @macroArguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "==== macro update exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    # Economic calendar refresh (official + fallback, times in ET). Never fails the booked paper run.
    $calArguments = @('-u', (Join-Path $projectRoot 'engine\macro_service.py'), 'refresh')
    & $python @calArguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) { "==== macro calendar refresh exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    # Our macro nowcasts: incremental FRED/ALFRED import, outside estimates (display only), then publish
    # Massive\Research\macro\nowcast_latest.json (+ history). Raw data only. Never fails the booked paper run.
    if (-not $DryRun -and $env:TRADING_MACRO_NOWCAST -ne 'off') {
        & $python -u (Join-Path $projectRoot 'engine\macro_nowcast.py') refresh 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { "==== macro nowcast refresh exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
        if (-not $NoSync) {
            # Long index history ($SPX, $DJI, ... back to 1970) from Schwab market data (read-only).
            & $python -u (Join-Path $projectRoot 'engine\index_long.py') update 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
            if ($LASTEXITCODE -ne 0) { "==== index history update exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
        }
    }
    # Black swan / geopolitical poll (feeds + confirmation + paper playbook state). Never fails the booked paper run.
    if ($env:TRADING_BLACKSWAN -ne 'off') {
        & $python -u (Join-Path $projectRoot 'engine\black_swan.py') poll 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { "==== black swan poll exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    }
    # Options: chain snapshot (Schwab market data, skipped without auth) and the vol-sleeve / overlay PAPER runner
    # (no orders; writes its own decision-chat entries). Never fails the booked paper run. TRADING_OPTIONS_PAPER=off skips.
    if (-not $DryRun -and $env:TRADING_OPTIONS_PAPER -ne 'off') {
        & $python -u (Join-Path $projectRoot 'engine\options_data.py') snapshot 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { "==== options snapshot exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
        & $python -u (Join-Path $projectRoot 'engine\options_paper.py') run --refresh-vix 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
        if ($LASTEXITCODE -ne 0) { "==== options paper exit $LASTEXITCODE (paper run kept) ====" | Add-Content -Path $log }
    }
}
finally {
    Pop-Location
}
"==== exit $code ====" | Add-Content -Path $log
exit $code
