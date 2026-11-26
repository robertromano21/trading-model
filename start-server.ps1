<#
.SYNOPSIS
Starts the local Prices dashboard in the current PowerShell window.
.DESCRIPTION
Reads JSON and SQLite history from the project's Massive folder by default.
Uses ADX worksheet 2.0.xlsm in the project folder for Excel updates.
Set PRICES_DATA_DIR or ADX_WORKBOOK_PATH in PowerShell to use different paths.
Each launch updates SQLite with the latest daily market prices from Massive.
Set MASSIVE_API_KEY in the environment or .env.local to enable downloads.
Local JSON imports need no API key.
Choose a date, a date range, or all pending files in the dashboard.
Use the History tab to build local SQLite history and look up a ticker.
Select Update Excel with latest session to refresh the workbook.
Stop the server with Ctrl+C, or close the last dashboard browser tab
(it exits after a short grace period so reloads and other tabs stay up).
Set DASHBOARD_AUTO_SHUTDOWN=off to keep it running with no tab open.
A long Schwab daily backfill keeps running after the dashboard exits.
Use -NoBrowser to skip opening localhost.
The default launch reuses a running dashboard or picks a free port when
starting with a different configuration.
Run this script as .\start-server.ps1 (note the backslash).
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 3003,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runtimeDirectory = Join-Path $projectRoot '.runtime'
$serverDirectory = Join-Path $projectRoot 'server'

function Test-LocalPortInUse([int]$LocalPort) {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $client.Connect('127.0.0.1', $LocalPort)
        return $true
    }
    catch [System.Net.Sockets.SocketException] {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

$requestedPort = $Port
$usesOverrides = $env:MASSIVE_API_KEY -or $env:PRICES_DATA_DIR -or $env:ADX_WORKBOOK_PATH -or $env:PRICES_PYTHON
while (Test-LocalPortInUse $Port) {
    $url = "http://localhost:$Port"
    if (-not $usesOverrides) {
        $health = $null
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/health" -UseBasicParsing -TimeoutSec 2
        }
        catch {
            # A different service may own this port.
        }
        if ($health.service -eq 'prices-dashboard') {
            $localVersion = (Get-FileHash -LiteralPath (Join-Path $serverDirectory 'server.js') -Algorithm SHA256).Hash
            if ($health.serverVersion -ne $localVersion) {
                throw "An older Prices Dashboard is running on port $Port. Stop it with Ctrl+C in its PowerShell window and run start-server.ps1 again."
            }
            $sync = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/api/sync-prices" -Method Post `
                -ContentType 'application/json' -Body '{}' -UseBasicParsing -TimeoutSec 5
            Write-Host "Prices dashboard is already running at $url"
            if ($sync.queued) { Write-Host 'Price update queued after the current job.' }
            else { Write-Host 'Checking for the latest market prices.' }
            if (-not $NoBrowser) { Start-Process $url }
            return
        }
    }
    if ($PSBoundParameters.ContainsKey('Port')) {
        throw "Port $Port is already in use. Choose a free port with -Port, or omit -Port to select one automatically."
    }
    $Port += 1
    if ($Port -gt 65535) { throw 'No free dashboard port was found.' }
}
if ($Port -ne $requestedPort) {
    Write-Host "Port $requestedPort is in use. Starting Prices Dashboard at http://localhost:$Port"
}

$node = (Get-Command node.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $node) {
    $node = (Get-ChildItem -LiteralPath $runtimeDirectory -Directory -ErrorAction SilentlyContinue |
        Where-Object Name -Like 'node-*' |
        ForEach-Object { Join-Path $_.FullName 'node.exe' } |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
}
if (-not $node) { throw 'Install Node.js 22 or newer and reopen PowerShell.' }
$nodeVersion = & $node -p "process.versions.node.split('.')[0]"
if ($LASTEXITCODE -ne 0 -or [int]$nodeVersion -lt 22) { throw 'Node.js 22 or newer is required.' }

$python = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    $basePython = $env:PRICES_PYTHON
    if (-not $basePython) {
        $basePython = (Get-Command python.exe -All -ErrorAction SilentlyContinue |
            Where-Object Source -NotLike '*\Microsoft\WindowsApps\*' | Select-Object -First 1).Source
    }
    if (-not $basePython) {
        $basePython = (Get-ChildItem -LiteralPath $runtimeDirectory -Directory -ErrorAction SilentlyContinue |
            Where-Object Name -Like 'python-*' |
            ForEach-Object { Join-Path $_.FullName 'tools/python.exe' } |
            Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1)
    }
    if (-not $basePython) {
        $pythonLauncher = (Get-Command py.exe -ErrorAction SilentlyContinue | Select-Object -First 1).Source
        if ($pythonLauncher) { $basePython = & $pythonLauncher -3 -c 'import sys; print(sys.executable)' }
    }
    if (-not $basePython) { throw 'Install Python 3.10 or newer and reopen PowerShell.' }
    & $basePython -c 'import sys; sys.exit(sys.version_info < (3, 10))'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 or newer is required.' }
    & $basePython -m venv (Join-Path $projectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
}

& $python -c "import importlib.util, sys; sys.exit(any(importlib.util.find_spec(name) is None for name in ('win32com', 'numpy', 'pandas', 'exchange_calendars')))"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing Python dependencies...'
    & $python -m pip --disable-pip-version-check install -r (Join-Path $projectRoot 'engine/requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Could not install Python dependencies.' }
}

$previousPython = $env:PRICES_PYTHON
$previousPort = $env:PRICES_PORT
$previousBrowser = $env:PRICES_OPEN_BROWSER
Push-Location -LiteralPath $serverDirectory
try {
    & $node --input-type=module -e "await Promise.all(['express','winston','winston-daily-rotate-file','ws','chart.js'].map(name => import(name).catch(() => process.exit(1))))"
    if ($LASTEXITCODE -ne 0) {
        $npm = Join-Path (Split-Path -Parent $node) 'npm.cmd'
        if (-not (Test-Path -LiteralPath $npm)) { throw 'npm is required to install Node.js dependencies.' }
        Write-Host 'Installing Node.js dependencies...'
        & $npm ci --no-fund
        if ($LASTEXITCODE -ne 0) { throw 'Could not install Node.js dependencies.' }
    }

    $env:PRICES_PYTHON = $python
    $env:PRICES_PORT = "$Port"
    $env:PRICES_OPEN_BROWSER = if ($NoBrowser) { '0' } else { '1' }
    & $node (Join-Path $serverDirectory 'server.js')
    if ($LASTEXITCODE -ne 0) { throw "The Node.js server stopped with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
    $env:PRICES_PYTHON = $previousPython
    $env:PRICES_PORT = $previousPort
    $env:PRICES_OPEN_BROWSER = $previousBrowser
}
