$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot "artifacts\runtime"
$PidFile = Join-Path $RuntimeDir "api-server.pid"
$PortFile = Join-Path $RuntimeDir "api-server.port"

if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "Basketball Analyzer is not running (no PID file)."
    exit 0
}

$recordedPid = 0
if (-not [int]::TryParse((Get-Content -LiteralPath $PidFile -Raw).Trim(), [ref]$recordedPid)) {
    Remove-Item -LiteralPath $PidFile, $PortFile -Force -ErrorAction SilentlyContinue
    Write-Host "Removed an invalid PID file."
    exit 0
}

$process = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedPid" -ErrorAction SilentlyContinue
if ($null -eq $process) {
    Remove-Item -LiteralPath $PidFile, $PortFile -Force -ErrorAction SilentlyContinue
    Write-Host "The recorded process no longer exists. Removed the stale PID file."
    exit 0
}

if ($process.CommandLine -notmatch "basketball_analyzer\.api:app") {
    throw "PID $recordedPid is not the Basketball Analyzer server. Refusing to stop it."
}

Stop-Process -Id $recordedPid -ErrorAction Stop
try {
    Wait-Process -Id $recordedPid -Timeout 10 -ErrorAction Stop
}
catch {
    Stop-Process -Id $recordedPid -Force -ErrorAction SilentlyContinue
}

Remove-Item -LiteralPath $PidFile, $PortFile -Force -ErrorAction SilentlyContinue
Write-Host "Basketball Analyzer stopped (PID $recordedPid)."
