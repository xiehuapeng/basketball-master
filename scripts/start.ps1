param(
    [int]$Port = 8000,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv311\Scripts\python.exe"
$RuntimeDir = Join-Path $ProjectRoot "artifacts\runtime"
$PidFile = Join-Path $RuntimeDir "api-server.pid"
$PortFile = Join-Path $RuntimeDir "api-server.port"
$StdoutLog = Join-Path $RuntimeDir "server.stdout.log"
$StderrLog = Join-Path $RuntimeDir "server.stderr.log"
$HealthUrl = "http://127.0.0.1:$Port/health"
$AppUrl = "http://127.0.0.1:$Port/"

function Get-RecordedServer {
    if (-not (Test-Path -LiteralPath $PidFile)) {
        return $null
    }

    $recordedPid = 0
    if (-not [int]::TryParse((Get-Content -LiteralPath $PidFile -Raw).Trim(), [ref]$recordedPid)) {
        Remove-Item -LiteralPath $PidFile, $PortFile -Force -ErrorAction SilentlyContinue
        return $null
    }

    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $recordedPid" -ErrorAction SilentlyContinue
    if ($null -eq $process -or $process.CommandLine -notmatch "basketball_analyzer\.api:app") {
        Remove-Item -LiteralPath $PidFile, $PortFile -Force -ErrorAction SilentlyContinue
        return $null
    }
    return $process
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found: $Python. Create .venv311 with Python 3.11 first."
}

New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null

$existing = Get-RecordedServer
if ($null -ne $existing) {
    Write-Host "Basketball Analyzer is already running (PID $($existing.ProcessId))."
    $recordedPort = 0
    if ((Test-Path -LiteralPath $PortFile) -and
        [int]::TryParse((Get-Content -LiteralPath $PortFile -Raw).Trim(), [ref]$recordedPort)) {
        $AppUrl = "http://127.0.0.1:$recordedPort/"
    }
    if (-not $NoBrowser) {
        Start-Process $AppUrl
    }
    exit 0
}

$listener = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $listener) {
    throw "Port $Port is already used by PID $($listener.OwningProcess). Stop that service or run scripts\start.ps1 -Port <port>."
}

Push-Location $ProjectRoot
try {
    & $Python -c "import basketball_analyzer, fastapi, uvicorn, ultralytics" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing project dependencies for the first run..."
        & $Python -m pip install -e ".[api,local]"
        if ($LASTEXITCODE -ne 0) {
            throw "Dependency installation failed."
        }
    }

    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot "models\basketball_yolo11.pt"))) {
        Write-Warning "Custom basketball model is missing. Local detection will fall back to yolov8n.pt."
    }

    Remove-Item -LiteralPath $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue
    $process = Start-Process -FilePath $Python `
        -ArgumentList @("-m", "uvicorn", "basketball_analyzer.api:app", "--host", "127.0.0.1", "--port", "$Port") `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StdoutLog `
        -RedirectStandardError $StderrLog `
        -PassThru
    Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
    Set-Content -LiteralPath $PortFile -Value $Port -Encoding ascii

    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Seconds 1
        if ($process.HasExited) {
            break
        }
        try {
            $response = Invoke-RestMethod -Uri $HealthUrl -TimeoutSec 2
            if ($response.status -eq "ok") {
                $ready = $true
                break
            }
        }
        catch {
            # The server is still starting.
        }
    }

    if (-not $ready) {
        Remove-Item -LiteralPath $PidFile, $PortFile -Force -ErrorAction SilentlyContinue
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
        $details = if (Test-Path -LiteralPath $StderrLog) {
            (Get-Content -LiteralPath $StderrLog -Tail 20) -join [Environment]::NewLine
        } else {
            "No server error log was created."
        }
        throw "Server did not become healthy within 60 seconds.`n$details"
    }

    Write-Host "Basketball Analyzer started successfully."
    Write-Host "URL: $AppUrl"
    Write-Host "PID: $($process.Id)"
    Write-Host "Logs: $RuntimeDir"
    if (-not $NoBrowser) {
        Start-Process $AppUrl
    }
}
finally {
    Pop-Location
}
