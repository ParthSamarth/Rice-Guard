# server/run_server.ps1
# Convenience launcher for the RiceGuard AI local API server.
# Run from anywhere -- resolves paths relative to this script's own location.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $root "..\.venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Error "Could not find the project venv at $venvPython -- run this from the project's existing .venv setup."
    exit 1
}

# Single source of truth for the port -- server/main.py reads the same
# RICEGUARD_PORT env var for its mDNS advertisement, so the two can never
# drift apart. Override with e.g. $env:RICEGUARD_PORT=8001 before running
# this script if 8000 is taken.
if (-not $env:RICEGUARD_PORT) { $env:RICEGUARD_PORT = "8000" }

Write-Host "Starting RiceGuard AI server (YOLOv8n @ 640 + ResNet50) ..." -ForegroundColor Green
Write-Host "Health check:  http://localhost:$($env:RICEGUARD_PORT)/health"
Write-Host "API docs:      http://localhost:$($env:RICEGUARD_PORT)/docs"
Write-Host "The Android app finds this server automatically on the same Wi-Fi (mDNS)."
Write-Host "If auto-discovery doesn't work, find this PC's LAN IP with 'ipconfig' and enter it manually in Settings."
Write-Host ""

Set-Location $root
& $venvPython -m uvicorn main:app --host 0.0.0.0 --port $env:RICEGUARD_PORT
