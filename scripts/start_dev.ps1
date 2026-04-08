#!/usr/bin/env pwsh
<#
.SYNOPSIS
Stop local MySQL (if any), start + bootstrap it, then run the ETF Momentum API (uvicorn).

.DESCRIPTION
Runs from repo root context (paths are resolved from this script location).
Requires: PowerShell, mysqld/mysql in PATH, and .venv with the project installed.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$localMysql = Join-Path $PSScriptRoot "local_mysql.ps1"
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $py)) {
  throw "Virtualenv Python not found: $py. From repo root run: python3 -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e '.[dev]'"
}

Push-Location $root
try {
  & $localMysql stop
  & $localMysql start
  & $localMysql bootstrap
  & $py -m uvicorn etf_momentum.app:app --reload --reload-dir "$root\src" --reload-exclude "tests/*" --port 8000
} finally {
  Pop-Location
}
