#!/usr/bin/env pwsh
<#
.SYNOPSIS
Local mysqld helper for this repo (Windows PowerShell version).

.DESCRIPTION
Goal: run a local MySQL with datadir under ./data/mysql/ so the database lives inside the project.

Requirements:
- mysqld available in PATH
- mysql client available in PATH (for bootstrap)
- MySQL 8.x recommended

Usage:
  .\scripts\local_mysql.ps1 init
  .\scripts\local_mysql.ps1 start
  .\scripts\local_mysql.ps1 bootstrap
  .\scripts\local_mysql.ps1 stop
  .\scripts\local_mysql.ps1 status

After init, put connection info into ./data/.env.local (gitignored).
#>

param(
  [Parameter(Position = 0)]
  [string]$Command
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$script:ScriptPath = $PSCommandPath

function Get-RootDir {
  return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function To-MySqlPath([string]$Path) {
  # MySQL config parsing is happiest with forward slashes.
  return ($Path -replace "\\", "/")
}

function Load-EnvLocal([string]$RootDir) {
  $envFile = Join-Path $RootDir "data\.env.local"
  if (-not (Test-Path -LiteralPath $envFile)) {
    return
  }

  foreach ($line in Get-Content -LiteralPath $envFile) {
    $t = $line.Trim()
    if ($t.Length -eq 0) { continue }
    if ($t.StartsWith("#")) { continue }

    # Support simple KEY=VALUE (no export, no quotes parsing beyond trimming)
    $m = [regex]::Match($t, '^(?<k>[A-Za-z_][A-Za-z0-9_]*)=(?<v>.*)$')
    if (-not $m.Success) { continue }
    $k = $m.Groups["k"].Value
    $v = $m.Groups["v"].Value.Trim()

    # Strip surrounding single/double quotes if present.
    if (($v.StartsWith('"') -and $v.EndsWith('"')) -or ($v.StartsWith("'") -and $v.EndsWith("'"))) {
      $v = $v.Substring(1, $v.Length - 2)
    }

    Set-Item -Path "Env:$k" -Value $v
  }
}

function Write-Conf(
  [string]$ConfFile,
  [string]$MysqlDir,
  [string]$SocketFile,
  [string]$PidFile,
  [string]$LogFile,
  [int]$Port
) {
  $shouldWrite = $true
  if (Test-Path -LiteralPath $ConfFile) {
    $shouldWrite = $false
    try {
      $bytes = [System.IO.File]::ReadAllBytes($ConfFile)
      if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        # UTF-8 BOM can trip mysqld config parsing on some builds.
        $shouldWrite = $true
      } else {
        $text = [System.IO.File]::ReadAllText($ConfFile)
        $trim = $text.TrimStart()
        if (-not $trim.StartsWith("[mysqld]")) {
          $shouldWrite = $true
        } elseif ($text -notmatch "(?m)^\s*port=$Port\s*$") {
          # Keep config in sync when the port changes.
          $shouldWrite = $true
        } elseif ($text -match "(?m)^\s*skip-name-resolve\s*=\s*1\s*$") {
          # On Windows, this can break local root bootstrap when root@localhost is present.
          $shouldWrite = $true
        }
      }
    } catch {
      $shouldWrite = $true
    }
  }

  if (-not $shouldWrite) {
    return
  }

  $mysqlDataDir = Join-Path $MysqlDir "data"

  $conf = @"
[mysqld]
datadir=$(To-MySqlPath $mysqlDataDir)
socket=$(To-MySqlPath $SocketFile)
pid-file=$(To-MySqlPath $PidFile)
log-error=$(To-MySqlPath $LogFile)
port=$Port
bind-address=127.0.0.1
mysqlx=0

[client]
socket=$(To-MySqlPath $SocketFile)
port=$Port
"@

  $confDir = Split-Path -Parent $ConfFile
  New-Item -ItemType Directory -Force -Path $confDir | Out-Null
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($ConfFile, $conf, $utf8NoBom)
}

function Ensure-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Required command not found in PATH: $Name"
  }
}

function Assert-LastExitCode([string]$What) {
  if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
    throw "$What failed with exit code $LASTEXITCODE"
  }
}

function Cmd-Init(
  [string]$ConfFile,
  [string]$MysqlDir
) {
  Ensure-Command "mysqld"

  $dataDir = Join-Path $MysqlDir "data"
  New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

  if (Test-Path -LiteralPath (Join-Path $dataDir "mysql")) {
    Write-Host "already initialized: $dataDir"
    return
  }

  Write-Host "initializing mysqld datadir=$dataDir (insecure, no root password)..."
  & mysqld "--defaults-file=$ConfFile" "--initialize-insecure"
  Assert-LastExitCode "mysqld --initialize-insecure"
  Write-Host "init done. now run: `"$script:ScriptPath`" start"
}

function Cmd-Start(
  [string]$RootDir,
  [string]$ConfFile,
  [string]$MysqlDir,
  [string]$PidFile,
  [string]$SocketFile,
  [string]$LogFile,
  [int]$Port
) {
  Ensure-Command "mysqld"

  Load-EnvLocal -RootDir $RootDir

  $dataDir = Join-Path $MysqlDir "data"
  New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

  if (Test-Path -LiteralPath $PidFile) {
    Write-Host "pid file exists: $PidFile (already running?)"
    return
  }

  Write-Host "starting mysqld ..."

  $helpText = ""
  try {
    $helpText = (& mysqld --help --verbose 2>$null | Out-String)
  } catch {
    $helpText = ""
  }

  # Prefer --daemonize if supported (MySQL 8). Fallback: background process.
  if ($helpText -match "--daemonize") {
    & mysqld "--defaults-file=$ConfFile" "--daemonize"
    Assert-LastExitCode "mysqld --daemonize"
  } else {
    # Start mysqld in background. PID file should be written by mysqld per config.
    Start-Process -FilePath "mysqld" -ArgumentList @("--defaults-file=$ConfFile") | Out-Null
  }

  # Wait briefly for startup; if it dies (e.g. port in use), pid-file won't appear.
  $started = $false
  for ($i = 0; $i -lt 20; $i++) {
    if (Test-Path -LiteralPath $PidFile) {
      $started = $true
      break
    }
    Start-Sleep -Milliseconds 500
  }

  if (-not $started) {
    $hint = "mysqld did not create pid file: $PidFile. Check log: $LogFile"
    $hint2 = "If you already have MySQL running, change MOMENTUM_MYSQL_PORT in data/.env.local (e.g. 3307) and re-run start."
    if (Test-Path -LiteralPath $LogFile) {
      $tail = (Get-Content -LiteralPath $LogFile -Tail 25 -ErrorAction SilentlyContinue) -join "`n"
      throw "$hint`n$hint2`n--- mysql.err (tail) ---`n$tail"
    }
    throw "$hint`n$hint2"
  }

  Write-Host "mysqld started. socket=$SocketFile port=$Port"
}

function Cmd-Bootstrap(
  [string]$RootDir,
  [string]$PidFile,
  [string]$SocketFile,
  [int]$Port
) {
  Ensure-Command "mysql"

  Load-EnvLocal -RootDir $RootDir

  if (-not (Test-Path -LiteralPath $PidFile)) {
    throw "mysqld does not look running (pid file missing: $PidFile). Run: `"$script:ScriptPath`" start"
  }

  $db = if ($env:MOMENTUM_MYSQL_DB) { $env:MOMENTUM_MYSQL_DB } else { "momentum" }
  $user = if ($env:MOMENTUM_MYSQL_USER) { $env:MOMENTUM_MYSQL_USER } else { "momentum" }
  $pass = if ($env:MOMENTUM_MYSQL_PASSWORD) { $env:MOMENTUM_MYSQL_PASSWORD } else { "momentum" }

  Write-Host "bootstrapping database/user on local mysqld ..."

  # Prefer socket if present; on Windows this may not work, so fallback to TCP.
  $mysqlArgs = New-Object System.Collections.Generic.List[string]
  if (Test-Path -LiteralPath $SocketFile) {
    $mysqlArgs.Add("--protocol=SOCKET")
    $mysqlArgs.Add("--socket=$(To-MySqlPath $SocketFile)")
  } else {
    $mysqlArgs.Add("--protocol=TCP")
    $mysqlArgs.Add("--host=localhost")
    $mysqlArgs.Add("--port=$Port")
  }
  $mysqlArgs.Add("-uroot")

  $sql = @"
CREATE DATABASE IF NOT EXISTS ``$db`` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER IF NOT EXISTS '$user'@'127.0.0.1' IDENTIFIED BY '$pass';
CREATE USER IF NOT EXISTS '$user'@'localhost' IDENTIFIED BY '$pass';
GRANT ALL PRIVILEGES ON ``$db``.* TO '$user'@'127.0.0.1';
GRANT ALL PRIVILEGES ON ``$db``.* TO '$user'@'localhost';
FLUSH PRIVILEGES;
"@

  $argsArray = $mysqlArgs.ToArray()
  $sql | & mysql @argsArray
  Assert-LastExitCode "mysql bootstrap"

  Write-Host "bootstrap done. db=$db user=$user"
}

function Cmd-Stop([string]$PidFile) {
  if (-not (Test-Path -LiteralPath $PidFile)) {
    Write-Host "no pid file: $PidFile (not running?)"
    return
  }

  $pidRaw = (Get-Content -LiteralPath $PidFile -ErrorAction Stop | Select-Object -First 1).Trim()
  [int]$mysqlPid = 0
  if (-not [int]::TryParse($pidRaw, [ref]$mysqlPid)) {
    Write-Host "invalid pid file contents: $PidFile"
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    return
  }

  Write-Host "stopping mysqld pid=$mysqlPid ..."
  Stop-Process -Id $mysqlPid -ErrorAction SilentlyContinue
  Start-Sleep -Seconds 1
  Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
  Write-Host "stopped."
}

function Cmd-Status(
  [string]$PidFile,
  [string]$SocketFile,
  [int]$Port
) {
  if (Test-Path -LiteralPath $PidFile) {
    $mysqlPid = (Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1).Trim()
    Write-Host "RUNNING pid=$mysqlPid socket=$SocketFile port=$Port"
  } else {
    Write-Host "STOPPED"
  }
}

$ROOT_DIR = Get-RootDir
$MYSQL_DIR = Join-Path $ROOT_DIR "data\mysql"
$CONF_FILE = Join-Path $MYSQL_DIR "my.cnf"
$SOCKET_FILE = Join-Path $MYSQL_DIR "mysql.sock"
$PID_FILE = Join-Path $MYSQL_DIR "mysql.pid"
$LOG_FILE = Join-Path $MYSQL_DIR "mysql.err"

New-Item -ItemType Directory -Force -Path $MYSQL_DIR | Out-Null

# Load repo-local env before reading MOMENTUM_MYSQL_PORT etc.
Load-EnvLocal -RootDir $ROOT_DIR

$PORT = 3306
if ($env:MOMENTUM_MYSQL_PORT) {
  [int]$tmp = 0
  if ([int]::TryParse($env:MOMENTUM_MYSQL_PORT, [ref]$tmp)) {
    $PORT = $tmp
  }
}

Write-Conf -ConfFile $CONF_FILE -MysqlDir $MYSQL_DIR -SocketFile $SOCKET_FILE -PidFile $PID_FILE -LogFile $LOG_FILE -Port $PORT

if (-not $Command) {
  Write-Host "usage: $($MyInvocation.MyCommand.Name) {init|start|bootstrap|stop|status}"
  exit 2
}

switch ($Command) {
  "init" { Cmd-Init -ConfFile $CONF_FILE -MysqlDir $MYSQL_DIR }
  "start" { Cmd-Start -RootDir $ROOT_DIR -ConfFile $CONF_FILE -MysqlDir $MYSQL_DIR -PidFile $PID_FILE -SocketFile $SOCKET_FILE -LogFile $LOG_FILE -Port $PORT }
  "bootstrap" { Cmd-Bootstrap -RootDir $ROOT_DIR -PidFile $PID_FILE -SocketFile $SOCKET_FILE -Port $PORT }
  "stop" { Cmd-Stop -PidFile $PID_FILE }
  "status" { Cmd-Status -PidFile $PID_FILE -SocketFile $SOCKET_FILE -Port $PORT }
  default {
    Write-Host "usage: $($MyInvocation.MyCommand.Name) {init|start|bootstrap|stop|status}"
    exit 2
  }
}

