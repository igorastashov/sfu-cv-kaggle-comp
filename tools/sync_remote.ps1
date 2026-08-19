<#
.SYNOPSIS
    Быстрая синхронизация кода и ноутбука на машину 192.168.52.118.

.EXAMPLE
    # Обновить stand + notebooks + tools (секунды)
    powershell -ExecutionPolicy Bypass -File tools\sync_remote.ps1

.EXAMPLE
    # Первый раз: всё включая data и models (~4 ГБ) + поднять стенд
    powershell -ExecutionPolicy Bypass -File tools\sync_remote.ps1 -Full -Start

.EXAMPLE
    # Обновить код и перезапустить контейнер
    powershell -ExecutionPolicy Bypass -File tools\sync_remote.ps1 -Restart
#>
[CmdletBinding()]
param(
    [string]$Host = "192.168.52.118",
    [string]$User = "dmd",
    [string]$Password = "",
    [switch]$Full,
    [switch]$Start,
    [switch]$Restart,
    [switch]$Status
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$argsPy = @("$PSScriptRoot\sync_remote.py", "--host", $Host, "--user", $User)
if ($Password) { $argsPy += @("--password", $Password) }
if ($Full)    { $argsPy += "--full" }
if ($Start)   { $argsPy += "--start" }
if ($Restart) { $argsPy += "--restart" }
if ($Status)  { $argsPy += "--status" }

& $py @argsPy
