<#
.SYNOPSIS
    Упаковка стенда в один архив для переноса на машину с ускорителем.

.DESCRIPTION
    Кладёт в архив всё, что нужно для запуска: код, ноутбуки, документы,
    описание контейнера, набор данных и веса моделей. Не кладёт то, что
    на целевой машине бесполезно или пересобирается: историю git,
    локальное окружение Windows, кеши Python.

    Сжатие отключено. Кадры внутри набора данных хранятся в сжатом виде,
    веса моделей сжимаются плохо, а проход по 4 ГБ занимает десятки минут.

    Имена переменных латиницей намеренно: Windows PowerShell 5.1 читает
    файл без метки порядка байтов как однобайтовую кодировку, и кириллица
    в именах ломает разбор. Сам файл сохранён с меткой.

.PARAMETER OutDir
    Куда положить архив. По умолчанию папка рядом с репозиторием.

.PARAMETER Token
    Пароль доступа к ноутбуку. По умолчанию создаётся случайный.

.PARAMETER NoData
    Не класть набор данных.

.PARAMETER NoModels
    Не класть веса моделей.

.PARAMETER NoChecksum
    Не считать контрольную сумму. Ускоряет упаковку, но переносить
    архив придётся без проверки целостности.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\pack_transfer.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\pack_transfer.ps1 -OutDir E:\ -Token moi-parol
#>
[CmdletBinding()]
param(
    [string]$OutDir = "",
    [string]$Token = "",
    [switch]$NoData,
    [switch]$NoModels,
    [switch]$NoChecksum
)

$ErrorActionPreference = "Stop"

# --- где что лежит -----------------------------------------------------------

$root   = Split-Path -Parent $PSScriptRoot
$leaf   = Split-Path -Leaf $root
$parent = Split-Path -Parent $root

if (-not $OutDir) { $OutDir = $parent }
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Force $OutDir | Out-Null }
$OutDir = (Resolve-Path $OutDir).Path

$stamp = Get-Date -Format "yyyyMMdd"
$zip   = Join-Path $OutDir "$leaf-$stamp.zip"

Write-Host "Стенд:  $root"
Write-Host "Архив:  $zip"
Write-Host ""

# --- проверка состава --------------------------------------------------------
# Пустая папка данных или моделей означает, что ноутбук на целевой машине
# не отработает. Лучше остановиться здесь, чем на площадке.

foreach ($item in @("stand", "notebooks", "docker", "requirements.txt")) {
    if (-not (Test-Path (Join-Path $root $item))) {
        throw "Нет $item. Запускать скрипт нужно из репозитория стенда."
    }
}

function Get-DirSize([string]$path) {
    if (-not (Test-Path $path)) { return 0 }
    $sum = (Get-ChildItem $path -Recurse -File -Force -ErrorAction SilentlyContinue |
            Measure-Object Length -Sum).Sum
    if ($null -eq $sum) { return 0 }
    return $sum
}

$dataSize   = Get-DirSize (Join-Path $root "data")
$modelsSize = Get-DirSize (Join-Path $root "models")

if (-not $NoData -and $dataSize -lt 100MB) {
    throw ("Набор данных пуст или неполон ({0} МБ). Порядок выгрузки в docs/10-waymo-download.md." -f [math]::Round($dataSize / 1MB))
}
if (-not $NoModels -and $modelsSize -lt 1GB) {
    throw ("Веса моделей отсутствуют ({0} МБ). Ожидается около 3,6 ГБ в models." -f [math]::Round($modelsSize / 1MB))
}

# --- пароль доступа ----------------------------------------------------------
# Файл .env читается описанием контейнера при запуске. Если пароль не задан,
# ноутбук откроется без него, поэтому пустое значение не допускается.

$envFile = Join-Path $root "docker\.env"

if (Test-Path $envFile) {
    Write-Host "Пароль доступа: взят из docker\.env"
    $line = @(Get-Content $envFile | Where-Object { $_ -match '^JUPYTER_TOKEN=' }) | Select-Object -First 1
    $Token = $line -replace '^JUPYTER_TOKEN=', ''
} else {
    if (-not $Token) {
        $bytes = New-Object byte[] 24
        [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $Token = ([Convert]::ToBase64String($bytes) -replace '[^A-Za-z0-9]', '')
    }
    $envText = @"
# Создан автоматически при упаковке $(Get-Date -Format 'yyyy-MM-dd HH:mm').
# Пароль доступа к ноутбуку. Пустое значение открывает доступ без пароля.
JUPYTER_TOKEN=$Token

# Порт, с которого ноутбук виден на целевой машине.
JUPYTER_PORT=8888

# Пути к набору данных и весам. Отсчитываются от папки docker.
DATA_PATH=../data
MODELS_PATH=../models
"@
    # Без метки порядка байтов: с ней первая переменная читается с искажённым именем.
    [IO.File]::WriteAllText($envFile, $envText, (New-Object Text.UTF8Encoding($false)))
    Write-Host "Пароль доступа: создан новый, записан в docker\.env"
}

if (-not $Token) { throw "В docker\.env пустой JUPYTER_TOKEN. Впишите пароль и повторите." }

# --- чем упаковываем ---------------------------------------------------------
# Встроенный в PowerShell 5.1 Compress-Archive не берёт файлы больше 2 ГБ,
# а веса модели сегментации весят 3,3 ГБ. Поэтому только эти два способа.

$seven = @("$env:ProgramFiles\7-Zip\7z.exe", "${env:ProgramFiles(x86)}\7-Zip\7z.exe") |
         Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $seven) { $seven = (Get-Command 7z.exe -ErrorAction SilentlyContinue).Source }

$skip = @(".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints",
          ".cache", ".pytest_cache", ".idea", ".vscode", ".claude")
if ($NoData)   { $skip += "data" }
if ($NoModels) { $skip += "models" }

if (Test-Path $zip) { Remove-Item $zip -Force }

$clock = [Diagnostics.Stopwatch]::StartNew()
Push-Location $parent
try {
    if ($seven) {
        Write-Host "Упаковщик: 7-Zip"
        # Ключ -mcu=on обязателен. Без него имена файлов с кириллицей пишутся
        # в однобайтовой кодировке DOS и на целевой машине распаковываются
        # искажёнными. Ролики занятия названы по-русски.
        $args7 = @("a", "-tzip", "-mx0", "-mcu=on", "-bso0", "-bsp1", $zip, $leaf)
        foreach ($s in $skip) { $args7 += "-xr!$s" }
        $args7 += "-xr!*.pyc"
        $args7 += "-xr!*.csv"
        & $seven @args7
        if ($LASTEXITCODE -ne 0) { throw "7-Zip завершился с кодом $LASTEXITCODE" }
    } else {
        Write-Host "Упаковщик: tar (7-Zip не найден)"
        $argsTar = @("-a", "-c", "-f", $zip)
        foreach ($s in $skip) { $argsTar += "--exclude=$s" }
        $argsTar += @("--exclude=*.pyc", "--exclude=*.csv", $leaf)
        & tar.exe @argsTar
        if ($LASTEXITCODE -ne 0) { throw "tar завершился с кодом $LASTEXITCODE" }
    }
} finally {
    Pop-Location
}
$clock.Stop()

$zipSize = (Get-Item $zip).Length

# --- контрольная сумма -------------------------------------------------------

$hash = ""
if (-not $NoChecksum) {
    Write-Host "Считаю контрольную сумму."
    # Формат и регистр как у sha256sum, иначе проверка на целевой машине
    # не находит правильно оформленных строк.
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    $line = "$hash  $(Split-Path -Leaf $zip)`n"
    [IO.File]::WriteAllText("$zip.sha256", $line, (New-Object Text.UTF8Encoding($false)))
}

# --- итог --------------------------------------------------------------------

Write-Host ""
Write-Host ("Готово за {0} мин" -f [math]::Round($clock.Elapsed.TotalMinutes, 1))
Write-Host ""
Write-Host ("{0,-22} {1,8:N0} МБ" -f "набор данных", ($dataSize / 1MB))
Write-Host ("{0,-22} {1,8:N0} МБ" -f "веса моделей", ($modelsSize / 1MB))
Write-Host ("{0,-22} {1,8:N2} ГБ" -f "архив", ($zipSize / 1GB))
if ($hash) { Write-Host ("{0,-22} {1}" -f "контрольная сумма", ($hash.Substring(0, 16) + "...")) }
Write-Host ""
Write-Host "Файл:   $zip"
Write-Host "Пароль: $Token"
Write-Host ""
Write-Host "Дальше по docs/40-perenos-stenda.md, раздел 2."
