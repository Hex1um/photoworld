<#
  PHOTOWORLD v3 启动器（兼容入口）
  ------------------------------------------------------------------
  真正的逻辑在 tools\start.py —— 零第三方依赖，不受 PowerShell 执行策略限制。
  双击仓库根目录的 start_v3.bat 即可；本文件仅为历史 .ps1 调用方保留。

  注意：直接运行 .ps1 仍需 -ExecutionPolicy Bypass（Windows 默认策略禁止脚本）。
  推荐：python tools/start.py
#>
[CmdletBinding()]
param(
  [switch]$Stop,
  [switch]$NoBrowser,
  [switch]$Visible,
  [int]$Port = 8199,
  [string]$World = '',
  [string]$Photo = '',
  [switch]$NoDiffusion,
  [switch]$Quiet,
  [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
# 找 Python：本机没有装系统 Python，PATH 上也没有 python/py。
# 实际只有 WorkBuddy 托管的那份（%USERPROFILE%/.workbuddy/binaries/python/versions/*）。
# 原来只查 PATH，于是桌面快捷方式永远报 [ERR] Python not found。
function Find-Python {
  if ($env:PHOTOWORLD_PY -and (Test-Path $env:PHOTOWORLD_PY)) { return $env:PHOTOWORLD_PY }
  foreach ($n in @('python', 'python3', 'py')) {
    $c = Get-Command $n -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c -and $c.Source -and (Test-Path $c.Source)) { return $c.Source }
  }
  $base = Join-Path $env:USERPROFILE '.workbuddy/binaries/python/versions'
  if (Test-Path $base) {
    foreach ($d in (Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
                    Sort-Object Name -Descending)) {
      $exe = Join-Path $d.FullName 'python.exe'
      if (Test-Path $exe) { return $exe }
    }
  }
  foreach ($root in @("$env:LOCALAPPDATA/Programs/Python", "$env:ProgramFiles")) {
    if (-not (Test-Path $root)) { continue }
    foreach ($d in (Get-ChildItem $root -Directory -Filter 'Python3*' -ErrorAction SilentlyContinue)) {
      $exe = Join-Path $d.FullName 'python.exe'
      if (Test-Path $exe) { return $exe }
    }
  }
  return $null
}
$py = Find-Python
if (-not $py) {
  Write-Host '[ERR] Python not found.' -ForegroundColor Red
  Write-Host '      install one, or: setx PHOTOWORLD_PY C://path//to//python.exe' -ForegroundColor Yellow
  exit 1
}

$argv = @("$Root\tools\start.py", '--port', "$Port")
if ($Stop) { $argv += '--stop' }
if ($NoBrowser) { $argv += '--no-browser' }
if ($Visible) { $argv += '--visible' }
if ($Quiet) { $argv += '--quiet' }
if ($World) { $argv += @('--world', $World) }
if ($Photo) { $argv += @('--photo', $Photo) }
if ($NoDiffusion) { $argv += '--no-diffusion' }
if ($Rest) { $argv += $Rest }
& $py @argv
exit $LASTEXITCODE
