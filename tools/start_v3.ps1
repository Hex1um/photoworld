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
$py = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $py) { $py = (Get-Command py -ErrorAction SilentlyContinue | Select-Object -First 1).Source }
if (-not $py) { $py = (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -First 1).Source }
if (-not $py) { Write-Host '[ERR] Python not found.' -ForegroundColor Red; exit 1 }

$args = @("$Root\tools\start.py", '--port', "$Port")
if ($Stop) { $args += '--stop' }
if ($NoBrowser) { $args += '--no-browser' }
if ($Visible) { $args += '--visible' }
if ($Quiet) { $args += '--quiet' }
if ($World) { $args += @('--world', $World) }
if ($Photo) { $args += @('--photo', $Photo) }
if ($NoDiffusion) { $args += '--no-diffusion' }
if ($Rest) { $args += $Rest }
& $py @args
exit $LASTEXITCODE
