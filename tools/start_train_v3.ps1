<#
  PHOTOWORLD v3 训练（参数搜索）启动器
  ------------------------------------------------------------------
  v3 没有可反传的网络，重建是"确定性管线 + 十来个几何旋钮"。所以这里的"训练"
  就是**用验收指标搜参数**：每组参数真的跑一遍 bake，再光栅化回原点视角拿
  PSNR/SSIM 打分。搜完给出排名与逐图最优（reports/v3_sweep/sweep.md）。

  双击「PHOTOWORLD 训练.bat」等价于：
    1. 起静态站（8199，已跑就复用）
    2. python tools/v3_sweep.py（默认 4 张内置世界 × 6 组参数，关扩散，几分钟）
    3. 打印报告路径，并提示下一步（转身观感要另跑 multiview）

  常用参数：
    -Limit 6                    多评几张
    -Grid "cliff=1.4,1.8,2.4"   自定义网格
    -Step 4                     网格抽稀更细（慢，但更接近交付口径）
    -WithDiffusion              连全景/地形一起评（慢约 4 倍）
#>
[CmdletBinding()]
param(
  [switch]$Stop,
  [int]$Port = 8199,
  [int]$Limit = 4,
  [string]$Grid = '',
  [int]$Step = 6,
  [switch]$WithDiffusion,
  [switch]$Visible
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root 'logs'

function Say([string]$msg, [string]$kind = 'info') {
  $c = switch ($kind) { 'ok' { 'Green' } 'warn' { 'Yellow' } 'err' { 'Red' } default { 'Gray' } }
  Write-Host $msg -ForegroundColor $c
}
function Die([string]$msg) {
  Say $msg 'err'
  # 非交互（被自动化调用）时 Read-Host 会抛异常，这里跳过等待
  if ([Environment]::UserInteractive) {
    try { Say '按回车关闭…' 'gray'; [void](Read-Host) } catch { }
  }
  exit 1
}
function Resolve-Python {
  $cands = @()
  $cands += (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
  $cands += (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -First 1).Source
  $cands += (Get-ChildItem "$env:USERPROFILE\.workbuddy\binaries\python\versions\*\python.exe" `
             -ErrorAction SilentlyContinue | Sort-Object FullName -Descending |
             Select-Object -First 1).FullName
  foreach ($c in $cands) { if ($c -and (Test-Path $c)) { return $c } }
  return $null
}

if ($Stop) {
  # 训练不需要常驻服务，-Stop 只是把静态站收掉，保持和启动器一致的使用习惯
  & (Join-Path $PSScriptRoot 'start_v3.ps1') -Stop -Port $Port
  exit 0
}

Say '==== PHOTOWORLD v3 训练（参数搜索）====' 'ok'
$py = Resolve-Python
if (-not $py) { Die '找不到 python。请先安装 Python 3.10+。' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

# 静态站：multiview 与"看结果"都要它；这里只确保它在跑，失败不阻塞训练
try { & (Join-Path $PSScriptRoot 'start_v3.ps1') -NoBrowser -Quiet -Port $Port } catch {
  Say "静态站没起来（不影响参数搜索）：$($_.Exception.Message)" 'warn'
}

$argsList = @('tools/v3_sweep.py', '--limit', "$Limit", '--step', "$Step")
if ($Grid) { $argsList += @('--grid', $Grid) }
if ($WithDiffusion) { $argsList += '--with-diffusion' }

Say "运行：python $($argsList -join ' ')" 'gray'
Push-Location $Root
try {
  & $py @argsList
  $code = $LASTEXITCODE
} finally {
  Pop-Location
}

if ($code -ne 0) { Die "参数搜索失败（退出码 $code）。" }

$md = Join-Path $Root 'reports/v3_sweep/sweep.md'
if (Test-Path $md) {
  Say ''
  Say "报告已生成：$md" 'ok'
  if ($Visible) { Start-Process $md }
}
Say ''
Say '下一步（转身观感，需要 Edge/Chrome）：' 'gray'
$mv = '  node tools/multiview.mjs --base http://127.0.0.1:' + $Port + ' --glb-list reports/v3_list.txt --plan v3 --views 32 --out shots/mv3 --json reports/mv3.json'
Say $mv 'gray'
Say ''
Say '把最优参数写进默认值：改 pipeline/v3/bake.py 里 frustum.build 的 cfg 默认（cliff/skirt/ground_tol）。' 'gray'
Start-Sleep -Milliseconds 1200
