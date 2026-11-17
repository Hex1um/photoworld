<#
  PHOTOWORLD v3 一键启动器
  ------------------------------------------------------------------
  双击 start_v3.bat（或桌面上的 v3 启动脚本）即可：
    1. 起静态站  仓库根 · python -m http.server 8199
    2. 等 index.html 真的可访问（不是"端口开了"就算）
    3. 打开浏览器，直接进 **v3 资产**（?glb=assets/world-v3-alpine.glb）

  与 v2 启动器的区别：
    * v2 需要常驻 GPU 重建服务（server/，8100）；v3 的重建是**离线烘焙**，
      烘焙产物就是 world.glb，所以启动器只需要一个静态站 —— 少一个服务、
      少一处"服务没起来但页面看起来正常"的坑。
    * 想现场烘焙一张新照片，用 -Photo <路径>。

  设计上刻意做的几件事（沿用 v2 的教训）：
    * **可重复执行**：静态站已经在跑就复用，不会因为端口占用报错；
    * **失败要说人话**：缺 python / 缺资产 / 端口被别的程序占，都给出下一步该敲什么；
    * **能停**：PID 记在 logs/pids_v3.json，stop 时按它精确关闭（不误杀别的 python）。
#>
[CmdletBinding()]
param(
  [switch]$Stop,
  [switch]$NoBrowser,
  [switch]$Visible,
  [int]$Port = 8199,
  # 留空 = 打开主页（画廊）；给名字才直接进某个世界
  [string]$World = '',
  [string]$Photo = '',
  [switch]$NoDiffusion,
  [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root 'logs'
$PidFile = Join-Path $LogDir 'pids_v3.json'
$StaticUrl = "http://127.0.0.1:$Port"

# PS 5.1 的 Start-Process 在带 -RedirectStandardOutput 时会把环境变量拼成字典，
# 一旦存在只有大小写不同的重复项（HTTPS_PROXY/https_proxy、Path/PATH…）就抛
# "已添加项。字典中的关键字…"，服务直接起不来。这里把**所有**大小写重复项合并成
# 一个（保留全大写那份的值）—— 实测本机是 Path/PATH 撞了。
function Remove-DuplicateEnv {
  try {
    $vars = [Environment]::GetEnvironmentVariables()
    $groups = @{}
    foreach ($k in @($vars.Keys)) {
      $low = ([string]$k).ToLowerInvariant()
      if (-not $groups.ContainsKey($low)) { $groups[$low] = @() }
      $groups[$low] += [string]$k
    }
    foreach ($low in @($groups.Keys)) {
      $names = @($groups[$low])
      if ($names.Count -le 1) { continue }
      $canon = @($names | Where-Object { $_ -ceq $_.ToUpperInvariant() })[0]
      if (-not $canon) { $canon = $names[0] }
      $val = $vars[$canon]
      foreach ($n in $names) { [Environment]::SetEnvironmentVariable($n, $null, 'Process') }
      [Environment]::SetEnvironmentVariable($canon, [string]$val, 'Process')
    }
  } catch { }
}
Remove-DuplicateEnv

function Say([string]$msg, [string]$kind = 'info') {
  if ($Quiet -and $kind -ne 'err') { return }
  $c = switch ($kind) { 'ok' { 'Green' } 'warn' { 'Yellow' } 'err' { 'Red' } default { 'Gray' } }
  Write-Host $msg -ForegroundColor $c
}

function Die([string]$msg) {
  Say $msg 'err'
  if ([Environment]::UserInteractive) {
    try { Say '按回车关闭…' 'gray'; [void](Read-Host) } catch { }
  }
  exit 1
}

function Get-PortOwner([int]$port) {
  try {
    $c = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop |
         Select-Object -First 1
    if ($c -and $c.OwningProcess) { return [int]$c.OwningProcess }
  } catch { }
  try {
    foreach ($line in (& netstat -ano -p tcp 2>$null)) {
      $f = @($line -split '\s+' | Where-Object { $_ })
      if ($f.Count -ge 5 -and $f[1] -match ":$port$" -and $f[3] -eq 'LISTENING') {
        return [int]$f[4]
      }
    }
  } catch { }
  return 0
}

function Test-Http([string]$url, [int]$timeoutMs = 1200) {
  try {
    $r = Invoke-WebRequest -Uri $url -TimeoutSec ([Math]::Max(1, [int]($timeoutMs / 1000))) `
                           -UseBasicParsing -ErrorAction Stop
    return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400)
  } catch { return $false }
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

function Start-Hidden([string]$file, [string[]]$argList, [string]$workDir, [string]$logName) {
  $out = Join-Path $LogDir $logName
  $err = Join-Path $LogDir ($logName -replace '\.log$', '.err.log')
  $p = Start-Process -FilePath $file -ArgumentList $argList -WorkingDirectory $workDir `
        -WindowStyle $(if ($Visible) { 'Normal' } else { 'Hidden' }) `
        -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
  return $p
}

function Save-Pids([hashtable]$map) {
  if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }
  $map | ConvertTo-Json | Set-Content -Path $PidFile -Encoding utf8
}

# ------------------------------------------------------------------ 停止
if ($Stop) {
  $killed = @()
  if (Test-Path $PidFile) {
    try {
      $j = Get-Content $PidFile -Raw | ConvertFrom-Json
      foreach ($pp in @($j.PSObject.Properties)) {
        $procId = [int]$pp.Value
        if ($procId -gt 0 -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) {
          Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
          $killed += "$($pp.Name)=$procId"
        }
      }
    } catch { }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  }
  $owner = Get-PortOwner $Port
  if ($owner -gt 0) {
    try { Stop-Process -Id $owner -Force -ErrorAction Stop; $killed += "port$Port=$owner" } catch { }
  }
  if ($killed.Count) { Say ("已停止：" + ($killed -join ', ')) 'ok' }
  else { Say '没有在跑的 PHOTOWORLD v3 服务。' 'gray' }
  Start-Sleep -Milliseconds 400
  exit 0
}

# ------------------------------------------------------------------ 启动
Say '==== PHOTOWORLD v3 ====' 'ok'
Say "仓库：$Root" 'gray'

$py = Resolve-Python
if (-not $py) { Die '找不到 python。请先安装 Python 3.10+。' }
Say "python：$py" 'gray'

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

$needStart = $true
if (Test-Http "$StaticUrl/index.html") {
  Say "静态站已在 $Port 端口运行，直接复用。" 'gray'
  $needStart = $false
} else {
  $owner = Get-PortOwner $Port
  if ($owner -gt 0) {
    Die "端口 $Port 被 PID=$owner 占用且不是 PHOTOWORLD 静态站。换端口：start_v3.bat -Port 8200"
  }
}

if ($needStart) {
  Say "启动静态站 http://127.0.0.1:$Port …"
  $p = Start-Hidden $py @('-m', 'http.server', "$Port", '--bind', '127.0.0.1') $Root 'v3_static.log'
  Save-Pids @{ static = $p.Id }
  $ok = $false
  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 250
    if (Test-Http "$StaticUrl/index.html") { $ok = $true; break }
  }
  if (-not $ok) {
    Die "静态站没起来（连 10s 都无法访问 $StaticUrl/index.html）。看日志：logs\v3_static.err.log"
  }
  Say "静态站就绪（PID=$($p.Id)）。" 'ok'
}

$target = ''
if (-not $Photo -and -not $World) {
  # 默认打开**主页**（画廊/展厅），不硬跳进某个世界
  $url = "$StaticUrl/"
  if ($NoBrowser) { Say "（-NoBrowser）请手动打开：$url" 'gray' }
  else { Say "打开主页：$url" 'ok'; Start-Process $url }
  if (-not $Quiet) {
    Say ''
    Say '提示：' 'gray'
    Say '  · 直接进某个世界：start_v3.bat -World alpine|desert|coast|winter' 'gray'
    Say '  · 烘焙自己的照片：start_v3.bat -Photo D://pics//a.jpg' 'gray'
    Say '  · 关闭服务：桌面「PHOTOWORLD v3 - 停止.bat」' 'gray'
    Start-Sleep -Milliseconds 900
  }
  exit 0
}
if ($Photo) {
  if (-not (Test-Path $Photo)) { Die "照片不存在：$Photo" }
  $name = [IO.Path]::GetFileNameWithoutExtension($Photo)
  $outDir = Join-Path $Root "out/v3/$name"
  Say "烘焙中：$Photo → out/v3/$name（GPU，约 2 分钟/张）…" 'warn'
  $bakeArgs = @('pipeline/v3/bake.py', $Photo, '--out', $outDir, '--step', '4')
  if ($NoDiffusion) { $bakeArgs += '--no-diffusion' }
  Push-Location $Root
  try { & $py @bakeArgs } finally { Pop-Location }
  if ($LASTEXITCODE -ne 0) { Die "烘焙失败（退出码 $LASTEXITCODE）。看上面的输出。" }
  $target = "out/v3/$name/world.glb"
} else {
  $candidates = @(
    "assets/world-v3-$World.glb",
    "out/v3/worlds/$World/world.glb",
    "out/v3/$World/world.glb"
  )
  foreach ($rel in $candidates) {
    if (Test-Path (Join-Path $Root $rel)) { $target = $rel; break }
  }
  if (-not $target) {
    Say "没有找到 $World 的 v3 资产。可用的内置世界：" 'warn'
    Get-ChildItem (Join-Path $Root 'assets') -Filter 'world-v3-*.glb' -ErrorAction SilentlyContinue |
      ForEach-Object { Say ("  " + ($_.Name -replace '^world-v3-|\.glb$', '')) 'gray' }
    Die "先烘焙：python pipeline/v3/bake.py worlds/$World.jpg --out out/v3/worlds/$World --step 4"
  }
}
Say "资产：$target" 'gray'

$url = "$StaticUrl/?glb=" + [Uri]::EscapeDataString($target) + '&instant=1'
if ($NoBrowser) {
  Say "（-NoBrowser）请手动打开：$url" 'gray'
} else {
  Say "打开：$url" 'ok'
  Start-Process $url
}

if (-not $Quiet) {
  Say ''
  Say '提示：' 'gray'
  Say '  · 鼠标拖动环视；点击画面锁定指针；WASD 行走，Shift 奔跑' 'gray'
  Say '  · 换世界：start_v3.bat -World desert|coast|winter' 'gray'
  Say '  · 烘焙自己的照片：start_v3.bat -Photo D:\pics\a.jpg' 'gray'
  Say '  · 关闭服务：桌面「PHOTOWORLD v3 - 停止.bat」' 'gray'
  Start-Sleep -Milliseconds 900
}
