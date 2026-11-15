<#
  PHOTOWORLD v2 一键启动器
  ------------------------------------------------------------------
  双击 start_v2.bat（或桌面上的快捷启动脚本）即可：
    1. 起 GPU 重建服务  server/  · uvicorn app:app  · 127.0.0.1:8100
    2. 起静态站       仓库根  · python -m http.server 8099
    3. 等 /health 真的就绪（不是"端口开了"就算，模型加载要几秒）
    4. 打开浏览器到带 ?api= 的地址，直接进 v2 链路

  设计上刻意做的几件事：
    * **可重复执行**：服务已经在跑就复用，不会因为端口占用报错；
    * **失败要说人话**：缺依赖 / 缺权重 / 端口被别的程序占，都给出下一步该敲什么；
    * **后台静默**：两个服务隐藏运行，日志写到 logs/，控制台只留启动进度；
    * **能停**：PID 记在 logs/pids.json，stop_v2.bat 按它精确关闭（不误杀别的 python）。
#>
[CmdletBinding()]
param(
  [switch]$Stop,
  [switch]$NoBrowser,
  [switch]$Visible,
  [int]$StaticPort = 8099,
  [int]$ApiPort = 8100
)

$ErrorActionPreference = 'Stop'
# 探测 /health 用的是 Invoke-WebRequest，它在控制台里会画进度条，
# 每秒一次轮询会把输出刷成一屏乱码（实测）。这里直接关掉进度渲染。
$ProgressPreference = 'SilentlyContinue'
$Root = Split-Path -Parent $PSScriptRoot          # tools/ 的上一级 = 仓库根
$LogDir = Join-Path $Root 'logs'
$PidFile = Join-Path $LogDir 'pids.json'
$ApiUrl = "http://127.0.0.1:$ApiPort"
$StaticUrl = "http://127.0.0.1:$StaticPort"
$OpenUrl = "$StaticUrl/?api=http://127.0.0.1:$ApiPort"

# PS 5.1 的 Start-Process 在 -RedirectStandardOutput 时会把环境变量拼成一个字典；
# 一旦同时存在 HTTPS_PROXY 和 https_proxy（本机就是），它会抛
# "已添加项。字典中的关键字…" 而直接起不来服务。这里在启动前把重复的大小写变体去掉
# （Windows 的环境变量本身大小写不敏感，去掉一个不影响子进程读到代理设置）。
foreach ($name in @('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY')) {
  try {
    $vars = [Environment]::GetEnvironmentVariables()
    foreach ($d in @($vars.Keys | Where-Object { $_ -ceq $name.ToLower() })) {
      [Environment]::SetEnvironmentVariable($d, $null, 'Process')
    }
  } catch { }
}

function Say([string]$msg, [string]$kind = 'info') {
  $c = switch ($kind) { 'ok' { 'Green' } 'warn' { 'Yellow' } 'err' { 'Red' } default { 'Gray' } }
  Write-Host $msg -ForegroundColor $c
}

# 双击运行时窗口会一闪而过；出错必须停下来让人看见
function Die([string]$msg) {
  Say $msg 'err'
  Say '按回车关闭…' 'gray'
  [void](Read-Host)
  exit 1
}

function Get-PortOwner([int]$port) {
  # 优先用 Get-NetTCPConnection，但它在某些 -NoProfile 子进程里会拿不到结果
  # （实测：端口明明在监听却返回 0，导致"没有需要停止的进程"而端口还占着）。
  # 所以再兜一层 netstat 解析：netstat 在任何 Windows 上都在，输出列固定为
  #   Proto  LocalAddress  ForeignAddress  State  PID
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

function Test-Http([string]$url, [int]$timeoutMs = 1500) {
  try {
    $r = Invoke-WebRequest -Uri $url -TimeoutSec ([Math]::Max(1, [int]($timeoutMs / 1000))) `
                           -UseBasicParsing -ErrorAction Stop
    return @{ ok = ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400); body = $r.Content }
  } catch { return @{ ok = $false; body = '' } }
}

function Resolve-Python {
  $cands = @()
  $cands += (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1).Source
  $cands += (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -First 1).Source
  # WorkBuddy 自带的解释器：PATH 里没有时的兜底
  $cands += (Get-ChildItem "$env:USERPROFILE\.workbuddy\binaries\python\versions\*\python.exe" `
             -ErrorAction SilentlyContinue | Sort-Object FullName -Descending |
             Select-Object -First 1).FullName
  foreach ($c in $cands) {
    if ($c -and (Test-Path $c)) { return $c }
  }
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

# ---------------------------------------------------------------- 停止
if ($Stop) {
  # 逐项报告，不做静默计数：早先的版本只累计一个 $killed 数字，
  # 一旦某一步悄悄失败（JSON 解析、Get-Process 拿不到），用户只会看到
  # "没有需要停止的进程"，而端口其实还占着 —— 完全无法定位。
  $acted = 0
  Say ''
  Say '  停止 PHOTOWORLD v2 服务' 'ok'

  $targets = @()
  if (Test-Path $PidFile) {
    try {
      $info = Get-Content $PidFile -Raw | ConvertFrom-Json
      foreach ($k in @('api', 'static')) {
        $procId = [int]$info.$k
        if ($procId -gt 0) { $targets += @{ port = "-"; pid = $procId; src = "pids.json:$k" } }
      }
    } catch {
      Say "  pids.json 读取失败（$($_.Exception.Message)），改按端口找" 'warn'
    }
  } else {
    Say '  pids.json 不存在，按端口找' 'gray'
  }
  # 兜底：按端口找监听者（pid 文件丢了 / 被别的方式启动过）
  foreach ($pair in @(@{ port = $ApiPort; name = 'GPU 服务' }, @{ port = $StaticPort; name = '静态站' })) {
    $procId = Get-PortOwner $pair.port
    if ($procId -gt 0 -and -not ($targets | Where-Object { $_.pid -eq $procId })) {
      $targets += @{ port = $pair.port; pid = $procId; src = "$($pair.name) 端口" }
    }
  }

  if (-not $targets.Count) { Say '  没有在跑的服务（端口都空着）' 'gray' }

  foreach ($t in $targets) {
    $p = Get-Process -Id $t.pid -ErrorAction SilentlyContinue
    if (-not $p) {
      Say "  pid $($t.pid)（$($t.src)）已经不在了，跳过" 'gray'
      continue
    }
    try {
      Stop-Process -Id $t.pid -Force -ErrorAction Stop
      Say "  已停止 pid $($t.pid)  $($p.ProcessName)  ← $($t.src)" 'ok'
      $acted++
    } catch {
      Say "  停不掉 pid $($t.pid)（$($_.Exception.Message)）" 'err'
      Say "    可以试：taskkill /F /PID $($t.pid)" 'gray'
    }
  }

  Start-Sleep -Milliseconds 700
  $left = @(Get-NetTCPConnection -LocalPort $ApiPort, $StaticPort -State Listen -ErrorAction SilentlyContinue)
  if ($left.Count -gt 0) {
    Say ("  注意：端口仍被占用 -> " + (($left | ForEach-Object { "$($_.LocalPort):pid$($_.OwningProcess)" }) -join ', ')) 'warn'
    Say '  可能是别的程序占了这个端口（不是本脚本起的），需要你手动确认。' 'gray'
  } else {
    Say "  端口 $ApiPort / $StaticPort 已释放" 'ok'
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  Say ''
  exit 0
}

# ---------------------------------------------------------------- 启动
Say ''
Say '  PHOTOWORLD v2 启动器' 'ok'
Say "  仓库：$Root" 'gray'
Say ''

if (-not (Test-Path (Join-Path $Root 'server\app.py'))) {
  Die "找不到 server\app.py —— 这个脚本必须放在 photoworld 仓库里（现在在 $Root）"
}

$py = Resolve-Python
if (-not $py) { Die '找不到 python。装一个 Python 3.10+ 并勾选 Add to PATH，或改本脚本里的路径。' }
Say "  python：$py" 'gray'

# 依赖自检：缺了直接说缺什么，不要等到 500 才发现
# 注意两点：① 必须 import importlib.util（只 import importlib 拿不到 .util）；
# ② 原生程序写到 stderr 的内容在 EAP=Stop 下会变成终止错误，探测期间要临时放宽。
$probe = ''
$oldEap = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
  $code = "import importlib.util as u;mods=['fastapi','uvicorn','torch','transformers','scipy'];print(','.join(m for m in mods if u.find_spec(m) is None))"
  $probe = (& $py -c $code 2>$null) | Out-String
  $rc = $LASTEXITCODE
} catch {
  $probe = ''; $rc = 1
} finally {
  $ErrorActionPreference = $oldEap
}
if ($rc -ne 0) { Die "python 无法运行：$py" }
$probe = ($probe -replace '\s', '')
if ($probe) { Die "缺少依赖：$probe`n    python -m pip install fastapi uvicorn torch transformers scipy pillow numpy" }

$weights = Join-Path $Root 'models\da2_large\model.safetensors'
if (-not (Test-Path $weights)) {
  Say '  注意：没找到 models\da2_large\model.safetensors（约 1.3GB）' 'warn'
  Say '        服务能起来，但 /api/reconstruct 会报"缺少权重"；下载命令见 pipeline/v2/README.md' 'warn'
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---- 1) GPU 服务 ----
$apiPid = 0
$health = Test-Http "$ApiUrl/health"
if ($health.ok) {
  Say "  GPU 服务已在运行（复用）：$ApiUrl" 'ok'
  $apiPid = Get-PortOwner $ApiPort
} else {
  if ((Get-PortOwner $ApiPort) -ne 0) {
    Die "端口 $ApiPort 被别的程序占用，而且不是 PHOTOWORLD 服务。`n    换个端口：start_v2.bat -ApiPort 8110`n    或先关掉占用者：netstat -ano | findstr :$ApiPort"
  }
  Say "  启动 GPU 服务（首次要加载 1.3GB 权重，约 10~30 秒）…" 'gray'
  $p = Start-Hidden $py @('-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', "$ApiPort") `
        (Join-Path $Root 'server') 'api.log'
  $apiPid = $p.Id
  $ok = $false
  for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Milliseconds 1000
    if ((Test-Http "$ApiUrl/health").ok) { $ok = $true; break }
    if ($p.HasExited) { break }
  }
  if (-not $ok) {
    $tail = ''
    $errLog = Join-Path $LogDir 'api.err.log'
    if (Test-Path $errLog) { $tail = (Get-Content $errLog -Tail 12) -join "`n" }
    Die "GPU 服务没能就绪。日志：$LogDir\api.err.log`n$tail"
  }
  Say "  GPU 服务就绪：$ApiUrl" 'ok'
}

# ---- 2) 静态站 ----
$staticPid = 0
$page = Test-Http "$StaticUrl/"
if ($page.ok -and $page.body -match 'PHOTOWORLD') {
  Say "  静态站已在运行（复用）：$StaticUrl" 'ok'
  $staticPid = Get-PortOwner $StaticPort
} else {
  if ((Get-PortOwner $StaticPort) -ne 0) {
    Die "端口 $StaticPort 被别的程序占用。`n    换个端口：start_v2.bat -StaticPort 8091"
  }
  $p2 = Start-Hidden $py @('-m', 'http.server', "$StaticPort", '--bind', '127.0.0.1') `
         $Root 'static.log'
  $staticPid = $p2.Id
  $ok2 = $false
  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 400
    $r = Test-Http "$StaticUrl/"
    if ($r.ok -and $r.body -match 'PHOTOWORLD') { $ok2 = $true; break }
    if ($p2.HasExited) { break }
  }
  if (-not $ok2) {
    $tail = ''
    $errLog = Join-Path $LogDir 'static.err.log'
    if (Test-Path $errLog) { $tail = (Get-Content $errLog -Tail 12) -join "`n" }
    Die "静态站没能就绪（端口 $StaticPort）。日志：$LogDir\static.err.log`n$tail"
  }
  Say "  静态站就绪：$StaticUrl" 'ok'
}

# ---- 3) 记下 PID，供 stop_v2.bat 精确关闭 ----
@{ api = $apiPid; static = $staticPid; started = (Get-Date).ToString('s') } |
  ConvertTo-Json | Set-Content -Path $PidFile -Encoding UTF8

# ---- 4) 开浏览器 ----
Say ''
Say "  待打开的地址：$OpenUrl" 'gray'
if (-not $NoBrowser) {
  Start-Process $OpenUrl
  Say '  已打开浏览器。第一次进会重建这张照片（10 秒左右），之后同一张命中缓存。' 'ok'
} else {
  Say '  已跳过开浏览器（-NoBrowser）。' 'gray'
}
Say ''
Say '  关闭服务：双击 stop_v2.bat（或 start_v2.bat -Stop）' 'gray'
Say "  日志目录：$LogDir" 'gray'
Say ''
