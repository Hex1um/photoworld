<#
  PHOTOWORLD 训练启动器（SkillOpt）
  ------------------------------------------------------------------
  与 start_v2.ps1 分开：那个是"打开世界看看"，这个是"跑训练"。
  双击"PHOTOWORLD 训练"即可：

    1. 起静态站    仓库根  · python -m http.server 8199
    2. 起评分服务  tools/score_server.py · 127.0.0.1:8300（--warm，先把深度模型加载好）
    3. 等 /health 真的就绪
    4. 前台跑训练  tools_skillopt/scripts/train.py
       （前台是有意为之：训练要看好几分钟，进度得能看见）

  为什么必须由这个脚本起评分服务：
    * 评分服务是常驻进程，深度模型只加载一次，单图 60~80s → 15~25s；
    * 训练脚本自己起的子进程在 shell/任务结束时会被回收，
      表现为 `WinError 10061 拒绝连接`，整轮 rollout 全 0 分。

  停止：PHOTOWORLD 训练 -停止.bat（按 PID 精确关闭，不误杀别的 python）。
#>
[CmdletBinding()]
param(
  [switch]$Stop,
  [switch]$Visible,
  [switch]$SkipTrain,
  [switch]$NoPause,
  [int]$StaticPort = 8199,
  [int]$ScorePort = 8300,
  [string]$Config = 'configs/photoworld/default.yaml',
  [string]$SkillOptDir = ''
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Root = Split-Path -Parent $PSScriptRoot                 # tools/ 的上一级 = photoworld 根
$LogDir = Join-Path $Root 'logs'
$PidFile = Join-Path $LogDir 'pids_train.json'
$ScoreUrl = "http://127.0.0.1:$ScorePort"
$StaticUrl = "http://127.0.0.1:$StaticPort"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# PS 5.1：同时存在 HTTPS_PROXY / https_proxy 时 Start-Process 会抛
# "已添加项。字典中的关键字…"，把重复的大小写变体清掉（Windows 环境变量大小写不敏感）
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
function Die([string]$msg) {
  Say $msg 'err'
  if (-not $NoPause) { Say '按回车关闭…' 'gray'; [void](Read-Host) }
  exit 1
}
function Test-Http([string]$url, [int]$timeoutSec = 2) {
  try {
    $r = Invoke-WebRequest -Uri $url -TimeoutSec $timeoutSec -UseBasicParsing -ErrorAction Stop
    return @{ ok = ($r.StatusCode -ge 200 -and $r.StatusCode -lt 400); body = $r.Content }
  } catch { return @{ ok = $false; body = '' } }
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
  return Start-Process -FilePath $file -ArgumentList $argList -WorkingDirectory $workDir `
    -WindowStyle $(if ($Visible) { 'Normal' } else { 'Hidden' }) `
    -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
}

# ---------------------------------------------------------------- 停止
if ($Stop) {
  Say ''
  Say '== 停止训练相关进程 ==' 'warn'
  if (-not (Test-Path $PidFile)) { Say '没有 PID 记录（可能没通过本脚本启动过）' 'gray' }
  else {
    $rec = Get-Content $PidFile -Raw | ConvertFrom-Json
    foreach ($k in @('static', 'score', 'train')) {
      $id = $rec.$k
      if (-not $id) { continue }
      try {
        $p = Get-Process -Id $id -ErrorAction Stop
        Stop-Process -Id $id -Force -ErrorAction Stop
        Say ("  已停止 {0} (PID {1})" -f $k, $id) 'ok'
      } catch {
        Say ("  {0} (PID {1}) 已经不在了" -f $k, $id) 'gray'
      }
    }
    Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
  }
  # 残留的浏览器/子进程：按命令行精确匹配（只碰本项目的）
  Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*score_server.py*" -or $_.CommandLine -like "*train.py*--config*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  Say '完成。' 'ok'
  Start-Sleep -Seconds 1
  exit 0
}

# ---------------------------------------------------------------- 前置检查
Say ''
Say '  PHOTOWORLD 训练启动器' 'ok'

$py = Resolve-Python
if (-not $py) { Die '  [错误] 找不到 python。装一个，或把 python 加进 PATH。' }

$skillDir = $SkillOptDir
if (-not $skillDir) { $skillDir = Join-Path (Split-Path -Parent $Root) 'tools_skillopt' }
if (-not (Test-Path (Join-Path $skillDir 'scripts\train.py'))) {
  Die ("  [错误] 找不到 SkillOpt：{0}`n        先把 microsoft/SkillOpt 克隆到 photoworld 的同级目录：" +
       "`n          git clone https://github.com/microsoft/SkillOpt.git tools_skillopt" -f $skillDir)
}
if (-not $env:DEEPSEEK_API_KEY -and -not $env:OPENAI_COMPATIBLE_API_KEY) {
  Die ("  [错误] 没有模型凭据。设置其中之一：`n" +
       "          setx DEEPSEEK_API_KEY ""sk-...""     （然后重开这个窗口）")
}

Say ("  仓库：{0}" -f $Root)
Say ("  SkillOpt：{0}" -f $skillDir)

# ---------------------------------------------------------------- 1) 静态站
$pids = [ordered]@{ static = $null; score = $null; train = $null }
if ((Test-Http $StaticUrl).ok) {
  Say ("  静态站已在运行：{0}" -f $StaticUrl) 'gray'
} else {
  $p = Start-Hidden $py @('-m', 'http.server', "$StaticPort", '--bind', '127.0.0.1') $Root 'static_train.log'
  $pids.static = $p.Id
  $ok = $false
  for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 400
    if ((Test-Http $StaticUrl).ok) { $ok = $true; break }
  }
  if (-not $ok) { Die ("  [错误] 静态站没起来，看日志：{0}" -f (Join-Path $LogDir 'static_train.err.log')) }
  Say ("  静态站就绪：{0}" -f $StaticUrl) 'ok'
}

# ---------------------------------------------------------------- 2) 评分服务
if ((Test-Http "$ScoreUrl/health").ok) {
  Say ("  评分服务已在运行：{0}" -f $ScoreUrl) 'gray'
} else {
  Say '  启动评分服务（先加载深度模型，约 20~30 秒）…'
  $p = Start-Hidden $py @((Join-Path $Root 'tools\score_server.py'), '--port', "$ScorePort",
                          '--base', $StaticUrl, '--warm') $Root 'score_server.log'
  $pids.score = $p.Id
  $ok = $false
  for ($i = 0; $i -lt 150; $i++) {
    Start-Sleep -Milliseconds 500
    $h = Test-Http "$ScoreUrl/health"
    if ($h.ok -and $h.body -match '"loaded":\s*true') { $ok = $true; break }
  }
  if (-not $ok) { Die ("  [错误] 评分服务没就绪，看日志：{0}" -f (Join-Path $LogDir 'score_server.err.log')) }
  Say ("  评分服务就绪：{0}" -f $ScoreUrl) 'ok'
}

# ---------------------------------------------------------------- 3) 训练
if ($SkipTrain) {
  Say '  [SkipTrain] 只起服务，不跑训练。' 'warn'
} else {
  $env:OPENAI_COMPATIBLE_BASE_URL = 'https://api.deepseek.com/v1'
  if ($env:DEEPSEEK_API_KEY) { $env:OPENAI_COMPATIBLE_API_KEY = $env:DEEPSEEK_API_KEY }
  if (-not $env:OPENAI_COMPATIBLE_MODEL) { $env:OPENAI_COMPATIBLE_MODEL = 'deepseek-chat' }
  $env:PW_SCORE_SERVER = $ScoreUrl
  $env:PW_BASE = $StaticUrl

  $cfgPath = Join-Path $skillDir $Config
  if (-not (Test-Path $cfgPath)) { Die ("  [错误] 找不到训练配置：{0}" -f $cfgPath) }

  Say ''
  Say '== 开始训练（Ctrl+C 可中断） ==' 'warn'
  Say ("  模型：{0}  配置：{1}" -f $env:OPENAI_COMPATIBLE_MODEL, $Config) 'gray'
  Say ''
  $trainLog = Join-Path $LogDir 'train.log'
  # 前台运行并同时写日志。
  # 不用 Tee-Object：PS 5.1 的 Tee 写的是 **UTF-16**，用普通文本工具看会变成
  # 每个字符间夹空格的乱码（实测）。这里逐行 Add-Content -Encoding utf8。
  if (Test-Path $trainLog) { Remove-Item $trainLog -Force -ErrorAction SilentlyContinue }
  Push-Location $skillDir
  try {
    & $py 'scripts\train.py' '--config' $Config 2>&1 | ForEach-Object {
      $line = "$_"
      Write-Host $line
      Add-Content -Path $trainLog -Value $line -Encoding utf8
    }
  } finally {
    Pop-Location
  }
  Say ''
  Say ("  训练结束，日志：{0}" -f $trainLog) 'ok'
}

# ---------------------------------------------------------------- 收尾
$pids.train = $null
try { ($pids | ConvertTo-Json) | Set-Content -Path $PidFile -Encoding utf8 } catch { }
Say ''
Say '  服务仍在后台运行；不需要时运行「PHOTOWORLD 训练 -停止」关闭。' 'gray'
if (-not $NoPause) {
  Say '（按任意键关闭本窗口，服务不受影响）' 'gray'
  [void](Read-Host)
}
