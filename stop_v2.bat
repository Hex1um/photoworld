@echo off
rem Stop PHOTOWORLD v2 services (GPU service + static server).
setlocal
set "PS=powershell"
where pwsh >nul 2>nul && set "PS=pwsh"
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_v2.ps1" -Stop %*
if errorlevel 1 pause
endlocal
