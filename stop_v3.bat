@echo off
rem Stop PHOTOWORLD v3 services (static server on port 8199 by default).
setlocal
set "PS=powershell"
where pwsh >nul 2>nul && set "PS=pwsh"
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_v3.ps1" -Stop %*
if errorlevel 1 pause
endlocal
