@echo off
rem PHOTOWORLD v2 launcher. ASCII only on purpose: cmd chokes on UTF-8 batch files.
rem All logic + messages live in tools\start_v2.ps1 (UTF-8 with BOM).
setlocal
set "PS=powershell"
where pwsh >nul 2>nul && set "PS=pwsh"
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_v2.ps1" %*
if errorlevel 1 pause
endlocal
