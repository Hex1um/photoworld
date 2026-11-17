@echo off
rem PHOTOWORLD v3 launcher. ASCII only on purpose: cmd chokes on UTF-8 batch files.
rem v3 rebuilds offline (pipeline/v3/bake.py -> world.glb), so this only needs the
rem static server. All logic + messages live in tools\start_v3.ps1 (UTF-8 with BOM).
setlocal
set "PS=powershell"
where pwsh >nul 2>nul && set "PS=pwsh"
"%PS%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\start_v3.ps1" %*
if errorlevel 1 pause
endlocal
