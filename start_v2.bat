@echo off
rem PHOTOWORLD unified launcher. ASCII only on purpose: cmd chokes on UTF-8 batch.
rem Logic lives in tools\start.py (Python, zero third-party deps).
setlocal
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY (where python3 >nul 2>nul && set "PY=python3")
if not defined PY (
  echo [ERR] Python not found. Install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)
%PY% "%~dp0tools\start.py" --v2 %*
if errorlevel 1 pause
endlocal
