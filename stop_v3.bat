@echo off
rem Stop PHOTOWORLD v3 services. Delegates to tools\start.py.
setlocal
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (where py >nul 2>nul && set "PY=py -3")
if not defined PY (where python3 >nul 2>nul && set "PY=python3")
if not defined PY (echo [ERR] Python not found. & pause & exit /b 1)
%PY% "%~dp0tools\start.py" --stop %*
if errorlevel 1 pause
endlocal
