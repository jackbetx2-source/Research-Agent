@echo off
cd /d "%~dp0"
set "URL=http://127.0.0.1:8000"

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-WebRequest -UseBasicParsing '%URL%/health' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if %ERRORLEVEL% EQU 0 (
  start "" "%URL%"
  exit /b 0
)

start "Research Agent Server" cmd /k "cd /d "%~dp0" && python web_app.py 8000"
timeout /t 2 /nobreak >nul
start "" "%URL%"
