@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop.ps1"
if errorlevel 1 (
  echo.
  echo Shutdown failed. See the message above.
  pause
  exit /b 1
)
endlocal
