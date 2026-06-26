@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Local environment is missing.
  echo Run setup-local-env.bat first.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" ".\tools\run_launcher.py"

if errorlevel 1 (
  echo.
  echo [ERROR] Launch failed.
  pause
  exit /b 1
)
