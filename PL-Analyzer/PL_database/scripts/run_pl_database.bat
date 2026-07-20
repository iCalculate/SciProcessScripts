@echo off
setlocal
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Missing Python virtual environment.
    echo Run scripts\setup_env.bat first.
    exit /b 1
)

if not exist "frontend\node_modules" (
    echo [ERROR] Missing frontend dependencies.
    echo Run scripts\setup_env.bat first.
    exit /b 1
)

powershell -ExecutionPolicy Bypass -File "%~dp0run_pl_database.ps1"
