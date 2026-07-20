@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Missing .venv\Scripts\python.exe
    echo Run scripts\setup_env.bat first.
    exit /b 1
)
".venv\Scripts\python.exe" -m uvicorn backend.app:app --host 127.0.0.1 --port 8110 --reload --no-access-log
