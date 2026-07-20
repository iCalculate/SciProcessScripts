@echo off
setlocal
cd /d "%~dp0.."

set "PLDB_BUILD_FRONTEND=0"
if /I "%~1"=="--build" set "PLDB_BUILD_FRONTEND=1"

if not exist "config.yaml" (
    if exist "config.example.yaml" (
        echo [setup] Creating local config.yaml from config.example.yaml...
        copy /Y "config.example.yaml" "config.yaml" >nul
        if errorlevel 1 exit /b 1
        echo [setup] Review config.yaml and keep importer.mock_mode=false for real WITio imports.
    ) else (
        echo [setup] config.example.yaml not found. The backend will fall back to built-in defaults.
    )
)

where uv >nul 2>nul
if errorlevel 1 (
    echo [ERROR] uv is not installed or not on PATH.
    echo Install uv from https://docs.astral.sh/uv/ and rerun this script.
    exit /b 1
)

echo [setup] Ensuring Python 3.11 is available through uv...
uv python install 3.11
if errorlevel 1 exit /b 1

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating .venv with uv...
    uv venv .venv --python 3.11
    if errorlevel 1 exit /b 1
)

echo [setup] Installing Python dependencies with uv...
uv pip install --python ".venv\Scripts\python.exe" -r requirements.txt
if errorlevel 1 exit /b 1

cd frontend
echo [setup] Installing frontend dependencies...
npm install
if errorlevel 1 exit /b 1
if "%PLDB_BUILD_FRONTEND%"=="1" (
    echo [setup] Building frontend...
    npm run build
    if errorlevel 1 exit /b 1
) else (
    echo [setup] Skipping frontend build. Run "scripts\setup_env.bat --build" when you need frontend\dist.
)
cd ..

echo [setup] Python importer backend configured through config.yaml: importer.backend=witio
echo [setup] Environment setup complete.
