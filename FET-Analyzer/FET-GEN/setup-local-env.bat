@echo off
setlocal

cd /d "%~dp0"

set "UV_CACHE_DIR=%CD%\_uv-cache"
set "UV_PYTHON_INSTALL_DIR=%CD%\_uv-python"

echo [1/5] Ensuring Python 3.12 is available via uv...
uv python install 3.12
if errorlevel 1 goto :fail

echo.
echo [2/5] Creating local virtual environment...
if exist ".venv\Scripts\python.exe" (
  echo Reusing existing virtual environment at .venv
) else (
  uv venv --python 3.12 .venv
  if errorlevel 1 goto :fail
)

echo.
echo [3/5] Syncing Python dependencies...
uv sync --python .venv\Scripts\python.exe --extra dev
if errorlevel 1 goto :fail

echo.
echo [4/5] Preparing pnpm through corepack...
set "COREPACK_ENABLE_AUTO_PIN=0"
call corepack pnpm --version
if errorlevel 1 goto :fail

echo.
echo [5/5] Installing frontend dependencies and building the app shell...
pushd frontend
call corepack pnpm install --frozen-lockfile
if errorlevel 1 (
  popd
  goto :fail
)
call corepack pnpm build
if errorlevel 1 (
  popd
  goto :fail
)
popd

echo.
echo Local environment is ready.
echo Backend venv: %CD%\.venv
echo Frontend build: %CD%\frontend\dist
exit /b 0

:fail
echo.
echo [ERROR] Local environment setup failed.
pause
exit /b 1
