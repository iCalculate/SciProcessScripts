@echo off
setlocal
cd /d "%~dp0"

set "PLDB_FORCE_SETUP=0"
if /I "%~1"=="--build" set "PLDB_FORCE_SETUP=1"
if /I "%~1"=="--setup" set "PLDB_FORCE_SETUP=1"

if "%PLDB_FORCE_SETUP%"=="1" goto setup
if not exist ".venv\Scripts\python.exe" goto setup
if not exist "frontend\node_modules" goto setup
goto run

:setup
call scripts\setup_env.bat %*
if errorlevel 1 exit /b 1

:run
call scripts\run_pl_database.bat
exit /b %errorlevel%
