@echo off
REM Launch MOSFET Data Plotter in the project's own uv-managed environment.
REM uv creates/updates .venv from pyproject.toml + uv.lock automatically.
REM Usage:  run.bat  [file.csv | folder]
cd /d "%~dp0"
uv run python b1500_plotter.py %*
