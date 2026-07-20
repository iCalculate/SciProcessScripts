# PL_database

PL_database is a local WITec PL and Raman spectral database project built for laboratory use. It adds a modern web workbench on top of a Python-native `witio` import path:

- Python reads `.wip` files directly through the [`witio`](https://pypi.org/project/witio/) package.
- Python manages the local API, SQLite index, HDF5 trace storage, and spectrum analysis.
- React + Vite provides the WebUI for import control, search, metadata editing, and batch analysis.
- Real Python import is the default path. Mock data is generated only when you explicitly import `mock://demo` or turn `mock_mode` back on.

## Layout

```text
PL_database/
|-- backend/
|-- data/
|-- frontend/
|-- scripts/
|-- tests/
|-- config.example.yaml
|-- config.yaml         # local only, auto-created from config.example.yaml
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Core features

- Recursive `.wip` import through Python `witio`
- Direct SQLite + HDF5 ingest without intermediate CSV files
- SQLite metadata index and HDF5 spectral array storage
- Import job tracking, progress polling, and local log files
- Plot-based database preview with overlay selection
- Metadata editing for selected spectra, a source file, a folder, or all rows
- Baseline correction, smoothing, normalization, peak detection, and Gaussian/Lorentzian fitting
- Mock import mode for frontend and pipeline testing without real `.wip` input

## Prerequisites

- `uv` on PATH
- Node.js + npm on PATH
- local `config.yaml`, which is auto-created from tracked `config.example.yaml` on first setup or first backend start
- Python environment with the backend dependencies installed
- `witio` available in that environment, which is handled by the tracked Python dependencies

## One-click run

From the project root:

```powershell
start_pl_database.bat
```

This root-level launcher:

- bootstraps the environment through `uv` when `.venv` or `frontend/node_modules` is missing
- reuses the existing environment on later runs for a faster start
- auto-creates a local `config.yaml` from `config.example.yaml` if you do not have one yet
- launches the backend and frontend through the existing combined launcher

Force a fresh setup step before launch:

```powershell
start_pl_database.bat --setup
```

## Setup

Run the environment bootstrap from the project root:

```powershell
scripts\setup_env.bat
```

This script:

- installs Python 3.11 through `uv`
- creates `.venv` through `uv`
- installs backend dependencies with `uv pip`
- installs frontend dependencies with `npm`
- creates a local ignored `config.yaml` from tracked `config.example.yaml` when missing
- skips the frontend production build by default to keep local setup source-only

Build the static frontend bundle only when you need `frontend/dist`:

```powershell
scripts\setup_env.bat --build
```

The repository tracks [`config.example.yaml`](./config.example.yaml) and ignores your local `config.yaml`, so personal data choices stay out of Git. The backend now uses the Python-native importer configured in your local `config.yaml`:

```yaml
importer:
  backend: "witio"
  mock_mode: true
```

`mock_mode` is `false` by default. Keep it that way for real `.wip` imports through `witio`.

## Run

### One-click launcher with combined logs

```powershell
scripts\run_pl_database.bat
```

This launcher:

- stops any older backend/frontend instance for this same project before starting a fresh one
- starts the backend on `http://127.0.0.1:8110`
- starts the frontend on `http://127.0.0.1:5173`
- writes live logs to a per-run folder like `data/logs/launcher/20260701-112448-12345/backend.log`
- streams both logs in the launcher console
- stops the backend and frontend again when you press `Ctrl+C` or close that launcher terminal

The combined launcher intentionally starts the backend without `--reload` so the backend process stays tied to that terminal session. If you need a standalone backend dev loop with auto-reload, use `scripts\run_backend.bat` separately.

### Backend only

```powershell
scripts\run_backend.bat
```

### Frontend only

```powershell
scripts\run_frontend.bat
```

## Real Python import

Once `mock_mode` is disabled, the backend reads `.wip` files directly in Python:

- `witio.read(file)` loads the WITec project
- `TDGraph` entries are classified into point spectra, line scans, area maps, and series scans
- traces are written directly into SQLite + HDF5 without temporary CSV files
- `force_reimport=true` clears older rows for the same source file before reinserting

The importer is intentionally conservative. It skips non-spectral graph-like data such as masks and only indexes datasets that can be classified into supported acquisition modes.

## Mock mode

The explicit mock import path is `mock://demo`.

That mode generates synthetic PL-like spectra and exercises:

- job creation
- progress polling
- direct SQLite + HDF5 ingest
- frontend search and plotting
- analysis pipeline

This makes the project usable even before a real `.wip` file is available.

## Testing

Run backend tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the quick mock import smoke test:

```powershell
.\.venv\Scripts\python.exe scripts\test_import_pipeline.py
```

## Notes

- The import pipeline is intentionally conservative: it prefers missing ambiguous data rather than exporting images or hyperspectral maps by accident.
- HDF5 arrays are stored under `/spectra/{spectrum_id}/x_axis` and `/spectra/{spectrum_id}/intensity`.
- Metadata editing is currently centered on the indexed SQLite rows; sample-level normalization can be expanded later if you want a richer sample registry UI.
