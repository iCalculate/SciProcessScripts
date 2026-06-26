# DeviceCurveGen

DeviceCurveGen is a runnable MVP for physics-informed generation of synthetic
FET transfer curves. It combines an interpretable device-physics baseline with
a stochastic residual model in log-current space:

```text
log10(I_final) = log10(I_physics) + AI strength * learned residual
```

`AI strength = 0` is pure physics. The default trained conditional VAE learns
the distribution of real measurement residuals while conditioning on Ion,
Ioff, Vth, SS, voltage range, polarity, and sweep direction. When aligned Ig
measurements are available, the checkpoint adds a second gate-current morphology
channel. If no checkpoint is available, the application uses a clearly labelled
procedural residual prior.

## Included

- Physics-informed n-type and p-type transfer-curve generation
- Forward/reverse sweeps, draggable hysteresis regions, linear-domain current,
  shot/read noise, current-resolution quantization, Vgs/Vgd gate leakage, seeds,
  and variants
- Independent Ion, Ioff, Vth, SS, hysteresis, mobility, and contact-resistance
  sigma controls for candidate dispersion
- Side-by-side transfer and output characteristic plots
- Physical constraint projection and generated-feature validation
- CSV/TXT inspection, column mapping, cleaning, sweep segmentation, and quality labels
- Ion, Ioff, Ion/Ioff, Vth, SS, gm, Von, and hysteresis extraction
- Dual-channel Ids/Ig conditional VAE and compact latent-PCA residual checkpoints
- Single-run or quick multi-trial parameter search with automatic best-checkpoint activation
- FastAPI, Typer CLI, and a React/Plotly generator workbench
- Final-model curve inspector in the Models panel for concrete Ids/Ig review
- Direct Vth/Ion/Ioff editing, draggable hysteresis bounds, and manual data-column remapping

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
cd frontend
pnpm install
pnpm build
cd ..
devicegen serve
```

Open `http://127.0.0.1:8010`.

For frontend development, run the API in one terminal and Vite in another:

```powershell
devicegen serve
```

```powershell
cd frontend
pnpm dev
```

Open `http://127.0.0.1:5173`. The frontend proxies `/api` and `/health` to the
FastAPI server on port `8010`.

## Local scripts

This repo keeps only two user-facing Windows entrypoints:

- `setup-local-env.bat`
  Creates or reuses `.venv` with `uv`, syncs Python dependencies, installs frontend dependencies with `pnpm`, and refreshes `frontend/dist`. It is safe to rerun whenever dependencies or local setup change.
- `run-fet-gen.bat`
  Opens an interactive console launcher. Use arrow keys to move, space to toggle modules, and Enter to launch. Shortcuts:
  - `D`: database only
  - `A`: database + analysis
  - `F`: full mode

The launcher can start any combination of:

- local project database on `127.0.0.1:3307`
- FastAPI analyzer backend on `127.0.0.1:8010`
- Vite frontend dev server on `127.0.0.1:5173`

The project-local database uses the installed MariaDB server binary with the copied
`.mysql/` data directory. It runs on port `3307` and keeps that data outside git.
This is separate from any default MariaDB service you may have on `3306`.

## CLI

```powershell
devicegen generate --output generated.csv
devicegen ingest .\measurements --output data\dataset.json
devicegen inspect measurement.csv
devicegen extract measurement.csv
devicegen train .\measurements --output models\residual-pca.npz
devicegen train-neural --dataset data\b1500_test_dataset_all --output models\residual-cvae.npz
devicegen serve
```

The default `models/residual-cvae.npz` checkpoint is loaded automatically,
falling back to `models/residual-pca.npz`.
Set `DEVICEGEN_MODEL_PATH` to use a different `.npz` checkpoint. The API reports
whether it is using `conditional_vae`, `learned_pca`, or `procedural_prior`.

The neural trainer can read the MySQL database directly by omitting `--dataset`.
It uses a source-file-grouped validation split, KL warm-up, Adam optimization,
early stopping, and an atomic compressed checkpoint. Exported datasets can
include `log10_abs_ig` beside `log10_abs_id`; missing Ig rows are masked rather
than synthesized during training.

## API

- `POST /api/generate`
- `GET /api/export`
- `POST /api/extract`
- `POST /api/inspect`
- `POST /api/train`
- `GET /api/model`
- `GET /api/examples/{path}`
- `GET /health`

The MVP intentionally targets transfer curves in CSV/TXT files. Original files
are never modified.

Use `examples/sample_transfer.csv` to exercise the Data workspace immediately.
Use the three files under `examples/training/` to exercise residual training.
Uploads are limited to 25 MB per file.

See [docs/STATUS.md](docs/STATUS.md) for the verified MVP boundary and deferred
foundation-model roadmap.
