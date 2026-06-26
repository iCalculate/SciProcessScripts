# DeviceCurveGen implementation status

This repository is a complete transfer-curve MVP, not the final multi-device
foundation model described in the product vision.

## Implemented and verified

- CSV/TXT/TSV/DAT inspection with header and delimiter detection
- Automatic and manual Vg/Id column mapping
- Invalid-row cleaning, duplicate handling, sweep segmentation, quality labels,
  and 201-point aligned log-current representations
- Transfer-curve extraction: polarity, Ion, Ioff, Ion/Ioff, Vth, SS, gm, Von,
  hysteresis, leakage/current floor, noise, and ambipolar-strength indicators
- Physics-informed n-type and p-type transfer generation
- Forward/reverse sweeps, linear-current noise and current-resolution sampling,
  hysteresis, mobility and contact
  resistance, independent Ion/Ioff/Vth/SS/hysteresis/mobility/resistance dispersion,
  output characteristic families, seeds, diversity, candidates,
  physical strictness, residual strength, latent codes, and constraint scoring
- Procedural residual prior, trainable latent-PCA checkpoints, and a conditional
  VAE residual generator with optional joint Ids/Ig output channels
- Quick multi-trial parameter search with automatic selection by weighted Ids
  reconstruction error plus the configured Ig error contribution
- Source-grouped holdout result: 3,064 validation curves, 0.197-decade
  reconstruction RMSE, 12-dimensional latent space
- FastAPI endpoints for generation, extraction, inspection, training, examples,
  model status, and health
- CLI commands for ingest, inspect, extract, train, generate, and serve
- React workbench with direct guide-line editing, candidate comparison, CSV
  export, manual data mapping, aligned preview, model training, trial comparison,
  and final-checkpoint Ids/Ig curve inspection

## Deliberately deferred

- XLSX/XLS ingestion
- Output, leakage, multi-terminal, and non-FET curve families
- Persistent SQLite/Parquet dataset catalogue and user accounts
- Diffusion training and larger neural residual architectures
- Learned conditional material/process embeddings
- Parameter locks and semantic latent-space navigation
- Statistical real-vs-generated dataset evaluation dashboards
- Laboratory database, cloud repository, and instrument-control connectors

These deferred items require additional product decisions and representative
licensed measurement datasets. They are not silently simulated by the MVP.
