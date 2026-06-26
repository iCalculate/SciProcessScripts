from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import numpy as np

from .harmonize import inspect_measurement
from .physics import _physics_log_current
from .schemas import GenerationCondition, TrainingResult


def train_residual_checkpoint(
    inputs: list[Path],
    output: Path,
    *,
    components: int = 8,
) -> TrainingResult:
    """Fit a compact latent residual distribution from real transfer sweeps."""

    grid = np.linspace(-1.0, 1.0, 201)
    if components < 1:
        raise ValueError("components must be at least 1")
    residuals: list[np.ndarray] = []
    skipped: list[str] = []
    for path in inputs:
        try:
            inspection = inspect_measurement(path.name, path.read_bytes())
        except (OSError, ValueError) as error:
            skipped.append(f"{path.name}: {error}")
            continue
        for segment in inspection.segments:
            features = segment.features
            if (
                features is None
                or features.vth is None
                or features.ss_mv_dec is None
                or features.ion <= features.ioff
                or features.polarity == "unknown"
            ):
                continue
            voltage = np.asarray(segment.voltage)
            current = np.abs(np.asarray(segment.current))
            condition = GenerationCondition(
                target_ion=features.ion,
                target_ioff=max(features.ioff, np.finfo(float).tiny),
                target_vth=features.vth,
                target_ss_mv_dec=features.ss_mv_dec,
                polarity=features.polarity,
                hysteresis_v=0.0,
                noise_sigma_a=0.0,
                ai_residual_strength=0.0,
                physical_strictness=1.0,
                voltage_min=float(np.min(voltage)),
                voltage_max=float(np.max(voltage)),
                points=max(51, int(voltage.size)),
                variants=1,
            )
            physics_log = _physics_log_current(voltage, condition, reverse=False)
            measured_log = np.log10(np.clip(current, np.finfo(float).tiny, None))
            x = 2.0 * (voltage - voltage.min()) / max(np.ptp(voltage), 1e-12) - 1.0
            order = np.argsort(x)
            sorted_x = x[order]
            sorted_residual = (measured_log - physics_log)[order]
            unique_x, unique_indices = np.unique(sorted_x, return_index=True)
            residuals.append(np.interp(grid, unique_x, sorted_residual[unique_indices]))

    if len(residuals) < 3:
        raise ValueError("At least three valid transfer sweeps are required")
    matrix = np.vstack(residuals)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    count = max(1, min(components, vt.shape[0], matrix.shape[0] - 1))
    scales = singular_values[:count] / np.sqrt(max(matrix.shape[0] - 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary_output,
            grid=grid,
            mean=mean,
            components=vt[:count],
            scales=scales,
        )
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
    return TrainingResult(
        curves=len(residuals),
        components=count,
        output=str(output),
        files_processed=len(inputs) - len(skipped),
        files_skipped=len(skipped),
        skipped=skipped,
    )
