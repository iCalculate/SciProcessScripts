from __future__ import annotations

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlalchemy import select

from .database import (
    aligned_gate_points,
    aligned_points,
    create_database_engine,
    create_schema,
    curves,
    source_files,
)
from .features import analyze_transfer_curve
from .schemas import GenerationCondition, NeuralTrainingResult

CONDITION_NAMES = (
    "log10_ion",
    "log10_ioff",
    "log10_dynamic_range",
    "vth_normalized",
    "log10_ss_mv_dec",
    "log10_voltage_span",
    "voltage_center_scaled",
    "polarity_sign",
    "direction_sign",
)


@dataclass
class NeuralDataset:
    grid: np.ndarray
    log_current: np.ndarray
    log_gate_current: np.ndarray | None
    conditions: np.ndarray
    groups: np.ndarray
    source: str


@dataclass
class NeuralTrainingConfig:
    method: str = "physics_cvae"
    latent_dim: int = 12
    hidden_dim: int = 96
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    beta: float = 0.005
    validation_fraction: float = 0.1
    patience: int = 7
    seed: int = 12345
    max_curves: int | None = None
    low_current_weight: float = 1.5
    subthreshold_weight: float = 2.5
    slope_weight: float = 0.10
    gate_loss_weight: float = 0.5
    pca_components: int = 12
    feature_eval_limit: int = 512

    def validate(self) -> None:
        if self.method not in {"physics_cvae", "latent_pca"}:
            raise ValueError("method must be physics_cvae or latent_pca")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be at least 1")
        if self.hidden_dim < 4:
            raise ValueError("hidden_dim must be at least 4")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.beta < 0:
            raise ValueError("beta must be non-negative")
        if not 0.01 <= self.validation_fraction <= 0.5:
            raise ValueError("validation_fraction must be between 0.01 and 0.5")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        if self.max_curves is not None and self.max_curves < 10:
            raise ValueError("max_curves must be at least 10")
        if self.low_current_weight < 0:
            raise ValueError("low_current_weight must be non-negative")
        if self.subthreshold_weight < 0:
            raise ValueError("subthreshold_weight must be non-negative")
        if self.slope_weight < 0:
            raise ValueError("slope_weight must be non-negative")
        if self.gate_loss_weight < 0:
            raise ValueError("gate_loss_weight must be non-negative")
        if not 1 <= self.pca_components <= 64:
            raise ValueError("pca_components must be between 1 and 64")
        if self.feature_eval_limit < 0:
            raise ValueError("feature_eval_limit must be non-negative")


def condition_vector(
    *,
    ion: float,
    ioff: float,
    vth: float,
    ss_mv_dec: float,
    voltage_min: float,
    voltage_max: float,
    polarity: str,
    direction: str,
) -> np.ndarray:
    tiny = np.finfo(np.float32).tiny
    safe_ion = max(float(ion), tiny)
    safe_ioff = max(min(float(ioff), safe_ion), tiny)
    span = max(float(voltage_max) - float(voltage_min), 1e-6)
    center = 0.5 * (float(voltage_min) + float(voltage_max))
    return np.asarray(
        [
            np.log10(safe_ion),
            np.log10(safe_ioff),
            np.log10(max(safe_ion / safe_ioff, 1.0)),
            2.0 * (float(vth) - float(voltage_min)) / span - 1.0,
            np.log10(max(float(ss_mv_dec), 1e-3)),
            np.log10(span),
            center / max(span, 1.0),
            1.0 if polarity == "n-type" else -1.0,
            -1.0 if direction == "reverse" else 1.0 if direction == "forward" else 0.0,
        ],
        dtype=np.float32,
    )


def condition_from_generation(
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    return condition_vector(
        ion=condition.target_ion,
        ioff=condition.target_ioff,
        vth=condition.target_vth,
        ss_mv_dec=condition.target_ss_mv_dec,
        voltage_min=condition.voltage_min,
        voltage_max=condition.voltage_max,
        polarity=condition.polarity,
        direction="reverse" if reverse else "forward",
    )


def _conditions_from_frame(frame: pd.DataFrame) -> np.ndarray:
    return np.vstack(
        [
            condition_vector(
                ion=row.feature_ion,
                ioff=row.feature_ioff,
                vth=row.feature_vth,
                ss_mv_dec=row.feature_ss_mv_dec,
                voltage_min=row.voltage_min_v,
                voltage_max=row.voltage_max_v,
                polarity=row.feature_polarity,
                direction=row.direction,
            )
            for row in frame.itertuples(index=False)
        ]
    )


def _validate_dataset(dataset: NeuralDataset) -> NeuralDataset:
    if dataset.grid.ndim != 1 or dataset.grid.size < 16:
        raise ValueError("Training grid must be a vector with at least 16 points")
    if not np.all(np.diff(dataset.grid) > 0):
        raise ValueError("Training grid must be strictly increasing")
    expected = (dataset.conditions.shape[0], dataset.grid.size)
    if dataset.log_current.shape != expected:
        raise ValueError(
            f"Current matrix has shape {dataset.log_current.shape}, expected {expected}"
        )
    if dataset.conditions.shape[1] != len(CONDITION_NAMES):
        raise ValueError("Training condition matrix has incompatible shape")
    if (
        dataset.log_gate_current is not None
        and dataset.log_gate_current.shape != dataset.log_current.shape
    ):
        raise ValueError("Gate-current matrix has incompatible shape")
    if dataset.groups.shape != (dataset.log_current.shape[0],):
        raise ValueError("Training groups have incompatible shape")
    if dataset.log_current.shape[0] < 10:
        raise ValueError("At least ten valid curves are required for neural training")
    arrays = (dataset.grid, dataset.log_current, dataset.conditions)
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Training dataset contains non-finite values")
    if dataset.log_gate_current is not None:
        finite_gate = np.isfinite(dataset.log_gate_current)
        valid_gate_rows = np.any(finite_gate, axis=1)
        if np.any(valid_gate_rows & ~np.all(finite_gate, axis=1)):
            raise ValueError("Gate-current rows must be complete or fully missing")
    return dataset


def load_exported_neural_dataset(path: Path) -> NeuralDataset:
    root = path.expanduser().resolve()
    matrix_path = root / "aligned_curves.npz"
    metadata_path = root / "curves.csv"
    if not matrix_path.is_file() or not metadata_path.is_file():
        raise ValueError(
            "Dataset directory must contain aligned_curves.npz and curves.csv"
        )
    with np.load(matrix_path) as payload:
        try:
            curve_ids = np.asarray(payload["curve_id"]).astype(str)
            grid = np.asarray(payload["x_norm"], dtype=np.float32)
            log_current = np.asarray(payload["log10_abs_id"], dtype=np.float32)
            log_gate_current = (
                np.asarray(payload["log10_abs_ig"], dtype=np.float32)
                if "log10_abs_ig" in payload.files
                else None
            )
        except KeyError as error:
            raise ValueError(f"Missing dataset array: {error.args[0]}") from error
    frame = pd.read_csv(metadata_path)
    required = {
        "curve_id",
        "source_path",
        "direction",
        "voltage_min_v",
        "voltage_max_v",
        "feature_ion",
        "feature_ioff",
        "feature_polarity",
        "feature_vth",
        "feature_ss_mv_dec",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset metadata is missing columns: {', '.join(missing)}")
    if frame["curve_id"].duplicated().any():
        raise ValueError("Dataset metadata contains duplicate curve_id values")
    indexed = frame.set_index("curve_id")
    missing_ids = sorted(set(curve_ids) - set(indexed.index))
    if missing_ids:
        raise ValueError(f"Metadata is missing {len(missing_ids)} curve IDs")
    ordered = indexed.loc[curve_ids].reset_index()
    dataset = NeuralDataset(
        grid=grid,
        log_current=log_current,
        log_gate_current=log_gate_current,
        conditions=_conditions_from_frame(ordered),
        groups=ordered["source_path"].astype(str).to_numpy(),
        source=str(root),
    )
    return _validate_dataset(dataset)


def load_database_neural_dataset(database_url: str | None = None) -> NeuralDataset:
    engine = create_database_engine(database_url)
    create_schema(engine)
    metadata_query = (
        select(
            curves.c.curve_id,
            source_files.c.source_path,
            curves.c.direction,
            curves.c.voltage_min_v,
            curves.c.voltage_max_v,
            curves.c.ion.label("feature_ion"),
            curves.c.ioff.label("feature_ioff"),
            curves.c.polarity.label("feature_polarity"),
            curves.c.vth.label("feature_vth"),
            curves.c.ss_mv_dec.label("feature_ss_mv_dec"),
        )
        .select_from(curves.join(source_files, curves.c.source_file_id == source_files.c.id))
        .where(
            curves.c.vth.is_not(None),
            curves.c.ss_mv_dec.is_not(None),
            curves.c.polarity.in_(["n-type", "p-type"]),
        )
        .order_by(curves.c.curve_id)
    )
    with engine.connect() as connection:
        rows = connection.execute(metadata_query).mappings().all()
        if not rows:
            raise ValueError("Database contains no trainable transfer curves")
        frame = pd.DataFrame(rows)
        curve_ids = frame["curve_id"].astype(str).to_numpy()
        index_by_id = {curve_id: index for index, curve_id in enumerate(curve_ids)}
        point_count = int(
            connection.scalar(
                select(aligned_points.c.point_index)
                .order_by(aligned_points.c.point_index.desc())
                .limit(1)
            )
            or 0
        ) + 1
        if point_count < 16:
            raise ValueError("Database aligned curves have too few points")
        log_current = np.full((len(frame), point_count), np.nan, dtype=np.float32)
        first_curve_id = curve_ids[0]
        grid = np.asarray(
            connection.scalars(
                select(aligned_points.c.x_norm)
                .where(aligned_points.c.curve_id == first_curve_id)
                .order_by(aligned_points.c.point_index)
            ).all(),
            dtype=np.float32,
        )
    if engine.dialect.name in {"mysql", "mariadb"}:
        chunk_size = 128
        chunks = [
            curve_ids[start : start + chunk_size]
            for start in range(0, curve_ids.size, chunk_size)
        ]

        def fetch_chunk(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            placeholders = ",".join(["%s"] * len(chunk))
            query = (
                "SELECT curve_id, point_index, log10_abs_id "
                "FROM aligned_points "
                f"WHERE curve_id IN ({placeholders}) "
                "ORDER BY curve_id, point_index"
            )
            local_index = {curve_id: index for index, curve_id in enumerate(chunk)}
            matrix = np.full((len(chunk), point_count), np.nan, dtype=np.float32)
            raw_connection = engine.raw_connection()
            try:
                cursor = raw_connection.cursor()
                try:
                    cursor.execute(query, tuple(chunk.tolist()))
                    for curve_id, point_index, log_id in cursor:
                        if point_index < point_count:
                            matrix[local_index[str(curve_id)], point_index] = float(log_id)
                finally:
                    cursor.close()
            finally:
                raw_connection.close()
            return chunk, matrix

        worker_count = min(8, max(1, len(chunks)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for chunk, matrix in executor.map(fetch_chunk, chunks):
                row_indices = np.asarray([index_by_id[str(curve_id)] for curve_id in chunk])
                log_current[row_indices] = matrix
    else:
        with engine.connect() as connection:
            points_query = select(
                aligned_points.c.curve_id,
                aligned_points.c.point_index,
                aligned_points.c.log10_abs_id,
            ).order_by(aligned_points.c.curve_id, aligned_points.c.point_index)
            for partition in connection.execution_options(stream_results=True).execute(
                points_query
            ).partitions(20_000):
                for curve_id, point_index, log_id in partition:
                    row_index = index_by_id.get(str(curve_id))
                    if row_index is None or point_index >= point_count:
                        continue
                    log_current[row_index, point_index] = float(log_id)
    log_gate_current = np.full_like(log_current, np.nan, dtype=np.float32)
    with engine.connect() as connection:
        gate_query = select(
            aligned_gate_points.c.curve_id,
            aligned_gate_points.c.point_index,
            aligned_gate_points.c.log10_abs_ig,
        ).order_by(
            aligned_gate_points.c.curve_id,
            aligned_gate_points.c.point_index,
        )
        for partition in connection.execution_options(stream_results=True).execute(
            gate_query
        ).partitions(20_000):
            for curve_id, point_index, log_ig in partition:
                row_index = index_by_id.get(str(curve_id))
                if row_index is None or point_index >= point_count:
                    continue
                log_gate_current[row_index, point_index] = float(log_ig)
    if not np.any(np.all(np.isfinite(log_gate_current), axis=1)):
        log_gate_current = None
    dataset = NeuralDataset(
        grid=grid,
        log_current=log_current,
        log_gate_current=log_gate_current,
        conditions=_conditions_from_frame(frame),
        groups=frame["source_path"].astype(str).to_numpy(),
        source="database",
    )
    return _validate_dataset(dataset)


def _physics_baseline(dataset: NeuralDataset, batch_size: int = 2048) -> np.ndarray:
    output = np.empty_like(dataset.log_current, dtype=np.float32)
    x = dataset.grid[None, :]
    thermal_v = np.float32(0.025852)
    log_ten = np.float32(np.log(10.0))
    for start in range(0, dataset.log_current.shape[0], batch_size):
        stop = min(start + batch_size, dataset.log_current.shape[0])
        conditions = dataset.conditions[start:stop]
        log_ion = conditions[:, 0:1]
        log_ioff = conditions[:, 1:2]
        vth_normalized = conditions[:, 3:4]
        ss = np.power(10.0, conditions[:, 4:5], dtype=np.float32)
        span = np.power(10.0, conditions[:, 5:6], dtype=np.float32)
        center = conditions[:, 6:7] * np.maximum(span, 1.0)
        voltage_min = center - 0.5 * span
        voltage = voltage_min + 0.5 * (x + 1.0) * span
        vth = voltage_min + 0.5 * (vth_normalized + 1.0) * span
        polarity = conditions[:, 7:8]
        u = polarity * (voltage - vth)
        n_sub = np.maximum(0.78 * ss / (1000.0 * thermal_v * log_ten), 0.2)
        n_eff = n_sub * 1.03
        pinch_off = u / n_eff
        forward_charge = np.logaddexp(0.0, pinch_off / (2.0 * thermal_v)) ** 2
        reverse_charge = np.logaddexp(
            0.0, (pinch_off - 1.0 / 8.0) / (2.0 * thermal_v)
        ) ** 2
        normalized = np.maximum(forward_charge - reverse_charge, 0.0) * 1.02
        on_index = np.argmax(u, axis=1)
        on_reference = normalized[np.arange(stop - start), on_index][:, None]
        on_reference = np.maximum(on_reference, np.finfo(np.float32).tiny)
        ion = np.power(10.0, log_ion, dtype=np.float32)
        ioff = np.power(10.0, log_ioff, dtype=np.float32)
        current = ioff + np.maximum(ion - ioff, np.finfo(np.float32).tiny) * (
            normalized / on_reference
        )
        output[start:stop] = np.log10(
            np.maximum(current, np.finfo(np.float32).tiny)
        )
    return output


def _subsample_dataset(
    dataset: NeuralDataset,
    max_curves: int | None,
    rng: np.random.Generator,
) -> NeuralDataset:
    if max_curves is None or dataset.log_current.shape[0] <= max_curves:
        return dataset
    indices = np.sort(rng.choice(dataset.log_current.shape[0], max_curves, replace=False))
    return NeuralDataset(
        grid=dataset.grid,
        log_current=dataset.log_current[indices],
        log_gate_current=(
            None
            if dataset.log_gate_current is None
            else dataset.log_gate_current[indices]
        ),
        conditions=dataset.conditions[indices],
        groups=dataset.groups[indices],
        source=dataset.source,
    )


def _group_split(
    groups: np.ndarray,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    unique_groups = np.unique(groups)
    if unique_groups.size < 2:
        indices = rng.permutation(groups.size)
        validation_count = max(1, int(round(groups.size * validation_fraction)))
        return indices[validation_count:], indices[:validation_count]
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    validation_group_count = max(
        1, min(unique_groups.size - 1, int(round(unique_groups.size * validation_fraction)))
    )
    validation_groups = set(shuffled[:validation_group_count].tolist())
    validation_mask = np.asarray([group in validation_groups for group in groups])
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def _voltage_grid(dataset: NeuralDataset) -> np.ndarray:
    span = np.power(10.0, dataset.conditions[:, 5:6], dtype=np.float32)
    center = dataset.conditions[:, 6:7] * np.maximum(span, 1.0)
    voltage_min = center - 0.5 * span
    return voltage_min + 0.5 * (dataset.grid[None, :] + 1.0) * span


def _current_position(dataset: NeuralDataset) -> np.ndarray:
    log_ion = dataset.conditions[:, 0:1]
    log_ioff = dataset.conditions[:, 1:2]
    dynamic_range = np.maximum(log_ion - log_ioff, 1.0)
    return np.clip((dataset.log_current - log_ioff) / dynamic_range, 0.0, 1.0)


def _region_masks(dataset: NeuralDataset) -> tuple[np.ndarray, np.ndarray]:
    current_position = _current_position(dataset)
    x = dataset.grid[None, :]
    vth_normalized = dataset.conditions[:, 3:4]
    polarity = dataset.conditions[:, 7:8]
    ss_mv_dec = np.power(10.0, dataset.conditions[:, 4:5], dtype=np.float32)
    span = np.power(10.0, dataset.conditions[:, 5:6], dtype=np.float32)
    one_decade_width = np.clip(2.0 * (ss_mv_dec / 1000.0) / span, 0.015, 0.35)
    signed_distance = polarity * (x - vth_normalized)
    near_threshold = signed_distance <= 6.0 * one_decade_width
    low_current = current_position <= 0.35
    subthreshold = (
        (current_position >= 0.04)
        & (current_position <= 0.82)
        & near_threshold
    )
    return low_current, subthreshold


def _point_weights(dataset: NeuralDataset, config: NeuralTrainingConfig) -> np.ndarray:
    current_position = _current_position(dataset)
    x = dataset.grid[None, :]
    vth_normalized = dataset.conditions[:, 3:4]
    polarity = dataset.conditions[:, 7:8]
    ss_mv_dec = np.power(10.0, dataset.conditions[:, 4:5], dtype=np.float32)
    span = np.power(10.0, dataset.conditions[:, 5:6], dtype=np.float32)
    one_decade_width = np.clip(2.0 * (ss_mv_dec / 1000.0) / span, 0.015, 0.35)
    signed_distance = polarity * (x - vth_normalized)

    low_current_component = np.power(1.0 - current_position, 1.35)
    transition_by_current = np.exp(
        -0.5 * np.square((current_position - 0.35) / 0.24)
    )
    transition_by_voltage = np.exp(
        -0.5 * np.square(signed_distance / np.maximum(6.0 * one_decade_width, 0.04))
    )
    subthreshold_component = np.maximum(
        transition_by_current * (current_position <= 0.85),
        transition_by_voltage * (current_position <= 0.90),
    )

    weights = (
        1.0
        + config.low_current_weight * low_current_component
        + config.subthreshold_weight * subthreshold_component
    )
    weights = np.maximum(weights, 1e-3)
    weights /= np.maximum(weights.mean(axis=1, keepdims=True), 1e-6)
    return weights.astype(np.float32)


def _gate_row_mask(dataset: NeuralDataset) -> np.ndarray:
    if dataset.log_gate_current is None:
        return np.zeros(dataset.log_current.shape[0], dtype=bool)
    return np.all(np.isfinite(dataset.log_gate_current), axis=1)


def _gate_shape_target(dataset: NeuralDataset) -> np.ndarray:
    output = np.zeros_like(dataset.log_current, dtype=np.float32)
    gate_rows = _gate_row_mask(dataset)
    if not np.any(gate_rows) or dataset.log_gate_current is None:
        return output
    gate = dataset.log_gate_current[gate_rows]
    center = np.median(gate, axis=1, keepdims=True)
    centered = gate - center
    output[gate_rows] = np.clip(centered, -6.0, 6.0).astype(np.float32)
    return output


def _training_target(
    dataset: NeuralDataset,
    physics: np.ndarray,
) -> tuple[np.ndarray, int]:
    drain_residual = np.clip(dataset.log_current - physics, -8.0, 8.0).astype(
        np.float32
    )
    gate_rows = _gate_row_mask(dataset)
    if np.count_nonzero(gate_rows) < 3:
        return drain_residual, 1
    return np.concatenate([drain_residual, _gate_shape_target(dataset)], axis=1), 2


def _training_weights(
    dataset: NeuralDataset,
    config: NeuralTrainingConfig,
    drain_weights: np.ndarray,
    channels: int,
) -> np.ndarray:
    if channels == 1:
        return drain_weights
    gate_rows = _gate_row_mask(dataset).astype(np.float32)[:, None]
    gate_weights = np.broadcast_to(
        gate_rows * np.float32(config.gate_loss_weight),
        drain_weights.shape,
    )
    return np.concatenate([drain_weights, gate_weights], axis=1).astype(np.float32)


def _rmse(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    selected = values[mask] if mask is not None else values.ravel()
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return None
    return float(np.sqrt(np.mean(selected * selected)))


def _mae(values: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    selected = values[mask] if mask is not None else values.ravel()
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return None
    return float(np.mean(np.abs(selected)))


def _weighted_rmse(values: np.ndarray, weights: np.ndarray) -> float | None:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return None
    return float(np.sqrt(np.sum(weights[valid] * values[valid] ** 2) / np.sum(weights[valid])))


def _xavier(
    rng: np.random.Generator,
    input_dim: int,
    output_dim: int,
) -> np.ndarray:
    bound = np.sqrt(6.0 / (input_dim + output_dim))
    return rng.uniform(-bound, bound, (input_dim, output_dim)).astype(np.float32)


def _initialize_parameters(
    rng: np.random.Generator,
    points: int,
    condition_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> dict[str, np.ndarray]:
    encoder_input = points + condition_dim
    decoder_input = latent_dim + condition_dim
    return {
        "encoder_w": _xavier(rng, encoder_input, hidden_dim),
        "encoder_b": np.zeros(hidden_dim, dtype=np.float32),
        "mu_w": _xavier(rng, hidden_dim, latent_dim),
        "mu_b": np.zeros(latent_dim, dtype=np.float32),
        "logvar_w": _xavier(rng, hidden_dim, latent_dim),
        "logvar_b": np.zeros(latent_dim, dtype=np.float32),
        "decoder_w": _xavier(rng, decoder_input, hidden_dim),
        "decoder_b": np.zeros(hidden_dim, dtype=np.float32),
        "output_w": _xavier(rng, hidden_dim, points),
        "output_b": np.zeros(points, dtype=np.float32),
    }


def _forward(
    parameters: dict[str, np.ndarray],
    residual: np.ndarray,
    conditions: np.ndarray,
    rng: np.random.Generator | None,
) -> dict[str, np.ndarray]:
    encoder_input = np.concatenate([residual, conditions], axis=1)
    encoder_hidden = np.tanh(
        encoder_input @ parameters["encoder_w"] + parameters["encoder_b"]
    )
    mu = encoder_hidden @ parameters["mu_w"] + parameters["mu_b"]
    raw_logvar = encoder_hidden @ parameters["logvar_w"] + parameters["logvar_b"]
    logvar = np.clip(raw_logvar, -8.0, 4.0)
    epsilon = (
        rng.normal(size=mu.shape).astype(np.float32)
        if rng is not None
        else np.zeros_like(mu)
    )
    standard_deviation = np.exp(0.5 * logvar)
    latent = mu + standard_deviation * epsilon
    decoder_input = np.concatenate([latent, conditions], axis=1)
    decoder_hidden = np.tanh(
        decoder_input @ parameters["decoder_w"] + parameters["decoder_b"]
    )
    prediction = decoder_hidden @ parameters["output_w"] + parameters["output_b"]
    return {
        "encoder_input": encoder_input,
        "encoder_hidden": encoder_hidden,
        "mu": mu,
        "raw_logvar": raw_logvar,
        "logvar": logvar,
        "epsilon": epsilon,
        "standard_deviation": standard_deviation,
        "decoder_input": decoder_input,
        "decoder_hidden": decoder_hidden,
        "prediction": prediction,
    }


def _loss(
    forward: dict[str, np.ndarray],
    target: np.ndarray,
    beta: float,
    weights: np.ndarray | None = None,
    slope_weight: float = 0.0,
    residual_scale: np.ndarray | None = None,
    slope_points: int | None = None,
) -> tuple[float, float, float, float]:
    error = forward["prediction"] - target
    if weights is None:
        reconstruction = float(np.mean(error * error))
    else:
        reconstruction = float(
            np.sum(weights * error * error) / np.maximum(np.sum(weights), 1e-6)
        )
    slope = 0.0
    slope_width = target.shape[1] if slope_points is None else slope_points
    if slope_weight > 0 and slope_width > 1:
        scale = (
            np.ones(target.shape[1], dtype=np.float32)
            if residual_scale is None
            else residual_scale.astype(np.float32)
        )
        slope_error = np.diff(
            error[:, :slope_width] * scale[None, :slope_width],
            axis=1,
        )
        if weights is None:
            slope = float(np.mean(slope_error * slope_error))
        else:
            slope_weights = 0.5 * (
                weights[:, 1:slope_width] + weights[:, : slope_width - 1]
            )
            slope = float(
                np.sum(slope_weights * slope_error * slope_error)
                / np.maximum(np.sum(slope_weights), 1e-6)
            )
    kl = float(
        0.5
        * np.mean(
            forward["mu"] ** 2
            + np.exp(forward["logvar"])
            - 1.0
            - forward["logvar"]
        )
    )
    objective = reconstruction + slope_weight * slope + beta * kl
    return objective, reconstruction, kl, slope


def _gradients(
    parameters: dict[str, np.ndarray],
    forward: dict[str, np.ndarray],
    target: np.ndarray,
    beta: float,
    weights: np.ndarray | None = None,
    slope_weight: float = 0.0,
    residual_scale: np.ndarray | None = None,
    slope_points: int | None = None,
) -> dict[str, np.ndarray]:
    batch_size, points = target.shape
    latent_dim = forward["mu"].shape[1]
    error = forward["prediction"] - target
    if weights is None:
        prediction_gradient = 2.0 * error / (batch_size * points)
    else:
        prediction_gradient = (
            2.0 * weights * error / np.maximum(np.sum(weights), 1e-6)
        )
    slope_width = points if slope_points is None else slope_points
    if slope_weight > 0 and slope_width > 1:
        scale = (
            np.ones(points, dtype=np.float32)
            if residual_scale is None
            else residual_scale.astype(np.float32)
        )
        scaled_error = error[:, :slope_width] * scale[None, :slope_width]
        slope_error = np.diff(scaled_error, axis=1)
        if weights is None:
            slope_gradient_delta = (
                2.0 * slope_weight * slope_error / (batch_size * (slope_width - 1))
            )
        else:
            slope_weights = 0.5 * (
                weights[:, 1:slope_width] + weights[:, : slope_width - 1]
            )
            slope_gradient_delta = (
                2.0
                * slope_weight
                * slope_weights
                * slope_error
                / np.maximum(np.sum(slope_weights), 1e-6)
            )
        slope_gradient = np.zeros_like(error)
        slope_gradient[:, 1:slope_width] += (
            slope_gradient_delta * scale[None, 1:slope_width]
        )
        slope_gradient[:, : slope_width - 1] -= (
            slope_gradient_delta * scale[None, : slope_width - 1]
        )
        prediction_gradient += slope_gradient
    gradients: dict[str, np.ndarray] = {}
    gradients["output_w"] = forward["decoder_hidden"].T @ prediction_gradient
    gradients["output_b"] = prediction_gradient.sum(axis=0)
    decoder_hidden_gradient = prediction_gradient @ parameters["output_w"].T
    decoder_activation_gradient = decoder_hidden_gradient * (
        1.0 - forward["decoder_hidden"] ** 2
    )
    gradients["decoder_w"] = forward["decoder_input"].T @ decoder_activation_gradient
    gradients["decoder_b"] = decoder_activation_gradient.sum(axis=0)
    decoder_input_gradient = decoder_activation_gradient @ parameters["decoder_w"].T
    latent_gradient = decoder_input_gradient[:, :latent_dim]

    mu_gradient = latent_gradient + beta * forward["mu"] / (batch_size * latent_dim)
    logvar_gradient = (
        latent_gradient
        * forward["epsilon"]
        * 0.5
        * forward["standard_deviation"]
        + beta
        * 0.5
        * (np.exp(forward["logvar"]) - 1.0)
        / (batch_size * latent_dim)
    )
    logvar_gradient *= (
        (forward["raw_logvar"] >= -8.0) & (forward["raw_logvar"] <= 4.0)
    )
    gradients["mu_w"] = forward["encoder_hidden"].T @ mu_gradient
    gradients["mu_b"] = mu_gradient.sum(axis=0)
    gradients["logvar_w"] = forward["encoder_hidden"].T @ logvar_gradient
    gradients["logvar_b"] = logvar_gradient.sum(axis=0)
    encoder_hidden_gradient = (
        mu_gradient @ parameters["mu_w"].T
        + logvar_gradient @ parameters["logvar_w"].T
    )
    encoder_activation_gradient = encoder_hidden_gradient * (
        1.0 - forward["encoder_hidden"] ** 2
    )
    gradients["encoder_w"] = forward["encoder_input"].T @ encoder_activation_gradient
    gradients["encoder_b"] = encoder_activation_gradient.sum(axis=0)
    return {name: value.astype(np.float32) for name, value in gradients.items()}


def _adam_step(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    first_moment: dict[str, np.ndarray],
    second_moment: dict[str, np.ndarray],
    step: int,
    learning_rate: float,
) -> None:
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    for name, gradient in gradients.items():
        np.clip(gradient, -5.0, 5.0, out=gradient)
        first_moment[name] *= beta1
        first_moment[name] += (1.0 - beta1) * gradient
        second_moment[name] *= beta2
        second_moment[name] += (1.0 - beta2) * gradient * gradient
        corrected_first = first_moment[name] / (1.0 - beta1**step)
        corrected_second = second_moment[name] / (1.0 - beta2**step)
        parameters[name] -= (
            learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
        )


def _evaluate(
    parameters: dict[str, np.ndarray],
    residual: np.ndarray,
    conditions: np.ndarray,
    residual_scale: np.ndarray,
    beta: float,
    batch_size: int,
    weights: np.ndarray | None = None,
    low_current_mask: np.ndarray | None = None,
    subthreshold_mask: np.ndarray | None = None,
    slope_weight: float = 0.0,
    slope_points: int | None = None,
) -> tuple[float, float, float | None, float | None, float | None]:
    losses: list[float] = []
    squared_error = 0.0
    value_count = 0
    weighted_squared_error = 0.0
    weight_count = 0.0
    low_squared_error = 0.0
    low_count = 0
    sub_squared_error = 0.0
    sub_count = 0
    for start in range(0, residual.shape[0], batch_size):
        stop = min(start + batch_size, residual.shape[0])
        batch_weights = None if weights is None else weights[start:stop]
        forward = _forward(parameters, residual[start:stop], conditions[start:stop], None)
        loss, _, _, _ = _loss(
            forward,
            residual[start:stop],
            beta,
            weights=batch_weights,
            slope_weight=slope_weight,
            residual_scale=residual_scale,
            slope_points=slope_points,
        )
        losses.append(loss * (stop - start))
        error = (forward["prediction"] - residual[start:stop]) * residual_scale
        metric_width = error.shape[1] if slope_points is None else slope_points
        metric_error = error[:, :metric_width]
        squared_error += float(np.sum(metric_error * metric_error))
        value_count += metric_error.size
        if batch_weights is not None:
            metric_weights = batch_weights[:, :metric_width]
            weighted_squared_error += float(
                np.sum(metric_weights * metric_error * metric_error)
            )
            weight_count += float(np.sum(metric_weights))
        if low_current_mask is not None:
            batch_mask = low_current_mask[start:stop]
            low_squared_error += float(np.sum(metric_error[batch_mask] ** 2))
            low_count += int(np.count_nonzero(batch_mask))
        if subthreshold_mask is not None:
            batch_mask = subthreshold_mask[start:stop]
            sub_squared_error += float(np.sum(metric_error[batch_mask] ** 2))
            sub_count += int(np.count_nonzero(batch_mask))
    weighted_rmse = (
        float(np.sqrt(weighted_squared_error / weight_count))
        if weight_count > 0
        else None
    )
    low_rmse = (
        float(np.sqrt(low_squared_error / low_count)) if low_count > 0 else None
    )
    sub_rmse = (
        float(np.sqrt(sub_squared_error / sub_count)) if sub_count > 0 else None
    )
    return (
        sum(losses) / residual.shape[0],
        float(np.sqrt(squared_error / value_count)),
        weighted_rmse,
        low_rmse,
        sub_rmse,
    )


def _predict_raw_residual(
    parameters: dict[str, np.ndarray],
    residual: np.ndarray,
    conditions: np.ndarray,
    residual_mean: np.ndarray,
    residual_scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    predicted = np.empty_like(residual, dtype=np.float32)
    for start in range(0, residual.shape[0], batch_size):
        stop = min(start + batch_size, residual.shape[0])
        forward = _forward(parameters, residual[start:stop], conditions[start:stop], None)
        predicted[start:stop] = residual_mean + forward["prediction"] * residual_scale
    return predicted


def _feature_error_statistics(
    dataset: NeuralDataset,
    validation_indices: np.ndarray,
    true_log_current: np.ndarray,
    predicted_log_current: np.ndarray,
    *,
    limit: int,
) -> dict[str, float | int | None]:
    if limit <= 0 or validation_indices.size == 0:
        return {
            "feature_eval_curves": 0,
            "feature_vth_mae_v": None,
            "feature_ss_mae_mv_dec": None,
            "feature_log_ion_mae_decades": None,
            "feature_log_ioff_mae_decades": None,
        }
    count = min(limit, validation_indices.size)
    local_positions = np.unique(
        np.linspace(0, validation_indices.size - 1, count, dtype=np.int64)
    )
    selected_indices = validation_indices[local_positions]
    voltage = _voltage_grid(dataset)[selected_indices]
    true_subset = true_log_current[local_positions]
    predicted_subset = predicted_log_current[local_positions]

    vth_errors: list[float] = []
    ss_errors: list[float] = []
    ion_errors: list[float] = []
    ioff_errors: list[float] = []
    for row, dataset_index in enumerate(selected_indices):
        polarity = (
            "n-type" if dataset.conditions[dataset_index, 7] >= 0 else "p-type"
        )
        true_current = np.power(10.0, np.clip(true_subset[row], -300.0, 50.0))
        predicted_current = np.power(
            10.0,
            np.clip(predicted_subset[row], -300.0, 50.0),
        )
        true_features = analyze_transfer_curve(
            voltage[row],
            true_current,
            polarity=polarity,
        )
        predicted_features = analyze_transfer_curve(
            voltage[row],
            predicted_current,
            polarity=polarity,
        )
        if true_features.vth is not None and predicted_features.vth is not None:
            vth_errors.append(abs(predicted_features.vth - true_features.vth))
        if (
            true_features.ss_mv_dec is not None
            and predicted_features.ss_mv_dec is not None
        ):
            ss_errors.append(
                abs(predicted_features.ss_mv_dec - true_features.ss_mv_dec)
            )
        if true_features.ion > 0 and predicted_features.ion > 0:
            ion_errors.append(
                abs(np.log10(predicted_features.ion) - np.log10(true_features.ion))
            )
        if true_features.ioff > 0 and predicted_features.ioff > 0:
            ioff_errors.append(
                abs(np.log10(predicted_features.ioff) - np.log10(true_features.ioff))
            )

    return {
        "feature_eval_curves": int(local_positions.size),
        "feature_vth_mae_v": float(np.mean(vth_errors)) if vth_errors else None,
        "feature_ss_mae_mv_dec": float(np.mean(ss_errors)) if ss_errors else None,
        "feature_log_ion_mae_decades": float(np.mean(ion_errors)) if ion_errors else None,
        "feature_log_ioff_mae_decades": (
            float(np.mean(ioff_errors)) if ioff_errors else None
        ),
    }


def _validation_statistics(
    *,
    parameters: dict[str, np.ndarray],
    dataset: NeuralDataset,
    validation_indices: np.ndarray,
    physics: np.ndarray,
    standardized_residual: np.ndarray,
    standardized_conditions: np.ndarray,
    residual_mean: np.ndarray,
    residual_scale: np.ndarray,
    weights: np.ndarray,
    low_current_mask: np.ndarray,
    subthreshold_mask: np.ndarray,
    batch_size: int,
    feature_eval_limit: int,
) -> dict[str, float | int | None]:
    predicted_residual = _predict_raw_residual(
        parameters,
        standardized_residual[validation_indices],
        standardized_conditions[validation_indices],
        residual_mean,
        residual_scale,
        batch_size,
    )
    points = dataset.grid.size
    predicted_drain_residual = predicted_residual[:, :points]
    true_log_current = dataset.log_current[validation_indices]
    predicted_log_current = physics[validation_indices] + predicted_drain_residual
    error = predicted_log_current - true_log_current
    absolute_error = np.abs(error)
    validation_weights = weights[validation_indices]
    validation_low_mask = low_current_mask[validation_indices]
    validation_sub_mask = subthreshold_mask[validation_indices]

    voltage = _voltage_grid(dataset)[validation_indices]
    voltage_delta = np.diff(voltage, axis=1)
    valid_delta = np.abs(voltage_delta) > 1e-9
    true_slope = np.divide(
        np.diff(true_log_current, axis=1),
        voltage_delta,
        out=np.zeros_like(voltage_delta),
        where=valid_delta,
    )
    predicted_slope = np.divide(
        np.diff(predicted_log_current, axis=1),
        voltage_delta,
        out=np.zeros_like(voltage_delta),
        where=valid_delta,
    )
    slope_mask = (
        (validation_sub_mask[:, 1:] | validation_sub_mask[:, :-1])
        & valid_delta
    )
    slope_error = predicted_slope - true_slope

    statistics: dict[str, float | int | None] = {
        "validation_rmse_decades": _rmse(error),
        "validation_mae_decades": _mae(error),
        "validation_p95_error_decades": float(np.percentile(absolute_error, 95)),
        "validation_weighted_rmse_decades": _weighted_rmse(error, validation_weights),
        "validation_low_current_rmse_decades": _rmse(error, validation_low_mask),
        "validation_subthreshold_rmse_decades": _rmse(error, validation_sub_mask),
        "validation_subthreshold_slope_rmse_dec_per_v": _rmse(
            slope_error,
            slope_mask,
        ),
        "validation_gate_rmse_decades": None,
    }
    if predicted_residual.shape[1] >= 2 * points:
        gate_rows = _gate_row_mask(dataset)[validation_indices]
        if np.any(gate_rows):
            true_gate_shape = _gate_shape_target(dataset)[validation_indices][gate_rows]
            predicted_gate_shape = predicted_residual[gate_rows, points : 2 * points]
            statistics["validation_gate_rmse_decades"] = _rmse(
                predicted_gate_shape - true_gate_shape
            )
    statistics.update(
        _feature_error_statistics(
            dataset,
            validation_indices,
            true_log_current,
            predicted_log_current,
            limit=feature_eval_limit,
        )
    )
    return statistics


def _save_checkpoint(
    output: Path,
    *,
    dataset: NeuralDataset,
    parameters: dict[str, np.ndarray],
    condition_mean: np.ndarray,
    condition_scale: np.ndarray,
    residual_mean: np.ndarray,
    residual_scale: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp.npz")
    payload: dict[str, Any] = {
        "model_type": np.asarray("conditional_vae"),
        "format_version": np.asarray(2, dtype=np.int64),
        "grid": dataset.grid.astype(np.float32),
        "condition_names": np.asarray(CONDITION_NAMES),
        "condition_mean": condition_mean.astype(np.float32),
        "condition_scale": condition_scale.astype(np.float32),
        "residual_mean": residual_mean.astype(np.float32),
        "residual_scale": residual_scale.astype(np.float32),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        **parameters,
    }
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _save_pca_checkpoint(
    output: Path,
    *,
    dataset: NeuralDataset,
    mean: np.ndarray,
    components: np.ndarray,
    scales: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            model_type=np.asarray("learned_pca"),
            format_version=np.asarray(2, dtype=np.int64),
            grid=dataset.grid.astype(np.float32),
            mean=mean.astype(np.float32),
            components=components.astype(np.float32),
            scales=scales.astype(np.float32),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _train_latent_pca_checkpoint(
    output: Path,
    *,
    dataset: NeuralDataset,
    training_indices: np.ndarray,
    validation_indices: np.ndarray,
    physics: np.ndarray,
    config: NeuralTrainingConfig,
    progress: Callable[[dict[str, float | int | None]], None] | None,
) -> NeuralTrainingResult:
    target, channels = _training_target(dataset, physics)
    training_target = target[training_indices]
    mean = training_target.mean(axis=0)
    centered = training_target - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    component_count = max(
        1,
        min(
            config.pca_components,
            vt.shape[0],
            max(training_indices.size - 1, 1),
        ),
    )
    components = vt[:component_count]
    scales = singular_values[:component_count] / np.sqrt(
        max(training_indices.size - 1, 1)
    )
    validation_target = target[validation_indices]
    validation_centered = validation_target - mean
    reconstruction = mean + (validation_centered @ components.T) @ components
    points = dataset.grid.size
    drain_error = reconstruction[:, :points] - validation_target[:, :points]
    drain_weights = _point_weights(dataset, config)[validation_indices]
    low_current_mask, subthreshold_mask = _region_masks(dataset)
    validation_rmse = float(_rmse(drain_error) or 0.0)
    weighted_rmse = _weighted_rmse(drain_error, drain_weights)
    low_rmse = _rmse(drain_error, low_current_mask[validation_indices])
    subthreshold_rmse = _rmse(
        drain_error,
        subthreshold_mask[validation_indices],
    )
    gate_rows = _gate_row_mask(dataset)
    validation_gate_rows = gate_rows[validation_indices]
    gate_rmse = None
    if channels == 2 and np.any(validation_gate_rows):
        gate_error = (
            reconstruction[validation_gate_rows, points : 2 * points]
            - validation_target[validation_gate_rows, points : 2 * points]
        )
        gate_rmse = _rmse(gate_error)
    predicted_log_current = physics[validation_indices] + reconstruction[:, :points]
    feature_statistics = _feature_error_statistics(
        dataset,
        validation_indices,
        dataset.log_current[validation_indices],
        predicted_log_current,
        limit=config.feature_eval_limit,
    )
    selection_score = float(
        (weighted_rmse or validation_rmse)
        + (
            config.gate_loss_weight * float(gate_rmse)
            if gate_rmse is not None
            else 0.0
        )
    )
    metric = {
        "epoch": 1,
        "train_loss": float(np.mean(centered * centered)),
        "validation_loss": float(np.mean(drain_error * drain_error)),
        "validation_rmse_decades": validation_rmse,
        "validation_weighted_rmse_decades": weighted_rmse,
        "validation_low_current_rmse_decades": low_rmse,
        "validation_subthreshold_rmse_decades": subthreshold_rmse,
    }
    if progress is not None:
        progress(metric)
    gate_curve_count = int(np.count_nonzero(gate_rows))
    metadata = {
        "objective": "dual_channel_latent_pca_reconstruction",
        "residual_space": "log10_abs_current_decades",
        "architecture": "latent_pca",
        "method": "latent_pca",
        "channels": ["Ids", "Ig"] if channels == 2 else ["Ids"],
        "curves": int(dataset.log_current.shape[0]),
        "gate_curves": gate_curve_count,
        "training_curves": int(training_indices.size),
        "validation_curves": int(validation_indices.size),
        "epochs_completed": 1,
        "best_epoch": 1,
        "latent_dim": component_count,
        "hidden_dim": None,
        "train_loss": metric["train_loss"],
        "validation_loss": metric["validation_loss"],
        "validation_rmse_decades": validation_rmse,
        "validation_weighted_rmse_decades": weighted_rmse,
        "validation_low_current_rmse_decades": low_rmse,
        "validation_subthreshold_rmse_decades": subthreshold_rmse,
        "validation_gate_rmse_decades": gate_rmse,
        "selection_score": selection_score,
        "best_trial": 1,
        **feature_statistics,
        "source": dataset.source,
        "seed": config.seed,
        "training_config": {
            "method": config.method,
            "pca_components": config.pca_components,
            "validation_fraction": config.validation_fraction,
            "seed": config.seed,
            "max_curves": config.max_curves,
            "gate_loss_weight": config.gate_loss_weight,
            "feature_eval_limit": config.feature_eval_limit,
        },
        "training_history": [metric],
    }
    _save_pca_checkpoint(
        output.expanduser().resolve(),
        dataset=dataset,
        mean=mean,
        components=components,
        scales=scales,
        metadata=metadata,
    )
    return NeuralTrainingResult(
        method="latent_pca",
        curves=dataset.log_current.shape[0],
        gate_curves=gate_curve_count,
        generated_channels=["Ids", "Ig"] if channels == 2 else ["Ids"],
        training_curves=training_indices.size,
        validation_curves=validation_indices.size,
        epochs_completed=1,
        best_epoch=1,
        latent_dim=component_count,
        hidden_dim=0,
        train_loss=float(metric["train_loss"]),
        validation_loss=float(metric["validation_loss"]),
        validation_rmse_decades=validation_rmse,
        validation_weighted_rmse_decades=weighted_rmse,
        validation_low_current_rmse_decades=low_rmse,
        validation_subthreshold_rmse_decades=subthreshold_rmse,
        validation_gate_rmse_decades=gate_rmse,
        feature_vth_mae_v=feature_statistics.get("feature_vth_mae_v"),
        feature_ss_mae_mv_dec=feature_statistics.get("feature_ss_mae_mv_dec"),
        selection_score=selection_score,
        output=str(output.expanduser().resolve()),
        source=dataset.source,
        stopped_early=False,
    )


def train_neural_checkpoint(
    output: Path,
    *,
    dataset_path: Path | None = None,
    database_url: str | None = None,
    config: NeuralTrainingConfig | None = None,
    progress: Callable[[dict[str, float | int | None]], None] | None = None,
) -> NeuralTrainingResult:
    training_config = config or NeuralTrainingConfig()
    training_config.validate()
    rng = np.random.default_rng(training_config.seed)
    dataset = (
        load_exported_neural_dataset(dataset_path)
        if dataset_path is not None
        else load_database_neural_dataset(database_url)
    )
    dataset = _subsample_dataset(dataset, training_config.max_curves, rng)
    training_indices, validation_indices = _group_split(
        dataset.groups, training_config.validation_fraction, rng
    )
    if training_indices.size < 2 or validation_indices.size < 1:
        raise ValueError("Training/validation split is too small")

    physics = _physics_baseline(dataset)
    if training_config.method == "latent_pca":
        return _train_latent_pca_checkpoint(
            output,
            dataset=dataset,
            training_indices=training_indices,
            validation_indices=validation_indices,
            physics=physics,
            config=training_config,
            progress=progress,
        )
    raw_residual, channels = _training_target(dataset, physics)
    condition_mean = dataset.conditions[training_indices].mean(axis=0)
    condition_scale = dataset.conditions[training_indices].std(axis=0)
    condition_scale = np.where(condition_scale < 1e-6, 1.0, condition_scale)
    standardized_conditions = (
        (dataset.conditions - condition_mean) / condition_scale
    ).astype(np.float32)
    residual_mean = raw_residual[training_indices].mean(axis=0)
    residual_scale = raw_residual[training_indices].std(axis=0)
    residual_scale = np.maximum(residual_scale, 0.1)
    standardized_residual = (
        (raw_residual - residual_mean) / residual_scale
    ).astype(np.float32)
    drain_weights = _point_weights(dataset, training_config)
    point_weights = _training_weights(
        dataset,
        training_config,
        drain_weights,
        channels,
    )
    low_current_mask, subthreshold_mask = _region_masks(dataset)

    parameters = _initialize_parameters(
        rng,
        raw_residual.shape[1],
        len(CONDITION_NAMES),
        training_config.hidden_dim,
        training_config.latent_dim,
    )
    first_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    second_moment = {name: np.zeros_like(value) for name, value in parameters.items()}
    best_parameters = {name: value.copy() for name, value in parameters.items()}
    best_validation = float("inf")
    best_epoch = 0
    best_train_loss = float("inf")
    stale_epochs = 0
    optimizer_step = 0
    epochs_completed = 0
    history: list[dict[str, float | int | None]] = []

    for epoch in range(1, training_config.epochs + 1):
        shuffled = rng.permutation(training_indices)
        beta = training_config.beta * min(1.0, epoch / max(5, training_config.epochs // 4))
        epoch_loss = 0.0
        seen = 0
        for start in range(0, shuffled.size, training_config.batch_size):
            indices = shuffled[start : start + training_config.batch_size]
            forward = _forward(
                parameters,
                standardized_residual[indices],
                standardized_conditions[indices],
                rng,
            )
            loss, _, _, _ = _loss(
                forward,
                standardized_residual[indices],
                beta,
                weights=point_weights[indices],
                slope_weight=training_config.slope_weight,
                residual_scale=residual_scale,
                slope_points=dataset.grid.size,
            )
            gradients = _gradients(
                parameters,
                forward,
                standardized_residual[indices],
                beta,
                weights=point_weights[indices],
                slope_weight=training_config.slope_weight,
                residual_scale=residual_scale,
                slope_points=dataset.grid.size,
            )
            optimizer_step += 1
            _adam_step(
                parameters,
                gradients,
                first_moment,
                second_moment,
                optimizer_step,
                training_config.learning_rate,
            )
            epoch_loss += loss * indices.size
            seen += indices.size
        train_loss = epoch_loss / seen
        (
            validation_loss,
            validation_rmse,
            validation_weighted_rmse,
            validation_low_rmse,
            validation_subthreshold_rmse,
        ) = _evaluate(
            parameters,
            standardized_residual[validation_indices],
            standardized_conditions[validation_indices],
            residual_scale,
            training_config.beta,
            training_config.batch_size,
            weights=point_weights[validation_indices],
            low_current_mask=low_current_mask[validation_indices],
            subthreshold_mask=subthreshold_mask[validation_indices],
            slope_weight=training_config.slope_weight,
            slope_points=dataset.grid.size,
        )
        epochs_completed = epoch
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_rmse_decades": validation_rmse,
            "validation_weighted_rmse_decades": validation_weighted_rmse,
            "validation_low_current_rmse_decades": validation_low_rmse,
            "validation_subthreshold_rmse_decades": validation_subthreshold_rmse,
        }
        history.append(epoch_metrics)
        if progress is not None:
            progress(epoch_metrics)
        if validation_loss < best_validation - 1e-5:
            best_validation = validation_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_parameters = {name: value.copy() for name, value in parameters.items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= training_config.patience:
                break

    validation_statistics = _validation_statistics(
        parameters=best_parameters,
        dataset=dataset,
        validation_indices=validation_indices,
        physics=physics,
        standardized_residual=standardized_residual,
        standardized_conditions=standardized_conditions,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
        weights=drain_weights,
        low_current_mask=low_current_mask,
        subthreshold_mask=subthreshold_mask,
        batch_size=training_config.batch_size,
        feature_eval_limit=training_config.feature_eval_limit,
    )
    validation_rmse = float(validation_statistics["validation_rmse_decades"] or 0.0)
    physics_baseline_rmse = float(
        np.sqrt(
            np.mean(
                raw_residual[validation_indices, : dataset.grid.size] ** 2
            )
        )
    )
    physics_baseline_weighted_rmse = _weighted_rmse(
        raw_residual[validation_indices, : dataset.grid.size],
        drain_weights[validation_indices],
    )
    physics_baseline_low_rmse = _rmse(
        raw_residual[validation_indices, : dataset.grid.size],
        low_current_mask[validation_indices],
    )
    physics_baseline_subthreshold_rmse = _rmse(
        raw_residual[validation_indices, : dataset.grid.size],
        subthreshold_mask[validation_indices],
    )
    improvement = (
        100.0 * (1.0 - validation_rmse / physics_baseline_rmse)
        if physics_baseline_rmse > 0
        else 0.0
    )
    weighted_validation_rmse = validation_statistics.get(
        "validation_weighted_rmse_decades"
    )
    weighted_improvement = (
        100.0 * (1.0 - float(weighted_validation_rmse) / physics_baseline_weighted_rmse)
        if weighted_validation_rmse is not None
        and physics_baseline_weighted_rmse is not None
        and physics_baseline_weighted_rmse > 0
        else 0.0
    )
    training_config_payload = {
        "latent_dim": training_config.latent_dim,
        "hidden_dim": training_config.hidden_dim,
        "epochs": training_config.epochs,
        "batch_size": training_config.batch_size,
        "learning_rate": training_config.learning_rate,
        "beta": training_config.beta,
        "validation_fraction": training_config.validation_fraction,
        "patience": training_config.patience,
        "seed": training_config.seed,
        "max_curves": training_config.max_curves,
        "low_current_weight": training_config.low_current_weight,
        "subthreshold_weight": training_config.subthreshold_weight,
        "slope_weight": training_config.slope_weight,
        "gate_loss_weight": training_config.gate_loss_weight,
        "pca_components": training_config.pca_components,
        "method": training_config.method,
        "feature_eval_limit": training_config.feature_eval_limit,
    }
    gate_curves = int(np.count_nonzero(_gate_row_mask(dataset)))
    gate_rmse = validation_statistics.get("validation_gate_rmse_decades")
    selection_score = float(
        (weighted_validation_rmse or validation_rmse)
        + (
            training_config.gate_loss_weight * float(gate_rmse)
            if gate_rmse is not None
            else 0.0
        )
    )
    metadata = {
        "objective": "weighted_dual_channel_log_residual_cvae",
        "residual_space": "log10_abs_current_decades",
        "architecture": "conditional_vae",
        "method": training_config.method,
        "channels": ["Ids", "Ig"] if channels == 2 else ["Ids"],
        "curves": int(dataset.log_current.shape[0]),
        "gate_curves": gate_curves,
        "training_curves": int(training_indices.size),
        "validation_curves": int(validation_indices.size),
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "latent_dim": training_config.latent_dim,
        "hidden_dim": training_config.hidden_dim,
        "train_loss": best_train_loss,
        "validation_loss": best_validation,
        **validation_statistics,
        "physics_baseline_rmse_decades": physics_baseline_rmse,
        "physics_baseline_weighted_rmse_decades": physics_baseline_weighted_rmse,
        "physics_baseline_low_current_rmse_decades": physics_baseline_low_rmse,
        "physics_baseline_subthreshold_rmse_decades": (
            physics_baseline_subthreshold_rmse
        ),
        "rmse_improvement_percent": improvement,
        "weighted_rmse_improvement_percent": weighted_improvement,
        "selection_score": selection_score,
        "best_trial": 1,
        "source": dataset.source,
        "seed": training_config.seed,
        "beta": training_config.beta,
        "training_config": training_config_payload,
        "training_history": history,
    }
    _save_checkpoint(
        output.expanduser().resolve(),
        dataset=dataset,
        parameters=best_parameters,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
        residual_mean=residual_mean,
        residual_scale=residual_scale,
        metadata=metadata,
    )
    return NeuralTrainingResult(
        method="physics_cvae",
        curves=dataset.log_current.shape[0],
        gate_curves=gate_curves,
        generated_channels=["Ids", "Ig"] if channels == 2 else ["Ids"],
        training_curves=training_indices.size,
        validation_curves=validation_indices.size,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        latent_dim=training_config.latent_dim,
        hidden_dim=training_config.hidden_dim,
        train_loss=best_train_loss,
        validation_loss=best_validation,
        validation_rmse_decades=validation_rmse,
        validation_weighted_rmse_decades=validation_statistics.get(
            "validation_weighted_rmse_decades"
        ),
        validation_low_current_rmse_decades=validation_statistics.get(
            "validation_low_current_rmse_decades"
        ),
        validation_subthreshold_rmse_decades=validation_statistics.get(
            "validation_subthreshold_rmse_decades"
        ),
        validation_subthreshold_slope_rmse_dec_per_v=validation_statistics.get(
            "validation_subthreshold_slope_rmse_dec_per_v"
        ),
        validation_gate_rmse_decades=gate_rmse,
        feature_vth_mae_v=validation_statistics.get("feature_vth_mae_v"),
        feature_ss_mae_mv_dec=validation_statistics.get("feature_ss_mae_mv_dec"),
        selection_score=selection_score,
        output=str(output.expanduser().resolve()),
        source=dataset.source,
        stopped_early=epochs_completed < training_config.epochs,
    )
