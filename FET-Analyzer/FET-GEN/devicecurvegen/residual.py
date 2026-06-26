from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .neural import CONDITION_NAMES, condition_from_generation
from .schemas import GenerationCondition, ModelInfo


@dataclass
class ResidualSample:
    values: np.ndarray
    mode: str
    latent_code: list[float]
    gate_values: np.ndarray | None = None


class ResidualEngine:
    """Sample learned residual morphology on a normalized voltage coordinate."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        discover_default: bool = True,
    ) -> None:
        configured = checkpoint_path or os.getenv("DEVICEGEN_MODEL_PATH")
        model_root = Path(__file__).resolve().parents[1] / "models"
        default_checkpoints = [
            model_root / "residual-cvae.npz",
            model_root / "residual-pca.npz",
        ]
        if configured is None and discover_default:
            configured = next((path for path in default_checkpoints if path.exists()), None)
        self.path = Path(configured).expanduser().resolve() if configured else None
        self.grid: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.components: np.ndarray | None = None
        self.scales: np.ndarray | None = None
        self.condition_mean: np.ndarray | None = None
        self.condition_scale: np.ndarray | None = None
        self.residual_mean: np.ndarray | None = None
        self.residual_scale: np.ndarray | None = None
        self.decoder_w: np.ndarray | None = None
        self.decoder_b: np.ndarray | None = None
        self.output_w: np.ndarray | None = None
        self.output_b: np.ndarray | None = None
        self.metadata: dict = {}
        if self.path:
            if not self.path.exists():
                raise ValueError(f"Residual checkpoint not found: {self.path}")
            self._load(self.path)

    def _load(self, path: Path) -> None:
        try:
            with np.load(path) as payload:
                model_type = (
                    str(payload["model_type"].item())
                    if "model_type" in payload.files
                    else "learned_pca"
                )
                if model_type == "conditional_vae":
                    self._load_conditional_vae(payload, path)
                    return
                if model_type != "learned_pca":
                    raise ValueError(f"Unsupported residual model type: {model_type}")
                grid = np.asarray(payload["grid"], dtype=float)
                mean = np.asarray(payload["mean"], dtype=float)
                components = np.asarray(payload["components"], dtype=float)
                scales = np.asarray(payload["scales"], dtype=float)
                metadata = {}
                if "metadata_json" in payload.files:
                    try:
                        metadata = json.loads(str(payload["metadata_json"].item()))
                    except (TypeError, ValueError):
                        metadata = {}
        except (OSError, KeyError, ValueError) as error:
            raise ValueError(f"Invalid residual checkpoint: {path}") from error
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
            raise ValueError("Residual checkpoint grid must be a strictly increasing vector")
        if mean.shape not in {(grid.size,), (2 * grid.size,)}:
            raise ValueError("Residual checkpoint mean must contain one or two channels")
        if components.ndim != 2 or components.shape[1] != mean.size:
            raise ValueError("Residual checkpoint components have incompatible shape")
        if scales.shape != (components.shape[0],):
            raise ValueError("Residual checkpoint scales have incompatible shape")
        if not all(np.all(np.isfinite(values)) for values in (grid, mean, components, scales)):
            raise ValueError("Residual checkpoint contains non-finite values")
        self._clear_neural()
        self.path = path
        self.grid = grid
        self.mean = mean
        self.components = components
        self.scales = scales
        self.metadata = metadata

    def _clear_neural(self) -> None:
        self.condition_mean = None
        self.condition_scale = None
        self.residual_mean = None
        self.residual_scale = None
        self.decoder_w = None
        self.decoder_b = None
        self.output_w = None
        self.output_b = None
        self.metadata = {}

    def _load_conditional_vae(self, payload, path: Path) -> None:
        required = (
            "grid",
            "condition_names",
            "condition_mean",
            "condition_scale",
            "residual_mean",
            "residual_scale",
            "decoder_w",
            "decoder_b",
            "output_w",
            "output_b",
        )
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(f"Conditional VAE checkpoint is missing: {', '.join(missing)}")
        grid = np.asarray(payload["grid"], dtype=np.float32)
        condition_names = tuple(np.asarray(payload["condition_names"]).astype(str).tolist())
        condition_mean = np.asarray(payload["condition_mean"], dtype=np.float32)
        condition_scale = np.asarray(payload["condition_scale"], dtype=np.float32)
        residual_mean = np.asarray(payload["residual_mean"], dtype=np.float32)
        residual_scale = np.asarray(payload["residual_scale"], dtype=np.float32)
        decoder_w = np.asarray(payload["decoder_w"], dtype=np.float32)
        decoder_b = np.asarray(payload["decoder_b"], dtype=np.float32)
        output_w = np.asarray(payload["output_w"], dtype=np.float32)
        output_b = np.asarray(payload["output_b"], dtype=np.float32)
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
            raise ValueError("Residual checkpoint grid must be a strictly increasing vector")
        if condition_names != CONDITION_NAMES:
            raise ValueError("Conditional VAE checkpoint condition schema is incompatible")
        condition_shape = (len(CONDITION_NAMES),)
        if condition_mean.shape != condition_shape or condition_scale.shape != condition_shape:
            raise ValueError("Conditional VAE checkpoint condition scaler is incompatible")
        if residual_mean.shape not in {(grid.size,), (2 * grid.size,)}:
            raise ValueError("Conditional VAE checkpoint residual scaler is incompatible")
        if residual_scale.shape != residual_mean.shape:
            raise ValueError("Conditional VAE checkpoint residual scaler is incompatible")
        if decoder_w.ndim != 2 or decoder_b.shape != (decoder_w.shape[1],):
            raise ValueError("Conditional VAE decoder hidden layer is incompatible")
        if output_w.shape != (decoder_w.shape[1], residual_mean.size):
            raise ValueError("Conditional VAE decoder output layer is incompatible")
        if output_b.shape != residual_mean.shape:
            raise ValueError("Conditional VAE decoder output layer is incompatible")
        if decoder_w.shape[0] <= len(CONDITION_NAMES):
            raise ValueError("Conditional VAE latent dimension must be positive")
        values = (
            grid,
            condition_mean,
            condition_scale,
            residual_mean,
            residual_scale,
            decoder_w,
            decoder_b,
            output_w,
            output_b,
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Conditional VAE checkpoint contains non-finite values")
        metadata = {}
        if "metadata_json" in payload.files:
            try:
                metadata = json.loads(str(payload["metadata_json"].item()))
            except (TypeError, ValueError):
                metadata = {}
        self.path = path
        self.grid = grid
        self.mean = None
        self.components = None
        self.scales = None
        self.condition_mean = condition_mean
        self.condition_scale = np.maximum(condition_scale, 1e-6)
        self.residual_mean = residual_mean
        self.residual_scale = np.maximum(residual_scale, 1e-6)
        self.decoder_w = decoder_w
        self.decoder_b = decoder_b
        self.output_w = output_w
        self.output_b = output_b
        self.metadata = metadata

    def reload(self, checkpoint_path: str | Path) -> None:
        self._load(Path(checkpoint_path).expanduser().resolve())

    @property
    def mode(self) -> str:
        if self.decoder_w is not None:
            return "conditional_vae"
        return "learned_pca" if self.components is not None else "procedural_prior"

    @property
    def model_name(self) -> str:
        if self.mode in {"conditional_vae", "learned_pca"}:
            return self.path.stem if self.path else "residual-pca"
        return "residual-prior-v1"

    def info(self) -> ModelInfo:
        return ModelInfo(
            residual_mode=self.mode,
            model_name=self.model_name,
            checkpoint_path=str(self.path) if self.path and self.path.exists() else None,
            components=(
                int(self.decoder_w.shape[0] - len(CONDITION_NAMES))
                if self.decoder_w is not None
                else int(self.components.shape[0])
                if self.components is not None
                else 0
            ),
            objective=self.metadata.get("objective"),
            residual_space=self.metadata.get("residual_space"),
            architecture=self.metadata.get("architecture"),
            curves=self.metadata.get("curves"),
            training_curves=self.metadata.get("training_curves"),
            validation_curves=self.metadata.get("validation_curves"),
            hidden_dim=self.metadata.get("hidden_dim"),
            epochs_completed=self.metadata.get("epochs_completed"),
            best_epoch=self.metadata.get("best_epoch"),
            train_loss=self.metadata.get("train_loss"),
            validation_loss=self.metadata.get("validation_loss"),
            validation_rmse_decades=self.metadata.get("validation_rmse_decades"),
            validation_mae_decades=self.metadata.get("validation_mae_decades"),
            validation_p95_error_decades=self.metadata.get(
                "validation_p95_error_decades"
            ),
            validation_weighted_rmse_decades=self.metadata.get(
                "validation_weighted_rmse_decades"
            ),
            validation_low_current_rmse_decades=self.metadata.get(
                "validation_low_current_rmse_decades"
            ),
            validation_subthreshold_rmse_decades=self.metadata.get(
                "validation_subthreshold_rmse_decades"
            ),
            validation_subthreshold_slope_rmse_dec_per_v=self.metadata.get(
                "validation_subthreshold_slope_rmse_dec_per_v"
            ),
            validation_gate_rmse_decades=self.metadata.get(
                "validation_gate_rmse_decades"
            ),
            gate_curves=self.metadata.get("gate_curves"),
            generated_channels=self.metadata.get("channels", ["Ids"]),
            selection_score=self.metadata.get("selection_score"),
            best_trial=self.metadata.get("best_trial"),
            tuning_trials=self.metadata.get("tuning_trials", []),
            feature_eval_curves=self.metadata.get("feature_eval_curves"),
            feature_vth_mae_v=self.metadata.get("feature_vth_mae_v"),
            feature_ss_mae_mv_dec=self.metadata.get("feature_ss_mae_mv_dec"),
            feature_log_ion_mae_decades=self.metadata.get(
                "feature_log_ion_mae_decades"
            ),
            feature_log_ioff_mae_decades=self.metadata.get(
                "feature_log_ioff_mae_decades"
            ),
            physics_baseline_rmse_decades=self.metadata.get(
                "physics_baseline_rmse_decades"
            ),
            physics_baseline_weighted_rmse_decades=self.metadata.get(
                "physics_baseline_weighted_rmse_decades"
            ),
            physics_baseline_low_current_rmse_decades=self.metadata.get(
                "physics_baseline_low_current_rmse_decades"
            ),
            physics_baseline_subthreshold_rmse_decades=self.metadata.get(
                "physics_baseline_subthreshold_rmse_decades"
            ),
            rmse_improvement_percent=self.metadata.get("rmse_improvement_percent"),
            weighted_rmse_improvement_percent=self.metadata.get(
                "weighted_rmse_improvement_percent"
            ),
            source=self.metadata.get("source"),
            training_config=self.metadata.get("training_config"),
            training_history=self.metadata.get("training_history", []),
        )

    def sample(
        self,
        normalized_voltage: np.ndarray,
        *,
        seed: int,
        diversity: float,
        sweep_phase: float,
        condition: GenerationCondition | None = None,
        reverse: bool = False,
    ) -> ResidualSample:
        rng = np.random.default_rng(seed)
        if self.decoder_w is not None and self.grid is not None:
            if condition is None:
                raise ValueError("Conditional VAE sampling requires a generation condition")
            if any(
                value is None
                for value in (
                    self.condition_mean,
                    self.condition_scale,
                    self.residual_mean,
                    self.residual_scale,
                    self.decoder_b,
                    self.output_w,
                    self.output_b,
                )
            ):
                raise ValueError("Conditional VAE checkpoint is incomplete")
            latent_dim = self.decoder_w.shape[0] - len(CONDITION_NAMES)
            latent = rng.normal(0.0, 0.35 + diversity, latent_dim).astype(np.float32)
            raw_condition = condition_from_generation(condition, reverse=reverse)
            standardized_condition = (
                (raw_condition - self.condition_mean) / self.condition_scale
            ).astype(np.float32)
            decoder_input = np.concatenate([latent, standardized_condition])
            hidden = np.tanh(decoder_input @ self.decoder_w + self.decoder_b)
            standardized_residual = hidden @ self.output_w + self.output_b
            residual = self.residual_mean + standardized_residual * self.residual_scale
            drain_residual = residual[: self.grid.size]
            gate_residual = (
                residual[self.grid.size : 2 * self.grid.size]
                if residual.size >= 2 * self.grid.size
                else None
            )
            sampled = np.interp(normalized_voltage, self.grid, drain_residual)
            sampled_gate = (
                np.interp(normalized_voltage, self.grid, gate_residual)
                if gate_residual is not None
                else None
            )
            return ResidualSample(
                values=sampled,
                mode=self.mode,
                latent_code=latent.astype(float).tolist(),
                gate_values=sampled_gate,
            )
        if self.components is not None and self.grid is not None:
            count = self.components.shape[0]
            latent = rng.normal(0.0, 1.0, count)
            scales = self.scales if self.scales is not None else np.ones(count)
            residual = self.mean + (latent * scales) @ self.components
            drain_residual = residual[: self.grid.size]
            gate_residual = (
                residual[self.grid.size : 2 * self.grid.size]
                if residual.size >= 2 * self.grid.size
                else None
            )
            sampled = np.interp(normalized_voltage, self.grid, drain_residual)
            sampled_gate = (
                np.interp(normalized_voltage, self.grid, gate_residual)
                if gate_residual is not None
                else None
            )
            return ResidualSample(
                values=sampled * (0.35 + diversity),
                mode=self.mode,
                latent_code=latent.astype(float).tolist(),
                gate_values=(
                    sampled_gate * (0.35 + diversity)
                    if sampled_gate is not None
                    else None
                ),
            )

        x = normalized_voltage
        residual = np.zeros_like(x)
        frequencies = np.array([0.55, 1.0, 1.7, 2.8])
        amplitudes = rng.normal(0.0, [0.07, 0.045, 0.025, 0.014])
        phases = rng.uniform(-np.pi, np.pi, frequencies.size) + sweep_phase
        for amplitude, frequency, phase in zip(amplitudes, frequencies, phases, strict=True):
            residual += amplitude * np.sin(np.pi * frequency * x + phase)
        center = rng.uniform(-0.25, 0.45)
        width = rng.uniform(0.10, 0.28)
        residual += rng.normal(0.0, 0.08) * np.exp(-0.5 * ((x - center) / width) ** 2)
        residual += rng.normal(0.0, 0.035) * np.tanh(3.0 * x)
        residual -= np.mean(residual)
        latent_code = [*amplitudes.tolist(), *phases.tolist(), float(center), float(width)]
        return ResidualSample(
            values=residual * (0.45 + 0.9 * diversity),
            mode=self.mode,
            latent_code=latent_code,
        )
