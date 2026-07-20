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


def _finite_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _target_transfer_vth(condition: GenerationCondition, *, reverse: bool) -> float:
    sign = 1.0 if condition.polarity == "n-type" else -1.0
    hysteresis_shift = (0.5 if reverse else -0.5) * condition.hysteresis_v
    return condition.target_vth + sign * hysteresis_shift


def _split_residual_channels(
    residual: np.ndarray,
    points: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    drain_residual = residual[:points]
    gate_residual = residual[points : 2 * points] if residual.size >= 2 * points else None
    return drain_residual, gate_residual


def _weighted_local_slope(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-9:
        return 0.0
    x_mean = float(np.sum(weights * x) / total_weight)
    y_mean = float(np.sum(weights * y) / total_weight)
    centered_x = x - x_mean
    denominator = float(np.sum(weights * centered_x * centered_x))
    if denominator <= 1e-12:
        return 0.0
    numerator = float(np.sum(weights * centered_x * (y - y_mean)))
    return numerator / denominator


def _weighted_local_affine(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    center_x: float,
) -> tuple[float, float]:
    slope = _weighted_local_slope(x, y, weights)
    total_weight = float(np.sum(weights))
    if total_weight <= 1e-9:
        return 0.0, slope
    intercept = float(np.sum(weights * (y - slope * (x - center_x))) / total_weight)
    return intercept, slope


def _threshold_focus_envelope(
    grid: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
    strength: float,
    window_scale: float,
    min_window_v: float,
) -> np.ndarray:
    if strength <= 0:
        return np.ones_like(grid, dtype=np.float32)
    span = max(condition.voltage_max - condition.voltage_min, 1e-6)
    effective_vth = _target_transfer_vth(condition, reverse=reverse)
    effective_vth_normalized = (
        2.0 * (effective_vth - condition.voltage_min) / span - 1.0
    )
    window_normalized = max(
        2.0 * window_scale * (condition.target_ss_mv_dec / 1000.0) / span,
        2.0 * min_window_v / span,
        0.015,
    )
    gaussian = np.exp(
        -0.5 * np.square((grid - effective_vth_normalized) / max(window_normalized, 1e-3))
    )
    centered = gaussian - float(np.mean(gaussian))
    envelope = 1.0 + strength * centered
    envelope = np.clip(envelope, 0.70, 1.0 + strength)
    return envelope.astype(np.float32)


def _restore_threshold_local_alignment(
    grid: np.ndarray,
    residual: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    effective_vth = _target_transfer_vth(condition, reverse=reverse)
    span = max(condition.voltage_max - condition.voltage_min, 1e-6)
    effective_vth_normalized = (
        2.0 * (effective_vth - condition.voltage_min) / span - 1.0
    )
    shifted_grid = grid - effective_vth_normalized
    return np.interp(
        shifted_grid,
        grid,
        residual,
        left=float(residual[0]),
        right=float(residual[-1]),
    ).astype(np.float32)


def _restore_drain_delta_rows(values: np.ndarray) -> np.ndarray:
    axis = 1 if values.ndim > 1 else 0
    return np.cumsum(values, axis=axis, dtype=np.float32)


def _aligned_threshold_local_envelope(
    grid: np.ndarray,
    condition: GenerationCondition,
    *,
    window_scale: float,
    min_window_v: float,
    floor: float = 0.03,
) -> np.ndarray:
    span = max(condition.voltage_max - condition.voltage_min, 1e-6)
    window_normalized = max(
        2.0 * window_scale * (condition.target_ss_mv_dec / 1000.0) / span,
        2.0 * min_window_v / span,
        0.012,
    )
    gaussian = np.exp(-0.5 * np.square(grid / max(window_normalized, 1e-3)))
    return np.clip(gaussian, floor, 1.0).astype(np.float32)


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
            model_root / "residual-hybrid-threshold-pca.npz",
            model_root / "residual-cvae.npz",
            model_root / "residual-conditional-pca.npz",
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
        self.condition_names: tuple[str, ...] = CONDITION_NAMES
        self.latent_w: np.ndarray | None = None
        self.latent_b: np.ndarray | None = None
        self.latent_noise: np.ndarray | None = None
        self.latent_clip: np.ndarray | None = None
        self.guide_mean: np.ndarray | None = None
        self.guide_components: np.ndarray | None = None
        self.guide_scales: np.ndarray | None = None
        self.reverse_guide_mean: np.ndarray | None = None
        self.reverse_guide_components: np.ndarray | None = None
        self.reverse_guide_scales: np.ndarray | None = None
        self.reverse_condition_mean: np.ndarray | None = None
        self.reverse_condition_scale: np.ndarray | None = None
        self.reverse_latent_w: np.ndarray | None = None
        self.reverse_latent_b: np.ndarray | None = None
        self.reverse_latent_noise: np.ndarray | None = None
        self.reverse_latent_clip: np.ndarray | None = None
        self.hybrid_local_blend: float | None = None
        self.hybrid_global_blend: float | None = None
        self.hybrid_window_scale: float | None = None
        self.hybrid_min_window_v: float | None = None
        self.hybrid_base_scale_multiplier: float | None = None
        self.hybrid_guide_align_strength: float | None = None
        self.hybrid_guide_align_window_scale: float | None = None
        self.hybrid_guide_delta_clip_decades: float | None = None
        self.hybrid_guide_delta_anchor_strength: float | None = None
        self.hybrid_guide_delta_preserve_affine_strength: float | None = None
        self.hybrid_post_vth_align_strength: float | None = None
        self.hybrid_post_vth_align_reverse_only: bool | None = None
        self.hybrid_post_vth_align_local_window_scale: float | None = None
        self.hybrid_post_vth_align_local_min_window_v: float | None = None
        self.hybrid_guide_as_local_delta: bool | None = None
        self.hybrid_reverse_on_state_blend_scale: float | None = None
        self.hybrid_reverse_on_state_delta_scale: float | None = None
        self.hybrid_reverse_on_state_onset_u_scale: float | None = None
        self.hybrid_reverse_on_state_window_scale: float | None = None
        self.threshold_focus_strength: float | None = None
        self.threshold_focus_window_scale: float | None = None
        self.threshold_focus_min_window_v: float | None = None
        self.threshold_local_align_window_scale: float | None = None
        self.threshold_local_align_min_window_v: float | None = None
        self.threshold_local_delta_transform: bool | None = None
        self.affine_w: np.ndarray | None = None
        self.affine_b: np.ndarray | None = None
        self.affine_clip: np.ndarray | None = None
        self.latent_prior_mean: np.ndarray | None = None
        self.latent_prior_std: np.ndarray | None = None
        self.residual_mean: np.ndarray | None = None
        self.residual_scale: np.ndarray | None = None
        self.decoder_w: np.ndarray | None = None
        self.decoder_b: np.ndarray | None = None
        self.skip_w: np.ndarray | None = None
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
                if model_type == "hybrid_threshold_pca":
                    self._load_hybrid_threshold_pca(payload, path)
                    return
                if model_type == "conditional_pca":
                    self._load_conditional_pca(payload, path)
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
        self.condition_names = CONDITION_NAMES
        self.latent_w = None
        self.latent_b = None
        self.latent_noise = None
        self.latent_clip = None
        self.guide_mean = None
        self.guide_components = None
        self.guide_scales = None
        self.reverse_guide_mean = None
        self.reverse_guide_components = None
        self.reverse_guide_scales = None
        self.reverse_condition_mean = None
        self.reverse_condition_scale = None
        self.reverse_latent_w = None
        self.reverse_latent_b = None
        self.reverse_latent_noise = None
        self.reverse_latent_clip = None
        self.hybrid_local_blend = None
        self.hybrid_global_blend = None
        self.hybrid_window_scale = None
        self.hybrid_min_window_v = None
        self.hybrid_base_scale_multiplier = None
        self.hybrid_guide_align_strength = None
        self.hybrid_guide_align_window_scale = None
        self.hybrid_guide_delta_clip_decades = None
        self.hybrid_guide_delta_anchor_strength = None
        self.hybrid_guide_delta_preserve_affine_strength = None
        self.hybrid_post_vth_align_strength = None
        self.hybrid_post_vth_align_reverse_only = None
        self.hybrid_post_vth_align_local_window_scale = None
        self.hybrid_post_vth_align_local_min_window_v = None
        self.hybrid_guide_as_local_delta = None
        self.hybrid_reverse_on_state_blend_scale = None
        self.hybrid_reverse_on_state_delta_scale = None
        self.hybrid_reverse_on_state_onset_u_scale = None
        self.hybrid_reverse_on_state_window_scale = None
        self.threshold_focus_strength = None
        self.threshold_focus_window_scale = None
        self.threshold_focus_min_window_v = None
        self.threshold_local_align_window_scale = None
        self.threshold_local_align_min_window_v = None
        self.threshold_local_delta_transform = None
        self.affine_w = None
        self.affine_b = None
        self.affine_clip = None
        self.latent_prior_mean = None
        self.latent_prior_std = None
        self.residual_mean = None
        self.residual_scale = None
        self.decoder_w = None
        self.decoder_b = None
        self.skip_w = None
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
        skip_w = (
            np.asarray(payload["skip_w"], dtype=np.float32)
            if "skip_w" in payload.files
            else None
        )
        output_w = np.asarray(payload["output_w"], dtype=np.float32)
        output_b = np.asarray(payload["output_b"], dtype=np.float32)
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
            raise ValueError("Residual checkpoint grid must be a strictly increasing vector")
        unsupported = [name for name in condition_names if name not in CONDITION_NAMES]
        if unsupported:
            raise ValueError(
                "Conditional VAE checkpoint condition schema is incompatible"
            )
        condition_shape = (len(condition_names),)
        if condition_mean.shape != condition_shape or condition_scale.shape != condition_shape:
            raise ValueError("Conditional VAE checkpoint condition scaler is incompatible")
        if residual_mean.shape not in {(grid.size,), (2 * grid.size,)}:
            raise ValueError("Conditional VAE checkpoint residual scaler is incompatible")
        if residual_scale.shape != residual_mean.shape:
            raise ValueError("Conditional VAE checkpoint residual scaler is incompatible")
        if decoder_w.ndim != 2 or decoder_b.shape != (decoder_w.shape[1],):
            raise ValueError("Conditional VAE decoder hidden layer is incompatible")
        if skip_w is None:
            skip_w = np.zeros((decoder_w.shape[0], residual_mean.size), dtype=np.float32)
        if skip_w.shape != (decoder_w.shape[0], residual_mean.size):
            raise ValueError("Conditional VAE decoder skip layer is incompatible")
        if output_w.shape != (decoder_w.shape[1], residual_mean.size):
            raise ValueError("Conditional VAE decoder output layer is incompatible")
        if output_b.shape != residual_mean.shape:
            raise ValueError("Conditional VAE decoder output layer is incompatible")
        if decoder_w.shape[0] <= len(condition_names):
            raise ValueError("Conditional VAE latent dimension must be positive")
        values = (
            grid,
            condition_mean,
            condition_scale,
            residual_mean,
            residual_scale,
            decoder_w,
            decoder_b,
            skip_w,
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
        self.condition_names = condition_names
        self.condition_mean = condition_mean
        self.condition_scale = np.maximum(condition_scale, 1e-6)
        self.residual_mean = residual_mean
        self.residual_scale = np.maximum(residual_scale, 1e-6)
        self.decoder_w = decoder_w
        self.decoder_b = decoder_b
        self.skip_w = skip_w
        self.output_w = output_w
        self.output_b = output_b
        latent_dim = decoder_w.shape[0] - len(condition_names)
        latent_prior_mean = np.asarray(
            metadata.get("latent_prior_mean", []),
            dtype=np.float32,
        )
        latent_prior_std = np.asarray(
            metadata.get("latent_prior_std", []),
            dtype=np.float32,
        )
        self.latent_prior_mean = (
            latent_prior_mean if latent_prior_mean.shape == (latent_dim,) else None
        )
        self.latent_prior_std = (
            np.maximum(latent_prior_std, 0.01)
            if latent_prior_std.shape == (latent_dim,)
            else None
        )
        self.threshold_local_align_window_scale = _finite_float(
            metadata.get("threshold_local_align_window_scale")
        )
        self.threshold_local_align_min_window_v = _finite_float(
            metadata.get("threshold_local_align_min_window_v")
        )
        self.threshold_local_delta_transform = bool(
            metadata.get("threshold_local_delta_transform", False)
        )
        self.metadata = metadata

    def _load_hybrid_threshold_pca(self, payload, path: Path) -> None:
        required = (
            "grid",
            "mean",
            "components",
            "scales",
            "guide_mean",
            "guide_components",
            "guide_scales",
            "condition_names",
            "condition_mean",
            "condition_scale",
            "latent_w",
            "latent_b",
            "latent_noise",
            "latent_clip",
            "hybrid_local_blend",
            "hybrid_global_blend",
            "hybrid_window_scale",
            "hybrid_min_window_v",
        )
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(
                f"Hybrid threshold PCA checkpoint is missing: {', '.join(missing)}"
            )
        grid = np.asarray(payload["grid"], dtype=np.float32)
        mean = np.asarray(payload["mean"], dtype=np.float32)
        components = np.asarray(payload["components"], dtype=np.float32)
        scales = np.asarray(payload["scales"], dtype=np.float32)
        guide_mean = np.asarray(payload["guide_mean"], dtype=np.float32)
        guide_components = np.asarray(payload["guide_components"], dtype=np.float32)
        guide_scales = np.asarray(payload["guide_scales"], dtype=np.float32)
        condition_names = tuple(np.asarray(payload["condition_names"]).astype(str).tolist())
        condition_mean = np.asarray(payload["condition_mean"], dtype=np.float32)
        condition_scale = np.asarray(payload["condition_scale"], dtype=np.float32)
        latent_w = np.asarray(payload["latent_w"], dtype=np.float32)
        latent_b = np.asarray(payload["latent_b"], dtype=np.float32)
        latent_noise = np.asarray(payload["latent_noise"], dtype=np.float32)
        latent_clip = np.asarray(payload["latent_clip"], dtype=np.float32)
        affine_w = (
            np.asarray(payload["affine_w"], dtype=np.float32)
            if "affine_w" in payload.files
            else None
        )
        affine_b = (
            np.asarray(payload["affine_b"], dtype=np.float32)
            if "affine_b" in payload.files
            else None
        )
        affine_clip = (
            np.asarray(payload["affine_clip"], dtype=np.float32)
            if "affine_clip" in payload.files
            else None
        )
        affine_w = (
            np.asarray(payload["affine_w"], dtype=np.float32)
            if "affine_w" in payload.files
            else None
        )
        affine_b = (
            np.asarray(payload["affine_b"], dtype=np.float32)
            if "affine_b" in payload.files
            else None
        )
        affine_clip = (
            np.asarray(payload["affine_clip"], dtype=np.float32)
            if "affine_clip" in payload.files
            else None
        )
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
            raise ValueError("Residual checkpoint grid must be a strictly increasing vector")
        if mean.shape not in {(grid.size,), (2 * grid.size,)}:
            raise ValueError("Hybrid base mean is incompatible")
        if components.ndim != 2 or components.shape[1] != mean.size:
            raise ValueError("Hybrid base components are incompatible")
        if scales.shape != (components.shape[0],):
            raise ValueError("Hybrid base scales are incompatible")
        if guide_mean.shape != mean.shape:
            raise ValueError("Hybrid guide mean is incompatible")
        if guide_components.ndim != 2 or guide_components.shape[1] != guide_mean.size:
            raise ValueError("Hybrid guide components are incompatible")
        if guide_scales.shape != (guide_components.shape[0],):
            raise ValueError("Hybrid guide scales are incompatible")
        unsupported = [name for name in condition_names if name not in CONDITION_NAMES]
        if unsupported:
            raise ValueError("Hybrid checkpoint condition schema is incompatible")
        condition_shape = (len(condition_names),)
        if condition_mean.shape != condition_shape or condition_scale.shape != condition_shape:
            raise ValueError("Hybrid checkpoint condition scaler is incompatible")
        if latent_w.shape != (len(condition_names), guide_components.shape[0]):
            raise ValueError("Hybrid checkpoint latent regression is incompatible")
        if latent_b.shape != (guide_components.shape[0],):
            raise ValueError("Hybrid checkpoint latent bias is incompatible")
        if latent_noise.shape != (guide_components.shape[0],):
            raise ValueError("Hybrid checkpoint latent noise is incompatible")
        if latent_clip.shape != (guide_components.shape[0],):
            raise ValueError("Hybrid checkpoint latent clip is incompatible")
        reverse_required = (
            "reverse_guide_mean",
            "reverse_guide_components",
            "reverse_guide_scales",
            "reverse_condition_mean",
            "reverse_condition_scale",
            "reverse_latent_w",
            "reverse_latent_b",
            "reverse_latent_noise",
            "reverse_latent_clip",
        )
        reverse_present = [name for name in reverse_required if name in payload.files]
        if reverse_present and len(reverse_present) != len(reverse_required):
            raise ValueError(
                "Hybrid checkpoint reverse guide is incomplete"
            )
        values = (
            grid,
            mean,
            components,
            scales,
            guide_mean,
            guide_components,
            guide_scales,
            condition_mean,
            condition_scale,
            latent_w,
            latent_b,
            latent_noise,
            latent_clip,
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Hybrid threshold PCA checkpoint contains non-finite values")
        reverse_values: tuple[np.ndarray, ...] = ()
        if reverse_present:
            reverse_guide_mean = np.asarray(payload["reverse_guide_mean"], dtype=np.float32)
            reverse_guide_components = np.asarray(
                payload["reverse_guide_components"],
                dtype=np.float32,
            )
            reverse_guide_scales = np.asarray(
                payload["reverse_guide_scales"],
                dtype=np.float32,
            )
            reverse_condition_mean = np.asarray(
                payload["reverse_condition_mean"],
                dtype=np.float32,
            )
            reverse_condition_scale = np.asarray(
                payload["reverse_condition_scale"],
                dtype=np.float32,
            )
            reverse_latent_w = np.asarray(payload["reverse_latent_w"], dtype=np.float32)
            reverse_latent_b = np.asarray(payload["reverse_latent_b"], dtype=np.float32)
            reverse_latent_noise = np.asarray(
                payload["reverse_latent_noise"],
                dtype=np.float32,
            )
            reverse_latent_clip = np.asarray(
                payload["reverse_latent_clip"],
                dtype=np.float32,
            )
            if reverse_guide_mean.shape != mean.shape:
                raise ValueError("Hybrid reverse guide mean is incompatible")
            if (
                reverse_guide_components.ndim != 2
                or reverse_guide_components.shape[1] != reverse_guide_mean.size
            ):
                raise ValueError("Hybrid reverse guide components are incompatible")
            if reverse_guide_scales.shape != (reverse_guide_components.shape[0],):
                raise ValueError("Hybrid reverse guide scales are incompatible")
            if (
                reverse_condition_mean.shape != condition_shape
                or reverse_condition_scale.shape != condition_shape
            ):
                raise ValueError("Hybrid reverse guide condition scaler is incompatible")
            if reverse_latent_w.shape != (
                len(condition_names),
                reverse_guide_components.shape[0],
            ):
                raise ValueError("Hybrid reverse guide latent regression is incompatible")
            if reverse_latent_b.shape != (reverse_guide_components.shape[0],):
                raise ValueError("Hybrid reverse guide latent bias is incompatible")
            if reverse_latent_noise.shape != (reverse_guide_components.shape[0],):
                raise ValueError("Hybrid reverse guide latent noise is incompatible")
            if reverse_latent_clip.shape != (reverse_guide_components.shape[0],):
                raise ValueError("Hybrid reverse guide latent clip is incompatible")
            reverse_values = (
                reverse_guide_mean,
                reverse_guide_components,
                reverse_guide_scales,
                reverse_condition_mean,
                reverse_condition_scale,
                reverse_latent_w,
                reverse_latent_b,
                reverse_latent_noise,
                reverse_latent_clip,
            )
            if not all(np.all(np.isfinite(value)) for value in reverse_values):
                raise ValueError(
                    "Hybrid threshold PCA reverse guide contains non-finite values"
                )
        metadata = {}
        if "metadata_json" in payload.files:
            try:
                metadata = json.loads(str(payload["metadata_json"].item()))
            except (TypeError, ValueError):
                metadata = {}
        self.path = path
        self.grid = grid
        self.mean = mean
        self.components = components
        self.scales = np.maximum(scales, 1e-6)
        self.guide_mean = guide_mean
        self.guide_components = guide_components
        self.guide_scales = np.maximum(guide_scales, 1e-6)
        self.condition_names = condition_names
        self.condition_mean = condition_mean
        self.condition_scale = np.maximum(condition_scale, 1e-6)
        self.latent_w = latent_w
        self.latent_b = latent_b
        self.latent_noise = np.maximum(latent_noise, 1e-4)
        self.latent_clip = np.maximum(latent_clip, 1.0)
        if reverse_present:
            self.reverse_guide_mean = reverse_values[0]
            self.reverse_guide_components = reverse_values[1]
            self.reverse_guide_scales = np.maximum(reverse_values[2], 1e-6)
            self.reverse_condition_mean = reverse_values[3]
            self.reverse_condition_scale = np.maximum(reverse_values[4], 1e-6)
            self.reverse_latent_w = reverse_values[5]
            self.reverse_latent_b = reverse_values[6]
            self.reverse_latent_noise = np.maximum(reverse_values[7], 1e-4)
            self.reverse_latent_clip = np.maximum(reverse_values[8], 1.0)
        self.hybrid_local_blend = float(np.asarray(payload["hybrid_local_blend"]).item())
        self.hybrid_global_blend = float(np.asarray(payload["hybrid_global_blend"]).item())
        self.hybrid_window_scale = float(np.asarray(payload["hybrid_window_scale"]).item())
        self.hybrid_min_window_v = float(np.asarray(payload["hybrid_min_window_v"]).item())
        self.hybrid_base_scale_multiplier = float(
            np.asarray(
                payload["hybrid_base_scale_multiplier"]
                if "hybrid_base_scale_multiplier" in payload.files
                else 1.0
            ).item()
        )
        self.hybrid_guide_align_strength = float(
            np.asarray(
                payload["hybrid_guide_align_strength"]
                if "hybrid_guide_align_strength" in payload.files
                else 0.0
            ).item()
        )
        self.hybrid_guide_align_window_scale = float(
            np.asarray(
                payload["hybrid_guide_align_window_scale"]
                if "hybrid_guide_align_window_scale" in payload.files
                else 2.0
            ).item()
        )
        self.hybrid_guide_delta_clip_decades = float(
            np.asarray(
                payload["hybrid_guide_delta_clip_decades"]
                if "hybrid_guide_delta_clip_decades" in payload.files
                else 0.0
            ).item()
        )
        self.hybrid_guide_delta_anchor_strength = float(
            np.asarray(
                payload["hybrid_guide_delta_anchor_strength"]
                if "hybrid_guide_delta_anchor_strength" in payload.files
                else 0.0
            ).item()
        )
        self.hybrid_guide_delta_preserve_affine_strength = float(
            np.asarray(
                payload["hybrid_guide_delta_preserve_affine_strength"]
                if "hybrid_guide_delta_preserve_affine_strength" in payload.files
                else 0.0
            ).item()
        )
        self.hybrid_post_vth_align_strength = float(
            np.asarray(
                payload["hybrid_post_vth_align_strength"]
                if "hybrid_post_vth_align_strength" in payload.files
                else 0.0
            ).item()
        )
        self.hybrid_post_vth_align_reverse_only = bool(
            int(
                np.asarray(
                    payload["hybrid_post_vth_align_reverse_only"]
                    if "hybrid_post_vth_align_reverse_only" in payload.files
                    else 0
                ).item()
            )
        )
        self.hybrid_post_vth_align_local_window_scale = float(
            np.asarray(
                payload["hybrid_post_vth_align_local_window_scale"]
                if "hybrid_post_vth_align_local_window_scale" in payload.files
                else 0.0
            ).item()
        )
        self.hybrid_post_vth_align_local_min_window_v = float(
            np.asarray(
                payload["hybrid_post_vth_align_local_min_window_v"]
                if "hybrid_post_vth_align_local_min_window_v" in payload.files
                else 0.18
            ).item()
        )
        self.hybrid_guide_as_local_delta = bool(
            int(
                np.asarray(
                    payload["hybrid_guide_as_local_delta"]
                    if "hybrid_guide_as_local_delta" in payload.files
                    else 0
                ).item()
            )
        )
        self.hybrid_reverse_on_state_blend_scale = float(
            np.asarray(
                payload["hybrid_reverse_on_state_blend_scale"]
                if "hybrid_reverse_on_state_blend_scale" in payload.files
                else 1.0
            ).item()
        )
        self.hybrid_reverse_on_state_delta_scale = float(
            np.asarray(
                payload["hybrid_reverse_on_state_delta_scale"]
                if "hybrid_reverse_on_state_delta_scale" in payload.files
                else 1.0
            ).item()
        )
        self.hybrid_reverse_on_state_onset_u_scale = float(
            np.asarray(
                payload["hybrid_reverse_on_state_onset_u_scale"]
                if "hybrid_reverse_on_state_onset_u_scale" in payload.files
                else 1.8
            ).item()
        )
        self.hybrid_reverse_on_state_window_scale = float(
            np.asarray(
                payload["hybrid_reverse_on_state_window_scale"]
                if "hybrid_reverse_on_state_window_scale" in payload.files
                else 1.2
            ).item()
        )
        self.residual_mean = None
        self.residual_scale = None
        self.decoder_w = None
        self.decoder_b = None
        self.skip_w = None
        self.output_w = None
        self.output_b = None
        self.metadata = metadata

    def _load_conditional_pca(self, payload, path: Path) -> None:
        required = (
            "grid",
            "mean",
            "components",
            "scales",
            "condition_names",
            "condition_mean",
            "condition_scale",
            "latent_w",
            "latent_b",
            "latent_noise",
            "latent_clip",
        )
        missing = [name for name in required if name not in payload.files]
        if missing:
            raise ValueError(f"Conditional PCA checkpoint is missing: {', '.join(missing)}")
        grid = np.asarray(payload["grid"], dtype=np.float32)
        mean = np.asarray(payload["mean"], dtype=np.float32)
        components = np.asarray(payload["components"], dtype=np.float32)
        scales = np.asarray(payload["scales"], dtype=np.float32)
        condition_names = tuple(np.asarray(payload["condition_names"]).astype(str).tolist())
        condition_mean = np.asarray(payload["condition_mean"], dtype=np.float32)
        condition_scale = np.asarray(payload["condition_scale"], dtype=np.float32)
        latent_w = np.asarray(payload["latent_w"], dtype=np.float32)
        latent_b = np.asarray(payload["latent_b"], dtype=np.float32)
        latent_noise = np.asarray(payload["latent_noise"], dtype=np.float32)
        latent_clip = np.asarray(payload["latent_clip"], dtype=np.float32)
        affine_w = (
            np.asarray(payload["affine_w"], dtype=np.float32)
            if "affine_w" in payload.files
            else None
        )
        affine_b = (
            np.asarray(payload["affine_b"], dtype=np.float32)
            if "affine_b" in payload.files
            else None
        )
        affine_clip = (
            np.asarray(payload["affine_clip"], dtype=np.float32)
            if "affine_clip" in payload.files
            else None
        )
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.diff(grid) > 0):
            raise ValueError("Residual checkpoint grid must be a strictly increasing vector")
        if mean.shape not in {(grid.size,), (2 * grid.size,)}:
            raise ValueError("Conditional PCA checkpoint mean is incompatible")
        if components.ndim != 2 or components.shape[1] != mean.size:
            raise ValueError("Conditional PCA checkpoint components are incompatible")
        if scales.shape != (components.shape[0],):
            raise ValueError("Conditional PCA checkpoint scales are incompatible")
        unsupported = [name for name in condition_names if name not in CONDITION_NAMES]
        if unsupported:
            raise ValueError("Conditional PCA checkpoint condition schema is incompatible")
        condition_shape = (len(condition_names),)
        if condition_mean.shape != condition_shape or condition_scale.shape != condition_shape:
            raise ValueError("Conditional PCA checkpoint condition scaler is incompatible")
        if latent_w.shape != (len(condition_names), components.shape[0]):
            raise ValueError("Conditional PCA checkpoint latent regression is incompatible")
        if latent_b.shape != (components.shape[0],):
            raise ValueError("Conditional PCA checkpoint latent bias is incompatible")
        if latent_noise.shape != (components.shape[0],):
            raise ValueError("Conditional PCA checkpoint latent noise is incompatible")
        if latent_clip.shape != (components.shape[0],):
            raise ValueError("Conditional PCA checkpoint latent clip is incompatible")
        if affine_w is not None and affine_w.shape != (len(condition_names), 2):
            raise ValueError("Conditional PCA checkpoint affine regression is incompatible")
        if affine_b is not None and affine_b.shape != (2,):
            raise ValueError("Conditional PCA checkpoint affine bias is incompatible")
        if affine_clip is not None and affine_clip.shape != (2,):
            raise ValueError("Conditional PCA checkpoint affine clip is incompatible")
        values = (
            grid,
            mean,
            components,
            scales,
            condition_mean,
            condition_scale,
            latent_w,
            latent_b,
            latent_noise,
            latent_clip,
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise ValueError("Conditional PCA checkpoint contains non-finite values")
        metadata = {}
        if "metadata_json" in payload.files:
            try:
                metadata = json.loads(str(payload["metadata_json"].item()))
            except (TypeError, ValueError):
                metadata = {}
        self.path = path
        self.grid = grid
        self.mean = mean
        self.components = components
        self.scales = np.maximum(scales, 1e-6)
        self.condition_names = condition_names
        self.condition_mean = condition_mean
        self.condition_scale = np.maximum(condition_scale, 1e-6)
        self.latent_w = latent_w
        self.latent_b = latent_b
        self.latent_noise = np.maximum(latent_noise, 1e-4)
        self.latent_clip = np.maximum(latent_clip, 1.0)
        self.affine_w = affine_w
        self.affine_b = affine_b
        self.affine_clip = (
            np.maximum(affine_clip, 1e-6) if affine_clip is not None else None
        )
        self.threshold_focus_strength = float(
            np.asarray(
                payload["threshold_focus_strength"]
                if "threshold_focus_strength" in payload.files
                else 0.0
            ).item()
        )
        self.threshold_focus_window_scale = float(
            np.asarray(
                payload["threshold_focus_window_scale"]
                if "threshold_focus_window_scale" in payload.files
                else 2.2
            ).item()
        )
        self.threshold_focus_min_window_v = float(
            np.asarray(
                payload["threshold_focus_min_window_v"]
                if "threshold_focus_min_window_v" in payload.files
                else 0.16
            ).item()
        )
        self.threshold_local_align_window_scale = float(
            np.asarray(
                payload["threshold_local_align_window_scale"]
                if "threshold_local_align_window_scale" in payload.files
                else 0.0
            ).item()
        )
        self.threshold_local_align_min_window_v = float(
            np.asarray(
                payload["threshold_local_align_min_window_v"]
                if "threshold_local_align_min_window_v" in payload.files
                else 0.12
            ).item()
        )
        self.threshold_local_delta_transform = bool(
            int(
                np.asarray(
                    payload["threshold_local_delta_transform"]
                    if "threshold_local_delta_transform" in payload.files
                    else 0
                ).item()
            )
        )
        self.residual_mean = None
        self.residual_scale = None
        self.decoder_w = None
        self.decoder_b = None
        self.skip_w = None
        self.output_w = None
        self.output_b = None
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
            components=int(
                self.metadata.get(
                    "display_components",
                    (
                        int(self.decoder_w.shape[0] - len(self.condition_names))
                        if self.decoder_w is not None
                        else int(self.latent_w.shape[1])
                        if self.latent_w is not None
                        else int(self.components.shape[0])
                        if self.components is not None
                        else 0
                    ),
                )
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
            condition_features=self.metadata.get(
                "condition_features",
                list(self.condition_names)
                if self.decoder_w is not None or self.latent_w is not None
                else [],
            ),
            sample_balance_strategy=self.metadata.get("sample_balance_strategy"),
            rare_curve_groups=self.metadata.get("rare_curve_groups"),
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
            latent_dim = self.decoder_w.shape[0] - len(self.condition_names)
            if (
                self.latent_prior_mean is not None
                and self.latent_prior_std is not None
                and self.latent_prior_mean.shape == (latent_dim,)
                and self.latent_prior_std.shape == (latent_dim,)
            ):
                latent_temperature = 0.25 + 0.75 * diversity
                latent = (
                    self.latent_prior_mean
                    + rng.normal(0.0, 1.0, latent_dim).astype(np.float32)
                    * self.latent_prior_std
                    * latent_temperature
                ).astype(np.float32)
            else:
                latent = rng.normal(0.0, 0.35 + diversity, latent_dim).astype(np.float32)
            raw_condition = condition_from_generation(
                condition,
                reverse=reverse,
                names=self.condition_names,
            )
            standardized_condition = (
                (raw_condition - self.condition_mean) / self.condition_scale
            ).astype(np.float32)
            decoder_input = np.concatenate([latent, standardized_condition])
            hidden = np.tanh(decoder_input @ self.decoder_w + self.decoder_b)
            standardized_residual = hidden @ self.output_w + self.output_b
            if self.skip_w is not None:
                standardized_residual = standardized_residual + decoder_input @ self.skip_w
            residual = self.residual_mean + standardized_residual * self.residual_scale
            drain_residual = residual[: self.grid.size]
            gate_residual = (
                residual[self.grid.size : 2 * self.grid.size]
                if residual.size >= 2 * self.grid.size
                else None
            )
            if (
                self.threshold_local_delta_transform
            ):
                drain_residual = _restore_threshold_local_alignment(
                    self.grid,
                    _restore_drain_delta_rows(drain_residual[None, :])[0],
                    condition,
                    reverse=reverse,
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
        if (
            self.components is not None
            and self.grid is not None
            and self.guide_mean is not None
            and self.guide_components is not None
            and self.guide_scales is not None
            and self.latent_w is not None
            and self.latent_b is not None
            and self.latent_noise is not None
            and self.latent_clip is not None
            and self.hybrid_local_blend is not None
            and self.hybrid_global_blend is not None
            and self.hybrid_window_scale is not None
            and self.hybrid_min_window_v is not None
            and self.hybrid_guide_align_strength is not None
            and self.hybrid_guide_align_window_scale is not None
            and self.hybrid_guide_delta_clip_decades is not None
            and self.hybrid_reverse_on_state_blend_scale is not None
            and self.hybrid_reverse_on_state_delta_scale is not None
            and self.hybrid_reverse_on_state_onset_u_scale is not None
            and self.hybrid_reverse_on_state_window_scale is not None
        ):
            if condition is None:
                raise ValueError("Hybrid threshold PCA sampling requires a generation condition")
            if self.condition_mean is None or self.condition_scale is None or self.mean is None:
                raise ValueError("Hybrid threshold PCA checkpoint is incomplete")
            base_count = self.components.shape[0]
            base_latent = rng.normal(0.0, 1.0, base_count)
            base_scales = (
                self.scales
                if self.scales is not None
                else np.ones(base_count, dtype=np.float32)
            )
            if self.hybrid_base_scale_multiplier is not None:
                base_scales = base_scales * self.hybrid_base_scale_multiplier
            base_residual = self.mean + (base_latent * base_scales) @ self.components
            guide_mean = self.guide_mean
            guide_components = self.guide_components
            guide_scales = self.guide_scales
            condition_mean = self.condition_mean
            condition_scale = self.condition_scale
            latent_w = self.latent_w
            latent_b = self.latent_b
            latent_noise = self.latent_noise
            latent_clip = self.latent_clip
            if (
                reverse
                and self.reverse_guide_mean is not None
                and self.reverse_guide_components is not None
                and self.reverse_guide_scales is not None
                and self.reverse_condition_mean is not None
                and self.reverse_condition_scale is not None
                and self.reverse_latent_w is not None
                and self.reverse_latent_b is not None
                and self.reverse_latent_noise is not None
                and self.reverse_latent_clip is not None
            ):
                guide_mean = self.reverse_guide_mean
                guide_components = self.reverse_guide_components
                guide_scales = self.reverse_guide_scales
                condition_mean = self.reverse_condition_mean
                condition_scale = self.reverse_condition_scale
                latent_w = self.reverse_latent_w
                latent_b = self.reverse_latent_b
                latent_noise = self.reverse_latent_noise
                latent_clip = self.reverse_latent_clip

            raw_condition = condition_from_generation(
                condition,
                reverse=reverse,
                names=self.condition_names,
            )
            standardized_condition = (
                (raw_condition - condition_mean) / condition_scale
            ).astype(np.float32)
            guide_latent = standardized_condition @ latent_w + latent_b
            guide_latent = np.clip(guide_latent, -latent_clip, latent_clip)
            noise_scale = (0.06 + 0.84 * diversity) * latent_noise
            guide_latent = guide_latent + rng.normal(
                0.0,
                noise_scale,
                latent_noise.shape[0],
            )
            guide_latent = np.clip(guide_latent, -latent_clip, latent_clip)
            guide_residual = guide_mean + (guide_latent * guide_scales) @ guide_components

            base_drain, base_gate = _split_residual_channels(base_residual, self.grid.size)
            guide_drain, guide_gate = _split_residual_channels(guide_residual, self.grid.size)

            grid_voltage = condition.voltage_min + 0.5 * (self.grid + 1.0) * (
                condition.voltage_max - condition.voltage_min
            )
            effective_vth = _target_transfer_vth(condition, reverse=reverse)
            span = max(condition.voltage_max - condition.voltage_min, 1e-6)
            local_window_v = max(
                self.hybrid_window_scale * condition.target_ss_mv_dec / 1000.0,
                self.hybrid_min_window_v,
                0.02 * span,
            )
            local_envelope = np.exp(
                -0.5 * ((grid_voltage - effective_vth) / local_window_v) ** 2
            )
            reverse_on_state_weight = np.zeros_like(local_envelope)
            if reverse:
                sign = 1.0 if condition.polarity == "n-type" else -1.0
                u = sign * (grid_voltage - effective_vth)
                onset_u = (
                    self.hybrid_reverse_on_state_onset_u_scale
                    * condition.target_ss_mv_dec
                    / 1000.0
                )
                transition_v = max(
                    self.hybrid_reverse_on_state_window_scale
                    * condition.target_ss_mv_dec
                    / 1000.0,
                    0.01 * span,
                    0.03,
                )
                reverse_on_state_weight = 1.0 / (
                    1.0
                    + np.exp(
                        -np.clip(
                            (u - onset_u) / transition_v,
                            -80.0,
                            80.0,
                        )
                    )
                )
            if self.hybrid_guide_align_strength > 0 and not self.hybrid_guide_as_local_delta:
                align_window_v = max(
                    self.hybrid_guide_align_window_scale
                    * condition.target_ss_mv_dec
                    / 1000.0,
                    0.5 * self.hybrid_min_window_v,
                    0.01 * span,
                )
                align_weights = np.exp(
                    -0.5 * ((grid_voltage - effective_vth) / align_window_v) ** 2
                )
                base_value = float(np.interp(effective_vth, grid_voltage, base_drain))
                guide_value = float(np.interp(effective_vth, grid_voltage, guide_drain))
                base_slope = _weighted_local_slope(grid_voltage, base_drain, align_weights)
                guide_slope = _weighted_local_slope(grid_voltage, guide_drain, align_weights)
                if abs(guide_slope) > 1e-6:
                    slope_scale = float(np.clip(base_slope / guide_slope, 0.45, 2.2))
                else:
                    slope_scale = 1.0
                aligned_guide_drain = base_value + slope_scale * (guide_drain - guide_value)
                align_strength = np.clip(
                    self.hybrid_guide_align_strength * align_weights,
                    0.0,
                    1.0,
                )
                guide_drain = (
                    (1.0 - align_strength) * guide_drain
                    + align_strength * aligned_guide_drain
                )
            if self.hybrid_guide_delta_clip_decades > 0:
                delta_limit = self.hybrid_guide_delta_clip_decades * (
                    0.30 + 0.70 * local_envelope
                )
                if reverse and self.hybrid_reverse_on_state_delta_scale != 1.0:
                    delta_limit = delta_limit * (
                        1.0
                        + (self.hybrid_reverse_on_state_delta_scale - 1.0)
                        * reverse_on_state_weight
                    )
                if self.hybrid_guide_as_local_delta:
                    guide_drain = np.clip(
                        guide_drain,
                        -delta_limit,
                        delta_limit,
                    )
                else:
                    guide_drain = base_drain + np.clip(
                        guide_drain - base_drain,
                        -delta_limit,
                        delta_limit,
                    )
            if (
                self.hybrid_guide_as_local_delta
                and self.hybrid_guide_delta_anchor_strength is not None
                and self.hybrid_guide_delta_anchor_strength > 0
            ):
                anchor_value = float(np.interp(effective_vth, grid_voltage, guide_drain))
                guide_drain = guide_drain - (
                    self.hybrid_guide_delta_anchor_strength * anchor_value * local_envelope
                )
            if (
                self.hybrid_guide_as_local_delta
                and self.hybrid_guide_delta_preserve_affine_strength is not None
                and self.hybrid_guide_delta_preserve_affine_strength > 0
            ):
                affine_intercept, affine_slope = _weighted_local_affine(
                    grid_voltage,
                    guide_drain,
                    local_envelope,
                    center_x=effective_vth,
                )
                affine_component = affine_intercept + affine_slope * (
                    grid_voltage - effective_vth
                )
                guide_drain = guide_drain - (
                    self.hybrid_guide_delta_preserve_affine_strength
                    * affine_component
                    * local_envelope
                )
            blend_grid = np.clip(
                self.hybrid_global_blend
                + (self.hybrid_local_blend - self.hybrid_global_blend) * local_envelope,
                0.0,
                1.0,
            )
            if reverse and self.hybrid_reverse_on_state_blend_scale != 1.0:
                blend_grid = np.clip(
                    blend_grid
                    * (
                        1.0
                        + (self.hybrid_reverse_on_state_blend_scale - 1.0)
                        * reverse_on_state_weight
                    ),
                    0.0,
                    1.0,
                )
            if self.hybrid_guide_as_local_delta:
                drain = base_drain + blend_grid * guide_drain
            else:
                drain = base_drain + blend_grid * (guide_drain - base_drain)
            if base_gate is not None or guide_gate is not None:
                base_gate_values = (
                    base_gate if base_gate is not None else np.zeros_like(base_drain)
                )
                if self.hybrid_guide_as_local_delta:
                    guide_gate_values = (
                        guide_gate if guide_gate is not None else np.zeros_like(base_gate_values)
                    )
                    gate = base_gate_values + blend_grid * 0.65 * guide_gate_values
                else:
                    guide_gate_values = (
                        guide_gate if guide_gate is not None else base_gate_values
                    )
                    gate = base_gate_values + blend_grid * 0.65 * (
                        guide_gate_values - base_gate_values
                    )
            else:
                gate = None
            latent_code = [*base_latent.astype(float).tolist(), *guide_latent.astype(float).tolist()]
            return ResidualSample(
                values=np.interp(normalized_voltage, self.grid, drain),
                mode=self.mode,
                latent_code=latent_code,
                gate_values=(
                    np.interp(normalized_voltage, self.grid, gate)
                    if gate is not None
                    else None
                ),
            )
        if (
            self.components is not None
            and self.grid is not None
            and self.latent_w is not None
            and self.latent_b is not None
            and self.latent_noise is not None
            and self.latent_clip is not None
        ):
            if condition is None:
                raise ValueError("Conditional PCA sampling requires a generation condition")
            if self.condition_mean is None or self.condition_scale is None or self.mean is None:
                raise ValueError("Conditional PCA checkpoint is incomplete")
            raw_condition = condition_from_generation(
                condition,
                reverse=reverse,
                names=self.condition_names,
            )
            standardized_condition = (
                (raw_condition - self.condition_mean) / self.condition_scale
            ).astype(np.float32)
            predicted_latent = standardized_condition @ self.latent_w + self.latent_b
            predicted_latent = np.clip(
                predicted_latent,
                -self.latent_clip,
                self.latent_clip,
            )
            noise_scale = (0.08 + 0.92 * diversity) * self.latent_noise
            latent = predicted_latent + rng.normal(
                0.0,
                noise_scale,
                self.latent_noise.shape[0],
            )
            scales = self.scales if self.scales is not None else np.ones(latent.shape[0])
            residual = self.mean + (latent * scales) @ self.components
            drain_residual = residual[: self.grid.size]
            if self.threshold_local_delta_transform:
                drain_residual = _restore_drain_delta_rows(drain_residual)
                if (
                    self.affine_w is not None
                    and self.affine_b is not None
                    and self.threshold_local_align_window_scale is not None
                    and self.threshold_local_align_min_window_v is not None
                ):
                    predicted_affine = (
                        standardized_condition @ self.affine_w + self.affine_b
                    ).astype(np.float32)
                    if self.affine_clip is not None:
                        predicted_affine = np.clip(
                            predicted_affine,
                            -self.affine_clip,
                            self.affine_clip,
                        )
                    aligned_weights = _aligned_threshold_local_envelope(
                        self.grid,
                        condition,
                        window_scale=self.threshold_local_align_window_scale,
                        min_window_v=self.threshold_local_align_min_window_v,
                    )
                    affine_component = (
                        predicted_affine[0] + predicted_affine[1] * self.grid
                    )
                    drain_residual = drain_residual + affine_component * aligned_weights
            if (
                self.threshold_focus_strength is not None
                and self.threshold_focus_window_scale is not None
                and self.threshold_focus_min_window_v is not None
                and self.threshold_focus_strength > 0
            ):
                focus = _threshold_focus_envelope(
                    self.grid,
                    condition,
                    reverse=reverse,
                    strength=self.threshold_focus_strength,
                    window_scale=self.threshold_focus_window_scale,
                    min_window_v=self.threshold_focus_min_window_v,
                )
                drain_residual = drain_residual / np.maximum(focus, 1e-6)
            if (
                self.threshold_local_align_window_scale is not None
                and self.threshold_local_align_min_window_v is not None
                and self.threshold_local_align_window_scale > 0
            ):
                drain_residual = _restore_threshold_local_alignment(
                    self.grid,
                    drain_residual,
                    condition,
                    reverse=reverse,
                )
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
