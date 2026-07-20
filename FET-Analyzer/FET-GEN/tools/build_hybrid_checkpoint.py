from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT
    / "experiments"
    / "model-experiments-20260627-220813"
    / "attempt_1_pca16.npz"
)
DEFAULT_GUIDE = (
    ROOT
    / "experiments"
    / "conditional-pca-component-sweep-20260627-224618"
    / "attempt_8_conditional_pca12_clipped.npz"
)
DEFAULT_OUTPUT = ROOT / "models" / "residual-hybrid-threshold-pca.npz"


def _load_metadata(payload) -> dict:
    if "metadata_json" not in payload.files:
        return {}
    try:
        return json.loads(str(payload["metadata_json"].item()))
    except (TypeError, ValueError):
        return {}


def build_hybrid_checkpoint(
    *,
    base_path: Path,
    guide_path: Path,
    reverse_guide_path: Path | None = None,
    guide_as_local_delta: bool = False,
    output_path: Path,
    base_scale_multiplier: float = 1.0,
    local_blend: float,
    global_blend: float,
    window_scale: float,
    min_window_v: float,
    guide_align_strength: float = 0.0,
    guide_align_window_scale: float = 2.0,
    guide_delta_clip_decades: float = 0.0,
    guide_delta_anchor_strength: float = 0.0,
    guide_delta_preserve_affine_strength: float = 0.0,
    post_vth_align_strength: float = 0.0,
    post_vth_align_reverse_only: bool = False,
    post_vth_align_local_window_scale: float = 0.0,
    post_vth_align_local_min_window_v: float = 0.18,
    reverse_on_state_blend_scale: float = 1.0,
    reverse_on_state_delta_scale: float = 1.0,
    reverse_on_state_onset_u_scale: float = 1.8,
    reverse_on_state_window_scale: float = 1.2,
) -> Path:
    with np.load(base_path) as base_payload, np.load(guide_path) as guide_payload:
        base_type = (
            str(base_payload["model_type"].item())
            if "model_type" in base_payload.files
            else "learned_pca"
        )
        guide_type = (
            str(guide_payload["model_type"].item())
            if "model_type" in guide_payload.files
            else "learned_pca"
        )
        if base_type != "learned_pca":
            raise ValueError(f"Base checkpoint must be learned_pca, got {base_type}")
        if guide_type != "conditional_pca":
            raise ValueError(f"Guide checkpoint must be conditional_pca, got {guide_type}")

        grid = np.asarray(base_payload["grid"], dtype=np.float32)
        guide_grid = np.asarray(guide_payload["grid"], dtype=np.float32)
        if grid.shape != guide_grid.shape or not np.allclose(grid, guide_grid):
            raise ValueError("Base and guide checkpoints must share the same grid")

        base_mean = np.asarray(base_payload["mean"], dtype=np.float32)
        base_components = np.asarray(base_payload["components"], dtype=np.float32)
        base_scales = np.asarray(base_payload["scales"], dtype=np.float32)

        guide_mean = np.asarray(guide_payload["mean"], dtype=np.float32)
        guide_components = np.asarray(guide_payload["components"], dtype=np.float32)
        guide_scales = np.asarray(guide_payload["scales"], dtype=np.float32)
        condition_names = np.asarray(guide_payload["condition_names"])
        condition_mean = np.asarray(guide_payload["condition_mean"], dtype=np.float32)
        condition_scale = np.asarray(guide_payload["condition_scale"], dtype=np.float32)
        latent_w = np.asarray(guide_payload["latent_w"], dtype=np.float32)
        latent_b = np.asarray(guide_payload["latent_b"], dtype=np.float32)
        latent_noise = np.asarray(guide_payload["latent_noise"], dtype=np.float32)
        latent_clip = np.asarray(guide_payload["latent_clip"], dtype=np.float32)

        base_metadata = _load_metadata(base_payload)
        guide_metadata = _load_metadata(guide_payload)

    reverse_payload: dict[str, np.ndarray] = {}
    reverse_metadata: dict = {}
    if reverse_guide_path is not None:
        with np.load(reverse_guide_path) as reverse_payload_npz:
            reverse_type = (
                str(reverse_payload_npz["model_type"].item())
                if "model_type" in reverse_payload_npz.files
                else "learned_pca"
            )
            if reverse_type != "conditional_pca":
                raise ValueError(
                    f"Reverse guide checkpoint must be conditional_pca, got {reverse_type}"
                )
            reverse_grid = np.asarray(reverse_payload_npz["grid"], dtype=np.float32)
            if grid.shape != reverse_grid.shape or not np.allclose(grid, reverse_grid):
                raise ValueError("Base and reverse guide checkpoints must share the same grid")
            reverse_condition_names = np.asarray(reverse_payload_npz["condition_names"])
            if reverse_condition_names.shape != condition_names.shape or not np.array_equal(
                reverse_condition_names,
                condition_names,
            ):
                raise ValueError("Guide and reverse guide checkpoints must share condition names")
            reverse_payload = {
                "reverse_guide_mean": np.asarray(
                    reverse_payload_npz["mean"],
                    dtype=np.float32,
                ),
                "reverse_guide_components": np.asarray(
                    reverse_payload_npz["components"],
                    dtype=np.float32,
                ),
                "reverse_guide_scales": np.asarray(
                    reverse_payload_npz["scales"],
                    dtype=np.float32,
                ),
                "reverse_condition_mean": np.asarray(
                    reverse_payload_npz["condition_mean"],
                    dtype=np.float32,
                ),
                "reverse_condition_scale": np.asarray(
                    reverse_payload_npz["condition_scale"],
                    dtype=np.float32,
                ),
                "reverse_latent_w": np.asarray(
                    reverse_payload_npz["latent_w"],
                    dtype=np.float32,
                ),
                "reverse_latent_b": np.asarray(
                    reverse_payload_npz["latent_b"],
                    dtype=np.float32,
                ),
                "reverse_latent_noise": np.asarray(
                    reverse_payload_npz["latent_noise"],
                    dtype=np.float32,
                ),
                "reverse_latent_clip": np.asarray(
                    reverse_payload_npz["latent_clip"],
                    dtype=np.float32,
                ),
            }
            reverse_metadata = _load_metadata(reverse_payload_npz)

    metadata = {
        **base_metadata,
        "objective": "hybrid_threshold_pca_guided_generation",
        "architecture": "hybrid_threshold_pca",
        "method": "hybrid_threshold_pca",
        "source": (
            f"base:{base_path}; guide:{guide_path}"
            + (
                f"; reverse_guide:{reverse_guide_path}"
                if reverse_guide_path is not None
                else ""
            )
        ),
        "hybrid_local_blend": local_blend,
        "hybrid_global_blend": global_blend,
        "hybrid_window_scale": window_scale,
        "hybrid_min_window_v": min_window_v,
        "hybrid_base_scale_multiplier": base_scale_multiplier,
        "hybrid_guide_align_strength": guide_align_strength,
        "hybrid_guide_align_window_scale": guide_align_window_scale,
        "hybrid_guide_delta_clip_decades": guide_delta_clip_decades,
        "hybrid_guide_delta_anchor_strength": guide_delta_anchor_strength,
        "hybrid_guide_delta_preserve_affine_strength": guide_delta_preserve_affine_strength,
        "hybrid_post_vth_align_strength": post_vth_align_strength,
        "hybrid_post_vth_align_reverse_only": post_vth_align_reverse_only,
        "hybrid_post_vth_align_local_window_scale": post_vth_align_local_window_scale,
        "hybrid_post_vth_align_local_min_window_v": post_vth_align_local_min_window_v,
        "hybrid_guide_as_local_delta": guide_as_local_delta,
        "hybrid_reverse_on_state_blend_scale": reverse_on_state_blend_scale,
        "hybrid_reverse_on_state_delta_scale": reverse_on_state_delta_scale,
        "hybrid_reverse_on_state_onset_u_scale": reverse_on_state_onset_u_scale,
        "hybrid_reverse_on_state_window_scale": reverse_on_state_window_scale,
        "base_model_name": base_path.stem,
        "guide_model_name": guide_path.stem,
        "base_validation_weighted_rmse_decades": base_metadata.get(
            "validation_weighted_rmse_decades"
        ),
        "guide_validation_weighted_rmse_decades": guide_metadata.get(
            "validation_weighted_rmse_decades"
        ),
        "guide_feature_vth_mae_v": guide_metadata.get("feature_vth_mae_v"),
        "guide_feature_ss_mae_mv_dec": guide_metadata.get("feature_ss_mae_mv_dec"),
        "reverse_guide_model_name": (
            reverse_guide_path.stem if reverse_guide_path is not None else None
        ),
        "reverse_guide_validation_weighted_rmse_decades": reverse_metadata.get(
            "validation_weighted_rmse_decades"
        ),
        "reverse_guide_feature_vth_mae_v": reverse_metadata.get("feature_vth_mae_v"),
        "reverse_guide_feature_ss_mae_mv_dec": reverse_metadata.get(
            "feature_ss_mae_mv_dec"
        ),
        "hybrid_directional_guides": reverse_guide_path is not None,
        "display_components": int(base_components.shape[0] + guide_components.shape[0]),
        "channels": base_metadata.get("channels", guide_metadata.get("channels", ["Ids"])),
        "condition_features": guide_metadata.get("condition_features", []),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            model_type=np.asarray("hybrid_threshold_pca"),
            format_version=np.asarray(1, dtype=np.int64),
            grid=grid,
            mean=base_mean,
            components=base_components,
            scales=base_scales,
            guide_mean=guide_mean,
            guide_components=guide_components,
            guide_scales=guide_scales,
            condition_names=condition_names,
            condition_mean=condition_mean,
            condition_scale=condition_scale,
            latent_w=latent_w,
            latent_b=latent_b,
            latent_noise=latent_noise,
            latent_clip=latent_clip,
            hybrid_local_blend=np.asarray(local_blend, dtype=np.float32),
            hybrid_global_blend=np.asarray(global_blend, dtype=np.float32),
            hybrid_window_scale=np.asarray(window_scale, dtype=np.float32),
            hybrid_min_window_v=np.asarray(min_window_v, dtype=np.float32),
            hybrid_base_scale_multiplier=np.asarray(
                base_scale_multiplier,
                dtype=np.float32,
            ),
            hybrid_guide_align_strength=np.asarray(
                guide_align_strength,
                dtype=np.float32,
            ),
            hybrid_guide_align_window_scale=np.asarray(
                guide_align_window_scale,
                dtype=np.float32,
            ),
            hybrid_guide_delta_clip_decades=np.asarray(
                guide_delta_clip_decades,
                dtype=np.float32,
            ),
            hybrid_guide_delta_anchor_strength=np.asarray(
                guide_delta_anchor_strength,
                dtype=np.float32,
            ),
            hybrid_guide_delta_preserve_affine_strength=np.asarray(
                guide_delta_preserve_affine_strength,
                dtype=np.float32,
            ),
            hybrid_post_vth_align_strength=np.asarray(
                post_vth_align_strength,
                dtype=np.float32,
            ),
            hybrid_post_vth_align_reverse_only=np.asarray(
                1 if post_vth_align_reverse_only else 0,
                dtype=np.int64,
            ),
            hybrid_post_vth_align_local_window_scale=np.asarray(
                post_vth_align_local_window_scale,
                dtype=np.float32,
            ),
            hybrid_post_vth_align_local_min_window_v=np.asarray(
                post_vth_align_local_min_window_v,
                dtype=np.float32,
            ),
            hybrid_guide_as_local_delta=np.asarray(
                1 if guide_as_local_delta else 0,
                dtype=np.int64,
            ),
            hybrid_reverse_on_state_blend_scale=np.asarray(
                reverse_on_state_blend_scale,
                dtype=np.float32,
            ),
            hybrid_reverse_on_state_delta_scale=np.asarray(
                reverse_on_state_delta_scale,
                dtype=np.float32,
            ),
            hybrid_reverse_on_state_onset_u_scale=np.asarray(
                reverse_on_state_onset_u_scale,
                dtype=np.float32,
            ),
            hybrid_reverse_on_state_window_scale=np.asarray(
                reverse_on_state_window_scale,
                dtype=np.float32,
            ),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            **reverse_payload,
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hybrid threshold-guided PCA checkpoint.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--guide", type=Path, default=DEFAULT_GUIDE)
    parser.add_argument("--reverse-guide", type=Path)
    parser.add_argument("--guide-as-local-delta", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--guide-delta-anchor-strength", type=float, default=0.0)
    parser.add_argument("--guide-delta-preserve-affine-strength", type=float, default=0.0)
    parser.add_argument("--post-vth-align-strength", type=float, default=0.0)
    parser.add_argument("--post-vth-align-reverse-only", action="store_true")
    parser.add_argument("--post-vth-align-local-window-scale", type=float, default=0.0)
    parser.add_argument("--post-vth-align-local-min-window-v", type=float, default=0.18)
    parser.add_argument("--local-blend", type=float, default=0.82)
    parser.add_argument("--global-blend", type=float, default=0.06)
    parser.add_argument("--window-scale", type=float, default=3.0)
    parser.add_argument("--min-window-v", type=float, default=0.22)
    parser.add_argument("--guide-align-strength", type=float, default=0.0)
    parser.add_argument("--guide-align-window-scale", type=float, default=2.0)
    parser.add_argument("--guide-delta-clip-decades", type=float, default=0.0)
    parser.add_argument("--reverse-on-state-blend-scale", type=float, default=1.0)
    parser.add_argument("--reverse-on-state-delta-scale", type=float, default=1.0)
    parser.add_argument("--reverse-on-state-onset-u-scale", type=float, default=1.8)
    parser.add_argument("--reverse-on-state-window-scale", type=float, default=1.2)
    args = parser.parse_args()
    built = build_hybrid_checkpoint(
        base_path=args.base.expanduser().resolve(),
        guide_path=args.guide.expanduser().resolve(),
        reverse_guide_path=(
            args.reverse_guide.expanduser().resolve()
            if args.reverse_guide is not None
            else None
        ),
        guide_as_local_delta=args.guide_as_local_delta,
        output_path=args.output.expanduser().resolve(),
        base_scale_multiplier=args.base_scale_multiplier,
        local_blend=args.local_blend,
        global_blend=args.global_blend,
        window_scale=args.window_scale,
        min_window_v=args.min_window_v,
        guide_align_strength=args.guide_align_strength,
        guide_align_window_scale=args.guide_align_window_scale,
        guide_delta_clip_decades=args.guide_delta_clip_decades,
        guide_delta_anchor_strength=args.guide_delta_anchor_strength,
        guide_delta_preserve_affine_strength=args.guide_delta_preserve_affine_strength,
        post_vth_align_strength=args.post_vth_align_strength,
        post_vth_align_reverse_only=args.post_vth_align_reverse_only,
        post_vth_align_local_window_scale=args.post_vth_align_local_window_scale,
        post_vth_align_local_min_window_v=args.post_vth_align_local_min_window_v,
        reverse_on_state_blend_scale=args.reverse_on_state_blend_scale,
        reverse_on_state_delta_scale=args.reverse_on_state_delta_scale,
        reverse_on_state_onset_u_scale=args.reverse_on_state_onset_u_scale,
        reverse_on_state_window_scale=args.reverse_on_state_window_scale,
    )
    print(built)


if __name__ == "__main__":
    main()
