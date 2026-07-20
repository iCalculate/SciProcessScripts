from __future__ import annotations

import json
import time
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from devicecurvegen.neural import NeuralTrainingConfig, train_neural_checkpoint
from devicecurvegen.residual import ResidualEngine
from tools.build_export_subset import build_export_subset
from tools.build_hybrid_checkpoint import build_hybrid_checkpoint
from tools.evaluate_postprocess_rechecks import _canonical_metrics
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiments"
EXPORT_SOURCE = ROOT / "experiments" / "db-export-subset-20260629-110358"
BASE_CHECKPOINT = (
    ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
)
ACTIVE_CHECKPOINT = ROOT / "models" / "residual-hybrid-threshold-pca.npz"
SUBSET_REFERENCE_CHECKPOINT = (
    ROOT / "experiments" / "postfix-model-sweep-20260629-114505" / "subset_conditional_pca20_ridge02.npz"
)


@dataclass
class DirectionalHybridSpec:
    name: str
    description: str
    forward_config: NeuralTrainingConfig
    reverse_config: NeuralTrainingConfig
    local_blend: float = 0.82
    global_blend: float = 0.06
    window_scale: float = 3.0
    min_window_v: float = 0.22
    guide_align_strength: float = 0.0
    guide_align_window_scale: float = 2.0
    guide_delta_clip_decades: float = 0.0
    reverse_on_state_blend_scale: float = 1.0
    reverse_on_state_delta_scale: float = 1.0
    reverse_on_state_onset_u_scale: float = 1.8
    reverse_on_state_window_scale: float = 1.2


def _specs() -> list[DirectionalHybridSpec]:
    common = dict(
        validation_fraction=0.12,
        feature_eval_limit=256,
        gate_loss_weight=0.6,
        rare_curve_weight=1.8,
        seed=12345,
    )
    return [
        DirectionalHybridSpec(
            name="dirsplit_cond20_cond20",
            description="Forward and reverse guides both use the best general conditioned PCA recipe on their own sweep-direction subsets.",
            forward_config=NeuralTrainingConfig(
                method="conditional_pca",
                pca_components=20,
                beta=0.02,
                **common,
            ),
            reverse_config=NeuralTrainingConfig(
                method="conditional_pca",
                pca_components=20,
                beta=0.02,
                **common,
            ),
        ),
        DirectionalHybridSpec(
            name="dirsplit_cond16_cond16",
            description="Lower-capacity conditioned PCA guides to test whether the smaller reverse subset benefits from a tighter latent basis.",
            forward_config=NeuralTrainingConfig(
                method="conditional_pca",
                pca_components=16,
                beta=0.02,
                **common,
            ),
            reverse_config=NeuralTrainingConfig(
                method="conditional_pca",
                pca_components=16,
                beta=0.02,
                **common,
            ),
        ),
        DirectionalHybridSpec(
            name="dirsplit_cond20_thresh16",
            description="Forward guide keeps the general conditioned PCA recipe, while the reverse guide uses a threshold-focused conditioned PCA to spend more capacity near Vth.",
            forward_config=NeuralTrainingConfig(
                method="conditional_pca",
                pca_components=20,
                beta=0.02,
                **common,
            ),
            reverse_config=NeuralTrainingConfig(
                method="threshold_conditional_pca",
                pca_components=16,
                beta=0.014,
                gate_loss_weight=0.7,
                rare_curve_weight=2.0,
                validation_fraction=0.12,
                feature_eval_limit=256,
                seed=12345,
            ),
        ),
    ]


def _baseline_rows() -> list[dict[str, object]]:
    rows = _sample_conditions(limit=72)
    baselines = [
        (
            "active_hybrid_onstate_retuned",
            "Current production hybrid checkpoint with the on-state postprocessor limiter.",
            ACTIVE_CHECKPOINT,
        ),
        (
            "subset_conditional_pca20_postprocess_recheck",
            "Best existing single-guide conditioned PCA checkpoint on the export subset.",
            SUBSET_REFERENCE_CHECKPOINT,
        ),
    ]
    output: list[dict[str, object]] = []
    for name, description, checkpoint in baselines:
        engine = ResidualEngine(checkpoint)
        info = engine.info().model_dump(mode="json")
        output.append(
            {
                "name": name,
                "description": description,
                "checkpoint_path": str(checkpoint),
                "result": info,
                "jump_metrics": _jump_metrics(engine, rows),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )
    return output


def run(
    *,
    source_export: Path = EXPORT_SOURCE,
    base_checkpoint: Path = BASE_CHECKPOINT,
    spec_names: set[str] | None = None,
) -> Path:
    output_dir = OUTPUT_ROOT / f"direction-split-hybrid-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    forward_dataset = build_export_subset(
        source_export,
        output_dir / "dataset-forward",
        directions=("forward",),
    )
    reverse_dataset = build_export_subset(
        source_export,
        output_dir / "dataset-reverse",
        directions=("reverse",),
    )
    sampled_conditions = _sample_conditions(limit=72)
    results = _baseline_rows()

    for spec in _specs():
        if spec_names is not None and spec.name not in spec_names:
            continue
        forward_checkpoint = output_dir / f"{spec.name}-forward-guide.npz"
        reverse_checkpoint = output_dir / f"{spec.name}-reverse-guide.npz"
        hybrid_checkpoint = output_dir / f"{spec.name}-hybrid.npz"
        forward_result = train_neural_checkpoint(
            forward_checkpoint,
            dataset_path=forward_dataset,
            config=spec.forward_config,
        )
        reverse_result = train_neural_checkpoint(
            reverse_checkpoint,
            dataset_path=reverse_dataset,
            config=spec.reverse_config,
        )
        build_hybrid_checkpoint(
            base_path=base_checkpoint,
            guide_path=forward_checkpoint,
            reverse_guide_path=reverse_checkpoint,
            output_path=hybrid_checkpoint,
            local_blend=spec.local_blend,
            global_blend=spec.global_blend,
            window_scale=spec.window_scale,
            min_window_v=spec.min_window_v,
            guide_align_strength=spec.guide_align_strength,
            guide_align_window_scale=spec.guide_align_window_scale,
            guide_delta_clip_decades=spec.guide_delta_clip_decades,
            reverse_on_state_blend_scale=spec.reverse_on_state_blend_scale,
            reverse_on_state_delta_scale=spec.reverse_on_state_delta_scale,
            reverse_on_state_onset_u_scale=spec.reverse_on_state_onset_u_scale,
            reverse_on_state_window_scale=spec.reverse_on_state_window_scale,
        )
        engine = ResidualEngine(hybrid_checkpoint)
        info = engine.info().model_dump(mode="json")
        results.append(
            {
                "name": spec.name,
                "description": spec.description,
                "checkpoint_path": str(hybrid_checkpoint),
                "result": info,
                "forward_training": forward_result.model_dump(mode="json"),
                "reverse_training": reverse_result.model_dump(mode="json"),
                "forward_config": asdict(spec.forward_config),
                "reverse_config": asdict(spec.reverse_config),
                "hybrid_params": {
                    "local_blend": spec.local_blend,
                    "global_blend": spec.global_blend,
                    "window_scale": spec.window_scale,
                    "min_window_v": spec.min_window_v,
                    "guide_align_strength": spec.guide_align_strength,
                    "guide_align_window_scale": spec.guide_align_window_scale,
                    "guide_delta_clip_decades": spec.guide_delta_clip_decades,
                    "reverse_on_state_blend_scale": spec.reverse_on_state_blend_scale,
                    "reverse_on_state_delta_scale": spec.reverse_on_state_delta_scale,
                    "reverse_on_state_onset_u_scale": spec.reverse_on_state_onset_u_scale,
                    "reverse_on_state_window_scale": spec.reverse_on_state_window_scale,
                },
                "jump_metrics": _jump_metrics(engine, sampled_conditions),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    ranked = sorted(
        results,
        key=lambda item: (
            float(item["jump_metrics"]["jump_p95_decades"]),
            float(item["canonical_metrics"]["canonical_jump_max_decades"]),
            float(item["jump_metrics"]["generated_vth_mae_v"]),
            float(
                item["result"].get("validation_weighted_rmse_decades")
                or item["result"].get("validation_rmse_decades")
                or 999.0
            ),
        ),
    )
    lines = [
        "# Direction-split hybrid experiment",
        "",
        "| Rank | Model | Jump P95 | Canonical max | Canonical reverse max | Gen. Vth | Gen. SS | Weighted RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(ranked, start=1):
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
        result = item["result"]
        lines.append(
            f"| {rank} | {item['name']} | "
            f"{jump['jump_p95_decades']:.4f} | "
            f"{canonical['canonical_jump_max_decades']:.4f} | "
            f"{canonical['canonical_reverse_jump_max_decades']:.4f} | "
            f"{jump['generated_vth_mae_v']:.3f} | "
            f"{jump['generated_ss_mae_mv_dec']:.1f} | "
            f"{(result.get('validation_weighted_rmse_decades') or result.get('validation_rmse_decades') or float('nan')):.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate direction-split hybrid guide experiments."
    )
    parser.add_argument("--source-export", type=Path, default=EXPORT_SOURCE)
    parser.add_argument("--base-checkpoint", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument(
        "--spec",
        dest="spec_names",
        action="append",
        help="Optional experiment spec name to run; repeat for multiple specs.",
    )
    args = parser.parse_args()
    print(
        run(
            source_export=args.source_export.expanduser().resolve(),
            base_checkpoint=args.base_checkpoint.expanduser().resolve(),
            spec_names=set(args.spec_names) if args.spec_names else None,
        )
    )
