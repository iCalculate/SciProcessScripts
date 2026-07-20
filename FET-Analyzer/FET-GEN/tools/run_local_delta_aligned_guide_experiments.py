from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from devicecurvegen.neural import NeuralTrainingConfig, train_neural_checkpoint
from devicecurvegen.residual import ResidualEngine
from tools.build_export_subset import build_export_subset
from tools.build_hybrid_checkpoint import build_hybrid_checkpoint
from tools.derive_local_delta_dataset import derive_local_delta_dataset
from tools.evaluate_postprocess_rechecks import _canonical_metrics
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]
FULL_EXPORT = ROOT / "experiments" / "db-export-forward-reverse-full-20260629-1453"
BASE = ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"


@dataclass
class Spec:
    name: str
    description: str
    split_guides: bool
    full_config: NeuralTrainingConfig | None = None
    forward_config: NeuralTrainingConfig | None = None
    reverse_config: NeuralTrainingConfig | None = None


def _config(components: int, *, seed: int) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        method="aligned_local_threshold_conditional_pca",
        pca_components=components,
        beta=0.02,
        validation_fraction=0.12,
        max_curves=None,
        feature_eval_limit=256,
        gate_loss_weight=0.6,
        rare_curve_weight=1.8,
        subthreshold_weight=3.0,
        slope_weight=0.35,
        seed=seed,
    )


def _specs() -> list[Spec]:
    return [
        Spec(
            name="aligned_localdelta_full16",
            description="Single aligned local-threshold guide trained on the base-relative local delta export.",
            split_guides=False,
            full_config=_config(16, seed=23345),
        ),
        Spec(
            name="aligned_localdelta_dirsplit16",
            description="Direction-specific aligned local-threshold guides trained on base-relative local threshold corrections.",
            split_guides=True,
            forward_config=_config(16, seed=23446),
            reverse_config=_config(16, seed=23547),
        ),
        Spec(
            name="aligned_localdelta_dirsplit20",
            description="Higher-capacity direction-specific aligned local-threshold guides.",
            split_guides=True,
            forward_config=_config(20, seed=23648),
            reverse_config=_config(20, seed=23749),
        ),
    ]


def _build_current_best_hybrid(
    output_path: Path,
    *,
    guide_path: Path,
    reverse_guide_path: Path | None = None,
) -> Path:
    build_hybrid_checkpoint(
        base_path=BASE,
        guide_path=guide_path,
        reverse_guide_path=reverse_guide_path,
        guide_as_local_delta=True,
        output_path=output_path,
        base_scale_multiplier=0.25,
        local_blend=1.0,
        global_blend=0.04,
        window_scale=3.0,
        min_window_v=0.22,
        guide_delta_clip_decades=0.08,
        post_vth_align_strength=1.15,
        post_vth_align_reverse_only=True,
        post_vth_align_local_window_scale=2.5,
        post_vth_align_local_min_window_v=0.18,
    )
    return output_path


def run() -> Path:
    output_dir = ROOT / "experiments" / f"local-delta-aligned-guide-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_full = derive_local_delta_dataset(
        FULL_EXPORT,
        BASE,
        output_dir / "dataset-local-delta-full",
        window_scale=1.8,
        min_window_v=0.12,
        floor=0.03,
    )
    derived_forward = build_export_subset(
        derived_full,
        output_dir / "dataset-local-delta-forward",
        directions=("forward",),
    )
    derived_reverse = build_export_subset(
        derived_full,
        output_dir / "dataset-local-delta-reverse",
        directions=("reverse",),
    )

    sampled_conditions = _sample_conditions(limit=72)
    results: list[dict[str, object]] = []
    active_engine = ResidualEngine(ACTIVE)
    results.append(
        {
            "name": "active_hybrid_reverse_local_vth_align115_w25",
            "description": "Current production hybrid checkpoint with reverse-only local-window post Vth alignment.",
            "checkpoint_path": str(ACTIVE),
            "result": active_engine.info().model_dump(mode="json"),
            "jump_metrics": _jump_metrics(active_engine, sampled_conditions),
            "canonical_metrics": _canonical_metrics(active_engine),
        }
    )

    for spec in _specs():
        if spec.split_guides:
            assert spec.forward_config is not None and spec.reverse_config is not None
            forward_checkpoint = output_dir / f"{spec.name}-forward-guide.npz"
            reverse_checkpoint = output_dir / f"{spec.name}-reverse-guide.npz"
            forward_result = train_neural_checkpoint(
                forward_checkpoint,
                dataset_path=derived_forward,
                config=spec.forward_config,
            )
            reverse_result = train_neural_checkpoint(
                reverse_checkpoint,
                dataset_path=derived_reverse,
                config=spec.reverse_config,
            )
            hybrid_checkpoint = _build_current_best_hybrid(
                output_dir / f"{spec.name}-hybrid.npz",
                guide_path=forward_checkpoint,
                reverse_guide_path=reverse_checkpoint,
            )
            engine = ResidualEngine(hybrid_checkpoint)
            results.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "checkpoint_path": str(hybrid_checkpoint),
                    "result": engine.info().model_dump(mode="json"),
                    "forward_training": forward_result.model_dump(mode="json"),
                    "reverse_training": reverse_result.model_dump(mode="json"),
                    "forward_config": asdict(spec.forward_config),
                    "reverse_config": asdict(spec.reverse_config),
                    "jump_metrics": _jump_metrics(engine, sampled_conditions),
                    "canonical_metrics": _canonical_metrics(engine),
                }
            )
            continue

        assert spec.full_config is not None
        guide_checkpoint = output_dir / f"{spec.name}-guide.npz"
        guide_result = train_neural_checkpoint(
            guide_checkpoint,
            dataset_path=derived_full,
            config=spec.full_config,
        )
        hybrid_checkpoint = _build_current_best_hybrid(
            output_dir / f"{spec.name}-hybrid.npz",
            guide_path=guide_checkpoint,
        )
        engine = ResidualEngine(hybrid_checkpoint)
        results.append(
            {
                "name": spec.name,
                "description": spec.description,
                "checkpoint_path": str(hybrid_checkpoint),
                "guide_checkpoint_path": str(guide_checkpoint),
                "result": engine.info().model_dump(mode="json"),
                "guide_result": guide_result.model_dump(mode="json"),
                "guide_config": asdict(spec.full_config),
                "jump_metrics": _jump_metrics(engine, sampled_conditions),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["jump_metrics"]["jump_p95_decades"]),
            float(item["jump_metrics"]["generated_vth_mae_v"]),
            float(item["canonical_metrics"]["canonical_jump_max_decades"]),
            float(item["jump_metrics"]["generated_ss_mae_mv_dec"]),
        ),
    )
    lines = [
        "# Local delta aligned-guide experiments",
        "",
        "| Rank | Model | Jump P95 | Canonical max | Canonical reverse max | Gen. Vth | Gen. SS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(ranked, start=1):
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
        lines.append(
            f"| {rank} | {item['name']} | "
            f"{jump['jump_p95_decades']:.4f} | "
            f"{canonical['canonical_jump_max_decades']:.4f} | "
            f"{canonical['canonical_reverse_jump_max_decades']:.4f} | "
            f"{jump['generated_vth_mae_v']:.3f} | "
            f"{jump['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
