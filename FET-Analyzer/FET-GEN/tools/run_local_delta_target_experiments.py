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
        method="conditional_pca",
        pca_components=components,
        beta=0.02,
        validation_fraction=0.12,
        max_curves=None,
        feature_eval_limit=256,
        gate_loss_weight=0.6,
        rare_curve_weight=1.8,
        seed=seed,
    )


def _specs() -> list[Spec]:
    return [
        Spec(
            name="localdelta_full16",
            description="Train a conditioned PCA guide on the base-relative local threshold correction export, then add it back as a local delta.",
            split_guides=False,
            full_config=_config(16, seed=22345),
        ),
        Spec(
            name="localdelta_full20",
            description="Same local delta target, with a slightly larger conditioned PCA latent basis.",
            split_guides=False,
            full_config=_config(20, seed=22446),
        ),
        Spec(
            name="localdelta_dirsplit16",
            description="Direction-specific conditioned PCA guides trained on base-relative local threshold corrections.",
            split_guides=True,
            forward_config=_config(16, seed=22547),
            reverse_config=_config(16, seed=22648),
        ),
    ]


def run() -> Path:
    output_dir = ROOT / "experiments" / f"local-delta-target-{time.strftime('%Y%m%d-%H%M%S')}"
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
    results = []
    active_engine = ResidualEngine(ACTIVE)
    results.append(
        {
            "name": "active_hybrid_onstate_retuned",
            "description": "Current production hybrid checkpoint with the on-state onset limiter.",
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
            hybrid_checkpoint = output_dir / f"{spec.name}-hybrid.npz"
            build_hybrid_checkpoint(
                base_path=BASE,
                guide_path=forward_checkpoint,
                reverse_guide_path=reverse_checkpoint,
                guide_as_local_delta=True,
                output_path=hybrid_checkpoint,
                local_blend=0.94,
                global_blend=0.0,
                window_scale=3.0,
                min_window_v=0.22,
                guide_delta_clip_decades=0.12,
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
        hybrid_checkpoint = output_dir / f"{spec.name}-hybrid.npz"
        build_hybrid_checkpoint(
            base_path=BASE,
            guide_path=guide_checkpoint,
            guide_as_local_delta=True,
            output_path=hybrid_checkpoint,
            local_blend=0.94,
            global_blend=0.0,
            window_scale=3.0,
            min_window_v=0.22,
            guide_delta_clip_decades=0.12,
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
            float(item["canonical_metrics"]["canonical_jump_max_decades"]),
            float(item["jump_metrics"]["generated_vth_mae_v"]),
        ),
    )
    lines = [
        "# Local delta target experiment",
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
