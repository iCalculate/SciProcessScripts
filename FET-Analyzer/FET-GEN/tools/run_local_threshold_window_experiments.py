from __future__ import annotations

import json
import time
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
FULL_EXPORT = ROOT / "experiments" / "db-export-forward-reverse-full-20260629-1453"
BASE_CHECKPOINT = (
    ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
)
ACTIVE_CHECKPOINT = ROOT / "models" / "residual-hybrid-threshold-pca.npz"


@dataclass
class ExperimentSpec:
    name: str
    description: str
    split_guides: bool
    full_config: NeuralTrainingConfig | None = None
    forward_config: NeuralTrainingConfig | None = None
    reverse_config: NeuralTrainingConfig | None = None
    local_blend: float = 0.82
    global_blend: float = 0.06
    window_scale: float = 3.0
    min_window_v: float = 0.22
    guide_align_strength: float = 0.0
    guide_align_window_scale: float = 2.0
    guide_delta_clip_decades: float = 0.0


def _local_config(components: int, *, seed: int) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        method="local_threshold_conditional_pca",
        pca_components=components,
        beta=0.012,
        validation_fraction=0.12,
        max_curves=None,
        feature_eval_limit=256,
        gate_loss_weight=0.6,
        rare_curve_weight=1.8,
        subthreshold_weight=3.2,
        slope_weight=0.35,
        seed=seed,
    )


def _specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="localwin_full16_hybrid",
            description="Single full-database local-threshold conditioned PCA guide with a stable latent-PCA base.",
            split_guides=False,
            full_config=_local_config(16, seed=12345),
            guide_delta_clip_decades=0.18,
        ),
        ExperimentSpec(
            name="localwin_full20_hybrid",
            description="Higher-capacity single full-database local-threshold guide to test whether extra latent capacity helps Vth curvature without reintroducing the jump.",
            split_guides=False,
            full_config=_local_config(20, seed=12446),
            guide_delta_clip_decades=0.18,
        ),
        ExperimentSpec(
            name="localwin_dirsplit16_hybrid",
            description="Forward and reverse guides both use local-threshold conditioned PCA trained on their own direction subsets.",
            split_guides=True,
            forward_config=_local_config(16, seed=12547),
            reverse_config=_local_config(16, seed=12648),
            guide_delta_clip_decades=0.18,
        ),
    ]


def _baseline_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for name, description, checkpoint in [
        (
            "active_hybrid_onstate_retuned",
            "Current production hybrid checkpoint with the on-state onset limiter.",
            ACTIVE_CHECKPOINT,
        ),
    ]:
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


def run() -> Path:
    output_dir = OUTPUT_ROOT / f"local-threshold-window-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_conditions = _sample_conditions(limit=72)
    results = _baseline_rows(sampled_conditions)
    forward_dataset = build_export_subset(FULL_EXPORT, output_dir / "dataset-forward", directions=("forward",))
    reverse_dataset = build_export_subset(FULL_EXPORT, output_dir / "dataset-reverse", directions=("reverse",))

    for spec in _specs():
        if spec.split_guides:
            assert spec.forward_config is not None and spec.reverse_config is not None
            forward_checkpoint = output_dir / f"{spec.name}-forward-guide.npz"
            reverse_checkpoint = output_dir / f"{spec.name}-reverse-guide.npz"
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
            hybrid_checkpoint = output_dir / f"{spec.name}-hybrid.npz"
            build_hybrid_checkpoint(
                base_path=BASE_CHECKPOINT,
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
            dataset_path=FULL_EXPORT,
            config=spec.full_config,
        )
        hybrid_checkpoint = output_dir / f"{spec.name}-hybrid.npz"
        build_hybrid_checkpoint(
            base_path=BASE_CHECKPOINT,
            guide_path=guide_checkpoint,
            output_path=hybrid_checkpoint,
            local_blend=spec.local_blend,
            global_blend=spec.global_blend,
            window_scale=spec.window_scale,
            min_window_v=spec.min_window_v,
            guide_align_strength=spec.guide_align_strength,
            guide_align_window_scale=spec.guide_align_window_scale,
            guide_delta_clip_decades=spec.guide_delta_clip_decades,
        )
        guide_engine = ResidualEngine(guide_checkpoint)
        hybrid_engine = ResidualEngine(hybrid_checkpoint)
        results.append(
            {
                "name": spec.name,
                "description": spec.description,
                "checkpoint_path": str(hybrid_checkpoint),
                "guide_checkpoint_path": str(guide_checkpoint),
                "result": hybrid_engine.info().model_dump(mode="json"),
                "guide_result": guide_result.model_dump(mode="json"),
                "guide_config": asdict(spec.full_config),
                "guide_jump_metrics": _jump_metrics(guide_engine, sampled_conditions),
                "guide_canonical_metrics": _canonical_metrics(guide_engine),
                "jump_metrics": _jump_metrics(hybrid_engine, sampled_conditions),
                "canonical_metrics": _canonical_metrics(hybrid_engine),
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
        ),
    )
    lines = [
        "# Local threshold-window experiment",
        "",
        "| Rank | Model | Jump P95 | Canonical max | Canonical reverse max | Gen. Vth | Gen. SS | Weighted RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(ranked, start=1):
        result = item["result"]
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
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
    print(run())
