from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from devicecurvegen.neural import NeuralTrainingConfig, train_neural_checkpoint
from devicecurvegen.residual import ResidualEngine
from tools.build_hybrid_checkpoint import build_hybrid_checkpoint
from tools.evaluate_postprocess_rechecks import _canonical_metrics
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"
DATASET_ROOT = ROOT / "experiments" / "local-delta-aligned-guide-20260630-111556"
FORWARD_DATASET = DATASET_ROOT / "dataset-local-delta-forward"
REVERSE_DATASET = DATASET_ROOT / "dataset-local-delta-reverse"


@dataclass
class SweepSpec:
    name: str
    description: str
    forward_config: NeuralTrainingConfig
    reverse_config: NeuralTrainingConfig


def _cfg(
    components: int,
    *,
    beta: float,
    subthreshold_weight: float,
    slope_weight: float,
    seed: int,
) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        method="aligned_local_threshold_conditional_pca",
        pca_components=components,
        beta=beta,
        validation_fraction=0.12,
        max_curves=None,
        feature_eval_limit=256,
        gate_loss_weight=0.6,
        rare_curve_weight=1.8,
        subthreshold_weight=subthreshold_weight,
        slope_weight=slope_weight,
        seed=seed,
    )


def _specs() -> list[SweepSpec]:
    return [
        SweepSpec(
            name="aligned_dirsplit24_balanced",
            description="Higher-capacity aligned guides with balanced threshold emphasis.",
            forward_config=_cfg(24, beta=0.018, subthreshold_weight=3.1, slope_weight=0.35, seed=24051),
            reverse_config=_cfg(24, beta=0.018, subthreshold_weight=3.1, slope_weight=0.35, seed=24052),
        ),
        SweepSpec(
            name="aligned_dirsplit24_slopefocus",
            description="Higher-capacity aligned guides with stronger slope emphasis near threshold.",
            forward_config=_cfg(24, beta=0.018, subthreshold_weight=3.0, slope_weight=0.45, seed=24161),
            reverse_config=_cfg(24, beta=0.018, subthreshold_weight=3.0, slope_weight=0.45, seed=24162),
        ),
        SweepSpec(
            name="aligned_dirsplit24_lowbeta",
            description="Higher-capacity aligned guides with a lighter latent regularizer.",
            forward_config=_cfg(24, beta=0.012, subthreshold_weight=3.2, slope_weight=0.40, seed=24271),
            reverse_config=_cfg(24, beta=0.012, subthreshold_weight=3.2, slope_weight=0.40, seed=24272),
        ),
        SweepSpec(
            name="aligned_dirsplit28_lowbeta",
            description="Largest aligned guide in this sweep, pairing extra capacity with a lighter beta.",
            forward_config=_cfg(28, beta=0.012, subthreshold_weight=3.1, slope_weight=0.40, seed=24381),
            reverse_config=_cfg(28, beta=0.012, subthreshold_weight=3.1, slope_weight=0.40, seed=24382),
        ),
        SweepSpec(
            name="aligned_dirsplit20_reversefocus",
            description="Keep forward guide moderate and push more threshold emphasis into the reverse guide.",
            forward_config=_cfg(20, beta=0.018, subthreshold_weight=2.8, slope_weight=0.30, seed=24491),
            reverse_config=_cfg(24, beta=0.015, subthreshold_weight=3.5, slope_weight=0.45, seed=24492),
        ),
    ]


def _build_hybrid(output_path: Path, *, guide_path: Path, reverse_guide_path: Path) -> Path:
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
    output_dir = ROOT / "experiments" / f"local-delta-aligned-training-sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_conditions(limit=72)
    results: list[dict[str, object]] = []

    active_engine = ResidualEngine(ACTIVE)
    results.append(
        {
            "name": "active_hybrid_reverse_local_vth_align115_w25",
            "description": "Current production hybrid checkpoint with reverse-only local-window post Vth alignment.",
            "checkpoint_path": str(ACTIVE),
            "result": active_engine.info().model_dump(mode="json"),
            "jump_metrics": _jump_metrics(active_engine, rows),
            "canonical_metrics": _canonical_metrics(active_engine),
        }
    )

    for spec in _specs():
        forward_checkpoint = output_dir / f"{spec.name}-forward-guide.npz"
        reverse_checkpoint = output_dir / f"{spec.name}-reverse-guide.npz"
        forward_result = train_neural_checkpoint(
            forward_checkpoint,
            dataset_path=FORWARD_DATASET,
            config=spec.forward_config,
        )
        reverse_result = train_neural_checkpoint(
            reverse_checkpoint,
            dataset_path=REVERSE_DATASET,
            config=spec.reverse_config,
        )
        hybrid_checkpoint = _build_hybrid(
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
                "jump_metrics": _jump_metrics(engine, rows),
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
            float(item["jump_metrics"]["generated_ss_mae_mv_dec"]),
        ),
    )
    lines = [
        "# Local delta aligned-guide training sweep",
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
