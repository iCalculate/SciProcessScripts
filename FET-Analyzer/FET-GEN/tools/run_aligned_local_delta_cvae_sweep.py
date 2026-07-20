from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from devicecurvegen.neural import NeuralTrainingConfig, train_neural_checkpoint
from devicecurvegen.residual import ResidualEngine
from tools.evaluate_postprocess_rechecks import _canonical_metrics
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"
BASELINE_CVAE = (
    ROOT
    / "experiments"
    / "postfix-model-sweep-20260629-114505"
    / "subset_cvae_jumpfocus12.npz"
)
DATASET = (
    ROOT
    / "experiments"
    / "local-delta-aligned-guide-20260630-111556"
    / "dataset-local-delta-full"
)


@dataclass
class SweepSpec:
    name: str
    description: str
    config: NeuralTrainingConfig


def _cfg(
    latent_dim: int,
    hidden_dim: int,
    *,
    epochs: int,
    learning_rate: float,
    beta: float,
    low_current_weight: float,
    subthreshold_weight: float,
    slope_weight: float,
    gate_loss_weight: float,
    rare_curve_weight: float,
    seed: int,
) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        method="aligned_local_delta_cvae",
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=256,
        learning_rate=learning_rate,
        beta=beta,
        validation_fraction=0.12,
        patience=4,
        seed=seed,
        max_curves=None,
        low_current_weight=low_current_weight,
        subthreshold_weight=subthreshold_weight,
        slope_weight=slope_weight,
        gate_loss_weight=gate_loss_weight,
        rare_curve_weight=rare_curve_weight,
        feature_eval_limit=256,
    )


def _specs() -> list[SweepSpec]:
    return [
        SweepSpec(
            name="aligned_delta_cvae_l12_h96_balanced",
            description="Direct CVAE baseline in aligned local-delta residual space with moderate threshold weighting.",
            config=_cfg(
                12,
                96,
                epochs=8,
                learning_rate=1e-3,
                beta=0.0045,
                low_current_weight=1.7,
                subthreshold_weight=3.0,
                slope_weight=0.16,
                gate_loss_weight=0.6,
                rare_curve_weight=1.6,
                seed=41011,
            ),
        ),
        SweepSpec(
            name="aligned_delta_cvae_l16_h128_slopefocus",
            description="Higher-capacity aligned delta CVAE with stronger local slope pressure around Vth.",
            config=_cfg(
                16,
                128,
                epochs=10,
                learning_rate=8e-4,
                beta=0.0030,
                low_current_weight=1.9,
                subthreshold_weight=3.6,
                slope_weight=0.28,
                gate_loss_weight=0.7,
                rare_curve_weight=1.9,
                seed=41121,
            ),
        ),
        SweepSpec(
            name="aligned_delta_cvae_l20_h160_lowbeta",
            description="Lower-beta aligned delta CVAE that lets the decoder spend more capacity on threshold-local morphology.",
            config=_cfg(
                20,
                160,
                epochs=12,
                learning_rate=6e-4,
                beta=0.0020,
                low_current_weight=2.0,
                subthreshold_weight=3.8,
                slope_weight=0.36,
                gate_loss_weight=0.75,
                rare_curve_weight=2.1,
                seed=41231,
            ),
        ),
        SweepSpec(
            name="aligned_delta_cvae_l24_h192_rarefocus",
            description="Largest aligned delta CVAE in this sweep, emphasizing rare-curve coverage without pushing slope loss as hard.",
            config=_cfg(
                24,
                192,
                epochs=12,
                learning_rate=5e-4,
                beta=0.0025,
                low_current_weight=2.2,
                subthreshold_weight=3.4,
                slope_weight=0.22,
                gate_loss_weight=0.8,
                rare_curve_weight=2.5,
                seed=41341,
            ),
        ),
    ]


def run() -> Path:
    output_dir = ROOT / "experiments" / f"aligned-local-delta-cvae-sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_conditions(limit=72)
    results: list[dict[str, object]] = []

    for name, description, checkpoint in [
        (
            "active_hybrid_reverse_local_vth_align115_w25",
            "Current production hybrid checkpoint with reverse-only local-window post Vth alignment.",
            ACTIVE,
        ),
        (
            "baseline_cvae_jumpfocus12",
            "Best earlier CVAE checkpoint before the aligned local-delta residual-space branch.",
            BASELINE_CVAE,
        ),
    ]:
        engine = ResidualEngine(checkpoint)
        results.append(
            {
                "name": name,
                "description": description,
                "checkpoint_path": str(checkpoint),
                "result": engine.info().model_dump(mode="json"),
                "jump_metrics": _jump_metrics(engine, rows),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )

    for spec in _specs():
        checkpoint = output_dir / f"{spec.name}.npz"
        training_result = train_neural_checkpoint(
            checkpoint,
            dataset_path=DATASET,
            config=spec.config,
        )
        engine = ResidualEngine(checkpoint)
        results.append(
            {
                "name": spec.name,
                "description": spec.description,
                "checkpoint_path": str(checkpoint),
                "result": engine.info().model_dump(mode="json"),
                "training": training_result.model_dump(mode="json"),
                "config": asdict(spec.config),
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
        "# Threshold-aligned local-delta CVAE sweep",
        "",
        "| Rank | Model | Jump P95 | Canonical max | Canonical reverse max | Gen. Vth | Gen. SS | Weighted RMSE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(ranked, start=1):
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
        result = item["result"]
        weighted_rmse = result.get("validation_weighted_rmse_decades") or result.get(
            "validation_rmse_decades"
        )
        lines.append(
            f"| {rank} | {item['name']} | "
            f"{jump['jump_p95_decades']:.4f} | "
            f"{canonical['canonical_jump_max_decades']:.4f} | "
            f"{canonical['canonical_reverse_jump_max_decades']:.4f} | "
            f"{jump['generated_vth_mae_v']:.3f} | "
            f"{jump['generated_ss_mae_mv_dec']:.1f} | "
            f"{float(weighted_rmse):.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
