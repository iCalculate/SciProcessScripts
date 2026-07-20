from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from tools.run_model_experiments import _jump_metrics, _sample_conditions
from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition


ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    {
        "name": "active_hybrid_onstate_retuned",
        "description": "Current production hybrid checkpoint with the added on-state onset step limiter in the physics-side postprocessor.",
        "checkpoint": ROOT / "models" / "residual-hybrid-threshold-pca.npz",
    },
    {
        "name": "threshold_conditional_pca16_postprocess_recheck",
        "description": "Best canonical threshold-focused conditioned PCA re-evaluated under the current postprocessor.",
        "checkpoint": ROOT
        / "experiments"
        / "postfix-model-sweep-20260629-114505"
        / "subset_threshold_conditional_pca16_focus10.npz",
    },
    {
        "name": "conditional_pca20_postprocess_recheck",
        "description": "Best general conditioned PCA re-evaluated under the current postprocessor.",
        "checkpoint": ROOT
        / "experiments"
        / "postfix-model-sweep-20260629-114505"
        / "subset_conditional_pca20_ridge02.npz",
    },
    {
        "name": "cvae_jumpfocus12_postprocess_recheck",
        "description": "Best CVAE jump-focused checkpoint re-evaluated under the current postprocessor.",
        "checkpoint": ROOT
        / "experiments"
        / "postfix-model-sweep-20260629-114505"
        / "subset_cvae_jumpfocus12.npz",
    },
]


def _canonical_condition(gate_on: bool) -> GenerationCondition:
    return GenerationCondition(
        target_ion=1e-5,
        target_ioff=1e-15,
        target_vth=0.0,
        target_ss_mv_dec=230.0,
        hysteresis_v=1.5,
        ai_residual_strength=1.0,
        gate_ai_residual_strength=1.0 if gate_on else 0.0,
        diversity=0.65,
        seed=12345,
        points=601,
        variants=1,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        output_noise_gain=0.0,
        gate_leakage_a=1e-15,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
    )


def _canonical_metrics(engine: ResidualEngine) -> dict[str, float]:
    condition = _canonical_condition("Ig" in engine.info().generated_channels)
    candidate = generate_curves(condition, engine).candidates[0]
    forward = np.clip(
        np.asarray(candidate.forward_current, dtype=float),
        np.finfo(float).tiny,
        None,
    )
    reverse = np.clip(
        np.asarray(candidate.reverse_current, dtype=float),
        np.finfo(float).tiny,
        None,
    )
    forward_delta = np.abs(np.diff(np.log10(forward)))
    reverse_delta = np.abs(np.diff(np.log10(reverse)))
    combined = np.concatenate([forward_delta, reverse_delta])
    return {
        "canonical_jump_p95_decades": float(np.percentile(combined, 95)),
        "canonical_jump_max_decades": float(np.max(combined)),
        "canonical_forward_jump_max_decades": float(np.max(forward_delta)),
        "canonical_reverse_jump_max_decades": float(np.max(reverse_delta)),
    }


def run() -> Path:
    output_dir = ROOT / "experiments" / f"postprocess-onstate-retune-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_conditions(limit=72)
    results: list[dict[str, object]] = []
    for item in MODELS:
        engine = ResidualEngine(item["checkpoint"])
        info = engine.info().model_dump(mode="json")
        results.append(
            {
                "name": item["name"],
                "description": item["description"],
                "method": info.get("architecture") or info.get("residual_mode") or "unknown",
                "checkpoint_path": str(item["checkpoint"]),
                "result": info,
                "jump_metrics": _jump_metrics(engine, rows),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )

    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Postprocess on-state retune recheck",
        "",
        "| Model | Method | Weighted RMSE | Jump P95 | Spike rate | Canonical jump max | Gen. Vth MAE | Gen. SS MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        result = item["result"]
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
        assert isinstance(result, dict)
        assert isinstance(jump, dict)
        assert isinstance(canonical, dict)
        lines.append(
            f"| {item['name']} | "
            f"{result.get('architecture') or result.get('residual_mode')} | "
            f"{(result.get('validation_weighted_rmse_decades') or result.get('validation_rmse_decades') or float('nan')):.4f} | "
            f"{jump['jump_p95_decades']:.4f} | "
            f"{jump['jump_spike_rate']:.4f} | "
            f"{canonical['canonical_jump_max_decades']:.4f} | "
            f"{jump['generated_vth_mae_v']:.3f} | "
            f"{jump['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
