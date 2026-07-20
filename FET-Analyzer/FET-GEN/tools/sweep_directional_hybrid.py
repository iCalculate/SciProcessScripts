from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition
from tools.build_hybrid_checkpoint import build_hybrid_checkpoint
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
)
GUIDE = (
    ROOT
    / "experiments"
    / "conditional-pca-component-sweep-20260627-224618"
    / "attempt_8_conditional_pca12_clipped.npz"
)


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


def _score(item: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        item["jump_p95_decades"],
        item["canonical_jump_max_decades"],
        item["generated_vth_mae_v"],
        item["generated_ss_mae_mv_dec"],
    )


def _parameter_grid() -> list[dict[str, float]]:
    grid = []
    for blend_scale, delta_scale, onset_u_scale, window_scale in itertools.product(
        [0.65, 0.80, 0.90],
        [0.55, 0.70, 0.85],
        [1.4, 1.8],
        [0.9, 1.2],
    ):
        grid.append(
            {
                "reverse_on_state_blend_scale": blend_scale,
                "reverse_on_state_delta_scale": delta_scale,
                "reverse_on_state_onset_u_scale": onset_u_scale,
                "reverse_on_state_window_scale": window_scale,
            }
        )
    return grid


def _evaluate(
    output_dir: Path,
    params: dict[str, float],
    sampled_conditions: list[dict],
) -> dict[str, float]:
    checkpoint = output_dir / (
        "dirhyb"
        f"_rb{int(round(params['reverse_on_state_blend_scale'] * 100)):02d}"
        f"_rd{int(round(params['reverse_on_state_delta_scale'] * 100)):02d}"
        f"_ru{int(round(params['reverse_on_state_onset_u_scale'] * 10)):02d}"
        f"_rw{int(round(params['reverse_on_state_window_scale'] * 10)):02d}.npz"
    )
    build_hybrid_checkpoint(
        base_path=BASE,
        guide_path=GUIDE,
        output_path=checkpoint,
        local_blend=0.82,
        global_blend=0.06,
        window_scale=3.0,
        min_window_v=0.22,
        guide_align_strength=0.0,
        guide_align_window_scale=2.0,
        guide_delta_clip_decades=0.0,
        **params,
    )
    engine = ResidualEngine(checkpoint)
    return {
        **params,
        **_jump_metrics(engine, sampled_conditions),
        **_canonical_metrics(engine),
    }


def run() -> Path:
    output_dir = ROOT / "experiments" / f"directional-hybrid-sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_conditions = _sample_conditions(limit=24)
    full_conditions = _sample_conditions(limit=72)
    coarse = [_evaluate(output_dir, params, coarse_conditions) for params in _parameter_grid()]
    coarse.sort(key=_score)
    shortlist = coarse[:8]
    full = [
        _evaluate(
            output_dir,
            {
                "reverse_on_state_blend_scale": item["reverse_on_state_blend_scale"],
                "reverse_on_state_delta_scale": item["reverse_on_state_delta_scale"],
                "reverse_on_state_onset_u_scale": item["reverse_on_state_onset_u_scale"],
                "reverse_on_state_window_scale": item["reverse_on_state_window_scale"],
            },
            full_conditions,
        )
        for item in shortlist
    ]
    full.sort(key=_score)
    payload = {"coarse_top": coarse[:16], "full_top": full}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Directional hybrid sweep",
        "",
        "| Rank | rev blend | rev delta | onset u | window | Jump P95 | Canonical max | Gen. Vth | Gen. SS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(full, start=1):
        lines.append(
            f"| {index} | "
            f"{item['reverse_on_state_blend_scale']:.2f} | "
            f"{item['reverse_on_state_delta_scale']:.2f} | "
            f"{item['reverse_on_state_onset_u_scale']:.2f} | "
            f"{item['reverse_on_state_window_scale']:.2f} | "
            f"{item['jump_p95_decades']:.4f} | "
            f"{item['canonical_jump_max_decades']:.4f} | "
            f"{item['generated_vth_mae_v']:.3f} | "
            f"{item['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
