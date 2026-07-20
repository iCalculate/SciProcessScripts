from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from devicecurvegen import physics
from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]


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
    gate_on = "Ig" in engine.info().generated_channels
    condition = _canonical_condition(gate_on)
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


def _parameter_grid() -> list[dict[str, float]]:
    grid: list[dict[str, float]] = []
    for onset_log_offset, onset_u_multiplier, slope_base, slope_ai in itertools.product(
        [3.8, 4.0, 4.2],
        [1.6, 1.8, 2.0],
        [0.90, 0.95, 1.00],
        [0.10, 0.15, 0.20],
    ):
        grid.append(
            {
                "onset_log_offset_decades": onset_log_offset,
                "onset_u_multiplier": onset_u_multiplier,
                "slope_multiplier_base": slope_base,
                "slope_multiplier_ai": slope_ai,
            }
        )
    return grid


def _evaluate_candidate(
    engine: ResidualEngine,
    params: dict[str, float],
    sampled_conditions: list[dict],
) -> dict[str, float]:
    physics.ON_STATE_STEP_ONSET_LOG_OFFSET_DECADES = params["onset_log_offset_decades"]
    physics.ON_STATE_STEP_ONSET_U_MULTIPLIER = params["onset_u_multiplier"]
    physics.ON_STATE_STEP_SLOPE_MULTIPLIER_BASE = params["slope_multiplier_base"]
    physics.ON_STATE_STEP_SLOPE_MULTIPLIER_AI = params["slope_multiplier_ai"]

    jump_metrics = _jump_metrics(engine, sampled_conditions)
    canonical_metrics = _canonical_metrics(engine)
    return {
        **params,
        **jump_metrics,
        **canonical_metrics,
    }


def _coarse_score(item: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        item["jump_p95_decades"],
        item["canonical_jump_max_decades"],
        item["generated_vth_mae_v"],
        item["generated_ss_mae_mv_dec"],
    )


def run() -> Path:
    output_dir = ROOT / "experiments" / f"onstate-postprocessor-sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = ResidualEngine()
    coarse_conditions = _sample_conditions(limit=24)
    full_conditions = _sample_conditions(limit=48)
    original = {
        "onset_log_offset_decades": physics.ON_STATE_STEP_ONSET_LOG_OFFSET_DECADES,
        "onset_u_multiplier": physics.ON_STATE_STEP_ONSET_U_MULTIPLIER,
        "slope_multiplier_base": physics.ON_STATE_STEP_SLOPE_MULTIPLIER_BASE,
        "slope_multiplier_ai": physics.ON_STATE_STEP_SLOPE_MULTIPLIER_AI,
    }

    try:
        coarse_results = [
            _evaluate_candidate(engine, params, coarse_conditions)
            for params in _parameter_grid()
        ]
        coarse_results.sort(key=_coarse_score)
        shortlist = coarse_results[:8]
        full_results = [
            _evaluate_candidate(
                engine,
                {
                    "onset_log_offset_decades": item["onset_log_offset_decades"],
                    "onset_u_multiplier": item["onset_u_multiplier"],
                    "slope_multiplier_base": item["slope_multiplier_base"],
                    "slope_multiplier_ai": item["slope_multiplier_ai"],
                },
                full_conditions,
            )
            for item in shortlist
        ]
        full_results.sort(key=_coarse_score)
    finally:
        physics.ON_STATE_STEP_ONSET_LOG_OFFSET_DECADES = original["onset_log_offset_decades"]
        physics.ON_STATE_STEP_ONSET_U_MULTIPLIER = original["onset_u_multiplier"]
        physics.ON_STATE_STEP_SLOPE_MULTIPLIER_BASE = original["slope_multiplier_base"]
        physics.ON_STATE_STEP_SLOPE_MULTIPLIER_AI = original["slope_multiplier_ai"]

    payload = {
        "current": original,
        "coarse_top": coarse_results[:20],
        "full_top": full_results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# On-state postprocessor sweep",
        "",
        "## Current",
        "",
        json.dumps(original, ensure_ascii=False),
        "",
        "## Full recheck top candidates",
        "",
        "| Rank | onset log | onset u | slope base | slope ai | Jump P95 | Canonical max | Gen. Vth | Gen. SS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(full_results, start=1):
        lines.append(
            f"| {index} | "
            f"{item['onset_log_offset_decades']:.2f} | "
            f"{item['onset_u_multiplier']:.2f} | "
            f"{item['slope_multiplier_base']:.2f} | "
            f"{item['slope_multiplier_ai']:.2f} | "
            f"{item['jump_p95_decades']:.4f} | "
            f"{item['canonical_jump_max_decades']:.4f} | "
            f"{item['generated_vth_mae_v']:.3f} | "
            f"{item['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    directory = run()
    print(directory)
