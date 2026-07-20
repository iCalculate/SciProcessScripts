from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sqlalchemy import select

from devicecurvegen.database import create_database_engine, curves
from devicecurvegen.neural import NeuralTrainingConfig, train_neural_checkpoint
from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiments"
DATABASE_URL = "mysql+pymysql://root@127.0.0.1:3307/devicecurvegen"


@dataclass
class ExperimentSpec:
    name: str
    description: str
    config: NeuralTrainingConfig


def _sample_conditions(limit: int = 72) -> list[dict[str, float | str | int | None]]:
    engine = create_database_engine(DATABASE_URL)
    query = (
        select(
            curves.c.curve_id,
            curves.c.direction,
            curves.c.voltage_min_v,
            curves.c.voltage_max_v,
            curves.c.ion,
            curves.c.ioff,
            curves.c.vth,
            curves.c.ss_mv_dec,
            curves.c.polarity,
            curves.c.hysteresis_v,
            curves.c.leakage_level,
            curves.c.has_gate_current,
        )
        .where(
            curves.c.vth.is_not(None),
            curves.c.ss_mv_dec.is_not(None),
            curves.c.ss_mv_dec >= 20.0,
            curves.c.ss_mv_dec <= 5000.0,
            curves.c.polarity.in_(["n-type", "p-type"]),
        )
        .order_by(curves.c.curve_id)
    )
    with engine.connect() as connection:
        rows = [dict(row) for row in connection.execute(query).mappings()]
    if len(rows) <= limit:
        return rows
    indices = np.unique(np.linspace(0, len(rows) - 1, limit, dtype=np.int64))
    return [rows[int(index)] for index in indices]


def _generation_condition(row: dict[str, float | str | int | None], *, gate_on: bool) -> GenerationCondition:
    gate_leakage = float(row["leakage_level"] or row["ioff"] or 1e-14)
    voltage_min = float(row["voltage_min_v"])
    voltage_max = float(row["voltage_max_v"])
    target_vth = float(np.clip(float(row["vth"]), voltage_min, voltage_max))
    return GenerationCondition(
        polarity=str(row["polarity"]),
        target_ion=float(row["ion"]),
        target_ioff=float(row["ioff"]),
        target_vth=target_vth,
        target_ss_mv_dec=float(np.clip(float(row["ss_mv_dec"]), 20.0, 5000.0)),
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        hysteresis_v=float(row["hysteresis_v"] or 0.0),
        gate_leakage_a=max(gate_leakage, 1e-15),
        ai_residual_strength=1.0,
        gate_ai_residual_strength=1.0 if gate_on else 0.0,
        physical_strictness=0.0,
        diversity=0.65,
        points=401,
        variants=1,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        output_noise_gain=0.0,
    )


def _jump_metrics(
    engine: ResidualEngine,
    sampled_conditions: list[dict[str, float | str | int | None]],
) -> dict[str, float]:
    jump_values: list[float] = []
    vth_errors: list[float] = []
    ss_errors: list[float] = []
    spike_count = 0
    total_curves = 0
    for index, row in enumerate(sampled_conditions):
        gate_on = bool(row["has_gate_current"]) and "Ig" in engine.info().generated_channels
        condition = _generation_condition(row, gate_on=gate_on).model_copy(
            update={"seed": 10_000 + index}
        )
        generated = generate_curves(condition, engine).candidates[0]
        voltage = np.asarray(generated.voltage, dtype=float)
        log_forward = np.log10(np.clip(np.asarray(generated.forward_current, dtype=float), np.finfo(float).tiny, None))
        target_vth = float(row["vth"])
        target_ss = float(row["ss_mv_dec"])
        window = max(0.25, 2.8 * target_ss / 1000.0)
        mask = np.abs(voltage - target_vth) <= window
        if np.count_nonzero(mask) >= 2:
            local_voltage = voltage[mask]
            local_log = log_forward[mask]
            deltas = np.abs(np.diff(local_log))
            if deltas.size > 0:
                jump_values.append(float(np.max(deltas)))
                spike_count += int(np.count_nonzero(deltas > 0.55))
                total_curves += int(deltas.size)
        extracted_vth = generated.features.vth
        plausible_vth_min = float(row["voltage_min_v"]) - 0.5 * (
            float(row["voltage_max_v"]) - float(row["voltage_min_v"])
        )
        plausible_vth_max = float(row["voltage_max_v"]) + 0.5 * (
            float(row["voltage_max_v"]) - float(row["voltage_min_v"])
        )
        if (
            extracted_vth is not None
            and np.isfinite(extracted_vth)
            and plausible_vth_min <= float(extracted_vth) <= plausible_vth_max
        ):
            vth_errors.append(abs(float(extracted_vth) - target_vth))
        extracted_ss = generated.features.ss_mv_dec
        if (
            extracted_ss is not None
            and np.isfinite(extracted_ss)
            and 20.0 <= float(extracted_ss) <= 10_000.0
        ):
            ss_errors.append(abs(float(extracted_ss) - target_ss))
    array = np.asarray(jump_values, dtype=float) if jump_values else np.asarray([0.0])
    return {
        "jump_p50_decades": float(np.percentile(array, 50)),
        "jump_p95_decades": float(np.percentile(array, 95)),
        "jump_max_decades": float(np.max(array)),
        "jump_spike_rate": float(spike_count / max(total_curves, 1)),
        "generated_vth_mae_v": float(np.mean(vth_errors)) if vth_errors else float("nan"),
        "generated_ss_mae_mv_dec": float(np.mean(ss_errors)) if ss_errors else float("nan"),
    }


def _specs() -> list[ExperimentSpec]:
    common = dict(
        seed=12345,
        validation_fraction=0.12,
        max_curves=8000,
        feature_eval_limit=256,
    )
    return [
        ExperimentSpec(
            name="attempt_1_pca16",
            description="Linear latent PCA baseline with database-balanced weighting.",
            config=NeuralTrainingConfig(
                method="latent_pca",
                pca_components=16,
                gate_loss_weight=0.6,
                rare_curve_weight=1.35,
                **common,
            ),
        ),
        ExperimentSpec(
            name="attempt_2_cvae_default",
            description="Current residual-skip CVAE with moderate weighting.",
            config=NeuralTrainingConfig(
                method="physics_cvae",
                latent_dim=12,
                hidden_dim=96,
                epochs=8,
                batch_size=256,
                learning_rate=1e-3,
                beta=0.005,
                low_current_weight=1.5,
                subthreshold_weight=2.5,
                slope_weight=0.10,
                gate_loss_weight=0.6,
                rare_curve_weight=1.35,
                **common,
            ),
        ),
        ExperimentSpec(
            name="attempt_3_cvae_slope_focus",
            description="CVAE with stronger subthreshold and slope supervision to suppress Vth jumps.",
            config=NeuralTrainingConfig(
                method="physics_cvae",
                latent_dim=16,
                hidden_dim=128,
                epochs=10,
                batch_size=256,
                learning_rate=8e-4,
                beta=0.003,
                low_current_weight=1.8,
                subthreshold_weight=3.2,
                slope_weight=0.22,
                gate_loss_weight=0.7,
                rare_curve_weight=1.8,
                **common,
            ),
        ),
        ExperimentSpec(
            name="attempt_4_cvae_rare_balance",
            description="Larger CVAE emphasizing rare shapes and threshold smoothness.",
            config=NeuralTrainingConfig(
                method="physics_cvae",
                latent_dim=20,
                hidden_dim=160,
                epochs=12,
                batch_size=256,
                learning_rate=6e-4,
                beta=0.002,
                low_current_weight=2.0,
                subthreshold_weight=3.8,
                slope_weight=0.30,
                gate_loss_weight=0.8,
                rare_curve_weight=2.4,
                **common,
            ),
        ),
        ExperimentSpec(
            name="attempt_5_conditional_pca16",
            description="Conditioned PCA predicts latent residual coefficients from the requested device features.",
            config=NeuralTrainingConfig(
                method="conditional_pca",
                pca_components=16,
                beta=0.008,
                gate_loss_weight=0.6,
                rare_curve_weight=1.35,
                **common,
            ),
        ),
        ExperimentSpec(
            name="attempt_6_conditional_pca24_balanced",
            description="Higher-capacity conditioned PCA with stronger rare-curve balancing and ridge regularization.",
            config=NeuralTrainingConfig(
                method="conditional_pca",
                pca_components=24,
                beta=0.015,
                gate_loss_weight=0.75,
                rare_curve_weight=1.8,
                **common,
            ),
        ),
    ]


def run() -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_ROOT / f"model-experiments-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_conditions = _sample_conditions()
    results: list[dict[str, object]] = []
    for spec in _specs():
        print(f"\n=== {spec.name} ===", flush=True)
        checkpoint = output_dir / f"{spec.name}.npz"
        started = time.perf_counter()
        result = train_neural_checkpoint(
            checkpoint,
            database_url=DATABASE_URL,
            config=spec.config,
        )
        elapsed = time.perf_counter() - started
        engine = ResidualEngine(checkpoint)
        jump_metrics = _jump_metrics(engine, sampled_conditions)
        payload = {
            "name": spec.name,
            "description": spec.description,
            "seconds": elapsed,
            "config": asdict(spec.config),
            "result": result.model_dump(mode="json"),
            "jump_metrics": jump_metrics,
        }
        results.append(payload)
        print(json.dumps(payload, indent=2), flush=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = [
        "# FET-GEN model experiment summary",
        "",
        "| Attempt | Method | Weighted RMSE | Vth MAE | SS MAE | Jump P95 | Spike rate | Time (s) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        result = item["result"]
        jump = item["jump_metrics"]
        assert isinstance(result, dict)
        assert isinstance(jump, dict)
        markdown.append(
            "| "
            f"{item['name']} | "
            f"{result['method']} | "
            f"{result.get('validation_weighted_rmse_decades') or result.get('validation_rmse_decades'):.4f} | "
            f"{(result.get('feature_vth_mae_v') or float('nan')):.4f} | "
            f"{(result.get('feature_ss_mae_mv_dec') or float('nan')):.2f} | "
            f"{jump['jump_p95_decades']:.4f} | "
            f"{jump['jump_spike_rate']:.4f} | "
            f"{item['seconds']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    final_dir = run()
    print(f"\nSaved experiment outputs to {final_dir}", flush=True)
