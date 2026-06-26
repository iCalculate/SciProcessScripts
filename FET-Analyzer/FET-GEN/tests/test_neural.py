import time
from pathlib import Path

import numpy as np
import pandas as pd

from devicecurvegen.neural import NeuralTrainingConfig, train_neural_checkpoint
from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition, NeuralTrainingRequest
from devicecurvegen.training_service import NeuralTrainingManager


def _write_neural_dataset(
    root: Path,
    curves: int = 16,
    *,
    include_gate: bool = False,
) -> None:
    root.mkdir()
    grid = np.linspace(-1.0, 1.0, 201, dtype=np.float32)
    curve_ids: list[str] = []
    matrix: list[np.ndarray] = []
    gate_matrix: list[np.ndarray] = []
    metadata: list[dict] = []
    for index in range(curves):
        curve_id = f"curve-{index:03d}"
        voltage_min = -5.0
        voltage_max = 10.0
        voltage = voltage_min + 0.5 * (grid + 1.0) * (voltage_max - voltage_min)
        vth = 1.5 + 0.08 * index
        ion = 1e-5 * (1.0 + 0.02 * index)
        ioff = 1e-11 * (1.0 + 0.03 * index)
        current = ioff + (ion - ioff) / (1.0 + np.exp(-(voltage - vth) / 0.35))
        residual_shape = 0.08 * np.sin(np.pi * grid * (1.0 + index % 3))
        curve_ids.append(curve_id)
        matrix.append(np.log10(current) + residual_shape)
        if include_gate:
            gate = 1e-13 * (
                1.0
                + 2.5 * np.abs(grid + 0.08 * (index % 3)) ** 0.8
                + 0.05 * np.sin((index % 4 + 1) * np.pi * grid)
            )
            gate_matrix.append(np.log10(gate))
        metadata.append(
            {
                "curve_id": curve_id,
                "source_path": f"device-{index // 2}.csv",
                "direction": "forward" if index % 2 == 0 else "reverse",
                "voltage_min_v": voltage_min,
                "voltage_max_v": voltage_max,
                "feature_ion": ion,
                "feature_ioff": ioff,
                "feature_polarity": "n-type",
                "feature_vth": vth,
                "feature_ss_mv_dec": 110.0 + index,
            }
        )
    payload = {
        "curve_id": np.asarray(curve_ids),
        "x_norm": grid,
        "log10_abs_id": np.asarray(matrix, dtype=np.float32),
    }
    if include_gate:
        payload["log10_abs_ig"] = np.asarray(gate_matrix, dtype=np.float32)
    np.savez_compressed(root / "aligned_curves.npz", **payload)
    pd.DataFrame(metadata).to_csv(root / "curves.csv", index=False)


def test_train_load_and_sample_conditional_vae(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "residual-cvae.npz"
    _write_neural_dataset(dataset)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            latent_dim=3,
            hidden_dim=12,
            epochs=3,
            batch_size=4,
            patience=3,
            seed=7,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            voltage_min=-5.0,
            voltage_max=10.0,
            target_vth=2.0,
            points=101,
        ),
        engine,
    )

    assert result.curves == 16
    assert result.best_epoch >= 1
    assert np.isfinite(result.validation_loss)
    assert engine.mode == "conditional_vae"
    assert engine.info().components == 3
    assert len(generated.candidates[0].latent_code) == 3
    assert np.all(np.isfinite(generated.candidates[0].forward_current))


def test_dual_channel_cvae_generates_ids_and_ig(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "dual-cvae.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            latent_dim=3,
            hidden_dim=12,
            epochs=2,
            batch_size=4,
            patience=2,
            seed=11,
            gate_loss_weight=1.0,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert result.generated_channels == ["Ids", "Ig"]
    assert result.gate_curves == 16
    assert result.validation_gate_rmse_decades is not None
    assert engine.info().generated_channels == ["Ids", "Ig"]
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.gate_forward_current))


def test_latent_pca_trains_from_aligned_dual_channel_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "dual-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="latent_pca",
            pca_components=4,
            seed=13,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
        ),
        engine,
    )

    assert result.method == "latent_pca"
    assert result.latent_dim == 4
    assert result.generated_channels == ["Ids", "Ig"]
    assert engine.mode == "learned_pca"
    assert generated.candidates[0].gate_latent_code


def test_quick_search_activates_best_trial(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "best.npz"
    _write_neural_dataset(dataset, curves=12, include_gate=True)
    monkeypatch.setenv("DEVICEGEN_NEURAL_MODEL_OUTPUT", str(output))
    activated: list[Path] = []
    manager = NeuralTrainingManager(activated.append)
    manager.start(
        NeuralTrainingRequest(
            dataset_path=str(dataset),
            method="physics_cvae",
            search_strategy="quick",
            search_trials=2,
            latent_dim=2,
            hidden_dim=8,
            epochs=1,
            batch_size=4,
            patience=1,
            feature_eval_limit=4,
        )
    )

    deadline = time.monotonic() + 10
    status = manager.snapshot()
    while status.status == "running" and time.monotonic() < deadline:
        time.sleep(0.02)
        status = manager.snapshot()

    assert status.status == "completed"
    assert status.result is not None
    assert len(status.trials) == 2
    assert status.result.best_trial in {1, 2}
    assert status.result.output == str(output)
    assert activated == [output]
    assert output.is_file()
    assert ResidualEngine(output).info().tuning_trials
