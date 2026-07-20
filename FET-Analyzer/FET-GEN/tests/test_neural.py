import time
from pathlib import Path

import numpy as np
import pandas as pd

from devicecurvegen.neural import NeuralTrainingConfig, train_neural_checkpoint
from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition, NeuralTrainingRequest
from devicecurvegen.training_service import NeuralTrainingManager
from tools.build_hybrid_checkpoint import build_hybrid_checkpoint


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
    assert "log10_gm_max" in engine.info().condition_features
    assert len(generated.candidates[0].latent_code) == 3
    assert np.all(np.isfinite(generated.candidates[0].forward_current))


def test_aligned_local_delta_cvae_trains_loads_and_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "aligned-local-delta-cvae.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="aligned_local_delta_cvae",
            latent_dim=4,
            hidden_dim=16,
            epochs=3,
            batch_size=4,
            patience=3,
            slope_weight=0.25,
            subthreshold_weight=3.0,
            seed=8,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            voltage_min=-5.0,
            voltage_max=10.0,
            target_vth=2.2,
            target_ss_mv_dec=135.0,
            points=101,
            gate_ai_residual_strength=1.0,
        ),
        engine,
    )

    assert result.method == "aligned_local_delta_cvae"
    assert result.best_epoch >= 1
    assert np.isfinite(result.validation_loss)
    assert engine.mode == "conditional_vae"
    assert engine.info().architecture == "aligned_local_delta_conditional_vae_residual_skip"
    assert engine.metadata["threshold_local_align_window_scale"] > 0
    assert engine.metadata["threshold_local_align_min_window_v"] > 0
    assert len(engine.metadata["latent_prior_std"]) == 4
    assert engine.threshold_local_delta_transform is True
    assert len(generated.candidates[0].latent_code) == 4
    assert generated.candidates[0].gate_latent_code
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


def test_conditional_pca_trains_and_samples_from_conditions(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "conditional-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="conditional_pca",
            pca_components=5,
            beta=0.01,
            seed=17,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.3,
            target_ss_mv_dec=140.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert result.method == "conditional_pca"
    assert result.latent_dim == 5
    assert engine.mode == "learned_pca"
    assert engine.info().architecture == "conditional_pca"
    assert "log10_gm_max" in engine.info().condition_features
    assert len(candidate.latent_code) == 5
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))


def test_threshold_conditional_pca_trains_and_samples_from_conditions(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "threshold-conditional-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="threshold_conditional_pca",
            pca_components=5,
            beta=0.01,
            subthreshold_weight=3.2,
            slope_weight=0.35,
            seed=29,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.1,
            target_ss_mv_dec=135.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert result.method == "threshold_conditional_pca"
    assert result.latent_dim == 5
    assert engine.mode == "learned_pca"
    assert engine.info().architecture == "threshold_conditional_pca"
    assert len(candidate.latent_code) == 5
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))


def test_local_threshold_conditional_pca_trains_and_samples_from_conditions(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "local-threshold-conditional-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="local_threshold_conditional_pca",
            pca_components=5,
            beta=0.012,
            subthreshold_weight=3.2,
            slope_weight=0.35,
            seed=43,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.1,
            target_ss_mv_dec=135.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert result.method == "local_threshold_conditional_pca"
    assert result.latent_dim == 5
    assert engine.mode == "learned_pca"
    assert engine.info().architecture == "local_threshold_conditional_pca"
    assert len(candidate.latent_code) == 5
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))


def test_aligned_local_threshold_conditional_pca_trains_and_samples_from_conditions(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "aligned-local-threshold-conditional-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="aligned_local_threshold_conditional_pca",
            pca_components=5,
            beta=0.012,
            subthreshold_weight=3.2,
            slope_weight=0.35,
            seed=44,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.1,
            target_ss_mv_dec=135.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert result.method == "aligned_local_threshold_conditional_pca"
    assert result.latent_dim == 5
    assert engine.mode == "learned_pca"
    assert engine.info().architecture == "aligned_local_threshold_conditional_pca"
    assert engine.metadata["threshold_local_align_window_scale"] > 0
    assert engine.metadata["threshold_local_align_min_window_v"] > 0
    assert len(candidate.latent_code) == 5
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))


def test_aligned_local_delta_conditional_pca_trains_and_samples_from_conditions(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "aligned-local-delta-conditional-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="aligned_local_delta_conditional_pca",
            pca_components=5,
            beta=0.012,
            subthreshold_weight=3.2,
            slope_weight=0.40,
            seed=54,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.1,
            target_ss_mv_dec=135.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert result.method == "aligned_local_delta_conditional_pca"
    assert result.latent_dim == 5
    assert engine.mode == "learned_pca"
    assert engine.info().architecture == "aligned_local_delta_conditional_pca"
    assert engine.metadata["threshold_local_align_window_scale"] > 0
    assert engine.metadata["threshold_local_align_min_window_v"] > 0
    assert engine.threshold_local_delta_transform is True
    assert len(candidate.latent_code) == 5
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))


def test_aligned_local_affine_delta_conditional_pca_trains_and_samples_from_conditions(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    checkpoint = tmp_path / "aligned-local-affine-delta-conditional-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    result = train_neural_checkpoint(
        checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="aligned_local_affine_delta_conditional_pca",
            pca_components=5,
            beta=0.012,
            subthreshold_weight=3.2,
            slope_weight=0.40,
            seed=64,
        ),
    )

    engine = ResidualEngine(checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.1,
            target_ss_mv_dec=135.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert result.method == "aligned_local_affine_delta_conditional_pca"
    assert result.latent_dim == 5
    assert engine.mode == "learned_pca"
    assert engine.info().architecture == "aligned_local_affine_delta_conditional_pca"
    assert engine.metadata["threshold_local_align_window_scale"] > 0
    assert engine.metadata["threshold_local_align_min_window_v"] > 0
    assert engine.metadata["threshold_local_affine_restore"] is True
    assert engine.threshold_local_delta_transform is True
    assert engine.affine_w is not None
    assert engine.affine_b is not None
    assert len(candidate.latent_code) == 5
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))


def test_hybrid_threshold_pca_builds_and_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    base_checkpoint = tmp_path / "base-pca.npz"
    guide_checkpoint = tmp_path / "guide-conditional-pca.npz"
    hybrid_checkpoint = tmp_path / "hybrid-threshold-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    train_neural_checkpoint(
        base_checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="latent_pca",
            pca_components=4,
            seed=19,
        ),
    )
    train_neural_checkpoint(
        guide_checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="conditional_pca",
            pca_components=4,
            beta=0.01,
            seed=23,
        ),
    )
    build_hybrid_checkpoint(
        base_path=base_checkpoint,
        guide_path=guide_checkpoint,
        output_path=hybrid_checkpoint,
        local_blend=0.82,
        global_blend=0.06,
        window_scale=3.0,
        min_window_v=0.22,
    )

    engine = ResidualEngine(hybrid_checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.3,
            target_ss_mv_dec=140.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert engine.info().architecture == "hybrid_threshold_pca"
    assert engine.info().components == 8
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))


def test_hybrid_threshold_pca_supports_separate_reverse_guide(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    base_checkpoint = tmp_path / "base-pca.npz"
    guide_checkpoint = tmp_path / "guide-conditional-pca.npz"
    reverse_guide_checkpoint = tmp_path / "reverse-guide-threshold-pca.npz"
    hybrid_checkpoint = tmp_path / "hybrid-threshold-directional-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    train_neural_checkpoint(
        base_checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="latent_pca",
            pca_components=4,
            seed=31,
        ),
    )
    train_neural_checkpoint(
        guide_checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="conditional_pca",
            pca_components=4,
            beta=0.01,
            seed=37,
        ),
    )
    train_neural_checkpoint(
        reverse_guide_checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="threshold_conditional_pca",
            pca_components=4,
            beta=0.01,
            subthreshold_weight=3.0,
            slope_weight=0.3,
            seed=41,
        ),
    )
    build_hybrid_checkpoint(
        base_path=base_checkpoint,
        guide_path=guide_checkpoint,
        reverse_guide_path=reverse_guide_checkpoint,
        output_path=hybrid_checkpoint,
        local_blend=0.82,
        global_blend=0.06,
        window_scale=3.0,
        min_window_v=0.22,
    )

    engine = ResidualEngine(hybrid_checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.3,
            target_ss_mv_dec=140.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert engine.metadata["hybrid_directional_guides"] is True
    assert engine.metadata["reverse_guide_model_name"] == reverse_guide_checkpoint.stem
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))
    assert np.all(np.isfinite(candidate.reverse_current))


def test_hybrid_threshold_pca_supports_local_delta_guide(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    base_checkpoint = tmp_path / "base-pca.npz"
    guide_checkpoint = tmp_path / "local-guide-threshold-pca.npz"
    hybrid_checkpoint = tmp_path / "hybrid-threshold-local-delta-pca.npz"
    _write_neural_dataset(dataset, include_gate=True)
    train_neural_checkpoint(
        base_checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="latent_pca",
            pca_components=4,
            seed=47,
        ),
    )
    train_neural_checkpoint(
        guide_checkpoint,
        dataset_path=dataset,
        config=NeuralTrainingConfig(
            method="local_threshold_conditional_pca",
            pca_components=4,
            beta=0.012,
            subthreshold_weight=3.2,
            slope_weight=0.35,
            seed=53,
        ),
    )
    build_hybrid_checkpoint(
        base_path=base_checkpoint,
        guide_path=guide_checkpoint,
        guide_as_local_delta=True,
        output_path=hybrid_checkpoint,
        base_scale_multiplier=0.5,
        local_blend=0.92,
        global_blend=0.0,
        window_scale=3.0,
        min_window_v=0.22,
        guide_delta_clip_decades=0.18,
        guide_delta_anchor_strength=1.0,
        guide_delta_preserve_affine_strength=0.5,
        post_vth_align_strength=0.25,
        post_vth_align_reverse_only=True,
        post_vth_align_local_window_scale=2.5,
        post_vth_align_local_min_window_v=0.2,
    )

    engine = ResidualEngine(hybrid_checkpoint)
    generated = generate_curves(
        GenerationCondition(
            variants=1,
            points=101,
            gate_ai_residual_strength=1.0,
            target_vth=2.3,
            target_ss_mv_dec=140.0,
        ),
        engine,
    )
    candidate = generated.candidates[0]

    assert engine.metadata["hybrid_guide_as_local_delta"] is True
    assert engine.metadata["hybrid_base_scale_multiplier"] == 0.5
    assert engine.metadata["hybrid_guide_delta_anchor_strength"] == 1.0
    assert engine.metadata["hybrid_guide_delta_preserve_affine_strength"] == 0.5
    assert engine.metadata["hybrid_post_vth_align_strength"] == 0.25
    assert engine.metadata["hybrid_post_vth_align_reverse_only"] is True
    assert engine.metadata["hybrid_post_vth_align_local_window_scale"] == 2.5
    assert engine.metadata["hybrid_post_vth_align_local_min_window_v"] == 0.2
    assert candidate.gate_latent_code
    assert np.all(np.isfinite(candidate.forward_current))
    assert np.all(np.isfinite(candidate.reverse_current))


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
