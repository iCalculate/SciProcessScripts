from pathlib import Path

import numpy as np
import pytest

from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition
from devicecurvegen.training import train_residual_checkpoint


def _write_curve(path: Path, index: int, *, reverse: bool = False) -> None:
    voltage = np.linspace(-5, 10, 101)
    if reverse:
        voltage = voltage[::-1]
    current = 1e-11 + (1e-5 * (1 + index * 0.05)) / (
        1 + np.exp(-(voltage - (2 + index * 0.1)) / 0.35)
    )
    path.write_text(
        "Vg,Id\n" + "\n".join(f"{vg},{ids}" for vg, ids in zip(voltage, current, strict=True)),
        encoding="utf-8",
    )


def test_train_and_load_checkpoint_with_reverse_sweep(tmp_path: Path) -> None:
    inputs = []
    for index in range(3):
        path = tmp_path / f"curve-{index}.csv"
        _write_curve(path, index, reverse=index == 1)
        inputs.append(path)
    output = tmp_path / "residual.npz"
    result = train_residual_checkpoint(inputs, output, components=8)
    engine = ResidualEngine(output)
    generated = generate_curves(GenerationCondition(variants=1), engine)
    assert result.curves == 3
    assert result.components == 2
    assert engine.mode == "learned_pca"
    assert len(generated.candidates[0].latent_code) == 2


def test_training_requires_positive_component_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="components"):
        train_residual_checkpoint([], tmp_path / "x.npz", components=0)


def test_malformed_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(
        path,
        grid=np.array([1.0, 0.0]),
        mean=np.zeros(2),
        components=np.zeros((1, 2)),
        scales=np.ones(1),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        ResidualEngine(path)


def test_missing_checkpoint_is_not_silently_ignored(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        ResidualEngine(tmp_path / "missing.npz")


def test_short_valid_curves_can_train(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        voltage = np.linspace(-5, 10, 16)
        current = 1e-11 + 1e-5 / (1 + np.exp(-(voltage - 2 - index * 0.1) / 0.4))
        path = tmp_path / f"short-{index}.csv"
        path.write_text(
            "Vg,Id\n" + "\n".join(f"{vg},{ids}" for vg, ids in zip(voltage, current, strict=True)),
            encoding="utf-8",
        )
        paths.append(path)
    result = train_residual_checkpoint(paths, tmp_path / "short-model.npz")
    assert result.curves == 3
