from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from devicecurvegen.cli import app

runner = CliRunner()


def _write_curve(path: Path, offset: float = 0.0) -> None:
    voltage = np.linspace(-5, 10, 61)
    current = 1e-11 + 1e-5 / (1 + np.exp(-(voltage - 2 - offset) / 0.4))
    path.write_text(
        "Vg,Id\n" + "\n".join(f"{vg},{ids}" for vg, ids in zip(voltage, current, strict=True)),
        encoding="utf-8",
    )


def test_generate_inspect_extract_and_ingest_commands(tmp_path: Path) -> None:
    curve = tmp_path / "curve.csv"
    _write_curve(curve)
    generated = tmp_path / "generated.csv"
    result = runner.invoke(
        app,
        ["generate", "--output", str(generated), "--variants", "2", "--points", "51"],
    )
    assert result.exit_code == 0
    assert generated.read_text(encoding="utf-8").splitlines()[0].endswith("Id_physics_reverse")

    inspected = runner.invoke(app, ["inspect", str(curve)])
    assert inspected.exit_code == 0
    assert '"curve_type": "transfer"' in inspected.stdout

    extracted = runner.invoke(app, ["extract", str(curve)])
    assert extracted.exit_code == 0
    assert '"features"' in extracted.stdout

    dataset = tmp_path / "dataset.json"
    ingested = runner.invoke(
        app,
        ["ingest", str(tmp_path), "--output", str(dataset)],
    )
    assert ingested.exit_code == 0
    assert dataset.exists()


def test_train_command_and_bad_path(tmp_path: Path) -> None:
    for index in range(3):
        _write_curve(tmp_path / f"curve-{index}.csv", offset=index * 0.1)
    checkpoint = tmp_path / "model.npz"
    trained = runner.invoke(
        app,
        ["train", str(tmp_path), "--output", str(checkpoint), "--components", "4"],
    )
    assert trained.exit_code == 0
    assert checkpoint.exists()

    missing = runner.invoke(app, ["inspect", str(tmp_path / "missing.csv")])
    assert missing.exit_code != 0
