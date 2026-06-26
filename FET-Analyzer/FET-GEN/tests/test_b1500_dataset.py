from pathlib import Path

import numpy as np

from devicecurvegen.b1500_dataset import (
    build_b1500_dataset,
    choose_columns,
    parse_source,
)


def test_b1500_csv_transfer_is_classified_from_data(tmp_path: Path) -> None:
    rows = [
        "SetupTitle, arbitrary name",
        "PrimitiveTest, I/V Sweep",
        "TestParameter, Output.Graph.XAxis.Data, Vg",
        "DataName, Vg, Vd, Id, Ig",
    ]
    rows.extend(
        f"DataValue, {value}, 1, {1e-12 * (10 ** (value / 2))}, 1e-13"
        for value in range(10)
    )
    path = tmp_path / "not_named_transfer.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    table = parse_source(path)[0]
    choice = choose_columns(table)

    assert choice is not None
    assert choice.curve_type == "transfer"
    assert choice.voltage == "Vg"
    assert choice.current == "Id"


def test_b1500_csv_output_is_not_misclassified_as_transfer(tmp_path: Path) -> None:
    rows = [
        "SetupTitle, misleading transfer label",
        "PrimitiveTest, I/V Sweep",
        "TestParameter, Output.Graph.XAxis.Data, Vd",
        "DataName, Vd, Vg, Id",
    ]
    rows.extend(f"DataValue, {value / 10}, 1, {1e-6 * value}" for value in range(10))
    path = tmp_path / "transfer-looking-name.csv"
    path.write_text("\n".join(rows), encoding="utf-8")

    table = parse_source(path)[0]
    choice = choose_columns(table)

    assert choice is not None
    assert choice.curve_type == "output"
    assert choice.voltage == "Vd"


def test_b1500_export_includes_aligned_gate_current(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "dataset"
    source.mkdir()
    rows = [
        "SetupTitle, dual channel transfer",
        "PrimitiveTest, I/V Sweep",
        "TestParameter, Output.Graph.XAxis.Data, Vg",
        "DataName, Vg, Vd, Id, Ig",
    ]
    voltage = np.linspace(-3.0, 3.0, 41)
    current = 1e-13 + 1e-5 / (1.0 + np.exp(-(voltage - 0.2) / 0.25))
    gate = 1e-14 * (1.0 + 2.0 * np.abs(voltage) ** 0.8)
    rows.extend(
        f"DataValue, {vg}, 1, {ids}, {ig}"
        for vg, ids, ig in zip(voltage, current, gate, strict=True)
    )
    (source / "dual.csv").write_text("\n".join(rows), encoding="utf-8")

    summary = build_b1500_dataset(source, output)

    with np.load(output / "aligned_curves.npz") as payload:
        assert "log10_abs_ig" in payload.files
        assert payload["log10_abs_ig"].shape == payload["log10_abs_id"].shape
        assert np.all(np.isfinite(payload["log10_abs_ig"]))
    assert summary["segments_with_gate_current"] == 1
    assert (output / "aligned_gate_curves.csv").is_file()
