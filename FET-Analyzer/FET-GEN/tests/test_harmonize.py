import pytest

from devicecurvegen.harmonize import inspect_measurement


def test_inspect_maps_and_segments_transfer_curve() -> None:
    rows = ["Gate Voltage,Drain Current"]
    rows.extend(f"{value},{1e-12 * (10 ** (value / 2))}" for value in range(6))
    rows.extend(f"{value},{1e-12 * (10 ** (value / 2))}" for value in range(5, -1, -1))
    result = inspect_measurement("curve.csv", "\n".join(rows).encode())
    assert result.mapping.voltage == "Gate Voltage"
    assert result.mapping.current == "Drain Current"
    assert result.curve_type == "transfer"
    assert len(result.segments) == 2
    assert result.segments[0].aligned_voltage
    assert result.segments[0].aligned_log_current


def test_headers_with_units_map_with_high_confidence() -> None:
    content = (
        b"Gate Voltage (V),Drain Current (A)\n0,1e-12\n1,1e-11\n2,1e-10\n3,1e-9\n4,1e-8\n5,1e-7"
    )
    result = inspect_measurement("curve.csv", content)
    assert result.mapping.voltage == "Gate Voltage (V)"
    assert result.mapping.current == "Drain Current (A)"
    assert result.mapping.confidence > 0.9
    assert result.curve_type == "transfer"


def test_turning_point_without_duplicate_is_preserved_in_both_segments() -> None:
    content = b"V,I\n0,1e-12\n1,1e-11\n2,1e-10\n3,1e-9\n2,1e-10\n1,1e-11\n0,1e-12"
    result = inspect_measurement(
        "curve.csv",
        content,
        voltage_column="V",
        current_column="I",
    )
    assert [(segment.direction, segment.rows) for segment in result.segments] == [
        ("forward", 4),
        ("reverse", 4),
    ]
    assert result.segments[0].voltage[-1] == result.segments[1].voltage[0] == 3


def test_invalid_manual_mapping_is_rejected() -> None:
    content = b"V,I\n0,1e-12\n1,1e-11\n2,1e-10\n3,1e-9"
    with pytest.raises(ValueError, match="Unknown voltage column"):
        inspect_measurement("curve.csv", content, voltage_column="missing")


def test_too_few_clean_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="Fewer than four"):
        inspect_measurement("curve.csv", b"Vg,Id\n0,1e-12\n1,1e-11\n2,NaN")
