from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, insert

from devicecurvegen.database import (
    aligned_gate_points,
    aligned_points,
    analyze_curves,
    backfill_b1500_gate_points,
    create_schema,
    curves,
    database_status,
    export_curve_rows,
    get_curve_detail,
    list_curves,
    match_matrix_sites,
    raw_gate_points,
    raw_points,
    source_files,
    test_configs,
)


def test_database_schema_and_curve_browser_queries(tmp_path: Path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'curves.db'}"
    engine = create_engine(url, future=True)
    create_schema(engine)
    with engine.begin() as connection:
        source_id = connection.execute(
            insert(source_files).values(
                source_path="User/sample.csv",
                extension=".csv",
                size_bytes=123,
                modified_at=datetime(2026, 6, 20, 12, 0, 0),
                sha1=None,
                created_at=datetime(2026, 6, 20, 12, 0, 0),
            )
        ).inserted_primary_key[0]
        config_id = connection.execute(
            insert(test_configs).values(
                source_file_id=source_id,
                table_name="DataBlock1",
                source_kind="b1500_csv",
                setup_title="IdVg",
                primitive_test="I/V Sweep",
                x_axis_data="Vg",
                voltage_column="Vg",
                current_column="Id",
                gate_current_column="Ig",
                classification="transfer",
                classification_reason="graph x-axis is swept gate voltage",
                classification_confidence=0.98,
                columns_json=["Vg", "Id"],
                metadata_json={"SetupTitle": ["IdVg"]},
            )
        ).inserted_primary_key[0]
        connection.execute(
            insert(curves).values(
                curve_id="abc123",
                source_file_id=source_id,
                test_config_id=config_id,
                segment_index=1,
                direction="forward",
                rows_clean=2,
                voltage_min_v=0,
                voltage_max_v=1,
                ion=1e-6,
                ioff=1e-12,
                ion_ioff_ratio=1e6,
                polarity="n-type",
                has_gate_current=1,
                vth=0.4,
                ss_mv_dec=90,
                ss_fit_r2=0.99,
                gm_max=1e-7,
                vth_gmmax=0.45,
                von=0.3,
                hysteresis_v=None,
                leakage_level=1e-12,
                noise_log_sigma=0.01,
                ambipolar_strength=0.0,
                current_floor=1e-12,
                imported_at=datetime(2026, 6, 20, 12, 0, 0),
            )
        )
        connection.execute(
            insert(raw_points),
            [
                {"curve_id": "abc123", "point_index": 0, "voltage_v": 0, "current_a": 1e-12},
                {"curve_id": "abc123", "point_index": 1, "voltage_v": 1, "current_a": 1e-6},
            ],
        )
        connection.execute(
            insert(raw_gate_points),
            [
                {"curve_id": "abc123", "point_index": 0, "voltage_v": 0, "current_a": 2e-13},
                {"curve_id": "abc123", "point_index": 1, "voltage_v": 1, "current_a": 3e-13},
            ],
        )
        connection.execute(
            insert(aligned_points),
            [
                {
                    "curve_id": "abc123",
                    "point_index": 0,
                    "x_norm": -1,
                    "voltage_v": 0,
                    "log10_abs_id": -12,
                    "abs_id_a": 1e-12,
                },
                {
                    "curve_id": "abc123",
                    "point_index": 1,
                    "x_norm": 1,
                    "voltage_v": 1,
                    "log10_abs_id": -6,
                    "abs_id_a": 1e-6,
                },
            ],
        )
        connection.execute(
            insert(aligned_gate_points),
            [
                {
                    "curve_id": "abc123",
                    "point_index": 0,
                    "x_norm": -1,
                    "voltage_v": 0,
                    "log10_abs_ig": -12.7,
                    "abs_ig_a": 2e-13,
                },
                {
                    "curve_id": "abc123",
                    "point_index": 1,
                    "x_norm": 1,
                    "voltage_v": 1,
                    "log10_abs_ig": -12.5,
                    "abs_ig_a": 3e-13,
                },
            ],
        )

    status = database_status(url)
    assert status["configured"] is True
    assert status["curves"] == 1
    assert status["raw_points"] == 2
    assert status["aligned_points"] == 2
    assert status["gate_points"] == 2
    assert status["aligned_gate_points"] == 2
    assert status["curves_with_ig"] == 1

    listed = list_curves(url, polarity="n-type", has_gate_current="true")
    assert listed["total"] == 1
    assert listed["items"][0]["curve_id"] == "abc123"
    assert listed["items"][0]["log_ratio"] == 6
    assert listed["items"][0]["has_gate_current"] is True

    detail = get_curve_detail(url, "abc123")
    assert detail is not None
    assert detail["metadata_json"] == {"SetupTitle": ["IdVg"]}
    assert len(detail["raw_points"]) == 2
    assert len(detail["gate_points"]) == 2
    assert len(detail["aligned_points"]) == 2
    assert len(detail["aligned_gate_points"]) == 2
    assert detail["gate_current_column"] == "Ig"

    analysis = analyze_curves(url, curve_ids=["abc123"])
    assert analysis["count"] == 1
    assert analysis["metrics"]["logRatio"]["mean"] == 6
    assert analysis["sample_count"] == 1
    assert analysis["samples"][0]["curve_id"] == "abc123"
    assert analysis["samples"][0]["logRatio"] == 6
    assert sum(bin_["count"] for bin_ in analysis["distributions"]["logRatio"]) == 1
    assert analysis["correlations"]["features"]
    assert analysis["pca"]["points"] == []
    assert analysis["categorical"]["has_ig"] == {"yes": 1, "no": 0}

    exported = export_curve_rows(url, curve_ids=["abc123"])
    assert exported["curves"][0]["logRatio"] == 6
    assert exported["curves"][0]["has_ig"] is True
    assert len(exported["raw_points"]) == 2
    assert len(exported["gate_points"]) == 2
    assert len(exported["aligned_gate_points"]) == 2


def test_matrix_site_matching_prefers_nearest_unique_curve(tmp_path: Path) -> None:
    url = f"sqlite+pysqlite:///{tmp_path / 'curves.db'}"
    engine = create_engine(url, future=True)
    create_schema(engine)
    with engine.begin() as connection:
        source_id = connection.execute(
            insert(source_files).values(
                source_path="User/matrix.csv",
                extension=".csv",
                size_bytes=123,
                modified_at=datetime(2026, 6, 20, 12, 0, 0),
                sha1=None,
                created_at=datetime(2026, 6, 20, 12, 0, 0),
            )
        ).inserted_primary_key[0]
        config_id = connection.execute(
            insert(test_configs).values(
                source_file_id=source_id,
                table_name="DataBlock1",
                source_kind="b1500_csv",
                setup_title="IdVg",
                primitive_test="I/V Sweep",
                x_axis_data="Vg",
                voltage_column="Vg",
                current_column="Id",
                gate_current_column=None,
                classification="transfer",
                classification_reason="graph x-axis is swept gate voltage",
                classification_confidence=0.98,
                columns_json=["Vg", "Id"],
                metadata_json={"SetupTitle": ["IdVg"]},
            )
        ).inserted_primary_key[0]
        for curve_id, vth, ion in [("lowvth", 0.2, 1e-6), ("highvth", 1.8, 8e-6)]:
            connection.execute(
                insert(curves).values(
                    curve_id=curve_id,
                    source_file_id=source_id,
                    test_config_id=config_id,
                    segment_index=1,
                    direction="forward",
                    rows_clean=2,
                    voltage_min_v=0,
                    voltage_max_v=2,
                    ion=ion,
                    ioff=1e-12,
                    ion_ioff_ratio=ion / 1e-12,
                    polarity="n-type",
                    has_gate_current=0,
                    vth=vth,
                    ss_mv_dec=100,
                    ss_fit_r2=0.99,
                    gm_max=1e-7,
                    vth_gmmax=vth,
                    von=0.1,
                    hysteresis_v=None,
                    leakage_level=1e-12,
                    noise_log_sigma=0.01,
                    ambipolar_strength=0.0,
                    current_floor=1e-12,
                    imported_at=datetime(2026, 6, 20, 12, 0, 0),
                )
            )

    assignments = match_matrix_sites(
        url,
        site_targets=[
            {"site": "A1", "row": 1, "col": 1, "parameters": {"target_vth": 0.25}},
            {"site": "B1", "row": 1, "col": 2, "parameters": {"target_vth": 0.3}},
        ],
        filters={"polarity": "n-type"},
        duplicate_mode="avoid",
    )

    assert assignments[0]["curve_id"] == "lowvth"
    assert assignments[1]["curve_id"] == "highvth"
    assert assignments[1]["source"] == "database"


def test_backfill_b1500_gate_points_imports_raw_and_aligned_ig(tmp_path: Path) -> None:
    source = tmp_path / "source"
    data_dir = source / "User"
    data_dir.mkdir(parents=True)
    rows = [
        "SetupTitle, IdVg",
        "PrimitiveTest, I/V Sweep",
        "TestParameter, Output.Graph.XAxis.Data, Vg",
        "DataName, Vg, Vd, Id, Ig",
    ]
    for index in range(25):
        vg = float(index)
        rows.append(f"DataValue, {vg}, 1, {1e-12 * (10 ** (index / 4))}, {1e-13 * (index + 1)}")
    (data_dir / "sample.csv").write_text("\n".join(rows), encoding="utf-8")

    url = f"sqlite+pysqlite:///{tmp_path / 'curves.db'}"
    engine = create_engine(url, future=True)
    create_schema(engine)
    with engine.begin() as connection:
        source_id = connection.execute(
            insert(source_files).values(
                source_path="User/sample.csv",
                extension=".csv",
                size_bytes=123,
                modified_at=datetime(2026, 6, 20, 12, 0, 0),
                sha1=None,
                created_at=datetime(2026, 6, 20, 12, 0, 0),
            )
        ).inserted_primary_key[0]
        config_id = connection.execute(
            insert(test_configs).values(
                source_file_id=source_id,
                table_name="DataBlock1",
                source_kind="b1500_csv",
                setup_title="IdVg",
                primitive_test="I/V Sweep",
                x_axis_data="Vg",
                voltage_column="Vg",
                current_column="Id",
                gate_current_column="Ig",
                classification="transfer",
                classification_reason="graph x-axis is swept gate voltage",
                classification_confidence=0.98,
                columns_json=["Vg", "Vd", "Id", "Ig"],
                metadata_json={"SetupTitle": ["IdVg"]},
            )
        ).inserted_primary_key[0]
        connection.execute(
            insert(curves).values(
                curve_id="needsig",
                source_file_id=source_id,
                test_config_id=config_id,
                segment_index=1,
                direction="single",
                rows_clean=25,
                voltage_min_v=0,
                voltage_max_v=24,
                ion=1e-6,
                ioff=1e-12,
                ion_ioff_ratio=1e6,
                polarity="n-type",
                has_gate_current=1,
                vth=0.4,
                ss_mv_dec=90,
                ss_fit_r2=0.99,
                gm_max=1e-7,
                vth_gmmax=0.45,
                von=0.3,
                hysteresis_v=None,
                leakage_level=1e-12,
                noise_log_sigma=0.01,
                ambipolar_strength=0.0,
                current_floor=1e-12,
                imported_at=datetime(2026, 6, 20, 12, 0, 0),
            )
        )
        connection.execute(
            insert(raw_points),
            [
                {
                    "curve_id": "needsig",
                    "point_index": index,
                    "voltage_v": float(index),
                    "current_a": 1e-12 * (10 ** (index / 4)),
                }
                for index in range(25)
            ],
        )

    dry_run = backfill_b1500_gate_points(source, url, dry_run=True)
    assert dry_run["candidate_curves"] == 1
    assert dry_run["curves_backfilled"] == 1
    assert database_status(url)["gate_points"] == 0

    summary = backfill_b1500_gate_points(source, url)
    assert summary["applied"] is True
    assert summary["curves_backfilled"] == 1
    assert summary["raw_gate_points"] == 25
    assert summary["aligned_gate_points"] == 201

    detail = get_curve_detail(url, "needsig")
    assert detail is not None
    assert len(detail["gate_points"]) == 25
    assert len(detail["aligned_gate_points"]) == 201
    assert detail["gate_points"][0]["voltage_v"] == 0.0
    assert detail["gate_points"][0]["current_a"] == 1e-13
    assert detail["aligned_gate_points"][0]["x_norm"] == -1.0
    assert detail["aligned_gate_points"][-1]["x_norm"] == 1.0
