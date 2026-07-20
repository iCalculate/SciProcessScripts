from __future__ import annotations

from pathlib import Path

from backend.config import AppSettings, DatabaseSettings, ImporterSettings
from backend.services.database import DatabaseService
from backend.services.metadata_parser import build_media_id, build_spectrum_id


def build_settings(tmp_path: Path) -> AppSettings:
    settings = AppSettings(
        project_root=tmp_path,
        config_path=tmp_path / "config.yaml",
        importer=ImporterSettings(mock_mode=True),
        database=DatabaseSettings(
            sqlite_path="data/database/test.sqlite3",
            hdf5_path="data/hdf5/test.h5",
            export_root="data/exported",
            log_root="data/logs",
        ),
    )
    settings.ensure_runtime_dirs()
    return settings


def test_insert_and_fetch_spectrum(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    spectrum_id = database.insert_spectrum(
        {
            "spectrum_id": "test-spectrum",
            "sample_id": "sample-001",
            "source_wip": str(tmp_path / "demo.wip"),
            "source_tree_path": "/WITioRaw/Point/001/demo",
            "spectrum_type": "PL",
            "x_axis_unit": "nm",
            "csv_path": str(tmp_path / "demo.csv"),
            "source": "MoS2",
        },
        [650.0, 651.0, 652.0],
        [0.1, 0.5, 0.2],
    )
    detail = database.get_spectrum(spectrum_id)
    assert detail is not None
    assert detail["sample_id"] == "sample-001"
    assert detail["source"] == "MoS2"
    assert detail["file_path"] == str(tmp_path / "demo.wip")
    assert detail["x_axis"] == [650.0, 651.0, 652.0]
    assert detail["intensity"] == [0.1, 0.5, 0.2]


def test_update_spectrum_analysis_summary_updates_entry_fields(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    spectrum_id = database.insert_spectrum(
        {
            "spectrum_id": "analysis-summary-spectrum",
            "source_wip": str(tmp_path / "demo.wip"),
            "source_tree_path": "/WITioRaw/Point/001/demo",
            "spectrum_type": "PL",
            "x_axis_unit": "nm",
            "source": "unknown",
        },
        [740.0, 750.0, 760.0],
        [0.1, 0.5, 0.2],
    )

    updated = database.update_spectrum_analysis_summary(
        spectrum_id,
        {
            "material": "WSe2",
            "spectrum_family": "PL",
            "method_version": "material-aware-v1",
            "material_confidence": 0.88,
            "features": {"A_to_B_intensity_ratio": 2.1},
            "fit": {"r2": 0.95},
            "peaks": [],
        },
        method="material-aware-v1",
    )

    detail = database.get_spectrum(spectrum_id)
    assert updated == 1
    assert detail is not None
    assert detail["source"] == "WSe2"
    assert detail["analysis_material"] == "WSe2"
    assert detail["analysis_family"] == "PL"
    assert detail["analysis_summary"]["features"]["A_to_B_intensity_ratio"] == 2.1


def test_list_spectra_hides_mock_rows_by_default(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    source_file = tmp_path / "real_input.wip"

    database.insert_spectrum(
        {
            "spectrum_id": "mock-like-spectrum",
            "source_wip": str(source_file),
            "source_tree_path": "/Project/Data/Graph/0",
            "x_axis_unit": "nm",
            "n_points": 3,
        },
        [650.0, 651.0, 652.0],
        [0.1, 0.2, 0.3],
    )
    database.insert_spectrum(
        {
            "spectrum_id": "real-spectrum",
            "source_wip": str(source_file),
            "source_tree_path": "/WITioRaw/Point/001/demo",
            "x_axis_unit": "nm",
            "n_points": 3,
        },
        [700.0, 701.0, 702.0],
        [0.3, 0.4, 0.5],
    )

    assert database.list_spectra(limit=10)["total"] == 1
    assert database.list_spectra(limit=10)["items"][0]["spectrum_id"] == "real-spectrum"
    assert database.list_spectra(limit=10, include_mock=True)["total"] == 2


def test_insert_spectrum_normalizes_source_path(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    source_file = tmp_path / "nested" / "demo.wip"
    source_file.parent.mkdir(parents=True, exist_ok=True)

    spectrum_id = database.insert_spectrum(
        {
            "spectrum_id": "normalized-spectrum",
            "source_wip": source_file.as_posix(),
            "source_tree_path": "/WITioRaw/Point/001/demo",
            "x_axis_unit": "nm",
            "n_points": 3,
        },
        [710.0, 711.0, 712.0],
        [0.5, 0.6, 0.7],
    )

    detail = database.get_spectrum(spectrum_id)
    assert detail is not None
    assert detail["file_path"] == str(source_file)
    assert detail["acquisition_mode"] == "point_spectrum"


def test_belonging_is_inferred_and_exposed_in_filters(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    source_file = tmp_path / "User" / "Alice" / "batch" / "demo.wip"
    source_file.parent.mkdir(parents=True, exist_ok=True)

    spectrum_id = database.insert_spectrum(
        {
            "spectrum_id": "belonging-spectrum",
            "source_wip": str(source_file),
            "source_tree_path": "/WITioRaw/Point/001/demo",
            "x_axis_unit": "nm",
            "n_points": 3,
        },
        [710.0, 711.0, 712.0],
        [0.5, 0.6, 0.7],
    )

    detail = database.get_spectrum(spectrum_id)
    assert detail is not None
    assert detail["belonging"] == "Alice"

    listing = database.list_spectra(limit=10, include_mock=True, filters={"belonging": "Alice"})
    assert listing["total"] == 1
    assert listing["items"][0]["belonging"] == "Alice"

    options = database.list_filter_options(include_mock=True)
    assert "Alice" in options["belonging"]


def test_line_scan_rows_are_grouped_in_database_listing(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    source_file = tmp_path / "scan_demo.wip"

    for trace_index in range(3):
        database.insert_spectrum(
            {
                "spectrum_id": f"line-trace-{trace_index}",
                "source_wip": str(source_file),
                "source_tree_path": f"/WITioRaw/line_scan/001/Line_A/trace-{trace_index:04d}",
                "spectrum_type": "PL",
                "acquisition_mode": "line_scan",
                "x_axis_unit": "nm",
                "trace_index": trace_index,
                "trace_count": 3,
                "scan_size_x": 3,
                "scan_size_y": 1,
                "grid_x": trace_index,
                "grid_y": 0,
                "n_points": 3,
            },
            [700.0, 701.0, 702.0],
            [0.1 + trace_index, 0.2 + trace_index, 0.3 + trace_index],
        )

    listing = database.list_spectra(limit=10, include_mock=True)
    assert listing["total"] == 1
    item = listing["items"][0]
    assert item["acquisition_mode"] == "line_scan"
    assert item["member_count"] == 3

    detail = database.get_spectrum(item["spectrum_id"])
    assert detail is not None
    assert detail["member_count"] == 3
    assert detail["preview_series"] is not None
    assert len(detail["preview_series"]) == 3


def test_list_spectra_filters_and_sorts_grouped_rows(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    small_id = database.insert_spectrum(
        {
            "spectrum_id": "small-point",
            "sample_id": "sample-small",
            "source_wip": str(tmp_path / "small.wip"),
            "source_tree_path": "/WITioRaw/Point/001/small",
            "spectrum_type": "PL",
            "x_axis_unit": "nm",
            "n_points": 3,
            "source": "MoS2",
        },
        [650.0, 651.0, 652.0],
        [0.1, 0.2, 0.3],
    )
    large_id = database.insert_spectrum(
        {
            "spectrum_id": "large-point",
            "sample_id": "sample-large",
            "source_wip": str(tmp_path / "large.wip"),
            "source_tree_path": "/WITioRaw/Point/001/large",
            "spectrum_type": "Raman",
            "x_axis_unit": "cm^-1",
            "n_points": 7,
            "source": "WSe2",
        },
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0],
        [0.1, 0.4, 0.8, 0.6, 0.3, 0.2, 0.1],
    )
    for trace_index in range(3):
        database.insert_spectrum(
            {
                "spectrum_id": f"group-trace-{trace_index}",
                "sample_id": "sample-grouped",
                "source_wip": str(tmp_path / "grouped.wip"),
                "source_tree_path": f"/WITioRaw/line_scan/001/Line_A/trace-{trace_index:04d}",
                "spectrum_type": "PL",
                "acquisition_mode": "line_scan",
                "x_axis_unit": "nm",
                "trace_index": trace_index,
                "trace_count": 3,
                "scan_size_x": 3,
                "scan_size_y": 1,
                "grid_x": trace_index,
                "grid_y": 0,
                "n_points": 5,
                "source": "WS2",
            },
            [700.0, 701.0, 702.0, 703.0, 704.0],
            [0.1 + trace_index, 0.2 + trace_index, 0.3 + trace_index, 0.4 + trace_index, 0.5 + trace_index],
        )

    database.update_spectrum_analysis_summary(
        large_id,
        {"material": "WSe2", "spectrum_family": "Raman", "method_version": "test-v1", "peaks": []},
        method="test-v1",
    )

    numeric_filtered = database.list_spectra(
        include_mock=True,
        filters={"n_points_min": 5, "n_points_max": 7, "member_count_min": 2},
        limit=10,
    )
    assert numeric_filtered["total"] == 1
    assert numeric_filtered["items"][0]["member_count"] == 3

    analysis_filtered = database.list_spectra(
        include_mock=True,
        filters={"analysis_material": "WSe2", "analysis_family": "Raman", "analysis_status": "processed"},
        limit=10,
    )
    assert analysis_filtered["total"] == 1
    assert analysis_filtered["items"][0]["representative_spectrum_id"] == large_id

    sorted_by_points = database.list_spectra(
        include_mock=True,
        limit=10,
        sort_by="n_points",
        sort_dir="asc",
    )
    assert [item["n_points"] for item in sorted_by_points["items"]] == [3, 5, 7]

    sorted_by_members = database.list_spectra(
        include_mock=True,
        limit=10,
        sort_by="member_count",
        sort_dir="desc",
    )
    assert sorted_by_members["items"][0]["member_count"] == 3

    source_filtered = database.list_spectra(include_mock=True, filters={"source": "MoS2"}, limit=10)
    sample_filtered = database.list_spectra(include_mock=True, filters={"sample_id": "sample-small"}, limit=10)
    assert source_filtered["total"] == 1
    assert source_filtered["items"][0]["representative_spectrum_id"] == small_id
    assert sample_filtered["total"] == 1
    assert sample_filtered["items"][0]["representative_spectrum_id"] == small_id


def test_area_map_rows_are_grouped_with_preview_grid(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    source_file = tmp_path / "map_demo.wip"

    trace_index = 0
    for grid_y in range(2):
        for grid_x in range(2):
            database.insert_spectrum(
                {
                    "spectrum_id": f"area-trace-{trace_index}",
                    "source_wip": str(source_file),
                    "source_tree_path": f"/WITioRaw/area_map/001/Map_A/trace-{trace_index:04d}",
                    "spectrum_type": "PL",
                    "acquisition_mode": "area_map",
                    "x_axis_unit": "nm",
                    "trace_index": trace_index,
                    "trace_count": 4,
                    "scan_size_x": 2,
                    "scan_size_y": 2,
                    "grid_x": grid_x,
                    "grid_y": grid_y,
                    "n_points": 3,
                },
                [700.0, 701.0, 702.0],
                [1.0 + trace_index, 2.0 + trace_index, 3.0 + trace_index],
            )
            trace_index += 1

    listing = database.list_spectra(limit=10, include_mock=True)
    assert listing["total"] == 1
    item = listing["items"][0]
    assert item["acquisition_mode"] == "area_map"
    assert item["member_count"] == 4

    detail = database.get_spectrum(item["spectrum_id"])
    assert detail is not None
    assert detail["member_count"] == 4
    assert detail["preview_grid"] is not None
    assert len(detail["preview_grid"]) == 2
    assert len(detail["preview_grid"][0]) == 2


def test_database_migrates_legacy_spectrum_ids_to_hashed_ids(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    source_file = tmp_path / "20260127-GrayScale.wip"
    source_tree_path = "/WITioRaw/point/0062/Spectrum--553--Spec.Data_1"

    legacy_id = database.insert_spectrum(
        {
            "spectrum_id": "20260127-GrayScale_007_003817",
            "sample_id": "20260127",
            "source_wip": str(source_file),
            "source_tree_path": source_tree_path,
            "x_axis_unit": "eV",
            "trace_index": 0,
            "source": "GrayScale",
        },
        [1.5, 1.6, 1.7],
        [10.0, 11.0, 12.0],
    )
    assert legacy_id == "20260127-GrayScale_007_003817"

    database = DatabaseService(settings)
    migrated_id = build_spectrum_id(str(source_file), source_tree_path, 0)

    detail = database.get_spectrum(migrated_id)
    assert detail is not None
    assert detail["spectrum_id"] == migrated_id
    assert detail["source"] == "20260127-GrayScale"
    assert detail["file_path"] == str(source_file)
    assert database.get_spectrum(legacy_id) is None


def test_dashboard_counts_duplicate_file_hash_once(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)

    database.insert_spectrum(
        {
            "spectrum_id": "hash-spectrum-a",
            "dataset_id": "dataset-a",
            "source_wip": str(tmp_path / "copy_a.wip"),
            "source_file_hash": "same-hash",
            "source_tree_path": "/WITioRaw/Point/001/demo-a",
            "x_axis_unit": "nm",
            "measurement_time": "2026-07-01T10:00:00+00:00",
            "n_points": 3,
        },
        [700.0, 701.0, 702.0],
        [0.1, 0.2, 0.3],
    )
    database.insert_spectrum(
        {
            "spectrum_id": "hash-spectrum-b",
            "dataset_id": "dataset-b",
            "source_wip": str(tmp_path / "copy_b.wip"),
            "source_file_hash": "same-hash",
            "source_tree_path": "/WITioRaw/Point/001/demo-b",
            "x_axis_unit": "nm",
            "measurement_time": "2026-07-02T10:00:00+00:00",
            "n_points": 3,
        },
        [703.0, 704.0, 705.0],
        [0.4, 0.5, 0.6],
    )

    dashboard = database.dashboard_summary(include_mock=True)
    assert dashboard["imported_files"] == 1
    assert len(dashboard["measurement_timeline"]) == 2


def test_delete_spectra_for_source_hash_removes_all_matching_rows(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)

    for index in range(2):
        database.insert_spectrum(
            {
                "spectrum_id": f"duplicate-hash-{index}",
                "dataset_id": f"duplicate-dataset-{index}",
                "source_wip": str(tmp_path / f"copy_{index}.wip"),
                "source_file_hash": "delete-me",
                "source_tree_path": f"/WITioRaw/Point/001/demo-{index}",
                "x_axis_unit": "nm",
                "n_points": 3,
            },
            [710.0, 711.0, 712.0],
            [0.5, 0.6, 0.7],
        )

    assert database.count_spectra_for_source_hash("delete-me") == 2
    assert database.delete_spectra_for_source_hash("delete-me") == 2
    assert database.count_spectra_for_source_hash("delete-me") == 0


def test_insert_media_asset_counts_toward_imported_files(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    source_file = tmp_path / "camera_only.wip"
    image_bytes = b"\xff\xd8\xff\xd9"

    media_id = database.insert_media_asset(
        {
            "media_id": build_media_id(str(source_file), "/WITioRaw/photo/0001/demo"),
            "source_wip": str(source_file),
            "source_file_hash": "camera-hash",
            "source_tree_path": "/WITioRaw/photo/0001/demo",
            "media_kind": "photo_image",
            "entry_class": "TDImage",
            "caption": "demo",
            "asset_format": "jpeg",
            "width_px": 480,
            "height_px": 360,
            "original_width_px": 1600,
            "original_height_px": 1200,
            "channel_count": 1,
            "bit_depth": 8,
            "measurement_config": {"caption": "demo"},
        },
        image_bytes,
    )

    assert media_id.startswith("plmedia-")
    assert database.count_media_assets_for_source_hash("camera-hash") == 1
    assert database.count_import_records_for_source_hash("camera-hash") == 1
    assert database.dashboard_summary(include_mock=True)["imported_files"] == 1


def test_list_imported_upload_history_uses_relative_upload_paths(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)

    database.upsert_import_job(
        {
            "job_id": "job-history-1",
            "input_path": str(tmp_path / "data" / "raw_wip" / "upload-demo"),
            "status": "finished",
            "total_files": 1,
            "processed_files": 1,
            "exported_spectra": 12,
            "failed_files": 0,
            "current_file": None,
            "start_time": "2026-07-03T15:00:00+00:00",
            "end_time": "2026-07-03T15:01:00+00:00",
            "log_path": None,
            "summary_path": None,
            "message": "Imported 12 spectra from 1 file",
            "details": {
                "source_kind": "folder_upload",
                "display_input_path": "Selected folder item: WPJ/20250912 SnSe transfer/SiO2 transfer.wip",
                "relative_input_path": "WPJ/20250912 SnSe transfer/SiO2 transfer.wip",
                "result_summary": {
                    "imported_spectra": 12,
                    "imported_media_assets": 0,
                },
            },
        }
    )

    history = database.list_imported_upload_history(root_name="WPJ")

    assert len(history) == 1
    assert history[0]["relative_path"] == "WPJ/20250912 SnSe transfer/SiO2 transfer.wip"
    assert history[0]["imported_spectra"] == 12
