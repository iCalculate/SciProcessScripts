from __future__ import annotations

import time
from pathlib import Path
from types import MethodType

from backend.config import AppSettings, DatabaseSettings, ImporterSettings
from backend.services.database import DatabaseService
from backend.services.import_bridge import ImportBridge
from backend.services.importer import ImportManager


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


def test_mock_import_pipeline(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    bridge = ImportBridge(settings, database)
    manager = ImportManager(settings, database, bridge)

    job = manager.start_job(
        input_path="mock://demo",
        recursive=True,
        force_reimport=True,
        import_options={"include_point_spectra": True},
    )
    deadline = time.time() + 10
    while time.time() < deadline:
        snapshot = manager.get_job(job["job_id"])
        assert snapshot is not None
        if snapshot["status"] not in {"pending", "running"}:
            break
        time.sleep(0.1)
    snapshot = manager.get_job(job["job_id"])
    assert snapshot is not None
    assert snapshot["status"] in {"finished", "partially_failed"}
    assert snapshot["details"]["detected_inventory"]["point_spectrum"] >= 1
    assert snapshot["details"]["result_summary"]["imported_dataset_count"] >= 1
    assert snapshot["details"]["result_summary"]["imported_spectra"] >= 1
    assert database.list_spectra(limit=50)["total"] == 0
    spectra = database.list_spectra(limit=50, include_mock=True)
    assert spectra["total"] >= 1
    assert not list(settings.export_root.rglob("*.csv"))


def test_existing_real_file_uses_python_import_path_even_when_mock_mode_is_enabled(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    settings.importer.mock_mode = True
    database = DatabaseService(settings)
    bridge = ImportBridge(settings, database)
    real_file = tmp_path / "real_input.wip"
    real_file.write_text("placeholder", encoding="utf-8")

    def fake_run_witio_import(
        self: ImportBridge,
        *,
        job_id,
        files,
        input_path,
        output_root,
        log_path,
        force_reimport,
        import_options,
        progress_callback,
        spectrum_callback,
        media_callback,
        should_stop,
    ):
        return {
            "summary_path": str(output_root / "import_summary.json"),
            "log_path": str(log_path),
            "summary": {
                "input_path": str(input_file),
                "mock_mode": False,
                "total_files": 1,
                "processed_files": 1,
                "exported_spectra": 0,
                "failed_files": 0,
                "skipped_files": 0,
                "files": [],
            },
        }

    input_file = real_file
    bridge._run_witio_import = MethodType(fake_run_witio_import, bridge)
    result = bridge.export_input(
        input_path=str(real_file),
        job_id="job-real",
        recursive=False,
        force_reimport=True,
    )
    assert result["summary"]["mock_mode"] is False
    assert result["summary"]["input_path"] == str(real_file)


def test_failed_upload_is_recorded_as_terminal_import_job(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    bridge = ImportBridge(settings, database)
    manager = ImportManager(settings, database, bridge)

    job = manager.record_failed_upload(
        input_path="Nexstrom/Frank/bad_remote_file.wip",
        message="Upload staging failed: network path disappeared",
        import_options={"include_point_spectra": True},
        details={
            "display_input_path": "Selected folder item: Nexstrom/Frank/bad_remote_file.wip",
            "relative_input_path": "Nexstrom/Frank/bad_remote_file.wip",
            "source_kind": "folder_upload",
        },
    )

    persisted = database.get_import_job(job["job_id"])
    assert persisted is not None
    assert persisted["status"] == "failed"
    assert persisted["processed_files"] == 1
    assert persisted["failed_files"] == 1
    assert persisted["details"]["phase"] == "failed"
    assert persisted["details"]["source_kind"] == "folder_upload"
    assert persisted["details"]["result_summary"]["failed_file_count"] == 1
    assert "network path disappeared" in str(persisted["details"]["single_file_summary"]["error_message"])


def test_import_jobs_are_dispatched_through_single_worker_queue(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    database = DatabaseService(settings)
    bridge = ImportBridge(settings, database)
    manager = ImportManager(settings, database, bridge)

    first = manager.start_job(
        input_path="mock://first",
        recursive=True,
        force_reimport=True,
        import_options={"include_point_spectra": True},
    )
    second = manager.start_job(
        input_path="mock://second",
        recursive=True,
        force_reimport=True,
        import_options={"include_point_spectra": True},
    )

    assert set(manager._threads) == {"worker"}

    deadline = time.time() + 10
    while time.time() < deadline:
        snapshots = [manager.get_job(first["job_id"]), manager.get_job(second["job_id"])]
        assert all(snapshot is not None for snapshot in snapshots)
        if all(snapshot["status"] not in {"pending", "running"} for snapshot in snapshots if snapshot):
            break
        time.sleep(0.1)

    assert manager.get_job(first["job_id"])["status"] in {"finished", "partially_failed"}
    assert manager.get_job(second["job_id"])["status"] in {"finished", "partially_failed"}
