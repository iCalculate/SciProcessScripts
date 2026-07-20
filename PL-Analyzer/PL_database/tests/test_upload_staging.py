from __future__ import annotations

from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from backend.config import AppSettings, DatabaseSettings, ImporterSettings
from backend.services.upload_staging import sanitize_upload_relative_path, stage_upload_files


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


def test_stage_upload_files_preserves_relative_structure_and_filters_non_wip(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    files = [
        UploadFile(file=BytesIO(b"first"), filename="demo_a.wip"),
        UploadFile(file=BytesIO(b"second"), filename="demo_b.WIP"),
        UploadFile(file=BytesIO(b"skip"), filename="notes.txt"),
    ]

    staged_root, uploaded_count = stage_upload_files(
        settings,
        files=files,
        relative_paths=[
            "Parent/Batch_A/demo_a.wip",
            "Parent/Batch_B/demo_b.WIP",
            "Parent/readme.txt",
        ],
        root_name="Parent",
    )

    assert uploaded_count == 2
    assert (staged_root / "Parent" / "Batch_A" / "demo_a.wip").read_bytes() == b"first"
    assert (staged_root / "Parent" / "Batch_B" / "demo_b.WIP").read_bytes() == b"second"
    assert not (staged_root / "Parent" / "readme.txt").exists()


def test_sanitize_upload_relative_path_rejects_parent_traversal() -> None:
    try:
        sanitize_upload_relative_path("../escape/demo.wip", default_name="demo.wip")
    except ValueError as error:
        assert "parent folders" in str(error)
    else:
        raise AssertionError("Expected parent traversal to be rejected")
