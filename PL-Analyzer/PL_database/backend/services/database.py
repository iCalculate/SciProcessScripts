from __future__ import annotations

import csv
import json
import sqlite3
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from ..config import AppSettings
from .file_fingerprint import compute_file_sha256
from .metadata_parser import (
    build_dataset_id,
    build_media_id,
    build_spectrum_id,
    infer_dataset_label_from_tree_path,
    infer_belonging_from_path,
    infer_legacy_source_label_from_path,
    infer_spectrum_type_with_context,
    is_mock_source,
    normalize_dataset_tree_path,
    normalize_source_path,
    parse_metadata_from_path,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _maybe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_iso_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text[: len(fmt)], fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _auto_timeline_granularity(day_span: int) -> str:
    if day_span <= 120:
        return "day"
    if day_span <= 730:
        return "week"
    return "month"


def _bucket_date(value: date, granularity: str) -> date:
    if granularity == "week":
        return value - timedelta(days=value.weekday())
    if granularity == "month":
        return value.replace(day=1)
    return value


def _coerce_number(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _matches_numeric_ranges(
    item: dict[str, object],
    filters: dict[str, object | None],
    fields: list[str],
) -> bool:
    for field in fields:
        value = _coerce_number(item.get(field))
        minimum = _coerce_number(filters.get(f"{field}_min"))
        maximum = _coerce_number(filters.get(f"{field}_max"))
        if minimum is not None and (value is None or value < minimum):
            return False
        if maximum is not None and (value is None or value > maximum):
            return False
    return True


def _sort_grouped_spectra(
    items: list[dict[str, object]],
    *,
    sort_by: str | None,
    sort_dir: str | None,
) -> list[dict[str, object]]:
    sort_key_map = {
        "spectrum_id": "dataset_id",
        "sample_id": "sample_id",
        "belonging": "belonging",
        "spectrum_type": "spectrum_type",
        "acquisition_mode": "acquisition_mode",
        "source": "material",
        "analysis_material": "analysis_material",
        "analysis_family": "analysis_family",
        "analysis_status": "analysis_status",
        "member_count": "member_count",
        "x_axis_unit": "x_axis_unit",
        "n_points": "n_points",
        "trace_count": "trace_count",
        "scan_size_x": "scan_size_x",
        "scan_size_y": "scan_size_y",
        "grid_x": "grid_x",
        "grid_y": "grid_y",
        "measurement_time": "measurement_time",
        "file_path": "source_wip",
        "source_tree_path": "source_tree_path",
    }
    numeric_fields = {
        "member_count",
        "n_points",
        "trace_count",
        "scan_size_x",
        "scan_size_y",
        "grid_x",
        "grid_y",
    }
    normalized_sort_by = str(sort_by or "").strip()
    if not normalized_sort_by:
        return sorted(
            items,
            key=lambda item: (str(item.get("_sort_key") or ""), str(item.get("dataset_id") or "")),
            reverse=True,
        )
    field = sort_key_map.get(normalized_sort_by)
    if field is None:
        return sorted(
            items,
            key=lambda item: (str(item.get("_sort_key") or ""), str(item.get("dataset_id") or "")),
            reverse=True,
        )

    descending = str(sort_dir or "").lower() == "desc"

    def value_for(item: dict[str, object]) -> object:
        if normalized_sort_by in numeric_fields:
            return _coerce_number(item.get(field))
        value = item.get(field)
        return str(value).casefold() if value not in (None, "") else None

    present = [item for item in items if value_for(item) is not None]
    missing = [item for item in items if value_for(item) is None]
    return sorted(
        present,
        key=lambda item: (value_for(item), str(item.get("dataset_id") or "")),
        reverse=descending,
    ) + missing


class DatabaseService:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self._hdf5_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._data_revision = 0
        self._filter_options_cache: dict[bool, tuple[int, dict[str, list[str]]]] = {}
        self._upload_history_cache: dict[str, tuple[int, list[dict[str, object]]]] = {}
        self._maintenance_thread: threading.Thread | None = None
        self._maintenance_started = False
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.settings.sqlite_path, check_same_thread=False, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    @contextmanager
    def bulk_import_context(self):
        connection = self._connect()
        self._hdf5_lock.acquire()
        handle = h5py.File(self.settings.hdf5_path, "a")
        try:
            yield connection, handle
            connection.commit()
        finally:
            handle.close()
            self._hdf5_lock.release()
            connection.close()

    def initialize(self) -> None:
        self.settings.ensure_runtime_dirs()
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    sample_id TEXT PRIMARY KEY,
                    material TEXT,
                    substrate TEXT,
                    device_id TEXT,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS spectra_index (
                    spectrum_id TEXT PRIMARY KEY,
                    dataset_id TEXT,
                    dataset_label TEXT,
                    sample_id TEXT,
                    source_wip TEXT NOT NULL,
                    source_file_hash TEXT,
                    source_tree_path TEXT NOT NULL,
                    spectrum_type TEXT,
                    acquisition_mode TEXT,
                    x_axis_unit TEXT,
                    n_points INTEGER NOT NULL,
                    h5_path TEXT NOT NULL,
                    h5_group TEXT NOT NULL,
                    csv_path TEXT,
                    import_time TEXT NOT NULL,
                    measurement_time TEXT,
                    measurement_config_json TEXT,
                    trace_index INTEGER,
                    trace_count INTEGER,
                    trace_preview_value REAL,
                    scan_size_x INTEGER,
                    scan_size_y INTEGER,
                    grid_x INTEGER,
                    grid_y INTEGER,
                    laser_wavelength TEXT,
                    laser_power TEXT,
                    integration_time TEXT,
                    grating TEXT,
                    objective TEXT,
                    material TEXT,
                    analysis_material TEXT,
                    analysis_family TEXT,
                    analysis_status TEXT,
                    analysis_method_version TEXT,
                    analysis_summary_json TEXT,
                    analysis_updated_at TEXT,
                    belonging TEXT,
                    substrate TEXT,
                    device_id TEXT,
                    notes TEXT,
                    folder_path TEXT,
                    FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
                );

                CREATE TABLE IF NOT EXISTS media_assets (
                    media_id TEXT PRIMARY KEY,
                    source_wip TEXT NOT NULL,
                    source_file_hash TEXT,
                    source_tree_path TEXT NOT NULL,
                    media_kind TEXT NOT NULL,
                    entry_class TEXT NOT NULL,
                    caption TEXT,
                    asset_format TEXT NOT NULL,
                    h5_path TEXT NOT NULL,
                    h5_group TEXT NOT NULL,
                    width_px INTEGER NOT NULL,
                    height_px INTEGER NOT NULL,
                    original_width_px INTEGER,
                    original_height_px INTEGER,
                    channel_count INTEGER NOT NULL,
                    bit_depth INTEGER NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    import_time TEXT NOT NULL,
                    measurement_time TEXT,
                    measurement_config_json TEXT,
                    folder_path TEXT
                );

                CREATE TABLE IF NOT EXISTS import_jobs (
                    job_id TEXT PRIMARY KEY,
                    input_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_files INTEGER NOT NULL DEFAULT 0,
                    processed_files INTEGER NOT NULL DEFAULT 0,
                    exported_spectra INTEGER NOT NULL DEFAULT 0,
                    failed_files INTEGER NOT NULL DEFAULT 0,
                    current_file TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    log_path TEXT,
                    summary_path TEXT,
                    message TEXT,
                    details_json TEXT
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    result_id TEXT PRIMARY KEY,
                    spectrum_id TEXT NOT NULL,
                    method TEXT NOT NULL,
                    parameters_json TEXT NOT NULL,
                    peaks_json TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    fit_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(spectrum_id) REFERENCES spectra_index(spectrum_id)
                );
                """
            )
            self._ensure_schema_columns(connection)
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_source_file_hash ON spectra_index(source_file_hash)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_dataset_id ON spectra_index(dataset_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_measurement_time ON spectra_index(measurement_time)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_import_time ON spectra_index(import_time)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_spectrum_type ON spectra_index(spectrum_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_material ON spectra_index(material)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_belonging ON spectra_index(belonging)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_acquisition_mode ON spectra_index(acquisition_mode)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_substrate ON spectra_index(substrate)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_x_axis_unit ON spectra_index(x_axis_unit)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_spectra_sample_id ON spectra_index(sample_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_assets_source_file_hash ON media_assets(source_file_hash)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_media_assets_source_wip ON media_assets(source_wip)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_import_jobs_status_end_time ON import_jobs(status, end_time, start_time)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_import_jobs_start_time ON import_jobs(start_time DESC)"
            )

        if not self.settings.hdf5_path.exists():
            with h5py.File(self.settings.hdf5_path, "w") as handle:
                handle.require_group("spectra")
        self._migrate_spectrum_ids_to_hashes()
        self._finalize_stale_import_jobs()

    def start_background_maintenance(self) -> None:
        if self._maintenance_started:
            return
        self._maintenance_started = True
        self._maintenance_thread = threading.Thread(
            target=self._run_background_maintenance,
            name="pldb-startup-maintenance",
            daemon=True,
        )
        self._maintenance_thread.start()

    def _run_background_maintenance(self) -> None:
        time.sleep(10)
        self._run_startup_maintenance("normalize_existing_paths", self._normalize_existing_paths)
        self._run_startup_maintenance("backfill_derived_metadata", self._backfill_derived_metadata)

    def _run_startup_maintenance(self, label: str, action) -> None:
        try:
            action()
        except sqlite3.OperationalError as error:
            message = str(error).lower()
            if "database is locked" in message or "database schema is locked" in message:
                print(f"[database] startup maintenance skipped ({label}): {error}")
                return
            raise
        except Exception as error:
            print(f"[database] startup maintenance failed ({label}): {error}")

    def _invalidate_data_cache(self) -> None:
        with self._cache_lock:
            self._data_revision += 1
            self._filter_options_cache.clear()
            self._upload_history_cache.clear()

    def _finalize_stale_import_jobs(self) -> None:
        now = utc_now()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT job_id, message, details_json
                FROM import_jobs
                WHERE status IN ('pending', 'running') AND end_time IS NULL
                """
            ).fetchall()
            for row in rows:
                details = self._deserialize_measurement_config(row["details_json"])
                legacy_phase = details.get("phase")
                details["phase"] = "interrupted"
                if legacy_phase:
                    details["legacy_phase"] = legacy_phase
                message = str(row["message"] or "").strip()
                if not message or "MATLAB" in message or "export pipeline" in message.lower():
                    message = "Import interrupted before backend restart"
                connection.execute(
                    """
                    UPDATE import_jobs
                    SET status = ?, end_time = ?, message = ?, details_json = ?
                    WHERE job_id = ?
                    """,
                    [
                        "failed",
                        now,
                        message,
                        json.dumps(details, ensure_ascii=False),
                        row["job_id"],
                    ],
                )

    def _ensure_schema_columns(self, connection: sqlite3.Connection) -> None:
        spectra_columns = {
            "dataset_id": "TEXT",
            "dataset_label": "TEXT",
            "source_file_hash": "TEXT",
            "acquisition_mode": "TEXT",
            "measurement_time": "TEXT",
            "measurement_config_json": "TEXT",
            "trace_index": "INTEGER",
            "trace_count": "INTEGER",
            "trace_preview_value": "REAL",
            "scan_size_x": "INTEGER",
            "scan_size_y": "INTEGER",
            "grid_x": "INTEGER",
            "grid_y": "INTEGER",
            "belonging": "TEXT",
            "analysis_material": "TEXT",
            "analysis_family": "TEXT",
            "analysis_status": "TEXT",
            "analysis_method_version": "TEXT",
            "analysis_summary_json": "TEXT",
            "analysis_updated_at": "TEXT",
        }
        import_job_columns = {
            "details_json": "TEXT",
        }
        media_columns = {
            "source_file_hash": "TEXT",
            "caption": "TEXT",
            "measurement_time": "TEXT",
            "measurement_config_json": "TEXT",
            "original_width_px": "INTEGER",
            "original_height_px": "INTEGER",
            "folder_path": "TEXT",
        }

        existing_spectra = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(spectra_index)").fetchall()
        }
        for column_name, column_type in spectra_columns.items():
            if column_name not in existing_spectra:
                connection.execute(f"ALTER TABLE spectra_index ADD COLUMN {column_name} {column_type}")

        existing_jobs = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(import_jobs)").fetchall()
        }
        for column_name, column_type in import_job_columns.items():
            if column_name not in existing_jobs:
                connection.execute(f"ALTER TABLE import_jobs ADD COLUMN {column_name} {column_type}")

        existing_media = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(media_assets)").fetchall()
        }
        for column_name, column_type in media_columns.items():
            if column_name not in existing_media:
                connection.execute(f"ALTER TABLE media_assets ADD COLUMN {column_name} {column_type}")

    def _normalize_existing_paths(self) -> None:
        with self._connection() as connection:
            spectrum_rows = connection.execute(
                "SELECT spectrum_id, source_wip, source_tree_path, folder_path, material, belonging, dataset_id, dataset_label FROM spectra_index"
            ).fetchall()
            for row in spectrum_rows:
                normalized_source = normalize_source_path(str(row["source_wip"]))
                normalized_folder = row["folder_path"]
                if normalized_source != row["source_wip"] and not is_mock_source(normalized_source):
                    normalized_folder = str(Path(normalized_source).parent)
                source_from_name = parse_metadata_from_path(normalized_source).get("source")
                legacy_source = infer_legacy_source_label_from_path(normalized_source)
                inferred_belonging = infer_belonging_from_path(normalized_source)
                normalized_material = row["material"]
                if source_from_name and (not normalized_material or normalized_material == legacy_source):
                    normalized_material = source_from_name
                normalized_tree = normalize_dataset_tree_path(str(row["source_tree_path"]))
                dataset_id = row["dataset_id"] or build_dataset_id(normalized_source, normalized_tree)
                dataset_label = row["dataset_label"] or infer_dataset_label_from_tree_path(normalized_tree)
                if (
                    normalized_source != row["source_wip"]
                    or normalized_folder != row["folder_path"]
                    or normalized_material != row["material"]
                    or row["belonging"] != inferred_belonging
                    or row["dataset_id"] != dataset_id
                    or row["dataset_label"] != dataset_label
                ):
                    connection.execute(
                        """
                        UPDATE spectra_index
                        SET source_wip = ?, folder_path = ?, material = ?, belonging = ?, dataset_id = ?, dataset_label = ?
                        WHERE spectrum_id = ?
                        """,
                        [
                            normalized_source,
                            normalized_folder,
                            normalized_material,
                            inferred_belonging,
                            dataset_id,
                            dataset_label,
                            row["spectrum_id"],
                        ],
                    )

            media_rows = connection.execute(
                "SELECT media_id, source_wip, folder_path FROM media_assets"
            ).fetchall()
            for row in media_rows:
                normalized_source = normalize_source_path(str(row["source_wip"]))
                normalized_folder = row["folder_path"]
                if normalized_source != row["source_wip"] and not is_mock_source(normalized_source):
                    normalized_folder = str(Path(normalized_source).parent)
                if normalized_source != row["source_wip"] or normalized_folder != row["folder_path"]:
                    connection.execute(
                        """
                        UPDATE media_assets
                        SET source_wip = ?, folder_path = ?
                        WHERE media_id = ?
                        """,
                        [
                            normalized_source,
                            normalized_folder,
                            row["media_id"],
                        ],
                    )

            job_rows = connection.execute(
                "SELECT job_id, input_path, current_file FROM import_jobs"
            ).fetchall()
            for row in job_rows:
                input_path = normalize_source_path(str(row["input_path"]))
                current_file = row["current_file"]
                normalized_current = normalize_source_path(str(current_file)) if current_file else current_file
                if input_path != row["input_path"] or normalized_current != current_file:
                    connection.execute(
                        "UPDATE import_jobs SET input_path = ?, current_file = ? WHERE job_id = ?",
                        [input_path, normalized_current, row["job_id"]],
                    )

    def _backfill_source_file_hashes(self) -> None:
        with self._connection() as connection:
            source_rows = connection.execute(
                """
                SELECT DISTINCT source_wip
                FROM (
                    SELECT source_wip
                    FROM spectra_index
                    WHERE COALESCE(source_file_hash, '') = ''
                    UNION
                    SELECT source_wip
                    FROM media_assets
                    WHERE COALESCE(source_file_hash, '') = ''
                )
                """
            ).fetchall()
            for row in source_rows:
                source_wip = normalize_source_path(str(row["source_wip"]))
                if not source_wip or is_mock_source(source_wip):
                    continue
                source_path = Path(source_wip)
                if not source_path.exists() or not source_path.is_file():
                    continue
                try:
                    source_hash = compute_file_sha256(source_path)
                except OSError:
                    continue
                connection.execute(
                    "UPDATE spectra_index SET source_file_hash = ? WHERE source_wip = ?",
                    [source_hash, source_wip],
                )
                connection.execute(
                    "UPDATE media_assets SET source_file_hash = ? WHERE source_wip = ?",
                    [source_hash, source_wip],
                )

    def _backfill_derived_metadata(self) -> None:
        if not self.settings.hdf5_path.exists():
            return

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT
                    spectrum_id, source_wip, source_tree_path, spectrum_type, acquisition_mode,
                    x_axis_unit, h5_group, material, belonging, measurement_config_json, dataset_id, dataset_label, grating
                FROM spectra_index
                """
            ).fetchall()

            with h5py.File(self.settings.hdf5_path, "r") as handle:
                for row in rows:
                    updates: dict[str, object] = {}
                    parsed = parse_metadata_from_path(str(row["source_wip"]))
                    source_from_name = parsed.get("source")
                    inferred_belonging = parsed.get("belonging") or infer_belonging_from_path(str(row["source_wip"]))
                    legacy_source = infer_legacy_source_label_from_path(str(row["source_wip"]))
                    if source_from_name and (not row["material"] or row["material"] == legacy_source):
                        updates["material"] = source_from_name
                    if inferred_belonging != row["belonging"]:
                        updates["belonging"] = inferred_belonging

                    normalized_tree = normalize_dataset_tree_path(str(row["source_tree_path"]))
                    dataset_id = row["dataset_id"] or build_dataset_id(str(row["source_wip"]), normalized_tree)
                    dataset_label = row["dataset_label"] or infer_dataset_label_from_tree_path(normalized_tree)
                    if row["dataset_id"] != dataset_id:
                        updates["dataset_id"] = dataset_id
                    if row["dataset_label"] != dataset_label:
                        updates["dataset_label"] = dataset_label

                    inferred_mode = self._infer_acquisition_mode_from_path(str(row["source_tree_path"]))
                    if inferred_mode != "unknown" and inferred_mode != row["acquisition_mode"]:
                        updates["acquisition_mode"] = inferred_mode

                    if row["h5_group"] in handle:
                        x_axis = handle[str(row["h5_group"])]["x_axis"][:].astype(float).tolist()
                        inferred_type = infer_spectrum_type_with_context(
                            x_axis,
                            row["x_axis_unit"],
                            grating=row["grating"],
                        )
                        if inferred_type != "unknown" and inferred_type != row["spectrum_type"]:
                            updates["spectrum_type"] = inferred_type

                    config = self._deserialize_measurement_config(row["measurement_config_json"])
                    if not config:
                        default_config = {
                            "acquisition_mode": updates.get("acquisition_mode", row["acquisition_mode"]),
                        }
                        updates["measurement_config_json"] = json.dumps(
                            {key: value for key, value in default_config.items() if value},
                            ensure_ascii=False,
                        )

                    if updates:
                        set_clause = ", ".join(f"{field} = ?" for field in updates)
                        connection.execute(
                            f"UPDATE spectra_index SET {set_clause} WHERE spectrum_id = ?",
                            [*updates.values(), row["spectrum_id"]],
                        )

    def _migrate_spectrum_ids_to_hashes(self) -> None:
        if not self.settings.hdf5_path.exists():
            return

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT spectrum_id, source_wip, source_tree_path, trace_index, h5_group, material
                FROM spectra_index
                WHERE spectrum_id < 'plspec-' OR spectrum_id >= 'plspec.'
                """
            ).fetchall()

            migrations: list[tuple[str, str, str, str, str | None]] = []
            for row in rows:
                old_spectrum_id = str(row["spectrum_id"])
                if old_spectrum_id.startswith("plspec-"):
                    continue
                normalized_source = str(row["source_wip"])
                new_spectrum_id = build_spectrum_id(
                    normalized_source,
                    str(row["source_tree_path"]),
                    int(row["trace_index"] or 0),
                )
                if new_spectrum_id == old_spectrum_id:
                    continue
                existing = connection.execute(
                    "SELECT 1 FROM spectra_index WHERE spectrum_id = ?",
                    [new_spectrum_id],
                ).fetchone()
                if existing is not None:
                    continue
                old_h5_group = str(row["h5_group"] or f"/spectra/{old_spectrum_id}")
                new_h5_group = f"/spectra/{new_spectrum_id}"
                parsed_source = parse_metadata_from_path(normalized_source).get("source")
                legacy_source = infer_legacy_source_label_from_path(normalized_source)
                material = row["material"]
                normalized_material = (
                    parsed_source
                    if parsed_source and (not material or material == legacy_source)
                    else material
                )
                migrations.append((old_spectrum_id, new_spectrum_id, old_h5_group, new_h5_group, normalized_material))

            if not migrations:
                return

            with h5py.File(self.settings.hdf5_path, "a") as handle:
                for old_spectrum_id, new_spectrum_id, old_h5_group, new_h5_group, normalized_material in migrations:
                    if old_h5_group in handle:
                        if new_h5_group in handle:
                            continue
                        handle.move(old_h5_group, new_h5_group)
                    elif new_h5_group not in handle:
                        continue

                    connection.execute(
                        "UPDATE analysis_results SET spectrum_id = ? WHERE spectrum_id = ?",
                        [new_spectrum_id, old_spectrum_id],
                    )
                    connection.execute(
                        """
                        UPDATE spectra_index
                        SET spectrum_id = ?, h5_group = ?, material = ?
                        WHERE spectrum_id = ?
                        """,
                        [new_spectrum_id, new_h5_group, normalized_material, old_spectrum_id],
                    )

    def _append_real_only_clause(self, clauses: list[str]) -> None:
        clauses.append(
            "("
            "lower(source_wip) NOT LIKE 'mock%' AND "
            "source_tree_path NOT LIKE 'mock://%' AND "
            "source_tree_path NOT LIKE '/Project/Data/Graph/%'"
            ")"
        )

    def _infer_acquisition_mode_from_path(self, source_tree_path: str) -> str:
        lowered = str(source_tree_path).lower()
        if any(token in lowered for token in ("/point/", "point_spectrum")):
            return "point_spectrum"
        if "series" in lowered or "time" in lowered or "power" in lowered:
            return "series_scan"
        if any(token in lowered for token in ("/line/", "line_scan")):
            return "line_scan"
        if any(token in lowered for token in ("/image/", "area_map", "mapping", "map/")):
            return "area_map"
        return "unknown"

    def _serialize_measurement_config(self, metadata: dict[str, object]) -> str:
        base_config = metadata.get("measurement_config")
        if isinstance(base_config, dict):
            config = dict(base_config)
        else:
            config = {}

        for key in [
            "measurement_time",
            "laser_wavelength",
            "laser_power",
            "integration_time",
            "grating",
            "objective",
            "acquisition_mode",
            "trace_index",
            "trace_count",
            "scan_size_x",
            "scan_size_y",
            "grid_x",
            "grid_y",
            "secondary_axis_kind",
            "secondary_axis_unit",
            "secondary_axis_value",
            "scan_label",
            "extraction_backend",
        ]:
            value = metadata.get(key)
            if value not in (None, "", []):
                config[key] = value

        return json.dumps(config, ensure_ascii=False)

    def _deserialize_measurement_config(self, raw_value: object) -> dict[str, object]:
        if not raw_value:
            return {}
        if isinstance(raw_value, dict):
            return raw_value
        try:
            decoded = json.loads(str(raw_value))
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def _compute_trace_preview_value(self, x_values: np.ndarray, y_values: np.ndarray) -> float:
        if len(x_values) < 2 or len(y_values) < 2:
            return float(np.sum(y_values))
        try:
            return float(np.trapezoid(y_values, x_values))
        except AttributeError:
            return float(np.trapz(y_values, x_values))

    def _row_to_api_payload(self, row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
        payload = dict(row)
        payload["file_path"] = payload.pop("source_wip")
        payload["source"] = payload.pop("material")
        payload["measurement_config"] = self._deserialize_measurement_config(payload.pop("measurement_config_json", None))
        payload["analysis_summary"] = self._deserialize_measurement_config(payload.pop("analysis_summary_json", None))
        payload.pop("csv_path", None)
        payload.pop("h5_path", None)
        payload.pop("h5_group", None)
        payload.pop("import_time", None)
        payload.pop("source_file_hash", None)
        return payload

    def _group_row_to_api_payload(self, row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
        payload = dict(row)
        dataset_id = payload.pop("dataset_id")
        representative_spectrum_id = payload.pop("representative_spectrum_id")
        member_count = int(payload.get("member_count") or 0)
        if member_count == 1 and payload.get("acquisition_mode") == "point_spectrum":
            payload["spectrum_id"] = representative_spectrum_id
        else:
            payload["spectrum_id"] = dataset_id
        payload["representative_spectrum_id"] = representative_spectrum_id
        payload["file_path"] = payload.pop("source_wip")
        payload["source"] = payload.pop("material")
        payload["analysis_summary"] = self._deserialize_measurement_config(payload.pop("analysis_summary_json", None))
        payload["source_tree_path"] = normalize_dataset_tree_path(str(payload["source_tree_path"]))
        payload["measurement_config"] = {}
        payload["member_count"] = member_count
        payload.pop("source_file_hash", None)
        return payload

    def _decode_job_payload(self, row: sqlite3.Row | dict[str, object]) -> dict[str, object]:
        payload = dict(row)
        payload["details"] = self._deserialize_measurement_config(payload.pop("details_json", None))
        phase = payload["details"].get("phase")
        if phase == "matlab_export":
            payload["details"]["phase"] = "reading_wip"
            payload["details"]["legacy_phase"] = "matlab_export"
        message = str(payload.get("message") or "")
        if message == "Starting MATLAB export":
            payload["message"] = "Import interrupted before backend restart"
        return payload

    def _fetch_dataset_rows(self, identifier: str) -> list[sqlite3.Row]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM spectra_index
                WHERE dataset_id = ? OR spectrum_id = ?
                ORDER BY COALESCE(trace_index, 0), spectrum_id
                """,
                [identifier, identifier],
            ).fetchall()
        return list(rows)

    def _build_dataset_detail(self, rows: list[sqlite3.Row], identifier: str) -> dict[str, object] | None:
        if not rows:
            return None

        representative = rows[0]
        for row in rows:
            if row["trace_index"] in (None, 0):
                representative = row
                break

        payload = self._row_to_api_payload(representative)
        if len(rows) == 1 and representative["acquisition_mode"] == "point_spectrum":
            payload["spectrum_id"] = str(representative["spectrum_id"])
        else:
            payload["spectrum_id"] = str(representative["dataset_id"] or identifier)
        payload["representative_spectrum_id"] = str(representative["spectrum_id"])
        payload["source_tree_path"] = normalize_dataset_tree_path(str(representative["source_tree_path"]))
        payload["member_count"] = len(rows)
        payload["trace_summaries"] = []
        payload["preview_series"] = None
        payload["preview_grid"] = None

        trace_summaries: list[dict[str, object]] = []
        for row in rows:
            trace_summaries.append(
                {
                    "spectrum_id": row["spectrum_id"],
                    "trace_index": row["trace_index"],
                    "grid_x": row["grid_x"],
                    "grid_y": row["grid_y"],
                    "preview_value": row["trace_preview_value"],
                }
            )
        payload["trace_summaries"] = trace_summaries

        mode = payload.get("acquisition_mode")
        if mode in {"line_scan", "series_scan"}:
            payload["preview_series"] = [
                {
                    "trace_index": item["trace_index"],
                    "value": item["preview_value"],
                }
                for item in trace_summaries
            ]
        elif mode == "area_map":
            size_x = int(payload.get("scan_size_x") or 0)
            size_y = int(payload.get("scan_size_y") or 0)
            if size_x > 0 and size_y > 0:
                grid = [[None for _ in range(size_x)] for _ in range(size_y)]
                for item in trace_summaries:
                    grid_x = item["grid_x"]
                    grid_y = item["grid_y"]
                    if isinstance(grid_x, int) and isinstance(grid_y, int):
                        if 0 <= grid_x < size_x and 0 <= grid_y < size_y:
                            grid[grid_y][grid_x] = item["preview_value"]
                payload["preview_grid"] = grid

        with self._hdf5_lock:
            with h5py.File(self.settings.hdf5_path, "r") as handle:
                group = handle[str(representative["h5_group"])]
                payload["x_axis"] = group["x_axis"][:].astype(float).tolist()
                payload["intensity"] = group["intensity"][:].astype(float).tolist()
        return payload

    def upsert_import_job(self, payload: dict[str, object]) -> None:
        fields = {
            "job_id": payload["job_id"],
            "input_path": payload.get("input_path", ""),
            "status": payload.get("status", "pending"),
            "total_files": int(payload.get("total_files", 0) or 0),
            "processed_files": int(payload.get("processed_files", 0) or 0),
            "exported_spectra": int(payload.get("exported_spectra", 0) or 0),
            "failed_files": int(payload.get("failed_files", 0) or 0),
            "current_file": payload.get("current_file"),
            "start_time": payload.get("start_time", utc_now()),
            "end_time": payload.get("end_time"),
            "log_path": payload.get("log_path"),
            "summary_path": payload.get("summary_path"),
            "message": payload.get("message"),
            "details_json": json.dumps(payload.get("details") or {}, ensure_ascii=False),
        }
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO import_jobs (
                    job_id, input_path, status, total_files, processed_files,
                    exported_spectra, failed_files, current_file, start_time,
                    end_time, log_path, summary_path, message, details_json
                ) VALUES (
                    :job_id, :input_path, :status, :total_files, :processed_files,
                    :exported_spectra, :failed_files, :current_file, :start_time,
                    :end_time, :log_path, :summary_path, :message, :details_json
                )
                ON CONFLICT(job_id) DO UPDATE SET
                    input_path=excluded.input_path,
                    status=excluded.status,
                    total_files=excluded.total_files,
                    processed_files=excluded.processed_files,
                    exported_spectra=excluded.exported_spectra,
                    failed_files=excluded.failed_files,
                    current_file=excluded.current_file,
                    start_time=excluded.start_time,
                    end_time=excluded.end_time,
                    log_path=excluded.log_path,
                    summary_path=excluded.summary_path,
                    message=excluded.message,
                    details_json=excluded.details_json
                """,
                fields,
            )

    def _upsert_sample(self, metadata: dict[str, object]) -> None:
        with self._connection() as connection:
            self._upsert_sample_with_connection(connection, metadata)

    def _upsert_sample_with_connection(
        self,
        connection: sqlite3.Connection,
        metadata: dict[str, object],
    ) -> None:
        sample_id = metadata.get("sample_id")
        if not sample_id:
            return
        now = utc_now()
        payload = {
            "sample_id": str(sample_id),
            "material": metadata.get("material") or metadata.get("source"),
            "substrate": metadata.get("substrate"),
            "device_id": metadata.get("device_id"),
            "notes": metadata.get("notes"),
            "created_at": now,
            "updated_at": now,
        }
        connection.execute(
            """
            INSERT INTO samples (
                sample_id, material, substrate, device_id, notes, created_at, updated_at
            ) VALUES (
                :sample_id, :material, :substrate, :device_id, :notes, :created_at, :updated_at
            )
            ON CONFLICT(sample_id) DO UPDATE SET
                material=COALESCE(excluded.material, samples.material),
                substrate=COALESCE(excluded.substrate, samples.substrate),
                device_id=COALESCE(excluded.device_id, samples.device_id),
                notes=COALESCE(excluded.notes, samples.notes),
                updated_at=excluded.updated_at
            """,
            payload,
        )

    def insert_spectrum(
        self,
        metadata: dict[str, object],
        x_axis: list[float] | np.ndarray,
        intensity: list[float] | np.ndarray,
    ) -> str:
        with self._connection() as connection:
            with self._hdf5_lock:
                with h5py.File(self.settings.hdf5_path, "a") as handle:
                    return self._insert_spectrum_with_handles(connection, handle, metadata, x_axis, intensity)

    def insert_spectrum_bulk(
        self,
        connection: sqlite3.Connection,
        h5_handle: h5py.File,
        metadata: dict[str, object],
        x_axis: list[float] | np.ndarray,
        intensity: list[float] | np.ndarray,
    ) -> str:
        return self._insert_spectrum_with_handles(connection, h5_handle, metadata, x_axis, intensity)

    def _insert_spectrum_with_handles(
        self,
        connection: sqlite3.Connection,
        h5_handle: h5py.File,
        metadata: dict[str, object],
        x_axis: list[float] | np.ndarray,
        intensity: list[float] | np.ndarray,
    ) -> str:
        spectrum_id = str(metadata["spectrum_id"])
        x_values = np.asarray(x_axis, dtype=float)
        y_values = np.asarray(intensity, dtype=float)
        h5_group = f"/spectra/{spectrum_id}"

        if h5_group in h5_handle:
            del h5_handle[h5_group]
        group = h5_handle.require_group(h5_group)
        group.create_dataset("x_axis", data=x_values)
        group.create_dataset("intensity", data=y_values)

        metadata = dict(metadata)
        metadata["source_wip"] = normalize_source_path(str(metadata["source_wip"]))
        metadata["material"] = metadata.get("source") or metadata.get("material")
        metadata.setdefault(
            "belonging",
            metadata.get("belonging") or infer_belonging_from_path(str(metadata["source_wip"])),
        )
        metadata.setdefault("import_time", utc_now())
        metadata.setdefault("source_file_hash", metadata.get("source_file_hash"))
        metadata.setdefault("h5_path", str(self.settings.hdf5_path))
        metadata.setdefault("h5_group", h5_group)
        metadata.setdefault("n_points", int(len(x_values)))
        metadata.setdefault(
            "dataset_id",
            build_dataset_id(str(metadata["source_wip"]), str(metadata.get("source_tree_path", ""))),
        )
        metadata.setdefault(
            "dataset_label",
            infer_dataset_label_from_tree_path(str(metadata.get("source_tree_path", ""))),
        )
        metadata.setdefault(
            "acquisition_mode",
            self._infer_acquisition_mode_from_path(str(metadata.get("source_tree_path", ""))),
        )
        metadata.setdefault("measurement_time", metadata.get("measurement_time"))
        metadata.setdefault("measurement_config_json", self._serialize_measurement_config(metadata))
        metadata.setdefault("trace_index", _maybe_int(metadata.get("trace_index")))
        metadata.setdefault("trace_count", _maybe_int(metadata.get("trace_count")))
        metadata.setdefault("trace_preview_value", self._compute_trace_preview_value(x_values, y_values))
        metadata.setdefault("scan_size_x", _maybe_int(metadata.get("scan_size_x")))
        metadata.setdefault("scan_size_y", _maybe_int(metadata.get("scan_size_y")))
        metadata.setdefault("grid_x", _maybe_int(metadata.get("grid_x")))
        metadata.setdefault("grid_y", _maybe_int(metadata.get("grid_y")))
        if is_mock_source(str(metadata["source_wip"])):
            metadata.setdefault("folder_path", "")
        else:
            metadata.setdefault("folder_path", str(Path(str(metadata["source_wip"])).parent))

        self._upsert_sample_with_connection(connection, metadata)

        record = {
            "spectrum_id": spectrum_id,
            "dataset_id": metadata.get("dataset_id"),
            "dataset_label": metadata.get("dataset_label"),
            "sample_id": metadata.get("sample_id"),
            "source_wip": str(metadata["source_wip"]),
            "source_file_hash": metadata.get("source_file_hash"),
            "source_tree_path": str(metadata.get("source_tree_path", "")),
            "spectrum_type": metadata.get("spectrum_type"),
            "acquisition_mode": metadata.get("acquisition_mode"),
            "x_axis_unit": metadata.get("x_axis_unit"),
            "n_points": int(metadata["n_points"]),
            "h5_path": str(metadata["h5_path"]),
            "h5_group": str(metadata["h5_group"]),
            "csv_path": metadata.get("csv_path"),
            "import_time": str(metadata["import_time"]),
            "measurement_time": metadata.get("measurement_time"),
            "measurement_config_json": metadata.get("measurement_config_json"),
            "trace_index": _maybe_int(metadata.get("trace_index")),
            "trace_count": _maybe_int(metadata.get("trace_count")),
            "trace_preview_value": metadata.get("trace_preview_value"),
            "scan_size_x": _maybe_int(metadata.get("scan_size_x")),
            "scan_size_y": _maybe_int(metadata.get("scan_size_y")),
            "grid_x": _maybe_int(metadata.get("grid_x")),
            "grid_y": _maybe_int(metadata.get("grid_y")),
            "laser_wavelength": metadata.get("laser_wavelength"),
            "laser_power": metadata.get("laser_power"),
            "integration_time": metadata.get("integration_time"),
            "grating": metadata.get("grating"),
            "objective": metadata.get("objective"),
            "material": metadata.get("material"),
            "belonging": metadata.get("belonging"),
            "substrate": metadata.get("substrate"),
            "device_id": metadata.get("device_id"),
            "notes": metadata.get("notes"),
            "folder_path": metadata.get("folder_path"),
        }

        connection.execute(
            """
            INSERT INTO spectra_index (
                spectrum_id, dataset_id, dataset_label, sample_id, source_wip, source_file_hash, source_tree_path, spectrum_type,
                acquisition_mode, x_axis_unit, n_points, h5_path, h5_group, csv_path, import_time,
                measurement_time, measurement_config_json, trace_index, trace_count, trace_preview_value,
                scan_size_x, scan_size_y, grid_x, grid_y,
                laser_wavelength, laser_power, integration_time, grating, objective,
                material, belonging, substrate, device_id, notes, folder_path
            ) VALUES (
                :spectrum_id, :dataset_id, :dataset_label, :sample_id, :source_wip, :source_file_hash, :source_tree_path, :spectrum_type,
                :acquisition_mode, :x_axis_unit, :n_points, :h5_path, :h5_group, :csv_path, :import_time,
                :measurement_time, :measurement_config_json, :trace_index, :trace_count, :trace_preview_value,
                :scan_size_x, :scan_size_y, :grid_x, :grid_y,
                :laser_wavelength, :laser_power, :integration_time, :grating, :objective,
                :material, :belonging, :substrate, :device_id, :notes, :folder_path
            )
            ON CONFLICT(spectrum_id) DO UPDATE SET
                dataset_id=excluded.dataset_id,
                dataset_label=excluded.dataset_label,
                sample_id=excluded.sample_id,
                source_wip=excluded.source_wip,
                source_file_hash=excluded.source_file_hash,
                source_tree_path=excluded.source_tree_path,
                spectrum_type=excluded.spectrum_type,
                acquisition_mode=excluded.acquisition_mode,
                x_axis_unit=excluded.x_axis_unit,
                n_points=excluded.n_points,
                h5_path=excluded.h5_path,
                h5_group=excluded.h5_group,
                csv_path=excluded.csv_path,
                import_time=excluded.import_time,
                measurement_time=excluded.measurement_time,
                measurement_config_json=excluded.measurement_config_json,
                trace_index=excluded.trace_index,
                trace_count=excluded.trace_count,
                trace_preview_value=excluded.trace_preview_value,
                scan_size_x=excluded.scan_size_x,
                scan_size_y=excluded.scan_size_y,
                grid_x=excluded.grid_x,
                grid_y=excluded.grid_y,
                laser_wavelength=excluded.laser_wavelength,
                laser_power=excluded.laser_power,
                integration_time=excluded.integration_time,
                grating=excluded.grating,
                objective=excluded.objective,
                material=excluded.material,
                belonging=excluded.belonging,
                substrate=excluded.substrate,
                device_id=excluded.device_id,
                notes=excluded.notes,
                folder_path=excluded.folder_path
            """,
            record,
        )
        self._invalidate_data_cache()
        return spectrum_id

    def count_spectra_for_source(self, source_wip: str) -> int:
        normalized_source = normalize_source_path(source_wip)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM spectra_index WHERE source_wip = ?",
                [normalized_source],
            ).fetchone()
        if row is None:
            return 0
        return int(row["count"] or 0)

    def insert_media_asset(self, metadata: dict[str, object], image_bytes: bytes) -> str:
        with self._connection() as connection:
            with self._hdf5_lock:
                with h5py.File(self.settings.hdf5_path, "a") as h5_handle:
                    media_id = self._insert_media_asset_with_handles(connection, h5_handle, metadata, image_bytes)
            connection.commit()
        return media_id

    def insert_media_asset_with_handles(
        self,
        connection: sqlite3.Connection,
        h5_handle: h5py.File,
        metadata: dict[str, object],
        image_bytes: bytes,
    ) -> str:
        return self._insert_media_asset_with_handles(connection, h5_handle, metadata, image_bytes)

    def _insert_media_asset_with_handles(
        self,
        connection: sqlite3.Connection,
        h5_handle: h5py.File,
        metadata: dict[str, object],
        image_bytes: bytes,
    ) -> str:
        metadata = dict(metadata)
        metadata["source_wip"] = normalize_source_path(str(metadata["source_wip"]))
        metadata.setdefault("import_time", utc_now())
        metadata.setdefault("source_file_hash", metadata.get("source_file_hash"))
        metadata.setdefault("asset_format", "jpeg")
        metadata.setdefault(
            "media_id",
            build_media_id(str(metadata["source_wip"]), str(metadata.get("source_tree_path", ""))),
        )
        media_id = str(metadata["media_id"])
        h5_group = f"/media/{media_id}"

        if h5_group in h5_handle:
            del h5_handle[h5_group]
        group = h5_handle.require_group(h5_group)
        group.create_dataset("data", data=np.frombuffer(image_bytes, dtype=np.uint8))

        metadata.setdefault("h5_path", str(self.settings.hdf5_path))
        metadata.setdefault("h5_group", h5_group)
        metadata.setdefault("file_size_bytes", int(len(image_bytes)))
        metadata.setdefault("measurement_time", metadata.get("measurement_time"))
        metadata.setdefault("measurement_config_json", self._serialize_measurement_config(metadata))
        metadata.setdefault("original_width_px", _maybe_int(metadata.get("original_width_px")))
        metadata.setdefault("original_height_px", _maybe_int(metadata.get("original_height_px")))
        metadata.setdefault("width_px", _maybe_int(metadata.get("width_px")) or 0)
        metadata.setdefault("height_px", _maybe_int(metadata.get("height_px")) or 0)
        metadata.setdefault("channel_count", _maybe_int(metadata.get("channel_count")) or 1)
        metadata.setdefault("bit_depth", _maybe_int(metadata.get("bit_depth")) or 8)
        if is_mock_source(str(metadata["source_wip"])):
            metadata.setdefault("folder_path", "")
        else:
            metadata.setdefault("folder_path", str(Path(str(metadata["source_wip"])).parent))

        record = {
            "media_id": media_id,
            "source_wip": str(metadata["source_wip"]),
            "source_file_hash": metadata.get("source_file_hash"),
            "source_tree_path": str(metadata.get("source_tree_path", "")),
            "media_kind": str(metadata.get("media_kind") or "photo_image"),
            "entry_class": str(metadata.get("entry_class") or ""),
            "caption": metadata.get("caption"),
            "asset_format": str(metadata.get("asset_format") or "jpeg"),
            "h5_path": str(metadata["h5_path"]),
            "h5_group": str(metadata["h5_group"]),
            "width_px": int(metadata["width_px"]),
            "height_px": int(metadata["height_px"]),
            "original_width_px": _maybe_int(metadata.get("original_width_px")),
            "original_height_px": _maybe_int(metadata.get("original_height_px")),
            "channel_count": int(metadata["channel_count"]),
            "bit_depth": int(metadata["bit_depth"]),
            "file_size_bytes": int(metadata["file_size_bytes"]),
            "import_time": str(metadata["import_time"]),
            "measurement_time": metadata.get("measurement_time"),
            "measurement_config_json": metadata.get("measurement_config_json"),
            "folder_path": metadata.get("folder_path"),
        }
        connection.execute(
            """
            INSERT INTO media_assets (
                media_id, source_wip, source_file_hash, source_tree_path, media_kind, entry_class,
                caption, asset_format, h5_path, h5_group, width_px, height_px,
                original_width_px, original_height_px, channel_count, bit_depth, file_size_bytes,
                import_time, measurement_time, measurement_config_json, folder_path
            ) VALUES (
                :media_id, :source_wip, :source_file_hash, :source_tree_path, :media_kind, :entry_class,
                :caption, :asset_format, :h5_path, :h5_group, :width_px, :height_px,
                :original_width_px, :original_height_px, :channel_count, :bit_depth, :file_size_bytes,
                :import_time, :measurement_time, :measurement_config_json, :folder_path
            )
            ON CONFLICT(media_id) DO UPDATE SET
                source_wip=excluded.source_wip,
                source_file_hash=excluded.source_file_hash,
                source_tree_path=excluded.source_tree_path,
                media_kind=excluded.media_kind,
                entry_class=excluded.entry_class,
                caption=excluded.caption,
                asset_format=excluded.asset_format,
                h5_path=excluded.h5_path,
                h5_group=excluded.h5_group,
                width_px=excluded.width_px,
                height_px=excluded.height_px,
                original_width_px=excluded.original_width_px,
                original_height_px=excluded.original_height_px,
                channel_count=excluded.channel_count,
                bit_depth=excluded.bit_depth,
                file_size_bytes=excluded.file_size_bytes,
                import_time=excluded.import_time,
                measurement_time=excluded.measurement_time,
                measurement_config_json=excluded.measurement_config_json,
                folder_path=excluded.folder_path
            """,
            record,
        )
        self._invalidate_data_cache()
        return media_id

    def count_spectra_for_source_hash(self, source_file_hash: str) -> int:
        normalized_hash = str(source_file_hash or "").strip()
        if not normalized_hash:
            return 0
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM spectra_index WHERE source_file_hash = ?",
                [normalized_hash],
            ).fetchone()
        if row is None:
            return 0
        return int(row["count"] or 0)

    def count_media_assets_for_source(self, source_wip: str) -> int:
        normalized_source = normalize_source_path(source_wip)
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM media_assets WHERE source_wip = ?",
                [normalized_source],
            ).fetchone()
        if row is None:
            return 0
        return int(row["count"] or 0)

    def count_media_assets_for_source_hash(self, source_file_hash: str) -> int:
        normalized_hash = str(source_file_hash or "").strip()
        if not normalized_hash:
            return 0
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM media_assets WHERE source_file_hash = ?",
                [normalized_hash],
            ).fetchone()
        if row is None:
            return 0
        return int(row["count"] or 0)

    def count_import_records_for_source(self, source_wip: str) -> int:
        return self.count_spectra_for_source(source_wip) + self.count_media_assets_for_source(source_wip)

    def count_import_records_for_source_hash(self, source_file_hash: str) -> int:
        return self.count_spectra_for_source_hash(source_file_hash) + self.count_media_assets_for_source_hash(source_file_hash)

    def list_sources_for_hash(self, source_file_hash: str) -> list[str]:
        normalized_hash = str(source_file_hash or "").strip()
        if not normalized_hash:
            return []
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT source_wip
                FROM (
                    SELECT source_wip
                    FROM spectra_index
                    WHERE source_file_hash = ?
                    UNION
                    SELECT source_wip
                    FROM media_assets
                    WHERE source_file_hash = ?
                )
                ORDER BY source_wip
                """,
                [normalized_hash, normalized_hash],
            ).fetchall()
        return [str(row["source_wip"]) for row in rows]

    def delete_spectra_for_source(self, source_wip: str) -> int:
        normalized_source = normalize_source_path(source_wip)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT spectrum_id, h5_group FROM spectra_index WHERE source_wip = ?",
                [normalized_source],
            ).fetchall()
            if rows:
                placeholders = ", ".join("?" for _ in rows)
                connection.execute(
                    f"DELETE FROM analysis_results WHERE spectrum_id IN ({placeholders})",
                    [str(row["spectrum_id"]) for row in rows],
                )
            connection.execute(
                "DELETE FROM spectra_index WHERE source_wip = ?",
                [normalized_source],
            )

        if not rows:
            return 0

        with self._hdf5_lock:
            with h5py.File(self.settings.hdf5_path, "a") as handle:
                for row in rows:
                    h5_group = str(row["h5_group"])
                    if h5_group in handle:
                        del handle[h5_group]
        self._invalidate_data_cache()
        return len(rows)

    def delete_spectra_for_source_hash(self, source_file_hash: str) -> int:
        normalized_hash = str(source_file_hash or "").strip()
        if not normalized_hash:
            return 0
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT spectrum_id, h5_group FROM spectra_index WHERE source_file_hash = ?",
                [normalized_hash],
            ).fetchall()
            if rows:
                placeholders = ", ".join("?" for _ in rows)
                connection.execute(
                    f"DELETE FROM analysis_results WHERE spectrum_id IN ({placeholders})",
                    [str(row["spectrum_id"]) for row in rows],
                )
            connection.execute(
                "DELETE FROM spectra_index WHERE source_file_hash = ?",
                [normalized_hash],
            )

        if not rows:
            return 0

        with self._hdf5_lock:
            with h5py.File(self.settings.hdf5_path, "a") as handle:
                for row in rows:
                    h5_group = str(row["h5_group"])
                    if h5_group in handle:
                        del handle[h5_group]
        self._invalidate_data_cache()
        return len(rows)

    def delete_media_assets_for_source(self, source_wip: str) -> int:
        normalized_source = normalize_source_path(source_wip)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT media_id, h5_group FROM media_assets WHERE source_wip = ?",
                [normalized_source],
            ).fetchall()
            connection.execute(
                "DELETE FROM media_assets WHERE source_wip = ?",
                [normalized_source],
            )

        if not rows:
            return 0

        with self._hdf5_lock:
            with h5py.File(self.settings.hdf5_path, "a") as handle:
                for row in rows:
                    h5_group = str(row["h5_group"])
                    if h5_group in handle:
                        del handle[h5_group]
        self._invalidate_data_cache()
        return len(rows)

    def delete_media_assets_for_source_hash(self, source_file_hash: str) -> int:
        normalized_hash = str(source_file_hash or "").strip()
        if not normalized_hash:
            return 0
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT media_id, h5_group FROM media_assets WHERE source_file_hash = ?",
                [normalized_hash],
            ).fetchall()
            connection.execute(
                "DELETE FROM media_assets WHERE source_file_hash = ?",
                [normalized_hash],
            )

        if not rows:
            return 0

        with self._hdf5_lock:
            with h5py.File(self.settings.hdf5_path, "a") as handle:
                for row in rows:
                    h5_group = str(row["h5_group"])
                    if h5_group in handle:
                        del handle[h5_group]
        self._invalidate_data_cache()
        return len(rows)

    def delete_import_records_for_source(self, source_wip: str) -> int:
        deleted_spectra = self.delete_spectra_for_source(source_wip)
        deleted_media = self.delete_media_assets_for_source(source_wip)
        return deleted_spectra + deleted_media

    def delete_import_records_for_source_hash(self, source_file_hash: str) -> int:
        deleted_spectra = self.delete_spectra_for_source_hash(source_file_hash)
        deleted_media = self.delete_media_assets_for_source_hash(source_file_hash)
        return deleted_spectra + deleted_media

    def list_spectra(
        self,
        *,
        search: str | None = None,
        filters: dict[str, object | None] | None = None,
        include_mock: bool = False,
        limit: int = 100,
        offset: int = 0,
        sort_by: str | None = None,
        sort_dir: str | None = None,
    ) -> dict[str, object]:
        filters = filters or {}
        clauses: list[str] = []
        params: list[object] = []
        if not include_mock:
            self._append_real_only_clause(clauses)

        alias_map = {
            "spectrum_type": "spectrum_type",
            "source": "material",
            "belonging": "belonging",
            "acquisition_mode": "acquisition_mode",
            "substrate": "substrate",
            "x_axis_unit": "x_axis_unit",
            "sample_id": "sample_id",
            "analysis_material": "analysis_material",
            "analysis_family": "analysis_family",
            "analysis_status": "analysis_status",
        }
        for filter_name, db_field in alias_map.items():
            value = filters.get(filter_name)
            if value:
                clauses.append(f"{db_field} = ?")
                params.append(value)

        if search:
            token = f"%{search.lower()}%"
            clauses.append(
                "("
                "lower(source_wip) LIKE ? OR lower(sample_id) LIKE ? OR lower(material) LIKE ? OR "
                "lower(substrate) LIKE ? OR lower(notes) LIKE ? OR lower(dataset_label) LIKE ? OR lower(source_tree_path) LIKE ? OR "
                "lower(coalesce(acquisition_mode, '')) LIKE ? OR lower(coalesce(belonging, '')) LIKE ?"
                ")"
            )
            params.extend([token, token, token, token, token, token, token, token, token])
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    dataset_id,
                    spectrum_id,
                    sample_id,
                    source_wip,
                    source_tree_path,
                    spectrum_type,
                    acquisition_mode,
                    x_axis_unit,
                    n_points,
                    measurement_time,
                    dataset_label,
                    laser_wavelength,
                    laser_power,
                    integration_time,
                    grating,
                    objective,
                    material,
                    analysis_material,
                    analysis_family,
                    analysis_status,
                    analysis_method_version,
                    analysis_summary_json,
                    analysis_updated_at,
                    belonging,
                    substrate,
                    device_id,
                    notes,
                    folder_path,
                    trace_index,
                    trace_count,
                    scan_size_x,
                    scan_size_y,
                    grid_x,
                    grid_y,
                    import_time
                FROM spectra_index
                {where_clause}
                """,
                params,
            ).fetchall()

        grouped: dict[str, dict[str, object]] = {}
        for row in rows:
            dataset_id = str(row["dataset_id"] or "")
            if not dataset_id:
                continue
            payload = grouped.get(dataset_id)
            sort_key = str(row["measurement_time"] or row["import_time"] or "")
            if payload is None:
                payload = dict(row)
                payload["representative_spectrum_id"] = row["spectrum_id"]
                payload["member_count"] = 1
                payload["_sort_key"] = sort_key
                grouped[dataset_id] = payload
                continue

            payload["member_count"] = int(payload.get("member_count") or 0) + 1
            if row["trace_index"] in (None, 0) and payload.get("trace_index") not in (None, 0):
                for key, value in dict(row).items():
                    if key == "dataset_id":
                        continue
                    payload[key] = value
                payload["representative_spectrum_id"] = row["spectrum_id"]
            else:
                payload["n_points"] = max(int(payload.get("n_points") or 0), int(row["n_points"] or 0))
                payload["trace_count"] = max(int(payload.get("trace_count") or 0), int(row["trace_count"] or 0))
                payload["scan_size_x"] = max(int(payload.get("scan_size_x") or 0), int(row["scan_size_x"] or 0))
                payload["scan_size_y"] = max(int(payload.get("scan_size_y") or 0), int(row["scan_size_y"] or 0))
            payload["_sort_key"] = max(str(payload.get("_sort_key") or ""), sort_key)

        items = [
            item
            for item in grouped.values()
            if _matches_numeric_ranges(
                item,
                filters,
                [
                    "n_points",
                    "member_count",
                    "trace_count",
                    "scan_size_x",
                    "scan_size_y",
                    "grid_x",
                    "grid_y",
                ],
            )
        ]
        items = _sort_grouped_spectra(items, sort_by=sort_by, sort_dir=sort_dir)
        paged_items = items[offset : offset + limit]
        for item in paged_items:
            item.pop("_sort_key", None)
        return {
            "total": len(items),
            "items": [self._group_row_to_api_payload(row) for row in paged_items],
        }

    def get_spectrum(self, spectrum_id: str) -> dict[str, object] | None:
        rows = self._fetch_dataset_rows(spectrum_id)
        return self._build_dataset_detail(rows, spectrum_id)

    def list_spectrum_identifiers(
        self,
        *,
        search: str | None = None,
        filters: dict[str, str | None] | None = None,
        include_mock: bool = False,
    ) -> list[str]:
        listing = self.list_spectra(
            search=search,
            filters=filters,
            include_mock=include_mock,
            limit=1_000_000,
            offset=0,
        )
        identifiers: list[str] = []
        seen: set[str] = set()
        for item in listing["items"]:
            identifier = str(item.get("representative_spectrum_id") or item.get("spectrum_id") or "")
            if identifier and identifier not in seen:
                seen.add(identifier)
                identifiers.append(identifier)
        return identifiers

    def list_filter_options(self, *, include_mock: bool = False) -> dict[str, list[str]]:
        with self._cache_lock:
            cached = self._filter_options_cache.get(include_mock)
            if cached and cached[0] == self._data_revision:
                return dict(cached[1])

        options = {
            "spectrum_type": set(),
            "source": set(),
            "belonging": set(),
            "acquisition_mode": set(),
            "substrate": set(),
            "x_axis_unit": set(),
            "sample_id": set(),
            "analysis_material": set(),
            "analysis_family": set(),
            "analysis_status": set(),
        }
        clauses: list[str] = []
        if not include_mock:
            clauses.append(
                "lower(source_wip) NOT LIKE 'mock%' AND "
                "source_tree_path NOT LIKE 'mock://%' AND "
                "source_tree_path NOT LIKE '/Project/Data/Graph/%'"
            )
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    spectrum_type,
                    material,
                    belonging,
                    acquisition_mode,
                    substrate,
                    x_axis_unit,
                    sample_id,
                    analysis_material,
                    analysis_family,
                    analysis_status
                FROM spectra_index
                {where_clause}
                """
            ).fetchall()

        for row in rows:
            if row["spectrum_type"] not in (None, ""):
                options["spectrum_type"].add(str(row["spectrum_type"]))
            if row["material"] not in (None, ""):
                options["source"].add(str(row["material"]))
            if row["belonging"] not in (None, ""):
                options["belonging"].add(str(row["belonging"]))
            if row["acquisition_mode"] not in (None, ""):
                options["acquisition_mode"].add(str(row["acquisition_mode"]))
            if row["substrate"] not in (None, ""):
                options["substrate"].add(str(row["substrate"]))
            if row["x_axis_unit"] not in (None, ""):
                options["x_axis_unit"].add(str(row["x_axis_unit"]))
            if row["sample_id"] not in (None, ""):
                options["sample_id"].add(str(row["sample_id"]))
            if row["analysis_material"] not in (None, ""):
                options["analysis_material"].add(str(row["analysis_material"]))
            if row["analysis_family"] not in (None, ""):
                options["analysis_family"].add(str(row["analysis_family"]))
            if row["analysis_status"] not in (None, ""):
                options["analysis_status"].add(str(row["analysis_status"]))

        serialized = {key: sorted(value) for key, value in options.items()}
        with self._cache_lock:
            self._filter_options_cache[include_mock] = (self._data_revision, dict(serialized))
        return serialized

    def update_metadata(
        self,
        *,
        spectrum_ids: list[str],
        apply_mode: str,
        scope_value: str | None,
        metadata: dict[str, str | None],
    ) -> int:
        allowed = {
            "sample_id",
            "source",
            "substrate",
            "device_id",
            "laser_wavelength",
            "laser_power",
            "integration_time",
            "grating",
            "objective",
            "measurement_time",
            "notes",
        }
        updates = {key: value for key, value in metadata.items() if key in allowed}
        if not updates:
            return 0

        db_updates = {
            ("material" if key == "source" else key): value
            for key, value in updates.items()
        }

        clauses: list[str] = []
        params: list[object] = []
        if apply_mode == "selected":
            if not spectrum_ids:
                return 0
            placeholders = ",".join("?" for _ in spectrum_ids)
            clauses.append(f"(spectrum_id IN ({placeholders}) OR dataset_id IN ({placeholders}))")
            params.extend(spectrum_ids)
            params.extend(spectrum_ids)
        elif apply_mode == "source_file" and scope_value:
            clauses.append("source_wip = ?")
            params.append(scope_value)
        elif apply_mode == "folder" and scope_value:
            clauses.append("folder_path = ?")
            params.append(scope_value)
        elif apply_mode == "all":
            pass
        else:
            return 0

        set_clause = ", ".join(f"{field} = ?" for field in db_updates)
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE spectra_index SET {set_clause} {where_clause}",
                [*db_updates.values(), *params],
            )
        if updates.get("sample_id"):
            self._upsert_sample(updates)
        if int(cursor.rowcount or 0) > 0:
            self._invalidate_data_cache()
        return int(cursor.rowcount)

    def insert_analysis_result(
        self,
        spectrum_id: str,
        result: dict[str, object],
        *,
        method: str,
        parameters: dict[str, object],
    ) -> str:
        result_id = f"analysis-{uuid.uuid4().hex[:12]}"
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO analysis_results (
                    result_id, spectrum_id, method, parameters_json, peaks_json,
                    metrics_json, fit_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    result_id,
                    spectrum_id,
                    method,
                    json.dumps(parameters),
                    json.dumps(result.get("peaks", [])),
                    json.dumps(result.get("metrics", {})),
                    json.dumps(result.get("fit", {})),
                    utc_now(),
                ],
            )
        return result_id

    def update_spectrum_analysis_summary(
        self,
        spectrum_id: str,
        result: dict[str, object],
        *,
        method: str,
        status: str = "processed",
    ) -> int:
        features = result.get("features") if isinstance(result.get("features"), dict) else {}
        summary = {
            "features": features,
            "material_confidence": result.get("material_confidence"),
            "material_candidates": result.get("material_candidates", []),
            "fit_r2": (result.get("fit") or {}).get("r2") if isinstance(result.get("fit"), dict) else None,
            "fit_rmse": (result.get("fit") or {}).get("rmse") if isinstance(result.get("fit"), dict) else None,
            "fit_snr": (result.get("fit") or {}).get("signal_to_noise") if isinstance(result.get("fit"), dict) else None,
            "fit_model": (result.get("fit") or {}).get("model") if isinstance(result.get("fit"), dict) else None,
            "physical_score": (result.get("fit") or {}).get("physical_score") if isinstance(result.get("fit"), dict) else None,
            "resolution_score": (result.get("fit") or {}).get("resolution_score") if isinstance(result.get("fit"), dict) else None,
            "fit_peaks": (result.get("fit") or {}).get("peaks", []) if isinstance(result.get("fit"), dict) else [],
            "fit_quality": result.get("quality", {}),
            "axis": result.get("axis", {}),
            "detected_peak_count": len(result.get("peaks", [])) if isinstance(result.get("peaks"), list) else 0,
            "fit_peak_count": len((result.get("fit") or {}).get("peaks", [])) if isinstance(result.get("fit"), dict) else 0,
        }
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE spectra_index
                SET
                    material = COALESCE(NULLIF(?, ''), material),
                    analysis_material = ?,
                    analysis_family = ?,
                    analysis_status = ?,
                    analysis_method_version = ?,
                    analysis_summary_json = ?,
                    analysis_updated_at = ?
                WHERE spectrum_id = ?
                """,
                [
                    result.get("material") if result.get("material") != "unknown" else None,
                    result.get("material"),
                    result.get("spectrum_family"),
                    status,
                    result.get("method_version") or method,
                    json.dumps(summary, ensure_ascii=False),
                    utc_now(),
                    spectrum_id,
                ],
            )
        if int(cursor.rowcount or 0) > 0:
            self._invalidate_data_cache()
        return int(cursor.rowcount or 0)

    def list_analysis_results(self, *, spectrum_id: str | None = None, limit: int = 200) -> list[dict[str, object]]:
        sql = "SELECT * FROM analysis_results"
        params: list[object] = []
        if spectrum_id:
            sql += " WHERE spectrum_id = ?"
            params.append(spectrum_id)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        payload = [dict(row) for row in rows]
        for item in payload:
            item["parameters"] = json.loads(item.pop("parameters_json"))
            item["peaks"] = json.loads(item.pop("peaks_json"))
            item["metrics"] = json.loads(item.pop("metrics_json"))
            item["fit"] = json.loads(item.pop("fit_json"))
        return payload

    def _group_counts(self, connection: sqlite3.Connection, field: str, where_clause: str, params: list[object]) -> dict[str, int]:
        rows = connection.execute(
            f"""
            SELECT {field} AS label, COUNT(DISTINCT dataset_id) AS count
            FROM spectra_index
            {where_clause}
            AND {field} IS NOT NULL
            AND {field} <> ''
            GROUP BY {field}
            ORDER BY count DESC, label ASC
            """
            if where_clause
            else f"""
            SELECT {field} AS label, COUNT(DISTINCT dataset_id) AS count
            FROM spectra_index
            WHERE {field} IS NOT NULL
            AND {field} <> ''
            GROUP BY {field}
            ORDER BY count DESC, label ASC
            """,
            params,
        ).fetchall()
        return {str(row["label"]): int(row["count"]) for row in rows}

    def _measurement_timeline(
        self,
        connection: sqlite3.Connection,
        where_clause: str,
        params: list[object],
    ) -> list[dict[str, object]]:
        rows = connection.execute(
            f"""
            SELECT dataset_id, measurement_time
            FROM spectra_index
            {where_clause}
            AND measurement_time IS NOT NULL
            AND measurement_time <> ''
            """
            if where_clause
            else """
            SELECT dataset_id, measurement_time
            FROM spectra_index
            WHERE measurement_time IS NOT NULL
            AND measurement_time <> ''
            """,
            params,
        ).fetchall()

        parsed_dates: list[date] = []
        seen_dataset_ids: set[str] = set()
        dataset_dates: list[date] = []
        for row in rows:
            dataset_id = str(row["dataset_id"] or "").strip()
            if not dataset_id or dataset_id in seen_dataset_ids:
                continue
            parsed = _coerce_iso_datetime(row["measurement_time"])
            if parsed is None:
                continue
            seen_dataset_ids.add(dataset_id)
            parsed_dates.append(parsed.date())
            dataset_dates.append(parsed.date())

        if not parsed_dates:
            return []

        minimum = min(parsed_dates)
        maximum = max(parsed_dates)
        granularity = _auto_timeline_granularity((maximum - minimum).days)
        counts = Counter(_bucket_date(item, granularity) for item in dataset_dates)
        timeline: list[dict[str, object]] = []
        for bucket_date in sorted(counts):
            timeline.append(
                {
                    "bucket": bucket_date.isoformat(),
                    "count": int(counts[bucket_date]),
                    "granularity": granularity,
                }
            )
        return timeline

    def dashboard_summary(self, *, include_mock: bool = False) -> dict[str, object]:
        spectrum_clauses: list[str] = []
        if not include_mock:
            self._append_real_only_clause(spectrum_clauses)
        spectrum_where = f"WHERE {' AND '.join(spectrum_clauses)}" if spectrum_clauses else ""
        media_where = spectrum_where
        with self._connection() as connection:
            imported_files = connection.execute(
                f"""
                SELECT COUNT(DISTINCT source_key) AS count
                FROM (
                    SELECT COALESCE(NULLIF(source_file_hash, ''), source_wip) AS source_key
                    FROM spectra_index
                    {spectrum_where}
                    UNION
                    SELECT COALESCE(NULLIF(source_file_hash, ''), source_wip) AS source_key
                    FROM media_assets
                    {media_where}
                )
                """
            ).fetchone()["count"]
            spectra_count = connection.execute(
                f"SELECT COUNT(DISTINCT dataset_id) AS count FROM spectra_index {spectrum_where}"
            ).fetchone()["count"]
            failed_imports = connection.execute(
                "SELECT COUNT(*) AS count FROM import_jobs WHERE status IN ('failed', 'partially_failed')"
            ).fetchone()["count"]
            latest_job = connection.execute(
                "SELECT * FROM import_jobs ORDER BY start_time DESC LIMIT 1"
            ).fetchone()
            type_counts = self._group_counts(connection, "spectrum_type", spectrum_where, [])
            acquisition_counts = self._group_counts(connection, "acquisition_mode", spectrum_where, [])
            measurement_timeline = self._measurement_timeline(connection, spectrum_where, [])

        sqlite_size = self.settings.sqlite_path.stat().st_size if self.settings.sqlite_path.exists() else 0
        hdf5_size = self.settings.hdf5_path.stat().st_size if self.settings.hdf5_path.exists() else 0
        return {
            "imported_files": int(imported_files or 0),
            "spectra_count": int(spectra_count or 0),
            "failed_imports": int(failed_imports or 0),
            "database_size_mb": round((sqlite_size + hdf5_size) / (1024 * 1024), 3),
            "type_counts": type_counts,
            "acquisition_counts": acquisition_counts,
            "measurement_timeline": measurement_timeline,
            "latest_job": self._decode_job_payload(latest_job) if latest_job else None,
        }

    def list_import_jobs(self, limit: int = 30) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM import_jobs ORDER BY start_time DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [self._decode_job_payload(row) for row in rows]

    def get_import_job(self, job_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM import_jobs WHERE job_id = ?",
                [job_id],
            ).fetchone()
        return self._decode_job_payload(row) if row else None

    def list_imported_upload_history(self, *, root_name: str | None = None) -> list[dict[str, object]]:
        normalized_root = str(root_name or "").strip().replace("\\", "/").strip("/")
        prefix = f"{normalized_root}/".lower() if normalized_root else ""
        cache_key = normalized_root.lower()
        with self._cache_lock:
            cached = self._upload_history_cache.get(cache_key)
            if cached and cached[0] == self._data_revision:
                return [dict(item) for item in cached[1]]

        with self._connection() as connection:
            job_rows = connection.execute(
                """
                SELECT
                    job_id,
                    status,
                    start_time,
                    end_time,
                    message,
                    json_extract(details_json, '$.source_kind') AS source_kind,
                    json_extract(details_json, '$.relative_input_path') AS relative_input_path,
                    json_extract(details_json, '$.display_input_path') AS display_input_path,
                    json_extract(details_json, '$.result_summary.imported_spectra') AS imported_spectra,
                    json_extract(details_json, '$.result_summary.imported_media_assets') AS imported_media_assets
                FROM import_jobs
                WHERE status IN ('finished', 'partially_failed')
                ORDER BY COALESCE(end_time, start_time) DESC
                """
            ).fetchall()
            source_rows = connection.execute(
                """
                SELECT
                    source_key,
                    MIN(source_wip) AS source_wip,
                    MAX(import_time) AS import_time,
                    SUM(imported_spectra) AS imported_spectra,
                    SUM(imported_media_assets) AS imported_media_assets
                FROM (
                    SELECT
                        COALESCE(NULLIF(source_file_hash, ''), source_wip) AS source_key,
                        source_wip,
                        MAX(import_time) AS import_time,
                        COUNT(DISTINCT dataset_id) AS imported_spectra,
                        0 AS imported_media_assets
                    FROM spectra_index
                    GROUP BY COALESCE(NULLIF(source_file_hash, ''), source_wip), source_wip

                    UNION ALL

                    SELECT
                        COALESCE(NULLIF(source_file_hash, ''), source_wip) AS source_key,
                        source_wip,
                        MAX(import_time) AS import_time,
                        0 AS imported_spectra,
                        COUNT(*) AS imported_media_assets
                    FROM media_assets
                    GROUP BY COALESCE(NULLIF(source_file_hash, ''), source_wip), source_wip
                )
                GROUP BY source_key
                ORDER BY MAX(import_time) DESC
                """
            ).fetchall()

        history: dict[str, dict[str, object]] = {}
        for row in job_rows:
            source_kind = str(row["source_kind"] or "")
            if source_kind not in {"file_upload", "folder_upload"}:
                continue

            relative_path = str(
                row["relative_input_path"]
                or self._extract_relative_upload_path(row["display_input_path"])
                or ""
            ).strip().replace("\\", "/")
            self._record_imported_upload_history_item(
                history,
                relative_path=relative_path,
                normalized_root=normalized_root,
                prefix=prefix,
                payload={
                    "relative_path": relative_path,
                    "source_kind": source_kind,
                    "job_id": row["job_id"],
                    "status": row["status"],
                    "imported_spectra": int(row["imported_spectra"] or 0),
                    "imported_media_assets": int(row["imported_media_assets"] or 0),
                    "ended_at": row["end_time"] or row["start_time"],
                    "message": row["message"],
                },
            )

        for row in source_rows:
            relative_path = self._extract_relative_source_path(row["source_wip"], normalized_root)
            self._record_imported_upload_history_item(
                history,
                relative_path=relative_path,
                normalized_root=normalized_root,
                prefix=prefix,
                payload={
                    "relative_path": relative_path,
                    "source_kind": "database_index",
                    "job_id": f"derived::{row['source_key']}",
                    "status": "finished",
                    "imported_spectra": int(row["imported_spectra"] or 0),
                    "imported_media_assets": int(row["imported_media_assets"] or 0),
                    "ended_at": row["import_time"],
                    "message": "Derived from previously imported database contents",
                },
            )

        items = list(history.values())
        with self._cache_lock:
            self._upload_history_cache[cache_key] = (
                self._data_revision,
                [dict(item) for item in items],
            )
        return items

    def _record_imported_upload_history_item(
        self,
        history: dict[str, dict[str, object]],
        *,
        relative_path: str | None,
        normalized_root: str,
        prefix: str,
        payload: dict[str, object],
    ) -> None:
        normalized_path = self._normalize_relative_upload_path(relative_path)
        if not normalized_path:
            return
        lowered_path = normalized_path.lower()
        normalized_root_lower = normalized_root.lower()
        if prefix and lowered_path != normalized_root_lower and not lowered_path.startswith(prefix):
            return
        if lowered_path in history:
            return

        history[lowered_path] = {
            **payload,
            "relative_path": normalized_path,
        }

    def _normalize_relative_upload_path(self, relative_path: object) -> str:
        return str(relative_path or "").strip().replace("\\", "/").strip("/")

    def _extract_relative_source_path(self, source_wip: object, normalized_root: str) -> str | None:
        normalized_source = normalize_source_path(str(source_wip or ""))
        if not normalized_source:
            return None

        parts = [part for part in Path(normalized_source).as_posix().split("/") if part]
        if not parts:
            return None

        if normalized_root:
            root_lower = normalized_root.lower()
            for index, part in enumerate(parts):
                if part.lower() == root_lower:
                    return "/".join(parts[index:])

        lowered_parts = [part.lower() for part in parts]
        for index in range(len(lowered_parts) - 2):
            if lowered_parts[index] != "data" or lowered_parts[index + 1] != "raw_wip":
                continue
            upload_index = index + 2
            if upload_index >= len(parts) or not parts[upload_index].lower().startswith("upload-"):
                continue
            trailing = parts[upload_index + 1 :]
            if trailing:
                return "/".join(trailing)

        return None

    def _extract_relative_upload_path(self, display_input_path: object) -> str | None:
        text = str(display_input_path or "").strip()
        for prefix in ("Selected folder item:", "Selected file:"):
            if text.startswith(prefix):
                value = text.removeprefix(prefix).strip()
                return value or None
        return None

    def resolve_raw_spectrum_ids(self, spectrum_ids: list[str]) -> list[str]:
        resolved: list[str] = []
        seen: set[str] = set()
        with self._connection() as connection:
            for identifier in spectrum_ids:
                rows = connection.execute(
                    """
                    SELECT spectrum_id
                    FROM spectra_index
                    WHERE spectrum_id = ? OR dataset_id = ?
                    ORDER BY COALESCE(trace_index, 0), spectrum_id
                    """,
                    [identifier, identifier],
                ).fetchall()
                for row in rows:
                    spectrum_id = str(row["spectrum_id"])
                    if spectrum_id not in seen:
                        seen.add(spectrum_id)
                        resolved.append(spectrum_id)
        return resolved

    def export_spectra_to_csv(self, spectrum_ids: list[str], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["spectrum_id", "x_axis", "intensity"])
            for spectrum_id in self.resolve_raw_spectrum_ids(spectrum_ids):
                detail = self.get_spectrum(spectrum_id)
                if not detail:
                    continue
                for x_value, y_value in zip(detail["x_axis"], detail["intensity"], strict=True):
                    writer.writerow([spectrum_id, x_value, y_value])
        return destination
