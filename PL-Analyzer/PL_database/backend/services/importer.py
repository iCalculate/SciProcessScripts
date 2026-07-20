from __future__ import annotations

import shutil
import queue
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..config import AppSettings
from .database import DatabaseService
from .import_bridge import ImportBridge, ImportCancelled
from .metadata_parser import (
    build_spectrum_id,
    infer_spectrum_type_with_context,
    infer_x_axis_unit,
    normalize_source_path,
    parse_metadata_from_path,
)


DEFAULT_IMPORT_OPTIONS = {
    "include_point_spectra": True,
    "include_line_scans": False,
    "include_area_maps": False,
    "include_series_scans": False,
    "include_photo_images": False,
}


class ImportManager:
    def __init__(self, settings: AppSettings, database: DatabaseService, bridge: ImportBridge):
        self.settings = settings
        self.database = database
        self.bridge = bridge
        self._jobs: dict[str, dict[str, object]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._job_queue: queue.Queue[str] = queue.Queue()
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start_job(
        self,
        *,
        input_path: str,
        recursive: bool,
        force_reimport: bool,
        import_options: dict[str, object] | None = None,
        details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        job_id = f"job-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        normalized_options = self._normalize_import_options(import_options)
        job_details = {
            "phase": "queued",
            "import_options": normalized_options,
        }
        if details:
            job_details.update(details)
        job = {
            "job_id": job_id,
            "input_path": input_path,
            "status": "pending",
            "total_files": 0,
            "processed_files": 0,
            "exported_spectra": 0,
            "failed_files": 0,
            "current_file": None,
            "start_time": datetime.now(UTC).isoformat(),
            "end_time": None,
            "log_path": None,
            "summary_path": None,
            "message": None,
            "recursive": recursive,
            "force_reimport": force_reimport,
            "details": job_details,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._stop_events[job_id] = threading.Event()
        self.database.upsert_import_job(job)
        self._ensure_worker()
        self._job_queue.put(job_id)
        self._console_log(job_id, f"queued import for {input_path}")
        return self.get_job(job_id) or job

    def record_failed_upload(
        self,
        *,
        input_path: str,
        message: str,
        import_options: dict[str, object] | None = None,
        details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        job_id = f"job-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        normalized_options = self._normalize_import_options(import_options)
        stage_detail = message or "Upload staging failed"
        failed_file_summary = {
            "source_wip": input_path,
            "source_file_hash": None,
            "status": "failed",
            "exported_spectra": 0,
            "imported_media_assets": 0,
            "detected_inventory": {},
            "dataset_mode_counts": {},
            "media_inventory": {},
            "datasets": [],
            "class_counts": {},
            "project_version": None,
            "duplicate_of_source": None,
            "error_message": stage_detail,
        }
        result_summary = {
            "imported_file_count": 0,
            "imported_dataset_count": 0,
            "imported_spectra": 0,
            "imported_media_assets": 0,
            "skipped_existing_count": 0,
            "duplicate_file_count": 0,
            "failed_file_count": 1,
            "dataset_mode_counts": {},
            "trace_mode_counts": {},
            "media_type_counts": {},
            "single_file_summary": failed_file_summary,
        }
        job_details = {
            "phase": "failed",
            "phase_label": "Upload failed",
            "stage_detail": stage_detail,
            "current_file_phase": "failed",
            "current_file_index": 1,
            "overall_progress": 100,
            "import_options": normalized_options,
            "result_summary": result_summary,
            "single_file_summary": failed_file_summary,
        }
        if details:
            job_details.update(details)
        job = {
            "job_id": job_id,
            "input_path": input_path,
            "status": "failed",
            "total_files": 1,
            "processed_files": 1,
            "exported_spectra": 0,
            "failed_files": 1,
            "current_file": input_path,
            "start_time": datetime.now(UTC).isoformat(),
            "end_time": datetime.now(UTC).isoformat(),
            "log_path": None,
            "summary_path": None,
            "message": stage_detail,
            "recursive": False,
            "force_reimport": False,
            "details": job_details,
        }
        with self._lock:
            self._jobs[job_id] = job
        self.database.upsert_import_job(job)
        self._console_log(job_id, f"recorded failed upload: {stage_detail}")
        return dict(job)

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            worker = threading.Thread(
                target=self._worker_loop,
                name="pldb-import-worker",
                daemon=True,
            )
            self._worker_thread = worker
            self._threads["worker"] = worker
            worker.start()

    def _worker_loop(self) -> None:
        while True:
            job_id = self._job_queue.get()
            try:
                self._run_job(job_id)
            finally:
                self._job_queue.task_done()

    def _normalize_import_options(self, import_options: dict[str, object] | None) -> dict[str, bool]:
        options = dict(DEFAULT_IMPORT_OPTIONS)
        if import_options:
            for key in DEFAULT_IMPORT_OPTIONS:
                if key in import_options:
                    options[key] = bool(import_options[key])
        return options

    def list_jobs(self) -> list[dict[str, object]]:
        jobs = {item["job_id"]: item for item in self.database.list_import_jobs(limit=100)}
        with self._lock:
            jobs.update({job_id: dict(payload) for job_id, payload in self._jobs.items()})
        return sorted(jobs.values(), key=lambda item: item["start_time"], reverse=True)

    def get_job(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            if job_id in self._jobs:
                return dict(self._jobs[job_id])
        return self.database.get_import_job(job_id)

    def request_stop(self, job_id: str) -> bool:
        with self._lock:
            stop_event = self._stop_events.get(job_id)
            job = dict(self._jobs.get(job_id, {}))
        if stop_event is None or not job:
            persisted_job = self.database.get_import_job(job_id)
            if not persisted_job or persisted_job.get("status") not in {"pending", "running"}:
                return False
            with self._lock:
                stop_event = self._stop_events.setdefault(job_id, threading.Event())
                self._jobs[job_id] = persisted_job
        if stop_event is None:
            return False
        stop_event.set()
        self._update_job(
            job_id,
            message="Stop requested. Finishing the current step before exiting.",
            details={
                "stop_requested": True,
                "phase_label": "Stopping import",
                "stage_detail": "Waiting for the current upload or decode step to stop cleanly",
            },
            persist=False,
        )
        return True

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        try:
            if self._should_stop(job_id):
                raise ImportCancelled("Import was stopped before it started")
            details = dict(job.get("details") or {})
            import_options = dict(details.get("import_options") or DEFAULT_IMPORT_OPTIONS)
            self._update_job(
                job_id,
                status="running",
                message="Starting Python WITec import",
                details={
                    "phase": "reading_wip",
                    "phase_label": "Preparing import",
                    "stage_detail": "Collecting input files and building the import plan",
                    "overall_progress": 0,
                },
            )
            self._console_log(job_id, "starting import worker")
            ingested = 0
            ingested_media_assets = 0
            last_progress_exported = 0
            last_progress_media_assets = 0
            last_progress_time = time.monotonic()

            with self.database.bulk_import_context() as (connection, h5_handle):

                def on_spectrum(
                    source_wip: str,
                    spectrum_meta: dict[str, object],
                    x_axis: np.ndarray,
                    intensity: np.ndarray,
                ) -> None:
                    nonlocal ingested, last_progress_exported, last_progress_time
                    spectrum_id = self._ingest_decoded_spectrum(
                        spectrum_meta,
                        x_axis,
                        intensity,
                        connection=connection,
                        h5_handle=h5_handle,
                    )
                    if not spectrum_id:
                        return
                    ingested += 1
                    now = time.monotonic()
                    if ingested - last_progress_exported >= 25 or now - last_progress_time >= 0.5:
                        self._update_job(
                            job_id,
                            exported_spectra=ingested,
                            current_file=source_wip,
                            details={
                                "phase": "ingesting",
                                "phase_label": "Writing imported data",
                                "stage_detail": self._build_ingest_progress_detail(
                                    imported_spectra=ingested,
                                    imported_media_assets=ingested_media_assets,
                                ),
                                "imported_media_assets": ingested_media_assets,
                            },
                            persist=False,
                        )
                        self._console_log(job_id, f"ingested {ingested} spectra so far ({source_wip})")
                        last_progress_exported = ingested
                        last_progress_time = now

                def on_media(
                    source_wip: str,
                    media_meta: dict[str, object],
                    image_bytes: bytes,
                ) -> None:
                    nonlocal ingested_media_assets, last_progress_media_assets, last_progress_time
                    media_id = self.database.insert_media_asset_with_handles(
                        connection,
                        h5_handle,
                        media_meta,
                        image_bytes,
                    )
                    if not media_id:
                        return
                    ingested_media_assets += 1
                    now = time.monotonic()
                    if ingested_media_assets - last_progress_media_assets >= 5 or now - last_progress_time >= 0.5:
                        self._update_job(
                            job_id,
                            current_file=source_wip,
                            details={
                                "phase": "ingesting",
                                "phase_label": "Writing imported data",
                                "stage_detail": self._build_ingest_progress_detail(
                                    imported_spectra=ingested,
                                    imported_media_assets=ingested_media_assets,
                                ),
                                "imported_media_assets": ingested_media_assets,
                            },
                            persist=False,
                        )
                        self._console_log(job_id, f"ingested {ingested_media_assets} photo assets so far ({source_wip})")
                        last_progress_media_assets = ingested_media_assets
                        last_progress_time = now

                bridge_result = self.bridge.export_input(
                    input_path=str(job["input_path"]),
                    job_id=job_id,
                    recursive=bool(job["recursive"]),
                    force_reimport=bool(job["force_reimport"]),
                    import_options=import_options,
                    progress_callback=lambda payload: self._update_job(job_id, persist=False, **payload),
                    spectrum_callback=on_spectrum,
                    media_callback=on_media,
                    should_stop=lambda: self._should_stop(job_id),
                )
            summary = bridge_result["summary"]
            detected_inventory = summary.get("detected_inventory") or self._aggregate_inventory(summary)
            dataset_mode_counts = self._aggregate_field_counts(summary, field_name="dataset_mode_counts")
            media_inventory = self._aggregate_field_counts(summary, field_name="media_inventory")
            result_summary = self._build_result_summary(
                summary,
                ingested=ingested,
                imported_media_assets=ingested_media_assets,
            )
            self._update_job(
                job_id,
                total_files=int(summary.get("total_files", 0)),
                log_path=bridge_result.get("log_path"),
                summary_path=bridge_result.get("summary_path"),
                exported_spectra=ingested,
                message="Import complete, finalizing job",
                details={
                    "phase": "finalizing",
                    "phase_label": "Finalizing import",
                    "stage_detail": "Compiling the final import summary",
                    "detected_inventory": detected_inventory,
                    "dataset_mode_counts": dataset_mode_counts,
                    "media_inventory": media_inventory,
                    "imported_media_assets": ingested_media_assets,
                    "result_summary": result_summary,
                },
            )
            failed_files = int(summary.get("failed_files", 0))
            skipped_files = int(summary.get("skipped_files", 0))
            processed_files = int(summary.get("processed_files", 0))
            if failed_files == 0:
                status = "finished"
            elif failed_files < processed_files:
                status = "partially_failed"
            else:
                status = "failed"

            message = self._build_completion_message(
                ingested=ingested,
                skipped_files=skipped_files,
                failed_files=failed_files,
                result_summary=result_summary,
            )
            self._update_job(
                job_id,
                status=status,
                processed_files=processed_files,
                failed_files=failed_files,
                exported_spectra=ingested,
                current_file=None,
                end_time=datetime.now(UTC).isoformat(),
                message=message,
                details={
                    "phase": "finished" if status == "finished" else status,
                    "phase_label": "Import complete" if status == "finished" else "Import finished with issues",
                    "stage_detail": message,
                    "detected_inventory": detected_inventory,
                    "dataset_mode_counts": dataset_mode_counts,
                    "media_inventory": media_inventory,
                    "imported_media_assets": ingested_media_assets,
                    "result_summary": result_summary,
                    "single_file_summary": result_summary.get("single_file_summary"),
                },
            )
            self._console_log(job_id, message)
        except ImportCancelled as error:
            message = str(error) or "Import stopped by the user"
            self._update_job(
                job_id,
                status="cancelled",
                end_time=datetime.now(UTC).isoformat(),
                message=message,
                details={
                    "phase": "cancelled",
                    "phase_label": "Import stopped",
                    "stage_detail": message,
                    "stop_requested": True,
                },
            )
            self._console_log(job_id, message)
        except Exception as error:
            self._update_job(
                job_id,
                status="failed",
                end_time=datetime.now(UTC).isoformat(),
                message=str(error),
                details={
                    "phase": "failed",
                    "phase_label": "Import failed",
                    "stage_detail": str(error),
                },
            )
            self._console_log(job_id, f"failed: {error}")
        finally:
            latest_job = self.get_job(job_id)
            if latest_job:
                self._cleanup_staged_upload(job_id, latest_job)
            with self._lock:
                self._stop_events.pop(job_id, None)

    def _aggregate_inventory(self, summary: dict[str, object]) -> dict[str, int]:
        return self._aggregate_field_counts(summary, field_name="detected_inventory")

    def _cleanup_staged_upload(self, job_id: str, job: dict[str, object]) -> None:
        details = dict(job.get("details") or {})
        if not bool(details.get("cleanup_staged_upload")):
            return

        try:
            staged_root = Path(str(job["input_path"])).resolve()
            raw_root = (self.settings.project_root / "data" / "raw_wip").resolve()
        except Exception:
            return

        if raw_root not in staged_root.parents or not staged_root.name.startswith("upload-"):
            return
        if not staged_root.exists():
            return

        shutil.rmtree(staged_root, ignore_errors=True)
        self._console_log(job_id, f"cleaned staged upload folder {staged_root}")

    def _aggregate_field_counts(self, summary: dict[str, object], *, field_name: str) -> dict[str, int]:
        aggregate: dict[str, int] = {}
        files = summary.get("files", [])
        if not isinstance(files, list):
            return aggregate
        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            inventory = file_entry.get(field_name)
            if not isinstance(inventory, dict):
                continue
            for key, value in inventory.items():
                aggregate[str(key)] = aggregate.get(str(key), 0) + int(value or 0)
        return aggregate

    def _build_result_summary(
        self,
        summary: dict[str, object],
        *,
        ingested: int,
        imported_media_assets: int,
    ) -> dict[str, object]:
        files = summary.get("files", [])
        if not isinstance(files, list):
            files = []

        imported_file_count = 0
        imported_dataset_count = 0
        skipped_existing_count = 0
        duplicate_file_count = int(summary.get("duplicate_files", 0) or 0)
        failed_file_count = int(summary.get("failed_files", 0) or 0)
        single_file_summary: dict[str, object] | None = None

        for file_entry in files:
            if not isinstance(file_entry, dict):
                continue
            status = str(file_entry.get("status") or "")
            dataset_counts = file_entry.get("dataset_mode_counts")
            media_asset_count = int(file_entry.get("imported_media_assets") or 0)
            if isinstance(dataset_counts, dict):
                imported_dataset_count += sum(int(value or 0) for value in dataset_counts.values())
            if status == "success" and (
                int(file_entry.get("exported_spectra") or 0) > 0 or media_asset_count > 0
            ):
                imported_file_count += 1
            elif status == "skipped_existing":
                skipped_existing_count += 1
            if single_file_summary is None:
                single_file_summary = file_entry

        result_summary = {
            "imported_file_count": imported_file_count,
            "imported_dataset_count": imported_dataset_count,
            "imported_spectra": ingested,
            "imported_media_assets": imported_media_assets,
            "skipped_existing_count": skipped_existing_count,
            "duplicate_file_count": duplicate_file_count,
            "failed_file_count": failed_file_count,
            "dataset_mode_counts": self._aggregate_field_counts(summary, field_name="dataset_mode_counts"),
            "trace_mode_counts": self._aggregate_inventory(summary),
            "media_type_counts": self._aggregate_field_counts(summary, field_name="media_inventory"),
        }
        if int(summary.get("total_files", 0) or 0) == 1 and single_file_summary is not None:
            result_summary["single_file_summary"] = single_file_summary
        return result_summary

    def _build_completion_message(
        self,
        *,
        ingested: int,
        skipped_files: int,
        failed_files: int,
        result_summary: dict[str, object],
    ) -> str:
        imported_file_count = int(result_summary.get("imported_file_count") or 0)
        imported_dataset_count = int(result_summary.get("imported_dataset_count") or 0)
        imported_media_assets = int(result_summary.get("imported_media_assets") or 0)
        duplicate_file_count = int(result_summary.get("duplicate_file_count") or 0)
        if ingested > 0 or imported_media_assets > 0:
            parts: list[str] = []
            if ingested > 0:
                parts.append(f"{ingested} spectra")
            if imported_media_assets > 0:
                parts.append(f"{imported_media_assets} photo assets")
            message = f"Imported {' and '.join(parts)} from {imported_file_count} file"
            if imported_file_count != 1:
                message += "s"
            if imported_dataset_count > 0:
                message += f" across {imported_dataset_count} datasets"
            if skipped_files:
                message += f" ({skipped_files} files skipped)"
            return message
        if skipped_files and failed_files == 0:
            if duplicate_file_count:
                return f"No new data imported ({duplicate_file_count} duplicate files skipped)"
            return f"No new data imported ({skipped_files} files skipped)"
        return "No data imported"

    def _build_ingest_progress_detail(self, *, imported_spectra: int, imported_media_assets: int) -> str:
        if imported_spectra > 0 and imported_media_assets > 0:
            return f"Persisted {imported_spectra} spectra and {imported_media_assets} photo assets so far"
        if imported_spectra > 0:
            return f"Persisted {imported_spectra} spectra so far"
        if imported_media_assets > 0:
            return f"Persisted {imported_media_assets} photo assets so far"
        return "Waiting for decoded datasets"

    def _ingest_decoded_spectrum(
        self,
        spectrum_meta: dict[str, object],
        x_axis: list[float] | np.ndarray,
        intensity: list[float] | np.ndarray,
        *,
        connection,
        h5_handle,
    ) -> str | None:
        source_wip = normalize_source_path(str(spectrum_meta.get("source_wip", "unknown")))
        metadata = parse_metadata_from_path(source_wip)
        metadata.update(spectrum_meta)
        metadata["source_wip"] = source_wip
        metadata.setdefault("source_tree_path", "/WITioRaw/graph/unknown")

        x_values = np.asarray(x_axis, dtype=float)
        y_values = np.asarray(intensity, dtype=float)

        x_axis_unit = str(metadata.get("x_axis_unit") or infer_x_axis_unit(x_values.tolist()))
        metadata["x_axis_unit"] = x_axis_unit
        metadata.setdefault(
            "spectrum_id",
            build_spectrum_id(
                source_wip,
                str(metadata.get("source_tree_path", "")),
                int(metadata.get("trace_index") or 0),
            ),
        )
        metadata["spectrum_type"] = infer_spectrum_type_with_context(
            x_values.tolist(),
            x_axis_unit,
            grating=metadata.get("grating"),
        )
        metadata["source"] = str(metadata.get("source") or metadata.get("material") or metadata.get("sample_id") or "")

        measurement_config = metadata.get("measurement_config")
        if not isinstance(measurement_config, dict):
            measurement_config = {}
        measurement_config.update(
            {
                "x_min": float(x_values.min()) if len(x_values) else None,
                "x_max": float(x_values.max()) if len(x_values) else None,
                "n_points": int(len(x_values)),
                "x_axis_unit": x_axis_unit,
            }
        )
        metadata["measurement_config"] = measurement_config

        spectrum_id = self.database.insert_spectrum_bulk(
            connection,
            h5_handle,
            metadata,
            x_values,
            y_values,
        )
        return spectrum_id

    def _update_job(self, job_id: str, *, persist: bool = True, **changes: object) -> None:
        with self._lock:
            job = dict(self._jobs.get(job_id, {}))
            if "details" in changes:
                merged_details = dict(job.get("details") or {})
                next_details = changes.get("details")
                if isinstance(next_details, dict):
                    merged_details.update(next_details)
                changes = dict(changes)
                changes["details"] = merged_details
            job.update(changes)
            self._jobs[job_id] = job
        if job and persist:
            self.database.upsert_import_job(job)

    def _should_stop(self, job_id: str) -> bool:
        with self._lock:
            stop_event = self._stop_events.get(job_id)
        return bool(stop_event and stop_event.is_set())

    def _console_log(self, job_id: str, message: str) -> None:
        print(f"[{datetime.now(UTC).isoformat()}] [import {job_id}] {message}", flush=True)
