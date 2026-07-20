from __future__ import annotations

import threading
import uuid
from collections import Counter
from datetime import UTC, datetime

from .database import DatabaseService
from .material_analysis import ANALYSIS_METHOD_VERSION, analyze_material_spectrum


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class MaterialAnalysisJobManager:
    def __init__(self, database: DatabaseService):
        self.database = database
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, object]] = {}
        self._stop_requests: set[str] = set()

    def start_job(
        self,
        *,
        spectrum_ids: list[str],
        options: dict[str, object],
        save_results: bool,
        update_entries: bool,
    ) -> dict[str, object]:
        job_id = f"analysis-job-{uuid.uuid4().hex[:12]}"
        unique_ids = list(dict.fromkeys(str(item) for item in spectrum_ids if str(item).strip()))
        job = {
            "job_id": job_id,
            "status": "queued",
            "total": len(unique_ids),
            "processed": 0,
            "failed": 0,
            "updated": 0,
            "started_at": utc_now(),
            "ended_at": None,
            "current_spectrum_id": None,
            "message": "Queued material-aware analysis",
            "method_version": str(options.get("method_version") or ANALYSIS_METHOD_VERSION),
            "logs": [],
            "latest_result": None,
            "spectrum_ids": unique_ids,
            "result_summaries": {},
            "failed_ids": {},
            "summary": {
                "materials": {},
                "families": {},
            },
        }
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, unique_ids, dict(options), save_results, update_entries),
            name=f"pldb-material-analysis-{job_id}",
            daemon=True,
        )
        thread.start()
        return self.get_job(job_id) or job

    def get_job(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            payload = {
                **job,
                "logs": list(job.get("logs", [])),
                "summary": dict(job.get("summary", {})),
            }
        payload["queue_window"] = self._build_queue_window(payload)
        payload.pop("spectrum_ids", None)
        payload.pop("result_summaries", None)
        payload.pop("failed_ids", None)
        return payload

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._stop_requests.add(job_id)
            self._append_log_locked(job_id, "Stop requested; current spectrum will finish first.")
            return True

    def _run_job(
        self,
        job_id: str,
        spectrum_ids: list[str],
        options: dict[str, object],
        save_results: bool,
        update_entries: bool,
    ) -> None:
        material_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        updated_count = 0
        failed_count = 0
        self._set_job(job_id, status="running", message="Running material-aware analysis")
        self._append_log(job_id, f"Started analysis for {len(spectrum_ids)} spectrum entries.")

        for index, identifier in enumerate(spectrum_ids, start=1):
            with self._lock:
                if job_id in self._stop_requests:
                    self._jobs[job_id]["status"] = "stopped"
                    self._jobs[job_id]["ended_at"] = utc_now()
                    self._jobs[job_id]["message"] = "Stopped by user"
                    self._append_log_locked(job_id, "Stopped before queue completion.")
                    return
                self._jobs[job_id]["current_spectrum_id"] = identifier

            try:
                detail = self.database.get_spectrum(identifier)
                if detail is None:
                    raise ValueError("Spectrum not found")
                target_id = str(detail.get("representative_spectrum_id") or identifier)
                result = analyze_material_spectrum(
                    detail["x_axis"],
                    detail["intensity"],
                    {
                        "spectrum_type": detail.get("spectrum_type"),
                        "x_axis_unit": detail.get("x_axis_unit"),
                        "laser_wavelength": detail.get("laser_wavelength"),
                        "grating": detail.get("grating"),
                        "material": detail.get("source"),
                    },
                    options,
                )
                result_id = None
                if save_results:
                    result_id = self.database.insert_analysis_result(
                        target_id,
                        result,
                        method=str(result.get("method_version") or ANALYSIS_METHOD_VERSION),
                        parameters=options,
                    )
                updated = self.database.update_spectrum_analysis_summary(
                    target_id,
                    result,
                    method=str(result.get("method_version") or ANALYSIS_METHOD_VERSION),
                ) if update_entries else 0
                material = str(result.get("material") or "unknown")
                family = str(result.get("spectrum_family") or "unknown")
                quality = result.get("quality") if isinstance(result.get("quality"), dict) else {}
                updated_count += int(updated > 0)
                material_counts[material] += 1
                family_counts[family] += 1
                latest = {
                    "spectrum_id": identifier,
                    "representative_spectrum_id": target_id,
                    "result_id": result_id,
                    "file_path": detail.get("file_path"),
                    "analysis": result,
                }
                result_summary = {
                    "material": material,
                    "family": family,
                    "fit_quality": quality.get("score"),
                    "fit_quality_label": quality.get("label"),
                    "fit_r2": result.get("metrics", {}).get("r2") if isinstance(result.get("metrics"), dict) else None,
                    "updated": bool(updated > 0),
                }
                self._set_job(
                    job_id,
                    processed=index,
                    updated=updated_count,
                    latest_result=latest,
                    result_summaries={
                        **self._snapshot_result_summaries(job_id),
                        identifier: result_summary,
                    },
                    summary={"materials": dict(material_counts), "families": dict(family_counts)},
                    message=f"Processed {index} of {len(spectrum_ids)}",
                )
                self._append_log(job_id, f"{index}/{len(spectrum_ids)} {identifier}: {family} {material}, updated={int(updated > 0)}")
            except Exception as error:
                failed_count += 1
                self._set_job(
                    job_id,
                    processed=index,
                    failed=failed_count,
                    failed_ids={
                        **self._snapshot_failed_ids(job_id),
                        identifier: str(error),
                    },
                    message=f"Processed {index} of {len(spectrum_ids)} with errors",
                )
                self._append_log(job_id, f"{index}/{len(spectrum_ids)} {identifier}: failed - {error}")

        self._set_job(
            job_id,
            status="finished",
            ended_at=utc_now(),
            current_spectrum_id=None,
            message=f"Finished {len(spectrum_ids)} spectrum entries",
            summary={"materials": dict(material_counts), "families": dict(family_counts)},
        )
        self._append_log(job_id, "Completed material-aware analysis.")

    def _set_job(self, job_id: str, **updates: object) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(updates)

    def _append_log(self, job_id: str, message: str) -> None:
        with self._lock:
            self._append_log_locked(job_id, message)

    def _append_log_locked(self, job_id: str, message: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        logs = list(job.get("logs", []))
        logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        job["logs"] = logs[-80:]

    def _snapshot_result_summaries(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id) or {}
            return dict(job.get("result_summaries") or {})

    def _snapshot_failed_ids(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id) or {}
            return dict(job.get("failed_ids") or {})

    def _build_queue_window(self, job: dict[str, object]) -> list[dict[str, object]]:
        spectrum_ids = [str(item) for item in job.get("spectrum_ids", []) if str(item)]
        if not spectrum_ids:
            return []
        processed = int(job.get("processed") or 0)
        start = max(0, processed - 5)
        end = min(len(spectrum_ids), processed + 7)
        result_summaries = dict(job.get("result_summaries") or {})
        failed_ids = dict(job.get("failed_ids") or {})
        current_id = str(job.get("current_spectrum_id") or "")

        window: list[dict[str, object]] = []
        for order_index in range(start, end):
            identifier = spectrum_ids[order_index]
            summary = result_summaries.get(identifier) if isinstance(result_summaries.get(identifier), dict) else {}
            status = self._queue_item_status(
                identifier,
                order_index,
                processed=processed,
                current_id=current_id,
                failed_ids=failed_ids,
                job_status=str(job.get("status") or ""),
            )
            detail = self.database.get_spectrum(identifier)
            window.append(
                {
                    "spectrum_id": identifier,
                    "order": order_index + 1,
                    "status": status,
                    "sample_id": detail.get("sample_id") if detail else None,
                    "source": detail.get("source") if detail else None,
                    "spectrum_type": detail.get("spectrum_type") if detail else None,
                    "acquisition_mode": detail.get("acquisition_mode") if detail else None,
                    "sparkline": self._sparkline(detail.get("intensity", []) if detail else []),
                    "material": summary.get("material") if isinstance(summary, dict) else None,
                    "family": summary.get("family") if isinstance(summary, dict) else None,
                    "fit_quality": summary.get("fit_quality") if isinstance(summary, dict) else None,
                    "fit_quality_label": summary.get("fit_quality_label") if isinstance(summary, dict) else None,
                    "error": failed_ids.get(identifier),
                }
            )
        return window

    def _queue_item_status(
        self,
        identifier: str,
        order_index: int,
        *,
        processed: int,
        current_id: str,
        failed_ids: dict[str, object],
        job_status: str,
    ) -> str:
        if identifier == current_id and job_status in {"queued", "running"}:
            return "running"
        if order_index < processed:
            return "failed" if identifier in failed_ids else "processed"
        return "pending"

    def _sparkline(self, values: object, *, size: int = 48) -> list[float]:
        try:
            raw = [float(value) for value in values]
        except Exception:
            return []
        if not raw:
            return []
        if len(raw) > size:
            step = len(raw) / size
            sampled = [raw[min(len(raw) - 1, int(index * step))] for index in range(size)]
        else:
            sampled = raw
        minimum = min(sampled)
        maximum = max(sampled)
        span = maximum - minimum
        if span <= 0:
            return [0.5 for _ in sampled]
        return [(value - minimum) / span for value in sampled]
