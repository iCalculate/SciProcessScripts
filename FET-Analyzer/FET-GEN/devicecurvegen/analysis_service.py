from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any
from uuid import uuid4

from .database import analyze_curves
from .schemas import DatabaseAnalysisStatus


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class DatabaseAnalysisManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="db-analysis")
        self._state = DatabaseAnalysisStatus()
        self._started_monotonic: float | None = None

    def snapshot(self) -> DatabaseAnalysisStatus:
        with self._lock:
            state = self._state.model_copy(deep=True)
            if state.status == "running" and self._started_monotonic is not None:
                state.elapsed_seconds = time.monotonic() - self._started_monotonic
            return state

    def start(
        self,
        *,
        curve_ids: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        selected_count: int = 0,
    ) -> DatabaseAnalysisStatus:
        with self._lock:
            if self._state.status == "running":
                raise RuntimeError("A database analysis job is already running")
            job_id = uuid4().hex
            self._started_monotonic = time.monotonic()
            self._state = DatabaseAnalysisStatus(
                status="running",
                stage="loading_selection",
                job_id=job_id,
                message="Loading selected curves from the database",
                started_at=_timestamp(),
                progress_fraction=0.0,
                selected_count=selected_count,
            )
        self._executor.submit(self._run, job_id, curve_ids or None, filters or {})
        return self.snapshot()

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            if self._state.job_id != job_id:
                return
            self._state = self._state.model_copy(update=changes)

    def _run(
        self,
        job_id: str,
        curve_ids: list[str] | None,
        filters: dict[str, Any],
    ) -> None:
        try:
            result = analyze_curves(
                curve_ids=curve_ids,
                filters=filters,
                progress=lambda update: self._handle_progress(job_id, update),
            )
            elapsed = (
                time.monotonic() - self._started_monotonic
                if self._started_monotonic is not None
                else 0.0
            )
            self._update(
                job_id,
                status="completed",
                stage="completed",
                message="Analysis completed",
                completed_at=_timestamp(),
                elapsed_seconds=elapsed,
                progress_fraction=1.0,
                selected_count=result.get("count", 0),
                result=result,
                error=None,
            )
        except Exception as error:
            elapsed = (
                time.monotonic() - self._started_monotonic
                if self._started_monotonic is not None
                else 0.0
            )
            self._update(
                job_id,
                status="failed",
                stage="failed",
                message="Analysis failed",
                completed_at=_timestamp(),
                elapsed_seconds=elapsed,
                error=str(error),
            )

    def _handle_progress(self, job_id: str, update: dict[str, Any]) -> None:
        progress_fraction = update.get("progress_fraction")
        if isinstance(progress_fraction, (int, float)):
            update["progress_fraction"] = max(0.0, min(0.99, float(progress_fraction)))
        self._update(job_id, **update)
