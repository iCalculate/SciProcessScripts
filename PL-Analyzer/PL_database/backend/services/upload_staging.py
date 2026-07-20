from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from fastapi import UploadFile

from ..config import AppSettings


def stage_upload_files(
    settings: AppSettings,
    *,
    files: list[UploadFile],
    relative_paths: list[str] | None = None,
    root_name: str | None = None,
) -> tuple[Path, int]:
    staged_root = settings.project_root / "data" / "raw_wip" / _build_upload_folder_name(root_name)
    staged_root.mkdir(parents=True, exist_ok=True)

    uploaded_count = 0
    relative_values = list(relative_paths or [])

    try:
        for index, upload in enumerate(files):
            filename = Path(str(upload.filename or f"upload_{index + 1}.wip")).name
            relative_value = relative_values[index] if index < len(relative_values) else filename
            safe_relative = sanitize_upload_relative_path(relative_value, default_name=filename)
            if safe_relative.suffix.lower() != ".wip":
                continue

            target_path = _allocate_unique_target(staged_root / safe_relative)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                upload.file.seek(0)
            except Exception:
                pass
            with target_path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            uploaded_count += 1
    except Exception:
        shutil.rmtree(staged_root, ignore_errors=True)
        raise

    if uploaded_count == 0:
        shutil.rmtree(staged_root, ignore_errors=True)
        raise ValueError("No .wip files were uploaded.")

    return staged_root, uploaded_count


def sanitize_upload_relative_path(relative_path: str | None, *, default_name: str) -> Path:
    fallback_name = Path(default_name).name or "uploaded_file.wip"
    raw_value = str(relative_path or "").strip().replace("\\", "/")
    if not raw_value:
        raw_value = fallback_name

    candidate = PurePosixPath(raw_value)
    if candidate.is_absolute() or re.match(r"^[A-Za-z]:", raw_value):
        raise ValueError(f"Upload path must be relative: {relative_path!r}")

    parts = [part for part in candidate.parts if part not in {"", "."}]
    if not parts:
        parts = [fallback_name]
    if any(part == ".." for part in parts):
        raise ValueError(f"Upload path cannot traverse parent folders: {relative_path!r}")

    return Path(*parts)


def _build_upload_folder_name(root_name: str | None) -> str:
    cleaned_root = Path(str(root_name or "").strip()).name
    cleaned_root = re.sub(r"[^\w.-]+", "_", cleaned_root, flags=re.UNICODE).strip("._")
    label = cleaned_root or "selected_folder"
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"upload-{timestamp}-{uuid4().hex[:6]}-{label}"


def _allocate_unique_target(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    suffix = target_path.suffix
    stem = target_path.stem
    counter = 2
    while True:
        candidate = target_path.with_name(f"{stem}__{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
