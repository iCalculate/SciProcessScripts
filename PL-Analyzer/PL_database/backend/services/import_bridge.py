from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import AppSettings
from .database import DatabaseService
from .file_fingerprint import compute_file_sha256
from .media_assets import encode_photo_asset
from .metadata_parser import (
    build_media_id,
    build_spectrum_id,
    infer_spectrum_type_with_context,
    infer_x_axis_unit,
    normalize_source_path,
    parse_metadata_from_path,
    strip_wrapped_quotes,
)
from .witio_import import (
    AREA_MAP,
    LINE_SCAN,
    POINT_SPECTRUM,
    SERIES_SCAN,
    build_witio_trace_metadata_lookup,
    describe_witio_photo_entry,
    describe_witio_graph,
    extract_witio_project_metadata,
    iter_witio_dataset_traces,
    load_witio,
    load_witio_dataset_arrays,
    probe_witio_file,
    should_include_descriptor,
    should_include_photo_descriptor,
)


ProgressCallback = Callable[[dict[str, object]], None]
SpectrumCallback = Callable[[str, dict[str, object], np.ndarray, np.ndarray], None]
MediaCallback = Callable[[str, dict[str, object], bytes], None]
ShouldStopCallback = Callable[[], bool]


class ImportCancelled(RuntimeError):
    pass


class ImportBridge:
    """Python-native WITec importer.

    The class name intentionally avoids MATLAB terminology, but it keeps the old
    `export_input()` entrypoint so the rest of the app can stay stable while we
    switch the import backend to `witio`.
    """

    def __init__(self, settings: AppSettings, database: DatabaseService | None = None):
        self.settings = settings
        self.database = database

    def export_input(
        self,
        *,
        input_path: str,
        job_id: str,
        recursive: bool,
        force_reimport: bool,
        import_options: dict[str, object] | None = None,
        progress_callback: ProgressCallback | None = None,
        spectrum_callback: SpectrumCallback | None = None,
        media_callback: MediaCallback | None = None,
        should_stop: ShouldStopCallback | None = None,
    ) -> dict[str, object]:
        normalized_input = strip_wrapped_quotes(input_path.strip())
        resolved_input = Path(normalized_input).expanduser()
        output_root = self.settings.export_root / job_id
        output_root.mkdir(parents=True, exist_ok=True)
        log_path = self.settings.log_root / f"{job_id}.log"

        if self._should_use_mock(normalized_input, resolved_input):
            return self._run_mock_export(
                resolved_input,
                output_root,
                log_path,
                job_id=job_id,
                recursive=recursive,
                force_reimport=force_reimport,
                import_options=import_options or {},
                progress_callback=progress_callback,
                spectrum_callback=spectrum_callback,
                media_callback=media_callback,
                should_stop=should_stop,
            )

        if not resolved_input.exists():
            raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

        files = self._collect_input_files(resolved_input, recursive=recursive)
        if not files:
            raise FileNotFoundError(f"No .wip files found under: {resolved_input}")

        return self._run_witio_import(
            job_id=job_id,
            files=files,
            input_path=resolved_input,
            output_root=output_root,
            log_path=log_path,
            force_reimport=force_reimport,
            import_options=import_options or {},
            progress_callback=progress_callback,
            spectrum_callback=spectrum_callback,
            media_callback=media_callback,
            should_stop=should_stop,
        )

    def probe_input_file(self, input_path: str) -> dict[str, object]:
        normalized_input = strip_wrapped_quotes(input_path.strip())
        resolved_input = Path(normalized_input).expanduser()
        if not resolved_input.exists():
            raise FileNotFoundError(f"Input path does not exist: {resolved_input}")
        if not resolved_input.is_file():
            raise FileNotFoundError(f"Probe expects a single .wip file: {resolved_input}")
        report, _ = probe_witio_file(resolved_input)
        return report

    def _run_witio_import(
        self,
        *,
        job_id: str,
        files: list[Path],
        input_path: Path,
        output_root: Path,
        log_path: Path,
        force_reimport: bool,
        import_options: dict[str, object],
        progress_callback: ProgressCallback | None,
        spectrum_callback: SpectrumCallback | None,
        media_callback: MediaCallback | None,
        should_stop: ShouldStopCallback | None,
    ) -> dict[str, object]:
        witio = load_witio()
        total_files = len(files)
        summary_files: list[dict[str, object]] = []
        exported_spectra = 0
        imported_media_assets = 0
        failed_files = 0
        skipped_files = 0
        duplicate_files = 0
        seen_hash_sources: dict[str, str] = {}
        log_lines = [
            f"[{self._timestamp()}] import_backend=witio",
            f"[{self._timestamp()}] input={input_path}",
            f"[{self._timestamp()}] force_reimport={force_reimport}",
            f"[{self._timestamp()}] import_options={json.dumps(import_options, ensure_ascii=False, sort_keys=True)}",
        ]
        self._console_log(job_id, f"discovered {total_files} WIP files under {input_path}")

        for index, file_path in enumerate(files, start=1):
            self._raise_if_cancelled(should_stop)
            normalized_source = normalize_source_path(str(file_path))
            if progress_callback:
                progress_callback(
                    {
                        "status": "running",
                        "current_file": normalized_source,
                        "processed_files": index - 1,
                        "total_files": total_files,
                        "details": {
                            "phase": "hashing_file",
                            "phase_label": "Hashing current file",
                            "stage_detail": f"Preparing file {index} of {total_files}",
                            "current_file_phase": "hashing",
                            "current_file_index": index,
                            "overall_progress": round(((index - 1) / total_files) * 100) if total_files > 0 else 0,
                        },
                    }
                )
            self._console_log(job_id, f"[{index}/{total_files}] hashing {normalized_source}")

            source_file_hash = compute_file_sha256(file_path)
            existing_count = self.database.count_import_records_for_source(normalized_source) if self.database is not None else 0
            existing_hash_count = (
                self.database.count_import_records_for_source_hash(source_file_hash) if self.database is not None else 0
            )
            existing_hash_sources = (
                self.database.list_sources_for_hash(source_file_hash) if self.database is not None and existing_hash_count > 0 else []
            )
            duplicate_source = seen_hash_sources.get(source_file_hash)
            if duplicate_source is None:
                duplicate_source = next((item for item in existing_hash_sources if item != normalized_source), None)

            if duplicate_source is not None or (existing_hash_count > 0 and not force_reimport):
                skipped_files += 1
                status = "skipped_duplicate" if duplicate_source else "skipped_existing"
                if duplicate_source:
                    duplicate_files += 1
                    error_message = f"Skipped duplicate file content (same as {duplicate_source})"
                else:
                    error_message = f"Skipped existing import ({existing_hash_count} spectra already indexed)"
                file_summary = {
                    "source_wip": normalized_source,
                    "source_file_hash": source_file_hash,
                    "status": status,
                    "exported_spectra": 0,
                    "imported_media_assets": 0,
                    "detected_inventory": {},
                    "dataset_mode_counts": {},
                    "media_inventory": {},
                    "datasets": [],
                    "class_counts": {},
                    "project_version": None,
                    "duplicate_of_source": duplicate_source,
                    "error_message": error_message,
                }
                summary_files.append(file_summary)
                log_lines.append(
                    f"[{self._timestamp()}] {status} {normalized_source}: "
                    f"existing_hash_count={existing_hash_count} duplicate_of={duplicate_source or normalized_source}"
                )
                self._console_log(job_id, f"[{index}/{total_files}] {status} {normalized_source}")
                if progress_callback:
                    progress_callback(
                        {
                            "status": "running",
                            "current_file": normalized_source,
                            "processed_files": index,
                            "total_files": total_files,
                            "details": {
                                "phase": "analyzing_file",
                                "phase_label": "Skipping file",
                                "stage_detail": file_summary["error_message"],
                                "current_file_phase": "skipped",
                                "current_file_index": index,
                                "overall_progress": round((index / total_files) * 100) if total_files > 0 else 100,
                                "last_file_inventory": {},
                                "last_file_dataset_counts": {},
                                "last_file_media_inventory": {},
                                "last_file_summary": file_summary,
                            },
                        }
                    )
                seen_hash_sources.setdefault(source_file_hash, duplicate_source or normalized_source)
                continue

            if existing_hash_count > 0 and force_reimport:
                deleted = self.database.delete_import_records_for_source_hash(source_file_hash) if self.database is not None else 0
                log_lines.append(
                    f"[{self._timestamp()}] cleared previous import by file hash {normalized_source}: deleted={deleted}"
                )
            elif existing_count > 0 and force_reimport:
                deleted = self.database.delete_import_records_for_source(normalized_source) if self.database is not None else 0
                log_lines.append(f"[{self._timestamp()}] cleared previous import {normalized_source}: deleted={deleted}")

            try:
                if progress_callback:
                    progress_callback(
                        {
                            "status": "running",
                            "current_file": normalized_source,
                            "processed_files": index - 1,
                            "total_files": total_files,
                            "details": {
                                "phase": "analyzing_file",
                                "phase_label": "Analyzing WIP structure",
                                "stage_detail": f"Inspecting file {index} of {total_files}",
                                "current_file_phase": "analyzing",
                                "current_file_index": index,
                                "overall_progress": round(((index - 1) / total_files) * 100) if total_files > 0 else 0,
                            },
                        }
                    )
                self._console_log(job_id, f"[{index}/{total_files}] analyzing {normalized_source}")
                project = witio.read(file_path)
                trace_metadata_lookup = build_witio_trace_metadata_lookup(project)
                project_metadata = extract_witio_project_metadata(project)
                class_counts: dict[str, int] = {}
                file_inventory: dict[str, int] = {}
                file_dataset_counts: dict[str, int] = {}
                file_media_inventory: dict[str, int] = {}
                dataset_summaries: list[dict[str, object]] = []
                file_exported_spectra = 0
                file_imported_media_assets = 0
                base_path_metadata = parse_metadata_from_path(normalized_source)

                for entry in project.data:
                    self._raise_if_cancelled(should_stop)
                    class_name = str(getattr(entry, "class_name", "unknown"))
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                    descriptor = describe_witio_graph(
                        entry,
                        trace_metadata_lookup=trace_metadata_lookup,
                        project_metadata=project_metadata,
                    )
                    if descriptor is not None and should_include_descriptor(descriptor, import_options):
                        dataset_summary, x_axis_values, graph_array, position_x, position_y = load_witio_dataset_arrays(
                            entry,
                            descriptor=descriptor,
                        )
                        dataset_summaries.append(dataset_summary)
                        mode = str(descriptor["acquisition_mode"])
                        dataset_trace_count = int(descriptor["trace_count"])
                        file_inventory[mode] = file_inventory.get(mode, 0) + dataset_trace_count
                        file_dataset_counts[mode] = file_dataset_counts.get(mode, 0) + 1

                        for decoded_trace in iter_witio_dataset_traces(
                            descriptor=descriptor,
                            source_wip=file_path,
                            x_axis_values=x_axis_values,
                            graph_array=graph_array,
                            position_x=position_x,
                            position_y=position_y,
                        ):
                            self._raise_if_cancelled(should_stop)
                            metadata = dict(base_path_metadata)
                            metadata.update(decoded_trace.metadata)
                            metadata["source_wip"] = normalized_source
                            metadata["source_file_hash"] = source_file_hash
                            metadata.setdefault(
                                "spectrum_id",
                                build_spectrum_id(
                                    normalized_source,
                                    str(metadata.get("source_tree_path", "")),
                                    int(metadata.get("trace_index") or 0),
                                ),
                            )
                            metadata.setdefault(
                                "x_axis_unit",
                                str(metadata.get("x_axis_unit") or infer_x_axis_unit(decoded_trace.x_axis.tolist())),
                            )
                            metadata["spectrum_type"] = infer_spectrum_type_with_context(
                                decoded_trace.x_axis.tolist(),
                                str(metadata["x_axis_unit"]),
                                grating=metadata.get("grating"),
                            )
                            metadata["source"] = str(
                                metadata.get("source") or metadata.get("material") or metadata.get("sample_id") or ""
                            )
                            if spectrum_callback:
                                spectrum_callback(
                                    normalized_source,
                                    metadata,
                                    decoded_trace.x_axis,
                                    decoded_trace.intensity,
                                )
                            file_exported_spectra += 1
                            exported_spectra += 1

                    photo_descriptor = describe_witio_photo_entry(
                        entry,
                        trace_metadata_lookup=trace_metadata_lookup,
                        project_metadata=project_metadata,
                    )
                    if photo_descriptor is None or not should_include_photo_descriptor(import_options):
                        continue

                    encoded_photo = encode_photo_asset(entry.array())
                    self._raise_if_cancelled(should_stop)
                    entry_metadata = photo_descriptor.get("entry_metadata")
                    measurement_config = dict(entry_metadata) if isinstance(entry_metadata, dict) else {}
                    measurement_config.update(
                        {
                            "entry_id": photo_descriptor.get("entry_id"),
                            "entry_class": photo_descriptor.get("entry_class"),
                            "caption": photo_descriptor.get("caption"),
                            "extraction_backend": "witio",
                            "estimated_raw_array_mb": photo_descriptor.get("estimated_raw_array_mb"),
                            "original_width_px": photo_descriptor.get("width_px"),
                            "original_height_px": photo_descriptor.get("height_px"),
                            "compressed_width_px": encoded_photo.width_px,
                            "compressed_height_px": encoded_photo.height_px,
                            "channel_count": encoded_photo.channel_count,
                            "bit_depth": encoded_photo.bit_depth,
                        }
                    )
                    photo_metadata = dict(base_path_metadata)
                    if isinstance(entry_metadata, dict):
                        photo_metadata.update(entry_metadata)
                    photo_metadata.update(
                        {
                            "media_id": build_media_id(normalized_source, str(photo_descriptor["source_tree_path"])),
                            "source_wip": normalized_source,
                            "source_file_hash": source_file_hash,
                            "source_tree_path": str(photo_descriptor["source_tree_path"]),
                            "media_kind": str(photo_descriptor["media_kind"]),
                            "entry_class": str(photo_descriptor["entry_class"]),
                            "caption": str(photo_descriptor["caption"]),
                            "asset_format": encoded_photo.asset_format,
                            "width_px": encoded_photo.width_px,
                            "height_px": encoded_photo.height_px,
                            "original_width_px": encoded_photo.original_width_px,
                            "original_height_px": encoded_photo.original_height_px,
                            "channel_count": encoded_photo.channel_count,
                            "bit_depth": encoded_photo.bit_depth,
                            "measurement_time": photo_descriptor.get("measurement_time"),
                            "measurement_config": measurement_config,
                        }
                    )
                    if media_callback:
                        media_callback(normalized_source, photo_metadata, encoded_photo.image_bytes)
                    media_kind = str(photo_descriptor["media_kind"])
                    file_media_inventory[media_kind] = file_media_inventory.get(media_kind, 0) + 1
                    file_imported_media_assets += 1
                    imported_media_assets += 1

                file_summary = {
                    "source_wip": normalized_source,
                    "source_file_hash": source_file_hash,
                    "status": "success",
                    "exported_spectra": file_exported_spectra,
                    "imported_media_assets": file_imported_media_assets,
                    "detected_inventory": file_inventory,
                    "dataset_mode_counts": file_dataset_counts,
                    "media_inventory": file_media_inventory,
                    "datasets": dataset_summaries,
                    "class_counts": class_counts,
                    "project_version": project.version,
                    "duplicate_of_source": None,
                    "error_message": "",
                }
                summary_files.append(file_summary)
                log_lines.append(
                    f"[{self._timestamp()}] success {normalized_source} spectra={file_exported_spectra} "
                    f"datasets={file_dataset_counts} inventory={file_inventory}"
                )
                self._console_log(
                    job_id,
                    f"[{index}/{total_files}] imported {normalized_source} spectra={file_exported_spectra} "
                    f"photos={file_imported_media_assets} datasets={file_dataset_counts} traces={file_inventory}",
                )
                if progress_callback:
                    progress_callback(
                        {
                            "status": "running",
                            "current_file": normalized_source,
                            "processed_files": index,
                            "total_files": total_files,
                            "details": {
                                "phase": "analyzing_file",
                                "phase_label": "File imported",
                                "stage_detail": f"Completed file {index} of {total_files}",
                                "current_file_phase": "completed",
                                "current_file_index": index,
                                "overall_progress": round((index / total_files) * 100) if total_files > 0 else 100,
                                "last_file_inventory": file_inventory,
                                "last_file_dataset_counts": file_dataset_counts,
                                "last_file_media_inventory": file_media_inventory,
                                "last_file_summary": file_summary,
                            },
                        }
                    )
                seen_hash_sources[source_file_hash] = normalized_source
            except Exception as error:
                failed_files += 1
                file_summary = {
                    "source_wip": normalized_source,
                    "source_file_hash": source_file_hash,
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
                    "error_message": str(error),
                }
                summary_files.append(file_summary)
                log_lines.append(f"[{self._timestamp()}] failed {normalized_source}: {error}")
                self._console_log(job_id, f"[{index}/{total_files}] failed {normalized_source}: {error}")
                if progress_callback:
                    progress_callback(
                        {
                            "status": "running",
                            "current_file": normalized_source,
                            "processed_files": index,
                            "total_files": total_files,
                            "details": {
                                "phase": "analyzing_file",
                                "phase_label": "File failed",
                                "stage_detail": str(error),
                                "current_file_phase": "failed",
                                "current_file_index": index,
                                "overall_progress": round((index / total_files) * 100) if total_files > 0 else 100,
                                "last_file_inventory": {},
                                "last_file_dataset_counts": {},
                                "last_file_media_inventory": {},
                                "last_file_summary": file_summary,
                            },
                        }
                    )

        summary = {
            "input_path": str(input_path),
            "import_backend": "witio",
            "mock_mode": False,
            "import_options": import_options,
            "total_files": total_files,
            "processed_files": total_files,
            "exported_spectra": exported_spectra,
            "imported_media_assets": imported_media_assets,
            "failed_files": failed_files,
            "skipped_files": skipped_files,
            "duplicate_files": duplicate_files,
            "files": summary_files,
            "detected_inventory": self._aggregate_inventory(summary_files),
            "dataset_mode_counts": self._aggregate_mode_counts(summary_files, field_name="dataset_mode_counts"),
            "media_inventory": self._aggregate_mode_counts(summary_files, field_name="media_inventory"),
            "generated_at": self._timestamp(),
        }
        self._console_log(
            job_id,
            f"completed import: files={total_files} imported_spectra={exported_spectra} "
            f"imported_photos={imported_media_assets} "
            f"failed_files={failed_files} skipped_files={skipped_files} duplicate_files={duplicate_files}",
        )
        return self._finalize_summary(output_root, log_path, log_lines, summary)

    def _run_mock_export(
        self,
        input_path: Path,
        output_root: Path,
        log_path: Path,
        *,
        job_id: str,
        recursive: bool,
        force_reimport: bool,
        import_options: dict[str, object],
        progress_callback: ProgressCallback | None,
        spectrum_callback: SpectrumCallback | None,
        media_callback: MediaCallback | None,
        should_stop: ShouldStopCallback | None,
    ) -> dict[str, object]:
        files = self._collect_mock_files(input_path, recursive=recursive)
        summary_files: list[dict[str, object]] = []
        exported_spectra = 0
        imported_media_assets = 0
        failed_files = 0
        log_lines = [
            f"[{self._timestamp()}] import_backend=mock",
            f"[{self._timestamp()}] input={input_path}",
            f"[{self._timestamp()}] force_reimport={force_reimport}",
        ]
        self._console_log(job_id, f"mock import prepared {len(files)} files under {input_path}")

        for index, file_path in enumerate(files, start=1):
            self._raise_if_cancelled(should_stop)
            normalized_source = normalize_source_path(str(file_path))
            if progress_callback:
                progress_callback(
                    {
                        "status": "running",
                        "current_file": normalized_source,
                        "processed_files": index - 1,
                        "total_files": len(files),
                        "details": {"phase": "mock_import"},
                    }
                )

            file_inventory: dict[str, int] = {}
            file_dataset_counts: dict[str, int] = {}
            file_media_inventory: dict[str, int] = {}
            file_exported_spectra = 0
            file_imported_media_assets = 0
            try:
                point_count = 1 + ((index - 1) % 3)
                line_count = 3 if bool(import_options.get("include_line_scans", False)) else 0
                area_count = 4 if bool(import_options.get("include_area_maps", False)) else 0
                photo_count = 1 if bool(import_options.get("include_photo_images", False)) else 0
                base_path_metadata = parse_metadata_from_path(normalized_source)

                if bool(import_options.get("include_point_spectra", True)):
                    for spectrum_index in range(point_count):
                        self._raise_if_cancelled(should_stop)
                        x_axis, intensity = self._generate_mock_trace(seed=index * 100 + spectrum_index, spectrum_type="PL")
                        point_tree_path = f"/WITioRaw/point/{index:04d}/point-{spectrum_index:04d}"
                        metadata = {
                            "spectrum_id": build_spectrum_id(normalized_source, point_tree_path, spectrum_index),
                            "source_wip": normalized_source,
                            "source_tree_path": point_tree_path,
                            "acquisition_mode": POINT_SPECTRUM,
                            "trace_index": 0,
                            "trace_count": 1,
                            "scan_size_x": 1,
                            "scan_size_y": 1,
                            "grid_x": 0,
                            "grid_y": 0,
                            "x_axis_unit": infer_x_axis_unit(x_axis.tolist()),
                            "measurement_config": {
                                "scan_label": "mock_point",
                                "extraction_backend": "mock",
                            },
                        }
                        metadata.update(base_path_metadata)
                        metadata["spectrum_type"] = infer_spectrum_type_with_context(
                            x_axis.tolist(),
                            str(metadata["x_axis_unit"]),
                        )
                        if spectrum_callback:
                            spectrum_callback(normalized_source, metadata, x_axis, intensity)
                        file_exported_spectra += 1
                    file_inventory[POINT_SPECTRUM] = point_count
                    file_dataset_counts[POINT_SPECTRUM] = point_count

                if line_count:
                    x_axis, _ = self._generate_mock_trace(seed=index * 1000, spectrum_type="Raman")
                    for trace_index in range(line_count):
                        self._raise_if_cancelled(should_stop)
                        _, intensity = self._generate_mock_trace(seed=index * 1000 + trace_index, spectrum_type="Raman")
                        metadata = {
                            "spectrum_id": build_spectrum_id(normalized_source, f"/WITioRaw/line/{index:04d}/trace-{trace_index:04d}", trace_index),
                            "source_wip": normalized_source,
                            "source_tree_path": f"/WITioRaw/line/{index:04d}/trace-{trace_index:04d}",
                            "acquisition_mode": LINE_SCAN,
                            "trace_index": trace_index,
                            "trace_count": line_count,
                            "scan_size_x": line_count,
                            "scan_size_y": 1,
                            "grid_x": trace_index,
                            "grid_y": 0,
                            "x_axis_unit": infer_x_axis_unit(x_axis.tolist()),
                            "measurement_config": {
                                "scan_label": "mock_line",
                                "extraction_backend": "mock",
                            },
                            "secondary_axis_kind": "position",
                            "secondary_axis_unit": "um",
                            "secondary_axis_value": float(trace_index),
                        }
                        metadata.update(base_path_metadata)
                        metadata["spectrum_type"] = infer_spectrum_type_with_context(
                            x_axis.tolist(),
                            str(metadata["x_axis_unit"]),
                        )
                        if spectrum_callback:
                            spectrum_callback(normalized_source, metadata, x_axis, intensity)
                        file_exported_spectra += 1
                    file_inventory[LINE_SCAN] = line_count
                    file_dataset_counts[LINE_SCAN] = 1

                if area_count:
                    x_axis, _ = self._generate_mock_trace(seed=index * 2000, spectrum_type="PL")
                    trace_index = 0
                    for grid_y in range(2):
                        for grid_x in range(2):
                            self._raise_if_cancelled(should_stop)
                            _, intensity = self._generate_mock_trace(seed=index * 2000 + trace_index, spectrum_type="PL")
                            metadata = {
                                "spectrum_id": build_spectrum_id(
                                    normalized_source,
                                    f"/WITioRaw/image/{index:04d}/trace-{trace_index:04d}",
                                    trace_index,
                                ),
                                "source_wip": normalized_source,
                                "source_tree_path": f"/WITioRaw/image/{index:04d}/trace-{trace_index:04d}",
                                "acquisition_mode": AREA_MAP,
                                "trace_index": trace_index,
                                "trace_count": area_count,
                                "scan_size_x": 2,
                                "scan_size_y": 2,
                                "grid_x": grid_x,
                                "grid_y": grid_y,
                                "x_axis_unit": infer_x_axis_unit(x_axis.tolist()),
                                "measurement_config": {
                                    "scan_label": "mock_area",
                                    "extraction_backend": "mock",
                                },
                            }
                            metadata.update(base_path_metadata)
                            metadata["spectrum_type"] = infer_spectrum_type_with_context(
                                x_axis.tolist(),
                                str(metadata["x_axis_unit"]),
                            )
                            if spectrum_callback:
                                spectrum_callback(normalized_source, metadata, x_axis, intensity)
                            file_exported_spectra += 1
                            trace_index += 1
                    file_inventory[AREA_MAP] = area_count
                    file_dataset_counts[AREA_MAP] = 1

                if photo_count:
                    self._raise_if_cancelled(should_stop)
                    mock_photo = np.linspace(0, 4095, num=640 * 512, dtype=np.float32).reshape(640, 512)
                    encoded_photo = encode_photo_asset(mock_photo)
                    photo_tree_path = f"/WITioRaw/photo/{index:04d}/mock_photo"
                    photo_metadata = {
                        "media_id": build_media_id(normalized_source, photo_tree_path),
                        "source_wip": normalized_source,
                        "source_file_hash": f"mock-hash-{index}",
                        "source_tree_path": photo_tree_path,
                        "media_kind": "photo_image",
                        "entry_class": "TDImage",
                        "caption": "mock_photo",
                        "asset_format": encoded_photo.asset_format,
                        "width_px": encoded_photo.width_px,
                        "height_px": encoded_photo.height_px,
                        "original_width_px": encoded_photo.original_width_px,
                        "original_height_px": encoded_photo.original_height_px,
                        "channel_count": encoded_photo.channel_count,
                        "bit_depth": encoded_photo.bit_depth,
                        "measurement_config": {
                            "entry_class": "TDImage",
                            "caption": "mock_photo",
                            "extraction_backend": "mock",
                        },
                    }
                    photo_metadata.update(base_path_metadata)
                    if media_callback:
                        media_callback(normalized_source, photo_metadata, encoded_photo.image_bytes)
                    file_media_inventory["photo_image"] = photo_count
                    file_imported_media_assets += photo_count
                    imported_media_assets += photo_count

                summary_files.append(
                    {
                        "source_wip": normalized_source,
                        "status": "success",
                        "exported_spectra": file_exported_spectra,
                        "imported_media_assets": file_imported_media_assets,
                        "detected_inventory": file_inventory,
                        "dataset_mode_counts": file_dataset_counts,
                        "media_inventory": file_media_inventory,
                        "datasets": [],
                        "class_counts": {},
                        "project_version": None,
                        "duplicate_of_source": None,
                        "error_message": "",
                    }
                )
                exported_spectra += file_exported_spectra
                log_lines.append(
                    f"[{self._timestamp()}] success {normalized_source} spectra={file_exported_spectra} inventory={file_inventory}"
                )
                if progress_callback:
                    progress_callback(
                        {
                            "status": "running",
                            "current_file": normalized_source,
                            "processed_files": index,
                            "total_files": len(files),
                            "details": {
                                "phase": "mock_import",
                                "last_file_inventory": file_inventory,
                                "last_file_dataset_counts": file_dataset_counts,
                                "last_file_media_inventory": file_media_inventory,
                            },
                        }
                    )
            except Exception as error:  # pragma: no cover - defensive
                failed_files += 1
                summary_files.append(
                    {
                        "source_wip": normalized_source,
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
                        "error_message": str(error),
                    }
                )
                log_lines.append(f"[{self._timestamp()}] failed {normalized_source}: {error}")

        summary = {
            "input_path": str(input_path),
            "import_backend": "mock",
            "mock_mode": True,
            "import_options": import_options,
            "total_files": len(files),
            "processed_files": len(files),
            "exported_spectra": exported_spectra,
            "imported_media_assets": imported_media_assets,
            "failed_files": failed_files,
            "skipped_files": 0,
            "files": summary_files,
            "detected_inventory": self._aggregate_inventory(summary_files),
            "dataset_mode_counts": self._aggregate_mode_counts(summary_files, field_name="dataset_mode_counts"),
            "media_inventory": self._aggregate_mode_counts(summary_files, field_name="media_inventory"),
            "generated_at": self._timestamp(),
        }
        return self._finalize_summary(output_root, log_path, log_lines, summary)

    def _finalize_summary(
        self,
        output_root: Path,
        log_path: Path,
        log_lines: list[str],
        summary: dict[str, object],
    ) -> dict[str, object]:
        summary_path = output_root / "import_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        return {
            "summary_path": str(summary_path),
            "log_path": str(log_path),
            "summary": summary,
        }

    def _aggregate_inventory(self, summary_files: list[dict[str, object]]) -> dict[str, int]:
        return self._aggregate_mode_counts(summary_files, field_name="detected_inventory")

    def _aggregate_mode_counts(self, summary_files: list[dict[str, object]], *, field_name: str) -> dict[str, int]:
        aggregate: dict[str, int] = {}
        for item in summary_files:
            inventory = item.get(field_name)
            if not isinstance(inventory, dict):
                continue
            for key, value in inventory.items():
                aggregate[str(key)] = aggregate.get(str(key), 0) + int(value or 0)
        return aggregate

    def _collect_input_files(self, input_path: Path, *, recursive: bool) -> list[Path]:
        if input_path.is_file():
            return [input_path]
        pattern = "**/*.wip" if recursive else "*.wip"
        return sorted(input_path.glob(pattern))

    def _collect_mock_files(self, input_path: Path, *, recursive: bool) -> list[Path]:
        if input_path.exists() and input_path.is_file():
            return [input_path]
        if input_path.exists() and input_path.is_dir():
            pattern = "**/*.wip" if recursive else "*.wip"
            collected = sorted(input_path.glob(pattern))
            if collected:
                return collected
            return [input_path / "mock_sample_01.wip", input_path / "mock_sample_02.wip"]
        return [Path("mock_demo_01.wip"), Path("mock_demo_02.wip"), Path("mock_demo_03.wip")]

    def _generate_mock_trace(self, *, seed: int, spectrum_type: str) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        if spectrum_type == "PL":
            x_axis = np.linspace(650.0, 900.0, 1024)
            centers = [730.0 + rng.normal(0, 8), 790.0 + rng.normal(0, 12)]
            widths = [18.0 + rng.normal(0, 2), 24.0 + rng.normal(0, 3)]
            amplitudes = [1.0 + rng.normal(0, 0.05), 0.45 + rng.normal(0, 0.04)]
        else:
            x_axis = np.linspace(100.0, 2200.0, 1024)
            centers = [380.0 + rng.normal(0, 5), 520.0 + rng.normal(0, 7)]
            widths = [8.0 + rng.normal(0, 1), 12.0 + rng.normal(0, 1)]
            amplitudes = [1.0 + rng.normal(0, 0.05), 0.5 + rng.normal(0, 0.05)]

        baseline = 0.08 + 0.0002 * (x_axis - x_axis.min())
        signal = baseline.copy()
        for amplitude, center, width in zip(amplitudes, centers, widths, strict=True):
            signal += amplitude * np.exp(-0.5 * ((x_axis - center) / width) ** 2)
        signal += rng.normal(scale=0.015, size=x_axis.shape)
        signal = np.clip(signal, a_min=0.0, a_max=None)
        return x_axis, signal

    def _should_use_mock(self, input_path: str, resolved_input: Path) -> bool:
        if input_path.lower().startswith("mock://"):
            return True
        if resolved_input.exists():
            return False
        return self.settings.importer.mock_mode

    def _timestamp(self) -> str:
        return datetime.now(UTC).isoformat()

    def _raise_if_cancelled(self, should_stop: ShouldStopCallback | None) -> None:
        if should_stop and should_stop():
            raise ImportCancelled("Import was stopped by the user")

    def _console_log(self, job_id: str, message: str) -> None:
        print(f"[{self._timestamp()}] [import {job_id}] {message}", flush=True)
