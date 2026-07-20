from __future__ import annotations

import shutil
from io import StringIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from . import __version__
from .config import load_settings
from .models.schemas import AnalysisRequest, BatchAnalysisRequest, ImportRequest, MaterialAnalysisStartRequest, MetadataUpdateRequest
from .services.analysis_jobs import MaterialAnalysisJobManager
from .services.database import DatabaseService
from .services.import_bridge import ImportBridge
from .services.importer import ImportManager
from .services.spectrum_processor import analyze_spectrum
from .services.upload_staging import stage_upload_files


settings = load_settings()
database = DatabaseService(settings)
bridge = ImportBridge(settings, database)
import_manager = ImportManager(settings, database, bridge)
analysis_jobs = MaterialAnalysisJobManager(database)
database.start_background_maintenance()

app = FastAPI(
    title="PL_database API",
    version=__version__,
    description="Local spectral database for WITec PL and Raman spectra.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "mock_mode": settings.importer.mock_mode,
        "import_backend": settings.importer.backend,
        "config_path": str(settings.config_path),
    }


@app.get("/api/config")
def config_view() -> dict[str, object]:
    return {
        "importer": {
            "backend": settings.importer.backend,
            "mock_mode": settings.importer.mock_mode,
        },
        "api": {"host": settings.api.host, "port": settings.api.port},
        "database": {
            "sqlite_path": str(settings.sqlite_path),
            "hdf5_path": str(settings.hdf5_path),
        },
    }


@app.get("/api/dashboard")
def dashboard(include_mock: bool = Query(default=False)) -> dict[str, object]:
    return database.dashboard_summary(include_mock=include_mock)


@app.post("/api/import/start")
def start_import(request: ImportRequest) -> dict[str, object]:
    return import_manager.start_job(
        input_path=request.input_path,
        recursive=request.recursive,
        force_reimport=request.force_reimport,
        import_options=request.options.model_dump(),
    )


@app.post("/api/import/upload-start")
async def upload_and_start_import(
    files: list[UploadFile] = File(...),
    relative_paths: list[str] | None = Form(default=None),
    root_name: str | None = Form(default=None),
    source_kind: str | None = Form(default=None),
    display_label: str | None = Form(default=None),
    force_reimport: bool = Form(default=False),
    include_point_spectra: bool = Form(default=True),
    include_line_scans: bool = Form(default=False),
    include_area_maps: bool = Form(default=False),
    include_series_scans: bool = Form(default=False),
    include_photo_images: bool = Form(default=False),
) -> dict[str, object]:
    if not files:
        raise HTTPException(status_code=400, detail="No upload files were provided")

    normalized_source_kind = str(source_kind or "").strip().lower()
    source_kind_value = "file_upload" if normalized_source_kind == "file_upload" else "folder_upload"
    relative_input_path = relative_paths[0] if relative_paths else (files[0].filename or None)
    if source_kind_value == "file_upload":
        resolved_display_label = display_label or f"Selected file: {root_name or files[0].filename or 'uploaded.wip'}"
    else:
        resolved_display_label = display_label or f"Selected folder item: {relative_input_path or root_name or 'uploaded.wip'}"
    import_options = {
        "include_point_spectra": include_point_spectra,
        "include_line_scans": include_line_scans,
        "include_area_maps": include_area_maps,
        "include_series_scans": include_series_scans,
        "include_photo_images": include_photo_images,
    }

    try:
        staged_root, uploaded_count = stage_upload_files(
            settings,
            files=files,
            relative_paths=relative_paths,
            root_name=root_name,
        )
    except Exception as error:
        return import_manager.record_failed_upload(
            input_path=str(relative_input_path or root_name or files[0].filename or "uploaded.wip"),
            message=f"Upload staging failed: {error}",
            import_options=import_options,
            details={
                "display_input_path": resolved_display_label,
                "relative_input_path": relative_input_path,
                "source_kind": source_kind_value,
                "uploaded_files": 0,
                "upload_root_name": root_name,
                "cleanup_staged_upload": False,
                "upload_stage": "staging",
            },
        )
    finally:
        for upload in files:
            await upload.close()

    if source_kind_value == "folder_upload" and not display_label:
        display_parts = []
        if root_name:
            display_parts.append(root_name)
        display_parts.append(f"{uploaded_count} .wip files")
        resolved_display_label = "Selected folder: " + " | ".join(display_parts)

    return import_manager.start_job(
        input_path=str(staged_root),
        recursive=True,
        force_reimport=force_reimport,
        import_options=import_options,
        details={
            "display_input_path": resolved_display_label,
            "relative_input_path": relative_input_path,
            "source_kind": source_kind_value,
            "uploaded_files": uploaded_count,
            "upload_root_name": root_name,
            "cleanup_staged_upload": True,
        },
    )


@app.post("/api/import/probe-upload")
async def probe_uploaded_wip(
    file: UploadFile = File(...),
) -> dict[str, object]:
    staged_root: Path | None = None
    try:
        staged_root, _ = stage_upload_files(
            settings,
            files=[file],
            relative_paths=[str(file.filename or "selected_file.wip")],
            root_name=Path(str(file.filename or "selected_file.wip")).stem,
        )
        staged_files = sorted(staged_root.rglob("*.wip"))
        if not staged_files:
            raise HTTPException(status_code=400, detail="No .wip file was uploaded")
        return bridge.probe_input_file(str(staged_files[0]))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        await file.close()
        if staged_root is not None:
            shutil.rmtree(staged_root, ignore_errors=True)


@app.get("/api/import/jobs")
def import_jobs() -> list[dict[str, object]]:
    return import_manager.list_jobs()


@app.post("/api/import/jobs/{job_id}/stop")
def stop_import_job(job_id: str) -> dict[str, object]:
    if not import_manager.request_stop(job_id):
        raise HTTPException(status_code=404, detail="Import job is not running or no longer exists")
    return {"job_id": job_id, "status": "stop_requested"}


@app.get("/api/import/upload-history")
def imported_upload_history(root_name: str | None = Query(default=None)) -> dict[str, object]:
    return {
        "items": database.list_imported_upload_history(root_name=root_name),
    }


@app.get("/api/import/jobs/{job_id}")
def import_job_detail(job_id: str) -> dict[str, object]:
    job = import_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Import job not found")
    return job


@app.get("/api/database/options")
def database_options(include_mock: bool = Query(default=False)) -> dict[str, list[str]]:
    return database.list_filter_options(include_mock=include_mock)


@app.get("/api/database/spectra")
def list_spectra(
    search: str | None = None,
    spectrum_type: str | None = None,
    source: str | None = None,
    belonging: str | None = None,
    acquisition_mode: str | None = None,
    substrate: str | None = None,
    x_axis_unit: str | None = None,
    sample_id: str | None = None,
    analysis_material: str | None = None,
    analysis_family: str | None = None,
    analysis_status: str | None = None,
    n_points_min: int | None = None,
    n_points_max: int | None = None,
    member_count_min: int | None = None,
    member_count_max: int | None = None,
    trace_count_min: int | None = None,
    trace_count_max: int | None = None,
    scan_size_x_min: int | None = None,
    scan_size_x_max: int | None = None,
    scan_size_y_min: int | None = None,
    scan_size_y_max: int | None = None,
    grid_x_min: int | None = None,
    grid_x_max: int | None = None,
    grid_y_min: int | None = None,
    grid_y_max: int | None = None,
    sort_by: str | None = None,
    sort_dir: str | None = None,
    include_mock: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    filters = {
        "spectrum_type": spectrum_type,
        "source": source,
        "belonging": belonging,
        "acquisition_mode": acquisition_mode,
        "substrate": substrate,
        "x_axis_unit": x_axis_unit,
        "sample_id": sample_id,
        "analysis_material": analysis_material,
        "analysis_family": analysis_family,
        "analysis_status": analysis_status,
        "n_points_min": n_points_min,
        "n_points_max": n_points_max,
        "member_count_min": member_count_min,
        "member_count_max": member_count_max,
        "trace_count_min": trace_count_min,
        "trace_count_max": trace_count_max,
        "scan_size_x_min": scan_size_x_min,
        "scan_size_x_max": scan_size_x_max,
        "scan_size_y_min": scan_size_y_min,
        "scan_size_y_max": scan_size_y_max,
        "grid_x_min": grid_x_min,
        "grid_x_max": grid_x_max,
        "grid_y_min": grid_y_min,
        "grid_y_max": grid_y_max,
    }
    return database.list_spectra(
        search=search,
        filters=filters,
        include_mock=include_mock,
        limit=limit,
        offset=offset,
        sort_by=sort_by,
        sort_dir=sort_dir,
    )


@app.get("/api/database/spectra/{spectrum_id}")
def get_spectrum(spectrum_id: str) -> dict[str, object]:
    detail = database.get_spectrum(spectrum_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Spectrum not found")
    return detail


@app.post("/api/database/metadata")
def update_metadata(request: MetadataUpdateRequest) -> dict[str, object]:
    updated = database.update_metadata(
        spectrum_ids=request.spectrum_ids,
        apply_mode=request.apply_mode,
        scope_value=request.scope_value,
        metadata=request.metadata,
    )
    return {"updated_rows": updated}


@app.get("/api/database/export")
def export_selection(
    spectrum_ids: str = Query(min_length=1, description="Comma-separated spectrum IDs"),
) -> Response:
    ids = [item.strip() for item in spectrum_ids.split(",") if item.strip()]
    if not ids:
        raise HTTPException(status_code=422, detail="No spectrum IDs provided")
    export_path = settings.export_root / "downloads" / "selected_spectra.csv"
    database.export_spectra_to_csv(ids, export_path)
    return FileResponse(export_path)


@app.post("/api/analysis/run")
def run_analysis(request: AnalysisRequest) -> dict[str, object]:
    results: list[dict[str, object]] = []
    for spectrum_id in request.spectrum_ids:
        detail = database.get_spectrum(spectrum_id)
        if detail is None:
            continue
        analysis_target_id = str(detail.get("representative_spectrum_id") or spectrum_id)
        result = analyze_spectrum(detail["x_axis"], detail["intensity"], request.options.model_dump())
        payload = {
            "spectrum_id": spectrum_id,
            "file_path": detail["file_path"],
            "analysis": result,
        }
        if request.save_results:
            payload["result_id"] = database.insert_analysis_result(
                analysis_target_id,
                result,
                method="baseline+smooth+peakfit",
                parameters=request.options.model_dump(),
            )
        results.append(payload)
    return {"results": results}


@app.post("/api/analysis/batch")
def run_batch_analysis(request: BatchAnalysisRequest) -> dict[str, object]:
    outcome = run_analysis(
        AnalysisRequest(
            spectrum_ids=request.spectrum_ids,
            options=request.options,
            save_results=request.save_results,
        )
    )
    metrics = [
        {
            "spectrum_id": item["spectrum_id"],
            **item["analysis"]["metrics"],
        }
        for item in outcome["results"]
    ]
    return {"summary": metrics, "results": outcome["results"]}


@app.post("/api/analysis/material/start")
def start_material_analysis(request: MaterialAnalysisStartRequest) -> dict[str, object]:
    spectrum_ids = request.spectrum_ids
    if not spectrum_ids:
        spectrum_ids = database.list_spectrum_identifiers(
            search=request.search,
            filters=request.filters,
            include_mock=request.include_mock,
        )
    if not spectrum_ids:
        raise HTTPException(status_code=422, detail="No spectrum entries matched the analysis request")
    return analysis_jobs.start_job(
        spectrum_ids=spectrum_ids,
        options=request.options.model_dump(),
        save_results=request.save_results,
        update_entries=request.update_entries,
    )


@app.get("/api/analysis/material/jobs/{job_id}")
def material_analysis_job_detail(job_id: str) -> dict[str, object]:
    job = analysis_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return job


@app.post("/api/analysis/material/jobs/{job_id}/stop")
def stop_material_analysis_job(job_id: str) -> dict[str, object]:
    if not analysis_jobs.stop_job(job_id):
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return {"job_id": job_id, "status": "stop_requested"}


@app.get("/api/analysis/results")
def list_analysis_results(spectrum_id: str | None = None, limit: int = Query(default=200, ge=1, le=1000)) -> list[dict[str, object]]:
    return database.list_analysis_results(spectrum_id=spectrum_id, limit=limit)


frontend_dist = settings.frontend_dist
if frontend_dist.exists():

    def _frontend_response(path: Path) -> FileResponse:
        response = FileResponse(path)
        if path.suffix in {".html", ".js", ".css"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        requested = (frontend_dist / path).resolve()
        if requested.is_relative_to(frontend_dist) and requested.is_file():
            return _frontend_response(requested)
        return _frontend_response(frontend_dist / "index.html")
