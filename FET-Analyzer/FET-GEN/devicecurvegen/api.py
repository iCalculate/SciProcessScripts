from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from . import __version__
from .database import (
    analyze_curves,
    calendar_curves,
    database_status,
    export_curve_rows,
    get_curve_detail,
    get_curve_previews,
    import_b1500_to_mysql,
    list_curves,
    search_options,
)
from .features import analyze_transfer_curve
from .harmonize import inspect_measurement
from .physics import generate_curves
from .residual import ResidualEngine
from .schemas import (
    ExtractedFeatures,
    ExtractionRequest,
    GenerationCondition,
    GenerationResponse,
    InspectionResponse,
    ModelInfo,
    NeuralTrainingRequest,
    NeuralTrainingStatus,
    TrainingResult,
)
from .training import train_residual_checkpoint
from .training_service import NeuralTrainingManager

app = FastAPI(
    title="DeviceCurveGen API",
    version="0.1.0",
    description="Physics-informed FET transfer-curve generation",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_SUFFIXES = {".csv", ".txt", ".tsv", ".dat"}
examples_root = (Path(__file__).resolve().parents[1] / "examples").resolve()
model_load_error: str | None = None
try:
    residual_engine = ResidualEngine()
except ValueError as error:
    residual_engine = ResidualEngine(discover_default=False)
    model_load_error = str(error)


def _activate_neural_checkpoint(path: Path) -> None:
    global model_load_error, residual_engine
    residual_engine = ResidualEngine(path)
    model_load_error = None


neural_training_manager = NeuralTrainingManager(_activate_neural_checkpoint)


class DatabaseSelectionRequest(BaseModel):
    curve_ids: list[str] = Field(default_factory=list)
    filters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class CurvePreviewRequest(BaseModel):
    curve_ids: list[str] = Field(default_factory=list, max_length=64)


class DatabaseImportRequest(BaseModel):
    source_path: str = Field(min_length=1)
    suffixes: list[str] = Field(default_factory=list)
    max_xml_mb: float = Field(default=128.0, ge=1.0, le=4096.0)
    hash_files: bool = False
    replace: bool = False


@app.get("/api/runtime-config")
def runtime_config() -> dict[str, str]:
    mode = os.getenv("DEVICEGEN_APP_MODE", "full").strip().lower() or "full"
    return {"app_mode": mode}


@app.get("/health")
def health() -> dict[str, str | None]:
    return {
        "status": "ok",
        "version": __version__,
        "residual_mode": residual_engine.mode,
        "model_load_error": model_load_error,
    }


@app.get("/api/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    info = residual_engine.info()
    return info.model_copy(update={"load_error": model_load_error})


@app.get("/api/neural-training/status", response_model=NeuralTrainingStatus)
def neural_training_status() -> NeuralTrainingStatus:
    return neural_training_manager.snapshot()


@app.post("/api/neural-training/start", response_model=NeuralTrainingStatus)
def start_neural_training(request: NeuralTrainingRequest) -> NeuralTrainingStatus:
    try:
        return neural_training_manager.start(request)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


def _database_error(error: Exception) -> HTTPException:
    if isinstance(error, ValueError):
        return HTTPException(status_code=503, detail=str(error))
    if isinstance(error, SQLAlchemyError):
        return HTTPException(status_code=503, detail=f"Database unavailable: {error}")
    return HTTPException(status_code=500, detail=str(error))


@app.get("/api/database/status")
def database_status_endpoint() -> dict:
    try:
        return database_status()
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


@app.get("/api/database/options")
def database_options_endpoint() -> dict:
    try:
        return search_options()
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


@app.get("/api/database/curves")
def database_curves_endpoint(
    limit: int = 100,
    offset: int = 0,
    order_by: str = "modified_at_desc",
    polarity: str | None = None,
    direction: str | None = None,
    source_kind: str | None = None,
    source_search: str | None = None,
    has_gate_current: str | None = None,
    hysteresis_available: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    ion_min: float | None = None,
    ion_max: float | None = None,
    ioff_min: float | None = None,
    ioff_max: float | None = None,
    ion_ioff_ratio_min: float | None = None,
    ion_ioff_ratio_max: float | None = None,
    vth_min: float | None = None,
    vth_max: float | None = None,
    ss_mv_dec_min: float | None = None,
    ss_mv_dec_max: float | None = None,
) -> dict:
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")
    try:
        return list_curves(
            limit=limit,
            offset=offset,
            order_by=order_by,
            polarity=polarity,
            direction=direction,
            source_kind=source_kind,
            source_search=source_search,
            has_gate_current=has_gate_current,
            hysteresis_available=hysteresis_available,
            date_from=date_from,
            date_to=date_to,
            ion_min=ion_min,
            ion_max=ion_max,
            ioff_min=ioff_min,
            ioff_max=ioff_max,
            ion_ioff_ratio_min=ion_ioff_ratio_min,
            ion_ioff_ratio_max=ion_ioff_ratio_max,
            vth_min=vth_min,
            vth_max=vth_max,
            ss_mv_dec_min=ss_mv_dec_min,
            ss_mv_dec_max=ss_mv_dec_max,
        )
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


@app.get("/api/database/calendar")
def database_calendar_endpoint(
    limit: int = 10_000,
    polarity: str | None = None,
    direction: str | None = None,
    source_kind: str | None = None,
    source_search: str | None = None,
    has_gate_current: str | None = None,
    hysteresis_available: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> dict:
    if limit < 1 or limit > 25_000:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 25000")
    try:
        return calendar_curves(
            limit=limit,
            polarity=polarity,
            direction=direction,
            source_kind=source_kind,
            source_search=source_search,
            has_gate_current=has_gate_current,
            hysteresis_available=hysteresis_available,
            date_from=date_from,
            date_to=date_to,
        )
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


@app.post("/api/database/analyze")
def database_analyze_endpoint(request: DatabaseSelectionRequest) -> dict:
    try:
        return analyze_curves(
            curve_ids=request.curve_ids or None,
            filters=request.filters,
        )
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


@app.post("/api/database/previews")
def database_previews_endpoint(request: CurvePreviewRequest) -> list[dict]:
    try:
        return get_curve_previews(None, request.curve_ids)
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


@app.post("/api/database/import")
async def database_import_endpoint(request: DatabaseImportRequest) -> dict:
    try:
        source = Path(request.source_path).expanduser().resolve()
        return await run_in_threadpool(
            import_b1500_to_mysql,
            source,
            None,
            replace=request.replace,
            suffixes=set(request.suffixes) if request.suffixes else None,
            max_xml_mb=request.max_xml_mb,
            hash_files=request.hash_files,
        )
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


@app.post("/api/database/import-upload")
async def database_import_upload_endpoint(
    files: Annotated[list[UploadFile], File()],
    relative_paths_json: Annotated[str, Form()],
    suffixes_json: Annotated[str | None, Form()] = None,
    max_xml_mb: Annotated[float, Form(ge=1.0, le=4096.0)] = 128.0,
    hash_files: Annotated[bool, Form()] = False,
    replace: Annotated[bool, Form()] = False,
) -> dict:
    if not files:
        raise HTTPException(status_code=422, detail="Select at least one file from a folder.")
    try:
        relative_paths = json.loads(relative_paths_json)
        suffixes = json.loads(suffixes_json) if suffixes_json else []
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail="Invalid folder import metadata.") from error
    if not isinstance(relative_paths, list) or not all(isinstance(item, str) for item in relative_paths):
        raise HTTPException(status_code=422, detail="Folder import paths must be a string list.")
    if len(relative_paths) != len(files):
        raise HTTPException(status_code=422, detail="Folder import file metadata is inconsistent.")
    if suffixes and (not isinstance(suffixes, list) or not all(isinstance(item, str) for item in suffixes)):
        raise HTTPException(status_code=422, detail="Suffix list must be a string list.")

    with tempfile.TemporaryDirectory(prefix="devicecurvegen-import-") as temp:
        temp_root = Path(temp)
        for file, relative_path in zip(files, relative_paths, strict=True):
            sanitized = Path(relative_path.replace("\\", "/"))
            if sanitized.is_absolute() or ".." in sanitized.parts or sanitized.name == "":
                raise HTTPException(status_code=422, detail=f"Invalid relative path: {relative_path}")
            content = await _read_upload(file)
            destination = (temp_root / sanitized).resolve()
            if not destination.is_relative_to(temp_root):
                raise HTTPException(status_code=422, detail=f"Unsafe relative path: {relative_path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        try:
            return await run_in_threadpool(
                import_b1500_to_mysql,
                temp_root,
                None,
                replace=replace,
                suffixes=set(suffixes) if suffixes else None,
                max_xml_mb=max_xml_mb,
                hash_files=hash_files,
            )
        except (ValueError, SQLAlchemyError) as error:
            raise _database_error(error) from error


def _write_csv_to_zip(archive: ZipFile, filename: str, rows: list[dict]) -> None:
    output = StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    archive.writestr(filename, output.getvalue())


@app.post("/api/database/export")
def database_export_endpoint(request: DatabaseSelectionRequest) -> Response:
    try:
        payload = export_curve_rows(
            curve_ids=request.curve_ids or None,
            filters=request.filters,
        )
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        _write_csv_to_zip(archive, "curves.csv", payload["curves"])
        _write_csv_to_zip(archive, "raw_id_points.csv", payload["raw_points"])
        _write_csv_to_zip(archive, "raw_ig_points.csv", payload["gate_points"])
        _write_csv_to_zip(archive, "aligned_ig_points.csv", payload["aligned_gate_points"])
        archive.writestr(
            "analysis.json",
            json.dumps(payload["analysis"], indent=2, ensure_ascii=False),
        )
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "format": "devicecurvegen.database-selection.v1",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "curves": len(payload["curves"]),
                    "raw_id_points": len(payload["raw_points"]),
                    "raw_ig_points": len(payload["gate_points"]),
                    "aligned_ig_points": len(payload["aligned_gate_points"]),
                    "selection": {
                        "mode": "ids" if request.curve_ids else "filters",
                        "curve_ids": request.curve_ids,
                        "filters": request.filters,
                    },
                    "files": [
                        "curves.csv",
                        "raw_id_points.csv",
                        "raw_ig_points.csv",
                        "aligned_ig_points.csv",
                        "analysis.json",
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
    filename = "devicecurvegen-database-selection.zip"
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/database/curves/{curve_id}")
def database_curve_detail_endpoint(curve_id: str) -> dict:
    try:
        detail = get_curve_detail(None, curve_id)
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error
    if detail is None:
        raise HTTPException(status_code=404, detail="Curve not found")
    return detail


@app.get("/api/examples/{path:path}", include_in_schema=False)
def example_file(path: str) -> FileResponse:
    requested = (examples_root / path).resolve()
    if (
        not requested.is_relative_to(examples_root)
        or not requested.is_file()
        or requested.suffix.lower() not in SUPPORTED_SUFFIXES
    ):
        raise HTTPException(status_code=404, detail="Example file not found")
    return FileResponse(requested)


@app.post("/api/generate", response_model=GenerationResponse)
def generate(condition: GenerationCondition) -> GenerationResponse:
    return generate_curves(condition, residual_engine)


@app.get("/api/export")
def export_candidate(condition: str, seed: int, candidate_id: int = 1) -> Response:
    try:
        parsed = GenerationCondition.model_validate_json(condition).model_copy(
            update={"seed": seed, "variants": 1}
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid generation condition") from error
    candidate = generate_curves(parsed, residual_engine).candidates[0]
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        [
            "Vg",
            "Id_forward",
            "Id_reverse",
            "Ig_forward",
            "Ig_reverse",
            "Id_physics_forward",
            "Id_physics_reverse",
        ]
    )
    writer.writerows(
        zip(
            candidate.voltage,
            candidate.forward_current,
            candidate.reverse_current,
            candidate.gate_forward_current,
            candidate.gate_reverse_current,
            candidate.physics_forward_current,
            candidate.physics_reverse_current,
            strict=True,
        )
    )
    filename = f"devicecurvegen-candidate-{candidate_id}-seed-{seed}.csv"
    return Response(
        output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/extract", response_model=ExtractedFeatures)
def extract(request: ExtractionRequest) -> ExtractedFeatures:
    return analyze_transfer_curve(
        request.voltage,
        request.current,
        polarity=request.polarity,
    )


async def _read_upload(file: UploadFile) -> bytes:
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )
    return content


@app.post("/api/inspect", response_model=InspectionResponse)
async def inspect(
    file: Annotated[UploadFile, File()],
    voltage_column: Annotated[str | None, Form()] = None,
    current_column: Annotated[str | None, Form()] = None,
) -> InspectionResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=415, detail="MVP supports CSV/TXT/TSV/DAT files")
    try:
        content = await _read_upload(file)
        return await run_in_threadpool(
            inspect_measurement,
            file.filename or "measurement",
            content,
            voltage_column=voltage_column,
            current_column=current_column,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/api/train", response_model=TrainingResult)
async def train(
    files: Annotated[list[UploadFile], File()],
    components: Annotated[int, Form(ge=1, le=64)] = 8,
) -> TrainingResult:
    if not 3 <= len(files) <= 100:
        raise HTTPException(status_code=422, detail="Upload between 3 and 100 files")
    output = (
        Path(
            os.getenv(
                "DEVICEGEN_MODEL_OUTPUT",
                Path(__file__).resolve().parents[1] / "models" / "residual-pca.npz",
            )
        )
        .expanduser()
        .resolve()
    )
    with tempfile.TemporaryDirectory(prefix="devicecurvegen-train-") as temp:
        temp_root = Path(temp)
        paths: list[Path] = []
        for index, file in enumerate(files):
            filename = Path(file.filename or f"curve-{index}.csv").name
            if Path(filename).suffix.lower() not in SUPPORTED_SUFFIXES:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported training file: {filename}",
                )
            content = await _read_upload(file)
            path = temp_root / f"{index:03d}-{filename}"
            path.write_bytes(content)
            paths.append(path)
        try:
            result = await run_in_threadpool(
                train_residual_checkpoint,
                paths,
                output,
                components=components,
            )
            residual_engine.reload(output)
            global model_load_error
            model_load_error = None
            return result
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error


frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    frontend_root = frontend_dist.resolve()
    assets = frontend_dist / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        requested = (frontend_root / path).resolve()
        if requested.is_relative_to(frontend_root) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(frontend_root / "index.html")
