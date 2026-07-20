from __future__ import annotations

import csv
import json
import os
import time
import tempfile
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Annotated, Any, Literal
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from . import __version__
from .analysis_service import DatabaseAnalysisManager
from .database import (
    analyze_curves,
    calendar_curves,
    database_status,
    export_curve_rows,
    get_curve_detail,
    get_curve_previews,
    import_b1500_to_mysql,
    list_curves,
    match_matrix_sites,
    search_options,
)
from .features import analyze_transfer_curve
from .harmonize import inspect_measurement
from .physics import generate_curves
from .residual import ResidualEngine
from .schemas import (
    DatabaseAnalysisStatus,
    ExperimentLeaderboardEntry,
    ExperimentLeaderboardResponse,
    ExtractedFeatures,
    ExtractionRequest,
    GenerationCondition,
    GenerationResponse,
    InspectionResponse,
    ModelComparisonItem,
    ModelComparisonRequest,
    ModelComparisonResponse,
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
experiments_root = (Path(__file__).resolve().parents[1] / "experiments").resolve()
model_load_error: str | None = None
_bootstrap_started_at = time.perf_counter()
print("[startup] API bootstrap: loading residual engine", flush=True)
try:
    residual_engine = ResidualEngine()
except ValueError as error:
    residual_engine = ResidualEngine(discover_default=False)
    model_load_error = str(error)
print(
    f"[startup] API bootstrap: residual engine mode={residual_engine.mode} "
    f"load_error={'none' if model_load_error is None else model_load_error}",
    flush=True,
)


def _activate_neural_checkpoint(path: Path) -> None:
    global model_load_error, residual_engine
    residual_engine = ResidualEngine(path)
    model_load_error = None


neural_training_manager = NeuralTrainingManager(_activate_neural_checkpoint)
database_analysis_manager = DatabaseAnalysisManager()


@app.on_event("startup")
def log_startup_configuration() -> None:
    elapsed = time.perf_counter() - _bootstrap_started_at
    print(
        "[startup] Runtime config: "
        f"app_mode={os.getenv('DEVICEGEN_APP_MODE', 'full')} "
        f"database_url={os.getenv('DEVICEGEN_DATABASE_URL', 'sqlite')}",
        flush=True,
    )
    print(
        "[startup] FastAPI ready: "
        f"version={__version__} residual_mode={residual_engine.mode} "
        f"bootstrap_elapsed={elapsed:.2f}s",
        flush=True,
    )


class DatabaseSelectionRequest(BaseModel):
    curve_ids: list[str] = Field(default_factory=list)
    filters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    export_options: "DatabaseExportOptions" = Field(default_factory=lambda: DatabaseExportOptions())


class DatabaseExportOptions(BaseModel):
    xyxy_curves: bool = True
    curve_metadata: bool = True
    raw_id_points: bool = True
    include_ig: bool = True
    raw_ig_points: bool = True
    aligned_ig_points: bool = True
    analysis_json: bool = True


class CurvePreviewRequest(BaseModel):
    curve_ids: list[str] = Field(default_factory=list, max_length=64)


class DatabaseImportRequest(BaseModel):
    source_path: str = Field(min_length=1)
    suffixes: list[str] = Field(default_factory=list)
    max_xml_mb: float = Field(default=128.0, ge=1.0, le=4096.0)
    hash_files: bool = False
    replace: bool = False


MatrixParameterKey = Literal[
    "target_ion",
    "target_ioff",
    "ion_ioff_ratio",
    "target_vth",
    "target_ss_mv_dec",
    "hysteresis_v",
    "mobility_cm2_vs",
    "contact_resistance_ohm",
    "gate_leakage_a",
]


class MatrixParameterRequest(BaseModel):
    key: MatrixParameterKey
    values: list[list[float]]


class MatrixSynthesisRequest(BaseModel):
    rows: int = Field(default=4, ge=1, le=32)
    cols: int = Field(default=4, ge=1, le=32)
    mode: Literal["database", "generate"] = "database"
    duplicate_mode: Literal["allow", "avoid", "generate_on_duplicate"] = "allow"
    parameters: list[MatrixParameterRequest] = Field(default_factory=list, min_length=1, max_length=8)
    filters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    generation_condition: GenerationCondition = Field(default_factory=GenerationCondition)


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


@app.get("/api/model/leaderboard", response_model=ExperimentLeaderboardResponse)
def model_leaderboard(limit: int = 18) -> ExperimentLeaderboardResponse:
    return _load_experiment_leaderboard(limit)


@app.get("/api/model/leaderboard-artifact/latest")
def model_leaderboard_artifact() -> FileResponse:
    artifact_path = _latest_comparison_artifact_path()
    if artifact_path is None:
        raise HTTPException(status_code=404, detail="No comparison artifact available")
    return FileResponse(artifact_path, media_type="image/svg+xml")


@app.post("/api/model/compare", response_model=ModelComparisonResponse)
def model_compare(request: ModelComparisonRequest) -> ModelComparisonResponse:
    base = request.condition.model_copy(update={"variants": 1})
    active_info = model_info()
    leaderboard = _load_experiment_leaderboard(limit=10_000)
    active_entry = _entry_by_checkpoint(leaderboard.entries, active_info.checkpoint_path)
    prior_engine = ResidualEngine(discover_default=False)
    prior_info = prior_engine.info()
    profiles = [
        (
            "physics_only",
            "Physics only",
            "Pure analytical baseline with all learned residual channels disabled.",
            residual_engine,
            base.model_copy(
                update={
                    "ai_residual_strength": 0.0,
                    "gate_ai_residual_strength": 0.0,
                }
            ),
            active_info,
            None,
        ),
        (
            "active_model",
            "Active model",
            "Current best checkpoint blended with the single physics-to-AI balance slider.",
            residual_engine,
            base,
            active_info,
            active_entry,
        ),
    ]
    best_jump = leaderboard.best_jump_entry or (leaderboard.entries[0] if leaderboard.entries else None)
    best_canonical = leaderboard.best_canonical_entry or _best_canonical_entry(leaderboard.entries)
    best_weighted = leaderboard.best_weighted_entry or _best_weighted_entry(leaderboard.entries)
    active_checkpoint = active_info.checkpoint_path
    used_checkpoints = {
        checkpoint
        for checkpoint in [active_checkpoint]
        if checkpoint is not None
    }
    if best_jump and best_jump.checkpoint_path and best_jump.checkpoint_path != active_checkpoint:
        try:
            jump_engine = ResidualEngine(best_jump.checkpoint_path)
            jump_info = jump_engine.info()
            profiles.append(
                (
                    "best_jump_model",
                    "Best jump model",
                    "Experiment winner for the lowest threshold jump metric on the held-out sweep.",
                    jump_engine,
                    base,
                    jump_info,
                    best_jump,
                )
            )
            used_checkpoints.add(best_jump.checkpoint_path)
        except ValueError:
            pass
    if (
        best_canonical
        and best_canonical.checkpoint_path
        and best_canonical.checkpoint_path not in used_checkpoints
    ):
        try:
            canonical_engine = ResidualEngine(best_canonical.checkpoint_path)
            canonical_info = canonical_engine.info()
            profiles.append(
                (
                    "best_canonical_model",
                    "Best canonical model",
                    "Experiment winner for the lowest 100% AI canonical threshold jump.",
                    canonical_engine,
                    base,
                    canonical_info,
                    best_canonical,
                )
            )
            used_checkpoints.add(best_canonical.checkpoint_path)
        except ValueError:
            pass
    if (
        best_weighted
        and best_weighted.checkpoint_path
        and best_weighted.checkpoint_path not in used_checkpoints
    ):
        try:
            stable_engine = ResidualEngine(best_weighted.checkpoint_path)
            stable_info = stable_engine.info()
            profiles.append(
                (
                    "best_weighted_model",
                    "Best stable model",
                    "Experiment winner for the lowest weighted reconstruction RMSE.",
                    stable_engine,
                    base,
                    stable_info,
                    best_weighted,
                )
            )
        except ValueError:
            pass
    profiles.append(
        (
            "procedural_prior",
            "Procedural prior",
            "Fallback stochastic residual prior without a trained checkpoint.",
            prior_engine,
            base,
            prior_info,
            None,
        )
    )
    items: list[ModelComparisonItem] = []
    for key, label, description, engine, condition, info, experiment_summary in profiles:
        response = generate_curves(condition, engine)
        items.append(
            ModelComparisonItem(
                key=key,
                label=label,
                description=description,
                residual_mode=response.residual_mode,
                model_name=response.model_name,
                checkpoint_path=info.checkpoint_path,
                ai_residual_strength=condition.ai_residual_strength,
                gate_ai_residual_strength=condition.gate_ai_residual_strength,
                model=info,
                candidate=response.candidates[0],
                experiment_summary=experiment_summary,
            )
        )
    return ModelComparisonResponse(condition=base, items=items)


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


def _experiment_sort_key(
    entry: ExperimentLeaderboardEntry,
) -> tuple[float, float, float, float]:
    return (
        entry.jump_p95_decades if entry.jump_p95_decades is not None else float("inf"),
        entry.generated_vth_mae_v
        if entry.generated_vth_mae_v is not None
        else float("inf"),
        entry.generated_ss_mae_mv_dec
        if entry.generated_ss_mae_mv_dec is not None
        else float("inf"),
        entry.validation_weighted_rmse_decades
        if entry.validation_weighted_rmse_decades is not None
        else float("inf"),
    )


def _leaderboard_entry_from_payload(
    item: dict,
    *,
    experiment_path: Path,
) -> ExperimentLeaderboardEntry:
    result = item.get("result") if isinstance(item.get("result"), dict) else {}
    jump = item.get("jump_metrics") if isinstance(item.get("jump_metrics"), dict) else item
    canonical = (
        item.get("canonical_metrics")
        if isinstance(item.get("canonical_metrics"), dict)
        else item
    )
    method = result.get("method") or item.get("method")
    if method is None:
        method = (
            "hybrid_threshold_pca"
            if str(item.get("name", "")).startswith("hybrid_")
            else "unknown"
        )
    architecture = result.get("architecture") or item.get("architecture") or method
    return ExperimentLeaderboardEntry(
        name=str(item.get("name", experiment_path.name)),
        description=item.get("description"),
        method=str(method),
        architecture=str(architecture) if architecture is not None else None,
        experiment_path=str(experiment_path),
        checkpoint_path=result.get("output") or item.get("checkpoint_path"),
        seconds=float(item["seconds"]) if item.get("seconds") is not None else None,
        validation_rmse_decades=(
            float(result["validation_rmse_decades"])
            if result.get("validation_rmse_decades") is not None
            else None
        ),
        validation_weighted_rmse_decades=(
            float(
                result.get(
                    "validation_weighted_rmse_decades",
                    item.get("validation_weighted_rmse_decades"),
                )
            )
            if result.get("validation_weighted_rmse_decades") is not None
            or item.get("validation_weighted_rmse_decades") is not None
            else None
        ),
        feature_vth_mae_v=(
            float(result["feature_vth_mae_v"])
            if result.get("feature_vth_mae_v") is not None
            else None
        ),
        feature_ss_mae_mv_dec=(
            float(result["feature_ss_mae_mv_dec"])
            if result.get("feature_ss_mae_mv_dec") is not None
            else None
        ),
        jump_p95_decades=(
            float(jump["jump_p95_decades"])
            if jump.get("jump_p95_decades") is not None
            else None
        ),
        jump_spike_rate=(
            float(jump["jump_spike_rate"])
            if jump.get("jump_spike_rate") is not None
            else None
        ),
        generated_vth_mae_v=(
            float(jump["generated_vth_mae_v"])
            if jump.get("generated_vth_mae_v") is not None
            else None
        ),
        generated_ss_mae_mv_dec=(
            float(jump["generated_ss_mae_mv_dec"])
            if jump.get("generated_ss_mae_mv_dec") is not None
            else None
        ),
        canonical_jump_p95_decades=(
            float(canonical["canonical_jump_p95_decades"])
            if canonical.get("canonical_jump_p95_decades") is not None
            else None
        ),
        canonical_jump_max_decades=(
            float(canonical["canonical_jump_max_decades"])
            if canonical.get("canonical_jump_max_decades") is not None
            else None
        ),
    )


def _latest_comparison_artifact_path() -> Path | None:
    if not experiments_root.is_dir():
        return None
    artifacts = [
        path
        for path in experiments_root.glob("*/canonical-model-comparison.svg")
        if path.is_file()
    ]
    if not artifacts:
        return None
    return max(artifacts, key=lambda path: path.stat().st_mtime)


def _load_experiment_leaderboard(limit: int = 18) -> ExperimentLeaderboardResponse:
    entries: list[ExperimentLeaderboardEntry] = []
    if experiments_root.is_dir():
        for summary_path in experiments_root.glob("*/summary.json"):
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if not isinstance(payload, list):
                continue
            for item in payload:
                if not isinstance(item, dict):
                    continue
                try:
                    entries.append(
                        _leaderboard_entry_from_payload(
                            item,
                            experiment_path=summary_path.parent,
                        )
                    )
                except (TypeError, ValueError):
                    continue
    entries.sort(key=_experiment_sort_key)
    best_jump = entries[0] if entries else None
    best_canonical = _best_canonical_entry(entries)
    best_weighted = _best_weighted_entry(entries)
    report_path = experiments_root / "model-selection-report-20260627.md"
    artifact_path = _latest_comparison_artifact_path()
    artifact_url = None
    if artifact_path is not None:
        artifact_url = (
            "/api/model/leaderboard-artifact/latest"
            f"?mtime={int(artifact_path.stat().st_mtime)}"
        )
    return ExperimentLeaderboardResponse(
        entries=entries[: max(limit, 1)],
        best_jump_entry=best_jump,
        best_canonical_entry=best_canonical,
        best_weighted_entry=best_weighted,
        report_path=str(report_path) if report_path.exists() else None,
        comparison_artifact_url=artifact_url,
    )


def _best_weighted_entry(
    entries: list[ExperimentLeaderboardEntry],
) -> ExperimentLeaderboardEntry | None:
    weighted = [
        entry for entry in entries if entry.validation_weighted_rmse_decades is not None
    ]
    if not weighted:
        return None
    return min(
        weighted,
        key=lambda entry: (
            entry.validation_weighted_rmse_decades
            if entry.validation_weighted_rmse_decades is not None
            else float("inf")
        ),
    )


def _best_canonical_entry(
    entries: list[ExperimentLeaderboardEntry],
) -> ExperimentLeaderboardEntry | None:
    canonical = [
        entry for entry in entries if entry.canonical_jump_max_decades is not None
    ]
    if not canonical:
        return None
    return min(
        canonical,
        key=lambda entry: (
            entry.canonical_jump_max_decades
            if entry.canonical_jump_max_decades is not None
            else float("inf"),
            entry.generated_vth_mae_v
            if entry.generated_vth_mae_v is not None
            else float("inf"),
            entry.validation_weighted_rmse_decades
            if entry.validation_weighted_rmse_decades is not None
            else float("inf"),
        ),
    )


def _entry_by_checkpoint(
    entries: list[ExperimentLeaderboardEntry],
    checkpoint_path: str | None,
) -> ExperimentLeaderboardEntry | None:
    if checkpoint_path is None:
        return None
    for entry in entries:
        if entry.checkpoint_path == checkpoint_path:
            return entry
    return None


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


@app.get("/api/database/analyze/status", response_model=DatabaseAnalysisStatus)
def database_analysis_status() -> DatabaseAnalysisStatus:
    return database_analysis_manager.snapshot()


@app.post("/api/database/analyze/start", response_model=DatabaseAnalysisStatus)
def start_database_analysis(request: DatabaseSelectionRequest) -> DatabaseAnalysisStatus:
    selected_count = len(request.curve_ids) if request.curve_ids else 0
    try:
        return database_analysis_manager.start(
            curve_ids=request.curve_ids or None,
            filters=request.filters,
            selected_count=selected_count,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/database/previews")
def database_previews_endpoint(request: CurvePreviewRequest) -> list[dict]:
    try:
        return get_curve_previews(None, request.curve_ids)
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


def _matrix_column_label(index: int) -> str:
    label = ""
    current = index
    while True:
        label = chr(ord("A") + (current % 26)) + label
        current = current // 26 - 1
        if current < 0:
            return label


def _matrix_site_label(row_index: int, col_index: int) -> str:
    return f"{_matrix_column_label(col_index)}{row_index + 1}"


def _matrix_sites(request: MatrixSynthesisRequest) -> list[dict[str, Any]]:
    for parameter in request.parameters:
        if len(parameter.values) != request.rows:
            raise HTTPException(status_code=422, detail=f"{parameter.key} heatmap row count does not match")
        if any(len(row) != request.cols for row in parameter.values):
            raise HTTPException(status_code=422, detail=f"{parameter.key} heatmap column count does not match")
    sites: list[dict[str, Any]] = []
    for row_index in range(request.rows):
        for col_index in range(request.cols):
            parameters = {
                parameter.key: float(parameter.values[row_index][col_index])
                for parameter in request.parameters
            }
            sites.append(
                {
                    "site": _matrix_site_label(row_index, col_index),
                    "row": row_index + 1,
                    "col": col_index + 1,
                    "parameters": parameters,
                }
            )
    return sites


def _generate_matrix_assignment(
    assignment: dict[str, Any],
    base_condition: GenerationCondition,
    index: int,
) -> dict[str, Any]:
    parameters = dict(assignment.get("parameters") or {})
    update = {
        key: value
        for key, value in parameters.items()
        if key in GenerationCondition.model_fields and key != "ion_ioff_ratio"
    }
    update["variants"] = 1
    update["seed"] = base_condition.seed + index
    if "target_ion" in update and "target_ioff" in update:
        ion = float(update["target_ion"])
        ioff = float(update["target_ioff"])
        if ion <= ioff:
            update["target_ion"] = max(ioff * 10.0, ion + abs(ion) * 0.1, 1e-12)
    condition = base_condition.model_copy(update=update)
    candidate = generate_curves(condition, residual_engine).candidates[0]
    return {
        **assignment,
        "source": "generated",
        "generated": {
            "seed": candidate.seed,
            "quality_score": candidate.quality_score,
            "features": candidate.features.model_dump(),
            "voltage": candidate.voltage,
            "forward_current": candidate.forward_current,
            "reverse_current": candidate.reverse_current,
            "gate_forward_current": candidate.gate_forward_current,
            "gate_reverse_current": candidate.gate_reverse_current,
        },
    }


def _complete_matrix_synthesis(request: MatrixSynthesisRequest) -> dict[str, Any]:
    site_count = request.rows * request.cols
    if site_count > 256:
        raise HTTPException(status_code=422, detail="Matrix synthesis is limited to 256 device sites")
    sites = _matrix_sites(request)
    if request.mode == "database":
        assignments = match_matrix_sites(
            None,
            site_targets=sites,
            filters=request.filters,
            duplicate_mode=request.duplicate_mode,
        )
    else:
        assignments = [{**site, "source": "generated"} for site in sites]
    completed: list[dict[str, Any]] = []
    for index, assignment in enumerate(assignments):
        if assignment.get("source") == "generated":
            completed.append(_generate_matrix_assignment(assignment, request.generation_condition, index))
        else:
            completed.append(assignment)
    return {
        "rows": request.rows,
        "cols": request.cols,
        "mode": request.mode,
        "duplicate_mode": request.duplicate_mode,
        "assignments": completed,
        "matched_count": sum(1 for item in completed if item.get("source") == "database"),
        "generated_count": sum(1 for item in completed if item.get("source") == "generated"),
        "unmatched_count": sum(1 for item in completed if item.get("source") == "unmatched"),
        "reused_count": sum(1 for item in completed if item.get("reused")),
    }


@app.post("/api/database/matrix-synthesize")
def database_matrix_synthesize_endpoint(request: MatrixSynthesisRequest) -> dict:
    try:
        return _complete_matrix_synthesis(request)
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error


def _xlsx_cell_ref(row_index: int, col_index: int) -> str:
    label = ""
    current = col_index
    while current >= 0:
        label = chr(ord("A") + (current % 26)) + label
        current = current // 26 - 1
    return f"{label}{row_index + 1}"


def _xlsx_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _xlsx_sheet_xml(rows: list[list[object]]) -> str:
    body: list[str] = []
    for row_index, row in enumerate(rows):
        cells: list[str] = []
        for col_index, value in enumerate(row):
            if value is None:
                continue
            ref = _xlsx_cell_ref(row_index, col_index)
            if isinstance(value, bool):
                cells.append(f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>')
            elif isinstance(value, int | float) and not isinstance(value, bool):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{_xlsx_escape(value)}</t></is></c>')
        body.append(f'<row r="{row_index + 1}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(body)}</sheetData>'
        "</worksheet>"
    )


def _write_xlsx_workbook(sheets: list[tuple[str, list[list[object]]]]) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for index in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{_xlsx_escape(name[:31])}" sheetId="{index}" r:id="rId{index}"/>'
                for index, (name, _) in enumerate(sheets, start=1)
            )
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
                for index in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _xlsx_sheet_xml(rows))
    return buffer.getvalue()


def _matrix_export_rows(result: dict[str, Any]) -> tuple[list[list[object]], list[list[object]], list[list[object]]]:
    parameter_keys = sorted({
        key
        for assignment in result["assignments"]
        for key in (assignment.get("parameters") or {}).keys()
    })
    matrix_rows: list[list[object]] = [["site", "row", "col", "parameter", "target_value"]]
    data_rows: list[list[object]] = [[
        "site", "row", "col", "source", "curve_id", "score", "score_features",
        "reused", "reason", "polarity", "direction", "source_kind", "source_path",
        "generated_seed", *[f"target_{key}" for key in parameter_keys],
    ]]
    curve_series: list[tuple[str, list[tuple[object, object]]]] = []
    for assignment in result["assignments"]:
        params = assignment.get("parameters") or {}
        for key in parameter_keys:
            matrix_rows.append([
                assignment.get("site"),
                assignment.get("row"),
                assignment.get("col"),
                key,
                params.get(key),
            ])
        generated = assignment.get("generated") or {}
        data_rows.append([
            assignment.get("site"),
            assignment.get("row"),
            assignment.get("col"),
            assignment.get("source"),
            assignment.get("curve_id"),
            assignment.get("score"),
            ", ".join(assignment.get("score_features") or []),
            bool(assignment.get("reused")),
            assignment.get("reason"),
            assignment.get("polarity"),
            assignment.get("direction"),
            assignment.get("source_kind"),
            assignment.get("source_path"),
            generated.get("seed"),
            *[params.get(key) for key in parameter_keys],
        ])
        if assignment.get("source") == "database" and assignment.get("curve_id"):
            detail = get_curve_detail(None, str(assignment["curve_id"]))
            if detail:
                label_base = f"{assignment.get('site')}_{assignment.get('curve_id')}"
                raw_points_series = [
                    (point.get("voltage_v"), point.get("current_a"))
                    for point in detail.get("raw_points", [])
                ]
                if raw_points_series:
                    curve_series.append((f"{label_base}_raw_id", raw_points_series))
                gate_points_series = [
                    (point.get("voltage_v"), point.get("current_a"))
                    for point in detail.get("gate_points", [])
                ]
                if gate_points_series:
                    curve_series.append((f"{label_base}_raw_ig", gate_points_series))
        elif assignment.get("source") == "generated":
            voltage = generated.get("voltage") or []
            channels = [
                ("id_forward", generated.get("forward_current") or []),
                ("id_reverse", generated.get("reverse_current") or []),
                ("ig_forward", generated.get("gate_forward_current") or []),
                ("ig_reverse", generated.get("gate_reverse_current") or []),
            ]
            for channel, currents in channels:
                series = [
                    (voltage_value, currents[index] if index < len(currents) else None)
                    for index, voltage_value in enumerate(voltage)
                ]
                if series:
                    curve_series.append((f"{assignment.get('site')}_seed{generated.get('seed')}_{channel}", series))
    curve_headers: list[object] = ["point_index"]
    for label, _series in curve_series:
        curve_headers.extend([f"{label}_x", f"{label}_y"])
    max_points = max((len(series) for _label, series in curve_series), default=0)
    curve_rows: list[list[object]] = [curve_headers]
    for index in range(max_points):
        row: list[object] = [index]
        for _label, series in curve_series:
            if index < len(series):
                x_value, y_value = series[index]
                row.extend([x_value, y_value])
            else:
                row.extend([None, None])
        curve_rows.append(row)
    return matrix_rows, data_rows, curve_rows


@app.post("/api/database/matrix-export")
def database_matrix_export_endpoint(request: MatrixSynthesisRequest) -> Response:
    try:
        result = _complete_matrix_synthesis(request)
        matrix_rows, data_rows, curve_rows = _matrix_export_rows(result)
    except (ValueError, SQLAlchemyError) as error:
        raise _database_error(error) from error
    workbook = _write_xlsx_workbook([
        ("Matrix", matrix_rows),
        ("Data", data_rows),
        ("Curves", curve_rows),
    ])
    return Response(
        workbook,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="devicecurvegen-matrix-output.xlsx"'},
    )


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


def _write_xyxy_csv_to_zip(
    archive: ZipFile,
    filename: str,
    payload: dict[str, Any],
    *,
    include_ig: bool = True,
) -> None:
    series_by_label: list[tuple[str, list[tuple[object, object]]]] = []
    curve_order = [str(row["curve_id"]) for row in payload["curves"]]
    raw_by_curve: dict[str, list[tuple[object, object]]] = {curve_id: [] for curve_id in curve_order}
    gate_by_curve: dict[str, list[tuple[object, object]]] = {curve_id: [] for curve_id in curve_order}

    for point in payload["raw_points"]:
        raw_by_curve.setdefault(str(point["curve_id"]), []).append(
            (point.get("voltage_v"), point.get("current_a"))
        )
    if include_ig:
        for point in payload["gate_points"]:
            gate_by_curve.setdefault(str(point["curve_id"]), []).append(
                (point.get("voltage_v"), point.get("current_a"))
            )

    for curve_id in curve_order:
        raw_series = raw_by_curve.get(curve_id, [])
        if raw_series:
            series_by_label.append((f"{curve_id}_raw_id", raw_series))
        gate_series = gate_by_curve.get(curve_id, [])
        if gate_series:
            series_by_label.append((f"{curve_id}_raw_ig", gate_series))

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    header: list[object] = ["point_index"]
    for label, _series in series_by_label:
        header.extend([f"{label}_x", f"{label}_y"])
    writer.writerow(header)
    max_points = max((len(series) for _label, series in series_by_label), default=0)
    for index in range(max_points):
        row: list[object] = [index]
        for _label, series in series_by_label:
            if index < len(series):
                row.extend(series[index])
            else:
                row.extend([None, None])
        writer.writerow(row)
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
    options = request.export_options
    files: list[str] = []
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        if options.xyxy_curves:
            _write_xyxy_csv_to_zip(
                archive,
                "xyxy_curves.csv",
                payload,
                include_ig=options.include_ig,
            )
            files.append("xyxy_curves.csv")
        if options.curve_metadata:
            _write_csv_to_zip(archive, "curves.csv", payload["curves"])
            files.append("curves.csv")
        if options.raw_id_points:
            _write_csv_to_zip(archive, "raw_id_points.csv", payload["raw_points"])
            files.append("raw_id_points.csv")
        if options.include_ig and options.raw_ig_points:
            _write_csv_to_zip(archive, "raw_ig_points.csv", payload["gate_points"])
            files.append("raw_ig_points.csv")
        if options.include_ig and options.aligned_ig_points:
            _write_csv_to_zip(archive, "aligned_ig_points.csv", payload["aligned_gate_points"])
            files.append("aligned_ig_points.csv")
        if options.analysis_json:
            archive.writestr(
                "analysis.json",
                json.dumps(payload["analysis"], indent=2, ensure_ascii=False),
            )
            files.append("analysis.json")
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
                    "export_options": options.model_dump(),
                    "files": files,
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

    def frontend_file_response(path: Path) -> FileResponse:
        response = FileResponse(path)
        if path.suffix in {".html", ".js", ".css"}:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        requested = (frontend_root / path).resolve()
        if requested.is_relative_to(frontend_root) and requested.is_file():
            return frontend_file_response(requested)
        return frontend_file_response(frontend_root / "index.html")
