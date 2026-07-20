from __future__ import annotations

import hashlib
import os
import statistics
import time
from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from sqlalchemy import (
    JSON,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    bindparam,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    or_,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError

from .b1500_dataset import (
    SUPPORTED_B1500_SUFFIXES,
    X_NORM_GRID,
    _align_segment,
    _clean_pair,
    _gate_current_rank,
    _is_gate_current,
    _metadata_axis,
    _quality_rejection,
    _safe_relpath,
    _segment_sweeps,
    _stable_curve_id,
    choose_columns,
    parse_source,
)
from .features import analyze_transfer_curve

DEFAULT_DATABASE_ENV = "DEVICEGEN_DATABASE_URL"
BATCH_SIZE = 10_000
ANALYSIS_SAMPLE_LIMIT = 2_400
ANALYSIS_HISTOGRAM_BINS = 32
ANALYSIS_PCA_MAX_COMPONENTS = 6

ANALYSIS_NUMERIC_FEATURES = (
    "logIon",
    "logIoff",
    "logRatio",
    "vth",
    "ss_mv_dec",
    "logGm",
    "noise_log_sigma",
    "ambipolar_strength",
    "hysteresis_v",
    "rows_clean",
    "voltage_span",
)

ANALYSIS_METRIC_SOURCE_KEYS = {
    "ion": "ion",
    "ioff": "ioff",
    "logIon": "logIon",
    "logIoff": "logIoff",
    "logRatio": "logRatio",
    "vth": "vth",
    "ss_mv_dec": "ss_mv_dec",
    "gm_max": "gm_max",
    "logGm": "logGm",
    "noise_log_sigma": "noise_log_sigma",
    "ambipolar_strength": "ambipolar_strength",
    "hysteresis_v": "hysteresis_v",
    "rows_clean": "rows_clean",
    "voltage_span": "voltage_span",
}

_schema_lock = Lock()
_initialized_schema_urls: set[str] = set()

metadata = MetaData()

source_files = Table(
    "source_files",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_path", String(1024), nullable=False, unique=True),
    Column("extension", String(16), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("modified_at", DateTime, nullable=True),
    Column("sha1", String(40), nullable=True),
    Column("created_at", DateTime, nullable=False, default=datetime.utcnow),
)

test_configs = Table(
    "test_configs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_file_id", ForeignKey("source_files.id", ondelete="CASCADE"), nullable=False),
    Column("table_name", String(255), nullable=False),
    Column("source_kind", String(64), nullable=False),
    Column("setup_title", String(512), nullable=True),
    Column("primitive_test", String(255), nullable=True),
    Column("x_axis_data", String(128), nullable=True),
    Column("voltage_column", String(128), nullable=False),
    Column("current_column", String(128), nullable=False),
    Column("gate_current_column", String(128), nullable=True),
    Column("classification", String(32), nullable=False),
    Column("classification_reason", String(255), nullable=False),
    Column("classification_confidence", Float, nullable=False),
    Column("columns_json", JSON, nullable=False),
    Column("metadata_json", JSON, nullable=False),
)

curves = Table(
    "curves",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("curve_id", String(32), nullable=False, unique=True),
    Column("source_file_id", ForeignKey("source_files.id", ondelete="CASCADE"), nullable=False),
    Column("test_config_id", ForeignKey("test_configs.id", ondelete="CASCADE"), nullable=False),
    Column("segment_index", Integer, nullable=False),
    Column("direction", String(16), nullable=False),
    Column("sweep_pair_id", String(40), nullable=True),
    Column("test_time", DateTime, nullable=True),
    Column("rows_clean", Integer, nullable=False),
    Column("voltage_min_v", Float, nullable=False),
    Column("voltage_max_v", Float, nullable=False),
    Column("ion", Float, nullable=False),
    Column("ioff", Float, nullable=False),
    Column("ion_ioff_ratio", Float, nullable=False),
    Column("polarity", String(16), nullable=False),
    Column("has_gate_current", Integer, nullable=False, default=0),
    Column("vth", Float, nullable=True),
    Column("ss_mv_dec", Float, nullable=True),
    Column("ss_fit_r2", Float, nullable=True),
    Column("gm_max", Float, nullable=True),
    Column("vth_gmmax", Float, nullable=True),
    Column("von", Float, nullable=True),
    Column("hysteresis_v", Float, nullable=True),
    Column("leakage_level", Float, nullable=True),
    Column("noise_log_sigma", Float, nullable=True),
    Column("ambipolar_strength", Float, nullable=True),
    Column("current_floor", Float, nullable=True),
    Column("imported_at", DateTime, nullable=False, default=datetime.utcnow),
)

raw_points = Table(
    "raw_points",
    metadata,
    Column(
        "curve_id",
        String(32),
        ForeignKey("curves.curve_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("point_index", Integer, primary_key=True),
    Column("voltage_v", Float, nullable=False),
    Column("current_a", Float, nullable=False),
)

raw_gate_points = Table(
    "raw_gate_points",
    metadata,
    Column(
        "curve_id",
        String(32),
        ForeignKey("curves.curve_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("point_index", Integer, primary_key=True),
    Column("voltage_v", Float, nullable=False),
    Column("current_a", Float, nullable=False),
)

aligned_gate_points = Table(
    "aligned_gate_points",
    metadata,
    Column(
        "curve_id",
        String(32),
        ForeignKey("curves.curve_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("point_index", Integer, primary_key=True),
    Column("x_norm", Float, nullable=False),
    Column("voltage_v", Float, nullable=False),
    Column("log10_abs_ig", Float, nullable=False),
    Column("abs_ig_a", Float, nullable=False),
)

aligned_points = Table(
    "aligned_points",
    metadata,
    Column(
        "curve_id",
        String(32),
        ForeignKey("curves.curve_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column("point_index", Integer, primary_key=True),
    Column("x_norm", Float, nullable=False),
    Column("voltage_v", Float, nullable=False),
    Column("log10_abs_id", Float, nullable=False),
    Column("abs_id_a", Float, nullable=False),
)

rejected_entries = Table(
    "rejected_entries",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("source_file_id", ForeignKey("source_files.id", ondelete="CASCADE"), nullable=True),
    Column("source_path", String(1024), nullable=False),
    Column("table_name", String(255), nullable=True),
    Column("reason", Text, nullable=False),
)

Index("ix_curves_polarity", curves.c.polarity)
Index("ix_curves_direction", curves.c.direction)
Index("ix_curves_test_time", curves.c.test_time)
Index("ix_curves_polarity_test_time", curves.c.polarity, curves.c.test_time)
Index("ix_curves_direction_test_time", curves.c.direction, curves.c.test_time)
Index("ix_curves_hysteresis_test_time", curves.c.hysteresis_v, curves.c.test_time)
Index("ix_curves_sweep_pair", curves.c.sweep_pair_id)
Index("ix_curves_ion", curves.c.ion)
Index("ix_curves_ioff", curves.c.ioff)
Index("ix_curves_vth", curves.c.vth)
Index("ix_curves_ss", curves.c.ss_mv_dec)
Index("ix_source_files_modified", source_files.c.modified_at)
Index("ix_test_configs_source_kind", test_configs.c.source_kind)


def resolve_database_url(database_url: str | None = None) -> str:
    resolved = database_url or os.getenv(DEFAULT_DATABASE_ENV)
    if not resolved:
        raise ValueError(
            f"MySQL database URL is required. Set {DEFAULT_DATABASE_ENV} or pass --database-url."
        )
    return resolved


@lru_cache(maxsize=8)
def _cached_database_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True, pool_recycle=1800, future=True)


def create_database_engine(database_url: str | None = None) -> Engine:
    return _cached_database_engine(resolve_database_url(database_url))


def create_schema(engine: Engine) -> None:
    schema_key = str(engine.url)
    if schema_key in _initialized_schema_urls:
        return
    with _schema_lock:
        if schema_key in _initialized_schema_urls:
            return
        metadata.create_all(engine)
        _ensure_schema_compat(engine)
        _initialized_schema_urls.add(schema_key)


def _ensure_schema_compat(engine: Engine) -> None:
    """Add non-destructive columns introduced after the first database schema."""

    inspector = inspect(engine)
    table_columns = {
        table_name: {column["name"] for column in inspector.get_columns(table_name)}
        for table_name in inspector.get_table_names()
    }
    statements: list[str] = []
    needs_scan_backfill = False
    if (
        "test_configs" in table_columns
        and "gate_current_column" not in table_columns["test_configs"]
    ):
        statements.append(
            "ALTER TABLE test_configs ADD COLUMN gate_current_column VARCHAR(128) NULL"
        )
    if "curves" in table_columns and "has_gate_current" not in table_columns["curves"]:
        statements.append(
            "ALTER TABLE curves ADD COLUMN has_gate_current INTEGER NOT NULL DEFAULT 0"
        )
    if "curves" in table_columns and "sweep_pair_id" not in table_columns["curves"]:
        statements.append("ALTER TABLE curves ADD COLUMN sweep_pair_id VARCHAR(40) NULL")
        needs_scan_backfill = True
    if "curves" in table_columns and "test_time" not in table_columns["curves"]:
        statements.append("ALTER TABLE curves ADD COLUMN test_time DATETIME NULL")
        needs_scan_backfill = True
    with engine.begin() as connection:
        for statement in statements:
            try:
                connection.execute(text(statement))
            except SQLAlchemyError as error:
                message = str(error).lower()
                if "duplicate column" in message or "duplicate column name" in message:
                    continue
                raise
    compatibility_indexes = {
        "ix_curves_test_time",
        "ix_curves_polarity_test_time",
        "ix_curves_direction_test_time",
        "ix_curves_hysteresis_test_time",
        "ix_curves_sweep_pair",
    }
    for index in curves.indexes:
        if index.name not in compatibility_indexes:
            continue
        index.create(engine, checkfirst=True)
    if needs_scan_backfill:
        _backfill_curve_scan_metadata(engine)


def _backfill_curve_scan_metadata(engine: Engine) -> None:
    with engine.begin() as connection:
        missing_time = connection.scalar(
            select(func.count()).select_from(curves).where(curves.c.test_time.is_(None))
        )
        if missing_time:
            rows = connection.execute(
                select(curves.c.curve_id, source_files.c.modified_at)
                .select_from(curves.join(source_files))
                .where(curves.c.test_time.is_(None))
            ).all()
            connection.execute(
                update(curves)
                .where(curves.c.curve_id == bindparam("curve_key"))
                .values(test_time=bindparam("new_test_time")),
                [
                    {"curve_key": curve_id, "new_test_time": modified_at}
                    for curve_id, modified_at in rows
                ],
            )
        pair_rows = connection.execute(
            select(
                curves.c.curve_id,
                curves.c.test_config_id,
                curves.c.segment_index,
                curves.c.direction,
                curves.c.vth,
                curves.c.sweep_pair_id,
            )
            .where(curves.c.direction.in_(("forward", "reverse")))
            .order_by(curves.c.test_config_id, curves.c.segment_index)
        ).mappings()
        pending: list[dict[str, Any]] = []
        previous: dict[str, Any] | None = None
        for row in pair_rows:
            current = dict(row)
            if (
                previous is not None
                and previous["test_config_id"] == current["test_config_id"]
                and current["segment_index"] == previous["segment_index"] + 1
                and previous["direction"] != current["direction"]
                and previous["vth"] is not None
                and current["vth"] is not None
            ):
                pair_id = hashlib.sha1(
                    f"{current['test_config_id']}:{previous['segment_index']}".encode()
                ).hexdigest()[:20]
                hysteresis = abs(float(current["vth"]) - float(previous["vth"]))
                pending.extend(
                    [
                        {
                            "curve_key": previous["curve_id"],
                            "pair_id": pair_id,
                            "hysteresis": hysteresis,
                        },
                        {
                            "curve_key": current["curve_id"],
                            "pair_id": pair_id,
                            "hysteresis": hysteresis,
                        },
                    ]
                )
                previous = None
            else:
                previous = current
        if pending:
            connection.execute(
                update(curves)
                .where(curves.c.curve_id == bindparam("curve_key"))
                .values(
                    sweep_pair_id=bindparam("pair_id"),
                    hysteresis_v=bindparam("hysteresis"),
                ),
                pending,
            )


def reset_schema_data(engine: Engine) -> None:
    ordered = [
        aligned_gate_points,
        aligned_points,
        raw_gate_points,
        raw_points,
        curves,
        test_configs,
        rejected_entries,
        source_files,
    ]
    with engine.begin() as connection:
        for table in ordered:
            connection.execute(delete(table))


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_metadata(metadata_json: dict[str, list[str]], key: str) -> str | None:
    values = metadata_json.get(key, [])
    return values[0] if values else None


def _record_rejection(
    connection: Any,
    *,
    source_file_id: int | None,
    source_path: str,
    table_name: str | None,
    reason: str,
) -> None:
    connection.execute(
        insert(rejected_entries).values(
            source_file_id=source_file_id,
            source_path=source_path,
            table_name=table_name,
            reason=reason,
        )
    )


def _insert_many(connection: Any, table: Table, rows: list[dict[str, Any]]) -> None:
    if rows:
        connection.execute(insert(table), rows)
        rows.clear()


def _align_current_to_grid(
    voltage: np.ndarray,
    current: np.ndarray,
    physical_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(voltage)
    voltage = voltage[order]
    current = current[order]
    unique_voltage, unique_indices = np.unique(voltage, return_index=True)
    unique_current = np.abs(current[unique_indices])
    if unique_voltage.size < 2:
        raise ValueError("gate segment has fewer than two unique voltage points")
    log_current = np.log10(np.clip(unique_current, np.finfo(float).tiny, None))
    aligned_log_current = np.interp(physical_grid, unique_voltage, log_current)
    aligned_current = np.power(10.0, aligned_log_current)
    return aligned_log_current, aligned_current


def _normalize_suffixes(suffixes: set[str] | None) -> set[str]:
    if suffixes is None:
        return SUPPORTED_B1500_SUFFIXES
    normalized = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}" for suffix in suffixes
    }
    unsupported = normalized - SUPPORTED_B1500_SUFFIXES
    if unsupported:
        raise ValueError(f"Unsupported suffixes: {', '.join(sorted(unsupported))}")
    return normalized


def _source_file_is_unchanged(
    existing: dict[str, Any],
    *,
    size_bytes: int,
    modified_at: datetime,
    sha1: str | None,
) -> bool:
    if int(existing["size_bytes"]) != int(size_bytes):
        return False
    existing_modified = existing.get("modified_at")
    if existing_modified is None:
        return False
    if abs(existing_modified.timestamp() - modified_at.timestamp()) > 1:
        return False
    existing_sha1 = existing.get("sha1")
    if sha1 is not None or existing_sha1 is not None:
        return sha1 is not None and existing_sha1 == sha1
    return True


def _mysql_lock_error_code(error: OperationalError) -> int | None:
    original = getattr(error, "orig", None)
    args = getattr(original, "args", ())
    if not args:
        return None
    try:
        return int(args[0])
    except (TypeError, ValueError):
        return None


def _run_with_mysql_lock_retry(operation: Callable[[], Any], *, attempts: int = 6) -> Any:
    for attempt in range(attempts):
        try:
            return operation()
        except OperationalError as error:
            if _mysql_lock_error_code(error) not in {1205, 1213} or attempt == attempts - 1:
                raise
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError("unreachable")


def _source_file_row(connection: Any, source_path: str) -> dict[str, Any] | None:
    row = connection.execute(
        select(
            source_files.c.id,
            source_files.c.source_path,
            source_files.c.size_bytes,
            source_files.c.modified_at,
            source_files.c.sha1,
        ).where(source_files.c.source_path == source_path)
    ).mappings().first()
    return dict(row) if row is not None else None


def _record_import_rejection_with_retry(
    engine: Engine,
    *,
    source_path: str,
    table_name: str | None,
    reason: str,
) -> None:
    def operation() -> None:
        with engine.begin() as connection:
            existing = _source_file_row(connection, source_path)
            _record_rejection(
                connection,
                source_file_id=int(existing["id"]) if existing else None,
                source_path=source_path,
                table_name=table_name,
                reason=reason,
            )

    _run_with_mysql_lock_retry(operation)


def _import_b1500_file_rows(
    engine: Engine,
    *,
    path: Path,
    relative_path: str,
    stat: os.stat_result,
    modified_at: datetime,
    sha1: str | None,
    tables: list[Any],
) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        raw_batch: list[dict[str, Any]] = []
        gate_batch: list[dict[str, Any]] = []
        aligned_batch: list[dict[str, Any]] = []
        aligned_gate_batch: list[dict[str, Any]] = []
        summary = {
            "files_imported": 0,
            "files_updated": 0,
            "accepted_transfer_segments": 0,
            "rejected_entries": 0,
            "source_file": {
                "id": None,
                "source_path": relative_path,
                "size_bytes": stat.st_size,
                "modified_at": modified_at,
                "sha1": sha1,
            },
        }
        with engine.begin() as connection:
            existing = _source_file_row(connection, relative_path)
            if existing:
                connection.execute(
                    delete(source_files).where(source_files.c.id == int(existing["id"]))
                )
                summary["files_updated"] = 1
            else:
                summary["files_imported"] = 1
            source_result = connection.execute(
                insert(source_files).values(
                    source_path=relative_path,
                    extension=path.suffix.lower(),
                    size_bytes=stat.st_size,
                    modified_at=modified_at,
                    sha1=sha1,
                    created_at=datetime.utcnow(),
                )
            )
            source_file_id = int(source_result.inserted_primary_key[0])
            summary["source_file"]["id"] = source_file_id

            for table in tables:
                choice = choose_columns(table)
                if choice is None:
                    _record_rejection(
                        connection,
                        source_file_id=source_file_id,
                        source_path=relative_path,
                        table_name=table.table_name,
                        reason="no content-level FET transfer/output column pattern",
                    )
                    summary["rejected_entries"] += 1
                    continue
                metadata_json = _clean_json(table.metadata)
                config_result = connection.execute(
                    insert(test_configs).values(
                        source_file_id=source_file_id,
                        table_name=table.table_name,
                        source_kind=table.source_kind,
                        setup_title=_first_metadata(metadata_json, "SetupTitle"),
                        primitive_test=_first_metadata(metadata_json, "PrimitiveTest"),
                        x_axis_data=_metadata_axis(table.metadata),
                        voltage_column=choice.voltage,
                        current_column=choice.current,
                        gate_current_column=choice.gate_current,
                        classification=choice.curve_type,
                        classification_reason=choice.reason,
                        classification_confidence=choice.confidence,
                        columns_json=list(map(str, table.frame.columns)),
                        metadata_json=metadata_json,
                    )
                )
                config_id = int(config_result.inserted_primary_key[0])
                if choice.curve_type != "transfer":
                    _record_rejection(
                        connection,
                        source_file_id=source_file_id,
                        source_path=relative_path,
                        table_name=table.table_name,
                        reason=f"content classified as {choice.curve_type}, not transfer",
                    )
                    summary["rejected_entries"] += 1
                    continue

                cleaned = _clean_pair(table.frame, choice.voltage, choice.current)
                gate_cleaned = (
                    _clean_pair(table.frame, choice.voltage, choice.gate_current)
                    if choice.gate_current is not None
                    else None
                )
                rejection = _quality_rejection(cleaned)
                if rejection:
                    _record_rejection(
                        connection,
                        source_file_id=source_file_id,
                        source_path=relative_path,
                        table_name=table.table_name,
                        reason=rejection,
                    )
                    summary["rejected_entries"] += 1
                    continue

                segments = _segment_sweeps(
                    cleaned["voltage_v"].to_numpy(dtype=float),
                    cleaned["current_a"].to_numpy(dtype=float),
                )
                gate_segments = (
                    _segment_sweeps(
                        gate_cleaned["voltage_v"].to_numpy(dtype=float),
                        gate_cleaned["current_a"].to_numpy(dtype=float),
                    )
                    if gate_cleaned is not None
                    else []
                )
                scan_metadata: dict[int, tuple[str, float]] = {}
                for pair_start in range(len(segments) - 1):
                    first = segments[pair_start]
                    second = segments[pair_start + 1]
                    if (
                        first.direction == second.direction
                        or "single" in {first.direction, second.direction}
                        or first.features is None
                        or second.features is None
                        or first.features.vth is None
                        or second.features.vth is None
                    ):
                        continue
                    pair_id = hashlib.sha1(
                        f"{relative_path}:{table.table_name}:{pair_start + 1}".encode()
                    ).hexdigest()[:20]
                    hysteresis = abs(float(second.features.vth) - float(first.features.vth))
                    scan_metadata[pair_start + 1] = (pair_id, hysteresis)
                    scan_metadata[pair_start + 2] = (pair_id, hysteresis)
                for segment_index, segment in enumerate(segments, start=1):
                    gate_segment = (
                        gate_segments[segment_index - 1]
                        if len(gate_segments) >= segment_index
                        else None
                    )
                    features = segment.features
                    if (
                        segment.rows < 20
                        or features is None
                        or features.vth is None
                        or features.ss_mv_dec is None
                        or features.polarity == "unknown"
                        or features.ion_ioff_ratio < 20
                    ):
                        _record_rejection(
                            connection,
                            source_file_id=source_file_id,
                            source_path=relative_path,
                            table_name=table.table_name,
                            reason="segment lacks stable transfer features",
                        )
                        summary["rejected_entries"] += 1
                        continue

                    physical_grid, aligned_log_current, aligned_current = _align_segment(
                        np.asarray(segment.voltage, dtype=float),
                        np.asarray(segment.current, dtype=float),
                    )
                    curve_id = _stable_curve_id(path, table.table_name, segment_index)
                    feature_payload = features.model_dump(mode="json")
                    pair_id, hysteresis_v = scan_metadata.get(
                        segment_index,
                        (None, None),
                    )
                    curve_result = connection.execute(
                        insert(curves).values(
                            curve_id=curve_id,
                            source_file_id=source_file_id,
                            test_config_id=config_id,
                            segment_index=segment_index,
                            direction=segment.direction,
                            sweep_pair_id=pair_id,
                            test_time=datetime.fromtimestamp(stat.st_mtime),
                            rows_clean=segment.rows,
                            voltage_min_v=float(np.min(segment.voltage)),
                            voltage_max_v=float(np.max(segment.voltage)),
                            ion=feature_payload["ion"],
                            ioff=feature_payload["ioff"],
                            ion_ioff_ratio=feature_payload["ion_ioff_ratio"],
                            polarity=feature_payload["polarity"],
                            has_gate_current=1 if gate_segment is not None else 0,
                            vth=feature_payload["vth"],
                            ss_mv_dec=feature_payload["ss_mv_dec"],
                            ss_fit_r2=feature_payload["ss_fit_r2"],
                            gm_max=feature_payload["gm_max"],
                            vth_gmmax=feature_payload["vth_gmmax"],
                            von=feature_payload["von"],
                            hysteresis_v=hysteresis_v,
                            leakage_level=feature_payload["leakage_level"],
                            noise_log_sigma=feature_payload["noise_log_sigma"],
                            ambipolar_strength=feature_payload["ambipolar_strength"],
                            current_floor=feature_payload["current_floor"],
                            imported_at=datetime.utcnow(),
                        )
                    )
                    if curve_result.rowcount != 1:
                        continue
                    for point_index, (voltage_v, current_a) in enumerate(
                        zip(segment.voltage, segment.current, strict=True)
                    ):
                        raw_batch.append(
                            {
                                "curve_id": curve_id,
                                "point_index": point_index,
                                "voltage_v": float(voltage_v),
                                "current_a": float(current_a),
                            }
                        )
                        if len(raw_batch) >= BATCH_SIZE:
                            _insert_many(connection, raw_points, raw_batch)
                    if gate_segment is not None:
                        aligned_log_gate, aligned_gate_current = _align_current_to_grid(
                            np.asarray(gate_segment.voltage, dtype=float),
                            np.asarray(gate_segment.current, dtype=float),
                            physical_grid,
                        )
                        for point_index, (voltage_v, current_a) in enumerate(
                            zip(gate_segment.voltage, gate_segment.current, strict=True)
                        ):
                            gate_batch.append(
                                {
                                    "curve_id": curve_id,
                                    "point_index": point_index,
                                    "voltage_v": float(voltage_v),
                                    "current_a": float(current_a),
                                }
                            )
                            if len(gate_batch) >= BATCH_SIZE:
                                _insert_many(connection, raw_gate_points, gate_batch)
                        for point_index, (x_norm, voltage_v, log_ig, abs_ig) in enumerate(
                            zip(
                                X_NORM_GRID,
                                physical_grid,
                                aligned_log_gate,
                                aligned_gate_current,
                                strict=True,
                            )
                        ):
                            aligned_gate_batch.append(
                                {
                                    "curve_id": curve_id,
                                    "point_index": point_index,
                                    "x_norm": float(x_norm),
                                    "voltage_v": float(voltage_v),
                                    "log10_abs_ig": float(log_ig),
                                    "abs_ig_a": float(abs_ig),
                                }
                            )
                            if len(aligned_gate_batch) >= BATCH_SIZE:
                                _insert_many(connection, aligned_gate_points, aligned_gate_batch)
                    for point_index, (x_norm, voltage_v, log_id, abs_id) in enumerate(
                        zip(
                            X_NORM_GRID,
                            physical_grid,
                            aligned_log_current,
                            aligned_current,
                            strict=True,
                        )
                    ):
                        aligned_batch.append(
                            {
                                "curve_id": curve_id,
                                "point_index": point_index,
                                "x_norm": float(x_norm),
                                "voltage_v": float(voltage_v),
                                "log10_abs_id": float(log_id),
                                "abs_id_a": float(abs_id),
                            }
                        )
                        if len(aligned_batch) >= BATCH_SIZE:
                            _insert_many(connection, aligned_points, aligned_batch)
                    summary["accepted_transfer_segments"] += 1
            _insert_many(connection, raw_points, raw_batch)
            _insert_many(connection, raw_gate_points, gate_batch)
            _insert_many(connection, aligned_points, aligned_batch)
            _insert_many(connection, aligned_gate_points, aligned_gate_batch)
        return summary

    return _run_with_mysql_lock_retry(operation)


def import_b1500_to_mysql(
    source: Path,
    database_url: str | None = None,
    *,
    replace: bool = False,
    suffixes: set[str] | None = None,
    max_xml_mb: float = 128.0,
    hash_files: bool = False,
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    engine = create_database_engine(database_url)
    create_schema(engine)
    if replace:
        reset_schema_data(engine)

    allowed_suffixes = _normalize_suffixes(suffixes)
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )
    accepted_segments = 0
    rejected_count = 0
    skipped_files = 0
    imported_files = 0
    updated_files = 0

    with engine.connect() as connection:
        existing_by_path = {
            str(row["source_path"]): dict(row)
            for row in connection.execute(
                select(
                    source_files.c.id,
                    source_files.c.source_path,
                    source_files.c.size_bytes,
                    source_files.c.modified_at,
                    source_files.c.sha1,
                )
            ).mappings()
        }

    for file_index, path in enumerate(files, start=1):
        stat = path.stat()
        relative_path = _safe_relpath(path, source)
        modified_at = datetime.fromtimestamp(stat.st_mtime)
        existing = existing_by_path.get(relative_path)
        sha1 = _file_sha1(path) if hash_files else None
        if existing and _source_file_is_unchanged(
            existing,
            size_bytes=stat.st_size,
            modified_at=modified_at,
            sha1=sha1,
        ):
            skipped_files += 1
            continue
        try:
            tables = parse_source(path, max_xml_mb=max_xml_mb)
        except (OSError, ValueError) as error:
            _record_import_rejection_with_retry(
                engine,
                source_path=relative_path,
                table_name=None,
                reason=str(error),
            )
            rejected_count += 1
            continue
        if not tables:
            _record_import_rejection_with_retry(
                engine,
                source_path=relative_path,
                table_name=None,
                reason="no numeric B1500 data table found",
            )
            rejected_count += 1
            continue

        file_summary = _import_b1500_file_rows(
            engine,
            path=path,
            relative_path=relative_path,
            stat=stat,
            modified_at=modified_at,
            sha1=sha1,
            tables=tables,
        )
        imported_files += int(file_summary["files_imported"])
        updated_files += int(file_summary["files_updated"])
        accepted_segments += int(file_summary["accepted_transfer_segments"])
        rejected_count += int(file_summary["rejected_entries"])
        existing_by_path[relative_path] = dict(file_summary["source_file"])
        if file_index % 500 == 0:
            print(
                f"Imported {file_index}/{len(files)} files; "
                f"accepted {accepted_segments} segments",
                flush=True,
            )

    return {
        "source": str(source),
        "files_discovered": len(files),
        "files_imported": imported_files,
        "files_updated": updated_files,
        "files_skipped": skipped_files,
        "accepted_transfer_segments": accepted_segments,
        "rejected_entries": rejected_count,
    }


def _flush_gate_backfill_batch(
    engine: Engine,
    curve_ids: set[str],
    raw_batch: list[dict[str, Any]],
    aligned_batch: list[dict[str, Any]],
) -> None:
    if not curve_ids and not raw_batch and not aligned_batch:
        return
    batch_curve_ids = sorted(curve_ids)
    raw_rows = sorted(
        raw_batch,
        key=lambda row: (str(row["curve_id"]), int(row["point_index"])),
    )
    aligned_rows = sorted(
        aligned_batch,
        key=lambda row: (str(row["curve_id"]), int(row["point_index"])),
    )
    for attempt in range(5):
        try:
            with engine.begin() as connection:
                if batch_curve_ids:
                    connection.execute(
                        delete(raw_gate_points).where(
                            raw_gate_points.c.curve_id.in_(batch_curve_ids)
                        )
                    )
                    connection.execute(
                        delete(aligned_gate_points).where(
                            aligned_gate_points.c.curve_id.in_(batch_curve_ids)
                        )
                    )
                if raw_rows:
                    connection.execute(insert(raw_gate_points), raw_rows)
                if aligned_rows:
                    connection.execute(insert(aligned_gate_points), aligned_rows)
            break
        except OperationalError as exc:
            error_code = exc.orig.args[0] if exc.orig.args else None
            if error_code not in {1205, 1213} or attempt == 4:
                raise
            time.sleep(0.25 * (attempt + 1))
    curve_ids.clear()
    raw_batch.clear()
    aligned_batch.clear()


def backfill_b1500_gate_points(
    source: Path,
    database_url: str | None = None,
    *,
    dry_run: bool = False,
    replace: bool = False,
    limit: int | None = None,
    max_xml_mb: float = 128.0,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    """Populate raw and aligned Ig points for already-imported B1500 curves."""

    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")
    engine = create_database_engine(database_url)
    create_schema(engine)

    joined = curves.join(
        source_files,
        curves.c.source_file_id == source_files.c.id,
    ).join(
        test_configs,
        curves.c.test_config_id == test_configs.c.id,
    )
    query = (
        select(
            curves.c.curve_id,
            curves.c.segment_index,
            source_files.c.source_path,
            test_configs.c.table_name,
            test_configs.c.voltage_column,
            test_configs.c.current_column,
            test_configs.c.gate_current_column,
        )
        .select_from(joined)
        .where(curves.c.has_gate_current == 1)
        .where(test_configs.c.gate_current_column.is_not(None))
        .order_by(source_files.c.source_path, test_configs.c.table_name, curves.c.segment_index)
    )

    with engine.connect() as connection:
        rows = [dict(row) for row in connection.execute(query).mappings()]
        if shard_count > 1:
            rows = [
                row
                for row in rows
                if int.from_bytes(
                    hashlib.sha1(str(row["source_path"]).encode("utf-8")).digest()[:8],
                    "big",
                )
                % shard_count
                == shard_index
            ]
        if not replace:
            candidate_ids = sorted(str(row["curve_id"]) for row in rows)
            existing_ids: dict[str, set[str]] = {}
            for point_table in (raw_gate_points, aligned_gate_points):
                table_ids: set[str] = set()
                for start in range(0, len(candidate_ids), 1_000):
                    batch_ids = candidate_ids[start : start + 1_000]
                    table_ids.update(
                        connection.scalars(
                            select(point_table.c.curve_id).where(
                                point_table.c.curve_id.in_(batch_ids),
                                point_table.c.point_index == 0,
                            )
                        )
                    )
                existing_ids[point_table.name] = table_ids
            raw_curve_ids = existing_ids[raw_gate_points.name]
            aligned_curve_ids = existing_ids[aligned_gate_points.name]
            rows = [
                row
                for row in rows
                if row["curve_id"] not in raw_curve_ids
                or row["curve_id"] not in aligned_curve_ids
            ]
    if limit is not None:
        rows = rows[:limit]

    summary = {
        "dry_run": dry_run,
        "replace": replace,
        "source": str(source),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "candidate_curves": len(rows),
        "source_files_parsed": 0,
        "curves_backfilled": 0,
        "raw_gate_points": 0,
        "aligned_gate_points": 0,
        "missing_files": 0,
        "parse_errors": 0,
        "missing_tables": 0,
        "missing_columns": 0,
        "missing_segments": 0,
        "alignment_errors": 0,
    }

    parsed_source_path: Path | None = None
    parsed_tables: dict[str, Any] = {}
    raw_batch: list[dict[str, Any]] = []
    aligned_batch: list[dict[str, Any]] = []
    replace_ids: set[str] = set()

    for row_index, row in enumerate(rows, start=1):
        curve_id = str(row["curve_id"])
        relative_path = Path(str(row["source_path"]))
        source_path = source / relative_path
        if source_path != parsed_source_path:
            parsed_source_path = source_path
            parsed_tables = {}
            if not source_path.is_file():
                summary["missing_files"] += 1
                continue
            try:
                parsed = parse_source(source_path, max_xml_mb=max_xml_mb)
            except (OSError, ValueError):
                summary["parse_errors"] += 1
                continue
            parsed_tables = {table.table_name: table for table in parsed}
            summary["source_files_parsed"] += 1
        table = parsed_tables.get(str(row["table_name"]))
        if table is None:
            summary["missing_tables"] += 1
            continue

        voltage_column = str(row["voltage_column"])
        current_column = str(row["current_column"])
        gate_current_column = str(row["gate_current_column"])
        if any(
            column not in table.frame.columns
            for column in (voltage_column, current_column, gate_current_column)
        ):
            summary["missing_columns"] += 1
            continue

        try:
            cleaned = _clean_pair(table.frame, voltage_column, current_column)
            gate_cleaned = _clean_pair(table.frame, voltage_column, gate_current_column)
            segments = _segment_sweeps(
                cleaned["voltage_v"].to_numpy(dtype=float),
                cleaned["current_a"].to_numpy(dtype=float),
            )
            gate_segments = _segment_sweeps(
                gate_cleaned["voltage_v"].to_numpy(dtype=float),
                gate_cleaned["current_a"].to_numpy(dtype=float),
            )
        except (KeyError, ValueError):
            summary["missing_columns"] += 1
            continue

        segment_index = int(row["segment_index"]) - 1
        if (
            segment_index < 0
            or segment_index >= len(segments)
            or segment_index >= len(gate_segments)
        ):
            summary["missing_segments"] += 1
            continue
        segment = segments[segment_index]
        gate_segment = gate_segments[segment_index]
        try:
            physical_grid, _, _ = _align_segment(
                np.asarray(segment.voltage, dtype=float),
                np.asarray(segment.current, dtype=float),
            )
            aligned_log_gate, aligned_gate_current = _align_current_to_grid(
                np.asarray(gate_segment.voltage, dtype=float),
                np.asarray(gate_segment.current, dtype=float),
                physical_grid,
            )
        except ValueError:
            summary["alignment_errors"] += 1
            continue

        replace_ids.add(curve_id)
        for point_index, (voltage_v, current_a) in enumerate(
            zip(gate_segment.voltage, gate_segment.current, strict=True)
        ):
            raw_batch.append(
                {
                    "curve_id": curve_id,
                    "point_index": point_index,
                    "voltage_v": float(voltage_v),
                    "current_a": float(current_a),
                }
            )
        for point_index, (x_norm, voltage_v, log_ig, abs_ig) in enumerate(
            zip(
                X_NORM_GRID,
                physical_grid,
                aligned_log_gate,
                aligned_gate_current,
                strict=True,
            )
        ):
            aligned_batch.append(
                {
                    "curve_id": curve_id,
                    "point_index": point_index,
                    "x_norm": float(x_norm),
                    "voltage_v": float(voltage_v),
                    "log10_abs_ig": float(log_ig),
                    "abs_ig_a": float(abs_ig),
                }
            )
        summary["curves_backfilled"] += 1
        summary["raw_gate_points"] += len(gate_segment.voltage)
        summary["aligned_gate_points"] += len(physical_grid)

        if len(raw_batch) + len(aligned_batch) >= BATCH_SIZE:
            if dry_run:
                replace_ids.clear()
                raw_batch.clear()
                aligned_batch.clear()
            else:
                _flush_gate_backfill_batch(engine, replace_ids, raw_batch, aligned_batch)
        if row_index % 500 == 0:
            print(
                f"Checked {row_index}/{len(rows)} gate candidates; "
                f"backfilled {summary['curves_backfilled']}",
                flush=True,
            )

    if not dry_run:
        _flush_gate_backfill_batch(engine, replace_ids, raw_batch, aligned_batch)
        summary["applied"] = True
    return summary


def backfill_b1500_metadata(
    source: Path,
    database_url: str | None = None,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    max_xml_mb: float = 128.0,
) -> dict[str, Any]:
    """Refresh Ig and polarity metadata without rewriting raw/aligned point data."""

    source = source.resolve()
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    _ = max_xml_mb
    engine = create_database_engine(database_url)
    create_schema(engine)
    with engine.connect() as connection:
        config_rows = [
            dict(row)
            for row in connection.execute(
                select(
                    test_configs.c.id,
                    test_configs.c.current_column,
                    test_configs.c.gate_current_column,
                    test_configs.c.columns_json,
                )
                .order_by(test_configs.c.id)
                .limit(limit)
                if limit is not None
                else select(
                    test_configs.c.id,
                    test_configs.c.current_column,
                    test_configs.c.gate_current_column,
                    test_configs.c.columns_json,
                ).order_by(test_configs.c.id)
            ).mappings()
        ]
        curve_query = select(curves.c.curve_id, curves.c.polarity).order_by(curves.c.curve_id)
        if limit is not None:
            curve_query = curve_query.limit(limit)
        curve_rows = [dict(row) for row in connection.execute(curve_query).mappings()]
        curve_counts_by_config = dict(
            connection.execute(
                select(curves.c.test_config_id, func.count()).group_by(curves.c.test_config_id)
            ).all()
        )

    gate_updates: dict[int, str | None] = {}
    has_gate_updates: dict[int, int] = {}
    polarity_updates: dict[str, str] = {}

    for row in config_rows:
        columns = [str(column) for column in (row["columns_json"] or [])]
        current_column = str(row["current_column"] or "")
        candidates = [
            column
            for column in columns
            if column != current_column and _is_gate_current(column)
        ]
        next_gate_column = sorted(candidates, key=_gate_current_rank)[0] if candidates else None
        config_id = int(row["id"])
        if row["gate_current_column"] != next_gate_column:
            gate_updates[config_id] = next_gate_column
        has_gate_updates[config_id] = 1 if next_gate_column is not None else 0

    batch_size = 500
    with engine.connect() as connection:
        for batch_start in range(0, len(curve_rows), batch_size):
            batch = curve_rows[batch_start : batch_start + batch_size]
            batch_ids = [row["curve_id"] for row in batch]
            points_by_curve: dict[str, list[tuple[float, float]]] = {
                curve_id: [] for curve_id in batch_ids
            }
            point_rows = connection.execute(
                select(raw_points.c.curve_id, raw_points.c.voltage_v, raw_points.c.current_a)
                .where(raw_points.c.curve_id.in_(batch_ids))
                .order_by(raw_points.c.curve_id, raw_points.c.point_index)
            )
            for curve_id, voltage_v, current_a in point_rows:
                points_by_curve[str(curve_id)].append((float(voltage_v), float(current_a)))
            for row in batch:
                points = points_by_curve.get(row["curve_id"], [])
                if len(points) < 6:
                    continue
                voltage = [point[0] for point in points]
                current = [point[1] for point in points]
                features = analyze_transfer_curve(voltage, current)
                next_polarity = features.polarity
                if (
                    next_polarity in {"n-type", "p-type", "bipolar"}
                    and next_polarity != row["polarity"]
                ):
                    polarity_updates[row["curve_id"]] = next_polarity
            if (batch_start // batch_size + 1) % 10 == 0:
                print(
                    f"Checked {min(batch_start + batch_size, len(curve_rows))}/"
                    f"{len(curve_rows)} curves; "
                    f"polarity updates {len(polarity_updates)}",
                    flush=True,
                )

    has_gate_curve_count = int(
        sum(
            curve_counts_by_config.get(config_id, 0)
            for config_id, value in has_gate_updates.items()
            if value
        )
    )
    summary = {
        "dry_run": dry_run,
        "source": str(source),
        "configs_checked": len(config_rows),
        "curves_checked": len(curve_rows),
        "gate_config_updates": len(gate_updates),
        "has_gate_config_updates": len(has_gate_updates),
        "has_gate_curves_after_update": has_gate_curve_count,
        "polarity_updates": len(polarity_updates),
    }
    if dry_run:
        return summary

    with engine.begin() as connection:
        for config_id, gate_column in gate_updates.items():
            connection.execute(
                update(test_configs)
                .where(test_configs.c.id == config_id)
                .values(gate_current_column=gate_column)
            )
        for config_id, has_gate in has_gate_updates.items():
            connection.execute(
                update(curves)
                .where(curves.c.test_config_id == config_id)
                .values(has_gate_current=has_gate)
            )
        for curve_id, polarity in polarity_updates.items():
            connection.execute(
                update(curves).where(curves.c.curve_id == curve_id).values(polarity=polarity)
            )
    summary["applied"] = True
    return summary


def database_status(database_url: str | None = None) -> dict[str, Any]:
    url = resolve_database_url(database_url)
    engine = create_database_engine(url)
    create_schema(engine)
    with engine.connect() as connection:
        curve_count = connection.scalar(select(func.count()).select_from(curves)) or 0
        raw_count = connection.scalar(select(func.sum(curves.c.rows_clean))) or 0
        if engine.dialect.name in {"mysql", "mariadb"}:
            aligned_count = int(curve_count) * int(X_NORM_GRID.size)
        else:
            aligned_count = connection.scalar(select(func.count()).select_from(aligned_points)) or 0
        curves_with_ig = connection.scalar(
            select(func.count()).select_from(curves).where(curves.c.has_gate_current == 1)
        ) or 0
        if engine.dialect.name in {"mysql", "mariadb"}:
            gate_count = connection.scalar(
                select(func.sum(curves.c.rows_clean)).where(curves.c.has_gate_current == 1)
            ) or 0
            aligned_gate_count = int(curves_with_ig) * int(X_NORM_GRID.size)
        else:
            gate_count = connection.scalar(select(func.count()).select_from(raw_gate_points)) or 0
            aligned_gate_count = (
                connection.scalar(select(func.count()).select_from(aligned_gate_points)) or 0
            )
        rejected_count = connection.scalar(select(func.count()).select_from(rejected_entries)) or 0
        source_count = connection.scalar(select(func.count()).select_from(source_files)) or 0
        belonger_counts: dict[str, int] = {}
        for source_path in connection.scalars(select(source_files.c.source_path)):
            belonger = _source_belonger(str(source_path))
            belonger_counts[belonger] = belonger_counts.get(belonger, 0) + 1
        polarity = dict(
            connection.execute(
                select(curves.c.polarity, func.count()).group_by(curves.c.polarity)
            ).all()
        )
        source_kinds = dict(
            connection.execute(
                select(test_configs.c.source_kind, func.count())
                .select_from(
                    curves.join(test_configs, curves.c.test_config_id == test_configs.c.id)
                )
                .group_by(test_configs.c.source_kind)
            ).all()
        )
    return {
        "configured": True,
        "database_url": _redact_url(url),
        "source_files": int(source_count),
        "curves": int(curve_count),
        "raw_points": int(raw_count),
        "aligned_points": int(aligned_count),
        "gate_points": int(gate_count),
        "aligned_gate_points": int(aligned_gate_count),
        "curves_with_ig": int(curves_with_ig),
        "rejected_entries": int(rejected_count),
        "polarity_counts": polarity,
        "source_kind_counts": source_kinds,
        "belonger_counts": belonger_counts,
    }


def _redact_url(url: str) -> str:
    if "@" not in url:
        return url
    prefix, suffix = url.rsplit("@", 1)
    scheme = prefix.split("://", 1)[0] if "://" in prefix else "mysql"
    return f"{scheme}://***@{suffix}"


def _source_belonger(source_path: str) -> str:
    normalized = str(source_path).replace("\\", "/").strip("/")
    if not normalized:
        return "Unknown"
    return normalized.split("/", 1)[0] or "Unknown"


def _curve_filter_conditions(filters: dict[str, Any]) -> list[Any]:
    conditions: list[Any] = []
    if filters.get("polarity"):
        conditions.append(curves.c.polarity == filters["polarity"])
    if filters.get("direction"):
        conditions.append(curves.c.direction == filters["direction"])
    if filters.get("source_kind"):
        conditions.append(test_configs.c.source_kind == filters["source_kind"])
    if filters.get("source_search"):
        pattern = f"%{filters['source_search']}%"
        conditions.append(
            or_(
                source_files.c.source_path.like(pattern),
                test_configs.c.source_kind.like(pattern),
                test_configs.c.setup_title.like(pattern),
                test_configs.c.primitive_test.like(pattern),
            )
        )
    if filters.get("has_gate_current") is not None:
        value = str(filters["has_gate_current"]).lower()
        if value in {"1", "true", "yes"}:
            conditions.append(curves.c.has_gate_current == 1)
        elif value in {"0", "false", "no"}:
            conditions.append(curves.c.has_gate_current == 0)
    if filters.get("hysteresis_available") is not None:
        value = str(filters["hysteresis_available"]).lower()
        if value in {"1", "true", "yes", "paired"}:
            conditions.append(curves.c.hysteresis_v.is_not(None))
        elif value in {"0", "false", "no", "single"}:
            conditions.append(curves.c.hysteresis_v.is_(None))
    if filters.get("date_from"):
        conditions.append(curves.c.test_time >= filters["date_from"])
    if filters.get("date_to"):
        conditions.append(curves.c.test_time <= filters["date_to"])
    for key, column in (
        ("ion", curves.c.ion),
        ("ioff", curves.c.ioff),
        ("ion_ioff_ratio", curves.c.ion_ioff_ratio),
        ("vth", curves.c.vth),
        ("ss_mv_dec", curves.c.ss_mv_dec),
    ):
        minimum = filters.get(f"{key}_min")
        maximum = filters.get(f"{key}_max")
        if minimum is not None:
            conditions.append(column >= minimum)
        if maximum is not None:
            conditions.append(column <= maximum)
    return conditions


def list_curves(
    database_url: str | None = None,
    *,
    limit: int = 100,
    offset: int = 0,
    order_by: str = "modified_at_desc",
    **filters: Any,
) -> dict[str, Any]:
    engine = create_database_engine(database_url)
    create_schema(engine)
    joined = curves.join(
        source_files,
        curves.c.source_file_id == source_files.c.id,
    ).join(
        test_configs,
        curves.c.test_config_id == test_configs.c.id,
    )
    conditions = _curve_filter_conditions(filters)
    where_clause = and_(*conditions) if conditions else None
    order_column = {
        "modified_at_asc": curves.c.test_time.asc(),
        "ion_desc": curves.c.ion.desc(),
        "ion_asc": curves.c.ion.asc(),
        "ratio_desc": curves.c.ion_ioff_ratio.desc(),
        "vth_asc": curves.c.vth.asc(),
        "vth_desc": curves.c.vth.desc(),
    }.get(order_by, curves.c.test_time.desc())
    base = select(
        curves.c.curve_id,
        curves.c.direction,
        curves.c.sweep_pair_id,
        curves.c.test_time,
        curves.c.rows_clean,
        curves.c.voltage_min_v,
        curves.c.voltage_max_v,
        curves.c.ion,
        curves.c.ioff,
        curves.c.ion_ioff_ratio,
        curves.c.polarity,
        curves.c.has_gate_current,
        curves.c.vth,
        curves.c.ss_mv_dec,
        curves.c.gm_max,
        curves.c.noise_log_sigma,
        curves.c.hysteresis_v,
        source_files.c.source_path,
        source_files.c.modified_at,
        test_configs.c.source_kind,
        test_configs.c.setup_title,
        test_configs.c.primitive_test,
        test_configs.c.voltage_column,
        test_configs.c.current_column,
        test_configs.c.gate_current_column,
    ).select_from(joined)
    count_from = joined if filters.get("source_kind") or filters.get("source_search") else curves
    count_query = select(func.count()).select_from(count_from)
    if where_clause is not None:
        base = base.where(where_clause)
        count_query = count_query.where(where_clause)
    with engine.connect() as connection:
        total = connection.scalar(count_query) or 0
        rows = connection.execute(
            base.order_by(order_column, curves.c.curve_id).limit(limit).offset(offset)
        ).mappings()
        items = [dict(row) for row in rows]
    for item in items:
        if item["modified_at"] is not None:
            item["modified_at"] = item["modified_at"].isoformat()
        if item["test_time"] is not None:
            item["test_time"] = item["test_time"].isoformat()
        ratio = item.get("ion_ioff_ratio")
        item["log_ratio"] = float(np.log10(ratio)) if ratio and ratio > 0 else None
        item["has_gate_current"] = bool(item.get("has_gate_current"))
    return {"total": int(total), "limit": limit, "offset": offset, "items": items}


MATRIX_MATCH_COLUMNS = {
    "target_ion": curves.c.ion,
    "target_ioff": curves.c.ioff,
    "ion_ioff_ratio": curves.c.ion_ioff_ratio,
    "target_vth": curves.c.vth,
    "target_ss_mv_dec": curves.c.ss_mv_dec,
    "hysteresis_v": curves.c.hysteresis_v,
}


def _matrix_score(row: dict[str, Any], targets: dict[str, float]) -> dict[str, Any] | None:
    score = 0.0
    weight = 0
    features: list[str] = []
    for key, target in targets.items():
        if key not in MATRIX_MATCH_COLUMNS or target is None:
            continue
        value = row.get(key)
        if value is None:
            continue
        try:
            measured = float(value)
            requested = float(target)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(measured) or not np.isfinite(requested):
            continue
        if key in {"target_ion", "target_ioff", "ion_ioff_ratio"}:
            if measured <= 0 or requested <= 0:
                continue
            score += abs(np.log10(measured) - np.log10(requested))
        elif key == "target_ss_mv_dec":
            score += abs(measured - requested) / max(abs(requested) * 0.2, 20.0)
        else:
            score += abs(measured - requested) / max(abs(requested) * 0.2, 0.25)
        weight += 1
        features.append(key)
    if weight == 0:
        return None
    return {"score": float(score / weight), "features": features}


def match_matrix_sites(
    database_url: str | None = None,
    *,
    site_targets: list[dict[str, Any]],
    filters: dict[str, Any] | None = None,
    duplicate_mode: str = "allow",
    max_candidates: int = 50_000,
) -> list[dict[str, Any]]:
    engine = create_database_engine(database_url)
    create_schema(engine)
    joined = curves.join(
        source_files,
        curves.c.source_file_id == source_files.c.id,
    ).join(
        test_configs,
        curves.c.test_config_id == test_configs.c.id,
    )
    conditions = _curve_filter_conditions(filters or {})
    query = (
        select(
            curves.c.curve_id,
            curves.c.polarity,
            curves.c.direction,
            curves.c.ion.label("target_ion"),
            curves.c.ioff.label("target_ioff"),
            curves.c.ion_ioff_ratio.label("ion_ioff_ratio"),
            curves.c.vth.label("target_vth"),
            curves.c.ss_mv_dec.label("target_ss_mv_dec"),
            curves.c.hysteresis_v.label("hysteresis_v"),
            source_files.c.source_path,
            test_configs.c.source_kind,
        )
        .select_from(joined)
        .order_by(curves.c.curve_id)
        .limit(max_candidates)
    )
    if conditions:
        query = query.where(and_(*conditions))
    with engine.connect() as connection:
        candidates = [dict(row) for row in connection.execute(query).mappings()]

    used_curve_ids: set[str] = set()
    assignments: list[dict[str, Any]] = []
    for site in site_targets:
        targets = {
            key: value
            for key, value in (site.get("parameters") or {}).items()
            if key in MATRIX_MATCH_COLUMNS and value is not None
        }
        scored = []
        for row in candidates:
            curve_id = str(row["curve_id"])
            score_result = _matrix_score(row, targets)
            if score_result is None:
                continue
            if duplicate_mode in {"avoid", "generate_on_duplicate"} and curve_id in used_curve_ids:
                continue
            scored.append((float(score_result["score"]), row, score_result["features"]))
        if not scored and duplicate_mode == "generate_on_duplicate":
            assignments.append({**site, "source": "generated", "reason": "duplicate_fallback"})
            continue
        if not scored and duplicate_mode == "avoid":
            assignments.append({**site, "source": "unmatched", "reason": "no_unique_match"})
            continue
        if not scored:
            assignments.append({**site, "source": "unmatched", "reason": "no_match"})
            continue
        score, row, score_features = min(
            scored,
            key=lambda item: (item[0], str(item[1]["curve_id"])),
        )
        curve_id = str(row["curve_id"])
        reused = curve_id in used_curve_ids
        used_curve_ids.add(curve_id)
        assignments.append(
            {
                **site,
                "source": "database",
                "curve_id": curve_id,
                "score": score,
                "score_features": score_features,
                "reused": reused,
                "matched": {
                    key: row.get(key)
                    for key in [
                        "target_ion",
                        "target_ioff",
                        "ion_ioff_ratio",
                        "target_vth",
                        "target_ss_mv_dec",
                        "hysteresis_v",
                    ]
                },
                "polarity": row.get("polarity"),
                "direction": row.get("direction"),
                "source_kind": row.get("source_kind"),
                "source_path": row.get("source_path"),
            }
        )
    return assignments


def calendar_curves(
    database_url: str | None = None,
    *,
    limit: int = 10_000,
    **filters: Any,
) -> dict[str, Any]:
    engine = create_database_engine(database_url)
    create_schema(engine)
    joined = curves.join(
        source_files,
        curves.c.source_file_id == source_files.c.id,
    ).join(
        test_configs,
        curves.c.test_config_id == test_configs.c.id,
    )
    conditions = _curve_filter_conditions(filters)
    query = (
        select(
            curves.c.curve_id,
            curves.c.test_time,
            curves.c.polarity,
            curves.c.direction,
            curves.c.hysteresis_v,
            curves.c.sweep_pair_id,
            curves.c.vth,
            curves.c.ion,
            curves.c.ioff,
            source_files.c.source_path,
            test_configs.c.source_kind,
            test_configs.c.setup_title,
        )
        .select_from(joined)
        .where(curves.c.test_time.is_not(None))
        .order_by(curves.c.test_time, curves.c.curve_id)
        .limit(limit + 1)
    )
    if conditions:
        query = query.where(and_(*conditions))
    with engine.connect() as connection:
        rows = [dict(row) for row in connection.execute(query).mappings()]
    truncated = len(rows) > limit
    items = rows[:limit]
    day_counts: dict[str, int] = {}
    for item in items:
        test_time = item["test_time"]
        item["test_time"] = test_time.isoformat()
        day = test_time.date().isoformat()
        day_counts[day] = day_counts.get(day, 0) + 1
    return {
        "items": items,
        "day_counts": day_counts,
        "truncated": truncated,
        "limit": limit,
    }


def get_curve_detail(database_url: str | None, curve_id: str) -> dict[str, Any] | None:
    engine = create_database_engine(database_url)
    create_schema(engine)
    joined = curves.join(
        source_files,
        curves.c.source_file_id == source_files.c.id,
    ).join(
        test_configs,
        curves.c.test_config_id == test_configs.c.id,
    )
    query = (
        select(
            curves,
            source_files.c.source_path,
            source_files.c.extension,
            source_files.c.size_bytes,
            source_files.c.modified_at,
            test_configs.c.source_kind,
            test_configs.c.table_name,
            test_configs.c.setup_title,
            test_configs.c.primitive_test,
            test_configs.c.x_axis_data,
            test_configs.c.voltage_column,
            test_configs.c.current_column,
            test_configs.c.gate_current_column,
            test_configs.c.classification_reason,
            test_configs.c.classification_confidence,
            test_configs.c.columns_json,
            test_configs.c.metadata_json,
        )
        .select_from(joined)
        .where(curves.c.curve_id == curve_id)
    )
    with engine.connect() as connection:
        row = connection.execute(query).mappings().first()
        if row is None:
            return None
        raw = [
            dict(point)
            for point in connection.execute(
                select(
                    raw_points.c.point_index,
                    raw_points.c.voltage_v,
                    raw_points.c.current_a,
                )
                .where(raw_points.c.curve_id == curve_id)
                .order_by(raw_points.c.point_index)
            ).mappings()
        ]
        aligned = [
            dict(point)
            for point in connection.execute(
                select(
                    aligned_points.c.point_index,
                    aligned_points.c.x_norm,
                    aligned_points.c.voltage_v,
                    aligned_points.c.log10_abs_id,
                    aligned_points.c.abs_id_a,
                )
                .where(aligned_points.c.curve_id == curve_id)
                .order_by(aligned_points.c.point_index)
            ).mappings()
        ]
        gate = [
            dict(point)
            for point in connection.execute(
                select(
                    raw_gate_points.c.point_index,
                    raw_gate_points.c.voltage_v,
                    raw_gate_points.c.current_a,
                )
                .where(raw_gate_points.c.curve_id == curve_id)
                .order_by(raw_gate_points.c.point_index)
            ).mappings()
        ]
        aligned_gate = [
            dict(point)
            for point in connection.execute(
                select(
                    aligned_gate_points.c.point_index,
                    aligned_gate_points.c.x_norm,
                    aligned_gate_points.c.voltage_v,
                    aligned_gate_points.c.log10_abs_ig,
                    aligned_gate_points.c.abs_ig_a,
                )
                .where(aligned_gate_points.c.curve_id == curve_id)
                .order_by(aligned_gate_points.c.point_index)
            ).mappings()
        ]
    payload = dict(row)
    if payload["modified_at"] is not None:
        payload["modified_at"] = payload["modified_at"].isoformat()
    if payload["test_time"] is not None:
        payload["test_time"] = payload["test_time"].isoformat()
    ratio = payload.get("ion_ioff_ratio")
    payload["log_ratio"] = float(np.log10(ratio)) if ratio and ratio > 0 else None
    payload["has_gate_current"] = bool(payload.get("has_gate_current"))
    payload["raw_points"] = raw
    payload["gate_points"] = gate
    payload["aligned_points"] = aligned
    payload["aligned_gate_points"] = aligned_gate
    return payload


def get_curve_previews(
    database_url: str | None,
    curve_ids: list[str],
) -> list[dict[str, Any]]:
    ids = list(dict.fromkeys(curve_ids))[:64]
    if not ids:
        return []
    engine = create_database_engine(database_url)
    create_schema(engine)
    summary_query = (
        select(
            curves.c.curve_id,
            curves.c.polarity,
            curves.c.direction,
            source_files.c.source_path,
        )
        .select_from(curves.join(source_files))
        .where(curves.c.curve_id.in_(ids))
    )
    points_query = (
        select(
            raw_points.c.curve_id,
            raw_points.c.point_index,
            raw_points.c.voltage_v,
            raw_points.c.current_a,
        )
        .where(raw_points.c.curve_id.in_(ids))
        .order_by(raw_points.c.curve_id, raw_points.c.point_index)
    )
    with engine.connect() as connection:
        summaries = {
            row["curve_id"]: dict(row)
            for row in connection.execute(summary_query).mappings()
        }
        for summary in summaries.values():
            summary["raw_points"] = []
        for point in connection.execute(points_query).mappings():
            summary = summaries.get(point["curve_id"])
            if summary is None:
                continue
            summary["raw_points"].append(
                {
                    "point_index": point["point_index"],
                    "voltage_v": point["voltage_v"],
                    "current_a": point["current_a"],
                }
            )
    return [summaries[curve_id] for curve_id in ids if curve_id in summaries]


def _selected_curve_conditions(
    curve_ids: list[str] | None,
    filters: dict[str, Any] | None,
) -> list[Any]:
    if curve_ids:
        return [curves.c.curve_id.in_(curve_ids)]
    return _curve_filter_conditions(filters or {})


def _analysis_metric(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "max": max(finite),
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.pstdev(finite) if len(finite) > 1 else 0.0,
    }


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _safe_log10(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None or number <= 0:
        return None
    return float(np.log10(number))


def _analysis_sample(row: dict[str, Any]) -> dict[str, Any]:
    voltage_span = _safe_float(row["voltage_max_v"] - row["voltage_min_v"])
    return {
        "curve_id": row["curve_id"],
        "polarity": row["polarity"],
        "direction": row["direction"],
        "source_kind": row["source_kind"],
        "source_path": row["source_path"],
        "has_ig": bool(row["has_gate_current"]),
        "rows_clean": _safe_float(row["rows_clean"]),
        "voltage_span": voltage_span,
        "ion": _safe_float(row["ion"]),
        "ioff": _safe_float(row["ioff"]),
        "ion_ioff_ratio": _safe_float(row["ion_ioff_ratio"]),
        "logIon": _safe_log10(row["ion"]),
        "logIoff": _safe_log10(row["ioff"]),
        "logRatio": _safe_log10(row["ion_ioff_ratio"]),
        "vth": _safe_float(row["vth"]),
        "ss_mv_dec": _safe_float(row["ss_mv_dec"]),
        "gm_max": _safe_float(row["gm_max"]),
        "logGm": _safe_log10(row["gm_max"]),
        "noise_log_sigma": _safe_float(row["noise_log_sigma"]),
        "ambipolar_strength": _safe_float(row["ambipolar_strength"]),
        "hysteresis_v": _safe_float(row["hysteresis_v"]),
        "test_time": (
            row["test_time"].isoformat()
            if isinstance(row.get("test_time"), datetime)
            else row.get("test_time")
        ),
    }


def _analysis_display_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(samples) <= ANALYSIS_SAMPLE_LIMIT:
        return samples
    step = len(samples) / ANALYSIS_SAMPLE_LIMIT
    return [
        samples[min(int(index * step), len(samples) - 1)]
        for index in range(ANALYSIS_SAMPLE_LIMIT)
    ]


def _analysis_histogram(values: list[float]) -> list[dict[str, Any]]:
    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return []
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 0.5)
        minimum -= padding
        maximum += padding
    counts, edges = np.histogram(finite, bins=ANALYSIS_HISTOGRAM_BINS, range=(minimum, maximum))
    return [
        {
            "start": float(edges[index]),
            "end": float(edges[index + 1]),
            "center": float((edges[index] + edges[index + 1]) / 2),
            "count": int(count),
        }
        for index, count in enumerate(counts)
    ]


def _analysis_metric_and_histogram(
    values: list[float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    finite = np.asarray([float(value) for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return (
            {"count": 0, "min": None, "max": None, "mean": None, "median": None, "std": None},
            [],
        )
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    mean = float(np.mean(finite))
    median = float(np.median(finite))
    std = float(np.std(finite))
    hist_minimum = minimum
    hist_maximum = maximum
    if hist_minimum == hist_maximum:
        padding = max(abs(hist_minimum) * 0.05, 0.5)
        hist_minimum -= padding
        hist_maximum += padding
    counts, edges = np.histogram(
        finite,
        bins=ANALYSIS_HISTOGRAM_BINS,
        range=(hist_minimum, hist_maximum),
    )
    return (
        {
            "count": int(finite.size),
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "median": median,
            "std": std,
        },
        [
            {
                "start": float(edges[index]),
                "end": float(edges[index + 1]),
                "center": float((edges[index] + edges[index + 1]) / 2),
                "count": int(count),
            }
            for index, count in enumerate(counts)
        ],
    )


def _analysis_correlations(samples: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: list[list[float | None]] = []
    counts: list[list[int]] = []
    pairs: list[dict[str, Any]] = []
    feature_values = {
        feature: np.asarray(
            [
                np.nan if sample[feature] is None else float(sample[feature])
                for sample in samples
            ],
            dtype=float,
        )
        for feature in ANALYSIS_NUMERIC_FEATURES
    }
    for x_feature in ANALYSIS_NUMERIC_FEATURES:
        row_values: list[float | None] = []
        row_counts: list[int] = []
        for y_feature in ANALYSIS_NUMERIC_FEATURES:
            x_all = feature_values[x_feature]
            y_all = feature_values[y_feature]
            finite = np.isfinite(x_all) & np.isfinite(y_all)
            count = int(np.count_nonzero(finite))
            row_counts.append(count)
            if count < 2:
                row_values.append(None)
                continue
            x_values = x_all[finite]
            y_values = y_all[finite]
            if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
                coefficient = None
            else:
                coefficient = float(np.corrcoef(x_values, y_values)[0, 1])
            row_values.append(coefficient)
            if x_feature < y_feature and coefficient is not None:
                pairs.append(
                    {
                        "x": x_feature,
                        "y": y_feature,
                        "r": coefficient,
                        "count": count,
                    }
                )
        matrix.append(row_values)
        counts.append(row_counts)
    pairs.sort(key=lambda pair: abs(pair["r"]), reverse=True)
    return {
        "features": list(ANALYSIS_NUMERIC_FEATURES),
        "matrix": matrix,
        "counts": counts,
        "strongest": pairs[:12],
    }


def _analysis_pca(samples: list[dict[str, Any]]) -> dict[str, Any]:
    usable_features: list[str] = []
    columns: list[np.ndarray] = []
    for feature in ANALYSIS_NUMERIC_FEATURES:
        values = np.asarray(
            [np.nan if sample[feature] is None else float(sample[feature]) for sample in samples],
            dtype=float,
        )
        finite = values[np.isfinite(values)]
        if finite.size < 2 or float(np.std(finite)) == 0.0:
            continue
        median = float(np.median(finite))
        values = np.where(np.isfinite(values), values, median)
        columns.append(values)
        usable_features.append(feature)
    if len(samples) < 2 or len(columns) < 2:
        return {
            "features": usable_features,
            "components": [],
            "points": [],
            "sampled": len(samples) > ANALYSIS_SAMPLE_LIMIT,
        }

    data = np.column_stack(columns)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    standardized = (data - means) / np.where(stds == 0, 1, stds)
    _, singular_values, vt = np.linalg.svd(standardized, full_matrices=False)
    component_count = min(
        ANALYSIS_PCA_MAX_COMPONENTS,
        vt.shape[0],
        standardized.shape[0] - 1,
        standardized.shape[1],
    )
    if component_count < 1:
        return {
            "features": usable_features,
            "components": [],
            "points": [],
            "sampled": len(samples) > ANALYSIS_SAMPLE_LIMIT,
        }

    scores = standardized @ vt[:component_count].T
    variances = (singular_values**2) / max(standardized.shape[0] - 1, 1)
    total_variance = float(np.sum(variances))
    explained = (
        variances[:component_count] / total_variance
        if total_variance > 0
        else np.zeros(component_count)
    )
    components = [
        {
            "name": f"PC{index + 1}",
            "explained_variance_ratio": float(explained[index]),
            "loadings": {
                feature: float(vt[index, feature_index])
                for feature_index, feature in enumerate(usable_features)
            },
        }
        for index in range(component_count)
    ]
    return {
        "features": usable_features,
        "components": components,
        "points": [
            {
                "curve_id": sample["curve_id"],
                "polarity": sample["polarity"],
                "direction": sample["direction"],
                "source_kind": sample["source_kind"],
                "scores": [float(value) for value in scores[row_index, :component_count]],
            }
            for row_index, sample in enumerate(samples)
        ],
        "sampled": len(samples) > ANALYSIS_SAMPLE_LIMIT,
    }


def analyze_curves(
    database_url: str | None = None,
    *,
    curve_ids: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    def report(stage: str, progress_fraction: float, message: str) -> None:
        if progress is None:
            return
        progress(
            {
                "stage": stage,
                "progress_fraction": progress_fraction,
                "message": message,
            }
        )

    report("loading_selection", 0.02, "Opening the database and reading the selected curves")
    engine = create_database_engine(database_url)
    create_schema(engine)
    joined = curves.join(
        source_files,
        curves.c.source_file_id == source_files.c.id,
    ).join(
        test_configs,
        curves.c.test_config_id == test_configs.c.id,
    )
    conditions = _selected_curve_conditions(curve_ids, filters)
    query = select(
        curves.c.curve_id,
        curves.c.direction,
        curves.c.rows_clean,
        curves.c.voltage_min_v,
        curves.c.voltage_max_v,
        curves.c.ion,
        curves.c.ioff,
        curves.c.ion_ioff_ratio,
        curves.c.polarity,
        curves.c.has_gate_current,
        curves.c.vth,
        curves.c.ss_mv_dec,
        curves.c.gm_max,
        curves.c.noise_log_sigma,
        curves.c.ambipolar_strength,
        curves.c.hysteresis_v,
        curves.c.test_time,
        source_files.c.source_path,
        test_configs.c.source_kind,
    ).select_from(joined)
    if conditions:
        query = query.where(and_(*conditions))
    with engine.connect() as connection:
        rows = [dict(row) for row in connection.execute(query).mappings()]
        report(
            "loading_selection",
            0.22,
            f"Loaded {len(rows):,} curves from the database selection",
        )
        gate_points = 0
        aligned_gate_point_count = 0
        if rows:
            if engine.dialect.name in {"mysql", "mariadb"}:
                gate_points = int(
                    sum(row["rows_clean"] for row in rows if row["has_gate_current"])
                )
                aligned_gate_point_count = int(
                    sum(1 for row in rows if row["has_gate_current"]) * X_NORM_GRID.size
                )
            else:
                ids = [row["curve_id"] for row in rows]
                gate_points = int(
                    connection.scalar(
                        select(func.count()).select_from(raw_gate_points).where(
                            raw_gate_points.c.curve_id.in_(ids)
                        )
                    )
                    or 0
                )
                aligned_gate_point_count = int(
                    connection.scalar(
                        select(func.count()).select_from(aligned_gate_points).where(
                            aligned_gate_points.c.curve_id.in_(ids)
                        )
                    )
                    or 0
                )
    report("building_samples", 0.28, "Converting selected curves into analysis samples")
    all_samples: list[dict[str, Any]] = []
    metric_sources: dict[str, list[float]] = {
        metric: [] for metric in ANALYSIS_METRIC_SOURCE_KEYS
    }
    polarity_counts: dict[str, int] = {}
    direction_counts: dict[str, int] = {}
    source_kind_counts: dict[str, int] = {}
    curves_with_ig = 0
    hysteresis_paired = 0
    unique_sources: set[str] = set()
    raw_points = 0
    sample_count = len(rows)
    progress_step = max(250, min(2_000, sample_count // 16 if sample_count else 250))
    for index, row in enumerate(rows, start=1):
        sample = _analysis_sample(row)
        all_samples.append(sample)
        for metric, source_key in ANALYSIS_METRIC_SOURCE_KEYS.items():
            value = sample[source_key]
            if value is not None:
                metric_sources[metric].append(value)
        polarity = str(row["polarity"])
        direction = str(row["direction"])
        source_kind = str(row["source_kind"])
        polarity_counts[polarity] = polarity_counts.get(polarity, 0) + 1
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        source_kind_counts[source_kind] = source_kind_counts.get(source_kind, 0) + 1
        if row["has_gate_current"]:
            curves_with_ig += 1
        if row["hysteresis_v"] is not None:
            hysteresis_paired += 1
        unique_sources.add(str(row["source_path"]))
        raw_points += int(row["rows_clean"] or 0)
        if index == sample_count or index % progress_step == 0:
            sample_progress = index / max(sample_count, 1)
            report(
                "building_samples",
                0.28 + 0.28 * sample_progress,
                f"Prepared {index:,}/{sample_count:,} analysis samples",
            )
    display_samples = _analysis_display_samples(all_samples)
    report(
        "building_metrics",
        0.6,
        f"Building summary metrics for {len(display_samples):,} plotted samples",
    )
    spans = metric_sources["voltage_span"]
    report("computing_correlations", 0.78, "Computing correlation trends across the sampled curves")
    correlations = _analysis_correlations(display_samples)
    report("computing_pca", 0.88, "Computing PCA structure and cluster-ready coordinates")
    pca = _analysis_pca(display_samples)
    pca["sampled"] = len(display_samples) < len(all_samples)
    report("finalizing", 0.92, "Computing metric summaries and distributions")
    metrics: dict[str, dict[str, Any]] = {}
    distributions: dict[str, list[dict[str, Any]]] = {}
    for metric, values in metric_sources.items():
        metric_stats, histogram = _analysis_metric_and_histogram(values)
        metrics[metric] = metric_stats
        distributions[metric] = histogram
    report("finalizing", 0.97, "Preparing categorical summaries and processing totals")
    report("finalizing", 0.995, "Packaging the analysis response")
    return {
        "count": len(rows),
        "selected_mode": "ids" if curve_ids else "filters",
        "sample_count": len(display_samples),
        "sample_limit": ANALYSIS_SAMPLE_LIMIT,
        "samples": display_samples,
        "metrics": metrics,
        "distributions": distributions,
        "correlations": correlations,
        "pca": pca,
        "categorical": {
            "polarity": polarity_counts,
            "direction": direction_counts,
            "source_kind": source_kind_counts,
            "has_ig": {"yes": curves_with_ig, "no": len(rows) - curves_with_ig},
            "hysteresis": {
                "paired": hysteresis_paired,
                "NA": len(rows) - hysteresis_paired,
            },
        },
        "processing": {
            "sources": len(unique_sources),
            "raw_points": raw_points,
            "aligned_points": int(len(rows) * X_NORM_GRID.size),
            "gate_points": gate_points,
            "aligned_gate_points": aligned_gate_point_count,
            "curves_with_ig": curves_with_ig,
            "rows_clean_mean": metrics["rows_clean"]["mean"],
            "voltage_span_min": min(spans) if spans else None,
            "voltage_span_max": max(spans) if spans else None,
        },
    }


def export_curve_rows(
    database_url: str | None = None,
    *,
    curve_ids: list[str] | None = None,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = create_database_engine(database_url)
    create_schema(engine)
    joined = curves.join(
        source_files,
        curves.c.source_file_id == source_files.c.id,
    ).join(
        test_configs,
        curves.c.test_config_id == test_configs.c.id,
    )
    conditions = _selected_curve_conditions(curve_ids, filters)
    metadata_query = select(
        curves.c.curve_id,
        curves.c.direction,
        curves.c.rows_clean,
        curves.c.voltage_min_v,
        curves.c.voltage_max_v,
        curves.c.ion,
        curves.c.ioff,
        curves.c.ion_ioff_ratio,
        curves.c.polarity,
        curves.c.has_gate_current,
        curves.c.vth,
        curves.c.ss_mv_dec,
        curves.c.gm_max,
        curves.c.noise_log_sigma,
        curves.c.ambipolar_strength,
        source_files.c.source_path,
        source_files.c.modified_at,
        test_configs.c.source_kind,
        test_configs.c.setup_title,
        test_configs.c.primitive_test,
        test_configs.c.voltage_column,
        test_configs.c.current_column,
        test_configs.c.gate_current_column,
    ).select_from(joined)
    if conditions:
        metadata_query = metadata_query.where(and_(*conditions))
    with engine.connect() as connection:
        curve_rows = [dict(row) for row in connection.execute(metadata_query).mappings()]
        ids = [row["curve_id"] for row in curve_rows]
        if not ids:
            return {
                "curves": [],
                "raw_points": [],
                "gate_points": [],
                "aligned_gate_points": [],
                "analysis": analyze_curves(database_url, curve_ids=curve_ids, filters=filters),
            }
        raw_rows = [
            dict(row)
            for row in connection.execute(
                select(raw_points).where(raw_points.c.curve_id.in_(ids)).order_by(
                    raw_points.c.curve_id,
                    raw_points.c.point_index,
                )
            ).mappings()
        ]
        gate_rows = [
            dict(row)
            for row in connection.execute(
                select(raw_gate_points).where(raw_gate_points.c.curve_id.in_(ids)).order_by(
                    raw_gate_points.c.curve_id,
                    raw_gate_points.c.point_index,
                )
            ).mappings()
        ]
        aligned_gate_rows = [
            dict(row)
            for row in connection.execute(
                select(aligned_gate_points)
                .where(aligned_gate_points.c.curve_id.in_(ids))
                .order_by(
                    aligned_gate_points.c.curve_id,
                    aligned_gate_points.c.point_index,
                )
            ).mappings()
        ]
    for row in curve_rows:
        if row["modified_at"] is not None:
            row["modified_at"] = row["modified_at"].isoformat()
        ratio = row["ion_ioff_ratio"]
        row["logRatio"] = float(np.log10(ratio)) if ratio and ratio > 0 else None
        row["has_ig"] = bool(row.pop("has_gate_current"))
    return {
        "curves": curve_rows,
        "raw_points": raw_rows,
        "gate_points": gate_rows,
        "aligned_gate_points": aligned_gate_rows,
        "analysis": analyze_curves(database_url, curve_ids=curve_ids, filters=filters),
    }


def search_options(database_url: str | None = None) -> dict[str, Any]:
    engine = create_database_engine(database_url)
    create_schema(engine)
    with engine.connect() as connection:
        source_kinds = [
            row[0]
            for row in connection.execute(
                select(test_configs.c.source_kind).distinct().order_by(test_configs.c.source_kind)
            )
        ]
        polarities = [
            row[0]
            for row in connection.execute(
                select(curves.c.polarity).distinct().order_by(curves.c.polarity)
            )
        ]
        directions = [
            row[0]
            for row in connection.execute(
                select(curves.c.direction).distinct().order_by(curves.c.direction)
            )
        ]
    return {
        "source_kinds": source_kinds,
        "polarities": polarities,
        "directions": directions,
    }
