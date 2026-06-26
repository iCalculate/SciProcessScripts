from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd

from .harmonize import _segment_sweeps

SUPPORTED_B1500_SUFFIXES = {".csv", ".txt", ".tsv", ".dat", ".ztr", ".xtr", ".xml"}
TEXT_SUFFIXES = {".csv", ".txt", ".tsv", ".dat"}
XML_SUFFIXES = {".xtr", ".xml"}
X_NORM_GRID = np.linspace(-1.0, 1.0, 201)
XML_GATE_MARKERS = (b'name="vg"', b'name="vgs"', b"name='vg'", b"name='vgs'")
XML_CURRENT_MARKERS = (
    b'name="id"',
    b'name="ids"',
    b'name="absid"',
    b"name='id'",
    b"name='ids'",
    b"name='absid'",
)
TEST_DATA_PATTERN = re.compile(
    rb"<(?:\w+:)?TestData\b.*?</(?:\w+:)?TestData>",
    flags=re.IGNORECASE | re.DOTALL,
)
DATA_VECTOR_PATTERN = re.compile(
    rb"<(?:\w+:)?DataVector\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?:\w+:)?DataVector>",
    flags=re.IGNORECASE | re.DOTALL,
)
XML_NAME_PATTERN = re.compile(rb"\bName\s*=\s*['\"](?P<name>[^'\"]+)['\"]", flags=re.IGNORECASE)
XML_VALUE_PATTERN = re.compile(
    rb"<(?:\w+:)?Value>\s*(?P<value>.*?)\s*</(?:\w+:)?Value>",
    flags=re.IGNORECASE | re.DOTALL,
)
XML_TITLE_PATTERN = re.compile(
    rb"<(?:\w+:)?Title>\s*(?P<value>.*?)\s*</(?:\w+:)?Title>",
    flags=re.IGNORECASE | re.DOTALL,
)
XML_PRIMITIVE_PATTERN = re.compile(
    rb"<(?:\w+:)?PrimitiveTest\b(?P<attrs>[^>]*)>",
    flags=re.IGNORECASE,
)
XML_PARAMETER_PATTERN = re.compile(
    rb"<(?:\w+:)?Parameter\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?:\w+:)?Parameter>",
    flags=re.IGNORECASE | re.DOTALL,
)
XML_INDEX_PATTERN = re.compile(rb"\bIndex\s*=\s*['\"](?P<index>\d+)['\"]", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ParsedTable:
    source_path: Path
    table_name: str
    frame: pd.DataFrame
    metadata: dict[str, list[str]]
    source_kind: str


@dataclass(frozen=True)
class ColumnChoice:
    voltage: str
    current: str
    gate_current: str | None
    curve_type: str
    confidence: float
    reason: str


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _to_number_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    return numeric


def _finite_unique_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0
    return int(values.nunique())


def _finite_range(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    return float(values.max() - values.min())


def _is_gate_voltage(name: str) -> bool:
    normalized = _norm(name)
    return normalized in {"vg", "vgs", "gatevoltage", "vgate"} or normalized.endswith("vg")


def _is_drain_voltage(name: str) -> bool:
    normalized = _norm(name)
    return normalized in {"vd", "vds", "drainvoltage", "vdrain"} or normalized.endswith("vd")


def _is_drain_current(name: str) -> bool:
    normalized = _norm(name)
    return normalized in {
        "id",
        "ids",
        "absid",
        "draincurrent",
        "draini",
        "idrain",
    } or normalized.endswith("id")


def _is_gate_current(name: str) -> bool:
    normalized = _norm(name)
    return normalized in {
        "ig",
        "igs",
        "gatecurrent",
        "gatei",
        "igate",
    } or normalized.endswith("ig")


def _current_rank(name: str) -> int:
    normalized = _norm(name)
    if normalized in {"id", "ids"}:
        return 0
    if normalized == "absid":
        return 1
    if _is_drain_current(name):
        return 2
    return 99


def _gate_current_rank(name: str) -> int:
    normalized = _norm(name)
    if normalized in {"ig", "igs"}:
        return 0
    if _is_gate_current(name):
        return 1
    return 99


def _parse_b1500_csv(content: bytes, path: Path) -> list[ParsedTable]:
    text = _decode(content)
    rows = list(csv.reader(io.StringIO(text)))
    metadata: dict[str, list[str]] = {}
    blocks: list[tuple[list[str], list[list[str]]]] = []
    current_header: list[str] | None = None
    current_rows: list[list[str]] = []

    for row in rows:
        if not row:
            continue
        kind = row[0].strip()
        if kind == "TestParameter" and len(row) >= 2:
            metadata[row[1].strip()] = [value.strip() for value in row[2:]]
        elif kind in {"SetupTitle", "PrimitiveTest"}:
            metadata[kind] = [value.strip() for value in row[1:]]
        elif kind == "DataName":
            if current_header is not None:
                blocks.append((current_header, current_rows))
            current_header = [value.strip() for value in row[1:]]
            current_rows = []
        elif kind == "DataValue" and current_header is not None:
            current_rows.append([value.strip() for value in row[1:]])

    if current_header is not None:
        blocks.append((current_header, current_rows))

    parsed: list[ParsedTable] = []
    for index, (header, values) in enumerate(blocks):
        if len(header) < 2 or len(values) < 4:
            continue
        width = len(header)
        clean_rows = [row[:width] + [""] * max(0, width - len(row)) for row in values]
        frame = pd.DataFrame(clean_rows, columns=header)
        parsed.append(
            ParsedTable(
                source_path=path,
                table_name=f"DataBlock{index + 1}",
                frame=frame,
                metadata=metadata,
                source_kind="b1500_csv",
            )
        )
    if parsed:
        return parsed

    return _parse_plain_table(text, path)


def _parse_plain_table(text: str, path: Path) -> list[ParsedTable]:
    sample = "\n".join(text.splitlines()[:30])
    delimiter = "\t" if "\t" in sample and sample.count("\t") >= sample.count(",") else ","
    best: pd.DataFrame | None = None
    for skiprows in range(min(25, max(1, len(text.splitlines()) - 2))):
        try:
            frame = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                engine="python",
                skiprows=skiprows,
            )
        except (UnicodeError, pd.errors.ParserError):
            continue
        if frame.shape[1] < 2:
            continue
        numeric_columns = sum(
            pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 4
            for column in frame.columns
        )
        if numeric_columns >= 2:
            best = frame
            break
    if best is None:
        return []
    best.columns = [str(column).strip() for column in best.columns]
    return [
        ParsedTable(
            source_path=path,
            table_name="Table1",
            frame=best,
            metadata={},
            source_kind="plain_table",
        )
    ]


def _xml_text_value(element: Any) -> str:
    for child in list(element):
        text = child.text
        if text is not None:
            return text.strip()
    return (element.text or "").strip()


def _parse_xtr_xml(content: bytes, path: Path, *, source_kind: str) -> list[ParsedTable]:
    lowered = content.lower()
    if not any(marker in lowered for marker in XML_GATE_MARKERS) or not any(
        marker in lowered for marker in XML_CURRENT_MARKERS
    ):
        return []
    parsed: list[ParsedTable] = []
    test_blocks = TEST_DATA_PATTERN.findall(content) or [content]
    for table_index, test_data in enumerate(test_blocks, start=1):
        metadata: dict[str, list[str]] = {}
        title_match = XML_TITLE_PATTERN.search(test_data)
        if title_match:
            metadata["SetupTitle"] = [
                title_match.group("value").decode("utf-8", errors="replace").strip()
            ]
        primitive_match = XML_PRIMITIVE_PATTERN.search(test_data)
        if primitive_match:
            name_match = XML_NAME_PATTERN.search(primitive_match.group("attrs"))
            if name_match:
                metadata["PrimitiveTest"] = [
                    name_match.group("name").decode("utf-8", errors="replace").strip()
                ]
        parameters: dict[str, dict[int, str]] = {}
        for parameter in XML_PARAMETER_PATTERN.finditer(test_data):
            name_match = XML_NAME_PATTERN.search(parameter.group("attrs"))
            if not name_match:
                continue
            index_match = XML_INDEX_PATTERN.search(parameter.group("attrs"))
            index = int(index_match.group("index")) if index_match else 0
            value_match = XML_VALUE_PATTERN.search(parameter.group("body"))
            if not value_match:
                continue
            name = name_match.group("name").decode("utf-8", errors="replace").strip()
            value = value_match.group("value").decode("utf-8", errors="replace").strip()
            parameters.setdefault(name, {})[index] = value
        for name, indexed in parameters.items():
            max_index = max(indexed) if indexed else -1
            metadata[name] = [indexed.get(index, "") for index in range(max_index + 1)]
        vectors: dict[str, list[float]] = {}
        for vector in DATA_VECTOR_PATTERN.finditer(test_data):
            name_match = XML_NAME_PATTERN.search(vector.group("attrs"))
            if not name_match:
                continue
            name = name_match.group("name").decode("utf-8", errors="replace").strip()
            values: list[float] = []
            for value_match in XML_VALUE_PATTERN.finditer(vector.group("body")):
                try:
                    values.append(float(value_match.group("value").strip()))
                except ValueError:
                    values.append(math.nan)
            if values:
                vectors[name] = values
        if len(vectors) < 2:
            continue
        min_length = min(len(values) for values in vectors.values())
        if min_length < 4:
            continue
        frame = pd.DataFrame({name: values[:min_length] for name, values in vectors.items()})
        parsed.append(
            ParsedTable(
                source_path=path,
                table_name=f"TestData{table_index}",
                frame=frame,
                metadata=metadata,
                source_kind=source_kind,
            )
        )
    return parsed


def parse_source(path: Path, *, max_xml_mb: float = 128.0) -> list[ParsedTable]:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return _parse_b1500_csv(path.read_bytes(), path)
    if suffix == ".ztr":
        parsed: list[ParsedTable] = []
        try:
            with ZipFile(path) as archive:
                for name in archive.namelist():
                    if Path(name).suffix.lower() in XML_SUFFIXES:
                        parsed.extend(
                            _parse_xtr_xml(
                                archive.read(name),
                                path,
                                source_kind=f"ztr:{Path(name).suffix.lower()}",
                            )
                        )
        except (BadZipFile, OSError):
            return []
        return parsed
    if suffix in XML_SUFFIXES:
        if path.stat().st_size > max_xml_mb * 1024 * 1024:
            raise ValueError(f"XML source is larger than {max_xml_mb:.0f} MB")
        return _parse_xtr_xml(path.read_bytes(), path, source_kind=suffix.lstrip("."))
    return []


def _metadata_axis(metadata: dict[str, list[str]]) -> str | None:
    values = metadata.get("Output.Graph.XAxis.Data", [])
    return values[0] if values else None


def choose_columns(table: ParsedTable) -> ColumnChoice | None:
    frame = _to_number_frame(table.frame)
    usable = [
        column
        for column in frame.columns
        if pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 4
    ]
    if len(usable) < 2:
        return None

    current_columns = [column for column in usable if _is_drain_current(column)]
    if not current_columns:
        return None
    current = sorted(current_columns, key=_current_rank)[0]

    gate_columns = [
        column
        for column in usable
        if _is_gate_voltage(column) and _finite_unique_count(frame[column]) >= 8
    ]
    drain_voltage_columns = [
        column
        for column in usable
        if _is_drain_voltage(column) and _finite_unique_count(frame[column]) >= 8
    ]
    x_axis = _metadata_axis(table.metadata)
    gate_current_columns = [column for column in usable if _is_gate_current(column)]
    gate_current = (
        sorted(gate_current_columns, key=_gate_current_rank)[0]
        if gate_current_columns
        else None
    )
    if x_axis in gate_columns:
        return ColumnChoice(
            x_axis,
            current,
            gate_current,
            "transfer",
            0.98,
            "graph x-axis is swept gate voltage",
        )
    if x_axis in drain_voltage_columns:
        return ColumnChoice(
            x_axis,
            current,
            gate_current,
            "output",
            0.98,
            "graph x-axis is swept drain voltage",
        )

    gate_columns = [column for column in gate_columns if _finite_range(frame[column]) >= 0.2]
    drain_voltage_columns = [
        column for column in drain_voltage_columns if _finite_range(frame[column]) >= 0.2
    ]
    if gate_columns and not drain_voltage_columns:
        return ColumnChoice(
            gate_columns[0],
            current,
            gate_current,
            "transfer",
            0.9,
            "only gate voltage sweeps",
        )
    if drain_voltage_columns and not gate_columns:
        return ColumnChoice(
            drain_voltage_columns[0],
            current,
            gate_current,
            "output",
            0.9,
            "only drain voltage sweeps",
        )
    if gate_columns and drain_voltage_columns:
        gate_range = max(_finite_range(frame[column]) for column in gate_columns)
        drain_range = max(_finite_range(frame[column]) for column in drain_voltage_columns)
        if gate_range >= drain_range:
            return ColumnChoice(
                gate_columns[0],
                current,
                gate_current,
                "transfer",
                0.72,
                "gate sweep dominates",
            )
        return ColumnChoice(
            drain_voltage_columns[0],
            current,
            gate_current,
            "output",
            0.72,
            "drain sweep dominates",
        )
    return None


def _clean_pair(frame: pd.DataFrame, voltage_column: str, current_column: str) -> pd.DataFrame:
    selected = frame[[voltage_column, current_column]].copy()
    selected.columns = ["voltage_v", "current_a"]
    selected["voltage_v"] = pd.to_numeric(selected["voltage_v"], errors="coerce")
    selected["current_a"] = pd.to_numeric(selected["current_a"], errors="coerce")
    selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    selected.dropna(inplace=True)
    selected = selected[selected["current_a"].abs() > 0]
    duplicated = selected["voltage_v"].eq(selected["voltage_v"].shift()) & selected[
        "current_a"
    ].eq(selected["current_a"].shift())
    selected = selected.loc[~duplicated].copy()
    return selected


def _clean_transfer_columns(frame: pd.DataFrame, choice: ColumnChoice) -> pd.DataFrame:
    columns = [choice.voltage, choice.current]
    if choice.gate_current is not None:
        columns.append(choice.gate_current)
    selected = frame[columns].copy()
    selected.columns = [
        "voltage_v",
        "current_a",
        *(["gate_current_a"] if choice.gate_current is not None else []),
    ]
    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    selected.dropna(subset=["voltage_v", "current_a"], inplace=True)
    selected = selected[selected["current_a"].abs() > 0]
    duplicated = selected["voltage_v"].eq(selected["voltage_v"].shift()) & selected[
        "current_a"
    ].eq(selected["current_a"].shift())
    return selected.loc[~duplicated].copy()


def _segment_bounds(voltage: np.ndarray) -> list[tuple[int, int]]:
    if voltage.size < 5:
        return []
    delta = np.diff(voltage)
    nonzero = np.sign(delta)
    for index in range(1, nonzero.size):
        if nonzero[index] == 0:
            nonzero[index] = nonzero[index - 1]
    if nonzero.size and nonzero[0] == 0:
        first = next((value for value in nonzero if value != 0), 1)
        nonzero[nonzero == 0] = first
    changes = np.flatnonzero(nonzero[1:] != nonzero[:-1]) + 1
    bounds: list[tuple[int, int]] = []
    start = 0
    for turning_point in changes.tolist():
        end = turning_point + 1
        if end - start >= 4:
            bounds.append((start, end))
        start = turning_point
    if voltage.size - start >= 4:
        bounds.append((start, voltage.size))
    return bounds or [(0, voltage.size)]


def _quality_rejection(selected: pd.DataFrame) -> str | None:
    if selected.shape[0] < 20:
        return "fewer than 20 finite transfer rows"
    voltage_range = float(selected["voltage_v"].max() - selected["voltage_v"].min())
    if voltage_range < 0.5:
        return "swept voltage range below 0.5 V"
    abs_current = selected["current_a"].abs().to_numpy(dtype=float)
    dynamic_range = float(np.max(abs_current) / max(np.min(abs_current), np.finfo(float).tiny))
    if dynamic_range < 20:
        return "current dynamic range below 20x"
    return None


def _align_segment(
    voltage: np.ndarray, current: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(voltage)
    voltage = voltage[order]
    current = current[order]
    unique_voltage, unique_indices = np.unique(voltage, return_index=True)
    unique_current = np.abs(current[unique_indices])
    if unique_voltage.size < 2:
        raise ValueError("segment has fewer than two unique voltage points")
    physical_grid = np.interp(
        X_NORM_GRID,
        np.linspace(-1.0, 1.0, unique_voltage.size),
        unique_voltage,
    )
    log_current = np.log10(np.clip(unique_current, np.finfo(float).tiny, None))
    aligned_log_current = np.interp(physical_grid, unique_voltage, log_current)
    aligned_current = np.power(10.0, aligned_log_current)
    return physical_grid, aligned_log_current, aligned_current


def _align_current_on_grid(
    voltage: np.ndarray,
    current: np.ndarray,
    physical_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(voltage)
    sorted_voltage = voltage[order]
    sorted_current = np.abs(current[order])
    unique_voltage, unique_indices = np.unique(sorted_voltage, return_index=True)
    if unique_voltage.size < 2:
        raise ValueError("segment has fewer than two unique voltage points")
    log_current = np.log10(
        np.clip(sorted_current[unique_indices], np.finfo(float).tiny, None)
    )
    aligned_log_current = np.interp(physical_grid, unique_voltage, log_current)
    return aligned_log_current, np.power(10.0, aligned_log_current)


def _stable_curve_id(path: Path, table_name: str, segment_index: int) -> str:
    digest = hashlib.sha1(f"{path}|{table_name}|{segment_index}".encode()).hexdigest()
    return digest[:16]


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def build_b1500_dataset(
    source: Path,
    output: Path,
    *,
    max_xml_mb: float = 128.0,
    suffixes: set[str] | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    allowed_suffixes = suffixes or SUPPORTED_B1500_SUFFIXES
    allowed_suffixes = {
        suffix.lower() if suffix.startswith(".") else f".{suffix.lower()}"
        for suffix in allowed_suffixes
    }
    unsupported = allowed_suffixes - SUPPORTED_B1500_SUFFIXES
    if unsupported:
        raise ValueError(f"Unsupported suffixes: {', '.join(sorted(unsupported))}")
    files = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_suffixes
    )
    records: list[dict[str, Any]] = []
    aligned_rows: list[dict[str, Any]] = []
    aligned_gate_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for file_index, path in enumerate(files, start=1):
        try:
            tables = parse_source(path, max_xml_mb=max_xml_mb)
        except (OSError, ValueError) as error:
            rejected.append(
                {
                    "source_path": _safe_relpath(path, source),
                    "table_name": "",
                    "reason": str(error),
                }
            )
            continue
        if not tables:
            rejected.append(
                {
                    "source_path": _safe_relpath(path, source),
                    "table_name": "",
                    "reason": "no numeric B1500 data table found",
                }
            )
            continue
        for table in tables:
            choice = choose_columns(table)
            if choice is None:
                rejected.append(
                    {
                        "source_path": _safe_relpath(path, source),
                        "table_name": table.table_name,
                        "reason": "no content-level FET transfer/output column pattern",
                    }
                )
                continue
            if choice.curve_type != "transfer":
                rejected.append(
                    {
                        "source_path": _safe_relpath(path, source),
                        "table_name": table.table_name,
                        "reason": f"content classified as {choice.curve_type}, not transfer",
                    }
                )
                continue

            cleaned = _clean_transfer_columns(table.frame, choice)
            rejection = _quality_rejection(cleaned)
            if rejection:
                rejected.append(
                    {
                        "source_path": _safe_relpath(path, source),
                        "table_name": table.table_name,
                        "reason": rejection,
                    }
                )
                continue

            voltage = cleaned["voltage_v"].to_numpy(dtype=float)
            current = cleaned["current_a"].to_numpy(dtype=float)
            segments = _segment_sweeps(voltage, current)
            segment_bounds = _segment_bounds(voltage)
            for segment_index, (segment, bounds) in enumerate(
                zip(segments, segment_bounds, strict=True),
                start=1,
            ):
                if segment.rows < 20 or segment.features is None:
                    continue
                features = segment.features
                if (
                    features.vth is None
                    or features.ss_mv_dec is None
                    or features.polarity == "unknown"
                    or features.ion_ioff_ratio < 20
                ):
                    rejected.append(
                        {
                            "source_path": _safe_relpath(path, source),
                            "table_name": table.table_name,
                            "reason": "segment lacks stable transfer features",
                        }
                    )
                    continue
                physical_grid, aligned_log_current, aligned_current = _align_segment(
                    np.asarray(segment.voltage, dtype=float),
                    np.asarray(segment.current, dtype=float),
                )
                curve_id = _stable_curve_id(path, table.table_name, segment_index)
                aligned_log_gate: np.ndarray | None = None
                aligned_gate_current: np.ndarray | None = None
                if "gate_current_a" in cleaned.columns:
                    start, stop = bounds
                    gate_voltage = voltage[start:stop]
                    gate_current = cleaned["gate_current_a"].to_numpy(dtype=float)[start:stop]
                    gate_valid = np.isfinite(gate_current) & (np.abs(gate_current) > 0)
                    if np.count_nonzero(gate_valid) >= 2:
                        aligned_log_gate, aligned_gate_current = _align_current_on_grid(
                            gate_voltage[gate_valid],
                            gate_current[gate_valid],
                            physical_grid,
                        )
                feature_payload = features.model_dump(mode="json")
                records.append(
                    {
                        "curve_id": curve_id,
                        "source_path": _safe_relpath(path, source),
                        "source_kind": table.source_kind,
                        "table_name": table.table_name,
                        "segment_index": segment_index,
                        "direction": segment.direction,
                        "rows_clean": segment.rows,
                        "voltage_column": choice.voltage,
                        "current_column": choice.current,
                        "gate_current_column": choice.gate_current,
                        "has_gate_current": aligned_log_gate is not None,
                        "classification_reason": choice.reason,
                        "classification_confidence": choice.confidence,
                        "voltage_min_v": float(np.min(segment.voltage)),
                        "voltage_max_v": float(np.max(segment.voltage)),
                        **{f"feature_{key}": value for key, value in feature_payload.items()},
                    }
                )
                for point_index, (x_norm, voltage_v, log_id, abs_id) in enumerate(
                    zip(
                        X_NORM_GRID,
                        physical_grid,
                        aligned_log_current,
                        aligned_current,
                        strict=True,
                    )
                ):
                    aligned_rows.append(
                        {
                            "curve_id": curve_id,
                            "point_index": point_index,
                            "x_norm": float(x_norm),
                            "voltage_v": float(voltage_v),
                            "log10_abs_id": float(log_id),
                            "abs_id_a": float(abs_id),
                        }
                    )
                if aligned_log_gate is not None and aligned_gate_current is not None:
                    for point_index, (x_norm, voltage_v, log_ig, abs_ig) in enumerate(
                        zip(
                            X_NORM_GRID,
                            physical_grid,
                            aligned_log_gate,
                            aligned_gate_current,
                            strict=True,
                        )
                    ):
                        aligned_gate_rows.append(
                            {
                                "curve_id": curve_id,
                                "point_index": point_index,
                                "x_norm": float(x_norm),
                                "voltage_v": float(voltage_v),
                                "log10_abs_ig": float(log_ig),
                                "abs_ig_a": float(abs_ig),
                            }
                        )

        if file_index % 500 == 0:
            print(
                f"Processed {file_index}/{len(files)} files; accepted {len(records)} segments",
                flush=True,
            )

    curves_frame = pd.DataFrame(records)
    aligned_frame = pd.DataFrame(aligned_rows)
    aligned_gate_frame = pd.DataFrame(aligned_gate_rows)
    rejected_frame = pd.DataFrame(rejected)

    curves_csv = output / "curves.csv"
    aligned_csv = output / "aligned_curves.csv"
    aligned_gate_csv = output / "aligned_gate_curves.csv"
    rejected_csv = output / "rejected_files.csv"
    manifest_json = output / "manifest.json"
    npz_path = output / "aligned_curves.npz"
    report_md = output / "README.md"

    curves_frame.to_csv(curves_csv, index=False)
    aligned_frame.to_csv(aligned_csv, index=False)
    aligned_gate_frame.to_csv(aligned_gate_csv, index=False)
    rejected_frame.to_csv(rejected_csv, index=False)

    if not curves_frame.empty:
        matrix = (
            aligned_frame.pivot(index="curve_id", columns="point_index", values="log10_abs_id")
            .loc[curves_frame["curve_id"]]
            .to_numpy(dtype=float)
        )
    else:
        matrix = np.empty((0, X_NORM_GRID.size), dtype=float)
    gate_matrix = np.full(matrix.shape, np.nan, dtype=float)
    if not curves_frame.empty and not aligned_gate_frame.empty:
        gate_pivot = aligned_gate_frame.pivot(
            index="curve_id",
            columns="point_index",
            values="log10_abs_ig",
        )
        available_gate_ids = curves_frame["curve_id"].isin(gate_pivot.index)
        if available_gate_ids.any():
            gate_matrix[available_gate_ids.to_numpy()] = (
                gate_pivot.loc[curves_frame.loc[available_gate_ids, "curve_id"]]
                .to_numpy(dtype=float)
            )
    np.savez_compressed(
        npz_path,
        curve_id=curves_frame["curve_id"].to_numpy(dtype=str) if not curves_frame.empty else [],
        x_norm=X_NORM_GRID,
        log10_abs_id=matrix,
        log10_abs_ig=gate_matrix,
    )

    summary = {
        "source": str(source),
        "output": str(output),
        "files_discovered": len(files),
        "accepted_transfer_segments": int(len(records)),
        "aligned_points_per_segment": int(X_NORM_GRID.size),
        "segments_with_gate_current": int(
            curves_frame["has_gate_current"].sum()
            if not curves_frame.empty and "has_gate_current" in curves_frame
            else 0
        ),
        "rejected_entries": int(len(rejected)),
        "accepted_by_source_kind": (
            curves_frame["source_kind"].value_counts().to_dict() if not curves_frame.empty else {}
        ),
        "polarity_counts": (
            curves_frame["feature_polarity"].value_counts().to_dict()
            if not curves_frame.empty and "feature_polarity" in curves_frame
            else {}
        ),
    }
    manifest_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_md.write_text(_render_report(summary, curves_frame, rejected_frame), encoding="utf-8")
    return summary


def _render_report(
    summary: dict[str, Any], curves_frame: pd.DataFrame, rejected_frame: pd.DataFrame
) -> str:
    lines = [
        "# B1500 local transfer test dataset",
        "",
        f"- Source: `{summary['source']}`",
        f"- Files discovered: {summary['files_discovered']}",
        f"- Accepted transfer segments: {summary['accepted_transfer_segments']}",
        f"- Aligned points per segment: {summary['aligned_points_per_segment']}",
        f"- Rejected/skipped entries: {summary['rejected_entries']}",
        "",
        "## Content-level screening rules",
        "",
        "- Parse B1500 `DataName/DataValue` blocks in CSV/TXT and `DataVector` blocks "
        "in ZTR/XTR/XML.",
        "- Classify transfer curves only when a swept gate-voltage column and "
        "drain-current column are present in the parsed data.",
        "- Reject output curves when the swept content axis is drain voltage.",
        "- Reject direct-control, DRAM, C-V, I-T, one-column, low-range, and "
        "low-dynamic-range tables.",
        "- Split forward/reverse sweeps by voltage turning points and align every "
        "accepted segment to 201 normalized points.",
        "",
    ]
    if not curves_frame.empty:
        ion_min = curves_frame["feature_ion"].min()
        ion_max = curves_frame["feature_ion"].max()
        ioff_min = curves_frame["feature_ioff"].min()
        ioff_max = curves_frame["feature_ioff"].max()
        ratio_min = curves_frame["feature_ion_ioff_ratio"].min()
        ratio_max = curves_frame["feature_ion_ioff_ratio"].max()
        voltage_min = curves_frame["voltage_min_v"].min()
        voltage_max = curves_frame["voltage_max_v"].max()
        lines.extend(
            [
                "## Accepted curve feature ranges",
                "",
                f"- Ion: {ion_min:.3e} to {ion_max:.3e} A",
                f"- Ioff: {ioff_min:.3e} to {ioff_max:.3e} A",
                f"- Ion/Ioff: {ratio_min:.2f} to {ratio_max:.2f}",
                f"- Voltage min/max: {voltage_min:.3g} to {voltage_max:.3g} V",
                "",
            ]
        )
    if not rejected_frame.empty:
        reasons = rejected_frame["reason"].value_counts().head(12)
        lines.extend(["## Top rejection reasons", ""])
        lines.extend(f"- {reason}: {count}" for reason, count in reasons.items())
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            "- `curves.csv`: one row per accepted transfer segment with source metadata "
            "and extracted features.",
            "- `aligned_curves.csv`: long-form 201-point aligned log-current curves.",
            "- `aligned_gate_curves.csv`: aligned Ig curves when a gate-current column exists.",
            "- `aligned_curves.npz`: compact matrix form for local model/testing code.",
            "- `rejected_files.csv`: audit trail for skipped tables and files.",
            "- `manifest.json`: machine-readable summary.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a local B1500 transfer-curve test dataset.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/b1500_test_dataset"))
    parser.add_argument("--max-xml-mb", type=float, default=128.0)
    args = parser.parse_args()
    summary = build_b1500_dataset(args.source, args.output, max_xml_mb=args.max_xml_mb)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
