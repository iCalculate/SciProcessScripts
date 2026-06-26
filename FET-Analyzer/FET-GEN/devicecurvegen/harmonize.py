from __future__ import annotations

import csv
import io
import re

import numpy as np
import pandas as pd

from .features import analyze_transfer_curve
from .schemas import (
    ColumnMapping,
    CurveSegment,
    InspectionResponse,
)

VOLTAGE_ALIASES = {
    "vg",
    "gatevoltage",
    "gatev",
    "vgs",
    "gatebias",
    "voltage",
}
CURRENT_ALIASES = {
    "id",
    "draincurrent",
    "draini",
    "ids",
    "absid",
    "current",
}


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:30])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t; ")
        return dialect.delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def _read_frame(text: str, delimiter: str) -> pd.DataFrame:
    lines = text.splitlines()
    best: pd.DataFrame | None = None
    for skiprows in range(min(25, max(1, len(lines) - 2))):
        try:
            frame = pd.read_csv(
                io.StringIO(text),
                sep=delimiter,
                engine="python",
                skiprows=skiprows,
                comment="#",
            )
        except (pd.errors.ParserError, UnicodeError):
            continue
        if frame.shape[1] < 2:
            continue
        numeric_columns = sum(
            pd.to_numeric(frame[column], errors="coerce").notna().sum() >= 2
            for column in frame.columns
        )
        if numeric_columns >= 2:
            best = frame
            break
    if best is None:
        raise ValueError("Could not find a numeric table with at least two columns")
    best.columns = [str(column).strip() for column in best.columns]
    return best


def _map_columns(
    columns: list[str],
    voltage_override: str | None,
    current_override: str | None,
) -> ColumnMapping:
    if voltage_override is not None and voltage_override not in columns:
        raise ValueError(f"Unknown voltage column: {voltage_override}")
    if current_override is not None and current_override not in columns:
        raise ValueError(f"Unknown current column: {current_override}")
    if (
        voltage_override is not None
        and current_override is not None
        and voltage_override == current_override
    ):
        raise ValueError("Voltage and current columns must be different")

    normalized = {_normalize_column(column): column for column in columns}
    voltage = voltage_override
    current = current_override
    if voltage is None:
        voltage = next(
            (
                column
                for alias in VOLTAGE_ALIASES
                for normalized_name, column in normalized.items()
                if normalized_name == alias
                or normalized_name.startswith(f"{alias}v")
                or normalized_name.startswith(f"{alias}volt")
            ),
            None,
        )
    if current is None:
        current = next(
            (
                column
                for alias in CURRENT_ALIASES
                for normalized_name, column in normalized.items()
                if normalized_name == alias
                or normalized_name.startswith(f"{alias}a")
                or normalized_name.startswith(f"{alias}amp")
            ),
            None,
        )
    if voltage == current:
        current = None
    confidence = (
        1.0
        if voltage_override and current_override
        else 0.97
        if voltage_override or current_override
        else 0.92
    )
    if voltage is None or current is None:
        confidence = 0.25
        remaining = [column for column in columns if column != voltage]
        voltage = voltage or (columns[0] if columns else None)
        current = current or (remaining[0] if remaining else None)
    return ColumnMapping(voltage=voltage, current=current, confidence=confidence)


def _aligned_curve(
    voltage: np.ndarray, current: np.ndarray, points: int = 201
) -> tuple[list[float], list[float]]:
    order = np.argsort(voltage)
    sorted_voltage = voltage[order]
    sorted_log_current = np.log10(np.clip(np.abs(current[order]), np.finfo(float).tiny, None))
    unique_voltage, unique_indices = np.unique(sorted_voltage, return_index=True)
    if unique_voltage.size < 2:
        return unique_voltage.tolist(), sorted_log_current[unique_indices].tolist()
    grid = np.linspace(float(unique_voltage[0]), float(unique_voltage[-1]), points)
    aligned = np.interp(grid, unique_voltage, sorted_log_current[unique_indices])
    return grid.tolist(), aligned.tolist()


def _segment_sweeps(voltage: np.ndarray, current: np.ndarray) -> list[CurveSegment]:
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
    raw: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for turning_point in changes.tolist():
        end = turning_point + 1
        if end - start >= 4:
            raw.append((voltage[start:end], current[start:end]))
        start = turning_point
    if voltage.size - start >= 4:
        raw.append((voltage[start:], current[start:]))
    if not raw:
        raw = [(voltage, current)]

    segments: list[CurveSegment] = []
    for vg, ids in raw:
        direction = "single" if len(raw) == 1 else "forward" if vg[-1] >= vg[0] else "reverse"
        aligned_voltage, aligned_log_current = _aligned_curve(vg, ids)
        segments.append(
            CurveSegment(
                direction=direction,
                rows=int(vg.size),
                voltage=vg.tolist(),
                current=ids.tolist(),
                aligned_voltage=aligned_voltage,
                aligned_log_current=aligned_log_current,
                features=analyze_transfer_curve(vg, ids),
            )
        )
    return segments


def inspect_measurement(
    filename: str,
    content: bytes,
    *,
    voltage_column: str | None = None,
    current_column: str | None = None,
) -> InspectionResponse:
    text = _decode(content)
    delimiter = _detect_delimiter(text)
    frame = _read_frame(text, delimiter)
    original_rows = int(frame.shape[0])
    mapping = _map_columns(list(frame.columns), voltage_column, current_column)
    if mapping.voltage is None or mapping.current is None:
        raise ValueError("A voltage and current column are required")

    selected = frame[[mapping.voltage, mapping.current]].copy()
    selected.columns = ["voltage", "current"]
    selected["voltage"] = pd.to_numeric(selected["voltage"], errors="coerce")
    selected["current"] = pd.to_numeric(selected["current"], errors="coerce")
    selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    selected.dropna(inplace=True)
    consecutive_duplicate = selected["voltage"].eq(selected["voltage"].shift()) & selected[
        "current"
    ].eq(selected["current"].shift())
    selected = selected.loc[~consecutive_duplicate].copy()
    cleaned_rows = int(selected.shape[0])
    if cleaned_rows < 4:
        raise ValueError("Fewer than four valid numeric rows remain after cleaning")
    voltage = selected["voltage"].to_numpy(dtype=float)
    current = selected["current"].to_numpy(dtype=float)
    segments = _segment_sweeps(voltage, current)

    labels: list[str] = []
    if cleaned_rows < 12:
        labels.append("incomplete")
    removed_fraction = (original_rows - cleaned_rows) / original_rows if original_rows else 0.0
    if removed_fraction > 0:
        labels.append("partially_valid")
    if cleaned_rows >= 7:
        log_current = np.log10(np.clip(np.abs(current), np.finfo(float).tiny, None))
        derivative = np.diff(log_current)
        median = np.median(derivative)
        mad = np.median(np.abs(derivative - median)) + np.finfo(float).eps
        if np.any(np.abs(derivative - median) > 12 * mad):
            labels.append("noisy")
    abs_current = np.abs(current)
    dynamic_range = float(np.max(abs_current) / max(np.min(abs_current), np.finfo(float).tiny))
    top = np.sort(abs_current)[-min(5, abs_current.size) :]
    if top.size >= 3 and np.ptp(top) <= max(np.max(top) * 1e-5, np.finfo(float).tiny):
        labels.append("compliance_limited")
    if dynamic_range < 10 or not segments:
        labels.append("possible_failure")
    if mapping.confidence < 0.5:
        labels.append("ambiguous")
    if not labels:
        labels.append("valid")

    preview_frame = frame.head(12).where(pd.notna(frame), None)
    preview = preview_frame.to_dict(orient="records")
    return InspectionResponse(
        filename=filename,
        delimiter="tab" if delimiter == "\t" else delimiter,
        columns=list(frame.columns),
        mapping=mapping,
        curve_type="transfer" if mapping.confidence >= 0.5 else "unknown",
        quality_labels=labels,
        original_rows=original_rows,
        cleaned_rows=cleaned_rows,
        removed_rows=original_rows - cleaned_rows,
        segments=segments,
        preview=preview,
    )
