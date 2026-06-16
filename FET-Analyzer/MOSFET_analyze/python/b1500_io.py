"""Robust reader for Keysight B1500A / EasyEXPERT / Clarius CSV exports.

The B1500 ``.csv`` export is a mix of free-form metadata and a tabular data
block.  A single file may contain **several test records** concatenated one
after another (each introduced by a ``SetupTitle`` line), the **column order
is not fixed**, and a record may hold **one or many curves** (an I/V sweep with
a VAR2 secondary axis produces ``Dimension2`` curves of ``Dimension1`` points).

This module hides all of that behind a small data model:

    file  ->  list[Measurement]  ->  Measurement.curves : list[Curve]

Each :class:`Curve` exposes named data columns (``Vg``, ``Id``, ``Ig`` ...) so
downstream code never depends on physical column positions.  The plotter and any
future analysis code consume :class:`Curve` / :class:`Measurement` only.

The reader is intentionally defensive: anything it cannot interpret is skipped
rather than raised, so a folder of heterogeneous exports always loads.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Curve", "Measurement", "read_b1500_csv", "load_folder", "MeasurementKind"]


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

class MeasurementKind:
    TRANSFER = "transfer"   # Id vs Vg
    OUTPUT = "output"       # Id vs Vd
    UNKNOWN = "unknown"


@dataclass
class Curve:
    """A single trace within a measurement (one secondary/VAR2 step)."""

    columns: Dict[str, np.ndarray]          # column name -> values
    label: str = ""                         # human readable, e.g. "Vd = 1 V"
    secondary_name: str = ""                # e.g. "Vd"
    secondary_value: Optional[float] = None  # constant value over this curve

    def get(self, name: str) -> Optional[np.ndarray]:
        """Case-insensitive column lookup; returns None when absent."""
        if name in self.columns:
            return self.columns[name]
        low = name.lower()
        for key, val in self.columns.items():
            if key.lower() == low:
                return val
        return None

    def has(self, name: str) -> bool:
        return self.get(name) is not None

    @property
    def names(self) -> List[str]:
        return list(self.columns.keys())


@dataclass
class Measurement:
    """One test record: a coherent set of curves sharing axes and parameters."""

    kind: str = MeasurementKind.UNKNOWN
    title: str = ""                         # SetupTitle
    x_name: str = ""                        # primary sweep variable (VAR1)
    secondary_name: str = ""                # VAR2 variable, "" if single curve
    curves: List[Curve] = field(default_factory=list)
    params: Dict[str, List[str]] = field(default_factory=dict)
    source_file: str = ""
    record_index: int = 0                   # position within the source file

    @property
    def n_curves(self) -> int:
        return len(self.curves)

    @property
    def name(self) -> str:
        """Short, unique-ish display name for UI lists."""
        base = os.path.basename(self.source_file)
        stem = os.path.splitext(base)[0]
        tag = self.title or self.kind
        return f"{stem}  [{tag}]" if tag else stem

    def default_y_name(self) -> str:
        """Best column to plot on Y for this kind of measurement."""
        for cand in ("Id", "absId", "Is", "Ig", "absIg"):
            if self.curves and self.curves[0].has(cand):
                return cand
        # fall back to first non-axis numeric column
        if self.curves:
            for n in self.curves[0].names:
                if n not in (self.x_name, "Vs"):
                    return n
        return "Id"


# --------------------------------------------------------------------------- #
# Low-level line parsing
# --------------------------------------------------------------------------- #

def _read_text(path: str) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    # last resort: ignore undecodable bytes
    with open(path, "rb") as fh:
        return fh.read().decode("latin-1", errors="ignore")


def _split_fields(line: str) -> List[str]:
    """Split a B1500 CSV line and strip surrounding whitespace from each field."""
    return [f.strip() for f in line.split(",")]


def _to_float(token: str) -> float:
    try:
        return float(token)
    except (ValueError, TypeError):
        return np.nan


# --------------------------------------------------------------------------- #
# Record assembly
# --------------------------------------------------------------------------- #

@dataclass
class _RawRecord:
    title: str = ""
    params: Dict[str, List[str]] = field(default_factory=dict)
    data_names: List[str] = field(default_factory=list)
    dim1: List[int] = field(default_factory=list)
    dim2: List[int] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    def has_data(self) -> bool:
        return bool(self.data_names) and bool(self.rows)


def _iter_raw_records(text: str):
    """Yield :class:`_RawRecord` objects, splitting at each ``SetupTitle`` line.

    Lines before the first ``SetupTitle`` (rare) seed an anonymous record.
    """
    rec = _RawRecord()
    started = False

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        fields = _split_fields(raw_line)
        key = fields[0]

        if key == "SetupTitle":
            if started and (rec.has_data() or rec.params):
                yield rec
            rec = _RawRecord(title=fields[1] if len(fields) > 1 else "")
            started = True
            continue

        started = True
        if key == "TestParameter":
            if len(fields) >= 2:
                rec.params[fields[1]] = fields[2:]
        elif key == "Dimension1":
            rec.dim1 = [int(_to_float(f)) for f in fields[1:] if f != ""]
        elif key == "Dimension2":
            rec.dim2 = [int(_to_float(f)) for f in fields[1:] if f != ""]
        elif key == "DataName":
            rec.data_names = [f for f in fields[1:]]
        elif key == "DataValue":
            rec.rows.append(fields[1:])
        # MetaData / AutoAnalysis / unknown keys are ignored on purpose

    if started and (rec.has_data() or rec.params):
        yield rec


# --------------------------------------------------------------------------- #
# Interpretation
# --------------------------------------------------------------------------- #

def _channel_role(params: Dict[str, List[str]], role: str) -> Optional[str]:
    """Return the V-name of the channel whose Func == role (VAR1 / VAR2)."""
    funcs = params.get("Channel.Func")
    vnames = params.get("Channel.VName")
    if not funcs or not vnames:
        return None
    for f, v in zip(funcs, vnames):
        if f.upper() == role.upper():
            return v
    return None


def _infer_axes(rec: _RawRecord, columns: Dict[str, np.ndarray]):
    """Determine (kind, x_name, secondary_name) from params + column names."""
    x_name = _channel_role(rec.params, "VAR1")
    secondary_name = _channel_role(rec.params, "VAR2")

    title = (rec.title or "").lower()
    # Title is the most reliable kind hint when present.
    if "output" in title:
        kind = MeasurementKind.OUTPUT
    elif "trans" in title:
        kind = MeasurementKind.TRANSFER
    else:
        kind = MeasurementKind.UNKNOWN

    # Fall back to column names to pick the sweep axis / kind.
    if not x_name:
        if kind == MeasurementKind.OUTPUT and "Vd" in columns:
            x_name = "Vd"
        elif kind == MeasurementKind.TRANSFER and "Vg" in columns:
            x_name = "Vg"
        elif "Vg" in columns:
            x_name = "Vg"
        elif "Vd" in columns:
            x_name = "Vd"
        else:
            x_name = rec.data_names[0] if rec.data_names else ""

    if kind == MeasurementKind.UNKNOWN:
        if x_name == "Vd":
            kind = MeasurementKind.OUTPUT
        elif x_name == "Vg":
            kind = MeasurementKind.TRANSFER

    # Secondary axis fallback: the other voltage if more than one curve exists.
    if not secondary_name:
        if kind == MeasurementKind.OUTPUT and "Vg" in columns:
            secondary_name = "Vg"
        elif kind == MeasurementKind.TRANSFER and "Vd" in columns:
            secondary_name = "Vd"

    return kind, x_name or "", secondary_name or ""


def _format_secondary_label(name: str, value: Optional[float]) -> str:
    if name == "" or value is None or np.isnan(value):
        return ""
    # Compact numeric formatting (drop trailing zeros).
    if abs(value) >= 1 or value == 0:
        txt = f"{value:g}"
    else:
        txt = f"{value:.3g}"
    return f"{name} = {txt} V"


def _build_measurement(rec: _RawRecord, source_file: str, index: int) -> Optional[Measurement]:
    if not rec.has_data():
        return None

    names = rec.data_names
    n_cols = len(names)
    # Coerce rows to a numeric matrix, padding/truncating to n_cols.
    mat = np.full((len(rec.rows), n_cols), np.nan, dtype=float)
    for i, row in enumerate(rec.rows):
        for j in range(min(n_cols, len(row))):
            mat[i, j] = _to_float(row[j])

    columns = {names[j]: mat[:, j] for j in range(n_cols)}

    kind, x_name, secondary_name = _infer_axes(rec, columns)

    # Points-per-curve / number-of-curves from Dimension fields (defensive).
    n_total = mat.shape[0]
    pts = rec.dim1[0] if rec.dim1 else n_total
    n_curves = rec.dim2[0] if rec.dim2 else 1
    if pts <= 0:
        pts = n_total
    if n_curves <= 0:
        n_curves = 1
    if pts * n_curves != n_total:
        # Trust the total: try to recover an integer split, else single curve.
        if n_total % max(pts, 1) == 0 and pts > 0:
            n_curves = n_total // pts
        else:
            pts, n_curves = n_total, 1

    sec_col = columns.get(secondary_name) if secondary_name else None

    curves: List[Curve] = []
    for c in range(n_curves):
        sl = slice(c * pts, (c + 1) * pts)
        cdata = {name: columns[name][sl] for name in names}
        sec_val = None
        if sec_col is not None:
            seg = sec_col[sl]
            seg = seg[~np.isnan(seg)]
            if seg.size:
                sec_val = float(np.median(seg))
        label = _format_secondary_label(secondary_name, sec_val)
        if not label:
            label = f"curve {c + 1}" if n_curves > 1 else (rec.title or "curve 1")
        curves.append(Curve(columns=cdata, label=label,
                            secondary_name=secondary_name, secondary_value=sec_val))

    return Measurement(
        kind=kind, title=rec.title, x_name=x_name,
        secondary_name=secondary_name if n_curves > 1 else "",
        curves=curves, params=rec.params,
        source_file=source_file, record_index=index,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _data_signature(m: Measurement) -> bytes:
    """Hash of a measurement's numeric content, used to drop duplicate records.

    B1500 exports frequently emit the same sweep twice (a raw record plus an
    auto-analysis copy, e.g. ``Trans.`` + ``2DFET_Transfer``).  Hashing the x
    column and every plotted-candidate column collapses those without merging
    genuinely different sweeps.
    """
    import hashlib

    h = hashlib.md5()
    h.update(f"{m.kind}|{m.x_name}|{m.n_curves}".encode())
    for c in m.curves:
        for name in (m.x_name, "Id", "Ig", "absId", "absIg"):
            arr = c.get(name)
            if arr is not None:
                h.update(np.nan_to_num(arr).astype("<f8").tobytes())
    return h.digest()


def read_b1500_csv(path: str, deduplicate: bool = True) -> List[Measurement]:
    """Parse one CSV file into a list of :class:`Measurement` records.

    When *deduplicate* is True (default), records with identical numeric
    content are collapsed to the first occurrence.
    """
    text = _read_text(path)
    out: List[Measurement] = []
    seen = set()
    idx = 0
    for rec in _iter_raw_records(text):
        m = _build_measurement(rec, path, idx)
        if m is None or not m.curves:
            continue
        if deduplicate:
            sig = _data_signature(m)
            if sig in seen:
                continue
            seen.add(sig)
        m.record_index = idx
        out.append(m)
        idx += 1
    return out


def _fmt_value(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(f):
        return ""
    return f"{f:.10g}"


def write_b1500_csv(path: str, measurement: "Measurement", get_column=None) -> None:
    """Write *measurement* back out in the B1500 CSV format it was read from.

    The file round-trips through :func:`read_b1500_csv`: same ``SetupTitle``,
    ``TestParameter`` block, ``Dimension``/``DataName`` headers and the curves
    concatenated as ``DataValue`` rows — so re-importing it reproduces the exact
    same plot with no further processing.

    *get_column(curve, name) -> array* lets the caller substitute post-processed
    values for a column; when omitted the raw column is written.
    """
    if get_column is None:
        def get_column(curve, name):
            return curve.get(name)

    curves = measurement.curves
    if not curves:
        return
    names = curves[0].names
    pts = int(np.asarray(get_column(curves[0], names[0])).size)

    lines: List[str] = [f"SetupTitle, {measurement.title}"]
    for key, vals in measurement.params.items():
        lines.append("TestParameter, " + key + (", " + ", ".join(vals) if vals else ", "))
    lines.append("Dimension1, " + ", ".join([str(pts)] * len(names)))
    lines.append("Dimension2, " + ", ".join([str(len(curves))] * len(names)))
    lines.append("DataName, " + ", ".join(names))
    for curve in curves:
        cols = [np.asarray(get_column(curve, n)) for n in names]
        npts = cols[0].size
        for i in range(npts):
            lines.append("DataValue, " + ", ".join(_fmt_value(c[i]) for c in cols))

    with open(path, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _looks_like_b1500(path: str) -> bool:
    if not path.lower().endswith(".csv"):
        return False
    try:
        with open(path, "r", encoding="latin-1", errors="ignore") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    return ("DataName" in head) or ("SetupTitle" in head) or ("PrimitiveTest" in head)


def load_folder(folder: str, recursive: bool = True) -> List[Measurement]:
    """Load every B1500 CSV under *folder*, sorted by filename then record order."""
    measurements: List[Measurement] = []
    walker = os.walk(folder) if recursive else [(folder, [], os.listdir(folder))]
    files: List[str] = []
    for root, _dirs, names in walker:
        for n in names:
            files.append(os.path.join(root, n))
    for path in sorted(files, key=lambda p: p.lower()):
        if _looks_like_b1500(path):
            try:
                measurements.extend(read_b1500_csv(path))
            except Exception:  # one bad file must not break the batch
                continue
    return measurements


if __name__ == "__main__":  # quick manual smoke test
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    ms = load_folder(target) if os.path.isdir(target) else read_b1500_csv(target)
    for m in ms:
        print(f"{m.name:60s} kind={m.kind:8s} x={m.x_name:3s} "
              f"sec={m.secondary_name or '-':3s} curves={m.n_curves} "
              f"cols={m.curves[0].names if m.curves else []}")
