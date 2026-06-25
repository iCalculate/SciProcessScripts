from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RAWDATA_DIR = ROOT / "rawdata"
OUTPUT_DIR = ROOT / "analysis_output"
PLOTS_DIR = OUTPUT_DIR / "plots"

TARGET_RE = re.compile(r"APMS2-(R\d+C\d+)-(H|V)")
INDEX_RE = re.compile(r"\((\d+)\)")


@dataclass
class CurveRecord:
    device: str
    orientation: str
    group_id: str
    source_file: str
    sort_index: int
    record_time: str | None
    remarks: str
    curve_label: str
    data: pd.DataFrame


def sanitize_label(remarks: str, fallback_index: int) -> str:
    text = (remarks or "").strip()
    if not text:
        return f"Curve {fallback_index}"
    return text.replace("/", "_")


def _finalize_curve_segment(
    *,
    path: Path,
    device: str,
    orientation: str,
    sort_index: int,
    record_time: str | None,
    remarks: str,
    rows: list[tuple[float, float, float, float]],
    segment_index: int,
) -> CurveRecord | None:
    if not rows:
        return None

    data = pd.DataFrame(rows, columns=["Vd", "Vg", "Ig", "Id"]).drop_duplicates()
    data = data.sort_values(["Vd", "Id"], kind="mergesort").reset_index(drop=True)

    return CurveRecord(
        device=device,
        orientation=orientation,
        group_id=f"{device}-{orientation}",
        source_file=f"{path.name}::segment{segment_index}",
        sort_index=sort_index * 10 + segment_index,
        record_time=record_time,
        remarks=remarks,
        curve_label="",
        data=data,
    )


def parse_curve_file_multi(path: Path) -> list[CurveRecord]:
    match = TARGET_RE.search(path.name)
    if not match:
        return []

    device, orientation = match.group(1), match.group(2)
    sort_match = INDEX_RE.search(path.name)
    sort_index = int(sort_match.group(1)) if sort_match else 10**9

    curves: list[CurveRecord] = []
    remarks = ""
    record_time = None
    rows: list[tuple[float, float, float, float]] = []
    segment_index = 0
    in_true_entry = False
    collecting_data = False

    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 3 and row[0] == "MetaData" and row[1].strip() == "TestRecord.EntryPoint":
                # Start of a new measurement block.
                entry_value = row[2].strip().lower()
                if rows:
                    curve = _finalize_curve_segment(
                        path=path,
                        device=device,
                        orientation=orientation,
                        sort_index=sort_index,
                        record_time=record_time,
                        remarks=remarks,
                        rows=rows,
                        segment_index=segment_index,
                    )
                    if curve is not None:
                        curves.append(curve)
                rows = []
                remarks = ""
                record_time = None
                collecting_data = False
                in_true_entry = entry_value == "true"
                if in_true_entry:
                    segment_index += 1
                continue

            if not in_true_entry:
                continue

            if len(row) >= 3 and row[0] == "MetaData" and row[1].strip() == "TestRecord.RecordTime":
                record_time = row[2].strip()
            elif len(row) >= 3 and row[0] == "MetaData" and row[1].strip() == "TestRecord.Remarks":
                remarks = row[2].strip()
            elif len(row) >= 5 and row[0] == "DataValue":
                collecting_data = True
                try:
                    rows.append(tuple(float(item.strip()) for item in row[1:5]))
                except ValueError:
                    continue
            elif collecting_data and row and row[0] != "DataValue":
                curve = _finalize_curve_segment(
                    path=path,
                    device=device,
                    orientation=orientation,
                    sort_index=sort_index,
                    record_time=record_time,
                    remarks=remarks,
                    rows=rows,
                    segment_index=segment_index,
                )
                if curve is not None:
                    curves.append(curve)
                rows = []
                remarks = ""
                record_time = None
                collecting_data = False
                in_true_entry = False

    if rows:
        curve = _finalize_curve_segment(
            path=path,
            device=device,
            orientation=orientation,
            sort_index=sort_index,
            record_time=record_time,
            remarks=remarks,
            rows=rows,
            segment_index=segment_index,
        )
        if curve is not None:
            curves.append(curve)

    return curves


def parse_curve_file(path: Path) -> CurveRecord | None:
    curves = parse_curve_file_multi(path)
    return curves[0] if curves else None


def classify_curve(curve: CurveRecord, fallback_index: int) -> str:
    remark = (curve.remarks or "").strip().lower()
    if "dark" in remark:
        return "Dark"
    if "light" in remark:
        return curve.remarks.strip()
    return f"Curve {fallback_index}"


def assign_group_labels(group_curves: list[CurveRecord]) -> None:
    explicit = any(("dark" in (curve.remarks or "").strip().lower()) or ("light" in (curve.remarks or "").strip().lower()) for curve in group_curves)
    if not explicit and len(group_curves) >= 2:
        group_curves[0].curve_label = "Dark"
        group_curves[1].curve_label = "Light"
        for idx, curve in enumerate(group_curves[2:], start=3):
            curve.curve_label = f"Curve {idx}"
        return

    for index, curve in enumerate(group_curves, start=1):
        curve.curve_label = classify_curve(curve, index)


def nearest_value(data: pd.DataFrame, target_vd: float) -> float:
    idx = (data["Vd"] - target_vd).abs().idxmin()
    return float(data.loc[idx, "Id"])


def fit_iv_segment(data: pd.DataFrame, v_min: float, v_max: float) -> tuple[float | None, float | None, int]:
    segment = data[(data["Vd"] >= v_min) & (data["Vd"] <= v_max)].copy()
    segment = segment.sort_values("Vd")
    if len(segment) < 3:
        return None, None, len(segment)

    x = segment["Vd"].to_numpy(dtype=float)
    y = segment["Id"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept), len(segment)


def slope_to_resistance(slope: float | None) -> float | None:
    if slope is None or abs(slope) < 1e-30:
        return None
    return 1.0 / slope


def log_abs(value: float) -> float:
    return math.log10(max(abs(value), 1e-15))


def ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if abs(b) < 1e-30:
        return None
    return abs(a) / abs(b)


def draw_line_plot(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    curves: list[CurveRecord],
    y_limit: float,
    colors: list[tuple[int, int, int]],
    labels: list[str],
) -> None:
    left, top, right, bottom = bbox
    pad_l, pad_t, pad_r, pad_b = 34, 16, 10, 24
    plot_box = (left + pad_l, top + pad_t, right - pad_r, bottom - pad_b)
    px0, py0, px1, py1 = plot_box
    width = max(px1 - px0, 1)
    height = max(py1 - py0, 1)

    draw.rectangle([left, top, right, bottom], outline=(210, 210, 210), width=1)
    draw.rectangle([px0, py0, px1, py1], outline=(180, 180, 180), width=1)

    # Zero axes.
    x_zero = px0 + width * 0.5
    y_zero = py0 + height * 0.5
    draw.line([px0, y_zero, px1, y_zero], fill=(220, 220, 220), width=1)
    draw.line([x_zero, py0, x_zero, py1], fill=(220, 220, 220), width=1)

    def x_map(v: float) -> float:
        return px0 + (v + 1.0) / 2.0 * width

    def y_map(i: float) -> float:
        return py1 - (i + y_limit) / (2.0 * y_limit) * height

    for curve, color in zip(curves, colors):
        pts = [(x_map(float(v)), y_map(float(i))) for v, i in zip(curve.data["Vd"], curve.data["Id"])]
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=2)

    font = ImageFont.load_default()
    draw.text((left + 4, top + 3), curves[0].group_id, fill=(20, 20, 20), font=font)
    draw.text((px0 - 6, py0 - 6), f"{y_limit*1e9:.1f} nA", fill=(90, 90, 90), font=font, anchor="ra")
    draw.text((px0 - 6, py1 - 2), f"-{y_limit*1e9:.1f} nA", fill=(90, 90, 90), font=font, anchor="ra")
    draw.text((px0, py1 + 4), "-1 V", fill=(90, 90, 90), font=font)
    draw.text((px1, py1 + 4), "+1 V", fill=(90, 90, 90), font=font, anchor="ra")

    legend_y = top + 3
    for idx, (label, color) in enumerate(zip(labels, colors)):
        lx = right - 88
        ly = legend_y + idx * 11
        draw.line([lx, ly + 5, lx + 12, ly + 5], fill=color, width=2)
        draw.text((lx + 16, ly), label[:11], fill=(50, 50, 50), font=font)


def draw_heatmap(
    matrix: pd.DataFrame,
    title: str,
    output_path: Path,
    value_formatter,
    color_mode: str = "diverging",
) -> None:
    cell_w, cell_h = 120, 56
    margin_left, margin_top, margin_right, margin_bottom = 84, 70, 30, 48
    rows, cols = matrix.shape
    img = Image.new(
        "RGB",
        (margin_left + cols * cell_w + margin_right, margin_top + rows * cell_h + margin_bottom),
        "white",
    )
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((margin_left, 18), title, fill=(10, 10, 10), font=font)
    valid = matrix.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    finite = valid[np.isfinite(valid)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if math.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0

    def cell_color(val: float | None) -> tuple[int, int, int]:
        if val is None or not math.isfinite(val):
            return (242, 242, 242)
        if color_mode == "diverging":
            max_abs = max(abs(vmin), abs(vmax), 1e-12)
            norm = max(-1.0, min(1.0, val / max_abs))
            if norm >= 0:
                mix = norm
                return (
                    int(240 - 55 * mix),
                    int(245 - 160 * mix),
                    int(255 - 160 * mix),
                )
            mix = -norm
            return (
                int(255 - 50 * mix),
                int(240 - 180 * mix),
                int(240 - 55 * mix),
            )
        norm = (val - vmin) / (vmax - vmin)
        return (
            int(245 - 125 * norm),
            int(245 - 60 * norm),
            int(255 - 170 * norm),
        )

    for c, label in enumerate(matrix.columns):
        draw.text((margin_left + c * cell_w + cell_w / 2, margin_top - 24), str(label), fill=(30, 30, 30), font=font, anchor="mm")
    for r, label in enumerate(matrix.index):
        draw.text((margin_left - 26, margin_top + r * cell_h + cell_h / 2), str(label), fill=(30, 30, 30), font=font, anchor="mm")
        for c in range(cols):
            x0 = margin_left + c * cell_w
            y0 = margin_top + r * cell_h
            x1 = x0 + cell_w - 2
            y1 = y0 + cell_h - 2
            value = matrix.iloc[r, c]
            fill = cell_color(None if pd.isna(value) else float(value))
            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(220, 220, 220))
            text = value_formatter(value)
            draw.text((x0 + cell_w / 2, y0 + cell_h / 2), text, fill=(20, 20, 20), font=font, anchor="mm")

    img.save(output_path)


def build_panel_figure(selected: dict[str, list[CurveRecord]], output_path: Path) -> None:
    ordered_groups = sorted(selected)
    cols = 4
    rows = math.ceil(len(ordered_groups) / cols)
    panel_w, panel_h = 320, 220
    img = Image.new("RGB", (cols * panel_w + 20, rows * panel_h + 20), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((12, 6), "SDIV IV curves (up to first 2 records per device/orientation)", fill=(10, 10, 10), font=font)

    all_ids = []
    for curves in selected.values():
        for curve in curves:
            all_ids.extend(curve.data["Id"].tolist())
    y_limit = max(np.quantile(np.abs(all_ids), 0.98), 1e-12)
    y_limit = float(math.ceil(y_limit * 1e9) / 1e9)

    palette = [(35, 99, 180), (208, 64, 58)]
    for idx, group_id in enumerate(ordered_groups):
        row, col = divmod(idx, cols)
        left = 10 + col * panel_w
        top = 24 + row * panel_h
        curves = selected[group_id]
        labels = [curve.curve_label for curve in curves]
        draw_line_plot(draw, (left, top, left + panel_w - 12, top + panel_h - 12), curves, y_limit, palette[: len(curves)], labels)

    img.save(output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    curves: list[CurveRecord] = []
    for path in sorted(RAWDATA_DIR.glob("*.csv")):
        curves.extend(parse_curve_file_multi(path))

    groups: dict[str, list[CurveRecord]] = {}
    for curve in curves:
        groups.setdefault(curve.group_id, []).append(curve)

    selected: dict[str, list[CurveRecord]] = {}
    omitted_rows: list[dict[str, object]] = []
    for group_id, group_curves in groups.items():
        group_curves.sort(key=lambda item: (item.sort_index, item.record_time or "", item.source_file))
        kept = group_curves[:2]
        assign_group_labels(kept)
        selected[group_id] = kept
        for omitted in group_curves[2:]:
            omitted_rows.append(
                {
                    "group_id": group_id,
                    "source_file": omitted.source_file,
                    "remarks": omitted.remarks,
                    "sort_index": omitted.sort_index,
                    "record_time": omitted.record_time,
                }
            )

    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for group_id, kept in sorted(selected.items()):
        curve_map = {}
        for curve in kept:
            id_pos_1v = nearest_value(curve.data, 1.0)
            id_neg_1v = nearest_value(curve.data, -1.0)
            max_abs = float(curve.data["Id"].abs().max())
            slope_pos, intercept_pos, n_pos = fit_iv_segment(curve.data, 0.0, 1.0)
            slope_neg, intercept_neg, n_neg = fit_iv_segment(curve.data, -1.0, 0.0)
            curve_map[curve.curve_label.lower()] = curve
            curve_rows.append(
                {
                    "group_id": group_id,
                    "device": curve.device,
                    "orientation": curve.orientation,
                    "curve_label": curve.curve_label,
                    "remarks": curve.remarks,
                    "source_file": curve.source_file,
                    "sort_index": curve.sort_index,
                    "record_time": curve.record_time,
                    "points": len(curve.data),
                    "id_at_+1V_A": id_pos_1v,
                    "id_at_-1V_A": id_neg_1v,
                    "abs_id_at_+1V_A": abs(id_pos_1v),
                    "abs_id_at_-1V_A": abs(id_neg_1v),
                    "max_abs_id_A": max_abs,
                    "slope_0_to_1_A_per_V": slope_pos,
                    "intercept_0_to_1_A": intercept_pos,
                    "fit_points_0_to_1": n_pos,
                    "resistance_0_to_1_ohm": slope_to_resistance(slope_pos),
                    "slope_-1_to_0_A_per_V": slope_neg,
                    "intercept_-1_to_0_A": intercept_neg,
                    "fit_points_-1_to_0": n_neg,
                    "resistance_-1_to_0_ohm": slope_to_resistance(slope_neg),
                }
            )

        dark_curve = next((curve for curve in kept if curve.curve_label.lower() == "dark"), None)
        light_curve = next((curve for curve in kept if "light" in curve.curve_label.lower()), None)
        paired_light_dark = dark_curve is not None and light_curve is not None
        curve1 = kept[0] if kept else None
        curve2 = kept[1] if len(kept) > 1 else None
        summary_rows.append(
            {
                "group_id": group_id,
                "device": kept[0].device,
                "orientation": kept[0].orientation,
                "num_available_records": len(groups[group_id]),
                "num_used_records": len(kept),
                "curve_1_label": curve1.curve_label if curve1 else None,
                "curve_2_label": curve2.curve_label if curve2 else None,
                "curve_1_abs_id_+1V_nA": abs(nearest_value(curve1.data, 1.0)) * 1e9 if curve1 else None,
                "curve_1_abs_id_-1V_nA": abs(nearest_value(curve1.data, -1.0)) * 1e9 if curve1 else None,
                "curve_2_abs_id_+1V_nA": abs(nearest_value(curve2.data, 1.0)) * 1e9 if curve2 else None,
                "curve_2_abs_id_-1V_nA": abs(nearest_value(curve2.data, -1.0)) * 1e9 if curve2 else None,
                "curve_1_slope_0_to_1_A_per_V": fit_iv_segment(curve1.data, 0.0, 1.0)[0] if curve1 else None,
                "curve_1_slope_-1_to_0_A_per_V": fit_iv_segment(curve1.data, -1.0, 0.0)[0] if curve1 else None,
                "curve_2_slope_0_to_1_A_per_V": fit_iv_segment(curve2.data, 0.0, 1.0)[0] if curve2 else None,
                "curve_2_slope_-1_to_0_A_per_V": fit_iv_segment(curve2.data, -1.0, 0.0)[0] if curve2 else None,
                "curve2_to_curve1_ratio_+1V": ratio(
                    nearest_value(curve2.data, 1.0) if curve2 else None,
                    nearest_value(curve1.data, 1.0) if curve1 else None,
                ),
                "curve2_to_curve1_ratio_-1V": ratio(
                    nearest_value(curve2.data, -1.0) if curve2 else None,
                    nearest_value(curve1.data, -1.0) if curve1 else None,
                ),
                "curve2_to_curve1_slope_ratio_0_to_1": ratio(
                    fit_iv_segment(curve2.data, 0.0, 1.0)[0] if curve2 else None,
                    fit_iv_segment(curve1.data, 0.0, 1.0)[0] if curve1 else None,
                ),
                "curve2_to_curve1_slope_ratio_-1_to_0": ratio(
                    fit_iv_segment(curve2.data, -1.0, 0.0)[0] if curve2 else None,
                    fit_iv_segment(curve1.data, -1.0, 0.0)[0] if curve1 else None,
                ),
                "paired_dark_light": paired_light_dark,
                "light_to_dark_ratio_+1V": ratio(
                    nearest_value(light_curve.data, 1.0) if light_curve else None,
                    nearest_value(dark_curve.data, 1.0) if dark_curve else None,
                ),
                "light_to_dark_ratio_-1V": ratio(
                    nearest_value(light_curve.data, -1.0) if light_curve else None,
                    nearest_value(dark_curve.data, -1.0) if dark_curve else None,
                ),
                "light_to_dark_slope_ratio_0_to_1": ratio(
                    fit_iv_segment(light_curve.data, 0.0, 1.0)[0] if light_curve else None,
                    fit_iv_segment(dark_curve.data, 0.0, 1.0)[0] if dark_curve else None,
                ),
                "light_to_dark_slope_ratio_-1_to_0": ratio(
                    fit_iv_segment(light_curve.data, -1.0, 0.0)[0] if light_curve else None,
                    fit_iv_segment(dark_curve.data, -1.0, 0.0)[0] if dark_curve else None,
                ),
            }
        )

    curve_df = pd.DataFrame(curve_rows).sort_values(["device", "orientation", "curve_label"])
    summary_df = pd.DataFrame(summary_rows).sort_values(["device", "orientation"])
    omitted_df = pd.DataFrame(omitted_rows).sort_values(["group_id", "sort_index"]) if omitted_rows else pd.DataFrame(columns=["group_id", "source_file", "remarks", "sort_index", "record_time"])

    curve_df.to_csv(OUTPUT_DIR / "curve_metrics.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "device_summary.csv", index=False)
    omitted_df.to_csv(OUTPUT_DIR / "omitted_records.csv", index=False)

    build_panel_figure(selected, PLOTS_DIR / "iv_panels.png")
    paired_selected = {
        group_id: curves_for_group
        for group_id, curves_for_group in selected.items()
        if len(curves_for_group) == 2
    }
    if paired_selected:
        build_panel_figure(paired_selected, PLOTS_DIR / "paired_iv_panels.png")

    curve1_plus = summary_df.assign(row=summary_df["device"].str.extract(r"R(\d+)")[0].astype(int), col=summary_df["device"].str.extract(r"C(\d+)")[0].astype(int))
    curve1_plus["curve_1_logabs_+1V"] = curve1_plus["curve_1_abs_id_+1V_nA"].apply(lambda x: np.log10(max(x, 1e-6)) if pd.notna(x) else np.nan)
    curve1_minus = curve1_plus.copy()
    curve1_minus["curve_1_logabs_-1V"] = curve1_minus["curve_1_abs_id_-1V_nA"].apply(lambda x: np.log10(max(x, 1e-6)) if pd.notna(x) else np.nan)

    def pivot_matrix(frame: pd.DataFrame, value_col: str, orientation: str) -> pd.DataFrame:
        subset = frame[frame["orientation"] == orientation]
        matrix = subset.pivot(index="row", columns="col", values=value_col)
        return matrix.sort_index().sort_index(axis=1)

    for orientation in ["H", "V"]:
        plus_matrix = pivot_matrix(curve1_plus, "curve_1_logabs_+1V", orientation)
        minus_matrix = pivot_matrix(curve1_minus, "curve_1_logabs_-1V", orientation)
        ratio_matrix = pivot_matrix(curve1_plus, "light_to_dark_ratio_+1V", orientation)

        draw_heatmap(
            plus_matrix,
            f"{orientation}: log10(|I| at +1V) of curve 1 [nA]",
            PLOTS_DIR / f"{orientation.lower()}_curve1_logabs_plus1V.png",
            value_formatter=lambda v: "" if pd.isna(v) else f"{10 ** float(v):.2f}",
            color_mode="sequential",
        )
        draw_heatmap(
            minus_matrix,
            f"{orientation}: log10(|I| at -1V) of curve 1 [nA]",
            PLOTS_DIR / f"{orientation.lower()}_curve1_logabs_minus1V.png",
            value_formatter=lambda v: "" if pd.isna(v) else f"{10 ** float(v):.2f}",
            color_mode="sequential",
        )
        draw_heatmap(
            ratio_matrix,
            f"{orientation}: light/dark ratio at +1V",
            PLOTS_DIR / f"{orientation.lower()}_light_dark_ratio_plus1V.png",
            value_formatter=lambda v: "" if pd.isna(v) else f"{float(v):.2f}",
            color_mode="sequential",
        )

    one_curve = int((summary_df["num_used_records"] == 1).sum())
    two_curves = int((summary_df["num_used_records"] == 2).sum())
    over_two = int((summary_df["num_available_records"] > 2).sum())
    paired = int(summary_df["paired_dark_light"].fillna(False).sum())

    report_lines = [
        "# SDIV rawdata analysis",
        "",
        f"- Total CSV files parsed: {len(curves)}",
        f"- Unique device-orientation groups: {len(summary_df)}",
        f"- Groups with 1 usable curve: {one_curve}",
        f"- Groups with 2 usable curves: {two_curves}",
        f"- Groups truncated to first 2 records: {over_two}",
        f"- Groups with explicit dark/light pairing: {paired}",
        "",
        "## Notes",
        "",
        "- Each CSV was parsed from the first contiguous `DataValue` block only, which avoids counting the duplicated second block inside a file.",
        "- Records were grouped by `R?C?-H/V` and sorted by the sequence number in the filename, then only the first two records were kept.",
        "- `curve_metrics.csv` contains per-curve current metrics; `device_summary.csv` contains per-device summaries and pair ratios.",
        "- Heatmaps use curve 1 for coverage across the whole map because many devices only have one retained curve.",
    ]

    paired_top = summary_df[summary_df["paired_dark_light"] & summary_df["light_to_dark_ratio_+1V"].notna()].copy()
    if not paired_top.empty:
        paired_top = paired_top.sort_values("light_to_dark_ratio_+1V", ascending=False).head(5)
        report_lines.extend(["", "## Strongest light response at +1V", ""])
        for _, row in paired_top.iterrows():
            report_lines.append(
                f"- {row['group_id']}: light/dark ratio = {row['light_to_dark_ratio_+1V']:.2f}"
            )

    incomplete = summary_df[summary_df["num_used_records"] == 1]["group_id"].tolist()
    if incomplete:
        report_lines.extend(["", "## Single-curve groups", "", f"- {', '.join(incomplete)}"])

    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
