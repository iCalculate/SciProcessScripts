from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path

import numpy as np

from devicecurvegen.physics import generate_curves
from devicecurvegen.residual import ResidualEngine
from devicecurvegen.schemas import GenerationCondition


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = max(
    (path for path in (ROOT / "experiments").glob("postfix-model-sweep-*") if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
)

DEFAULT_MODELS = [
    "active_hybrid_patched_projection",
    "subset_threshold_conditional_pca16_focus10",
    "subset_conditional_pca20_ridge02",
    "subset_cvae_jumpfocus12",
]

COLORS = {
    "active_hybrid_patched_projection": "#1769ff",
    "subset_threshold_conditional_pca16_focus10": "#c4375a",
    "subset_conditional_pca20_ridge02": "#1b8f6a",
    "subset_cvae_jumpfocus12": "#d97706",
    "active_hybrid_onstate_retuned": "#4b5563",
    "active_hybrid_reverse_vth_align050": "#1769ff",
    "reverse_local_vth_align_1.00_w2.5": "#0f9d58",
    "reverse_local_vth_align_1.00_w3.0": "#d97706",
}


def _canonical_condition(gate_on: bool) -> GenerationCondition:
    return GenerationCondition(
        target_ion=1e-5,
        target_ioff=1e-15,
        target_vth=0.0,
        target_ss_mv_dec=230.0,
        hysteresis_v=1.5,
        ai_residual_strength=1.0,
        gate_ai_residual_strength=1.0 if gate_on else 0.0,
        diversity=0.65,
        seed=12345,
        points=601,
        variants=1,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        output_noise_gain=0.0,
        gate_leakage_a=1e-15,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
    )


def _load_summary(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return [item for item in payload if isinstance(item, dict)]


def _entry_by_name(entries: list[dict], name: str) -> dict:
    for entry in entries:
        if entry.get("name") == name:
            return entry
    raise ValueError(f"Could not find model entry named {name}")


def _label(name: str) -> str:
    if name == "active_hybrid_patched_projection":
        return "Active hybrid patched"
    if name == "active_hybrid_onstate_retuned":
        return "Active hybrid baseline"
    if name == "active_hybrid_reverse_vth_align050":
        return "Reverse global Vth align"
    if name == "reverse_local_vth_align_1.00_w2.5":
        return "Reverse local Vth align w2.5"
    if name == "reverse_local_vth_align_1.00_w3.0":
        return "Reverse local Vth align w3.0"
    if name == "subset_threshold_conditional_pca16_focus10":
        return "Best canonical threshold PCA"
    if name == "subset_conditional_pca20_ridge02":
        return "Best general conditional PCA"
    if name == "subset_cvae_jumpfocus12":
        return "Best CVAE jump-focus"
    return name.replace("_", " ")


def _format_metric(value: float | None, digits: int = 4, suffix: str = "") -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}{suffix}"


def _transfer_map(
    voltage: np.ndarray,
    current: np.ndarray,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    safe_current = np.clip(np.asarray(current, dtype=float), np.finfo(float).tiny, None)
    log_current = np.log10(safe_current)
    log_min = math.log10(y_min)
    log_max = math.log10(y_max)
    points: list[tuple[float, float]] = []
    for x_value, log_value in zip(np.asarray(voltage, dtype=float), log_current, strict=True):
        x = left + ((float(x_value) - x_min) / (x_max - x_min)) * width
        y = top + (1.0 - ((float(log_value) - log_min) / (log_max - log_min))) * height
        points.append((x, y))
    return points


def _linear_map(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x_value, y_value in zip(np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float), strict=True):
        x = left + ((float(x_value) - x_min) / (x_max - x_min)) * width
        y = top + (1.0 - ((float(y_value) - y_min) / (y_max - y_min))) * height
        points.append((x, y))
    return points


def _path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    commands = [f"M {points[0][0]:.2f} {points[0][1]:.2f}"]
    commands.extend(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    return " ".join(commands)


def _polyline(points: list[tuple[float, float]], color: str, width: float, dash: str | None = None) -> str:
    if not points:
        return ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<path d="{_path(points)}" fill="none" stroke="{color}" '
        f'stroke-width="{width:.2f}" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
    )


def _text(x: float, y: float, value: str, *, size: int = 12, weight: int = 400, fill: str = "#24364d") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size}" '
        f'font-weight="{weight}" font-family="Segoe UI, Arial, sans-serif">{escape(value)}</text>'
    )


def _step_profile(forward: np.ndarray, reverse: np.ndarray) -> np.ndarray:
    log_forward = np.log10(np.clip(np.asarray(forward, dtype=float), np.finfo(float).tiny, None))
    log_reverse = np.log10(np.clip(np.asarray(reverse, dtype=float), np.finfo(float).tiny, None))
    return np.maximum(np.abs(np.diff(log_forward)), np.abs(np.diff(log_reverse)))


def render(experiment_dir: Path, model_names: list[str], output: Path) -> Path:
    summary_path = experiment_dir / "summary.json"
    entries = _load_summary(summary_path)

    plot_x_min = -2.0
    plot_x_max = 2.0
    ids_y_min = 1e-15
    ids_y_max = 4e-5
    step_y_min = 0.0
    step_y_max = 0.45

    width = 1520
    height = 960
    transfer_box = (72.0, 132.0, 900.0, 390.0)
    step_box = (72.0, 598.0, 900.0, 210.0)
    side_box = (1022.0, 132.0, 424.0, 676.0)

    rendered: list[dict[str, object]] = []
    physics_forward: np.ndarray | None = None
    physics_reverse: np.ndarray | None = None
    physics_voltage: np.ndarray | None = None

    for name in model_names:
        entry = _entry_by_name(entries, name)
        result = entry.get("result", {})
        checkpoint_output = result.get("output") or entry.get("checkpoint_path")
        if not checkpoint_output:
            raise ValueError(f"Model {name} is missing a checkpoint output path")
        checkpoint = Path(str(checkpoint_output))
        if not checkpoint.is_absolute():
            checkpoint = (ROOT / checkpoint).resolve()
        engine = ResidualEngine(checkpoint)
        condition = _canonical_condition("Ig" in engine.info().generated_channels)
        candidate = generate_curves(condition, engine).candidates[0]
        voltage = np.asarray(candidate.voltage, dtype=float)
        forward = np.asarray(candidate.forward_current, dtype=float)
        reverse = np.asarray(candidate.reverse_current, dtype=float)
        if physics_forward is None:
            physics_forward = np.asarray(candidate.physics_forward_current, dtype=float)
            physics_reverse = np.asarray(candidate.physics_reverse_current, dtype=float)
            physics_voltage = voltage

        step_voltage = 0.5 * (voltage[1:] + voltage[:-1])
        step_profile = _step_profile(forward, reverse)

        rendered.append(
            {
                "name": name,
                "label": _label(name),
                "color": COLORS.get(name, "#1769ff"),
                "voltage": voltage,
                "forward": forward,
                "reverse": reverse,
                "step_voltage": step_voltage,
                "step_profile": step_profile,
                "canonical_jump_max": entry.get("canonical_metrics", {}).get("canonical_jump_max_decades"),
                "heldout_jump_p95": entry.get("jump_metrics", {}).get("jump_p95_decades"),
                "weighted_rmse": result.get("validation_weighted_rmse_decades"),
                "generated_vth_mae": entry.get("jump_metrics", {}).get("generated_vth_mae_v"),
                "generated_ss_mae": entry.get("jump_metrics", {}).get("generated_ss_mae_mv_dec"),
            }
        )

    if physics_forward is None or physics_reverse is None or physics_voltage is None:
        raise ValueError("No model traces were rendered")

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" fill="none">',
        '<rect width="100%" height="100%" fill="#f5f8fc"/>',
        '<rect x="36" y="36" width="1448" height="888" rx="24" fill="#ffffff" stroke="#dce5ef"/>',
        _text(72, 72, "Canonical 100% AI model comparison", size=26, weight=700, fill="#163250"),
        _text(
            72,
            102,
            "Same device condition, same seed, 100% AI blend. Solid = forward, dashed = reverse, lower panel = worst local log-step.",
            size=13,
            fill="#5a708a",
        ),
    ]

    def draw_axes(box: tuple[float, float, float, float], x_ticks: list[float], y_ticks: list[float], *, y_log: bool, y_label: str, x_label: str) -> None:
        left, top, box_width, box_height = box
        svg.append(f'<rect x="{left:.2f}" y="{top:.2f}" width="{box_width:.2f}" height="{box_height:.2f}" rx="14" fill="#fbfdff" stroke="#dfe8f1"/>')
        for tick in x_ticks:
            x = left + ((tick - plot_x_min) / (plot_x_max - plot_x_min)) * box_width
            svg.append(f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + box_height:.2f}" stroke="#e9eff6" stroke-width="1"/>')
            svg.append(_text(x - 10, top + box_height + 24, f"{tick:g}", size=11, fill="#6b7e93"))
        if y_log:
            log_min = math.log10(ids_y_min)
            log_max = math.log10(ids_y_max)
            for tick in y_ticks:
                y = top + (1.0 - ((math.log10(tick) - log_min) / (log_max - log_min))) * box_height
                svg.append(f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + box_width:.2f}" y2="{y:.2f}" stroke="#e9eff6" stroke-width="1"/>')
                svg.append(_text(left - 56, y + 4, f"1e{int(round(math.log10(tick)))}", size=11, fill="#6b7e93"))
        else:
            for tick in y_ticks:
                y = top + (1.0 - ((tick - step_y_min) / (step_y_max - step_y_min))) * box_height
                svg.append(f'<line x1="{left:.2f}" y1="{y:.2f}" x2="{left + box_width:.2f}" y2="{y:.2f}" stroke="#e9eff6" stroke-width="1"/>')
                svg.append(_text(left - 42, y + 4, f"{tick:.2f}", size=11, fill="#6b7e93"))
        zero_x = left + ((0.0 - plot_x_min) / (plot_x_max - plot_x_min)) * box_width
        svg.append(f'<line x1="{zero_x:.2f}" y1="{top:.2f}" x2="{zero_x:.2f}" y2="{top + box_height:.2f}" stroke="#9bb0c5" stroke-width="1.2" stroke-dasharray="4 4"/>')
        svg.append(_text(left + box_width * 0.5 - 58, top + box_height + 48, x_label, size=12, weight=600, fill="#315073"))
        svg.append(
            f'<text x="{left - 76:.2f}" y="{top + box_height * 0.5:.2f}" fill="#315073" font-size="12" font-weight="600" '
            f'font-family="Segoe UI, Arial, sans-serif" transform="rotate(-90 {left - 76:.2f} {top + box_height * 0.5:.2f})">{escape(y_label)}</text>'
        )

    draw_axes(
        transfer_box,
        x_ticks=[-2, -1, 0, 1, 2],
        y_ticks=[1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5],
        y_log=True,
        y_label="|Ids| (A)",
        x_label="Gate voltage Vg (V)",
    )
    draw_axes(
        step_box,
        x_ticks=[-2, -1, 0, 1, 2],
        y_ticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        y_log=False,
        y_label="max local log-step (dec)",
        x_label="Gate voltage midpoint (V)",
    )

    svg.append(_text(78, 124, "Transfer curve near Vth", size=16, weight=700, fill="#1e3b5c"))
    svg.append(_text(78, 590, "Threshold-region jump profile", size=16, weight=700, fill="#1e3b5c"))

    baseline_forward = _transfer_map(
        physics_voltage,
        physics_forward,
        left=transfer_box[0],
        top=transfer_box[1],
        width=transfer_box[2],
        height=transfer_box[3],
        x_min=plot_x_min,
        x_max=plot_x_max,
        y_min=ids_y_min,
        y_max=ids_y_max,
    )
    baseline_reverse = _transfer_map(
        physics_voltage,
        physics_reverse,
        left=transfer_box[0],
        top=transfer_box[1],
        width=transfer_box[2],
        height=transfer_box[3],
        x_min=plot_x_min,
        x_max=plot_x_max,
        y_min=ids_y_min,
        y_max=ids_y_max,
    )
    svg.append(_polyline(baseline_forward, "#9aa8b8", 1.6, "7 6"))
    svg.append(_polyline(baseline_reverse, "#c1ccd8", 1.1, "3 5"))

    for item in rendered:
        color = str(item["color"])
        svg.append(
            _polyline(
                _transfer_map(
                    item["voltage"],
                    item["forward"],
                    left=transfer_box[0],
                    top=transfer_box[1],
                    width=transfer_box[2],
                    height=transfer_box[3],
                    x_min=plot_x_min,
                    x_max=plot_x_max,
                    y_min=ids_y_min,
                    y_max=ids_y_max,
                ),
                color,
                2.9,
            )
        )
        svg.append(
            _polyline(
                _transfer_map(
                    item["voltage"],
                    item["reverse"],
                    left=transfer_box[0],
                    top=transfer_box[1],
                    width=transfer_box[2],
                    height=transfer_box[3],
                    x_min=plot_x_min,
                    x_max=plot_x_max,
                    y_min=ids_y_min,
                    y_max=ids_y_max,
                ),
                color,
                1.8,
                "6 5",
            )
        )
        svg.append(
            _polyline(
                _linear_map(
                    item["step_voltage"],
                    np.clip(item["step_profile"], step_y_min, step_y_max),
                    left=step_box[0],
                    top=step_box[1],
                    width=step_box[2],
                    height=step_box[3],
                    x_min=plot_x_min,
                    x_max=plot_x_max,
                    y_min=step_y_min,
                    y_max=step_y_max,
                ),
                color,
                2.5,
            )
        )

    side_left, side_top, side_width, side_height = side_box
    svg.append(f'<rect x="{side_left:.2f}" y="{side_top:.2f}" width="{side_width:.2f}" height="{side_height:.2f}" rx="18" fill="#f8fbff" stroke="#dce5ef"/>')
    svg.append(_text(side_left + 22, side_top + 32, "What this figure answers", size=17, weight=700, fill="#1e3b5c"))
    svg.append(_text(side_left + 22, side_top + 60, "Does 100% AI still create an unnatural threshold jump?", size=13, fill="#5a708a"))
    svg.append(_text(side_left + 22, side_top + 92, "Condition", size=12, weight=700, fill="#315073"))
    svg.append(_text(side_left + 22, side_top + 116, "Ion 1e-5 A, Ioff 1e-15 A, Vth 0.0 V, SS 230 mV/dec", size=12, fill="#4d647d"))
    svg.append(_text(side_left + 22, side_top + 138, "Hysteresis 1.5 V, diversity 65%, seed 12345, AI balance 100%", size=12, fill="#4d647d"))

    legend_y = side_top + 176
    svg.append(_text(side_left + 22, legend_y, "Legend", size=12, weight=700, fill="#315073"))
    svg.append(f'<line x1="{side_left + 22:.2f}" y1="{legend_y + 20:.2f}" x2="{side_left + 60:.2f}" y2="{legend_y + 20:.2f}" stroke="#9aa8b8" stroke-width="2" stroke-dasharray="7 6"/>')
    svg.append(_text(side_left + 70, legend_y + 24, "physics baseline", size=12, fill="#4d647d"))
    svg.append(f'<line x1="{side_left + 22:.2f}" y1="{legend_y + 44:.2f}" x2="{side_left + 60:.2f}" y2="{legend_y + 44:.2f}" stroke="#1769ff" stroke-width="3"/>')
    svg.append(_text(side_left + 70, legend_y + 48, "solid = forward, dashed = reverse", size=12, fill="#4d647d"))

    block_y = side_top + 264
    for item in rendered:
        color = str(item["color"])
        svg.append(f'<rect x="{side_left + 22:.2f}" y="{block_y - 14:.2f}" width="{side_width - 44:.2f}" height="102" rx="14" fill="#ffffff" stroke="#dde6ef"/>')
        svg.append(f'<circle cx="{side_left + 40:.2f}" cy="{block_y + 6:.2f}" r="6" fill="{color}"/>')
        svg.append(_text(side_left + 56, block_y + 11, str(item["label"]), size=14, weight=700, fill="#1f3b5a"))
        svg.append(_text(side_left + 32, block_y + 36, f"canonical jump max: {_format_metric(item['canonical_jump_max'], 4)} dec", size=12, fill="#425972"))
        svg.append(_text(side_left + 32, block_y + 58, f"held-out jump P95: {_format_metric(item['heldout_jump_p95'], 4)} dec", size=12, fill="#425972"))
        svg.append(_text(side_left + 32, block_y + 80, f"weighted RMSE: {_format_metric(item['weighted_rmse'], 4)} dec", size=12, fill="#425972"))
        svg.append(_text(side_left + 230, block_y + 36, f"generated Vth MAE: {_format_metric(item['generated_vth_mae'], 3, ' V')}", size=12, fill="#425972"))
        svg.append(_text(side_left + 230, block_y + 58, f"generated SS MAE: {_format_metric(item['generated_ss_mae'], 1, ' mV/dec')}", size=12, fill="#425972"))
        local_peak = float(np.max(np.asarray(item["step_profile"], dtype=float)))
        svg.append(_text(side_left + 230, block_y + 80, f"peak local log-step: {_format_metric(local_peak, 4)} dec", size=12, fill="#425972"))
        block_y += 118

    svg.append(_text(72, 870, f"Source summary: {summary_path}", size=11, fill="#6f8398"))
    svg.append(_text(72, 892, f"Output generated by {Path(__file__).name}", size=11, fill="#6f8398"))
    svg.append("</svg>")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a post-patch canonical model comparison SVG.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR,
        help="Experiment directory containing summary.json and model checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "canonical-model-comparison.svg",
        help="Output SVG path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model entry names to render from summary.json.",
    )
    args = parser.parse_args()
    output = render(
        experiment_dir=args.experiment_dir.expanduser().resolve(),
        model_names=list(args.models),
        output=args.output.expanduser().resolve(),
    )
    print(output)


if __name__ == "__main__":
    main()
