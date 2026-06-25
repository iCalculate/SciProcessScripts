from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_sdiv_iv import assign_group_labels, parse_curve_file_multi


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "rawdata"
OUTPUT_DIR = ROOT / "device_iv_plots"

COLORS = {
    "Dark": "#4D4D4D",
    "Light": "#0F4D92",
    "1/4 Light": "#0F4D92",
    "1/2 Light": "#7C6CCF",
}


def load_device_curves(device_code: str):
    device_code = device_code.upper()
    grouped = defaultdict(list)

    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        for curve in parse_curve_file_multi(csv_path):
            if curve.device.upper() == device_code:
                grouped[curve.group_id].append(curve)

    selected = {}
    for group_id, curves in grouped.items():
        curves.sort(key=lambda item: (item.sort_index, item.record_time or "", item.source_file))
        kept = curves[:2]
        assign_group_labels(kept)
        selected[group_id] = kept

    return selected


def plot_orientation(ax, curves, orientation: str) -> None:
    if not curves:
        ax.text(0.5, 0.5, f"No {orientation} data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    for curve in curves:
        label = curve.curve_label
        color = COLORS.get(label, "#3775BA")
        linestyle = "--" if label == "Dark" else "-"
        ax.plot(curve.data["Vd"], curve.data["Id"] * 1e9, lw=1.8, color=color, ls=linestyle, label=label)

    ax.axhline(0, color="#D7D7D7", lw=0.9, zorder=0)
    ax.axvline(0, color="#D7D7D7", lw=0.9, zorder=0)
    ax.set_title(f"{orientation} orientation", fontsize=10)
    ax.set_xlabel(r"$V_d$ (V)")
    ax.set_ylabel(r"$I_d$ (nA)")
    ax.legend(fontsize=8, loc="best")


def make_figure(device_code: str, output_stem: Path | None = None) -> Path:
    selected = load_device_curves(device_code)
    if not selected:
        raise ValueError(f"No data found for device {device_code}")

    h_curves = selected.get(f"{device_code.upper()}-H", [])
    v_curves = selected.get(f"{device_code.upper()}-V", [])

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    plot_orientation(axes[0], h_curves, "H")
    plot_orientation(axes[1], v_curves, "V")

    fig.suptitle(f"{device_code.upper()} dark/light IV curves", fontsize=12, fontweight="bold")

    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = output_stem or (OUTPUT_DIR / f"{device_code.upper()}_iv")
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dark/light IV curves for a selected device RC code.")
    parser.add_argument("device", help="Device code, for example R7C2")
    parser.add_argument("--output", help="Optional output file stem, without extension", default=None)
    args = parser.parse_args()

    stem = Path(args.output) if args.output else None
    output_png = make_figure(args.device, stem)
    print(f"Saved figure to: {output_png}")


if __name__ == "__main__":
    main()
