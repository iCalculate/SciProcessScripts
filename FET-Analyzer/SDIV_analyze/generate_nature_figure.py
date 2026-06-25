from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from analyze_sdiv_iv import parse_curve_file


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "rawdata"
ANALYSIS_DIR = ROOT / "analysis_output"
FIG_DIR = ANALYSIS_DIR / "nature_figure"
SOURCE_DIR = FIG_DIR / "source_data"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
    "teal": "#42949E",
    "lilac": "#B9A7E8",
    "violet": "#7C6CCF",
    "gain": "#2E9E44",
}


def apply_publication_style(font_size: int = 8, axes_linewidth: float = 0.8) -> None:
    plt.rcParams["font.size"] = font_size
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = axes_linewidth
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = axes_linewidth
    plt.rcParams["ytick.major.width"] = axes_linewidth
    plt.rcParams["xtick.minor.width"] = axes_linewidth
    plt.rcParams["ytick.minor.width"] = axes_linewidth


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.14,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")


def load_summary() -> pd.DataFrame:
    df = pd.read_csv(ANALYSIS_DIR / "device_summary.csv")
    df["row"] = df["device"].str.extract(r"R(\d+)").astype(int)
    df["col"] = df["device"].str.extract(r"C(\d+)").astype(int)
    return df


def build_selected_curves():
    curves = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        curve = parse_curve_file(path)
        if curve is not None:
            curves.append(curve)

    groups = {}
    for curve in curves:
        groups.setdefault(curve.group_id, []).append(curve)

    selected = {}
    for group_id, group_curves in groups.items():
        group_curves.sort(key=lambda item: (item.sort_index, item.record_time or "", item.source_file))
        kept = group_curves[:2]
        for idx, curve in enumerate(kept, start=1):
            remark = (curve.remarks or "").strip().lower()
            if "dark" in remark:
                curve.curve_label = "Dark"
            elif "light" in remark:
                curve.curve_label = curve.remarks.strip()
            else:
                curve.curve_label = f"Curve {idx}"
        selected[group_id] = kept
    return selected


def heatmap_matrix(summary: pd.DataFrame, orientation: str, value_col: str) -> pd.DataFrame:
    subset = summary[summary["orientation"] == orientation]
    return subset.pivot(index="row", columns="col", values=value_col).sort_index().sort_index(axis=1)


def plot_heatmap(ax, matrix: pd.DataFrame, title: str, cmap: str = "Greens") -> None:
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, cmap=cmap, aspect="equal")

    ax.set_title(title, fontsize=8, pad=6)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns.tolist())
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index.tolist())
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.tick_params(length=0)

    for r in range(matrix.shape[0] + 1):
        ax.axhline(r - 0.5, color="white", lw=0.8)
    for c in range(matrix.shape[1] + 1):
        ax.axvline(c - 0.5, color="white", lw=0.8)

    for r_idx, row in enumerate(matrix.index):
        for c_idx, col in enumerate(matrix.columns):
            val = matrix.loc[row, col]
            if pd.notna(val):
                ax.text(c_idx, r_idx, f"{val:.2f}", ha="center", va="center", fontsize=6.3, color=PALETTE["neutral_black"])

    return im


def annotate_paired_sites(ax, summary: pd.DataFrame, orientation: str) -> None:
    subset = summary[(summary["orientation"] == orientation) & (summary["paired_dark_light"] == True)]
    cols = sorted(summary["col"].unique())
    rows = sorted(summary["row"].unique())
    col_lookup = {value: idx for idx, value in enumerate(cols)}
    row_lookup = {value: idx for idx, value in enumerate(rows)}
    for _, row in subset.iterrows():
        x = col_lookup[row["col"]]
        y = row_lookup[row["row"]]
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, lw=1.4, ec=PALETTE["red_strong"]))


def plot_ratio_panel(ax, summary: pd.DataFrame) -> pd.DataFrame:
    paired = summary[summary["paired_dark_light"] == True].copy()
    paired = paired.sort_values("light_to_dark_ratio_+1V", ascending=False)
    paired["label"] = paired["group_id"]
    paired["ratio"] = paired["light_to_dark_ratio_+1V"].astype(float)
    paired["color"] = paired["orientation"].map({"H": PALETTE["blue_main"], "V": PALETTE["teal"]})

    y = np.arange(len(paired))
    ax.barh(y, paired["ratio"], color=paired["color"], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(paired["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Light / dark current at +1 V")
    ax.set_title("Photocurrent enhancement in explicit dark/light pairs", fontsize=8, pad=6)
    ax.axvline(1, lw=0.9, ls="--", color=PALETTE["neutral_mid"])
    ax.set_xlim(0, max(1.6, paired["ratio"].max() * 1.18))
    for yi, ratio in zip(y, paired["ratio"]):
        ax.text(ratio + 0.12, yi, f"{ratio:.2f}", va="center", fontsize=6.8, color=PALETTE["neutral_black"])

    legend_handles = [
        Rectangle((0, 0), 1, 1, color=PALETTE["blue_main"], label="H orientation"),
        Rectangle((0, 0), 1, 1, color=PALETTE["teal"], label="V orientation"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=6.8, handlelength=1.1)
    return paired[["group_id", "orientation", "ratio"]]


def plot_iv_panel(ax, curves, group_ids: list[str], title: str) -> pd.DataFrame:
    trace_rows = []
    cond_colors = {"Dark": PALETTE["neutral_dark"], "1/4 Light": PALETTE["blue_main"], "1/2 Light": PALETTE["violet"]}

    for group_id in group_ids:
        if group_id not in curves:
            continue
        for curve in curves[group_id]:
            label = curve.curve_label
            color = cond_colors.get(label, PALETTE["blue_secondary"])
            linestyle = "-" if "Light" in label else "--" if label == "Dark" else "-"
            linewidth = 1.3 if "Light" in label else 1.1
            ax.plot(curve.data["Vd"], curve.data["Id"] * 1e9, color=color, lw=linewidth, ls=linestyle, alpha=0.95)

            x_end = float(curve.data["Vd"].iloc[-1])
            y_end = float(curve.data["Id"].iloc[-1] * 1e9)
            ax.text(
                x_end + 0.03,
                y_end,
                f"{group_id} {label}",
                fontsize=6.2,
                color=color,
                va="center",
            )

            trace_frame = curve.data.copy()
            trace_frame["group_id"] = group_id
            trace_frame["curve_label"] = label
            trace_rows.append(trace_frame)

    ax.axhline(0, color=PALETTE["neutral_light"], lw=0.8, zorder=0)
    ax.axvline(0, color=PALETTE["neutral_light"], lw=0.8, zorder=0)
    ax.set_xlim(-1.05, 1.25)
    ax.set_xlabel("$V_d$ (V)")
    ax.set_ylabel("$I_d$ (nA)")
    ax.set_title(title, fontsize=8, pad=6)

    return pd.concat(trace_rows, ignore_index=True) if trace_rows else pd.DataFrame()


def main() -> None:
    apply_publication_style()
    FIG_DIR.mkdir(exist_ok=True)
    SOURCE_DIR.mkdir(exist_ok=True)

    summary = load_summary()
    curves = build_selected_curves()

    summary["curve_1_log10_abs_+1V_nA"] = np.log10(summary["curve_1_abs_id_+1V_nA"].clip(lower=1e-6))
    h_matrix = heatmap_matrix(summary, "H", "curve_1_log10_abs_+1V_nA")
    v_matrix = heatmap_matrix(summary, "V", "curve_1_log10_abs_+1V_nA")

    fig = plt.figure(figsize=(7.1, 7.2))
    outer = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[1.18, 1.0],
        height_ratios=[0.82, 1.18],
        wspace=0.36,
        hspace=0.38,
    )

    left = outer[:, 0].subgridspec(2, 1, hspace=0.18)
    ax_a1 = fig.add_subplot(left[0, 0])
    ax_a2 = fig.add_subplot(left[1, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    right_bottom = outer[1, 1].subgridspec(2, 1, hspace=0.44)
    ax_c = fig.add_subplot(right_bottom[0, 0])
    ax_d = fig.add_subplot(right_bottom[1, 0])

    im1 = plot_heatmap(ax_a1, h_matrix, "Array map, H orientation: log10(|I| at +1 V) [nA]", cmap="Greens")
    annotate_paired_sites(ax_a1, summary, "H")
    add_panel_label(ax_a1, "a")
    cbar1 = fig.colorbar(im1, ax=ax_a1, fraction=0.046, pad=0.02)
    cbar1.set_label("log10(nA)", fontsize=6.6)
    cbar1.ax.tick_params(labelsize=6.2)

    im2 = plot_heatmap(ax_a2, v_matrix, "Array map, V orientation: log10(|I| at +1 V) [nA]", cmap="Greens")
    annotate_paired_sites(ax_a2, summary, "V")
    cbar2 = fig.colorbar(im2, ax=ax_a2, fraction=0.046, pad=0.02)
    cbar2.set_label("log10(nA)", fontsize=6.6)
    cbar2.ax.tick_params(labelsize=6.2)
    ax_a2.text(
        0.02,
        -0.25,
        "Red outlines mark positions with explicit dark/light pairing.",
        transform=ax_a2.transAxes,
        fontsize=6.2,
        color=PALETTE["neutral_mid"],
    )

    ratio_source = plot_ratio_panel(ax_b, summary)
    add_panel_label(ax_b, "b")

    h_source = plot_iv_panel(ax_c, curves, ["R7C2-H", "R7C3-H"], "Representative H-direction IV curves")
    add_panel_label(ax_c, "c")

    v_source = plot_iv_panel(ax_d, curves, ["R7C2-V", "R7C3-V"], "Representative V-direction IV curves")
    add_panel_label(ax_d, "d")

    ylim_min = min(ax_c.get_ylim()[0], ax_d.get_ylim()[0])
    ylim_max = max(ax_c.get_ylim()[1], ax_d.get_ylim()[1])
    ax_c.set_ylim(ylim_min, ylim_max)
    ax_d.set_ylim(ylim_min, ylim_max)

    fig.suptitle(
        "Localized photocurrent response emerges near R7C2-R7C3 while most array sites remain low-current",
        x=0.52,
        y=0.995,
        fontsize=9.4,
        fontweight="bold",
    )

    stem = FIG_DIR / "sdiv_nature_result_figure"
    save_figure(fig, stem)
    plt.close(fig)

    h_export = h_matrix.rename_axis("row").reset_index()
    v_export = v_matrix.rename_axis("row").reset_index()
    h_export.to_csv(SOURCE_DIR / "panel_a_h_map.csv", index=False)
    v_export.to_csv(SOURCE_DIR / "panel_a_v_map.csv", index=False)
    ratio_source.to_csv(SOURCE_DIR / "panel_b_ratios.csv", index=False)
    h_source.to_csv(SOURCE_DIR / "panel_c_h_curves.csv", index=False)
    v_source.to_csv(SOURCE_DIR / "panel_d_v_curves.csv", index=False)


if __name__ == "__main__":
    main()
