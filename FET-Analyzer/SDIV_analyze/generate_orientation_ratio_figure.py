from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd

from analyze_sdiv_iv import parse_curve_file_multi, assign_group_labels


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Graphik", "Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "rawdata"
OUT_DIR = ROOT / "analysis_output" / "orientation_ratio_figure"
SOURCE_DIR = OUT_DIR / "source_data"

PALETTE = {
    "blue": "#0F4D92",
    "blue_soft": "#3775BA",
    "blue_text": "#1E5FAF",
    "teal": "#42949E",
    "teal_soft": "#7DB8BF",
    "red": "#B64342",
    "violet": "#7C6CCF",
    "gray": "#6F6F6F",
    "light_gray": "#D7D7D7",
    "orange": "#C46A1A",
    "black": "#272727",
}


def apply_style() -> None:
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.8
    plt.rcParams["ytick.major.width"] = 0.8


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.14, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")


def save_figure(fig, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=500, bbox_inches="tight")


def nearest_current(curve, target_vd: float) -> float:
    idx = (curve.data["Vd"] - target_vd).abs().idxmin()
    return float(curve.data.loc[idx, "Id"])


def load_selected_curves():
    curves = []
    for path in sorted(RAW_DIR.glob("*.csv")):
        curves.extend(parse_curve_file_multi(path))

    # Measurement notes indicate rows 5-7 had H/V labels swapped at acquisition time.
    # Normalize here so all downstream analyses use the corrected orientation.
    for curve in curves:
        if curve.device.startswith(("R5", "R6", "R7")):
            curve.orientation = "V" if curve.orientation == "H" else "H"
            curve.group_id = f"{curve.device}-{curve.orientation}"

    groups = defaultdict(list)
    for curve in curves:
        groups[curve.group_id].append(curve)

    selected = {}
    for group_id, group_curves in groups.items():
        group_curves.sort(key=lambda item: (item.sort_index, item.record_time or "", item.source_file))
        kept = group_curves[:2]
        assign_group_labels(kept)
        selected[group_id] = kept
    return selected


def build_orientation_tables(selected):
    site_map = defaultdict(lambda: {"H": {}, "V": {}})
    for group_id, curves in selected.items():
        device, orientation = group_id.rsplit("-", 1)
        for curve in curves:
            site_map[device][orientation][curve.curve_label] = curve

    rows = []
    ratio_traces = []
    for device in sorted(site_map):
        h_map = site_map[device]["H"]
        v_map = site_map[device]["V"]
        common_labels = sorted(set(h_map) & set(v_map))
        for label in common_labels:
            h_curve = h_map[label]
            v_curve = v_map[label]
            h_pos = abs(nearest_current(h_curve, 1.0))
            v_pos = abs(nearest_current(v_curve, 1.0))
            h_neg = abs(nearest_current(h_curve, -1.0))
            v_neg = abs(nearest_current(v_curve, -1.0))
            rows.append(
                {
                    "device": device,
                    "label": label,
                    "row": int(device.split("C")[0][1:]),
                    "col": int(device.split("C")[1]),
                    "H_abs_id_+1V_nA": h_pos * 1e9,
                    "V_abs_id_+1V_nA": v_pos * 1e9,
                    "H_abs_id_-1V_nA": h_neg * 1e9,
                    "V_abs_id_-1V_nA": v_neg * 1e9,
                    "V_over_H_+1V": v_pos / h_pos if h_pos > 0 else np.nan,
                    "V_over_H_-1V": v_neg / h_neg if h_neg > 0 else np.nan,
                    "H_slope_0_to_1_A_per_V": abs(np.polyfit(
                        h_curve.data[(h_curve.data["Vd"] >= 0) & (h_curve.data["Vd"] <= 1)]["Vd"],
                        h_curve.data[(h_curve.data["Vd"] >= 0) & (h_curve.data["Vd"] <= 1)]["Id"],
                        1,
                    )[0]),
                    "V_slope_0_to_1_A_per_V": abs(np.polyfit(
                        v_curve.data[(v_curve.data["Vd"] >= 0) & (v_curve.data["Vd"] <= 1)]["Vd"],
                        v_curve.data[(v_curve.data["Vd"] >= 0) & (v_curve.data["Vd"] <= 1)]["Id"],
                        1,
                    )[0]),
                    "H_slope_-1_to_0_A_per_V": abs(np.polyfit(
                        h_curve.data[(h_curve.data["Vd"] >= -1) & (h_curve.data["Vd"] <= 0)]["Vd"],
                        h_curve.data[(h_curve.data["Vd"] >= -1) & (h_curve.data["Vd"] <= 0)]["Id"],
                        1,
                    )[0]),
                    "V_slope_-1_to_0_A_per_V": abs(np.polyfit(
                        v_curve.data[(v_curve.data["Vd"] >= -1) & (v_curve.data["Vd"] <= 0)]["Vd"],
                        v_curve.data[(v_curve.data["Vd"] >= -1) & (v_curve.data["Vd"] <= 0)]["Id"],
                        1,
                    )[0]),
                }
            )

            merged = pd.merge(
                h_curve.data[["Vd", "Id"]].rename(columns={"Id": "Id_H"}),
                v_curve.data[["Vd", "Id"]].rename(columns={"Id": "Id_V"}),
                on="Vd",
                how="inner",
            ).sort_values("Vd")
            merged["ratio_abs_V_over_H"] = merged["Id_V"].abs() / merged["Id_H"].abs().clip(lower=1e-30)
            merged["device"] = device
            merged["label"] = label
            ratio_traces.append(merged)

    summary = pd.DataFrame(rows).sort_values(["row", "col", "label"])
    summary["V_over_H_slope_0_to_1"] = summary["V_slope_0_to_1_A_per_V"] / summary["H_slope_0_to_1_A_per_V"].clip(lower=1e-30)
    summary["V_over_H_slope_-1_to_0"] = summary["V_slope_-1_to_0_A_per_V"] / summary["H_slope_-1_to_0_A_per_V"].clip(lower=1e-30)
    traces = pd.concat(ratio_traces, ignore_index=True) if ratio_traces else pd.DataFrame()
    return summary, traces


def physical_col(row_val: int, logical_col: int) -> int:
    return 2 * logical_col - 1 if row_val % 2 == 1 else 2 * logical_col


def build_triangular_map(summary: pd.DataFrame, label_name: str | list[str]) -> pd.DataFrame:
    if isinstance(label_name, list):
        subset = summary[summary["label"].isin(label_name)].copy()
    else:
        subset = summary[summary["label"] == label_name].copy()
    subset = subset.sort_values(["row", "col", "label"]).drop_duplicates(subset=["device"], keep="first")
    subset["physical_col"] = [physical_col(r, c) for r, c in zip(subset["row"], subset["col"])]
    return subset.sort_values(["row", "physical_col"])


def triangular_heatmap_limits(*frames: pd.DataFrame) -> float:
    vals = []
    for frame in frames:
        for col in ["V_over_H_slope_0_to_1", "V_over_H_slope_-1_to_0"]:
            if col in frame:
                clean = frame[col].replace([np.inf, -np.inf], np.nan).dropna().tolist()
                vals.extend([abs(math.log2(v)) for v in clean if v > 0])
    vmax = max(vals, default=1.0)
    return vmax if np.isfinite(vmax) and vmax > 0 else 1.0


def ratio_score(value: float) -> float:
    return math.log2(value)


def ratio_text(value: float) -> tuple[str, str]:
    if not np.isfinite(value) or value <= 0:
        return "", PALETTE["black"]
    if abs(value - 1.0) < 0.03:
        return "1.0", PALETTE["black"]
    if value > 1:
        return f"{value:.1f}", PALETTE["orange"]
    inv = 1.0 / value
    inv_text = f"{inv:.1f}"
    return rf"${inv_text}^{{-1}}$", PALETTE["blue_text"]


def score_color(score: float, vmax: float):
    intensity = min(abs(score) / max(vmax, 1e-9), 1.0)
    cmap = plt.get_cmap("Oranges") if score >= 0 else plt.get_cmap("Blues")
    return cmap(0.18 + 0.62 * intensity)


def draw_triangular_heatmap(ax, frame: pd.DataFrame, title: str, vmax: float):
    norm = Normalize(vmin=-vmax, vmax=vmax)

    for _, row in frame.iterrows():
        x = float(row["physical_col"])
        y = float(row["row"])
        x0, x1 = x - 0.5, x + 0.5
        y0, y1 = y - 0.5, y + 0.5
        pos_val = float(row["V_over_H_slope_0_to_1"])
        neg_val = float(row["V_over_H_slope_-1_to_0"])
        pos_score = ratio_score(pos_val)
        neg_score = ratio_score(neg_val)

        upper = Polygon([[x0, y1], [x1, y1], [x1, y0]], closed=True,
                        facecolor=score_color(pos_score, vmax), edgecolor="white", linewidth=1.0)
        lower = Polygon([[x0, y1], [x0, y0], [x1, y0]], closed=True,
                        facecolor=score_color(neg_score, vmax), edgecolor="white", linewidth=1.0)
        ax.add_patch(upper)
        ax.add_patch(lower)
        ax.plot([x0, x1], [y1, y0], color="white", lw=1.0)
        ax.add_patch(Rectangle((x0, y0), 1, 1, fill=False, ec=PALETTE["light_gray"], lw=0.8))

        pos_text, pos_color = ratio_text(pos_val)
        neg_text, neg_color = ratio_text(neg_val)
        ax.text(x0 + 0.10, y0 + 0.16, pos_text, ha="left", va="top", fontsize=5.8, color=pos_color)
        ax.text(x1 - 0.10, y1 - 0.08, neg_text, ha="right", va="bottom", fontsize=5.8, color=neg_color)

    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(7.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(1, 8))
    ax.set_xticklabels(["1", "1", "2", "2", "3", "3", "4"])
    ax.set_yticks(range(1, 8))
    ax.set_xlabel("Logical column")
    ax.set_ylabel("Row")
    ax.set_title(title, fontsize=8, pad=6)
    ax.tick_params(length=0)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_visible(False)
    for xpos in range(1, 8):
        ax.axvline(xpos + 0.5, color="#F3F3F3", lw=0.6, zorder=0)
    for ypos in range(1, 8):
        ax.axhline(ypos + 0.5, color="#F3F3F3", lw=0.6, zorder=0)
    return plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu_r"))


def plot_grouped_bar(ax, panel_df: pd.DataFrame) -> None:
    devices = sorted(panel_df["device"].unique())
    x = np.arange(len(devices))
    width = 0.34
    dark = [panel_df.loc[(panel_df["device"] == d) & (panel_df["label"] == "Dark"), "V_over_H_+1V"].iloc[0] for d in devices]
    light = [panel_df.loc[(panel_df["device"] == d) & (panel_df["label"].str.contains("Light")), "V_over_H_slope_0_to_1"].iloc[0] for d in devices]
    dark = [panel_df.loc[(panel_df["device"] == d) & (panel_df["label"] == "Dark"), "V_over_H_slope_0_to_1"].iloc[0] for d in devices]
    ax.bar(x - width / 2, dark, width=width, color=PALETTE["gray"], label="Dark")
    ax.bar(x + width / 2, light, width=width, color=PALETTE["blue"], label="Light")
    ax.axhline(1, color=PALETTE["light_gray"], lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(devices)
    ax.set_ylabel(r"$|G_V|/|G_H|$ from 0 to +1 V fit")
    ax.set_title("Orientation anisotropy from forward-bias linear fits", fontsize=8, pad=6)
    ax.legend(loc="upper right", fontsize=6.8, handlelength=1.2)
    ymax = max(max(dark), max(light)) * 1.18
    ax.set_ylim(0, ymax)
    for xpos, val in zip(x - width / 2, dark):
        ax.text(xpos, val + ymax * 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=6.2)
    for xpos, val in zip(x + width / 2, light):
        ax.text(xpos, val + ymax * 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=6.2)


def pick_row_representatives(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row_id, group in summary.groupby("row"):
        group = group.copy()
        group["score"] = group[["V_over_H_slope_0_to_1", "V_over_H_slope_-1_to_0"]].apply(
            lambda s: max(abs(math.log2(max(s.iloc[0], 1e-30))), abs(math.log2(max(s.iloc[1], 1e-30)))), axis=1
        )
        best = group.sort_values(["score", "device"], ascending=[False, True]).iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows).sort_values("row")


def plot_conductance_bars(ax, panel: pd.DataFrame, title: str, state: str) -> None:
    devices = [f"R{int(r)}:{d}" for r, d in zip(panel["row"], panel["device"])]
    x = np.arange(len(devices))
    width = 0.34
    h_vals = panel["H_slope_0_to_1_A_per_V"].abs().to_numpy() * 1e9
    v_vals = panel["V_slope_0_to_1_A_per_V"].abs().to_numpy() * 1e9

    ax.bar(x - width / 2, h_vals, width=width, color=PALETTE["teal"], label="H slope")
    ax.bar(x + width / 2, v_vals, width=width, color=PALETTE["blue"], label="V slope")
    ax.set_xticks(x)
    ax.set_xticklabels(devices, rotation=25, ha="right")
    ax.set_ylabel(r"$|dI/dV|$ from 0 to +1 V fit (nA/V)")
    ax.set_title(title, fontsize=8, pad=6)
    ax.legend(loc="upper right", fontsize=6.8, handlelength=1.2)

    positive_vals = np.concatenate([h_vals[h_vals > 0], v_vals[v_vals > 0]]) if len(h_vals) else np.array([1e-6])
    ymin = max(np.min(positive_vals) * 0.6, 1e-6)
    ymax = max(np.max(positive_vals) * 1.8, ymin * 10)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    for xpos, val in zip(x - width / 2, h_vals):
        ax.text(xpos, val * 1.15, f"{val:.1e}", ha="center", va="bottom", fontsize=5.6, rotation=90)
    for xpos, val in zip(x + width / 2, v_vals):
        ax.text(xpos, val * 1.15, f"{val:.1e}", ha="center", va="bottom", fontsize=5.6, rotation=90)

    for xpos, label in zip(x, panel["device"]):
        ax.text(xpos, ymax * 0.02, label, ha="center", va="bottom", fontsize=5.6, color=PALETTE["gray"])


def plot_anisotropy_summary(ax, panel: pd.DataFrame) -> None:
    x = np.arange(len(panel))
    width = 0.34
    dark_vals = panel["V_over_H_slope_0_to_1"].to_numpy()
    light_vals = panel["light_ratio"].to_numpy()

    ax.bar(x - width / 2, dark_vals, width=width, color=PALETTE["gray"], label="Dark")
    ax.bar(x + width / 2, light_vals, width=width, color=PALETTE["blue"], label="Light")
    ax.axhline(1, color=PALETTE["light_gray"], lw=1.0, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"R{int(r)}" for r in panel["row"]])
    ax.set_ylabel(r"$|G_V|/|G_H|$")
    ax.set_title("Representative anisotropy by row", fontsize=8, pad=6)
    ax.legend(loc="upper right", fontsize=6.8, handlelength=1.2)
    positive_vals = np.concatenate([dark_vals[dark_vals > 0], light_vals[light_vals > 0]]) if len(panel) else np.array([1e-3])
    ymin = max(np.min(positive_vals) * 0.7, 1e-3)
    ymax = max(np.max(positive_vals) * 1.6, ymin * 10)
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    for xpos, val in zip(x - width / 2, dark_vals):
        ax.text(xpos, val * 1.12, f"{val:.1f}", ha="center", va="bottom", fontsize=5.8)
    for xpos, val in zip(x + width / 2, light_vals):
        ax.text(xpos, val * 1.12, f"{val:.1f}", ha="center", va="bottom", fontsize=5.8)


def build_standard_layout_figure(dark_map, light_map, vmax, representatives, light_labels, light_label):
    fig = plt.figure(figsize=(7.2, 7.0))
    outer = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.08, 1.0], height_ratios=[0.9, 1.1], wspace=0.35, hspace=0.42)
    left = outer[:, 0].subgridspec(2, 1, hspace=0.22)
    right = outer[:, 1].subgridspec(3, 1, hspace=0.48)

    ax_a1 = fig.add_subplot(left[0, 0])
    ax_a2 = fig.add_subplot(left[1, 0])
    ax_b = fig.add_subplot(right[0, 0])
    ax_c = fig.add_subplot(right[1, 0])
    ax_d = fig.add_subplot(right[2, 0])

    sm1 = draw_triangular_heatmap(ax_a1, dark_map, r"Dark-state map: upper 0 to +1 V fit, lower -1 to 0 V fit", vmax)
    add_panel_label(ax_a1, "a")
    cbar1 = fig.colorbar(sm1, ax=ax_a1, fraction=0.046, pad=0.03)
    cbar1.set_label(r"sign $\times \log_2$ ratio magnitude", fontsize=6.6)
    cbar1.ax.tick_params(labelsize=6.1)

    sm2 = draw_triangular_heatmap(ax_a2, light_map, rf"{light_label}-state map: upper 0 to +1 V fit, lower -1 to 0 V fit", vmax)
    cbar2 = fig.colorbar(sm2, ax=ax_a2, fraction=0.046, pad=0.03)
    cbar2.set_label(r"sign $\times \log_2$ ratio magnitude", fontsize=6.6)
    cbar2.ax.tick_params(labelsize=6.1)
    ax_a2.text(0.02, -0.24, "Each square is split diagonally: upper triangle uses the 0 to +1 V linear-fit slope and lower triangle uses the -1 to 0 V slope.", transform=ax_a2.transAxes, fontsize=6.1, color=PALETTE["gray"])

    plot_anisotropy_summary(ax_b, representatives.dropna(subset=["light_ratio"]))
    add_panel_label(ax_b, "b")

    plot_conductance_bars(ax_c, representatives[["row", "device", "H_slope_0_to_1_A_per_V", "V_slope_0_to_1_A_per_V"]].copy(), r"Representative dark-state conductance by row", "Dark")
    add_panel_label(ax_c, "c")

    light_panel = representatives[["row", "device", "light_H_slope", "light_V_slope"]].rename(
        columns={"light_H_slope": "H_slope_0_to_1_A_per_V", "light_V_slope": "V_slope_0_to_1_A_per_V"}
    ).dropna()
    plot_conductance_bars(ax_d, light_panel, r"Representative light-state conductance by row", "Light")
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "Orientation anisotropy is strongest around R7C2-R7C3 and changes between dark and illuminated states",
        x=0.52,
        y=0.995,
        fontsize=9.3,
        fontweight="bold",
    )
    return fig


def build_horizontal_layout_figure(dark_map, light_map, vmax, representatives, light_labels, light_label):
    fig = plt.figure(figsize=(11.8, 6.6))
    outer = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[0.14, 1.0, 0.08, 1.0, 0.14], height_ratios=[1.08, 0.92], hspace=0.34, wspace=0.0)
    top = outer[0, 1:4].subgridspec(1, 2, wspace=0.26)
    bottom = outer[1, :].subgridspec(1, 3, wspace=0.32)

    ax_a1 = fig.add_subplot(top[0, 0])
    ax_a2 = fig.add_subplot(top[0, 1])
    ax_b = fig.add_subplot(bottom[0, 0])
    ax_c = fig.add_subplot(bottom[0, 1])
    ax_d = fig.add_subplot(bottom[0, 2])

    sm1 = draw_triangular_heatmap(ax_a1, dark_map, r"Dark-state map: upper 0 to +1 V fit, lower -1 to 0 V fit", vmax)
    add_panel_label(ax_a1, "a")
    cbar1 = fig.colorbar(sm1, ax=ax_a1, fraction=0.046, pad=0.02)
    cbar1.set_label(r"sign $\times \log_2$ ratio magnitude", fontsize=6.6)
    cbar1.ax.tick_params(labelsize=6.1)

    sm2 = draw_triangular_heatmap(ax_a2, light_map, rf"{light_label}-state map: upper 0 to +1 V fit, lower -1 to 0 V fit", vmax)
    cbar2 = fig.colorbar(sm2, ax=ax_a2, fraction=0.046, pad=0.02)
    cbar2.set_label(r"sign $\times \log_2$ ratio magnitude", fontsize=6.6)
    cbar2.ax.tick_params(labelsize=6.1)

    plot_anisotropy_summary(ax_b, representatives.dropna(subset=["light_ratio"]))
    add_panel_label(ax_b, "b")

    plot_conductance_bars(ax_c, representatives[["row", "device", "H_slope_0_to_1_A_per_V", "V_slope_0_to_1_A_per_V"]].copy(), r"Representative dark-state conductance by row", "Dark")
    add_panel_label(ax_c, "c")

    light_panel = representatives[["row", "device", "light_H_slope", "light_V_slope"]].rename(
        columns={"light_H_slope": "H_slope_0_to_1_A_per_V", "light_V_slope": "V_slope_0_to_1_A_per_V"}
    ).dropna()
    plot_conductance_bars(ax_d, light_panel, r"Representative light-state conductance by row", "Light")
    add_panel_label(ax_d, "d")

    fig.suptitle(
        "Orientation anisotropy is strongest around R7C2-R7C3 and changes between dark and illuminated states",
        x=0.5,
        y=0.99,
        fontsize=9.5,
        fontweight="bold",
    )
    return fig


def main() -> None:
    apply_style()
    OUT_DIR.mkdir(exist_ok=True)
    SOURCE_DIR.mkdir(exist_ok=True)

    selected = load_selected_curves()
    summary, traces = build_orientation_tables(selected)
    summary.to_csv(SOURCE_DIR / "orientation_ratio_summary.csv", index=False)
    traces.to_csv(SOURCE_DIR / "orientation_ratio_traces.csv", index=False)

    light_labels = [label for label in ["Light", "1/4 Light", "1/2 Light"] if label in summary["label"].unique()]
    representatives = pick_row_representatives(summary[summary["label"] == "Dark"].copy())
    light_pick = summary[summary["label"].isin(light_labels if light_labels else ["Light"])].copy()
    light_pick = light_pick.sort_values(["row", "device", "label"]).drop_duplicates(subset=["row", "device"], keep="first")
    representatives = representatives.merge(
        light_pick[["row", "device", "V_over_H_slope_0_to_1", "H_slope_0_to_1_A_per_V", "V_slope_0_to_1_A_per_V"]].rename(
            columns={
                "V_over_H_slope_0_to_1": "light_ratio",
                "H_slope_0_to_1_A_per_V": "light_H_slope",
                "V_slope_0_to_1_A_per_V": "light_V_slope",
            }
        ),
        on=["row", "device"],
        how="left",
    )
    dark_map = build_triangular_map(summary, "Dark")
    light_label = " / ".join(light_labels) if light_labels else "Light"
    light_map = build_triangular_map(summary, light_labels if light_labels else "Light")
    vmax = triangular_heatmap_limits(dark_map, light_map)

    fig_standard = build_standard_layout_figure(dark_map, light_map, vmax, representatives, light_labels, light_label)
    save_figure(fig_standard, OUT_DIR / "sdiv_orientation_ratio_figure")
    plt.close(fig_standard)

    fig_horizontal = build_horizontal_layout_figure(dark_map, light_map, vmax, representatives, light_labels, light_label)
    save_figure(fig_horizontal, OUT_DIR / "sdiv_orientation_ratio_figure_horizontal")
    plt.close(fig_horizontal)


if __name__ == "__main__":
    main()
