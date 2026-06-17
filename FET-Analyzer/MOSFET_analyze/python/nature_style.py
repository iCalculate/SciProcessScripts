"""Nature-style figure defaults for the B1500 plotter.

Centralises every visual convention so the GUI and any future batch/export
script share one source of truth.  Values follow Nature's print guidance:

* sans-serif type (Arial / Helvetica family), small label sizes;
* single-column width 89 mm (3.5 in), double-column 183 mm (7.2 in);
* thin black spines, inward ticks on all four sides, minor ticks for log axes;
* a restrained, colour-blind-safe qualitative palette used sparingly.

Nothing here touches the global ``matplotlib.rcParams`` of the host process —
:func:`apply_style` is given an explicit Figure/Axes so the tool never leaks
state into a user's other plots.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import List

import matplotlib as mpl

# matplotlib's mathtext can't render an arbitrary TTF (e.g. Graphik) and many such
# fonts also lack the Unicode superscript block — so log-axis tick labels are
# written as plain ASCII scientific notation, which every body font can render.
warnings.filterwarnings("ignore", message="Glyph .* missing from current font")
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import Formatter, LogLocator

class SciLogFormatter(Formatter):
    """Label decade ticks as ``$10^{n}$`` via mathtext, rendered in the body font
    (matplotlib >= 3.8 with a custom mathtext fontset)."""

    def __call__(self, x, pos=None):
        if x <= 0:
            return ""
        exp = math.log10(x)
        if abs(exp - round(exp)) > 1e-6:   # only label exact decades
            return ""
        return rf"$10^{{{int(round(exp))}}}$"

# Millimetre <-> inch helpers (Nature specifies widths in mm).
MM = 1.0 / 25.4
SINGLE_COLUMN_IN = 89 * MM     # ~3.50 in
ONE_HALF_COLUMN_IN = 120 * MM  # ~4.72 in
DOUBLE_COLUMN_IN = 183 * MM    # ~7.20 in

# --- Canonical Nature palette (from the nature-figure skill, references/api.md) ---
# One neutral family, one signal family, one accent family — used sparingly.
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_3": "#8BCF8B", "red_strong": "#B64342",
    "teal": "#42949E", "violet": "#9A4D8E", "gold": "#FFD700",
    "neutral_light": "#CFCECE", "neutral_mid": "#767676",
    "neutral_dark": "#4D4D4D", "neutral_black": "#272727",
}

# Categorical default order (skill DEFAULT_COLORS): use when colour carries a
# discrete semantic role rather than an ordered sweep value.
NATURE_PALETTE: List[str] = [
    PALETTE["blue_main"],     # hero / Id
    PALETTE["red_strong"],    # contrast / Ig
    PALETTE["green_3"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["gold"],
    PALETTE["neutral_mid"],
    PALETTE["blue_secondary"],
]

# Sequential ramps for *stepped families* (VAR2 sweeps such as Id–Vg at many Vd,
# or Id–Vd at many Vg).  A single restrained hue ramped light->dark reads as one
# ordered series — the Nature convention — instead of a categorical rainbow.
SEQUENTIAL_RAMPS = {
    "Blue":   ["#DCE6F2", PALETTE["blue_main"]],
    "Teal":   ["#D6ECEC", PALETTE["teal"]],
    "Red":    ["#F6D9D7", PALETTE["red_strong"]],
    "Violet": ["#E7DCEA", PALETTE["violet"]],
    "Grey":   ["#DCDCDC", PALETTE["neutral_black"]],
    "Viridis": None,   # use matplotlib's perceptual map
}

# Named font stacks the GUI offers; the first installed family wins.
FONT_CHOICES = {
    # "Graphik-Regular" covers systems where the family registers under the full
    # name rather than "Graphik".
    "Graphik": ["Graphik", "Graphik-Regular", "Arial", "Helvetica", "DejaVu Sans"],
    "Arial": ["Arial", "Helvetica", "DejaVu Sans"],
    "Helvetica": ["Helvetica", "Arial", "DejaVu Sans"],
    "DejaVu Sans": ["DejaVu Sans"],
    "Times New Roman": ["Times New Roman", "DejaVu Serif"],
}


BUNDLED_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def register_user_fonts(families=("Graphik",)) -> None:
    """Register on-disk .ttf/.otf fonts (e.g. Graphik) matplotlib hasn't indexed.

    Looks first in the project's bundled ``fonts/`` directory (so the app is
    self-contained and portable), then in the Windows per-user and system font
    folders, preferring the licensed weight over any ``*-Trial`` file.
    """
    import glob

    # 1) Always load everything shipped in the bundled fonts/ directory.
    if os.path.isdir(BUNDLED_FONT_DIR):
        for path in (glob.glob(os.path.join(BUNDLED_FONT_DIR, "*.ttf"))
                     + glob.glob(os.path.join(BUNDLED_FONT_DIR, "*.otf"))):
            try:
                mpl.font_manager.fontManager.addfont(path)
            except Exception:
                continue

    # 2) Fall back to system folders for any family still missing.
    folders = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     "Microsoft", "Windows", "Fonts"),
        os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
    ]
    already = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for fam in families:
        if fam in already:
            continue
        for folder in folders:
            if not os.path.isdir(folder):
                continue
            candidates = sorted(
                glob.glob(os.path.join(folder, f"{fam}*.ttf"))
                + glob.glob(os.path.join(folder, f"{fam}*.otf")),
                key=lambda p: ("trial" in p.lower(), "italic" in p.lower()))
            for path in candidates:
                try:
                    mpl.font_manager.fontManager.addfont(path)
                except Exception:
                    continue


@dataclass
class StyleConfig:
    """User-tunable subset of the Nature style, driven by the GUI controls."""

    # Figure geometry (inches). Defaults requested: 3 wide x 4 tall.
    width_in: float = 3.0
    height_in: float = 4.0
    dpi: int = 300           # export DPI

    # Typography — Graphik at 16 pt by default.
    font_family: str = "Graphik"
    font_size: float = 16.0   # base size; axis-title/tick sizes derive from it

    # Line / marker weights
    line_width: float = 2.0       # data lines
    axes_line_width: float = 1.5  # spines / frame
    tick_width: float = 1.5       # edge tick-mark thickness (user-adjustable)
    tick_length: float = 4.0

    # Frame — open top/right spines is the Nature default; full box is opt-in.
    full_box: bool = False
    minor_ticks: bool = True

    # Colour strategy for multi-curve stepped families.
    color_mode: str = "sequential"   # "sequential" | "categorical"
    ramp: str = "Blue"               # key into SEQUENTIAL_RAMPS

    palette: List[str] = field(default_factory=lambda: list(NATURE_PALETTE))

    # Derived sizes ------------------------------------------------------- #
    @property
    def label_size(self) -> float:
        return self.font_size

    @property
    def tick_label_size(self) -> float:
        return max(self.font_size - 1.0, 5.0)

    @property
    def legend_size(self) -> float:
        return max(self.font_size - 1.0, 5.0)

    def font_stack(self) -> List[str]:
        return FONT_CHOICES.get(self.font_family, ["DejaVu Sans"])

    def color(self, i: int) -> str:
        return self.palette[i % len(self.palette)]

    def sequence_colors(self, n: int):
        """Return *n* colours for a stepped family using the chosen ramp.

        ``sequential`` maps the curve index onto a single restrained hue ramped
        light->dark (the Nature convention for an ordered sweep); ``categorical``
        falls back to the discrete palette.
        """
        import matplotlib as _mpl
        import numpy as _np
        if self.color_mode == "categorical" or n <= 1:
            return [self.color(i) for i in range(n)]
        return [_mpl.colors.to_hex(c) for c in build_cmap(self.ramp)(
            _np.linspace(0.18, 1.0, n))]


def build_cmap(ramp: str):
    """Build a matplotlib colormap for a named sequential ramp."""
    import matplotlib as _mpl
    spec = SEQUENTIAL_RAMPS.get(ramp)
    if spec is None:
        return _mpl.colormaps["viridis"]
    return _mpl.colors.LinearSegmentedColormap.from_list(f"nat_{ramp}", spec)


def rc_context(cfg: StyleConfig) -> dict:
    """Return an rcParams dict to be used inside ``plt.rc_context``."""
    serif = cfg.font_family == "Times New Roman"
    stack = cfg.font_stack()
    # Resolve the first installed family in the stack so mathtext maps to the
    # same concrete font as the body text (handles "Graphik" vs "Graphik-Regular").
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    primary = next((f for f in stack if f in installed), stack[0])
    rc = {
        "figure.dpi": cfg.dpi,
        "savefig.dpi": cfg.dpi,
        "font.family": "serif" if serif else "sans-serif",
        "font.weight": "normal",
        ("font.serif" if serif else "font.sans-serif"): stack,
        # Render mathtext (subscripts/superscripts in labels & tick values) in the
        # SAME font as the body text. matplotlib >= 3.8 honours a custom TTF here.
        "mathtext.fontset": "custom",
        "mathtext.rm": primary,
        "mathtext.it": primary,
        "mathtext.bf": primary,
        "mathtext.sf": primary,
        "mathtext.cal": primary,
        "mathtext.tt": primary,
        "mathtext.default": "regular",
        "axes.unicode_minus": False,   # ASCII '-' (some TTFs lack U+2212)
        "font.size": cfg.font_size,
        "axes.titlesize": cfg.font_size,
        "axes.labelsize": cfg.label_size,
        "xtick.labelsize": cfg.tick_label_size,
        "ytick.labelsize": cfg.tick_label_size,
        "legend.fontsize": cfg.legend_size,
        "axes.linewidth": cfg.axes_line_width,
        "lines.linewidth": cfg.line_width,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": cfg.tick_length,
        "ytick.major.size": cfg.tick_length,
        "xtick.minor.size": cfg.tick_length * 0.6,
        "ytick.minor.size": cfg.tick_length * 0.6,
        "xtick.major.width": cfg.tick_width,
        "ytick.major.width": cfg.tick_width,
        "xtick.minor.width": cfg.tick_width * 0.8,
        "ytick.minor.width": cfg.tick_width * 0.8,
        "xtick.top": cfg.full_box,
        "ytick.right": cfg.full_box,
        "axes.grid": False,
        "legend.frameon": False,
        "svg.fonttype": "none",      # keep text editable in exported SVG
        "pdf.fonttype": 42,          # embed TrueType so text stays selectable
        "ps.fonttype": 42,
    }
    return rc


def style_axes(ax: Axes, cfg: StyleConfig, *, log_y: bool = False,
               right_axis: bool = False) -> None:
    """Apply per-axes Nature conventions (spines, ticks, minor locators).

    Spine visibility:
      * full box -> all four spines shown;
      * open frame, primary axis -> hide top & right;
      * open frame, twin (right) axis -> show only the right spine, so the
        figure never grows a top edge line just because a second axis exists.
    """
    for side, spine in ax.spines.items():
        spine.set_linewidth(cfg.axes_line_width)
        if cfg.full_box:
            spine.set_visible(True)
        elif right_axis:
            spine.set_visible(side == "right")
        elif side in ("top", "right"):
            spine.set_visible(False)

    # Which sides carry ticks.  A twin (right) axis is the *data* axis on the
    # right, so its right-side y ticks are always on; its x ticks stay off
    # (the primary axis already draws them).
    if right_axis:
        ytick = dict(left=False, right=True, labelright=True, labelleft=False)
        xtick = dict(top=False, bottom=False, labeltop=False, labelbottom=False)
    else:
        ytick = dict(left=True, right=cfg.full_box)
        xtick = dict(bottom=True, top=cfg.full_box)

    if cfg.minor_ticks:
        ax.minorticks_on()
    for which, scale in (("major", 1.0), ("minor", 0.6)):
        if which == "minor" and not cfg.minor_ticks:
            continue
        w = cfg.tick_width if which == "major" else cfg.tick_width * 0.8
        ax.tick_params(axis="y", which=which, direction="in",
                       length=cfg.tick_length * scale, width=w, **ytick)
        ax.tick_params(axis="x", which=which, direction="in",
                       length=cfg.tick_length * scale, width=w, **xtick)
    if log_y:
        # Plain ASCII decade labels, so the body font renders them.
        ax.yaxis.set_major_formatter(SciLogFormatter())
        if cfg.minor_ticks:
            ax.yaxis.set_minor_locator(
                LogLocator(base=10.0, subs=tuple(range(2, 10)), numticks=100))


def add_panel_label(ax, label: str, x: float = -0.16, y: float = 1.02,
                    cfg: "StyleConfig" = None) -> None:
    """Nature-style panel label: small bold lowercase letter, top-left edge."""
    size = (cfg.font_size + 1.5) if cfg else 9
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size,
            fontweight="bold", ha="left", va="bottom")


def save_publication(fig, path_no_ext: str, cfg: "StyleConfig",
                     formats=("svg", "pdf", "tiff")) -> List[str]:
    """Export the Nature submission bundle.

    SVG is the primary, text-editable master (``svg.fonttype='none'``); PDF keeps
    vector text; TIFF/PNG are rasterised at >=600 dpi for journal upload.  Mirrors
    the nature-figure skill's ``save_pub_py`` / ``finalize_figure`` policy.
    """
    import os
    saved = []
    base, ext = os.path.splitext(path_no_ext)
    if ext.lstrip(".").lower() in ("svg", "pdf", "tiff", "tif", "png", "eps"):
        base, formats = base, (ext.lstrip("."),)
    for fmt in formats:
        out = f"{base}.{fmt}"
        dpi = max(cfg.dpi, 600) if fmt.lower() in ("tiff", "tif", "png") else cfg.dpi
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        saved.append(out)
    return saved


def get_available_fonts() -> List[str]:
    """Font families from FONT_CHOICES that are actually installed."""
    register_user_fonts()  # make sure Graphik is indexed before we check
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    out = []
    for label, stack in FONT_CHOICES.items():
        if any(name in installed for name in stack):
            out.append(label)
    return out or ["DejaVu Sans"]


# Register on-disk Graphik (and similar) once, at import.
register_user_fonts()
