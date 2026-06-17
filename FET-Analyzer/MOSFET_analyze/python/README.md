# MOSFET Data Plotter (Python)

A GUI tool that turns Keysight **B1500A** CSV exports into publication-ready
**Transfer** (Id–Vg) and **Output** (Id–Vd) curves following Nature figure
conventions. A menu bar (**File / Edit / View / Help**) and an always-on status bar
(showing **Ready / Busy** plus the current measurement) sit around the plot.
Switching the data source in the **Data** dropdown **keeps your current
settings** (axis ranges, labels, scaling, smoothing …) — only the curve list
follows the data. Use the **Reset** button (next to *Apply / Replot*) or
**File → Reset All Settings** to return everything to defaults (axes back to
Auto).

**Edit → Preferences** sets the default style, grouped into Typography (font,
size), Figure (width, height, DPI), Lines & frame (line / tick / border width,
tick length, full-box, minor ticks) and Colours (default mode, ramp, and the
editable categorical **palette swatches**). Font, palette, tick length and
minor-ticks apply immediately; the other numeric defaults take effect on Reset.
Preferences persist across sessions in `preferences.json`.

```
run.bat                                 # Windows: launches in the uv env
run.bat <file.csv | folder>            # preload a file or folder
```

or directly with [uv](https://docs.astral.sh/uv/):

```
uv run python b1500_plotter.py [<file.csv> | <folder>]
```

### Environment (uv-managed, self-contained)

The project carries its **own isolated environment**, managed by `uv` and pinned
in `pyproject.toml` + `uv.lock` (Python 3.11, `matplotlib`, `PySide6`, `numpy`,
`scipy`). It does **not** use any system / conda Python. `uv run` (and `run.bat`)
create/update `.venv` automatically from the lockfile on first launch — nothing
to install by hand.

Python ≥ 3.11 with matplotlib ≥ 3.8 is required so that **mathtext labels render
in the Graphik font** (older matplotlib cannot use an arbitrary TTF for math, so
`$V_\mathrm{g}$` / `$10^{-9}$` would fall back to a different font).

Common commands (run inside this folder):

```
uv sync            # create/refresh the environment from the lockfile
uv add <pkg>       # add a dependency (updates pyproject.toml + uv.lock)
uv run python ...  # run anything inside the project environment
```

`pyproject.toml` and `uv.lock` are version-controlled; `.venv/` is git-ignored.

---

## What the UI gives you

The control panel is organised into three tabs.

### Plot tab

| Section   | Controls |
|-----------|----------|
| **Figure**| Width / height (inches = export size & aspect), font size, **line width**, **tick width**, **border width** (spine thickness), DPI; Nature presets; full-box vs open-spine frame. The preview is **true WYSIWYG**: the figure is rendered at its fixed W×H and scaled uniformly to fit, so fonts/lines keep the exact same proportion at any window size and match the Copy/Export output. The font family is set in **Edit → Preferences**, not here. |
| **Title** | Collapsible. Optional plot title. |
| **Axes**  | Collapsible. X / left-Y labels, ranges (min/max boxes always present; when on auto they display the live autoscaled value in grey — overwrite to override), linear/log, `abs()`. **Id + |Ig|**: overlay the gate current on the **same axis**, or on a **right axis whose numeric range is locked equal to the left** axis. With an open frame the top edge line is never drawn, even with the second axis. |
| **Curves**| **Colour strategy** (sequential ramp vs categorical); per-curve show/hide, colour picker, editable label; legend or **colourbar**. |

### Preprocess tab

Refines the same plot without changing any Plot-tab styling:

- **Scaling** — independent multipliers for Id and Ig.
- **Smoothing** — **Id and Ig smoothed independently**, each with its own method
  (**Savitzky–Golay / moving average / median / Gaussian**) and a live strength
  readout. Smoothing is computed in **linear** current space and then displayed
  on whatever axis (linear or log) you choose.
- **Noise floor** — optional, **applied after smoothing** (per channel). Adds a
  Gaussian noise floor of a chosen RMS **Level (A)** to restore a realistic
  baseline on an over-smoothed off-state (the on-state is unaffected). The
  **Seed** is either fixed (reproducible — no flicker on incidental redraws) or
  set to **Random**: in random mode the pattern re-rolls automatically whenever
  you change a preprocessing parameter, and the **↻** button draws a fresh one
  on demand.

### Analyze tab

- **Compute parameters** runs the FET extraction (ported from the MATLAB
  analyzer) per transfer curve and lists **SS, Ion/Ioff, Ion, Ioff, Vth, Von,
  g\_m,max, SS-fit R²** in a table — **parameters as rows, one column per curve**
  (e.g. per Vd step).
- **Annotate on plot** overlays the selected results: the **subthreshold-slope
  tangent**, the **Ion / Ioff horizontal levels**, the **Vth** line, and the
  **g\_m,max** line. Text labels are shown when ≤ 3 curves are visible to avoid
  clutter on stepped families.

Default style: **Graphik** font, 3 × 4 in, 16 pt, line width 2, DPI 300.
`Graphik-Regular.ttf` is bundled in `fonts/` and registered at startup; a custom
mathtext fontset maps math to it, so axis titles and tick values render in
Graphik **with proper subscripts/superscripts** (`$V_\mathrm{g}$`, `$|I_\mathrm{d}|$`,
`$10^{-9}$`) — real labels, not text overlays. (This needs matplotlib ≥ 3.8; see
**Environment** above.) Drop other `.ttf`/`.otf` files into `fonts/` to add choices.

**Copy image** puts the figure (rendered at the export geometry) straight on the
clipboard.

**Export…** opens a dialog to pick any combination of:

- **Image** — `PNG` (≥300 dpi raster) and/or `SVG` (vector, editable text).
- **Data** — `CSV` of the **post-processed** data, written in the **original
  B1500 format with the original filename**, so it re-imports and reproduces the
  exact same figure with no further processing.
- **Configuration** — a `JSON` capturing all plot and preprocessing parameters
  (style, axes, curves, smoothing/noise, analysis selections).
- **Select all** exports everything at once into a chosen folder.

### Nature style, by default

These conventions come from the `nature-figure` skill and are baked into
`nature_style.py`:

- **Restrained palette.** A `VAR2`-stepped family (Id–Vg at many Vd, or Id–Vd at
  many Vg) is drawn as **one hue ramped light→dark** (the ordered-sweep
  convention) instead of a categorical rainbow, with an optional **colourbar**
  replacing a crowded legend. Ramps: Blue / Teal / Red / Violet / Grey / Viridis.
- **Canonical Nature colours** (`#0F4D92`, `#B64342`, `#8BCF8B`, `#42949E`, …) for
  categorical mode.
- **Compact print sizing**: 7–8 pt sans-serif (Arial/Helvetica), 0.8 pt axes,
  open top/right spines, inward major+minor ticks.
- **Editable vector text** (`svg.fonttype='none'`, `pdf.fonttype=42`).

### Figure contract (what each plot argues)

- **Transfer (Id–Vg)** — defends *switching behaviour*: log |Id| vs Vg shows the
  on/off ratio and subthreshold slope; turn on the right-Y |Ig| axis to show that
  gate leakage stays low.
- **Output (Id–Vd)** — defends *current modulation/saturation*: Id vs Vd, one
  ramped family across the gate steps.

---

## File / data-format support

The reader (`b1500_io.py`) is deliberately tolerant of the many shapes a B1500
export can take:

- **Single-curve** sweeps and **multi-curve** sweeps (a VAR2 secondary step
  produces `Dimension2` curves of `Dimension1` points each).
- **Multiple test records concatenated in one file** — each `SetupTitle` block
  is parsed separately; byte-for-byte duplicate records (the common
  raw + auto-analysis pair) are collapsed automatically.
- **Varying column order and extra analysis columns** (`gm`, `Vth`, `Von`,
  `absId`, …) — columns are keyed by their `DataName`, never by position.
- Transfer vs. output and the sweep/secondary variables are inferred from
  `Channel.Func` (VAR1/VAR2) and fall back to the title and column names.

---

## Architecture (built for extension)

```
b1500_io.py       parse CSV  -> list[Measurement] -> list[Curve]   (no UI, no mpl)
nature_style.py   StyleConfig + rcParams + axis styling + fonts     (single source of truth)
preprocess.py     PreprocessConfig: Id/Ig scaling + linear smoothing
fet_analysis.py   FET parameter extraction (ported from MATLAB) + annotation geometry
analysis.py       legacy Annotation / Analysis registry (kept for custom overlays)
b1500_plotter.py  PySide6 (Qt) + matplotlib GUI that wires the above together
```

### Extending the analysis

`fet_analysis.analyze_transfer_curve()` returns a `FetParams` dataclass; add a
field + an entry in `PARAM_TABLE` to surface a new metric in the Analyze table,
and a branch in `PlotterWindow._draw_annotations` to overlay it. The older
`analysis.py` registry remains available for free-form overlays:

```python
@register(Analysis(key="vth", label="Threshold voltage (mark Vth)",
                   applies_to={"transfer"}))
def vth(curve, ctx):
    # ... compute, then return drawable Annotation objects ...
    return [Annotation("vline", x=v_th, text=f"  V_th={v_th:.2f} V")]
```

It renders the
returned annotations against the correct axis. Annotation kinds: `point`,
`vline`, `hline`, `line`, `text`.

Further parameter work (mobility, hysteresis, contact resistance …) extends
`fet_analysis.py`, reusing the MATLAB logic in `../matlab/`.
