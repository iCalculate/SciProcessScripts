"""MOSFET Data Plotter — Nature-style B1500 transfer/output curves, Qt (PySide6).

Run::

    python b1500_plotter.py                # open empty, then load data from the UI
    python b1500_plotter.py <file-or-dir>  # preload a CSV file or a folder

Left: a tabbed control panel.  Right: a live Nature-styled matplotlib plot.

Tabs
    Plot        figure size/aspect, font, line & tick width, frame; title and
                axes settings (collapsible); Id + optional |Ig| overlay (same
                axis, or a right axis with a shared numeric range); colour
                strategy and per-curve show/hide/colour.
    Preprocess  rescale Id / Ig and smooth each channel independently (computed
                in linear space, then shown on any axis).  Refines the same plot
                without changing the Plot-tab styling.
    Analyze     compute FET parameters (subthreshold swing, on/off ratio, Vth,
                Von, gm,max — ported from the MATLAB analyzer), list them, and
                overlay the selected ones (SS tangent, Ion/Ioff levels, ...).

Only this file is Qt-specific; b1500_io, nature_style, fet_analysis and
preprocess are toolkit-independent so the framework stays easy to extend.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Dict, List, Optional

import numpy as np


# --- Qt binding selection (prefer PySide6; matplotlib needs Qt >= 5.10) ------ #
def _configure_qt_plugin_path(module) -> None:
    """Pin Qt's plugin path + DLL search to *this* binding's own directory.

    With several Qt bindings in one environment (e.g. Anaconda's PyQt5/Qt5 next
    to a pip PySide6/Qt6) the wrong platform plugin can shadow the chosen one and
    cause "no Qt platform plugin could be initialized".  Forcing the paths fixes
    it deterministically.
    """
    base = os.path.dirname(getattr(module, "__file__", "") or "")
    if not base:
        return
    plugins = os.path.join(base, "plugins")
    if os.path.isdir(plugins):
        os.environ["QT_PLUGIN_PATH"] = plugins
        platforms = os.path.join(plugins, "platforms")
        if os.path.isdir(platforms):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms
    os.environ["PATH"] = base + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory") and os.path.isdir(base):
        try:
            os.add_dll_directory(base)
        except OSError:
            pass


def _select_qt_binding() -> str:
    import importlib
    for api, module in (("pyside6", "PySide6"), ("pyqt6", "PyQt6"),
                        ("pyside2", "PySide2"), ("pyqt5", "PyQt5")):
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        os.environ["QT_API"] = api
        _configure_qt_plugin_path(mod)
        return module
    raise ImportError(
        "No Qt binding found. Install one of: PySide6 (recommended), PyQt6.")


_QT_MODULE = _select_qt_binding()

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import b1500_io as io  # noqa: E402
import fet_analysis as fa  # noqa: E402
import nature_style as ns  # noqa: E402
import preprocess as pp  # noqa: E402

CURRENT_COLUMNS_PREF = ["Id", "absId", "Ig", "absIg", "Is", "Jd", "gm"]

APP_NAME = "MOSFET Data Plotter"
APP_VERSION = "1.0"

# On-screen preview render resolution. Proportions are identical at any DPI;
# this only sets the base crispness before the image is scaled to the window.
PREVIEW_DPI = 200


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def pretty_axis_label(col: str) -> str:
    # mathtext subscripts; rendered in the body font via the custom mathtext
    # fontset (matplotlib >= 3.8).
    table = {
        "Vg": r"$V_\mathrm{g}$ (V)", "Vd": r"$V_\mathrm{d}$ (V)",
        "Vs": r"$V_\mathrm{s}$ (V)", "Vds": r"$V_\mathrm{ds}$ (V)",
        "Id": r"$I_\mathrm{d}$ (A)", "absId": r"$|I_\mathrm{d}|$ (A)",
        "Ig": r"$I_\mathrm{g}$ (A)", "absIg": r"$|I_\mathrm{g}|$ (A)",
        "Is": r"$I_\mathrm{s}$ (A)", "gm": r"$g_\mathrm{m}$ (S)",
        "Jd": r"$J_\mathrm{d}$ (A/mm)",
    }
    return table.get(col, col)


def _parse_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Small custom widgets
# --------------------------------------------------------------------------- #

class CollapsibleBox(QtWidgets.QWidget):
    """A titled section whose body can be expanded/collapsed by clicking it."""

    def __init__(self, title: str, expanded: bool = True):
        super().__init__()
        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(expanded)
        self.toggle.setStyleSheet(
            "QToolButton { border: none; font-weight: bold; }")
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(
            QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.toggle.toggled.connect(self._on_toggle)

        self._content = QtWidgets.QWidget()
        self._content.setVisible(expanded)
        self._body = QtWidgets.QVBoxLayout(self._content)
        self._body.setContentsMargins(10, 2, 2, 2)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.toggle)
        lay.addWidget(self._content)

    def _on_toggle(self, checked: bool):
        self.toggle.setArrowType(
            QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow)
        self._content.setVisible(checked)

    def body(self) -> QtWidgets.QVBoxLayout:
        return self._body


class RangeEdit(QtWidgets.QLineEdit):
    """Axis min/max field that shows the live auto value in grey until edited.

    While in *auto* mode it displays the current autoscaled limit (grey) and is
    treated as "no manual limit"; once the user types a value it turns black and
    overrides the autoscale.  Clearing the field returns it to auto.
    """

    def __init__(self, on_change):
        super().__init__()
        self.auto = True
        self._on_change = on_change
        self.setFixedWidth(66)
        self.textEdited.connect(self._user_edited)
        self.editingFinished.connect(self._finished)
        self._restyle()

    def _user_edited(self, txt):
        self.auto = txt.strip() == ""
        self._restyle()

    def _finished(self):
        if self.text().strip() == "":
            self.auto = True
        self._restyle()
        self._on_change()

    def _restyle(self):
        # Muted grey for auto (visible on light & dark themes); empty stylesheet
        # for manual so it inherits the theme's normal text colour.
        self.setStyleSheet("color:#8a8a8a;" if self.auto else "")

    def show_auto_value(self, value: float):
        """If in auto mode, display the current autoscaled limit (grey)."""
        if self.auto and value is not None and np.isfinite(value):
            self.blockSignals(True)
            self.setText(f"{value:g}")
            self.blockSignals(False)
            self._restyle()

    def manual_value(self):
        return None if self.auto else _parse_float(self.text())


class SmoothControls(QtWidgets.QWidget):
    """One channel's smoothing controls: enable, method, params, live strength."""

    def __init__(self, channel_label: str, on_change):
        super().__init__()
        self._on_change = on_change
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)

        self.enable = QtWidgets.QCheckBox(f"Smooth {channel_label}")
        self.enable.toggled.connect(self._changed)
        grid.addWidget(self.enable, 0, 0, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Method"), 1, 0)
        self.method = QtWidgets.QComboBox()
        self.method.addItems(pp.SMOOTH_METHODS)
        self.method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.method.setMinimumContentsLength(10)
        self.method.currentIndexChanged.connect(self._changed)
        grid.addWidget(self.method, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Window"), 2, 0)
        self.window = QtWidgets.QSpinBox()
        self.window.setRange(3, 201)
        self.window.setSingleStep(2)
        self.window.setValue(7)
        self.window.setMinimumWidth(78)
        self.window.valueChanged.connect(self._changed)
        grid.addWidget(self.window, 2, 1)

        self.poly_label = QtWidgets.QLabel("Poly order")
        grid.addWidget(self.poly_label, 3, 0)
        self.poly = QtWidgets.QSpinBox()
        self.poly.setRange(1, 6)
        self.poly.setValue(2)
        self.poly.setMinimumWidth(78)
        self.poly.valueChanged.connect(self._changed)
        grid.addWidget(self.poly, 3, 1)

        self.sigma_label = QtWidgets.QLabel("Sigma")
        grid.addWidget(self.sigma_label, 4, 0)
        self.sigma = QtWidgets.QDoubleSpinBox()
        self.sigma.setRange(0.3, 50)
        self.sigma.setSingleStep(0.5)
        self.sigma.setValue(2.0)
        self.sigma.setMinimumWidth(78)
        self.sigma.valueChanged.connect(self._changed)
        grid.addWidget(self.sigma, 4, 1)
        grid.setColumnStretch(1, 0)
        for s in (self.window, self.poly, self.sigma):
            s.setMaximumWidth(90)

        self.strength = QtWidgets.QLabel()
        self.strength.setStyleSheet("color:#666;")
        grid.addWidget(self.strength, 5, 0, 1, 2)

        # --- Optional noise floor, applied AFTER smoothing ---
        self.noise_enable = QtWidgets.QCheckBox("Add noise floor")
        self.noise_enable.setToolTip("Add a noise floor after smoothing")
        self.noise_enable.toggled.connect(self._changed)
        grid.addWidget(self.noise_enable, 6, 0, 1, 2)
        self.noise_level_label = QtWidgets.QLabel("Level (A)")
        grid.addWidget(self.noise_level_label, 7, 0)
        self.noise_level = QtWidgets.QLineEdit("1e-12")
        self.noise_level.setFixedWidth(80)
        self.noise_level.editingFinished.connect(self._changed)
        grid.addWidget(self.noise_level, 7, 1)
        self.noise_seed_label = QtWidgets.QLabel("Seed")
        grid.addWidget(self.noise_seed_label, 8, 0)
        seed_row = QtWidgets.QHBoxLayout()
        self.noise_seed = QtWidgets.QSpinBox()
        self.noise_seed.setRange(0, 99999)
        self.noise_seed.valueChanged.connect(self._changed)
        seed_row.addWidget(self.noise_seed, 1)
        self.noise_random = QtWidgets.QCheckBox("Random")
        self.noise_random.toggled.connect(self._changed)
        seed_row.addWidget(self.noise_random)
        self.noise_reroll = QtWidgets.QPushButton("↻")
        self.noise_reroll.setFixedWidth(28)
        self.noise_reroll.setToolTip("Draw a new random noise pattern")
        self.noise_reroll.clicked.connect(self._reroll_noise)
        seed_row.addWidget(self.noise_reroll)
        grid.addLayout(seed_row, 8, 1)

        self._random_seed = random.randint(0, 99999)
        self._update_visibility()

    def _reroll_noise(self):
        self._random_seed = random.randint(0, 99999)
        self._update_visibility()
        self._on_change()

    def _changed(self, *_):
        # In random-seed mode, any parameter change draws a fresh noise pattern.
        if self.noise_random.isChecked():
            self._random_seed = random.randint(0, 99999)
        self._update_visibility()
        self._on_change()

    def _update_visibility(self):
        method = self.method.currentText()
        is_sg = method == "Savitzky–Golay"
        is_g = method == "Gaussian"
        smoothing = self.enable.isChecked()
        for w in (self.method, self.window, self.poly_label, self.poly,
                  self.sigma_label, self.sigma, self.strength):
            w.setEnabled(smoothing)
        self.poly_label.setVisible(is_sg)
        self.poly.setVisible(is_sg)
        self.sigma_label.setVisible(is_g)
        self.sigma.setVisible(is_g)
        self.window.setEnabled(smoothing and not is_g)
        self.strength.setText(self.spec().strength_text())
        noise_on = self.noise_enable.isChecked()
        random_on = self.noise_random.isChecked()
        for w in (self.noise_level_label, self.noise_level,
                  self.noise_seed_label, self.noise_random):
            w.setEnabled(noise_on)
        self.noise_reroll.setEnabled(noise_on and random_on)
        # Manual spinbox only when not random; show the live random seed otherwise.
        self.noise_seed.setEnabled(noise_on and not random_on)
        if random_on:
            self.noise_seed.blockSignals(True)
            self.noise_seed.setValue(self._random_seed)
            self.noise_seed.blockSignals(False)

    def spec(self) -> pp.SmoothSpec:
        return pp.SmoothSpec(
            enabled=self.enable.isChecked(), method=self.method.currentText(),
            window=self.window.value(), polyorder=self.poly.value(),
            sigma=self.sigma.value())

    def noise_spec(self) -> pp.NoiseFloorSpec:
        seed = self._random_seed if self.noise_random.isChecked() \
            else self.noise_seed.value()
        return pp.NoiseFloorSpec(
            enabled=self.noise_enable.isChecked(),
            level=_parse_float(self.noise_level.text()) or 0.0,
            seed=seed)

    def set_channel_available(self, available: bool):
        self.setEnabled(available)
        if not available:
            self.enable.setChecked(False)
            self.noise_enable.setChecked(False)

    def reset(self):
        """Restore default smoothing + noise settings (no replot signal)."""
        widgets = (self.enable, self.method, self.window, self.poly, self.sigma,
                   self.noise_enable, self.noise_level, self.noise_seed,
                   self.noise_random)
        for w in widgets:
            w.blockSignals(True)
        self.enable.setChecked(False)
        self.method.setCurrentIndex(0)
        self.window.setValue(7)
        self.poly.setValue(2)
        self.sigma.setValue(2.0)
        self.noise_enable.setChecked(False)
        self.noise_level.setText("1e-12")
        self.noise_seed.setValue(0)
        self.noise_random.setChecked(False)
        self._random_seed = random.randint(0, 99999)
        for w in widgets:
            w.blockSignals(False)
        self._update_visibility()


class CurveRow:
    """Widgets + state for a single curve in the Curves table."""

    def __init__(self, curve: io.Curve, color: str, on_change, on_pick):
        self.curve = curve
        self.color = color
        self.customized = False

        self.chk = QtWidgets.QCheckBox()
        self.chk.setChecked(True)
        self.chk.toggled.connect(lambda *_: on_change())

        self.swatch = QtWidgets.QPushButton()
        self.swatch.setFixedSize(30, 20)
        self.swatch.clicked.connect(lambda *_: on_pick(self))
        self._apply_swatch()

        self.edit = QtWidgets.QLineEdit(curve.label)
        self.edit.editingFinished.connect(lambda *_: on_change())

    def _apply_swatch(self):
        self.swatch.setStyleSheet(
            f"background-color: {self.color}; border: 1px solid #888;")

    def set_color(self, hexc: str, customized: bool = True):
        self.color = hexc
        self.customized = customized or self.customized
        self._apply_swatch()

    @property
    def visible(self) -> bool:
        return self.chk.isChecked()

    @property
    def label(self) -> str:
        return self.edit.text()


class PreviewLabel(QtWidgets.QLabel):
    """True-WYSIWYG preview: shows the figure rendered at its fixed W×H size,
    scaled *uniformly* to fit the area.

    Because the whole rendered image is scaled as one unit, every element —
    fonts, line widths, spacing — keeps the exact same proportion no matter how
    the window is resized, and matches the Copy Image / Export output.
    """

    def __init__(self):
        super().__init__()
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(80, 80)
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#3a3a3a"))
        self.setPalette(pal)
        self._pix = None

    def set_figure_pixmap(self, pix: "QtGui.QPixmap"):
        self._pix = pix
        self._rescale()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self):
        if self._pix is None or self._pix.isNull():
            return
        super().setPixmap(self._pix.scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


class ExportDialog(QtWidgets.QDialog):
    """Lets the user pick which artifacts to export (image / data / config)."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Export")
        lay = QtWidgets.QVBoxLayout(self)

        img = QtWidgets.QGroupBox("Image")
        il = QtWidgets.QVBoxLayout(img)
        self.png_chk = QtWidgets.QCheckBox("PNG (raster, ≥300 dpi)")
        self.png_chk.setChecked(True)
        self.svg_chk = QtWidgets.QCheckBox("SVG (vector, editable text)")
        il.addWidget(self.png_chk)
        il.addWidget(self.svg_chk)
        lay.addWidget(img)

        data = QtWidgets.QGroupBox("Data")
        dl = QtWidgets.QVBoxLayout(data)
        self.csv_chk = QtWidgets.QCheckBox(
            "CSV — post-processed, original B1500 format & filename")
        dl.addWidget(self.csv_chk)
        lay.addWidget(data)

        cfgb = QtWidgets.QGroupBox("Configuration")
        cl = QtWidgets.QVBoxLayout(cfgb)
        self.json_chk = QtWidgets.QCheckBox("Plot + preprocess settings (JSON)")
        cl.addWidget(self.json_chk)
        lay.addWidget(cfgb)

        btns = QtWidgets.QHBoxLayout()
        all_btn = QtWidgets.QPushButton("Select all")
        all_btn.clicked.connect(self._select_all)
        btns.addWidget(all_btn)
        btns.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        btns.addWidget(bb)
        lay.addLayout(btns)

    def _select_all(self):
        for chk in (self.png_chk, self.svg_chk, self.csv_chk, self.json_chk):
            chk.setChecked(True)

    def selections(self) -> dict:
        return {"png": self.png_chk.isChecked(), "svg": self.svg_chk.isChecked(),
                "csv": self.csv_chk.isChecked(), "json": self.json_chk.isChecked()}


class PreferencesDialog(QtWidgets.QDialog):
    """Grouped editor for the default style parameters (typography, figure,
    lines/frame, colours)."""

    def __init__(self, parent, prefs: "ns.StyleConfig"):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(420, 620)
        outer = QtWidgets.QVBoxLayout(self)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        body = QtWidgets.QWidget()
        scroll.setWidget(body)
        v = QtWidgets.QVBoxLayout(body)
        outer.addWidget(scroll, 1)

        def group(title):
            g = QtWidgets.QGroupBox(title)
            v.addWidget(g)
            return QtWidgets.QFormLayout(g)

        def dspin(lo, hi, step, val, decimals=2):
            s = QtWidgets.QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setSingleStep(step)
            s.setDecimals(decimals)
            s.setValue(val)
            s.setMinimumWidth(90)
            return s

        # --- Typography ---
        t = group("Typography")
        self.font_combo = QtWidgets.QComboBox()
        fonts = ns.get_available_fonts()
        self.font_combo.addItems(fonts)
        if prefs.font_family in fonts:
            self.font_combo.setCurrentText(prefs.font_family)
        t.addRow("Font", self.font_combo)
        self.fontsize_spin = dspin(4, 40, 0.5, prefs.font_size)
        t.addRow("Font size", self.fontsize_spin)

        # --- Figure size ---
        f = group("Figure")
        self.width_spin = dspin(1.0, 20, 0.1, prefs.width_in)
        self.height_spin = dspin(1.0, 20, 0.1, prefs.height_in)
        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(prefs.dpi)
        self.dpi_spin.setMinimumWidth(90)
        f.addRow("Width (in)", self.width_spin)
        f.addRow("Height (in)", self.height_spin)
        f.addRow("DPI", self.dpi_spin)

        # --- Lines & frame ---
        lf = group("Lines & frame")
        self.linewidth_spin = dspin(0.3, 6, 0.1, prefs.line_width)
        self.tickwidth_spin = dspin(0.3, 6, 0.1, prefs.tick_width)
        self.borderwidth_spin = dspin(0.3, 6, 0.1, prefs.axes_line_width)
        self.ticklen_spin = dspin(0, 12, 0.5, prefs.tick_length)
        lf.addRow("Line width", self.linewidth_spin)
        lf.addRow("Tick width", self.tickwidth_spin)
        lf.addRow("Border width", self.borderwidth_spin)
        lf.addRow("Tick length", self.ticklen_spin)
        self.fullbox_chk = QtWidgets.QCheckBox("Full box (4 spines)")
        self.fullbox_chk.setChecked(prefs.full_box)
        lf.addRow(self.fullbox_chk)
        self.minor_chk = QtWidgets.QCheckBox("Minor ticks")
        self.minor_chk.setChecked(prefs.minor_ticks)
        lf.addRow(self.minor_chk)

        # --- Colours ---
        cg = group("Colours")
        self.colormode_combo = QtWidgets.QComboBox()
        self.colormode_combo.addItems(["sequential", "categorical"])
        self.colormode_combo.setCurrentText(prefs.color_mode)
        cg.addRow("Default mode", self.colormode_combo)
        self.ramp_combo = QtWidgets.QComboBox()
        self.ramp_combo.addItems(list(ns.SEQUENTIAL_RAMPS.keys()))
        self.ramp_combo.setCurrentText(prefs.ramp)
        cg.addRow("Default ramp", self.ramp_combo)

        self._palette = list(prefs.palette)
        self._swatches = []
        sw_row = QtWidgets.QHBoxLayout()
        sw_row.setSpacing(4)
        for i, col in enumerate(self._palette):
            b = QtWidgets.QPushButton()
            b.setFixedSize(26, 22)
            b.setStyleSheet(f"background:{col}; border:1px solid #888;")
            b.clicked.connect(lambda _=False, idx=i: self._pick_palette(idx))
            self._swatches.append(b)
            sw_row.addWidget(b)
        sw_row.addStretch(1)
        cg.addRow("Categorical", self._wrap(sw_row))

        note = QtWidgets.QLabel(
            "Font, palette, tick length and minor-ticks apply immediately; the "
            "other numeric defaults take effect on Reset / next launch. Saved "
            "across sessions.")
        note.setStyleSheet("color:#888;")
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    @staticmethod
    def _wrap(layout):
        w = QtWidgets.QWidget()
        w.setLayout(layout)
        return w

    def _pick_palette(self, idx):
        col = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(self._palette[idx]), self, "Palette colour")
        if col.isValid():
            self._palette[idx] = col.name()
            self._swatches[idx].setStyleSheet(
                f"background:{col.name()}; border:1px solid #888;")

    def updated_prefs(self, base: "ns.StyleConfig") -> "ns.StyleConfig":
        import dataclasses
        return dataclasses.replace(
            base, font_family=self.font_combo.currentText(),
            width_in=self.width_spin.value(), height_in=self.height_spin.value(),
            font_size=self.fontsize_spin.value(),
            line_width=self.linewidth_spin.value(),
            tick_width=self.tickwidth_spin.value(),
            axes_line_width=self.borderwidth_spin.value(),
            tick_length=self.ticklen_spin.value(),
            full_box=self.fullbox_chk.isChecked(),
            minor_ticks=self.minor_chk.isChecked(),
            color_mode=self.colormode_combo.currentText(),
            ramp=self.ramp_combo.currentText(),
            dpi=int(self.dpi_spin.value()),
            palette=list(self._palette))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

class PlotterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 900)

        # Preferred defaults (Edit → Preferences), persisted across sessions.
        self.prefs = self._load_preferences()
        import dataclasses
        self.cfg = dataclasses.replace(self.prefs)
        self.pre = pp.PreprocessConfig()
        self.measurements: List[io.Measurement] = []
        self.measurement: Optional[io.Measurement] = None
        self.rows: List[CurveRow] = []
        self.fet_params: Dict[int, fa.FetParams] = {}
        self._col_combos: Dict[str, QtWidgets.QComboBox] = {}

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._refresh_plot)

        self._build_ui()
        self._refresh_plot()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_ui(self):
        self._build_menus()
        self._build_statusbar()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        # ---- Left: Data (always visible) + Tabs + Actions ----
        left = QtWidgets.QWidget()
        left.setFixedWidth(470)
        left_lay = QtWidgets.QVBoxLayout(left)
        left_lay.setContentsMargins(6, 6, 6, 6)
        self._build_data_section(left_lay)

        self.tabs = QtWidgets.QTabWidget()
        left_lay.addWidget(self.tabs, 1)
        plot_lay = self._make_tab("Plot")
        pre_lay = self._make_tab("Preprocess")
        ana_lay = self._make_tab("Analyze")

        self._build_figure_section(plot_lay)
        self._build_title_section(plot_lay)
        self._build_axes_section(plot_lay)
        self._build_curves_section(plot_lay)
        plot_lay.addStretch(1)

        self._build_preprocess_section(pre_lay)
        pre_lay.addStretch(1)

        self._build_analyze_section(ana_lay)
        ana_lay.addStretch(1)

        self._build_actions(left_lay)
        root.addWidget(left)

        # ---- Right: fixed-proportion WYSIWYG preview ----
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)
        # The Figure stays at the chosen export size; the canvas is used purely
        # as an off-screen Agg renderer, and the result is shown scaled in the
        # preview label (so proportions never change with the window).
        self.fig = Figure(figsize=(self.cfg.width_in, self.cfg.height_in),
                          constrained_layout=True, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.preview = PreviewLabel()
        rlay.addWidget(self.preview, 1)
        root.addWidget(right, 1)

    def _make_tab(self, name: str) -> QtWidgets.QVBoxLayout:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        # Never overflow horizontally — keep everything within the panel width.
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        lay = QtWidgets.QVBoxLayout(content)
        lay.setSpacing(8)
        self.tabs.addTab(scroll, name)
        return lay

    def _group(self, lay, title: str) -> QtWidgets.QVBoxLayout:
        box = QtWidgets.QGroupBox(title)
        lay.addWidget(box)
        inner = QtWidgets.QVBoxLayout(box)
        inner.setSpacing(5)
        return inner

    def _schedule(self):
        self._timer.start()

    # ---- Menus & status bar ------------------------------------------- #
    def _build_menus(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")
        file_menu.addAction("Open File…", self.load_file)
        file_menu.addAction("Open Folder…", self.load_folder)
        file_menu.addSeparator()
        file_menu.addAction("Export…", self.export)
        file_menu.addSeparator()
        reset_act = file_menu.addAction("Reset All Settings", self._reset_settings)
        reset_act.setStatusTip("Restore every plot and preprocessing setting to defaults")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction("Preferences…", self._open_preferences)
        edit_menu.addAction("Reset All Settings", self._reset_settings)

        view_menu = mb.addMenu("&View")
        view_menu.addAction("Copy Image to Clipboard", self.copy_to_clipboard)
        view_menu.addAction("Reset Axes to Auto", self._reset_axes_auto)

        help_menu = mb.addMenu("&Help")
        help_menu.addAction("About", self._show_about)

    def _build_statusbar(self):
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)
        # Permanent indicator on the right that is always visible.
        self.status_state = QtWidgets.QLabel("Ready")
        self.status_state.setStyleSheet("color:#2a7f2a; padding:0 8px;")
        sb.addPermanentWidget(self.status_state)
        self.status_info = QtWidgets.QLabel("No data loaded")
        sb.addWidget(self.status_info)

    def _set_state(self, text: str, busy: bool = False):
        self.status_state.setText(text)
        self.status_state.setStyleSheet(
            ("color:#b06a00;" if busy else "color:#2a7f2a;") + " padding:0 8px;")
        self.status_state.repaint()   # show immediately, even mid-operation

    def _reset_settings(self):
        """Restore every plot and preprocessing setting to the preferred defaults."""
        import dataclasses
        self._set_state("Resetting…", busy=True)
        c = dataclasses.replace(self.prefs)   # defaults from Edit → Preferences
        self.cfg = c
        self.pre = pp.PreprocessConfig()
        widgets = [self.width_spin, self.height_spin, self.fontsize_spin,
                   self.linewidth_spin, self.tickwidth_spin, self.borderwidth_spin,
                   self.dpi_spin, self.fullbox_chk, self.colormode_combo,
                   self.ramp_combo, self.legend_chk, self.colorbar_chk,
                   self.title_edit, self.show_ig_chk, self.ig_axis_combo,
                   self.xscale_spin, self.xoffset_spin, self.idscale_spin,
                   self.igscale_spin, self.ann_ss_chk, self.ann_onoff_chk,
                   self.ann_vth_chk, self.ann_gm_chk]
        for w in widgets:
            w.blockSignals(True)
        self.width_spin.setValue(c.width_in)
        self.height_spin.setValue(c.height_in)
        self.fontsize_spin.setValue(c.font_size)
        self.linewidth_spin.setValue(c.line_width)
        self.tickwidth_spin.setValue(c.tick_width)
        self.borderwidth_spin.setValue(c.axes_line_width)
        self.dpi_spin.setValue(c.dpi)
        self.fullbox_chk.setChecked(c.full_box)
        self.colormode_combo.setCurrentText(c.color_mode)
        self.ramp_combo.setCurrentText(c.ramp)
        self.legend_chk.setChecked(True)
        self.colorbar_chk.setChecked(False)
        self.title_edit.clear()
        self.show_ig_chk.setChecked(False)
        self.ig_axis_combo.setCurrentIndex(0)
        self.xscale_spin.setValue(1.0)
        self.xoffset_spin.setValue(0.0)
        self.idscale_spin.setValue(1.0)
        self.igscale_spin.setValue(1.0)
        for a in (self.ann_ss_chk, self.ann_onoff_chk, self.ann_vth_chk,
                  self.ann_gm_chk):
            a.setChecked(False)
        for w in widgets:
            w.blockSignals(False)
        self.id_smooth_ctrl.reset()
        self.ig_smooth_ctrl.reset()
        # Re-apply per-measurement defaults (axis labels, columns, ranges→auto).
        if self.measurement:
            self._load_measurement(apply_defaults=True)
        else:
            self._refresh_plot()
        self._set_state("Ready")
        self.statusBar().showMessage("All settings reset to defaults", 3000)

    def _reset_axes_auto(self):
        for e in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            e.auto = True
            e.blockSignals(True)
            e.clear()
            e.blockSignals(False)
        self._refresh_plot()

    def _show_about(self):
        QtWidgets.QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<b>{APP_NAME}</b> v{APP_VERSION}<br><br>"
            "Nature-style plotting and analysis for Keysight B1500A "
            "transfer / output curves.")

    # ---- Preferences (defaults) --------------------------------------- #
    _PREF_KEYS = ("font_family", "width_in", "height_in", "font_size",
                  "line_width", "tick_width", "axes_line_width", "tick_length",
                  "dpi", "full_box", "minor_ticks", "color_mode", "ramp",
                  "palette")

    def _pref_path(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "preferences.json")

    def _load_preferences(self) -> "ns.StyleConfig":
        cfg = ns.StyleConfig()
        try:
            import json
            with open(self._pref_path(), encoding="utf-8") as fh:
                data = json.load(fh)
            for k in self._PREF_KEYS:
                if k in data:
                    setattr(cfg, k, data[k])
        except (OSError, ValueError):
            pass
        return cfg

    def _save_preferences(self):
        import json
        data = {k: getattr(self.prefs, k) for k in self._PREF_KEYS}
        try:
            with open(self._pref_path(), "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError:
            pass

    def _open_preferences(self):
        dlg = PreferencesDialog(self, self.prefs)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        self.prefs = dlg.updated_prefs(self.prefs)
        self._save_preferences()
        # Apply the styles that have no per-figure control immediately; the rest
        # are defaults that take effect on Reset.
        self.cfg.font_family = self.prefs.font_family
        self.cfg.palette = list(self.prefs.palette)
        self.cfg.tick_length = self.prefs.tick_length
        self.cfg.minor_ticks = self.prefs.minor_ticks
        if self.rows:
            self._assign_curve_colors()   # repaint swatches with the new palette
        self._refresh_plot()
        self.statusBar().showMessage("Preferences saved", 3000)

    # ---- Data ---------------------------------------------------------- #
    def _build_data_section(self, parent):
        lay = self._group(parent, "Data")
        btns = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Open file…")
        b1.clicked.connect(self.load_file)
        b2 = QtWidgets.QPushButton("Open folder…")
        b2.clicked.connect(self.load_folder)
        btns.addWidget(b1)
        btns.addWidget(b2)
        lay.addLayout(btns)
        self.meas_combo = QtWidgets.QComboBox()
        self.meas_combo.currentIndexChanged.connect(self._on_measurement_selected)
        lay.addWidget(self.meas_combo)
        self.meas_info = QtWidgets.QLabel("No data loaded.")
        self.meas_info.setStyleSheet("color:#666;")
        self.meas_info.setWordWrap(True)
        lay.addWidget(self.meas_info)

    # ---- Figure -------------------------------------------------------- #
    def _build_figure_section(self, parent):
        lay = self._group(parent, "Figure")
        grid = QtWidgets.QGridLayout()
        lay.addLayout(grid)

        self.width_spin = self._dspin(1.0, 20, 0.1, self.cfg.width_in)
        self.height_spin = self._dspin(1.0, 20, 0.1, self.cfg.height_in)
        self.fontsize_spin = self._dspin(4, 40, 0.5, self.cfg.font_size)
        self.linewidth_spin = self._dspin(0.3, 6, 0.1, self.cfg.line_width)
        self.tickwidth_spin = self._dspin(0.3, 6, 0.1, self.cfg.tick_width)
        self.borderwidth_spin = self._dspin(0.3, 6, 0.1, self.cfg.axes_line_width)
        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(self.cfg.dpi)
        self.dpi_spin.setMinimumWidth(78)
        self.dpi_spin.valueChanged.connect(self._schedule)
        for spin in (self.width_spin, self.height_spin):
            spin.valueChanged.connect(self._on_size_changed)
        for spin in (self.fontsize_spin, self.linewidth_spin, self.tickwidth_spin,
                     self.borderwidth_spin):
            spin.valueChanged.connect(self._schedule)

        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        def cell(text, widget, r, c):
            lab = QtWidgets.QLabel(text)
            lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(lab, r, c * 2)
            grid.addWidget(widget, r, c * 2 + 1)

        cell("Width", self.width_spin, 0, 0)
        cell("Height", self.height_spin, 0, 1)
        cell("Font", self.fontsize_spin, 1, 0)
        cell("Line", self.linewidth_spin, 1, 1)
        cell("Tick", self.tickwidth_spin, 2, 0)
        cell("Border", self.borderwidth_spin, 2, 1)
        cell("DPI", self.dpi_spin, 3, 0)
        # Spin columns share the slack so values fill the panel neatly.
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)

        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Preset:"))
        p1 = QtWidgets.QPushButton("89 mm")
        p1.setToolTip("Nature single column (89 mm)")
        p1.clicked.connect(lambda: self._set_size(ns.SINGLE_COLUMN_IN,
                                                  ns.SINGLE_COLUMN_IN * 0.8))
        p2 = QtWidgets.QPushButton("183 mm")
        p2.setToolTip("Nature double column (183 mm)")
        p2.clicked.connect(lambda: self._set_size(ns.DOUBLE_COLUMN_IN,
                                                 ns.DOUBLE_COLUMN_IN * 0.62))
        presets.addWidget(p1)
        presets.addWidget(p2)
        presets.addStretch(1)
        lay.addLayout(presets)

        row = QtWidgets.QHBoxLayout()
        self.fullbox_chk = QtWidgets.QCheckBox("Full box")
        self.fullbox_chk.setToolTip("Draw all four axis spines")
        self.fullbox_chk.setChecked(self.cfg.full_box)
        self.fullbox_chk.toggled.connect(self._schedule)
        row.addWidget(self.fullbox_chk)
        row.addStretch(1)
        lay.addLayout(row)

    def _dspin(self, lo, hi, step, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(2)
        s.setValue(val)
        s.setMinimumWidth(78)
        return s

    def _on_size_changed(self):
        # Figure W:H drives the preview proportions directly via _render_preview.
        self._schedule()

    def _set_size(self, w, h):
        for spin, v in ((self.width_spin, w), (self.height_spin, h)):
            spin.blockSignals(True)
            spin.setValue(round(v, 2))
            spin.blockSignals(False)
        self._on_size_changed()

    # ---- Title (collapsible) ------------------------------------------ #
    def _build_title_section(self, parent):
        box = CollapsibleBox("Title", expanded=False)
        parent.addWidget(box)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Title"))
        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.editingFinished.connect(self._schedule)
        row.addWidget(self.title_edit, 1)
        box.body().addLayout(row)

    # ---- Axes (collapsible) ------------------------------------------- #
    def _build_axes_section(self, parent):
        box = CollapsibleBox("Axes", expanded=True)
        parent.addWidget(box)
        lay = box.body()

        self.xlabel_edit, self.xmin_edit, self.xmax_edit = self._axis_row(lay, "X label")
        (self.ycol_combo, self.ylabel_edit, self.ymin_edit, self.ymax_edit,
         self.ylog_chk, self.yabs_chk) = self._ycol_row(lay, "Left Y")

        # --- Id + Ig content (requirement 1a) ---
        r1 = QtWidgets.QHBoxLayout()
        r1.setSpacing(4)
        self.show_ig_chk = QtWidgets.QCheckBox("Overlay |Ig|")
        self.show_ig_chk.toggled.connect(self._schedule)
        r1.addWidget(self.show_ig_chk)
        r1.addWidget(QtWidgets.QLabel("axis"))
        self.ig_axis_combo = QtWidgets.QComboBox()
        self.ig_axis_combo.addItems(["Same as Id", "Right (shared)"])
        self.ig_axis_combo.setToolTip(
            "Right axis shares the same numeric range as the left axis")
        self.ig_axis_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ig_axis_combo.setMinimumContentsLength(8)
        self.ig_axis_combo.currentIndexChanged.connect(self._schedule)
        r1.addWidget(self.ig_axis_combo, 1)
        lay.addLayout(r1)

        r2 = QtWidgets.QHBoxLayout()
        r2.setSpacing(4)
        r2.addWidget(QtWidgets.QLabel("Right Y"))
        self.y2label_edit = QtWidgets.QLineEdit(pretty_axis_label("absIg"))
        self.y2label_edit.editingFinished.connect(self._schedule)
        r2.addWidget(self.y2label_edit, 1)
        lay.addLayout(r2)

    def _range_row(self):
        """A compact min/max line that never overflows the panel width."""
        rng = QtWidgets.QHBoxLayout()
        rng.setSpacing(3)
        rng.addWidget(QtWidgets.QLabel("min"))
        mn = RangeEdit(self._schedule)
        rng.addWidget(mn)
        rng.addSpacing(6)
        rng.addWidget(QtWidgets.QLabel("max"))
        mx = RangeEdit(self._schedule)
        rng.addWidget(mx)
        return rng, mn, mx

    def _axis_row(self, lay, label):
        v = QtWidgets.QVBoxLayout()
        v.setSpacing(3)
        lay.addLayout(v)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(label))
        lab = QtWidgets.QLineEdit()
        lab.editingFinished.connect(self._schedule)
        top.addWidget(lab, 1)
        v.addLayout(top)
        rng, mn, mx = self._range_row()
        rng.addStretch(1)
        v.addLayout(rng)
        return lab, mn, mx

    def _ycol_row(self, lay, label):
        v = QtWidgets.QVBoxLayout()
        v.setSpacing(3)
        lay.addLayout(v)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(label))
        combo = QtWidgets.QComboBox()
        combo.currentIndexChanged.connect(self._on_ycol_change)
        self._col_combos[label] = combo
        top.addWidget(combo)
        lab = QtWidgets.QLineEdit()
        lab.editingFinished.connect(self._schedule)
        top.addWidget(lab, 1)
        v.addLayout(top)
        rng, mn, mx = self._range_row()
        rng.addSpacing(8)
        log = QtWidgets.QCheckBox("log")
        log.toggled.connect(self._schedule)
        abschk = QtWidgets.QCheckBox("abs")
        abschk.toggled.connect(self._schedule)
        rng.addWidget(log)
        rng.addWidget(abschk)
        rng.addStretch(1)
        v.addLayout(rng)
        return combo, lab, mn, mx, log, abschk

    # ---- Curves -------------------------------------------------------- #
    def _build_curves_section(self, parent):
        lay = self._group(parent, "Curves")
        crow = QtWidgets.QHBoxLayout()
        crow.setSpacing(4)
        crow.addWidget(QtWidgets.QLabel("Colour"))
        self.colormode_combo = QtWidgets.QComboBox()
        self.colormode_combo.addItems(["sequential", "categorical"])
        self.colormode_combo.setCurrentText(self.cfg.color_mode)
        self.colormode_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.colormode_combo.setMinimumContentsLength(7)
        self.colormode_combo.currentIndexChanged.connect(self._recolor_and_plot)
        crow.addWidget(self.colormode_combo, 1)
        crow.addWidget(QtWidgets.QLabel("Ramp"))
        self.ramp_combo = QtWidgets.QComboBox()
        self.ramp_combo.addItems(list(ns.SEQUENTIAL_RAMPS.keys()))
        self.ramp_combo.setCurrentText(self.cfg.ramp)
        self.ramp_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.ramp_combo.setMinimumContentsLength(6)
        self.ramp_combo.currentIndexChanged.connect(self._recolor_and_plot)
        crow.addWidget(self.ramp_combo, 1)
        lay.addLayout(crow)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        ball = QtWidgets.QPushButton("Show")
        ball.setToolTip("Show all curves")
        ball.clicked.connect(lambda: self._set_all_visible(True))
        bnone = QtWidgets.QPushButton("Hide")
        bnone.setToolTip("Hide all curves")
        bnone.clicked.connect(lambda: self._set_all_visible(False))
        bar.addWidget(ball)
        bar.addWidget(bnone)
        bar.addStretch(1)
        self.colorbar_chk = QtWidgets.QCheckBox("Bar")
        self.colorbar_chk.setToolTip("Show a colourbar instead of a legend")
        self.colorbar_chk.toggled.connect(self._schedule)
        self.legend_chk = QtWidgets.QCheckBox("Legend")
        self.legend_chk.setChecked(True)
        self.legend_chk.toggled.connect(self._schedule)
        bar.addWidget(self.colorbar_chk)
        bar.addWidget(self.legend_chk)
        lay.addLayout(bar)

        self.curves_box = QtWidgets.QVBoxLayout()
        self.curves_box.setSpacing(2)
        lay.addLayout(self.curves_box)

    def _build_curve_rows(self):
        self._clear_layout(self.curves_box)
        self.rows = []
        for c in self.measurement.curves:
            row = CurveRow(c, "#000000", self._schedule, self._pick_color)
            self.rows.append(row)
            line = QtWidgets.QHBoxLayout()
            line.addWidget(row.chk)
            line.addWidget(row.swatch)
            line.addWidget(row.edit, 1)
            holder = QtWidgets.QWidget()
            holder.setLayout(line)
            self.curves_box.addWidget(holder)
        self._assign_curve_colors()

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())

    def _pick_color(self, row: CurveRow):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor(row.color), self,
                                              "Curve colour")
        if col.isValid():
            row.set_color(col.name(), customized=True)
            self._refresh_plot()

    def _set_all_visible(self, val: bool):
        for r in self.rows:
            r.chk.blockSignals(True)
            r.chk.setChecked(val)
            r.chk.blockSignals(False)
        self._refresh_plot()

    def _assign_curve_colors(self):
        self.cfg.color_mode = self.colormode_combo.currentText()
        self.cfg.ramp = self.ramp_combo.currentText()
        seq = self.cfg.sequence_colors(len(self.rows))
        for r, c in zip(self.rows, seq):
            if not r.customized:
                r.set_color(c, customized=False)

    def _recolor_and_plot(self):
        if self.rows:
            self._assign_curve_colors()
        self._refresh_plot()

    # ---- Preprocess ---------------------------------------------------- #
    def _build_preprocess_section(self, parent):
        # X (sweep) axis transform: x_out = x * scale + offset.
        xg = self._group(parent, "X axis (× scale + offset)")
        xgrid = QtWidgets.QGridLayout()
        xg.addLayout(xgrid)
        xgrid.addWidget(QtWidgets.QLabel("Scale ×"), 0, 0)
        self.xscale_spin = QtWidgets.QDoubleSpinBox()
        self.xscale_spin.setRange(-1e6, 1e6)
        self.xscale_spin.setDecimals(4)
        self.xscale_spin.setSingleStep(0.1)
        self.xscale_spin.setValue(1.0)
        self.xscale_spin.setMinimumWidth(96)
        self.xscale_spin.valueChanged.connect(self._schedule)
        xgrid.addWidget(self.xscale_spin, 0, 1)
        xgrid.addWidget(QtWidgets.QLabel("Offset +"), 1, 0)
        self.xoffset_spin = QtWidgets.QDoubleSpinBox()
        self.xoffset_spin.setRange(-1e9, 1e9)
        self.xoffset_spin.setDecimals(4)
        self.xoffset_spin.setSingleStep(1.0)
        self.xoffset_spin.setValue(0.0)
        self.xoffset_spin.setMinimumWidth(96)
        self.xoffset_spin.valueChanged.connect(self._schedule)
        xgrid.addWidget(self.xoffset_spin, 1, 1)
        xgrid.setColumnStretch(2, 1)

        scale = self._group(parent, "Scaling (× current)")
        grid = QtWidgets.QGridLayout()
        scale.addLayout(grid)
        grid.addWidget(QtWidgets.QLabel("Id × "), 0, 0)
        self.idscale_spin = self._dspin(1e-6, 1e6, 0.1, 1.0)
        self.idscale_spin.setDecimals(4)
        self.idscale_spin.setFixedWidth(96)
        self.idscale_spin.valueChanged.connect(self._schedule)
        grid.addWidget(self.idscale_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Ig × "), 1, 0)
        self.igscale_spin = self._dspin(1e-6, 1e6, 0.1, 1.0)
        self.igscale_spin.setDecimals(4)
        self.igscale_spin.setFixedWidth(96)
        self.igscale_spin.valueChanged.connect(self._schedule)
        grid.addWidget(self.igscale_spin, 1, 1)
        grid.setColumnStretch(2, 1)

        # Independent smoothing for the drain and gate currents.
        id_g = self._group(parent, "Id smoothing")
        self.id_smooth_ctrl = SmoothControls("Id", self._schedule)
        id_g.addWidget(self.id_smooth_ctrl)

        ig_g = self._group(parent, "Ig smoothing")
        self.ig_smooth_ctrl = SmoothControls("Ig", self._schedule)
        ig_g.addWidget(self.ig_smooth_ctrl)

        note = QtWidgets.QLabel(
            "Smoothing is computed in linear current space, then displayed on "
            "whatever axis (linear or log) you choose.")
        note.setStyleSheet("color:#666;")
        note.setWordWrap(True)
        ig_g.addWidget(note)

    # ---- Analyze ------------------------------------------------------- #
    def _build_analyze_section(self, parent):
        lay = self._group(parent, "FET parameters")
        self.analyze_btn = QtWidgets.QPushButton("Compute parameters")
        self.analyze_btn.clicked.connect(self._compute_parameters)
        lay.addWidget(self.analyze_btn)
        self.analyze_hint = QtWidgets.QLabel(
            "Computes per-curve parameters for transfer measurements.")
        self.analyze_hint.setStyleSheet("color:#666;")
        self.analyze_hint.setWordWrap(True)
        lay.addWidget(self.analyze_hint)

        # Parameters are rows; each curve (e.g. a Vd step) is a column.
        self.param_table = QtWidgets.QTableWidget()
        self.param_table.setRowCount(len(fa.PARAM_TABLE))
        self.param_table.setVerticalHeaderLabels([lbl for _, lbl, _ in fa.PARAM_TABLE])
        self.param_table.horizontalHeader().setDefaultSectionSize(96)
        self.param_table.verticalHeader().setDefaultSectionSize(22)
        self.param_table.setMinimumHeight(220)
        lay.addWidget(self.param_table)

        ann = self._group(parent, "Annotate on plot")
        self.ann_ss_chk = QtWidgets.QCheckBox("Subthreshold slope (SS tangent)")
        self.ann_onoff_chk = QtWidgets.QCheckBox("On/Off levels (Ion, Ioff lines)")
        self.ann_vth_chk = QtWidgets.QCheckBox("Vth (vertical line)")
        self.ann_gm_chk = QtWidgets.QCheckBox("gm,max (vertical line)")
        for chk in (self.ann_ss_chk, self.ann_onoff_chk, self.ann_vth_chk, self.ann_gm_chk):
            chk.toggled.connect(self._schedule)
            ann.addWidget(chk)

    def _compute_parameters(self):
        self.fet_params = {}
        if not self.measurement:
            return
        if self.measurement.kind != io.MeasurementKind.TRANSFER:
            self.analyze_hint.setText("Parameter extraction applies to transfer "
                                      "(Id–Vg) measurements only.")
            self.param_table.setColumnCount(0)
            return
        self._set_state("Computing…", busy=True)
        id_spec = self.id_smooth_ctrl.spec()
        sm_win = id_spec.window if id_spec.enabled else 5
        # Columns = curves; rows = parameters (transposed layout).
        self.param_table.setColumnCount(len(self.rows))
        self.param_table.setHorizontalHeaderLabels([r.label for r in self.rows])
        for i, r in enumerate(self.rows):
            vg = r.curve.get(self.measurement.x_name)
            idd = r.curve.get("Id")
            if idd is None:
                idd = r.curve.get("absId")
            p = fa.analyze_transfer_curve(vg, idd, secondary_value=r.curve.secondary_value,
                                          smoothing_pts=sm_win)
            self.fet_params[i] = p
            for j, (key, _lbl, fmt) in enumerate(fa.PARAM_TABLE):
                val = getattr(p, key)
                txt = fmt(val) if val is not None and np.isfinite(val) else "—"
                self.param_table.setItem(j, i, QtWidgets.QTableWidgetItem(txt))
        self.analyze_hint.setText(
            "Computed. Tick an annotation below to overlay it on the plot.")
        self._refresh_plot()

    # ---- Actions ------------------------------------------------------- #
    def _build_actions(self, parent):
        row = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("Reset")
        reset.setToolTip("Reset all settings to defaults (axes back to Auto)")
        reset.clicked.connect(self._reset_settings)
        replot = QtWidgets.QPushButton("Apply / Replot")
        replot.clicked.connect(self._refresh_plot)
        copy = QtWidgets.QPushButton("Copy image")
        copy.clicked.connect(self.copy_to_clipboard)
        export = QtWidgets.QPushButton("Export…")
        export.clicked.connect(self.export)
        row.addWidget(reset)
        row.addWidget(replot)
        row.addWidget(copy)
        row.addWidget(export)
        parent.addLayout(row)

    # ------------------------------------------------------------------ #
    # Clipboard
    # ------------------------------------------------------------------ #
    def copy_to_clipboard(self):
        """Render the figure at export geometry and put it on the clipboard."""
        if not self.measurement:
            return
        import io as _io
        self._set_state("Copying…", busy=True)
        self._sync_cfg()
        buf = _io.BytesIO()
        # Render exactly like the preview (full figure, same layout), just at the
        # export DPI — so the clipboard image matches the preview pixel-for-pixel
        # in proportion.
        with matplotlib.rc_context(ns.rc_context(self.cfg)):
            self.fig.set_dpi(self.cfg.dpi)
            self.fig.set_size_inches(self.cfg.width_in, self.cfg.height_in)
            self.fig.savefig(buf, format="png", dpi=self.cfg.dpi)
        self._render_preview()
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)
        self._set_state("Ready")
        self.statusBar().showMessage("Figure copied to clipboard", 2500)

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open B1500 CSV", "", "CSV files (*.csv);;All files (*)")
        if path:
            self._ingest(io.read_b1500_csv(path), path)

    def load_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Open folder of B1500 CSV files")
        if folder:
            self._ingest(io.load_folder(folder), folder)

    def preload(self, path: str):
        if os.path.isdir(path):
            self._ingest(io.load_folder(path), path)
        elif os.path.isfile(path):
            self._ingest(io.read_b1500_csv(path), path)

    def _ingest(self, measurements, source):
        if not measurements:
            QtWidgets.QMessageBox.warning(
                self, "No data", f"No B1500 measurements found in:\n{source}")
            return
        self._set_state("Loading…", busy=True)
        self.measurements = measurements
        self.meas_combo.blockSignals(True)
        self.meas_combo.clear()
        self.meas_combo.addItems(
            [f"{i+1:02d}. {m.name}" for i, m in enumerate(measurements)])
        self.meas_combo.setCurrentIndex(0)
        self.meas_combo.blockSignals(False)
        # A freshly loaded dataset starts from defaults.
        self._load_measurement(apply_defaults=True)

    # ------------------------------------------------------------------ #
    # Measurement selection
    # ------------------------------------------------------------------ #
    def _on_measurement_selected(self, *_):
        # Switching the data source must NOT wipe the user's settings — only the
        # curve list (and column choices) follow the data; labels, ranges, scale,
        # smoothing, etc. are preserved. Use Reset to return to defaults.
        self._load_measurement(apply_defaults=False)

    def _load_measurement(self, apply_defaults: bool):
        idx = self.meas_combo.currentIndex()
        if idx < 0 or idx >= len(self.measurements):
            return
        m = self.measurements[idx]
        self.measurement = m
        self.fet_params = {}
        self.param_table.setColumnCount(0)

        self.meas_info.setText(
            f"{m.kind}  •  {m.n_curves} curve(s)  •  x = {m.x_name}"
            + (f"  •  {m.secondary_name}-stepped" if m.secondary_name else ""))

        avail = [c for c in CURRENT_COLUMNS_PREF if m.curves and m.curves[0].has(c)]
        if not avail:
            avail = [n for n in (m.curves[0].names if m.curves else []) if n != m.x_name]
        prev_y = self.ycol_combo.currentText()
        self._set_combo_items(self.ycol_combo, avail)

        has_ig = bool(m.curves and (m.curves[0].has("Ig") or m.curves[0].has("absIg")))
        self.show_ig_chk.setEnabled(has_ig)
        if not has_ig:
            self.show_ig_chk.setChecked(False)
        self.ig_smooth_ctrl.set_channel_available(has_ig)

        if apply_defaults or prev_y not in avail:
            # First load / Reset, or the previous column no longer exists.
            self._apply_measurement_defaults(m, avail)
        else:
            # Preserve the user's column selection across the switch.
            self._set_combo_text(self.ycol_combo, prev_y)

        self._build_curve_rows()
        self._refresh_plot()

    def _apply_measurement_defaults(self, m, avail):
        """Set the default columns/labels/scales/ranges for a measurement kind."""
        if m.kind == io.MeasurementKind.TRANSFER:
            ydef = "absId" if "absId" in avail else ("Id" if "Id" in avail else avail[0])
            self.ylog_chk.setChecked(True)
            self.yabs_chk.setChecked(not ydef.startswith("abs"))
        else:
            ydef = "Id" if "Id" in avail else avail[0]
            self.ylog_chk.setChecked(False)
            self.yabs_chk.setChecked(False)
        self._set_combo_text(self.ycol_combo, ydef)
        self.xlabel_edit.setText(pretty_axis_label(m.x_name))
        self.ylabel_edit.setText(pretty_axis_label(ydef))
        self.y2label_edit.setText(pretty_axis_label("absIg"))
        for e in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            e.auto = True
            e.blockSignals(True)
            e.clear()
            e.blockSignals(False)

    def _set_combo_items(self, combo, items):
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.blockSignals(False)

    def _set_combo_text(self, combo, text):
        combo.blockSignals(True)
        i = combo.findText(text)
        if i >= 0:
            combo.setCurrentIndex(i)
        combo.blockSignals(False)

    def _on_ycol_change(self, *_):
        col = self.ycol_combo.currentText()
        if col:
            self.ylabel_edit.setText(pretty_axis_label(col))
        self._refresh_plot()

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def _sync_cfg(self):
        self.cfg.width_in = float(self.width_spin.value())
        self.cfg.height_in = float(self.height_spin.value())
        self.cfg.dpi = int(self.dpi_spin.value())
        self.cfg.font_size = float(self.fontsize_spin.value())
        self.cfg.line_width = float(self.linewidth_spin.value())
        self.cfg.tick_width = float(self.tickwidth_spin.value())
        self.cfg.axes_line_width = float(self.borderwidth_spin.value())
        # Font family is a preference (Edit → Preferences), not a per-figure field.
        self.cfg.font_family = self.prefs.font_family
        self.cfg.full_box = self.fullbox_chk.isChecked()
        self.cfg.color_mode = self.colormode_combo.currentText()
        self.cfg.ramp = self.ramp_combo.currentText()
        self.pre.x_scale = float(self.xscale_spin.value())
        self.pre.x_offset = float(self.xoffset_spin.value())
        self.pre.id_scale = float(self.idscale_spin.value())
        self.pre.ig_scale = float(self.igscale_spin.value())
        self.pre.id_smooth = self.id_smooth_ctrl.spec()
        self.pre.ig_smooth = self.ig_smooth_ctrl.spec()
        self.pre.id_noise = self.id_smooth_ctrl.noise_spec()
        self.pre.ig_noise = self.ig_smooth_ctrl.noise_spec()

    def _series(self, curve, col, use_abs, log):
        x = curve.get(self.measurement.x_name)
        y = curve.get(col)
        if x is None or y is None:
            return None, None
        x = np.asarray(x, float)
        # Preprocess (scale + linear smoothing) on the raw current before
        # abs/log masking — smoothing is always done in linear space.
        x, y = pp.apply_series(self.pre, x, np.asarray(y, float), col)
        if use_abs:
            y = np.abs(y)
        mask = np.isfinite(x) & np.isfinite(y)
        if log:
            mask &= y > 0
        return x[mask], y[mask]

    def _refresh_plot(self):
        self._sync_cfg()
        self._set_state("Plotting…", busy=True)
        with matplotlib.rc_context(ns.rc_context(self.cfg)):
            self.fig.clear()
            ax = self.fig.add_subplot(111)

            if not self.measurement:
                ax.text(0.5, 0.5, "Open a B1500 CSV file or folder to begin",
                        ha="center", va="center", transform=ax.transAxes,
                        color="#888")
                ns.style_axes(ax, self.cfg, log_y=False)
                self._render_preview()
                self._set_state("Ready")
                return

            ycol = self.ycol_combo.currentText()
            log_y = self.ylog_chk.isChecked()
            self._draw_axis(ax, ycol, self.yabs_chk.isChecked(), log_y, False)

            # --- Id + Ig content (requirement 1a) ---
            ax2 = None
            if self.show_ig_chk.isChecked() and self.show_ig_chk.isEnabled():
                ig_col = "absIg" if self.measurement.curves[0].has("absIg") else "Ig"
                if self.ig_axis_combo.currentIndex() == 0:       # same axis as Id
                    self._draw_axis(ax, ig_col, True, log_y, False, dashed=True)
                else:                                            # right axis
                    ax2 = ax.twinx()
                    self._draw_axis(ax2, ig_col, True, log_y, True, dashed=True)

            ax.set_xlabel(self.xlabel_edit.text())
            ax.set_ylabel(self.ylabel_edit.text())
            if self.title_edit.text().strip():
                ax.set_title(self.title_edit.text().strip())
            self._apply_range(ax, "x", self.xmin_edit, self.xmax_edit)
            self._apply_range(ax, "y", self.ymin_edit, self.ymax_edit)

            if log_y:
                ax.set_yscale("log")
            if ax2 is not None:
                # Shared numeric range between the two axes (requirement 1a).
                if log_y:
                    ax2.set_yscale("log")
                ax2.set_ylim(ax.get_ylim())
                ax2.set_ylabel(self.y2label_edit.text())

            self._draw_annotations(ax)
            if not self._maybe_colorbar(ax):
                self._draw_legend(ax)

            ns.style_axes(ax, self.cfg, log_y=log_y)
            if ax2 is not None:
                ns.style_axes(ax2, self.cfg, log_y=log_y, right_axis=True)

            self._refresh_auto_ranges(ax, ax2)
            self._render_preview()
        n_vis = sum(1 for r in self.rows if r.visible)
        self.status_info.setText(
            f"{self.measurement.kind} · {n_vis}/{len(self.rows)} curve(s) shown")
        self._set_state("Ready")

    def _render_preview(self):
        """Render the figure at its fixed export size and show it scaled to fit.

        The figure is always drawn at width×height inches; only the preview DPI
        sets pixel density. Because the *whole* image is then scaled uniformly to
        the preview area, font/line proportions are locked to the figure and the
        preview matches the exported file exactly.
        """
        import numpy as _np
        self.fig.set_dpi(PREVIEW_DPI)
        self.fig.set_size_inches(self.cfg.width_in, self.cfg.height_in)
        self.canvas.draw()
        arr = _np.asarray(self.canvas.buffer_rgba())
        h, w = arr.shape[:2]
        qimg = QtGui.QImage(arr.data, w, h, QtGui.QImage.Format_RGBA8888).copy()
        self.preview.set_figure_pixmap(QtGui.QPixmap.fromImage(qimg))

    def _draw_axis(self, ax, col, use_abs, log, right, dashed=False):
        for r in self.rows:
            if not r.visible:
                continue
            x, y = self._series(r.curve, col, use_abs, log)
            if x is None or x.size == 0:
                continue
            style = dict(color=r.color, linewidth=self.cfg.line_width, label=r.label)
            if dashed:
                style["linestyle"] = "--"
                style["label"] = "_nolegend_"
            if right:
                style["label"] = "_nolegend_"
            ax.plot(x, y, **style)

    def _apply_range(self, ax, axis, min_edit, max_edit):
        lo, hi = min_edit.manual_value(), max_edit.manual_value()
        if lo is None and hi is None:
            return
        cur = ax.get_xlim() if axis == "x" else ax.get_ylim()
        lo = cur[0] if lo is None else lo
        hi = cur[1] if hi is None else hi
        if axis == "x":
            ax.set_xlim(lo, hi)
        else:
            ax.set_ylim(lo, hi)

    def _refresh_auto_ranges(self, ax, ax2):
        """Show the live autoscaled limits (grey) in any 'auto' range field."""
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        self.xmin_edit.show_auto_value(xlo)
        self.xmax_edit.show_auto_value(xhi)
        self.ymin_edit.show_auto_value(ylo)
        self.ymax_edit.show_auto_value(yhi)

    def _maybe_colorbar(self, ax) -> bool:
        m = self.measurement
        if not self.colorbar_chk.isChecked() or self.cfg.color_mode != "sequential":
            return False
        vis = [r for r in self.rows if r.visible]
        if len(vis) < 2 or not m.secondary_name:
            return False
        vals = [r.curve.secondary_value for r in vis
                if r.curve.secondary_value is not None]
        if len(vals) < 2:
            return False
        cmap = ns.build_cmap(self.cfg.ramp)
        norm = matplotlib.colors.Normalize(vmin=min(vals), vmax=max(vals))
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
        cbar.outline.set_linewidth(self.cfg.axes_line_width)
        cbar.ax.tick_params(width=self.cfg.tick_width, length=self.cfg.tick_length,
                            labelsize=self.cfg.tick_label_size)
        cbar.set_label(pretty_axis_label(m.secondary_name),
                       fontsize=self.cfg.label_size)
        return True

    # ---- Analysis annotations ----------------------------------------- #
    def _draw_annotations(self, ax):
        if not self.fet_params:
            return
        show_ss = self.ann_ss_chk.isChecked()
        show_onoff = self.ann_onoff_chk.isChecked()
        show_vth = self.ann_vth_chk.isChecked()
        show_gm = self.ann_gm_chk.isChecked()
        if not any((show_ss, show_onoff, show_vth, show_gm)):
            return
        n_vis = sum(1 for r in self.rows if r.visible)
        label_ok = n_vis <= 3   # avoid label spam on dense families
        for i, r in enumerate(self.rows):
            if not r.visible:
                continue
            p = self.fet_params.get(i)
            if p is None:
                continue
            color = r.color
            xt = self.pre.x_transform   # match the plotted (scaled/offset) x axis
            if show_ss:
                vg_l, id_l = p.subthreshold_line()
                if vg_l.size:
                    vg_l = xt(vg_l)
                    ax.plot(vg_l, id_l, color=color, linestyle="-",
                            linewidth=self.cfg.axes_line_width, alpha=0.9)
                    if label_ok and np.isfinite(p.ss_mV_dec):
                        mid = vg_l.size // 2
                        ax.annotate(f"SS={p.ss_mV_dec:.0f} mV/dec",
                                    (vg_l[mid], id_l[mid]), color=color,
                                    fontsize=self.cfg.legend_size,
                                    xytext=(4, 0), textcoords="offset points")
            if show_onoff:
                for lvl, name in ((p.ion, r"$I_\mathrm{on}$"),
                                  (p.ioff, r"$I_\mathrm{off}$")):
                    if lvl is not None and np.isfinite(lvl) and lvl > 0:
                        ax.axhline(lvl, color=color, linestyle="--",
                                   linewidth=self.cfg.axes_line_width, alpha=0.8)
                        if label_ok:
                            ax.annotate(name, (0.01, lvl), xycoords=("axes fraction", "data"),
                                        color=color, fontsize=self.cfg.legend_size,
                                        va="bottom")
            if show_vth and np.isfinite(p.vth):
                vth = float(xt(p.vth))
                ax.axvline(vth, color=color, linestyle=":",
                           linewidth=self.cfg.axes_line_width, alpha=0.8)
                if label_ok:
                    ax.annotate(r"$V_\mathrm{th}$", (vth, 0.02),
                                xycoords=("data", "axes fraction"), color=color,
                                fontsize=self.cfg.legend_size)
            if show_gm and np.isfinite(p.vth_gmmax):
                ax.axvline(float(xt(p.vth_gmmax)), color=color, linestyle="-.",
                           linewidth=self.cfg.axes_line_width, alpha=0.6)

    def _draw_legend(self, ax):
        if not self.legend_chk.isChecked():
            return
        handles, labels = ax.get_legend_handles_labels()
        keep = [(h, l) for h, l in zip(handles, labels) if l and not l.startswith("_")]
        if keep:
            ax.legend([h for h, _ in keep], [l for _, l in keep],
                      frameon=False, fontsize=self.cfg.legend_size)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    CURRENT_DATA_COLS = {"Id", "absId", "Ig", "absIg", "Is", "Jd"}

    def export(self):
        if not self.measurement:
            QtWidgets.QMessageBox.information(
                self, "Nothing to export", "Load and plot data first.")
            return
        dlg = ExportDialog(self)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return
        sel = dlg.selections()
        if not any(sel.values()):
            return

        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose export folder")
        if not directory:
            return
        self._set_state("Exporting…", busy=True)
        stem = os.path.splitext(os.path.basename(self.measurement.source_file))[0] \
            or "figure"
        self._sync_cfg()
        saved: List[str] = []

        # --- Image (PNG / SVG) — rendered identically to the preview ---
        if sel["png"] or sel["svg"]:
            with matplotlib.rc_context(ns.rc_context(self.cfg)):
                self.fig.set_dpi(self.cfg.dpi)
                self.fig.set_size_inches(self.cfg.width_in, self.cfg.height_in)
                if sel["png"]:
                    p = os.path.join(directory, stem + ".png")
                    self.fig.savefig(p, dpi=max(self.cfg.dpi, 300))
                    saved.append(p)
                if sel["svg"]:
                    p = os.path.join(directory, stem + ".svg")
                    self.fig.savefig(p)
                    saved.append(p)
            self._render_preview()

        # --- Data CSV: post-processed, original B1500 format & filename ---
        if sel["csv"]:
            p = os.path.join(directory, os.path.basename(self.measurement.source_file))
            if not p.lower().endswith(".csv"):
                p += ".csv"
            io.write_b1500_csv(p, self.measurement, self._processed_column)
            saved.append(p)

        # --- Configuration JSON ---
        if sel["json"]:
            import json
            p = os.path.join(directory, stem + ".plotcfg.json")
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self._collect_config(), fh, indent=2, ensure_ascii=False)
            saved.append(p)

        self._set_state("Ready")
        self.statusBar().showMessage(f"Exported {len(saved)} file(s)", 3000)
        QtWidgets.QMessageBox.information(self, "Exported", "Saved:\n" + "\n".join(saved))

    def _processed_column(self, curve, name):
        """Column values as plotted: current columns get the preprocess pipeline."""
        if name in self.CURRENT_DATA_COLS:
            _, y = pp.apply_series(self.pre, curve.get(self.measurement.x_name),
                                   curve.get(name), name)
            return y
        return curve.get(name)

    def _collect_config(self) -> dict:
        import dataclasses
        return {
            "measurement": os.path.basename(self.measurement.source_file),
            "style": dataclasses.asdict(self.cfg),
            "preprocess": dataclasses.asdict(self.pre),
            "axes": {
                "x_label": self.xlabel_edit.text(),
                "x_min": self.xmin_edit.manual_value(),
                "x_max": self.xmax_edit.manual_value(),
                "y_column": self.ycol_combo.currentText(),
                "y_label": self.ylabel_edit.text(),
                "y_min": self.ymin_edit.manual_value(),
                "y_max": self.ymax_edit.manual_value(),
                "y_log": self.ylog_chk.isChecked(),
                "y_abs": self.yabs_chk.isChecked(),
                "title": self.title_edit.text(),
                "show_ig": self.show_ig_chk.isChecked(),
                "ig_axis_mode": self.ig_axis_combo.currentIndex(),
                "y2_label": self.y2label_edit.text(),
            },
            "curves": {
                "color_mode": self.colormode_combo.currentText(),
                "ramp": self.ramp_combo.currentText(),
                "legend": self.legend_chk.isChecked(),
                "colorbar": self.colorbar_chk.isChecked(),
                "rows": [{"label": r.label, "color": r.color, "visible": r.visible}
                         for r in self.rows],
            },
            "analysis": {
                "ss": self.ann_ss_chk.isChecked(),
                "onoff": self.ann_onoff_chk.isChecked(),
                "vth": self.ann_vth_chk.isChecked(),
                "gm": self.ann_gm_chk.isChecked(),
            },
        }


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    win = PlotterWindow()
    win.show()
    if len(sys.argv) > 1:
        win.preload(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
