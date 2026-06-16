"""B1500 Transfer / Output curve plotter — Nature-style, Qt (PySide6) GUI.

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
from matplotlib.backends.backend_qtagg import (  # noqa: E402
    FigureCanvasQTAgg, NavigationToolbar2QT)
from matplotlib.figure import Figure  # noqa: E402

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import b1500_io as io  # noqa: E402
import fet_analysis as fa  # noqa: E402
import nature_style as ns  # noqa: E402
import preprocess as pp  # noqa: E402

CURRENT_COLUMNS_PREF = ["Id", "absId", "Ig", "absIg", "Is", "Jd", "gm"]

APP_NAME = "MOSFET Data Plotter"
APP_VERSION = "1.0"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def pretty_axis_label(col: str) -> str:
    table = {
        "Vg": r"$V_\mathrm{g}$ (V)", "Vd": r"$V_\mathrm{d}$ (V)",
        "Vs": r"$V_\mathrm{s}$ (V)", "Vds": r"$V_\mathrm{ds}$ (V)",
        "Id": r"$I_\mathrm{d}$ (A)", "absId": r"$|I_\mathrm{d}|$ (A)",
        "Ig": r"$I_\mathrm{g}$ (A)", "absIg": r"$|I_\mathrm{g}|$ (A)",
        "Is": r"$I_\mathrm{s}$ (A)", "gm": r"$g_\mathrm{m}$ (S)",
        "Jd": r"$J_\mathrm{d}$ (A mm$^{-1}$)",
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
        self.setFixedWidth(78)
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
        self.setStyleSheet("color:#999;" if self.auto else "color:black;")

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
        self.method.currentIndexChanged.connect(self._changed)
        grid.addWidget(self.method, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Window (pts)"), 2, 0)
        self.window = QtWidgets.QSpinBox()
        self.window.setRange(3, 201)
        self.window.setSingleStep(2)
        self.window.setValue(7)
        self.window.valueChanged.connect(self._changed)
        grid.addWidget(self.window, 2, 1)

        self.poly_label = QtWidgets.QLabel("Poly order")
        grid.addWidget(self.poly_label, 3, 0)
        self.poly = QtWidgets.QSpinBox()
        self.poly.setRange(1, 6)
        self.poly.setValue(2)
        self.poly.valueChanged.connect(self._changed)
        grid.addWidget(self.poly, 3, 1)

        self.sigma_label = QtWidgets.QLabel("Sigma (pts)")
        grid.addWidget(self.sigma_label, 4, 0)
        self.sigma = QtWidgets.QDoubleSpinBox()
        self.sigma.setRange(0.3, 50)
        self.sigma.setSingleStep(0.5)
        self.sigma.setValue(2.0)
        self.sigma.valueChanged.connect(self._changed)
        grid.addWidget(self.sigma, 4, 1)

        self.strength = QtWidgets.QLabel()
        self.strength.setStyleSheet("color:#666;")
        grid.addWidget(self.strength, 5, 0, 1, 2)

        # --- Optional noise floor, applied AFTER smoothing ---
        self.noise_enable = QtWidgets.QCheckBox("Add noise floor (after smoothing)")
        self.noise_enable.toggled.connect(self._changed)
        grid.addWidget(self.noise_enable, 6, 0, 1, 2)
        self.noise_level_label = QtWidgets.QLabel("Level (A, RMS)")
        grid.addWidget(self.noise_level_label, 7, 0)
        self.noise_level = QtWidgets.QLineEdit("1e-12")
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


class AspectCanvas(QtWidgets.QWidget):
    """Hosts the matplotlib canvas and (optionally) letterboxes it to a ratio."""

    def __init__(self, canvas: FigureCanvasQTAgg):
        super().__init__()
        self._canvas = canvas
        canvas.setParent(self)
        self._ratio = 1.0
        self._lock = True
        # Dark-grey margins around a white figure so the page edge is obvious.
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor("#3a3a3a"))
        self.setPalette(pal)

    def set_ratio(self, ratio: float):
        self._ratio = max(ratio, 0.05)
        self._relayout()

    def set_lock(self, lock: bool):
        self._lock = lock
        self._relayout()

    def resizeEvent(self, event):
        self._relayout()
        super().resizeEvent(event)

    def _relayout(self):
        W, H = self.width(), self.height()
        if W < 10 or H < 10:
            return
        if not self._lock:
            self._canvas.setGeometry(0, 0, W, H)
            return
        w = W
        h = w / self._ratio
        if h > H:
            h = H
            w = h * self._ratio
        x, y = (W - w) / 2, (H - h) / 2
        self._canvas.setGeometry(int(x), int(y), int(max(w, 10)), int(max(h, 10)))


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


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #

class PlotterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1360, 900)

        self.cfg = ns.StyleConfig()
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
        left.setFixedWidth(410)
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

        # ---- Right: plot + toolbar ----
        right = QtWidgets.QWidget()
        rlay = QtWidgets.QVBoxLayout(right)
        rlay.setContentsMargins(4, 4, 4, 4)
        self.fig = Figure(figsize=(self.cfg.width_in, self.cfg.height_in),
                          constrained_layout=True, facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.aspect = AspectCanvas(self.canvas)
        self.aspect.set_ratio(self.cfg.width_in / self.cfg.height_in)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        rlay.addWidget(self.aspect, 1)
        rlay.addWidget(toolbar, 0)
        root.addWidget(right, 1)

    def _make_tab(self, name: str) -> QtWidgets.QVBoxLayout:
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
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

        view_menu = mb.addMenu("&View")
        view_menu.addAction("Copy Image to Clipboard", self.copy_to_clipboard)
        view_menu.addAction("Reset Axes to Auto", self._reset_axes_auto)
        self.lock_ratio_act = view_menu.addAction("Lock Preview Ratio")
        self.lock_ratio_act.setCheckable(True)
        self.lock_ratio_act.setChecked(True)
        self.lock_ratio_act.toggled.connect(self._toggle_lock_ratio)

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
        """Restore every plot and preprocessing setting to its default."""
        self._set_state("Resetting…", busy=True)
        self.cfg = ns.StyleConfig()
        self.pre = pp.PreprocessConfig()
        c = self.cfg
        widgets = [self.width_spin, self.height_spin, self.fontsize_spin,
                   self.linewidth_spin, self.tickwidth_spin, self.dpi_spin,
                   self.font_combo, self.fullbox_chk, self.lockaspect_chk,
                   self.colormode_combo, self.ramp_combo, self.legend_chk,
                   self.colorbar_chk, self.title_edit, self.show_ig_chk,
                   self.ig_axis_combo, self.idscale_spin, self.igscale_spin,
                   self.ann_ss_chk, self.ann_onoff_chk, self.ann_vth_chk,
                   self.ann_gm_chk]
        for w in widgets:
            w.blockSignals(True)
        self.width_spin.setValue(c.width_in)
        self.height_spin.setValue(c.height_in)
        self.fontsize_spin.setValue(c.font_size)
        self.linewidth_spin.setValue(c.line_width)
        self.tickwidth_spin.setValue(c.tick_width)
        self.dpi_spin.setValue(c.dpi)
        if c.font_family in ns.get_available_fonts():
            self.font_combo.setCurrentText(c.font_family)
        self.fullbox_chk.setChecked(c.full_box)
        self.lockaspect_chk.setChecked(True)
        self.colormode_combo.setCurrentText(c.color_mode)
        self.ramp_combo.setCurrentText(c.ramp)
        self.legend_chk.setChecked(True)
        self.colorbar_chk.setChecked(False)
        self.title_edit.clear()
        self.show_ig_chk.setChecked(False)
        self.ig_axis_combo.setCurrentIndex(0)
        self.idscale_spin.setValue(1.0)
        self.igscale_spin.setValue(1.0)
        for a in (self.ann_ss_chk, self.ann_onoff_chk, self.ann_vth_chk,
                  self.ann_gm_chk):
            a.setChecked(False)
        for w in widgets:
            w.blockSignals(False)
        self.id_smooth_ctrl.reset()
        self.ig_smooth_ctrl.reset()
        self.aspect.set_lock(True)
        self.aspect.set_ratio(c.width_in / c.height_in)
        if hasattr(self, "lock_ratio_act"):
            self.lock_ratio_act.setChecked(True)
        # Re-apply per-measurement defaults (axis labels, columns, curve rows).
        if self.measurement:
            self._on_measurement_selected()
        else:
            self._refresh_plot()
        self._set_state("Ready")
        self.statusBar().showMessage("All settings reset to defaults", 3000)

    def _toggle_lock_ratio(self, on: bool):
        self.lockaspect_chk.setChecked(on)  # keeps the panel checkbox in sync

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
        self.dpi_spin = QtWidgets.QSpinBox()
        self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setSingleStep(50)
        self.dpi_spin.setValue(self.cfg.dpi)
        self.dpi_spin.valueChanged.connect(self._schedule)
        for spin in (self.width_spin, self.height_spin):
            spin.valueChanged.connect(self._on_size_changed)
        for spin in (self.fontsize_spin, self.linewidth_spin, self.tickwidth_spin):
            spin.valueChanged.connect(self._schedule)

        grid.addWidget(QtWidgets.QLabel("Width (in)"), 0, 0)
        grid.addWidget(self.width_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Height (in)"), 0, 2)
        grid.addWidget(self.height_spin, 0, 3)
        grid.addWidget(QtWidgets.QLabel("Font size"), 1, 0)
        grid.addWidget(self.fontsize_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Line width"), 1, 2)
        grid.addWidget(self.linewidth_spin, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Tick width"), 2, 0)
        grid.addWidget(self.tickwidth_spin, 2, 1)
        grid.addWidget(QtWidgets.QLabel("DPI"), 2, 2)
        grid.addWidget(self.dpi_spin, 2, 3)
        grid.addWidget(QtWidgets.QLabel("Font"), 3, 0)
        self.font_combo = QtWidgets.QComboBox()
        fonts = ns.get_available_fonts()
        self.font_combo.addItems(fonts)
        if self.cfg.font_family in fonts:
            self.font_combo.setCurrentText(self.cfg.font_family)
        self.font_combo.currentIndexChanged.connect(self._schedule)
        grid.addWidget(self.font_combo, 3, 1, 1, 3)

        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Preset:"))
        p1 = QtWidgets.QPushButton("Single (89 mm)")
        p1.clicked.connect(lambda: self._set_size(ns.SINGLE_COLUMN_IN,
                                                  ns.SINGLE_COLUMN_IN * 0.8))
        p2 = QtWidgets.QPushButton("Double (183 mm)")
        p2.clicked.connect(lambda: self._set_size(ns.DOUBLE_COLUMN_IN,
                                                 ns.DOUBLE_COLUMN_IN * 0.62))
        presets.addWidget(p1)
        presets.addWidget(p2)
        lay.addLayout(presets)

        row = QtWidgets.QHBoxLayout()
        self.fullbox_chk = QtWidgets.QCheckBox("Full box (4 spines)")
        self.fullbox_chk.setChecked(self.cfg.full_box)
        self.fullbox_chk.toggled.connect(self._schedule)
        self.lockaspect_chk = QtWidgets.QCheckBox("Lock preview ratio")
        self.lockaspect_chk.setChecked(True)
        self.lockaspect_chk.toggled.connect(
            lambda v: (self.aspect.set_lock(v), self._schedule()))
        row.addWidget(self.fullbox_chk)
        row.addWidget(self.lockaspect_chk)
        row.addStretch(1)
        lay.addLayout(row)

    def _dspin(self, lo, hi, step, val):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(2)
        s.setValue(val)
        return s

    def _on_size_changed(self):
        self.aspect.set_ratio(self.width_spin.value() / max(self.height_spin.value(), 0.1))
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
        gl = QtWidgets.QGridLayout()
        lay.addLayout(gl)
        self.show_ig_chk = QtWidgets.QCheckBox("Overlay |Ig|")
        self.show_ig_chk.toggled.connect(self._schedule)
        gl.addWidget(self.show_ig_chk, 0, 0)
        gl.addWidget(QtWidgets.QLabel("Ig axis"), 0, 1)
        self.ig_axis_combo = QtWidgets.QComboBox()
        self.ig_axis_combo.addItems(["Same as Id", "Right axis (shared range)"])
        self.ig_axis_combo.currentIndexChanged.connect(self._schedule)
        gl.addWidget(self.ig_axis_combo, 0, 2)
        gl.addWidget(QtWidgets.QLabel("Right Y label"), 1, 1)
        self.y2label_edit = QtWidgets.QLineEdit(pretty_axis_label("absIg"))
        self.y2label_edit.editingFinished.connect(self._schedule)
        gl.addWidget(self.y2label_edit, 1, 2)

    def _axis_row(self, lay, label):
        grid = QtWidgets.QGridLayout()
        lay.addLayout(grid)
        grid.addWidget(QtWidgets.QLabel(label), 0, 0)
        lab = QtWidgets.QLineEdit()
        lab.editingFinished.connect(self._schedule)
        grid.addWidget(lab, 0, 1, 1, 3)
        grid.addWidget(QtWidgets.QLabel("min"), 1, 0)
        mn = RangeEdit(self._schedule)
        grid.addWidget(mn, 1, 1)
        grid.addWidget(QtWidgets.QLabel("max"), 1, 2)
        mx = RangeEdit(self._schedule)
        grid.addWidget(mx, 1, 3)
        return lab, mn, mx

    def _ycol_row(self, lay, label):
        grid = QtWidgets.QGridLayout()
        lay.addLayout(grid)
        grid.addWidget(QtWidgets.QLabel(label), 0, 0)
        combo = QtWidgets.QComboBox()
        combo.currentIndexChanged.connect(self._on_ycol_change)
        self._col_combos[label] = combo
        grid.addWidget(combo, 0, 1)
        lab = QtWidgets.QLineEdit()
        lab.editingFinished.connect(self._schedule)
        grid.addWidget(lab, 0, 2, 1, 2)
        grid.addWidget(QtWidgets.QLabel("min"), 1, 0)
        mn = RangeEdit(self._schedule)
        grid.addWidget(mn, 1, 1)
        grid.addWidget(QtWidgets.QLabel("max"), 1, 2)
        mx = RangeEdit(self._schedule)
        grid.addWidget(mx, 1, 3)
        opts = QtWidgets.QHBoxLayout()
        log = QtWidgets.QCheckBox("log")
        log.toggled.connect(self._schedule)
        abschk = QtWidgets.QCheckBox("abs")
        abschk.toggled.connect(self._schedule)
        opts.addWidget(log)
        opts.addWidget(abschk)
        opts.addStretch(1)
        grid.addLayout(opts, 2, 0, 1, 4)
        return combo, lab, mn, mx, log, abschk

    # ---- Curves -------------------------------------------------------- #
    def _build_curves_section(self, parent):
        lay = self._group(parent, "Curves")
        crow = QtWidgets.QHBoxLayout()
        crow.addWidget(QtWidgets.QLabel("Colour"))
        self.colormode_combo = QtWidgets.QComboBox()
        self.colormode_combo.addItems(["sequential", "categorical"])
        self.colormode_combo.setCurrentText(self.cfg.color_mode)
        self.colormode_combo.currentIndexChanged.connect(self._recolor_and_plot)
        crow.addWidget(self.colormode_combo)
        crow.addWidget(QtWidgets.QLabel("Ramp"))
        self.ramp_combo = QtWidgets.QComboBox()
        self.ramp_combo.addItems(list(ns.SEQUENTIAL_RAMPS.keys()))
        self.ramp_combo.setCurrentText(self.cfg.ramp)
        self.ramp_combo.currentIndexChanged.connect(self._recolor_and_plot)
        crow.addWidget(self.ramp_combo)
        lay.addLayout(crow)

        bar = QtWidgets.QHBoxLayout()
        ball = QtWidgets.QPushButton("Show all")
        ball.clicked.connect(lambda: self._set_all_visible(True))
        bnone = QtWidgets.QPushButton("Hide all")
        bnone.clicked.connect(lambda: self._set_all_visible(False))
        bar.addWidget(ball)
        bar.addWidget(bnone)
        bar.addStretch(1)
        self.colorbar_chk = QtWidgets.QCheckBox("Colourbar")
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
        scale = self._group(parent, "Scaling (multiplies current)")
        grid = QtWidgets.QGridLayout()
        scale.addLayout(grid)
        grid.addWidget(QtWidgets.QLabel("Id × "), 0, 0)
        self.idscale_spin = self._dspin(1e-6, 1e6, 0.1, 1.0)
        self.idscale_spin.setDecimals(4)
        self.idscale_spin.valueChanged.connect(self._schedule)
        grid.addWidget(self.idscale_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Ig × "), 1, 0)
        self.igscale_spin = self._dspin(1e-6, 1e6, 0.1, 1.0)
        self.igscale_spin.setDecimals(4)
        self.igscale_spin.valueChanged.connect(self._schedule)
        grid.addWidget(self.igscale_spin, 1, 1)

        # Independent smoothing for the drain and gate currents.
        id_g = self._group(parent, "Drain current Id smoothing")
        self.id_smooth_ctrl = SmoothControls("Id", self._schedule)
        id_g.addWidget(self.id_smooth_ctrl)

        ig_g = self._group(parent, "Gate current Ig smoothing")
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
        replot = QtWidgets.QPushButton("Apply / Replot")
        replot.clicked.connect(self._refresh_plot)
        copy = QtWidgets.QPushButton("Copy image")
        copy.clicked.connect(self.copy_to_clipboard)
        export = QtWidgets.QPushButton("Export…")
        export.clicked.connect(self.export)
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
        self._sync_cfg()
        buf = _io.BytesIO()
        old_size = self.fig.get_size_inches().copy()
        old_dpi = self.fig.get_dpi()
        self.fig.set_size_inches(self.cfg.width_in, self.cfg.height_in)
        try:
            with matplotlib.rc_context(ns.rc_context(self.cfg)):
                self.fig.savefig(buf, format="png", dpi=self.cfg.dpi,
                                 bbox_inches="tight")
        finally:
            self.fig.set_size_inches(*old_size)
            self.fig.set_dpi(old_dpi)
            self.canvas.draw_idle()
        img = QtGui.QImage.fromData(buf.getvalue(), "PNG")
        QtWidgets.QApplication.clipboard().setImage(img)
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
        self.measurements = measurements
        self.meas_combo.blockSignals(True)
        self.meas_combo.clear()
        self.meas_combo.addItems(
            [f"{i+1:02d}. {m.name}" for i, m in enumerate(measurements)])
        self.meas_combo.setCurrentIndex(0)
        self.meas_combo.blockSignals(False)
        self._on_measurement_selected()

    # ------------------------------------------------------------------ #
    # Measurement selection
    # ------------------------------------------------------------------ #
    def _on_measurement_selected(self, *_):
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
        self._set_combo_items(self.ycol_combo, avail)

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
        has_ig = bool(m.curves and (m.curves[0].has("Ig") or m.curves[0].has("absIg")))
        self.show_ig_chk.setEnabled(has_ig)
        if not has_ig:
            self.show_ig_chk.setChecked(False)
        self.ig_smooth_ctrl.set_channel_available(has_ig)
        for e in (self.xmin_edit, self.xmax_edit, self.ymin_edit, self.ymax_edit):
            e.auto = True
            e.blockSignals(True)
            e.clear()
            e.blockSignals(False)

        self._build_curve_rows()
        self._refresh_plot()

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
        self.cfg.font_family = self.font_combo.currentText()
        self.cfg.full_box = self.fullbox_chk.isChecked()
        self.cfg.color_mode = self.colormode_combo.currentText()
        self.cfg.ramp = self.ramp_combo.currentText()
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
                self.canvas.draw_idle()
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
            self.canvas.draw_idle()
        n_vis = sum(1 for r in self.rows if r.visible)
        self.status_info.setText(
            f"{self.measurement.kind} · {n_vis}/{len(self.rows)} curve(s) shown")
        self._set_state("Ready")

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
            if show_ss:
                vg_l, id_l = p.subthreshold_line()
                if vg_l.size:
                    ax.plot(vg_l, id_l, color=color, linestyle="-",
                            linewidth=self.cfg.axes_line_width, alpha=0.9)
                    if label_ok and np.isfinite(p.ss_mV_dec):
                        mid = vg_l.size // 2
                        ax.annotate(f"SS={p.ss_mV_dec:.0f} mV/dec",
                                    (vg_l[mid], id_l[mid]), color=color,
                                    fontsize=self.cfg.legend_size,
                                    xytext=(4, 0), textcoords="offset points")
            if show_onoff:
                for lvl, name in ((p.ion, "I$_\\mathrm{on}$"),
                                  (p.ioff, "I$_\\mathrm{off}$")):
                    if lvl is not None and np.isfinite(lvl) and lvl > 0:
                        ax.axhline(lvl, color=color, linestyle="--",
                                   linewidth=self.cfg.axes_line_width, alpha=0.8)
                        if label_ok:
                            ax.annotate(name, (0.01, lvl), xycoords=("axes fraction", "data"),
                                        color=color, fontsize=self.cfg.legend_size,
                                        va="bottom")
            if show_vth and np.isfinite(p.vth):
                ax.axvline(p.vth, color=color, linestyle=":",
                           linewidth=self.cfg.axes_line_width, alpha=0.8)
                if label_ok:
                    ax.annotate("V$_\\mathrm{th}$", (p.vth, 0.02),
                                xycoords=("data", "axes fraction"), color=color,
                                fontsize=self.cfg.legend_size)
            if show_gm and np.isfinite(p.vth_gmmax):
                ax.axvline(p.vth_gmmax, color=color, linestyle="-.",
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
        stem = os.path.splitext(os.path.basename(self.measurement.source_file))[0] \
            or "figure"
        self._sync_cfg()
        saved: List[str] = []

        # --- Image (PNG / SVG) at the export geometry ---
        if sel["png"] or sel["svg"]:
            old_size = self.fig.get_size_inches().copy()
            old_dpi = self.fig.get_dpi()
            self.fig.set_size_inches(self.cfg.width_in, self.cfg.height_in)
            try:
                with matplotlib.rc_context(ns.rc_context(self.cfg)):
                    if sel["png"]:
                        p = os.path.join(directory, stem + ".png")
                        self.fig.savefig(p, dpi=max(self.cfg.dpi, 300),
                                         bbox_inches="tight")
                        saved.append(p)
                    if sel["svg"]:
                        p = os.path.join(directory, stem + ".svg")
                        self.fig.savefig(p, bbox_inches="tight")
                        saved.append(p)
            finally:
                self.fig.set_size_inches(*old_size)
                self.fig.set_dpi(old_dpi)
                self.canvas.draw_idle()

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
    win = PlotterWindow()
    win.show()
    if len(sys.argv) > 1:
        win.preload(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
