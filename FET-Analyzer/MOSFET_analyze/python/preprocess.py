"""Data preprocessing applied to curves before plotting.

These transforms sit between the raw :class:`b1500_io.Curve` and the plot: they
rescale the drain / gate currents and optionally smooth them.  They do not touch
any styling — the plot parameters chosen on the Plot tab still apply on top.

Smoothing is always performed in **linear** current space (the physical signal);
the log axis, if any, is applied only afterwards for display.  Drain (Id) and
gate (Ig) currents are smoothed **independently** — each has its own on/off,
method and strength.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.signal import medfilt, savgol_filter
    from scipy.ndimage import gaussian_filter1d, uniform_filter1d
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


GATE_COLUMNS = {"ig", "absig"}

# Smoothing methods offered in the UI.
SMOOTH_METHODS = ["Savitzky–Golay", "Moving average", "Median", "Gaussian"]


def channel_of(column: str) -> str:
    """'g' for gate-current columns (Ig/absIg), else 'd' (drain & friends)."""
    return "g" if column.lower() in GATE_COLUMNS else "d"


@dataclass
class SmoothSpec:
    """Independent smoothing settings for one current channel."""

    enabled: bool = False
    method: str = "Savitzky–Golay"
    window: int = 7        # window length in points (forced odd where needed)
    polyorder: int = 2     # Savitzky–Golay polynomial order
    sigma: float = 2.0     # Gaussian kernel width (points)

    def strength_text(self) -> str:
        """Human-readable smoothing strength, shown next to the controls."""
        if self.method == "Gaussian":
            return f"strength: σ = {self.sigma:g} pts"
        return f"strength: window = {self.window} pts"


@dataclass
class NoiseFloorSpec:
    """Optional baseline noise added *after* smoothing, to restore a realistic
    noise floor on an over-smoothed off-state."""

    enabled: bool = False
    level: float = 1e-12   # RMS current of the added Gaussian noise floor (A)
    seed: int = 0          # fixed seed -> reproducible, no jitter on every redraw


@dataclass
class PreprocessConfig:
    """User-tunable preprocessing, driven by the Preprocess tab."""

    # X (sweep) axis transform: x_out = x * x_scale + x_offset.
    x_scale: float = 1.0
    x_offset: float = 0.0

    id_scale: float = 1.0
    ig_scale: float = 1.0
    id_smooth: SmoothSpec = field(default_factory=SmoothSpec)
    ig_smooth: SmoothSpec = field(default_factory=SmoothSpec)
    id_noise: NoiseFloorSpec = field(default_factory=NoiseFloorSpec)
    ig_noise: NoiseFloorSpec = field(default_factory=NoiseFloorSpec)

    def x_transform(self, v):
        """Apply the X scale+offset to a scalar/array (for annotation positions)."""
        return np.asarray(v, float) * self.x_scale + self.x_offset

    def scale_for(self, column: str) -> float:
        return self.ig_scale if channel_of(column) == "g" else self.id_scale

    def smooth_for(self, column: str) -> SmoothSpec:
        return self.ig_smooth if channel_of(column) == "g" else self.id_smooth

    def noise_for(self, column: str) -> NoiseFloorSpec:
        return self.ig_noise if channel_of(column) == "g" else self.id_noise


def _odd_window(win: int, n: int) -> int:
    win = int(win)
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    return max(win, 3)


def _smooth(y: np.ndarray, spec: SmoothSpec) -> np.ndarray:
    n = y.size
    if n < 5:
        return y
    method = spec.method
    if method == "Gaussian":
        if _HAVE_SCIPY:
            return gaussian_filter1d(y, max(spec.sigma, 1e-3), mode="nearest")
        method = "Moving average"  # fallback below

    win = _odd_window(spec.window, n)
    if win < 3:
        return y
    if method == "Savitzky–Golay":
        poly = min(spec.polyorder, win - 1)
        if _HAVE_SCIPY:
            return savgol_filter(y, win, poly)
        method = "Moving average"
    if method == "Median":
        if _HAVE_SCIPY:
            return medfilt(y, win)
        method = "Moving average"
    # Moving average (and the no-scipy fallback for every method).
    if _HAVE_SCIPY:
        return uniform_filter1d(y, win, mode="nearest")
    kernel = np.ones(win) / win
    return np.convolve(y, kernel, mode="same")


def _add_noise_floor(y: np.ndarray, spec: NoiseFloorSpec) -> np.ndarray:
    """Add reproducible Gaussian noise of RMS ``spec.level`` to the finite samples.

    The seed is mixed with a hash of the data so each curve gets a distinct but
    repeatable pattern (no flicker when the plot is redrawn)."""
    if not spec.enabled or spec.level <= 0:
        return y
    finite = np.isfinite(y)
    if not finite.any():
        return y
    h = hash(y[finite].tobytes()) & 0xFFFFFFFF
    rng = np.random.default_rng(((int(spec.seed) & 0xFFFFFFFF) ^ h) & 0xFFFFFFFF)
    out = y.copy()
    out[finite] = y[finite] + rng.normal(0.0, spec.level, size=int(finite.sum()))
    return out


def apply_series(cfg: PreprocessConfig, x: np.ndarray, y: np.ndarray,
                 column: str):
    """Return (x, y) after scaling, (linear) smoothing and an optional noise floor.

    Pipeline (all in linear current space): scale -> smooth -> add noise floor.
    The X (sweep) axis is independently scaled/offset.  The caller applies
    abs()/log afterwards for display.
    """
    x = np.asarray(x, float) * cfg.x_scale + cfg.x_offset
    y = np.asarray(y, float) * cfg.scale_for(column)
    spec = cfg.smooth_for(column)
    if spec.enabled:
        finite = np.isfinite(y)
        if finite.sum() >= 5:
            ys = y.copy()
            ys[finite] = _smooth(y[finite], spec)
            y = ys
    y = _add_noise_floor(y, cfg.noise_for(column))
    return x, y
