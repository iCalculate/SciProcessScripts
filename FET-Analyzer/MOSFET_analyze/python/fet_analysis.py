"""FET transfer-curve parameter extraction.

Python port of the core logic in ``../matlab/analyze_transfer_curves.m`` —
subthreshold swing, on/off ratio, peak transconductance, threshold and turn-on
voltages — packaged so the GUI can both tabulate the numbers and overlay the
matching annotations (the subthreshold tangent, the Ion/Ioff levels, ...).

The extraction is per-curve and defensive: on noisy or degenerate input it
returns a :class:`FetParams` with NaN fields rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_EPS = np.finfo(float).eps


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #

@dataclass
class FetParams:
    n: int = 0
    secondary_value: Optional[float] = None      # e.g. Vd for this curve

    # Subthreshold fit:  log10(|Id|) = a * Vg + b   over [vg_lo, vg_hi]
    slope_a: float = np.nan                       # decades / V
    intercept_b: float = np.nan
    ss_mV_dec: float = np.nan                     # subthreshold swing
    ss_r2: float = np.nan
    vg_lo: float = np.nan
    vg_hi: float = np.nan

    # Currents
    ion: float = np.nan
    ioff: float = np.nan                          # robust off level (median low 10%)
    ion_ioff_ratio: float = np.nan

    # Transconductance / thresholds
    gm_max: float = np.nan
    vth_gmmax: float = np.nan
    vth: float = np.nan                           # extrapolation to Ioff*10
    von: float = np.nan                           # linear-fit turn-on voltage

    def subthreshold_line(self, decades: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Return (Vg, |Id|) for the subthreshold tangent, for plotting.

        The line is the log-linear fit evaluated across the detected window,
        extended by ``decades`` of current on each side so the slope is visible.
        """
        if not np.isfinite(self.slope_a) or self.slope_a == 0:
            return np.array([]), np.array([])
        # Vg span that covers the window plus +-`decades` of current.
        dv = decades / abs(self.slope_a)
        v0 = self.vg_lo - dv
        v1 = self.vg_hi + dv
        vg = np.linspace(v0, v1, 50)
        idd = 10.0 ** (self.slope_a * vg + self.intercept_b)
        return vg, idd


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _movmean(y: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average that ignores NaNs (mirrors MATLAB movmean)."""
    win = max(1, int(round(win)))
    if win <= 1:
        return y
    y = np.asarray(y, float)
    out = np.full(y.shape, np.nan)
    half = win // 2
    for i in range(y.size):
        seg = y[max(0, i - half): i + half + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = seg.mean()
    return out


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ybar = np.mean(y)
    ss_tot = np.sum((y - ybar) ** 2)
    ss_res = np.sum((y - yhat) ** 2)
    return float(1.0 - ss_res / max(ss_tot, _EPS))


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def analyze_transfer_curve(
    vg, idd, *,
    secondary_value: Optional[float] = None,
    smoothing_pts: int = 5,
    ioff_frac: float = 3.0,
    peak_frac: float = 0.8,
    min_win_pts: int = 6,
) -> FetParams:
    """Extract FET parameters from one Id-Vg sweep (see module docstring)."""
    vg = np.asarray(vg, float)
    idd = np.asarray(idd, float)
    m = np.isfinite(vg) & np.isfinite(idd)
    vg, idd = vg[m], idd[m]
    if vg.size < max(4, min_win_pts):
        return FetParams(n=vg.size, secondary_value=secondary_value)

    # Sort by Vg ascending so gradients are well defined.
    order = np.argsort(vg)
    vg, idd = vg[order], idd[order]
    abs_id = np.abs(idd)

    res = FetParams(n=vg.size, secondary_value=secondary_value)

    valid = abs_id > 0
    if valid.sum() < 4:
        return res

    # --- Ioff estimate: median of the lowest 10 % of |Id| --------------- #
    n_low = max(5, int(round(0.10 * abs_id.size)))
    ioff_est = float(np.nanmedian(np.sort(abs_id)[:n_low]))
    res.ioff = ioff_est

    # --- d(log10|Id|)/dVg, smoothed, for window detection --------------- #
    safe = np.where(abs_id > 0, abs_id, np.nan)
    log_id = np.log10(safe)
    log_for_slope = _movmean(log_id, smoothing_pts)
    log_for_slope = _fill_nearest(log_for_slope)
    d_vg = np.gradient(vg)
    dlog = np.gradient(log_for_slope) / np.maximum(d_vg, _EPS)
    dlog = _movmean(dlog, smoothing_pts)

    # First index that has clearly left the off state.
    above = np.where(abs_id >= ioff_est * ioff_frac)[0]
    leave = int(above[0]) if above.size else 0

    seg = dlog[leave:]
    if not np.any(np.isfinite(seg)):
        return res
    peak_idx = int(np.nanargmax(seg)) + leave
    peak_slope = float(dlog[peak_idx])
    if not np.isfinite(peak_slope):
        return res

    abs_thr = 0.5 * max(peak_slope, _EPS) * (1.0 - peak_frac)
    thr = max(peak_frac * peak_slope, abs_thr)
    lo = hi = peak_idx
    while lo > 0 and dlog[lo - 1] >= thr:
        lo -= 1
    while hi < vg.size - 1 and dlog[hi + 1] >= thr:
        hi += 1
    if (hi - lo + 1) < min_win_pts:
        pad = int(np.ceil((min_win_pts - (hi - lo + 1)) / 2))
        lo = max(0, lo - pad)
        hi = min(vg.size - 1, hi + pad)

    # --- Subthreshold linear fit: log10(|Id|) = a*Vg + b ---------------- #
    vx = vg[lo:hi + 1]
    vy = log_id[lo:hi + 1]
    good = np.isfinite(vx) & np.isfinite(vy)
    vx, vy = vx[good], vy[good]
    if vx.size < 3:
        return res
    a, b = np.polyfit(vx, vy, 1)
    res.slope_a, res.intercept_b = float(a), float(b)
    res.vg_lo, res.vg_hi = float(vg[lo]), float(vg[hi])
    res.ss_r2 = _r2(vy, a * vx + b)
    if a > 0:
        res.ss_mV_dec = float((1.0 / max(a, _EPS)) * 1e3)

    # --- Ion / Ioff ----------------------------------------------------- #
    res.ion = float(np.max(abs_id[valid]))
    if ioff_est > 0:
        res.ion_ioff_ratio = float(res.ion / ioff_est)

    # --- Peak transconductance gm = dId/dVg ----------------------------- #
    gm_win = max(3, 2 * (smoothing_pts // 2) + 1)
    gm = np.gradient(_fill_nearest(_movmean(idd, gm_win))) / np.maximum(d_vg, _EPS)
    if np.any(np.isfinite(gm)):
        gi = int(np.nanargmax(gm))
        res.gm_max = float(gm[gi])
        res.vth_gmmax = float(vg[gi])

    # --- Vth by subthreshold extrapolation to Ioff*10 ------------------- #
    if np.isfinite(a) and a > 0 and ioff_est > 0:
        res.vth = float((np.log10(ioff_est * 10.0) - b) / a)

    # --- Von from linear fit of the high-current band (top 20 %) -------- #
    res.von = _von_linear(vg, idd)
    return res


def _fill_nearest(y: np.ndarray) -> np.ndarray:
    """Replace NaNs by nearest finite value (simple forward/backward fill)."""
    y = np.asarray(y, float).copy()
    idx = np.where(np.isfinite(y))[0]
    if idx.size == 0:
        return np.zeros_like(y)
    y[: idx[0]] = y[idx[0]]
    y[idx[-1] + 1:] = y[idx[-1]]
    for i in range(1, y.size):
        if not np.isfinite(y[i]):
            y[i] = y[i - 1]
    return y


def _von_linear(vg: np.ndarray, idd: np.ndarray, pct: float = 80.0,
                min_pts: int = 5, min_dvg: float = 0.02) -> float:
    a_id = np.abs(idd)
    finite = np.isfinite(a_id) & np.isfinite(vg)
    if finite.sum() < min_pts:
        return np.nan
    thr = np.percentile(a_id[finite], pct)
    sel = finite & (a_id >= thr)
    if sel.sum() < min_pts:
        return np.nan
    vs, is_ = vg[sel], idd[sel]
    if (vs.max() - vs.min()) < min_dvg:
        return np.nan
    a_lin, b_lin = np.polyfit(vs, is_, 1)
    if not np.isfinite(a_lin) or abs(a_lin) < 1e-12:
        return np.nan
    return float(-b_lin / a_lin)


# Catalogue of reportable parameters: key -> (label, formatter).
PARAM_TABLE = [
    ("ss_mV_dec", "SS (mV/dec)", lambda v: f"{v:.1f}"),
    ("ion_ioff_ratio", "Ion/Ioff", lambda v: f"{v:.2e}"),
    ("ion", "Ion (A)", lambda v: f"{v:.2e}"),
    ("ioff", "Ioff (A)", lambda v: f"{v:.2e}"),
    ("vth", "Vth (V)", lambda v: f"{v:.2f}"),
    ("von", "Von (V)", lambda v: f"{v:.2f}"),
    ("gm_max", "gm,max (S)", lambda v: f"{v:.2e}"),
    ("ss_r2", "SS fit R²", lambda v: f"{v:.3f}"),
]
