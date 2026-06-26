from __future__ import annotations

import numpy as np

from .schemas import ExtractedFeatures


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _moving_average(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size < 3 or window <= 1:
        return values.copy()
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.copy()
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _deduplicate_voltage(voltage: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse duplicate voltage points to a median current.

    Duplicate setpoints are common at sweep turning points. Numerical gradients
    require strictly increasing coordinates, so preserving all duplicates here
    creates divide-by-zero warnings and invalid gm values.
    """

    unique, inverse = np.unique(voltage, return_inverse=True)
    if unique.size == voltage.size:
        return voltage, current
    aggregated = np.empty(unique.size, dtype=float)
    for index in range(unique.size):
        aggregated[index] = float(np.median(current[inverse == index]))
    return unique, aggregated


def infer_polarity(voltage: np.ndarray, abs_current: np.ndarray) -> str:
    if voltage.size >= 12:
        edge_count = min(max(3, int(round(voltage.size * 0.08))), voltage.size // 3)
        left_edge = float(np.median(abs_current[:edge_count]))
        right_edge = float(np.median(abs_current[-edge_count:]))
        peak = float(np.max(abs_current))
        valley_idx = int(np.nanargmin(abs_current))
        valley_inside = edge_count <= valley_idx < voltage.size - edge_count
        if valley_inside and min(left_edge, right_edge) / max(peak, np.finfo(float).tiny) >= 0.08:
            return "bipolar"
    high_idx = int(np.nanargmax(abs_current))
    low_idx = int(np.nanargmin(abs_current))
    if np.isclose(voltage[high_idx], voltage[low_idx]):
        return "unknown"
    return "n-type" if voltage[high_idx] > voltage[low_idx] else "p-type"


def analyze_transfer_curve(
    voltage: list[float] | np.ndarray,
    current: list[float] | np.ndarray,
    *,
    polarity: str | None = None,
    min_window_points: int = 6,
) -> ExtractedFeatures:
    """Extract robust scalar features from one transfer sweep.

    The implementation follows the documented analyzer method: median low-tail
    Ioff, a contiguous peak log-slope window, log-linear SS/Vth fitting,
    numerical gm, and a high-current linear Von estimate.
    """

    vg = np.asarray(voltage, dtype=float)
    ids = np.asarray(current, dtype=float)
    valid = np.isfinite(vg) & np.isfinite(ids) & (np.abs(ids) > 0)
    vg, ids = vg[valid], ids[valid]
    if vg.size < max(6, min_window_points):
        return ExtractedFeatures(ion=0.0, ioff=0.0, ion_ioff_ratio=0.0)

    abs_ids = np.abs(ids)
    inferred_polarity = polarity
    if inferred_polarity is None:
        inferred_polarity = infer_polarity(vg, abs_ids)
    sign = -1.0 if inferred_polarity == "p-type" else 1.0

    u = sign * vg
    order = np.argsort(u)
    u, vg, ids, abs_ids = u[order], vg[order], ids[order], abs_ids[order]
    u, ids = _deduplicate_voltage(u, ids)
    vg = sign * u
    abs_ids = np.abs(ids)
    if u.size < max(6, min_window_points):
        return ExtractedFeatures(
            ion=float(np.max(abs_ids)),
            ioff=float(np.min(abs_ids)),
            ion_ioff_ratio=float(np.max(abs_ids) / max(np.min(abs_ids), np.finfo(float).tiny)),
            polarity=(
                inferred_polarity
                if inferred_polarity in {"n-type", "p-type", "bipolar"}
                else "unknown"
            ),
        )

    n_low = min(abs_ids.size, max(5, int(round(0.10 * abs_ids.size))))
    ioff = float(np.median(np.sort(abs_ids)[:n_low]))
    ion = float(np.max(abs_ids))
    ratio = ion / ioff if ioff > 0 else 0.0
    edge_count = min(max(3, int(round(abs_ids.size * 0.08))), abs_ids.size)
    low_edge = float(np.median(abs_ids[:edge_count]))
    high_edge = float(np.median(abs_ids[-edge_count:]))
    ambipolar_strength = min(low_edge, high_edge) / max(ion, np.finfo(float).tiny)

    log_ids = np.log10(np.clip(abs_ids, np.finfo(float).tiny, None))
    smooth_log = _moving_average(log_ids, 5)
    noise_log_sigma = float(
        1.4826 * np.median(np.abs((log_ids - smooth_log) - np.median(log_ids - smooth_log)))
    )
    slope = np.gradient(smooth_log, u)
    slope = _moving_average(slope, 5)

    leave_candidates = np.flatnonzero(abs_ids >= ioff * 3.0)
    leave = int(leave_candidates[0]) if leave_candidates.size else 0
    slope_tail = slope[leave:]
    peak_idx = int(np.nanargmax(slope_tail)) + leave
    peak = max(float(slope[peak_idx]), np.finfo(float).eps)
    threshold = 0.8 * peak
    lo = hi = peak_idx
    while lo > 0 and slope[lo - 1] >= threshold:
        lo -= 1
    while hi < u.size - 1 and slope[hi + 1] >= threshold:
        hi += 1
    while hi - lo + 1 < min_window_points and (lo > 0 or hi < u.size - 1):
        lo = max(0, lo - 1)
        hi = min(u.size - 1, hi + 1)

    fit_u = u[lo : hi + 1]
    fit_y = log_ids[lo : hi + 1]
    fit_slope = np.nan
    fit_intercept = np.nan
    fit_r2 = np.nan
    ss = np.nan
    vth = np.nan
    if fit_u.size >= 3 and np.ptp(fit_u) > 0:
        fit_slope, fit_intercept = np.polyfit(fit_u, fit_y, 1)
        predicted = fit_slope * fit_u + fit_intercept
        ss_total = float(np.sum((fit_y - np.mean(fit_y)) ** 2))
        ss_res = float(np.sum((fit_y - predicted) ** 2))
        fit_r2 = 1.0 - ss_res / max(ss_total, np.finfo(float).eps)
        if fit_slope > 0:
            ss = 1000.0 / fit_slope
            vth_u = (np.log10(ioff * 10.0) - fit_intercept) / fit_slope
            vth = sign * vth_u

    smooth_linear = _moving_average(ids, 5)
    gm = np.gradient(smooth_linear, vg)
    gm_effective = sign * gm
    gm_idx = int(np.nanargmax(gm_effective))
    gm_max = float(gm_effective[gm_idx])
    vth_gmmax = float(vg[gm_idx])

    threshold_current = np.percentile(abs_ids, 80)
    high = abs_ids >= threshold_current
    von = np.nan
    if int(high.sum()) >= 5 and np.ptp(vg[high]) > 0.02:
        linear_slope, linear_intercept = np.polyfit(vg[high], ids[high], 1)
        if abs(linear_slope) >= 1e-12:
            von = -linear_intercept / linear_slope

    return ExtractedFeatures(
        ion=ion,
        ioff=ioff,
        ion_ioff_ratio=ratio,
        polarity=(
            inferred_polarity
            if inferred_polarity in {"n-type", "p-type", "bipolar"}
            else "unknown"
        ),
        vth=_finite_or_none(vth),
        ss_mv_dec=_finite_or_none(ss),
        ss_fit_r2=_finite_or_none(fit_r2),
        gm_max=_finite_or_none(gm_max),
        vth_gmmax=_finite_or_none(vth_gmmax),
        von=_finite_or_none(von),
        leakage_level=ioff,
        noise_log_sigma=noise_log_sigma,
        ambipolar_strength=ambipolar_strength,
        current_floor=ioff,
    )


def combine_sweep_features(
    forward: ExtractedFeatures,
    reverse: ExtractedFeatures,
) -> ExtractedFeatures:
    values = forward.model_dump()
    if forward.vth is not None and reverse.vth is not None:
        values["vth"] = 0.5 * (forward.vth + reverse.vth)
        values["hysteresis_v"] = abs(reverse.vth - forward.vth)
    if forward.ss_mv_dec is not None and reverse.ss_mv_dec is not None:
        values["ss_mv_dec"] = 0.5 * (forward.ss_mv_dec + reverse.ss_mv_dec)
    if forward.ss_fit_r2 is not None and reverse.ss_fit_r2 is not None:
        values["ss_fit_r2"] = 0.5 * (forward.ss_fit_r2 + reverse.ss_fit_r2)
    return ExtractedFeatures(**values)
