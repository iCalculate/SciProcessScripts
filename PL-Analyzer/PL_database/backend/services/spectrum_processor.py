from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter


def _as_float_array(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Spectrum arrays must be one-dimensional")
    return array


def baseline_correction(
    x_axis: list[float] | np.ndarray,
    intensity: list[float] | np.ndarray,
    *,
    order: int = 3,
    quantile: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = _as_float_array(x_axis)
    y_values = _as_float_array(intensity)
    if len(x_values) != len(y_values):
        raise ValueError("x_axis and intensity must have the same length")
    if len(x_values) <= order + 2:
        baseline = np.full_like(y_values, np.median(y_values))
        return baseline, y_values - baseline

    threshold = np.quantile(y_values, quantile)
    mask = y_values <= threshold
    if mask.sum() <= order + 1:
        mask = np.argsort(y_values)[: max(order + 2, len(y_values) // 4)]
        selected_x = x_values[mask]
        selected_y = y_values[mask]
    else:
        selected_x = x_values[mask]
        selected_y = y_values[mask]

    degree = min(order, max(1, len(selected_x) - 1))
    coefficients = np.polyfit(selected_x, selected_y, deg=degree)
    baseline = np.polyval(coefficients, x_values)
    corrected = y_values - baseline
    return baseline, corrected


def smooth_spectrum(
    intensity: list[float] | np.ndarray,
    *,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    y_values = _as_float_array(intensity)
    if len(y_values) < 5:
        return y_values.copy()
    window = min(window_length, len(y_values) if len(y_values) % 2 == 1 else len(y_values) - 1)
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    poly = min(polyorder, window - 1)
    return savgol_filter(y_values, window_length=window, polyorder=poly, mode="interp")


def normalize_spectrum(intensity: list[float] | np.ndarray, method: str = "max") -> np.ndarray:
    y_values = _as_float_array(intensity)
    if method == "none":
        return y_values.copy()
    if method == "max":
        scale = float(np.max(np.abs(y_values))) or 1.0
        return y_values / scale
    if method == "area":
        scale = float(np.trapezoid(np.abs(y_values))) or 1.0
        return y_values / scale
    if method == "zscore":
        std = float(np.std(y_values)) or 1.0
        return (y_values - float(np.mean(y_values))) / std
    raise ValueError(f"Unsupported normalization method: {method}")


def detect_peaks(
    x_axis: list[float] | np.ndarray,
    intensity: list[float] | np.ndarray,
    *,
    prominence: float = 0.05,
    height: float | None = None,
    distance: int | None = None,
) -> list[dict[str, float]]:
    x_values = _as_float_array(x_axis)
    y_values = _as_float_array(intensity)
    peaks, properties = find_peaks(
        y_values,
        prominence=prominence,
        height=height,
        distance=distance,
    )
    results: list[dict[str, float]] = []
    widths = properties.get("widths")
    for index, peak in enumerate(peaks):
        results.append(
            {
                "index": int(peak),
                "position": float(x_values[peak]),
                "height": float(y_values[peak]),
                "prominence": float(properties["prominences"][index]),
                "width_points": float(widths[index]) if widths is not None else math.nan,
            }
        )
    return results


def _gaussian_sum(x_values: np.ndarray, *params: float) -> np.ndarray:
    total = np.zeros_like(x_values)
    for start in range(0, len(params), 3):
        amplitude, center, sigma = params[start : start + 3]
        total += amplitude * np.exp(-0.5 * ((x_values - center) / sigma) ** 2)
    return total


def _lorentzian_sum(x_values: np.ndarray, *params: float) -> np.ndarray:
    total = np.zeros_like(x_values)
    for start in range(0, len(params), 3):
        amplitude, center, gamma = params[start : start + 3]
        total += amplitude / (1.0 + ((x_values - center) / gamma) ** 2)
    return total


def fit_peaks(
    x_axis: list[float] | np.ndarray,
    intensity: list[float] | np.ndarray,
    peaks: list[dict[str, float]],
    *,
    model: str = "gaussian",
    max_peaks: int = 4,
) -> dict[str, object]:
    x_values = _as_float_array(x_axis)
    y_values = _as_float_array(intensity)
    selected = sorted(peaks, key=lambda item: item["height"], reverse=True)[:max_peaks]
    if not selected:
        return {"model": model, "fit_curve": [], "r2": None, "peaks": []}

    fit_fn: Callable[..., np.ndarray]
    if model == "gaussian":
        fit_fn = _gaussian_sum
    elif model == "lorentzian":
        fit_fn = _lorentzian_sum
    else:
        raise ValueError(f"Unsupported fit model: {model}")

    span = float(x_values.max() - x_values.min()) or 1.0
    sigma_guess = span / max(len(x_values), 20) * 6
    initial_params: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for peak in selected:
        initial_params.extend([peak["height"], peak["position"], sigma_guess])
        lower_bounds.extend([0.0, float(x_values.min()), span / 1_000_000])
        upper_bounds.extend([float(y_values.max()) * 5 or 1.0, float(x_values.max()), span])

    try:
        parameters, _ = curve_fit(
            fit_fn,
            x_values,
            y_values,
            p0=initial_params,
            bounds=(lower_bounds, upper_bounds),
            maxfev=20_000,
        )
    except Exception:
        return {"model": model, "fit_curve": [], "r2": None, "peaks": []}

    fitted = fit_fn(x_values, *parameters)
    residual = y_values - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2)) or 1.0
    r2 = 1.0 - ss_res / ss_tot

    fit_peaks_payload: list[dict[str, float]] = []
    for index in range(0, len(parameters), 3):
        amplitude = float(parameters[index])
        center = float(parameters[index + 1])
        width = float(parameters[index + 2])
        if model == "gaussian":
            fwhm = 2.0 * math.sqrt(2.0 * math.log(2.0)) * width
            area = amplitude * width * math.sqrt(2.0 * math.pi)
        else:
            fwhm = 2.0 * width
            area = amplitude * math.pi * width
        fit_peaks_payload.append(
            {
                "amplitude": amplitude,
                "center": center,
                "width": width,
                "fwhm": fwhm,
                "area": area,
            }
        )

    return {
        "model": model,
        "fit_curve": fitted.tolist(),
        "r2": r2,
        "peaks": fit_peaks_payload,
    }


def analyze_spectrum(
    x_axis: list[float] | np.ndarray,
    intensity: list[float] | np.ndarray,
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    config = options or {}
    baseline, corrected = baseline_correction(
        x_axis,
        intensity,
        order=int(config.get("baseline_order", 3)),
        quantile=float(config.get("baseline_quantile", 0.25)),
    )
    smoothed = smooth_spectrum(
        corrected,
        window_length=int(config.get("smoothing_window", 11)),
        polyorder=int(config.get("smoothing_polyorder", 3)),
    )
    normalized = normalize_spectrum(smoothed, method=str(config.get("normalization", "max")))
    peaks = detect_peaks(
        x_axis,
        normalized,
        prominence=float(config.get("prominence", 0.05)),
        height=(float(config["height"]) if config.get("height") is not None else None),
        distance=(int(config["distance"]) if config.get("distance") is not None else None),
    )
    fit_result = fit_peaks(
        x_axis,
        normalized,
        peaks,
        model=str(config.get("fit_model", "gaussian")),
        max_peaks=int(config.get("max_peaks", 4)),
    )

    baseline_noise = float(np.std(corrected[: max(10, len(corrected) // 8)])) or 1.0
    peak_max = float(np.max(normalized)) if len(normalized) else 0.0
    integrated = float(np.trapezoid(np.maximum(normalized, 0.0), x=_as_float_array(x_axis)))

    return {
        "x_axis": _as_float_array(x_axis).tolist(),
        "raw_intensity": _as_float_array(intensity).tolist(),
        "baseline": baseline.tolist(),
        "corrected_intensity": corrected.tolist(),
        "smoothed_intensity": smoothed.tolist(),
        "normalized_intensity": normalized.tolist(),
        "peaks": peaks,
        "fit": fit_result,
        "metrics": {
            "integrated_intensity": integrated,
            "peak_max": peak_max,
            "signal_to_noise_ratio": peak_max / baseline_noise,
            "n_detected_peaks": len(peaks),
        },
    }
