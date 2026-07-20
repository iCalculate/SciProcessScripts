from __future__ import annotations

import math
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

from .metadata_parser import infer_raman_excitation_from_nm_axis
from .spectrum_processor import baseline_correction, normalize_spectrum, smooth_spectrum


ANALYSIS_METHOD_VERSION = "material-aware-v2"
HC_EV_NM = 1239.841984
COMMON_LASER_LINES_NM = [325, 355, 405, 442, 457, 473, 488, 514, 532, 561, 594, 633, 660, 785, 830, 1064]


@dataclass(frozen=True)
class PeakTemplate:
    label: str
    center: float
    tolerance: float
    width_bounds: tuple[float, float]


@dataclass(frozen=True)
class MaterialTemplate:
    material: str
    family: Literal["Raman", "PL"]
    peaks: tuple[PeakTemplate, ...]


RAMAN_TEMPLATES: tuple[MaterialTemplate, ...] = (
    MaterialTemplate("MoS2", "Raman", (PeakTemplate("E", 383.0, 12.0, (1.0, 18.0)), PeakTemplate("G", 408.0, 12.0, (1.0, 18.0)))),
    MaterialTemplate("WSe2", "Raman", (PeakTemplate("E", 250.0, 14.0, (1.0, 20.0)), PeakTemplate("G", 260.0, 14.0, (1.0, 20.0)))),
    MaterialTemplate("WS2", "Raman", (PeakTemplate("E", 356.0, 14.0, (1.0, 22.0)), PeakTemplate("G", 419.0, 14.0, (1.0, 22.0)))),
    MaterialTemplate("MoSe2", "Raman", (PeakTemplate("E", 287.0, 14.0, (1.0, 22.0)), PeakTemplate("G", 240.0, 14.0, (1.0, 22.0)))),
    MaterialTemplate("MoTe2", "Raman", (PeakTemplate("E", 235.0, 16.0, (1.0, 28.0)), PeakTemplate("G", 171.0, 16.0, (1.0, 28.0)))),
    MaterialTemplate("WTe2", "Raman", (PeakTemplate("E", 212.0, 18.0, (1.0, 30.0)), PeakTemplate("G", 164.0, 18.0, (1.0, 30.0)))),
)

PL_TEMPLATES: tuple[MaterialTemplate, ...] = (
    MaterialTemplate("MoS2", "PL", (PeakTemplate("trion", 1.80, 0.045, (0.008, 0.12)), PeakTemplate("A", 1.84, 0.055, (0.008, 0.14)), PeakTemplate("B", 2.00, 0.075, (0.01, 0.18)))),
    MaterialTemplate("WSe2", "PL", (PeakTemplate("trion", 1.62, 0.045, (0.008, 0.12)), PeakTemplate("A", 1.66, 0.055, (0.008, 0.14)), PeakTemplate("B", 2.08, 0.09, (0.01, 0.20)))),
    MaterialTemplate("WS2", "PL", (PeakTemplate("trion", 1.98, 0.05, (0.008, 0.13)), PeakTemplate("A", 2.02, 0.055, (0.008, 0.14)), PeakTemplate("B", 2.39, 0.10, (0.01, 0.22)))),
    MaterialTemplate("MoSe2", "PL", (PeakTemplate("trion", 1.52, 0.045, (0.008, 0.12)), PeakTemplate("A", 1.56, 0.055, (0.008, 0.14)), PeakTemplate("B", 1.74, 0.08, (0.01, 0.18)))),
    MaterialTemplate("MoTe2", "PL", (PeakTemplate("trion", 1.07, 0.06, (0.01, 0.16)), PeakTemplate("A", 1.10, 0.07, (0.01, 0.18)), PeakTemplate("B", 1.29, 0.10, (0.01, 0.22)))),
)


def analyze_material_spectrum(
    x_axis: list[float] | np.ndarray,
    intensity: list[float] | np.ndarray,
    metadata: dict[str, object] | None = None,
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    config = options or {}
    meta = metadata or {}
    x_values = np.asarray(x_axis, dtype=float)
    y_values = np.asarray(intensity, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1 or len(x_values) != len(y_values):
        raise ValueError("Spectrum arrays must be one-dimensional and equally sized")
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if len(x_values) == 0:
        raise ValueError("Spectrum contains no finite data points")
    family = _resolve_family(x_values, meta, str(config.get("spectrum_family") or "auto"))
    analysis_x, axis_unit, axis_note = _convert_axis(x_values, family, meta)
    finite_axis = np.isfinite(analysis_x) & np.isfinite(y_values)
    analysis_x = analysis_x[finite_axis]
    y_values = y_values[finite_axis]
    if len(analysis_x) < 5:
        raise ValueError("Spectrum contains too few finite points after axis conversion")
    if float(np.max(analysis_x) - np.min(analysis_x)) <= 0 or len(np.unique(analysis_x)) < 3:
        raise ValueError("Spectrum analysis axis is degenerate after conversion")
    order = np.argsort(analysis_x)
    analysis_x = analysis_x[order]
    y_values = y_values[order]

    baseline, corrected = baseline_correction(
        analysis_x,
        y_values,
        order=int(config.get("baseline_order", 3)),
        quantile=float(config.get("baseline_quantile", 0.25)),
    )
    smoothed = smooth_spectrum(
        corrected,
        window_length=int(config.get("smoothing_window", 11)),
        polyorder=int(config.get("smoothing_polyorder", 3)),
    )
    normalization_method = str(config.get("normalization", "max"))
    normalized = normalize_spectrum(smoothed, method=normalization_method)
    noise_reference = normalize_spectrum(corrected, method=normalization_method)
    detected = _detect_candidate_peaks(
        analysis_x,
        normalized,
        prominence=float(config.get("prominence", 0.035 if family == "PL" else 0.025)),
        distance=int(config.get("distance") or max(3, len(analysis_x) // 90)),
    )

    templates = PL_TEMPLATES if family == "PL" else RAMAN_TEMPLATES
    hint = _normalize_material_name(config.get("material_hint") or meta.get("material") or meta.get("source"))
    ranked = _rank_materials(templates, detected, hint)
    best_template = ranked[0]["template"] if ranked else templates[0]
    requested_profile = str(config.get("fit_model") or "auto")
    fit_profile = ("lorentzian" if family == "Raman" else "pseudo_voigt") if requested_profile == "auto" else requested_profile
    fit = _fit_template_peaks(
        analysis_x,
        normalized,
        best_template,
        fit_profile,
        noise_values=noise_reference,
        detected_peaks=detected,
    )
    features = _build_features(family, fit["peaks"])
    evidence_confidence = float(ranked[0]["confidence"] if ranked else 0.0)
    fit_evidence = float(fit.get("evidence_score") or 0.0)
    material_confidence = min(1.0, 0.72 * evidence_confidence + 0.28 * fit_evidence)
    fit["material_evidence"] = evidence_confidence
    fit["matched_template_peaks"] = int(ranked[0]["matched_peaks"] if ranked else 0)
    fit_quality = _fit_quality(fit)
    material = best_template.material if ranked and material_confidence >= float(config.get("min_material_confidence", 0.28)) else "unknown"

    return {
        "method_version": str(config.get("method_version") or ANALYSIS_METHOD_VERSION),
        "spectrum_family": family,
        "material": material,
        "material_confidence": material_confidence,
        "material_candidates": [
            {"material": item["template"].material, "confidence": item["confidence"], "matched_peaks": item["matched_peaks"]}
            for item in ranked[:5]
        ],
        "axis": {"unit": axis_unit, "note": axis_note},
        "x_axis": analysis_x.tolist(),
        "raw_intensity": y_values.tolist(),
        "baseline": baseline.tolist(),
        "corrected_intensity": corrected.tolist(),
        "smoothed_intensity": smoothed.tolist(),
        "normalized_intensity": normalized.tolist(),
        "peaks": detected,
        "fit": fit,
        "features": features,
        "metrics": {
            **features,
            "fit_quality": fit_quality["score"],
            "n_detected_peaks": len(detected),
            "r2": fit["r2"],
            "fit_rmse": fit.get("rmse"),
            "fit_snr": fit.get("signal_to_noise"),
            "fit_window": fit.get("fit_window"),
            "peak_max": float(np.max(normalized)) if len(normalized) else 0.0,
            "integrated_intensity": float(np.trapezoid(np.maximum(normalized, 0.0), x=analysis_x)),
        },
        "quality": fit_quality,
    }


def _as_float_array(values: list[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Spectrum arrays must be one-dimensional")
    finite = np.isfinite(array)
    return array[finite]


def _resolve_family(x_values: np.ndarray, metadata: dict[str, object], requested: str) -> Literal["Raman", "PL"]:
    if requested in {"Raman", "PL"}:
        return requested  # type: ignore[return-value]
    text = f"{metadata.get('spectrum_type') or ''} {metadata.get('grating') or ''}".lower()
    if "raman" in text or re.search(r"\bg3\b", text):
        return "Raman"
    if "pl" in text or "photoluminescence" in text or re.search(r"\bg[12]\b", text):
        return "PL"
    unit = str(metadata.get("x_axis_unit") or "").lower()
    if any(token in unit for token in ("cm^-1", "cm-1", "1/cm", "raman")):
        return "Raman"
    if any(token in unit for token in ("ev", "mev")):
        return "PL"
    if any(token in unit for token in ("nm", "nanometer", "nanometre")):
        return "Raman" if infer_raman_excitation_from_nm_axis(x_values.tolist()) is not None else "PL"
    return "PL" if x_values.size and 0.6 <= float(np.nanmedian(x_values)) <= 3.2 else "Raman"


def _convert_axis(x_values: np.ndarray, family: str, metadata: dict[str, object]) -> tuple[np.ndarray, str, str | None]:
    unit = str(metadata.get("x_axis_unit") or "").strip().lower()
    if family == "Raman":
        if any(token in unit for token in ("cm^-1", "cm-1", "1/cm", "raman")):
            return x_values.copy(), "cm^-1", None
        laser = _parse_laser_wavelength(metadata.get("laser_wavelength")) or infer_raman_excitation_from_nm_axis(x_values.tolist())
        if laser:
            with np.errstate(divide="ignore", invalid="ignore"):
                return (1e7 / float(laser)) - (1e7 / x_values), "cm^-1", f"converted from nm using {laser:g} nm excitation"
        return x_values.copy(), str(metadata.get("x_axis_unit") or "x_axis"), "Raman axis could not be converted without laser wavelength"

    if "mev" in unit:
        return x_values / 1000.0, "eV", "converted from meV"
    if any(token in unit for token in ("ev", "electron")):
        return x_values.copy(), "eV", None
    if any(token in unit for token in ("nm", "nanometer", "nanometre")) or (x_values.size and float(np.nanmedian(x_values)) > 10):
        with np.errstate(divide="ignore", invalid="ignore"):
            return HC_EV_NM / x_values, "eV", "converted from nm"
    return x_values.copy(), str(metadata.get("x_axis_unit") or "eV"), None


def _parse_laser_wavelength(value: object) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    if not match:
        return None
    parsed = float(match.group(1))
    if parsed in COMMON_LASER_LINES_NM or 250 <= parsed <= 1200:
        return parsed
    return None


def _detect_candidate_peaks(x_values: np.ndarray, y_values: np.ndarray, *, prominence: float, distance: int) -> list[dict[str, float]]:
    peak_indices, properties = find_peaks(y_values, prominence=prominence, distance=distance)
    if len(peak_indices) == 0:
        return []
    widths = peak_widths(y_values, peak_indices, rel_height=0.5)[0]
    step = float(np.median(np.abs(np.diff(x_values)))) if len(x_values) > 1 else 1.0
    peaks: list[dict[str, float]] = []
    for index, peak_index in enumerate(peak_indices):
        peaks.append(
            {
                "index": int(peak_index),
                "position": float(x_values[peak_index]),
                "height": float(y_values[peak_index]),
                "prominence": float(properties["prominences"][index]),
                "width_points": float(widths[index]),
                "fwhm_estimate": float(widths[index] * step),
            }
        )
    return sorted(peaks, key=lambda item: item["prominence"], reverse=True)


def _normalize_material_name(value: object) -> str | None:
    text = str(value or "")
    for material in ("MoS2", "WSe2", "WS2", "MoSe2", "MoTe2", "WTe2"):
        if material.lower() in text.lower().replace(" ", ""):
            return material
    return None


def _rank_materials(
    templates: tuple[MaterialTemplate, ...],
    detected: list[dict[str, float]],
    hint: str | None,
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for template in templates:
        available = list(detected)
        matched = 0
        score = 0.0
        total_weight = 0.0
        ordered_peaks = sorted(template.peaks, key=lambda peak: _classification_weight(peak.label), reverse=True)
        for expected in ordered_peaks:
            weight = _classification_weight(expected.label)
            total_weight += weight
            nearest = _nearest_peak(available, expected.center)
            if nearest is None:
                continue
            distance = abs(float(nearest["position"]) - expected.center)
            if distance <= expected.tolerance:
                matched += 1
                position_score = math.exp(-0.5 * (distance / max(expected.tolerance * 0.55, 1e-9)) ** 2)
                prominence = max(0.0, float(nearest["prominence"]))
                prominence_score = prominence / (prominence + (0.045 if template.family == "PL" else 0.035))
                score += weight * position_score * (0.45 + 0.55 * prominence_score)
                available.remove(nearest)
        if hint == template.material:
            score += 0.08 * total_weight
        confidence = min(1.0, score / max(total_weight, 1e-9))
        ranked.append({"template": template, "confidence": confidence, "matched_peaks": matched})
    return sorted(ranked, key=lambda item: (float(item["confidence"]), int(item["matched_peaks"])), reverse=True)


def _classification_weight(label: str) -> float:
    if label in {"E", "G"}:
        return 0.5
    if label == "A":
        return 0.55
    if label == "B":
        return 0.25
    return 0.20


def _nearest_peak(peaks: list[dict[str, float]], center: float) -> dict[str, float] | None:
    if not peaks:
        return None
    return min(peaks, key=lambda item: abs(float(item["position"]) - center))


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


def _fit_template_peaks(
    x_values: np.ndarray,
    y_values: np.ndarray,
    template: MaterialTemplate,
    profile: str,
    *,
    noise_values: np.ndarray | None = None,
    detected_peaks: list[dict[str, float]] | None = None,
) -> dict[str, object]:
    x_min = float(x_values.min())
    x_max = float(x_values.max())
    visible = [
        peak
        for peak in template.peaks
        if x_min <= peak.center + peak.tolerance * 0.65 and x_max >= peak.center - peak.tolerance * 0.65
    ]
    if not visible:
        return _empty_fit(profile)

    fit_low = max(x_min, min(peak.center - peak.tolerance * 1.25 for peak in visible))
    fit_high = min(x_max, max(peak.center + peak.tolerance * 1.25 for peak in visible))
    fit_mask = (x_values >= fit_low) & (x_values <= fit_high)
    if int(np.count_nonzero(fit_mask)) < max(12, len(visible) * 5):
        return _empty_fit(profile, fit_window=[fit_low, fit_high])

    fit_indices = np.flatnonzero(fit_mask)
    noise_source = noise_values[fit_mask] if noise_values is not None and len(noise_values) == len(y_values) else y_values[fit_mask]
    noise = _robust_noise(noise_source)
    if len(fit_indices) > 700:
        fit_indices = fit_indices[np.linspace(0, len(fit_indices) - 1, 700, dtype=int)]
    x_fit = x_values[fit_indices]
    y_fit = y_values[fit_indices]
    x_mid = float((fit_low + fit_high) / 2.0)
    x_span = max(float(fit_high - fit_low) / 2.0, 1e-9)
    signal_scale = max(float(np.percentile(y_fit, 95) - np.percentile(y_fit, 5)), noise * 3.0, 1e-6)
    subsets = _candidate_peak_subsets(template.family, visible, detected_peaks or [])
    candidates: list[dict[str, object]] = []
    for subset in subsets:
        candidate = _fit_peak_subset(
            x_fit,
            y_fit,
            subset,
            profile,
            x_mid=x_mid,
            x_span=x_span,
            noise=noise,
            signal_scale=signal_scale,
        )
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return _empty_fit(profile, fit_window=[fit_low, fit_high])

    chosen = min(candidates, key=lambda item: float(item["bic"]))
    chosen_peaks = chosen["templates"]
    params = np.asarray(chosen["params"], dtype=float)
    uncertainties = np.asarray(chosen["uncertainties"], dtype=float)
    full_curve = _constrained_model(x_values, params, chosen_peaks, profile, x_mid=x_mid, x_span=x_span)
    fitted_peaks: list[dict[str, object]] = []
    present_weights = 0.0
    total_weights = sum(_classification_weight(peak.label) for peak in visible)
    boundary_terms: list[float] = []
    for index, peak in enumerate(chosen_peaks):
        amplitude = float(params[index * 3])
        center = float(params[index * 3 + 1])
        fwhm = float(params[index * 3 + 2])
        amplitude_stderr = float(uncertainties[index * 3])
        center_stderr = float(uncertainties[index * 3 + 1])
        fwhm_stderr = float(uncertainties[index * 3 + 2])
        peak_snr = amplitude / max(noise, 1e-9)
        nearest = _nearest_peak(detected_peaks or [], center)
        detected_support = bool(
            nearest is not None
            and abs(float(nearest["position"]) - center) <= max(fwhm * 0.8, peak.tolerance * 0.45)
            and float(nearest["prominence"]) >= (0.018 if template.family == "PL" else 0.014)
        )
        center_edge = abs(center - peak.center) / max(peak.tolerance, 1e-9)
        width_edge = min(
            abs(fwhm - peak.width_bounds[0]),
            abs(peak.width_bounds[1] - fwhm),
        ) / max(peak.width_bounds[1] - peak.width_bounds[0], 1e-9)
        at_boundary = center_edge >= 0.97 or width_edge <= 0.015
        identifiable = (
            math.isfinite(center_stderr)
            and math.isfinite(fwhm_stderr)
            and center_stderr <= peak.tolerance * 0.45
            and fwhm_stderr <= max(fwhm * 0.75, peak.width_bounds[0])
        )
        present = amplitude >= max(0.012, noise * 2.5) and peak_snr >= 2.5 and detected_support
        if present:
            present_weights += _classification_weight(peak.label)
            boundary_terms.append(0.25 if at_boundary else (1.0 if identifiable else 0.55))
        area = _profile_area(amplitude, fwhm, profile)
        fitted_peaks.append(
            {
                "label": peak.label,
                "amplitude": amplitude,
                "center": center,
                "width": fwhm,
                "fwhm": fwhm,
                "area": area,
                "snr": peak_snr,
                "present": present,
                "detected_support": detected_support,
                "at_boundary": at_boundary,
                "identifiable": identifiable,
                "amplitude_stderr": amplitude_stderr if math.isfinite(amplitude_stderr) else None,
                "center_stderr": center_stderr if math.isfinite(center_stderr) else None,
                "fwhm_stderr": fwhm_stderr if math.isfinite(fwhm_stderr) else None,
            }
        )
    fitted_peaks = [peak for peak in fitted_peaks if bool(peak["present"])]
    signal_to_noise = max((float(peak["snr"]) for peak in fitted_peaks), default=0.0)
    presence_score = present_weights / max(total_weights, 1e-9)
    evidence_score = min(1.0, 0.68 * presence_score + 0.32 * (1.0 - math.exp(-signal_to_noise / 5.0)))
    boundary_score = float(np.mean(boundary_terms)) if boundary_terms else 0.0
    resolution_score = _peak_resolution_score(fitted_peaks)
    physical_score = 0.60 * boundary_score + 0.40 * resolution_score
    return {
        "model": profile,
        "fit_curve": full_curve.tolist(),
        "r2": chosen["r2"],
        "rmse": chosen["rmse"],
        "bic": chosen["bic"],
        "peaks": fitted_peaks,
        "fit_window": [fit_low, fit_high],
        "signal_to_noise": signal_to_noise,
        "evidence_score": evidence_score,
        "physical_score": physical_score,
        "resolution_score": resolution_score,
        "baseline": {"offset": float(params[-2]), "slope": float(params[-1])},
        "converged": bool(chosen["converged"]),
    }


def _candidate_peak_subsets(
    family: str,
    visible: list[PeakTemplate],
    detected: list[dict[str, float]],
) -> list[tuple[PeakTemplate, ...]]:
    if family != "PL":
        return [tuple(visible)]
    a_peak = next((peak for peak in visible if peak.label == "A"), None)
    if a_peak is None:
        return [tuple(visible)]
    optional = [
        peak
        for peak in visible
        if peak.label != "A"
        and (nearest := _nearest_peak(detected, peak.center)) is not None
        and abs(float(nearest["position"]) - peak.center) <= peak.tolerance
    ]
    subsets: list[tuple[PeakTemplate, ...]] = []
    for count in range(len(optional) + 1):
        for selected in combinations(optional, count):
            subsets.append(tuple(sorted((a_peak, *selected), key=lambda peak: peak.center)))
    return subsets


def _fit_peak_subset(
    x_values: np.ndarray,
    y_values: np.ndarray,
    peaks: tuple[PeakTemplate, ...],
    profile: str,
    *,
    x_mid: float,
    x_span: float,
    noise: float,
    signal_scale: float,
) -> dict[str, object] | None:
    initial: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    amplitude_upper = max(2.5, float(np.max(y_values) - np.min(y_values)) * 4.0)
    for peak in peaks:
        local_mask = (x_values >= peak.center - peak.tolerance) & (x_values <= peak.center + peak.tolerance)
        if np.any(local_mask):
            local_x = x_values[local_mask]
            local_y = y_values[local_mask]
            local_index = int(np.argmax(local_y))
            center_guess = float(local_x[local_index])
            height_guess = float(local_y[local_index] - np.percentile(y_values, 15))
        else:
            center_guess = peak.center
            height_guess = signal_scale * 0.15
        fwhm_guess = math.sqrt(peak.width_bounds[0] * peak.width_bounds[1])
        initial.extend([max(0.015, height_guess), center_guess, fwhm_guess])
        lower.extend([0.0, peak.center - peak.tolerance, peak.width_bounds[0]])
        upper.extend([amplitude_upper, peak.center + peak.tolerance, peak.width_bounds[1]])

    baseline_offset = float(np.percentile(y_values, 12))
    initial.extend([baseline_offset, 0.0])
    lower.extend([float(np.min(y_values) - signal_scale), -signal_scale * 2.0])
    upper.extend([float(np.max(y_values) + signal_scale), signal_scale * 2.0])

    def residual(parameters: np.ndarray) -> np.ndarray:
        model = _constrained_model(x_values, parameters, peaks, profile, x_mid=x_mid, x_span=x_span)
        penalties = _ordering_penalties(parameters, peaks, noise)
        return np.concatenate((model - y_values, penalties))

    try:
        result = least_squares(
            residual,
            np.asarray(initial, dtype=float),
            bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
            loss="soft_l1",
            f_scale=max(noise * 2.0, 0.012),
            max_nfev=4_000,
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return None
    if not _ordering_is_valid(result.x, peaks):
        return None
    fitted = _constrained_model(x_values, result.x, peaks, profile, x_mid=x_mid, x_span=x_span)
    raw_residual = y_values - fitted
    rss = max(float(np.sum(raw_residual**2)), 1e-15)
    ss_tot = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r2 = 1.0 - rss / ss_tot if ss_tot > 1e-15 else None
    rmse = math.sqrt(rss / len(y_values)) / max(signal_scale, 1e-9)
    parameter_count = len(result.x)
    bic = len(y_values) * math.log(rss / len(y_values)) + parameter_count * math.log(len(y_values))
    uncertainties = _parameter_uncertainties(result.jac[: len(y_values), :], rss, len(y_values), parameter_count)
    return {
        "params": result.x,
        "templates": peaks,
        "r2": r2,
        "rmse": rmse,
        "bic": bic,
        "converged": result.success,
        "uncertainties": uncertainties,
    }


def _constrained_model(
    x_values: np.ndarray,
    params: np.ndarray,
    peaks: tuple[PeakTemplate, ...],
    profile: str,
    *,
    x_mid: float,
    x_span: float,
) -> np.ndarray:
    x_scaled = np.clip((x_values - x_mid) / x_span, -1.0, 1.0)
    total = params[-2] + params[-1] * x_scaled
    for index, _peak in enumerate(peaks):
        amplitude, center, fwhm = params[index * 3 : index * 3 + 3]
        total = total + _unit_height_profile(x_values, amplitude, center, fwhm, profile)
    return total


def _unit_height_profile(x_values: np.ndarray, amplitude: float, center: float, fwhm: float, profile: str) -> np.ndarray:
    safe_fwhm = max(float(fwhm), 1e-12)
    gaussian = amplitude * np.exp(-4.0 * math.log(2.0) * ((x_values - center) / safe_fwhm) ** 2)
    if profile == "gaussian":
        return gaussian
    lorentzian = amplitude / (1.0 + 4.0 * ((x_values - center) / safe_fwhm) ** 2)
    if profile == "lorentzian":
        return lorentzian
    return 0.65 * gaussian + 0.35 * lorentzian


def _profile_area(amplitude: float, fwhm: float, profile: str) -> float:
    gaussian_area = amplitude * fwhm * math.sqrt(math.pi / (4.0 * math.log(2.0)))
    if profile == "gaussian":
        return gaussian_area
    lorentzian_area = amplitude * math.pi * fwhm / 2.0
    if profile == "lorentzian":
        return lorentzian_area
    return 0.65 * gaussian_area + 0.35 * lorentzian_area


def _ordering_penalties(params: np.ndarray, peaks: tuple[PeakTemplate, ...], noise: float) -> np.ndarray:
    centers = {peak.label: float(params[index * 3 + 1]) for index, peak in enumerate(peaks)}
    scale = max(noise, 0.01) * 8.0
    penalties: list[float] = []
    if "trion" in centers and "A" in centers:
        penalties.append(max(0.0, centers["trion"] + 0.008 - centers["A"]) * scale / 0.008)
    if "A" in centers and "B" in centers:
        penalties.append(max(0.0, centers["A"] + 0.08 - centers["B"]) * scale / 0.08)
    if "E" in centers and "G" in centers:
        expected = {peak.label: peak.center for peak in peaks}
        expected_sign = 1.0 if expected["G"] >= expected["E"] else -1.0
        observed_delta = expected_sign * (centers["G"] - centers["E"])
        penalties.append(max(0.0, 1.0 - observed_delta) * scale)
    return np.asarray(penalties, dtype=float)


def _ordering_is_valid(params: np.ndarray, peaks: tuple[PeakTemplate, ...]) -> bool:
    centers = {peak.label: float(params[index * 3 + 1]) for index, peak in enumerate(peaks)}
    if "trion" in centers and "A" in centers and centers["trion"] >= centers["A"] - 0.005:
        return False
    if "A" in centers and "B" in centers and centers["B"] <= centers["A"] + 0.06:
        return False
    if "E" in centers and "G" in centers:
        expected = {peak.label: peak.center for peak in peaks}
        expected_sign = 1.0 if expected["G"] >= expected["E"] else -1.0
        if expected_sign * (centers["G"] - centers["E"]) < 1.0:
            return False
    return True


def _parameter_uncertainties(
    jacobian: np.ndarray,
    rss: float,
    observation_count: int,
    parameter_count: int,
) -> np.ndarray:
    if observation_count <= parameter_count or jacobian.shape != (observation_count, parameter_count):
        return np.full(parameter_count, np.inf)
    try:
        information = jacobian.T @ jacobian
        if not np.all(np.isfinite(information)) or np.linalg.cond(information) > 1e14:
            return np.full(parameter_count, np.inf)
        covariance = np.linalg.inv(information) * (rss / (observation_count - parameter_count))
        diagonal = np.maximum(np.diag(covariance), 0.0)
        return np.sqrt(diagonal)
    except np.linalg.LinAlgError:
        return np.full(parameter_count, np.inf)


def _peak_resolution_score(peaks: list[dict[str, object]]) -> float:
    if not peaks:
        return 0.0
    if len(peaks) == 1:
        return 1.0
    ordered = sorted(peaks, key=lambda peak: float(peak["center"]))
    scores: list[float] = []
    for left, right in zip(ordered, ordered[1:]):
        separation = float(right["center"]) - float(left["center"])
        mean_fwhm = 0.5 * (float(left["fwhm"]) + float(right["fwhm"]))
        scores.append(max(0.0, min(1.0, separation / max(mean_fwhm, 1e-9))))
    return float(np.mean(scores))


def _robust_noise(values: np.ndarray) -> float:
    if len(values) < 3:
        return max(float(np.std(values)), 1e-6)
    differences = np.diff(values)
    median = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median)))
    return max(mad / (0.67448975 * math.sqrt(2.0)), 1e-6)


def _empty_fit(profile: str, fit_window: list[float] | None = None) -> dict[str, object]:
    return {
        "model": profile,
        "fit_curve": [],
        "r2": None,
        "rmse": None,
        "bic": None,
        "peaks": [],
        "fit_window": fit_window,
        "signal_to_noise": 0.0,
        "evidence_score": 0.0,
        "physical_score": 0.0,
        "resolution_score": 0.0,
        "baseline": None,
        "converged": False,
    }


def _local_height(x_values: np.ndarray, y_values: np.ndarray, center: float, tolerance: float) -> float:
    mask = (x_values >= center - tolerance) & (x_values <= center + tolerance)
    if not np.any(mask):
        return 0.05
    return float(np.max(y_values[mask]))


def _build_features(family: str, peaks: object) -> dict[str, float | None]:
    by_label = {str(item.get("label")): item for item in peaks if isinstance(item, dict)}
    if family == "Raman":
        e_peak = by_label.get("E")
        g_peak = by_label.get("G")
        return {
            "E_intensity": _peak_value(e_peak, "amplitude"),
            "G_intensity": _peak_value(g_peak, "amplitude"),
            "E_to_G_intensity_ratio": _ratio(_peak_value(e_peak, "amplitude"), _peak_value(g_peak, "amplitude")),
            "E_to_G_area_ratio": _ratio(_peak_value(e_peak, "area"), _peak_value(g_peak, "area")),
            "E_center": _peak_value(e_peak, "center"),
            "G_center": _peak_value(g_peak, "center"),
            "E_fwhm": _peak_value(e_peak, "fwhm"),
            "G_fwhm": _peak_value(g_peak, "fwhm"),
            "E_G_separation": _delta(_peak_value(g_peak, "center"), _peak_value(e_peak, "center")),
        }
    trion = by_label.get("trion")
    a_peak = by_label.get("A")
    b_peak = by_label.get("B")
    return {
        "A_intensity": _peak_value(a_peak, "amplitude"),
        "B_intensity": _peak_value(b_peak, "amplitude"),
        "trion_intensity": _peak_value(trion, "amplitude"),
        "A_to_B_intensity_ratio": _ratio(_peak_value(a_peak, "amplitude"), _peak_value(b_peak, "amplitude")),
        "trion_to_A_intensity_ratio": _ratio(_peak_value(trion, "amplitude"), _peak_value(a_peak, "amplitude")),
        "A_to_B_area_ratio": _ratio(_peak_value(a_peak, "area"), _peak_value(b_peak, "area")),
        "trion_to_A_area_ratio": _ratio(_peak_value(trion, "area"), _peak_value(a_peak, "area")),
        "A_center": _peak_value(a_peak, "center"),
        "B_center": _peak_value(b_peak, "center"),
        "trion_center": _peak_value(trion, "center"),
        "A_fwhm": _peak_value(a_peak, "fwhm"),
        "B_fwhm": _peak_value(b_peak, "fwhm"),
        "trion_fwhm": _peak_value(trion, "fwhm"),
    }


def _fit_quality(fit: dict[str, object]) -> dict[str, object]:
    r2_value = fit.get("r2")
    r2 = float(r2_value) if isinstance(r2_value, (int, float)) and math.isfinite(float(r2_value)) else None
    if r2 is None:
        return {"score": None, "label": "unfit", "reasons": ["No stable fit was available in the material window."]}
    rmse_value = fit.get("rmse")
    rmse = float(rmse_value) if isinstance(rmse_value, (int, float)) and math.isfinite(float(rmse_value)) else 1.0
    snr_value = fit.get("signal_to_noise")
    snr = float(snr_value) if isinstance(snr_value, (int, float)) and math.isfinite(float(snr_value)) else 0.0
    evidence_value = fit.get("evidence_score")
    evidence = float(evidence_value) if isinstance(evidence_value, (int, float)) else 0.0
    physical_value = fit.get("physical_score")
    physical = float(physical_value) if isinstance(physical_value, (int, float)) else 0.0
    material_evidence_value = fit.get("material_evidence")
    material_evidence = float(material_evidence_value) if isinstance(material_evidence_value, (int, float)) else 0.0
    matched_template_peaks = int(fit.get("matched_template_peaks") or 0)
    r2_score = max(0.0, min(1.0, r2))
    residual_score = math.exp(-2.2 * max(0.0, rmse))
    snr_score = 1.0 - math.exp(-max(0.0, snr) / 6.0)
    score = 0.55 * r2_score + 0.15 * residual_score + 0.15 * snr_score + 0.15 * max(0.0, min(1.0, physical))
    reasons: list[str] = []
    if evidence < 0.30:
        score = min(score, 0.68)
        reasons.append("Peak evidence is weak in the expected material window.")
    if matched_template_peaks == 0 or material_evidence < 0.18:
        score = min(score, 0.40)
        reasons.append("No reliable detected peak supports the selected material template.")
    elif material_evidence < 0.28:
        score = min(score, 0.68)
        reasons.append("Material discrimination is below the automatic acceptance threshold.")
    if physical < 0.45:
        score = min(score, 0.68)
        reasons.append("Peak separation or parameter identifiability is weak.")
    if not bool(fit.get("converged")):
        score = min(score, 0.60)
        reasons.append("The robust optimizer did not fully converge.")
    if r2 < 0.65:
        reasons.append("The constrained components explain less than 65% of local variance.")
    if snr < 3.0:
        reasons.append("Resolved peak signal-to-noise is below 3.")
    score = max(0.0, min(1.0, score))
    if score >= 0.90:
        label = "excellent"
    elif score >= 0.78:
        label = "good"
    elif score >= 0.60:
        label = "check"
    else:
        label = "poor"
    return {"score": score, "label": label, "reasons": reasons}


def _peak_value(peak: object, key: str) -> float | None:
    if not isinstance(peak, dict):
        return None
    value = peak.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return float(numerator / denominator)


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)
