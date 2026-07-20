from __future__ import annotations

import numpy as np
import pytest

from backend.services.spectrum_processor import analyze_spectrum
from backend.services.material_analysis import analyze_material_spectrum


def test_analyze_spectrum_finds_peak() -> None:
    x_axis = np.linspace(650.0, 850.0, 512)
    intensity = 0.05 + np.exp(-0.5 * ((x_axis - 735.0) / 8.0) ** 2)
    result = analyze_spectrum(x_axis.tolist(), intensity.tolist())
    assert result["metrics"]["n_detected_peaks"] >= 1
    assert result["fit"]["model"] == "gaussian"


def test_material_analysis_identifies_mos2_raman_features() -> None:
    x_axis = np.linspace(330.0, 450.0, 600)
    intensity = (
        0.04
        + 0.75 * np.exp(-0.5 * ((x_axis - 383.0) / 4.5) ** 2)
        + 1.0 * np.exp(-0.5 * ((x_axis - 408.0) / 4.2) ** 2)
    )
    result = analyze_material_spectrum(
        x_axis.tolist(),
        intensity.tolist(),
        {"spectrum_type": "Raman", "x_axis_unit": "cm^-1"},
        {"spectrum_family": "Raman", "fit_model": "lorentzian"},
    )
    assert result["material"] == "MoS2"
    assert result["features"]["E_to_G_intensity_ratio"] is not None
    assert 0.4 < result["features"]["E_to_G_intensity_ratio"] < 1.2


def test_material_analysis_identifies_wse2_pl_features_from_nm_axis() -> None:
    energy = np.linspace(1.45, 2.18, 700)
    wavelength_nm = 1239.841984 / energy
    intensity = (
        0.03
        + 0.45 * np.exp(-0.5 * ((energy - 1.62) / 0.025) ** 2)
        + 1.0 * np.exp(-0.5 * ((energy - 1.66) / 0.035) ** 2)
        + 0.38 * np.exp(-0.5 * ((energy - 2.08) / 0.055) ** 2)
    )
    result = analyze_material_spectrum(
        wavelength_nm.tolist(),
        intensity.tolist(),
        {"spectrum_type": "PL", "x_axis_unit": "nm"},
        {"spectrum_family": "PL", "fit_model": "gaussian"},
    )
    assert result["material"] == "WSe2"
    assert result["axis"]["unit"] == "eV"
    assert result["features"]["trion_to_A_intensity_ratio"] is not None
    assert result["features"]["A_to_B_intensity_ratio"] is not None


def test_material_analysis_rejects_degenerate_nm_axis_without_runtime_warning() -> None:
    with pytest.raises(ValueError, match="too few finite points|degenerate"):
        analyze_material_spectrum(
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.2, 0.1],
            {"spectrum_type": "PL", "x_axis_unit": "nm"},
            {"spectrum_family": "PL"},
        )


def test_material_analysis_does_not_treat_monotonic_pl_background_as_a_good_fit() -> None:
    energy = np.linspace(1.45, 2.15, 700)
    intensity = 0.2 + 0.35 * energy + 0.08 * energy**2
    result = analyze_material_spectrum(
        energy.tolist(),
        intensity.tolist(),
        {"spectrum_type": "PL", "x_axis_unit": "eV"},
        {"spectrum_family": "PL", "fit_model": "auto"},
    )
    assert result["material"] == "unknown"
    assert result["quality"]["label"] == "poor"
    assert result["fit"]["peaks"] == []


def test_material_analysis_keeps_pl_peak_order_and_uncertainty() -> None:
    rng = np.random.default_rng(17)
    energy = np.linspace(1.45, 2.18, 900)
    intensity = (
        0.025
        + 0.38 * np.exp(-4.0 * np.log(2.0) * ((energy - 1.615) / 0.055) ** 2)
        + 1.00 * np.exp(-4.0 * np.log(2.0) * ((energy - 1.665) / 0.075) ** 2)
        + 0.32 * np.exp(-4.0 * np.log(2.0) * ((energy - 2.075) / 0.12) ** 2)
        + rng.normal(0.0, 0.004, energy.size)
    )
    result = analyze_material_spectrum(
        energy.tolist(),
        intensity.tolist(),
        {"spectrum_type": "PL", "x_axis_unit": "eV"},
        {"spectrum_family": "PL", "fit_model": "auto"},
    )
    peaks = {peak["label"]: peak for peak in result["fit"]["peaks"]}
    assert peaks["trion"]["center"] < peaks["A"]["center"] < peaks["B"]["center"]
    assert np.isfinite(peaks["A"]["center_stderr"])
    assert 0.04 < peaks["A"]["fwhm"] < 0.12


def test_material_analysis_is_robust_to_outlier_away_from_raman_window() -> None:
    x_axis = np.linspace(250.0, 650.0, 1600)
    intensity = (
        0.03
        + 0.65 * np.exp(-4.0 * np.log(2.0) * ((x_axis - 383.0) / 8.0) ** 2)
        + 0.92 * np.exp(-4.0 * np.log(2.0) * ((x_axis - 408.0) / 7.0) ** 2)
    )
    intensity[np.argmin(np.abs(x_axis - 575.0))] = 8.0
    result = analyze_material_spectrum(
        x_axis.tolist(),
        intensity.tolist(),
        {"spectrum_type": "Raman", "x_axis_unit": "cm^-1"},
        {"spectrum_family": "Raman", "fit_model": "auto"},
    )
    assert result["material"] == "MoS2"
    assert result["quality"]["label"] in {"good", "excellent"}
    assert result["features"]["E_to_G_intensity_ratio"] is not None
