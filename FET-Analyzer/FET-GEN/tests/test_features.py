import warnings

import numpy as np
import pytest

from devicecurvegen.features import analyze_transfer_curve
from devicecurvegen.physics import (
    _effective_ai_balance,
    _gate_leakage_components,
    _gate_leakage_current,
    _output_noise_sigma,
    _physics_log_current,
    _quantize_current,
    _stabilize_on_state_log_current,
    generate_curves,
)
from devicecurvegen.residual import ResidualSample
from devicecurvegen.schemas import GenerationCondition


def test_generated_curve_tracks_requested_features() -> None:
    condition = GenerationCondition(
        target_ion=1e-5,
        target_ioff=1e-11,
        target_vth=5.0,
        target_ss_mv_dec=120.0,
        hysteresis_v=1.0,
        ai_residual_strength=0.0,
        noise_sigma_a=0.0,
        physical_strictness=1.0,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]
    assert candidate.features.ion == np.max(candidate.forward_current)
    assert candidate.features.ion_ioff_ratio > 1e4
    assert candidate.features.vth is not None
    assert abs(candidate.features.vth - condition.target_vth) < 1.0
    assert candidate.features.ss_mv_dec is not None
    assert abs(candidate.features.ss_mv_dec - condition.target_ss_mv_dec) < 35


def test_physics_baseline_has_smooth_subthreshold_transition() -> None:
    condition = GenerationCondition(
        target_ion=1e-5,
        target_ioff=1e-11,
        target_vth=5.0,
        target_ss_mv_dec=120.0,
        voltage_min=0.0,
        voltage_max=12.0,
        ai_residual_strength=0.0,
        noise_sigma_a=0.0,
        gate_leakage_a=0.0,
        physical_strictness=1.0,
        points=601,
    )
    voltage = np.linspace(condition.voltage_min, condition.voltage_max, condition.points)
    log_current = _physics_log_current(voltage, condition, reverse=False)
    slope = np.diff(log_current) / np.diff(voltage)

    assert np.max(np.abs(np.diff(slope))) < 2.0


def test_generation_includes_floor_quantization_and_gate_leakage_terms() -> None:
    condition = GenerationCondition(
        target_ioff=1e-15,
        noise_floor_a=3e-12,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        ai_residual_strength=0.0,
        noise_sigma_a=0.0,
        physical_strictness=0.0,
        variants=1,
        points=301,
    )
    candidate = generate_curves(condition).candidates[0]
    current = np.asarray(candidate.forward_current)
    physics = np.asarray(candidate.physics_forward_current)
    off_state = current[:40]

    assert np.min(physics[:40]) < condition.noise_floor_a * 0.1
    assert np.std(off_state) > condition.noise_floor_a * 0.25
    assert np.min(off_state) < condition.noise_floor_a * 0.5


def test_quantization_step_is_the_minimum_current_resolution() -> None:
    step = 1e-12
    current = np.asarray([0.2, 0.49, 0.51, 1.49, 1.51, 3.2]) * step

    quantized = _quantize_current(current, step)

    assert np.allclose(
        quantized,
        np.asarray([1.0, 1.0, 1.0, 1.0, 2.0, 3.0]) * step,
    )
    assert np.allclose(quantized / step, np.round(quantized / step))


def test_generated_id_and_ig_use_the_configured_current_resolution() -> None:
    step = 1e-13
    condition = GenerationCondition(
        quantization_step_a=step,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]

    for series in [
        candidate.forward_current,
        candidate.reverse_current,
        candidate.gate_forward_current,
        candidate.gate_reverse_current,
        *[curve.current for curve in candidate.output_curves],
    ]:
        levels = np.asarray(series) / step
        assert np.allclose(levels, np.round(levels), atol=1e-8)


def test_learned_residual_cannot_create_macroscopic_on_state_ndc() -> None:
    class BroadPeakResidual:
        mode = "conditional_vae"
        model_name = "broad-peak-test"

        def sample(self, normalized_voltage, **_kwargs):
            values = 1.4 * np.exp(
                -0.5 * ((np.asarray(normalized_voltage) - 0.45) / 0.12) ** 2
            )
            return ResidualSample(values, self.mode, [1.4, 0.45, 0.12])

    condition = GenerationCondition(
        voltage_min=-5.0,
        voltage_max=5.0,
        target_vth=0.0,
        target_ss_mv_dec=300.0,
        physical_strictness=0.0,
        ai_residual_strength=1.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
    )
    candidate = generate_curves(condition, BroadPeakResidual()).candidates[0]
    voltage = np.asarray(candidate.voltage)
    log_current = np.log10(np.asarray(candidate.forward_current))
    on_state = voltage > condition.target_vth + 2.0 * condition.target_ss_mv_dec / 1000.0
    running_peak = np.maximum.accumulate(log_current[on_state])
    drawdown = running_peak - log_current[on_state]

    assert np.max(drawdown) <= 0.031


def test_ai_residual_is_smooth_in_large_current_region() -> None:
    class HighFrequencyOnResidual:
        mode = "conditional_vae"
        model_name = "high-frequency-on-state-test"

        def sample(self, normalized_voltage, **_kwargs):
            x = np.asarray(normalized_voltage)
            gate = 1.0 / (1.0 + np.exp(-24.0 * (x - 0.05)))
            values = 0.45 * gate * np.sin(38.0 * np.pi * x)
            return ResidualSample(values, self.mode, [0.45, 38.0])

    condition = GenerationCondition(
        voltage_min=-5.0,
        voltage_max=5.0,
        target_vth=0.0,
        target_ss_mv_dec=240.0,
        physical_strictness=0.0,
        ai_residual_strength=1.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
        points=801,
    )
    candidate = generate_curves(condition, HighFrequencyOnResidual()).candidates[0]
    voltage = np.asarray(candidate.voltage)
    log_current = np.log10(np.asarray(candidate.forward_current))
    on_state = voltage > condition.target_vth + 2.0 * condition.target_ss_mv_dec / 1000.0
    on_log = log_current[on_state]
    on_voltage = voltage[on_state]
    slope = np.diff(on_log) / np.diff(on_voltage)

    assert np.quantile(np.abs(np.diff(slope)), 0.95) < 0.15


def test_threshold_jump_limiter_softens_local_ai_spikes() -> None:
    class StepThresholdResidual:
        mode = "conditional_vae"
        model_name = "step-threshold-test"

        def sample(self, normalized_voltage, **_kwargs):
            x = np.asarray(normalized_voltage)
            values = np.where(x > -0.02, 1.8, -1.2)
            return ResidualSample(values, self.mode, [1.8, -1.2])

    condition = GenerationCondition(
        voltage_min=-5.0,
        voltage_max=5.0,
        target_vth=0.0,
        target_ss_mv_dec=230.0,
        hysteresis_v=1.5,
        ai_residual_strength=1.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
        points=801,
    )
    candidate = generate_curves(condition, StepThresholdResidual()).candidates[0]
    voltage = np.asarray(candidate.voltage)
    log_current = np.log10(np.asarray(candidate.forward_current))
    threshold_band = np.abs(voltage - condition.target_vth) <= 1.5
    local_jump = np.max(np.abs(np.diff(log_current[threshold_band])))

    assert local_jump < 0.5


def test_low_ai_balance_does_not_rewrite_a_smooth_physics_curve() -> None:
    condition = GenerationCondition(
        voltage_min=-20.0,
        voltage_max=20.0,
        target_vth=0.0,
        target_ss_mv_dec=230.0,
        hysteresis_v=1.5,
        ai_residual_strength=0.01,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
        points=601,
    )
    voltage = np.linspace(condition.voltage_min, condition.voltage_max, condition.points)
    baseline = _physics_log_current(voltage, condition, reverse=False)
    stabilized = _stabilize_on_state_log_current(
        voltage,
        baseline,
        condition,
        reverse=False,
    )

    assert np.max(np.abs(stabilized - baseline)) < 0.01


def test_low_ai_balance_has_a_deadband_before_residual_cut_in() -> None:
    class ThresholdStepResidual:
        mode = "conditional_vae"
        model_name = "threshold-step-deadband-test"

        def sample(self, normalized_voltage, **_kwargs):
            x = np.asarray(normalized_voltage)
            values = np.where(x > 0.0, 2.0, -1.5)
            return ResidualSample(values, self.mode, [2.0, -1.5])

    physics_first = GenerationCondition(
        voltage_min=-5.0,
        voltage_max=5.0,
        target_vth=0.0,
        target_ss_mv_dec=230.0,
        hysteresis_v=1.5,
        ai_residual_strength=0.0,
        gate_ai_residual_strength=0.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
        points=801,
    )
    low_ai = physics_first.model_copy(
        update={
            "ai_residual_strength": 0.04,
            "gate_ai_residual_strength": 0.04,
        }
    )

    baseline = generate_curves(physics_first, ThresholdStepResidual()).candidates[0]
    low_balance = generate_curves(low_ai, ThresholdStepResidual()).candidates[0]

    np.testing.assert_allclose(
        np.asarray(low_balance.forward_current),
        np.asarray(baseline.forward_current),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(low_balance.reverse_current),
        np.asarray(baseline.reverse_current),
        rtol=0.0,
        atol=0.0,
    )
    assert _effective_ai_balance(low_ai.ai_residual_strength) == 0.0


def test_gate_leakage_depends_on_vgs_and_vgd_fields() -> None:
    condition = GenerationCondition(
        target_vth=0.0,
        voltage_min=-2.0,
        voltage_max=2.0,
        gate_leakage_a=1e-15,
        gate_leakage_v_char=0.7,
        gate_leakage_exponent=0.8,
        hysteresis_v=0.0,
    )
    voltage = np.linspace(-2.0, 2.0, 9)
    leakage = _gate_leakage_current(voltage, condition, reverse=False)
    softer_dielectric = condition.model_copy(update={"gate_leakage_v_char": 1.4})
    softer_leakage = _gate_leakage_current(voltage, softer_dielectric, reverse=False)

    assert np.ptp(leakage) > condition.gate_leakage_a
    assert leakage[0] > leakage[4]
    assert leakage[0] > leakage[-1]
    assert np.max(leakage) > np.max(softer_leakage)


def test_gate_leakage_is_terminal_based_not_hysteresis_shifted() -> None:
    condition = GenerationCondition(
        target_vth=5.0,
        hysteresis_v=2.0,
        gate_leakage_a=1e-15,
        voltage_min=0.0,
        voltage_max=12.0,
    )
    voltage = np.linspace(condition.voltage_min, condition.voltage_max, 101)
    forward = _gate_leakage_current(voltage, condition, reverse=False)
    reverse = _gate_leakage_current(voltage, condition, reverse=True)

    assert np.allclose(forward, reverse)


def test_off_state_forward_reverse_overlap_with_residual_enabled() -> None:
    condition = GenerationCondition(
        target_vth=5.0,
        target_ioff=1e-14,
        hysteresis_v=1.5,
        ai_residual_strength=1.0,
        physical_strictness=0.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]
    voltage = np.asarray(candidate.voltage)
    forward = np.asarray(candidate.forward_current)
    reverse = np.asarray(candidate.reverse_current)
    off_state = voltage < condition.target_vth - condition.hysteresis_v - 1.0
    relative_delta = np.abs(reverse[off_state] - forward[off_state]) / condition.target_ioff

    assert np.quantile(relative_delta, 0.95) < 0.05


@pytest.mark.parametrize("physical_strictness", [0.0, 0.7, 1.0])
def test_forward_reverse_sweeps_close_at_both_endpoints(
    physical_strictness: float,
) -> None:
    condition = GenerationCondition(
        physical_strictness=physical_strictness,
        ai_residual_strength=1.0,
        noise_sigma_a=2e-13,
        noise_floor_a=1e-13,
        quantization_step_a=1e-14,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]
    forward = np.asarray(candidate.forward_current)
    reverse = np.asarray(candidate.reverse_current)

    assert forward[0] == reverse[0]
    assert forward[-1] == reverse[-1]
    center = slice(len(forward) // 3, 2 * len(forward) // 3)
    assert np.max(np.abs(np.log10(forward[center]) - np.log10(reverse[center]))) > 0.01


def test_variants_have_independent_seeded_parameter_jitter() -> None:
    condition = GenerationCondition(
        variants=24,
        ion_sigma_fraction=0.12,
        ioff_sigma_fraction=0.20,
        vth_sigma_v=0.25,
        ss_sigma_fraction=0.12,
        hysteresis_sigma_v=0.20,
        ai_residual_strength=0.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
    )
    response = generate_curves(condition)
    ions = np.asarray([candidate.features.ion for candidate in response.candidates])
    ioffs = np.asarray([candidate.features.ioff for candidate in response.candidates])
    vths = np.asarray(
        [candidate.features.vth for candidate in response.candidates if candidate.features.vth]
    )
    ss_values = np.asarray(
        [
            candidate.features.ss_mv_dec
            for candidate in response.candidates
            if candidate.features.ss_mv_dec is not None
        ]
    )
    hysteresis = np.asarray(
        [
            candidate.features.hysteresis_v
            for candidate in response.candidates
            if candidate.features.hysteresis_v is not None
        ]
    )

    assert len(vths) == condition.variants
    assert np.std(ions) > condition.target_ion * 0.03
    assert np.std(ioffs) > condition.target_ioff * 0.03
    assert np.std(vths) > 0.05
    assert np.std(ss_values) > condition.target_ss_mv_dec * 0.02
    assert np.std(hysteresis) > 0.03


def test_generation_noise_is_added_in_linear_current_domain() -> None:
    condition = GenerationCondition(
        target_ioff=1e-10,
        hysteresis_v=0.0,
        ai_residual_strength=0.0,
        physical_strictness=0.0,
        noise_sigma_a=2e-12,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        variants=1,
        points=1001,
    )
    noisy = np.asarray(generate_curves(condition).candidates[0].forward_current)
    clean_condition = condition.model_copy(update={"noise_sigma_a": 0.0})
    clean = np.asarray(generate_curves(clean_condition).candidates[0].forward_current)
    off_state = clean < condition.target_ioff * 1.1

    assert np.std(noisy[off_state] - clean[off_state]) == pytest.approx(
        condition.noise_sigma_a,
        rel=0.25,
    )


def test_gate_leakage_default_does_not_explode_over_wide_voltage_range() -> None:
    condition = GenerationCondition(
        voltage_min=-20.0,
        voltage_max=20.0,
        target_vth=0.0,
    )
    voltage = np.linspace(condition.voltage_min, condition.voltage_max, 401)
    leakage = _gate_leakage_current(voltage, condition, reverse=False)

    assert np.max(leakage) < condition.target_ion * 0.01


def test_generation_includes_output_characteristic_family() -> None:
    condition = GenerationCondition(
        target_ioff=1e-15,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]

    assert len(candidate.output_curves) == 5
    assert candidate.output_drain_voltage[0] == 0.0
    assert candidate.output_drain_voltage[-1] == pytest.approx(condition.vd)
    assert all(
        len(curve.current) == len(candidate.output_drain_voltage)
        for curve in candidate.output_curves
    )
    assert all(curve.current[0] == pytest.approx(0.0) for curve in candidate.output_curves)
    terminal_currents = [curve.current[-1] for curve in candidate.output_curves]
    assert terminal_currents == sorted(terminal_currents)


def test_output_noise_has_floor_and_current_scaled_component() -> None:
    condition = GenerationCondition(
        target_ion=1e-5,
        target_ioff=1e-14,
        noise_sigma_a=2e-12,
        noise_floor_a=4e-13,
        quantization_step_a=0.0,
        variants=1,
    )
    current = np.geomspace(1e-14, 1e-5, 128)
    sigma = _output_noise_sigma(current, condition)

    assert sigma[0] == pytest.approx(condition.noise_floor_a, rel=0.02)
    assert sigma[-1] > sigma[0]
    assert sigma[-1] > (
        condition.noise_floor_a
        + condition.noise_sigma_a * condition.output_noise_gain * 0.75
    )
    assert sigma[-1] - sigma[-16] > sigma[16] - sigma[0]


def test_output_noise_gain_controls_large_current_noise() -> None:
    condition = GenerationCondition(
        target_ion=1e-5,
        target_ioff=1e-14,
        noise_sigma_a=2e-12,
        noise_floor_a=4e-13,
        output_noise_gain=1.0,
        quantization_step_a=0.0,
        variants=1,
    )
    boosted = condition.model_copy(update={"output_noise_gain": 8.0})
    current = np.asarray([condition.target_ioff, condition.target_ion])
    base_sigma = _output_noise_sigma(current, condition)
    boosted_sigma = _output_noise_sigma(current, boosted)

    assert boosted_sigma[0] == pytest.approx(base_sigma[0])
    assert boosted_sigma[-1] > base_sigma[-1] * 4.0


def test_output_gate_range_uses_ss_to_span_off_and_on_states() -> None:
    condition = GenerationCondition(
        voltage_min=-10.0,
        voltage_max=10.0,
        target_vth=1.0,
        target_ion=1e-5,
        target_ioff=1e-11,
        target_ss_mv_dec=200.0,
        contact_resistance_ohm=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]
    gates = np.asarray([curve.gate_voltage for curve in candidate.output_curves])
    expected_span = 2.0 * condition.ss_region_width_v

    assert gates[0] < condition.target_vth < gates[-1]
    assert gates[-1] - gates[0] == pytest.approx(expected_span, rel=0.05)
    assert gates[0] + gates[-1] == pytest.approx(2.0 * condition.target_vth)


def test_gate_current_includes_linear_domain_shot_noise() -> None:
    condition = GenerationCondition(
        gate_leakage_a=1e-11,
        noise_sigma_a=0.0,
        noise_floor_a=1e-20,
        quantization_step_a=0.0,
        variants=1,
        points=601,
    )
    candidate = generate_curves(condition).candidates[0]
    voltage = np.asarray(candidate.voltage)
    gate_source, gate_drain = _gate_leakage_components(
        voltage,
        condition,
        reverse=False,
    )
    deterministic = gate_source + gate_drain
    measured = np.asarray(candidate.gate_forward_current)

    assert np.std(measured - deterministic) > 0.0
    assert np.mean(np.abs(measured - deterministic)) < np.mean(deterministic) * 0.1


def test_generation_defaults_match_relaxed_physics_first_controls() -> None:
    condition = GenerationCondition()

    assert condition.diversity == 1.0
    assert condition.ai_residual_strength == 0.0
    assert condition.target_vth == 0.0
    assert condition.vth_sigma_v == 0.20
    assert condition.target_ss_mv_dec == 230.0
    assert condition.ss_sigma_fraction == 0.10
    assert condition.ss_region_width_v == 0.5
    assert condition.quantization_step_a == 1e-15
    assert condition.output_noise_gain == 4.0


def test_ss_region_width_controls_output_gate_span_without_transfer_plateau() -> None:
    base = GenerationCondition(
        target_vth=0.0,
        target_ss_mv_dec=230.0,
        ss_region_width_v=2.3,
        voltage_min=-5.0,
        voltage_max=20.0,
        ai_residual_strength=0.0,
        physical_strictness=0.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        contact_resistance_ohm=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
    )
    narrow = generate_curves(base).candidates[0]
    wide = generate_curves(base.model_copy(update={"ss_region_width_v": 12.0})).candidates[0]
    voltage = np.asarray(narrow.voltage)
    narrow_current = np.asarray(narrow.forward_current)
    narrow_gates = np.asarray([curve.gate_voltage for curve in narrow.output_curves])
    wide_gates = np.asarray([curve.gate_voltage for curve in wide.output_curves])
    current_at_10v = narrow_current[np.argmin(np.abs(voltage - 10.0))]
    current_at_20v = narrow_current[np.argmin(np.abs(voltage - 20.0))]

    assert current_at_20v > current_at_10v * 1.4
    assert wide_gates[-1] - wide_gates[0] > narrow_gates[-1] - narrow_gates[0]


def test_transfer_linear_ids_keeps_rising_in_on_state() -> None:
    condition = GenerationCondition(
        target_vth=0.0,
        target_ss_mv_dec=230.0,
        voltage_min=-20.0,
        voltage_max=20.0,
        ai_residual_strength=0.0,
        physical_strictness=0.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]
    voltage = np.asarray(candidate.voltage)
    current = np.asarray(candidate.forward_current)

    current_at_10v = current[np.argmin(np.abs(voltage - 10.0))]
    current_at_20v = current[np.argmin(np.abs(voltage - 20.0))]

    assert current_at_20v > current_at_10v * 1.4
    assert candidate.features.ion == pytest.approx(condition.target_ion, rel=0.12)


def test_output_on_curve_remains_quasi_linear_at_high_vds() -> None:
    condition = GenerationCondition(
        ai_residual_strength=0.0,
        physical_strictness=0.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        variants=1,
    )
    candidate = generate_curves(condition).candidates[0]
    high_gate_current = np.asarray(candidate.output_curves[-1].current)
    midpoint = high_gate_current.size // 2
    high_vds_gain = high_gate_current[-1] - high_gate_current[midpoint]

    assert high_vds_gain > high_gate_current[-1] * 0.20


def test_mobility_and_contact_resistance_affect_device_current() -> None:
    base = GenerationCondition(
        mobility_cm2_vs=20.0,
        contact_resistance_ohm=0.0,
        mobility_sigma_fraction=0.0,
        contact_resistance_sigma_fraction=0.0,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        ai_residual_strength=0.0,
        physical_strictness=1.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
        variants=1,
    )
    low_mobility = generate_curves(
        base.model_copy(update={"mobility_cm2_vs": 10.0})
    ).candidates[0]
    high_mobility = generate_curves(
        base.model_copy(update={"mobility_cm2_vs": 40.0})
    ).candidates[0]
    high_resistance = generate_curves(
        base.model_copy(update={"contact_resistance_ohm": 100_000.0})
    ).candidates[0]

    assert high_mobility.features.ion > low_mobility.features.ion
    assert high_resistance.features.ion < generate_curves(base).candidates[0].features.ion
    assert (
        high_resistance.output_curves[-1].current[-1]
        < high_mobility.output_curves[-1].current[-1]
    )


def test_mobility_and_contact_resistance_have_seeded_dispersion() -> None:
    condition = GenerationCondition(
        variants=24,
        mobility_sigma_fraction=0.20,
        contact_resistance_sigma_fraction=0.25,
        ion_sigma_fraction=0.0,
        ioff_sigma_fraction=0.0,
        vth_sigma_v=0.0,
        ss_sigma_fraction=0.0,
        hysteresis_sigma_v=0.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
    )
    candidates = generate_curves(condition).candidates

    assert np.std([candidate.mobility_cm2_vs for candidate in candidates]) > 1.0
    assert np.std([candidate.contact_resistance_ohm for candidate in candidates]) > 100.0


def test_p_type_feature_extraction() -> None:
    voltage = np.linspace(-10, 10, 201)
    current = 1e-12 + 1e-5 / (1 + np.exp((voltage + 1.5) / 0.45))
    features = analyze_transfer_curve(voltage, current, polarity="p-type")
    assert features.ion > 1e-6
    assert features.ioff < 1e-9
    assert features.vth is not None
    assert features.polarity == "p-type"


def test_bipolar_feature_extraction_marks_two_edge_conduction() -> None:
    voltage = np.linspace(-10, 10, 301)
    current = (
        1e-12
        + 4e-6 / (1 + np.exp((voltage + 5.0) / 0.45))
        + 5e-6 / (1 + np.exp(-(voltage - 5.0) / 0.45))
    )

    features = analyze_transfer_curve(voltage, current)

    assert features.polarity == "bipolar"
    assert features.ambipolar_strength is not None
    assert features.ambipolar_strength > 0.05


def test_duplicate_voltage_points_do_not_emit_gradient_warnings() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = analyze_transfer_curve(
            [0, 0, 1, 2, 3, 4, 5],
            [1e-12, 2e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7],
        )
    assert len(caught) == 0
    assert features.gm_max is not None
    assert features.noise_log_sigma is not None
    assert features.current_floor == features.ioff


def test_generation_is_deterministic_and_exposes_latent_code() -> None:
    condition = GenerationCondition(seed=77, variants=2, points=101)
    first = generate_curves(condition)
    second = generate_curves(condition)
    assert first == second
    assert first.candidates[0].latent_code


@pytest.mark.parametrize("polarity", ["n-type", "p-type"])
def test_physical_strictness_is_ignored_for_backward_compatibility(
    polarity: str,
) -> None:
    base = GenerationCondition(
        polarity=polarity,
        variants=2,
        ai_residual_strength=1.0,
        noise_sigma_a=0.0,
        noise_floor_a=0.0,
        quantization_step_a=0.0,
        gate_leakage_a=0.0,
    )
    legacy = GenerationCondition.model_validate(
        {
            **base.model_dump(),
            "physical_strictness": 1.0,
        }
    )
    base_candidates = generate_curves(base).candidates
    legacy_candidates = generate_curves(legacy).candidates

    assert len(base_candidates) == len(legacy_candidates)
    assert "physical_strictness" not in legacy.model_dump()
    for baseline, deprecated in zip(base_candidates, legacy_candidates, strict=True):
        baseline_forward = np.asarray(baseline.forward_current)
        deprecated_forward = np.asarray(deprecated.forward_current)
        assert np.all(baseline_forward > 0)
        np.testing.assert_allclose(
            deprecated_forward,
            baseline_forward,
            rtol=0.0,
            atol=0.0,
        )
        assert sum(item.passed for item in deprecated.constraints) >= 4
