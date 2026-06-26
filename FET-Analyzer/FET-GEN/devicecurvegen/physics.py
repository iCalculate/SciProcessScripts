from __future__ import annotations

import numpy as np

from .features import analyze_transfer_curve, combine_sweep_features
from .residual import ResidualEngine
from .schemas import (
    ConstraintResult,
    GeneratedCandidate,
    GenerationCondition,
    GenerationResponse,
    OutputCurve,
)


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, values)


def _gate_leakage_current(
    voltage: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    return _gate_leakage_components(voltage, condition, reverse=reverse)[1]


def _gate_leakage_components(
    voltage: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if condition.gate_leakage_a <= 0:
        zeros = np.zeros_like(voltage)
        return zeros, zeros
    vgs = np.asarray(voltage, dtype=float)
    vgd = vgs - condition.vd
    v_char = max(condition.gate_leakage_v_char, 1e-9)
    exponent = condition.gate_leakage_exponent
    gs_field = np.clip((np.abs(vgs) / v_char) ** exponent, 0.0, 80.0)
    gd_field = np.clip((np.abs(vgd) / v_char) ** exponent, 0.0, 80.0)
    gate_source = condition.gate_leakage_a * gs_field
    gate_drain = condition.gate_leakage_a * gd_field
    return gate_source, gate_drain


def _apply_deterministic_parasitics(
    channel_current: np.ndarray,
    voltage: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    current = np.asarray(channel_current, dtype=float)
    current = current + _gate_leakage_current(voltage, condition, reverse=reverse)
    return np.clip(current, np.finfo(float).tiny, None)


def _apply_current_domain_noise(
    current: np.ndarray,
    condition: GenerationCondition,
    rng: np.random.Generator,
) -> np.ndarray:
    if condition.noise_sigma_a <= 0 and condition.noise_floor_a <= 0:
        return np.clip(current, np.finfo(float).tiny, None)
    electron_charge = 1.602176634e-19
    bandwidth_hz = 2e4
    sigma = np.sqrt(
        2.0 * electron_charge * np.maximum(current, 0.0) * bandwidth_hz
        + condition.noise_sigma_a**2
        + condition.noise_floor_a**2
    )
    measured = current + rng.normal(0.0, sigma, current.size)
    return np.clip(np.abs(measured), np.finfo(float).tiny, None)


def _output_noise_sigma(
    current: np.ndarray,
    condition: GenerationCondition,
) -> np.ndarray:
    current_magnitude = np.maximum(np.asarray(current, dtype=float), 0.0)
    fixed_floor = max(condition.noise_floor_a, 0.0)
    if condition.noise_sigma_a <= 0 or condition.output_noise_gain <= 0:
        return np.full_like(current_magnitude, fixed_floor)

    reference_current = max(
        _effective_on_current(condition),
        condition.target_ioff,
        np.finfo(float).tiny,
    )
    relative_current = current_magnitude / reference_current
    current_noise = (
        condition.noise_sigma_a
        * condition.output_noise_gain
        * np.sqrt(relative_current)
    )
    return np.sqrt(fixed_floor**2 + current_noise**2)


def _apply_output_domain_noise(
    current: np.ndarray,
    condition: GenerationCondition,
    rng: np.random.Generator,
) -> np.ndarray:
    if condition.noise_sigma_a <= 0 and condition.noise_floor_a <= 0:
        return np.clip(current, np.finfo(float).tiny, None)
    sigma = _output_noise_sigma(current, condition)
    measured = current + rng.normal(0.0, sigma, current.size)
    return np.clip(np.abs(measured), np.finfo(float).tiny, None)


def _quantize_current(
    current: np.ndarray,
    step: float,
) -> np.ndarray:
    if step <= 0:
        return np.clip(np.abs(current), np.finfo(float).tiny, None)
    levels = np.rint(np.abs(current) / step)
    return np.maximum(levels, 1.0) * step


def _contact_limited_current(
    intrinsic_current: np.ndarray,
    drain_voltage: float | np.ndarray,
    contact_resistance_ohm: float,
) -> np.ndarray:
    current = np.asarray(intrinsic_current, dtype=float)
    drain_magnitude = np.maximum(np.abs(drain_voltage), 1e-12)
    if contact_resistance_ohm <= 0:
        return np.clip(current, np.finfo(float).tiny, None)
    limited = current / (1.0 + current * contact_resistance_ohm / drain_magnitude)
    return np.clip(limited, np.finfo(float).tiny, None)


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    size = values.size
    if size < 3:
        return values.copy()
    window = min(max(window, 3), size if size % 2 == 1 else size - 1)
    if window < 3:
        return values.copy()
    if window % 2 == 0:
        window -= 1
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def _effective_on_current(condition: GenerationCondition) -> float:
    mobility_scale = condition.mobility_cm2_vs / 20.0
    intrinsic = condition.target_ioff + (
        condition.target_ion - condition.target_ioff
    ) * mobility_scale
    return float(
        _contact_limited_current(
            np.asarray([intrinsic]),
            max(abs(condition.vd), 1e-9),
            condition.contact_resistance_ohm,
        )[0]
    )


def _linear_extrapolation_vth_offset(condition: GenerationCondition) -> float:
    dynamic_span = (
        condition.target_ss_mv_dec
        / 1000.0
        * np.log10(max(condition.target_ion / condition.target_ioff, 1.0))
    )
    return 0.72 * dynamic_span


def _internal_transfer_vth(
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> float:
    sign = 1.0 if condition.polarity == "n-type" else -1.0
    hysteresis_shift = (0.5 if reverse else -0.5) * condition.hysteresis_v
    return condition.target_vth + sign * (
        _linear_extrapolation_vth_offset(condition) + hysteresis_shift
    )


def _target_transfer_vth(
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> float:
    sign = 1.0 if condition.polarity == "n-type" else -1.0
    hysteresis_shift = (0.5 if reverse else -0.5) * condition.hysteresis_v
    return condition.target_vth + sign * hysteresis_shift


def _normalized_transfer_charge(
    u: np.ndarray | float,
    vds: float | np.ndarray,
    condition: GenerationCondition,
) -> np.ndarray:
    thermal_v = 0.025852
    drain_voltage = np.maximum(np.asarray(vds, dtype=float), 1e-12)
    n_sub = max(condition.target_ss_mv_dec / (1000.0 * thermal_v * np.log(10.0)), 0.2)
    n_eff = n_sub * (1.0 + 0.03 * drain_voltage)
    kappa_sat = 8.0
    channel_length_modulation = 0.02
    pinch_off = np.asarray(u, dtype=float) / n_eff
    forward_charge = _softplus(pinch_off / (2.0 * thermal_v)) ** 2
    reverse_charge = _softplus((pinch_off - drain_voltage / kappa_sat) / (2.0 * thermal_v)) ** 2
    normalized = np.maximum(forward_charge - reverse_charge, 0.0)
    return normalized * (1.0 + channel_length_modulation * drain_voltage)


def _on_state_reference_charge(
    maximum_u: float,
    vds: float | np.ndarray,
    condition: GenerationCondition,
) -> np.ndarray:
    reference_u = min(max(condition.ss_region_width_v, 1e-6), max(maximum_u, 1e-6))
    return np.maximum(
        _normalized_transfer_charge(reference_u, vds, condition),
        np.finfo(float).tiny,
    )


def _physics_log_current(
    voltage: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    sign = 1.0 if condition.polarity == "n-type" else -1.0
    effective_vth = _internal_transfer_vth(condition, reverse=reverse)
    u = sign * (voltage - effective_vth)
    vds = max(abs(condition.vd), 1e-9)
    normalized_ekv = _normalized_transfer_charge(u, vds, condition)
    on_reference = float(
        np.maximum(
            _normalized_transfer_charge(float(np.max(u)), vds, condition),
            np.finfo(float).tiny,
        )
    )
    drive = normalized_ekv / on_reference
    mobility_scale = condition.mobility_cm2_vs / 20.0
    channel_span = max(condition.target_ion - condition.target_ioff, np.finfo(float).tiny)
    intrinsic_current = (
        condition.target_ioff
        + mobility_scale * channel_span * drive
    )
    channel_current = _contact_limited_current(
        intrinsic_current,
        max(abs(condition.vd), 1e-9),
        condition.contact_resistance_ohm,
    )
    measured_current = _apply_deterministic_parasitics(
        channel_current,
        voltage,
        condition,
        reverse=reverse,
    )
    return np.log10(measured_current)


def _stabilize_on_state_log_current(
    voltage: np.ndarray,
    log_current: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    if (
        condition.ai_residual_strength <= 0
        or condition.physical_strictness >= 0.95
        or log_current.size < 9
    ):
        return log_current
    sign = 1.0 if condition.polarity == "n-type" else -1.0
    effective_vth = _target_transfer_vth(condition, reverse=reverse)
    u = sign * (voltage - effective_vth)
    order = np.argsort(u)
    ordered_u = u[order]
    ordered_log = log_current[order].copy()
    ss_v = condition.target_ss_mv_dec / 1000.0
    onset = 1.6 * ss_v
    blend_width = max(1.2 * ss_v, (condition.voltage_max - condition.voltage_min) * 0.015)
    on_weight = 1.0 / (1.0 + np.exp(-np.clip((ordered_u - onset) / blend_width, -80.0, 80.0)))
    if np.max(on_weight) < 0.05:
        return log_current

    window = min(
        max(9, int(round(ordered_log.size * 0.080)) | 1),
        61,
    )
    smoothed = _smooth_series(ordered_log, window)
    robust_log = (1.0 - 0.95 * on_weight) * ordered_log + 0.95 * on_weight * smoothed

    on_region = on_weight > 0.5
    if np.count_nonzero(on_region) >= 3:
        values = robust_log[on_region]
        center = _smooth_series(values, min(max(5, int(round(values.size * 0.12)) | 1), 31))
        local_delta = np.clip(values - center, -0.025, 0.025)
        robust_log[on_region] = center + local_delta

    stabilized = log_current.copy()
    stabilized[order] = robust_log
    return stabilized


def _project_residual(
    voltage: np.ndarray,
    physics_log: np.ndarray,
    residual: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    sign = 1.0 if condition.polarity == "n-type" else -1.0
    effective_vth = _target_transfer_vth(condition, reverse=reverse)
    u = sign * (voltage - effective_vth)
    span = max(condition.voltage_max - condition.voltage_min, 1e-6)
    normalized_distance = np.abs(voltage - effective_vth) / span

    corrected = residual.copy()
    strictness = condition.physical_strictness
    residual_gate = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                u / max(2.0 * condition.target_ss_mv_dec / 1000.0, span * 0.015),
                -80.0,
                80.0,
            )
        )
    )
    corrected *= residual_gate
    off_mask = u < -3.0 * condition.target_ss_mv_dec / 1000.0
    on_mask = u > np.percentile(u, 85)
    if off_mask.any():
        corrected -= (
            strictness
            * np.mean(corrected[off_mask])
            * np.exp(-np.maximum(u, 0.0) / max(span * 0.15, 1e-6))
        )
    if on_mask.any():
        corrected -= (
            strictness
            * np.mean(corrected[on_mask])
            * (1.0 - np.exp(-np.maximum(u, 0.0) / max(span * 0.12, 1e-6)))
        )

    threshold_idx = int(np.argmin(np.abs(voltage - effective_vth)))
    threshold_error = corrected[threshold_idx]
    threshold_guard = np.exp(
        -0.5
        * ((voltage - effective_vth) / max(3.0 * condition.target_ss_mv_dec / 1000.0, span * 0.025))
        ** 2
    )
    corrected -= strictness * threshold_error * threshold_guard
    corrected *= 1.0 - 0.18 * strictness * normalized_distance

    combined = physics_log + condition.ai_residual_strength * corrected
    log_ioff = np.log10(condition.target_ioff)
    log_ion = np.log10(_effective_on_current(condition))
    slack = 0.45 * (1.0 - strictness) + 0.05
    combined = np.clip(combined, log_ioff, log_ion + slack)

    if strictness > 0:
        order = np.argsort(u)
        monotonic = np.maximum.accumulate(combined[order])
        blend = strictness**2
        combined[order] = (1.0 - blend) * combined[order] + blend * monotonic
    combined = _stabilize_on_state_log_current(
        voltage,
        combined,
        condition,
        reverse=reverse,
    )
    return combined


def _limit_on_state_drawdown(
    voltage: np.ndarray,
    log_current: np.ndarray,
    condition: GenerationCondition,
    *,
    reverse: bool,
) -> np.ndarray:
    sign = 1.0 if condition.polarity == "n-type" else -1.0
    effective_vth = _target_transfer_vth(condition, reverse=reverse)
    u = sign * (voltage - effective_vth)
    order = np.argsort(u)
    ordered_log = log_current[order].copy()
    on_current_threshold = np.log10(_effective_on_current(condition)) - 3.0
    on_candidates = np.flatnonzero(ordered_log >= on_current_threshold)
    if on_candidates.size == 0:
        return log_current
    on_indices = np.arange(on_candidates[0], ordered_log.size)
    if on_indices.size < 2:
        return log_current
    on_values = ordered_log[on_indices]
    running_peak = np.maximum.accumulate(on_values)
    maximum_drawdown_decades = 0.03
    ordered_log[on_indices] = np.maximum(
        on_values,
        running_peak - maximum_drawdown_decades,
    )
    constrained = log_current.copy()
    constrained[order] = ordered_log
    return constrained


def _generate_sweep(
    voltage: np.ndarray,
    condition: GenerationCondition,
    residual_engine: ResidualEngine,
    *,
    seed: int,
    reverse: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], list[float]]:
    physics_log = _physics_log_current(voltage, condition, reverse=reverse)
    normalized = (
        2.0 * (voltage - condition.voltage_min) / (condition.voltage_max - condition.voltage_min)
        - 1.0
    )
    residual_sample = residual_engine.sample(
        normalized,
        seed=seed + (7919 if reverse else 0),
        diversity=condition.diversity,
        sweep_phase=0.7 if reverse else 0.0,
        condition=condition,
        reverse=reverse,
    )
    final_log = _project_residual(
        voltage,
        physics_log,
        residual_sample.values,
        condition,
        reverse=reverse,
    )
    final_log = _limit_on_state_drawdown(
        voltage,
        final_log,
        condition,
        reverse=reverse,
    )

    rng = np.random.default_rng(seed + 104729)
    log_ioff = np.log10(condition.target_ioff)
    log_ion = np.log10(_effective_on_current(condition))
    slack = 0.45 * (1.0 - condition.physical_strictness) + 0.05
    lower_bound = np.log10(max(condition.target_ioff, np.finfo(float).tiny))
    final_log = np.clip(final_log, lower_bound - slack, log_ion + slack)
    if condition.physical_strictness > 0:
        sign = 1.0 if condition.polarity == "n-type" else -1.0
        order = np.argsort(sign * voltage)
        ordered_log = final_log[order]
        monotonic = np.maximum.accumulate(ordered_log)
        blend = condition.physical_strictness**2
        ordered_log = (1.0 - blend) * ordered_log + blend * monotonic
        edge_count = min(max(5, int(round(ordered_log.size * 0.10))), ordered_log.size)
        off_error = log_ioff - float(np.median(ordered_log[:edge_count]))
        on_error = log_ion - float(np.max(ordered_log[-edge_count:]))
        position = np.linspace(0.0, 1.0, ordered_log.size)
        smooth_position = position * position * (3.0 - 2.0 * position)
        ordered_log += condition.physical_strictness * (
            (1.0 - smooth_position) * off_error + smooth_position * on_error
        )
        ordered_log = (1.0 - blend) * ordered_log + blend * np.maximum.accumulate(ordered_log)
        final_log[order] = np.clip(ordered_log, lower_bound - slack, log_ion + slack)
    final_current = np.power(10.0, final_log)
    final_current = _apply_current_domain_noise(final_current, condition, rng)
    final_current = _quantize_current(final_current, condition.quantization_step_a)
    gate_source, gate_drain = _gate_leakage_components(voltage, condition, reverse=reverse)
    gate_baseline = np.clip(gate_source + gate_drain, np.finfo(float).tiny, None)
    if (
        residual_sample.gate_values is not None
        and condition.gate_ai_residual_strength > 0
    ):
        gate_log = np.log10(gate_baseline)
        gate_delta = np.clip(residual_sample.gate_values, -4.0, 4.0)
        gate_baseline = np.power(
            10.0,
            gate_log + condition.gate_ai_residual_strength * gate_delta,
        )
    gate_measured = _apply_current_domain_noise(
        gate_baseline,
        condition,
        rng,
    )
    gate_current = _quantize_current(
        gate_measured,
        condition.quantization_step_a,
    )
    return (
        np.power(10.0, physics_log),
        final_current,
        gate_current,
        residual_sample.latent_code,
        (
            residual_sample.latent_code
            if residual_sample.gate_values is not None
            else []
        ),
    )


def _condition_for_variant(condition: GenerationCondition, seed: int) -> GenerationCondition:
    if (
        condition.ion_sigma_fraction <= 0
        and condition.ioff_sigma_fraction <= 0
        and condition.vth_sigma_v <= 0
        and condition.ss_sigma_fraction <= 0
        and condition.hysteresis_sigma_v <= 0
        and condition.mobility_sigma_fraction <= 0
        and condition.contact_resistance_sigma_fraction <= 0
    ):
        return condition
    rng = np.random.default_rng(seed + 3571)
    ion_multiplier = float(
        np.exp(
            rng.normal(
                -0.5 * condition.ion_sigma_fraction**2,
                condition.ion_sigma_fraction,
            )
        )
    )
    ioff_multiplier = float(
        np.exp(
            rng.normal(
                -0.5 * condition.ioff_sigma_fraction**2,
                condition.ioff_sigma_fraction,
            )
        )
    )
    next_ion = max(condition.target_ion * ion_multiplier, np.finfo(float).tiny)
    next_ioff = min(
        max(condition.target_ioff * ioff_multiplier, np.finfo(float).tiny),
        next_ion * 0.99,
    )
    next_vth = float(
        np.clip(
            condition.target_vth + rng.normal(0.0, condition.vth_sigma_v),
            condition.voltage_min,
            condition.voltage_max,
        )
    )
    ss_multiplier = float(
        np.exp(
            rng.normal(
                -0.5 * condition.ss_sigma_fraction**2,
                condition.ss_sigma_fraction,
            )
        )
    )
    next_ss = float(
        np.clip(
            condition.target_ss_mv_dec * ss_multiplier,
            20.0,
            5000.0,
        )
    )
    next_hysteresis = float(
        np.clip(
            condition.hysteresis_v + rng.normal(0.0, condition.hysteresis_sigma_v),
            0.0,
            condition.voltage_max - condition.voltage_min,
        )
    )
    mobility_multiplier = float(
        np.exp(
            rng.normal(
                -0.5 * condition.mobility_sigma_fraction**2,
                condition.mobility_sigma_fraction,
            )
        )
    )
    resistance_multiplier = float(
        np.exp(
            rng.normal(
                -0.5 * condition.contact_resistance_sigma_fraction**2,
                condition.contact_resistance_sigma_fraction,
            )
        )
    )
    return condition.model_copy(
        update={
            "target_ion": next_ion,
            "target_ioff": next_ioff,
            "target_vth": next_vth,
            "target_ss_mv_dec": next_ss,
            "hysteresis_v": next_hysteresis,
            "mobility_cm2_vs": max(
                condition.mobility_cm2_vs * mobility_multiplier,
                np.finfo(float).tiny,
            ),
            "contact_resistance_ohm": max(
                condition.contact_resistance_ohm * resistance_multiplier,
                0.0,
            ),
        }
    )


def _normalized_ekv_current(
    gate_voltage: float,
    drain_magnitude: np.ndarray,
    condition: GenerationCondition,
) -> np.ndarray:
    polarity_sign = 1.0 if condition.polarity == "n-type" else -1.0
    vds = np.maximum(np.asarray(drain_magnitude, dtype=float), 1e-12)
    dibl_v = 0.04 * vds
    internal_vth = condition.target_vth + polarity_sign * _linear_extrapolation_vth_offset(
        condition
    )
    u = polarity_sign * (gate_voltage - internal_vth) + dibl_v
    raw = _normalized_transfer_charge(u, vds, condition)
    reference = _on_state_reference_charge(
        max(condition.ss_region_width_v, float(np.max(u))),
        vds,
        condition,
    )
    return np.minimum(raw / reference, 1.0)


def _physical_output_current(
    gate_voltage: float,
    drain_magnitude: np.ndarray,
    condition: GenerationCondition,
) -> np.ndarray:
    polarity_sign = 1.0 if condition.polarity == "n-type" else -1.0
    internal_vth = condition.target_vth + polarity_sign * _linear_extrapolation_vth_offset(
        condition
    )
    reference_gate = internal_vth + polarity_sign * condition.ss_region_width_v
    reference_vds = max(abs(condition.vd), 1e-9)
    dibl_v = 0.04 * np.maximum(np.asarray(drain_magnitude, dtype=float), 1e-12)
    u = polarity_sign * (gate_voltage - internal_vth) + dibl_v
    reference_u = polarity_sign * (reference_gate - internal_vth) + 0.04 * reference_vds
    raw = _normalized_transfer_charge(u, drain_magnitude, condition)
    on_reference = float(_normalized_transfer_charge(reference_u, reference_vds, condition))
    on_reference = max(on_reference, np.finfo(float).tiny)
    mobility_scale = condition.mobility_cm2_vs / 20.0
    channel_span = max(condition.target_ion - condition.target_ioff, np.finfo(float).tiny)
    intrinsic = (
        condition.target_ioff
        + mobility_scale
        * channel_span
        * raw
        / on_reference
    )
    intrinsic[drain_magnitude <= 0] = 0.0
    return _contact_limited_current(
        intrinsic,
        drain_magnitude,
        condition.contact_resistance_ohm,
    )


def _generate_output_curves(
    voltage: np.ndarray,
    transfer_current: np.ndarray,
    condition: GenerationCondition,
    *,
    seed: int,
) -> tuple[np.ndarray, list[OutputCurve]]:
    drain_sign = 1.0 if condition.vd >= 0 else -1.0
    drain_limit = max(abs(condition.vd), 1.0)
    drain_magnitude = np.linspace(0.0, drain_limit, 121)
    drain_voltage = drain_sign * drain_magnitude
    polarity_sign = 1.0 if condition.polarity == "n-type" else -1.0
    gate_half_span = max(
        condition.ss_region_width_v,
        condition.target_ss_mv_dec / 1000.0,
    )
    off_gate = condition.target_vth - polarity_sign * gate_half_span
    on_gate = condition.target_vth + polarity_sign * gate_half_span
    gate_voltages = np.linspace(off_gate, on_gate, 5)
    gate_voltages = np.clip(
        gate_voltages,
        condition.voltage_min,
        condition.voltage_max,
    )
    voltage_order = np.argsort(voltage)
    sorted_voltage = voltage[voltage_order]
    sorted_current = transfer_current[voltage_order]
    output: list[OutputCurve] = []
    rng = np.random.default_rng(seed + 524287)
    reference_voltage = abs(condition.vd) if abs(condition.vd) > 1e-9 else drain_limit
    reference_index = int(np.argmin(np.abs(drain_magnitude - reference_voltage)))
    previous_terminal_current = 0.0
    for gate_voltage in gate_voltages:
        terminal_current = float(np.interp(gate_voltage, sorted_voltage, sorted_current))
        terminal_current = max(terminal_current, previous_terminal_current)
        previous_terminal_current = terminal_current
        physical_output = _physical_output_current(
            float(gate_voltage),
            drain_magnitude,
            condition,
        )
        physical_reference = max(
            float(physical_output[reference_index]),
            np.finfo(float).tiny,
        )
        normalized = physical_output / physical_reference
        output_current = terminal_current * normalized
        output_current = _apply_output_domain_noise(output_current, condition, rng)
        if condition.quantization_step_a > 0:
            output_current = np.rint(
                output_current / condition.quantization_step_a
            ) * condition.quantization_step_a
            output_current = np.maximum(output_current, condition.quantization_step_a)
        output_current[0] = 0.0
        output.append(
            OutputCurve(
                gate_voltage=float(gate_voltage),
                current=output_current.tolist(),
            )
        )
    return drain_voltage, output


def _close_sweep_endpoints(
    forward: np.ndarray,
    reverse: np.ndarray,
    quantization_step: float,
    polarity: str,
    physical_strictness: float,
) -> tuple[np.ndarray, np.ndarray]:
    forward_log = np.log10(np.clip(forward, np.finfo(float).tiny, None))
    reverse_log = np.log10(np.clip(reverse, np.finfo(float).tiny, None))
    common_log = 0.5 * (forward_log + reverse_log)
    size = forward.size
    edge_count = min(max(5, int(round(size * 0.08))), max(size // 3, 1))
    edge_position = np.linspace(1.0, 0.0, edge_count)
    edge_weight = edge_position * edge_position * (3.0 - 2.0 * edge_position)
    weight = np.zeros(size)
    weight[:edge_count] = edge_weight
    weight[-edge_count:] = edge_weight[::-1]
    closed_forward = 10.0 ** ((1.0 - weight) * forward_log + weight * common_log)
    closed_reverse = 10.0 ** ((1.0 - weight) * reverse_log + weight * common_log)
    closed_forward = _quantize_current(closed_forward, quantization_step)
    closed_reverse = _quantize_current(closed_reverse, quantization_step)
    common_start = _quantize_current(
        np.asarray([10.0 ** common_log[0]]),
        quantization_step,
    )[0]
    common_end = _quantize_current(
        np.asarray([10.0 ** common_log[-1]]),
        quantization_step,
    )[0]
    closed_forward[0] = common_start
    closed_reverse[0] = common_start
    closed_forward[-1] = common_end
    closed_reverse[-1] = common_end
    if physical_strictness > 0:
        order = np.arange(size) if polarity == "n-type" else np.arange(size - 1, -1, -1)
        ordered_forward = np.maximum.accumulate(closed_forward[order])
        ordered_reverse = np.maximum.accumulate(closed_reverse[order])
        off_common = min(ordered_forward[0], ordered_reverse[0])
        on_common = max(ordered_forward[-1], ordered_reverse[-1])
        ordered_forward[0] = off_common
        ordered_reverse[0] = off_common
        ordered_forward[-1] = on_common
        ordered_reverse[-1] = on_common
        closed_forward[order] = ordered_forward
        closed_reverse[order] = ordered_reverse
    return closed_forward, closed_reverse


def _relative_error(measured: float | None, target: float) -> float:
    if measured is None or not np.isfinite(measured):
        return float("inf")
    return abs(measured - target) / max(abs(target), np.finfo(float).eps)


def _constraint_results(condition: GenerationCondition, features) -> list[ConstraintResult]:
    hysteresis = features.hysteresis_v
    checks = [
        ("Ion", _effective_on_current(condition), features.ion, 0.20, "relative"),
        ("Ioff", condition.target_ioff, features.ioff, 0.50, "relative"),
        (
            "Vth",
            condition.target_vth,
            features.vth,
            max(0.25, condition.hysteresis_v * 0.2),
            "absolute",
        ),
        ("SS", condition.target_ss_mv_dec, features.ss_mv_dec, 0.25, "relative"),
        (
            "Hysteresis",
            condition.hysteresis_v,
            hysteresis,
            max(0.25, condition.hysteresis_v * 0.25),
            "absolute",
        ),
    ]
    output: list[ConstraintResult] = []
    for name, target, measured, tolerance, tolerance_kind in checks:
        if tolerance_kind == "absolute":
            error = (
                abs(measured - target) / tolerance
                if measured is not None and tolerance > 0
                else float("inf")
            )
        else:
            error = _relative_error(measured, target) / tolerance
        passed = error <= 1.0
        output.append(
            ConstraintResult(
                name=name,
                target=target,
                measured=measured,
                tolerance=tolerance,
                tolerance_kind=tolerance_kind,
                normalized_error=error if np.isfinite(error) else None,
                passed=bool(passed),
            )
        )
    return output


def _quality_score(constraints: list[ConstraintResult], ss_r2: float | None) -> float:
    constraint_scores = [
        max(0.0, 1.0 - (item.normalized_error or 0.0)) if item.normalized_error is not None else 0.0
        for item in constraints
    ]
    constraint_quality = sum(constraint_scores) / len(constraint_scores)
    fit_quality = max(0.0, min(1.0, ss_r2 if ss_r2 is not None else 0.0))
    return round(0.75 * constraint_quality + 0.25 * fit_quality, 3)


def generate_curves(
    condition: GenerationCondition,
    residual_engine: ResidualEngine | None = None,
) -> GenerationResponse:
    engine = residual_engine or ResidualEngine(discover_default=False)
    voltage = np.linspace(condition.voltage_min, condition.voltage_max, condition.points)
    candidates: list[GeneratedCandidate] = []
    for index in range(condition.variants):
        seed = condition.seed + index
        variant_condition = _condition_for_variant(condition, seed)
        (
            physics_forward,
            forward,
            gate_forward,
            latent_code,
            gate_latent_code,
        ) = _generate_sweep(
            voltage, variant_condition, engine, seed=seed, reverse=False
        )
        physics_reverse, reverse, gate_reverse, _, _ = _generate_sweep(
            voltage, variant_condition, engine, seed=seed, reverse=True
        )
        forward, reverse = _close_sweep_endpoints(
            forward,
            reverse,
            variant_condition.quantization_step_a,
            variant_condition.polarity,
            variant_condition.physical_strictness,
        )
        forward_log = _limit_on_state_drawdown(
            voltage,
            np.log10(np.clip(forward, np.finfo(float).tiny, None)),
            variant_condition,
            reverse=False,
        )
        reverse_log = _limit_on_state_drawdown(
            voltage,
            np.log10(np.clip(reverse, np.finfo(float).tiny, None)),
            variant_condition,
            reverse=True,
        )
        forward_log = _stabilize_on_state_log_current(
            voltage,
            forward_log,
            variant_condition,
            reverse=False,
        )
        reverse_log = _stabilize_on_state_log_current(
            voltage,
            reverse_log,
            variant_condition,
            reverse=True,
        )
        forward = _quantize_current(
            10.0**forward_log,
            variant_condition.quantization_step_a,
        )
        reverse = _quantize_current(
            10.0**reverse_log,
            variant_condition.quantization_step_a,
        )
        if variant_condition.polarity == "n-type":
            off_common = min(forward[0], reverse[0])
            on_common = max(forward[-1], reverse[-1])
            forward[0] = reverse[0] = off_common
            forward[-1] = reverse[-1] = on_common
        else:
            on_common = max(forward[0], reverse[0])
            off_common = min(forward[-1], reverse[-1])
            forward[0] = reverse[0] = on_common
            forward[-1] = reverse[-1] = off_common
        forward, reverse = _close_sweep_endpoints(
            forward,
            reverse,
            variant_condition.quantization_step_a,
            variant_condition.polarity,
            variant_condition.physical_strictness,
        )
        forward, reverse = _close_sweep_endpoints(
            forward,
            reverse,
            variant_condition.quantization_step_a,
            variant_condition.polarity,
            variant_condition.physical_strictness,
        )
        forward_log = _stabilize_on_state_log_current(
            voltage,
            np.log10(np.clip(forward, np.finfo(float).tiny, None)),
            variant_condition,
            reverse=False,
        )
        reverse_log = _stabilize_on_state_log_current(
            voltage,
            np.log10(np.clip(reverse, np.finfo(float).tiny, None)),
            variant_condition,
            reverse=True,
        )
        forward_log = _limit_on_state_drawdown(
            voltage,
            forward_log,
            variant_condition,
            reverse=False,
        )
        reverse_log = _limit_on_state_drawdown(
            voltage,
            reverse_log,
            variant_condition,
            reverse=True,
        )
        forward = _quantize_current(
            10.0**forward_log,
            variant_condition.quantization_step_a,
        )
        reverse = _quantize_current(
            10.0**reverse_log,
            variant_condition.quantization_step_a,
        )
        if variant_condition.polarity == "n-type":
            off_common = min(forward[0], reverse[0])
            on_common = max(forward[-1], reverse[-1])
            forward[0] = reverse[0] = off_common
            forward[-1] = reverse[-1] = on_common
        else:
            on_common = max(forward[0], reverse[0])
            off_common = min(forward[-1], reverse[-1])
            forward[0] = reverse[0] = on_common
            forward[-1] = reverse[-1] = off_common
        forward_features = analyze_transfer_curve(
            voltage, forward, polarity=variant_condition.polarity
        )
        reverse_features = analyze_transfer_curve(
            voltage, reverse, polarity=variant_condition.polarity
        )
        features = combine_sweep_features(forward_features, reverse_features)
        constraints = _constraint_results(variant_condition, features)
        output_drain_voltage, output_curves = _generate_output_curves(
            voltage,
            forward,
            variant_condition,
            seed=seed,
        )
        candidates.append(
            GeneratedCandidate(
                candidate_id=index + 1,
                seed=seed,
                voltage=voltage.tolist(),
                forward_current=forward.tolist(),
                reverse_current=reverse.tolist(),
                gate_forward_current=gate_forward.tolist(),
                gate_reverse_current=gate_reverse.tolist(),
                physics_forward_current=physics_forward.tolist(),
                physics_reverse_current=physics_reverse.tolist(),
                output_drain_voltage=output_drain_voltage.tolist(),
                output_curves=output_curves,
                mobility_cm2_vs=variant_condition.mobility_cm2_vs,
                contact_resistance_ohm=variant_condition.contact_resistance_ohm,
                latent_code=latent_code,
                gate_latent_code=gate_latent_code,
                features=features,
                quality_score=_quality_score(constraints, features.ss_fit_r2),
                constraints=constraints,
            )
        )
    return GenerationResponse(
        condition=condition,
        candidates=candidates,
        residual_mode=engine.mode,
        model_name=engine.model_name,
    )
