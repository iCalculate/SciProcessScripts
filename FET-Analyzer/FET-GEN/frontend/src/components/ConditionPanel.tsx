import { Dices, Info } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { GenerationCondition } from "../types";

interface ConditionPanelProps {
  condition: GenerationCondition;
  disabled: boolean;
  onChange: (patch: Partial<GenerationCondition>) => void;
}

interface NumberFieldProps {
  label: string;
  value: number;
  unit: string;
  step: number;
  min?: number;
  max?: number;
  integer?: boolean;
  compact?: boolean;
  onCommit: (value: number) => void;
}

const ENGINEERING_PREFIXES: Record<string, number> = {
  f: 1e-15,
  p: 1e-12,
  n: 1e-9,
  u: 1e-6,
  "µ": 1e-6,
  m: 1e-3,
  k: 1e3,
  K: 1e3,
  M: 1e6,
  G: 1e9
};

function parseEngineeringNumber(input: string): number | null {
  const trimmed = input.trim();
  if (!trimmed) return null;
  const match = trimmed.match(
    /^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([fpnuµmkKMG]?)(?:[a-zA-Zµ]*)?$/u
  );
  if (!match) return null;
  const value = Number(match[1]);
  if (!Number.isFinite(value)) return null;
  return value * (ENGINEERING_PREFIXES[match[2]] ?? 1);
}

function formatNumber(value: number): string {
  if (value === 0) return "0";
  const absValue = Math.abs(value);
  if (absValue < 1e-3 || absValue >= 1e4) return value.toExponential(3);
  return Number(value.toPrecision(8)).toString();
}

export function NumberField({
  label,
  value,
  unit,
  step,
  min,
  max,
  integer = false,
  compact = false,
  onCommit
}: NumberFieldProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [draft, setDraft] = useState(() => formatNumber(value));

  useEffect(() => {
    if (document.activeElement !== inputRef.current) setDraft(formatNumber(value));
  }, [value]);

  function reset() {
    setDraft(formatNumber(value));
  }

  function commit() {
    const parsed = parseEngineeringNumber(draft);
    if (
      parsed === null ||
      (min !== undefined && parsed < min) ||
      (max !== undefined && parsed > max)
    ) {
      reset();
      return;
    }
    const nextValue = integer ? Math.round(parsed) : parsed;
    setDraft(formatNumber(nextValue));
    onCommit(nextValue);
  }

  return (
    <label className={`field-row${compact ? " compact-number-field" : ""}`}>
      <span className="field-label">
        {label}
        <Info size={12} />
      </span>
      <span className="input-with-unit">
        <input
          ref={inputRef}
          type="text"
          inputMode="decimal"
          value={draft}
          data-step={step}
          onChange={(event) => setDraft(event.currentTarget.value)}
          onBlur={reset}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commit();
            } else if (event.key === "Escape") {
              event.preventDefault();
              reset();
            }
          }}
        />
        <span>{unit}</span>
      </span>
    </label>
  );
}

function MaterialField({
  value,
  onCommit
}: {
  value: string;
  onCommit: (value: string) => void;
}) {
  const [draft, setDraft] = useState(value);
  useEffect(() => setDraft(value), [value]);
  return (
    <label className="field-row">
      <span className="field-label">
        Material <Info size={12} />
      </span>
      <input
        className="text-input"
        value={draft}
        onChange={(event) => setDraft(event.currentTarget.value)}
        onBlur={() => setDraft(value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            const nextValue = draft.trim();
            if (nextValue) onCommit(nextValue);
            else setDraft(value);
          } else if (event.key === "Escape") {
            setDraft(value);
          }
        }}
      />
    </label>
  );
}

interface SliderFieldProps {
  label: string;
  value: number;
  left: string;
  right: string;
  percent?: boolean;
  onChange: (value: number) => void;
}

function SliderField({
  label,
  value,
  left,
  right,
  percent = false,
  onChange
}: SliderFieldProps) {
  return (
    <div className="slider-field">
      <div className="slider-heading">
        <span>
          {label} <Info size={12} />
        </span>
        <strong>{percent ? `${Math.round(value * 100)}%` : value.toFixed(2)}</strong>
      </div>
      <input
        aria-label={label}
        type="range"
        min="0"
        max="1"
        step="0.01"
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <div className="slider-ends">
        <span>{left}</span>
        <span>{right}</span>
      </div>
    </div>
  );
}

export function ConditionPanel({
  condition,
  disabled,
  onChange
}: ConditionPanelProps) {
  return (
    <aside className="side-panel conditions-panel" aria-busy={disabled}>
      <div className="panel-title">Conditions</div>
      <fieldset className="panel-scroll panel-fieldset" disabled={disabled}>
        <MaterialField value={condition.material} onCommit={(material) => onChange({ material })} />
        <label className="field-row">
          <span className="field-label">
            Polarity <Info size={12} />
          </span>
          <select
            value={condition.polarity}
            onChange={(event) =>
              onChange({
                polarity: event.target.value as GenerationCondition["polarity"]
              })
            }
          >
            <option value="n-type">n-type</option>
            <option value="p-type">p-type</option>
          </select>
        </label>
        <NumberField label="Drain voltage Vd" value={condition.vd} unit="V" step={0.1} onCommit={(vd) => onChange({ vd })} />

        <div className="parameter-pair-row">
          <NumberField compact label="Mobility μ" value={condition.mobility_cm2_vs} unit="cm²/V·s" step={1} min={0.01} onCommit={(mobility_cm2_vs) => onChange({ mobility_cm2_vs })} />
          <NumberField compact label="Mobility σ" value={condition.mobility_sigma_fraction * 100} unit="%" step={1} min={0} max={100} onCommit={(value) => onChange({ mobility_sigma_fraction: value / 100 })} />
        </div>
        <div className="parameter-pair-row">
          <NumberField compact label="Contact resistance" value={condition.contact_resistance_ohm / 1000} unit="kΩ" step={1} min={0} onCommit={(value) => onChange({ contact_resistance_ohm: value * 1000 })} />
          <NumberField compact label="Resistance σ" value={condition.contact_resistance_sigma_fraction * 100} unit="%" step={1} min={0} max={100} onCommit={(value) => onChange({ contact_resistance_sigma_fraction: value / 100 })} />
        </div>
        <div className="parameter-pair-row">
          <NumberField compact label="Target Ion" value={condition.target_ion} unit="A" step={1e-6} min={0} onCommit={(target_ion) => onChange({ target_ion })} />
          <NumberField compact label="Ion σ" value={condition.ion_sigma_fraction * 100} unit="%" step={1} min={0} max={100} onCommit={(value) => onChange({ ion_sigma_fraction: value / 100 })} />
        </div>
        <div className="parameter-pair-row">
          <NumberField compact label="Target Ioff" value={condition.target_ioff} unit="A" step={1e-15} min={0} onCommit={(target_ioff) => onChange({ target_ioff })} />
          <NumberField compact label="Ioff σ" value={condition.ioff_sigma_fraction * 100} unit="%" step={1} min={0} max={100} onCommit={(value) => onChange({ ioff_sigma_fraction: value / 100 })} />
        </div>
        <div className="parameter-pair-row">
          <NumberField compact label="Threshold Vth" value={condition.target_vth} unit="V" step={0.1} onCommit={(target_vth) => onChange({ target_vth })} />
          <NumberField compact label="Vth σ" value={condition.vth_sigma_v} unit="V" step={0.01} min={0} onCommit={(vth_sigma_v) => onChange({ vth_sigma_v })} />
        </div>
        <div className="parameter-pair-row">
          <NumberField compact label="SS" value={condition.target_ss_mv_dec} unit="mV/dec" step={5} min={20} onCommit={(target_ss_mv_dec) => onChange({ target_ss_mv_dec })} />
          <NumberField compact label="SS σ" value={condition.ss_sigma_fraction * 100} unit="%" step={1} min={0} max={100} onCommit={(value) => onChange({ ss_sigma_fraction: value / 100 })} />
        </div>
        <div className="parameter-pair-row">
          <NumberField compact label="Hysteresis" value={condition.hysteresis_v} unit="V" step={0.1} min={0} onCommit={(hysteresis_v) => onChange({ hysteresis_v })} />
          <NumberField compact label="Hysteresis σ" value={condition.hysteresis_sigma_v} unit="V" step={0.01} min={0} onCommit={(hysteresis_sigma_v) => onChange({ hysteresis_sigma_v })} />
        </div>

        <SliderField label="Diversity" value={condition.diversity} left="Low" right="High" onChange={(diversity) => onChange({ diversity })} />
        <SliderField label="AI residual strength" value={condition.ai_residual_strength} left="Physics" right="Learned residual" percent onChange={(ai_residual_strength) => onChange({ ai_residual_strength })} />
        <SliderField label="Ig residual strength" value={condition.gate_ai_residual_strength} left="Analytical Ig" right="Learned Ig" percent onChange={(gate_ai_residual_strength) => onChange({ gate_ai_residual_strength })} />
        <SliderField label="Physical strictness" value={condition.physical_strictness} left="Relaxed" right="Strict" onChange={(physical_strictness) => onChange({ physical_strictness })} />

        <details className="parameter-section">
          <summary>Parameters</summary>
          <div className="parameter-section-body">
            <div className="section-label compact-section-label">Measurement</div>
            <NumberField label="Current noise σ" value={condition.noise_sigma_a} unit="A" step={1e-13} min={0} onCommit={(noise_sigma_a) => onChange({ noise_sigma_a })} />
            <NumberField label="Read noise σ" value={condition.noise_floor_a} unit="A" step={1e-13} min={0} onCommit={(noise_floor_a) => onChange({ noise_floor_a })} />
            <NumberField label="Current resolution" value={condition.quantization_step_a} unit="A/LSB" step={1e-15} min={0} onCommit={(quantization_step_a) => onChange({ quantization_step_a })} />
            <NumberField label="Output noise gain" value={condition.output_noise_gain} unit="x" step={0.5} min={0} max={50} onCommit={(output_noise_gain) => onChange({ output_noise_gain })} />
            <NumberField label="SS region width" value={condition.ss_region_width_v} unit="V" step={0.1} min={0.01} onCommit={(ss_region_width_v) => onChange({ ss_region_width_v })} />
            <NumberField label="Gate leakage I0" value={condition.gate_leakage_a} unit="A" step={1e-13} min={0} onCommit={(gate_leakage_a) => onChange({ gate_leakage_a })} />
            <NumberField label="Gate leakage Vchar" value={condition.gate_leakage_v_char} unit="V" step={0.05} min={0.01} onCommit={(gate_leakage_v_char) => onChange({ gate_leakage_v_char })} />
            <NumberField label="Gate leakage exponent" value={condition.gate_leakage_exponent} unit="" step={0.05} min={0.01} max={3} onCommit={(gate_leakage_exponent) => onChange({ gate_leakage_exponent })} />

            <div className="section-label compact-section-label">Randomization</div>
            <div className="seed-field">
              <NumberField label="Random seed" value={condition.seed} unit="" step={1} min={0} integer onCommit={(seed) => onChange({ seed })} />
              <button
                type="button"
                aria-label="Randomize seed"
                onClick={() => onChange({ seed: Math.floor(Math.random() * 1_000_000) })}
              >
                <Dices size={16} />
              </button>
            </div>
          </div>
        </details>
      </fieldset>
    </aside>
  );
}
