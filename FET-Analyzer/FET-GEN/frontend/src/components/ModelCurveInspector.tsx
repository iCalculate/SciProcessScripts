import { CheckCircle2, FlaskConical, RefreshCw, Sparkles } from "lucide-react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";
import type { Data, Layout } from "plotly.js";
import { useCallback, useEffect, useMemo, useState } from "react";
import { generateCurves } from "../api";
import type {
  GeneratedCandidate,
  GenerationCondition,
  GenerationResponse,
  ModelInfo
} from "../types";

const Plot = createPlotlyComponent(Plotly);

const PREVIEW_CONDITION: GenerationCondition = {
  curve_type: "transfer",
  material: "MoS2",
  polarity: "n-type",
  vd: 1,
  target_ion: 1e-5,
  target_ioff: 1e-13,
  target_vth: 0,
  target_ss_mv_dec: 230,
  ss_region_width_v: 0.5,
  hysteresis_v: 1.5,
  noise_sigma_a: 0,
  noise_floor_a: 0,
  quantization_step_a: 0,
  output_noise_gain: 0,
  gate_leakage_a: 1e-14,
  gate_leakage_v_char: 0.7,
  gate_leakage_exponent: 0.8,
  ion_sigma_fraction: 0.04,
  ioff_sigma_fraction: 0.08,
  vth_sigma_v: 0.08,
  ss_sigma_fraction: 0.05,
  hysteresis_sigma_v: 0.05,
  mobility_cm2_vs: 20,
  mobility_sigma_fraction: 0.05,
  contact_resistance_ohm: 1e4,
  contact_resistance_sigma_fraction: 0.08,
  ai_residual_strength: 1,
  gate_ai_residual_strength: 1,
  physical_strictness: 0.45,
  diversity: 0.8,
  seed: 12345,
  voltage_min: -20,
  voltage_max: 20,
  points: 401,
  variants: 4
};

function InspectorField({
  label,
  value,
  step,
  onChange
}: {
  label: string;
  value: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <input
        type="number"
        value={value}
        step={step}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  );
}

function finitePositiveRange(values: number[][], fallback: [number, number]) {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = 0;
  for (const series of values) {
    for (const value of series) {
      if (!Number.isFinite(value) || value <= 0) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
  }
  if (!Number.isFinite(minimum) || maximum <= 0) return fallback;
  return [Math.log10(minimum) - 0.3, Math.log10(maximum) + 0.3] as [
    number,
    number
  ];
}

function buildTraces(
  response: GenerationResponse,
  selected: GeneratedCandidate
): Data[] {
  const otherTraces = response.candidates
    .filter((candidate) => candidate.candidate_id !== selected.candidate_id)
    .flatMap(
      (candidate) =>
        [
          {
            x: candidate.voltage,
            y: candidate.forward_current,
            type: "scatter",
            mode: "lines",
            xaxis: "x",
            yaxis: "y",
            hoverinfo: "skip",
            line: { color: "rgba(23, 105, 255, 0.13)", width: 1 },
            showlegend: false
          },
          {
            x: candidate.voltage,
            y: candidate.gate_forward_current,
            type: "scatter",
            mode: "lines",
            xaxis: "x2",
            yaxis: "y2",
            hoverinfo: "skip",
            line: { color: "rgba(217, 95, 2, 0.14)", width: 1 },
            showlegend: false
          }
        ] as Data[]
    );
  return [
    ...otherTraces,
    {
      x: selected.voltage,
      y: selected.physics_forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Ids physics",
      line: { color: "#7d8ca0", width: 1.3, dash: "dash" }
    },
    {
      x: selected.voltage,
      y: selected.forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Ids forward",
      line: { color: "#1769ff", width: 2.4 }
    },
    {
      x: selected.voltage,
      y: selected.reverse_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Ids reverse",
      line: { color: "#1769ff", width: 1.9, dash: "dot" }
    },
    {
      x: selected.voltage,
      y: selected.gate_forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x2",
      yaxis: "y2",
      name: "Ig forward",
      line: { color: "#d95f02", width: 2.2 }
    },
    {
      x: selected.voltage,
      y: selected.gate_reverse_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x2",
      yaxis: "y2",
      name: "Ig reverse",
      line: { color: "#d95f02", width: 1.8, dash: "dot" }
    }
  ];
}

export function ModelCurveInspector({
  model,
  disabled
}: {
  model: ModelInfo | null;
  disabled: boolean;
}) {
  const [condition, setCondition] =
    useState<GenerationCondition>(PREVIEW_CONDITION);
  const [response, setResponse] = useState<GenerationResponse | null>(null);
  const [selectedId, setSelectedId] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runPreview = useCallback(async (next: GenerationCondition) => {
    setLoading(true);
    setError(null);
    try {
      const generated = await generateCurves(next);
      setResponse(generated);
      setSelectedId(1);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Curve preview failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const supportsGate = model?.generated_channels.includes("Ig") ?? false;
    const next = {
      ...PREVIEW_CONDITION,
      gate_ai_residual_strength: supportsGate ? 1 : 0
    };
    setCondition(next);
    void runPreview(next);
  }, [model?.checkpoint_path, model?.generated_channels, runPreview]);

  const selected = useMemo(
    () =>
      response?.candidates.find(
        (candidate) => candidate.candidate_id === selectedId
      ) ??
      response?.candidates[0] ??
      null,
    [response, selectedId]
  );
  const traces = useMemo(
    () => (response && selected ? buildTraces(response, selected) : []),
    [response, selected]
  );
  const idsRange = selected
    ? finitePositiveRange(
        [selected.forward_current, selected.reverse_current],
        [-15, -4]
      )
    : [-15, -4];
  const igRange = selected
    ? finitePositiveRange(
        [selected.gate_forward_current, selected.gate_reverse_current],
        [-16, -10]
      )
    : [-16, -10];
  const passedConstraints =
    selected?.constraints.filter((constraint) => constraint.passed).length ?? 0;

  const layout: Partial<Layout> = {
    autosize: true,
    height: 510,
    margin: { l: 68, r: 24, t: 34, b: 52 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    legend: {
      orientation: "h",
      x: 0,
      y: 1.08,
      font: { size: 9 }
    },
    xaxis: {
      domain: [0, 1],
      anchor: "y",
      showticklabels: false,
      gridcolor: "#e8edf4",
      zeroline: false
    },
    yaxis: {
      domain: [0.57, 1],
      type: "log",
      range: idsRange,
      title: { text: "|Ids| (A)" },
      gridcolor: "#e8edf4",
      zeroline: false
    },
    xaxis2: {
      domain: [0, 1],
      anchor: "y2",
      title: { text: "Gate voltage Vg (V)" },
      gridcolor: "#e8edf4",
      zeroline: false
    },
    yaxis2: {
      domain: [0, 0.4],
      type: "log",
      range: igRange,
      title: { text: "|Ig| (A)" },
      gridcolor: "#e8edf4",
      zeroline: false
    },
    hovermode: "x unified"
  };

  function patchCondition(patch: Partial<GenerationCondition>) {
    setCondition((current) => ({ ...current, ...patch }));
  }

  return (
    <section className="model-curve-inspector">
      <div className="panel-title-row">
        <div>
          <h2>Final-model curve inspection</h2>
          <p>
            Generate concrete Ids and Ig transfer curves from the active checkpoint
            before accepting the training result.
          </p>
        </div>
        <div className="model-inspector-badges">
          <span>
            <FlaskConical size={13} />
            {response?.model_name ?? model?.model_name ?? "No active model"}
          </span>
          <span>
            Channels {model?.generated_channels.join(" + ") ?? "Ids"}
          </span>
        </div>
      </div>

      <div className="model-inspector-grid">
        <aside className="model-inspector-controls">
          <InspectorField
            label="Target Ion (A)"
            value={condition.target_ion}
            step={1e-6}
            onChange={(target_ion) => patchCondition({ target_ion })}
          />
          <InspectorField
            label="Target Ioff (A)"
            value={condition.target_ioff}
            step={1e-14}
            onChange={(target_ioff) => patchCondition({ target_ioff })}
          />
          <InspectorField
            label="Vth (V)"
            value={condition.target_vth}
            step={0.1}
            onChange={(target_vth) => patchCondition({ target_vth })}
          />
          <InspectorField
            label="SS (mV/dec)"
            value={condition.target_ss_mv_dec}
            step={10}
            onChange={(target_ss_mv_dec) =>
              patchCondition({ target_ss_mv_dec })
            }
          />
          <InspectorField
            label="Ig baseline (A)"
            value={condition.gate_leakage_a}
            step={1e-14}
            onChange={(gate_leakage_a) => patchCondition({ gate_leakage_a })}
          />
          <InspectorField
            label="Seed"
            value={condition.seed}
            step={1}
            onChange={(seed) => patchCondition({ seed: Math.max(0, Math.round(seed)) })}
          />
          <label className="model-inspector-slider">
            <span>Ids learned residual</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={condition.ai_residual_strength}
              onChange={(event) =>
                patchCondition({
                  ai_residual_strength: Number(event.currentTarget.value)
                })
              }
            />
            <strong>{condition.ai_residual_strength.toFixed(2)}</strong>
          </label>
          <label className="model-inspector-slider">
            <span>Ig learned residual</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={condition.gate_ai_residual_strength}
              disabled={!model?.generated_channels.includes("Ig")}
              onChange={(event) =>
                patchCondition({
                  gate_ai_residual_strength: Number(event.currentTarget.value)
                })
              }
            />
            <strong>{condition.gate_ai_residual_strength.toFixed(2)}</strong>
          </label>
          <button
            type="button"
            className="button primary"
            disabled={disabled || loading}
            onClick={() => void runPreview(condition)}
          >
            {loading ? (
              <RefreshCw size={15} className="spin" />
            ) : (
              <Sparkles size={15} />
            )}
            Generate inspection curves
          </button>
          {error ? <div className="model-inspector-error">{error}</div> : null}
        </aside>

        <div className="model-inspector-plot">
          {selected ? (
            <Plot
              data={traces}
              layout={layout}
              config={{
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ["lasso2d", "select2d"]
              }}
              useResizeHandler
              style={{ width: "100%", height: "510px" }}
            />
          ) : (
            <div className="model-inspector-empty">
              {loading ? "Generating curves" : "No inspection curves available"}
            </div>
          )}
        </div>
      </div>

      {response && selected ? (
        <div className="model-inspector-summary">
          <div className="model-inspector-candidates">
            {response.candidates.map((candidate) => (
              <button
                type="button"
                key={candidate.candidate_id}
                className={candidate.candidate_id === selectedId ? "active" : undefined}
                onClick={() => setSelectedId(candidate.candidate_id)}
              >
                #{candidate.candidate_id}
                <small>Q {candidate.quality_score.toFixed(3)}</small>
              </button>
            ))}
          </div>
          <dl>
            <div>
              <dt>Ion</dt>
              <dd>{selected.features.ion.toExponential(3)} A</dd>
            </div>
            <div>
              <dt>Ioff</dt>
              <dd>{selected.features.ioff.toExponential(3)} A</dd>
            </div>
            <div>
              <dt>Vth</dt>
              <dd>
                {selected.features.vth === null
                  ? "n/a"
                  : `${selected.features.vth.toFixed(3)} V`}
              </dd>
            </div>
            <div>
              <dt>SS</dt>
              <dd>
                {selected.features.ss_mv_dec === null
                  ? "n/a"
                  : `${selected.features.ss_mv_dec.toFixed(1)} mV/dec`}
              </dd>
            </div>
            <div>
              <dt>Constraints</dt>
              <dd>
                <CheckCircle2 size={13} />
                {passedConstraints}/{selected.constraints.length}
              </dd>
            </div>
          </dl>
        </div>
      ) : null}
    </section>
  );
}
