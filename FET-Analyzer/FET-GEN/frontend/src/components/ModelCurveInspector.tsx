import { CheckCircle2, FlaskConical, RefreshCw, Sparkles } from "lucide-react";
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";
import type { Data, Layout } from "plotly.js";
import { useCallback, useEffect, useMemo, useState } from "react";
import { compareModels } from "../api";
import type {
  GenerationCondition,
  ModelComparisonItem,
  ModelComparisonResponse,
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
  ai_residual_strength: 1.0,
  gate_ai_residual_strength: 1.0,
  diversity: 0.8,
  seed: 12345,
  voltage_min: -20,
  voltage_max: 20,
  points: 401,
  variants: 1
};

const PROFILE_COLORS: Record<string, string> = {
  physics_only: "#7d8ca0",
  procedural_prior: "#d95f02",
  active_model: "#1769ff",
  best_jump_model: "#d1495b",
  best_canonical_model: "#6d28d9",
  best_weighted_model: "#1b8f6a"
};

function profileColor(key: string) {
  return PROFILE_COLORS[key] ?? "#1769ff";
}

function residualLabel(item: ModelComparisonItem) {
  if (item.model.architecture === "hybrid_threshold_pca") return "Hybrid threshold PCA";
  if (item.model.architecture === "conditional_pca") return "Conditioned PCA";
  return item.residual_mode === "conditional_vae"
    ? "Conditional VAE"
    : item.residual_mode === "learned_pca"
      ? "Learned PCA"
      : "Procedural prior";
}

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
  comparison: ModelComparisonResponse,
  focusedKey: string
): Data[] {
  const traces: Data[] = [];
  const focused = comparison.items.find((item) => item.key === focusedKey) ?? comparison.items[0];
  if (focused) {
    traces.push({
      x: focused.candidate.voltage,
      y: focused.candidate.physics_forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Physics baseline",
      line: { color: "#97a5b7", width: 1.3, dash: "dash" }
    });
  }
  comparison.items.forEach((item) => {
    const color = profileColor(item.key);
    const emphasized = item.key === focusedKey;
    traces.push(
      {
        x: item.candidate.voltage,
        y: item.candidate.forward_current,
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y",
        name: `${item.label} Ids`,
        legendgroup: item.key,
        line: {
          color,
          width: emphasized ? 3.1 : 2.0
        }
      },
      {
        x: item.candidate.voltage,
        y: item.candidate.reverse_current,
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y",
        showlegend: false,
        legendgroup: item.key,
        hoverinfo: "skip",
        line: {
          color,
          width: emphasized ? 2.4 : 1.5,
          dash: "dot"
        }
      },
      {
        x: item.candidate.voltage,
        y: item.candidate.gate_forward_current,
        type: "scatter",
        mode: "lines",
        xaxis: "x2",
        yaxis: "y2",
        name: `${item.label} Ig`,
        legendgroup: `${item.key}-ig`,
        line: {
          color,
          width: emphasized ? 2.6 : 1.8
        }
      },
      {
        x: item.candidate.voltage,
        y: item.candidate.gate_reverse_current,
        type: "scatter",
        mode: "lines",
        xaxis: "x2",
        yaxis: "y2",
        showlegend: false,
        legendgroup: `${item.key}-ig`,
        hoverinfo: "skip",
        line: {
          color,
          width: emphasized ? 2.1 : 1.3,
          dash: "dot"
        }
      }
    );
  });
  return traces;
}

function metricLabel(value: number | null) {
  return value === null || !Number.isFinite(value) ? "n/a" : value.toFixed(3);
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
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(null);
  const [focusedKey, setFocusedKey] = useState("active_model");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const gateChannelKey = model?.generated_channels.join(",") ?? "";

  const runComparison = useCallback(async (next: GenerationCondition) => {
    setLoading(true);
    setError(null);
    try {
      const request = {
        ...next,
        variants: 1,
        gate_ai_residual_strength: next.ai_residual_strength
      };
      const generated = await compareModels(request);
      setComparison(generated);
      setFocusedKey((current) =>
        generated.items.some((item) => item.key === current) ? current : "active_model"
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Model comparison failed");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const supportsGate = model?.generated_channels.includes("Ig") ?? false;
    const next = {
      ...PREVIEW_CONDITION,
      gate_ai_residual_strength: supportsGate ? PREVIEW_CONDITION.ai_residual_strength : 0
    };
    setCondition(next);
    void runComparison(next);
  }, [gateChannelKey, model?.checkpoint_path, runComparison]);

  const focused = useMemo(
    () =>
      comparison?.items.find((item) => item.key === focusedKey) ??
      comparison?.items.find((item) => item.key === "active_model") ??
      comparison?.items[0] ??
      null,
    [comparison, focusedKey]
  );

  const traces = useMemo(
    () => (comparison ? buildTraces(comparison, focused?.key ?? "active_model") : []),
    [comparison, focused?.key]
  );

  const idsRange = focused
    ? finitePositiveRange(
        comparison?.items.flatMap((item) => [
          item.candidate.forward_current,
          item.candidate.reverse_current
        ]) ?? [],
        [-15, -4]
      )
    : [-15, -4];
  const igRange = focused
    ? finitePositiveRange(
        comparison?.items.flatMap((item) => [
          item.candidate.gate_forward_current,
          item.candidate.gate_reverse_current
        ]) ?? [],
        [-16, -10]
      )
    : [-16, -10];

  const layout: Partial<Layout> = {
    autosize: true,
    height: 540,
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
    setCondition((current) => {
      const next = { ...current, ...patch };
      if (patch.ai_residual_strength !== undefined) {
        next.gate_ai_residual_strength = patch.ai_residual_strength;
      }
      return next;
    });
  }

  return (
    <section className="model-curve-inspector">
      <div className="panel-title-row">
        <div>
          <h2>Model comparison lab</h2>
          <p>
            Compare pure physics, the active model, and the best leaderboard
            checkpoints under one shared physics-to-AI balance slider. The
            default preview opens at 100% AI to expose threshold-jump behavior
            directly.
          </p>
        </div>
        <div className="model-inspector-badges">
          <span>
            <FlaskConical size={13} />
            {model?.model_name ?? "No active model"}
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
            onChange={(target_ss_mv_dec) => patchCondition({ target_ss_mv_dec })}
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
          <label className="model-inspector-slider model-inspector-slider-wide">
            <span>Physics {"<->"} AI balance</span>
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
            <strong>{Math.round(condition.ai_residual_strength * 100)}%</strong>
            <small>0 = pure formula, 100 = strongest learned morphology blend.</small>
          </label>
          <button
            type="button"
            className="button primary"
            disabled={disabled || loading}
            onClick={() => void runComparison(condition)}
          >
            {loading ? (
              <RefreshCw size={15} className="spin" />
            ) : (
              <Sparkles size={15} />
            )}
            Generate comparison
          </button>
          {error ? <div className="model-inspector-error">{error}</div> : null}
        </aside>

        <div className="model-inspector-plot">
          {comparison && focused ? (
            <Plot
              data={traces}
              layout={layout}
              config={{
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ["lasso2d", "select2d"]
              }}
              useResizeHandler
              style={{ width: "100%", height: "540px" }}
            />
          ) : (
            <div className="model-inspector-empty">
              {loading ? "Generating model comparison" : "No model comparison available"}
            </div>
          )}
        </div>
      </div>

      {comparison ? (
        <div className="model-comparison-cards">
          {comparison.items.map((item) => {
            const passedConstraints = item.candidate.constraints.filter(
              (constraint) => constraint.passed
            ).length;
            const active = focused?.key === item.key;
            const experiment = item.experiment_summary;
            return (
              <button
                type="button"
                key={item.key}
                className={`model-comparison-card${active ? " active" : ""}`}
                onClick={() => setFocusedKey(item.key)}
              >
                <div className="model-comparison-card-top">
                  <span
                    className="model-comparison-swatch"
                    style={{ backgroundColor: profileColor(item.key) }}
                  />
                  <div>
                    <strong>{item.label}</strong>
                    <small>{residualLabel(item)}</small>
                  </div>
                </div>
                <p>{item.description}</p>
                <dl>
                  <div>
                    <dt>Quality</dt>
                    <dd>{item.candidate.quality_score.toFixed(3)}</dd>
                  </div>
                  <div>
                    <dt>Blend</dt>
                    <dd>{Math.round(item.ai_residual_strength * 100)}%</dd>
                  </div>
                  <div>
                    <dt>Vth</dt>
                    <dd>{metricLabel(item.candidate.features.vth)} V</dd>
                  </div>
                  <div>
                    <dt>SS</dt>
                    <dd>{metricLabel(item.candidate.features.ss_mv_dec)} mV/dec</dd>
                  </div>
                  <div>
                    <dt>Ioff</dt>
                    <dd>{item.candidate.features.ioff.toExponential(2)} A</dd>
                  </div>
                  <div>
                    <dt>Constraints</dt>
                    <dd>
                      <CheckCircle2 size={13} />
                      {passedConstraints}/{item.candidate.constraints.length}
                    </dd>
                  </div>
                  {experiment ? (
                    <>
                      <div>
                        <dt>Jump P95</dt>
                        <dd>{metricLabel(experiment.jump_p95_decades)}</dd>
                      </div>
                      <div>
                        <dt>Canon. jump</dt>
                        <dd>{metricLabel(experiment.canonical_jump_max_decades)}</dd>
                      </div>
                      <div>
                        <dt>Gen. Vth</dt>
                        <dd>{metricLabel(experiment.generated_vth_mae_v)} V</dd>
                      </div>
                      <div>
                        <dt>Gen. SS</dt>
                        <dd>{metricLabel(experiment.generated_ss_mae_mv_dec)} mV/dec</dd>
                      </div>
                      <div>
                        <dt>Weighted</dt>
                        <dd>{metricLabel(experiment.validation_weighted_rmse_decades)}</dd>
                      </div>
                    </>
                  ) : null}
                </dl>
              </button>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}
