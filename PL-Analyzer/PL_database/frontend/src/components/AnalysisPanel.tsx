import { useState } from "react";
import Plot from "react-plotly.js";
import type { AnalysisOptions, BatchAnalysisResponse, SpectrumDetail } from "../types";
import { buildPreviewAxis, formatPreviewAxisSummary } from "../utils/spectrumPreview";

interface AnalysisPanelProps {
  analysisResponse: BatchAnalysisResponse | null;
  selectedSpectra: SpectrumDetail[];
  onRun: (options: AnalysisOptions, saveResults: boolean) => Promise<void>;
}

const DEFAULT_OPTIONS: AnalysisOptions = {
  baseline_order: 3,
  baseline_quantile: 0.25,
  smoothing_window: 11,
  smoothing_polyorder: 3,
  normalization: "max",
  prominence: 0.05,
  height: null,
  distance: null,
  fit_model: "gaussian",
  max_peaks: 4,
  spectrum_family: "auto",
  material_hint: null,
  method_version: "material-aware-v2",
  min_material_confidence: 0.28
};

export function AnalysisPanel(props: AnalysisPanelProps) {
  const [options, setOptions] = useState<AnalysisOptions>(DEFAULT_OPTIONS);
  const [saveResults, setSaveResults] = useState(true);
  const [running, setRunning] = useState(false);

  const leadResult = props.analysisResponse?.results[0];
  const leadSpectrum = leadResult
    ? props.selectedSpectra.find((spectrum) => spectrum.spectrum_id === leadResult.spectrum_id) ?? props.selectedSpectra[0] ?? null
    : null;
  const leadPreview = leadSpectrum ? buildPreviewAxis(leadSpectrum) : null;
  const analysisXAxis =
    leadResult && leadPreview && leadPreview.values.length === leadResult.analysis.x_axis.length
      ? leadPreview.values
      : leadResult?.analysis.x_axis ?? [];

  async function handleRun() {
    setRunning(true);
    try {
      await props.onRun(options, saveResults);
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="panel-grid">
      <div className="card card-span-1">
        <div className="card-head">
          <div>
            <p className="eyebrow">Pipeline controls</p>
            <h2>Processing options</h2>
          </div>
        </div>
        <div className="form-grid">
          <NumericField
            label="Baseline order"
            value={options.baseline_order}
            onChange={(value) => setOptions({ ...options, baseline_order: value })}
          />
          <NumericField
            label="Baseline quantile"
            value={options.baseline_quantile}
            step="0.05"
            onChange={(value) => setOptions({ ...options, baseline_quantile: value })}
          />
          <NumericField
            label="Savgol window"
            value={options.smoothing_window}
            onChange={(value) => setOptions({ ...options, smoothing_window: value })}
          />
          <NumericField
            label="Peak prominence"
            value={options.prominence}
            step="0.01"
            onChange={(value) => setOptions({ ...options, prominence: value })}
          />
          <label className="field">
            <span>Normalization</span>
            <select
              value={options.normalization}
              onChange={(event) =>
                setOptions({ ...options, normalization: event.target.value as AnalysisOptions["normalization"] })
              }
            >
              <option value="none">none</option>
              <option value="max">max</option>
              <option value="area">area</option>
              <option value="zscore">zscore</option>
            </select>
          </label>
          <label className="field">
            <span>Fit model</span>
            <select
              value={options.fit_model}
              onChange={(event) =>
                setOptions({ ...options, fit_model: event.target.value as AnalysisOptions["fit_model"] })
              }
            >
              <option value="gaussian">gaussian</option>
              <option value="lorentzian">lorentzian</option>
            </select>
          </label>
          <NumericField
            label="Max peaks"
            value={options.max_peaks}
            onChange={(value) => setOptions({ ...options, max_peaks: value })}
          />
          <label className="toggle-row">
            <input
              type="checkbox"
              checked={saveResults}
              onChange={(event) => setSaveResults(event.target.checked)}
            />
            <span>Save analysis results to SQLite</span>
          </label>
          <button className="primary-button" disabled={running || props.selectedSpectra.length === 0} onClick={handleRun} type="button">
            {running ? "Running..." : "Run analysis"}
          </button>
        </div>
      </div>
      <div className="card card-span-2">
        <div className="card-head">
          <div>
            <p className="eyebrow">Processed trace</p>
            <h2>Raw vs processed signal</h2>
          </div>
        </div>
        {leadResult ? (
          <>
            {leadPreview?.mode === "raman_shift" ? (
              <p className="helper-copy helper-copy-tight">{formatPreviewAxisSummary(leadPreview)}</p>
            ) : null}
            <Plot
              className="plot-frame"
              data={[
                {
                  x: analysisXAxis,
                  y: leadResult.analysis.raw_intensity,
                  type: "scatter",
                  mode: "lines",
                  name: "Raw"
                },
                {
                  x: analysisXAxis,
                  y: leadResult.analysis.normalized_intensity,
                  type: "scatter",
                  mode: "lines",
                  name: "Normalized"
                },
                {
                  x: analysisXAxis,
                  y: leadResult.analysis.fit.fit_curve,
                  type: "scatter",
                  mode: "lines",
                  name: "Fit"
                }
              ]}
              layout={{
                autosize: true,
                paper_bgcolor: "rgba(0,0,0,0)",
                plot_bgcolor: "rgba(0,0,0,0)",
                margin: { t: 24, r: 24, b: 56, l: 56 },
                xaxis: { title: leadPreview?.axisTitle ?? "x_axis" },
                yaxis: { title: "Intensity (normalized)" }
              }}
              config={{ responsive: true, displaylogo: false }}
              useResizeHandler
            />
          </>
        ) : (
          <p className="empty-state">Run analysis on the selected spectra to populate this panel.</p>
        )}
      </div>
      <div className="card card-span-3">
        <div className="card-head">
          <div>
            <p className="eyebrow">Results</p>
            <h2>Peak summary table</h2>
          </div>
        </div>
        {props.analysisResponse ? (
          <div className="table-shell">
            <table>
              <thead>
                <tr>
                  <th>Spectrum ID</th>
                  <th>Integrated intensity</th>
                  <th>Peak max</th>
                  <th>SNR</th>
                  <th>Detected peaks</th>
                </tr>
              </thead>
              <tbody>
                {props.analysisResponse.summary.map((row) => (
                  <tr key={row.spectrum_id}>
                    <td>{row.spectrum_id}</td>
                    <td>{row.integrated_intensity.toFixed(4)}</td>
                    <td>{row.peak_max.toFixed(4)}</td>
                    <td>{row.signal_to_noise_ratio.toFixed(2)}</td>
                    <td>{row.n_detected_peaks}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="empty-state">Batch metrics will appear here after analysis completes.</p>
        )}
      </div>
    </section>
  );
}

interface NumericFieldProps {
  label: string;
  value: number;
  step?: string;
  onChange: (value: number) => void;
}

function NumericField(props: NumericFieldProps) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        type="number"
        step={props.step ?? "1"}
        value={props.value}
        onChange={(event) => props.onChange(Number(event.target.value))}
      />
    </label>
  );
}
