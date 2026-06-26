import { CheckCircle2, Download, RefreshCw, XCircle } from "lucide-react";
import {
  candidateExportHref,
  candidateFilename,
  fixed,
  scientific
} from "../format";
import type { GeneratedCandidate, GenerationResponse } from "../types";

interface ResultsPanelProps {
  candidate: GeneratedCandidate;
  response: GenerationResponse;
  loading: boolean;
  regenerateDisabled: boolean;
  onRegenerate: () => void;
}

export function ResultsPanel({
  candidate,
  response,
  loading,
  regenerateDisabled,
  onRegenerate
}: ResultsPanelProps) {
  const features = candidate.features;
  const parameters = [
    ["Ion", scientific(features.ion), "A"],
    ["Ioff", scientific(features.ioff), "A"],
    ["Ion / Ioff", scientific(features.ion_ioff_ratio), ""],
    ["Vth", fixed(features.vth, 3), "V"],
    ["SS", fixed(features.ss_mv_dec, 1), "mV/dec"],
    ["Hysteresis", fixed(features.hysteresis_v, 3), "V"],
    ["Mobility", fixed(candidate.mobility_cm2_vs, 2), "cm²/V·s"],
    ["Contact resistance", fixed(candidate.contact_resistance_ohm / 1000, 2), "kΩ"],
    ["Noise", fixed(features.noise_log_sigma, 4), "dec"],
    ["Ambipolar", fixed(features.ambipolar_strength, 4), ""]
  ];

  return (
    <aside className="side-panel results-panel">
      <div className="panel-title">Results</div>
      <div className="panel-scroll">
        <details className="results-parameter-section">
          <summary>Generated parameters</summary>
          <div className="measurement-context">
            Measured @ Vd = {response.condition.vd.toFixed(2)} V
          </div>
          <dl className="parameter-list">
            {parameters.map(([label, value, unit]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>
                  {value} <span>{unit}</span>
                </dd>
              </div>
            ))}
            <div className="quality-row">
              <dt>Quality score</dt>
              <dd>{candidate.quality_score.toFixed(2)} / 1.00</dd>
            </div>
          </dl>
        </details>

        <div className="section-label">Constraint status</div>
        <div className="constraint-list">
          {candidate.constraints.map((constraint) => (
            <div key={constraint.name}>
              <span>{constraint.name}</span>
              <span className={constraint.passed ? "constraint-pass" : "constraint-fail"}>
                {constraint.measured === null
                  ? "—"
                  : constraint.name === "Ion" || constraint.name === "Ioff"
                    ? scientific(constraint.measured)
                    : fixed(constraint.measured, 2)}
                {constraint.passed ? <CheckCircle2 size={15} /> : <XCircle size={15} />}
              </span>
            </div>
          ))}
        </div>

        <div className="action-stack">
          <button
            className="button primary"
            onClick={onRegenerate}
            disabled={loading || regenerateDisabled}
          >
            <RefreshCw size={16} className={loading ? "spin" : ""} />
            {loading ? "Generating…" : "Regenerate"}
          </button>
          <a
            className="button secondary"
            href={candidateExportHref(candidate, response.condition)}
            download={candidateFilename(candidate)}
          >
            <Download size={16} />
            Download CSV
          </a>
        </div>

        <details className="results-parameter-section model-section">
          <summary>Model</summary>
          <dl className="model-facts">
            <div>
              <dt>Name</dt>
              <dd>{response.model_name}</dd>
            </div>
            <div>
              <dt>Residual mode</dt>
              <dd>
                {response.residual_mode === "conditional_vae"
                  ? "Conditional VAE"
                  : response.residual_mode === "learned_pca"
                    ? "Learned PCA"
                    : "Procedural prior"}
              </dd>
            </div>
            <div>
              <dt>Blend</dt>
              <dd>{Math.round(response.condition.ai_residual_strength * 100)}% AI residual</dd>
            </div>
            <div>
              <dt>Latent dimensions</dt>
              <dd>{candidate.latent_code.length}</dd>
            </div>
          </dl>
        </details>
      </div>
    </aside>
  );
}
