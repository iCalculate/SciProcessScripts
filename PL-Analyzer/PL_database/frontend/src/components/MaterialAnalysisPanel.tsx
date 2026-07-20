import { useEffect, useMemo, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { fetchMaterialAnalysisJob, fetchSpectrum, startMaterialAnalysis, stopMaterialAnalysisJob } from "../api";
import type { AnalysisOptions, MaterialAnalysisJob, SpectrumDetail, SpectrumFilters, SpectrumRow } from "../types";

interface MaterialAnalysisPanelProps {
  filters: SpectrumFilters;
  matchedCount: number;
  selectedIds: string[];
  selectedSpectra: SpectrumDetail[];
  visibleRows: SpectrumRow[];
  onCompleted: () => Promise<void>;
}

type RunScope = "selected" | "filtered" | "all";
type QueueItem = MaterialAnalysisJob["queue_window"][number];

const DEFAULT_OPTIONS: AnalysisOptions = {
  baseline_order: 3,
  baseline_quantile: 0.25,
  smoothing_window: 11,
  smoothing_polyorder: 3,
  normalization: "max",
  prominence: 0.035,
  height: null,
  distance: null,
  fit_model: "auto",
  max_peaks: 4,
  spectrum_family: "auto",
  material_hint: null,
  method_version: "material-aware-v2",
  min_material_confidence: 0.28
};

export function MaterialAnalysisPanel(props: MaterialAnalysisPanelProps) {
  const [options, setOptions] = useState<AnalysisOptions>(DEFAULT_OPTIONS);
  const [scope, setScope] = useState<RunScope>("selected");
  const [saveResults, setSaveResults] = useState(true);
  const [updateEntries, setUpdateEntries] = useState(true);
  const [job, setJob] = useState<MaterialAnalysisJob | null>(null);
  const [previewDetails, setPreviewDetails] = useState<Record<string, SpectrumDetail>>({});
  const [error, setError] = useState("");
  const logRef = useRef<HTMLPreElement | null>(null);
  const queueRef = useRef<HTMLDivElement | null>(null);

  const active = job?.status === "queued" || job?.status === "running";
  const latest = job?.latest_result ?? null;
  const quality = latest?.analysis.quality ?? null;
  const percent = job && job.total > 0 ? Math.round((job.processed / job.total) * 100) : 0;
  const runSize = scope === "selected" ? props.selectedIds.length : scope === "filtered" ? props.matchedCount : null;
  const previewIds = useMemo(() => buildPreviewIds(scope, props.selectedIds, props.visibleRows), [scope, props.selectedIds, props.visibleRows]);
  const initialQueue = useMemo(
    () => buildInitialQueue(previewIds, props.visibleRows, props.selectedSpectra, previewDetails),
    [previewIds, props.visibleRows, props.selectedSpectra, previewDetails]
  );
  const queueItems = job?.queue_window?.length ? job.queue_window : initialQueue;
  const logs = useMemo(() => buildDisplayLogs(job, error, previewIds.length, scope), [job, error, previewIds.length, scope]);
  const featureRows = useMemo(() => Object.entries(latest?.analysis.features ?? {}), [latest]);

  useEffect(() => {
    if (!job || !active) {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const next = await fetchMaterialAnalysisJob(job.job_id);
        setJob(next);
        if (next.status === "finished" || next.status === "stopped") {
          await props.onCompleted();
        }
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    }, 850);
    return () => window.clearInterval(timer);
  }, [active, job?.job_id]);

  useEffect(() => {
    const missingIds = previewIds.filter((id) => !previewDetails[id]).slice(0, 12);
    if (missingIds.length === 0 || active) {
      return;
    }
    let cancelled = false;
    void Promise.allSettled(missingIds.map((id) => fetchSpectrum(id))).then((results) => {
      if (cancelled) {
        return;
      }
      const next: Record<string, SpectrumDetail> = {};
      results.forEach((result) => {
        if (result.status === "fulfilled") {
          next[result.value.spectrum_id] = result.value;
          if (result.value.representative_spectrum_id) {
            next[result.value.representative_spectrum_id] = result.value;
          }
        }
      });
      if (Object.keys(next).length > 0) {
        setPreviewDetails((current) => ({ ...current, ...next }));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [active, previewIds.join("|")]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  useEffect(() => {
    const runningRow = queueRef.current?.querySelector("[data-running='true']");
    runningRow?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [job?.processed, job?.current_spectrum_id]);

  async function handleStart() {
    setError("");
    const { filters, search } = resolveScopeFilters(scope, props.filters);
    try {
      const nextJob = await startMaterialAnalysis({
        spectrum_ids: scope === "selected" ? props.selectedIds : [],
        filters,
        search,
        include_mock: false,
        options,
        save_results: saveResults,
        update_entries: updateEntries
      });
      setJob(nextJob);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleStop() {
    if (!job) {
      return;
    }
    setError("");
    await stopMaterialAnalysisJob(job.job_id);
    setJob(await fetchMaterialAnalysisJob(job.job_id));
  }

  return (
    <section className="analysis-console">
      <div className="analysis-run-band">
        <div className="card analysis-launch-card">
          <div className="analysis-launch-top">
            <div>
              <p className="eyebrow">Material-aware extraction</p>
              <h2>{job?.message ?? "Ready to classify spectra"}</h2>
            </div>
            <span className={`analysis-state analysis-state-${job?.status ?? "idle"}`}>{job?.status ?? "idle"}</span>
          </div>
          <div className="analysis-launch-controls">
            <div className="segmented-control" role="tablist" aria-label="Analysis scope">
              {(["selected", "filtered", "all"] as RunScope[]).map((item) => (
                <button className={scope === item ? "segmented-active" : ""} disabled={active} key={item} onClick={() => setScope(item)} type="button">
                  {scopeLabel(item)}
                </button>
              ))}
            </div>
            <div className="analysis-run-count">
              <strong>{runSize == null ? "All" : runSize}</strong>
              <span>{scope === "selected" ? "selected" : scope === "filtered" ? "matched" : "indexed"}</span>
            </div>
            <button className="primary-button analysis-run-button" disabled={active || (scope === "selected" && props.selectedIds.length === 0)} onClick={handleStart} type="button">
              {active ? "Running" : "Run"}
            </button>
            <button className="secondary-button analysis-stop-button" disabled={!active} onClick={handleStop} type="button">
              Stop
            </button>
          </div>
          <div className="analysis-progress-row">
            <div className="progress-shell">
              <div className="progress-fill" style={{ width: `${percent}%` }} />
            </div>
            <strong>{percent}%</strong>
          </div>
          <div className="analysis-kpi-grid analysis-kpi-grid-compact">
            <Kpi label="Processed" value={`${job?.processed ?? 0}/${job?.total ?? 0}`} />
            <Kpi label="Updated" value={String(job?.updated ?? 0)} />
            <Kpi label="Failed" value={String(job?.failed ?? 0)} />
            <Kpi label="Fit quality" value={quality?.score == null ? "-" : `${Math.round(quality.score * 100)}%`} tone={quality?.label ?? "idle"} />
          </div>
          <details className="analysis-settings-panel">
            <summary>Advanced settings</summary>
            <div className="form-grid">
              <SelectSetting disabled={active} label="Family" value={options.spectrum_family} options={["auto", "PL", "Raman"]} onChange={(value) => setOptions({ ...options, spectrum_family: value as AnalysisOptions["spectrum_family"] })} />
              <SelectSetting disabled={active} label="Peak profile" value={options.fit_model} options={["auto", "lorentzian", "gaussian", "pseudo_voigt"]} onChange={(value) => setOptions({ ...options, fit_model: value as AnalysisOptions["fit_model"] })} />
              <SelectSetting disabled={active} label="Material hint" value={options.material_hint ?? ""} options={["", "MoS2", "WSe2", "WS2", "MoSe2", "MoTe2", "WTe2"]} onChange={(value) => setOptions({ ...options, material_hint: value || null })} />
              <NumericField disabled={active} label="Prominence" step="0.005" value={options.prominence} onChange={(value) => setOptions({ ...options, prominence: value })} />
              <NumericField disabled={active} label="Savgol window" value={options.smoothing_window} onChange={(value) => setOptions({ ...options, smoothing_window: value })} />
              <NumericField disabled={active} label="Min confidence" step="0.05" value={options.min_material_confidence} onChange={(value) => setOptions({ ...options, min_material_confidence: value })} />
              <label className="toggle-row">
                <input checked={saveResults} disabled={active} onChange={(event) => setSaveResults(event.target.checked)} type="checkbox" />
                <span>Save full results</span>
              </label>
              <label className="toggle-row">
                <input checked={updateEntries} disabled={active} onChange={(event) => setUpdateEntries(event.target.checked)} type="checkbox" />
                <span>Update database entries</span>
              </label>
            </div>
          </details>
        </div>

        <div className="card analysis-log-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Log</p>
              <h2>{job?.current_spectrum_id ?? "Pipeline timeline"}</h2>
            </div>
          </div>
          <pre ref={logRef} className="log-box analysis-log-box">{logs}</pre>
          <div className="analysis-summary-strip">
            {Object.entries(job?.summary.materials ?? {}).slice(0, 8).map(([material, count]) => (
              <span key={material}>
                <strong>{material}</strong>
                {count}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="card analysis-queue-card">
        <div className="card-head">
          <div>
            <p className="eyebrow">Rolling database</p>
            <h2>{job ? "Processed and upcoming entries" : "Queued preview"}</h2>
          </div>
        </div>
        <div ref={queueRef} className="analysis-queue-list">
          {queueItems.length > 0 ? (
            queueItems.map((item) => (
              <div className={`analysis-queue-row analysis-queue-${item.status} quality-${item.fit_quality_label ?? "idle"}`} data-running={item.status === "running"} key={`${item.order}-${item.spectrum_id}`}>
                <div className="analysis-queue-index">{item.order}</div>
                <Sparkline values={item.sparkline} />
                <div className="analysis-queue-main">
                  <strong>{item.sample_id || item.spectrum_id}</strong>
                  <span>{item.source ?? item.spectrum_type ?? "-"} · {item.acquisition_mode ?? "-"}</span>
                </div>
                <div className="analysis-queue-fit">
                  <span>{item.material ?? item.status}</span>
                  <strong>{item.fit_quality == null ? "-" : `${Math.round(item.fit_quality * 100)}%`}</strong>
                </div>
              </div>
            ))
          ) : (
            <p className="empty-state">Select rows in Database, or use the filtered/current database scope.</p>
          )}
        </div>
      </div>

      <div className="analysis-fit-grid">
        <div className="card analysis-plot-card">
          <div className="card-head analysis-fit-head">
            <div>
              <p className="eyebrow">Latest fit</p>
              <h2>{latest ? `${latest.analysis.spectrum_family} ${latest.analysis.material}` : "Fit preview"}</h2>
            </div>
            {quality ? <span className={`quality-pill quality-${quality.label}`}>{quality.label}</span> : <span className="quality-pill quality-idle">waiting</span>}
          </div>
          {latest ? <FitPlot latest={latest} /> : <FitPlaceholder />}
        </div>

        <div className="card analysis-feature-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Extracted parameters</p>
              <h2>Latest feature set</h2>
            </div>
          </div>
          {featureRows.length > 0 ? (
            <div className="analysis-feature-grid">
              {featureRows.map(([key, value]) => (
                <div className="analysis-feature-chip" key={key}>
                  <span>{key}</span>
                  <strong>{formatFeatureValue(value)}</strong>
                </div>
              ))}
            </div>
          ) : (
            <div className="analysis-feature-grid">
              {["material", "family", "fit_quality", "E/G or A/B ratio", "center", "FWHM"].map((label) => (
                <div className="analysis-feature-chip analysis-feature-empty" key={label}>
                  <span>{label}</span>
                  <strong>-</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function FitPlot(props: { latest: NonNullable<MaterialAnalysisJob["latest_result"]> }) {
  const latest = props.latest;
  return (
    <Plot
      className="plot-frame analysis-fit-plot"
      data={[
        { x: latest.analysis.x_axis, y: latest.analysis.normalized_intensity, type: "scatter", mode: "lines", name: "Normalized", line: { width: 2.2, color: "#18354d" } },
        { x: latest.analysis.x_axis, y: latest.analysis.fit.fit_curve, type: "scatter", mode: "lines", name: "Constrained fit", line: { width: 2.8, color: "#e27a2e" } },
        ...latest.analysis.fit.peaks.map((peak) => ({
          x: latest.analysis.x_axis,
          y: peakCurve(latest.analysis.x_axis, peak, latest.analysis.fit.model),
          type: "scatter" as const,
          mode: "lines" as const,
          name: peak.label ?? "peak",
          line: { dash: "dot" as const, width: 1.6 }
        }))
      ]}
      layout={{
        autosize: true,
        uirevision: latest.spectrum_id,
        transition: { duration: 220, easing: "cubic-in-out" },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        margin: { t: 10, r: 18, b: 48, l: 54 },
        xaxis: { title: latest.analysis.axis?.unit ?? "analysis axis", zeroline: false, gridcolor: "rgba(17, 34, 59, 0.07)" },
        yaxis: { title: "Intensity (normalized)", zeroline: false, gridcolor: "rgba(17, 34, 59, 0.07)" },
        legend: { orientation: "h", y: 1.08, x: 0, font: { size: 11 } }
      }}
      config={{ responsive: true, displaylogo: false }}
      useResizeHandler
    />
  );
}

function FitPlaceholder() {
  return (
    <div className="analysis-fit-placeholder">
      <svg viewBox="0 0 720 260" aria-hidden="true">
        <path d="M20 214 C105 184 134 98 210 126 C285 154 304 56 376 74 C456 94 464 194 542 160 C610 130 642 96 700 110" />
        <path className="fit-placeholder-fit" d="M20 218 C112 190 132 112 210 132 C286 151 309 65 376 79 C456 96 469 185 542 155 C610 125 642 102 700 113" />
      </svg>
      <p className="empty-state">Run material analysis to stream constrained peak fits here.</p>
    </div>
  );
}

function buildPreviewIds(scope: RunScope, selectedIds: string[], visibleRows: SpectrumRow[]): string[] {
  if (scope === "selected" && selectedIds.length > 0) {
    return selectedIds.slice(0, 12);
  }
  return visibleRows.map((row) => row.representative_spectrum_id || row.spectrum_id).filter(Boolean).slice(0, 12);
}

function buildInitialQueue(
  ids: string[],
  visibleRows: SpectrumRow[],
  selectedSpectra: SpectrumDetail[],
  previewDetails: Record<string, SpectrumDetail>
): QueueItem[] {
  const rowsById = new Map<string, SpectrumRow>();
  visibleRows.forEach((row) => {
    rowsById.set(row.spectrum_id, row);
    if (row.representative_spectrum_id) {
      rowsById.set(row.representative_spectrum_id, row);
    }
  });
  const detailsById = new Map<string, SpectrumDetail>();
  selectedSpectra.forEach((detail) => {
    detailsById.set(detail.spectrum_id, detail);
    if (detail.representative_spectrum_id) {
      detailsById.set(detail.representative_spectrum_id, detail);
    }
  });
  Object.entries(previewDetails).forEach(([id, detail]) => detailsById.set(id, detail));

  return ids.map((id, index) => {
    const detail = detailsById.get(id);
    const row = rowsById.get(id) ?? detail;
    return {
      spectrum_id: id,
      order: index + 1,
      status: "pending",
      sample_id: row?.sample_id ?? null,
      source: row?.source ?? null,
      spectrum_type: row?.spectrum_type ?? null,
      acquisition_mode: row?.acquisition_mode ?? null,
      sparkline: detail ? normalizeSparkline(detail.intensity) : fallbackSparkline(index),
      material: row?.analysis_material ?? null,
      family: row?.analysis_family ?? null,
      fit_quality: readStoredQuality(row),
      fit_quality_label: readStoredQualityLabel(row),
      error: null
    };
  });
}

function readStoredQuality(row: SpectrumRow | undefined): number | null {
  const summary = row?.analysis_summary as { fit_quality?: { score?: unknown } } | undefined;
  const value = summary?.fit_quality?.score;
  return typeof value === "number" ? value : null;
}

function readStoredQualityLabel(row: SpectrumRow | undefined): string | null {
  const summary = row?.analysis_summary as { fit_quality?: { label?: unknown } } | undefined;
  return typeof summary?.fit_quality?.label === "string" ? summary.fit_quality.label : null;
}

function normalizeSparkline(values: number[]): number[] {
  if (values.length === 0) {
    return [];
  }
  const size = 48;
  const sampled = values.length > size
    ? Array.from({ length: size }, (_, index) => values[Math.min(values.length - 1, Math.floor((index / size) * values.length))])
    : values;
  const minimum = Math.min(...sampled);
  const maximum = Math.max(...sampled);
  const span = maximum - minimum;
  if (span <= 0) {
    return sampled.map(() => 0.5);
  }
  return sampled.map((value) => (value - minimum) / span);
}

function fallbackSparkline(seed: number): number[] {
  return Array.from({ length: 48 }, (_, index) => {
    const x = index / 47;
    return Math.max(0.04, Math.min(0.96, 0.2 + 0.58 * Math.exp(-((x - 0.35 - (seed % 4) * 0.06) ** 2) / 0.018) + 0.18 * Math.exp(-((x - 0.72) ** 2) / 0.03)));
  });
}

function buildDisplayLogs(job: MaterialAnalysisJob | null, error: string, previewCount: number, scope: RunScope): string {
  const lines = job?.logs?.length
    ? job.logs
    : [
        `[ready] Scope: ${scopeLabel(scope)} spectra`,
        `[ready] Preview queue loaded: ${previewCount} entries`,
        "[ready] Press Run to start material recognition and constrained peak fitting.",
      ];
  if (error) {
    return [...lines, `[error] ${error}`].join("\n");
  }
  return lines.join("\n");
}

function resolveScopeFilters(scope: RunScope, filters: SpectrumFilters): { filters: SpectrumFilters; search?: string } {
  if (scope === "all") {
    return { filters: {} };
  }
  const { search, ...rest } = filters;
  return { filters: rest, search };
}

function scopeLabel(scope: RunScope): string {
  if (scope === "selected") {
    return "Selected";
  }
  if (scope === "filtered") {
    return "Filtered";
  }
  return "All";
}

function peakCurve(xAxis: number[], peak: { amplitude: number; center: number; width: number }, model: string): number[] {
  return xAxis.map((x) => {
    const scaled = (x - peak.center) / Math.max(peak.width, Number.EPSILON);
    const gaussian = peak.amplitude * Math.exp(-4 * Math.log(2) * scaled ** 2);
    const lorentzian = peak.amplitude / (1 + 4 * scaled ** 2);
    if (model === "lorentzian") {
      return lorentzian;
    }
    if (model === "pseudo_voigt") {
      return 0.65 * gaussian + 0.35 * lorentzian;
    }
    return gaussian;
  });
}

function formatFeatureValue(value: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "-";
  }
  return Math.abs(value) >= 100 ? value.toFixed(2) : value.toFixed(4);
}

function Sparkline(props: { values: number[] }) {
  const points = props.values.length > 0
    ? props.values.map((value, index) => `${(index / Math.max(1, props.values.length - 1)) * 92 + 4},${34 - value * 28}`).join(" ")
    : "";
  return (
    <svg className="analysis-sparkline" viewBox="0 0 100 40" aria-hidden="true">
      <polyline points={points} />
    </svg>
  );
}

function Kpi(props: { label: string; value: string; tone?: string }) {
  return (
    <div className={`analysis-kpi quality-${props.tone ?? "idle"}`}>
      <span>{props.label}</span>
      <strong>{props.value}</strong>
    </div>
  );
}

function SelectSetting(props: { disabled: boolean; label: string; value: string; options: string[]; onChange: (value: string) => void }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <select disabled={props.disabled} value={props.value} onChange={(event) => props.onChange(event.target.value)}>
        {props.options.map((option) => (
          <option key={option || "auto"} value={option}>
            {option || "auto"}
          </option>
        ))}
      </select>
    </label>
  );
}

function NumericField(props: { disabled: boolean; label: string; step?: string; value: number; onChange: (value: number) => void }) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input disabled={props.disabled} step={props.step ?? "1"} type="number" value={props.value} onChange={(event) => props.onChange(Number(event.target.value))} />
    </label>
  );
}
