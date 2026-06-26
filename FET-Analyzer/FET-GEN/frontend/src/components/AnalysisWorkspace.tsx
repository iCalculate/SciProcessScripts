import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";
import type { Data, Layout, PlotMouseEvent, PlotSelectionEvent } from "plotly.js";
import {
  Activity,
  BarChart3,
  ChartLine,
  Database,
  Download,
  Eye,
  RefreshCw,
  ScatterChart,
  Sigma,
  SlidersHorizontal,
  Table2,
  Target
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  exportDatabaseSelection,
  getDatabaseAnalysisStatus,
  getDatabaseCurvePreviews,
  startDatabaseAnalysis
} from "../api";
import { fixed, scientific } from "../format";
import type {
  AnalysisDistributionBin,
  AnalysisMetricStats,
  AnalysisPcaPoint,
  AnalysisSample,
  CurvePreview,
  DatabaseAnalysisStatus,
  DatabaseAnalysisResponse,
  DatabaseSelectionState
} from "../types";

const Plot = createPlotlyComponent(Plotly);

const PREVIEW_CURVE_LIMIT = 32;
const ANALYSIS_SCATTER_RENDER_LIMIT = 1_600;
const ANALYSIS_PCA_CLUSTER_LIMIT = 900;
const EMPTY_PREVIEW = {
  title: "Preview",
  detail: "No active selection",
  ids: [] as string[]
};

const METRIC_LABELS: Record<string, string> = {
  ion: "Ion",
  ioff: "Ioff",
  logIon: "log10 Ion",
  logIoff: "log10 Ioff",
  logRatio: "logRatio",
  vth: "Vth",
  ss_mv_dec: "SS",
  gm_max: "Gm max",
  logGm: "log10 Gm",
  noise_log_sigma: "Noise",
  ambipolar_strength: "Ambipolar",
  hysteresis_v: "Hysteresis",
  rows_clean: "Rows",
  voltage_span: "V span"
};

const METRIC_OPTIONS = [
  "logIon",
  "logIoff",
  "logRatio",
  "vth",
  "ss_mv_dec",
  "logGm",
  "noise_log_sigma",
  "ambipolar_strength",
  "hysteresis_v",
  "rows_clean",
  "voltage_span"
];

const RAW_METRIC_OPTIONS = [
  "ion",
  "ioff",
  "logIon",
  "logIoff",
  "logRatio",
  "vth",
  "ss_mv_dec",
  "gm_max",
  "logGm",
  "noise_log_sigma",
  "ambipolar_strength",
  "hysteresis_v",
  "rows_clean",
  "voltage_span"
];

const PALETTE = [
  "#1769ff",
  "#079669",
  "#df7d10",
  "#8b5cf6",
  "#dc3f72",
  "#0f9ca6",
  "#6f5d4f",
  "#4f6b8a"
];

type PreviewRequest = typeof EMPTY_PREVIEW;
type ColorMode = "cluster" | "polarity" | "source_kind" | "direction";

function metricLabel(name: string): string {
  return METRIC_LABELS[name] ?? name;
}

function metricValue(name: string, value: number | null): string {
  if (value === null) return "-";
  if (name === "ion" || name === "ioff" || name === "gm_max") return scientific(value);
  if (name === "rows_clean") return fixed(value, 0);
  return fixed(value, name.startsWith("log") ? 2 : 3);
}

function sourceName(path: string): string {
  return path.split(/[\\/]/).at(-1) ?? path;
}

function sampleMetric(sample: AnalysisSample, metric: string): number | null {
  const value = sample[metric as keyof AnalysisSample];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function finiteMetricPairs(samples: AnalysisSample[], metric: string) {
  return samples.flatMap((sample) => {
    const value = sampleMetric(sample, metric);
    return value === null ? [] : [{ sample, value }];
  });
}

function histogramBins(values: number[], binCount = 24): AnalysisDistributionBin[] {
  if (values.length === 0) return [];
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    const padding = Math.max(Math.abs(min) * 0.05, 0.5);
    min -= padding;
    max += padding;
  }
  const width = (max - min) / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    start: min + index * width,
    end: min + (index + 1) * width,
    center: min + (index + 0.5) * width,
    count: 0
  }));
  values.forEach((value) => {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - min) / width)));
    bins[index].count += 1;
  });
  return bins;
}

function histogramByWidth(
  values: number[],
  requestedMin: number | null,
  requestedMax: number | null,
  requestedWidth: number | null
): AnalysisDistributionBin[] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) return [];
  const dataMin = Math.min(...finite);
  const dataMax = Math.max(...finite);
  const min = requestedMin ?? dataMin;
  const max = requestedMax ?? dataMax;
  if (!(max > min)) return histogramBins(finite);
  const fallbackWidth = (max - min) / 24;
  const width = requestedWidth && requestedWidth > 0 ? requestedWidth : fallbackWidth;
  const count = Math.max(1, Math.min(160, Math.ceil((max - min) / width)));
  const bins = Array.from({ length: count }, (_, index) => {
    const start = min + index * width;
    const end = Math.min(max, start + width);
    return { start, end, center: 0.5 * (start + end), count: 0 };
  });
  finite.forEach((value) => {
    if (value < min || value > max) return;
    const index = Math.min(count - 1, Math.floor((value - min) / width));
    bins[index].count += 1;
  });
  return bins;
}

function smoothTrendFromBins(bins: AnalysisDistributionBin[], windowRadius = 2) {
  if (bins.length === 0) return { x: [] as number[], y: [] as number[] };
  const x = bins.map((bin) => bin.center);
  const y = bins.map((_, index) => {
    let sum = 0;
    let weight = 0;
    for (let offset = -windowRadius; offset <= windowRadius; offset += 1) {
      const candidate = bins[index + offset];
      if (!candidate) continue;
      const w = windowRadius + 1 - Math.abs(offset);
      sum += candidate.count * w;
      weight += w;
    }
    return weight > 0 ? sum / weight : 0;
  });
  return { x, y };
}

function nextMetric(options: string[], current: string) {
  const index = options.indexOf(current);
  if (index < 0) return options[0] ?? current;
  return options[(index + 1) % options.length] ?? current;
}

function densityCurve(values: number[], min: number, max: number, countScale: number) {
  if (values.length < 2 || !Number.isFinite(min) || !Number.isFinite(max) || min === max) {
    return { x: [] as number[], y: [] as number[] };
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, values.length - 1);
  const std = Math.sqrt(Math.max(variance, Number.EPSILON));
  const bandwidth = Math.max(
    1.06 * std * values.length ** -0.2,
    (max - min) / 120,
    Number.EPSILON
  );
  const x = Array.from({ length: 96 }, (_, index) => min + ((max - min) * index) / 95);
  const raw = x.map((point) => {
    const kernelSum = values.reduce((sum, value) => {
      const z = (point - value) / bandwidth;
      return sum + Math.exp(-0.5 * z * z);
    }, 0);
    return kernelSum / (values.length * bandwidth * Math.sqrt(2 * Math.PI));
  });
  const maxDensity = Math.max(...raw, Number.EPSILON);
  return { x, y: raw.map((value) => (value / maxDensity) * countScale) };
}

function pearsonFor(samples: AnalysisSample[], xMetric: string, yMetric: string) {
  const points = samples.flatMap((sample) => {
    const x = sampleMetric(sample, xMetric);
    const y = sampleMetric(sample, yMetric);
    return x === null || y === null ? [] : [{ sample, x, y }];
  });
  if (points.length < 2) {
    return { points, r: null as number | null, r2: null as number | null, slope: null as number | null };
  }
  const meanX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const meanY = points.reduce((sum, point) => sum + point.y, 0) / points.length;
  let covariance = 0;
  let varianceX = 0;
  let varianceY = 0;
  points.forEach((point) => {
    const dx = point.x - meanX;
    const dy = point.y - meanY;
    covariance += dx * dy;
    varianceX += dx * dx;
    varianceY += dy * dy;
  });
  const denominator = Math.sqrt(varianceX * varianceY);
  const r = denominator > 0 ? covariance / denominator : null;
  return {
    points,
    r,
    r2: r === null ? null : r * r,
    slope: varianceX > 0 ? covariance / varianceX : null
  };
}

function colorFor(label: string): string {
  let hash = 0;
  for (const char of label) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return PALETTE[hash % PALETTE.length];
}

function plotConfig(selectable = false) {
  void selectable;
  return {
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
    editable: false,
    scrollZoom: false
  };
}

function sampleEvenly<T>(items: T[], limit: number): T[] {
  if (items.length <= limit) return items;
  const step = items.length / limit;
  return Array.from({ length: limit }, (_, index) => items[Math.min(items.length - 1, Math.floor(index * step))]);
}

function AnalysisLoadingPanel({
  selectionCount,
  status
}: {
  selectionCount: number;
  status: DatabaseAnalysisStatus | null;
}) {
  const progress = Math.max(0, Math.min(1, status?.progress_fraction ?? 0));
  const progressLabel = status?.stage ?? "idle";
  const progressDetail = status?.message ?? "Waiting for the analysis worker to start.";
  const elapsedSeconds = status?.elapsed_seconds ?? 0;

  return (
    <section className="analysis-loading-panel" aria-live="polite">
      <div className="analysis-loading-hero">
        <div className="analysis-math-loader" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div className="analysis-loading-copy">
          <strong>Analysis in progress</strong>
          <p>{progressDetail} Progress is reported by the backend from the active analysis job.</p>
        </div>
        <span>{Math.round(progress * 100)}%</span>
      </div>
      <div className="analysis-loading-track active-pulse">
        <div className="analysis-loading-fill" style={{ width: `${progress * 100}%` }} />
      </div>
      <div className="analysis-inline-stats">
        <span>Selected scope <strong>{selectionCount.toLocaleString()}</strong></span>
        <span>Stage <strong>{progressLabel}</strong></span>
        <span>Elapsed <strong>{elapsedSeconds.toFixed(1)}s</strong></span>
      </div>
    </section>
  );
}

function SummaryCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="analysis-summary-card">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{hint}</small>
    </div>
  );
}

function MetricRow({ name, stats }: { name: string; stats: AnalysisMetricStats }) {
  return (
    <tr>
      <th>{metricLabel(name)}</th>
      <td>{stats.count.toLocaleString()}</td>
      <td>{metricValue(name, stats.min)}</td>
      <td>{metricValue(name, stats.median)}</td>
      <td>{metricValue(name, stats.mean)}</td>
      <td>{metricValue(name, stats.max)}</td>
      <td>{metricValue(name, stats.std)}</td>
    </tr>
  );
}

function CountBars({ title, counts }: { title: string; counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, count]) => count));
  return (
    <section className="analysis-count-panel">
      <h2>{title}</h2>
      <div className="analysis-bars">
        {entries.length > 0 ? entries.map(([label, count]) => (
          <div key={label}>
            <span>{label}</span>
            <b><i style={{ width: `${(100 * count) / max}%` }} /></b>
            <strong>{count.toLocaleString()}</strong>
          </div>
        )) : <p>No data</p>}
      </div>
    </section>
  );
}

function HistogramExplorer({
  analysis,
  metric,
  onMetricChange,
  onPreview
}: {
  analysis: DatabaseAnalysisResponse;
  metric: string;
  onMetricChange: (metric: string) => void;
  onPreview: (request: PreviewRequest) => void;
}) {
  const pairs = useMemo(() => finiteMetricPairs(analysis.samples, metric), [analysis.samples, metric]);
  const values = useMemo(() => pairs.map((pair) => pair.value), [pairs]);
  const dataMin = values.length ? Math.min(...values) : 0;
  const dataMax = values.length ? Math.max(...values) : 1;
  const [xMin, setXMin] = useState<number | null>(null);
  const [xMax, setXMax] = useState<number | null>(null);
  const [binWidth, setBinWidth] = useState<number | null>(null);
  const bins = useMemo(
    () => histogramByWidth(values, xMin, xMax, binWidth),
    [binWidth, values, xMax, xMin]
  );
  const [activeBin, setActiveBin] = useState<AnalysisDistributionBin | null>(null);
  const maxCount = Math.max(1, ...bins.map((bin) => bin.count));
  const trend = useMemo(() => smoothTrendFromBins(bins, 2), [bins]);

  useEffect(() => {
    setActiveBin(null);
    setXMin(null);
    setXMax(null);
    setBinWidth(null);
  }, [metric]);

  const data: Data[] = [
    {
      type: "bar",
      x: bins.map((bin) => bin.center),
      y: bins.map((bin) => bin.count),
      width: bins.map((bin) => bin.end - bin.start),
      marker: {
        color: bins.map((bin) =>
          activeBin?.start === bin.start && activeBin.end === bin.end ? "#079669" : "#6f99d4"
        ),
        line: { color: "#ffffff", width: 1 }
      },
      hovertemplate: `${metricLabel(metric)} %{x:.3g}<br>count %{y}<extra></extra>`
    } as Data,
    {
      type: "scatter",
      mode: "lines",
      x: trend.x,
      y: trend.y,
      line: { color: "#df7d10", width: 2.5, shape: "spline", smoothing: 0.8 },
      fill: "tozeroy",
      fillcolor: "rgba(223, 125, 16, 0.14)",
      hovertemplate: `Trend<br>${metricLabel(metric)} %{x:.3g}<br>smoothed count %{y:.2f}<extra></extra>`
    } as Data
  ];

  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 48, r: 18, t: 8, b: 42 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    showlegend: false,
    bargap: 0,
    xaxis: {
      title: { text: metricLabel(metric), standoff: 10 },
      gridcolor: "#eef2f7",
      zeroline: false,
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    yaxis: {
      title: { text: "Count", standoff: 8 },
      gridcolor: "#eef2f7",
      rangemode: "tozero",
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    uirevision: metric
  };

  function selectBin(event?: Readonly<PlotMouseEvent>) {
    const pointIndex = event?.points?.[0]?.pointIndex;
    if (typeof pointIndex !== "number") return;
    const bin = bins[pointIndex];
    if (!bin) return;
    setActiveBin(bin);
    const ids = pairs
      .filter((pair) => pair.value >= bin.start && pair.value <= bin.end)
      .map((pair) => pair.sample.curve_id);
    onPreview({
      title: `${metricLabel(metric)} ${metricValue(metric, bin.start)} - ${metricValue(metric, bin.end)}`,
      detail: `${ids.length.toLocaleString()} curves in bin`,
      ids
    });
  }

  return (
    <section className="analysis-chart-panel analysis-histogram-panel">
      <div className="analysis-panel-header">
        <div>
          <BarChart3 size={16} />
          <h2>Distribution</h2>
        </div>
        <div className="analysis-control-row analysis-control-row-grid">
          <label>
            Metric
            <select value={metric} onChange={(event) => onMetricChange(event.target.value)}>
              {RAW_METRIC_OPTIONS.filter((option) => analysis.metrics[option]).map((option) => (
                <option key={option} value={option}>{metricLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>X min<input type="number" value={xMin ?? ""} placeholder={String(Number(dataMin.toPrecision(4)))} onChange={(event) => setXMin(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>X max<input type="number" value={xMax ?? ""} placeholder={String(Number(dataMax.toPrecision(4)))} onChange={(event) => setXMax(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>Bin width<input type="number" min="0" value={binWidth ?? ""} placeholder="auto" onChange={(event) => setBinWidth(event.target.value ? Number(event.target.value) : null)} /></label>
        </div>
      </div>
      <Plot
        data={data}
        layout={layout}
        config={plotConfig()}
        useResizeHandler
        className="analysis-plot analysis-histogram-plot"
        onClick={selectBin}
      />
      <div className="analysis-inline-stats">
        <span>N {analysis.metrics[metric]?.count.toLocaleString() ?? "0"}</span>
        <span>Median {metricValue(metric, analysis.metrics[metric]?.median ?? null)}</span>
        <span>Std {metricValue(metric, analysis.metrics[metric]?.std ?? null)}</span>
      </div>
    </section>
  );
}

function PreviewPanel({ request }: { request: PreviewRequest }) {
  const [details, setDetails] = useState<CurvePreview[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [xMin, setXMin] = useState<number | null>(null);
  const [xMax, setXMax] = useState<number | null>(null);
  const [hoveredCurveId, setHoveredCurveId] = useState<string | null>(null);
  const ids = useMemo(() => [...new Set(request.ids)].slice(0, PREVIEW_CURVE_LIMIT), [request.ids]);

  useEffect(() => {
    if (ids.length === 0) {
      setDetails([]);
      setLoading(false);
      setError(null);
      return undefined;
    }
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    setError(null);
    void getDatabaseCurvePreviews(ids, controller.signal)
      .then((responses) => {
        if (active) setDetails(responses);
      })
      .catch((caught) => {
        if (!active || (caught instanceof Error && caught.name === "AbortError")) return;
        setError(caught instanceof Error ? caught.message : "Could not load curve preview");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [ids]);

  useEffect(() => {
    setXMin(null);
    setXMax(null);
    setHoveredCurveId(null);
  }, [request.ids]);

  const allX = useMemo(
    () => details.flatMap((detail) => detail.raw_points.map((point) => point.voltage_v)),
    [details]
  );
  const previewXMin = allX.length > 0 ? Math.min(...allX) : null;
  const previewXMax = allX.length > 0 ? Math.max(...allX) : null;

  const traces: Data[] = details.map((detail, index) => ({
    type: "scatter",
    mode: "lines",
    x: detail.raw_points.map((point) => point.voltage_v),
    y: detail.raw_points.map((point) => Math.log10(Math.max(Math.abs(point.current_a), Number.MIN_VALUE))),
    customdata: detail.raw_points.map(() => detail.curve_id),
    name: sourceName(detail.source_path),
    text: detail.curve_id,
    showlegend: false,
    line: {
      color: colorFor(detail.polarity || String(index)),
      width:
        hoveredCurveId === detail.curve_id
          ? 3.2
          : details.length > 10
            ? 1.2
            : 1.8
    },
    opacity:
      hoveredCurveId === null || hoveredCurveId === detail.curve_id
        ? details.length > 10 ? 0.56 : 0.8
        : 0.18,
    hovertemplate: `${detail.curve_id}<br>Vg %{x:.3g} V<br>log |Id| %{y:.3g}<extra></extra>`
  } as Data));

  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 52, r: 16, t: 8, b: 40 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    showlegend: false,
    xaxis: {
      title: { text: "Gate voltage Vg (V)", standoff: 9 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      ticks: "outside",
      range: xMin !== null && xMax !== null && xMax > xMin ? [xMin, xMax] : undefined
    },
    yaxis: {
      title: { text: "log10 |Id|", standoff: 8 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    uirevision: request.ids.join("|")
  };

  return (
    <section className="analysis-chart-panel analysis-preview-panel">
      <div className="analysis-panel-header">
        <div>
          <Eye size={16} />
          <h2>{request.title}</h2>
        </div>
        <div className="analysis-control-row analysis-control-row-compact">
          <label>X min<input type="number" value={xMin ?? ""} placeholder={previewXMin === null ? "" : String(Number(previewXMin.toPrecision(4)))} onChange={(event) => setXMin(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>X max<input type="number" value={xMax ?? ""} placeholder={previewXMax === null ? "" : String(Number(previewXMax.toPrecision(4)))} onChange={(event) => setXMax(event.target.value ? Number(event.target.value) : null)} /></label>
        </div>
      </div>
      <span className="analysis-subtle">
        {request.ids.length > PREVIEW_CURVE_LIMIT
          ? `${details.length}/${request.ids.length.toLocaleString()} curves`
          : request.detail}
      </span>
      {error ? <div className="analysis-mini-error">{error}</div> : null}
      {loading ? (
        <div className="analysis-preview-loading"><RefreshCw className="spin" size={18} /> Loading preview</div>
      ) : null}
      {!loading && details.length === 0 ? (
        <div className="analysis-preview-empty"><Target size={22} /> No curves selected</div>
      ) : null}
      {details.length > 0 ? (
        <Plot
          data={traces}
          layout={layout}
          config={plotConfig()}
          useResizeHandler
          className="analysis-plot analysis-preview-plot"
          onHover={(event) => {
            const curveId = (event.points?.[0] as { customdata?: unknown } | undefined)?.customdata;
            setHoveredCurveId(typeof curveId === "string" ? curveId : null);
          }}
          onUnhover={() => setHoveredCurveId(null)}
        />
      ) : null}
    </section>
  );
}

function ScatterExplorer({
  analysis,
  xMetric,
  yMetric,
  onXMetricChange,
  onYMetricChange,
  onPreview
}: {
  analysis: DatabaseAnalysisResponse;
  xMetric: string;
  yMetric: string;
  onXMetricChange: (metric: string) => void;
  onYMetricChange: (metric: string) => void;
  onPreview: (request: PreviewRequest) => void;
}) {
  const [colorBy, setColorBy] = useState<Exclude<ColorMode, "cluster">>("polarity");
  const [xMin, setXMin] = useState<number | null>(null);
  const [xMax, setXMax] = useState<number | null>(null);
  const [yMin, setYMin] = useState<number | null>(null);
  const [yMax, setYMax] = useState<number | null>(null);
  const [openAxisMenu, setOpenAxisMenu] = useState<"x" | "y" | null>(null);
  const availableMetrics = useMemo(
    () => METRIC_OPTIONS.filter((option) => analysis.metrics[option]),
    [analysis.metrics]
  );
  const correlation = useMemo(() => pearsonFor(analysis.samples, xMetric, yMetric), [
    analysis.samples,
    xMetric,
    yMetric
  ]);
  const renderPoints = useMemo(
    () => sampleEvenly(correlation.points, ANALYSIS_SCATTER_RENDER_LIMIT),
    [correlation.points]
  );
  const groups = useMemo(() => {
    const buckets = new Map<string, typeof correlation.points>();
    renderPoints.forEach((point) => {
      const label = String(point.sample[colorBy]);
      buckets.set(label, [...(buckets.get(label) ?? []), point]);
    });
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [colorBy, renderPoints]);
  const filteredPoints = useMemo(
    () =>
      renderPoints.filter((point) => {
        if (xMin !== null && point.x < xMin) return false;
        if (xMax !== null && point.x > xMax) return false;
        if (yMin !== null && point.y < yMin) return false;
        if (yMax !== null && point.y > yMax) return false;
        return true;
      }),
    [renderPoints, xMax, xMin, yMax, yMin]
  );
  const filteredIds = useMemo(
    () => new Set(filteredPoints.map((point) => point.sample.curve_id)),
    [filteredPoints]
  );
  const xValues = filteredPoints.map((point) => point.x);
  const yValues = filteredPoints.map((point) => point.y);
  const xHist = histogramBins(xValues, 24);
  const yHist = histogramBins(yValues, 24);
  const xDensity = smoothTrendFromBins(xHist, 2);
  const yDensity = smoothTrendFromBins(yHist, 2);
  const filteredCorrelation = useMemo(() => {
    if (filteredPoints.length < 2) return { r: 0, r2: 0, slope: 0 };
    const xMean = xValues.reduce((sum, value) => sum + value, 0) / xValues.length;
    const yMean = yValues.reduce((sum, value) => sum + value, 0) / yValues.length;
    let numerator = 0;
    let xVariance = 0;
    let yVariance = 0;
    filteredPoints.forEach((point) => {
      const dx = point.x - xMean;
      const dy = point.y - yMean;
      numerator += dx * dy;
      xVariance += dx * dx;
      yVariance += dy * dy;
    });
    const r = xVariance > 0 && yVariance > 0 ? numerator / Math.sqrt(xVariance * yVariance) : 0;
    const slope = xVariance > 0 ? numerator / xVariance : 0;
    return { r, r2: r * r, slope };
  }, [filteredPoints, xValues, yValues]);

  const scatterTraces: Data[] = groups.map(([label, points]) => ({
    type: "scatter",
    mode: "markers",
    x: points
      .filter((point) => filteredIds.has(point.sample.curve_id))
      .map((point) => point.x),
    y: points
      .filter((point) => filteredIds.has(point.sample.curve_id))
      .map((point) => point.y),
    customdata: points
      .filter((point) => filteredIds.has(point.sample.curve_id))
      .map((point) => point.sample.curve_id),
    name: label,
    marker: {
      color: colorFor(label),
      size: 7,
      opacity: 0.72,
      line: { color: "#ffffff", width: 0.5 }
    },
    hovertemplate: `%{customdata}<br>${metricLabel(xMetric)} %{x:.3g}<br>${metricLabel(yMetric)} %{y:.3g}<extra>${label}</extra>`
  } as Data));
  const data: Data[] = [
    ...scatterTraces,
    {
      type: "bar",
      x: xHist.map((bin) => bin.center),
      y: xHist.map((bin) => bin.count),
      width: xHist.map((bin) => bin.end - bin.start),
      xaxis: "x",
      yaxis: "y2",
      marker: { color: "rgba(111, 153, 212, 0.52)" },
      hoverinfo: "skip",
      showlegend: false
    } as Data,
    {
      type: "scatter",
      mode: "lines",
      x: xDensity.x,
      y: xDensity.y,
      xaxis: "x",
      yaxis: "y2",
      line: { color: "#df7d10", width: 2.2, shape: "spline", smoothing: 0.8 },
      fill: "tozeroy",
      fillcolor: "rgba(223, 125, 16, 0.14)",
      hoverinfo: "skip",
      showlegend: false
    } as Data,
    {
      type: "bar",
      orientation: "h",
      y: yHist.map((bin) => bin.center),
      x: yHist.map((bin) => bin.count),
      width: yHist.map((bin) => bin.end - bin.start),
      xaxis: "x3",
      yaxis: "y",
      marker: { color: "rgba(7, 150, 105, 0.42)" },
      hoverinfo: "skip",
      showlegend: false
    } as Data,
    {
      type: "scatter",
      mode: "lines",
      x: yDensity.y,
      y: yDensity.x,
      xaxis: "x3",
      yaxis: "y",
      line: { color: "#8b5cf6", width: 2.2, shape: "spline", smoothing: 0.8 },
      fill: "tozerox",
      fillcolor: "rgba(139, 92, 246, 0.12)",
      hoverinfo: "skip",
      showlegend: false
    } as Data
  ];

  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 58, r: 18, t: 8, b: 48 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    showlegend: true,
    legend: { orientation: "v", x: 1.02, y: 0, yanchor: "bottom", font: { size: 9 } },
    dragmode: "select",
    xaxis: {
      domain: [0, 0.78],
      title: { text: "", standoff: 0 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      zeroline: false,
      ticks: "outside",
      range: xMin !== null && xMax !== null && xMax > xMin ? [xMin, xMax] : undefined
    },
    yaxis: {
      domain: [0, 0.72],
      title: { text: "", standoff: 0 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      zeroline: false,
      ticks: "outside",
      range: yMin !== null && yMax !== null && yMax > yMin ? [yMin, yMax] : undefined
    },
    yaxis2: {
      domain: [0.8, 1],
      anchor: "x",
      showticklabels: false,
      showgrid: false,
      zeroline: false
    },
    xaxis3: {
      domain: [0.84, 1],
      anchor: "y",
      showticklabels: false,
      showgrid: false,
      zeroline: false
    },
    uirevision: `${xMetric}-${yMetric}-${colorBy}`
  };

  function previewIds(ids: string[], label: string) {
    onPreview({
      title: label,
      detail: `${ids.length.toLocaleString()} curves`,
      ids
    });
  }

  function handleSelected(event?: Readonly<PlotSelectionEvent>) {
    const ids = (event?.points ?? [])
      .map((point) => (point as { customdata?: unknown }).customdata)
      .filter((value): value is string => typeof value === "string");
    if (ids.length === 0) return;
    previewIds(ids, `${metricLabel(xMetric)} x ${metricLabel(yMetric)} selection`);
  }

  function handleClick(event?: Readonly<PlotMouseEvent>) {
    const id = (event?.points?.[0] as { customdata?: unknown } | undefined)?.customdata;
    if (typeof id === "string") previewIds([id], id);
  }

  return (
    <section className="analysis-chart-panel analysis-scatter-panel">
      <div className="analysis-panel-header">
        <div>
          <ScatterChart size={16} />
          <h2>Scatter correlation</h2>
        </div>
        <div className="analysis-control-row analysis-control-row-grid">
          <label>X min<input type="number" value={xMin ?? ""} onChange={(event) => setXMin(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>X max<input type="number" value={xMax ?? ""} onChange={(event) => setXMax(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>Y min<input type="number" value={yMin ?? ""} onChange={(event) => setYMin(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>Y max<input type="number" value={yMax ?? ""} onChange={(event) => setYMax(event.target.value ? Number(event.target.value) : null)} /></label>
          <label className="analysis-color-control analysis-color-control-inline">
            Color
            <select value={colorBy} onChange={(event) => setColorBy(event.target.value as typeof colorBy)}>
              <option value="polarity">Polarity</option>
              <option value="source_kind">Source</option>
              <option value="direction">Direction</option>
            </select>
          </label>
        </div>
      </div>
      <div className="analysis-scatter-shell">
        <div className="analysis-axis-title analysis-axis-title-y">
          <button type="button" onClick={() => setOpenAxisMenu((current) => current === "y" ? null : "y")}>
            {metricLabel(yMetric)}
          </button>
          {openAxisMenu === "y" ? (
            <div className="analysis-axis-menu">
              {availableMetrics.map((option) => (
                <button
                  key={`y-${option}`}
                  type="button"
                  className={option === yMetric ? "active" : undefined}
                  onClick={() => {
                    onYMetricChange(option);
                    setOpenAxisMenu(null);
                  }}
                >
                  {metricLabel(option)}
                </button>
              ))}
            </div>
          ) : null}
        </div>
        <Plot
          data={data}
          layout={layout}
          config={plotConfig(true)}
          useResizeHandler
          className="analysis-plot analysis-scatter-plot"
          onSelected={handleSelected}
          onClick={handleClick}
        />
        <div className="analysis-axis-title analysis-axis-title-x">
          <button type="button" onClick={() => setOpenAxisMenu((current) => current === "x" ? null : "x")}>
            {metricLabel(xMetric)}
          </button>
          {openAxisMenu === "x" ? (
            <div className="analysis-axis-menu">
              {availableMetrics.map((option) => (
                <button
                  key={`x-${option}`}
                  type="button"
                  className={option === xMetric ? "active" : undefined}
                  onClick={() => {
                    onXMetricChange(option);
                    setOpenAxisMenu(null);
                  }}
                >
                  {metricLabel(option)}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      </div>
      <div className="analysis-inline-stats">
        <span>N {filteredPoints.length.toLocaleString()}</span>
        <span>Plot sample {renderPoints.length.toLocaleString()}</span>
        <span>r {fixed(filteredCorrelation.r, 3)}</span>
        <span>R2 {fixed(filteredCorrelation.r2, 3)}</span>
        <span>Slope {fixed(filteredCorrelation.slope, 3)}</span>
      </div>
    </section>
  );
}

function CorrelationPanel({ analysis }: { analysis: DatabaseAnalysisResponse }) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; column: number } | null>(null);
  const [pinnedCell, setPinnedCell] = useState<{ row: number; column: number } | null>(null);

  const orderedIndexes = useMemo(() => {
    const features = analysis.correlations.features;
    const matrix = analysis.correlations.matrix;
    if (features.length <= 2) return features.map((_, index) => index);
    const remaining = new Set(features.map((_, index) => index));
    const seed = features
      .map((_, index) => ({
        index,
        strength: matrix[index].reduce(
          (sum: number, value) => sum + (value === null ? 0 : Math.abs(value)),
          0
        )
      }))
      .sort((left, right) => right.strength - left.strength)[0]?.index ?? 0;
    const ordered = [seed];
    remaining.delete(seed);
    while (remaining.size > 0) {
      const last = ordered[ordered.length - 1];
      let bestIndex = -1;
      let bestScore = Number.NEGATIVE_INFINITY;
      remaining.forEach((candidate) => {
        const pairScore = Math.abs(matrix[last]?.[candidate] ?? 0);
        const globalScore = matrix[candidate].reduce(
          (sum: number, value) => sum + (value === null ? 0 : Math.abs(value)),
          0
        );
        const score = pairScore * 10 + globalScore;
        if (score > bestScore) {
          bestScore = score;
          bestIndex = candidate;
        }
      });
      ordered.push(bestIndex);
      remaining.delete(bestIndex);
    }
    return ordered;
  }, [analysis.correlations.features, analysis.correlations.matrix]);

  const orderedFeatures = orderedIndexes.map((index) => analysis.correlations.features[index]);
  const labels = orderedFeatures.map(metricLabel);
  const orderedMatrix = orderedIndexes.map((rowIndex) =>
    orderedIndexes.map((columnIndex) => analysis.correlations.matrix[rowIndex]?.[columnIndex] ?? null)
  );
  const orderedCounts = orderedIndexes.map((rowIndex) =>
    orderedIndexes.map((columnIndex) => analysis.correlations.counts[rowIndex]?.[columnIndex] ?? 0)
  );

  const activeCell = pinnedCell ?? hoveredCell;
  const activeRowMetric = activeCell ? orderedFeatures[activeCell.row] ?? null : null;
  const activeColMetric = activeCell ? orderedFeatures[activeCell.column] ?? null : null;
  const activeValue = activeCell ? orderedMatrix[activeCell.row]?.[activeCell.column] ?? null : null;
  const activeCount = activeCell ? orderedCounts[activeCell.row]?.[activeCell.column] ?? 0 : 0;

  const strongestPairs = useMemo(() => {
    const sourceMetric = activeRowMetric ?? orderedFeatures[0] ?? null;
    if (!sourceMetric) return [] as Array<{ other: string; r: number; count: number }>;
    const metricIndex = orderedFeatures.indexOf(sourceMetric);
    if (metricIndex < 0) return [];
    return orderedFeatures
      .map((feature, index) => ({
        other: feature,
        r: orderedMatrix[metricIndex]?.[index] ?? null,
        count: orderedCounts[metricIndex]?.[index] ?? 0
      }))
      .filter(
        (item): item is { other: string; r: number; count: number } =>
          item.r !== null && item.other !== sourceMetric
      )
      .sort((left, right) => Math.abs(right.r) - Math.abs(left.r))
      .slice(0, 5);
  }, [activeRowMetric, orderedCounts, orderedFeatures, orderedMatrix]);

  const textMatrix = orderedMatrix.map((row, rowIndex) =>
    row.map((value, columnIndex) => {
      if (value === null) return "";
      return rowIndex === columnIndex || Math.abs(value) >= 0.6 ? fixed(value, 2) : "";
    })
  );

  const heatmapData: Data[] = [
    {
      type: "heatmap",
      x: labels,
      y: labels,
      z: orderedMatrix.map((row) => row.map((value) => value ?? null)),
      text: textMatrix,
      texttemplate: "%{text}",
      textfont: { size: 9, color: "#20334d" },
      xgap: 2,
      ygap: 2,
      zmin: -1,
      zmax: 1,
      zmid: 0,
      colorscale: [
        [0, "#1d5fd1"],
        [0.5, "#f8fbff"],
        [1, "#cb2f69"]
      ],
      colorbar: {
        title: { text: "r", side: "right" },
        thickness: 14,
        len: 0.84,
        tickvals: [-1, -0.5, 0, 0.5, 1]
      },
      customdata: orderedCounts.map((row, rowIndex) =>
        row.map((count, columnIndex) => [
          orderedFeatures[rowIndex],
          orderedFeatures[columnIndex],
          count
        ])
      ),
      hovertemplate:
        "<b>%{customdata[0]}</b> vs <b>%{customdata[1]}</b><br>" +
        "r = %{z:.3f}<br>" +
        "N = %{customdata[2]}<extra></extra>"
    } as unknown as Data
  ];

  const heatmapLayout: Partial<Layout> = {
    autosize: true,
    margin: { l: 92, r: 38, t: 18, b: 84 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    xaxis: {
      side: "top",
      tickangle: -35,
      automargin: true,
      tickfont: { size: 10 }
    },
    yaxis: {
      automargin: true,
      autorange: "reversed",
      tickfont: { size: 10 },
      scaleanchor: "x",
      scaleratio: 1
    }
  };

  function focusMetric(metric: string) {
    const index = orderedFeatures.indexOf(metric);
    if (index < 0) return;
    setPinnedCell({ row: index, column: index });
  }

  function handleHeatmapClick(event?: Readonly<PlotMouseEvent>) {
    const point = event?.points?.[0] as
      | { pointNumber?: [number, number] | number[] }
      | undefined;
    const pair = point?.pointNumber;
    if (!pair || pair.length < 2) return;
    const [row, column] = pair;
    setPinnedCell((current) =>
      current?.row === row && current?.column === column ? null : { row, column }
    );
  }

  function handleHeatmapHover(event?: Readonly<PlotMouseEvent>) {
    const point = event?.points?.[0] as
      | { pointNumber?: [number, number] | number[] }
      | undefined;
    const pair = point?.pointNumber;
    if (!pair || pair.length < 2) return;
    const [row, column] = pair;
    setHoveredCell({ row, column });
  }

  return (
    <section className="analysis-chart-panel analysis-correlation-panel">
      <div className="analysis-panel-header">
        <div>
          <Sigma size={16} />
          <h2>Correlation map</h2>
        </div>
      </div>
      <div className="analysis-correlation-topbar">
        <div className="analysis-correlation-legend">
          <span>Negative</span>
          <b />
          <span>Positive</span>
        </div>
        <div className="analysis-correlation-focus">
          {activeCell ? (
            <>
              <strong>{metricLabel(activeRowMetric ?? "")} / {metricLabel(activeColMetric ?? "")}</strong>
              <span>{activeValue === null ? "No usable overlap" : `r ${fixed(activeValue, 3)} · N ${activeCount.toLocaleString()}`}</span>
            </>
          ) : (
            <span>Hover a cell to inspect one pair. Click to pin it and compare its strongest links.</span>
          )}
        </div>
      </div>
      <div className="analysis-correlation-layout">
        <div
          className="analysis-correlation-plot-wrap"
          onMouseLeave={() => setHoveredCell(null)}
        >
          <Plot
            data={heatmapData}
            layout={heatmapLayout}
            config={plotConfig(false)}
            useResizeHandler
            className="analysis-plot analysis-correlation-plot"
            onHover={handleHeatmapHover}
            onClick={handleHeatmapClick}
          />
        </div>
        <div className="analysis-correlation-detail">
          {activeCell ? (
            <>
              <strong>{metricLabel(activeRowMetric ?? "")} / {metricLabel(activeColMetric ?? "")}</strong>
              <b>{activeValue === null ? "NA" : fixed(activeValue, 3)}</b>
              <span>
                {activeValue === null
                  ? "This pair does not have enough finite values."
                  : activeValue >= 0
                    ? "Positive relationship across the filtered population."
                    : "Negative relationship across the filtered population."}
              </span>
            </>
          ) : (
            <>
              <strong>Interactive heatmap</strong>
              <span>
                The matrix is reordered by correlation structure so related metrics stay visually grouped.
              </span>
            </>
          )}
          {strongestPairs.map((pair) => (
            <button
              key={`${activeRowMetric ?? "seed"}-${pair.other}`}
              type="button"
              className="analysis-correlation-pair"
              onClick={() => focusMetric(pair.other)}
            >
              <span>{metricLabel(pair.other)}</span>
              <b>{fixed(pair.r, 3)}</b>
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

type NetworkNode = {
  id: string;
  label: string;
  degree: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
};

type NetworkEdge = {
  source: string;
  target: string;
  r: number;
};

function buildCorrelationNetwork(
  analysis: DatabaseAnalysisResponse,
  threshold: number
): { nodes: NetworkNode[]; edges: NetworkEdge[] } {
  const nodes = analysis.correlations.features.map((feature, index) => ({
    id: feature,
    label: metricLabel(feature),
    degree: 0,
    x: 0.5 + Math.cos((Math.PI * 2 * index) / Math.max(analysis.correlations.features.length, 1)) * 0.28,
    y: 0.5 + Math.sin((Math.PI * 2 * index) / Math.max(analysis.correlations.features.length, 1)) * 0.28,
    vx: 0,
    vy: 0
  }));
  const edges: NetworkEdge[] = [];
  analysis.correlations.matrix.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      if (columnIndex <= rowIndex || value === null || Math.abs(value) < threshold) return;
      edges.push({
        source: analysis.correlations.features[rowIndex],
        target: analysis.correlations.features[columnIndex],
        r: value
      });
    });
  });
  const degreeById = new Map<string, number>();
  edges.forEach((edge) => {
    degreeById.set(edge.source, (degreeById.get(edge.source) ?? 0) + 1);
    degreeById.set(edge.target, (degreeById.get(edge.target) ?? 0) + 1);
  });
  return {
    nodes: nodes.map((node) => ({ ...node, degree: degreeById.get(node.id) ?? 0 })),
    edges
  };
}

function CorrelationNetworkPanel({ analysis }: { analysis: DatabaseAnalysisResponse }) {
  const [threshold, setThreshold] = useState(0.45);
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const baseNetwork = useMemo(() => buildCorrelationNetwork(analysis, threshold), [analysis, threshold]);
  const [nodes, setNodes] = useState<NetworkNode[]>(baseNetwork.nodes);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragRef = useRef<string | null>(null);

  useEffect(() => {
    setNodes(baseNetwork.nodes);
    setActiveNode(null);
  }, [baseNetwork]);

  useEffect(() => {
    let frame = 0;
    let cancelled = false;

    const step = () => {
      setNodes((current) => {
        const indexById = new Map(current.map((node, index) => [node.id, index]));
        const next = current.map((node) => ({ ...node }));
        next.forEach((node, index) => {
          if (dragRef.current === node.id) {
            node.vx = 0;
            node.vy = 0;
            return;
          }
          let fx = 0;
          let fy = 0;
          next.forEach((other, otherIndex) => {
            if (index === otherIndex) return;
            const dx = node.x - other.x;
            const dy = node.y - other.y;
            const dist = Math.max(0.001, Math.sqrt(dx * dx + dy * dy));
            const repel = 0.0013 / (dist * dist);
            fx += (dx / dist) * repel;
            fy += (dy / dist) * repel;
          });
          baseNetwork.edges.forEach((edge) => {
            if (edge.source !== node.id && edge.target !== node.id) return;
            const otherIndex = indexById.get(edge.source === node.id ? edge.target : edge.source);
            if (otherIndex === undefined) return;
            const other = next[otherIndex];
            const dx = other.x - node.x;
            const dy = other.y - node.y;
            const dist = Math.max(0.001, Math.sqrt(dx * dx + dy * dy));
            const spring = (dist - 0.22) * 0.026 * (0.65 + Math.abs(edge.r));
            fx += (dx / dist) * spring;
            fy += (dy / dist) * spring;
          });
          fx += (0.5 - node.x) * 0.004;
          fy += (0.5 - node.y) * 0.004;
          node.vx = (node.vx + fx) * 0.84;
          node.vy = (node.vy + fy) * 0.84;
        });
        next.forEach((node) => {
          if (dragRef.current === node.id) return;
          node.x = Math.min(0.92, Math.max(0.08, node.x + node.vx));
          node.y = Math.min(0.92, Math.max(0.08, node.y + node.vy));
        });
        return next;
      });
      if (!cancelled) frame = window.requestAnimationFrame(step);
    };

    frame = window.requestAnimationFrame(step);
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [baseNetwork.edges]);

  useEffect(() => {
    function updateDrag(clientX: number, clientY: number) {
      const draggingId = dragRef.current;
      const svg = svgRef.current;
      if (!draggingId || !svg) return;
      const bounds = svg.getBoundingClientRect();
      const x = Math.min(0.92, Math.max(0.08, (clientX - bounds.left) / Math.max(bounds.width, 1)));
      const y = Math.min(0.92, Math.max(0.08, (clientY - bounds.top) / Math.max(bounds.height, 1)));
      setNodes((current) =>
        current.map((node) =>
          node.id === draggingId ? { ...node, x, y, vx: 0, vy: 0 } : node
        )
      );
    }

    const handleMove = (event: PointerEvent) => updateDrag(event.clientX, event.clientY);
    const handleUp = () => {
      dragRef.current = null;
    };
    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp);
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
  }, []);

  const connectedIds = useMemo(() => {
    if (activeNode === null) return new Set<string>();
    const ids = new Set<string>([activeNode]);
    baseNetwork.edges.forEach((edge) => {
      if (edge.source === activeNode) ids.add(edge.target);
      if (edge.target === activeNode) ids.add(edge.source);
    });
    return ids;
  }, [activeNode, baseNetwork.edges]);

  const activeLinks = useMemo(() => {
    if (activeNode === null) return [] as NetworkEdge[];
    return baseNetwork.edges
      .filter((edge) => edge.source === activeNode || edge.target === activeNode)
      .sort((left, right) => Math.abs(right.r) - Math.abs(left.r))
      .slice(0, 4);
  }, [activeNode, baseNetwork.edges]);

  return (
    <section className="analysis-chart-panel analysis-network-panel">
      <div className="analysis-panel-header">
        <div>
          <Target size={16} />
          <h2>Correlation network</h2>
        </div>
        <div className="analysis-control-row analysis-control-row-grid">
          <label>
            Threshold
            <input
              type="range"
              min={0.2}
              max={0.85}
              step={0.05}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
            <strong>{fixed(threshold, 2)}</strong>
          </label>
        </div>
      </div>
      <div className="analysis-network-shell">
        <svg
          ref={svgRef}
          className="analysis-network-svg"
          viewBox="0 0 1000 1000"
          role="img"
          aria-label="Draggable correlation network"
        >
          {baseNetwork.edges.map((edge) => {
            const source = nodes.find((node) => node.id === edge.source);
            const target = nodes.find((node) => node.id === edge.target);
            if (!source || !target) return null;
            const highlighted = activeNode !== null && (edge.source === activeNode || edge.target === activeNode);
            return (
              <line
                key={`${edge.source}-${edge.target}`}
                x1={source.x * 1000}
                y1={source.y * 1000}
                x2={target.x * 1000}
                y2={target.y * 1000}
                stroke={edge.r >= 0 ? "#cb2f69" : "#1d5fd1"}
                strokeOpacity={activeNode === null ? 0.34 : highlighted ? 0.88 : 0.1}
                strokeWidth={2 + Math.abs(edge.r) * 7}
              />
            );
          })}
          {nodes.map((node) => {
            const active = activeNode === node.id;
            const related = activeNode === null || connectedIds.has(node.id);
            const radius = 24 + Math.min(node.degree, 6) * 4;
            return (
              <g
                key={node.id}
                className="analysis-network-node"
                transform={`translate(${node.x * 1000} ${node.y * 1000})`}
                onMouseEnter={() => setActiveNode(node.id)}
                onMouseLeave={() => setActiveNode(null)}
                onPointerDown={(event) => {
                  dragRef.current = node.id;
                  setActiveNode(node.id);
                  event.currentTarget.setPointerCapture(event.pointerId);
                }}
              >
                <circle
                  r={radius}
                  fill={active ? "#1459cb" : "#ffffff"}
                  fillOpacity={related ? 0.98 : 0.42}
                  stroke={active ? "#1459cb" : "#c7d4e3"}
                  strokeWidth={active ? 4 : 2}
                />
                <text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fill={active ? "#ffffff" : "#263a55"}
                  fontSize="24"
                  fontWeight="700"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="analysis-inline-stats">
        <span>Nodes <strong>{nodes.length.toLocaleString()}</strong></span>
        <span>Edges <strong>{baseNetwork.edges.length.toLocaleString()}</strong></span>
        <span>Threshold <strong>{fixed(threshold, 2)}</strong></span>
        <span>{activeNode ? `Focus ${metricLabel(activeNode)}` : "Drag a node to reshape the graph"}</span>
      </div>
      {activeNode ? (
        <div className="analysis-correlation-list">
          {activeLinks.map((edge) => {
            const other = edge.source === activeNode ? edge.target : edge.source;
            return (
              <span key={`${activeNode}-${other}`}>
                {metricLabel(other)}
                <strong>{fixed(edge.r, 3)}</strong>
              </span>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}

type PcaClusterResult = {
  assignments: Map<string, number>;
  idsByCluster: Map<number, string[]>;
  clusterSizes: Record<string, number>;
  score: number;
};

function vectorDistanceSquared(left: number[], right: number[]) {
  return left.reduce((sum, value, index) => {
    const delta = value - right[index];
    return sum + delta * delta;
  }, 0);
}

function clusterPca(points: AnalysisPcaPoint[], depth: number, clusterCount: number): PcaClusterResult {
  const usable = points.filter((point) => point.scores.length >= depth);
  const assignments = new Map<string, number>();
  const idsByCluster = new Map<number, string[]>();
  const clusterSizes: Record<string, number> = {};
  const k = Math.min(clusterCount, usable.length);
  if (k <= 0) {
    return { assignments, idsByCluster, clusterSizes, score: -1 };
  }
  if (k === 1) {
    usable.forEach((point) => {
      assignments.set(point.curve_id, 0);
      idsByCluster.set(0, [...(idsByCluster.get(0) ?? []), point.curve_id]);
    });
    clusterSizes.C1 = usable.length;
    return { assignments, idsByCluster, clusterSizes, score: 0 };
  }
  const ordered = [...usable].sort((a, b) => a.curve_id.localeCompare(b.curve_id));
  const vectors = ordered.map((point) => point.scores.slice(0, depth));
  let centroids = Array.from({ length: k }, (_, index) => {
    const source = ordered[Math.round((index * (ordered.length - 1)) / (k - 1))];
    return source.scores.slice(0, depth);
  });
  let indexes = new Array<number>(ordered.length).fill(0);
  for (let iteration = 0; iteration < 32; iteration += 1) {
    const buckets = Array.from({ length: k }, () => [] as number[][]);
    indexes = ordered.map((_, pointIndex) => {
      const vector = vectors[pointIndex];
      let bestIndex = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      centroids.forEach((centroid, centroidIndex) => {
        const distance = vectorDistanceSquared(vector, centroid);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = centroidIndex;
        }
      });
      buckets[bestIndex].push(vector);
      return bestIndex;
    });
    centroids = centroids.map((centroid, centroidIndex) => {
      const bucket = buckets[centroidIndex];
      if (bucket.length === 0) return centroid;
      return centroid.map((_, valueIndex) =>
        bucket.reduce((sum, vector) => sum + vector[valueIndex], 0) / bucket.length
      );
    });
  }

  ordered.forEach((point, pointIndex) => {
    const clusterIndex = indexes[pointIndex];
    assignments.set(point.curve_id, clusterIndex);
    idsByCluster.set(clusterIndex, [...(idsByCluster.get(clusterIndex) ?? []), point.curve_id]);
  });
  idsByCluster.forEach((ids, clusterIndex) => {
    clusterSizes[`C${clusterIndex + 1}`] = ids.length;
  });
  const sampleIndexes = sampleEvenly(
    indexes.map((_, index) => index),
    Math.min(320, indexes.length)
  );
  const silhouetteValues = sampleIndexes.map((pointIndex) => {
    const clusterIndex = indexes[pointIndex];
    const vector = vectors[pointIndex];
    const ownCentroid = centroids[clusterIndex];
    const a = Math.sqrt(vectorDistanceSquared(vector, ownCentroid));
    let b = Number.POSITIVE_INFINITY;
    centroids.forEach((centroid, centroidIndex) => {
      if (centroidIndex === clusterIndex) return;
      b = Math.min(b, Math.sqrt(vectorDistanceSquared(vector, centroid)));
    });
    if (!Number.isFinite(b)) return 0;
    return (b - a) / Math.max(a, b, 1e-9);
  });
  const score =
    silhouetteValues.length > 0
      ? silhouetteValues.reduce((sum, value) => sum + value, 0) / silhouetteValues.length
      : -1;
  return { assignments, idsByCluster, clusterSizes, score };
}

function chooseAutoClusterCount(points: AnalysisPcaPoint[], depth: number, options: number[]) {
  const usable = points.filter((point) => point.scores.length >= depth);
  if (usable.length < 4) return Math.min(2, Math.max(1, usable.length));
  let bestCount = Math.min(3, usable.length);
  let bestScore = Number.NEGATIVE_INFINITY;
  options.forEach((option) => {
    if (option > usable.length || option < 2) return;
    const result = clusterPca(usable, depth, option);
    const normalizedPenalty = option / Math.max(usable.length, 8);
    const score = result.score - normalizedPenalty * 0.08;
    if (score > bestScore) {
      bestScore = score;
      bestCount = option;
    }
  });
  return bestCount;
}

function PcaExplorer({
  analysis,
  onPreview
}: {
  analysis: DatabaseAnalysisResponse;
  onPreview: (request: PreviewRequest) => void;
}) {
  const componentCount = analysis.pca.components.length;
  const plottedPcaPoints = useMemo(
    () => sampleEvenly(analysis.pca.points, ANALYSIS_PCA_CLUSTER_LIMIT),
    [analysis.pca.points]
  );
  const [depth, setDepth] = useState(2);
  const [clusterCount, setClusterCount] = useState<number | "auto">("auto");
  const [colorMode, setColorMode] = useState<ColorMode>("cluster");
  const [xComponent, setXComponent] = useState(0);
  const [yComponent, setYComponent] = useState(1);

  useEffect(() => {
    setDepth((current) => Math.min(Math.max(1, current), Math.max(1, componentCount)));
    setXComponent((current) => Math.min(current, Math.max(0, componentCount - 1)));
    setYComponent((current) => Math.min(Math.max(1, current), Math.max(0, componentCount - 1)));
  }, [componentCount]);

  const clusterOptions = useMemo(
    () => [2, 3, 4, 5, 6, 8].filter((value) => value <= Math.max(2, plottedPcaPoints.length)),
    [plottedPcaPoints.length]
  );
  const effectiveDepth = Math.min(depth, componentCount);
  const autoClusterCount = useMemo(
    () => chooseAutoClusterCount(plottedPcaPoints, effectiveDepth, clusterOptions),
    [plottedPcaPoints, clusterOptions, effectiveDepth]
  );
  const effectiveClusterCount = clusterCount === "auto" ? autoClusterCount : clusterCount;
  const clusterResult = useMemo(
    () => clusterPca(plottedPcaPoints, effectiveDepth, effectiveClusterCount),
    [plottedPcaPoints, effectiveClusterCount, effectiveDepth]
  );
  const grouped = useMemo(() => {
    const buckets = new Map<string, AnalysisPcaPoint[]>();
    plottedPcaPoints.forEach((point) => {
      if (point.scores.length <= Math.max(xComponent, yComponent)) return;
      const label = colorMode === "cluster"
        ? `Cluster ${(clusterResult.assignments.get(point.curve_id) ?? 0) + 1}`
        : String(point[colorMode]);
      buckets.set(label, [...(buckets.get(label) ?? []), point]);
    });
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [plottedPcaPoints, clusterResult.assignments, colorMode, xComponent, yComponent]);

  if (componentCount < 2) {
    return (
      <section className="analysis-chart-panel analysis-pca-panel">
        <div className="analysis-panel-header">
          <div>
            <SlidersHorizontal size={16} />
            <h2>PCA workspace</h2>
          </div>
        </div>
        <div className="analysis-preview-empty"><Sigma size={22} /> Not enough numeric variation</div>
      </section>
    );
  }

  const data: Data[] = grouped.map(([label, points]) => ({
    type: "scatter",
    mode: "markers",
    x: points.map((point) => point.scores[xComponent]),
    y: points.map((point) => point.scores[yComponent]),
    customdata: points.map((point) => point.curve_id),
    name: label,
    marker: {
      color: colorFor(label),
      size: 7,
      opacity: 0.72,
      line: { color: "#ffffff", width: 0.5 }
    },
    hovertemplate: `%{customdata}<br>${analysis.pca.components[xComponent].name} %{x:.3g}<br>${analysis.pca.components[yComponent].name} %{y:.3g}<extra>${label}</extra>`
  } as Data));
  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 58, r: 22, t: 8, b: 50 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    showlegend: true,
    legend: { orientation: "h", x: 0, y: 1.12, font: { size: 9 } },
    dragmode: "select",
    xaxis: {
      title: {
        text: `${analysis.pca.components[xComponent].name} (${fixed(100 * analysis.pca.components[xComponent].explained_variance_ratio, 1)}%)`,
        standoff: 10
      },
      gridcolor: "#eef2f7",
      zerolinecolor: "#cdd7e5",
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    yaxis: {
      title: {
        text: `${analysis.pca.components[yComponent].name} (${fixed(100 * analysis.pca.components[yComponent].explained_variance_ratio, 1)}%)`,
        standoff: 8
      },
      gridcolor: "#eef2f7",
      zerolinecolor: "#cdd7e5",
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    uirevision: `pca-${depth}-${effectiveClusterCount}-${colorMode}-${xComponent}-${yComponent}`
  };

  function previewIds(ids: string[], title: string) {
    onPreview({ title, detail: `${ids.length.toLocaleString()} curves`, ids });
  }

  function handleSelected(event?: Readonly<PlotSelectionEvent>) {
    const ids = (event?.points ?? [])
      .map((point) => (point as { customdata?: unknown }).customdata)
      .filter((value): value is string => typeof value === "string");
    if (ids.length === 0) return;
    previewIds(ids, "PCA selection");
  }

  function handleClick(event?: Readonly<PlotMouseEvent>) {
    const id = (event?.points?.[0] as { customdata?: unknown } | undefined)?.customdata;
    if (typeof id !== "string") return;
    if (colorMode === "cluster") {
      const clusterIndex = clusterResult.assignments.get(id);
      const ids = clusterIndex === undefined ? [id] : (clusterResult.idsByCluster.get(clusterIndex) ?? [id]);
      previewIds(ids, `Cluster ${clusterIndex === undefined ? "?" : clusterIndex + 1}`);
      return;
    }
    previewIds([id], id);
  }

  const clusterSizes = clusterResult.clusterSizes;
  const xLoadings = Object.entries(analysis.pca.components[xComponent].loadings)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, 5);
  const yLoadings = Object.entries(analysis.pca.components[yComponent].loadings)
    .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
    .slice(0, 5);

  return (
    <section className="analysis-chart-panel analysis-pca-panel">
      <div className="analysis-panel-header">
        <div>
          <SlidersHorizontal size={16} />
          <h2>PCA workspace</h2>
        </div>
      </div>
      <div className="analysis-pca-stage">
        <Plot
          data={data}
          layout={layout}
          config={plotConfig(true)}
          useResizeHandler
          className="analysis-plot analysis-pca-plot"
          onSelected={handleSelected}
          onClick={handleClick}
        />
      </div>
      <div className="analysis-pca-summary-grid">
        <section className="analysis-pca-side-card analysis-pca-compact-card">
          <h3>Explained variance</h3>
          <div className="analysis-pca-bars">
            {analysis.pca.components.map((component) => (
              <div key={component.name}>
                <span>{component.name}</span>
                <b><i style={{ width: `${Math.max(2, component.explained_variance_ratio * 100)}%` }} /></b>
                <strong>{fixed(component.explained_variance_ratio * 100, 1)}%</strong>
              </div>
            ))}
          </div>
        </section>
        <section className="analysis-pca-side-card analysis-pca-compact-card">
          <h3>Dominant loadings</h3>
          <div className="analysis-pca-loading-columns">
            <div className="analysis-loading-list">
              <span><strong>{analysis.pca.components[xComponent].name}</strong></span>
              {xLoadings.map(([feature, value]) => (
                <span key={`x-${feature}`}>{metricLabel(feature)} <strong>{fixed(value, 2)}</strong></span>
              ))}
            </div>
            <div className="analysis-loading-list">
              <span><strong>{analysis.pca.components[yComponent].name}</strong></span>
              {yLoadings.map(([feature, value]) => (
                <span key={`y-${feature}`}>{metricLabel(feature)} <strong>{fixed(value, 2)}</strong></span>
              ))}
            </div>
          </div>
        </section>
        <section className="analysis-pca-side-card analysis-pca-compact-card">
          <h3>Cluster size</h3>
          <div className="analysis-cluster-list">
            {Object.entries(clusterSizes).map(([cluster, count]) => (
              <button
                key={cluster}
                type="button"
                className="analysis-cluster-chip"
                onClick={() => {
                  const clusterIndex = Number(cluster.replace("C", "")) - 1;
                  const ids = clusterResult.idsByCluster.get(clusterIndex) ?? [];
                  if (ids.length > 0) previewIds(ids, `Cluster ${clusterIndex + 1}`);
                }}
              >
                <span>{cluster}</span>
                <strong>{count.toLocaleString()}</strong>
              </button>
            ))}
          </div>
        </section>
      </div>
      <div className="analysis-control-row analysis-control-row-grid analysis-control-row-bottom">
        <label>
          Depth
          <input
            type="range"
            min={1}
            max={componentCount}
            value={depth}
            onChange={(event) => setDepth(Number(event.target.value))}
          />
          <strong>{depth}</strong>
        </label>
        <label>
          Clusters
          <select
            value={String(clusterCount)}
            onChange={(event) =>
              setClusterCount(event.target.value === "auto" ? "auto" : Number(event.target.value))
            }
          >
            <option value="auto">Auto ({effectiveClusterCount})</option>
            {clusterOptions.map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          Color
          <select value={colorMode} onChange={(event) => setColorMode(event.target.value as ColorMode)}>
            <option value="cluster">Cluster</option>
            <option value="polarity">Polarity</option>
            <option value="source_kind">Source</option>
            <option value="direction">Direction</option>
          </select>
        </label>
        <label>
          X
          <select value={xComponent} onChange={(event) => setXComponent(Number(event.target.value))}>
            {analysis.pca.components.map((component, index) => (
              <option key={component.name} value={index}>{component.name}</option>
            ))}
          </select>
        </label>
        <label>
          Y
          <select value={yComponent} onChange={(event) => setYComponent(Number(event.target.value))}>
            {analysis.pca.components.map((component, index) => (
              <option key={component.name} value={index}>{component.name}</option>
            ))}
          </select>
        </label>
      </div>
    </section>
  );
}

function SelectedDataTable({ analysis }: { analysis: DatabaseAnalysisResponse }) {
  return (
    <section className="analysis-chart-panel analysis-data-panel">
      <div className="analysis-panel-header">
        <div>
          <Table2 size={16} />
          <h2>Selected data table</h2>
        </div>
        <span className="analysis-subtle">
          {Math.min(analysis.samples.length, 1000).toLocaleString()} / {analysis.count.toLocaleString()}
        </span>
      </div>
      <div className="analysis-data-table-wrap">
        <table className="analysis-data-table">
          <thead>
            <tr>
              <th>Test time</th>
              <th>Curve</th>
              <th>Polarity</th>
              <th>Direction</th>
              <th>Vth</th>
              <th>Ion</th>
              <th>Ioff</th>
              <th>SS</th>
              <th>Hysteresis</th>
            </tr>
          </thead>
          <tbody>
            {analysis.samples.slice(0, 1000).map((sample) => (
              <tr key={sample.curve_id}>
                <td>{sample.test_time?.replace("T", " ").slice(0, 19) ?? "—"}</td>
                <td>{sample.curve_id}</td>
                <td>{sample.polarity}</td>
                <td>{sample.direction}</td>
                <td>{fixed(sample.vth, 3)}</td>
                <td>{scientific(sample.ion)}</td>
                <td>{scientific(sample.ioff)}</td>
                <td>{fixed(sample.ss_mv_dec, 1)}</td>
                <td>{sample.hysteresis_v === null ? "NA" : fixed(sample.hysteresis_v, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function AnalysisWorkspace({ selection }: { selection: DatabaseSelectionState }) {
  const [analysis, setAnalysis] = useState<DatabaseAnalysisResponse | null>(null);
  const [histogramMetric, setHistogramMetric] = useState("logRatio");
  const [scatterXMetric, setScatterXMetric] = useState("vth");
  const [scatterYMetric, setScatterYMetric] = useState("logRatio");
  const [previewRequest, setPreviewRequest] = useState<PreviewRequest>(EMPTY_PREVIEW);
  const [loading, setLoading] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState<DatabaseAnalysisStatus | null>(null);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const effectiveSelection = useMemo<DatabaseSelectionState | null>(() => {
    if (selection.allFiltered || selection.selectedIds.length > 0) return selection;
    return null;
  }, [selection]);
  const selectionRequestKey = useMemo(() => {
    if (effectiveSelection === null) return "none";
    return JSON.stringify({
      allFiltered: effectiveSelection.allFiltered,
      total: effectiveSelection.total,
      selectedIds: effectiveSelection.selectedIds,
      filters: effectiveSelection.filters
    });
  }, [effectiveSelection]);

  useEffect(() => {
    if (effectiveSelection === null) {
      setAnalysis(null);
      setLoading(false);
      setAnalysisStatus(null);
      setError(null);
      setPreviewRequest(EMPTY_PREVIEW);
      return undefined;
    }
    let active = true;
    let timer: number | null = null;
    setLoading(true);
    setAnalysis(null);
    setAnalysisStatus(null);
    setError(null);

    const applyAnalysis = (response: DatabaseAnalysisResponse) => {
      setAnalysis(response);
      setPreviewRequest({
        title: "Curve data preview",
        detail: `${Math.min(response.samples.length, PREVIEW_CURVE_LIMIT)} filtered curves`,
        ids: response.samples.slice(0, PREVIEW_CURVE_LIMIT).map((sample) => sample.curve_id)
      });
      setHistogramMetric((current) =>
        response.metrics[current] ? current : Object.keys(response.metrics)[0] ?? "logRatio"
      );
    };

    const refreshStatus = async () => {
      const nextStatus = await getDatabaseAnalysisStatus();
      if (!active) return;
      setAnalysisStatus(nextStatus);
      if (nextStatus.status === "completed") {
        if (timer !== null) window.clearInterval(timer);
        if (nextStatus.result) {
          applyAnalysis(nextStatus.result);
          setLoading(false);
          return;
        }
        setError("Analysis completed without a result payload");
        setLoading(false);
        return;
      }
      if (nextStatus.status === "failed") {
        if (timer !== null) window.clearInterval(timer);
        setError(nextStatus.error ?? nextStatus.message ?? "Could not analyze database selection");
        setLoading(false);
      }
    };

    const beginPolling = () => {
      void refreshStatus().catch((caught) => {
        if (!active) return;
        setLoading(false);
        setError(
          caught instanceof Error ? caught.message : "Could not read analysis progress"
        );
      });
      timer = window.setInterval(() => {
        void refreshStatus().catch((caught) => {
          if (!active) return;
          if (timer !== null) window.clearInterval(timer);
          setLoading(false);
          setError(
            caught instanceof Error ? caught.message : "Could not read analysis progress"
          );
        });
      }, 700);
    };

    void getDatabaseAnalysisStatus()
      .then((currentStatus) => {
        if (!active) return;
        if (currentStatus.status === "running") {
          setAnalysisStatus(currentStatus);
          beginPolling();
          return;
        }
        return startDatabaseAnalysis(effectiveSelection).then((status) => {
          if (!active) return;
          setAnalysisStatus(status);
          beginPolling();
        });
      })
      .catch((caught) => {
        if (!active) return;
        if (caught instanceof ApiError && caught.status === 409) {
          beginPolling();
          return;
        }
        setLoading(false);
        setError(caught instanceof Error ? caught.message : "Could not start database analysis");
      });

    return () => {
      active = false;
      if (timer !== null) window.clearInterval(timer);
    };
  }, [selectionRequestKey]);

  async function exportSelection() {
    if (effectiveSelection === null) return;
    setExporting(true);
    setError(null);
    try {
      await exportDatabaseSelection(effectiveSelection);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not export database selection");
    } finally {
      setExporting(false);
    }
  }

  return (
    <main className="analysis-workspace">
      <section className="analysis-heading">
        <div>
          <Activity size={22} />
          <h1>Analysis</h1>
        </div>
        <div className="analysis-actions">
          <span>
            {effectiveSelection === null
              ? "Waiting for database selection"
              : effectiveSelection.allFiltered
              ? `${effectiveSelection.total.toLocaleString()} filtered curves`
              : `${effectiveSelection.selectedIds.length.toLocaleString()} selected curves`}
          </span>
          {analysis ? (
            <span>
              Based on current Database selection
              {analysis.sample_count < analysis.count
                ? ` · charts use ${analysis.sample_count.toLocaleString()} sampled curves`
                : ""}
            </span>
          ) : null}
          {analysis && analysis.sample_count < analysis.count ? (
            <span>{analysis.sample_count.toLocaleString()} plotted samples</span>
          ) : null}
          <button
            className="button secondary compact"
            onClick={exportSelection}
            disabled={exporting || effectiveSelection === null}
          >
            <Download size={15} />
            {exporting ? "Exporting" : "Export"}
          </button>
        </div>
      </section>
      {error ? <div className="error-banner" role="alert">{error}</div> : null}
      {effectiveSelection === null ? (
        <div className="analysis-loading">
          Select curves in Database before running analysis.
        </div>
      ) : null}
      {loading ? (
        <AnalysisLoadingPanel
          selectionCount={effectiveSelection?.allFiltered ? effectiveSelection.total : effectiveSelection?.selectedIds.length ?? 0}
          status={analysisStatus}
        />
      ) : null}
      {analysis ? (
        <div className="analysis-layout">
          <div className="analysis-six-grid">
            <PreviewPanel request={previewRequest} />
            <HistogramExplorer
              analysis={analysis}
              metric={histogramMetric}
              onMetricChange={setHistogramMetric}
              onPreview={setPreviewRequest}
            />
            <ScatterExplorer
              analysis={analysis}
              xMetric={scatterXMetric}
              yMetric={scatterYMetric}
              onXMetricChange={setScatterXMetric}
              onYMetricChange={setScatterYMetric}
              onPreview={setPreviewRequest}
            />
            <CorrelationPanel analysis={analysis} />
            <PcaExplorer analysis={analysis} onPreview={setPreviewRequest} />
            <CorrelationNetworkPanel analysis={analysis} />
          </div>
        </div>
      ) : null}
    </main>
  );
}
