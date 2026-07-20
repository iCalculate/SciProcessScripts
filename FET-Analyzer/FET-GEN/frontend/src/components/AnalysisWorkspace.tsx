import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";
import type { Data, Layout, PlotMouseEvent, PlotlyHTMLElement, PlotSelectionEvent } from "plotly.js";
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
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type InputHTMLAttributes,
  type MouseEvent as ReactMouseEvent
} from "react";
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
const PlotlyApi = Plotly as unknown as typeof import("plotly.js");

const PREVIEW_CURVE_LIMIT = 32;
const ANALYSIS_SCATTER_RENDER_LIMIT = 1_600;
const ANALYSIS_PCA_CLUSTER_LIMIT = 900;
const EMPTY_PREVIEW = {
  title: "Preview",
  detail: "No active selection",
  ids: [] as string[],
  focusCurveId: null as string | null
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

type PcaMarkerCustomData = [string, string, string];
type PcaHullCustomData = ["__cluster__", string];

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

function cross(origin: { x: number; y: number }, left: { x: number; y: number }, right: { x: number; y: number }) {
  return (left.x - origin.x) * (right.y - origin.y) - (left.y - origin.y) * (right.x - origin.x);
}

function convexHull(points: { x: number; y: number }[]) {
  if (points.length <= 1) return points;
  const sorted = [...points].sort((left, right) => (left.x === right.x ? left.y - right.y : left.x - right.x));
  const lower: { x: number; y: number }[] = [];
  sorted.forEach((point) => {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  });
  const upper: { x: number; y: number }[] = [];
  [...sorted].reverse().forEach((point) => {
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  });
  lower.pop();
  upper.pop();
  return [...lower, ...upper];
}

function circlePolygon(centerX: number, centerY: number, radiusX: number, radiusY: number, segments = 18) {
  return Array.from({ length: segments }, (_, index) => {
    const angle = (Math.PI * 2 * index) / segments;
    return {
      x: centerX + radiusX * Math.cos(angle),
      y: centerY + radiusY * Math.sin(angle)
    };
  });
}

function clusterEnvelope(
  points: { x: number; y: number }[],
  xSpan: number,
  ySpan: number
) {
  if (points.length === 0) return [];
  const safeXSpan = Math.max(Math.abs(xSpan), 1);
  const safeYSpan = Math.max(Math.abs(ySpan), 1);
  const radiusX = safeXSpan * 0.035;
  const radiusY = safeYSpan * 0.035;
  if (points.length === 1) {
    const [point] = points;
    return circlePolygon(point.x, point.y, radiusX, radiusY);
  }
  if (points.length === 2) {
    const centerX = 0.5 * (points[0]!.x + points[1]!.x);
    const centerY = 0.5 * (points[0]!.y + points[1]!.y);
    const dx = points[1]!.x - points[0]!.x;
    const dy = points[1]!.y - points[0]!.y;
    const length = Math.max(Math.hypot(dx, dy), Number.EPSILON);
    const nx = -dy / length;
    const ny = dx / length;
    return [
      { x: points[0]!.x + nx * radiusX, y: points[0]!.y + ny * radiusY },
      { x: centerX + nx * radiusX * 1.2, y: centerY + ny * radiusY * 1.2 },
      { x: points[1]!.x + nx * radiusX, y: points[1]!.y + ny * radiusY },
      { x: points[1]!.x - nx * radiusX, y: points[1]!.y - ny * radiusY },
      { x: centerX - nx * radiusX * 1.2, y: centerY - ny * radiusY * 1.2 },
      { x: points[0]!.x - nx * radiusX, y: points[0]!.y - ny * radiusY }
    ];
  }
  return convexHull(points);
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

function paletteAt(index: number): string {
  return PALETTE[((index % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

function hexToRgb(hex: string) {
  const normalized = hex.replace("#", "");
  const safe = normalized.length === 3
    ? normalized.split("").map((char) => `${char}${char}`).join("")
    : normalized;
  const value = Number.parseInt(safe, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255
  };
}

function rgba(hex: string, alpha: number) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function interpolateHex(left: string, right: string, t: number) {
  const clamped = Math.max(0, Math.min(1, t));
  const a = hexToRgb(left);
  const b = hexToRgb(right);
  const r = Math.round(a.r + (b.r - a.r) * clamped);
  const g = Math.round(a.g + (b.g - a.g) * clamped);
  const bValue = Math.round(a.b + (b.b - a.b) * clamped);
  return `rgb(${r}, ${g}, ${bValue})`;
}

function correlationColor(value: number) {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped < 0) return interpolateHex("#1d5fd1", "#fbfdff", clamped + 1);
  return interpolateHex("#fbfdff", "#cb2f69", clamped);
}

function sequentialGradientColor(index: number, total: number) {
  if (total <= 1) return "#2d7cff";
  return interpolateHex("#7fb2ff", "#1459cb", index / (total - 1));
}

function multiplySquareMatrix(left: number[][], right: number[][]) {
  const size = left.length;
  return Array.from({ length: size }, (_, rowIndex) =>
    Array.from({ length: size }, (_, columnIndex) => {
      let sum = 0;
      for (let pivot = 0; pivot < size; pivot += 1) {
        sum += (left[rowIndex]?.[pivot] ?? 0) * (right[pivot]?.[columnIndex] ?? 0);
      }
      return sum;
    })
  );
}

function correlationMatrixForOrder(matrix: (number | null)[][], order: number) {
  const base = matrix.map((row, rowIndex) =>
    row.map((value, columnIndex) => (rowIndex === columnIndex ? 0 : value ?? 0))
  );
  if (order <= 1) return base;
  let powered = base.map((row) => [...row]);
  for (let step = 1; step < order; step += 1) {
    powered = multiplySquareMatrix(powered, base);
  }
  const maxAbs = Math.max(
    1e-9,
    ...powered.flatMap((row, rowIndex) =>
      row.map((value, columnIndex) => (rowIndex === columnIndex ? 0 : Math.abs(value)))
    )
  );
  return powered.map((row, rowIndex) =>
    row.map((value, columnIndex) => (rowIndex === columnIndex ? 0 : value / maxAbs))
  );
}

function quantile(values: number[], fraction: number) {
  if (values.length === 0) return null;
  const ordered = [...values].sort((left, right) => left - right);
  const position = (ordered.length - 1) * fraction;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return ordered[lower] ?? null;
  const mix = position - lower;
  return (ordered[lower] ?? 0) * (1 - mix) + (ordered[upper] ?? 0) * mix;
}

function numericInputValue(value: number | null, fallback: number | null): string {
  if (value !== null && Number.isFinite(value)) return String(value);
  if (fallback !== null && Number.isFinite(fallback)) return String(fallback);
  return "";
}

function parseNumericInput(value: string): number | null {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function expandedRange(min: number | null, max: number | null, paddingRatio = 0.08) {
  if (min === null || max === null || !Number.isFinite(min) || !Number.isFinite(max)) return undefined;
  if (min === max) {
    const padding = Math.max(Math.abs(min) * paddingRatio, 1);
    return [min - padding, max + padding] as [number, number];
  }
  const padding = (max - min) * paddingRatio;
  return [min - padding, max + padding] as [number, number];
}

function clampInteger(value: number | null, min: number, max: number) {
  if (value === null) return null;
  return Math.max(min, Math.min(max, Math.round(value)));
}

function clampPositive(value: number | null) {
  if (value === null || value <= 0) return null;
  return value;
}

function CommitNumberInput({
  value,
  fallback = null,
  onCommit,
  normalize,
  ...props
}: Omit<InputHTMLAttributes<HTMLInputElement>, "type" | "value" | "onChange"> & {
  value: number | null;
  fallback?: number | null;
  onCommit: (value: number | null) => void;
  normalize?: (value: number | null) => number | null;
}) {
  const [draft, setDraft] = useState(() => numericInputValue(value, fallback));
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (!focused) setDraft(numericInputValue(value, fallback));
  }, [fallback, focused, value]);

  function commit() {
    const parsed = parseNumericInput(draft);
    onCommit(normalize ? normalize(parsed) : parsed);
    setFocused(false);
  }

  return (
    <input
      {...props}
      type="number"
      value={draft}
      onFocus={() => setFocused(true)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          event.currentTarget.blur();
          return;
        }
        if (event.key === "Escape") {
          setDraft(numericInputValue(value, fallback));
          event.preventDefault();
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function numericBounds(values: number[]) {
  if (values.length === 0) return { min: null as number | null, max: null as number | null };
  return {
    min: Math.min(...values),
    max: Math.max(...values)
  };
}

function resolvedRange(min: number | null, max: number | null): [number, number] | undefined {
  return min !== null && max !== null && max > min ? [min, max] : undefined;
}

function formatRangeLabel(metric: string, start: number, end: number) {
  return `${metricLabel(metric)} ${metricValue(metric, start)} to ${metricValue(metric, end)}`;
}

function plotConfig(
  selectable = false,
  options?: {
    scrollZoom?: boolean;
    doubleClick?: false | "reset+autosize" | "reset" | "autosize";
  }
) {
  void selectable;
  return {
    responsive: true,
    displaylogo: false,
    displayModeBar: false,
    editable: false,
    scrollZoom: options?.scrollZoom ?? false,
    doubleClick: options?.doubleClick ?? "reset+autosize"
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
  const { min: dataMin, max: dataMax } = useMemo(() => numericBounds(values), [values]);
  const [xMin, setXMin] = useState<number | null>(null);
  const [xMax, setXMax] = useState<number | null>(null);
  const [binWidth, setBinWidth] = useState<number | null>(null);
  const effectiveXMin = xMin ?? dataMin;
  const effectiveXMax = xMax ?? dataMax;
  const bins = useMemo(
    () => histogramByWidth(values, effectiveXMin, effectiveXMax, binWidth),
    [binWidth, effectiveXMax, effectiveXMin, values]
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
      ids,
      focusCurveId: null
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
          <label>X min<CommitNumberInput value={xMin} fallback={dataMin} onCommit={setXMin} /></label>
          <label>X max<CommitNumberInput value={xMax} fallback={dataMax} onCommit={setXMax} /></label>
          <label>Bin width<CommitNumberInput value={binWidth} min={0} placeholder="auto" onCommit={setBinWidth} normalize={clampPositive} /></label>
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
  const ids = useMemo(() => {
    const ordered = request.focusCurveId
      ? [request.focusCurveId, ...request.ids.filter((id) => id !== request.focusCurveId)]
      : request.ids;
    return [...new Set(ordered)].slice(0, PREVIEW_CURVE_LIMIT);
  }, [request.focusCurveId, request.ids]);

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
  }, [request.ids, request.focusCurveId]);

  const allX = useMemo(
    () => details.flatMap((detail) => detail.raw_points.map((point) => point.voltage_v)),
    [details]
  );
  const { min: previewXMin, max: previewXMax } = useMemo(() => numericBounds(allX), [allX]);
  const effectiveXMin = xMin ?? previewXMin;
  const effectiveXMax = xMax ?? previewXMax;

  const activeCurveId = hoveredCurveId ?? request.focusCurveId;
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
        activeCurveId === detail.curve_id
          ? 3.2
          : details.length > 10
            ? 1.2
            : 1.8
    },
    opacity:
      activeCurveId === null || activeCurveId === detail.curve_id
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
      range: resolvedRange(effectiveXMin, effectiveXMax)
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
          <label>X min<CommitNumberInput value={xMin} fallback={previewXMin} onCommit={setXMin} /></label>
          <label>X max<CommitNumberInput value={xMax} fallback={previewXMax} onCommit={setXMax} /></label>
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
  const axisMenuRef = useRef<HTMLDivElement | null>(null);
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
  const { min: dataXMin, max: dataXMax } = useMemo(
    () => numericBounds(renderPoints.map((point) => point.x)),
    [renderPoints]
  );
  const { min: dataYMin, max: dataYMax } = useMemo(
    () => numericBounds(renderPoints.map((point) => point.y)),
    [renderPoints]
  );
  const effectiveXMin = xMin ?? dataXMin;
  const effectiveXMax = xMax ?? dataXMax;
  const effectiveYMin = yMin ?? dataYMin;
  const effectiveYMax = yMax ?? dataYMax;
  useEffect(() => {
    setOpenAxisMenu(null);
    setXMin(null);
    setXMax(null);
    setYMin(null);
    setYMax(null);
  }, [xMetric, yMetric]);
  useEffect(() => {
    if (openAxisMenu === null) return undefined;
    const handlePointerDown = (event: PointerEvent) => {
      if (!axisMenuRef.current?.contains(event.target as Node)) setOpenAxisMenu(null);
    };
    window.addEventListener("pointerdown", handlePointerDown);
    return () => window.removeEventListener("pointerdown", handlePointerDown);
  }, [openAxisMenu]);
  const filteredPoints = useMemo(
    () =>
      renderPoints.filter((point) => {
        if (effectiveXMin !== null && point.x < effectiveXMin) return false;
        if (effectiveXMax !== null && point.x > effectiveXMax) return false;
        if (effectiveYMin !== null && point.y < effectiveYMin) return false;
        if (effectiveYMax !== null && point.y > effectiveYMax) return false;
        return true;
      }),
    [effectiveXMax, effectiveXMin, effectiveYMax, effectiveYMin, renderPoints]
  );
  const groups = useMemo(() => {
    const buckets = new Map<string, typeof filteredPoints>();
    filteredPoints.forEach((point) => {
      const label = String(point.sample[colorBy]);
      buckets.set(label, [...(buckets.get(label) ?? []), point]);
    });
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [colorBy, filteredPoints]);
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
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    customdata: points.map((point) => point.sample.curve_id),
    name: label,
    marker: {
      color: colorFor(label),
      size: 7.5,
      opacity: 0.74,
      line: { color: "#ffffff", width: 0.6 }
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
      showlegend: false,
      cliponaxis: false
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
    legend: {
      orientation: "h",
      x: 0,
      y: -0.16,
      yanchor: "top",
      font: { size: 9 },
      bgcolor: "rgba(255,255,255,0.82)"
    },
    dragmode: "select",
    xaxis: {
      domain: [0, 0.78],
      title: { text: "", standoff: 0 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      zeroline: false,
      ticks: "outside",
      range: resolvedRange(effectiveXMin, effectiveXMax)
    },
    yaxis: {
      domain: [0, 0.72],
      title: { text: "", standoff: 0 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      zeroline: false,
      ticks: "outside",
      range: resolvedRange(effectiveYMin, effectiveYMax)
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
      ids,
      focusCurveId: null
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
        <div className="analysis-control-row analysis-control-row-scatter">
          <label>X range
            <div className="analysis-range-inline">
              <CommitNumberInput value={xMin} fallback={dataXMin} onCommit={setXMin} />
              <span>to</span>
              <CommitNumberInput value={xMax} fallback={dataXMax} onCommit={setXMax} />
            </div>
          </label>
          <label>Y range
            <div className="analysis-range-inline">
              <CommitNumberInput value={yMin} fallback={dataYMin} onCommit={setYMin} />
              <span>to</span>
              <CommitNumberInput value={yMax} fallback={dataYMax} onCommit={setYMax} />
            </div>
          </label>
          <label className="analysis-color-pill">
            <span>Color</span>
            <select value={colorBy} onChange={(event) => setColorBy(event.target.value as typeof colorBy)}>
              <option value="polarity">Polarity</option>
              <option value="source_kind">Source</option>
              <option value="direction">Direction</option>
            </select>
          </label>
        </div>
      </div>
      <div ref={axisMenuRef} className="analysis-scatter-shell">
        <div className="analysis-axis-trigger analysis-axis-trigger-y">
          <button type="button" onClick={() => setOpenAxisMenu((current) => current === "y" ? null : "y")}>
            {metricLabel(yMetric)}
          </button>
          {openAxisMenu === "y" ? (
            <div className="analysis-axis-menu analysis-axis-menu-y">
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
        <div className="analysis-axis-trigger analysis-axis-trigger-x">
          <button type="button" onClick={() => setOpenAxisMenu((current) => current === "x" ? null : "x")}>
            {metricLabel(xMetric)}
          </button>
          {openAxisMenu === "x" ? (
            <div className="analysis-axis-menu analysis-axis-menu-x">
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
  const [correlationOrder, setCorrelationOrder] = useState(1);

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
  const orderedMatrix = useMemo(
    () =>
      orderedIndexes.map((rowIndex) =>
        orderedIndexes.map((columnIndex) => analysis.correlations.matrix[rowIndex]?.[columnIndex] ?? null)
      ),
    [analysis.correlations.matrix, orderedIndexes]
  );
  const poweredMatrix = useMemo(
    () => correlationMatrixForOrder(orderedMatrix, correlationOrder),
    [correlationOrder, orderedMatrix]
  );
  const squareSize = Math.max(18, Math.min(28, 260 / Math.max(orderedFeatures.length, 1)));
  const diagonalCells = orderedFeatures.map((feature) => ({
    x: metricLabel(feature),
    y: metricLabel(feature),
    text: '1.00'
  }));
  const upperTriangleCells = orderedFeatures.flatMap((rowFeature, rowIndex) =>
    orderedFeatures.flatMap((columnFeature, columnIndex) => {
      if (columnIndex <= rowIndex) return [];
      const value = poweredMatrix[rowIndex]?.[columnIndex] ?? 0;
      return [{
        x: metricLabel(columnFeature),
        y: metricLabel(rowFeature),
        value
      }];
    })
  );
  const offDiagonalCells = orderedFeatures.flatMap((rowFeature, rowIndex) =>
    orderedFeatures.flatMap((columnFeature, columnIndex) => {
      if (columnIndex >= rowIndex) return [];
      const value = poweredMatrix[rowIndex]?.[columnIndex] ?? 0;
      return [{
        x: metricLabel(columnFeature),
        y: metricLabel(rowFeature),
        value,
        text: Math.abs(value) >= 0.68 ? fixed(value, 2) : ''
      }];
    })
  );

  const heatmapData: Data[] = [
    {
      type: 'scatter',
      mode: 'markers',
      x: diagonalCells.map((cell) => cell.x),
      y: diagonalCells.map((cell) => cell.y),
      marker: {
        symbol: 'square',
        size: squareSize,
        color: '#eef2f7',
        line: { color: '#ffffff', width: 1 }
      },
      hoverinfo: 'skip',
      showlegend: false
    } as Data,
    {
      type: 'scatter',
      mode: 'markers',
      x: upperTriangleCells.map((cell) => cell.x),
      y: upperTriangleCells.map((cell) => cell.y),
      marker: {
        symbol: 'circle',
        size: upperTriangleCells.map((cell) => 4 + Math.abs(cell.value) * (squareSize - 6)),
        color: upperTriangleCells.map((cell) =>
          cell.value >= 0 ? 'rgba(203, 47, 105, 0.68)' : 'rgba(29, 95, 209, 0.68)'
        ),
        line: {
          color: '#ffffff',
          width: 0.8
        }
      },
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>|value| = %{marker.size:.1f}px<br>value = %{customdata:.3f}<extra></extra>',
      customdata: upperTriangleCells.map((cell) => cell.value),
      showlegend: false
    } as Data,
    {
      type: 'scatter',
      mode: 'markers',
      x: offDiagonalCells.map((cell) => cell.x),
      y: offDiagonalCells.map((cell) => cell.y),
      marker: {
        symbol: 'square',
        size: squareSize,
        color: offDiagonalCells.map((cell) => cell.value),
        cmin: -1,
        cmax: 1,
        colorscale: [
          [0, '#1d5fd1'],
          [0.5, '#ffffff'],
          [1, '#cb2f69']
        ],
        line: { color: '#ffffff', width: 1 },
        colorbar: {
          title: { text: correlationOrder === 1 ? 'r' : `order ${correlationOrder}`, side: 'right' },
          thickness: 12,
          len: 0.82,
          tickvals: [-1, -0.5, 0, 0.5, 1]
        }
      },
      hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>value = %{marker.color:.3f}<extra></extra>',
      showlegend: false
    } as Data,
    {
      type: 'scatter',
      mode: 'text',
      x: [
        ...diagonalCells.map((cell) => cell.x),
        ...offDiagonalCells.filter((cell) => cell.text).map((cell) => cell.x)
      ],
      y: [
        ...diagonalCells.map((cell) => cell.y),
        ...offDiagonalCells.filter((cell) => cell.text).map((cell) => cell.y)
      ],
      text: [
        ...diagonalCells.map((cell) => cell.text),
        ...offDiagonalCells.filter((cell) => cell.text).map((cell) => cell.text)
      ],
      textfont: { size: 9, color: '#20334d' },
      hoverinfo: 'skip',
      showlegend: false
    } as Data
  ];

  const heatmapLayout: Partial<Layout> = {
    autosize: true,
    margin: { l: 92, r: 56, t: 18, b: 84 },
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#fbfdff',
    font: { family: 'Segoe UI, system-ui, sans-serif', color: '#263a55', size: 10 },
    xaxis: {
      side: 'top',
      tickangle: -35,
      automargin: true,
      tickfont: { size: 10 },
      showgrid: false,
      zeroline: false
    },
    yaxis: {
      automargin: true,
      autorange: 'reversed',
      tickfont: { size: 10 },
      scaleanchor: 'x',
      scaleratio: 1,
      showgrid: false,
      zeroline: false
    }
  };

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
        <label className="analysis-correlation-order">
          <span>Order</span>
          <CommitNumberInput
            min={1}
            max={6}
            step={1}
            value={correlationOrder}
            onCommit={(value) => setCorrelationOrder(clampInteger(value, 1, 6) ?? 1)}
          />
        </label>
      </div>
      <div className="analysis-correlation-layout analysis-correlation-layout-simple">
        <div className="analysis-correlation-plot-wrap">
          <Plot
            data={heatmapData}
            layout={heatmapLayout}
            config={plotConfig(false)}
            useResizeHandler
            className="analysis-plot analysis-correlation-plot"
          />
        </div>
      </div>
      <div className="analysis-inline-stats analysis-inline-stats-wrap">
        <span>Diagonal self-correlation is shown in light gray.</span>
        <span>Zero correlation is white. The colorbar stays symmetric from -1 to 1.</span>
      </div>
    </section>
  );
}
function BinnedViolinPanel({
  analysis,
  onPreview
}: {
  analysis: DatabaseAnalysisResponse;
  onPreview: (request: PreviewRequest) => void;
}) {
  const metricOptions = useMemo(
    () => RAW_METRIC_OPTIONS.filter((option) => analysis.metrics[option]),
    [analysis.metrics]
  );
  const [xMetric, setXMetric] = useState("vth");
  const [yMetric, setYMetric] = useState("logRatio");
  const [binCount, setBinCount] = useState(6);
  const allPairs = useMemo(
    () =>
      analysis.samples.flatMap((sample) => {
        const x = sampleMetric(sample, xMetric);
        const y = sampleMetric(sample, yMetric);
        return x === null || y === null ? [] : [{ sample, x, y }];
      }),
    [analysis.samples, xMetric, yMetric]
  );
  const { min: dataXMin, max: dataXMax } = useMemo(
    () => numericBounds(allPairs.map((pair) => pair.x)),
    [allPairs]
  );
  const { min: dataYMin, max: dataYMax } = useMemo(
    () => numericBounds(allPairs.map((pair) => pair.y)),
    [allPairs]
  );
  const [xMin, setXMin] = useState<number | null>(null);
  const [xMax, setXMax] = useState<number | null>(null);
  const [yMin, setYMin] = useState<number | null>(null);
  const [yMax, setYMax] = useState<number | null>(null);
  const effectiveXMin = xMin ?? dataXMin;
  const effectiveXMax = xMax ?? dataXMax;
  const effectiveYMin = yMin ?? dataYMin;
  const effectiveYMax = yMax ?? dataYMax;

  useEffect(() => {
    setXMin(null);
    setXMax(null);
    setYMin(null);
    setYMax(null);
  }, [xMetric, yMetric]);

  const bins = useMemo(() => {
    if (
      effectiveXMin === null ||
      effectiveXMax === null ||
      effectiveXMax <= effectiveXMin ||
      binCount < 1
    ) {
      return [] as Array<{
        label: string;
        start: number;
        end: number;
        points: typeof allPairs;
      }>;
    }
    const width = (effectiveXMax - effectiveXMin) / binCount;
    return Array.from({ length: binCount }, (_, index) => {
      const start = effectiveXMin + index * width;
      const end = index === binCount - 1 ? effectiveXMax : start + width;
      const points = allPairs.filter((pair) => {
        if (pair.x < start || pair.x > end) return false;
        if (effectiveYMin !== null && pair.y < effectiveYMin) return false;
        if (effectiveYMax !== null && pair.y > effectiveYMax) return false;
        return index === binCount - 1 ? pair.x <= end : pair.x < end;
      });
      return {
        label: `${fixed(start, 2)}-${fixed(end, 2)}`,
        start,
        end,
        points
      };
    }).filter((bin) => bin.points.length > 0);
  }, [allPairs, binCount, effectiveXMax, effectiveXMin, effectiveYMax, effectiveYMin]);

  const yFloor = effectiveYMin ?? dataYMin ?? 0;
  const yCeiling = effectiveYMax ?? dataYMax ?? 1;
  const violinHalfWidth = Math.max(0.18, Math.min(0.42, 2.2 / Math.max(bins.length, 1)));
  const data: Data[] = bins.flatMap((bin, index) => {
    const color = sequentialGradientColor(index, bins.length);
    const center = index + 1;
    const values = bin.points.map((point) => point.y);
    const density = densityCurve(values, yFloor, yCeiling, violinHalfWidth);
    const q1 = quantile(values, 0.25);
    const median = quantile(values, 0.5);
    const q3 = quantile(values, 0.75);
    const left = density.x.map((_, pointIndex) => center - (density.y[pointIndex] ?? 0));
    const right = density.x.map((_, pointIndex) => center + (density.y[pointIndex] ?? 0)).reverse();
    const outlineX = [...left, ...right];
    const outlineY = [...density.x, ...[...density.x].reverse()];
    const jitteredX = bin.points.map((point, pointIndex) => {
      const seed = `${point.sample.curve_id}:${pointIndex}:${bin.label}`;
      let hash = 0;
      for (const char of seed) hash = (hash * 33 + char.charCodeAt(0)) >>> 0;
      return center + ((((hash % 1000) / 999) * 2 - 1) * violinHalfWidth * 0.82);
    });
    return [
      {
        type: "scatter",
        mode: "lines",
        x: outlineX,
        y: outlineY,
        line: { color, width: 1.6, shape: "spline", smoothing: 0.65 },
        fill: "toself",
        fillcolor: rgba(color, 0.22),
        hoverinfo: "skip",
        showlegend: false
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: [center, center],
        y: [q1 ?? yFloor, q3 ?? yCeiling],
        line: { color: rgba(color, 0.92), width: 4 },
        hoverinfo: "skip",
        showlegend: false
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: [center - violinHalfWidth * 0.48, center + violinHalfWidth * 0.48],
        y: [median ?? yFloor, median ?? yFloor],
        line: { color: rgba(color, 0.96), width: 2.4 },
        hoverinfo: "skip",
        showlegend: false
      } as Data,
      {
        type: "scatter",
        mode: "markers",
        name: bin.label,
        x: jitteredX,
        y: values,
        customdata: bin.points.map((point) => point.sample.curve_id),
        marker: {
          size: 5.5,
          opacity: 0.66,
          color,
          line: { color: "#ffffff", width: 0.7 }
        },
        hovertemplate:
          `%{customdata}<br>${metricLabel(xMetric)} bin ${bin.label}<br>${metricLabel(yMetric)} %{y:.3g}<extra></extra>`,
        showlegend: false
      } as Data
    ];
  });

  const layout = {
    autosize: true,
    margin: { l: 58, r: 16, t: 8, b: 56 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    showlegend: false,
    xaxis: {
      title: { text: `${metricLabel(xMetric)} bins`, standoff: 10 },
      tickangle: -22,
      linecolor: "#9cabc0",
      ticks: "outside",
      tickmode: "array",
      tickvals: bins.map((_, index) => index + 1),
      ticktext: bins.map((bin) => bin.label),
      range: bins.length > 0 ? [0.35, bins.length + 0.65] : undefined
    },
    yaxis: {
      title: { text: metricLabel(yMetric), standoff: 8 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      ticks: "outside",
      range: resolvedRange(effectiveYMin, effectiveYMax)
    },
    uirevision: `violin-${xMetric}-${yMetric}-${binCount}`
  } as Partial<Layout>;

  function previewBin(ids: string[], label: string) {
    if (ids.length === 0) return;
    onPreview({
      title: `${metricLabel(yMetric)} across ${metricLabel(xMetric)} bins`,
      detail: `${label} - ${ids.length.toLocaleString()} curves`,
      ids,
      focusCurveId: null
    });
  }

  function handleClick(event?: Readonly<PlotMouseEvent>) {
    const traceName = event?.points?.[0]?.data?.name;
    if (typeof traceName !== "string") return;
    const bin = bins.find((candidate) => candidate.label === traceName);
    if (!bin) return;
    previewBin(
      bin.points.map((point) => point.sample.curve_id),
      formatRangeLabel(xMetric, bin.start, bin.end)
    );
  }

  return (
    <section className="analysis-chart-panel analysis-violin-panel">
      <div className="analysis-panel-header">
        <div>
          <ChartLine size={16} />
          <h2>Binned violin</h2>
        </div>
        <div className="analysis-control-row analysis-control-row-violin">
          <label>X
            <select value={xMetric} onChange={(event) => setXMetric(event.target.value)}>
              {metricOptions.map((option) => (
                <option key={`violin-x-${option}`} value={option}>{metricLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>Y
            <select value={yMetric} onChange={(event) => setYMetric(event.target.value)}>
              {metricOptions.map((option) => (
                <option key={`violin-y-${option}`} value={option}>{metricLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>X range
            <div className="analysis-range-inline">
              <CommitNumberInput value={xMin} fallback={dataXMin} onCommit={setXMin} />
              <span>to</span>
              <CommitNumberInput value={xMax} fallback={dataXMax} onCommit={setXMax} />
            </div>
          </label>
          <label>Y range
            <div className="analysis-range-inline">
              <CommitNumberInput value={yMin} fallback={dataYMin} onCommit={setYMin} />
              <span>to</span>
              <CommitNumberInput value={yMax} fallback={dataYMax} onCommit={setYMax} />
            </div>
          </label>
          <label>Bins
            <CommitNumberInput
              min={2}
              max={24}
              step={1}
              value={binCount}
              onCommit={(value) => setBinCount(clampInteger(value, 2, 24) ?? 2)}
            />
          </label>
        </div>
      </div>
      <Plot
        data={data}
        layout={layout}
        config={plotConfig(true)}
        useResizeHandler
        className="analysis-plot analysis-violin-plot"
        onClick={handleClick}
      />
      <div className="analysis-inline-stats">
        <span>Bins {bins.length.toLocaleString()}</span>
        <span>Points {allPairs.length.toLocaleString()}</span>
        <span>Visible curves {bins.reduce((sum, bin) => sum + bin.points.length, 0).toLocaleString()}</span>
        <span>Click one violin to preview that bin in the curve plot.</span>
      </div>
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
  const [clusterCountInput, setClusterCountInput] = useState<number | null>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("cluster");
  const [xComponent, setXComponent] = useState(0);
  const [yComponent, setYComponent] = useState(1);
  const [viewportResetKey, setViewportResetKey] = useState(0);
  const lastClusterClickRef = useRef<{ label: string; at: number } | null>(null);
  const lastContextMenuRef = useRef(0);
  const hoverClearTimerRef = useRef<number | null>(null);
  const pcaGraphRef = useRef<PlotlyHTMLElement | null>(null);
  const styledClusterRef = useRef<string | null | undefined>(undefined);

  useEffect(() => {
    setXComponent((current) => Math.min(current, Math.max(0, componentCount - 1)));
    setYComponent((current) => Math.min(Math.max(1, current), Math.max(0, componentCount - 1)));
  }, [componentCount]);

  const clusteringDepth = Math.min(
    componentCount,
    Math.max(2, Math.min(componentCount, Math.max(xComponent, yComponent) + 2))
  );
  const clusterOptions = useMemo(
    () => [2, 3, 4, 5, 6, 8, 10].filter((value) => value <= Math.max(2, plottedPcaPoints.length)),
    [plottedPcaPoints.length]
  );
  const autoClusterCount = useMemo(
    () => chooseAutoClusterCount(plottedPcaPoints, clusteringDepth, clusterOptions),
    [clusteringDepth, plottedPcaPoints, clusterOptions]
  );
  const effectiveClusterCount =
    clusterCountInput !== null && clusterCountInput >= 2
      ? Math.min(clusterCountInput, Math.max(2, plottedPcaPoints.length))
      : autoClusterCount;
  const clusterResult = useMemo(
    () => clusterPca(plottedPcaPoints, clusteringDepth, effectiveClusterCount),
    [clusteringDepth, effectiveClusterCount, plottedPcaPoints]
  );
  const clusterLabelById = useMemo(() => {
    const labels = new Map<string, string>();
    plottedPcaPoints.forEach((point) => {
      const clusterIndex = clusterResult.assignments.get(point.curve_id) ?? 0;
      labels.set(point.curve_id, `Cluster ${clusterIndex + 1}`);
    });
    return labels;
  }, [clusterResult.assignments, plottedPcaPoints]);
  const clusterGroups = useMemo(() => {
    const buckets = new Map<string, AnalysisPcaPoint[]>();
    plottedPcaPoints.forEach((point) => {
      if (point.scores.length <= Math.max(xComponent, yComponent)) return;
      const label = clusterLabelById.get(point.curve_id) ?? "Cluster 1";
      buckets.set(label, [...(buckets.get(label) ?? []), point]);
    });
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [clusterLabelById, plottedPcaPoints, xComponent, yComponent]);
  const grouped = useMemo(() => {
    const buckets = new Map<string, AnalysisPcaPoint[]>();
    plottedPcaPoints.forEach((point) => {
      if (point.scores.length <= Math.max(xComponent, yComponent)) return;
      const label =
        colorMode === "cluster"
          ? clusterLabelById.get(point.curve_id) ?? "Cluster 1"
          : String(point[colorMode]);
      buckets.set(label, [...(buckets.get(label) ?? []), point]);
    });
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [clusterLabelById, colorMode, plottedPcaPoints, xComponent, yComponent]);
  const projectedPoints = useMemo(
    () =>
      plottedPcaPoints
        .filter((point) => point.scores.length > Math.max(xComponent, yComponent))
        .map((point) => ({
          x: point.scores[xComponent] ?? 0,
          y: point.scores[yComponent] ?? 0
        })),
    [plottedPcaPoints, xComponent, yComponent]
  );
  const { min: xMin, max: xMax } = useMemo(
    () => numericBounds(projectedPoints.map((point) => point.x)),
    [projectedPoints]
  );
  const { min: yMin, max: yMax } = useMemo(
    () => numericBounds(projectedPoints.map((point) => point.y)),
    [projectedPoints]
  );
  const baseXRange = useMemo(() => expandedRange(xMin, xMax, 0.08), [xMax, xMin]);
  const baseYRange = useMemo(() => expandedRange(yMin, yMax, 0.08), [yMax, yMin]);
  const hullTraceDefs = useMemo(() => {
    const spanX = (xMax ?? 1) - (xMin ?? 0);
    const spanY = (yMax ?? 1) - (yMin ?? 0);
    return clusterGroups.flatMap(([label, points]) => {
      const polygon = clusterEnvelope(
        points.map((point) => ({
          x: point.scores[xComponent] ?? 0,
          y: point.scores[yComponent] ?? 0
        })),
        spanX,
        spanY
      );
      if (polygon.length < 3) return [];
      return [{
        label,
        trace: {
          type: "scatter",
          mode: "lines",
          x: [...polygon.map((point) => point.x), polygon[0]!.x],
          y: [...polygon.map((point) => point.y), polygon[0]!.y],
          customdata: [...polygon.map(() => ["__cluster__", label] as PcaHullCustomData), ["__cluster__", label] as PcaHullCustomData],
          name: label,
          hoveron: "fills",
          line: {
            color: rgba(colorFor(label), 0.38),
            width: 1.2,
            shape: "spline",
            smoothing: 0.55
          },
          fill: "toself",
          fillcolor: rgba(colorFor(label), 0.08),
          opacity: 1,
          hovertemplate: `${label}<br>%{x:.3g}, %{y:.3g}<extra>Cluster region</extra>`,
          showlegend: false
        } as Data
      }];
    });
  }, [clusterGroups, xComponent, xMax, xMin, yComponent, yMax, yMin]);

  useEffect(() => {
    if (hoverClearTimerRef.current !== null) {
      window.clearTimeout(hoverClearTimerRef.current);
      hoverClearTimerRef.current = null;
    }
    lastClusterClickRef.current = null;
  }, [colorMode, effectiveClusterCount, xComponent, yComponent, viewportResetKey]);
  useEffect(() => () => {
    if (hoverClearTimerRef.current !== null) window.clearTimeout(hoverClearTimerRef.current);
  }, []);

  const markerTraceDefs = useMemo(() => {
    if (componentCount < 2) {
      return [] as Array<{
        label: string;
        pointClusterLabels: string[];
        trace: Data;
      }>;
    }
    return grouped.map(([label, points]) => ({
      label,
      pointClusterLabels: points.map((point) => clusterLabelById.get(point.curve_id) ?? "Cluster 1"),
      trace: {
        type: "scatter",
        mode: "markers",
        x: points.map((point) => point.scores[xComponent]),
        y: points.map((point) => point.scores[yComponent]),
        customdata: points.map((point) => [
          point.curve_id,
          label,
          clusterLabelById.get(point.curve_id) ?? "Cluster 1"
        ] as PcaMarkerCustomData),
        name: label,
        marker: {
          color: colorFor(label),
          size: 7.4,
          opacity: 0.78,
          line: { color: "rgba(255,255,255,0.72)", width: 0.55 }
        },
        hovertemplate: `%{customdata[0]}<br>${analysis.pca.components[xComponent].name} %{x:.3g}<br>${analysis.pca.components[yComponent].name} %{y:.3g}<extra>%{customdata[2]}</extra>`
      } as Data
    }));
  }, [analysis.pca.components, clusterLabelById, componentCount, grouped, xComponent, yComponent]);

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

  const data: Data[] = [
    ...hullTraceDefs.map((entry) => entry.trace),
    ...markerTraceDefs.map((entry) => entry.trace)
  ];
  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 58, r: 20, t: 8, b: 52 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 10 },
    showlegend: true,
    legend: {
      orientation: "v",
      x: 0.995,
      xanchor: "right",
      y: 0.995,
      yanchor: "top",
      font: { size: 9 },
      bgcolor: "rgba(255,255,255,0.86)",
      bordercolor: "#d8e3ef",
      borderwidth: 1,
      itemsizing: "constant"
    },
    dragmode: "pan",
    hovermode: "closest",
    xaxis: {
      title: {
        text: `${analysis.pca.components[xComponent].name} (${fixed(100 * analysis.pca.components[xComponent].explained_variance_ratio, 1)}%)`,
        standoff: 10
      },
      gridcolor: "#eef2f7",
      zerolinecolor: "#cdd7e5",
      linecolor: "#9cabc0",
      ticks: "outside",
      range: baseXRange
    },
    yaxis: {
      title: {
        text: `${analysis.pca.components[yComponent].name} (${fixed(100 * analysis.pca.components[yComponent].explained_variance_ratio, 1)}%)`,
        standoff: 8
      },
      gridcolor: "#eef2f7",
      zerolinecolor: "#cdd7e5",
      linecolor: "#9cabc0",
      ticks: "outside",
      range: baseYRange
    },
    uirevision: `pca-${clusteringDepth}-${effectiveClusterCount}-${colorMode}-${xComponent}-${yComponent}-${viewportResetKey}`
  };

  function applyPcaHoverStyle(clusterLabel: string | null) {
    const graph = pcaGraphRef.current;
    if (!graph) return;
    if (styledClusterRef.current === clusterLabel) return;
    styledClusterRef.current = clusterLabel;
    const hullCount = hullTraceDefs.length;
    hullTraceDefs.forEach((entry, traceIndex) => {
      const active = clusterLabel !== null && entry.label === clusterLabel;
      void PlotlyApi.restyle(graph, {
        "line.color": rgba(colorFor(entry.label), active ? 0.7 : 0.38),
        "line.width": active ? 2.2 : 1.2,
        fillcolor: rgba(
          colorFor(entry.label),
          active ? 0.14 : clusterLabel === null ? 0.08 : 0.035
        )
      } as unknown as Data, traceIndex);
    });
    markerTraceDefs.forEach((entry, markerIndex) => {
      const traceIndex = hullCount + markerIndex;
      void PlotlyApi.restyle(graph, {
        "marker.opacity": [
          entry.pointClusterLabels.map((pointClusterLabel) =>
            clusterLabel === null ? 0.78 : pointClusterLabel === clusterLabel ? 0.94 : 0.11
          )
        ]
      } as unknown as Data, traceIndex);
    });
  }

  useEffect(() => {
    styledClusterRef.current = undefined;
    applyPcaHoverStyle(null);
  }, [hullTraceDefs, markerTraceDefs, viewportResetKey]);

  function previewIds(ids: string[], title: string, focusCurveId: string | null = null) {
    onPreview({ title, detail: `${ids.length.toLocaleString()} curves`, ids, focusCurveId });
  }

  function idsForCluster(label: string) {
    const clusterIndex = Number.parseInt(label.replace("Cluster ", ""), 10) - 1;
    if (!Number.isFinite(clusterIndex) || clusterIndex < 0) return [];
    return clusterResult.idsByCluster.get(clusterIndex) ?? [];
  }

  function parsePcaEvent(event?: Readonly<PlotMouseEvent>) {
    const customdata = (event?.points?.[0] as { customdata?: unknown } | undefined)?.customdata;
    if (Array.isArray(customdata) && customdata[0] === "__cluster__" && typeof customdata[1] === "string") {
      return { clusterLabel: customdata[1], curveId: null as string | null };
    }
    if (Array.isArray(customdata) && typeof customdata[0] === "string") {
      const clusterLabel =
        typeof customdata[2] === "string"
          ? customdata[2]
          : typeof customdata[1] === "string"
            ? customdata[1]
            : null;
      return { clusterLabel, curveId: customdata[0] };
    }
    return { clusterLabel: null, curveId: null as string | null };
  }

  function handleClick(event?: Readonly<PlotMouseEvent>) {
    const { clusterLabel, curveId } = parsePcaEvent(event);
    if (clusterLabel === null && curveId === null) return;
    if (event?.event?.ctrlKey || event?.event?.metaKey) {
      if (curveId === null) return;
      const clusterIds = clusterLabel ? idsForCluster(clusterLabel) : [];
      previewIds(clusterIds.length > 0 ? clusterIds : [curveId], clusterLabel ?? curveId, curveId);
      return;
    }
    if (clusterLabel === null) return;
    const now = Date.now();
    const lastClick = lastClusterClickRef.current;
    if (lastClick && lastClick.label === clusterLabel && now - lastClick.at < 360) {
      const ids = idsForCluster(clusterLabel);
      if (ids.length > 0) previewIds(ids, clusterLabel);
      lastClusterClickRef.current = null;
      return;
    }
    lastClusterClickRef.current = { label: clusterLabel, at: now };
  }

  function handleRightDoubleClick(event: ReactMouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const now = Date.now();
    if (now - lastContextMenuRef.current < 360) {
      setViewportResetKey((current) => current + 1);
      lastContextMenuRef.current = 0;
      return;
    }
    lastContextMenuRef.current = now;
  }

  return (
    <section className="analysis-chart-panel analysis-pca-panel">
      <div className="analysis-panel-header">
        <div>
          <SlidersHorizontal size={16} />
          <h2>PCA workspace</h2>
        </div>
      </div>
      <div className="analysis-pca-stage" onContextMenu={handleRightDoubleClick}>
        <Plot
          data={data}
          layout={layout}
          config={plotConfig(true, { scrollZoom: true, doubleClick: false })}
          useResizeHandler
          className="analysis-plot analysis-pca-plot"
          onInitialized={(_, graphDiv) => {
            pcaGraphRef.current = graphDiv as PlotlyHTMLElement;
            styledClusterRef.current = undefined;
          }}
          onClick={handleClick}
          onHover={(event) => {
            if (hoverClearTimerRef.current !== null) {
              window.clearTimeout(hoverClearTimerRef.current);
              hoverClearTimerRef.current = null;
            }
            const { clusterLabel } = parsePcaEvent(event);
            applyPcaHoverStyle(clusterLabel);
          }}
          onUnhover={() => {
            if (hoverClearTimerRef.current !== null) window.clearTimeout(hoverClearTimerRef.current);
            hoverClearTimerRef.current = window.setTimeout(() => {
              applyPcaHoverStyle(null);
              hoverClearTimerRef.current = null;
            }, 90);
          }}
        />
      </div>
      <div className="analysis-control-row analysis-control-row-grid analysis-control-row-bottom analysis-control-row-pca">
        <label>
          Clusters
          <CommitNumberInput
            min={2}
            max={Math.max(2, plottedPcaPoints.length)}
            step={1}
            value={clusterCountInput}
            placeholder={`Auto (${autoClusterCount})`}
            onCommit={(value) =>
              setClusterCountInput(clampInteger(value, 2, Math.max(2, plottedPcaPoints.length)))
            }
          />
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
        ids: response.samples.slice(0, PREVIEW_CURVE_LIMIT).map((sample) => sample.curve_id),
        focusCurveId: null
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
                ? ` - charts use ${analysis.sample_count.toLocaleString()} sampled curves`
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
            <BinnedViolinPanel analysis={analysis} onPreview={setPreviewRequest} />
          </div>
        </div>
      ) : null}
    </main>
  );
}

