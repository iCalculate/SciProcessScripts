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
import { useEffect, useMemo, useState } from "react";
import {
  analyzeDatabaseSelection,
  exportDatabaseSelection,
  getDatabaseCurvePreviews
} from "../api";
import { fixed, scientific } from "../format";
import type {
  AnalysisDistributionBin,
  AnalysisMetricStats,
  AnalysisPcaPoint,
  AnalysisSample,
  CurvePreview,
  DatabaseAnalysisResponse,
  DatabaseSelectionState
} from "../types";

const Plot = createPlotlyComponent(Plotly);

const PREVIEW_CURVE_LIMIT = 32;
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
  const density = useMemo(() => {
    if (bins.length === 0) return { x: [] as number[], y: [] as number[] };
    return densityCurve(
      pairs.map((pair) => pair.value),
      bins[0].start,
      bins.at(-1)?.end ?? bins[0].end,
      maxCount
    );
  }, [bins, maxCount, pairs]);

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
      x: density.x,
      y: density.y,
      line: { color: "#df7d10", width: 2 },
      hoverinfo: "skip"
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
        <div className="analysis-control-row">
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

  const traces: Data[] = details.map((detail, index) => ({
    type: "scatter",
    mode: "lines",
    x: detail.raw_points.map((point) => point.voltage_v),
    y: detail.raw_points.map((point) => Math.log10(Math.max(Math.abs(point.current_a), Number.MIN_VALUE))),
    name: sourceName(detail.source_path),
    text: detail.curve_id,
    showlegend: false,
    line: {
      color: colorFor(detail.polarity || String(index)),
      width: details.length > 10 ? 1.2 : 1.8
    },
    opacity: details.length > 10 ? 0.42 : 0.72,
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
      ticks: "outside"
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
        <span className="analysis-subtle">
          {request.ids.length > PREVIEW_CURVE_LIMIT
            ? `${details.length}/${request.ids.length.toLocaleString()} curves`
            : request.detail}
        </span>
      </div>
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
  const correlation = useMemo(() => pearsonFor(analysis.samples, xMetric, yMetric), [
    analysis.samples,
    xMetric,
    yMetric
  ]);
  const groups = useMemo(() => {
    const buckets = new Map<string, typeof correlation.points>();
    correlation.points.forEach((point) => {
      const label = String(point.sample[colorBy]);
      buckets.set(label, [...(buckets.get(label) ?? []), point]);
    });
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [colorBy, correlation.points]);
  const xValues = correlation.points.map((point) => point.x);
  const yValues = correlation.points.map((point) => point.y);
  const xHist = histogramBins(xValues, 24);
  const yHist = histogramBins(yValues, 24);
  const xDensity = xHist.length > 0
    ? densityCurve(xValues, xHist[0].start, xHist.at(-1)?.end ?? xHist[0].end, Math.max(...xHist.map((bin) => bin.count), 1))
    : { x: [] as number[], y: [] as number[] };
  const yDensity = yHist.length > 0
    ? densityCurve(yValues, yHist[0].start, yHist.at(-1)?.end ?? yHist[0].end, Math.max(...yHist.map((bin) => bin.count), 1))
    : { x: [] as number[], y: [] as number[] };

  const scatterTraces: Data[] = groups.map(([label, points]) => ({
    type: "scatter",
    mode: "markers",
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    customdata: points.map((point) => point.sample.curve_id),
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
      line: { color: "#df7d10", width: 1.8 },
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
      line: { color: "#8b5cf6", width: 1.8 },
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
    legend: { orientation: "h", x: 0, y: 1.14, font: { size: 9 } },
    dragmode: "select",
    xaxis: {
      domain: [0, 0.78],
      title: { text: metricLabel(xMetric), standoff: 10 },
      gridcolor: "#eef2f7",
      linecolor: "#9cabc0",
      zeroline: false,
      ticks: "outside",
      range: xMin !== null && xMax !== null && xMax > xMin ? [xMin, xMax] : undefined
    },
    yaxis: {
      domain: [0, 0.72],
      title: { text: metricLabel(yMetric), standoff: 8 },
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
        <div className="analysis-control-row">
          <label>
            X
            <select value={xMetric} onChange={(event) => onXMetricChange(event.target.value)}>
              {METRIC_OPTIONS.map((option) => (
                <option key={option} value={option}>{metricLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>
            Y
            <select value={yMetric} onChange={(event) => onYMetricChange(event.target.value)}>
              {METRIC_OPTIONS.map((option) => (
                <option key={option} value={option}>{metricLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>
            Color
            <select value={colorBy} onChange={(event) => setColorBy(event.target.value as typeof colorBy)}>
              <option value="polarity">Polarity</option>
              <option value="source_kind">Source</option>
              <option value="direction">Direction</option>
            </select>
          </label>
          <label>X min<input type="number" value={xMin ?? ""} onChange={(event) => setXMin(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>X max<input type="number" value={xMax ?? ""} onChange={(event) => setXMax(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>Y min<input type="number" value={yMin ?? ""} onChange={(event) => setYMin(event.target.value ? Number(event.target.value) : null)} /></label>
          <label>Y max<input type="number" value={yMax ?? ""} onChange={(event) => setYMax(event.target.value ? Number(event.target.value) : null)} /></label>
        </div>
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
      <div className="analysis-inline-stats">
        <span>N {correlation.points.length.toLocaleString()}</span>
        <span>r {fixed(correlation.r, 3)}</span>
        <span>R2 {fixed(correlation.r2, 3)}</span>
        <span>Slope {fixed(correlation.slope, 3)}</span>
      </div>
    </section>
  );
}

function CorrelationPanel({ analysis }: { analysis: DatabaseAnalysisResponse }) {
  const labels = analysis.correlations.features.map(metricLabel);
  function cellColor(value: number | null): string {
    if (value === null) return "#f1f5f9";
    const strength = Math.min(1, Math.abs(value));
    const target = value >= 0 ? [23, 105, 255] : [220, 63, 114];
    const channel = (index: number) => Math.round(255 + (target[index] - 255) * strength);
    return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
  }
  return (
    <section className="analysis-chart-panel analysis-correlation-panel">
      <div className="analysis-panel-header">
        <div>
          <Sigma size={16} />
          <h2>Correlation map</h2>
        </div>
      </div>
      <div
        className="analysis-correlation-grid"
        style={{ gridTemplateColumns: `76px repeat(${labels.length}, minmax(22px, 1fr))` }}
      >
        <span />
        {labels.map((label) => <strong key={`x-${label}`}>{label}</strong>)}
        {labels.map((rowLabel, rowIndex) => (
          <div className="analysis-correlation-row" key={rowLabel}>
            <b>{rowLabel}</b>
            {analysis.correlations.matrix[rowIndex].map((value, columnIndex) => (
              <span
                key={`${rowLabel}-${labels[columnIndex]}`}
                style={{ background: cellColor(value), color: value !== null && Math.abs(value) > 0.55 ? "#fff" : "#263a55" }}
                title={`${rowLabel} / ${labels[columnIndex]}: ${value === null ? "NA" : value.toFixed(3)}`}
              >
                {value === null ? "—" : value.toFixed(2)}
              </span>
            ))}
          </div>
        ))}
      </div>
      <div className="analysis-correlation-list">
        {analysis.correlations.strongest.slice(0, 5).map((pair) => (
          <span key={`${pair.x}-${pair.y}`}>
            {metricLabel(pair.x)} / {metricLabel(pair.y)} <strong>{fixed(pair.r, 2)}</strong>
          </span>
        ))}
      </div>
    </section>
  );
}

function clusterPca(points: AnalysisPcaPoint[], depth: number, clusterCount: number) {
  const usable = points.filter((point) => point.scores.length >= depth);
  const assignments = new Map<string, number>();
  const k = Math.min(clusterCount, usable.length);
  if (k <= 0) return assignments;
  if (k === 1) {
    usable.forEach((point) => assignments.set(point.curve_id, 0));
    return assignments;
  }
  const ordered = [...usable].sort((a, b) => a.curve_id.localeCompare(b.curve_id));
  let centroids = Array.from({ length: k }, (_, index) => {
    const source = ordered[Math.round((index * (ordered.length - 1)) / (k - 1))];
    return source.scores.slice(0, depth);
  });
  for (let iteration = 0; iteration < 28; iteration += 1) {
    const buckets = Array.from({ length: k }, () => [] as number[][]);
    usable.forEach((point) => {
      const vector = point.scores.slice(0, depth);
      let bestIndex = 0;
      let bestDistance = Number.POSITIVE_INFINITY;
      centroids.forEach((centroid, centroidIndex) => {
        const distance = vector.reduce((sum, value, valueIndex) => {
          const delta = value - centroid[valueIndex];
          return sum + delta * delta;
        }, 0);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = centroidIndex;
        }
      });
      assignments.set(point.curve_id, bestIndex);
      buckets[bestIndex].push(vector);
    });
    centroids = centroids.map((centroid, centroidIndex) => {
      const bucket = buckets[centroidIndex];
      if (bucket.length === 0) return centroid;
      return centroid.map((_, valueIndex) =>
        bucket.reduce((sum, vector) => sum + vector[valueIndex], 0) / bucket.length
      );
    });
  }
  return assignments;
}

function PcaExplorer({
  analysis,
  onPreview
}: {
  analysis: DatabaseAnalysisResponse;
  onPreview: (request: PreviewRequest) => void;
}) {
  const componentCount = analysis.pca.components.length;
  const [depth, setDepth] = useState(2);
  const [clusterCount, setClusterCount] = useState(3);
  const [colorMode, setColorMode] = useState<ColorMode>("cluster");
  const [xComponent, setXComponent] = useState(0);
  const [yComponent, setYComponent] = useState(1);

  useEffect(() => {
    setDepth((current) => Math.min(Math.max(1, current), Math.max(1, componentCount)));
    setXComponent((current) => Math.min(current, Math.max(0, componentCount - 1)));
    setYComponent((current) => Math.min(Math.max(1, current), Math.max(0, componentCount - 1)));
  }, [componentCount]);

  const clusters = useMemo(
    () => clusterPca(analysis.pca.points, Math.min(depth, componentCount), clusterCount),
    [analysis.pca.points, clusterCount, componentCount, depth]
  );
  const grouped = useMemo(() => {
    const buckets = new Map<string, AnalysisPcaPoint[]>();
    analysis.pca.points.forEach((point) => {
      if (point.scores.length <= Math.max(xComponent, yComponent)) return;
      const label = colorMode === "cluster"
        ? `Cluster ${(clusters.get(point.curve_id) ?? 0) + 1}`
        : String(point[colorMode]);
      buckets.set(label, [...(buckets.get(label) ?? []), point]);
    });
    return [...buckets.entries()].sort(([a], [b]) => a.localeCompare(b));
  }, [analysis.pca.points, clusters, colorMode, xComponent, yComponent]);

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
    uirevision: `pca-${depth}-${clusterCount}-${colorMode}-${xComponent}-${yComponent}`
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
    if (typeof id === "string") previewIds([id], id);
  }

  const clusterSizes = [...clusters.values()].reduce<Record<string, number>>((result, cluster) => {
    const label = `C${cluster + 1}`;
    result[label] = (result[label] ?? 0) + 1;
    return result;
  }, {});

  return (
    <section className="analysis-chart-panel analysis-pca-panel">
      <div className="analysis-panel-header">
        <div>
          <SlidersHorizontal size={16} />
          <h2>PCA workspace</h2>
        </div>
        <div className="analysis-control-row">
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
            <select value={clusterCount} onChange={(event) => setClusterCount(Number(event.target.value))}>
              {[2, 3, 4, 5, 6, 8].filter((value) => value <= Math.max(2, analysis.pca.points.length)).map((value) => (
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
      </div>
      <div className="analysis-pca-content">
        <Plot
          data={data}
          layout={layout}
          config={plotConfig(true)}
          useResizeHandler
          className="analysis-plot analysis-pca-plot"
          onSelected={handleSelected}
          onClick={handleClick}
        />
        <aside className="analysis-pca-side">
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
          <h3>Loadings · {analysis.pca.components[xComponent].name}</h3>
          <div className="analysis-loading-list">
            {Object.entries(analysis.pca.components[xComponent].loadings)
              .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
              .slice(0, 5)
              .map(([feature, value]) => (
                <span key={feature}>{metricLabel(feature)} <strong>{fixed(value, 2)}</strong></span>
              ))}
          </div>
          <h3>Cluster size</h3>
          <div className="analysis-cluster-list">
            {Object.entries(clusterSizes).map(([cluster, count]) => (
              <span key={cluster}>{cluster}<strong>{count.toLocaleString()}</strong></span>
            ))}
          </div>
        </aside>
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
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const effectiveSelection = useMemo<DatabaseSelectionState | null>(() => {
    if (selection.allFiltered || selection.selectedIds.length > 0) return selection;
    return null;
  }, [selection]);

  useEffect(() => {
    if (effectiveSelection === null) {
      setAnalysis(null);
      setLoading(false);
      setError(null);
      setPreviewRequest(EMPTY_PREVIEW);
      return undefined;
    }
    const controller = new AbortController();
    let active = true;
    setLoading(true);
    setError(null);
    void analyzeDatabaseSelection(effectiveSelection, controller.signal)
      .then((response) => {
        if (!active) return;
        setAnalysis(response);
        setPreviewRequest({
          title: "Curve data preview",
          detail: `${Math.min(response.samples.length, PREVIEW_CURVE_LIMIT)} filtered curves`,
          ids: response.samples.slice(0, PREVIEW_CURVE_LIMIT).map((sample) => sample.curve_id)
        });
        setHistogramMetric((current) =>
          response.metrics[current] ? current : Object.keys(response.metrics)[0] ?? "logRatio"
        );
      })
      .catch((caught) => {
        if (!active || (caught instanceof Error && caught.name === "AbortError")) return;
        setError(caught instanceof Error ? caught.message : "Could not analyze database selection");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [effectiveSelection]);

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
      {loading && !analysis ? (
        <div className="analysis-loading"><RefreshCw className="spin" size={24} /> Loading analysis</div>
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
            <SelectedDataTable analysis={analysis} />
          </div>
        </div>
      ) : null}
    </main>
  );
}
