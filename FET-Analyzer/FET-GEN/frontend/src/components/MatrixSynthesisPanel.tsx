import {
  Braces,
  CheckSquare,
  Download,
  Grid3X3,
  Paintbrush,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from "react";
import { exportMatrixWorkbook, getDatabaseCurvePreviews, synthesizeMatrix } from "../api";
import { fixed, scientific } from "../format";
import type {
  CurveFilters,
  CurvePreview,
  GenerationCondition,
  MatrixAssignment,
  MatrixDuplicateMode,
  MatrixParameterKey,
  MatrixSynthesisRequest,
  MatrixSynthesisMode,
  MatrixSynthesisResponse
} from "../types";

type ParameterLayer = {
  id: string;
  key: MatrixParameterKey;
  min: number;
  max: number;
  values: number[][];
};

type HeatmapCell = {
  row: number;
  col: number;
};

type BrushSample = HeatmapCell & {
  weight: number;
};

const MATRIX_PIXEL_SIZE = 28;
const OVERLAY_LIMIT = 20;

const PARAMETER_OPTIONS: Array<{
  key: MatrixParameterKey;
  label: string;
  min: number;
  max: number;
}> = [
  { key: "target_ion", label: "Ion", min: 1e-8, max: 1e-4 },
  { key: "target_ioff", label: "Ioff", min: 1e-14, max: 1e-10 },
  { key: "target_vth", label: "Vth", min: -2, max: 8 },
  { key: "target_ss_mv_dec", label: "SS", min: 70, max: 450 },
  { key: "hysteresis_v", label: "Hysteresis", min: 0, max: 3 },
  { key: "mobility_cm2_vs", label: "Mobility", min: 2, max: 60 },
  { key: "contact_resistance_ohm", label: "Contact R", min: 1e3, max: 8e4 },
  { key: "gate_leakage_a", label: "Gate leak", min: 1e-15, max: 1e-11 }
];

const MATRIX_BASE_CONDITION: GenerationCondition = {
  curve_type: "transfer",
  material: "MoS2",
  polarity: "n-type",
  vd: 1,
  target_ion: 1e-5,
  target_ioff: 1e-15,
  target_vth: 0,
  target_ss_mv_dec: 230,
  ss_region_width_v: 0.5,
  hysteresis_v: 1.5,
  noise_sigma_a: 1e-13,
  noise_floor_a: 1e-13,
  quantization_step_a: 1e-15,
  output_noise_gain: 4,
  gate_leakage_a: 1e-14,
  gate_leakage_v_char: 0.7,
  gate_leakage_exponent: 0.8,
  ion_sigma_fraction: 0.08,
  ioff_sigma_fraction: 0.15,
  vth_sigma_v: 0.2,
  ss_sigma_fraction: 0.1,
  hysteresis_sigma_v: 0.1,
  mobility_cm2_vs: 20,
  mobility_sigma_fraction: 0.1,
  contact_resistance_ohm: 1e4,
  contact_resistance_sigma_fraction: 0.15,
  ai_residual_strength: 0,
  gate_ai_residual_strength: 0,
  diversity: 1,
  seed: 32001,
  voltage_min: -20,
  voltage_max: 20,
  points: 301,
  variants: 1
};

function parameterDefaults(key: MatrixParameterKey) {
  return PARAMETER_OPTIONS.find((option) => option.key === key) ?? PARAMETER_OPTIONS[0];
}

function makeMatrix(rows: number, cols: number, value: number): number[][] {
  return Array.from({ length: rows }, () => Array.from({ length: cols }, () => value));
}

function resizeMatrix(values: number[][], rows: number, cols: number, fill: number): number[][] {
  return Array.from({ length: rows }, (_, rowIndex) =>
    Array.from({ length: cols }, (_, colIndex) => values[rowIndex]?.[colIndex] ?? fill)
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function layerSpan(layer: Pick<ParameterLayer, "min" | "max">): number {
  return Math.max(Math.abs(layer.max - layer.min), Number.EPSILON);
}

function defaultBrushDelta(layer: Pick<ParameterLayer, "min" | "max">): number {
  return layerSpan(layer) / 80;
}

function brushDeltaLimit(layer: Pick<ParameterLayer, "min" | "max">): number {
  return layerSpan(layer) / 8;
}

function clampBrushDelta(value: number, layer: Pick<ParameterLayer, "min" | "max">): number {
  const limit = brushDeltaLimit(layer);
  return clamp(value, -limit, limit);
}

function valueLabel(value: number): string {
  const abs = Math.abs(value);
  if (abs > 0 && (abs < 1e-3 || abs >= 1e4)) return value.toExponential(2);
  return fixed(value, abs < 10 ? 2 : 0);
}

function cellColor(value: number, min: number, max: number): string {
  const t = clamp((value - min) / Math.max(max - min, Number.EPSILON), 0, 1);
  const hue = 210 - t * 170;
  const lightness = 92 - t * 42;
  return `hsl(${hue} 76% ${lightness}%)`;
}

function columnLabel(index: number): string {
  let label = "";
  let current = index;
  while (current >= 0) {
    label = String.fromCharCode(65 + (current % 26)) + label;
    current = Math.floor(current / 26) - 1;
  }
  return label;
}

function brushSamples(center: HeatmapCell, rows: number, cols: number, radius: number): BrushSample[] {
  const samples: BrushSample[] = [];
  if (radius <= 0) return [{ ...center, weight: 1 }];
  for (let row = center.row - radius; row <= center.row + radius; row += 1) {
    for (let col = center.col - radius; col <= center.col + radius; col += 1) {
      if (row < 0 || col < 0 || row >= rows || col >= cols) continue;
      const distance = Math.hypot(row - center.row, col - center.col);
      if (distance <= radius + 0.01) {
        const normalized = clamp(distance / Math.max(radius, 1), 0, 1);
        const weight = 0.08 + 0.92 * ((1 + Math.cos(Math.PI * normalized)) / 2);
        samples.push({ row, col, weight });
      }
    }
  }
  return samples;
}

function HeatmapCanvas({
  layer,
  active,
  rows,
  cols,
  brushDelta,
  brushRadius,
  onSelect,
  onApplyBrush,
  onAdjustCell
}: {
  layer: ParameterLayer;
  active: boolean;
  rows: number;
  cols: number;
  brushDelta: number;
  brushRadius: number;
  onSelect: () => void;
  onApplyBrush: (samples: BrushSample[], delta: number) => void;
  onAdjustCell: (cell: HeatmapCell, delta: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverCell, setHoverCell] = useState<HeatmapCell | null>(null);
  const paintingRef = useRef(false);

  const canvasWidth = cols * MATRIX_PIXEL_SIZE;
  const canvasHeight = rows * MATRIX_PIXEL_SIZE;

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const scale = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * scale;
    canvas.height = canvasHeight * scale;
    canvas.style.width = `${canvasWidth}px`;
    canvas.style.height = `${canvasHeight}px`;
    context.setTransform(scale, 0, 0, scale, 0, 0);
    context.clearRect(0, 0, canvasWidth, canvasHeight);

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const value = layer.values[row]?.[col] ?? layer.min;
        context.fillStyle = cellColor(value, layer.min, layer.max);
        context.fillRect(
          col * MATRIX_PIXEL_SIZE,
          row * MATRIX_PIXEL_SIZE,
          MATRIX_PIXEL_SIZE,
          MATRIX_PIXEL_SIZE
        );
      }
    }

    context.strokeStyle = "rgba(31, 48, 72, 0.18)";
    context.lineWidth = 1;
    for (let col = 0; col <= cols; col += 1) {
      const x = col * MATRIX_PIXEL_SIZE + 0.5;
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, canvasHeight);
      context.stroke();
    }
    for (let row = 0; row <= rows; row += 1) {
      const y = row * MATRIX_PIXEL_SIZE + 0.5;
      context.beginPath();
      context.moveTo(0, y);
      context.lineTo(canvasWidth, y);
      context.stroke();
    }

    if (hoverCell) {
      for (const sample of brushSamples(hoverCell, rows, cols, brushRadius)) {
        context.fillStyle = `rgba(255, 255, 255, ${0.06 + 0.32 * sample.weight})`;
        context.strokeStyle = `rgba(15, 23, 42, ${0.18 + 0.5 * sample.weight})`;
        context.lineWidth = 1 + sample.weight;
        context.fillRect(
          sample.col * MATRIX_PIXEL_SIZE,
          sample.row * MATRIX_PIXEL_SIZE,
          MATRIX_PIXEL_SIZE,
          MATRIX_PIXEL_SIZE
        );
        context.strokeRect(
          sample.col * MATRIX_PIXEL_SIZE + 2,
          sample.row * MATRIX_PIXEL_SIZE + 2,
          MATRIX_PIXEL_SIZE - 4,
          MATRIX_PIXEL_SIZE - 4
        );
      }
    }
  }, [brushRadius, canvasHeight, canvasWidth, cols, hoverCell, layer, rows]);

  function eventCell(event: ReactPointerEvent<HTMLCanvasElement> | ReactWheelEvent<HTMLCanvasElement>): HeatmapCell | null {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const col = Math.floor(((event.clientX - rect.left) / rect.width) * cols);
    const row = Math.floor(((event.clientY - rect.top) / rect.height) * rows);
    if (row < 0 || col < 0 || row >= rows || col >= cols) return null;
    return { row, col };
  }

  function paint(cell: HeatmapCell) {
    onApplyBrush(brushSamples(cell, rows, cols, brushRadius), brushDelta);
  }

  return (
    <div className={active ? "matrix-canvas-viewport active" : "matrix-canvas-viewport"}>
      <canvas
        ref={canvasRef}
        className="matrix-canvas"
        width={canvasWidth}
        height={canvasHeight}
        onPointerDown={(event) => {
          onSelect();
          const cell = eventCell(event);
          if (!cell) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          paintingRef.current = true;
          setHoverCell(cell);
          paint(cell);
        }}
        onPointerMove={(event) => {
          const cell = eventCell(event);
          setHoverCell(cell);
          if (cell && paintingRef.current) paint(cell);
        }}
        onPointerUp={() => {
          paintingRef.current = false;
        }}
        onPointerCancel={() => {
          paintingRef.current = false;
        }}
        onPointerLeave={() => {
          paintingRef.current = false;
          setHoverCell(null);
        }}
        onWheel={(event) => {
          onSelect();
          const cell = eventCell(event);
          if (!cell) return;
          event.preventDefault();
          const delta = (layer.max - layer.min) / 100;
          onAdjustCell(cell, event.deltaY < 0 ? delta : -delta);
        }}
      />
      {hoverCell ? (
        <div className="matrix-canvas-readout">
          {columnLabel(hoverCell.col)}{hoverCell.row + 1} / {valueLabel(layer.values[hoverCell.row]?.[hoverCell.col] ?? layer.min)}
        </div>
      ) : null}
    </div>
  );
}

function metricFromAssignment(assignment: MatrixAssignment, key: MatrixParameterKey): number | undefined {
  if (assignment.source === "generated") {
    const features = assignment.generated?.features;
    if (key === "target_ion") return features?.ion;
    if (key === "target_ioff") return features?.ioff;
    if (key === "target_vth") return features?.vth ?? undefined;
    if (key === "target_ss_mv_dec") return features?.ss_mv_dec ?? undefined;
    if (key === "hysteresis_v") return features?.hysteresis_v ?? undefined;
  }
  return assignment.matched?.[key];
}

type OverlaySeries = {
  key: string;
  label: string;
  voltage: number[];
  current: number[];
  source: "database" | "generated";
};

function stableHash(text: string): number {
  let hash = 2166136261;
  for (const char of text) {
    hash ^= char.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function sampledOverlaySeries(series: OverlaySeries[]): OverlaySeries[] {
  if (series.length <= OVERLAY_LIMIT) return series;
  return [...series]
    .sort((left, right) => stableHash(left.key) - stableHash(right.key))
    .slice(0, OVERLAY_LIMIT);
}

function OverlayPlot({
  assignments,
  previews,
  loading
}: {
  assignments: MatrixAssignment[];
  previews: CurvePreview[];
  loading: boolean;
}) {
  const plot = useMemo(() => {
    const previewsById = new Map(previews.map((preview) => [preview.curve_id, preview]));
    const series: OverlaySeries[] = [];
    for (const assignment of assignments) {
      if (assignment.source === "database" && assignment.curve_id) {
        const preview = previewsById.get(assignment.curve_id);
        if (!preview || preview.raw_points.length < 2) continue;
        series.push({
          key: `${assignment.site}-${assignment.curve_id}-raw_id`,
          label: `${assignment.site} ${assignment.curve_id}`,
          voltage: preview.raw_points.map((point) => point.voltage_v),
          current: preview.raw_points.map((point) => point.current_a),
          source: "database"
        });
      } else if (assignment.source === "generated" && assignment.generated) {
        const voltage = assignment.generated.voltage;
        if (voltage.length < 2) continue;
        series.push({
          key: `${assignment.site}-${assignment.generated.seed}-id-forward`,
          label: `${assignment.site} seed ${assignment.generated.seed} F`,
          voltage,
          current: assignment.generated.forward_current,
          source: "generated"
        });
        series.push({
          key: `${assignment.site}-${assignment.generated.seed}-id-reverse`,
          label: `${assignment.site} seed ${assignment.generated.seed} R`,
          voltage,
          current: assignment.generated.reverse_current,
          source: "generated"
        });
      }
    }
    const sampled = sampledOverlaySeries(series).filter((entry) => entry.current.length > 0);
    const points = sampled.flatMap((entry) =>
      entry.voltage.flatMap((voltage, index) => {
        const current = Math.abs(entry.current[index] ?? NaN);
        if (!Number.isFinite(voltage) || !Number.isFinite(current) || current <= 0) return [];
        return [{ voltage, logCurrent: Math.log10(current) }];
      })
    );
    if (points.length === 0) {
      return { series: sampled, paths: [] as { key: string; d: string; source: string }[], ticks: null };
    }
    const minX = Math.min(...points.map((point) => point.voltage));
    const maxX = Math.max(...points.map((point) => point.voltage));
    const minY = Math.floor(Math.min(...points.map((point) => point.logCurrent)));
    const maxY = Math.ceil(Math.max(...points.map((point) => point.logCurrent)));
    const x0 = minX === maxX ? minX - 1 : minX;
    const x1 = minX === maxX ? maxX + 1 : maxX;
    const y0 = minY === maxY ? minY - 1 : minY;
    const y1 = minY === maxY ? maxY + 1 : maxY;
    const left = 42;
    const right = 342;
    const top = 16;
    const bottom = 172;
    const project = (voltage: number, current: number) => {
      const logCurrent = Math.log10(Math.abs(current));
      return {
        x: left + ((voltage - x0) / (x1 - x0)) * (right - left),
        y: top + (1 - (logCurrent - y0) / (y1 - y0)) * (bottom - top)
      };
    };
    const paths = sampled.flatMap((entry) => {
      const coords = entry.voltage.flatMap((voltage, index) => {
        const current = entry.current[index] ?? NaN;
        if (!Number.isFinite(voltage) || !Number.isFinite(current) || Math.abs(current) <= 0) return [];
        return [project(voltage, current)];
      });
      if (coords.length < 2) return [];
      return [{
        key: entry.key,
        source: entry.source,
        d: coords.map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`).join(" ")
      }];
    });
    const yTickCount = Math.min(6, Math.max(2, y1 - y0 + 1));
    return {
      series: sampled,
      paths,
      ticks: {
        x: Array.from({ length: 5 }, (_, index) => {
          const value = x0 + ((x1 - x0) * index) / 4;
          return { value, x: left + ((right - left) * index) / 4 };
        }),
        y: Array.from({ length: yTickCount }, (_, index) => {
          const value = y0 + ((y1 - y0) * index) / Math.max(1, yTickCount - 1);
          return { value, y: top + (1 - (value - y0) / (y1 - y0)) * (bottom - top) };
        }),
        left,
        right,
        top,
        bottom,
        total: series.length,
        shown: sampled.length
      }
    };
  }, [assignments, previews]);

  return (
    <section className="matrix-overlay-panel">
      <header>
        <strong>Overlay plot</strong>
        <span>{loading ? "Loading curves" : `${plot.ticks?.shown ?? plot.series.length}/${plot.ticks?.total ?? plot.series.length} lines`}</span>
      </header>
      {plot.paths.length === 0 ? (
        <div className="matrix-overlay-empty">Run matrix matching to preview transfer curves.</div>
      ) : (
        <svg viewBox="0 0 360 210" role="img" aria-label="Overlay transfer curves on logarithmic current axis">
          {plot.ticks?.y.map((tick) => (
            <g key={`y-${tick.value}`}>
              <line className="overlay-grid" x1={plot.ticks?.left} x2={plot.ticks?.right} y1={tick.y} y2={tick.y} />
              <text x="35" y={tick.y + 4}>{tick.value.toFixed(0)}</text>
            </g>
          ))}
          {plot.ticks?.x.map((tick) => (
            <g key={`x-${tick.value}`}>
              <line className="overlay-grid" x1={tick.x} x2={tick.x} y1={plot.ticks?.top} y2={plot.ticks?.bottom} />
              <text x={tick.x} y="194">{fixed(tick.value, 1)}</text>
            </g>
          ))}
          {plot.ticks ? (
            <>
              <line x1={plot.ticks.left} x2={plot.ticks.right} y1={plot.ticks.bottom} y2={plot.ticks.bottom} />
              <line x1={plot.ticks.left} x2={plot.ticks.left} y1={plot.ticks.top} y2={plot.ticks.bottom} />
            </>
          ) : null}
          {plot.paths.map((path, index) => (
            <path
              key={path.key}
              className={path.source === "database" ? "database-line" : "generated-line"}
              d={path.d}
              style={{ opacity: 0.35 + 0.45 * (1 - index / Math.max(1, plot.paths.length)) }}
            />
          ))}
          <text className="axis-label" x="190" y="208">Vg (V)</text>
          <text className="axis-label" x="13" y="102" transform="rotate(-90 13 102)">log10 |Id|</text>
        </svg>
      )}
    </section>
  );
}

export function MatrixSynthesisPanel() {
  const [expanded, setExpanded] = useState(true);
  const [rows, setRows] = useState(4);
  const [cols, setCols] = useState(4);
  const [mode, setMode] = useState<MatrixSynthesisMode>("database");
  const [duplicateMode, setDuplicateMode] = useState<MatrixDuplicateMode>("generate_on_duplicate");
  const [filters, setFilters] = useState<CurveFilters>({});
  const [layers, setLayers] = useState<ParameterLayer[]>(() => {
    const initial = parameterDefaults("target_vth");
    return [{
      id: "target_vth",
      key: "target_vth",
      min: initial.min,
      max: initial.max,
      values: makeMatrix(4, 4, (initial.min + initial.max) / 2)
    }];
  });
  const [activeLayerId, setActiveLayerId] = useState("target_vth");
  const [brushDelta, setBrushDelta] = useState(0.125);
  const [brushRadius, setBrushRadius] = useState(1);
  const [response, setResponse] = useState<MatrixSynthesisResponse | null>(null);
  const [plotPreviews, setPlotPreviews] = useState<CurvePreview[]>([]);
  const [loadingPlot, setLoadingPlot] = useState(false);
  const [running, setRunning] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const activeLayer = useMemo(
    () => layers.find((layer) => layer.id === activeLayerId) ?? layers[0],
    [activeLayerId, layers]
  );
  const activeOption = activeLayer ? parameterDefaults(activeLayer.key) : PARAMETER_OPTIONS[0];
  const visibleLayers = layers.slice(0, 4);

  useEffect(() => {
    if (!activeLayer) return;
    setBrushDelta((value) => clampBrushDelta(value, activeLayer));
  }, [activeLayer?.id, activeLayer?.min, activeLayer?.max]);

  useEffect(() => {
    const databaseIds = Array.from(new Set(
      (response?.assignments ?? []).flatMap((assignment) =>
        assignment.source === "database" && assignment.curve_id ? [assignment.curve_id] : []
      )
    ));
    if (databaseIds.length === 0) {
      setPlotPreviews([]);
      setLoadingPlot(false);
      return undefined;
    }
    const selectedIds = sampledOverlaySeries(
      databaseIds.map((curveId) => ({
        key: curveId,
        label: curveId,
        voltage: [],
        current: [],
        source: "database" as const
      }))
    ).map((entry) => entry.key);
    const controller = new AbortController();
    setLoadingPlot(true);
    void getDatabaseCurvePreviews(selectedIds, controller.signal)
      .then(setPlotPreviews)
      .catch(() => setPlotPreviews([]))
      .finally(() => setLoadingPlot(false));
    return () => controller.abort();
  }, [response]);

  function updateDimensions(nextRows: number, nextCols: number) {
    const safeRows = clamp(Math.round(nextRows), 1, 32);
    const safeCols = clamp(Math.round(nextCols), 1, 32);
    setRows(safeRows);
    setCols(safeCols);
    setLayers((current) =>
      current.map((layer) => ({
        ...layer,
        values: resizeMatrix(layer.values, safeRows, safeCols, (layer.min + layer.max) / 2)
      }))
    );
  }

  function patchLayer(layerId: string, patch: Partial<ParameterLayer>) {
    setLayers((current) =>
      current.map((layer) => {
        if (layer.id !== layerId) return layer;
        const next = { ...layer, ...patch };
        if (patch.min !== undefined || patch.max !== undefined) {
          next.values = next.values.map((row) =>
            row.map((value) => clamp(value, next.min, next.max))
          );
          if (layerId === activeLayerId) {
            setBrushDelta((value) => clampBrushDelta(value, next));
          }
        }
        return next;
      })
    );
  }

  function setCell(layerId: string, rowIndex: number, colIndex: number, value: number) {
    const targetLayer = layers.find((layer) => layer.id === layerId);
    if (!targetLayer) return;
    const nextValue = clamp(value, targetLayer.min, targetLayer.max);
    setLayers((current) =>
      current.map((layer) =>
        layer.id === layerId
          ? {
              ...layer,
              values: layer.values.map((row, r) =>
                row.map((cell, c) => (r === rowIndex && c === colIndex ? nextValue : cell))
              )
            }
          : layer
      )
    );
  }

  function applyBrush(layerId: string, samples: BrushSample[], delta: number) {
    const targetLayer = layers.find((layer) => layer.id === layerId);
    if (!targetLayer) return;
    const sampleMap = new Map(samples.map((sample) => [`${sample.row}:${sample.col}`, sample.weight]));
    setLayers((current) =>
      current.map((layer) =>
        layer.id === layerId
          ? {
              ...layer,
              values: layer.values.map((row, rowIndex) =>
                row.map((cell, colIndex) => {
                  const weight = sampleMap.get(`${rowIndex}:${colIndex}`);
                  return weight === undefined
                    ? cell
                    : clamp(cell + delta * weight, layer.min, layer.max);
                })
              )
            }
          : layer
      )
    );
  }

  function addLayer() {
    if (layers.length >= 4) return;
    const option = PARAMETER_OPTIONS.find((candidate) => !layers.some((layer) => layer.key === candidate.key))
      ?? PARAMETER_OPTIONS[0];
    const id = `${option.key}-${Date.now()}`;
    setLayers((current) => [
      ...current,
      {
        id,
        key: option.key,
        min: option.min,
        max: option.max,
        values: makeMatrix(rows, cols, (option.min + option.max) / 2)
      }
    ]);
    setActiveLayerId(id);
    setBrushDelta(defaultBrushDelta(option));
  }

  function removeActiveLayer() {
    if (layers.length <= 1 || !activeLayer) return;
    const nextLayers = layers.filter((layer) => layer.id !== activeLayer.id);
    setLayers(nextLayers);
    setActiveLayerId(nextLayers[0].id);
    setBrushDelta(defaultBrushDelta(nextLayers[0]));
  }

  function randomizeActiveLayer() {
    if (!activeLayer) return;
    const span = activeLayer.max - activeLayer.min;
    setLayers((current) =>
      current.map((layer) =>
        layer.id === activeLayer.id
          ? {
              ...layer,
              values: layer.values.map((row) =>
                row.map((value) =>
                  clamp(value + (Math.random() - 0.5) * span * 0.12, layer.min, layer.max)
                )
              )
            }
          : layer
      )
    );
  }

  function matrixRequest(): MatrixSynthesisRequest {
    return {
      rows,
      cols,
      mode,
      duplicate_mode: duplicateMode,
      filters,
      generation_condition: MATRIX_BASE_CONDITION,
      parameters: layers.map((layer) => ({ key: layer.key, values: layer.values }))
    };
  }

  async function runSynthesis() {
    setRunning(true);
    setError(null);
    try {
      const result = await synthesizeMatrix(matrixRequest());
      setResponse(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Matrix synthesis failed");
    } finally {
      setRunning(false);
    }
  }

  async function exportWorkbook() {
    setExporting(true);
    setError(null);
    try {
      await exportMatrixWorkbook(matrixRequest());
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Matrix workbook export failed");
    } finally {
      setExporting(false);
    }
  }

  return (
    <section className={`matrix-panel${expanded ? " expanded" : ""}`}>
      <header className="matrix-panel-header">
        <div>
          <Grid3X3 size={18} />
          <h2>Parameter matrix panel</h2>
          <span>{rows} x {cols} sites</span>
        </div>
        <button className="button secondary compact" onClick={() => setExpanded((current) => !current)}>
          {expanded ? "Collapse" : "Open"}
        </button>
      </header>
      {expanded ? (
        <div className="matrix-panel-body">
          <aside className="matrix-controls">
            <section className="matrix-control-section">
              <h3>Matrix targets</h3>
              <div className="matrix-control-row two">
                <label>
                  Rows
                  <input type="number" min="1" max="32" value={rows} onChange={(event) => updateDimensions(Number(event.target.value), cols)} />
                </label>
                <label>
                  Columns
                  <input type="number" min="1" max="32" value={cols} onChange={(event) => updateDimensions(rows, Number(event.target.value))} />
                </label>
              </div>
              <div className="matrix-layer-actions">
                <button className="button secondary compact" onClick={addLayer} disabled={layers.length >= 4}><Plus size={14} /> Add</button>
                <button className="button secondary compact" onClick={removeActiveLayer} disabled={layers.length <= 1}><Trash2 size={14} /> Remove</button>
              </div>
            </section>
            <section className="matrix-control-section matrix-match-section">
              <h3>Output matching</h3>
              <div className="matrix-control-row two">
                <label>
                  Source
                  <select value={mode} onChange={(event) => setMode(event.target.value as MatrixSynthesisMode)}>
                    <option value="database">Nearest database</option>
                    <option value="generate">Generate</option>
                  </select>
                </label>
                <label>
                  Repeats
                  <select value={duplicateMode} onChange={(event) => setDuplicateMode(event.target.value as MatrixDuplicateMode)}>
                    <option value="allow">Allow</option>
                    <option value="avoid">Avoid</option>
                    <option value="generate_on_duplicate">Generate fallback</option>
                  </select>
                </label>
              </div>
            </section>
            <section className="matrix-control-section matrix-global-filters">
              <h3>Database global limits</h3>
              <div className="matrix-control-row two">
                <label>
                  Polarity
                  <select value={filters.polarity ?? ""} onChange={(event) => setFilters({ ...filters, polarity: event.target.value || undefined })}>
                    <option value="">Any</option>
                    <option value="n-type">n-type</option>
                    <option value="p-type">p-type</option>
                  </select>
                </label>
                <label>
                  Direction
                  <select value={filters.direction ?? ""} onChange={(event) => setFilters({ ...filters, direction: event.target.value || undefined })}>
                    <option value="">Any</option>
                    <option value="forward">forward</option>
                    <option value="reverse">reverse</option>
                    <option value="single">single</option>
                  </select>
                </label>
              </div>
              <div className="matrix-control-row two">
                <label>
                  Ion min
                  <input value={filters.ion_min ?? ""} onChange={(event) => setFilters({ ...filters, ion_min: event.target.value || undefined })} />
                </label>
                <label>
                  Ion max
                  <input value={filters.ion_max ?? ""} onChange={(event) => setFilters({ ...filters, ion_max: event.target.value || undefined })} />
                </label>
              </div>
              <div className="matrix-control-row two">
                <label>
                  Ioff min
                  <input value={filters.ioff_min ?? ""} onChange={(event) => setFilters({ ...filters, ioff_min: event.target.value || undefined })} />
                </label>
                <label>
                  Ioff max
                  <input value={filters.ioff_max ?? ""} onChange={(event) => setFilters({ ...filters, ioff_max: event.target.value || undefined })} />
                </label>
              </div>
              <div className="matrix-control-row two">
                <label>
                  On/off min
                  <input value={filters.ion_ioff_ratio_min ?? ""} onChange={(event) => setFilters({ ...filters, ion_ioff_ratio_min: event.target.value || undefined })} />
                </label>
                <label>
                  On/off max
                  <input value={filters.ion_ioff_ratio_max ?? ""} onChange={(event) => setFilters({ ...filters, ion_ioff_ratio_max: event.target.value || undefined })} />
                </label>
              </div>
              <div className="matrix-control-row two">
                <label>
                  Vth min
                  <input value={filters.vth_min ?? ""} onChange={(event) => setFilters({ ...filters, vth_min: event.target.value || undefined })} />
                </label>
                <label>
                  Vth max
                  <input value={filters.vth_max ?? ""} onChange={(event) => setFilters({ ...filters, vth_max: event.target.value || undefined })} />
                </label>
              </div>
              <div className="matrix-control-row two">
                <label>
                  SS min
                  <input value={filters.ss_mv_dec_min ?? ""} onChange={(event) => setFilters({ ...filters, ss_mv_dec_min: event.target.value || undefined })} />
                </label>
                <label>
                  SS max
                  <input value={filters.ss_mv_dec_max ?? ""} onChange={(event) => setFilters({ ...filters, ss_mv_dec_max: event.target.value || undefined })} />
                </label>
              </div>
              <div className="matrix-control-row two">
                <label>
                  Has Ig
                  <select value={filters.has_gate_current ?? ""} onChange={(event) => setFilters({ ...filters, has_gate_current: event.target.value || undefined })}>
                    <option value="">Any</option>
                    <option value="true">With Ig</option>
                    <option value="false">No Ig</option>
                  </select>
                </label>
                <label>
                  Hysteresis
                  <select value={filters.hysteresis_available ?? ""} onChange={(event) => setFilters({ ...filters, hysteresis_available: event.target.value || undefined })}>
                    <option value="">Any</option>
                    <option value="true">Paired</option>
                    <option value="false">NA / single</option>
                  </select>
                </label>
              </div>
            </section>
          </aside>
          <main className="matrix-heatmap-panel">
            <div className="matrix-heatmap-toolbar">
              <span><CheckSquare size={14} /> {activeOption.label} / {MATRIX_PIXEL_SIZE}px pixels</span>
              {activeLayer ? (
                <div className="matrix-brush-controls">
                  <label>
                    Delta
                    <input
                      type="range"
                      min={-brushDeltaLimit(activeLayer)}
                      max={brushDeltaLimit(activeLayer)}
                      step={layerSpan(activeLayer) / 240}
                      value={clampBrushDelta(brushDelta, activeLayer)}
                      onChange={(event) => setBrushDelta(Number(event.target.value))}
                    />
                    <strong>{valueLabel(clampBrushDelta(brushDelta, activeLayer))}</strong>
                  </label>
                  <label>
                    Size
                    <input
                      type="range"
                      min="0"
                      max="5"
                      step="1"
                      value={brushRadius}
                      onChange={(event) => setBrushRadius(Number(event.target.value))}
                    />
                    <strong>{brushRadius === 0 ? "1" : String(brushRadius * 2 + 1)}</strong>
                  </label>
                  <button className="button secondary compact matrix-noise-button" onClick={randomizeActiveLayer}>
                    <Paintbrush size={14} />
                    Noise
                  </button>
                </div>
              ) : null}
              <button className="button primary compact" onClick={() => void runSynthesis()} disabled={running}>
                {mode === "generate" ? <Sparkles size={14} /> : <RefreshCw size={14} className={running ? "spin" : undefined} />}
                {running ? "Running" : "Run matrix"}
              </button>
            </div>
            {error ? <div className="error-banner">{error}</div> : null}
            <div className="matrix-view-grid">
              {visibleLayers.map((layer) => (
                <section key={layer.id} className={layer.id === activeLayerId ? "matrix-view active" : "matrix-view"}>
                  <header>
                    <div className="matrix-view-title">
                      <button type="button" onClick={() => {
                        setActiveLayerId(layer.id);
                        setBrushDelta((value) => clampBrushDelta(value, layer));
                      }}>
                        <Braces size={13} />
                        {parameterDefaults(layer.key).label}
                      </button>
                      <span>{layer.id === activeLayerId ? "Active" : "Click to edit"}</span>
                    </div>
                    <div className="matrix-view-settings">
                      <select
                        value={layer.key}
                        onFocus={() => {
                          setActiveLayerId(layer.id);
                          setBrushDelta((value) => clampBrushDelta(value, layer));
                        }}
                        onChange={(event) => {
                          const option = parameterDefaults(event.target.value as MatrixParameterKey);
                          patchLayer(layer.id, {
                            key: option.key,
                            min: option.min,
                            max: option.max,
                            values: makeMatrix(rows, cols, (option.min + option.max) / 2)
                          });
                          setActiveLayerId(layer.id);
                          setBrushDelta(defaultBrushDelta(option));
                        }}
                      >
                        {PARAMETER_OPTIONS.map((option) => <option key={option.key} value={option.key}>{option.label}</option>)}
                      </select>
                      <input
                        aria-label={`${parameterDefaults(layer.key).label} min`}
                        type="number"
                        value={layer.min}
                        onFocus={() => {
                          setActiveLayerId(layer.id);
                          setBrushDelta((value) => clampBrushDelta(value, layer));
                        }}
                        onChange={(event) => patchLayer(layer.id, { min: Number(event.target.value) })}
                      />
                      <input
                        aria-label={`${parameterDefaults(layer.key).label} max`}
                        type="number"
                        value={layer.max}
                        onFocus={() => {
                          setActiveLayerId(layer.id);
                          setBrushDelta((value) => clampBrushDelta(value, layer));
                        }}
                        onChange={(event) => patchLayer(layer.id, { max: Number(event.target.value) })}
                      />
                    </div>
                  </header>
                  <HeatmapCanvas
                    layer={layer}
                    active={layer.id === activeLayerId}
                    rows={rows}
                    cols={cols}
                    brushDelta={clampBrushDelta(brushDelta, layer)}
                    brushRadius={brushRadius}
                    onSelect={() => {
                      setActiveLayerId(layer.id);
                      setBrushDelta((value) => clampBrushDelta(value, layer));
                    }}
                    onApplyBrush={(samples, delta) => applyBrush(layer.id, samples, delta)}
                    onAdjustCell={(cell, delta) =>
                      setCell(
                        layer.id,
                        cell.row,
                        cell.col,
                        (layer.values[cell.row]?.[cell.col] ?? layer.min) + delta
                      )
                    }
                  />
                </section>
              ))}
              {Array.from({ length: Math.max(0, 4 - visibleLayers.length) }, (_, index) => (
                <section key={`empty-${index}`} className="matrix-view empty">
                  <button type="button" onClick={addLayer} disabled={layers.length >= 4}>
                    <Plus size={18} />
                    Add parameter
                  </button>
                </section>
              ))}
            </div>
          </main>
          <aside className="matrix-results">
            <header>
              <strong>Output</strong>
              <button className="button secondary compact" onClick={() => void exportWorkbook()} disabled={exporting}>
                  <Download size={14} />
                  {exporting ? "Exporting" : "XLSX"}
              </button>
            </header>
            {response ? (
              <>
                <div className="matrix-summary">
                  <div><span>Matched</span><strong>{response.matched_count}</strong></div>
                  <div><span>Generated</span><strong>{response.generated_count}</strong></div>
                  <div><span>Unmatched</span><strong>{response.unmatched_count}</strong></div>
                  <div><span>Reused</span><strong>{response.reused_count}</strong></div>
                </div>
                <div className="matrix-output-table">
                  <table>
                    <thead><tr><th>Site</th><th>Source</th><th>Curve</th><th>Score</th><th>Features</th><th>Ion</th><th>Vth</th></tr></thead>
                    <tbody>
                      {response.assignments.map((assignment) => (
                        <tr key={assignment.site}>
                          <td>{assignment.site}</td>
                          <td>{assignment.source}</td>
                          <td>{assignment.curve_id ?? (assignment.generated ? `seed ${assignment.generated.seed}` : "-")}</td>
                          <td>{assignment.score === undefined ? "-" : fixed(assignment.score, 3)}</td>
                          <td>{assignment.score_features?.map((key) => parameterDefaults(key).label).join(", ") ?? "-"}</td>
                          <td>{scientific(metricFromAssignment(assignment, "target_ion") ?? null)}</td>
                          <td>{fixed(metricFromAssignment(assignment, "target_vth") ?? null, 2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <OverlayPlot
                  assignments={response.assignments}
                  previews={plotPreviews}
                  loading={loadingPlot}
                />
              </>
            ) : (
              <>
                <div className="matrix-empty-output">Run the matrix to assign A1, B2 style device sites.</div>
                <OverlayPlot assignments={[]} previews={[]} loading={false} />
              </>
            )}
          </aside>
        </div>
      ) : null}
    </section>
  );
}
