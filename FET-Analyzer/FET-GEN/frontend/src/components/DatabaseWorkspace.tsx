import {
  BarChart3,
  CalendarDays,
  CheckSquare,
  ChevronLeft,
  ChevronRight,
  Columns3,
  Database,
  Download,
  Filter,
  RefreshCw,
  RotateCcw,
  Search,
  SlidersHorizontal,
  Square,
  X
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  getDatabaseCurve,
  getDatabaseOptions,
  getDatabaseStatus,
  exportDatabaseSelection,
  listDatabaseCurves
} from "../api";
import { DatabaseCalendarPanel } from "./DatabaseCalendarPanel";
import { fixed, scientific } from "../format";
import type {
  CurveDetail,
  CurveFilters,
  CurveListResponse,
  CurveSummary,
  DatabaseSelectionState,
  DatabaseOptions,
  DatabaseStatus
} from "../types";

const DEFAULT_PAGE_SIZE = 100;
const PAGE_SIZE_OPTIONS = [50, 100, 250];

type FilterKey = keyof CurveFilters;

const NUMERIC_FILTER_KEYS: FilterKey[] = [
  "ion_min",
  "ion_max",
  "ioff_min",
  "ioff_max",
  "ion_ioff_ratio_min",
  "ion_ioff_ratio_max",
  "vth_min",
  "vth_max",
  "ss_mv_dec_min",
  "ss_mv_dec_max"
];

const FILTER_LABELS: Record<FilterKey, string> = {
  polarity: "Polarity",
  direction: "Direction",
  source_kind: "Source kind",
  source_search: "Source",
  has_gate_current: "Has Ig",
  hysteresis_available: "Hysteresis",
  date_from: "From",
  date_to: "To",
  ion_min: "Ion min",
  ion_max: "Ion max",
  ioff_min: "Ioff min",
  ioff_max: "Ioff max",
  ion_ioff_ratio_min: "Ion/Ioff min",
  ion_ioff_ratio_max: "Ion/Ioff max",
  vth_min: "Vth min",
  vth_max: "Vth max",
  ss_mv_dec_min: "SS min",
  ss_mv_dec_max: "SS max"
};

type MiniCurvePoint = {
  voltage: number;
  logCurrent: number;
};

type SvgPoint = {
  x: number;
  y: number;
};

type MiniCurvePlot = {
  idLine: string;
  igLine: string;
  idDots: SvgPoint[];
  igDots: SvgPoint[];
  xTicks: { value: number; x: number }[];
  yTicks: { value: number; y: number }[];
  hasIg: boolean;
};

const MINI_CURVE = {
  width: 360,
  height: 240,
  plotLeft: 54,
  plotRight: 338,
  plotTop: 28,
  plotBottom: 184
} as const;

const MINI_CURVE_EMPTY: MiniCurvePlot = {
  idLine: "",
  igLine: "",
  idDots: [],
  igDots: [],
  xTicks: [],
  yTicks: [],
  hasIg: false
};

const miniCurvePlotWidth = MINI_CURVE.plotRight - MINI_CURVE.plotLeft;
const miniCurvePlotHeight = MINI_CURVE.plotBottom - MINI_CURVE.plotTop;

function toMiniCurvePoint(point: { voltage_v: number; current_a: number }): MiniCurvePoint | null {
  const current = Math.abs(point.current_a);
  if (!Number.isFinite(point.voltage_v) || !Number.isFinite(current) || current <= 0) {
    return null;
  }
  return {
    voltage: point.voltage_v,
    logCurrent: Math.log10(current)
  };
}

function MiniCurve({ detail }: { detail: CurveDetail }) {
  const plot = useMemo(() => {
    const idPoints = detail.raw_points.flatMap((point) => {
      const nextPoint = toMiniCurvePoint(point);
      return nextPoint ? [nextPoint] : [];
    });
    const igPoints = detail.gate_points.flatMap((point) => {
      const nextPoint = toMiniCurvePoint(point);
      return nextPoint ? [nextPoint] : [];
    });
    const points = [...idPoints, ...igPoints];
    if (points.length === 0) {
      return MINI_CURVE_EMPTY;
    }

    const minVoltage = Math.min(...points.map((point) => point.voltage));
    const maxVoltage = Math.max(...points.map((point) => point.voltage));
    const minLogCurrent = Math.floor(Math.min(...points.map((point) => point.logCurrent)));
    const maxLogCurrent = Math.ceil(Math.max(...points.map((point) => point.logCurrent)));
    const minX = minVoltage === maxVoltage ? minVoltage - 1 : minVoltage;
    const maxX = minVoltage === maxVoltage ? maxVoltage + 1 : maxVoltage;
    const minY = minLogCurrent === maxLogCurrent ? minLogCurrent - 1 : minLogCurrent;
    const maxY = minLogCurrent === maxLogCurrent ? maxLogCurrent + 1 : maxLogCurrent;
    const xSpan = maxX - minX;
    const ySpan = maxY - minY;
    const project = (point: MiniCurvePoint): SvgPoint => ({
      x: MINI_CURVE.plotLeft + ((point.voltage - minX) / xSpan) * miniCurvePlotWidth,
      y: MINI_CURVE.plotTop + (1 - (point.logCurrent - minY) / ySpan) * miniCurvePlotHeight
    });
    const toLine = (series: SvgPoint[]) => series
      .map((point) => {
        return `${point.x.toFixed(1)},${point.y.toFixed(1)}`;
      })
      .join(" ");
    const idSeries = idPoints.map(project);
    const igSeries = igPoints.map(project);
    const xTicks = Array.from({ length: 5 }, (_, index) => {
      const value = minX + (xSpan * index) / 4;
      return {
        value,
        x: MINI_CURVE.plotLeft + (miniCurvePlotWidth * index) / 4
      };
    });
    const yTickCount = Math.min(6, Math.max(2, maxY - minY + 1));
    const yTicks = Array.from({ length: yTickCount }, (_, index) => {
      const value = minY + (ySpan * index) / Math.max(1, yTickCount - 1);
      return {
        value,
        y: MINI_CURVE.plotTop + (1 - (value - minY) / ySpan) * miniCurvePlotHeight
      };
    });
    return {
      idLine: idSeries.length > 1 ? toLine(idSeries) : "",
      igLine: igSeries.length > 1 ? toLine(igSeries) : "",
      idDots: idSeries.length === 1 ? idSeries : [],
      igDots: igSeries.length === 1 ? igSeries : [],
      xTicks,
      yTicks,
      hasIg: igSeries.length > 0
    };
  }, [detail]);

  return (
    <svg
      className="database-curve-preview"
      viewBox={`0 0 ${MINI_CURVE.width} ${MINI_CURVE.height}`}
      role="img"
      aria-label={plot.hasIg ? "Raw transfer curve preview with Id and Ig" : "Raw transfer curve preview with Id"}
    >
      {plot.yTicks.map((tick) => (
        <g key={`y-${tick.value}`}>
          <line
            className="preview-grid"
            x1={MINI_CURVE.plotLeft}
            y1={tick.y}
            x2={MINI_CURVE.plotRight}
            y2={tick.y}
          />
          <text className="preview-tick y-tick" x="47" y={tick.y + 4}>{tick.value.toFixed(0)}</text>
        </g>
      ))}
      {plot.xTicks.map((tick) => (
        <g key={`x-${tick.value}`}>
          <line
            className="preview-grid"
            x1={tick.x}
            y1={MINI_CURVE.plotTop}
            x2={tick.x}
            y2={MINI_CURVE.plotBottom}
          />
          <line x1={tick.x} y1={MINI_CURVE.plotBottom} x2={tick.x} y2={MINI_CURVE.plotBottom + 6} />
          <text className="preview-tick" x={tick.x} y="204">{fixed(tick.value, 1)}</text>
        </g>
      ))}
      <line x1={MINI_CURVE.plotLeft} y1={MINI_CURVE.plotBottom} x2={MINI_CURVE.plotRight} y2={MINI_CURVE.plotBottom} />
      <line x1={MINI_CURVE.plotLeft} y1={MINI_CURVE.plotTop} x2={MINI_CURVE.plotLeft} y2={MINI_CURVE.plotBottom} />
      {plot.idLine ? <polyline className="id-line" points={plot.idLine} /> : null}
      {plot.idDots.map((point) => (
        <circle key={`id-dot-${point.x}-${point.y}`} className="id-dot" cx={point.x} cy={point.y} r="3" />
      ))}
      {plot.igLine ? <polyline className="ig-line" points={plot.igLine} /> : null}
      {plot.igDots.map((point) => (
        <circle key={`ig-dot-${point.x}-${point.y}`} className="ig-dot" cx={point.x} cy={point.y} r="3" />
      ))}
      <text className="axis-label" x="196" y="228">Gate voltage Vg (V)</text>
      <text className="axis-label" x="15" y="106" transform="rotate(-90 15 106)">log10 |I| (A)</text>
      <g className="preview-legend" transform="translate(266 32)">
        <line className="id-line" x1="0" y1="0" x2="20" y2="0" />
        <text x="26" y="4">Id</text>
        {plot.hasIg ? (
          <>
            <line className="ig-line" x1="0" y1="18" x2="20" y2="18" />
            <text x="26" y="22">Ig</text>
          </>
        ) : null}
      </g>
    </svg>
  );
}

function StatStrip({ status }: { status: DatabaseStatus | null }) {
  return (
    <div className="database-stats">
      <div>
        <span>Curves</span>
        <strong>{status?.curves.toLocaleString() ?? "0"}</strong>
      </div>
      <div>
        <span>Raw points</span>
        <strong>{status?.raw_points.toLocaleString() ?? "0"}</strong>
      </div>
      <div>
        <span>Aligned points</span>
        <strong>{status?.aligned_points.toLocaleString() ?? "0"}</strong>
      </div>
      <div>
        <span>Curves with Ig</span>
        <strong>{status?.curves_with_ig.toLocaleString() ?? "0"}</strong>
      </div>
      <div>
        <span>Sources</span>
        <strong>{status?.source_files.toLocaleString() ?? "0"}</strong>
      </div>
    </div>
  );
}

function cleanFilters(filters: CurveFilters): CurveFilters {
  const next: CurveFilters = {};
  (Object.entries(filters) as [FilterKey, string | undefined][]).forEach(([key, value]) => {
    const trimmed = value?.trim();
    if (trimmed) next[key] = trimmed;
  });
  return next;
}

function patchFilter(
  filters: CurveFilters,
  key: FilterKey,
  value: string
): CurveFilters {
  const next = { ...filters };
  const trimmed = value.trim();
  if (trimmed) {
    next[key] = value;
  } else {
    delete next[key];
  }
  return next;
}

function removeFilter(filters: CurveFilters, key: FilterKey): CurveFilters {
  const next = { ...filters };
  delete next[key];
  return next;
}

function filterFingerprint(filters: CurveFilters): string {
  return JSON.stringify(
    (Object.keys(filters) as FilterKey[])
      .sort()
      .map((key) => [key, filters[key]])
  );
}

function numericFilterErrors(filters: CurveFilters): string[] {
  return NUMERIC_FILTER_KEYS.flatMap((key) => {
    const value = filters[key]?.trim();
    if (!value || Number.isFinite(Number(value))) return [];
    return [FILTER_LABELS[key]];
  });
}

function chipLabel(key: FilterKey, value: string): string {
  if (key === "date_from") return `From ${value}`;
  if (key === "date_to") return `To ${value}`;
  return `${FILTER_LABELS[key]}: ${value}`;
}

function sourceName(path: string): string {
  return path.split(/[\\/]/).at(-1) ?? path;
}

function logRatio(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return value.toFixed(2);
}

type ColumnKey =
  | "curve"
  | "polarity"
  | "direction"
  | "ion"
  | "ioff"
  | "logRatio"
  | "ig"
  | "vth"
  | "ss"
  | "sourceKind"
  | "source"
  | "modified";

const ALL_COLUMNS: { key: ColumnKey; label: string }[] = [
  { key: "curve", label: "Curve" },
  { key: "polarity", label: "Polarity" },
  { key: "direction", label: "Dir" },
  { key: "ion", label: "Ion" },
  { key: "ioff", label: "Ioff" },
  { key: "logRatio", label: "logRatio" },
  { key: "ig", label: "Ig" },
  { key: "vth", label: "Vth" },
  { key: "ss", label: "SS" },
  { key: "sourceKind", label: "Source kind" },
  { key: "source", label: "Source" },
  { key: "modified", label: "Modified" }
];

const DEFAULT_VISIBLE_COLUMNS = new Set<ColumnKey>([
  "curve",
  "polarity",
  "direction",
  "ion",
  "ioff",
  "logRatio",
  "ig",
  "vth",
  "ss",
  "source"
]);

function MetricCard({
  label,
  value,
  tone
}: {
  label: string;
  value: string;
  tone?: "primary" | "good";
}) {
  return (
    <div className={tone ? `detail-metric ${tone}` : "detail-metric"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function DatabaseWorkspace({
  selection,
  onSelectionChange
}: {
  selection: DatabaseSelectionState;
  onSelectionChange: (selection: DatabaseSelectionState) => void;
}) {
  const [status, setStatus] = useState<DatabaseStatus | null>(null);
  const [options, setOptions] = useState<DatabaseOptions>({
    source_kinds: [],
    polarities: [],
    directions: []
  });
  const [filters, setFilters] = useState<CurveFilters>({});
  const [appliedFilters, setAppliedFilters] = useState<CurveFilters>({});
  const [orderBy, setOrderBy] = useState("modified_at_desc");
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [offset, setOffset] = useState(0);
  const [reloadKey, setReloadKey] = useState(0);
  const [list, setList] = useState<CurveListResponse | null>(null);
  const [selected, setSelected] = useState<CurveSummary | null>(null);
  const [calendarSelectedId, setCalendarSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<CurveDetail | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [browseView, setBrowseView] = useState<"list" | "calendar">("list");
  const [visibleColumns, setVisibleColumns] = useState<Set<ColumnKey>>(
    () => new Set(DEFAULT_VISIBLE_COLUMNS)
  );
  const [showColumnChooser, setShowColumnChooser] = useState(false);

  const cleanedFilters = useMemo(() => cleanFilters(filters), [filters]);
  const activeFilters = useMemo(
    () => Object.entries(cleanedFilters) as [FilterKey, string][],
    [cleanedFilters]
  );
  const filterErrors = useMemo(() => numericFilterErrors(filters), [filters]);
  const hasFilterErrors = filterErrors.length > 0;
  const currentFilterFingerprint = useMemo(
    () => filterFingerprint(cleanedFilters),
    [cleanedFilters]
  );
  const appliedFilterFingerprint = useMemo(
    () => filterFingerprint(appliedFilters),
    [appliedFilters]
  );
  const filtersPending = !hasFilterErrors && currentFilterFingerprint !== appliedFilterFingerprint;
  const selectedIds = useMemo(() => new Set(selection.selectedIds), [selection.selectedIds]);
  const visibleColumnKeys = useMemo(() => visibleColumns, [visibleColumns]);
  const activeCurveId = calendarSelectedId ?? selected?.curve_id ?? null;

  useEffect(() => {
    void Promise.all([getDatabaseStatus(), getDatabaseOptions()])
      .then(([nextStatus, nextOptions]) => {
        setStatus(nextStatus);
        setOptions(nextOptions);
      })
      .catch((caught) => {
        setError(caught instanceof Error ? caught.message : "Database is unavailable");
      });
  }, []);

  useEffect(() => {
    if (hasFilterErrors) return undefined;
    const timer = window.setTimeout(() => {
      setOffset(0);
      setAppliedFilters(cleanedFilters);
    }, 350);
    return () => window.clearTimeout(timer);
  }, [cleanedFilters, currentFilterFingerprint, hasFilterErrors]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setLoadingList(true);
    setError(null);
    void listDatabaseCurves(appliedFilters, pageSize, offset, orderBy, controller.signal)
      .then((response) => {
        if (!active) return;
        setList(response);
        const nextSelected = response.items[0] ?? null;
        setSelected((current) => {
          const refreshed = response.items.find((item) => item.curve_id === current?.curve_id);
          return refreshed ?? nextSelected;
        });
      })
      .catch((caught) => {
        if (!active || (caught instanceof Error && caught.name === "AbortError")) return;
        setError(caught instanceof Error ? caught.message : "Could not load curves");
      })
      .finally(() => {
        if (active) setLoadingList(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [appliedFilters, offset, orderBy, pageSize, reloadKey]);

  useEffect(() => {
    if (!activeCurveId) {
      setDetail(null);
      return undefined;
    }
    const controller = new AbortController();
    let active = true;
    setLoadingDetail(true);
    setDetail(null);
    void getDatabaseCurve(activeCurveId, controller.signal)
      .then((response) => {
        if (active) setDetail(response);
      })
      .catch((caught) => {
        if (!active || (caught instanceof Error && caught.name === "AbortError")) return;
        setError(caught instanceof Error ? caught.message : "Could not load curve detail");
      })
      .finally(() => {
        if (active) setLoadingDetail(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [activeCurveId]);

  const total = list?.total ?? 0;
  const pageStart = total > 0 ? (list?.offset ?? 0) + 1 : 0;
  const pageEnd = Math.min((list?.offset ?? 0) + (list?.items.length ?? 0), total);
  const canGoBack = (list?.offset ?? 0) > 0;
  const canGoForward = pageEnd < total;
  const pageIds = useMemo(() => (list?.items ?? []).map((curve) => curve.curve_id), [list]);
  const pageSelected = pageIds.length > 0 && pageIds.every((curveId) => selectedIds.has(curveId));
  const selectionCount = selection.allFiltered ? total : selection.selectedIds.length;

  useEffect(() => {
    onSelectionChange({
      selectedIds: selection.selectedIds,
      allFiltered: selection.allFiltered,
      filters: appliedFilters,
      total
    });
  }, [appliedFilters, onSelectionChange, selection.allFiltered, selection.selectedIds, total]);

  function applyFiltersNow() {
    if (hasFilterErrors) return;
    setOffset(0);
    setAppliedFilters(cleanedFilters);
    setReloadKey((current) => current + 1);
  }

  function resetFilters() {
    setFilters({});
    setAppliedFilters({});
    setOffset(0);
  }

  function selectCurve(curve: CurveSummary) {
    setCalendarSelectedId(null);
    setSelected(curve);
  }

  function selectCurveById(curveId: string) {
    setCalendarSelectedId(curveId);
    const matched = list?.items.find((curve) => curve.curve_id === curveId);
    if (matched) {
      setSelected(matched);
    }
  }

  function toggleColumn(key: ColumnKey) {
    setVisibleColumns((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        if (next.size === 1) return next;
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  function toggleCurveSelection(curveId: string) {
    const next = new Set(selection.allFiltered ? [] : selection.selectedIds);
    if (next.has(curveId)) {
      next.delete(curveId);
    } else {
      next.add(curveId);
    }
    onSelectionChange({
      selectedIds: [...next],
      allFiltered: false,
      filters: appliedFilters,
      total
    });
  }

  function selectCurveIds(curveIds: string[]) {
    const next = new Set(selection.allFiltered ? [] : selection.selectedIds);
    curveIds.forEach((curveId) => next.add(curveId));
    onSelectionChange({
      selectedIds: [...next],
      allFiltered: false,
      filters: appliedFilters,
      total
    });
  }

  function togglePageSelection() {
    const next = new Set(selection.allFiltered ? [] : selection.selectedIds);
    if (pageSelected) {
      pageIds.forEach((curveId) => next.delete(curveId));
    } else {
      pageIds.forEach((curveId) => next.add(curveId));
    }
    onSelectionChange({
      selectedIds: [...next],
      allFiltered: false,
      filters: appliedFilters,
      total
    });
  }

  function selectAllFiltered() {
    onSelectionChange({
      selectedIds: [],
      allFiltered: true,
      filters: appliedFilters,
      total
    });
  }

  function clearSelection() {
    onSelectionChange({
      selectedIds: [],
      allFiltered: false,
      filters: appliedFilters,
      total
    });
  }

  async function exportSelection() {
    setExporting(true);
    setError(null);
    try {
      await exportDatabaseSelection({
        selectedIds: selection.allFiltered ? [] : selection.selectedIds,
        allFiltered: selection.allFiltered,
        filters: appliedFilters,
        total
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not export selection");
    } finally {
      setExporting(false);
    }
  }

  function renderCell(curve: CurveSummary, key: ColumnKey) {
    switch (key) {
      case "curve":
        return (
          <span className="curve-id-cell">
            <strong>{curve.curve_id}</strong>
            <small>{curve.setup_title ?? curve.source_kind}</small>
          </span>
        );
      case "polarity":
        return <span className="database-pill">{curve.polarity}</span>;
      case "direction":
        return curve.direction;
      case "ion":
        return scientific(curve.ion);
      case "ioff":
        return scientific(curve.ioff);
      case "logRatio":
        return logRatio(curve.log_ratio);
      case "ig":
        return curve.has_gate_current ? "yes" : "no";
      case "vth":
        return fixed(curve.vth, 2);
      case "ss":
        return fixed(curve.ss_mv_dec, 1);
      case "sourceKind":
        return curve.source_kind;
      case "source":
        return sourceName(curve.source_path);
      case "modified":
        return curve.modified_at?.slice(0, 10) ?? "-";
    }
  }

  return (
    <main className="database-workspace">
      <section className="database-heading">
        <div>
          <Database size={22} />
          <h1>Measurement database</h1>
        </div>
        <StatStrip status={status} />
      </section>
      {error ? <div className="error-banner" role="alert">{error}</div> : null}
      <div className="database-layout">
        <aside className="database-filters">
          <div className="filter-title-row">
            <div className="filter-title">
              <Filter size={16} />
              Filters
            </div>
            <span className={hasFilterErrors ? "filter-state invalid" : filtersPending ? "filter-state pending" : "filter-state"}>
              {hasFilterErrors ? "Invalid" : filtersPending ? "Updating" : "Current"}
            </span>
          </div>
          {activeFilters.length > 0 ? (
            <div className="active-filter-list" aria-label="Active filters">
              {activeFilters.map(([key, value]) => (
                <button
                  key={key}
                  type="button"
                  className="active-filter-chip"
                  onClick={() => setFilters((current) => removeFilter(current, key))}
                >
                  {chipLabel(key, value)}
                  <X size={12} />
                </button>
              ))}
            </div>
          ) : null}
          {hasFilterErrors ? (
            <div className="filter-error">
              Check numeric filters: {filterErrors.join(", ")}
            </div>
          ) : null}
          <label>
            Source text
            <div className="filter-search">
              <Search size={14} />
              <input
                value={filters.source_search ?? ""}
                onChange={(event) =>
                  setFilters((current) =>
                    patchFilter(current, "source_search", event.target.value)
                  )
                }
                placeholder="path, sample, material"
              />
            </div>
          </label>
          <label>
            Source kind
            <select
              value={filters.source_kind ?? ""}
              onChange={(event) =>
                setFilters((current) =>
                  patchFilter(current, "source_kind", event.target.value)
                )
              }
            >
              <option value="">All</option>
              {options.source_kinds.map((kind) => (
                <option key={kind} value={kind}>{kind}</option>
              ))}
            </select>
          </label>
          <label>
            Has Ig
            <select
              value={filters.has_gate_current ?? ""}
              onChange={(event) =>
                setFilters((current) =>
                  patchFilter(current, "has_gate_current", event.target.value)
                )
              }
            >
              <option value="">All</option>
              <option value="true">With Ig</option>
              <option value="false">No Ig</option>
            </select>
          </label>
          <label>
            Polarity
            <select
              value={filters.polarity ?? ""}
              onChange={(event) =>
                setFilters((current) => patchFilter(current, "polarity", event.target.value))
              }
            >
              <option value="">All</option>
              {options.polarities.map((polarity) => (
                <option key={polarity} value={polarity}>{polarity}</option>
              ))}
            </select>
          </label>
          <label>
            Sweep direction
            <select
              value={filters.direction ?? ""}
              onChange={(event) =>
                setFilters((current) => patchFilter(current, "direction", event.target.value))
              }
            >
              <option value="">All</option>
              {options.directions.map((direction) => (
                <option key={direction} value={direction}>{direction}</option>
              ))}
            </select>
          </label>
          <div className="filter-pair">
            <label>
              From
              <input
                type="date"
                value={filters.date_from ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "date_from", event.target.value))
                }
              />
            </label>
            <label>
              To
              <input
                type="date"
                value={filters.date_to ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "date_to", event.target.value))
                }
              />
            </label>
          </div>
          <div className="filter-pair">
            <label>
              Ion min
              <input
                inputMode="decimal"
                value={filters.ion_min ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "ion_min", event.target.value))
                }
                placeholder="1e-7"
              />
            </label>
            <label>
              Ion max
              <input
                inputMode="decimal"
                value={filters.ion_max ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "ion_max", event.target.value))
                }
                placeholder="1e-3"
              />
            </label>
          </div>
          <div className="filter-pair">
            <label>
              Ioff min
              <input
                inputMode="decimal"
                value={filters.ioff_min ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "ioff_min", event.target.value))
                }
                placeholder="1e-14"
              />
            </label>
            <label>
              Ioff max
              <input
                inputMode="decimal"
                value={filters.ioff_max ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "ioff_max", event.target.value))
                }
                placeholder="1e-8"
              />
            </label>
          </div>
          <div className="filter-pair">
            <label>
              Ion/Ioff min
              <input
                inputMode="decimal"
                value={filters.ion_ioff_ratio_min ?? ""}
                onChange={(event) =>
                  setFilters((current) =>
                    patchFilter(current, "ion_ioff_ratio_min", event.target.value)
                  )
                }
                placeholder="1e4"
              />
            </label>
            <label>
              Ion/Ioff max
              <input
                inputMode="decimal"
                value={filters.ion_ioff_ratio_max ?? ""}
                onChange={(event) =>
                  setFilters((current) =>
                    patchFilter(current, "ion_ioff_ratio_max", event.target.value)
                  )
                }
                placeholder="1e8"
              />
            </label>
          </div>
          <div className="filter-pair">
            <label>
              Vth min
              <input
                inputMode="decimal"
                value={filters.vth_min ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "vth_min", event.target.value))
                }
                placeholder="-5"
              />
            </label>
            <label>
              Vth max
              <input
                inputMode="decimal"
                value={filters.vth_max ?? ""}
                onChange={(event) =>
                  setFilters((current) => patchFilter(current, "vth_max", event.target.value))
                }
                placeholder="20"
              />
            </label>
          </div>
          <div className="filter-pair">
            <label>
              SS min
              <input
                inputMode="decimal"
                value={filters.ss_mv_dec_min ?? ""}
                onChange={(event) =>
                  setFilters((current) =>
                    patchFilter(current, "ss_mv_dec_min", event.target.value)
                  )
                }
                placeholder="50"
              />
            </label>
            <label>
              SS max
              <input
                inputMode="decimal"
                value={filters.ss_mv_dec_max ?? ""}
                onChange={(event) =>
                  setFilters((current) =>
                    patchFilter(current, "ss_mv_dec_max", event.target.value)
                  )
                }
                placeholder="500"
              />
            </label>
          </div>
          <div className="filter-actions">
            <button className="button primary" onClick={applyFiltersNow} disabled={hasFilterErrors}>
              <RefreshCw size={15} />
              Apply now
            </button>
            <button
              className="button secondary"
              onClick={resetFilters}
              disabled={activeFilters.length === 0 && Object.keys(appliedFilters).length === 0}
            >
              <RotateCcw size={15} />
              Reset
            </button>
          </div>
        </aside>
        <section className="database-results">
          <div className="database-toolbar">
            <div className="database-result-count">
              <strong>{total.toLocaleString()}</strong> curves
              {list && total > 0 ? <span> showing {pageStart}-{pageEnd}</span> : null}
              {selectionCount > 0 ? <span> selected {selectionCount.toLocaleString()}</span> : null}
              {loadingList ? <span className="database-sync-state">Loading</span> : null}
            </div>
            <div className="database-view-toggle">
              <button
                className={browseView === "list" ? "button secondary compact active" : "button secondary compact"}
                onClick={() => setBrowseView("list")}
              >
                <Search size={15} />
                List
              </button>
              <button
                className={browseView === "calendar" ? "button secondary compact active" : "button secondary compact"}
                onClick={() => setBrowseView("calendar")}
              >
                <CalendarDays size={15} />
                Calendar
              </button>
            </div>
            <div className="database-selection-actions">
              <button
                className="button secondary compact"
                onClick={selectAllFiltered}
                disabled={total === 0 || loadingList}
              >
                <CheckSquare size={15} />
                Select filtered
              </button>
              <button
                className="button secondary compact"
                onClick={clearSelection}
                disabled={selectionCount === 0}
              >
                <X size={15} />
                Clear
              </button>
              <button
                className="button primary compact"
                onClick={exportSelection}
                disabled={selectionCount === 0 || exporting}
              >
                <Download size={15} />
                {exporting ? "Exporting" : "Export"}
              </button>
            </div>
            <button
              className="button secondary compact"
              onClick={() => setReloadKey((current) => current + 1)}
              disabled={loadingList}
              aria-label="Refresh results"
              title="Refresh results"
            >
              <RefreshCw size={15} className={loadingList ? "spin" : undefined} />
            </button>
            {browseView === "list" ? (
              <>
                <div className="column-chooser">
                  <button
                    className="button secondary compact"
                    onClick={() => setShowColumnChooser((current) => !current)}
                    aria-expanded={showColumnChooser}
                  >
                    <Columns3 size={15} />
                    Columns
                  </button>
                  {showColumnChooser ? (
                    <div className="column-chooser-menu">
                      {ALL_COLUMNS.map((column) => (
                        <label key={column.key}>
                          <input
                            type="checkbox"
                            checked={visibleColumnKeys.has(column.key)}
                            onChange={() => toggleColumn(column.key)}
                          />
                          {column.label}
                        </label>
                      ))}
                    </div>
                  ) : null}
                </div>
                <label>
                  Sort
                  <select
                    value={orderBy}
                    onChange={(event) => {
                      setOrderBy(event.target.value);
                      setOffset(0);
                    }}
                  >
                    <option value="modified_at_desc">Newest file</option>
                    <option value="modified_at_asc">Oldest file</option>
                    <option value="ion_desc">Ion high to low</option>
                    <option value="ion_asc">Ion low to high</option>
                    <option value="ratio_desc">Ion/Ioff high to low</option>
                    <option value="vth_asc">Vth low to high</option>
                    <option value="vth_desc">Vth high to low</option>
                  </select>
                </label>
                <label className="page-size-control">
                  Rows
                  <select
                    value={pageSize}
                    onChange={(event) => {
                      setPageSize(Number(event.target.value));
                      setOffset(0);
                    }}
                  >
                    {PAGE_SIZE_OPTIONS.map((size) => (
                      <option key={size} value={size}>{size}</option>
                    ))}
                  </select>
                </label>
                <button
                  className="button secondary compact"
                  disabled={!canGoBack || loadingList}
                  onClick={() => setOffset(Math.max(0, offset - pageSize))}
                  aria-label="Previous page"
                >
                  <ChevronLeft size={15} />
                </button>
                <button
                  className="button secondary compact"
                  disabled={!canGoForward || loadingList}
                  onClick={() => setOffset(offset + pageSize)}
                  aria-label="Next page"
                >
                  <ChevronRight size={15} />
                </button>
              </>
            ) : null}
          </div>
          {browseView === "list" ? (
            <div className="database-table-wrap">
              <table className="database-table">
                <thead>
                  <tr>
                    <th className="selection-column">
                      <button
                        type="button"
                        onClick={togglePageSelection}
                        aria-label={pageSelected ? "Unselect page" : "Select page"}
                      >
                        {pageSelected ? <CheckSquare size={15} /> : <Square size={15} />}
                      </button>
                    </th>
                    {ALL_COLUMNS.filter((column) => visibleColumnKeys.has(column.key)).map((column) => (
                      <th key={column.key}>{column.label}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(list?.items ?? []).map((curve) => (
                    <tr
                      key={curve.curve_id}
                      className={activeCurveId === curve.curve_id ? "selected" : ""}
                      onClick={() => selectCurve(curve)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          selectCurve(curve);
                        }
                      }}
                      tabIndex={0}
                      aria-selected={activeCurveId === curve.curve_id}
                    >
                      <td className="selection-column">
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            toggleCurveSelection(curve.curve_id);
                          }}
                          aria-label={selectedIds.has(curve.curve_id) ? "Unselect curve" : "Select curve"}
                        >
                          {selectedIds.has(curve.curve_id) || selection.allFiltered ? (
                            <CheckSquare size={15} />
                          ) : (
                            <Square size={15} />
                          )}
                        </button>
                      </td>
                      {ALL_COLUMNS.filter((column) => visibleColumnKeys.has(column.key)).map((column) => (
                        <td
                          key={column.key}
                          title={column.key === "source" ? curve.source_path : undefined}
                        >
                          {renderCell(curve, column.key)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              {!loadingList && list && list.items.length === 0 ? (
                <div className="database-empty-results">
                  <Search size={24} />
                  <strong>No curves match</strong>
                  <button className="button secondary compact" onClick={resetFilters}>
                    Clear filters
                  </button>
                </div>
              ) : null}
              {loadingList ? <div className="table-loading">Loading database rows</div> : null}
            </div>
          ) : (
            <DatabaseCalendarPanel
              filters={appliedFilters}
              selectedCurveId={activeCurveId}
              selectedIds={selectedIds}
              onSelectCurve={selectCurveById}
              onSelectCurveIds={selectCurveIds}
              onToggleSelection={toggleCurveSelection}
            />
          )}
        </section>
        <aside className="database-detail">
          {!detail ? (
            <div className="database-empty-detail">
              {loadingDetail ? <RefreshCw size={30} className="spin" /> : <BarChart3 size={30} />}
              {loadingDetail ? "Loading curve detail" : "Select a curve"}
            </div>
          ) : (
            <>
              <div className="detail-heading">
                <div>
                  <span>{detail.source_kind}</span>
                  <h2>{detail.curve_id}</h2>
                </div>
                <strong>{detail.polarity}</strong>
              </div>
              <MiniCurve detail={detail} />
              <div className="detail-metric-grid">
                <MetricCard label="Ion" value={scientific(detail.ion)} tone="primary" />
                <MetricCard label="Ioff" value={scientific(detail.ioff)} />
                <MetricCard label="logRatio" value={logRatio(detail.log_ratio)} tone="good" />
                <MetricCard label="Vth" value={`${fixed(detail.vth, 2)} V`} />
                <MetricCard label="SS" value={`${fixed(detail.ss_mv_dec, 1)} mV/dec`} />
                <MetricCard label="Range" value={`${fixed(detail.voltage_min_v, 1)} to ${fixed(detail.voltage_max_v, 1)} V`} />
              </div>
              <dl className="detail-facts">
                <div><dt>Source</dt><dd title={detail.source_path}>{detail.source_path}</dd></div>
                <div><dt>Table</dt><dd>{detail.table_name}</dd></div>
                <div><dt>Setup</dt><dd>{detail.setup_title ?? "-"}</dd></div>
                <div><dt>Primitive</dt><dd>{detail.primitive_test ?? "-"}</dd></div>
                <div><dt>Columns</dt><dd>{detail.voltage_column} / {detail.current_column}</dd></div>
                <div><dt>Ig column</dt><dd>{detail.gate_current_column ?? "-"}</dd></div>
                <div><dt>Has Ig</dt><dd>{detail.has_gate_current ? "yes" : "no"}</dd></div>
                <div><dt>Raw points</dt><dd>{detail.raw_points.length}</dd></div>
                <div><dt>Ig points</dt><dd>{detail.gate_points.length}</dd></div>
                <div><dt>Aligned points</dt><dd>{detail.aligned_points.length}</dd></div>
                <div><dt>Direction</dt><dd>{detail.direction}</dd></div>
                <div><dt>Noise</dt><dd>{fixed(detail.noise_log_sigma, 3)}</dd></div>
              </dl>
              <div className="metadata-panel">
                <h3>
                  <SlidersHorizontal size={14} />
                  Test configuration
                </h3>
                {Object.entries(detail.metadata_json).slice(0, 14).map(([key, value]) => (
                  <div key={key}>
                    <span>{key}</span>
                    <code>{value.join(", ") || "-"}</code>
                  </div>
                ))}
              </div>
            </>
          )}
        </aside>
      </div>
    </main>
  );
}
