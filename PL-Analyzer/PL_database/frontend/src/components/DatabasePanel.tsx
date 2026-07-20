import { useEffect, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import Plot from "react-plotly.js";
import { getExportUrl } from "../api";
import type { FilterOptions, SpectrumDetail, SpectrumFilters, SpectrumRow, SpectraResponse } from "../types";
import { buildPreviewAxis, formatPreviewAxisSummary, summarizePreviewAxes } from "../utils/spectrumPreview";

interface DatabasePanelProps {
  activeSpectrumId: string | null;
  filterOptions: FilterOptions;
  filters: SpectrumFilters;
  selectedIds: string[];
  selectedSpectraLoading: boolean;
  selectedSpectra: SpectrumDetail[];
  spectra: SpectraResponse;
  onClearSelection: () => void;
  onFilterChange: (nextFilters: SpectrumFilters) => void;
  onToggleSelect: (spectrumId: string) => void;
}

type AxisKey = "x" | "y";
type AxisRange = [number, number];
type SortDirection = "asc" | "desc";
type CategoryFilterKey =
  | "spectrum_type"
  | "belonging"
  | "acquisition_mode"
  | "substrate"
  | "x_axis_unit"
  | "analysis_material"
  | "analysis_family"
  | "analysis_status";
type NumericFilterKey =
  | "n_points"
  | "member_count"
  | "trace_count"
  | "scan_size_x"
  | "scan_size_y"
  | "grid_x"
  | "grid_y";
type DynamicFilterKey = CategoryFilterKey | NumericFilterKey;
type ColumnKey =
  | "spectrum_id"
  | "sample_id"
  | "belonging"
  | "spectrum_type"
  | "acquisition_mode"
  | "source"
  | "substrate"
  | "analysis_material"
  | "analysis_family"
  | "analysis_status"
  | "member_count"
  | "x_axis_unit"
  | "n_points"
  | "trace_count"
  | "scan_size_x"
  | "scan_size_y"
  | "grid_x"
  | "grid_y"
  | "measurement_time"
  | "file_path"
  | "source_tree_path";

interface OverlayAxisRanges {
  x: AxisRange | null;
  y: AxisRange | null;
}

interface OverlayAxisDraft {
  xMin: string;
  xMax: string;
  yMin: string;
  yMax: string;
}

interface CategoryFilterDefinition {
  kind: "category";
  key: CategoryFilterKey;
  label: string;
  optionsKey: keyof FilterOptions;
}

interface NumericFilterDefinition {
  kind: "numeric";
  key: NumericFilterKey;
  label: string;
}

type FilterDefinition = CategoryFilterDefinition | NumericFilterDefinition;

interface ColumnDefinition {
  key: ColumnKey;
  label: string;
  className?: string;
  render: (row: SpectrumRow) => ReactNode;
}

const CATEGORY_FILTERS: CategoryFilterDefinition[] = [
  { kind: "category", key: "spectrum_type", label: "Type", optionsKey: "spectrum_type" },
  { kind: "category", key: "belonging", label: "Belonging", optionsKey: "belonging" },
  { kind: "category", key: "acquisition_mode", label: "Acquisition", optionsKey: "acquisition_mode" },
  { kind: "category", key: "substrate", label: "Substrate", optionsKey: "substrate" },
  { kind: "category", key: "x_axis_unit", label: "Axis unit", optionsKey: "x_axis_unit" },
  { kind: "category", key: "analysis_material", label: "Analysis material", optionsKey: "analysis_material" },
  { kind: "category", key: "analysis_family", label: "Analysis family", optionsKey: "analysis_family" },
  { kind: "category", key: "analysis_status", label: "Analysis status", optionsKey: "analysis_status" }
];

const NUMERIC_FILTERS: NumericFilterDefinition[] = [
  { kind: "numeric", key: "n_points", label: "Points" },
  { kind: "numeric", key: "member_count", label: "Members" },
  { kind: "numeric", key: "trace_count", label: "Trace count" },
  { kind: "numeric", key: "scan_size_x", label: "Scan size X" },
  { kind: "numeric", key: "scan_size_y", label: "Scan size Y" },
  { kind: "numeric", key: "grid_x", label: "Grid X" },
  { kind: "numeric", key: "grid_y", label: "Grid Y" }
];

const FILTER_DEFINITIONS: FilterDefinition[] = [...CATEGORY_FILTERS, ...NUMERIC_FILTERS];

const COLUMN_DEFINITIONS: ColumnDefinition[] = [
  { key: "spectrum_id", label: "Dataset ID", render: (row) => row.spectrum_id },
  { key: "sample_id", label: "Sample", render: (row) => row.sample_id ?? "-" },
  { key: "belonging", label: "Belonging", render: (row) => row.belonging ?? "-" },
  { key: "spectrum_type", label: "Type", render: (row) => row.spectrum_type ?? "-" },
  { key: "acquisition_mode", label: "Acquisition", render: (row) => row.acquisition_mode ?? "-" },
  { key: "source", label: "Source", render: (row) => row.source ?? "-" },
  { key: "substrate", label: "Substrate", render: (row) => row.substrate ?? "-" },
  { key: "analysis_material", label: "Material", render: (row) => row.analysis_material ?? "-" },
  { key: "analysis_family", label: "Family", render: (row) => row.analysis_family ?? "-" },
  { key: "analysis_status", label: "Analysis", render: (row) => row.analysis_status ?? "-" },
  { key: "member_count", label: "Members", render: (row) => row.member_count ?? 1 },
  { key: "x_axis_unit", label: "Unit", render: (row) => row.x_axis_unit ?? "-" },
  { key: "n_points", label: "Points", render: (row) => row.n_points },
  { key: "trace_count", label: "Traces", render: (row) => row.trace_count ?? "-" },
  { key: "scan_size_x", label: "Scan X", render: (row) => row.scan_size_x ?? "-" },
  { key: "scan_size_y", label: "Scan Y", render: (row) => row.scan_size_y ?? "-" },
  { key: "grid_x", label: "Grid X", render: (row) => row.grid_x ?? "-" },
  { key: "grid_y", label: "Grid Y", render: (row) => row.grid_y ?? "-" },
  { key: "measurement_time", label: "Measured", render: (row) => row.measurement_time ?? "-" },
  { key: "file_path", label: "File path", className: "truncate-cell", render: (row) => row.file_path },
  { key: "source_tree_path", label: "Tree path", className: "truncate-cell", render: (row) => row.source_tree_path }
];

const DEFAULT_VISIBLE_COLUMNS: ColumnKey[] = [
  "spectrum_id",
  "spectrum_type",
  "acquisition_mode",
  "source",
  "substrate",
  "analysis_material",
  "n_points"
];

export function DatabasePanel(props: DatabasePanelProps) {
  const [activeFilterKeys, setActiveFilterKeys] = useState<DynamicFilterKey[]>([]);
  const [nextFilterKey, setNextFilterKey] = useState<DynamicFilterKey>(FILTER_DEFINITIONS[0].key);
  const [visibleColumns, setVisibleColumns] = useState<ColumnKey[]>(DEFAULT_VISIBLE_COLUMNS);
  const [overlayRanges, setOverlayRanges] = useState<OverlayAxisRanges>(() => createEmptyOverlayRanges());
  const [overlayDraft, setOverlayDraft] = useState<OverlayAxisDraft>(() => createEmptyOverlayDraft());
  const activeRow = props.spectra.items.find((row) => row.spectrum_id === props.activeSpectrumId) ?? null;
  const activeSpectrum =
    props.selectedSpectra.find(
      (spectrum) =>
        spectrum.spectrum_id === props.activeSpectrumId ||
        spectrum.representative_spectrum_id === props.activeSpectrumId
    ) ?? null;
  const activePreview = activeSpectrum ? buildPreviewAxis(activeSpectrum) : null;
  const previewTraces = props.selectedSpectra.map((spectrum) => ({
    spectrum,
    preview: buildPreviewAxis(spectrum)
  }));
  const previewSummary = summarizePreviewAxes(previewTraces.map((item) => item.preview));
  const selectionRevision = props.selectedIds.join("|");
  const availableFilters = FILTER_DEFINITIONS.filter((definition) => !activeFilterKeys.includes(definition.key));
  const activeFilterDefinitions = activeFilterKeys
    .map((key) => FILTER_DEFINITIONS.find((definition) => definition.key === key))
    .filter((definition): definition is FilterDefinition => Boolean(definition));
  const visibleColumnDefinitions = COLUMN_DEFINITIONS.filter((definition) => visibleColumns.includes(definition.key));

  useEffect(() => {
    setOverlayRanges(createEmptyOverlayRanges());
    setOverlayDraft(createEmptyOverlayDraft());
  }, [previewSummary.axisTitle, selectionRevision]);

  useEffect(() => {
    const appliedKeys = readAppliedFilterKeys(props.filters);
    if (appliedKeys.length > 0) {
      setActiveFilterKeys((current) => [...current, ...appliedKeys.filter((key) => !current.includes(key))]);
    }
  }, [props.filters]);

  useEffect(() => {
    if (availableFilters.length === 0) {
      return;
    }
    if (!availableFilters.some((definition) => definition.key === nextFilterKey)) {
      setNextFilterKey(availableFilters[0].key);
    }
  }, [availableFilters, nextFilterKey]);

  const sortBy = props.filters.sort_by as ColumnKey | undefined;
  const sortDir = props.filters.sort_dir ?? "asc";

  function handleOverlayDraftChange(field: keyof OverlayAxisDraft, value: string) {
    setOverlayDraft((current) => ({
      ...current,
      [field]: value
    }));
  }

  function applyAxisRange(axis: AxisKey) {
    const nextRange =
      axis === "x"
        ? resolveAxisRangeDraft(overlayDraft.xMin, overlayDraft.xMax)
        : resolveAxisRangeDraft(overlayDraft.yMin, overlayDraft.yMax);

    if (!nextRange) {
      return;
    }

    setOverlayRanges((current) => ({
      ...current,
      [axis]: nextRange
    }));
  }

  function resetAxisRange(axis: AxisKey) {
    setOverlayRanges((current) => ({
      ...current,
      [axis]: null
    }));
    setOverlayDraft((current) =>
      axis === "x"
        ? { ...current, xMin: "", xMax: "" }
        : { ...current, yMin: "", yMax: "" }
    );
  }

  function handleOverlayRelayout(event: Readonly<Record<string, unknown>>) {
    const xRange = readAxisRange(event, "xaxis");
    const yRange = readAxisRange(event, "yaxis");
    const xAutorange = event["xaxis.autorange"] === true;
    const yAutorange = event["yaxis.autorange"] === true;

    if (!xRange && !yRange && !xAutorange && !yAutorange) {
      return;
    }

    setOverlayRanges((current) => ({
      x: xAutorange ? null : xRange ?? current.x,
      y: yAutorange ? null : yRange ?? current.y
    }));

    setOverlayDraft((current) => ({
      xMin: xAutorange ? "" : xRange ? formatAxisValue(xRange[0]) : current.xMin,
      xMax: xAutorange ? "" : xRange ? formatAxisValue(xRange[1]) : current.xMax,
      yMin: yAutorange ? "" : yRange ? formatAxisValue(yRange[0]) : current.yMin,
      yMax: yAutorange ? "" : yRange ? formatAxisValue(yRange[1]) : current.yMax
    }));
  }

  function updateFilter<K extends keyof SpectrumFilters>(key: K, value: SpectrumFilters[K]) {
    props.onFilterChange({
      ...props.filters,
      [key]: value
    });
  }

  function addFilter() {
    if (activeFilterKeys.includes(nextFilterKey)) {
      return;
    }
    setActiveFilterKeys((current) => [...current, nextFilterKey]);
  }

  function removeFilter(key: DynamicFilterKey) {
    const nextFilters = { ...props.filters };
    if (isNumericFilterKey(key)) {
      delete nextFilters[`${key}_min`];
      delete nextFilters[`${key}_max`];
    } else {
      delete nextFilters[key];
    }
    setActiveFilterKeys((current) => current.filter((item) => item !== key));
    props.onFilterChange(nextFilters);
  }

  function clearFilters() {
    setActiveFilterKeys([]);
    props.onFilterChange({
      sort_by: props.filters.sort_by,
      sort_dir: props.filters.sort_dir
    });
  }

  function updateNumericFilter(key: NumericFilterKey, boundary: "min" | "max", rawValue: string) {
    const filterKey = `${key}_${boundary}` as keyof SpectrumFilters;
    updateFilter(filterKey, rawValue === "" ? undefined : Number(rawValue));
  }

  function handleSort(column: ColumnKey) {
    const nextDirection: SortDirection =
      props.filters.sort_by === column && props.filters.sort_dir === "asc" ? "desc" : "asc";
    props.onFilterChange({
      ...props.filters,
      sort_by: column,
      sort_dir: nextDirection
    });
  }

  function toggleVisibleColumn(column: ColumnKey) {
    setVisibleColumns((current) => {
      if (current.includes(column)) {
        return current.length === 1 ? current : current.filter((item) => item !== column);
      }
      return [...current, column];
    });
  }

  return (
    <section className="database-shell">
      <aside className="card database-filter-panel">
        <div className="card-head database-panel-head">
          <div>
            <p className="eyebrow">Database</p>
            <h2>Filters</h2>
          </div>
          <button className="secondary-button compact-button" onClick={clearFilters} type="button">
            Clear
          </button>
        </div>

        <label className="field">
          <span>Search</span>
          <input
            placeholder="ID, path, notes..."
            value={props.filters.search ?? ""}
            onChange={(event) => updateFilter("search", event.target.value || undefined)}
          />
        </label>

        <div className="filter-builder">
          {activeFilterDefinitions.map((definition) => (
            <DynamicFilterControl
              key={definition.key}
              definition={definition}
              filterOptions={props.filterOptions}
              filters={props.filters}
              onCategoryChange={(key, value) => updateFilter(key, value || undefined)}
              onNumericChange={updateNumericFilter}
              onRemove={removeFilter}
            />
          ))}
        </div>

        <div className="add-filter-row">
          <label className="field">
            <span>Add parameter</span>
            <select
              disabled={availableFilters.length === 0}
              value={nextFilterKey}
              onChange={(event) => setNextFilterKey(event.target.value as DynamicFilterKey)}
            >
              {availableFilters.map((definition) => (
                <option key={definition.key} value={definition.key}>
                  {definition.label}
                </option>
              ))}
            </select>
          </label>
          <button className="primary-button add-filter-button" disabled={availableFilters.length === 0} onClick={addFilter} type="button">
            Add
          </button>
        </div>
      </aside>

      <section className="card database-table-panel">
        <div className="card-head database-panel-head">
          <div>
            <p className="eyebrow">List and preview</p>
            <h2>Data entries</h2>
          </div>
          <div className="action-row">
            <button
              className="secondary-button"
              disabled={props.selectedIds.length === 0}
              onClick={props.onClearSelection}
              type="button"
            >
              Clear selection
            </button>
            {props.selectedIds.length > 0 ? (
              <a className="secondary-link" href={getExportUrl(props.selectedIds)}>
                Download CSV
              </a>
            ) : null}
          </div>
        </div>

        <div className="database-toolbar">
          <div className="database-selection-summary">
            <strong>{props.spectra.total}</strong>
            <span>matched</span>
            <span className="database-selection-divider" />
            <strong>{props.selectedIds.length}</strong>
            <span>selected</span>
          </div>
          <span className="database-sort-summary">
            {sortBy ? `Sorted by ${columnLabel(sortBy)} ${sortDir}` : "Sorted by newest"}
          </span>
          <details className="database-column-config">
            <summary>Configure columns</summary>
            <div className="column-picker-actions">
              <button className="secondary-button compact-button" onClick={() => setVisibleColumns(DEFAULT_VISIBLE_COLUMNS)} type="button">
                Default
              </button>
              <button className="secondary-button compact-button" onClick={() => setVisibleColumns(COLUMN_DEFINITIONS.map((item) => item.key))} type="button">
                All
              </button>
            </div>
            <div className="column-chip-grid database-column-chip-grid">
              {COLUMN_DEFINITIONS.map((column) => (
                <label key={column.key} className={`column-chip ${visibleColumns.includes(column.key) ? "column-chip-active" : ""}`}>
                  <input
                    checked={visibleColumns.includes(column.key)}
                    disabled={visibleColumns.length === 1 && visibleColumns.includes(column.key)}
                    onChange={() => toggleVisibleColumn(column.key)}
                    type="checkbox"
                  />
                  <span>{column.label}</span>
                </label>
              ))}
            </div>
          </details>
        </div>

        <div className="table-shell database-table-shell">
          <table className="table-selectable database-data-table">
            <thead>
              <tr>
                {visibleColumnDefinitions.map((column) => (
                  <th key={column.key}>
                    <button className="table-sort-button" onClick={() => handleSort(column.key)} type="button">
                      <span>{column.label}</span>
                      <span className="sort-indicator">{sortBy === column.key ? (sortDir === "asc" ? "▲" : "▼") : "↕"}</span>
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {props.spectra.items.map((row) => {
                const isSelected = props.selectedIds.includes(row.spectrum_id);
                const isActive = row.spectrum_id === props.activeSpectrumId;

                return (
                  <tr
                    key={row.spectrum_id}
                    aria-selected={isSelected}
                    className={`${isSelected ? "selected-row" : ""} ${isActive ? "active-row" : ""}`}
                    onClick={() => props.onToggleSelect(row.spectrum_id)}
                    onKeyDown={(event) => handleRowKeyDown(event, row.spectrum_id, props.onToggleSelect)}
                    tabIndex={0}
                  >
                    {visibleColumnDefinitions.map((column) => (
                      <td key={column.key} className={column.className}>
                        {column.render(row)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <aside className="database-inspector-panel">
        <section className="card database-preview-card">
          <div className="card-head database-panel-head">
            <div>
              <p className="eyebrow">Selection preview</p>
              <h2>Overlay</h2>
            </div>
          </div>
          {previewTraces.length > 0 ? (
            <>
              <div className="overlay-controls">
                <AxisRangeEditor
                  axisLabel={previewSummary.axisTitle}
                  canApply={isAxisRangeDraftValid(overlayDraft.xMin, overlayDraft.xMax)}
                  maxValue={overlayDraft.xMax}
                  minValue={overlayDraft.xMin}
                  onApply={() => applyAxisRange("x")}
                  onAuto={() => resetAxisRange("x")}
                  onMaxChange={(value) => handleOverlayDraftChange("xMax", value)}
                  onMinChange={(value) => handleOverlayDraftChange("xMin", value)}
                />
                <AxisRangeEditor
                  axisLabel="Intensity"
                  canApply={isAxisRangeDraftValid(overlayDraft.yMin, overlayDraft.yMax)}
                  maxValue={overlayDraft.yMax}
                  minValue={overlayDraft.yMin}
                  onApply={() => applyAxisRange("y")}
                  onAuto={() => resetAxisRange("y")}
                  onMaxChange={(value) => handleOverlayDraftChange("yMax", value)}
                  onMinChange={(value) => handleOverlayDraftChange("yMin", value)}
                />
              </div>
              {previewSummary.note ? <p className="helper-copy helper-copy-tight">{previewSummary.note}</p> : null}
              <Plot
                className="plot-frame plot-frame-compact database-preview-plot"
                data={previewTraces.map(({ spectrum, preview }, index) => ({
                  x: preview.values,
                  y: spectrum.intensity,
                  type: "scatter",
                  mode: "lines",
                  name: spectrum.sample_id || spectrum.spectrum_id,
                  line: {
                    width: props.selectedSpectra.length > 3 ? 1.7 : 2.1
                  },
                  opacity: Math.max(0.42, 1 - index * 0.05),
                  hovertemplate: `${spectrum.sample_id || spectrum.spectrum_id}<br>%{x:.4g}, %{y:.4g}<extra></extra>`
                }))}
                layout={{
                  autosize: true,
                  uirevision: `database-overlay:${previewSummary.axisTitle}:${selectionRevision}`,
                  paper_bgcolor: "rgba(0,0,0,0)",
                  plot_bgcolor: "rgba(0,0,0,0)",
                  margin: { t: 10, r: 10, b: 38, l: 44 },
                  hovermode: "closest",
                  legend: {
                    orientation: "h",
                    y: 1.1,
                    x: 0,
                    font: { size: 11, color: "#5e7a8d" }
                  },
                  xaxis: {
                    title: { text: previewSummary.axisTitle, standoff: 6, font: { size: 11, color: "#648095" } },
                    autorange: overlayRanges.x ? false : true,
                    range: overlayRanges.x ?? undefined,
                    showline: false,
                    zeroline: false,
                    gridcolor: "rgba(17, 34, 59, 0.06)"
                  },
                  yaxis: {
                    title: { text: "Intensity", standoff: 6, font: { size: 11, color: "#648095" } },
                    autorange: overlayRanges.y ? false : true,
                    range: overlayRanges.y ?? undefined,
                    showline: false,
                    zeroline: false,
                    gridcolor: "rgba(17, 34, 59, 0.06)"
                  }
                }}
                config={{ responsive: true, displaylogo: false, doubleClick: "reset+autosize" }}
                onRelayout={handleOverlayRelayout}
                useResizeHandler
              />
            </>
          ) : props.selectedIds.length > 0 ? (
            <p className="empty-state">
              {props.selectedSpectraLoading
                ? "Loading the selected spectrum preview..."
                : "The selected spectrum details are not available yet. Try selecting the row again in a moment."}
            </p>
          ) : (
            <p className="empty-state">Select one or more spectra to preview them here.</p>
          )}
        </section>

        <section className="card database-detail-card">
          <div className="card-head database-panel-head">
            <div>
              <p className="eyebrow">Active entry</p>
              <h2>Details</h2>
            </div>
          </div>
          {activeSpectrum ? (
            <SpectrumSummaryBlock leadSpectrum={activeSpectrum} leadPreview={activePreview} />
          ) : activeRow ? (
            <SpectrumRowSummaryBlock row={activeRow} loading={props.selectedSpectraLoading && props.selectedIds.includes(activeRow.spectrum_id)} />
          ) : (
            <p className="empty-state">Click a table row to inspect its parameters.</p>
          )}
        </section>
      </aside>
    </section>
  );
}

interface DynamicFilterControlProps {
  definition: FilterDefinition;
  filterOptions: FilterOptions;
  filters: SpectrumFilters;
  onCategoryChange: (key: CategoryFilterKey, value: string) => void;
  onNumericChange: (key: NumericFilterKey, boundary: "min" | "max", rawValue: string) => void;
  onRemove: (key: DynamicFilterKey) => void;
}

function DynamicFilterControl(props: DynamicFilterControlProps) {
  const definition = props.definition;
  if (definition.kind === "category") {
    return (
      <div className="dynamic-filter-card">
        <div className="dynamic-filter-head">
          <span>{definition.label}</span>
          <button className="icon-text-button" onClick={() => props.onRemove(definition.key)} type="button">
            Remove
          </button>
        </div>
        <select
          value={props.filters[definition.key] ?? ""}
          onChange={(event) => props.onCategoryChange(definition.key, event.target.value)}
        >
          <option value="">All</option>
          {props.filterOptions[definition.optionsKey].map((option) => (
            <option key={option} value={option}>
              {option}
            </option>
          ))}
        </select>
      </div>
    );
  }

  const minKey = `${definition.key}_min` as keyof SpectrumFilters;
  const maxKey = `${definition.key}_max` as keyof SpectrumFilters;
  return (
    <div className="dynamic-filter-card">
      <div className="dynamic-filter-head">
        <span>{definition.label}</span>
        <button className="icon-text-button" onClick={() => props.onRemove(definition.key)} type="button">
          Remove
        </button>
      </div>
      <div className="range-filter-row">
        <input
          placeholder="min"
          step="1"
          type="number"
          value={props.filters[minKey] ?? ""}
          onChange={(event) => props.onNumericChange(definition.key, "min", event.target.value)}
        />
        <input
          placeholder="max"
          step="1"
          type="number"
          value={props.filters[maxKey] ?? ""}
          onChange={(event) => props.onNumericChange(definition.key, "max", event.target.value)}
        />
      </div>
    </div>
  );
}

function isNumericFilterKey(key: DynamicFilterKey): key is NumericFilterKey {
  return NUMERIC_FILTERS.some((definition) => definition.key === key);
}

function readAppliedFilterKeys(filters: SpectrumFilters): DynamicFilterKey[] {
  return FILTER_DEFINITIONS.filter((definition) => {
    if (definition.kind === "category") {
      return Boolean(filters[definition.key]);
    }
    return filters[`${definition.key}_min`] != null || filters[`${definition.key}_max`] != null;
  }).map((definition) => definition.key);
}

function columnLabel(key: ColumnKey): string {
  return COLUMN_DEFINITIONS.find((column) => column.key === key)?.label ?? key;
}

function formatGrid(spectrum: SpectrumDetail): string {
  if (spectrum.grid_x == null || spectrum.grid_y == null) {
    return "-";
  }
  const sizeX = spectrum.scan_size_x ?? "?";
  const sizeY = spectrum.scan_size_y ?? "?";
  return `${spectrum.grid_x}, ${spectrum.grid_y} of ${sizeX} x ${sizeY}`;
}

function formatMeasurementConfig(config: Record<string, unknown>): string {
  const entries = Object.entries(config ?? {}).filter(([, value]) => value !== null && value !== "");
  if (entries.length === 0) {
    return "-";
  }
  return entries
    .slice(0, 6)
    .map(([key, value]) => `${key}:${String(value)}`)
    .join("  ");
}

function SpectrumSummaryBlock(props: { leadSpectrum: SpectrumDetail; leadPreview: ReturnType<typeof buildPreviewAxis> | null }) {
  const { leadSpectrum, leadPreview } = props;
  return (
    <div className="summary-block">
      <div className="summary-row">
        <span>Spectrum ID</span>
        <strong>{leadSpectrum.spectrum_id}</strong>
      </div>
      <div className="summary-row">
        <span>Sample</span>
        <strong>{leadSpectrum.sample_id ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Belonging</span>
        <strong>{leadSpectrum.belonging ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Source</span>
        <strong>{leadSpectrum.source ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Type</span>
        <strong>{leadSpectrum.spectrum_type ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Acquisition</span>
        <strong>{leadSpectrum.acquisition_mode ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Analysis material</span>
        <strong>{leadSpectrum.analysis_material ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Analysis family</span>
        <strong>{leadSpectrum.analysis_family ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Analysis status</span>
        <strong>{leadSpectrum.analysis_status ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Dataset members</span>
        <strong>{leadSpectrum.member_count ?? 1}</strong>
      </div>
      <div className="summary-row">
        <span>Grid</span>
        <strong>{formatGrid(leadSpectrum)}</strong>
      </div>
      <div className="summary-row">
        <span>Measurement time</span>
        <strong>{leadSpectrum.measurement_time ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Laser</span>
        <strong>{leadSpectrum.laser_wavelength ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Preview axis</span>
        <strong>{leadPreview ? formatPreviewAxisSummary(leadPreview) : "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Stored axis</span>
        <strong>{leadSpectrum.x_axis_unit ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>File</span>
        <code>{leadSpectrum.file_path}</code>
      </div>
      <div className="summary-row">
        <span>Tree path</span>
        <code>{leadSpectrum.source_tree_path}</code>
      </div>
      <div className="summary-row">
        <span>Representative trace</span>
        <code>{leadSpectrum.representative_spectrum_id ?? leadSpectrum.spectrum_id}</code>
      </div>
      <div className="summary-row">
        <span>Config</span>
        <code>{formatMeasurementConfig(leadSpectrum.measurement_config)}</code>
      </div>
    </div>
  );
}

function SpectrumRowSummaryBlock(props: { row: SpectrumRow; loading: boolean }) {
  const { row, loading } = props;
  return (
    <div className="summary-block">
      <div className="summary-row">
        <span>Spectrum ID</span>
        <strong>{row.spectrum_id}</strong>
      </div>
      <div className="summary-row">
        <span>Sample</span>
        <strong>{row.sample_id ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Belonging</span>
        <strong>{row.belonging ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Source</span>
        <strong>{row.source ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Type</span>
        <strong>{row.spectrum_type ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Acquisition</span>
        <strong>{row.acquisition_mode ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Analysis material</span>
        <strong>{row.analysis_material ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Analysis family</span>
        <strong>{row.analysis_family ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Analysis status</span>
        <strong>{row.analysis_status ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Dataset members</span>
        <strong>{row.member_count ?? 1}</strong>
      </div>
      <div className="summary-row">
        <span>Measurement time</span>
        <strong>{row.measurement_time ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>Stored axis</span>
        <strong>{row.x_axis_unit ?? "-"}</strong>
      </div>
      <div className="summary-row">
        <span>File</span>
        <code>{row.file_path}</code>
      </div>
      <div className="summary-row">
        <span>Tree path</span>
        <code>{row.source_tree_path}</code>
      </div>
      <p className="helper-copy helper-copy-tight">
        {loading
          ? "Loading full spectrum arrays and preview payload..."
          : "Summary metadata is available for this active row."}
      </p>
    </div>
  );
}

interface AxisRangeEditorProps {
  axisLabel: string;
  canApply: boolean;
  maxValue: string;
  minValue: string;
  onApply: () => void;
  onAuto: () => void;
  onMaxChange: (value: string) => void;
  onMinChange: (value: string) => void;
}

function AxisRangeEditor(props: AxisRangeEditorProps) {
  return (
    <div className="overlay-inline-editor">
      <span className="overlay-inline-label">{props.axisLabel}</span>
      <input
        className="overlay-inline-input"
        placeholder="min"
        step="any"
        type="number"
        value={props.minValue}
        onChange={(event) => props.onMinChange(event.target.value)}
      />
      <input
        className="overlay-inline-input"
        placeholder="max"
        step="any"
        type="number"
        value={props.maxValue}
        onChange={(event) => props.onMaxChange(event.target.value)}
      />
      <button className="secondary-button overlay-inline-button" disabled={!props.canApply} onClick={props.onApply} type="button">
        Apply
      </button>
      <button className="secondary-button overlay-inline-button" onClick={props.onAuto} type="button">
        Auto
      </button>
    </div>
  );
}

function createEmptyOverlayRanges(): OverlayAxisRanges {
  return { x: null, y: null };
}

function createEmptyOverlayDraft(): OverlayAxisDraft {
  return { xMin: "", xMax: "", yMin: "", yMax: "" };
}

function resolveAxisRangeDraft(minText: string, maxText: string): AxisRange | null {
  const minimum = Number(minText);
  const maximum = Number(maxText);

  if (!Number.isFinite(minimum) || !Number.isFinite(maximum) || minimum >= maximum) {
    return null;
  }

  return [minimum, maximum];
}

function isAxisRangeDraftValid(minText: string, maxText: string): boolean {
  return resolveAxisRangeDraft(minText, maxText) !== null;
}

function readAxisRange(event: Readonly<Record<string, unknown>>, axisName: "xaxis" | "yaxis"): AxisRange | null {
  const rangeStart = event[`${axisName}.range[0]`];
  const rangeEnd = event[`${axisName}.range[1]`];

  if (typeof rangeStart === "number" && typeof rangeEnd === "number") {
    return [rangeStart, rangeEnd];
  }

  const compactRange = event[`${axisName}.range`];
  if (
    Array.isArray(compactRange) &&
    compactRange.length >= 2 &&
    typeof compactRange[0] === "number" &&
    typeof compactRange[1] === "number"
  ) {
    return [compactRange[0], compactRange[1]];
  }

  return null;
}

function formatAxisValue(value: number): string {
  if (Number.isInteger(value)) {
    return String(value);
  }
  return String(Number(value.toFixed(6)));
}

function handleRowKeyDown(
  event: KeyboardEvent<HTMLTableRowElement>,
  spectrumId: string,
  onToggleSelect: (spectrumId: string) => void
) {
  if (event.key !== "Enter" && event.key !== " ") {
    return;
  }

  event.preventDefault();
  onToggleSelect(spectrumId);
}
