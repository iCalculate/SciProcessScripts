import { ChevronLeft, ChevronRight, Download, Layers3, Sparkles, X } from "lucide-react";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  checkHealth,
  exportDatabaseSelection,
  generateCurves,
  getModelInfo,
  getRuntimeConfig
} from "./api";
import { AppHeader } from "./components/AppHeader";
import { CandidateTable } from "./components/CandidateTable";
import { ConditionPanel, NumberField } from "./components/ConditionPanel";
import {
  DatabaseWorkspace,
  DEFAULT_DATABASE_WORKSPACE_STATE,
  type DatabaseWorkspaceState
} from "./components/DatabaseWorkspace";
import { ImportWorkspace } from "./components/ImportWorkspace";
import { MatrixSynthesisPanel } from "./components/MatrixSynthesisPanel";
import { ModelsWorkspace } from "./components/ModelsWorkspace";
import { ResultsPanel } from "./components/ResultsPanel";
import { candidateExportHref, candidateFilename } from "./format";
import type {
  GeneratedCandidate,
  DatabaseSelectionState,
  DatabaseExportOptions,
  AppMode,
  GenerationCondition,
  GenerationResponse,
  ModelInfo,
  TabName
} from "./types";

const TABS_BY_MODE: Record<AppMode, TabName[]> = {
  full: ["Generate", "Import", "Database", "Matrix", "Analysis", "Models"],
  generate: ["Generate"],
  import: ["Import", "Database", "Matrix"],
  analysis: ["Import", "Database", "Matrix", "Analysis"],
  training: ["Database", "Matrix", "Models"]
};

const DEFAULT_DATABASE_EXPORT_OPTIONS: DatabaseExportOptions = {
  xyxy_curves: true,
  curve_metadata: true,
  raw_id_points: true,
  include_ig: true,
  raw_ig_points: true,
  aligned_ig_points: true,
  analysis_json: true
};

const CurvePlot = lazy(() =>
  import("./components/CurvePlot").then((module) => ({
    default: module.CurvePlot
  }))
);
const AnalysisWorkspace = lazy(() =>
  import("./components/AnalysisWorkspace").then((module) => ({
    default: module.AnalysisWorkspace
  }))
);

const initialCondition: GenerationCondition = {
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
  seed: 12345,
  voltage_min: -20,
  voltage_max: 20,
  points: 601,
  variants: 10
};

function validateCondition(condition: GenerationCondition): string | null {
  if (!condition.material.trim()) return "Material is required";
  if (!(condition.target_ion > condition.target_ioff && condition.target_ioff > 0)) {
    return "Ion must be greater than Ioff, and both must be positive";
  }
  if (condition.voltage_max <= condition.voltage_min) {
    return "Voltage maximum must be greater than voltage minimum";
  }
  if (
    condition.target_vth < condition.voltage_min ||
    condition.target_vth > condition.voltage_max
  ) {
    return "Vth must lie inside the voltage grid";
  }
  if (condition.points < 51 || condition.points > 2001) {
    return "Grid points must be between 51 and 2001";
  }
  if (condition.variants < 1 || condition.variants > 32) {
    return "Candidates must be between 1 and 32";
  }
  if (
    condition.noise_sigma_a < 0 ||
    condition.noise_floor_a < 0 ||
    condition.quantization_step_a < 0 ||
    condition.output_noise_gain < 0 ||
    condition.output_noise_gain > 50 ||
    condition.ss_region_width_v <= 0 ||
    condition.gate_leakage_a < 0 ||
    condition.gate_leakage_v_char <= 0 ||
    condition.gate_leakage_exponent <= 0 ||
    condition.gate_leakage_exponent > 3 ||
    condition.ion_sigma_fraction < 0 ||
    condition.ion_sigma_fraction > 1 ||
    condition.ioff_sigma_fraction < 0 ||
    condition.ioff_sigma_fraction > 1 ||
    condition.vth_sigma_v < 0 ||
    condition.ss_sigma_fraction < 0 ||
    condition.ss_sigma_fraction > 1 ||
    condition.hysteresis_sigma_v < 0 ||
    condition.mobility_cm2_vs <= 0 ||
    condition.mobility_sigma_fraction < 0 ||
    condition.mobility_sigma_fraction > 1 ||
    condition.contact_resistance_ohm < 0 ||
    condition.contact_resistance_sigma_fraction < 0 ||
    condition.contact_resistance_sigma_fraction > 1
  ) {
    return "Noise, quantization, and gate-leakage parameters must be valid";
  }
  return null;
}

export default function App() {
  const [appMode, setAppMode] = useState<AppMode>("full");
  const [activeTab, setActiveTab] = useState<TabName>("Generate");
  const [condition, setCondition] = useState(initialCondition);
  const [response, setResponse] = useState<GenerationResponse | null>(null);
  const [selectedId, setSelectedId] = useState(1);
  const [highlightAll, setHighlightAll] = useState(false);
  const [candidateTableExpanded, setCandidateTableExpanded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [apiOnline, setApiOnline] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [databaseExporting, setDatabaseExporting] = useState(false);
  const [showDatabaseExportSettings, setShowDatabaseExportSettings] = useState(false);
  const [databaseExportOptions, setDatabaseExportOptions] = useState<DatabaseExportOptions>(
    DEFAULT_DATABASE_EXPORT_OPTIONS
  );
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [databaseSelection, setDatabaseSelection] = useState<DatabaseSelectionState>({
    selectedIds: [],
    allFiltered: false,
    filters: {},
    total: 0
  });
  const [databaseWorkspaceState, setDatabaseWorkspaceState] = useState<DatabaseWorkspaceState>(
    DEFAULT_DATABASE_WORKSPACE_STATE
  );
  const requestIdRef = useRef(0);

  const runGeneration = useCallback(async (nextCondition: GenerationCondition) => {
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    setLoading(true);
    setError(null);
    try {
      const generated = await generateCurves(nextCondition);
      if (requestId !== requestIdRef.current) return;
      setResponse(generated);
      setCondition(generated.condition);
      setSelectedId(1);
      setApiOnline(true);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Generation failed");
      if (caught instanceof ApiError && caught.status === 0) {
        setApiOnline(false);
      }
    } finally {
      if (requestId === requestIdRef.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    void checkHealth().then(setApiOnline);
    void getRuntimeConfig()
      .then((config) => {
        setAppMode(config.app_mode);
        setActiveTab(TABS_BY_MODE[config.app_mode][0] ?? "Generate");
      })
      .catch(() => undefined);
    void getModelInfo().then(setModel).catch(() => undefined);
    void runGeneration(initialCondition);
    const healthTimer = window.setInterval(() => {
      void checkHealth().then(setApiOnline);
    }, 30_000);
    return () => window.clearInterval(healthTimer);
  }, [runGeneration]);

  const selected = useMemo<GeneratedCandidate | null>(() => {
    if (!response) return null;
    return (
      response.candidates.find((candidate) => candidate.candidate_id === selectedId) ??
      response.candidates[0] ??
      null
    );
  }, [response, selectedId]);
  const generatedCondition = response?.condition ?? condition;
  const availableTabs = TABS_BY_MODE[appMode];
  const hasPendingChanges =
    response !== null &&
    JSON.stringify(condition) !== JSON.stringify(response.condition);
  const conditionError = validateCondition(condition);
  const databaseSelectionCount = databaseSelection.allFiltered
    ? databaseSelection.total
    : databaseSelection.selectedIds.length;
  const canExportDatabaseSelection = databaseSelectionCount > 0;

  useEffect(() => {
    if (response === null || conditionError !== null) return undefined;
    if (JSON.stringify(condition) === JSON.stringify(response.condition)) return undefined;
    const timer = window.setTimeout(() => {
      void runGeneration(condition);
    }, 450);
    return () => window.clearTimeout(timer);
  }, [condition, conditionError, response, runGeneration]);

  function patchCondition(patch: Partial<GenerationCondition>) {
    setCondition((current) => {
      const next = { ...current, ...patch };
      if (
        patch.ai_residual_strength !== undefined &&
        patch.gate_ai_residual_strength === undefined
      ) {
        next.gate_ai_residual_strength = patch.ai_residual_strength;
      }
      if (
        patch.gate_ai_residual_strength !== undefined &&
        patch.ai_residual_strength === undefined
      ) {
        next.ai_residual_strength = patch.gate_ai_residual_strength;
      }
      return next;
    });
  }

  function moveCandidate(delta: number) {
    if (!response) return;
    const next = Math.min(
      response.candidates.length,
      Math.max(1, selectedId + delta)
    );
    setSelectedId(next);
  }

  function patchDatabaseExportOptions(patch: Partial<DatabaseExportOptions>) {
    setDatabaseExportOptions((current) => {
      const next = { ...current, ...patch };
      if (patch.include_ig === false) {
        next.raw_ig_points = false;
        next.aligned_ig_points = false;
      } else if (patch.include_ig === true) {
        next.raw_ig_points = true;
        next.aligned_ig_points = true;
      }
      return next;
    });
  }

  function requestDatabaseExport() {
    if (!canExportDatabaseSelection || databaseExporting) return;
    setShowDatabaseExportSettings(true);
  }

  async function exportActiveDatabaseSelection() {
    if (!canExportDatabaseSelection || databaseExporting) return;
    setDatabaseExporting(true);
    setError(null);
    try {
      await exportDatabaseSelection(databaseSelection, databaseExportOptions);
      setShowDatabaseExportSettings(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Database export failed");
    } finally {
      setDatabaseExporting(false);
    }
  }

  return (
    <div className="app">
      <AppHeader
        activeTab={activeTab}
        availableTabs={availableTabs}
        apiOnline={apiOnline}
        onTabChange={setActiveTab}
        exportHref={
          selected ? candidateExportHref(selected, generatedCondition) : null
        }
        exportFilename={selected ? candidateFilename(selected) : null}
        exportDisabled={
          activeTab === "Database"
            ? databaseExporting || !canExportDatabaseSelection
            : activeTab !== "Generate" || selected === null
        }
        exportLabel={activeTab === "Database" && databaseExporting ? "Exporting" : "Export"}
        onExportClick={
          activeTab === "Database"
            ? requestDatabaseExport
            : undefined
        }
      />
      {showDatabaseExportSettings ? (
        <div className="modal-backdrop" role="presentation">
          <section
            className="export-settings-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="database-export-title"
          >
            <header>
              <div>
                <span>Database selection</span>
                <h2 id="database-export-title">Export settings</h2>
              </div>
              <button
                type="button"
                className="icon-button"
                onClick={() => setShowDatabaseExportSettings(false)}
                aria-label="Close export settings"
              >
                <X size={17} />
              </button>
            </header>
            <div className="export-settings-summary">
              <strong>{databaseSelectionCount.toLocaleString()}</strong>
              <span>{databaseSelection.allFiltered ? "filtered curves" : "selected curves"}</span>
            </div>
            <div className="export-settings-list">
              <label>
                <input
                  type="checkbox"
                  checked={databaseExportOptions.xyxy_curves}
                  onChange={(event) => patchDatabaseExportOptions({ xyxy_curves: event.currentTarget.checked })}
                />
                <span>
                  <strong>XYXY curves</strong>
                  <small>Wide CSV with paired voltage/current columns.</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={databaseExportOptions.curve_metadata}
                  onChange={(event) => patchDatabaseExportOptions({ curve_metadata: event.currentTarget.checked })}
                />
                <span>
                  <strong>Curve metadata</strong>
                  <small>Summary metrics, source path, polarity, direction.</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={databaseExportOptions.raw_id_points}
                  onChange={(event) => patchDatabaseExportOptions({ raw_id_points: event.currentTarget.checked })}
                />
                <span>
                  <strong>Raw Id points</strong>
                  <small>Long-table drain current points.</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={databaseExportOptions.include_ig}
                  onChange={(event) => patchDatabaseExportOptions({ include_ig: event.currentTarget.checked })}
                />
                <span>
                  <strong>Include Ig</strong>
                  <small>Add gate-current data to XYXY and Ig files.</small>
                </span>
              </label>
              <label className={!databaseExportOptions.include_ig ? "disabled" : undefined}>
                <input
                  type="checkbox"
                  checked={databaseExportOptions.include_ig && databaseExportOptions.raw_ig_points}
                  disabled={!databaseExportOptions.include_ig}
                  onChange={(event) => patchDatabaseExportOptions({ raw_ig_points: event.currentTarget.checked })}
                />
                <span>
                  <strong>Raw Ig points</strong>
                  <small>Long-table gate current points.</small>
                </span>
              </label>
              <label className={!databaseExportOptions.include_ig ? "disabled" : undefined}>
                <input
                  type="checkbox"
                  checked={databaseExportOptions.include_ig && databaseExportOptions.aligned_ig_points}
                  disabled={!databaseExportOptions.include_ig}
                  onChange={(event) => patchDatabaseExportOptions({ aligned_ig_points: event.currentTarget.checked })}
                />
                <span>
                  <strong>Aligned Ig points</strong>
                  <small>Normalized gate-current points.</small>
                </span>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={databaseExportOptions.analysis_json}
                  onChange={(event) => patchDatabaseExportOptions({ analysis_json: event.currentTarget.checked })}
                />
                <span>
                  <strong>Analysis JSON</strong>
                  <small>Selection statistics and distributions.</small>
                </span>
              </label>
            </div>
            <footer>
              <button
                type="button"
                className="button secondary compact"
                onClick={() => setDatabaseExportOptions(DEFAULT_DATABASE_EXPORT_OPTIONS)}
                disabled={databaseExporting}
              >
                Reset
              </button>
              <button
                type="button"
                className="button primary compact"
                onClick={() => void exportActiveDatabaseSelection()}
                disabled={databaseExporting}
              >
                <Download size={15} />
                {databaseExporting ? "Exporting" : "Export"}
              </button>
            </footer>
          </section>
        </div>
      ) : null}

      {activeTab === "Generate" ? (
        <>
          {error ? <div className="error-banner app-error">{error}</div> : null}
          <div className="workbench">
            <ConditionPanel
              condition={condition}
              disabled={false}
              onChange={patchCondition}
            />
            <main className={`canvas-panel${candidateTableExpanded ? "" : " table-collapsed"}`}>
              <div className="canvas-toolbar">
                <div className="toolbar-primary">
                <div className="candidate-stepper">
                  <span>Candidate</span>
                  <button onClick={() => moveCandidate(-1)} aria-label="Previous candidate">
                    <ChevronLeft size={16} />
                  </button>
                  <strong>
                    {selectedId} / {response?.candidates.length ?? condition.variants}
                  </strong>
                  <button onClick={() => moveCandidate(1)} aria-label="Next candidate">
                    <ChevronRight size={16} />
                  </button>
                </div>
                <div className="sweep-controls">
                  <span className="sweep-label">Sweep</span>
                  <NumberField compact label="V min" value={condition.voltage_min} unit="V" step={1} onCommit={(voltage_min) => patchCondition({ voltage_min })} />
                  <NumberField compact label="V max" value={condition.voltage_max} unit="V" step={1} onCommit={(voltage_max) => patchCondition({ voltage_max })} />
                  <NumberField compact label="Points" value={condition.points} unit="pts" step={50} min={51} max={2001} integer onCommit={(points) => patchCondition({ points })} />
                  <NumberField compact label="Curves" value={condition.variants} unit="" step={1} min={1} max={32} integer onCommit={(variants) => patchCondition({ variants })} />
                </div>
                <button
                  className={`button compact highlight-toggle${highlightAll ? " active" : ""}`}
                  aria-pressed={highlightAll}
                  onClick={() => setHighlightAll((current) => !current)}
                >
                  <Layers3 size={15} />
                  Highlight all
                </button>
                <button
                  className="button primary compact"
                  disabled={loading || conditionError !== null}
                  onClick={() => void runGeneration(condition)}
                >
                  <Sparkles size={15} />
                  {loading ? "Generating..." : "Generate variants"}
                </button>
                {hasPendingChanges ? (
                  <span className="pending-state">Conditions changed</span>
                ) : null}
                {conditionError ? (
                  <span className="invalid-state">{conditionError}</span>
                ) : null}
                <div className="curve-title">
                  Transfer curve <span>(Vd = {generatedCondition.vd.toFixed(2)} V)</span>
                </div>
                </div>
              </div>
              {response && selected ? (
                <>
                  <Suspense
                    fallback={
                      <div className="plot-loading">
                        <span className="loading-ring" />
                        Loading curve canvas
                      </div>
                    }
                  >
                    <CurvePlot
                      response={response}
                      selected={selected}
                      condition={generatedCondition}
                      highlightAll={highlightAll}
                      onConstraintChange={patchCondition}
                    />
                  </Suspense>
                  <CandidateTable
                    candidates={response.candidates}
                    selectedId={selectedId}
                    expanded={candidateTableExpanded}
                    onSelect={setSelectedId}
                    onToggle={() => setCandidateTableExpanded((current) => !current)}
                  />
                </>
              ) : (
                <div className="plot-loading">
                  <span className="loading-ring" />
                  Waiting for the generation API
                </div>
              )}
            </main>
            {response && selected ? (
              <ResultsPanel
                candidate={selected}
                response={response}
                loading={loading}
                regenerateDisabled={conditionError !== null}
                onRegenerate={() =>
                  conditionError === null
                    ? void runGeneration({ ...condition, seed: condition.seed + 1 })
                    : undefined
                }
              />
            ) : (
              <aside className="side-panel results-panel loading-panel" />
            )}
          </div>
          <footer className="status-bar">
            <span className="status-bar-label"><strong>Parameter</strong></span>
          </footer>
        </>
      ) : activeTab === "Import" ? (
        <ImportWorkspace />
      ) : activeTab === "Database" ? (
        <>
          {error ? <div className="error-banner app-error">{error}</div> : null}
        <DatabaseWorkspace
          selection={databaseSelection}
          onSelectionChange={setDatabaseSelection}
          state={databaseWorkspaceState}
          onStateChange={setDatabaseWorkspaceState}
        />
        </>
      ) : activeTab === "Matrix" ? (
        <main className="matrix-workspace">
          <MatrixSynthesisPanel />
        </main>
      ) : activeTab === "Analysis" ? (
        <Suspense fallback={<div className="analysis-loading">Loading analysis workspace</div>}>
          <AnalysisWorkspace selection={databaseSelection} />
        </Suspense>
      ) : (
        <ModelsWorkspace model={model} onModelChange={setModel} />
      )}
    </div>
  );
}
