import { Database, FolderOpen, FolderSync, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { getDatabaseStatus, importDatabaseFolder, importDatabaseSource } from "../api";
import type {
  DatabaseImportRequest,
  DatabaseImportSummary,
  DatabaseStatus
} from "../types";

const ALL_SUFFIXES = [".csv", ".txt", ".tsv", ".dat", ".ztr", ".xtr", ".xml"];
const DEFAULT_SOURCE_PATH = "D:\\B1500";

const INITIAL_REQUEST: DatabaseImportRequest = {
  source_path: DEFAULT_SOURCE_PATH,
  suffixes: [...ALL_SUFFIXES],
  max_xml_mb: 128,
  hash_files: false,
  replace: false
};

const SERVER_PROGRESS_STEPS = [
  "Resolving source",
  "Scanning candidate files",
  "Cleaning raw tables",
  "Aligning transfer curves",
  "Writing database rows",
  "Refreshing database snapshot"
] as const;

type ImportTone = "idle" | "running" | "success";
type OverviewMode = "assets" | "polarity" | "source" | "belonger" | "gate";
type HiddenOverviewState = Record<OverviewMode, string[]>;

type ImportProgressState = {
  active: boolean;
  percent: number;
  title: string;
  detail: string;
  tone: ImportTone;
  steps: readonly string[];
  stepIndex: number;
};

type StatusDelta = {
  source_files: number;
  curves: number;
  curves_with_ig: number;
  rejected_entries: number;
  raw_points: number;
  aligned_points: number;
  gate_points: number;
};

type OverviewItem = {
  key: string;
  label: string;
  value: number;
  share: number;
  tone: "blue" | "green" | "amber" | "pink";
};

const PIE_TONE_COLORS: Record<OverviewItem["tone"], string> = {
  blue: "#2d7cff",
  green: "#08a06f",
  amber: "#d98a10",
  pink: "#ef476f"
};

const IDLE_PROGRESS: ImportProgressState = {
  active: false,
  percent: 0,
  title: "Ready to import",
  detail: "Choose a folder or keep the default source path, then start an incremental import.",
  tone: "idle",
  steps: SERVER_PROGRESS_STEPS,
  stepIndex: 0
};

const ZERO_DELTA: StatusDelta = {
  source_files: 0,
  curves: 0,
  curves_with_ig: 0,
  rejected_entries: 0,
  raw_points: 0,
  aligned_points: 0,
  gate_points: 0
};

function formatCount(value: number | string) {
  return typeof value === "number" ? value.toLocaleString() : value;
}

function formatDelta(value: number) {
  return value > 0 ? `+${value.toLocaleString()}` : null;
}

function ratio(value: number, total: number) {
  if (total <= 0) return 0;
  return value / total;
}

function toneForIndex(index: number): OverviewItem["tone"] {
  return (["blue", "green", "amber", "pink"] as const)[index % 4];
}

function StatusCard({
  label,
  value,
  delta
}: {
  label: string;
  value: number | string;
  delta?: number;
}) {
  const deltaLabel = delta !== undefined ? formatDelta(delta) : null;
  return (
    <div className="import-stat-card">
      <span>{label}</span>
      <strong>{formatCount(value)}</strong>
      {deltaLabel ? <em>{deltaLabel}</em> : <i>No change</i>}
    </div>
  );
}

function buildImportConclusion(summary: DatabaseImportSummary): string {
  const actions: string[] = [];
  if (summary.files_imported > 0) actions.push(`${summary.files_imported} new files`);
  if (summary.files_updated > 0) actions.push(`${summary.files_updated} updated files`);
  if (summary.files_skipped > 0) actions.push(`${summary.files_skipped} unchanged files skipped`);
  if (actions.length === 0) actions.push("no new rows needed to be written");
  return `Scanned ${summary.files_discovered.toLocaleString()} files, ${actions.join(", ")}. Accepted ${summary.accepted_transfer_segments.toLocaleString()} transfer segments and rejected ${summary.rejected_entries.toLocaleString()} invalid entries.`;
}

function buildDelta(previous: DatabaseStatus | null, next: DatabaseStatus): StatusDelta {
  return {
    source_files: next.source_files - (previous?.source_files ?? 0),
    curves: next.curves - (previous?.curves ?? 0),
    curves_with_ig: next.curves_with_ig - (previous?.curves_with_ig ?? 0),
    rejected_entries: next.rejected_entries - (previous?.rejected_entries ?? 0),
    raw_points: next.raw_points - (previous?.raw_points ?? 0),
    aligned_points: next.aligned_points - (previous?.aligned_points ?? 0),
    gate_points: next.gate_points - (previous?.gate_points ?? 0)
  };
}

function overviewItems(status: DatabaseStatus | null, mode: OverviewMode): OverviewItem[] {
  if (!status) return [];
  if (mode === "assets") {
    const total =
      status.source_files +
      status.curves +
      status.curves_with_ig +
      status.rejected_entries;
    return [
      { key: "sources", label: "Source files", value: status.source_files, share: ratio(status.source_files, total), tone: "blue" },
      { key: "curves", label: "Valid curves", value: status.curves, share: ratio(status.curves, total), tone: "green" },
      { key: "ig", label: "Curves with Ig", value: status.curves_with_ig, share: ratio(status.curves_with_ig, total), tone: "pink" },
      { key: "rejected", label: "Rejected entries", value: status.rejected_entries, share: ratio(status.rejected_entries, total), tone: "amber" }
    ];
  }
  if (mode === "gate") {
    const noGate = Math.max(status.curves - status.curves_with_ig, 0);
    return [
      { key: "with-ig", label: "With Ig", value: status.curves_with_ig, share: ratio(status.curves_with_ig, status.curves), tone: "green" },
      { key: "without-ig", label: "Without Ig", value: noGate, share: ratio(noGate, status.curves), tone: "blue" }
    ];
  }
  const countMap =
    mode === "polarity"
      ? status.polarity_counts
      : mode === "belonger"
        ? status.belonger_counts
        : status.source_kind_counts;
  const entries = Object.entries(countMap).sort((left, right) => right[1] - left[1]);
  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  return entries.map(([key, value], index) => ({
    key,
    label: key,
    value,
    share: ratio(value, total),
    tone: toneForIndex(index)
  }));
}

function polarToCartesian(cx: number, cy: number, radius: number, angleDeg: number) {
  const angleRad = ((angleDeg - 90) * Math.PI) / 180;
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad)
  };
}

function describeWedgePath(
  cx: number,
  cy: number,
  outerRadius: number,
  innerRadius: number,
  startAngle: number,
  endAngle: number
) {
  const outerStart = polarToCartesian(cx, cy, outerRadius, startAngle);
  const outerEnd = polarToCartesian(cx, cy, outerRadius, endAngle);
  const innerEnd = polarToCartesian(cx, cy, innerRadius, endAngle);
  const innerStart = polarToCartesian(cx, cy, innerRadius, startAngle);
  const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0;
  return [
    `M ${outerStart.x} ${outerStart.y}`,
    `A ${outerRadius} ${outerRadius} 0 ${largeArcFlag} 1 ${outerEnd.x} ${outerEnd.y}`,
    `L ${innerEnd.x} ${innerEnd.y}`,
    `A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${innerStart.x} ${innerStart.y}`,
    "Z"
  ].join(" ");
}

export function ImportWorkspace() {
  const [request, setRequest] = useState<DatabaseImportRequest>(INITIAL_REQUEST);
  const [selectedFolderFiles, setSelectedFolderFiles] = useState<File[]>([]);
  const [selectedFolderLabel, setSelectedFolderLabel] = useState<string>("");
  const [summary, setSummary] = useState<DatabaseImportSummary | null>(null);
  const [status, setStatus] = useState<DatabaseStatus | null>(null);
  const [delta, setDelta] = useState<StatusDelta>(ZERO_DELTA);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [runningImport, setRunningImport] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<ImportProgressState>(IDLE_PROGRESS);
  const [overviewMode, setOverviewMode] = useState<OverviewMode>("assets");
  const [hiddenOverview, setHiddenOverview] = useState<HiddenOverviewState>({
    assets: [],
    polarity: [],
    source: [],
    belonger: [],
    gate: []
  });
  const [activeOverviewKey, setActiveOverviewKey] = useState<string | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const progressTimerRef = useRef<number | null>(null);

  function stopProgressTimer() {
    if (progressTimerRef.current !== null) {
      window.clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  }

  function startServerProgress(basePercent: number, detail: string) {
    stopProgressTimer();
    setProgress({
      active: true,
      percent: basePercent,
      title: SERVER_PROGRESS_STEPS[0],
      detail,
      tone: "running",
      steps: SERVER_PROGRESS_STEPS,
      stepIndex: 0
    });
    progressTimerRef.current = window.setInterval(() => {
      setProgress((current) => {
        if (!current.active) return current;
        const nextPercent = Math.min(current.percent + (current.percent < 64 ? 5 : current.percent < 84 ? 2 : 1), 94);
        const nextStep = Math.min(
          Math.floor((nextPercent / 95) * SERVER_PROGRESS_STEPS.length),
          SERVER_PROGRESS_STEPS.length - 1
        );
        return {
          ...current,
          percent: nextPercent,
          title: SERVER_PROGRESS_STEPS[nextStep],
          detail,
          stepIndex: nextStep
        };
      });
    }, 800);
  }

  async function refreshStatus() {
    setLoadingStatus(true);
    try {
      const nextStatus = await getDatabaseStatus();
      setStatus(nextStatus);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not read database status");
    } finally {
      setLoadingStatus(false);
    }
  }

  useEffect(() => {
    void refreshStatus();
  }, []);

  useEffect(() => () => stopProgressTimer(), []);

  const suffixLabel = useMemo(() => {
    if (request.suffixes.length === ALL_SUFFIXES.length) return "All supported suffixes";
    if (request.suffixes.length === 0) return "No suffix selected";
    return request.suffixes.join(", ");
  }, [request.suffixes]);

  const activeOverview = useMemo(
    () => overviewItems(status, overviewMode),
    [overviewMode, status]
  );
  const visibleOverview = useMemo(
    () => activeOverview.filter((item) => !hiddenOverview[overviewMode].includes(item.key)),
    [activeOverview, hiddenOverview, overviewMode]
  );
  const pieSegments = useMemo(() => {
    let start = 0;
    return visibleOverview.map((item) => {
      const span = item.share * 360;
      const end = start + span;
      const segment = {
        ...item,
        startAngle: start,
        endAngle: end,
        path: describeWedgePath(110, 110, 102, 54, start, end)
      };
      start = end;
      return segment;
    });
  }, [visibleOverview]);

  function toggleSuffix(suffix: string) {
    setRequest((current) => {
      const selected = current.suffixes.includes(suffix)
        ? current.suffixes.filter((item) => item !== suffix)
        : [...current.suffixes, suffix];
      return { ...current, suffixes: selected };
    });
  }

  function toggleOverviewItem(itemKey: string) {
    setHiddenOverview((current) => {
      const hidden = current[overviewMode];
      const nextHidden = hidden.includes(itemKey)
        ? hidden.filter((key) => key !== itemKey)
        : [...hidden, itemKey];
      return { ...current, [overviewMode]: nextHidden };
    });
    setActiveOverviewKey((current) => (current === itemKey ? null : itemKey));
  }

  async function finalizeImport(result: DatabaseImportSummary, previousStatus: DatabaseStatus | null) {
    stopProgressTimer();
    setProgress({
      active: true,
      percent: 97,
      title: "Refreshing database snapshot",
      detail: "Import finished. Updating totals and category shares.",
      tone: "running",
      steps: SERVER_PROGRESS_STEPS,
      stepIndex: SERVER_PROGRESS_STEPS.length - 1
    });
    const nextStatus = await getDatabaseStatus();
    setStatus(nextStatus);
    setDelta(buildDelta(previousStatus, nextStatus));
    setSummary(result);
    setProgress({
      active: true,
      percent: 100,
      title: "Import completed",
      detail: buildImportConclusion(result),
      tone: "success",
      steps: SERVER_PROGRESS_STEPS,
      stepIndex: SERVER_PROGRESS_STEPS.length - 1
    });
  }

  async function runImport() {
    if (!request.source_path.trim()) {
      setError("Source path is required.");
      return;
    }
    if (request.suffixes.length === 0) {
      setError("Select at least one source suffix.");
      return;
    }
    const baseline = status;
    setRunningImport(true);
    setError(null);
    setSummary(null);
    setDelta(ZERO_DELTA);
    if (selectedFolderFiles.length > 0) {
      setProgress({
        active: true,
        percent: 4,
        title: "Uploading selected folder",
        detail: `Submitting ${selectedFolderFiles.length.toLocaleString()} files from ${selectedFolderLabel || "the selected folder"}.`,
        tone: "running",
        steps: ["Preparing upload", "Uploading folder", ...SERVER_PROGRESS_STEPS],
        stepIndex: 1
      });
      try {
        const result = await importDatabaseFolder(
          selectedFolderFiles,
          {
            suffixes: request.suffixes,
            max_xml_mb: request.max_xml_mb,
            hash_files: request.hash_files,
            replace: request.replace
          },
          (uploadProgress) => {
            if (uploadProgress < 1) {
              stopProgressTimer();
              setProgress({
                active: true,
                percent: Math.min(6 + Math.round(uploadProgress * 28), 34),
                title: "Uploading selected folder",
                detail: `Upload ${Math.round(uploadProgress * 100)}% complete. Server-side cleaning and alignment will begin next.`,
                tone: "running",
                steps: ["Preparing upload", "Uploading folder", ...SERVER_PROGRESS_STEPS],
                stepIndex: 1
              });
              return;
            }
            startServerProgress(36, "Folder uploaded. Cleaning, alignment, and incremental database updates are now running.");
          }
        );
        await finalizeImport(result, baseline);
      } catch (caught) {
        stopProgressTimer();
        setProgress(IDLE_PROGRESS);
        setError(caught instanceof Error ? caught.message : "Folder import failed");
      } finally {
        setRunningImport(false);
      }
      return;
    }

    startServerProgress(10, `Scanning ${request.source_path.trim()} and applying incremental cleaning, alignment, and import.`);
    try {
      const result = await importDatabaseSource({
        ...request,
        source_path: request.source_path.trim()
      });
      await finalizeImport(result, baseline);
    } catch (caught) {
      stopProgressTimer();
      setProgress(IDLE_PROGRESS);
      setError(caught instanceof Error ? caught.message : "Database import failed");
    } finally {
      setRunningImport(false);
    }
  }

  return (
    <main className="secondary-workspace import-workspace">
      <section className="workspace-heading">
        <div>
          <h1>Incremental Import</h1>
          <p>
            Use the default source path or choose a local folder. The import remains incremental by default and only rewrites data that actually changed.
          </p>
        </div>
        <div className="import-header-actions">
          <label className="button secondary upload-button">
            <FolderOpen size={16} />
            Choose Folder
            <input
              ref={folderInputRef}
              type="file"
              multiple
              webkitdirectory=""
              directory=""
              onChange={(event) => {
                const nextFiles = Array.from(event.currentTarget.files ?? []);
                setSelectedFolderFiles(nextFiles);
                if (nextFiles.length === 0) {
                  setSelectedFolderLabel("");
                  return;
                }
                const first = (nextFiles[0] as File & { webkitRelativePath?: string }).webkitRelativePath;
                setSelectedFolderLabel(first?.split("/")[0] ?? nextFiles[0].name);
              }}
            />
          </label>
          <button
            className="button primary"
            disabled={runningImport}
            onClick={() => void runImport()}
          >
            <FolderSync size={16} />
            {runningImport ? "Importing..." : "Import"}
          </button>
        </div>
      </section>

      {error ? <div className="error-banner" role="alert">{error}</div> : null}

      <section className={`import-progress-card ${progress.tone}`}>
        <div className="import-progress-header">
          <div className="import-progress-copy">
            <div className="math-loader" aria-hidden="true">
              <span />
              <span />
              <span />
            </div>
            <div>
              <strong>{progress.title}</strong>
              <p>{progress.detail}</p>
            </div>
          </div>
          <span>{progress.active ? `${progress.percent}%` : "Idle"}</span>
        </div>
        <div className="import-progress-track" aria-hidden="true">
          <div
            className="import-progress-fill"
            style={{ width: `${progress.active ? progress.percent : 0}%` }}
          />
        </div>
        <div className="import-progress-steps">
          {progress.steps.map((step, index) => (
            <div
              key={step}
              className={`import-progress-step${index <= progress.stepIndex && progress.active ? " active" : ""}`}
            >
              <span />
              <small>{step}</small>
            </div>
          ))}
        </div>
      </section>

      <section className="import-status-strip">
        <StatusCard label="Source files" value={status?.source_files ?? "-"} delta={delta.source_files} />
        <StatusCard label="Valid curves" value={status?.curves ?? "-"} delta={delta.curves} />
        <StatusCard label="Curves with Ig" value={status?.curves_with_ig ?? "-"} delta={delta.curves_with_ig} />
        <StatusCard label="Rejected entries" value={status?.rejected_entries ?? "-"} delta={delta.rejected_entries} />
      </section>

      <div className="import-grid single-aside">
        <section className="import-panel">
          <h2>Import settings</h2>
          <label className="import-field">
            <span>Source root</span>
            <input
              type="text"
              placeholder={DEFAULT_SOURCE_PATH}
              value={request.source_path}
              onChange={(event) =>
                setRequest((current) => ({ ...current, source_path: event.currentTarget.value }))
              }
            />
          </label>

          {selectedFolderFiles.length > 0 ? (
            <div className="import-summary-block">
              <strong>Selected folder</strong>
              <code>{selectedFolderLabel || "Local folder"} - {selectedFolderFiles.length.toLocaleString()} files ready</code>
            </div>
          ) : null}

          <label className="import-field">
            <span>Max XML/XTR size (MB)</span>
            <input
              type="number"
              min={1}
              max={4096}
              value={request.max_xml_mb}
              onChange={(event) =>
                setRequest((current) => ({
                  ...current,
                  max_xml_mb: Number(event.currentTarget.value) || 128
                }))
              }
            />
          </label>

          <div className="suffix-panel">
            <div className="suffix-panel-heading">
              <strong>Included suffixes</strong>
              <span>{suffixLabel}</span>
            </div>
            <div className="suffix-grid">
              {ALL_SUFFIXES.map((suffix) => (
                <label key={suffix} className="suffix-chip">
                  <input
                    type="checkbox"
                    checked={request.suffixes.includes(suffix)}
                    onChange={() => toggleSuffix(suffix)}
                  />
                  <span>{suffix}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="import-option-list">
            <label className="import-check">
              <input
                type="checkbox"
                checked={request.hash_files}
                onChange={(event) =>
                  setRequest((current) => ({ ...current, hash_files: event.currentTarget.checked }))
                }
              />
              <span>Use SHA1 for stricter change detection</span>
            </label>
            <label className="import-check warning">
              <input
                type="checkbox"
                checked={request.replace}
                onChange={(event) =>
                  setRequest((current) => ({ ...current, replace: event.currentTarget.checked }))
                }
              />
              <span>Replace the full database instead of updating incrementally</span>
            </label>
          </div>
        </section>

        <section className="import-results database-overview-panel">
          <div className="database-overview-header">
            <div>
              <h2>Database snapshot</h2>
              <p>Inspect current totals, fresh deltas, and category shares after each import.</p>
            </div>
            <button
              className="button compact secondary"
              disabled={loadingStatus || runningImport}
              onClick={() => void refreshStatus()}
            >
              <RefreshCw size={15} className={loadingStatus ? "spin" : ""} />
              Refresh
            </button>
          </div>

          <div className="database-overview-meta">
            <div>
              <span>Raw points</span>
              <strong>{status?.raw_points.toLocaleString() ?? "0"}</strong>
              {formatDelta(delta.raw_points) ? <em>{formatDelta(delta.raw_points)}</em> : null}
            </div>
            <div>
              <span>Aligned points</span>
              <strong>{status?.aligned_points.toLocaleString() ?? "0"}</strong>
              {formatDelta(delta.aligned_points) ? <em>{formatDelta(delta.aligned_points)}</em> : null}
            </div>
            <div>
              <span>Gate-current points</span>
              <strong>{status?.gate_points.toLocaleString() ?? "0"}</strong>
              {formatDelta(delta.gate_points) ? <em>{formatDelta(delta.gate_points)}</em> : null}
            </div>
          </div>

          <div className="database-overview-toggle">
            {([
              ["assets", "Asset mix"],
              ["polarity", "Polarity"],
              ["source", "Source kind"],
              ["belonger", "Belonger"],
              ["gate", "Gate-current"]
            ] as const).map(([mode, label]) => (
              <button
                key={mode}
                className={overviewMode === mode ? "active" : ""}
                onClick={() => {
                  setOverviewMode(mode);
                  setActiveOverviewKey(null);
                }}
                type="button"
              >
                {label}
              </button>
            ))}
          </div>

          <div className="database-overview-list">
            <div className="database-overview-chart">
              <div className="database-overview-pie-shell">
                <svg
                  className="database-overview-pie"
                  viewBox="0 0 220 220"
                  role="img"
                  aria-label={`${overviewMode} pie chart`}
                >
                  {pieSegments.length === 0 ? (
                    <circle cx="110" cy="110" r="102" fill="#dbe6f2" />
                  ) : pieSegments.map((item) => (
                    <path
                      key={item.key}
                      d={item.path}
                      fill={PIE_TONE_COLORS[item.tone]}
                      className={activeOverviewKey === item.key ? "active" : ""}
                      onClick={() => setActiveOverviewKey(item.key)}
                    />
                  ))}
                </svg>
                <div className="database-overview-pie-core">
                  <span>{visibleOverview.length > 0 ? "Total" : "Empty"}</span>
                  <strong>
                    {status
                      ? (overviewMode === "gate"
                        ? status.curves.toLocaleString()
                        : visibleOverview.reduce((sum, item) => sum + item.value, 0).toLocaleString())
                      : "0"}
                  </strong>
                </div>
              </div>
            </div>
            <div className="database-overview-legend">
              {activeOverview.map((item) => {
                const hidden = hiddenOverview[overviewMode].includes(item.key);
                const active = activeOverviewKey === item.key;
                return (
                  <button
                    key={item.key}
                    type="button"
                    className={`database-overview-legend-item${hidden ? " hidden" : ""}${active ? " active" : ""}`}
                    onClick={() => toggleOverviewItem(item.key)}
                  >
                    <b className={`database-overview-dot tone-${item.tone}`} />
                    <span>{item.label}</span>
                    <small>{item.value.toLocaleString()} - {(item.share * 100).toFixed(1)}%</small>
                  </button>
                );
              })}
            </div>
            {visibleOverview.map((item) => (
              <div
                key={item.key}
                className={`database-overview-item${activeOverviewKey === item.key ? " active" : ""}`}
              >
                <div className="database-overview-item-head">
                  <strong>
                    <b className={`database-overview-dot tone-${item.tone}`} />
                    {item.label}
                  </strong>
                  <span>{item.value.toLocaleString()} - {(item.share * 100).toFixed(1)}%</span>
                </div>
                <div className="database-overview-bar">
                  <div
                    className={`database-overview-bar-fill tone-${item.tone}`}
                    style={{ width: `${Math.max(item.share * 100, 4)}%` }}
                  />
                </div>
              </div>
            ))}
            {activeOverview.length === 0 ? (
              <div className="database-overview-empty">
                <Database size={22} />
                <span>No database snapshot available yet.</span>
              </div>
            ) : null}
          </div>

          {summary ? (
            <div className="import-summary-block emphasis">
              <strong>Latest import</strong>
              <p>{buildImportConclusion(summary)}</p>
            </div>
          ) : null}
        </section>
      </div>
    </main>
  );
}
