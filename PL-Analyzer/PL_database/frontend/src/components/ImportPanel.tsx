import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import type { ImportJob, ImportOptions, ImportProbeReport, UploadTransferProgress } from "../types";
import type { UploadedImportBatchResult } from "../api";

type ImportedUploadHistoryItem = {
  relative_path: string;
  imported_spectra: number;
  imported_media_assets: number;
  ended_at: string;
  status: string;
};

interface ImportPanelProps {
  jobs: ImportJob[];
  onFetchUploadHistory: (rootName?: string) => Promise<ImportedUploadHistoryItem[]>;
  onProbeFile: (file: File) => Promise<ImportProbeReport>;
  onStart: (
    inputPath: string,
    recursive: boolean,
    forceReimport: boolean,
    options: ImportOptions
  ) => Promise<void>;
  onStartUpload: (
    rootName: string,
    files: File[],
    sourceKind: "file_upload" | "folder_upload",
    forceReimport: boolean,
    options: ImportOptions,
    onProgress?: (progress: UploadTransferProgress) => void,
    onJobQueued?: (job: ImportJob) => void | Promise<void>,
    shouldStop?: () => boolean,
    registerAbort?: (abortCurrentUpload: (() => void) | null) => void
  ) => Promise<UploadedImportBatchResult>;
  onStopJobs: (jobIds: string[]) => Promise<void>;
}

const DEFAULT_OPTIONS: ImportOptions = {
  include_point_spectra: true,
  include_line_scans: false,
  include_area_maps: false,
  include_series_scans: false,
  include_photo_images: false
};

export function ImportPanel(props: ImportPanelProps) {
  const [inputPath, setInputPath] = useState("");
  const [recursive, setRecursive] = useState(true);
  const [forceReimport, setForceReimport] = useState(false);
  const [options, setOptions] = useState<ImportOptions>(DEFAULT_OPTIONS);
  const [starting, setStarting] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadTransferProgress | null>(null);
  const [uploadSourceLabel, setUploadSourceLabel] = useState("");
  const [batchJobIds, setBatchJobIds] = useState<string[]>([]);
  const [batchSourceLabel, setBatchSourceLabel] = useState("");
  const [batchExpectedFiles, setBatchExpectedFiles] = useState(0);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedFolderName, setSelectedFolderName] = useState("");
  const [selectedFolderFiles, setSelectedFolderFiles] = useState<File[]>([]);
  const [pendingFolderFiles, setPendingFolderFiles] = useState<File[]>([]);
  const [skippedFolderHistory, setSkippedFolderHistory] = useState<ImportedUploadHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [probeResult, setProbeResult] = useState<ImportProbeReport | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const uploadAbortRef = useRef<(() => void) | null>(null);
  const stopRequestedRef = useRef(false);
  const [stopping, setStopping] = useState(false);
  const uploadProgressHistoryRef = useRef<Array<{ at: number; loaded: number }>>([]);
  const [uploadSpeedHistory, setUploadSpeedHistory] = useState<number[]>([]);

  const selectionMode = selectedFile ? "file" : selectedFolderFiles.length > 0 ? "folder" : inputPath.trim() ? "path" : "none";
  const selectionLabel = useMemo(() => {
    if (selectedFile) {
      return `Selected file: ${selectedFile.name}`;
    }
    if (selectedFolderFiles.length > 0) {
      return `Selected folder: ${selectedFolderName || "selected-folder"} (${selectedFolderFiles.length} .wip files)`;
    }
    if (inputPath.trim()) {
      return `Manual path: ${inputPath.trim()}`;
    }
    return "No source selected yet.";
  }, [inputPath, selectedFile, selectedFolderFiles, selectedFolderName]);
  const pendingFolderCount = forceReimport ? selectedFolderFiles.length : pendingFolderFiles.length;
  const skippedFolderCount = forceReimport ? 0 : skippedFolderHistory.length;
  const canStartSelection = selectionMode !== "none" && !starting && !historyLoading && (selectionMode !== "folder" || pendingFolderCount > 0);

  const batchJobs = useMemo(
    () =>
      batchJobIds
        .map((jobId) => props.jobs.find((job) => job.job_id === jobId) ?? null)
        .filter((job): job is ImportJob => job !== null),
    [batchJobIds, props.jobs]
  );
  const batchIsActive = batchExpectedFiles > 0;
  const batchCurrentJob = batchJobs.find((job) => job.status === "running") ?? batchJobs.find((job) => job.status === "pending") ?? null;
  const activeJob = batchIsActive
    ? batchCurrentJob ?? batchJobs[0] ?? null
    : props.jobs.find((job) => job.status === "running" || job.status === "pending") ?? (uploadProgress ? null : (props.jobs[0] ?? null));
  const latestCompletedJob =
    props.jobs.find((job) => job.status !== "running" && job.status !== "pending") ?? props.jobs[0] ?? null;

  const batchCompletedJobs = batchJobs.filter((job) => isTerminalJob(job.status)).length;
  const batchRunningJobs = batchJobs.filter((job) => job.status === "running").length;
  const totalFiles = activeJob?.total_files ?? 0;
  const processedFiles = activeJob?.processed_files ?? 0;
  const progress = batchIsActive
    ? Math.min(
        100,
        Math.round(
          (batchJobs.reduce((sum, job) => sum + approximateJobCompletion(job), 0) / Math.max(1, batchExpectedFiles)) * 100
        )
      )
    : totalFiles > 0
      ? Math.min(100, Math.round((processedFiles / totalFiles) * 100))
      : 0;
  const uploadPercent = uploadProgress?.percent ?? 0;
  const inventory = batchIsActive
    ? aggregateJobCounts(batchJobs, "detected_inventory", "last_file_inventory")
    : activeJob?.details?.detected_inventory ?? activeJob?.details?.last_file_inventory ?? {};
  const datasetCounts = batchIsActive
    ? aggregateJobCounts(batchJobs, "dataset_mode_counts", "last_file_dataset_counts")
    : activeJob?.details?.dataset_mode_counts ?? activeJob?.details?.last_file_dataset_counts ?? {};
  const mediaInventory = batchIsActive
    ? aggregateJobCounts(batchJobs, "media_inventory", "last_file_media_inventory")
    : activeJob?.details?.media_inventory ?? activeJob?.details?.last_file_media_inventory ?? {};
  const activePhaseLabel = batchIsActive
    ? uploadProgress
      ? uploadProgress.stage === "queued"
        ? "Upload complete"
        : "Streaming files to backend"
      : batchCurrentJob
        ? batchCurrentJob.details?.phase_label ?? formatPhaseLabel(batchCurrentJob.details?.phase)
        : batchCompletedJobs >= batchExpectedFiles
          ? "Batch import complete"
          : "Waiting for queued jobs"
    : activeJob?.details?.phase_label ?? formatPhaseLabel(activeJob?.details?.phase);
  const activeStageDetail = batchIsActive
    ? uploadProgress
      ? `Uploading file ${uploadProgress.file_index} of ${uploadProgress.file_count}`
      : batchCurrentJob?.details?.stage_detail ??
        (batchCompletedJobs >= batchExpectedFiles ? "All streamed files have finished importing." : "Waiting for the next queued file.")
    : activeJob?.details?.stage_detail ?? null;
  const currentFileIndex = batchIsActive
    ? Math.min(batchExpectedFiles, batchCompletedJobs + (batchCurrentJob ? 1 : 0))
    : activeJob?.details?.current_file_index ??
      (activeJob?.status === "running" && totalFiles > 0 ? Math.min(processedFiles + 1, totalFiles) : processedFiles);
  const activeInputLabel = batchIsActive
    ? batchSourceLabel || uploadSourceLabel || selectionLabel
    : activeJob?.details?.display_input_path ?? activeJob?.input_path ?? (uploadProgress ? uploadSourceLabel || selectionLabel : "-");
  const currentFileLabel = batchIsActive
    ? batchCurrentJob?.details?.display_input_path ?? batchCurrentJob?.current_file ?? uploadProgress?.current_file_name ?? "-"
    : activeJob?.current_file ?? uploadProgress?.current_file_name ?? "-";
  const activeMessage = batchIsActive
    ? batchCurrentJob?.message ??
      (uploadProgress
        ? uploadProgress.stage === "queued"
          ? "Upload complete. Remaining files are continuing through analysis and import."
          : "Files are streaming to the backend one by one."
        : batchCompletedJobs >= batchExpectedFiles
          ? "All streamed files have finished importing."
          : "Queued import jobs are still running.")
    : activeJob?.message ??
      (uploadProgress
        ? uploadProgress.stage === "queued"
          ? "Upload complete. Waiting for backend import to appear."
          : "Files are being transferred to the backend staging area."
        : "Waiting for job events.");
  const latestResult = latestCompletedJob?.details?.result_summary ?? null;
  const latestSingleFileSummary = latestCompletedJob?.details?.single_file_summary ?? latestResult?.single_file_summary ?? null;
  const batchImportedSpectra = batchJobs.reduce((sum, job) => sum + job.exported_spectra, 0);
  const batchImportedMediaAssets = batchJobs.reduce(
    (sum, job) => sum + Number(job.details?.imported_media_assets ?? job.details?.result_summary?.imported_media_assets ?? 0),
    0
  );
  const activeImportedMediaAssets = batchIsActive
    ? batchImportedMediaAssets
    : Number(activeJob?.details?.imported_media_assets ?? activeJob?.details?.result_summary?.imported_media_assets ?? 0);
  const latestUploadRate = uploadSpeedHistory[uploadSpeedHistory.length - 1] ?? 0;
  const smoothedUploadRate = useMemo(() => {
    if (uploadSpeedHistory.length === 0) {
      return 0;
    }
    const recent = uploadSpeedHistory.slice(-24);
    const weighted = recent.reduce(
      (accumulator, value, index) => {
        const weight = index + 1;
        return {
          total: accumulator.total + value * weight,
          weight: accumulator.weight + weight
        };
      },
      { total: 0, weight: 0 }
    );
    return weighted.weight > 0 ? weighted.total / weighted.weight : 0;
  }, [uploadSpeedHistory]);
  const uploadEtaSeconds = uploadProgress && uploadProgress.stage === "uploading" && smoothedUploadRate > 0
    ? Math.max(0, (uploadProgress.total_bytes - uploadProgress.loaded_bytes) / smoothedUploadRate)
    : null;
  const uploadExpectedFinish = uploadEtaSeconds != null ? new Date(Date.now() + uploadEtaSeconds * 1000) : null;

  useEffect(() => {
    const folderInput = folderInputRef.current;
    if (!folderInput) {
      return;
    }
    folderInput.setAttribute("webkitdirectory", "");
    folderInput.setAttribute("directory", "");
  }, []);

  useEffect(() => {
    if (selectedFolderFiles.length === 0) {
      setPendingFolderFiles([]);
      setSkippedFolderHistory([]);
      setHistoryLoading(false);
      setHistoryError("");
      return;
    }
    if (forceReimport) {
      setPendingFolderFiles(selectedFolderFiles);
      setSkippedFolderHistory([]);
      setHistoryLoading(false);
      setHistoryError("");
      return;
    }

    let cancelled = false;
    setHistoryLoading(true);
    setHistoryError("");
    void props.onFetchUploadHistory(selectedFolderName || undefined)
      .then((items) => {
        if (cancelled) {
          return;
        }
        const historyByPath = new Map<string, ImportedUploadHistoryItem>();
        items.forEach((item) => {
          buildRelativeLookupKeys(item.relative_path, selectedFolderName || undefined).forEach((key) => {
            if (!historyByPath.has(key)) {
              historyByPath.set(key, item);
            }
          });
        });
        const nextPending: File[] = [];
        const nextSkipped: ImportedUploadHistoryItem[] = [];
        selectedFolderFiles.forEach((file) => {
          const relativePath = buildRelativeUploadPath(selectedFolderName || "selected-folder", file, "folder_upload");
          const match = buildRelativeLookupKeys(relativePath, selectedFolderName || undefined)
            .map((key) => historyByPath.get(key))
            .find((item): item is ImportedUploadHistoryItem => item != null);
          if (match) {
            nextSkipped.push(match);
            return;
          }
          nextPending.push(file);
        });
        setPendingFolderFiles(nextPending);
        setSkippedFolderHistory(nextSkipped);
      })
      .catch((caught) => {
        if (cancelled) {
          return;
        }
        setPendingFolderFiles(selectedFolderFiles);
        setSkippedFolderHistory([]);
        setHistoryError(caught instanceof Error ? caught.message : String(caught));
      })
      .finally(() => {
        if (!cancelled) {
          setHistoryLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [forceReimport, selectedFolderFiles, selectedFolderName]);

  useEffect(() => {
    if (!uploadProgress || uploadProgress.stage !== "uploading") {
      if (!uploadProgress) {
        uploadProgressHistoryRef.current = [];
        setUploadSpeedHistory([]);
      }
      return;
    }
    const nextPoint = { at: Date.now(), loaded: uploadProgress.loaded_bytes };
    const previous = uploadProgressHistoryRef.current[uploadProgressHistoryRef.current.length - 1];
    uploadProgressHistoryRef.current = [...uploadProgressHistoryRef.current.slice(-319), nextPoint];
    if (!previous || nextPoint.loaded <= previous.loaded) {
      return;
    }
    const elapsedSeconds = Math.max(0.001, (nextPoint.at - previous.at) / 1000);
    const bytesPerSecond = (nextPoint.loaded - previous.loaded) / elapsedSeconds;
    setUploadSpeedHistory((current) => [...current.slice(-359), bytesPerSecond]);
  }, [uploadProgress]);

  function handlePathChange(event: ChangeEvent<HTMLInputElement>) {
    const value = event.target.value;
    setInputPath(value);
    if (value.trim()) {
      clearFileSelection();
      clearFolderSelection();
      setProbeResult(null);
      setProbeError("");
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = Array.from(event.target.files ?? []).find((file) => file.name.toLowerCase().endsWith(".wip")) ?? null;
    setSelectedFile(nextFile);
    setProbeResult(null);
    setProbeError("");
    clearFolderSelection();
    setInputPath("");
  }

  function handleFolderChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []).filter((file) => file.name.toLowerCase().endsWith(".wip"));
    setSelectedFolderFiles(files);
    setSelectedFolderName(inferFolderName(files));
    clearFileSelection();
    setProbeResult(null);
    setProbeError("");
    setInputPath("");
  }

  function clearFileSelection() {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  function clearFolderSelection() {
    setSelectedFolderFiles([]);
    setSelectedFolderName("");
    setPendingFolderFiles([]);
    setSkippedFolderHistory([]);
    setHistoryError("");
    if (folderInputRef.current) {
      folderInputRef.current.value = "";
    }
  }

  function clearAllSelections() {
    clearFileSelection();
    clearFolderSelection();
    setInputPath("");
    setProbeResult(null);
    setProbeError("");
  }

  async function handleProbeSelectedFile() {
    if (!selectedFile) {
      return;
    }
    setProbing(true);
    setProbeError("");
    try {
      const result = await props.onProbeFile(selectedFile);
      setProbeResult(result);
    } catch (caught) {
      setProbeError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setProbing(false);
    }
  }

  async function handleSubmit() {
    const trimmedPath = inputPath.trim();
    if (selectionMode === "none") {
      return;
    }

    setStarting(true);
    stopRequestedRef.current = false;
    uploadAbortRef.current = null;
    uploadProgressHistoryRef.current = [];
    setUploadSpeedHistory([]);
    setUploadSourceLabel(selectionLabel);

    try {
      if (selectionMode === "file" && selectedFile) {
        let batchStarted = false;
        const markBatchStarted = () => {
          if (batchStarted) {
            return;
          }
          batchStarted = true;
          setStarting(false);
          clearAllSelections();
        };
        setBatchJobIds([]);
        setBatchSourceLabel(`Selected file: ${selectedFile.name}`);
        setBatchExpectedFiles(1);
        const result = await props.onStartUpload(
          selectedFile.name,
          [selectedFile],
          "file_upload",
          forceReimport,
          options,
          (progress) => {
            markBatchStarted();
            setUploadProgress(progress);
          },
          (job) => {
            markBatchStarted();
            setBatchJobIds((current) => (current.includes(job.job_id) ? current : [...current, job.job_id]));
          },
          () => stopRequestedRef.current,
          (abortCurrentUpload) => {
            uploadAbortRef.current = abortCurrentUpload;
          }
        );
        setBatchExpectedFiles(result.jobs.length);
        if (result.jobs.length === 0) {
          setBatchSourceLabel("");
          setBatchExpectedFiles(0);
        }
      } else if (selectionMode === "folder" && selectedFolderFiles.length > 0) {
        const folderFilesToUpload = forceReimport ? selectedFolderFiles : pendingFolderFiles;
        if (folderFilesToUpload.length === 0) {
          return;
        }
        let batchStarted = false;
        const markBatchStarted = () => {
          if (batchStarted) {
            return;
          }
          batchStarted = true;
          setStarting(false);
          clearAllSelections();
        };
        setBatchJobIds([]);
        setBatchSourceLabel(
          `Selected folder: ${selectedFolderName || "selected-folder"} (${folderFilesToUpload.length}/${selectedFolderFiles.length} .wip files queued)`
        );
        setBatchExpectedFiles(folderFilesToUpload.length);
        const result = await props.onStartUpload(
          selectedFolderName || "selected-folder",
          folderFilesToUpload,
          "folder_upload",
          forceReimport,
          options,
          (progress) => {
            markBatchStarted();
            setUploadProgress(progress);
          },
          (job) => {
            markBatchStarted();
            setBatchJobIds((current) => (current.includes(job.job_id) ? current : [...current, job.job_id]));
          },
          () => stopRequestedRef.current,
          (abortCurrentUpload) => {
            uploadAbortRef.current = abortCurrentUpload;
          }
        );
        setBatchExpectedFiles(result.jobs.length);
        const recordedFailedJobs = result.jobs.filter((job) => job.status === "failed").length;
        setBatchSourceLabel(
          recordedFailedJobs > 0 || result.failedUploads.length > 0
            ? `Selected folder: ${selectedFolderName || "selected-folder"} (${result.jobs.length} recorded, ${recordedFailedJobs + result.failedUploads.length} failed or skipped)`
            : `Selected folder: ${selectedFolderName || "selected-folder"} (${result.jobs.length}/${selectedFolderFiles.length} .wip files queued)`
        );
        if (result.jobs.length === 0) {
          setBatchSourceLabel("");
          setBatchExpectedFiles(0);
        }
      } else if (trimmedPath) {
        setBatchJobIds([]);
        setBatchSourceLabel("");
        setBatchExpectedFiles(0);
        await props.onStart(trimmedPath, recursive, forceReimport, options);
      }
    } catch (caught) {
      if (!(caught instanceof Error) || caught.message !== "UPLOAD_ABORTED") {
        throw caught;
      }
    } finally {
      setStarting(false);
      window.setTimeout(() => {
        setUploadProgress(null);
        setUploadSourceLabel("");
        uploadAbortRef.current = null;
      }, 1200);
    }
  }

  async function handleStopActiveImport() {
    stopRequestedRef.current = true;
    uploadAbortRef.current?.();
    uploadAbortRef.current = null;
    const jobIds = (batchIsActive ? batchJobs : activeJob ? [activeJob] : [])
      .filter((job) => !isTerminalJob(job.status) && job.status !== "cancelled")
      .map((job) => job.job_id);
    if (jobIds.length === 0) {
      return;
    }
    setStopping(true);
    try {
      await props.onStopJobs(jobIds);
    } finally {
      setStopping(false);
    }
  }

  return (
    <section className="panel-grid">
      <div className="card card-span-2">
        <div className="card-head">
          <div>
            <p className="eyebrow">Import pipeline</p>
            <h2>Pick a file or parent folder</h2>
          </div>
        </div>

        <div className="selection-grid">
          <div className={`selection-card ${selectionMode === "file" ? "selection-card-active" : ""}`}>
            <p className="eyebrow">Single file</p>
            <h3>Choose one `.wip`</h3>
            <p className="selection-copy">Use the file chooser, then inspect how many spectra datasets and photo entries the WITec project contains.</p>
            <div className="button-row">
              <button className="secondary-button" onClick={() => fileInputRef.current?.click()} type="button">
                Choose file
              </button>
              <button
                className="secondary-button"
                disabled={!selectedFile || probing}
                onClick={handleProbeSelectedFile}
                type="button"
              >
                {probing ? "Analyzing..." : "Analyze composition"}
              </button>
            </div>
            <input accept=".wip" onChange={handleFileChange} ref={fileInputRef} style={{ display: "none" }} type="file" />
            <p className="folder-picker-summary">{selectedFile ? selectedFile.name : "No single file selected yet."}</p>
          </div>

          <div className={`selection-card ${selectionMode === "folder" ? "selection-card-active" : ""}`}>
            <p className="eyebrow">Batch folder</p>
            <h3>Choose a parent folder</h3>
            <p className="selection-copy">
              Every nested `.wip` inside the selected folder, including all subfolders, is streamed one by one. Each
              file starts importing as soon as its upload finishes.
            </p>
            <div className="button-row">
              <button className="secondary-button" onClick={() => folderInputRef.current?.click()} type="button">
                Choose folder
              </button>
              <button
                className="secondary-button"
                disabled={selectedFolderFiles.length === 0}
                onClick={clearFolderSelection}
                type="button"
              >
                Clear folder
              </button>
            </div>
            <input
              accept=".wip"
              multiple
              onChange={handleFolderChange}
              ref={folderInputRef}
              style={{ display: "none" }}
              type="file"
            />
            <p className="folder-picker-summary">
              {selectedFolderFiles.length > 0
                ? `${selectedFolderName || "selected-folder"} | ${selectedFolderFiles.length} .wip files`
                : "No parent folder selected yet."}
            </p>
            {selectedFolderFiles.length > 0 ? (
              <div className="import-plan-summary">
                <span className="plan-pill">📦 Total {selectedFolderFiles.length}</span>
                <span className="plan-pill plan-pill-good">🆕 Pending {pendingFolderCount}</span>
                <span className="plan-pill">⏭️ Skip {skippedFolderCount}</span>
              </div>
            ) : null}
            {historyLoading ? <p className="helper-copy helper-copy-tight">Checking local import history...</p> : null}
            {historyError ? <p className="helper-copy helper-copy-tight">History lookup failed, all files will remain available for upload.</p> : null}
          </div>
        </div>

        <div className="selected-source-card">
          <div>
            <p className="eyebrow">Current source</p>
            <h3>{selectionLabel}</h3>
          </div>
          <div className="button-row">
            <button
              className="primary-button"
              disabled={!canStartSelection}
              onClick={handleSubmit}
              type="button"
            >
              {starting
                ? "Starting..."
                : historyLoading
                  ? "Checking history..."
                : selectionMode === "file"
                  ? "Import selected file"
                  : selectionMode === "folder"
                    ? pendingFolderCount > 0
                      ? `Import ${pendingFolderCount} new files`
                      : "Nothing new to import"
                    : "Start import"}
            </button>
            <button className="secondary-button" disabled={selectionMode === "none"} onClick={clearAllSelections} type="button">
              Clear selection
            </button>
            {(starting || uploadProgress || (activeJob && !isTerminalJob(activeJob.status)) || batchIsActive) ? (
              <button className="secondary-button" disabled={stopping} onClick={handleStopActiveImport} type="button">
                {stopping ? "Stopping..." : "Stop current import"}
              </button>
            ) : null}
          </div>
        </div>

        <details className="manual-import-shell">
          <summary>Manual path import for local or network shares</summary>
          <div className="form-grid import-form-grid">
            <label className="field field-wide">
              <span>Input path</span>
              <input
                placeholder="C:\\Users\\Admin\\Documents\\20260127-GrayScale.wip or \\\\server\\share\\PL\\batch_01"
                value={inputPath}
                onChange={handlePathChange}
              />
            </label>
            <label className="toggle-row">
              <input
                checked={recursive}
                disabled={selectionMode !== "path"}
                onChange={(event) => setRecursive(event.target.checked)}
                type="checkbox"
              />
              <span>Recursive folder import</span>
            </label>
          </div>
        </details>

        <div className="import-options-shell">
          <div className="import-option-section">
            <p className="eyebrow">Behavior</p>
            <div className="import-option-grid import-option-grid-single">
              <ImportOptionCard
                checked={forceReimport}
                description="Clear previously indexed content when the same file path or file hash is imported again."
                label="♻️ Force reimport"
                onChange={setForceReimport}
              />
            </div>
          </div>
          <div className="import-option-section">
            <p className="eyebrow">Content</p>
            <p className="helper-copy helper-copy-tight">
              Choose only the WITec content you want. Imported photos are compressed to a normal 8-bit JPEG at roughly 480p.
            </p>
            <div className="import-option-grid">
              <ImportOptionCard
                checked={options.include_point_spectra}
                description="Single-point spectra and standalone graph traces."
                label="🔬 Point spectra"
                onChange={(checked) => setOptions({ ...options, include_point_spectra: checked })}
              />
              <ImportOptionCard
                checked={options.include_line_scans}
                description="Sequential traces along one spatial scan path."
                label="📏 Line scans"
                onChange={(checked) => setOptions({ ...options, include_line_scans: checked })}
              />
              <ImportOptionCard
                checked={options.include_area_maps}
                description="Mapped spectra over 2D regions, including scan grids."
                label="🗺️ Area maps"
                onChange={(checked) => setOptions({ ...options, include_area_maps: checked })}
              />
              <ImportOptionCard
                checked={options.include_series_scans}
                description="Time-series or power-series measurements."
                label="⏱️ Series scans"
                onChange={(checked) => setOptions({ ...options, include_series_scans: checked })}
              />
              <ImportOptionCard
                checked={options.include_photo_images}
                description="TDImage and TDBitmap photo entries, resized and compressed to stay lightweight."
                label="🖼️ Photo images"
                onChange={(checked) => setOptions({ ...options, include_photo_images: checked })}
              />
            </div>
          </div>
        </div>
        <p className="helper-copy">
          File and folder choosers are the primary path now. Manual paths remain available for UNC shares such as
          <code>\\server\share\PL\batch_01</code>. Folder imports include nested subfolders by default and now stream
          file-by-file instead of waiting for the whole folder to finish uploading.
        </p>
      </div>

      <div className="card card-span-1">
        <div className="card-head">
          <div>
            <p className="eyebrow">Progress</p>
            <h2>{batchIsActive ? "Current batch" : "Current job"}</h2>
          </div>
        </div>
        {activeJob || uploadProgress ? (
          <div className="summary-block">
            <div className="summary-row">
              <span>Status</span>
              <strong>
                {batchIsActive
                  ? batchCompletedJobs >= batchExpectedFiles && !uploadProgress
                    ? "finished"
                    : "streaming"
                  : activeJob?.status ?? (uploadProgress?.stage === "queued" ? "queued" : "uploading")}
              </strong>
            </div>
            <div className="summary-row">
              <span>Phase</span>
              <strong>{activePhaseLabel}</strong>
            </div>
            {activeStageDetail ? (
              <div className="summary-row">
                <span>Stage detail</span>
                <span>{activeStageDetail}</span>
              </div>
            ) : null}
            <div className="progress-group">
              <div className="progress-meta">
                <span>Upload transfer</span>
                <strong>
                  {uploadProgress
                    ? `${uploadPercent}% | ${uploadProgress.file_index}/${uploadProgress.file_count} files`
                    : activeJob || batchIsActive
                      ? "Done"
                      : "Waiting"}
                </strong>
              </div>
              <div className="progress-shell">
                <div className="progress-fill" style={{ width: `${uploadProgress ? uploadPercent : activeJob || batchIsActive ? 100 : 0}%` }} />
              </div>
              <span className="progress-caption">
                {uploadProgress
                  ? `${formatBytes(uploadProgress.loaded_bytes)} / ${formatBytes(uploadProgress.total_bytes)} | ${uploadProgress.current_file_name ?? "-"}`
                  : batchIsActive
                    ? "Queued files have already been handed to the backend."
                    : activeJob
                      ? "Files have been transferred to the backend staging area."
                      : "No upload is currently in progress."}
              </span>
              {uploadProgress && uploadProgress.stage === "uploading" ? (
                <div className="upload-telemetry">
                  <div className="upload-telemetry-row">
                    <span>Current speed</span>
                    <strong>{latestUploadRate > 0 ? formatDataRate(latestUploadRate) : "Learning..."}</strong>
                  </div>
                  <div className="upload-telemetry-row">
                    <span>Rolling mean</span>
                    <strong>{smoothedUploadRate > 0 ? formatDataRate(smoothedUploadRate) : "Estimating..."}</strong>
                  </div>
                  <div className="upload-telemetry-row">
                    <span>ETA / finish</span>
                    <strong>
                      {uploadEtaSeconds != null && uploadExpectedFinish
                        ? `${formatDuration(uploadEtaSeconds)} | ${formatClockTime(uploadExpectedFinish)}`
                        : "Estimating..."}
                    </strong>
                  </div>
                  <UploadSpeedSparkline values={uploadSpeedHistory} />
                </div>
              ) : null}
            </div>
            <div className="progress-group">
              <div className="progress-meta">
                <span>Backend import</span>
                <strong>
                  {batchIsActive
                    ? `${progress}% | ${batchCompletedJobs}/${batchExpectedFiles} files complete`
                    : totalFiles > 0
                      ? `${progress}% | file ${currentFileIndex}/${totalFiles}`
                      : activeJob
                        ? activeJob.status
                        : "Waiting"}
                </strong>
              </div>
              <div className="progress-shell">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
              <span className="progress-caption">
                {batchIsActive
                  ? batchCurrentJob?.details?.display_input_path ?? batchCurrentJob?.current_file ?? "Waiting for the next queued file."
                  : activeJob?.current_file
                    ? `${activeJob.current_file}`
                    : activeJob
                      ? "Backend is waiting for the next step."
                      : "No backend import is currently running."}
              </span>
            </div>
            <div className="summary-row">
              <span>Files</span>
              <strong>
                {batchIsActive
                  ? `${batchCompletedJobs}/${batchExpectedFiles} completed`
                  : activeJob
                    ? `${activeJob.processed_files}/${activeJob.total_files}`
                    : "-"}
              </strong>
            </div>
            {batchIsActive ? (
              <div className="summary-row">
                <span>Active jobs</span>
                <strong>{batchRunningJobs}</strong>
              </div>
            ) : null}
            <div className="summary-row">
              <span>Imported spectra</span>
              <strong>{batchIsActive ? batchImportedSpectra : activeJob?.exported_spectra ?? "-"}</strong>
            </div>
            <div className="summary-row">
              <span>Imported photos</span>
              <strong>{activeImportedMediaAssets}</strong>
            </div>
            <div className="summary-row">
              <span>Input</span>
              <code>{activeInputLabel}</code>
            </div>
            <div className="summary-row">
              <span>Current file</span>
              <code>{currentFileLabel}</code>
            </div>
            <div className="summary-row">
              <span>Dataset mix</span>
              <strong>{formatModeCounts(datasetCounts)}</strong>
            </div>
            <div className="summary-row">
              <span>Trace mix</span>
              <strong>{formatModeCounts(inventory)}</strong>
            </div>
            <div className="summary-row">
              <span>Photo mix</span>
              <strong>{formatModeCounts(mediaInventory)}</strong>
            </div>
            <div className="summary-row">
              <span>Import modes</span>
              <strong>{formatImportOptions(activeJob?.details?.import_options ?? options)}</strong>
            </div>
            <div className="summary-row">
              <span>Log file</span>
              <code>{activeJob?.log_path ?? "-"}</code>
            </div>
            <div className="log-box">{activeMessage}</div>
          </div>
        ) : (
          <p className="empty-state">No import job has started yet.</p>
        )}
      </div>

      {selectedFile && (probeResult || probeError) ? (
        <div className="card card-span-3">
          <div className="card-head">
            <div>
              <p className="eyebrow">Single-file probe</p>
              <h2>Composition snapshot</h2>
            </div>
          </div>
          {probeError ? (
            <p className="empty-state">{probeError}</p>
          ) : probeResult ? (
            <div className="result-shell">
              <div className="result-hero">
                <p className="result-title">{selectedFile.name}</p>
                <p className="result-copy">{formatProbeCompositionSummary(probeResult)}</p>
              </div>
              <div className="result-stat-grid">
                <ResultStat label="Datasets" value={formatModeCounts(probeResult.inventory_by_mode)} />
                <ResultStat label="Traces" value={formatModeCounts(probeResult.traces_by_mode)} />
                <ResultStat label="Photos" value={formatModeCounts(probeResult.media_inventory ?? {})} />
                <ResultStat label="Entry classes" value={formatModeCounts(probeResult.class_counts)} />
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {latestCompletedJob && latestResult ? (
        <div className="card card-span-3">
          <div className="card-head">
            <div>
              <p className="eyebrow">Latest result</p>
              <h2>Newly loaded data</h2>
            </div>
            <span className={`pill ${latestCompletedJob.status === "finished" ? "pill-good" : "pill-warn"}`}>
              {latestCompletedJob.status}
            </span>
          </div>
          <div className="result-shell">
            <div className="result-hero">
              <p className="result-title">{latestCompletedJob.details?.display_input_path ?? latestCompletedJob.input_path}</p>
              <p className="result-copy">{latestCompletedJob.message ?? "Latest import completed."}</p>
            </div>
            <div className="result-stat-grid">
              <ResultStat label="New spectra" value={String(latestResult.imported_spectra)} />
              <ResultStat label="New datasets" value={String(latestResult.imported_dataset_count)} />
              <ResultStat label="New photos" value={String(latestResult.imported_media_assets ?? 0)} />
              <ResultStat label="Imported files" value={String(latestResult.imported_file_count)} />
              <ResultStat label="Skipped duplicates" value={String(latestResult.duplicate_file_count)} />
              <ResultStat label="Skipped existing" value={String(latestResult.skipped_existing_count)} />
              <ResultStat label="Failed files" value={String(latestResult.failed_file_count)} />
            </div>
            <div className="result-breakdown-grid">
              <div className="result-breakdown-card">
                <p className="eyebrow">Dataset mix</p>
                <strong>{formatModeCounts(latestResult.dataset_mode_counts)}</strong>
              </div>
              <div className="result-breakdown-card">
                <p className="eyebrow">Trace mix</p>
                <strong>{formatModeCounts(latestResult.trace_mode_counts)}</strong>
              </div>
              <div className="result-breakdown-card">
                <p className="eyebrow">Photo mix</p>
                <strong>{formatModeCounts(latestResult.media_type_counts ?? {})}</strong>
              </div>
              <div className="result-breakdown-card">
                <p className="eyebrow">Completed</p>
                <strong>{formatTimestamp(latestCompletedJob.end_time ?? latestCompletedJob.start_time)}</strong>
              </div>
            </div>

            {latestSingleFileSummary ? (
              <div className="single-file-summary-card">
                <p className="eyebrow">Single-file composition</p>
                <div className="summary-row">
                  <span>Source</span>
                  <code>{latestSingleFileSummary.source_wip}</code>
                </div>
                <div className="summary-row">
                  <span>Dataset composition</span>
                  <strong>{formatModeCounts(latestSingleFileSummary.dataset_mode_counts)}</strong>
                </div>
                <div className="summary-row">
                  <span>Trace composition</span>
                  <strong>{formatModeCounts(latestSingleFileSummary.detected_inventory)}</strong>
                </div>
                <div className="summary-row">
                  <span>Photo composition</span>
                  <strong>{formatModeCounts(latestSingleFileSummary.media_inventory ?? {})}</strong>
                </div>
                {latestSingleFileSummary.duplicate_of_source ? (
                  <div className="summary-row">
                    <span>Duplicate of</span>
                    <code>{latestSingleFileSummary.duplicate_of_source}</code>
                  </div>
                ) : null}
                {latestSingleFileSummary.error_message ? (
                  <div className="summary-row">
                    <span>Notes</span>
                    <span>{latestSingleFileSummary.error_message}</span>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="card card-span-3">
        <div className="card-head">
          <div>
            <p className="eyebrow">History</p>
            <h2>Import job log</h2>
          </div>
        </div>
        <div className="table-shell import-history-shell">
          <table>
            <thead>
              <tr>
                <th>Job ID</th>
                <th>Status</th>
                <th>Phase</th>
                <th>Input path</th>
                <th>Files</th>
                <th>Spectra</th>
                <th>Trace mix</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {props.jobs.map((job) => (
                <tr key={job.job_id}>
                  <td>{job.job_id}</td>
                  <td>{job.status}</td>
                  <td>{job.details?.phase ?? "-"}</td>
                  <td className="truncate-cell">{job.details?.display_input_path ?? job.input_path}</td>
                  <td>
                    {job.processed_files}/{job.total_files}
                  </td>
                  <td>{job.exported_spectra}</td>
                  <td>{formatModeCounts(job.details?.detected_inventory ?? {})}</td>
                  <td>{formatTimestamp(job.start_time)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function ResultStat(props: { label: string; value: string }) {
  return (
    <div className="result-stat-card">
      <span className="result-stat-label">{props.label}</span>
      <strong className="result-stat-value">{props.value}</strong>
    </div>
  );
}

function UploadSpeedSparkline(props: { values: number[] }) {
  const points = props.values.slice(-280);
  if (points.length < 2) {
    return <div className="upload-sparkline-empty">Speed history will appear after a few upload samples.</div>;
  }
  const width = 320;
  const height = 74;
  const maxValue = Math.max(...points, 1);
  const minValue = Math.min(...points, maxValue);
  const latestValue = points[points.length - 1] ?? 0;
  const normalizedSpan = Math.max(1, maxValue - minValue);
  const plotPoints = points.map((value, index) => {
    const x = (index / Math.max(1, points.length - 1)) * width;
    const y = height - 10 - (((value - minValue) / normalizedSpan) * (height - 22));
    return { x, y };
  });
  const linePath = buildSmoothPath(plotPoints);
  const areaPath = `${linePath} L ${width} ${height - 6} L 0 ${height - 6} Z`;

  return (
    <div className="upload-sparkline-shell">
      <div className="upload-sparkline-labels">
        <span>{formatDataRate(maxValue)}</span>
        <strong>{formatDataRate(latestValue)}</strong>
      </div>
      <svg className="upload-sparkline" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Upload speed history">
        <defs>
          <linearGradient id="upload-speed-fill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="rgba(34, 161, 179, 0.34)" />
            <stop offset="100%" stopColor="rgba(34, 161, 179, 0.02)" />
          </linearGradient>
        </defs>
        <path d={`M0 ${height - 6} L${width} ${height - 6}`} className="upload-sparkline-baseline" />
        <path d={`M0 ${height * 0.52} L${width} ${height * 0.52}`} className="upload-sparkline-grid" />
        <path className="upload-sparkline-fill" d={areaPath} />
        <path className="upload-sparkline-glow" d={linePath} />
        <path className="upload-sparkline-line" d={linePath} />
      </svg>
    </div>
  );
}

function buildSmoothPath(points: Array<{ x: number; y: number }>): string {
  if (points.length === 0) {
    return "";
  }
  if (points.length === 1) {
    return `M ${points[0].x} ${points[0].y}`;
  }

  let path = `M ${points[0].x} ${points[0].y}`;
  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const midX = (current.x + next.x) / 2;
    path += ` C ${midX} ${current.y}, ${midX} ${next.y}, ${next.x} ${next.y}`;
  }
  return path;
}

function ImportOptionCard(props: {
  checked: boolean;
  label: string;
  description: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label className={`import-option-card ${props.checked ? "import-option-card-active" : ""}`}>
      <span className="import-option-head">
        <input checked={props.checked} onChange={(event) => props.onChange(event.target.checked)} type="checkbox" />
        <span className="import-option-label">{props.label}</span>
      </span>
      <span className="import-option-copy">{props.description}</span>
    </label>
  );
}

function formatImportOptions(options: ImportOptions | null | undefined): string {
  if (!options) {
    return "-";
  }
  const labels: string[] = [];
  if (options.include_point_spectra) {
    labels.push("point");
  }
  if (options.include_line_scans) {
    labels.push("line");
  }
  if (options.include_area_maps) {
    labels.push("area");
  }
  if (options.include_series_scans) {
    labels.push("series");
  }
  if (options.include_photo_images) {
    labels.push("photo");
  }
  return labels.length > 0 ? labels.join(", ") : "none";
}

function formatProbeCompositionSummary(probeResult: ImportProbeReport): string {
  const datasetCount = Object.values(probeResult.inventory_by_mode ?? {}).reduce((sum, value) => sum + Number(value || 0), 0);
  const photoCount = Object.values(probeResult.media_inventory ?? {}).reduce((sum, value) => sum + Number(value || 0), 0);
  const projectLabel = probeResult.project_version ? `WITec project ${probeResult.project_version}` : "WITec project analyzed";
  const parts = [`${datasetCount} spectrum datasets`];
  if (photoCount > 0) {
    parts.push(`${photoCount} photos`);
  }
  return `${projectLabel} | ${parts.join(" | ")}`;
}

function buildRelativeUploadPath(
  rootName: string,
  file: File,
  sourceKind: "file_upload" | "folder_upload"
): string {
  const browserFile = file as File & { webkitRelativePath?: string };
  const fallbackRelativePath = sourceKind === "file_upload" ? file.name : `${rootName}/${file.name}`;
  return (browserFile.webkitRelativePath || fallbackRelativePath).trim();
}

function normalizeRelativeUploadPath(value: string): string {
  return String(value || "").trim().replace(/\\/g, "/").replace(/^\.?\//, "").replace(/\/+$/, "").toLowerCase();
}

function buildRelativeLookupKeys(value: string, rootName?: string): string[] {
  const normalized = normalizeRelativeUploadPath(value);
  if (!normalized) {
    return [];
  }

  const keys = new Set<string>([normalized]);
  const normalizedRoot = normalizeRelativeUploadPath(rootName || "");
  if (normalizedRoot && normalized.startsWith(`${normalizedRoot}/`)) {
    keys.add(normalized.slice(normalizedRoot.length + 1));
  }
  return [...keys];
}

function formatDataRate(bytesPerSecond: number): string {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) {
    return "0 B/s";
  }
  const units = ["B/s", "KB/s", "MB/s", "GB/s"];
  let value = bytesPerSecond;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  const digits = value >= 100 ? 0 : value >= 10 ? 1 : 2;
  return `${value.toFixed(digits)} ${units[unitIndex]}`;
}

function formatDuration(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds < 0) {
    return "-";
  }
  const seconds = Math.round(totalSeconds);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${remainingSeconds}s`;
  }
  return `${remainingSeconds}s`;
}

function formatClockTime(value: Date): string {
  return value.toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit"
  });
}

function formatModeCounts(counts: Record<string, number> | null | undefined): string {
  const entries = Object.entries(counts ?? {}).filter(([, value]) => Number(value) > 0);
  if (entries.length === 0) {
    return "-";
  }
  return entries
    .map(([label, value]) => `${friendlyModeLabel(label)} ${value}`)
    .join("  ");
}

function friendlyModeLabel(label: string): string {
  switch (label) {
    case "point_spectrum":
      return "point";
    case "line_scan":
      return "line";
    case "area_map":
      return "area";
    case "series_scan":
      return "series";
    case "photo_image":
      return "photo";
    default:
      return label.replace(/_/g, " ");
  }
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function formatPhaseLabel(phase: string | null | undefined): string {
  switch (phase) {
    case "queued":
      return "Queued";
    case "uploading":
      return "Uploading files";
    case "hashing_file":
      return "Hashing current file";
    case "analyzing_file":
      return "Analyzing WIP structure";
    case "ingesting":
      return "Writing traces to database";
    case "finalizing":
      return "Finalizing import";
    case "finished":
      return "Finished";
    case "partially_failed":
      return "Partially failed";
    case "failed":
      return "Failed";
    case "reading_wip":
      return "Reading WIP files";
    case "mock_import":
      return "Mock import";
    default:
      return phase ? phase.replace(/_/g, " ") : "-";
  }
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  const digits = size >= 100 || index === 0 ? 0 : size >= 10 ? 1 : 2;
  return `${size.toFixed(digits)} ${units[index]}`;
}

function isTerminalJob(status: string): boolean {
  return status === "finished" || status === "failed" || status === "partially_failed" || status === "cancelled";
}

function approximateJobCompletion(job: ImportJob): number {
  if (isTerminalJob(job.status)) {
    return 1;
  }
  const overallProgress = job.details?.overall_progress;
  if (typeof overallProgress === "number" && Number.isFinite(overallProgress) && overallProgress > 0) {
    return Math.max(0, Math.min(1, overallProgress / 100));
  }
  switch (job.details?.phase) {
    case "queued":
      return 0;
    case "reading_wip":
      return 0.08;
    case "hashing_file":
      return 0.2;
    case "analyzing_file":
      return 0.45;
    case "ingesting":
      return 0.8;
    case "finalizing":
      return 0.95;
    default:
      return job.status === "running" ? 0.15 : 0;
  }
}

function aggregateJobCounts(
  jobs: ImportJob[],
  primaryKey: "detected_inventory" | "dataset_mode_counts" | "media_inventory",
  fallbackKey: "last_file_inventory" | "last_file_dataset_counts" | "last_file_media_inventory"
): Record<string, number> {
  const totals: Record<string, number> = {};
  jobs.forEach((job) => {
    const counts = job.details?.[primaryKey] ?? job.details?.[fallbackKey] ?? {};
    Object.entries(counts).forEach(([key, value]) => {
      totals[key] = (totals[key] ?? 0) + Number(value ?? 0);
    });
  });
  return totals;
}

type BrowserFile = File & {
  webkitRelativePath?: string;
};

function inferFolderName(files: File[]): string {
  if (files.length === 0) {
    return "";
  }
  const relativePath = (files[0] as BrowserFile).webkitRelativePath?.trim();
  if (!relativePath) {
    return "selected-folder";
  }
  const parts = relativePath.split(/[\\/]/).filter(Boolean);
  return parts[0] ?? "selected-folder";
}
