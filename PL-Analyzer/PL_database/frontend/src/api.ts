import type {
  AnalysisOptions,
  BatchAnalysisResponse,
  DashboardSummary,
  FilterOptions,
  HealthPayload,
  ImportedUploadHistoryItem,
  ImportJob,
  ImportOptions,
  ImportProbeReport,
  MaterialAnalysisJob,
  SpectraResponse,
  SpectrumDetail,
  SpectrumFilters,
  UploadTransferProgress
} from "./types";

export interface UploadedImportFailure {
  relativePath: string;
  error: string;
}

export interface UploadedImportBatchResult {
  jobs: ImportJob[];
  failedUploads: UploadedImportFailure[];
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers ?? {});
  if (!(init?.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(`${API_BASE}${path}`, {
    headers,
    ...init
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

function buildQuery(filters: SpectrumFilters, limit = 100): string {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  });
  return params.toString();
}

export function getExportUrl(spectrumIds: string[]): string {
  const params = new URLSearchParams();
  params.set("spectrum_ids", spectrumIds.join(","));
  return `${API_BASE}/api/database/export?${params.toString()}`;
}

export function fetchHealth(): Promise<HealthPayload> {
  return request<HealthPayload>("/health");
}

export function fetchDashboard(): Promise<DashboardSummary> {
  return request<DashboardSummary>("/api/dashboard");
}

export function fetchImportJobs(): Promise<ImportJob[]> {
  return request<ImportJob[]>("/api/import/jobs");
}

export function stopImportJob(jobId: string): Promise<{ job_id: string; status: string }> {
  return request<{ job_id: string; status: string }>(`/api/import/jobs/${encodeURIComponent(jobId)}/stop`, {
    method: "POST"
  });
}

export function fetchImportedUploadHistory(rootName?: string): Promise<{ items: ImportedUploadHistoryItem[] }> {
  const params = new URLSearchParams();
  if (rootName?.trim()) {
    params.set("root_name", rootName.trim());
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  return request<{ items: ImportedUploadHistoryItem[] }>(`/api/import/upload-history${suffix}`);
}

export function startImport(
  inputPath: string,
  recursive: boolean,
  forceReimport: boolean,
  options: ImportOptions
): Promise<ImportJob> {
  return request<ImportJob>("/api/import/start", {
    method: "POST",
    body: JSON.stringify({
      input_path: inputPath,
      recursive,
      force_reimport: forceReimport,
      options
    })
  });
}

type BrowserFile = File & {
  webkitRelativePath?: string;
};

function buildUploadRelativePath(
  rootName: string,
  file: File,
  sourceKind: "file_upload" | "folder_upload"
): string {
  const fallbackRelativePath = sourceKind === "file_upload" ? file.name : `${rootName}/${file.name}`;
  return ((file as BrowserFile).webkitRelativePath || fallbackRelativePath).trim();
}

function buildUploadDisplayLabel(
  relativePath: string,
  sourceKind: "file_upload" | "folder_upload"
): string {
  return sourceKind === "file_upload" ? `Selected file: ${relativePath}` : `Selected folder item: ${relativePath}`;
}

function buildUploadBody(
  rootName: string,
  file: File,
  relativePath: string,
  sourceKind: "file_upload" | "folder_upload",
  forceReimport: boolean,
  options: ImportOptions,
  displayLabel: string
): FormData {
  const body = new FormData();
  body.append("root_name", rootName);
  body.append("source_kind", sourceKind);
  body.append("display_label", displayLabel);
  body.append("force_reimport", String(forceReimport));
  body.append("include_point_spectra", String(options.include_point_spectra));
  body.append("include_line_scans", String(options.include_line_scans));
  body.append("include_area_maps", String(options.include_area_maps));
  body.append("include_series_scans", String(options.include_series_scans));
  body.append("include_photo_images", String(options.include_photo_images));
  body.append("files", file, file.name);
  body.append("relative_paths", relativePath);
  return body;
}

function uploadSingleImport(
  rootName: string,
  file: File,
  relativePath: string,
  sourceKind: "file_upload" | "folder_upload",
  forceReimport: boolean,
  options: ImportOptions,
  progressContext: {
    completedBytes: number;
    totalBytes: number;
    fileIndex: number;
    fileCount: number;
  },
  onProgress?: (progress: UploadTransferProgress) => void,
  registerAbort?: (abortCurrentUpload: (() => void) | null) => void
): Promise<ImportJob> {
  const displayLabel = buildUploadDisplayLabel(relativePath, sourceKind);
  const body = buildUploadBody(rootName, file, relativePath, sourceKind, forceReimport, options, displayLabel);
  if (!onProgress) {
    return request<ImportJob>("/api/import/upload-start", {
      method: "POST",
      body
    });
  }

  return new Promise<ImportJob>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const endpoint = `${API_BASE}/api/import/upload-start`;
    xhr.open("POST", endpoint);
    xhr.responseType = "text";
    registerAbort?.(() => xhr.abort());
    onProgress({
      stage: "uploading",
      percent: progressContext.totalBytes > 0
        ? Math.min(100, Math.round((progressContext.completedBytes / progressContext.totalBytes) * 100))
        : 0,
      loaded_bytes: progressContext.completedBytes,
      total_bytes: progressContext.totalBytes,
      file_index: progressContext.fileIndex + 1,
      file_count: progressContext.fileCount,
      current_file_name: relativePath
    });

    xhr.upload.onprogress = (event) => {
      const totalBytes = progressContext.totalBytes;
      const currentFileBytes = event.lengthComputable && event.total > 0
        ? Math.min(file.size, Math.round(file.size * (event.loaded / event.total)))
        : Math.min(file.size, event.loaded);
      const loadedBytes = Math.min(totalBytes, progressContext.completedBytes + currentFileBytes);
      onProgress({
        stage: "uploading",
        percent: totalBytes > 0 ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100)) : 100,
        loaded_bytes: loadedBytes,
        total_bytes: totalBytes,
        file_index: progressContext.fileIndex + 1,
        file_count: progressContext.fileCount,
        current_file_name: relativePath
      });
    };

    xhr.onerror = () => {
      registerAbort?.(null);
      reject(new Error(`Upload failed before the server accepted ${relativePath}.`));
    };

    xhr.onabort = () => {
      registerAbort?.(null);
      reject(new Error("UPLOAD_ABORTED"));
    };

    xhr.onload = () => {
      registerAbort?.(null);
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(xhr.responseText || `Request failed: ${xhr.status}`));
        return;
      }
      onProgress({
        stage: progressContext.fileIndex + 1 >= progressContext.fileCount ? "queued" : "uploading",
        percent: progressContext.totalBytes > 0
          ? Math.min(100, Math.round(((progressContext.completedBytes + file.size) / progressContext.totalBytes) * 100))
          : 100,
        loaded_bytes: Math.min(progressContext.totalBytes, progressContext.completedBytes + file.size),
        total_bytes: progressContext.totalBytes,
        file_index: progressContext.fileIndex + 1,
        file_count: progressContext.fileCount,
        current_file_name: relativePath
      });
      try {
        resolve(JSON.parse(xhr.responseText) as ImportJob);
      } catch (error) {
        reject(error instanceof Error ? error : new Error(String(error)));
      }
    };

    xhr.send(body);
  });
}

export async function startUploadedImport(
  rootName: string,
  files: File[],
  sourceKind: "file_upload" | "folder_upload",
  forceReimport: boolean,
  options: ImportOptions,
  onProgress?: (progress: UploadTransferProgress) => void,
  onJobQueued?: (job: ImportJob) => void | Promise<void>,
  shouldStop?: () => boolean,
  registerAbort?: (abortCurrentUpload: (() => void) | null) => void
): Promise<UploadedImportBatchResult> {
  if (files.length === 0) {
    throw new Error("No .wip files were selected.");
  }

  const orderedFiles = [...files].sort((left, right) =>
    buildUploadRelativePath(rootName, left, sourceKind).localeCompare(buildUploadRelativePath(rootName, right, sourceKind))
  );
  const totalBytes = orderedFiles.reduce((sum, file) => sum + file.size, 0);
  const queuedJobs: ImportJob[] = [];
  const failedUploads: UploadedImportFailure[] = [];
  let completedBytes = 0;

  for (const [index, file] of orderedFiles.entries()) {
    if (shouldStop?.()) {
      throw new Error("UPLOAD_ABORTED");
    }
    const relativePath = buildUploadRelativePath(rootName, file, sourceKind);
    try {
      const job = await uploadSingleImport(
        rootName,
        file,
        relativePath,
        sourceKind,
        forceReimport,
        options,
        {
          completedBytes,
          totalBytes,
          fileIndex: index,
          fileCount: orderedFiles.length
        },
        onProgress,
        registerAbort
      );
      queuedJobs.push(job);
      await onJobQueued?.(job);
      completedBytes += file.size;
    } catch (error) {
      if (error instanceof Error && error.message === "UPLOAD_ABORTED") {
        throw error;
      }
      if (sourceKind === "file_upload") {
        throw error;
      }
      failedUploads.push({
        relativePath,
        error: error instanceof Error ? error.message : String(error)
      });
      completedBytes += file.size;
      onProgress?.({
        stage: index + 1 >= orderedFiles.length ? "queued" : "uploading",
        percent: totalBytes > 0 ? Math.min(100, Math.round((completedBytes / totalBytes) * 100)) : 100,
        loaded_bytes: Math.min(totalBytes, completedBytes),
        total_bytes: totalBytes,
        file_index: index + 1,
        file_count: orderedFiles.length,
        current_file_name: relativePath
      });
    }
  }

  if (onProgress) {
    const lastRelativePath =
      orderedFiles.length > 0
        ? buildUploadRelativePath(rootName, orderedFiles[orderedFiles.length - 1], sourceKind)
        : null;
    onProgress({
      stage: "queued",
      percent: 100,
      loaded_bytes: totalBytes,
      total_bytes: totalBytes,
      file_index: orderedFiles.length,
      file_count: orderedFiles.length,
      current_file_name: lastRelativePath
    });
  }

  return {
    jobs: queuedJobs,
    failedUploads
  };
}

export function probeUploadedFile(file: File): Promise<ImportProbeReport> {
  const body = new FormData();
  body.append("file", file, file.name);
  return request<ImportProbeReport>("/api/import/probe-upload", {
    method: "POST",
    body
  });
}

export function fetchSpectra(filters: SpectrumFilters): Promise<SpectraResponse> {
  const query = buildQuery(filters);
  return request<SpectraResponse>(`/api/database/spectra?${query}`);
}

export function fetchSpectrum(spectrumId: string): Promise<SpectrumDetail> {
  return request<SpectrumDetail>(`/api/database/spectra/${encodeURIComponent(spectrumId)}`);
}

export function fetchFilterOptions(): Promise<FilterOptions> {
  return request<FilterOptions>("/api/database/options");
}

export function applyMetadata(
  spectrumIds: string[],
  applyMode: "selected" | "source_file" | "folder" | "all",
  scopeValue: string | null,
  metadata: Record<string, string | null>
): Promise<{ updated_rows: number }> {
  return request<{ updated_rows: number }>("/api/database/metadata", {
    method: "POST",
    body: JSON.stringify({
      spectrum_ids: spectrumIds,
      apply_mode: applyMode,
      scope_value: scopeValue,
      metadata
    })
  });
}

export function runBatchAnalysis(
  spectrumIds: string[],
  options: AnalysisOptions,
  saveResults: boolean
): Promise<BatchAnalysisResponse> {
  return request<BatchAnalysisResponse>("/api/analysis/batch", {
    method: "POST",
    body: JSON.stringify({
      spectrum_ids: spectrumIds,
      options,
      save_results: saveResults
    })
  });
}

export function startMaterialAnalysis(
  payload: {
    spectrum_ids: string[];
    filters: SpectrumFilters;
    search?: string;
    include_mock?: boolean;
    options: AnalysisOptions;
    save_results: boolean;
    update_entries: boolean;
  }
): Promise<MaterialAnalysisJob> {
  return request<MaterialAnalysisJob>("/api/analysis/material/start", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchMaterialAnalysisJob(jobId: string): Promise<MaterialAnalysisJob> {
  return request<MaterialAnalysisJob>(`/api/analysis/material/jobs/${encodeURIComponent(jobId)}`);
}

export function stopMaterialAnalysisJob(jobId: string): Promise<{ job_id: string; status: string }> {
  return request<{ job_id: string; status: string }>(`/api/analysis/material/jobs/${encodeURIComponent(jobId)}/stop`, {
    method: "POST"
  });
}
