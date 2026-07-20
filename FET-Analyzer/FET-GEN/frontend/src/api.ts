import type {
  GenerationCondition,
  GenerationResponse,
  DatabaseAnalysisStatus,
  DatabaseAnalysisResponse,
  DatabaseSelectionState,
  CurveDetail,
  CurvePreview,
  CurveFilters,
  CurveListResponse,
  DatabaseExportOptions,
  CalendarCurveResponse,
  AppMode,
  DatabaseFolderImportOptions,
  DatabaseImportRequest,
  DatabaseImportSummary,
  DatabaseOptions,
  DatabaseStatus,
  ExperimentLeaderboardResponse,
  InspectionResponse,
  ModelComparisonResponse,
  ModelInfo,
  NeuralTrainingConfig,
  NeuralTrainingStatus,
  MatrixSynthesisRequest,
  MatrixSynthesisResponse,
  TrainingResult
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as
      | { detail?: string }
      | null;
    throw new ApiError(
      payload?.detail ?? `Request failed (${response.status})`,
      response.status
    );
  }
  return response.json() as Promise<T>;
}

async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      error instanceof Error ? error.message : "Network request failed",
      0
    );
  }
}

export async function generateCurves(
  condition: GenerationCondition
): Promise<GenerationResponse> {
  const response = await apiFetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(condition)
  });
  return readJson<GenerationResponse>(response);
}

export async function inspectFile(
  file: File,
  voltageColumn?: string,
  currentColumn?: string
): Promise<InspectionResponse> {
  const body = new FormData();
  body.append("file", file);
  if (voltageColumn) body.append("voltage_column", voltageColumn);
  if (currentColumn) body.append("current_column", currentColumn);
  const response = await apiFetch("/api/inspect", { method: "POST", body });
  return readJson<InspectionResponse>(response);
}

export async function getModelInfo(): Promise<ModelInfo> {
  const response = await apiFetch("/api/model");
  return readJson<ModelInfo>(response);
}

export async function compareModels(
  condition: GenerationCondition
): Promise<ModelComparisonResponse> {
  const response = await apiFetch("/api/model/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ condition })
  });
  return readJson<ModelComparisonResponse>(response);
}

export async function getModelLeaderboard(): Promise<ExperimentLeaderboardResponse> {
  const response = await apiFetch("/api/model/leaderboard");
  return readJson<ExperimentLeaderboardResponse>(response);
}

export async function getRuntimeConfig(): Promise<{ app_mode: AppMode }> {
  const response = await apiFetch("/api/runtime-config");
  return readJson<{ app_mode: AppMode }>(response);
}

export async function getNeuralTrainingStatus(): Promise<NeuralTrainingStatus> {
  const response = await apiFetch("/api/neural-training/status");
  return readJson<NeuralTrainingStatus>(response);
}

export async function startNeuralTraining(
  config: NeuralTrainingConfig
): Promise<NeuralTrainingStatus> {
  const response = await apiFetch("/api/neural-training/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(config)
  });
  return readJson<NeuralTrainingStatus>(response);
}

export async function checkHealth(): Promise<boolean> {
  try {
    return (await apiFetch("/health")).ok;
  } catch {
    return false;
  }
}

export async function trainModel(
  files: File[],
  components: number
): Promise<TrainingResult> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  body.append("components", String(components));
  const response = await apiFetch("/api/train", { method: "POST", body });
  return readJson<TrainingResult>(response);
}

export async function getExampleFile(path: string): Promise<File> {
  const response = await apiFetch(`/api/examples/${path}`);
  if (!response.ok) {
    throw new ApiError(`Example file not found (${response.status})`, response.status);
  }
  const blob = await response.blob();
  const name = path.split("/").at(-1) ?? "example.csv";
  return new File([blob], name, { type: blob.type || "text/csv" });
}

function appendFilter(params: URLSearchParams, key: string, value: string | undefined) {
  if (value !== undefined && value.trim() !== "") {
    params.set(key, value.trim());
  }
}

export async function getDatabaseStatus(): Promise<DatabaseStatus> {
  const response = await apiFetch("/api/database/status");
  return readJson<DatabaseStatus>(response);
}

export async function importDatabaseSource(
  request: DatabaseImportRequest
): Promise<DatabaseImportSummary> {
  const response = await apiFetch("/api/database/import", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  return readJson<DatabaseImportSummary>(response);
}

export async function importDatabaseFolder(
  files: File[],
  options: DatabaseFolderImportOptions,
  onUploadProgress?: (progressFraction: number) => void
): Promise<DatabaseImportSummary> {
  const relativePaths = files.map((file) => {
    const relativePath = (file as File & { webkitRelativePath?: string }).webkitRelativePath;
    return relativePath && relativePath.trim() ? relativePath : file.name;
  });
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  body.append("relative_paths_json", JSON.stringify(relativePaths));
  body.append("suffixes_json", JSON.stringify(options.suffixes));
  body.append("max_xml_mb", String(options.max_xml_mb));
  body.append("hash_files", String(options.hash_files));
  body.append("replace", String(options.replace));
  if (!onUploadProgress) {
    const response = await apiFetch("/api/database/import-upload", {
      method: "POST",
      body
    });
    return readJson<DatabaseImportSummary>(response);
  }
  return new Promise<DatabaseImportSummary>((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/database/import-upload");
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      onUploadProgress(event.loaded / event.total);
    };
    request.onerror = () => {
      reject(new ApiError("Network request failed", 0));
    };
    request.onload = () => {
      const response = request.response as DatabaseImportSummary | { detail?: string } | null;
      if (request.status >= 200 && request.status < 300) {
        onUploadProgress(1);
        resolve((response ?? {}) as DatabaseImportSummary);
        return;
      }
      reject(
        new ApiError(
          response && typeof response === "object" && "detail" in response
            ? response.detail ?? `Request failed (${request.status})`
            : `Request failed (${request.status})`,
          request.status
        )
      );
    };
    request.send(body);
  });
}

export async function getDatabaseOptions(): Promise<DatabaseOptions> {
  const response = await apiFetch("/api/database/options");
  return readJson<DatabaseOptions>(response);
}

export async function listDatabaseCurves(
  filters: CurveFilters,
  limit: number,
  offset: number,
  orderBy: string,
  signal?: AbortSignal
): Promise<CurveListResponse> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
    order_by: orderBy
  });
  appendFilter(params, "polarity", filters.polarity);
  appendFilter(params, "direction", filters.direction);
  appendFilter(params, "source_kind", filters.source_kind);
  appendFilter(params, "source_search", filters.source_search);
  appendFilter(params, "has_gate_current", filters.has_gate_current);
  appendFilter(params, "hysteresis_available", filters.hysteresis_available);
  appendFilter(
    params,
    "date_from",
    filters.date_from ? `${filters.date_from}T00:00:00` : undefined
  );
  appendFilter(
    params,
    "date_to",
    filters.date_to ? `${filters.date_to}T23:59:59` : undefined
  );
  appendFilter(params, "ion_min", filters.ion_min);
  appendFilter(params, "ion_max", filters.ion_max);
  appendFilter(params, "ioff_min", filters.ioff_min);
  appendFilter(params, "ioff_max", filters.ioff_max);
  appendFilter(params, "ion_ioff_ratio_min", filters.ion_ioff_ratio_min);
  appendFilter(params, "ion_ioff_ratio_max", filters.ion_ioff_ratio_max);
  appendFilter(params, "vth_min", filters.vth_min);
  appendFilter(params, "vth_max", filters.vth_max);
  appendFilter(params, "ss_mv_dec_min", filters.ss_mv_dec_min);
  appendFilter(params, "ss_mv_dec_max", filters.ss_mv_dec_max);
  const response = await apiFetch(`/api/database/curves?${params.toString()}`, { signal });
  return readJson<CurveListResponse>(response);
}

export async function listDatabaseCalendar(
  filters: CurveFilters,
  signal?: AbortSignal
): Promise<CalendarCurveResponse> {
  const params = new URLSearchParams({ limit: "10000" });
  appendFilter(params, "polarity", filters.polarity);
  appendFilter(params, "direction", filters.direction);
  appendFilter(params, "source_kind", filters.source_kind);
  appendFilter(params, "source_search", filters.source_search);
  appendFilter(params, "has_gate_current", filters.has_gate_current);
  appendFilter(params, "hysteresis_available", filters.hysteresis_available);
  appendFilter(
    params,
    "date_from",
    filters.date_from ? `${filters.date_from}T00:00:00` : undefined
  );
  appendFilter(
    params,
    "date_to",
    filters.date_to ? `${filters.date_to}T23:59:59` : undefined
  );
  const response = await apiFetch(`/api/database/calendar?${params.toString()}`, { signal });
  return readJson<CalendarCurveResponse>(response);
}

export async function getDatabaseCurve(
  curveId: string,
  signal?: AbortSignal
): Promise<CurveDetail> {
  const response = await apiFetch(`/api/database/curves/${curveId}`, { signal });
  return readJson<CurveDetail>(response);
}

export async function getDatabaseCurvePreviews(
  curveIds: string[],
  signal?: AbortSignal
): Promise<CurvePreview[]> {
  const response = await apiFetch("/api/database/previews", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ curve_ids: curveIds }),
    signal
  });
  return readJson<CurvePreview[]>(response);
}

function selectionBody(selection: DatabaseSelectionState): string {
  return JSON.stringify({
    curve_ids: selection.allFiltered ? [] : selection.selectedIds,
    filters: selection.allFiltered ? selection.filters : {}
  });
}

export async function analyzeDatabaseSelection(
  selection: DatabaseSelectionState,
  signal?: AbortSignal
): Promise<DatabaseAnalysisResponse> {
  const response = await apiFetch("/api/database/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: selectionBody(selection),
    signal
  });
  return readJson<DatabaseAnalysisResponse>(response);
}

export async function getDatabaseAnalysisStatus(): Promise<DatabaseAnalysisStatus> {
  const response = await apiFetch("/api/database/analyze/status");
  return readJson<DatabaseAnalysisStatus>(response);
}

export async function startDatabaseAnalysis(
  selection: DatabaseSelectionState
): Promise<DatabaseAnalysisStatus> {
  const response = await apiFetch("/api/database/analyze/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: selectionBody(selection)
  });
  return readJson<DatabaseAnalysisStatus>(response);
}

export async function exportDatabaseSelection(
  selection: DatabaseSelectionState,
  options?: DatabaseExportOptions
): Promise<void> {
  const response = await apiFetch("/api/database/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      curve_ids: selection.allFiltered ? [] : selection.selectedIds,
      filters: selection.allFiltered ? selection.filters : {},
      export_options: options
    })
  });
  if (!response.ok) {
    await readJson<never>(response);
    return;
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "devicecurvegen-database-selection.zip";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

export async function synthesizeMatrix(
  request: MatrixSynthesisRequest
): Promise<MatrixSynthesisResponse> {
  const response = await apiFetch("/api/database/matrix-synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  return readJson<MatrixSynthesisResponse>(response);
}

export async function exportMatrixWorkbook(
  request: MatrixSynthesisRequest
): Promise<void> {
  const response = await apiFetch("/api/database/matrix-export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request)
  });
  if (!response.ok) {
    await readJson<never>(response);
    return;
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "devicecurvegen-matrix-output.xlsx";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
