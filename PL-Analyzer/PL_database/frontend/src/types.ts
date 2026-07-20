export interface ImportOptions {
  include_point_spectra: boolean;
  include_line_scans: boolean;
  include_area_maps: boolean;
  include_series_scans: boolean;
  include_photo_images: boolean;
}

export interface HealthPayload {
  status: string;
  version: string;
  mock_mode: boolean;
  import_backend: string;
  config_path: string;
}

export interface ImportFileSummary {
  source_wip: string;
  source_file_hash?: string | null;
  status: string;
  exported_spectra: number;
  imported_media_assets?: number;
  detected_inventory: Record<string, number>;
  dataset_mode_counts: Record<string, number>;
  media_inventory?: Record<string, number>;
  datasets?: Array<Record<string, unknown>>;
  class_counts: Record<string, number>;
  project_version?: string | null;
  duplicate_of_source?: string | null;
  error_message: string;
}

export interface ImportResultSummary {
  imported_file_count: number;
  imported_dataset_count: number;
  imported_spectra: number;
  imported_media_assets?: number;
  skipped_existing_count: number;
  duplicate_file_count: number;
  failed_file_count: number;
  dataset_mode_counts: Record<string, number>;
  trace_mode_counts: Record<string, number>;
  media_type_counts?: Record<string, number>;
  single_file_summary?: ImportFileSummary | null;
}

export interface UploadTransferProgress {
  stage: "uploading" | "queued";
  percent: number;
  loaded_bytes: number;
  total_bytes: number;
  file_index: number;
  file_count: number;
  current_file_name: string | null;
}

export interface ImportedUploadHistoryItem {
  relative_path: string;
  source_kind: string;
  job_id: string;
  status: string;
  imported_spectra: number;
  imported_media_assets: number;
  ended_at: string;
  message: string | null;
}

export interface ImportJobDetails {
  phase?: string;
  phase_label?: string;
  stage_detail?: string;
  current_file_phase?: string;
  current_file_index?: number;
  overall_progress?: number;
  import_options?: ImportOptions;
  detected_inventory?: Record<string, number>;
  last_file_inventory?: Record<string, number>;
  last_file_dataset_counts?: Record<string, number>;
  media_inventory?: Record<string, number>;
  last_file_media_inventory?: Record<string, number>;
  imported_media_assets?: number;
  last_file_summary?: ImportFileSummary | null;
  dataset_mode_counts?: Record<string, number>;
  result_summary?: ImportResultSummary | null;
  single_file_summary?: ImportFileSummary | null;
  display_input_path?: string;
  source_kind?: string;
  uploaded_files?: number;
  upload_root_name?: string | null;
}

export interface ImportJob {
  job_id: string;
  input_path: string;
  status: string;
  total_files: number;
  processed_files: number;
  exported_spectra: number;
  failed_files: number;
  current_file: string | null;
  start_time: string;
  end_time: string | null;
  log_path: string | null;
  summary_path: string | null;
  message: string | null;
  details?: ImportJobDetails | null;
}

export interface DashboardSummary {
  imported_files: number;
  spectra_count: number;
  failed_imports: number;
  database_size_mb: number;
  type_counts: Record<string, number>;
  acquisition_counts: Record<string, number>;
  measurement_timeline: Array<{
    bucket: string;
    count: number;
    granularity: string;
  }>;
  latest_job: ImportJob | null;
}

export interface ImportProbeReport {
  file_path: string;
  project_version: string | null;
  data_count: number;
  class_counts: Record<string, number>;
  project_metadata: Record<string, unknown>;
  inventory_by_mode: Record<string, number>;
  traces_by_mode: Record<string, number>;
  media_inventory?: Record<string, number>;
  selected_datasets: Array<Record<string, unknown>>;
  selection_limits: Record<string, unknown>;
  notes: string[];
}

export interface SpectrumRow {
  spectrum_id: string;
  representative_spectrum_id?: string | null;
  sample_id: string | null;
  file_path: string;
  source_tree_path: string;
  spectrum_type: string | null;
  acquisition_mode: string | null;
  x_axis_unit: string | null;
  n_points: number;
  source: string | null;
  belonging: string | null;
  substrate: string | null;
  device_id: string | null;
  notes: string | null;
  folder_path: string | null;
  measurement_time: string | null;
  measurement_config: Record<string, unknown>;
  trace_index: number | null;
  trace_count: number | null;
  scan_size_x: number | null;
  scan_size_y: number | null;
  grid_x: number | null;
  grid_y: number | null;
  laser_wavelength: string | null;
  laser_power: string | null;
  integration_time: string | null;
  grating: string | null;
  objective: string | null;
  analysis_material: string | null;
  analysis_family: string | null;
  analysis_status: string | null;
  analysis_method_version: string | null;
  analysis_updated_at: string | null;
  analysis_summary: Record<string, unknown>;
  member_count?: number;
}

export interface TraceSummary {
  spectrum_id: string;
  trace_index: number | null;
  grid_x: number | null;
  grid_y: number | null;
  preview_value: number | null;
}

export interface SpectrumDetail extends SpectrumRow {
  x_axis: number[];
  intensity: number[];
  trace_summaries?: TraceSummary[];
  preview_series?: Array<{
    trace_index: number | null;
    value: number | null;
  }> | null;
  preview_grid?: Array<Array<number | null>> | null;
}

export interface SpectraResponse {
  total: number;
  items: SpectrumRow[];
}

export interface FilterOptions {
  spectrum_type: string[];
  source: string[];
  belonging: string[];
  acquisition_mode: string[];
  substrate: string[];
  x_axis_unit: string[];
  sample_id: string[];
  analysis_material: string[];
  analysis_family: string[];
  analysis_status: string[];
}

export interface SpectrumFilters {
  search?: string;
  spectrum_type?: string;
  source?: string;
  belonging?: string;
  acquisition_mode?: string;
  substrate?: string;
  x_axis_unit?: string;
  sample_id?: string;
  analysis_material?: string;
  analysis_family?: string;
  analysis_status?: string;
  n_points_min?: number;
  n_points_max?: number;
  member_count_min?: number;
  member_count_max?: number;
  trace_count_min?: number;
  trace_count_max?: number;
  scan_size_x_min?: number;
  scan_size_x_max?: number;
  scan_size_y_min?: number;
  scan_size_y_max?: number;
  grid_x_min?: number;
  grid_x_max?: number;
  grid_y_min?: number;
  grid_y_max?: number;
  sort_by?: string;
  sort_dir?: "asc" | "desc";
}

export interface AnalysisOptions {
  baseline_order: number;
  baseline_quantile: number;
  smoothing_window: number;
  smoothing_polyorder: number;
  normalization: "none" | "max" | "area" | "zscore";
  prominence: number;
  height: number | null;
  distance: number | null;
  fit_model: "auto" | "gaussian" | "lorentzian" | "pseudo_voigt";
  max_peaks: number;
  spectrum_family: "auto" | "PL" | "Raman";
  material_hint: string | null;
  method_version: string;
  min_material_confidence: number;
}

export interface PeakFit {
  label?: string;
  amplitude: number;
  center: number;
  width: number;
  fwhm: number;
  area: number;
}

export interface AnalysisPayload {
  x_axis: number[];
  raw_intensity: number[];
  baseline: number[];
  corrected_intensity: number[];
  smoothed_intensity: number[];
  normalized_intensity: number[];
  peaks: Array<{
    index: number;
    position: number;
    height: number;
    prominence: number;
    width_points: number;
  }>;
  fit: {
    model: string;
    fit_curve: number[];
    r2: number | null;
    peaks: PeakFit[];
  };
  metrics: {
    integrated_intensity: number;
    peak_max: number;
    signal_to_noise_ratio: number;
    n_detected_peaks: number;
    r2?: number | null;
    [key: string]: number | null | undefined;
  };
  method_version?: string;
  spectrum_family?: string;
  material?: string;
  material_confidence?: number;
  quality?: {
    score: number | null;
    label: string;
  };
  material_candidates?: Array<{
    material: string;
    confidence: number;
    matched_peaks: number;
  }>;
  axis?: {
    unit: string;
    note: string | null;
  };
  features?: Record<string, number | null>;
}

export interface AnalysisResult {
  spectrum_id: string;
  file_path: string;
  result_id?: string;
  analysis: AnalysisPayload;
}

export interface BatchAnalysisResponse {
  summary: Array<{
    spectrum_id: string;
    integrated_intensity: number;
    peak_max: number;
    signal_to_noise_ratio: number;
    n_detected_peaks: number;
  }>;
  results: AnalysisResult[];
}

export interface MaterialAnalysisJob {
  job_id: string;
  status: "queued" | "running" | "finished" | "failed" | "stopped";
  total: number;
  processed: number;
  failed: number;
  updated: number;
  started_at: string;
  ended_at: string | null;
  current_spectrum_id: string | null;
  message: string;
  method_version: string;
  logs: string[];
  latest_result: AnalysisResult | null;
  queue_window: Array<{
    spectrum_id: string;
    order: number;
    status: "pending" | "running" | "processed" | "failed";
    sample_id: string | null;
    source: string | null;
    spectrum_type: string | null;
    acquisition_mode: string | null;
    sparkline: number[];
    material: string | null;
    family: string | null;
    fit_quality: number | null;
    fit_quality_label: string | null;
    error: string | null;
  }>;
  summary: {
    materials?: Record<string, number>;
    families?: Record<string, number>;
  };
}
