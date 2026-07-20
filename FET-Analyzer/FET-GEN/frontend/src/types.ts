export type TabName = "Generate" | "Import" | "Database" | "Matrix" | "Analysis" | "Models";
export type AppMode = "full" | "generate" | "import" | "analysis" | "training";

export interface GenerationCondition {
  curve_type: "transfer";
  material: string;
  polarity: "n-type" | "p-type";
  vd: number;
  target_ion: number;
  target_ioff: number;
  target_vth: number;
  target_ss_mv_dec: number;
  ss_region_width_v: number;
  hysteresis_v: number;
  noise_sigma_a: number;
  noise_floor_a: number;
  quantization_step_a: number;
  output_noise_gain: number;
  gate_leakage_a: number;
  gate_leakage_v_char: number;
  gate_leakage_exponent: number;
  ion_sigma_fraction: number;
  ioff_sigma_fraction: number;
  vth_sigma_v: number;
  ss_sigma_fraction: number;
  hysteresis_sigma_v: number;
  mobility_cm2_vs: number;
  mobility_sigma_fraction: number;
  contact_resistance_ohm: number;
  contact_resistance_sigma_fraction: number;
  ai_residual_strength: number;
  gate_ai_residual_strength: number;
  diversity: number;
  seed: number;
  voltage_min: number;
  voltage_max: number;
  points: number;
  variants: number;
}

export interface ExtractedFeatures {
  ion: number;
  ioff: number;
  ion_ioff_ratio: number;
  polarity: "n-type" | "p-type" | "bipolar" | "unknown";
  vth: number | null;
  ss_mv_dec: number | null;
  ss_fit_r2: number | null;
  gm_max: number | null;
  vth_gmmax: number | null;
  von: number | null;
  hysteresis_v: number | null;
  leakage_level: number | null;
  noise_log_sigma: number | null;
  ambipolar_strength: number | null;
  current_floor: number | null;
}

export interface ConstraintResult {
  name: string;
  target: number | null;
  measured: number | null;
  tolerance: number | null;
  tolerance_kind: "relative" | "absolute";
  normalized_error: number | null;
  passed: boolean;
}

export interface GeneratedCandidate {
  candidate_id: number;
  seed: number;
  voltage: number[];
  forward_current: number[];
  reverse_current: number[];
  gate_forward_current: number[];
  gate_reverse_current: number[];
  physics_forward_current: number[];
  physics_reverse_current: number[];
  output_drain_voltage: number[];
  output_curves: {
    gate_voltage: number;
    current: number[];
  }[];
  mobility_cm2_vs: number;
  contact_resistance_ohm: number;
  latent_code: number[];
  gate_latent_code: number[];
  features: ExtractedFeatures;
  quality_score: number;
  constraints: ConstraintResult[];
}

export interface GenerationResponse {
  condition: GenerationCondition;
  candidates: GeneratedCandidate[];
  residual_mode: "conditional_vae" | "learned_pca" | "procedural_prior";
  model_name: string;
}

export interface ModelInfo {
  residual_mode: "conditional_vae" | "learned_pca" | "procedural_prior";
  model_name: string;
  checkpoint_path: string | null;
  components: number;
  objective: string | null;
  residual_space: string | null;
  architecture: string | null;
  curves: number | null;
  training_curves: number | null;
  validation_curves: number | null;
  hidden_dim: number | null;
  epochs_completed: number | null;
  best_epoch: number | null;
  train_loss: number | null;
  validation_loss: number | null;
  validation_rmse_decades: number | null;
  validation_mae_decades: number | null;
  validation_p95_error_decades: number | null;
  validation_weighted_rmse_decades: number | null;
  validation_low_current_rmse_decades: number | null;
  validation_subthreshold_rmse_decades: number | null;
  validation_subthreshold_slope_rmse_dec_per_v: number | null;
  validation_gate_rmse_decades: number | null;
  gate_curves: number | null;
  generated_channels: ("Ids" | "Ig")[];
  selection_score: number | null;
  best_trial: number | null;
  tuning_trials: NeuralTrialSummary[];
  feature_eval_curves: number | null;
  feature_vth_mae_v: number | null;
  feature_ss_mae_mv_dec: number | null;
  feature_log_ion_mae_decades: number | null;
  feature_log_ioff_mae_decades: number | null;
  physics_baseline_rmse_decades: number | null;
  physics_baseline_weighted_rmse_decades: number | null;
  physics_baseline_low_current_rmse_decades: number | null;
  physics_baseline_subthreshold_rmse_decades: number | null;
  rmse_improvement_percent: number | null;
  weighted_rmse_improvement_percent: number | null;
  source: string | null;
  training_config: Partial<NeuralTrainingConfig> | null;
  training_history: NeuralEpochMetric[];
  condition_features: string[];
  sample_balance_strategy: string | null;
  rare_curve_groups: number | null;
  load_error: string | null;
}

export interface NeuralEpochMetric {
  trial: number;
  epoch: number;
  train_loss: number;
  validation_loss: number;
  validation_rmse_decades: number;
  validation_weighted_rmse_decades: number | null;
  validation_low_current_rmse_decades: number | null;
  validation_subthreshold_rmse_decades: number | null;
}

export interface NeuralTrainingConfig {
  method:
    | "physics_cvae"
    | "aligned_local_delta_cvae"
    | "latent_pca"
    | "conditional_pca"
    | "threshold_conditional_pca"
    | "local_threshold_conditional_pca"
    | "aligned_local_threshold_conditional_pca"
    | "aligned_local_delta_conditional_pca"
    | "aligned_local_affine_delta_conditional_pca";
  search_strategy: "single" | "quick";
  search_trials: number;
  data_source: "export" | "database";
  dataset_path: string;
  latent_dim: number;
  hidden_dim: number;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  beta: number;
  validation_fraction: number;
  patience: number;
  seed: number;
  max_curves: number | null;
  low_current_weight: number;
  subthreshold_weight: number;
  slope_weight: number;
  gate_loss_weight: number;
  rare_curve_weight: number;
  pca_components: number;
  feature_eval_limit: number;
}

export interface ModelComparisonItem {
  key: string;
  label: string;
  description: string;
  residual_mode: "conditional_vae" | "learned_pca" | "procedural_prior";
  model_name: string;
  checkpoint_path: string | null;
  ai_residual_strength: number;
  gate_ai_residual_strength: number;
  model: ModelInfo;
  candidate: GeneratedCandidate;
  experiment_summary: ExperimentLeaderboardEntry | null;
}

export interface ModelComparisonResponse {
  condition: GenerationCondition;
  items: ModelComparisonItem[];
}

export interface NeuralTrainingResult {
  method:
    | "physics_cvae"
    | "aligned_local_delta_cvae"
    | "latent_pca"
    | "conditional_pca"
    | "threshold_conditional_pca"
    | "local_threshold_conditional_pca"
    | "aligned_local_threshold_conditional_pca"
    | "aligned_local_delta_conditional_pca"
    | "aligned_local_affine_delta_conditional_pca";
  curves: number;
  gate_curves: number;
  generated_channels: ("Ids" | "Ig")[];
  training_curves: number;
  validation_curves: number;
  epochs_completed: number;
  best_epoch: number;
  latent_dim: number;
  hidden_dim: number;
  train_loss: number;
  validation_loss: number;
  validation_rmse_decades: number;
  validation_weighted_rmse_decades: number | null;
  validation_low_current_rmse_decades: number | null;
  validation_subthreshold_rmse_decades: number | null;
  validation_subthreshold_slope_rmse_dec_per_v: number | null;
  validation_gate_rmse_decades: number | null;
  feature_vth_mae_v: number | null;
  feature_ss_mae_mv_dec: number | null;
  selection_score: number | null;
  best_trial: number;
  output: string;
  source: string;
  stopped_early: boolean;
}

export interface NeuralTrialSummary {
  trial: number;
  method:
    | "physics_cvae"
    | "aligned_local_delta_cvae"
    | "latent_pca"
    | "conditional_pca"
    | "threshold_conditional_pca"
    | "local_threshold_conditional_pca"
    | "aligned_local_threshold_conditional_pca"
    | "aligned_local_delta_conditional_pca"
    | "aligned_local_affine_delta_conditional_pca";
  latent_dim: number;
  hidden_dim: number;
  learning_rate: number;
  beta: number;
  validation_rmse_decades: number;
  validation_weighted_rmse_decades: number | null;
  validation_gate_rmse_decades: number | null;
  selection_score: number;
}

export interface NeuralTrainingStatus {
  status: "idle" | "running" | "completed" | "failed";
  stage:
    | "idle"
    | "loading_data"
    | "preparing"
    | "training"
    | "saving"
    | "completed"
    | "failed";
  job_id: string | null;
  message: string;
  started_at: string | null;
  completed_at: string | null;
  elapsed_seconds: number;
  current_epoch: number;
  total_epochs: number;
  progress_fraction: number;
  current_trial: number;
  total_trials: number;
  config: NeuralTrainingConfig;
  history: NeuralEpochMetric[];
  trials: NeuralTrialSummary[];
  result: NeuralTrainingResult | null;
  error: string | null;
}

export interface ExperimentLeaderboardEntry {
  name: string;
  description: string | null;
  method: string;
  architecture: string | null;
  experiment_path: string;
  checkpoint_path: string | null;
  seconds: number | null;
  validation_rmse_decades: number | null;
  validation_weighted_rmse_decades: number | null;
  feature_vth_mae_v: number | null;
  feature_ss_mae_mv_dec: number | null;
  jump_p95_decades: number | null;
  jump_spike_rate: number | null;
  generated_vth_mae_v: number | null;
  generated_ss_mae_mv_dec: number | null;
  canonical_jump_p95_decades: number | null;
  canonical_jump_max_decades: number | null;
}

export interface ExperimentLeaderboardResponse {
  entries: ExperimentLeaderboardEntry[];
  best_jump_entry: ExperimentLeaderboardEntry | null;
  best_canonical_entry: ExperimentLeaderboardEntry | null;
  best_weighted_entry: ExperimentLeaderboardEntry | null;
  report_path: string | null;
  comparison_artifact_url: string | null;
}

export interface DatabaseAnalysisStatus {
  status: "idle" | "running" | "completed" | "failed";
  stage:
    | "idle"
    | "loading_selection"
    | "building_samples"
    | "building_metrics"
    | "computing_correlations"
    | "computing_pca"
    | "finalizing"
    | "completed"
    | "failed";
  job_id: string | null;
  message: string;
  started_at: string | null;
  completed_at: string | null;
  elapsed_seconds: number;
  progress_fraction: number;
  selected_count: number;
  result: DatabaseAnalysisResponse | null;
  error: string | null;
}

export interface CurveSegment {
  direction: "forward" | "reverse" | "single";
  rows: number;
  voltage: number[];
  current: number[];
  aligned_voltage: number[];
  aligned_log_current: number[];
  features: ExtractedFeatures | null;
}

export interface InspectionResponse {
  filename: string;
  delimiter: string;
  columns: string[];
  mapping: {
    voltage: string | null;
    current: string | null;
    confidence: number;
  };
  curve_type: "transfer" | "unknown";
  quality_labels: string[];
  original_rows: number;
  cleaned_rows: number;
  removed_rows: number;
  segments: CurveSegment[];
  preview: Record<string, number | string | null>[];
}

export interface TrainingResult {
  curves: number;
  components: number;
  output: string;
  files_processed: number;
  files_skipped: number;
  skipped: string[];
}

export interface DatabaseStatus {
  configured: boolean;
  database_url: string;
  source_files: number;
  curves: number;
  raw_points: number;
  aligned_points: number;
  gate_points: number;
  aligned_gate_points: number;
  curves_with_ig: number;
  rejected_entries: number;
  polarity_counts: Record<string, number>;
  source_kind_counts: Record<string, number>;
  belonger_counts: Record<string, number>;
}

export interface DatabaseImportRequest {
  source_path: string;
  suffixes: string[];
  max_xml_mb: number;
  hash_files: boolean;
  replace: boolean;
}

export interface DatabaseImportSummary {
  source: string;
  files_discovered: number;
  files_imported: number;
  files_updated: number;
  files_skipped: number;
  accepted_transfer_segments: number;
  rejected_entries: number;
}

export interface RuntimeConfig {
  app_mode: AppMode;
}

export interface DatabaseFolderImportOptions {
  suffixes: string[];
  max_xml_mb: number;
  hash_files: boolean;
  replace: boolean;
}

export interface DatabaseOptions {
  source_kinds: string[];
  polarities: string[];
  directions: string[];
}

export interface CurveSummary {
  curve_id: string;
  direction: string;
  sweep_pair_id: string | null;
  test_time: string | null;
  rows_clean: number;
  voltage_min_v: number;
  voltage_max_v: number;
  ion: number;
  ioff: number;
  ion_ioff_ratio: number;
  log_ratio: number | null;
  polarity: string;
  has_gate_current: boolean;
  vth: number | null;
  ss_mv_dec: number | null;
  gm_max: number | null;
  noise_log_sigma: number | null;
  hysteresis_v: number | null;
  source_path: string;
  modified_at: string | null;
  source_kind: string;
  setup_title: string | null;
  primitive_test: string | null;
  voltage_column: string;
  current_column: string;
  gate_current_column: string | null;
}

export interface CurveListResponse {
  total: number;
  limit: number;
  offset: number;
  items: CurveSummary[];
}

export interface RawPoint {
  point_index: number;
  voltage_v: number;
  current_a: number;
}

export interface GatePoint {
  point_index: number;
  voltage_v: number;
  current_a: number;
}

export interface AlignedPoint {
  point_index: number;
  x_norm: number;
  voltage_v: number;
  log10_abs_id: number;
  abs_id_a: number;
}

export interface AlignedGatePoint {
  point_index: number;
  x_norm: number;
  voltage_v: number;
  log10_abs_ig: number;
  abs_ig_a: number;
}

export interface CurveDetail extends CurveSummary {
  extension: string;
  size_bytes: number;
  table_name: string;
  x_axis_data: string | null;
  classification_reason: string;
  classification_confidence: number;
  columns_json: string[];
  metadata_json: Record<string, string[]>;
  raw_points: RawPoint[];
  gate_points: GatePoint[];
  aligned_points: AlignedPoint[];
  aligned_gate_points: AlignedGatePoint[];
}

export interface CurvePreview {
  curve_id: string;
  polarity: string;
  direction: string;
  source_path: string;
  raw_points: RawPoint[];
}

export interface CurveFilters {
  polarity?: string;
  direction?: string;
  source_kind?: string;
  source_search?: string;
  date_from?: string;
  date_to?: string;
  ion_min?: string;
  ion_max?: string;
  ioff_min?: string;
  ioff_max?: string;
  ion_ioff_ratio_min?: string;
  ion_ioff_ratio_max?: string;
  has_gate_current?: string;
  hysteresis_available?: string;
  vth_min?: string;
  vth_max?: string;
  ss_mv_dec_min?: string;
  ss_mv_dec_max?: string;
}

export interface DatabaseSelectionState {
  selectedIds: string[];
  allFiltered: boolean;
  filters: CurveFilters;
  total: number;
}

export interface DatabaseExportOptions {
  xyxy_curves: boolean;
  curve_metadata: boolean;
  raw_id_points: boolean;
  include_ig: boolean;
  raw_ig_points: boolean;
  aligned_ig_points: boolean;
  analysis_json: boolean;
}

export interface AnalysisMetricStats {
  count: number;
  min: number | null;
  max: number | null;
  mean: number | null;
  median: number | null;
  std: number | null;
}

export interface AnalysisDistributionBin {
  start: number;
  end: number;
  center: number;
  count: number;
}

export interface AnalysisSample {
  curve_id: string;
  polarity: string;
  direction: string;
  source_kind: string;
  source_path: string;
  has_ig: boolean;
  rows_clean: number | null;
  voltage_span: number | null;
  ion: number | null;
  ioff: number | null;
  ion_ioff_ratio: number | null;
  logIon: number | null;
  logIoff: number | null;
  logRatio: number | null;
  vth: number | null;
  ss_mv_dec: number | null;
  gm_max: number | null;
  logGm: number | null;
  noise_log_sigma: number | null;
  ambipolar_strength: number | null;
  hysteresis_v: number | null;
  test_time: string | null;
}

export interface CalendarCurve {
  curve_id: string;
  test_time: string;
  polarity: string;
  direction: string;
  hysteresis_v: number | null;
  sweep_pair_id: string | null;
  vth: number | null;
  ion: number;
  ioff: number;
  source_path: string;
  source_kind: string;
  setup_title: string | null;
}

export interface CalendarCurveResponse {
  items: CalendarCurve[];
  day_counts: Record<string, number>;
  truncated: boolean;
  limit: number;
}

export interface AnalysisCorrelations {
  features: string[];
  matrix: (number | null)[][];
  counts: number[][];
  strongest: {
    x: string;
    y: string;
    r: number;
    count: number;
  }[];
}

export interface AnalysisPcaComponent {
  name: string;
  explained_variance_ratio: number;
  loadings: Record<string, number>;
}

export interface AnalysisPcaPoint {
  curve_id: string;
  polarity: string;
  direction: string;
  source_kind: string;
  scores: number[];
}

export interface AnalysisPca {
  features: string[];
  components: AnalysisPcaComponent[];
  points: AnalysisPcaPoint[];
  sampled: boolean;
}

export interface DatabaseAnalysisResponse {
  count: number;
  selected_mode: "ids" | "filters";
  sample_count: number;
  sample_limit: number;
  samples: AnalysisSample[];
  metrics: Record<string, AnalysisMetricStats>;
  distributions: Record<string, AnalysisDistributionBin[]>;
  correlations: AnalysisCorrelations;
  pca: AnalysisPca;
  categorical: {
    polarity: Record<string, number>;
    direction: Record<string, number>;
    source_kind: Record<string, number>;
    has_ig: Record<string, number>;
    hysteresis: Record<string, number>;
  };
  processing: {
    sources: number;
    raw_points: number;
    aligned_points: number;
    gate_points: number;
    aligned_gate_points?: number;
    curves_with_ig: number;
    rows_clean_mean: number | null;
    voltage_span_min: number | null;
    voltage_span_max: number | null;
  };
}

export type MatrixParameterKey =
  | "target_ion"
  | "target_ioff"
  | "ion_ioff_ratio"
  | "target_vth"
  | "target_ss_mv_dec"
  | "hysteresis_v"
  | "mobility_cm2_vs"
  | "contact_resistance_ohm"
  | "gate_leakage_a";

export type MatrixSynthesisMode = "database" | "generate";
export type MatrixDuplicateMode = "allow" | "avoid" | "generate_on_duplicate";

export interface MatrixParameterMap {
  key: MatrixParameterKey;
  values: number[][];
}

export interface MatrixSynthesisRequest {
  rows: number;
  cols: number;
  mode: MatrixSynthesisMode;
  duplicate_mode: MatrixDuplicateMode;
  parameters: MatrixParameterMap[];
  filters: CurveFilters;
  generation_condition: GenerationCondition;
}

export interface MatrixGeneratedPayload {
  seed: number;
  quality_score: number;
  features: ExtractedFeatures;
  voltage: number[];
  forward_current: number[];
  reverse_current: number[];
  gate_forward_current: number[];
  gate_reverse_current: number[];
}

export interface MatrixAssignment {
  site: string;
  row: number;
  col: number;
  parameters: Partial<Record<MatrixParameterKey, number>>;
  source: "database" | "generated" | "unmatched";
  reason?: string;
  curve_id?: string;
  score?: number;
  score_features?: MatrixParameterKey[];
  reused?: boolean;
  matched?: Partial<Record<MatrixParameterKey, number>> & {
    ion_ioff_ratio?: number;
  };
  polarity?: string;
  direction?: string;
  source_kind?: string;
  source_path?: string;
  generated?: MatrixGeneratedPayload;
}

export interface MatrixSynthesisResponse {
  rows: number;
  cols: number;
  mode: MatrixSynthesisMode;
  duplicate_mode: MatrixDuplicateMode;
  assignments: MatrixAssignment[];
  matched_count: number;
  generated_count: number;
  unmatched_count: number;
  reused_count: number;
}
