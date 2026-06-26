from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GenerationCondition(BaseModel):
    curve_type: Literal["transfer"] = "transfer"
    material: str = "MoS2"
    polarity: Literal["n-type", "p-type"] = "n-type"
    vd: float = 1.0
    target_ion: float = Field(1e-5, gt=0)
    target_ioff: float = Field(1e-15, gt=0)
    target_vth: float = 0.0
    target_ss_mv_dec: float = Field(230.0, ge=20.0, le=5000.0)
    ss_region_width_v: float = Field(0.5, gt=0.0, le=100.0)
    hysteresis_v: float = Field(1.5, ge=0.0)
    noise_sigma_a: float = Field(1e-13, ge=0.0)
    noise_floor_a: float = Field(1e-13, ge=0.0)
    quantization_step_a: float = Field(1e-15, ge=0.0)
    output_noise_gain: float = Field(4.0, ge=0.0, le=50.0)
    gate_leakage_a: float = Field(1e-14, ge=0.0)
    gate_leakage_v_char: float = Field(0.70, gt=0.0)
    gate_leakage_exponent: float = Field(0.80, gt=0.0, le=3.0)
    ion_sigma_fraction: float = Field(0.08, ge=0.0, le=1.0)
    ioff_sigma_fraction: float = Field(0.15, ge=0.0, le=1.0)
    vth_sigma_v: float = Field(0.20, ge=0.0)
    ss_sigma_fraction: float = Field(0.10, ge=0.0, le=1.0)
    hysteresis_sigma_v: float = Field(0.10, ge=0.0)
    mobility_cm2_vs: float = Field(20.0, gt=0.0)
    mobility_sigma_fraction: float = Field(0.10, ge=0.0, le=1.0)
    contact_resistance_ohm: float = Field(1e4, ge=0.0)
    contact_resistance_sigma_fraction: float = Field(0.15, ge=0.0, le=1.0)
    ai_residual_strength: float = Field(0.0, ge=0.0, le=1.0)
    gate_ai_residual_strength: float = Field(0.0, ge=0.0, le=1.0)
    physical_strictness: float = Field(0.0, ge=0.0, le=1.0)
    diversity: float = Field(1.0, ge=0.0, le=1.0)
    seed: int = Field(12345, ge=0)
    voltage_min: float = -20.0
    voltage_max: float = 20.0
    points: int = Field(601, ge=51, le=2001)
    variants: int = Field(10, ge=1, le=32)

    @model_validator(mode="after")
    def validate_ranges(self) -> GenerationCondition:
        if self.target_ion <= self.target_ioff:
            raise ValueError("target_ion must be greater than target_ioff")
        if self.voltage_max <= self.voltage_min:
            raise ValueError("voltage_max must be greater than voltage_min")
        if not self.voltage_min <= self.target_vth <= self.voltage_max:
            raise ValueError("target_vth must lie inside the voltage grid")
        return self


class ExtractedFeatures(BaseModel):
    ion: float
    ioff: float
    ion_ioff_ratio: float
    polarity: Literal["n-type", "p-type", "bipolar", "unknown"] = "unknown"
    vth: float | None = None
    ss_mv_dec: float | None = None
    ss_fit_r2: float | None = None
    gm_max: float | None = None
    vth_gmmax: float | None = None
    von: float | None = None
    hysteresis_v: float | None = None
    leakage_level: float | None = None
    noise_log_sigma: float | None = None
    ambipolar_strength: float | None = None
    current_floor: float | None = None


class ConstraintResult(BaseModel):
    name: str
    target: float | None = None
    measured: float | None = None
    tolerance: float | None = None
    tolerance_kind: Literal["relative", "absolute"] = "relative"
    normalized_error: float | None = None
    passed: bool


class OutputCurve(BaseModel):
    gate_voltage: float
    current: list[float]


class GeneratedCandidate(BaseModel):
    candidate_id: int
    seed: int
    voltage: list[float]
    forward_current: list[float]
    reverse_current: list[float]
    gate_forward_current: list[float]
    gate_reverse_current: list[float]
    physics_forward_current: list[float]
    physics_reverse_current: list[float]
    output_drain_voltage: list[float]
    output_curves: list[OutputCurve]
    mobility_cm2_vs: float
    contact_resistance_ohm: float
    latent_code: list[float]
    gate_latent_code: list[float] = Field(default_factory=list)
    features: ExtractedFeatures
    quality_score: float
    constraints: list[ConstraintResult]


class GenerationResponse(BaseModel):
    condition: GenerationCondition
    candidates: list[GeneratedCandidate]
    residual_mode: Literal["conditional_vae", "learned_pca", "procedural_prior"]
    model_name: str


class ColumnMapping(BaseModel):
    voltage: str | None
    current: str | None
    confidence: float


class CurveSegment(BaseModel):
    direction: Literal["forward", "reverse", "single"]
    rows: int
    voltage: list[float]
    current: list[float]
    aligned_voltage: list[float] = Field(default_factory=list)
    aligned_log_current: list[float] = Field(default_factory=list)
    features: ExtractedFeatures | None = None


class InspectionResponse(BaseModel):
    filename: str
    delimiter: str
    columns: list[str]
    mapping: ColumnMapping
    curve_type: Literal["transfer", "unknown"]
    quality_labels: list[str]
    original_rows: int
    cleaned_rows: int
    removed_rows: int
    segments: list[CurveSegment]
    preview: list[dict[str, float | str | None]]


class ExtractionRequest(BaseModel):
    voltage: list[float] = Field(min_length=4)
    current: list[float] = Field(min_length=4)
    polarity: Literal["n-type", "p-type"] | None = None

    @model_validator(mode="after")
    def matching_lengths(self) -> ExtractionRequest:
        if len(self.voltage) != len(self.current):
            raise ValueError("voltage and current must have the same length")
        return self


class TrainingResult(BaseModel):
    curves: int
    components: int
    output: str
    files_processed: int
    files_skipped: int
    skipped: list[str] = Field(default_factory=list)


class NeuralTrainingResult(BaseModel):
    method: Literal["physics_cvae", "latent_pca"] = "physics_cvae"
    curves: int
    gate_curves: int = 0
    generated_channels: list[Literal["Ids", "Ig"]] = Field(default_factory=lambda: ["Ids"])
    training_curves: int
    validation_curves: int
    epochs_completed: int
    best_epoch: int
    latent_dim: int
    hidden_dim: int
    train_loss: float
    validation_loss: float
    validation_rmse_decades: float
    validation_weighted_rmse_decades: float | None = None
    validation_low_current_rmse_decades: float | None = None
    validation_subthreshold_rmse_decades: float | None = None
    validation_subthreshold_slope_rmse_dec_per_v: float | None = None
    validation_gate_rmse_decades: float | None = None
    feature_vth_mae_v: float | None = None
    feature_ss_mae_mv_dec: float | None = None
    selection_score: float | None = None
    best_trial: int = 1
    output: str
    source: str
    stopped_early: bool


class NeuralEpochMetric(BaseModel):
    trial: int = 1
    epoch: int
    train_loss: float
    validation_loss: float
    validation_rmse_decades: float
    validation_weighted_rmse_decades: float | None = None
    validation_low_current_rmse_decades: float | None = None
    validation_subthreshold_rmse_decades: float | None = None


class NeuralTrialSummary(BaseModel):
    trial: int
    method: Literal["physics_cvae", "latent_pca"]
    latent_dim: int
    hidden_dim: int
    learning_rate: float
    beta: float
    validation_rmse_decades: float
    validation_weighted_rmse_decades: float | None = None
    validation_gate_rmse_decades: float | None = None
    selection_score: float


class NeuralTrainingRequest(BaseModel):
    method: Literal["physics_cvae", "latent_pca"] = "physics_cvae"
    search_strategy: Literal["single", "quick"] = "single"
    search_trials: int = Field(3, ge=1, le=8)
    data_source: Literal["export", "database"] = "export"
    dataset_path: str = "data/b1500_test_dataset_all"
    latent_dim: int = Field(12, ge=1, le=64)
    hidden_dim: int = Field(96, ge=8, le=1024)
    epochs: int = Field(40, ge=1, le=1000)
    batch_size: int = Field(256, ge=2, le=8192)
    learning_rate: float = Field(1e-3, gt=0, le=1.0)
    beta: float = Field(0.005, ge=0, le=1.0)
    validation_fraction: float = Field(0.1, ge=0.01, le=0.5)
    patience: int = Field(7, ge=1, le=200)
    seed: int = Field(12345, ge=0)
    max_curves: int | None = Field(None, ge=10)
    low_current_weight: float = Field(1.5, ge=0, le=20)
    subthreshold_weight: float = Field(2.5, ge=0, le=20)
    slope_weight: float = Field(0.10, ge=0, le=10)
    gate_loss_weight: float = Field(0.5, ge=0, le=10)
    pca_components: int = Field(12, ge=1, le=64)
    feature_eval_limit: int = Field(512, ge=0, le=10000)


class NeuralTrainingStatus(BaseModel):
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    stage: Literal[
        "idle",
        "loading_data",
        "preparing",
        "training",
        "saving",
        "completed",
        "failed",
    ] = "idle"
    job_id: str | None = None
    message: str = "Ready to train"
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    progress_fraction: float = 0.0
    current_trial: int = 0
    total_trials: int = 1
    config: NeuralTrainingRequest = Field(default_factory=NeuralTrainingRequest)
    history: list[NeuralEpochMetric] = Field(default_factory=list)
    trials: list[NeuralTrialSummary] = Field(default_factory=list)
    result: NeuralTrainingResult | None = None
    error: str | None = None


class DatabaseAnalysisStatus(BaseModel):
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    stage: Literal[
        "idle",
        "loading_selection",
        "building_samples",
        "building_metrics",
        "computing_correlations",
        "computing_pca",
        "finalizing",
        "completed",
        "failed",
    ] = "idle"
    job_id: str | None = None
    message: str = "Ready to analyze"
    started_at: str | None = None
    completed_at: str | None = None
    elapsed_seconds: float = 0.0
    progress_fraction: float = 0.0
    selected_count: int = 0
    result: dict[str, Any] | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    residual_mode: Literal["conditional_vae", "learned_pca", "procedural_prior"]
    model_name: str
    checkpoint_path: str | None = None
    components: int = 0
    objective: str | None = None
    residual_space: str | None = None
    architecture: str | None = None
    curves: int | None = None
    training_curves: int | None = None
    validation_curves: int | None = None
    hidden_dim: int | None = None
    epochs_completed: int | None = None
    best_epoch: int | None = None
    train_loss: float | None = None
    validation_loss: float | None = None
    validation_rmse_decades: float | None = None
    validation_mae_decades: float | None = None
    validation_p95_error_decades: float | None = None
    validation_weighted_rmse_decades: float | None = None
    validation_low_current_rmse_decades: float | None = None
    validation_subthreshold_rmse_decades: float | None = None
    validation_subthreshold_slope_rmse_dec_per_v: float | None = None
    validation_gate_rmse_decades: float | None = None
    gate_curves: int | None = None
    generated_channels: list[Literal["Ids", "Ig"]] = Field(default_factory=lambda: ["Ids"])
    selection_score: float | None = None
    best_trial: int | None = None
    tuning_trials: list[NeuralTrialSummary] = Field(default_factory=list)
    feature_eval_curves: int | None = None
    feature_vth_mae_v: float | None = None
    feature_ss_mae_mv_dec: float | None = None
    feature_log_ion_mae_decades: float | None = None
    feature_log_ioff_mae_decades: float | None = None
    physics_baseline_rmse_decades: float | None = None
    physics_baseline_weighted_rmse_decades: float | None = None
    physics_baseline_low_current_rmse_decades: float | None = None
    physics_baseline_subthreshold_rmse_decades: float | None = None
    rmse_improvement_percent: float | None = None
    weighted_rmse_improvement_percent: float | None = None
    source: str | None = None
    training_config: dict[str, Any] | None = None
    training_history: list[NeuralEpochMetric] = Field(default_factory=list)
    load_error: str | None = None
