import {
  Activity,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Database,
  Layers3,
  Play,
  RefreshCw,
  RotateCcw,
  Sigma,
  SlidersHorizontal,
  TriangleAlert
} from "lucide-react";
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  getModelLeaderboard,
  getModelInfo,
  getNeuralTrainingStatus,
  startNeuralTraining
} from "../api";
import type {
  ExperimentLeaderboardEntry,
  ExperimentLeaderboardResponse,
  ModelInfo,
  NeuralEpochMetric,
  NeuralTrainingConfig,
  NeuralTrainingStatus
} from "../types";
import { NeuralTrainingChart } from "./NeuralTrainingChart";

const ModelCurveInspector = lazy(() =>
  import("./ModelCurveInspector").then((module) => ({
    default: module.ModelCurveInspector
  }))
);

interface ModelsWorkspaceProps {
  model: ModelInfo | null;
  onModelChange: (model: ModelInfo) => void;
}

const DEFAULT_CONFIG: NeuralTrainingConfig = {
  method: "physics_cvae",
  search_strategy: "single",
  search_trials: 3,
  data_source: "export",
  dataset_path: "data/b1500_test_dataset_all",
  latent_dim: 12,
  hidden_dim: 96,
  epochs: 40,
  batch_size: 256,
  learning_rate: 0.001,
  beta: 0.005,
  validation_fraction: 0.1,
  patience: 7,
  seed: 12345,
  max_curves: null,
  low_current_weight: 1.5,
  subthreshold_weight: 2.5,
  slope_weight: 0.1,
  gate_loss_weight: 0.5,
  rare_curve_weight: 1.35,
  pca_components: 12,
  feature_eval_limit: 512
};

interface NumberFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  help: string;
  disabled: boolean;
  onChange: (value: number) => void;
}

function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  help,
  disabled,
  onChange
}: NumberFieldProps) {
  return (
    <label className="neural-field">
      <span>
        <strong>{label}</strong>
        <small>{help}</small>
      </span>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.currentTarget.value))}
      />
    </label>
  );
}

function formatDuration(seconds: number) {
  const rounded = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(rounded / 60);
  const remaining = rounded % 60;
  return minutes > 0 ? `${minutes}m ${remaining}s` : `${remaining}s`;
}

function metric(value: number | null | undefined, digits = 4) {
  return value === null || value === undefined || !Number.isFinite(value)
    ? "n/a"
    : value.toFixed(digits);
}

function compact(value: number | null | undefined) {
  return value === null || value === undefined
    ? "n/a"
    : value.toLocaleString();
}

function methodLabel(method: string, architecture: string | null) {
  if (architecture === "hybrid_threshold_pca") return "Hybrid";
  if (method === "threshold_conditional_pca") return "Thresh. Cond. PCA";
  if (method === "aligned_local_affine_delta_conditional_pca") return "Affine Delta PCA";
  if (method === "aligned_local_delta_conditional_pca") return "Aligned Delta PCA";
  if (method === "aligned_local_delta_cvae") return "Aligned Delta CVAE";
  if (method === "physics_cvae") return "CVAE";
  if (method === "conditional_pca") return "Cond. PCA";
  if (method === "latent_pca") return "PCA";
  return method;
}

function statusLabel(status: NeuralTrainingStatus | null) {
  if (!status) return "Loading";
  if (status.status === "running") return "Training";
  if (status.status === "completed") return "Completed";
  if (status.status === "failed") return "Failed";
  return "Ready";
}

function EvaluationMetric({
  label,
  value,
  unit,
  note
}: {
  label: string;
  value: string;
  unit?: string;
  note: string;
}) {
  return (
    <div className="evaluation-metric">
      <span>{label}</span>
      <strong>
        {value} {unit ? <small>{unit}</small> : null}
      </strong>
      <p>{note}</p>
    </div>
  );
}

export function ModelsWorkspace({ model, onModelChange }: ModelsWorkspaceProps) {
  const [config, setConfig] = useState<NeuralTrainingConfig>(DEFAULT_CONFIG);
  const [status, setStatus] = useState<NeuralTrainingStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [leaderboard, setLeaderboard] = useState<ExperimentLeaderboardResponse | null>(null);
  const [leaderboardError, setLeaderboardError] = useState<string | null>(null);
  const appliedModelConfig = useRef(false);
  const activatedJob = useRef<string | null>(null);

  useEffect(() => {
    if (appliedModelConfig.current || !model?.training_config) return;
    appliedModelConfig.current = true;
    setConfig((current) => ({
      ...current,
      ...model.training_config,
      data_source: "export",
      dataset_path:
        model.source && model.source !== "database"
          ? model.source
          : current.dataset_path
    }));
  }, [model]);

  useEffect(() => {
    let cancelled = false;

    async function refresh() {
      try {
        const nextStatus = await getNeuralTrainingStatus();
        if (!cancelled) {
          setStatus(nextStatus);
          setError(null);
        }
      } catch (caught) {
        if (!cancelled) {
          setError(
            caught instanceof Error
              ? caught.message
              : "Could not read training status"
          );
        }
      }
    }

    void refresh();
    const interval = window.setInterval(() => void refresh(), 1500);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function refreshLeaderboard() {
      try {
        const next = await getModelLeaderboard();
        if (!cancelled) {
          setLeaderboard(next);
          setLeaderboardError(null);
        }
      } catch (caught) {
        if (!cancelled) {
          setLeaderboardError(
            caught instanceof Error ? caught.message : "Could not load model leaderboard"
          );
        }
      }
    }

    void refreshLeaderboard();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (
      status?.status !== "completed" ||
      !status.job_id ||
      activatedJob.current === status.job_id
    ) {
      return;
    }
    activatedJob.current = status.job_id;
    void Promise.all([getModelInfo(), getModelLeaderboard()])
      .then(([nextModel, nextLeaderboard]) => {
        onModelChange(nextModel);
        setLeaderboard(nextLeaderboard);
        setLeaderboardError(null);
      })
      .catch((caught) => {
        setError(
          caught instanceof Error ? caught.message : "Could not refresh active model"
        );
      });
  }, [onModelChange, status?.job_id, status?.status]);

  const running = status?.status === "running";
  const rawHistory = useMemo<NeuralEpochMetric[]>(
    () =>
      status?.history && status.history.length > 0
        ? status.history
        : model?.training_history ?? [],
    [model?.training_history, status?.history]
  );
  const activeTrial = status?.current_trial || model?.best_trial || 1;
  const history = useMemo(
    () => rawHistory.filter((item) => (item.trial ?? 1) === activeTrial),
    [activeTrial, rawHistory]
  );
  const trialSummaries =
    status?.trials && status.trials.length > 0
      ? status.trials
      : model?.tuning_trials ?? [];
  const latest = history.at(-1);
  const bestMetric = history.reduce<NeuralEpochMetric | null>(
    (best, item) =>
      best === null || item.validation_loss < best.validation_loss ? item : best,
    null
  );
  const progress = Math.round((status?.progress_fraction ?? 0) * 100);
  const trainCurves =
    status?.result?.training_curves ?? model?.training_curves ?? null;
  const validationCurves =
    status?.result?.validation_curves ?? model?.validation_curves ?? null;
  const totalCurves = status?.result?.curves ?? model?.curves ?? null;
  const validationLoss =
    status?.result?.validation_loss ?? model?.validation_loss ?? null;
  const validationRmse =
    status?.result?.validation_rmse_decades ??
    model?.validation_rmse_decades ??
    null;
  const weightedRmse =
    status?.result?.validation_weighted_rmse_decades ??
    model?.validation_weighted_rmse_decades ??
    null;
  const lowCurrentRmse =
    status?.result?.validation_low_current_rmse_decades ??
    model?.validation_low_current_rmse_decades ??
    null;
  const subthresholdRmse =
    status?.result?.validation_subthreshold_rmse_decades ??
    model?.validation_subthreshold_rmse_decades ??
    null;
  const improvement =
    model?.weighted_rmse_improvement_percent ??
    model?.rmse_improvement_percent ??
    null;
  const gateRmse =
    status?.result?.validation_gate_rmse_decades ??
    model?.validation_gate_rmse_decades ??
    null;
  const gateCurves = status?.result?.gate_curves ?? model?.gate_curves ?? 0;
  const generatedChannels =
    status?.result?.generated_channels ?? model?.generated_channels ?? ["Ids"];
  const activeMethod =
    model?.architecture === "latent_pca"
      ? "latent_pca"
      : model?.architecture === "hybrid_threshold_pca"
        ? "conditional_pca"
      : model?.architecture === "aligned_local_affine_delta_conditional_pca"
        ? "aligned_local_affine_delta_conditional_pca"
      : model?.architecture === "aligned_local_delta_conditional_pca"
        ? "aligned_local_delta_conditional_pca"
      : model?.architecture === "aligned_local_delta_conditional_vae_residual_skip"
        ? "aligned_local_delta_cvae"
      : model?.architecture === "threshold_conditional_pca"
        ? "threshold_conditional_pca"
      : model?.architecture === "conditional_pca"
        ? "conditional_pca"
      : model?.architecture === "conditional_vae" ||
          model?.architecture === "conditional_vae_residual_skip"
        ? "physics_cvae"
        : config.method;
  const leaderboardEntries = leaderboard?.entries ?? [];
  const bestJumpEntry = leaderboard?.best_jump_entry ?? leaderboardEntries[0] ?? null;
  const bestCanonicalEntry = useMemo<ExperimentLeaderboardEntry | null>(
    () =>
      leaderboard?.best_canonical_entry ??
      leaderboardEntries.reduce<ExperimentLeaderboardEntry | null>((best, entry) => {
        if (entry.canonical_jump_max_decades === null) return best;
        if (best === null || best.canonical_jump_max_decades === null) return entry;
        if (entry.canonical_jump_max_decades < best.canonical_jump_max_decades) {
          return entry;
        }
        if (
          entry.canonical_jump_max_decades === best.canonical_jump_max_decades &&
          (entry.generated_vth_mae_v ?? Number.POSITIVE_INFINITY) <
            (best.generated_vth_mae_v ?? Number.POSITIVE_INFINITY)
        ) {
          return entry;
        }
        return best;
      }, null),
    [leaderboard?.best_canonical_entry, leaderboardEntries]
  );
  const bestWeightedEntry = useMemo<ExperimentLeaderboardEntry | null>(
    () =>
      leaderboard?.best_weighted_entry ??
      leaderboardEntries.reduce<ExperimentLeaderboardEntry | null>((best, entry) => {
        if (entry.validation_weighted_rmse_decades === null) return best;
        if (best === null || best.validation_weighted_rmse_decades === null) return entry;
        return entry.validation_weighted_rmse_decades <
          best.validation_weighted_rmse_decades
          ? entry
          : best;
      }, null),
    [leaderboard?.best_weighted_entry, leaderboardEntries]
  );

  function patchConfig(patch: Partial<NeuralTrainingConfig>) {
    setConfig((current) => ({ ...current, ...patch }));
  }

  async function handleStart() {
    setStarting(true);
    setError(null);
    try {
      const nextStatus = await startNeuralTraining(config);
      setStatus(nextStatus);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not start training");
    } finally {
      setStarting(false);
    }
  }

  return (
    <main className="models-workspace">
      <section className="models-heading">
        <div>
          <h1>Generative network training</h1>
          <p>
            Train and compare physics-aware CVAE and latent-PCA generators for
            joint log10(Ids) and log10(Ig) transfer-curve morphology.
          </p>
        </div>
        <div className={`training-state training-state-${status?.status ?? "idle"}`}>
          {running ? <RefreshCw size={15} className="spin" /> : <Activity size={15} />}
          <span>{statusLabel(status)}</span>
        </div>
      </section>

      {error ? (
        <div className="error-banner model-error" role="alert">
          {error}
        </div>
      ) : null}
      {status?.error ? (
        <div className="error-banner model-error" role="alert">
          {status.error}
        </div>
      ) : null}

      <section className="active-model-strip">
        <div className="active-model-name">
          <BrainCircuit size={25} />
          <div>
            <span>Active model</span>
            <strong>{model?.model_name ?? "Loading model"}</strong>
          </div>
        </div>
        <dl>
          <div>
            <dt>Objective</dt>
            <dd>{model?.objective ?? model?.architecture ?? "n/a"}</dd>
          </div>
          <div>
            <dt>Residual space</dt>
            <dd>{model?.residual_space ?? "log10 Id"}</dd>
          </div>
          <div>
            <dt>Latent</dt>
            <dd>{model?.components ?? "n/a"} dimensions</dd>
          </div>
          <div>
            <dt>Checkpoint</dt>
            <dd title={model?.checkpoint_path ?? ""}>
              {model?.checkpoint_path ?? "Not loaded"}
            </dd>
          </div>
        </dl>
      </section>

      <div className="models-dashboard">
        <aside className="training-parameter-panel">
          <div className="panel-section-heading">
            <SlidersHorizontal size={17} />
            <div>
              <h2>Training parameters</h2>
              <p>Changes apply to the next retraining job.</p>
            </div>
          </div>

          <label className="neural-field neural-field-wide">
            <span>
              <strong>Data source</strong>
              <small>Aligned curves used to construct residual targets.</small>
            </span>
            <select
              value={config.data_source}
              disabled={running}
              onChange={(event) =>
                patchConfig({
                  data_source: event.currentTarget.value as "export" | "database"
                })
              }
            >
              <option value="export">Aligned NPZ export</option>
              <option value="database">MySQL database</option>
            </select>
          </label>

          <label className="neural-field neural-field-wide">
            <span>
              <strong>Generation method</strong>
              <small>Choose between the stochastic CVAE, the unconditional PCA baseline, and threshold-aware conditioned PCA variants.</small>
            </span>
            <select
              value={config.method}
              disabled={running}
              onChange={(event) =>
                patchConfig({
                  method: event.currentTarget.value as
                    | "physics_cvae"
                    | "aligned_local_delta_cvae"
                    | "latent_pca"
                    | "conditional_pca"
                    | "threshold_conditional_pca"
                    | "aligned_local_delta_conditional_pca"
                    | "aligned_local_affine_delta_conditional_pca"
                })
              }
            >
              <option value="physics_cvae">Physics-aware conditional VAE</option>
              <option value="aligned_local_delta_cvae">
                Threshold-aligned delta CVAE
              </option>
              <option value="conditional_pca">Conditioned latent PCA</option>
              <option value="threshold_conditional_pca">
                Threshold-focused conditioned PCA
              </option>
              <option value="aligned_local_delta_conditional_pca">
                Threshold-aligned delta PCA
              </option>
              <option value="aligned_local_affine_delta_conditional_pca">
                Threshold-aligned affine-delta PCA
              </option>
              <option value="latent_pca">Physics residual + latent PCA</option>
            </select>
          </label>

          <label className="neural-field neural-field-wide">
            <span>
              <strong>Parameter search</strong>
              <small>Quick search trains several variants and activates the best score.</small>
            </span>
            <select
              value={config.search_strategy}
              disabled={running}
              onChange={(event) =>
                patchConfig({
                  search_strategy: event.currentTarget.value as "single" | "quick"
                })
              }
            >
              <option value="single">Single configuration</option>
              <option value="quick">Quick multi-trial search</option>
            </select>
          </label>

          {config.search_strategy === "quick" ? (
            <NumberField
              label="Search trials"
              help="Sequential configurations compared by joint Ids/Ig score."
              value={config.search_trials}
              min={2}
              max={8}
              disabled={running}
              onChange={(search_trials) => patchConfig({ search_trials })}
            />
          ) : null}

          <label className="neural-field neural-field-path">
            <span>
              <strong>Dataset path</strong>
              <small>Directory containing aligned_curves.npz and curves.csv.</small>
            </span>
            <input
              value={config.dataset_path}
              disabled={running || config.data_source === "database"}
              onChange={(event) => patchConfig({ dataset_path: event.currentTarget.value })}
            />
          </label>

          <div className="parameter-group-label">Model capacity</div>
          {config.method === "physics_cvae" || config.method === "aligned_local_delta_cvae" ? (
            <>
              <NumberField
                label="Latent dimensions"
                help="Capacity of the sampled morphology code."
                value={config.latent_dim}
                min={1}
                max={64}
                disabled={running}
                onChange={(latent_dim) => patchConfig({ latent_dim })}
              />
              <NumberField
                label="Hidden width"
                help="Encoder and decoder hidden layer width."
                value={config.hidden_dim}
                min={8}
                max={1024}
                step={8}
                disabled={running}
                onChange={(hidden_dim) => patchConfig({ hidden_dim })}
              />
            </>
          ) : (
            <NumberField
              label="PCA components"
              help={
                config.method === "conditional_pca" ||
                config.method === "threshold_conditional_pca" ||
                config.method === "aligned_local_delta_conditional_pca" ||
                config.method === "aligned_local_affine_delta_conditional_pca"
                  ? "Latent basis size before condition-to-latent regression."
                  : "Latent basis size for the stable residual baseline."
              }
              value={config.pca_components}
              min={1}
              max={64}
              disabled={running}
              onChange={(pca_components) => patchConfig({ pca_components })}
            />
          )}

          <div className="parameter-group-label">Log-space emphasis</div>
          <NumberField
            label="Low-current weight"
            help="Extra loss weight near the Ioff side of each curve."
            value={config.low_current_weight}
            min={0}
            max={20}
            step={0.1}
            disabled={running}
            onChange={(low_current_weight) => patchConfig({ low_current_weight })}
          />
          <NumberField
            label="Subthreshold weight"
            help="Extra loss weight around the log-slope transition."
            value={config.subthreshold_weight}
            min={0}
            max={20}
            step={0.1}
            disabled={running}
            onChange={(subthreshold_weight) => patchConfig({ subthreshold_weight })}
          />
          <NumberField
            label="Slope penalty"
            help="Derivative loss on adjacent log-residual points."
            value={config.slope_weight}
            min={0}
            max={10}
            step={0.01}
            disabled={running}
            onChange={(slope_weight) => patchConfig({ slope_weight })}
          />
          <NumberField
            label="Ig loss weight"
            help="Relative importance of learned gate-current morphology."
            value={config.gate_loss_weight}
            min={0}
            max={10}
            step={0.1}
            disabled={running}
            onChange={(gate_loss_weight) => patchConfig({ gate_loss_weight })}
          />
          <NumberField
            label="Rare-curve boost"
            help="Upweight sparse device regimes so the model learns uncommon but real shapes."
            value={config.rare_curve_weight}
            min={1}
            max={10}
            step={0.05}
            disabled={running}
            onChange={(rare_curve_weight) => patchConfig({ rare_curve_weight })}
          />

          {config.method === "physics_cvae" || config.method === "aligned_local_delta_cvae" ? (
            <>
              <div className="parameter-group-label">Optimization</div>
              <NumberField
                label="Epochs"
                help="Maximum complete passes over the training split."
                value={config.epochs}
                min={1}
                max={1000}
                disabled={running}
                onChange={(epochs) => patchConfig({ epochs })}
              />
              <NumberField
                label="Batch size"
                help="Curves processed per Adam update."
                value={config.batch_size}
                min={2}
                max={8192}
                step={16}
                disabled={running}
                onChange={(batch_size) => patchConfig({ batch_size })}
              />
              <NumberField
                label="Learning rate"
                help="Adam optimizer step size."
                value={config.learning_rate}
                min={0.000001}
                max={1}
                step={0.0001}
                disabled={running}
                onChange={(learning_rate) => patchConfig({ learning_rate })}
              />
              <NumberField
                label="KL beta"
                help="Regularization weight after warm-up."
                value={config.beta}
                min={0}
                max={1}
                step={0.001}
                disabled={running}
                onChange={(beta) => patchConfig({ beta })}
              />
            </>
          ) : config.method === "conditional_pca" ||
            config.method === "threshold_conditional_pca" ||
            config.method === "aligned_local_delta_conditional_pca" ||
            config.method === "aligned_local_affine_delta_conditional_pca" ? (
            <>
              <div className="parameter-group-label">Condition regression</div>
              <NumberField
                label="Ridge beta"
                help="Regularization on the condition-to-latent regression weights."
                value={config.beta}
                min={0}
                max={1}
                step={0.001}
                disabled={running}
                onChange={(beta) => patchConfig({ beta })}
              />
            </>
          ) : null}

          <div className="parameter-group-label">Validation and control</div>
          <NumberField
            label="Validation split"
            help="Source-grouped holdout fraction."
            value={config.validation_fraction}
            min={0.01}
            max={0.5}
            step={0.01}
            disabled={running}
            onChange={(validation_fraction) => patchConfig({ validation_fraction })}
          />
          {config.method === "physics_cvae" || config.method === "aligned_local_delta_cvae" ? (
            <NumberField
              label="Early stopping"
              help="Epochs without validation improvement."
              value={config.patience}
              min={1}
              max={200}
              disabled={running}
              onChange={(patience) => patchConfig({ patience })}
            />
          ) : null}
          <NumberField
            label="Feature eval curves"
            help="Validation curves used for SS/Vth feature metrics."
            value={config.feature_eval_limit}
            min={0}
            max={10000}
            step={64}
            disabled={running}
            onChange={(feature_eval_limit) => patchConfig({ feature_eval_limit })}
          />
          <NumberField
            label="Random seed"
            help="Controls split, initialization, and batches."
            value={config.seed}
            min={0}
            max={2147483647}
            disabled={running}
            onChange={(seed) => patchConfig({ seed })}
          />
          <label className="neural-field">
            <span>
              <strong>Max curves</strong>
              <small>Leave empty to use the full accepted dataset.</small>
            </span>
            <input
              type="number"
              min="10"
              value={config.max_curves ?? ""}
              disabled={running}
              placeholder="All"
              onChange={(event) =>
                patchConfig({
                  max_curves:
                    event.currentTarget.value === ""
                      ? null
                      : Number(event.currentTarget.value)
                })
              }
            />
          </label>

          <div className="training-actions">
            <button
              type="button"
              className="button secondary"
              disabled={running}
              onClick={() => setConfig(DEFAULT_CONFIG)}
            >
              <RotateCcw size={15} />
              Reset
            </button>
            <button
              type="button"
              className="button primary"
              disabled={running || starting}
              onClick={() => void handleStart()}
            >
              {running || starting ? (
                <RefreshCw size={15} className="spin" />
              ) : (
                <Play size={15} />
              )}
              {running
                ? "Training"
                : config.search_strategy === "quick"
                  ? "Run search"
                  : "Retrain"}
            </button>
          </div>
        </aside>

        <div className="training-main">
          <section className="training-progress-panel">
            <div className="panel-title-row">
              <div>
                <h2>Training progress</h2>
                <p>{status?.message ?? "Reading training service state"}</p>
              </div>
              <div className="epoch-counter">
                <span>Epoch</span>
                <strong>{status?.current_epoch ?? model?.epochs_completed ?? 0}</strong>
                <small>/ {status?.total_epochs || model?.epochs_completed || "n/a"}</small>
              </div>
            </div>
            <div className="training-progress-track" aria-label={`${progress}% complete`}>
              <span style={{ width: `${progress}%` }} />
            </div>
            <div className="training-progress-facts">
              <span>
                <Clock3 size={13} />
                Elapsed {formatDuration(status?.elapsed_seconds ?? 0)}
              </span>
              <span>Stage: {status?.stage.replaceAll("_", " ") ?? "idle"}</span>
              <span>
                Trial {status?.current_trial || model?.best_trial || 1} /{" "}
                {status?.total_trials || Math.max(trialSummaries.length, 1)}
              </span>
              <span>Best epoch: {bestMetric?.epoch ?? model?.best_epoch ?? "n/a"}</span>
              <span>
                Weighted RMSE:{" "}
                {metric(
                  latest?.validation_weighted_rmse_decades ?? weightedRmse,
                  4
                )}{" "}
                dec
              </span>
            </div>
          </section>

          <section className="loss-chart-panel">
            <div className="panel-title-row">
                <div>
                  <h2>Optimization history</h2>
                  <p>
                    {activeMethod === "physics_cvae"
                      ? "Weighted log-residual VAE objective on grouped source splits."
                      : activeMethod === "threshold_conditional_pca"
                        ? "Single-step threshold-focused conditioned PCA fit on grouped source splits."
                      : activeMethod === "conditional_pca"
                        ? "Single-step conditioned PCA fit on grouped source splits."
                        : "Single-step latent PCA reconstruction on grouped source splits."}
                  </p>
                </div>
              <div className="chart-current-values">
                <span>
                  Train <strong>{metric(latest?.train_loss ?? model?.train_loss, 4)}</strong>
                </span>
                <span>
                  Validation{" "}
                  <strong>{metric(latest?.validation_loss ?? validationLoss, 4)}</strong>
                </span>
              </div>
            </div>
            <NeuralTrainingChart history={history} />
          </section>

          {trialSummaries.length > 1 ? (
            <section className="trial-comparison-panel">
              <div className="panel-title-row">
                <div>
                  <h2>Parameter-search comparison</h2>
                  <p>
                    Lowest joint score wins: weighted Ids RMSE plus the configured
                    Ig error contribution.
                  </p>
                </div>
                <span className="trial-best-label">
                  Best trial {status?.result?.best_trial ?? model?.best_trial ?? "n/a"}
                </span>
              </div>
              <div className="trial-table-wrap">
                <table className="trial-table">
                  <thead>
                    <tr>
                      <th>Trial</th>
                      <th>Method</th>
                      <th>Latent</th>
                      <th>Hidden</th>
                      <th>Learning rate</th>
                      <th>Ids weighted RMSE</th>
                      <th>Ig RMSE</th>
                      <th>Joint score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trialSummaries.map((trial) => (
                      <tr
                        key={trial.trial}
                        className={
                          trial.trial ===
                          (status?.result?.best_trial ?? model?.best_trial)
                            ? "best"
                            : undefined
                        }
                      >
                        <td>{trial.trial}</td>
                        <td>
                          {trial.method === "physics_cvae"
                            ? "CVAE"
                            : trial.method === "threshold_conditional_pca"
                              ? "Thresh. Cond. PCA"
                            : trial.method === "conditional_pca"
                              ? "Cond. PCA"
                              : "PCA"}
                        </td>
                        <td>{trial.latent_dim}</td>
                        <td>{trial.hidden_dim || "—"}</td>
                        <td>{trial.learning_rate.toExponential(2)}</td>
                        <td>
                          {metric(
                            trial.validation_weighted_rmse_decades ??
                              trial.validation_rmse_decades,
                            4
                          )}
                        </td>
                        <td>{metric(trial.validation_gate_rmse_decades, 4)}</td>
                        <td>{metric(trial.selection_score, 4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}

          <section className="evaluation-panel">
            <div className="panel-title-row">
              <div>
                <h2>Residual and feature evaluation</h2>
                <p>
                  Metrics are computed in log10-current space, with separate
                  low-current and subthreshold checks.
                </p>
              </div>
              {improvement !== null && improvement !== undefined ? (
                <span className="evaluation-improvement">
                  <CheckCircle2 size={14} />
                  {improvement.toFixed(1)}% below physics baseline
                </span>
              ) : null}
            </div>
            <div className="evaluation-grid">
              <EvaluationMetric
                label="Global log RMSE"
                value={metric(validationRmse, 4)}
                unit="dec"
                note="Uniform holdout point error."
              />
              <EvaluationMetric
                label="Weighted log RMSE"
                value={metric(weightedRmse, 4)}
                unit="dec"
                note="Training-aligned weighted error."
              />
              <EvaluationMetric
                label="Low-current RMSE"
                value={metric(lowCurrentRmse, 4)}
                unit="dec"
                note="Ioff-side reconstruction error."
              />
              <EvaluationMetric
                label="Subthreshold RMSE"
                value={metric(subthresholdRmse, 4)}
                unit="dec"
                note="Log-slope transition error."
              />
              <EvaluationMetric
                label="Slope RMSE"
                value={metric(model?.validation_subthreshold_slope_rmse_dec_per_v, 4)}
                unit="dec/V"
                note="Derivative error inside subthreshold."
              />
              <EvaluationMetric
                label="SS MAE"
                value={metric(model?.feature_ss_mae_mv_dec, 1)}
                unit="mV/dec"
                note="Feature-level subthreshold swing error."
              />
              <EvaluationMetric
                label="Vth MAE"
                value={metric(model?.feature_vth_mae_v, 3)}
                unit="V"
                note="Feature extraction error after reconstruction."
              />
              <EvaluationMetric
                label="Ioff MAE"
                value={metric(model?.feature_log_ioff_mae_decades, 4)}
                unit="dec"
                note="Log off-current feature error."
              />
              <EvaluationMetric
                label="Ig shape RMSE"
                value={metric(gateRmse, 4)}
                unit="dec"
                note="Holdout error for centered gate-current morphology."
              />
            </div>
          </section>

          <section className="trial-comparison-panel">
            <div className="panel-title-row">
              <div>
                <h2>Experiment leaderboard</h2>
                <p>
                  Real database-backed attempts ranked by threshold-jump behavior,
                  so we can compare multiple model families side by side.
                </p>
              </div>
              <span className="trial-best-label">
                {leaderboard?.report_path ? "Report available" : "Top experiments"}
              </span>
            </div>
            {leaderboardError ? (
              <div className="error-banner model-error" role="alert">
                {leaderboardError}
              </div>
            ) : null}
            {leaderboardEntries.length > 0 ? (
              <div className="evaluation-grid">
                <EvaluationMetric
                  label="Best held-out jump"
                  value={metric(bestJumpEntry?.jump_p95_decades, 4)}
                  unit="dec"
                  note={
                    bestJumpEntry
                      ? `${bestJumpEntry.name} • ${methodLabel(
                          bestJumpEntry.method,
                          bestJumpEntry.architecture
                        )}`
                      : "n/a"
                  }
                />
                <EvaluationMetric
                  label="Best canonical 100% AI"
                  value={metric(bestCanonicalEntry?.canonical_jump_max_decades, 4)}
                  unit="dec"
                  note={
                    bestCanonicalEntry
                      ? `${bestCanonicalEntry.name} • ${methodLabel(
                          bestCanonicalEntry.method,
                          bestCanonicalEntry.architecture
                        )}`
                      : "n/a"
                  }
                />
                <EvaluationMetric
                  label="Best weighted RMSE"
                  value={metric(bestWeightedEntry?.validation_weighted_rmse_decades, 4)}
                  unit="dec"
                  note={
                    bestWeightedEntry
                      ? `${bestWeightedEntry.name} • ${methodLabel(
                          bestWeightedEntry.method,
                          bestWeightedEntry.architecture
                        )}`
                      : "n/a"
                  }
                />
              </div>
            ) : null}
            {leaderboard?.comparison_artifact_url ? (
              <div className="leaderboard-artifact-panel">
                <div className="leaderboard-artifact-copy">
                  <strong>Canonical 100% AI snapshot</strong>
                  <p>
                    This fixed SVG compares the strongest current candidates under
                    one identical condition, so the Vth-region jump is visible at
                    a glance before reading the table.
                  </p>
                  <a
                    className="button secondary compact"
                    href={leaderboard.comparison_artifact_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Open SVG
                  </a>
                </div>
                <img
                  src={leaderboard.comparison_artifact_url}
                  alt="Canonical 100% AI model comparison"
                />
              </div>
            ) : null}
            <div className="trial-table-wrap">
              <table className="trial-table">
                <thead>
                  <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>Method</th>
                    <th>Jump P95</th>
                    <th>Canon. max</th>
                    <th>Spike rate</th>
                    <th>Gen. Vth MAE</th>
                    <th>Gen. SS MAE</th>
                    <th>Weighted RMSE</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboardEntries.length === 0 ? (
                    <tr>
                      <td colSpan={9}>No experiment summaries discovered yet.</td>
                    </tr>
                  ) : (
                    leaderboardEntries.map((entry, index) => {
                      const active = entry.checkpoint_path === model?.checkpoint_path;
                      return (
                        <tr key={`${entry.experiment_path}-${entry.name}`} className={active ? "best" : undefined}>
                          <td>{index + 1}</td>
                          <td>
                            <strong>{entry.name}</strong>
                            <br />
                            <small>{entry.description ?? entry.experiment_path}</small>
                          </td>
                          <td>{methodLabel(entry.method, entry.architecture)}</td>
                          <td>{metric(entry.jump_p95_decades, 4)}</td>
                          <td>{metric(entry.canonical_jump_max_decades, 4)}</td>
                          <td>{metric(entry.jump_spike_rate, 4)}</td>
                          <td>{metric(entry.generated_vth_mae_v, 3)}</td>
                          <td>{metric(entry.generated_ss_mae_mv_dec, 1)}</td>
                          <td>{metric(entry.validation_weighted_rmse_decades, 4)}</td>
                        </tr>
                      );
                    })
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <div className="training-detail-grid">
            <section className="dataset-split-panel">
              <div className="panel-section-heading">
                <Database size={17} />
                <div>
                  <h2>Dataset split</h2>
                  <p>Grouped by source file to prevent leakage.</p>
                </div>
              </div>
              <div className="dataset-total">
                <strong>{compact(totalCurves)}</strong>
                <span>accepted curves</span>
              </div>
              <div className="dataset-split-bar">
                <span
                  className="dataset-train-bar"
                  style={{
                    width:
                      totalCurves && trainCurves
                        ? `${(trainCurves / totalCurves) * 100}%`
                        : "90%"
                  }}
                />
                <span className="dataset-validation-bar" />
              </div>
              <dl className="dataset-split-facts">
                <div>
                  <dt>Train</dt>
                  <dd>{compact(trainCurves)}</dd>
                </div>
                <div>
                  <dt>Validation</dt>
                  <dd>{compact(validationCurves)}</dd>
                </div>
                <div>
                  <dt>Feature eval</dt>
                  <dd>{compact(model?.feature_eval_curves)}</dd>
                </div>
                <div>
                  <dt>Ig curves</dt>
                  <dd>{compact(gateCurves)}</dd>
                </div>
                <div>
                  <dt>Channels</dt>
                  <dd>{generatedChannels.join(" + ")}</dd>
                </div>
              </dl>
              <code title={model?.source ?? ""}>{model?.source ?? config.dataset_path}</code>
            </section>

            <section className="architecture-panel">
              <div className="panel-section-heading">
                <Layers3 size={17} />
                <div>
                  <h2>Network structure</h2>
                  <p>
                    {activeMethod === "physics_cvae"
                      ? "Physics-residual conditional variational autoencoder with a direct condition skip path."
                      : model?.architecture === "hybrid_threshold_pca"
                        ? "Stable PCA backbone with a threshold-local conditioned PCA guide blended around Vth."
                      : activeMethod === "threshold_conditional_pca"
                        ? "Conditioned PCA trained in a threshold-emphasized residual space so more latent capacity is spent around Vth."
                      : activeMethod === "conditional_pca"
                        ? "Physics-residual PCA basis with a learned condition-to-latent regressor."
                        : "Physics-residual latent PCA baseline."}
                  </p>
                </div>
              </div>
              <div className="architecture-flow">
                <div>
                  <span>Channels</span>
                  <strong>{generatedChannels.length}</strong>
                  <small>{generatedChannels.join(" + ")}</small>
                </div>
                <b aria-hidden="true">+</b>
              <div>
                <span>Condition</span>
                  <strong>{model?.condition_features.length || 9}</strong>
                  <small>conditioning features</small>
                </div>
                <b aria-hidden="true">-&gt;</b>
                {activeMethod === "physics_cvae" ? (
                  <>
                    <div>
                      <span>Encoder</span>
                      <strong>{model?.hidden_dim ?? config.hidden_dim}</strong>
                      <small>tanh units</small>
                    </div>
                    <b aria-hidden="true">-&gt;</b>
                  </>
                ) : null}
                <div className="architecture-latent">
                  <Sigma size={14} />
                  <strong>
                    {model?.components ??
                      (activeMethod === "physics_cvae"
                        ? config.latent_dim
                        : config.pca_components)}
                  </strong>
                  <small>
                    {activeMethod === "physics_cvae"
                      ? "mu, logvar"
                      : model?.architecture === "hybrid_threshold_pca"
                        ? "base + guide basis"
                      : activeMethod === "threshold_conditional_pca"
                        ? "threshold-focused PCA basis"
                      : activeMethod === "conditional_pca"
                        ? "conditioned PCA basis"
                        : "PCA basis"}
                  </small>
                </div>
                <b aria-hidden="true">-&gt;</b>
                {activeMethod === "physics_cvae" ? (
                  <>
                    <div>
                      <span>Decoder</span>
                      <strong>{model?.hidden_dim ?? config.hidden_dim}</strong>
                      <small>tanh units</small>
                    </div>
                    <b aria-hidden="true">-&gt;</b>
                  </>
                ) : null}
                <div>
                  <span>Output</span>
                  <strong>{201 * generatedChannels.length}</strong>
                  <small>log residual points</small>
                </div>
              </div>
              <p className="architecture-note">
                {model?.architecture === "hybrid_threshold_pca"
                  ? "The active checkpoint samples a stable global PCA residual, predicts a conditioned PCA guide around threshold, and blends the two with a local Vth-centered envelope before the physics-side jump limiter runs."
                  : activeMethod === "threshold_conditional_pca"
                  ? "The active checkpoint predicts latent coefficients in a threshold-weighted residual space, then removes that weighting during sampling so more model capacity is spent around the Vth transition."
                  : activeMethod === "conditional_pca"
                  ? "The active checkpoint predicts a latent mean from the requested device conditions, adds learned latent noise for diversity, and reconstructs Ids/Ig residuals before blending them with the analytical baseline."
                  : "The active checkpoint generates Ids residuals and, when Ig training curves are available, centered gate-current morphology. Both are blended with analytical baselines before measurement noise and physical projection are applied."}
              </p>
              <div className="architecture-meta">
                <span>Balance: {model?.sample_balance_strategy ?? "uniform"}</span>
                <span>
                  Rare groups: {(model?.rare_curve_groups ?? 0).toLocaleString()}
                </span>
                <span>
                  Features:{" "}
                  {model?.condition_features.length
                    ? model.condition_features.join(", ")
                    : "legacy checkpoint"}
                </span>
              </div>
            </section>
          </div>

          <section className="training-log-panel">
            <div className="panel-title-row">
              <div>
                <h2>Recent training events</h2>
                <p>Latest optimizer and validation checkpoints.</p>
              </div>
              {status?.status === "failed" ? (
                <TriangleAlert size={18} className="training-log-error" />
              ) : (
                <CheckCircle2 size={18} className="training-log-ok" />
              )}
            </div>
            <div className="training-log">
              {history.length === 0 ? (
                <div className="training-log-empty">
                  No epoch history recorded yet.
                </div>
              ) : (
                history.slice(-8).reverse().map((item) => (
                  <div key={item.epoch}>
                    <time>Epoch {item.epoch.toString().padStart(3, "0")}</time>
                    <span>train {item.train_loss.toFixed(5)}</span>
                    <span>val {item.validation_loss.toFixed(5)}</span>
                    <strong>
                      weighted{" "}
                      {metric(
                        item.validation_weighted_rmse_decades ??
                          item.validation_rmse_decades,
                        4
                      )}{" "}
                      dec
                    </strong>
                  </div>
                ))
              )}
            </div>
          </section>

          <Suspense
            fallback={
              <section className="model-inspector-loading">
                Loading final-model curve inspector
              </section>
            }
          >
            <ModelCurveInspector model={model} disabled={running} />
          </Suspense>
        </div>
      </div>
    </main>
  );
}
