from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np

from .neural import NeuralTrainingConfig, train_neural_checkpoint
from .schemas import (
    NeuralEpochMetric,
    NeuralTrainingRequest,
    NeuralTrainingStatus,
    NeuralTrialSummary,
)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _trial_requests(request: NeuralTrainingRequest) -> list[NeuralTrainingRequest]:
    if request.search_strategy == "single" or request.search_trials <= 1:
        return [request]
    trials = [request]
    for index in range(1, request.search_trials):
        if request.method in {
            "latent_pca",
            "conditional_pca",
            "threshold_conditional_pca",
            "local_threshold_conditional_pca",
            "aligned_local_threshold_conditional_pca",
            "aligned_local_delta_conditional_pca",
            "aligned_local_affine_delta_conditional_pca",
        }:
            multipliers = (0.5, 1.5, 2.0, 0.75, 1.25, 2.5, 0.35)
            multiplier = multipliers[(index - 1) % len(multipliers)]
            trials.append(
                request.model_copy(
                    update={
                        "pca_components": max(
                            1,
                            min(64, int(round(request.pca_components * multiplier))),
                        ),
                        "rare_curve_weight": min(
                            10.0,
                            request.rare_curve_weight * (1.0 + 0.12 * index),
                        ),
                        "beta": min(1.0, max(1e-5, request.beta * (0.8 + 0.15 * index))),
                        "seed": request.seed + index * 101,
                    }
                )
            )
            continue
        variants = (
            {
                "latent_dim": min(64, request.latent_dim + 4),
                "hidden_dim": min(1024, max(8, int(round(request.hidden_dim * 1.5)))),
                "learning_rate": max(1e-6, request.learning_rate * 0.65),
                "beta": max(0.0, request.beta * 0.7),
                "low_current_weight": min(20.0, request.low_current_weight * 1.2),
                "subthreshold_weight": min(20.0, request.subthreshold_weight * 1.25),
                "rare_curve_weight": min(10.0, request.rare_curve_weight * 1.18),
            },
            {
                "latent_dim": max(4, request.latent_dim - 4),
                "hidden_dim": min(1024, max(8, int(round(request.hidden_dim * 1.2)))),
                "learning_rate": min(1.0, request.learning_rate * 1.25),
                "beta": min(1.0, request.beta * 1.5),
                "slope_weight": min(10.0, request.slope_weight * 1.5),
                "rare_curve_weight": max(1.0, request.rare_curve_weight * 0.92),
            },
            {
                "latent_dim": min(64, request.latent_dim + 8),
                "hidden_dim": min(1024, max(8, request.hidden_dim * 2)),
                "learning_rate": max(1e-6, request.learning_rate * 0.5),
                "beta": min(1.0, max(request.beta * 0.5, 0.001)),
                "gate_loss_weight": min(10.0, request.gate_loss_weight * 1.5),
                "rare_curve_weight": min(10.0, request.rare_curve_weight * 1.32),
            },
        )
        update = {
            **variants[(index - 1) % len(variants)],
            "seed": request.seed + index * 101,
        }
        trials.append(request.model_copy(update=update))
    return trials


def _training_config(request: NeuralTrainingRequest) -> NeuralTrainingConfig:
    return NeuralTrainingConfig(
        method=request.method,
        latent_dim=request.latent_dim,
        hidden_dim=request.hidden_dim,
        epochs=request.epochs,
        batch_size=request.batch_size,
        learning_rate=request.learning_rate,
        beta=request.beta,
        validation_fraction=request.validation_fraction,
        patience=request.patience,
        seed=request.seed,
        max_curves=request.max_curves,
        low_current_weight=request.low_current_weight,
        subthreshold_weight=request.subthreshold_weight,
        slope_weight=request.slope_weight,
        gate_loss_weight=request.gate_loss_weight,
        rare_curve_weight=request.rare_curve_weight,
        pca_components=request.pca_components,
        feature_eval_limit=request.feature_eval_limit,
    )


def _activate_best_checkpoint(
    source: Path,
    destination: Path,
    *,
    trials: list[NeuralTrialSummary],
    best_trial: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.activate.npz")
    try:
        with np.load(source) as payload:
            arrays = {name: payload[name] for name in payload.files}
        metadata = {}
        if "metadata_json" in arrays:
            try:
                metadata = json.loads(str(arrays["metadata_json"].item()))
            except (TypeError, ValueError):
                metadata = {}
        metadata.update(
            {
                "best_trial": best_trial,
                "tuning_trials": [trial.model_dump(mode="json") for trial in trials],
            }
        )
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


class NeuralTrainingManager:
    def __init__(self, activate_checkpoint: Callable[[Path], None]) -> None:
        self._activate_checkpoint = activate_checkpoint
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neural-training")
        self._state = NeuralTrainingStatus()
        self._started_monotonic: float | None = None

    def snapshot(self) -> NeuralTrainingStatus:
        with self._lock:
            state = self._state.model_copy(deep=True)
            if state.status == "running" and self._started_monotonic is not None:
                state.elapsed_seconds = time.monotonic() - self._started_monotonic
            return state

    def start(self, request: NeuralTrainingRequest) -> NeuralTrainingStatus:
        with self._lock:
            if self._state.status == "running":
                raise RuntimeError("A neural training job is already running")
            job_id = uuid4().hex
            self._started_monotonic = time.monotonic()
            self._state = NeuralTrainingStatus(
                status="running",
                stage="loading_data",
                job_id=job_id,
                message="Loading aligned transfer curves",
                started_at=_timestamp(),
                current_epoch=0,
                total_epochs=request.epochs,
                progress_fraction=0.0,
                current_trial=0,
                total_trials=(
                    request.search_trials
                    if request.search_strategy == "quick"
                    else 1
                ),
                config=request,
            )
        self._executor.submit(self._run, job_id, request)
        return self.snapshot()

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            if self._state.job_id != job_id:
                return
            self._state = self._state.model_copy(update=changes)

    def _run(self, job_id: str, request: NeuralTrainingRequest) -> None:
        project_root = Path(__file__).resolve().parents[1]
        default_output_name = (
            "residual-cvae.npz"
            if request.method in {"physics_cvae", "aligned_local_delta_cvae"}
            else "residual-conditional-pca.npz"
            if request.method in {"conditional_pca", "threshold_conditional_pca", "local_threshold_conditional_pca", "aligned_local_threshold_conditional_pca"}
            or request.method in {"aligned_local_delta_conditional_pca", "aligned_local_affine_delta_conditional_pca"}
            else "residual-pca.npz"
        )
        output = Path(
            os.getenv(
                "DEVICEGEN_NEURAL_MODEL_OUTPUT",
                project_root / "models" / default_output_name,
            )
        ).expanduser().resolve()
        dataset_path = (
            Path(request.dataset_path).expanduser()
            if Path(request.dataset_path).is_absolute()
            else project_root / request.dataset_path
        )
        trial_requests = _trial_requests(request)
        trial_outputs: list[Path] = []
        try:
            results = []
            trial_summaries: list[NeuralTrialSummary] = []
            for trial_index, trial_request in enumerate(trial_requests, start=1):
                trial_output = output.with_name(
                    f".{output.stem}.{job_id}.trial-{trial_index}.npz"
                )
                trial_outputs.append(trial_output)
                self._update(
                    job_id,
                    stage="preparing",
                    message=(
                        f"Preparing trial {trial_index}/{len(trial_requests)} "
                        f"({trial_request.method})"
                    ),
                    current_trial=trial_index,
                    current_epoch=0,
                )

                def progress(
                    metrics: dict[str, float | int | None],
                    *,
                    current_trial: int = trial_index,
                    current_request: NeuralTrainingRequest = trial_request,
                ) -> None:
                    metric = NeuralEpochMetric.model_validate(
                        {**metrics, "trial": current_trial}
                    )
                    tracked_rmse = (
                        metric.validation_weighted_rmse_decades
                        if metric.validation_weighted_rmse_decades is not None
                        else metric.validation_rmse_decades
                    )
                    trial_progress = (
                        metric.epoch / max(current_request.epochs, 1)
                        if current_request.method == "physics_cvae"
                        else 1.0
                    )
                    trial_epochs = (
                        current_request.epochs
                        if current_request.method == "physics_cvae"
                        else 1
                    )
                    with self._lock:
                        if self._state.job_id != job_id:
                            return
                        history = [*self._state.history, metric]
                        self._state = self._state.model_copy(
                            update={
                                "stage": "training",
                                "message": (
                                    f"Trial {current_trial}/{len(trial_requests)}, "
                                    f"epoch {metric.epoch}/{trial_epochs}: "
                                    f"weighted log RMSE {tracked_rmse:.4f} decades"
                                ),
                                "current_epoch": metric.epoch,
                                "progress_fraction": (
                                    (current_trial - 1 + trial_progress)
                                    / len(trial_requests)
                                ),
                                "history": history,
                            }
                        )

                result = train_neural_checkpoint(
                    trial_output,
                    dataset_path=(
                        dataset_path if request.data_source == "export" else None
                    ),
                    config=_training_config(trial_request),
                    progress=progress,
                )
                summary = NeuralTrialSummary(
                    trial=trial_index,
                    method=trial_request.method,
                    latent_dim=result.latent_dim,
                    hidden_dim=result.hidden_dim,
                    learning_rate=trial_request.learning_rate,
                    beta=trial_request.beta,
                    validation_rmse_decades=result.validation_rmse_decades,
                    validation_weighted_rmse_decades=(
                        result.validation_weighted_rmse_decades
                    ),
                    validation_gate_rmse_decades=result.validation_gate_rmse_decades,
                    selection_score=(
                        result.selection_score
                        if result.selection_score is not None
                        else result.validation_rmse_decades
                    ),
                )
                results.append(result)
                trial_summaries.append(summary)
                self._update(job_id, trials=trial_summaries)

            best_index = min(
                range(len(results)),
                key=lambda index: (
                    results[index].selection_score
                    if results[index].selection_score is not None
                    else results[index].validation_rmse_decades
                ),
            )
            best_trial = best_index + 1
            result = results[best_index].model_copy(
                update={
                    "best_trial": best_trial,
                    "output": str(output),
                }
            )
            self._update(job_id, stage="saving", message="Activating the best checkpoint")
            _activate_best_checkpoint(
                trial_outputs[best_index],
                output,
                trials=trial_summaries,
                best_trial=best_trial,
            )
            self._activate_checkpoint(output)
            elapsed = (
                time.monotonic() - self._started_monotonic
                if self._started_monotonic is not None
                else 0.0
            )
            self._update(
                job_id,
                status="completed",
                stage="completed",
                message="Training completed and checkpoint activated",
                completed_at=_timestamp(),
                elapsed_seconds=elapsed,
                current_epoch=result.epochs_completed,
                progress_fraction=1.0,
                current_trial=best_trial,
                total_trials=len(trial_requests),
                trials=trial_summaries,
                result=result,
                error=None,
            )
        except Exception as error:
            elapsed = (
                time.monotonic() - self._started_monotonic
                if self._started_monotonic is not None
                else 0.0
            )
            self._update(
                job_id,
                status="failed",
                stage="failed",
                message="Training failed",
                completed_at=_timestamp(),
                elapsed_seconds=elapsed,
                error=str(error),
            )
        finally:
            for trial_output in trial_outputs:
                trial_output.unlink(missing_ok=True)
