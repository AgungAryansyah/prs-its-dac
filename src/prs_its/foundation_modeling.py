from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

from prs_its.metrics import evaluate_probabilities
from prs_its.modeling import ID_COL, TARGET, FeatureSpec, PreparedFeatures, prepare_catboost_features


FOUNDATION_MODELS = ("tabicl-v2", "tabpfn-3")
DEFAULT_FOUNDATION_ESTIMATORS = 4
DEFAULT_FOUNDATION_PREDICTION_CHUNK_SIZE = 1024
DEFAULT_MIN_FREE_VRAM_GIB = 10.0


@dataclass(frozen=True)
class FoundationParams:
    model: str = "tabicl-v2"
    n_estimators: int = DEFAULT_FOUNDATION_ESTIMATORS
    estimator_batch_size: int = 1
    prediction_chunk_size: int = DEFAULT_FOUNDATION_PREDICTION_CHUNK_SIZE
    min_free_vram_gib: float = DEFAULT_MIN_FREE_VRAM_GIB

    def __post_init__(self) -> None:
        if self.model not in FOUNDATION_MODELS:
            raise ValueError(f"model must be one of {FOUNDATION_MODELS}.")
        if self.n_estimators < 1 or self.estimator_batch_size < 1:
            raise ValueError("Foundation estimator counts and batch size must be positive.")
        if self.prediction_chunk_size < 1:
            raise ValueError("prediction_chunk_size must be positive.")
        if self.min_free_vram_gib <= 0:
            raise ValueError("min_free_vram_gib must be positive.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_foundation_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: FeatureSpec,
) -> PreparedFeatures:
    prepared = prepare_catboost_features(train, test, spec)
    if list(prepared.X.columns) != list(prepared.X_test.columns):
        raise RuntimeError("Foundation train and test features are not aligned.")
    if ID_COL in prepared.X or TARGET in prepared.X:
        raise RuntimeError("claim_id and label must not be foundation features.")
    return prepared


def ensure_foundation_gpu_ready(min_free_vram_gib: float = DEFAULT_MIN_FREE_VRAM_GIB) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA PyTorch is unavailable for the foundation model. "
            "Run on the remote CUDA server or explicitly choose --task-type CPU."
        )
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    required_bytes = int(min_free_vram_gib * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            "Foundation model GPU preflight requires at least "
            f"{min_free_vram_gib:.1f} GiB free; only {free_bytes / 1024**3:.2f} GiB of "
            f"{total_bytes / 1024**3:.2f} GiB is available."
        )
    return (
        f"{torch.cuda.get_device_name(device)}; "
        f"free_vram_gib={free_bytes / 1024**3:.2f}; "
        f"total_vram_gib={total_bytes / 1024**3:.2f}; "
        f"torch={torch.__version__}; cuda={torch.version.cuda}"
    )


def train_foundation_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: list[str],
    cv: Any,
    params: FoundationParams,
    seed: int,
    task_type: str = "GPU",
    cache_dir: Path | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
    predict_test: bool = True,
    estimator_factory: Callable[[FoundationParams, int, Path | None, str], Any] | None = None,
) -> dict[str, Any]:
    task_type = task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    _validate_inputs(X, y, X_test, categorical_features)
    if task_type == "GPU":
        ensure_foundation_gpu_ready(params.min_free_vram_gib)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.Series(y, index=X.index, dtype=int)
    oof_pred = np.full(len(X), np.nan, dtype=float)
    fold_id = np.full(len(X), -1, dtype=int)
    test_fold_predictions: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    split_iterator = cv.split(X, labels)
    factory = estimator_factory or _create_estimator

    for fold, (train_idx, valid_idx) in enumerate(split_iterator):
        if progress_callback is not None:
            progress_callback("start", fold)
        started = time.monotonic()
        fold_cache = cache_dir / f"seed_{seed}/fold_{fold}" if cache_dir is not None else None
        estimator = factory(params, seed + fold, fold_cache, task_type)
        try:
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
            y_train = labels.iloc[train_idx].to_numpy()
            y_valid = labels.iloc[valid_idx].to_numpy()
            estimator.fit(X_train, y_train)
            valid_pred = _predict_probabilities(
                estimator, X_valid, params.prediction_chunk_size
            )
            oof_pred[valid_idx] = valid_pred
            fold_id[valid_idx] = fold
            if predict_test:
                test_fold_predictions.append(
                    _predict_probabilities(estimator, X_test, params.prediction_chunk_size)
                )
            metrics = evaluate_probabilities(y_valid, valid_pred)
            metrics.update(
                {
                    "fold": fold,
                    "train_size": len(train_idx),
                    "valid_size": len(valid_idx),
                    "train_fraud_prevalence": float(y_train.mean()),
                    "valid_fraud_prevalence": float(y_valid.mean()),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            fold_metrics.append(metrics)
        finally:
            del estimator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if progress_callback is not None:
            progress_callback("complete", fold)

    if (fold_id < 0).any():
        raise RuntimeError("Every training row must receive exactly one OOF prediction.")
    _validate_probabilities(oof_pred, "Foundation OOF predictions")
    test_fold_predictions_array = np.vstack(test_fold_predictions) if predict_test else None
    test_pred = (
        np.mean(test_fold_predictions_array, axis=0) if predict_test else None
    )
    if predict_test:
        assert test_pred is not None
        _validate_probabilities(test_pred, "Foundation test predictions")
    return {
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "test_fold_predictions": test_fold_predictions_array,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "fold_id": fold_id,
        "params": params.as_dict(),
    }


def run_foundation_preflight(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    cv: Any,
    params: FoundationParams,
    seed: int,
    task_type: str = "GPU",
    cache_dir: Path | None = None,
    estimator_factory: Callable[[FoundationParams, int, Path | None, str], Any] | None = None,
    prediction_rows: int = DEFAULT_FOUNDATION_PREDICTION_CHUNK_SIZE,
) -> dict[str, Any]:
    task_type = task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    _validate_inputs(X, y, X, categorical_features)
    gpu_status = (
        ensure_foundation_gpu_ready(params.min_free_vram_gib) if task_type == "GPU" else "CPU explicitly selected"
    )
    train_idx, valid_idx = next(iter(cv.split(X, y)))
    valid_idx = valid_idx[: min(prediction_rows, len(valid_idx))]
    factory = estimator_factory or _create_estimator
    fold_cache = cache_dir / f"preflight/seed_{seed}/fold_0" if cache_dir is not None else None
    if fold_cache is not None:
        fold_cache.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    estimator = factory(params, seed, fold_cache, task_type)
    try:
        estimator.fit(X.iloc[train_idx], pd.Series(y).iloc[train_idx].to_numpy())
        probabilities = _predict_probabilities(
            estimator, X.iloc[valid_idx], params.prediction_chunk_size
        )
        _validate_probabilities(probabilities, "Foundation preflight predictions")
        peak_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    except Exception as error:
        raise RuntimeError(
            f"Foundation {params.model} CUDA/model preflight failed: {error}"
        ) from error
    finally:
        del estimator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "model": params.model,
        "seed": seed,
        "train_rows": len(train_idx),
        "prediction_rows": len(valid_idx),
        "elapsed_seconds": time.monotonic() - started,
        "peak_allocated_bytes": peak_bytes,
        "gpu_status": gpu_status,
        "params": params.as_dict(),
    }


def _create_estimator(
    params: FoundationParams,
    seed: int,
    cache_dir: Path | None,
    task_type: str,
) -> Any:
    device = "cuda" if task_type == "GPU" else "cpu"
    if params.model == "tabicl-v2":
        try:
            from tabicl import TabICLClassifier
        except ImportError as error:
            raise RuntimeError(
                "TabICLv2 is unavailable. Install the locked project dependencies with `uv sync`."
            ) from error
        return TabICLClassifier(
            n_estimators=params.n_estimators,
            batch_size=params.estimator_batch_size,
            kv_cache=True,
            device=device,
            use_amp="auto",
            offload_mode="auto",
            disk_offload_dir=str(cache_dir) if cache_dir is not None else None,
            random_state=seed,
            verbose=False,
        )
    try:
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except ImportError as error:
        raise RuntimeError(
            "TabPFN-3 is unavailable. Install the optional dependency with `uv sync --extra tabpfn`."
        ) from error
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V3,
        n_estimators=params.n_estimators,
        auto_scale_n_estimators=False,
        device=device,
        fit_mode="low_memory",
        memory_saving_mode=True,
        inference_precision="auto",
        random_state=seed,
        show_progress_bar=False,
    )


def _predict_probabilities(estimator: Any, X: pd.DataFrame, chunk_size: int) -> np.ndarray:
    predictions = []
    for start in range(0, len(X), chunk_size):
        output = estimator.predict_proba(X.iloc[start : start + chunk_size])
        values = output.to_numpy() if isinstance(output, pd.DataFrame) else np.asarray(output)
        if values.ndim == 2:
            if values.shape[1] != 2:
                raise RuntimeError("Foundation classifier must return binary probabilities.")
            values = values[:, 1]
        values = np.asarray(values, dtype=float).reshape(-1)
        predictions.append(values)
    probabilities = np.concatenate(predictions) if predictions else np.empty(0, dtype=float)
    _validate_probabilities(probabilities, "Foundation predictions")
    return probabilities


def _validate_inputs(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    categorical_features: list[str],
) -> None:
    if not isinstance(X, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("Foundation features must be pandas DataFrames.")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("X and X_test must have the same feature order.")
    if ID_COL in X or TARGET in X or ID_COL in X_test or TARGET in X_test:
        raise ValueError("claim_id and label must not be foundation model inputs.")
    missing = [feature for feature in categorical_features if feature not in X]
    if missing:
        raise ValueError(f"Categorical features are unavailable: {missing}")
    labels = np.asarray(y, dtype=int).reshape(-1)
    if len(labels) != len(X):
        raise ValueError("X and y must have the same length.")
    if len(labels) == 0:
        raise ValueError("X and y must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y must contain only binary labels.")


def _validate_probabilities(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise RuntimeError(f"{name} must contain finite probabilities within [0, 1].")
