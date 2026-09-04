from __future__ import annotations

import gc
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from prs_its.foundation_modeling import DEFAULT_MIN_FREE_VRAM_GIB

DEFAULT_FINETUNE_EPOCHS = 12
DEFAULT_FINETUNE_LEARNING_RATE = 1e-5
DEFAULT_FINETUNE_WEIGHT_DECAY = 0.01
DEFAULT_FINETUNE_GRAD_CLIP = 1.0
DEFAULT_FINETUNE_PATIENCE = 3
DEFAULT_FINETUNE_MAX_DATA_SIZE = 4096
DEFAULT_FINETUNE_PREDICTION_CHUNK_SIZE = 256
DEFAULT_FINETUNE_MIN_PREDICTION_CHUNK_SIZE = 64
DEFAULT_FINETUNE_SUPPORT_CAP = 100_000
FINETUNE_DATA_SIZE_PROFILES = (4096, 2048, 1024)


@dataclass(frozen=True)
class TabICLFinetuneParams:
    epochs: int = DEFAULT_FINETUNE_EPOCHS
    learning_rate: float = DEFAULT_FINETUNE_LEARNING_RATE
    weight_decay: float = DEFAULT_FINETUNE_WEIGHT_DECAY
    grad_clip: float = DEFAULT_FINETUNE_GRAD_CLIP
    patience: int = DEFAULT_FINETUNE_PATIENCE
    max_data_size: int = DEFAULT_FINETUNE_MAX_DATA_SIZE
    prediction_chunk_size: int = DEFAULT_FINETUNE_PREDICTION_CHUNK_SIZE
    min_prediction_chunk_size: int = DEFAULT_FINETUNE_MIN_PREDICTION_CHUNK_SIZE
    support_cap: int = DEFAULT_FINETUNE_SUPPORT_CAP
    min_free_vram_gib: float = DEFAULT_MIN_FREE_VRAM_GIB
    n_estimators_finetune: int = 1
    n_estimators_validation: int = 1
    n_estimators_inference: int = 4
    validation_split_ratio: float = 0.10
    finetune_ctx_query_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Fine-tuning optimizer values must be valid.")
        if self.patience < 1:
            raise ValueError("patience must be positive.")
        if self.max_data_size not in FINETUNE_DATA_SIZE_PROFILES:
            raise ValueError(
                f"max_data_size must be one of {FINETUNE_DATA_SIZE_PROFILES}."
            )
        if self.prediction_chunk_size < self.min_prediction_chunk_size:
            raise ValueError(
                "prediction_chunk_size must be at least min_prediction_chunk_size."
            )
        if self.min_prediction_chunk_size < 1 or self.support_cap < 2:
            raise ValueError("Prediction and support limits must be positive.")
        if self.min_free_vram_gib <= 0:
            raise ValueError("min_free_vram_gib must be positive.")
        if (
            min(
                self.n_estimators_finetune,
                self.n_estimators_validation,
                self.n_estimators_inference,
            )
            < 1
        ):
            raise ValueError("TabICL ensemble counts must be positive.")
        if not 0 < self.validation_split_ratio < 0.5:
            raise ValueError("validation_split_ratio must be within (0, 0.5).")
        if not 0 < self.finetune_ctx_query_ratio < 1:
            raise ValueError("finetune_ctx_query_ratio must be within (0, 1).")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionProfile:
    initial_chunk_size: int
    effective_chunk_size: int
    attempted_chunk_sizes: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupportProfile:
    strategy: str
    support_rows: int
    support_cap: int
    attempted_rows: tuple[int, ...]
    selected_indices: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "support_rows": self.support_rows,
            "support_cap": self.support_cap,
            "attempted_rows": list(self.attempted_rows),
        }


@dataclass(frozen=True)
class FinetuneResult:
    checkpoint_path: Path
    checkpoint_dir: Path
    effective_max_data_size: int
    attempted_max_data_sizes: tuple[int, ...]
    validation_profile: PredictionProfile | None
    peak_gpu_allocation_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_dir": str(self.checkpoint_dir),
            "effective_max_data_size": self.effective_max_data_size,
            "attempted_max_data_sizes": list(self.attempted_max_data_sizes),
            "validation_profile": (
                self.validation_profile.as_dict()
                if self.validation_profile is not None
                else None
            ),
            "peak_gpu_allocation_bytes": self.peak_gpu_allocation_bytes,
        }


def ensure_tabicl_finetune_gpu_ready(
    min_free_vram_gib: float = DEFAULT_MIN_FREE_VRAM_GIB,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TabICL fine-tuning requires CUDA; CPU execution is not supported."
        )
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    required_bytes = int(min_free_vram_gib * 1024**3)
    if free_bytes < required_bytes:
        raise RuntimeError(
            "TabICL fine-tuning GPU preflight requires at least "
            f"{min_free_vram_gib:.1f} GiB free; only {free_bytes / 1024**3:.2f} GiB of "
            f"{total_bytes / 1024**3:.2f} GiB is available."
        )
    return {
        "device": torch.cuda.get_device_name(device),
        "free_vram_gib": round(free_bytes / 1024**3, 3),
        "total_vram_gib": round(total_bytes / 1024**3, 3),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def training_data_size_profiles(max_data_size: int) -> tuple[int, ...]:
    if max_data_size not in FINETUNE_DATA_SIZE_PROFILES:
        raise ValueError(f"max_data_size must be one of {FINETUNE_DATA_SIZE_PROFILES}.")
    start = FINETUNE_DATA_SIZE_PROFILES.index(max_data_size)
    return FINETUNE_DATA_SIZE_PROFILES[start:]


def adaptive_predict_probabilities(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    prediction_chunk_size: int,
    min_prediction_chunk_size: int,
) -> tuple[np.ndarray, PredictionProfile]:
    if (
        prediction_chunk_size < min_prediction_chunk_size
        or min_prediction_chunk_size < 1
    ):
        raise ValueError("Prediction chunk sizes are invalid.")
    chunk_size = prediction_chunk_size
    attempted = [chunk_size]
    predictions: list[np.ndarray] = []
    start = 0
    while start < len(X):
        stop = min(start + chunk_size, len(X))
        try:
            probabilities = np.asarray(
                estimator.predict_proba(_slice_rows(X, start, stop)), dtype=float
            )
            if probabilities.ndim != 2 or probabilities.shape[1] != 2:
                raise RuntimeError("TabICL must return binary class probabilities.")
            values = probabilities[:, 1]
            if (
                not np.isfinite(values).all()
                or not ((0 <= values) & (values <= 1)).all()
            ):
                raise RuntimeError(
                    "TabICL prediction values must be finite probabilities."
                )
            predictions.append(values)
            start = stop
        except Exception as error:
            if not is_cuda_oom(error) or chunk_size == min_prediction_chunk_size:
                raise
            release_cuda_memory()
            chunk_size = max(min_prediction_chunk_size, chunk_size // 2)
            attempted.append(chunk_size)
    return np.concatenate(predictions), PredictionProfile(
        initial_chunk_size=prediction_chunk_size,
        effective_chunk_size=chunk_size,
        attempted_chunk_sizes=tuple(attempted),
    )


def deterministic_stratified_support_indices(
    labels: pd.Series | np.ndarray,
    support_cap: int,
    random_state: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    if len(y) <= support_cap:
        return np.arange(len(y), dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError("Support fallback requires binary labels.")
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=support_cap, random_state=random_state
    )
    selected, _ = next(splitter.split(np.zeros(len(y)), y))
    return np.sort(selected.astype(int))


def fit_in_context_predictor(
    estimator_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    support_cap: int,
    random_state: int,
) -> tuple[Any, SupportProfile]:
    labels = np.asarray(y, dtype=int)
    attempted_rows = [len(X)]
    estimator = estimator_factory()
    try:
        estimator.fit(X, labels)
        return estimator, SupportProfile(
            strategy="full_support",
            support_rows=len(X),
            support_cap=support_cap,
            attempted_rows=tuple(attempted_rows),
            selected_indices=np.arange(len(X), dtype=int),
        )
    except Exception as error:
        _release_estimator(estimator)
        if not is_cuda_oom(error):
            raise
    release_cuda_memory()
    selected_indices = deterministic_stratified_support_indices(
        labels, support_cap, random_state
    )
    attempted_rows.append(len(selected_indices))
    estimator = estimator_factory()
    try:
        estimator.fit(X.iloc[selected_indices], labels[selected_indices])
    except Exception as error:
        _release_estimator(estimator)
        raise RuntimeError(
            "TabICL in-context support could not fit on CUDA after disk offload. "
            f"Attempted support profiles {attempted_rows}; {_cuda_memory_diagnostics()}"
        ) from error
    return estimator, SupportProfile(
        strategy="stratified_support_cap",
        support_rows=len(selected_indices),
        support_cap=support_cap,
        attempted_rows=tuple(attempted_rows),
        selected_indices=selected_indices,
    )


def fine_tune_with_oom_backoff(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame,
    y_val: pd.Series | np.ndarray,
    params: TabICLFinetuneParams,
    checkpoint_root: Path,
    cache_dir: Path,
    random_state: int,
    time_limit_seconds: float | None = None,
    finetuner_factory: Callable[..., Any] | None = None,
) -> FinetuneResult:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    attempted: list[int] = []
    factory = finetuner_factory or _create_finetuner
    for max_data_size in training_data_size_profiles(params.max_data_size):
        attempted.append(max_data_size)
        attempt_dir = _next_attempt_directory(checkpoint_root)
        attempt_dir.mkdir(parents=True, exist_ok=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        estimator = factory(
            params=params,
            max_data_size=max_data_size,
            output_dir=attempt_dir,
            cache_dir=cache_dir / attempt_dir.name,
            random_state=random_state,
            time_limit_seconds=time_limit_seconds,
        )
        _install_chunked_validation(estimator, params)
        try:
            estimator.fit(
                X_train,
                np.asarray(y_train, dtype=int),
                X_val=X_val,
                y_val=np.asarray(y_val, dtype=int),
                output_dir=attempt_dir,
            )
            checkpoint_path = attempt_dir / "best.ckpt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"TabICL fine-tuning did not save {checkpoint_path}."
                )
            validation_profile = getattr(estimator, "_prs_its_validation_profile", None)
            peak_allocation = (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else 0
            )
            return FinetuneResult(
                checkpoint_path=checkpoint_path,
                checkpoint_dir=attempt_dir,
                effective_max_data_size=max_data_size,
                attempted_max_data_sizes=tuple(attempted),
                validation_profile=validation_profile,
                peak_gpu_allocation_bytes=peak_allocation,
            )
        except Exception as error:
            if (
                not is_cuda_oom(error)
                or max_data_size == FINETUNE_DATA_SIZE_PROFILES[-1]
            ):
                raise
            _release_estimator(estimator)
            release_cuda_memory()
        finally:
            _release_estimator(estimator)
    raise RuntimeError(
        "TabICL fine-tuning exhausted CUDA memory for max_data_size profiles "
        f"{attempted}; {_cuda_memory_diagnostics()}"
    )


def create_tabicl_predictor(
    checkpoint_path: Path,
    params: TabICLFinetuneParams,
    cache_dir: Path,
    random_state: int,
) -> Any:
    from tabicl import TabICLClassifier

    cache_dir.mkdir(parents=True, exist_ok=True)
    return TabICLClassifier(
        n_estimators=params.n_estimators_inference,
        batch_size=1,
        kv_cache="repr",
        model_path=checkpoint_path,
        allow_auto_download=False,
        device="cuda",
        use_amp="auto",
        offload_mode="disk",
        disk_offload_dir=str(cache_dir),
        random_state=random_state,
        verbose=False,
    )


def is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()
    return (
        "cuda out of memory" in message
        or "outofmemoryerror" in error.__class__.__name__.lower()
    )


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _create_finetuner(
    *,
    params: TabICLFinetuneParams,
    max_data_size: int,
    output_dir: Path,
    cache_dir: Path,
    random_state: int,
    time_limit_seconds: float | None,
) -> Any:
    from tabicl import FinetunedTabICLClassifier

    cache_dir.mkdir(parents=True, exist_ok=True)
    return FinetunedTabICLClassifier(
        epochs=params.epochs,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        grad_clip=params.grad_clip,
        amp=True,
        n_estimators_finetune=params.n_estimators_finetune,
        n_estimators_validation=params.n_estimators_validation,
        n_estimators_inference=params.n_estimators_inference,
        max_data_size=max_data_size,
        finetune_ctx_query_ratio=params.finetune_ctx_query_ratio,
        validation_split_ratio=params.validation_split_ratio,
        early_stopping=True,
        patience=params.patience,
        time_limit=time_limit_seconds,
        save_interval=1,
        device="cuda",
        random_state=random_state,
        verbose=False,
        wandb_kwargs=None,
        eval_metric="roc_auc",
        extra_classifier_kwargs={
            "batch_size": 1,
            "kv_cache": "repr",
            "offload_mode": "disk",
            "disk_offload_dir": str(cache_dir),
            "use_amp": "auto",
        },
    )


def _install_chunked_validation(estimator: Any, params: TabICLFinetuneParams) -> None:
    from tabicl._finetune.base import ValidationMetrics

    def run_validation(
        self: Any,
        inner: Any,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        inner.fit(X_train, y_train)
        probabilities, profile = adaptive_predict_probabilities(
            inner,
            X_val,
            params.prediction_chunk_size,
            params.min_prediction_chunk_size,
        )
        self._prs_its_validation_profile = profile
        full_probabilities = np.column_stack((1 - probabilities, probabilities))
        roc = float(roc_auc_score(y_val, probabilities))
        ll = float(log_loss(y_val, full_probabilities, labels=inner.classes_))
        acc = float(
            accuracy_score(
                y_val, np.asarray(inner.classes_)[full_probabilities.argmax(axis=1)]
            )
        )
        primary = {"roc_auc": roc, "log_loss": -ll, "accuracy": acc}[self.eval_metric]
        return ValidationMetrics(
            primary=primary, secondary={"roc_auc": roc, "log_loss": ll, "accuracy": acc}
        )

    estimator._run_validation = types.MethodType(run_validation, estimator)


def _next_attempt_directory(checkpoint_root: Path) -> Path:
    attempts = [
        int(path.name.removeprefix("attempt-"))
        for path in checkpoint_root.iterdir()
        if path.is_dir()
        and path.name.startswith("attempt-")
        and path.name.removeprefix("attempt-").isdigit()
    ]
    return checkpoint_root / f"attempt-{max(attempts, default=0) + 1:02d}"


def _slice_rows(
    X: pd.DataFrame | np.ndarray, start: int, stop: int
) -> pd.DataFrame | np.ndarray:
    return X.iloc[start:stop] if isinstance(X, pd.DataFrame) else X[start:stop]


def _release_estimator(estimator: Any) -> None:
    del estimator
    release_cuda_memory()


def _cuda_memory_diagnostics() -> str:
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda"))
    return f"free_vram_gib={free_bytes / 1024**3:.2f}; total_vram_gib={total_bytes / 1024**3:.2f}"
