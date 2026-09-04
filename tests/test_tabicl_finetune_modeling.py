from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prs_its.tabicl_finetune_modeling as finetune_modeling
from prs_its.tabicl_finetune_modeling import (
    TabICLFinetuneParams,
    adaptive_predict_probabilities,
    deterministic_stratified_support_indices,
    ensure_tabicl_finetune_gpu_ready,
    fine_tune_with_oom_backoff,
    fit_in_context_predictor,
    tabicl_inference_kwargs,
    training_data_size_profiles,
)


class _ChunkLimitedEstimator:
    def __init__(self, maximum_rows: int):
        self.maximum_rows = maximum_rows
        self.calls: list[int] = []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.calls.append(len(X))
        if len(X) > self.maximum_rows:
            raise RuntimeError("CUDA out of memory")
        return np.tile(np.array([[0.7, 0.3]]), (len(X), 1))


def test_adaptive_predictions_back_off_after_cuda_oom() -> None:
    estimator = _ChunkLimitedEstimator(maximum_rows=128)
    X = pd.DataFrame({"value": np.arange(300)})

    predictions, profile = adaptive_predict_probabilities(
        estimator, X, prediction_chunk_size=256, min_prediction_chunk_size=64
    )

    assert len(predictions) == len(X)
    assert profile.effective_chunk_size == 128
    assert profile.attempted_chunk_sizes == (256, 128)
    assert estimator.calls == [256, 128, 128, 44]


def test_adaptive_predictions_raise_when_minimum_chunk_is_exhausted() -> None:
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        adaptive_predict_probabilities(
            _ChunkLimitedEstimator(maximum_rows=32),
            pd.DataFrame({"value": np.arange(64)}),
            prediction_chunk_size=64,
            min_prediction_chunk_size=64,
        )


def test_support_subset_is_deterministic_and_stratified() -> None:
    labels = np.array([0] * 90 + [1] * 10)

    first = deterministic_stratified_support_indices(
        labels, support_cap=40, random_state=42
    )
    second = deterministic_stratified_support_indices(
        labels, support_cap=40, random_state=42
    )

    assert np.array_equal(first, second)
    assert len(first) == 40
    assert labels[first].sum() == 4


def test_support_fit_retries_with_the_stratified_cap() -> None:
    X = pd.DataFrame({"value": np.arange(100)})
    y = np.array([0] * 90 + [1] * 10)
    fit_rows: list[int] = []

    class Estimator:
        def fit(self, values: pd.DataFrame, labels: np.ndarray) -> None:
            fit_rows.append(len(values))
            if len(values) > 40:
                raise RuntimeError("CUDA out of memory")

    estimator, profile = fit_in_context_predictor(
        Estimator,
        X,
        y,
        support_cap=40,
        random_state=42,
    )

    assert estimator is not None
    assert fit_rows == [100, 40]
    assert profile.strategy == "stratified_support_cap"
    assert profile.attempted_rows == (100, 40)
    assert y[profile.selected_indices].sum() == 4


def test_finetune_oom_restarts_from_a_new_checkpoint_attempt(tmp_path: Path) -> None:
    calls: list[int] = []

    class Finetuner:
        def __init__(self, max_data_size: int):
            self.max_data_size = max_data_size

        def fit(self, X, y, X_val, y_val, output_dir):
            calls.append(self.max_data_size)
            if self.max_data_size == 4096:
                raise RuntimeError("CUDA out of memory")
            Path(output_dir, "best.ckpt").write_text("checkpoint")

    def factory(**kwargs):
        return Finetuner(kwargs["max_data_size"])

    X = pd.DataFrame({"value": np.arange(20)})
    result = fine_tune_with_oom_backoff(
        X.iloc[:16],
        np.array([0, 1] * 8),
        X.iloc[16:],
        np.array([0, 1, 0, 1]),
        TabICLFinetuneParams(max_data_size=4096),
        checkpoint_root=tmp_path / "checkpoints",
        cache_dir=tmp_path / "cache",
        random_state=42,
        finetuner_factory=factory,
    )

    assert calls == [4096, 2048]
    assert result.effective_max_data_size == 2048
    assert result.attempted_max_data_sizes == (4096, 2048)
    assert (tmp_path / "checkpoints" / "attempt-01").exists()
    assert result.checkpoint_path.exists()


def test_training_profiles_are_fixed_and_descending() -> None:
    assert training_data_size_profiles(4096) == (4096, 2048, 1024)
    assert training_data_size_profiles(2048) == (2048, 1024)


def test_inference_offload_can_be_disabled_without_changing_default(tmp_path: Path) -> None:
    disk_kwargs = tabicl_inference_kwargs(TabICLFinetuneParams(), tmp_path)
    gpu_kwargs = tabicl_inference_kwargs(
        TabICLFinetuneParams(offload_mode=False), tmp_path
    )

    assert disk_kwargs["offload_mode"] == "disk"
    assert disk_kwargs["disk_offload_dir"] == str(tmp_path)
    assert gpu_kwargs["offload_mode"] is False
    assert "disk_offload_dir" not in gpu_kwargs


def test_gpu_preflight_rejects_cpu_execution(monkeypatch) -> None:
    monkeypatch.setattr(finetune_modeling.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        ensure_tabicl_finetune_gpu_ready()
