from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import prs_its.foundation_modeling as foundation_modeling
from prs_its.foundation_modeling import (
    FoundationParams,
    _predict_probabilities,
    ensure_foundation_gpu_ready,
    prepare_foundation_features,
    run_foundation_preflight,
    train_foundation_cv,
)
from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec


def _datasets(rows: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "claim_id": [f"CLM_{index:03d}" for index in range(rows)],
            "umur": np.arange(rows) % 70,
            "los": np.arange(rows) % 4,
            "dx2_a00_b99": np.arange(rows) % 3,
            "proc00_13": np.arange(rows) % 2,
        }
    )
    for index, column in enumerate(CATEGORICAL_CANDIDATES):
        frame[column] = [f"{index:02d}{row % 4}" for row in range(rows)]
    train = frame.copy()
    train["label"] = np.arange(rows) % 2
    test = frame.iloc[:8].copy()
    test["claim_id"] = [f"TST_{index:03d}" for index in range(len(test))]
    return train, test


class _FakeFoundationEstimator:
    def __init__(self, calls: list[tuple[int, str, Path | None]], seed: int, cache_dir: Path | None, task_type: str):
        self.calls = calls
        self.seed = seed
        self.cache_dir = cache_dir
        self.task_type = task_type

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> _FakeFoundationEstimator:
        self.calls.append((self.seed, self.task_type, self.cache_dir))
        self.train_rows = len(X)
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probability = np.full(len(X), self.rate, dtype=float)
        return np.column_stack([1 - probability, probability])


def _factory(calls: list[tuple[int, str, Path | None]]):
    def make(params, seed: int, cache_dir: Path | None, task_type: str):
        return _FakeFoundationEstimator(calls, seed, cache_dir, task_type)

    return make


def test_prepare_foundation_features_preserves_raw_schema() -> None:
    train, test = _datasets()
    prepared = prepare_foundation_features(train, test, make_feature_spec(train, test))

    assert list(prepared.X.columns) == list(prepared.X_test.columns)
    assert "claim_id" not in prepared.X
    assert "label" not in prepared.X
    assert "secondary_diagnosis_count" not in prepared.X
    assert prepared.categorical_features == CATEGORICAL_CANDIDATES
    assert all(pd.api.types.is_string_dtype(prepared.X[column]) for column in CATEGORICAL_CANDIDATES)


def test_foundation_cv_completes_oof_and_averages_test_predictions(tmp_path) -> None:
    train, test = _datasets()
    prepared = prepare_foundation_features(train, test, make_feature_spec(train, test))
    calls: list[tuple[int, str, Path | None]] = []
    events: list[tuple[str, int]] = []

    result = train_foundation_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        params=FoundationParams(model="tabicl-v2", prediction_chunk_size=3),
        seed=42,
        task_type="CPU",
        cache_dir=tmp_path / "cache",
        progress_callback=lambda event, fold: events.append((event, fold)),
        estimator_factory=_factory(calls),
    )

    assert np.isfinite(result["oof_pred"]).all()
    assert result["test_fold_predictions"].shape == (2, len(test))
    np.testing.assert_allclose(result["test_pred"], result["test_fold_predictions"].mean(axis=0))
    assert events == [("start", 0), ("complete", 0), ("start", 1), ("complete", 1)]
    assert [seed for seed, _, _ in calls] == [42, 43]
    assert all(task_type == "CPU" for _, task_type, _ in calls)
    assert all(cache_dir is not None for _, _, cache_dir in calls)


def test_foundation_cv_rejects_identifiers_and_schema_mismatch() -> None:
    train, test = _datasets()
    labels = train["label"]
    with pytest.raises(ValueError, match="claim_id"):
        train_foundation_cv(
            train.drop(columns="label"),
            labels,
            test,
            [],
            cv=StratifiedKFold(n_splits=2),
            params=FoundationParams(model="tabicl-v2"),
            seed=42,
            task_type="CPU",
            estimator_factory=_factory([]),
        )

    prepared = prepare_foundation_features(train, test, make_feature_spec(train, test))
    with pytest.raises(ValueError, match="same feature order"):
        train_foundation_cv(
            prepared.X,
            prepared.y,
            prepared.X_test.iloc[:, ::-1],
            prepared.categorical_features,
            cv=StratifiedKFold(n_splits=2),
            params=FoundationParams(model="tabicl-v2"),
            seed=42,
            task_type="CPU",
            estimator_factory=_factory([]),
        )


def test_preflight_runs_one_fold_and_records_resources(tmp_path) -> None:
    train, test = _datasets()
    prepared = prepare_foundation_features(train, test, make_feature_spec(train, test))
    calls: list[tuple[int, str, Path | None]] = []

    result = run_foundation_preflight(
        prepared.X,
        prepared.y,
        prepared.categorical_features,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        params=FoundationParams(model="tabicl-v2", prediction_chunk_size=2),
        seed=42,
        task_type="CPU",
        cache_dir=tmp_path / "cache",
        estimator_factory=_factory(calls),
        prediction_rows=3,
    )

    assert result["train_rows"] == 20
    assert result["prediction_rows"] == 3
    assert result["gpu_status"] == "CPU explicitly selected"
    assert len(calls) == 1


def test_gpu_preflight_rejects_cpu_only_torch(monkeypatch) -> None:
    monkeypatch.setattr(foundation_modeling.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA PyTorch is unavailable"):
        ensure_foundation_gpu_ready()


def test_gpu_preflight_rejects_insufficient_free_memory(monkeypatch) -> None:
    monkeypatch.setattr(foundation_modeling.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        foundation_modeling.torch.cuda,
        "mem_get_info",
        lambda device=None: (2 * 1024**3, 6 * 1024**3),
    )

    with pytest.raises(RuntimeError, match="requires at least 10.0 GiB"):
        ensure_foundation_gpu_ready()


def test_predict_probabilities_rejects_non_binary_output() -> None:
    class Estimator:
        def predict_proba(self, X):
            return np.ones((len(X), 3)) / 3

    with pytest.raises(RuntimeError, match="binary probabilities"):
        _predict_probabilities(Estimator(), pd.DataFrame({"x": [1, 2]}), 1)


def test_foundation_cv_retries_tabicl_with_representation_cache_on_oom(monkeypatch) -> None:
    train, test = _datasets(rows=20)
    prepared = prepare_foundation_features(train, test, make_feature_spec(train, test))
    monkeypatch.setattr(foundation_modeling, "ensure_foundation_gpu_ready", lambda *_: "gpu")
    calls: list[str] = []

    class Estimator:
        def __init__(self, cache_mode: str):
            self.cache_mode = cache_mode

        def fit(self, X, y):
            calls.append(self.cache_mode)
            if self.cache_mode == "auto":
                raise RuntimeError("CUDA out of memory")
            self.rate = float(np.mean(y))

        def predict_proba(self, X):
            values = np.full(len(X), self.rate)
            return np.column_stack([1 - values, values])

    def factory(params, seed, cache_dir, task_type):
        return Estimator(params.tabicl_cache_mode)

    result = train_foundation_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        params=FoundationParams(tabicl_cache_mode="auto"),
        seed=42,
        task_type="GPU",
        estimator_factory=factory,
    )

    assert calls == ["auto", "repr", "auto", "repr"]
    assert result["effective_tabicl_cache_mode"] == "repr"
    assert set(result["fold_metrics"]["tabicl_cache_mode"]) == {"repr"}
