from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import prs_its.xgb_training as xgb_training
import prs_its.xgb_modeling as xgb_modeling
from prs_its.metrics import evaluate_probabilities
from prs_its.modeling import CATEGORICAL_CANDIDATES


def _write_competition_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = project_root / "data"
    data_dir.mkdir()
    rows = 20
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
        frame[column] = [f"{index:02d}{row % 3}" for row in range(rows)]
    train = frame.copy()
    train["label"] = np.arange(rows) % 2
    test = frame.iloc[:6].copy()
    test["claim_id"] = [f"TST_{index:03d}" for index in range(len(test))]
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    return train, test


def _write_incumbent_oof(project_root: Path, train: pd.DataFrame) -> None:
    oof_dir = project_root / "outputs" / "runs" / "ctr-v1" / "oof"
    oof_dir.mkdir(parents=True)
    fold = np.repeat([0, 1], len(train) // 2)
    probabilities = np.where(train["label"].to_numpy() == 1, 0.7, 0.3)
    for seed in (42, 2026):
        pd.DataFrame(
            {
                "claim_id": train["claim_id"],
                "label": train["label"],
                "fold": fold,
                "random_seed": seed,
                "fraud_probability_raw": probabilities,
            }
        ).to_csv(oof_dir / f"catboost_oof_seed_{seed}.csv", index=False)


def _fake_train_xgb_cv(X, y, X_test, categorical_features, cv, params, task_type, **kwargs):
    callback = kwargs.get("progress_callback")
    if callback is not None:
        callback("start", 0)
        callback("complete", 0)
        callback("start", 1)
        callback("complete", 1)
    transformer = kwargs["target_encoder_factory"]().fit(X, y)
    probabilities = np.where(y.to_numpy() == 1, 0.8, 0.2)
    fold = np.repeat([0, 1], len(X) // 2)
    fold_metrics = []
    for fold_index in (0, 1):
        valid = fold == fold_index
        fold_metrics.append(
            {
                **evaluate_probabilities(y.iloc[valid], probabilities[valid]),
                "fold": fold_index,
                "train_size": len(X) // 2,
                "valid_size": len(X) // 2,
                "train_fraud_prevalence": 0.5,
                "valid_fraud_prevalence": 0.5,
                "best_iteration": 3,
                "iteration_cap": params["n_estimators"],
                "hit_iteration_cap": False,
            }
        )
    predict_test = kwargs.get("predict_test", True)
    test_fold_predictions = np.full((2, len(X_test)), 0.5) if predict_test else None
    return {
        "oof_pred": probabilities,
        "test_pred": np.full(len(X_test), 0.5) if predict_test else None,
        "test_fold_predictions": test_fold_predictions,
        "models": [],
        "fold_metrics": pd.DataFrame(fold_metrics),
        "fold_id": fold,
        "feature_importance": pd.DataFrame(
            {"feature": transformer.transform(X).columns, "importance": 1.0, "fold": 0}
        ),
        "model_features": transformer.transform(X).columns.tolist(),
        "fold_transformers": [transformer, transformer],
        "params": {**params, "device": "cpu" if task_type == "CPU" else "cuda"},
    }


def test_xgb_training_stages_and_writes_only_to_its_run(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    train, _ = _write_competition_data(tmp_path)
    _write_incumbent_oof(tmp_path, train)
    calls = []

    def tracked_train_xgb_cv(*args, **kwargs):
        transformer = kwargs["target_encoder_factory"]()
        calls.append(
            (
                kwargs["params"]["random_state"],
                transformer.config.add_support_counts,
                "dati2_typeppk" in transformer.categorical_features,
            )
        )
        return _fake_train_xgb_cv(*args, **kwargs)

    def select_support(results: pd.DataFrame) -> str:
        return "te_xgb_interactions" if len(results) == 2 else "te_xgb_support"

    monkeypatch.setattr(xgb_training, "train_xgb_cv", tracked_train_xgb_cv)
    monkeypatch.setattr(xgb_training, "_select_experiment", select_support)

    result = xgb_training.run_xgb_training(
        xgb_training.XGBTrainingConfig(
            project_root=tmp_path,
            run_name="xgb-check",
            task_type="CPU",
            n_splits=2,
            show_progress=False,
            n_bootstrap=5,
        )
    )

    run_dir = tmp_path / "outputs" / "runs" / "xgb-check"
    experiments = pd.read_csv(run_dir / "metrics" / "xgb_experiments.csv")
    assert result["selected_experiment"] == "te_xgb_support"
    assert set(experiments["experiment_name"]) == {
        "te_xgb_base",
        "te_xgb_interactions",
        "te_xgb_support",
    }
    assert calls == [
        (42, False, False),
        (42, False, True),
        (42, True, True),
        (42, True, True),
        (2026, True, True),
        (42, True, True),
    ]
    assert (run_dir / "oof" / "xgb_oof_seed_42.csv").exists()
    assert (run_dir / "oof" / "xgb_oof_seed_2026.csv").exists()
    assert (run_dir / "metrics" / "xgb_vs_ctr_seed_42_paired.csv").exists()
    assert (run_dir / "metrics" / "xgb_vs_ctr_ensemble_fairness_gaps.csv").exists()
    assert (run_dir / "submissions" / "xgb_target_encoding_submission.csv").exists()
    assert not (tmp_path / "outputs" / "models" / "xgb_final_config.json").exists()
    with (run_dir / "models" / "xgb_final_config.json").open() as file:
        final_config = json.load(file)
    assert final_config["experiment"]["add_support_counts"] is True
    assert final_config["target_encoder"]["artifacts_by_seed"]["42"] == [
        "xgb_seed_42_target_encoder_fold_0.joblib",
        "xgb_seed_42_target_encoder_fold_1.joblib",
    ]


def test_xgb_output_paths_require_an_isolated_run_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="required"):
        xgb_training.xgb_output_paths(tmp_path, "")


class _ProbeBooster:
    def __init__(self, device: str) -> None:
        self.device = device

    def save_config(self) -> str:
        return f'{{"learner": {{"generic_param": {{"device": "{self.device}"}}}}}}'


class _ProbeClassifier:
    def __init__(self, **params) -> None:
        self.device = params["device"]

    def fit(self, X, y, verbose=False):
        return self

    def get_booster(self) -> _ProbeBooster:
        return _ProbeBooster(self.device)


def test_cuda_probe_rejects_a_silent_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setattr(xgb_modeling, "XGBClassifier", _ProbeClassifier)

    assert xgb_modeling.ensure_xgb_gpu_ready() == "cuda"

    class _FallbackProbeClassifier(_ProbeClassifier):
        def get_booster(self) -> _ProbeBooster:
            return _ProbeBooster("cpu")

    monkeypatch.setattr(xgb_modeling, "XGBClassifier", _FallbackProbeClassifier)
    with pytest.raises(RuntimeError, match="CUDA XGBoost is unavailable"):
        xgb_modeling.ensure_xgb_gpu_ready()
