from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import prs_its.training as training
from prs_its.modeling import CATEGORICAL_CANDIDATES
from prs_its.metrics import evaluate_probabilities


def _write_competition_data(project_root: Path) -> None:
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


def test_python_training_orchestration_writes_core_artifacts(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    _write_competition_data(tmp_path)

    def fake_train_catboost_cv(X, y, X_test, categorical_features, cv, params, task_type, devices, **kwargs):
        probabilities = np.where(y.to_numpy() == 1, 0.8, 0.2)
        metrics = evaluate_probabilities(y, probabilities)
        fold_metrics = pd.DataFrame(
            [{**metrics, "best_iteration": 3}, {**metrics, "best_iteration": 4}]
        )
        return {
            "oof_pred": probabilities,
            "test_pred": np.full(len(X_test), 0.5),
            "test_fold_predictions": np.full((2, len(X_test)), 0.5),
            "models": [],
            "fold_metrics": fold_metrics,
            "fold_id": np.repeat([0, 1], len(X) // 2),
            "feature_importance": pd.DataFrame(
                {"feature": X.columns, "importance": np.ones(len(X.columns)), "fold": 0}
            ),
            "params": {**params, "task_type": task_type, "devices": devices},
        }

    monkeypatch.setattr(training, "train_catboost_cv", fake_train_catboost_cv)
    monkeypatch.setattr(training, "_save_figures", lambda *args: None)
    monkeypatch.setattr(training, "_save_explanations", lambda *args: None)

    result = training.run_training(
        training.TrainingConfig(project_root=tmp_path, task_type="CPU", n_splits=2)
    )

    assert result["submission_rows"] == 6
    assert result["submission_path"].exists()
    assert (tmp_path / "outputs" / "oof" / "catboost_oof.csv").exists()
    assert (tmp_path / "outputs" / "metrics" / "catboost_experiments.csv").exists()
    assert (tmp_path / "outputs" / "models" / "catboost_final_config.json").exists()
