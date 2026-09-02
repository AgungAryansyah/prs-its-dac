from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import prs_its.history_training as history_training
from prs_its.metrics import evaluate_probabilities
from prs_its.modeling import CATEGORICAL_CANDIDATES, train_catboost_cv


def _competition_data(rows: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "claim_id": [f"TRN_{index:03d}" for index in range(rows)],
            "umur": np.arange(rows) % 70,
            "los": np.arange(rows) % 4,
            "dx2_a00_b99": np.arange(rows) % 3,
            "proc00_13": np.arange(rows) % 2,
        }
    )
    for index, column in enumerate(CATEGORICAL_CANDIDATES):
        frame[column] = [f"{column}_{row % 4}" for row in range(rows)]
    train = frame.copy()
    train["label"] = np.arange(rows) % 2
    test = frame.iloc[:6].copy()
    test["claim_id"] = [f"TST_{index:03d}" for index in range(len(test))]
    history_rows = []
    current = pd.concat(
        [train.drop(columns="label"), test], ignore_index=True
    )
    for index, row in current.iterrows():
        is_train = row["claim_id"].startswith("TRN")
        event_at = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=index)
        history_rows.append(
            {
                "claim_id": row["claim_id"],
                "event_at": event_at,
                "adjudicated_at": event_at + pd.Timedelta(hours=1) if is_train else pd.NaT,
                "provider_id": f"P{index % 6}",
                "patient_id": f"PAT{index % 8}",
                "claim_amount": float(100 + index),
                "tariff_amount": float(80 + index),
                "adjudicated_label": int(train.loc[index, "label"]) if is_train else np.nan,
                "kdkc": row["kdkc"],
                "typeppk": row["typeppk"],
                "cmg": row["cmg"],
                "severitylevel": row["severitylevel"],
                "diagprimer": row["diagprimer"],
            }
        )
    return train, test, pd.DataFrame(history_rows)


def _write_project(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    (project_root / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    data_dir = project_root / "data"
    data_dir.mkdir()
    train, test, history = _competition_data()
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    history.to_csv(project_root / "history.csv", index=False)
    return train, test


def _fake_train_catboost_cv(X, y, X_test, categorical_features, cv, params, task_type, **kwargs):
    del categorical_features, task_type
    groups = kwargs.get("groups")
    split_iterator = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    fold_id = np.full(len(X), -1, dtype=int)
    oof = np.full(len(X), np.nan, dtype=float)
    fold_metrics = []
    for fold, (_, valid_idx) in enumerate(split_iterator):
        history_strength = (
            0.99
            if "history_provider_fraud_rate" in X
            else 0.90
            if "history_provider_prior_claim_count" in X
            else 0.80
            if "history_claim_amount" in X
            else 0.60
        )
        probabilities = np.where(y.iloc[valid_idx].to_numpy() == 1, history_strength, 1 - history_strength)
        oof[valid_idx] = probabilities
        fold_id[valid_idx] = fold
        fold_metrics.append(
            {
                **evaluate_probabilities(y.iloc[valid_idx], probabilities),
                "fold": fold,
                "train_size": len(X) - len(valid_idx),
                "valid_size": len(valid_idx),
                "train_fraud_prevalence": 0.5,
                "valid_fraud_prevalence": float(y.iloc[valid_idx].mean()),
                "best_iteration": 3,
                "iteration_cap": params["iterations"],
                "hit_iteration_cap": False,
                "group_overlap_count": 0,
            }
        )
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback("complete", fold)
    predict_test = kwargs.get("predict_test", True)
    test_fold_predictions = np.full((len(fold_metrics), len(X_test)), 0.5) if predict_test else None
    return {
        "oof_pred": oof,
        "test_pred": np.full(len(X_test), 0.5) if predict_test else None,
        "test_fold_predictions": test_fold_predictions,
        "models": [],
        "fold_metrics": pd.DataFrame(fold_metrics),
        "fold_id": fold_id,
        "feature_importance": pd.DataFrame(
            {"feature": X.columns, "importance": 1.0, "fold": 0}
        ),
        "model_features": list(X.columns),
        "fold_transformers": None,
        "params": {**params, "task_type": "CPU", "devices": "0"},
    }


def test_rolling_origin_cv_uses_only_adjudicated_pre_validation_claims() -> None:
    event_at = pd.Series(pd.date_range("2025-01-01", periods=20, tz="UTC"))
    adjudicated_at = event_at + pd.Timedelta(days=2)
    cv = history_training.RollingOriginCV(event_at, adjudicated_at, n_splits=2)

    folds = list(cv.split(pd.DataFrame(index=range(20))))

    assert cv.evaluation_mask.sum() == 16
    assert cv.fold_id[:4].tolist() == [-1, -1, -1, -1]
    assert folds[0][0].tolist() == [0, 1]
    assert folds[0][1].tolist() == list(range(4, 12))


def test_catboost_cv_allows_a_partial_temporal_oof_cohort() -> None:
    class PartialCV:
        def split(self, X, y):
            del X, y
            yield np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])

    X = pd.DataFrame({"value": np.arange(8, dtype=float)})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    result = train_catboost_cv(
        X,
        y,
        pd.DataFrame({"value": [8.0, 9.0]}),
        [],
        cv=PartialCV(),
        params={"iterations": 10, "depth": 2, "learning_rate": 0.1},
        task_type="CPU",
        early_stopping_rounds=2,
        require_complete_oof=False,
    )

    assert result["fold_id"].tolist() == [-1, -1, -1, -1, 0, 0, 0, 0]
    assert np.isnan(result["oof_pred"][:4]).all()
    assert np.isfinite(result["oof_pred"][4:]).all()


def test_history_preflight_reports_temporal_and_prior_coverage_without_a_run(tmp_path) -> None:
    train, _ = _write_project(tmp_path)
    history_path = tmp_path / "history.csv"
    history = pd.read_csv(history_path)
    prior = history.iloc[[0]].copy()
    prior["claim_id"] = "HIST_000"
    prior["event_at"] = "2024-12-01 00:00:00+00:00"
    prior["adjudicated_at"] = "2024-12-02 00:00:00+00:00"
    prior["adjudicated_label"] = 1
    pd.concat([prior, history], ignore_index=True).to_csv(history_path, index=False)

    report = history_training.preflight_history(
        history_training.HistoryTrainingConfig(
            project_root=tmp_path,
            history_path=history_path,
            run_name="preflight-only",
            task_type="CPU",
            n_splits=2,
            show_progress=False,
        )
    )

    assert report["coverage"]["matched_current_train_claims"] == len(train)
    assert report["prior_history"]["available"] is True
    assert report["temporal_eligibility"]["eligible"] is True
    assert report["ready_to_train"] is True
    assert not (tmp_path / "outputs" / "runs" / "preflight-only").exists()


def test_history_training_runs_all_stages_without_writing_baseline_paths(tmp_path, monkeypatch) -> None:
    train, _ = _write_project(tmp_path)
    calls = []

    def tracked_train(*args, **kwargs):
        calls.append(
            (
                kwargs["params"]["random_seed"],
                "history_provider_fraud_rate" in args[0],
                kwargs.get("groups") is not None,
            )
        )
        return _fake_train_catboost_cv(*args, **kwargs)

    monkeypatch.setattr(history_training, "train_catboost_cv", tracked_train)

    result = history_training.run_history_training(
        history_training.HistoryTrainingConfig(
            project_root=tmp_path,
            history_path=tmp_path / "history.csv",
            run_name="history-check",
            task_type="CPU",
            n_splits=2,
            n_bootstrap=5,
            show_progress=False,
        )
    )

    run_dir = tmp_path / "outputs" / "runs" / "history-check"
    experiments = pd.read_csv(run_dir / "metrics" / "history_experiments.csv")
    assert result["selected_experiment"] == "history_adjudication"
    assert result["promoted"] is False
    assert experiments["experiment_name"].tolist() == [
        "history_static_control",
        "history_financial",
        "history_behavioral",
        "history_adjudication",
    ]
    assert calls == [
        (42, False, False),
        (42, False, False),
        (42, False, False),
        (42, True, False),
        (2026, True, False),
        (2026, False, False),
        (2718, True, False),
        (2718, False, False),
        (42, True, True),
    ]
    assert (run_dir / "oof" / "history_adjudication_temporal_oof_seed_42.csv").exists()
    assert (run_dir / "metrics" / "history_vs_control_ensemble_paired.csv").exists()
    assert (run_dir / "metrics" / "history_provider_grouped_metrics.csv").exists()
    assert (run_dir / "models" / "history_final_config.json").exists()
    assert not list((run_dir / "submissions").glob("*.csv"))
    assert not (tmp_path / "outputs" / "models" / "history_final_config.json").exists()
    with (run_dir / "metrics" / "history_promotion_decision.json").open() as file:
        decision = json.load(file)
    assert decision["fraud_capture_meaningful"] is False
    assert len(train) == 30


def test_history_output_paths_reject_existing_run(tmp_path) -> None:
    root = tmp_path / "outputs" / "runs" / "occupied"
    root.mkdir(parents=True)
    (root / "artifact.txt").write_text("existing")

    try:
        history_training.history_output_paths(tmp_path, "occupied")
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected an existing output run to be rejected.")
