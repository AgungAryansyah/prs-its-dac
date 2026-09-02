from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import prs_its.tabm_tuning as tabm_tuning
from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec, prepare_catboost_features


def _write_competition_data(project_root) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _write_ctr_source(project_root, train: pd.DataFrame, test: pd.DataFrame) -> None:
    ctr_dir = project_root / "outputs" / "runs" / "ctr-v1"
    for directory in (ctr_dir / "models", ctr_dir / "oof"):
        directory.mkdir(parents=True, exist_ok=True)
    cv_config = {"type": "StratifiedKFold", "n_splits": 2, "shuffle": True, "random_state": 42}
    prepared = prepare_catboost_features(
        train,
        test,
        make_feature_spec(train, test),
        add_count_features=True,
        add_interaction_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features={"dati2_typeppk": ("dati2", "typeppk")},
    )
    (ctr_dir / "models" / "catboost_final_config.json").write_text(
        json.dumps(
            {
                "model": "CatBoostClassifier",
                "profile": "ctr",
                "features": list(prepared.X.columns),
                "categorical_features": prepared.categorical_features,
                "experiment": {"name": "ctr_dati2_typeppk"},
                "cv": cv_config,
            }
        )
    )
    folds = np.full(len(train), -1, dtype=int)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold, (_, valid_idx) in enumerate(cv.split(train, train["label"])):
        folds[valid_idx] = fold
    probabilities = np.where(train["label"].to_numpy() == 1, 0.7, 0.3)
    for seed in (42, 2026, 2718):
        pd.DataFrame(
            {
                "claim_id": train["claim_id"],
                "label": train["label"],
                "fold": folds,
                "fraud_probability_raw": probabilities,
            }
        ).to_csv(ctr_dir / "oof" / f"catboost_oof_seed_{seed}.csv", index=False)


def _fake_train_tabm_cv(calls):
    def train(X, y, X_test, categorical_features, cv, params, seed, task_type, **kwargs):
        del categorical_features, task_type
        calls.append(
            {
                "k": params.k,
                "seed": seed,
                "model_dir": kwargs["model_dir"],
                "groups": kwargs.get("groups") is not None,
                "predict_test": kwargs["predict_test"],
                "batch_size": params.batch_size,
            }
        )
        folds = np.full(len(y), -1, dtype=int)
        groups = kwargs.get("groups")
        splitter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        for fold, (_, valid_idx) in enumerate(splitter):
            folds[valid_idx] = fold
            kwargs["progress_callback"]("start", fold)
            kwargs["progress_callback"]("complete", fold)
        if kwargs["model_dir"] is not None:
            for fold in range(2):
                (kwargs["model_dir"] / f"{kwargs['model_prefix']}_fold_{fold}.pt").write_text("model")
                (kwargs["model_dir"] / f"{kwargs['model_prefix']}_preprocessor_fold_{fold}.joblib").write_text("preprocessor")
        probabilities = np.where(y.to_numpy() == 1, 0.9, 0.1)
        return {
            "oof_pred": probabilities,
            "test_pred": np.full(len(X_test), 0.65) if kwargs["predict_test"] else None,
            "test_fold_predictions": (
                np.full((2, len(X_test)), 0.65) if kwargs["predict_test"] else None
            ),
            "models": [],
            "fold_preprocessors": [],
            "fold_metrics": pd.DataFrame(
                [{"fold": 0, "best_epoch": 2}, {"fold": 1, "best_epoch": 3}]
            ),
            "fold_id": folds,
            "feature_importance": pd.DataFrame(
                {"feature": ["umur"], "importance": [0.1], "fold": [0]}
            ),
            "model_features": ["umur"],
            "fold_model_features": {0: ["umur"], 1: ["umur"]},
            "params": params.as_dict(),
        }

    return train


def test_trial_allocation_is_deterministic_and_output_paths_are_isolated(tmp_path) -> None:
    assert tabm_tuning.trial_allocation(30) == {16: 15, 32: 15}
    assert tabm_tuning.trial_allocation(2) == {16: 1, 32: 1}
    with pytest.raises(ValueError, match="divide evenly"):
        tabm_tuning.trial_allocation(3)

    paths = tabm_tuning.tabm_tuning_output_paths(tmp_path, "tabm-hpo", resume=False)
    (paths["metrics"] / "artifact.txt").write_text("existing")
    with pytest.raises(FileExistsError, match="already contains"):
        tabm_tuning.tabm_tuning_output_paths(tmp_path, "tabm-hpo", resume=False)
    resumed = tabm_tuning.tabm_tuning_output_paths(tmp_path, "tabm-hpo", resume=True)
    assert resumed["root"] == paths["root"]


def test_study_resume_restores_isolated_sqlite_and_sampler(tmp_path) -> None:
    paths = tabm_tuning.tabm_tuning_output_paths(tmp_path, "tabm-resume", resume=False)
    config = tabm_tuning.TabMTuningConfig(project_root=tmp_path, run_name="tabm-resume")
    study = tabm_tuning._open_study(paths, config, 16)
    trial = study.ask()
    study.tell(trial, 1.0)
    tabm_tuning._save_sampler(tabm_tuning._sampler_path(paths, 16), study.sampler)

    resumed_config = tabm_tuning.TabMTuningConfig(
        project_root=tmp_path,
        run_name="tabm-resume",
        resume=True,
    )
    resumed = tabm_tuning._open_study(paths, resumed_config, 16)

    assert len(resumed.trials) == 1
    assert tabm_tuning._study_database_path(paths, 16).exists()
    assert tabm_tuning._sampler_path(paths, 16).exists()


def test_hpo_keeps_trials_model_free_preserves_ctr_and_confirms_selected_seeds(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    train, test = _write_competition_data(tmp_path)
    _write_ctr_source(tmp_path, train, test)
    source_root = tmp_path / "outputs" / "runs" / "ctr-v1"
    source_before = {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    calls = []
    monkeypatch.setattr(tabm_tuning, "train_tabm_cv", _fake_train_tabm_cv(calls))

    result = tabm_tuning.run_tabm_tuning(
        tabm_tuning.TabMTuningConfig(
            project_root=tmp_path,
            run_name="tabm-hpo-check",
            task_type="CPU",
            trials=2,
            n_splits=2,
            n_bootstrap=2,
            show_progress=False,
        )
    )

    run_dir = tmp_path / "outputs" / "runs" / "tabm-hpo-check"
    source_after = {
        path.relative_to(source_root): path.read_bytes()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    assert result["promoted"] is False
    assert result["submission_path"] is None
    assert source_after == source_before
    assert [call["batch_size"] for call in calls[:2]] == [512, 512]
    assert [call["model_dir"] for call in calls[:2]] == [None, None]
    assert [call["predict_test"] for call in calls[:2]] == [False, False]
    final_calls = calls[2:]
    assert [call["seed"] for call in final_calls] == [42, 2026, 2718, 42]
    assert [call["groups"] for call in final_calls] == [False, False, False, True]
    assert not list((run_dir / "models").glob("tuning_*.pt"))
    assert (run_dir / "metrics" / "tabm_hpo_trials.csv").exists()
    assert (run_dir / "metrics" / "tabm_hpo_promotion_decision.json").exists()
    assert (run_dir / "studies" / "tabm_piecewise_k16.sqlite3").exists()
    assert (run_dir / "studies" / "tabm_piecewise_k32.sqlite3").exists()


def test_hpo_selection_requires_screen_guardrails() -> None:
    candidates = pd.DataFrame(
        [
            {
                "experiment_name": "ineligible",
                "screen_eligible": False,
                "fraud_caught_at_5pct": 100,
                "average_precision": 1.0,
                "brier_score": 0.0,
                "fold_normalized_recall_5_std": 0.0,
                "fairness_audit_rate_gap_5": 0.0,
                "tabm_weight": 0.05,
                "trial_number": 0,
                "params": tabm_tuning.largest_hpo_params().as_dict(),
            }
        ]
    )

    assert tabm_tuning._select_candidate(candidates) is None
