from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import prs_its.tabm_training as tabm_training
from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec, prepare_catboost_features


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


def _write_ctr_source(project_root: Path, train: pd.DataFrame, test: pd.DataFrame) -> None:
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
        for fold in range(2):
            (ctr_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm").write_text("model")


def _fake_train_tabm_cv(calls: list[tuple[str, int, bool]]):
    def train(X, y, X_test, categorical_features, cv, params, seed, task_type, **kwargs):
        groups = kwargs.get("groups")
        calls.append((params.variant, seed, groups is not None))
        folds = np.full(len(X), -1, dtype=int)
        splitter = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
        for fold, (_, valid_idx) in enumerate(splitter):
            folds[valid_idx] = fold
            kwargs["progress_callback"]("start", fold)
            kwargs["progress_callback"]("complete", fold)
        model_dir = kwargs.get("model_dir")
        if model_dir is not None:
            prefix = kwargs["model_prefix"]
            for fold in range(2):
                (model_dir / f"{prefix}_fold_{fold}.pt").write_text("model")
                (model_dir / f"{prefix}_preprocessor_fold_{fold}.joblib").write_text("preprocessor")
        probability = 0.9 if params.variant == "tabm_piecewise" else 0.8
        probabilities = np.where(y.to_numpy() == 1, probability, 1 - probability)
        return {
            "oof_pred": probabilities,
            "test_pred": np.full(len(X_test), 0.65),
            "test_fold_predictions": (
                np.full((2, len(X_test)), 0.65) if kwargs["predict_test"] else None
            ),
            "models": [],
            "fold_preprocessors": [],
            "fold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "best_epoch": 2, "train_size": len(X) // 2, "valid_size": len(X) // 2},
                    {"fold": 1, "best_epoch": 3, "train_size": len(X) // 2, "valid_size": len(X) // 2},
                ]
            ),
            "fold_id": folds,
            "feature_importance": pd.DataFrame(
                {"feature": ["umur"], "importance": [0.1], "fold": [0]}
            ),
            "model_features": ["umur", "dati2"],
            "fold_model_features": {0: ["umur", "dati2"], 1: ["umur", "dati2"]},
            "params": params.as_dict(),
        }

    return train


def test_tabm_output_paths_require_an_isolated_run(tmp_path) -> None:
    run_dir = tmp_path / "outputs" / "runs" / "tabm-existing"
    run_dir.mkdir(parents=True)
    (run_dir / "artifact.txt").write_text("existing")

    with pytest.raises(FileExistsError, match="already contains"):
        tabm_training.tabm_output_paths(tmp_path, "tabm-existing")


def test_tabm_training_screens_variants_confirms_seeds_and_preserves_ctr(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    train, test = _write_competition_data(tmp_path)
    _write_ctr_source(tmp_path, train, test)
    calls: list[tuple[str, int, bool]] = []
    source_files_before = sorted(
        str(path.relative_to(tmp_path))
        for path in (tmp_path / "outputs" / "runs" / "ctr-v1").rglob("*")
        if path.is_file()
    )
    monkeypatch.setattr(tabm_training, "train_tabm_cv", _fake_train_tabm_cv(calls))
    monkeypatch.setattr(
        tabm_training,
        "reconstruct_ctr_test_predictions",
        lambda *_: {seed: np.full(len(test), 0.5) for seed in (42, 2026, 2718)},
    )

    result = tabm_training.run_tabm_training(
        tabm_training.TabMTrainingConfig(
            project_root=tmp_path,
            run_name="tabm-screened",
            task_type="CPU",
            n_splits=2,
            n_bootstrap=2,
            show_progress=False,
        )
    )

    run_dir = tmp_path / "outputs" / "runs" / "tabm-screened"
    source_files_after = sorted(
        str(path.relative_to(tmp_path))
        for path in (tmp_path / "outputs" / "runs" / "ctr-v1").rglob("*")
        if path.is_file()
    )
    assert calls == [
        ("tabm_base", 42, False),
        ("tabm_piecewise", 42, False),
        ("tabm_piecewise", 2026, False),
        ("tabm_piecewise", 2718, False),
        ("tabm_piecewise", 42, True),
    ]
    assert result["selected_variant"] == "tabm_piecewise"
    assert result["promoted"] is False
    assert result["submission_status"] == "unpromoted"
    assert result["submission_path"].exists()
    assert result["submission_path"].name.endswith("_unpromoted_submission.csv")
    assert (run_dir / "metrics" / "tabm_experiments.csv").exists()
    assert (run_dir / "metrics" / "tabm_promotion_decision.json").exists()
    assert (run_dir / "oof" / "tabm_base_oof_seed_42.csv").exists()
    assert (run_dir / "oof" / "tabm_piecewise_oof_seed_2026.csv").exists()
    assert (run_dir / "metrics" / "tabm_piecewise_raw_screen_vs_ctr_paired.csv").exists()
    assert (run_dir / "metrics" / "tabm_piecewise_raw_ensemble_vs_ctr_paired.csv").exists()
    assert source_files_after == source_files_before


def test_tabm_training_rejects_missing_ctr_oof_before_creating_run(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    train, test = _write_competition_data(tmp_path)
    _write_ctr_source(tmp_path, train, test)
    (tmp_path / "outputs" / "runs" / "ctr-v1" / "oof" / "catboost_oof_seed_2026.csv").unlink()

    with pytest.raises(FileNotFoundError, match="catboost_oof_seed_2026"):
        tabm_training.run_tabm_training(
            tabm_training.TabMTrainingConfig(
                project_root=tmp_path,
                run_name="tabm-missing-source",
                task_type="CPU",
                n_splits=2,
                n_bootstrap=2,
                show_progress=False,
            )
        )

    assert not (tmp_path / "outputs" / "runs" / "tabm-missing-source").exists()


def test_tabm_training_names_promoted_submission_after_selected_experiment(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    train, test = _write_competition_data(tmp_path)
    _write_ctr_source(tmp_path, train, test)
    calls: list[tuple[str, int, bool]] = []
    monkeypatch.setattr(tabm_training, "train_tabm_cv", _fake_train_tabm_cv(calls))
    monkeypatch.setattr(
        tabm_training,
        "_promotion_decision",
        lambda *_: {
            "promoted": True,
            "fraud_capture_meaningful": True,
            "fresh_seed_noninferior": True,
        },
    )
    monkeypatch.setattr(
        tabm_training,
        "reconstruct_ctr_test_predictions",
        lambda *_: {seed: np.full(len(test), 0.5) for seed in (42, 2026, 2718)},
    )

    result = tabm_training.run_tabm_training(
        tabm_training.TabMTrainingConfig(
            project_root=tmp_path,
            run_name="tabm-promoted",
            task_type="CPU",
            n_splits=2,
            n_bootstrap=2,
            show_progress=False,
        )
    )

    assert result["promoted"] is True
    assert result["submission_path"].exists()
    assert result["submission_path"].name.startswith("tabm_piecewise_raw_")
    assert result["raw_submission_path"].exists()


def test_tabm_promotion_requires_meaningful_fraud_capture() -> None:
    def comparison(fraud_delta: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"metric": "fraud_caught", "audit_fraction": 0.05, "delta": fraud_delta, "ci_lower": 1.0},
                {"metric": "normalized_recall", "audit_fraction": 0.05, "delta": 0.1, "ci_lower": 0.01},
                {"metric": "average_precision", "audit_fraction": np.nan, "delta": 0.0, "ci_lower": 0.0},
                {"metric": "brier_score", "audit_fraction": np.nan, "delta": -0.01, "ci_lower": -0.02},
            ]
        )

    fairness = {
        "gap_intervals": pd.DataFrame(
            [
                {"group_variable": "gender", "audit_fraction": 0.05, "ci_lower": -0.01},
                {"group_variable": "age_group", "audit_fraction": 0.05, "ci_lower": -0.01},
            ]
        )
    }

    promoted = tabm_training._promotion_decision(comparison(20), fairness, comparison(20))
    rejected = tabm_training._promotion_decision(comparison(19), fairness, comparison(19))

    assert promoted["promoted"] is True
    assert rejected["promoted"] is False
