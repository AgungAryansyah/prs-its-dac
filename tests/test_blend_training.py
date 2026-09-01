from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import prs_its.blend_training as blend_training
from prs_its.blend_modeling import PromotionDecision, ScreenDecision
from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec, prepare_catboost_features
from prs_its.xgb_modeling import FoldTargetEncoderTransformer, prepare_xgb_features


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


def _write_source_runs(
    project_root: Path, train: pd.DataFrame, test: pd.DataFrame, xgb_scores: np.ndarray
) -> list[str]:
    ctr_dir = project_root / "outputs" / "runs" / "ctr-v1"
    xgb_dir = project_root / "outputs" / "runs" / "xgb-v1"
    for directory in (ctr_dir / "models", ctr_dir / "oof", xgb_dir / "models", xgb_dir / "oof"):
        directory.mkdir(parents=True, exist_ok=True)
    cv_config = {"type": "StratifiedKFold", "n_splits": 2, "shuffle": True, "random_state": 42}
    spec = make_feature_spec(train, test)
    ctr_prepared = prepare_catboost_features(
        train,
        test,
        spec,
        add_count_features=True,
        add_interaction_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features={"dati2_typeppk": ("dati2", "typeppk")},
    )
    xgb_prepared = prepare_xgb_features(train, test, spec, add_interaction_features=True)
    encoder = FoldTargetEncoderTransformer(
        xgb_prepared.categorical_features,
        inner_n_splits=2,
        random_state=42,
        add_support_counts=True,
    )
    xgb_model_features = encoder.fit_transform(xgb_prepared.X, train["label"]).columns.tolist()
    (ctr_dir / "models" / "catboost_final_config.json").write_text(
        json.dumps(
            {
                "model": "CatBoostClassifier",
                "profile": "ctr",
                "features": list(ctr_prepared.X.columns),
                "categorical_features": ctr_prepared.categorical_features,
                "experiment": {"name": "ctr_dati2_typeppk"},
                "cv": cv_config,
            }
        )
    )
    (xgb_dir / "models" / "xgb_final_config.json").write_text(
        json.dumps(
            {
                "model": "XGBClassifier",
                "features": xgb_model_features,
                "categorical_features": xgb_prepared.categorical_features,
                "params": {"n_estimators": 10},
                "experiment": {"name": "te_xgb_support", "add_support_counts": True},
                "target_encoder": {"inner_n_splits": 2, "smooth": 20.0},
                "cv": cv_config,
                "ensemble_seeds": [42, 2026],
            }
        )
    )
    folds = np.full(len(train), -1, dtype=int)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    for fold, (_, valid_idx) in enumerate(cv.split(train, train["label"])):
        folds[valid_idx] = fold
    ctr_scores = np.where(train["label"].to_numpy() == 1, 0.8, 0.2)
    for seed in (42, 2026, 2718):
        pd.DataFrame(
            {
                "claim_id": train["claim_id"],
                "label": train["label"],
                "fold": folds,
                "fraud_probability_raw": ctr_scores,
            }
        ).to_csv(ctr_dir / "oof" / f"catboost_oof_seed_{seed}.csv", index=False)
        for fold in range(2):
            (ctr_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm").write_text("model")
    for seed in (42, 2026):
        pd.DataFrame(
            {
                "claim_id": train["claim_id"],
                "label": train["label"],
                "fold": folds,
                "fraud_probability_raw": xgb_scores,
            }
        ).to_csv(xgb_dir / "oof" / f"xgb_oof_seed_{seed}.csv", index=False)
        pd.DataFrame(
            {
                "claim_id": test["claim_id"],
                "fold_0": np.full(len(test), 0.4),
                "fold_1": np.full(len(test), 0.6),
            }
        ).to_csv(xgb_dir / "oof" / f"xgb_test_fold_predictions_seed_{seed}.csv", index=False)
    return xgb_model_features


def _fake_train_xgb_cv(model_features: list[str], calls: list[int]):
    def train(X, y, X_test, categorical_features, cv, params, task_type, **kwargs):
        calls.append(params["random_state"])
        callback = kwargs["progress_callback"]
        folds = np.full(len(X), -1, dtype=int)
        for fold, (_, valid_idx) in enumerate(cv.split(X, y)):
            folds[valid_idx] = fold
            callback("start", fold)
            callback("complete", fold)
        model_dir = kwargs["model_dir"]
        prefix = kwargs["model_prefix"]
        for fold in range(2):
            (model_dir / f"{prefix}_fold_{fold}.json").write_text("model")
            (model_dir / f"{prefix}_target_encoder_fold_{fold}.joblib").write_text("encoder")
        probabilities = np.where(y.to_numpy() == 1, 0.9, 0.1)
        return {
            "oof_pred": probabilities,
            "test_pred": np.full(len(X_test), 0.7),
            "test_fold_predictions": np.full((2, len(X_test)), 0.7),
            "fold_metrics": pd.DataFrame(
                [
                    {"fold": 0, "best_iteration": 3},
                    {"fold": 1, "best_iteration": 4},
                ]
            ),
            "fold_id": folds,
            "feature_importance": pd.DataFrame(
                {"feature": model_features, "importance": 1.0, "fold": 0}
            ),
            "model_features": model_features,
            "params": params,
        }

    return train


def test_blend_output_paths_require_a_new_isolated_run(tmp_path) -> None:
    existing = tmp_path / "outputs" / "runs" / "blend-existing"
    existing.mkdir(parents=True)
    (existing / "artifact.txt").write_text("existing")

    with pytest.raises(FileExistsError, match="already contains"):
        blend_training.blend_output_paths(tmp_path, "blend-existing")


def test_blend_training_rejects_screen_without_training_xgb(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    train, test = _write_competition_data(tmp_path)
    _write_source_runs(tmp_path, train, test, np.where(train["label"].to_numpy() == 1, 0.1, 0.9))
    monkeypatch.setattr(
        blend_training,
        "train_xgb_cv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected XGBoost fit")),
    )

    result = blend_training.run_blend_training(
        blend_training.BlendTrainingConfig(
            project_root=tmp_path,
            run_name="blend-rejected",
            task_type="CPU",
            n_bootstrap=2,
            show_progress=False,
        )
    )

    run_dir = tmp_path / "outputs" / "runs" / "blend-rejected"
    assert result["status"] == "screen_rejected"
    assert result["submission_path"] is None
    assert (run_dir / "metrics" / "blend_experiments.csv").exists()
    assert (run_dir / "metrics" / "promotion_decision.json").exists()
    assert not any((run_dir / "submissions").iterdir())


def test_blend_training_confirms_only_fresh_xgb_seed_and_names_submission(tmp_path, monkeypatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test-project'\n")
    train, test = _write_competition_data(tmp_path)
    model_features = _write_source_runs(
        tmp_path, train, test, np.where(train["label"].to_numpy() == 1, 0.8, 0.2)
    )
    calls: list[int] = []
    source_files_before = sorted(
        str(path.relative_to(tmp_path))
        for path in (tmp_path / "outputs" / "runs").rglob("*")
        if path.is_file()
    )
    monkeypatch.setattr(blend_training, "train_xgb_cv", _fake_train_xgb_cv(model_features, calls))
    monkeypatch.setattr(
        blend_training,
        "select_screen_candidate",
        lambda _: ScreenDecision("ctr_xgb_raw_w02", 0.02, True, True, True, True),
    )
    monkeypatch.setattr(
        blend_training,
        "promotion_decision",
        lambda *_: PromotionDecision(True, True, True, True, True, True),
    )
    monkeypatch.setattr(
        blend_training,
        "reconstruct_ctr_test_predictions",
        lambda *_: {42: np.full(len(test), 0.5), 2026: np.full(len(test), 0.5), 2718: np.full(len(test), 0.5)},
    )

    result = blend_training.run_blend_training(
        blend_training.BlendTrainingConfig(
            project_root=tmp_path,
            run_name="blend-confirmed",
            task_type="CPU",
            n_bootstrap=2,
            show_progress=False,
        )
    )

    run_dir = tmp_path / "outputs" / "runs" / "blend-confirmed"
    source_files_after = sorted(
        str(path.relative_to(tmp_path))
        for path in (tmp_path / "outputs" / "runs").rglob("*")
        if path.is_file() and "blend-confirmed" not in path.parts
    )
    assert calls == [2718]
    assert result["status"] == "promoted"
    assert result["submission_path"] == run_dir / "submissions" / "ctr_xgb_raw_w02_submission.csv"
    assert result["submission_path"].exists()
    assert (run_dir / "models" / "xgb_seed_2718_fold_0.json").exists()
    assert (run_dir / "models" / "xgb_seed_2718_target_encoder_fold_1.joblib").exists()
    assert source_files_after == source_files_before
