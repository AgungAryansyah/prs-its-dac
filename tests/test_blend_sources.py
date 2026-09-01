from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import prs_its.blend_sources as blend_sources
from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec, prepare_catboost_features
from prs_its.xgb_modeling import prepare_xgb_features


def _competition_data(rows: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return train, test


def _write_source_run(project_root: Path, train: pd.DataFrame, test: pd.DataFrame) -> None:
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
                "features": list(xgb_prepared.X.columns),
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
    for seed in (42, 2026, 2718):
        pd.DataFrame(
            {
                "claim_id": train["claim_id"],
                "label": train["label"],
                "fold": folds,
                "fraud_probability_raw": np.where(train["label"].to_numpy() == 1, 0.7, 0.3),
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
                "fraud_probability_raw": np.where(train["label"].to_numpy() == 1, 0.8, 0.2),
            }
        ).to_csv(xgb_dir / "oof" / f"xgb_oof_seed_{seed}.csv", index=False)
        pd.DataFrame(
            {
                "claim_id": test["claim_id"],
                "fold_0": np.full(len(test), 0.4),
                "fold_1": np.full(len(test), 0.6),
            }
        ).to_csv(xgb_dir / "oof" / f"xgb_test_fold_predictions_seed_{seed}.csv", index=False)


class _FakeCatBoostClassifier:
    def load_model(self, path: Path) -> None:
        self.path = path

    def predict_proba(self, X) -> np.ndarray:
        return np.tile([0.4, 0.6], (len(X), 1))


def test_sources_validate_oof_and_rebuild_ctr_test_predictions(tmp_path, monkeypatch) -> None:
    train, test = _competition_data()
    _write_source_run(tmp_path, train, test)
    monkeypatch.setattr(blend_sources, "CatBoostClassifier", _FakeCatBoostClassifier)
    monkeypatch.setattr(blend_sources, "Pool", lambda X, cat_features: X)

    sources = blend_sources.load_blend_source_artifacts(tmp_path, "ctr-v1", "xgb-v1", train, test)
    predictions = blend_sources.reconstruct_ctr_test_predictions(sources, train, test)

    assert sources.n_splits == 2
    assert set(sources.xgb_test_by_seed) == {42, 2026}
    assert set(predictions) == {42, 2026, 2718}
    np.testing.assert_allclose(predictions[2718], 0.6)


def test_sources_reject_mismatched_oof_claim_ids(tmp_path) -> None:
    train, test = _competition_data()
    _write_source_run(tmp_path, train, test)
    path = tmp_path / "outputs" / "runs" / "xgb-v1" / "oof" / "xgb_oof_seed_42.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "claim_id"] = "wrong"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="claim_id"):
        blend_sources.load_blend_source_artifacts(tmp_path, "ctr-v1", "xgb-v1", train, test)
