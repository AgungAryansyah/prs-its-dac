from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import prs_its.xgb_modeling as xgb_modeling
from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec
from prs_its.xgb_modeling import (
    FoldTargetEncoderTransformer,
    prepare_xgb_features,
    train_xgb_cv,
)


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


def test_target_encoder_cross_fits_training_and_uses_outer_train_mappings(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "category": ["A", "A", "A", "A", "B", "B", "B", "B", None, None, "C", "C"],
            "numeric": np.arange(12),
        }
    )
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0])
    transformer = FoldTargetEncoderTransformer(
        ["category"], inner_n_splits=2, smooth=0, random_state=42, add_support_counts=True
    )

    cross_fitted = transformer.fit_transform(train, labels)
    full_mapping = transformer.transform(train)
    validation = transformer.transform(pd.DataFrame({"category": ["D", None], "numeric": [1, 2]}))

    assert not np.allclose(cross_fitted["te_category"], full_mapping["te_category"])
    assert validation.loc[0, "te_category"] == transformer.global_rate
    assert validation.loc[0, "support_count_category"] == 0
    assert validation.loc[1, "support_count_category"] == 2

    path = tmp_path / "encoder.joblib"
    transformer.save(path, seed=42, fold=0)
    payload = joblib.load(path)
    loaded = FoldTargetEncoderTransformer.load(path)
    assert payload["seed"] == 42
    assert payload["fold"] == 0
    pd.testing.assert_frame_equal(loaded.transform(train), full_mapping)


def test_prepare_xgb_features_adds_the_targeted_interaction_schema() -> None:
    train, test = _datasets()
    spec = make_feature_spec(train, test)

    prepared = prepare_xgb_features(train, test, spec, add_interaction_features=True)

    assert {"dati2_typeppk", "diagprimer_cmg", "cmg_severitylevel"} <= set(
        prepared.categorical_features
    )
    assert {"secondary_diagnosis_count", "procedure_count", "los_zero_indicator"} <= set(
        prepared.X
    )
    assert "claim_id" not in prepared.X
    assert list(prepared.X.columns) == list(prepared.X_test.columns)


class _MockXGBClassifier:
    def __init__(self, **params) -> None:
        self.params = params
        self.best_iteration = 2
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X, y, eval_set, verbose=False):
        self.feature_importances_ = np.full(X.shape[1], 1 / X.shape[1])
        return self

    def predict_proba(self, X):
        probabilities = np.linspace(0.2, 0.8, len(X))
        return np.column_stack([1 - probabilities, probabilities])

    def save_model(self, path: Path) -> None:
        path.write_text("mock xgboost model")


def test_xgb_cv_saves_fold_models_and_encoders_with_complete_oof(tmp_path, monkeypatch) -> None:
    train, test = _datasets()
    spec = make_feature_spec(train, test)
    prepared = prepare_xgb_features(train, test, spec, add_interaction_features=True)
    progress_events = []
    monkeypatch.setattr(xgb_modeling, "XGBClassifier", _MockXGBClassifier)

    result = train_xgb_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        params={"n_estimators": 10, "random_state": 2026},
        task_type="CPU",
        early_stopping_rounds=3,
        model_dir=tmp_path,
        model_prefix="xgb_seed_2026",
        progress_callback=lambda event, fold: progress_events.append((event, fold)),
        target_encoder_factory=lambda: FoldTargetEncoderTransformer(
            prepared.categorical_features,
            inner_n_splits=2,
            random_state=2026,
            add_support_counts=True,
        ),
    )

    assert (result["fold_id"] >= 0).all()
    assert np.isfinite(result["oof_pred"]).all()
    assert result["test_fold_predictions"].shape == (2, len(prepared.X_test))
    assert len(result["fold_transformers"]) == 2
    assert len(list(tmp_path.glob("xgb_seed_2026_fold_*.json"))) == 2
    assert len(list(tmp_path.glob("xgb_seed_2026_target_encoder_fold_*.joblib"))) == 2
    assert progress_events == [("start", 0), ("complete", 0), ("start", 1), ("complete", 1)]
    assert result["params"]["device"] == "cpu"


def test_xgb_cv_runs_with_the_real_cpu_estimator() -> None:
    train, test = _datasets()
    spec = make_feature_spec(train, test)
    prepared = prepare_xgb_features(train, test, spec)

    result = train_xgb_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        params={"n_estimators": 10, "max_depth": 2, "min_child_weight": 1},
        task_type="CPU",
        early_stopping_rounds=3,
        target_encoder_factory=lambda: FoldTargetEncoderTransformer(
            prepared.categorical_features, inner_n_splits=2, random_state=42
        ),
    )

    assert np.isfinite(result["oof_pred"]).all()
    assert np.all((0 <= result["test_pred"]) & (result["test_pred"] <= 1))
