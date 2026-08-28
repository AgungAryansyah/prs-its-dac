from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from prs_its.calibration import calibrate_test_predictions, cross_fit_calibration
from prs_its.fairness import age_groups, fairness_audit_rates
from prs_its.modeling import (
    CATEGORICAL_CANDIDATES,
    make_feature_spec,
    prepare_catboost_features,
    train_catboost_cv,
    validate_train_test_schema,
)
from prs_its.submission import make_submission, prediction_summary


def _datasets(rows: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def test_feature_preparation_preserves_schema_and_fills_categories() -> None:
    train, test = _datasets()
    train.loc[0, "kdkc"] = pd.NA
    test.loc[0, "kdkc"] = pd.NA
    spec = make_feature_spec(train, test)
    prepared = prepare_catboost_features(train, test, spec, add_count_features=True)

    assert "claim_id" not in prepared.X
    assert "label" not in prepared.X
    assert prepared.X.loc[0, "kdkc"] == "__MISSING__"
    assert prepared.X_test.loc[0, "kdkc"] == "__MISSING__"
    assert list(prepared.X.columns) == list(prepared.X_test.columns)
    assert {"secondary_diagnosis_count", "procedure_count"} <= set(prepared.X)


def test_schema_validation_rejects_mismatched_order() -> None:
    train, test = _datasets()
    reordered = test.loc[:, ["claim_id", *reversed(test.columns.drop("claim_id").tolist())]]

    with pytest.raises(ValueError, match="mismatch"):
        validate_train_test_schema(train, reordered)


def test_cpu_cv_produces_complete_oof_and_fold_models(tmp_path) -> None:
    train, test = _datasets()
    spec = make_feature_spec(train, test)
    prepared = prepare_catboost_features(train, test, spec)
    progress_events = []
    result = train_catboost_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        spec.categorical_features,
        cv=StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        params={"iterations": 10, "depth": 2, "learning_rate": 0.1},
        task_type="CPU",
        early_stopping_rounds=3,
        model_dir=tmp_path,
        progress_callback=lambda event, fold: progress_events.append((event, fold)),
    )

    assert (result["fold_id"] >= 0).all()
    assert np.isfinite(result["oof_pred"]).all()
    assert np.all((0 <= result["test_pred"]) & (result["test_pred"] <= 1))
    assert result["test_fold_predictions"].shape == (2, len(prepared.X_test))
    assert len(result["fold_metrics"]) == 2
    assert len(list(tmp_path.glob("catboost_fold_*.cbm"))) == 2
    assert progress_events == [("start", 0), ("complete", 0), ("start", 1), ("complete", 1)]


def test_cross_fit_calibration_and_submission_helpers(tmp_path) -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    raw = np.array([0.2, 0.8, 0.3, 0.7, 0.1, 0.9, 0.4, 0.6])
    folds = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    calibrated = cross_fit_calibration(labels, raw, folds, method="sigmoid")["oof_pred"]
    test_prediction = calibrate_test_predictions(raw, labels, [0.25, 0.75], method="isotonic")
    submission = make_submission(
        pd.Series(["A", "B"]), test_prediction, output_path=tmp_path / "submission.csv"
    )

    assert np.all((0 <= calibrated) & (calibrated <= 1))
    assert submission.columns.tolist() == ["claim_id", "fraud_probability"]
    assert prediction_summary(test_prediction)["n_unique"] >= 1


def test_fairness_limits_exposure_to_legitimate_claims() -> None:
    groups = pd.Series(["A", "A", "B", "B"], name="jkpst")
    rates = fairness_audit_rates(groups, [0, 1, 0, 0], [0.9, 0.8, 0.7, 0.1], 0.5)

    assert rates["nonfraud_count"].sum() == 3
    assert rates["audited_nonfraud_count"].sum() == 1
    assert not rates["eligible_for_comparison"].any()


def test_age_groups_use_required_boundaries() -> None:
    assert age_groups(pd.Series([0, 17, 18, 39, 40, 59, 60])).tolist() == [
        "0-17",
        "0-17",
        "18-39",
        "18-39",
        "40-59",
        "40-59",
        "60+",
    ]
