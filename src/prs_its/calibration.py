from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from prs_its.metrics import evaluate_probabilities


def calibration_curve_frame(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    observed, predicted = calibration_curve(labels, probabilities, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted_probability": predicted, "observed_fraud_rate": observed})


def prediction_distribution(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame({"label": np.asarray(y_true, dtype=int), "fraud_probability": y_prob})
    return (
        frame.groupby("label")["fraud_probability"]
        .agg(
            minimum="min",
            mean="mean",
            median="median",
            percentile_95=lambda values: values.quantile(0.95),
            percentile_99=lambda values: values.quantile(0.99),
            maximum="max",
        )
        .reset_index()
    )


def _make_calibrator(method: str) -> LogisticRegression | IsotonicRegression:
    if method == "sigmoid":
        return LogisticRegression(random_state=42, solver="lbfgs")
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip")
    raise ValueError("method must be 'sigmoid' or 'isotonic'.")


def _fit_calibrator(
    calibrator: LogisticRegression | IsotonicRegression,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression | IsotonicRegression:
    if isinstance(calibrator, LogisticRegression):
        calibrator.fit(probabilities.reshape(-1, 1), labels)
    else:
        calibrator.fit(probabilities, labels)
    return calibrator


def _transform_calibrator(
    calibrator: LogisticRegression | IsotonicRegression,
    probabilities: np.ndarray,
) -> np.ndarray:
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    return calibrator.predict(probabilities)


def cross_fit_calibration(
    y_true: Iterable[int] | np.ndarray,
    raw_oof_pred: Iterable[float] | np.ndarray,
    fold_id: Iterable[int] | np.ndarray,
    method: str,
) -> dict[str, object]:
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(raw_oof_pred, dtype=float)
    folds = np.asarray(fold_id, dtype=int)
    if not (len(labels) == len(probabilities) == len(folds)):
        raise ValueError("Labels, probabilities, and folds must have matching lengths.")
    if (folds < 0).any():
        raise ValueError("Every row must have a fold id.")

    calibrated = np.zeros(len(labels), dtype=float)
    calibrators: dict[int, LogisticRegression | IsotonicRegression] = {}
    for fold in np.unique(folds):
        train_mask = folds != fold
        valid_mask = ~train_mask
        calibrator = _fit_calibrator(
            _make_calibrator(method), probabilities[train_mask], labels[train_mask]
        )
        calibrated[valid_mask] = _transform_calibrator(calibrator, probabilities[valid_mask])
        calibrators[int(fold)] = calibrator
    if not np.isfinite(calibrated).all() or not ((0 <= calibrated) & (calibrated <= 1)).all():
        raise RuntimeError("Calibrated OOF predictions must be finite probabilities.")
    return {"oof_pred": calibrated, "calibrators": calibrators, "method": method}


def calibrate_test_predictions(
    raw_oof_pred: Iterable[float] | np.ndarray,
    y_true: Iterable[int] | np.ndarray,
    raw_test_pred: Iterable[float] | np.ndarray,
    method: str,
) -> np.ndarray:
    calibrator = _fit_calibrator(
        _make_calibrator(method), np.asarray(raw_oof_pred, dtype=float), np.asarray(y_true, dtype=int)
    )
    calibrated = _transform_calibrator(calibrator, np.asarray(raw_test_pred, dtype=float))
    if not np.isfinite(calibrated).all() or not ((0 <= calibrated) & (calibrated <= 1)).all():
        raise RuntimeError("Calibrated test predictions must be finite probabilities.")
    return calibrated


def calibration_comparison(
    y_true: Iterable[int] | np.ndarray,
    raw_oof_pred: Iterable[float] | np.ndarray,
    calibrated_oof_pred: Iterable[float] | np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, probabilities in (("raw", raw_oof_pred), ("calibrated", calibrated_oof_pred)):
        metrics = evaluate_probabilities(y_true, probabilities)
        metrics["prediction_type"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def should_select_calibration(
    raw_metrics: dict[str, float | int], calibrated_metrics: dict[str, float | int], tolerance: float = 0.005
) -> bool:
    return bool(
        calibrated_metrics["brier_score"] < raw_metrics["brier_score"]
        and raw_metrics["average_precision"] - calibrated_metrics["average_precision"] <= tolerance
        and raw_metrics["normalized_recall_at_5pct"]
        - calibrated_metrics["normalized_recall_at_5pct"]
        <= tolerance
    )
