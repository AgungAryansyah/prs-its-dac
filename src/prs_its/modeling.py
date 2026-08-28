from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from prs_its.metrics import evaluate_probabilities


ID_COL = "claim_id"
TARGET = "label"
RANDOM_STATE = 42
N_SPLITS = 5
CATEGORICAL_CANDIDATES = [
    "kdkc",
    "dati2",
    "typeppk",
    "jkpst",
    "jnspelsep",
    "cmg",
    "severitylevel",
    "diagprimer",
]
NUMERIC_CANDIDATES = ["umur", "los"]
BASE_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 2000,
    "learning_rate": 0.03,
    "depth": 6,
    "l2_leaf_reg": 5.0,
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
}


@dataclass(frozen=True)
class FeatureSpec:
    features: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    binary_features: list[str]
    count_features: list[str]
    excluded_features: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return asdict(self)


@dataclass
class PreparedFeatures:
    X: pd.DataFrame
    y: pd.Series
    X_test: pd.DataFrame
    spec: FeatureSpec


def code_like_dtypes(columns: Iterable[str] = CATEGORICAL_CANDIDATES) -> dict[str, str]:
    return {column: "string" for column in columns}


def validate_train_test_schema(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str = ID_COL,
    target: str = TARGET,
) -> list[str]:
    if id_col not in train or id_col not in test:
        raise ValueError(f"Both datasets must contain {id_col!r}.")
    if target not in train or target in test:
        raise ValueError(f"Training data must contain {target!r} and test data must not.")
    if not set(train[target].dropna().unique()).issubset({0, 1}):
        raise ValueError("The target must contain only 0 and 1.")

    features = [column for column in train.columns if column not in {id_col, target}]
    test_features = [column for column in test.columns if column != id_col]
    if features != test_features:
        missing_from_test = sorted(set(features) - set(test_features))
        unexpected_in_test = sorted(set(test_features) - set(features))
        raise ValueError(
            "Train/test feature mismatch. "
            f"Missing from test: {missing_from_test}; unexpected in test: {unexpected_in_test}; "
            f"same order: {features == test_features}."
        )
    return features


def make_feature_spec(
    train: pd.DataFrame,
    test: pd.DataFrame,
    id_col: str = ID_COL,
    target: str = TARGET,
) -> FeatureSpec:
    features = validate_train_test_schema(train, test, id_col=id_col, target=target)
    categorical_features = [column for column in CATEGORICAL_CANDIDATES if column in features]
    numeric_features = [column for column in NUMERIC_CANDIDATES if column in features]
    grouped_features = [
        column for column in features if column.startswith("dx2_") or column.startswith("proc")
    ]
    binary_features = []
    count_features = []
    for column in grouped_features:
        values = pd.to_numeric(train[column], errors="coerce").dropna().unique()
        if set(values).issubset({0, 1}):
            binary_features.append(column)
        else:
            count_features.append(column)
    undefined = set(features) - set(categorical_features) - set(numeric_features) - set(grouped_features)
    if undefined:
        raise ValueError(f"Unclassified features require a documented type: {sorted(undefined)}")
    return FeatureSpec(
        features=features,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        binary_features=binary_features,
        count_features=count_features,
        excluded_features=[id_col],
    )


def prepare_catboost_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: FeatureSpec,
    target: str = TARGET,
    add_count_features: bool = False,
) -> PreparedFeatures:
    X = train.loc[:, spec.features].copy()
    X_test = test.loc[:, spec.features].copy()
    y = train[target].astype(int).copy()

    for column in spec.categorical_features:
        X[column] = X[column].astype("string").fillna("__MISSING__").astype(str)
        X_test[column] = X_test[column].astype("string").fillna("__MISSING__").astype(str)

    if add_count_features:
        diagnosis_columns = [column for column in spec.features if column.startswith("dx2_")]
        procedure_columns = [column for column in spec.features if column.startswith("proc")]
        X["secondary_diagnosis_count"] = X[diagnosis_columns].apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1, min_count=1)
        X_test["secondary_diagnosis_count"] = X_test[diagnosis_columns].apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1, min_count=1)
        X["procedure_count"] = X[procedure_columns].apply(pd.to_numeric, errors="coerce").sum(
            axis=1, min_count=1
        )
        X_test["procedure_count"] = X_test[procedure_columns].apply(
            pd.to_numeric, errors="coerce"
        ).sum(axis=1, min_count=1)

    if list(X.columns) != list(X_test.columns):
        raise ValueError("Prepared train and test features are not aligned.")
    return PreparedFeatures(X=X, y=y, X_test=X_test, spec=spec)


def ensure_gpu_ready(devices: str = "0") -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed."
        raise RuntimeError(f"NVIDIA GPU is unavailable: {message}")

    probe_X = pd.DataFrame({"code": ["a", "b", "a", "b"], "value": [0.0, 1.0, 2.0, 3.0]})
    probe_y = [0, 1, 0, 1]
    try:
        CatBoostClassifier(
            iterations=1,
            task_type="GPU",
            devices=devices,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        ).fit(probe_X, probe_y, cat_features=["code"])
    except Exception as error:
        raise RuntimeError(f"CatBoost GPU probe failed: {error}") from error
    return result.stdout.strip()


def _validate_weighting(params: dict[str, Any]) -> None:
    weighting_options = ["class_weights", "auto_class_weights", "scale_pos_weight"]
    configured = [name for name in weighting_options if params.get(name) is not None]
    if len(configured) > 1:
        raise ValueError(f"Use only one class-weighting strategy, found: {configured}")


def train_catboost_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: list[str],
    cv: Any,
    params: dict[str, Any] | None = None,
    task_type: str = "GPU",
    devices: str = "0",
    early_stopping_rounds: int = 200,
    model_dir: Path | None = None,
    model_prefix: str = "catboost",
) -> dict[str, Any]:
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("X and X_test must have the same feature order.")
    if any(column not in X for column in categorical_features):
        raise ValueError("Every categorical feature must exist in X.")

    model_params = {**BASE_PARAMS, **(params or {})}
    _validate_weighting(model_params)
    model_params.update({"task_type": task_type, "devices": devices})
    y = pd.Series(y, index=X.index, dtype=int)
    oof_pred = np.zeros(len(X), dtype=float)
    fold_id = np.full(len(X), -1, dtype=int)
    test_fold_predictions: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    importance_records: list[pd.DataFrame] = []
    models: list[CatBoostClassifier] = []

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        model = CatBoostClassifier(**model_params)
        model.fit(
            X_train,
            y_train,
            cat_features=categorical_features,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )
        valid_pred = model.predict_proba(X_valid)[:, 1]
        oof_pred[valid_idx] = valid_pred
        fold_id[valid_idx] = fold
        test_fold_predictions.append(model.predict_proba(X_test)[:, 1])
        best_iteration = model.get_best_iteration()
        if best_iteration < 0:
            best_iteration = model.tree_count_ - 1
        metrics = evaluate_probabilities(y_valid, valid_pred)
        metrics.update(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                "train_fraud_prevalence": float(y_train.mean()),
                "valid_fraud_prevalence": float(y_valid.mean()),
                "best_iteration": int(best_iteration + 1),
            }
        )
        fold_metrics.append(metrics)
        importance_records.append(
            pd.DataFrame(
                {"feature": X.columns, "importance": model.get_feature_importance(), "fold": fold}
            )
        )
        if model_dir is not None:
            model.save_model(model_dir / f"{model_prefix}_fold_{fold}.cbm")
        models.append(model)

    if (fold_id < 0).any():
        raise RuntimeError("Every training row must receive exactly one OOF prediction.")
    if not np.isfinite(oof_pred).all() or not ((0 <= oof_pred) & (oof_pred <= 1)).all():
        raise RuntimeError("OOF predictions must be finite probabilities.")
    test_pred = np.mean(np.vstack(test_fold_predictions), axis=0)
    return {
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "models": models,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "fold_id": fold_id,
        "feature_importance": pd.concat(importance_records, ignore_index=True),
        "params": model_params,
    }


def aggregate_feature_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    return (
        feature_importance.groupby("feature", as_index=False)["importance"]
        .agg(mean_importance="mean", std_importance="std")
        .fillna({"std_importance": 0.0})
        .sort_values("mean_importance", ascending=False)
        .assign(rank=lambda frame: np.arange(1, len(frame) + 1))
        .reset_index(drop=True)
    )
