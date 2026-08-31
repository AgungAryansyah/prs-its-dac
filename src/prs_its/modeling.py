from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Iterable

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
INTERACTION_FEATURES = {
    "typeppk_cmg": ("typeppk", "cmg"),
    "cmg_severitylevel": ("cmg", "severitylevel"),
    "diagprimer_cmg": ("diagprimer", "cmg"),
}
SECONDARY_DIAGNOSIS_COUNT_BUCKET = "secondary_diagnosis_count_bucket"
PROCEDURE_COUNT_BUCKET = "procedure_count_bucket"
SECONDARY_DIAGNOSIS_ACTIVE_GROUP_COUNT = "secondary_diagnosis_active_group_count"
PROCEDURE_ACTIVE_GROUP_COUNT = "procedure_active_group_count"
SECONDARY_DIAGNOSIS_MAX_GROUP_COUNT = "secondary_diagnosis_max_group_count"
PROCEDURE_MAX_GROUP_COUNT = "procedure_max_group_count"
SECONDARY_DIAGNOSIS_CONCENTRATION = "secondary_diagnosis_concentration"
PROCEDURE_CONCENTRATION = "procedure_concentration"
CLINICAL_TOTAL_BURDEN = "clinical_total_burden"
PROCEDURE_BURDEN_SHARE = "procedure_burden_share"
CLINICAL_SHAPE_FEATURES = {
    "active_groups": (
        SECONDARY_DIAGNOSIS_ACTIVE_GROUP_COUNT,
        PROCEDURE_ACTIVE_GROUP_COUNT,
    ),
    "concentration": (
        SECONDARY_DIAGNOSIS_MAX_GROUP_COUNT,
        PROCEDURE_MAX_GROUP_COUNT,
        SECONDARY_DIAGNOSIS_CONCENTRATION,
        PROCEDURE_CONCENTRATION,
    ),
    "joint_burden": (
        CLINICAL_TOTAL_BURDEN,
        PROCEDURE_BURDEN_SHARE,
    ),
}
FREQUENCY_SOURCE_FEATURES = (
    "dati2",
    "kdkc",
    "typeppk",
    "cmg",
    "diagprimer",
    "dati2_typeppk",
)
FREQUENCY_RARE_SOURCE_FEATURES = ("dati2", "cmg", "diagprimer", "dati2_typeppk")
FREQUENCY_RARE_THRESHOLD = 25
BASE_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 10000,
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
    categorical_features: list[str]


@dataclass
class FrequencyFeatureTransformer:
    source_features: tuple[str, ...]
    mode: str
    rare_threshold: int = FREQUENCY_RARE_THRESHOLD
    frequency_maps: dict[str, dict[str, int]] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"count", "log_count", "rare_flag"}:
            raise ValueError("mode must be 'count', 'log_count', or 'rare_flag'.")
        if not self.source_features:
            raise ValueError("source_features must not be empty.")
        if self.rare_threshold < 0:
            raise ValueError("rare_threshold must be non-negative.")

    @property
    def output_features(self) -> list[str]:
        prefix = {
            "count": "frequency_count",
            "log_count": "frequency_log_count",
            "rare_flag": "frequency_rare",
        }[self.mode]
        return [f"{prefix}_{feature}" for feature in self.source_features]

    def fit(self, X: pd.DataFrame) -> FrequencyFeatureTransformer:
        self._validate_source_features(X)
        self.frequency_maps = {
            feature: self._feature_values(X[feature]).value_counts().astype(int).to_dict()
            for feature in self.source_features
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.frequency_maps is None:
            raise RuntimeError("FrequencyFeatureTransformer must be fitted before transform.")
        self._validate_source_features(X)
        transformed = X.copy()
        for source_feature, output_feature in zip(self.source_features, self.output_features, strict=True):
            counts = self._feature_values(X[source_feature]).map(
                self.frequency_maps[source_feature]
            ).fillna(0.0)
            if self.mode == "count":
                transformed[output_feature] = counts.astype(float)
            elif self.mode == "log_count":
                transformed[output_feature] = np.log1p(counts).astype(float)
            else:
                transformed[output_feature] = (counts <= self.rare_threshold).astype(int)
        return transformed

    def as_dict(self) -> dict[str, Any]:
        if self.frequency_maps is None:
            raise RuntimeError("FrequencyFeatureTransformer must be fitted before serialization.")
        return {
            "source_features": list(self.source_features),
            "mode": self.mode,
            "rare_threshold": self.rare_threshold,
            "frequency_maps": self.frequency_maps,
        }

    def save(self, path: Path) -> None:
        with path.open("w") as file:
            json.dump(self.as_dict(), file, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: Path) -> FrequencyFeatureTransformer:
        with path.open() as file:
            payload = json.load(file)
        return cls(
            source_features=tuple(payload["source_features"]),
            mode=payload["mode"],
            rare_threshold=int(payload["rare_threshold"]),
            frequency_maps={
                feature: {value: int(count) for value, count in counts.items()}
                for feature, counts in payload["frequency_maps"].items()
            },
        )

    @staticmethod
    def _feature_values(values: pd.Series) -> pd.Series:
        return values.astype("string").fillna("__MISSING__").astype(str)

    def _validate_source_features(self, X: pd.DataFrame) -> None:
        missing = [feature for feature in self.source_features if feature not in X]
        if missing:
            raise ValueError(f"Frequency source features are unavailable: {missing}")


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
    add_interaction_features: bool = False,
    add_los_features: bool = False,
    add_count_bucket_features: bool = False,
    clinical_shape_family: str | None = None,
    additional_interaction_features: dict[str, tuple[str, ...]] | None = None,
) -> PreparedFeatures:
    X = train.loc[:, spec.features].copy()
    X_test = test.loc[:, spec.features].copy()
    y = train[target].astype(int).copy()

    for column in spec.categorical_features:
        X[column] = X[column].astype("string").fillna("__MISSING__").astype(str)
        X_test[column] = X_test[column].astype("string").fillna("__MISSING__").astype(str)
    categorical_features = list(spec.categorical_features)

    if add_count_bucket_features and not add_count_features:
        raise ValueError("Count bucket features require add_count_features=True.")
    if clinical_shape_family is not None and not add_count_features:
        raise ValueError("Clinical-shape features require add_count_features=True.")
    if clinical_shape_family is not None and clinical_shape_family not in CLINICAL_SHAPE_FEATURES:
        raise ValueError(
            f"Unknown clinical-shape family: {clinical_shape_family!r}. "
            f"Expected one of {sorted(CLINICAL_SHAPE_FEATURES)}."
        )

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

    if clinical_shape_family is not None:
        for frame in (X, X_test):
            diagnosis_values = frame[diagnosis_columns].apply(pd.to_numeric, errors="coerce")
            procedure_values = frame[procedure_columns].apply(pd.to_numeric, errors="coerce")
            if clinical_shape_family == "active_groups":
                frame[SECONDARY_DIAGNOSIS_ACTIVE_GROUP_COUNT] = (
                    diagnosis_values.gt(0).sum(axis=1).where(diagnosis_values.notna().any(axis=1))
                )
                frame[PROCEDURE_ACTIVE_GROUP_COUNT] = (
                    procedure_values.gt(0).sum(axis=1).where(procedure_values.notna().any(axis=1))
                )
            elif clinical_shape_family == "concentration":
                diagnosis_maximum = diagnosis_values.max(axis=1)
                procedure_maximum = procedure_values.max(axis=1)
                frame[SECONDARY_DIAGNOSIS_MAX_GROUP_COUNT] = diagnosis_maximum
                frame[PROCEDURE_MAX_GROUP_COUNT] = procedure_maximum
                frame[SECONDARY_DIAGNOSIS_CONCENTRATION] = diagnosis_maximum.divide(
                    frame["secondary_diagnosis_count"].clip(lower=1)
                )
                frame[PROCEDURE_CONCENTRATION] = procedure_maximum.divide(
                    frame["procedure_count"].clip(lower=1)
                )
            else:
                total_burden = frame[["secondary_diagnosis_count", "procedure_count"]].sum(
                    axis=1, min_count=1
                )
                frame[CLINICAL_TOTAL_BURDEN] = total_burden
                frame[PROCEDURE_BURDEN_SHARE] = frame["procedure_count"].divide(
                    total_burden.clip(lower=1)
                )

    if add_count_bucket_features:
        for frame in (X, X_test):
            frame[SECONDARY_DIAGNOSIS_COUNT_BUCKET] = (
                pd.cut(
                    frame["secondary_diagnosis_count"],
                    bins=[-np.inf, 0, 1, 2, np.inf],
                    labels=["0", "1", "2", "3+"],
                    include_lowest=True,
                )
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )
            frame[PROCEDURE_COUNT_BUCKET] = (
                pd.cut(
                    frame["procedure_count"],
                    bins=[-np.inf, 0, 1, 2, 3, np.inf],
                    labels=["0", "1", "2", "3", "4+"],
                    include_lowest=True,
                )
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )
        categorical_features.extend(
            [SECONDARY_DIAGNOSIS_COUNT_BUCKET, PROCEDURE_COUNT_BUCKET]
        )

    interaction_features = {}
    if add_interaction_features:
        interaction_features.update(INTERACTION_FEATURES)
    if additional_interaction_features:
        interaction_features.update(additional_interaction_features)
    for name, columns in interaction_features.items():
        if name in X:
            raise ValueError(f"Interaction feature already exists: {name}.")
        if any(column not in X for column in columns):
            raise ValueError(f"Interaction {name!r} references an unavailable feature.")
        for frame in (X, X_test):
            frame[name] = (
                frame.loc[:, list(columns)]
                .astype("string")
                .fillna("__MISSING__")
                .agg("__".join, axis=1)
                .astype(str)
            )
        categorical_features.append(name)

    if add_los_features:
        for frame in (X, X_test):
            los = pd.to_numeric(frame["los"], errors="coerce")
            frame["los_zero_indicator"] = np.where(los.notna(), (los == 0).astype(int), np.nan)
            frame["los_bucket"] = (
                pd.cut(
                    los,
                    bins=[-np.inf, 0, 1, 3, 7, 14, np.inf],
                    labels=["0", "1", "2-3", "4-7", "8-14", "15+"],
                    include_lowest=True,
                )
                .astype("string")
                .fillna("__MISSING__")
                .astype(str)
            )
        categorical_features.append("los_bucket")

    if list(X.columns) != list(X_test.columns):
        raise ValueError("Prepared train and test features are not aligned.")
    return PreparedFeatures(
        X=X,
        y=y,
        X_test=X_test,
        spec=spec,
        categorical_features=categorical_features,
    )


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


def feature_signature_groups(X: pd.DataFrame) -> np.ndarray:
    if ID_COL in X or TARGET in X:
        raise ValueError("Feature signatures must not include claim_id or label.")
    return pd.util.hash_pandas_object(X, index=False).to_numpy(dtype=np.uint64)


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
    verbose: int | bool = False,
    progress_callback: Callable[[str, int], None] | None = None,
    groups: np.ndarray | pd.Series | None = None,
    predict_test: bool = True,
    fold_transformer_factory: Callable[[], FrequencyFeatureTransformer] | None = None,
) -> dict[str, Any]:
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("X and X_test must have the same feature order.")
    if any(column not in X for column in categorical_features):
        raise ValueError("Every categorical feature must exist in X.")
    if groups is not None and len(groups) != len(X):
        raise ValueError("groups must have the same length as X.")

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
    fold_transformers: list[FrequencyFeatureTransformer] | None = (
        [] if fold_transformer_factory is not None else None
    )
    model_features: list[str] | None = None

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    split_iterator = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for fold, (train_idx, valid_idx) in enumerate(split_iterator):
        if progress_callback is not None:
            progress_callback("start", fold)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        if fold_transformer_factory is None:
            X_train_model = X_train
            X_valid_model = X_valid
            X_test_model = X_test if predict_test else None
        else:
            transformer = fold_transformer_factory().fit(X_train)
            X_train_model = transformer.transform(X_train)
            X_valid_model = transformer.transform(X_valid)
            X_test_model = transformer.transform(X_test) if predict_test else None
            if model_dir is not None:
                transformer.save(model_dir / f"{model_prefix}_frequency_fold_{fold}.json")
            fold_transformers.append(transformer)
        if model_features is None:
            model_features = list(X_train_model.columns)
        if list(X_train_model.columns) != list(X_valid_model.columns):
            raise RuntimeError("Fold feature transformer produced mismatched train and validation schemas.")
        if X_test_model is not None and list(X_train_model.columns) != list(X_test_model.columns):
            raise RuntimeError("Fold feature transformer produced mismatched train and test schemas.")
        model = CatBoostClassifier(**model_params)
        model.fit(
            X_train_model,
            y_train,
            cat_features=categorical_features,
            eval_set=(X_valid_model, y_valid),
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        valid_pred = model.predict_proba(X_valid_model)[:, 1]
        oof_pred[valid_idx] = valid_pred
        fold_id[valid_idx] = fold
        if predict_test:
            test_fold_predictions.append(model.predict_proba(X_test_model)[:, 1])
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
                "iteration_cap": int(model_params["iterations"]),
                "hit_iteration_cap": bool(best_iteration + 1 >= model_params["iterations"]),
            }
        )
        if groups is not None:
            train_groups = np.asarray(groups)[train_idx]
            valid_groups = np.asarray(groups)[valid_idx]
            overlap_count = len(set(train_groups).intersection(valid_groups))
            if overlap_count:
                raise RuntimeError(f"Fold {fold} has {overlap_count} overlapping feature groups.")
            metrics.update(
                {
                    "train_group_count": int(len(np.unique(train_groups))),
                    "valid_group_count": int(len(np.unique(valid_groups))),
                    "group_overlap_count": overlap_count,
                }
            )
        fold_metrics.append(metrics)
        importance_records.append(
            pd.DataFrame(
                {
                    "feature": X_train_model.columns,
                    "importance": model.get_feature_importance(),
                    "fold": fold,
                }
            )
        )
        if model_dir is not None:
            model.save_model(model_dir / f"{model_prefix}_fold_{fold}.cbm")
        models.append(model)
        if progress_callback is not None:
            progress_callback("complete", fold)

    if (fold_id < 0).any():
        raise RuntimeError("Every training row must receive exactly one OOF prediction.")
    if not np.isfinite(oof_pred).all() or not ((0 <= oof_pred) & (oof_pred <= 1)).all():
        raise RuntimeError("OOF predictions must be finite probabilities.")
    test_fold_predictions_array = np.vstack(test_fold_predictions) if predict_test else None
    test_pred = np.mean(test_fold_predictions_array, axis=0) if predict_test else None
    return {
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "test_fold_predictions": test_fold_predictions_array,
        "models": models,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "fold_id": fold_id,
        "feature_importance": pd.concat(importance_records, ignore_index=True),
        "model_features": model_features or list(X.columns),
        "fold_transformers": fold_transformers,
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
