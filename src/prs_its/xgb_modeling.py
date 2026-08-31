from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from prs_its.metrics import evaluate_probabilities
from prs_its.modeling import (
    ID_COL,
    INTERACTION_FEATURES,
    RANDOM_STATE,
    TARGET,
    FeatureSpec,
    PreparedFeatures,
    prepare_catboost_features,
)


XGB_BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_estimators": 10_000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 10.0,
    "random_state": RANDOM_STATE,
}
XGB_INTERACTION_FEATURES = {
    "dati2_typeppk": ("dati2", "typeppk"),
    "diagprimer_cmg": INTERACTION_FEATURES["diagprimer_cmg"],
    "cmg_severitylevel": INTERACTION_FEATURES["cmg_severitylevel"],
}
MISSING_CATEGORY = "__MISSING__"


@dataclass(frozen=True)
class TargetEncodingConfig:
    categorical_features: tuple[str, ...]
    inner_n_splits: int = 5
    smooth: float = 20.0
    random_state: int = RANDOM_STATE
    add_support_counts: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class FoldTargetEncoderTransformer:
    def __init__(
        self,
        categorical_features: list[str] | tuple[str, ...],
        inner_n_splits: int = 5,
        smooth: float = 20.0,
        random_state: int = RANDOM_STATE,
        add_support_counts: bool = False,
    ) -> None:
        if inner_n_splits < 2:
            raise ValueError("inner_n_splits must be at least 2.")
        if smooth < 0:
            raise ValueError("smooth must be non-negative.")
        self.config = TargetEncodingConfig(
            categorical_features=tuple(categorical_features),
            inner_n_splits=inner_n_splits,
            smooth=smooth,
            random_state=random_state,
            add_support_counts=add_support_counts,
        )
        self.encoder: TargetEncoder | None = None
        self.global_rate: float | None = None
        self.support_maps: dict[str, dict[str, int]] = {}
        self.feature_names: list[str] | None = None

    @property
    def categorical_features(self) -> list[str]:
        return list(self.config.categorical_features)

    @property
    def encoded_features(self) -> list[str]:
        return [f"te_{feature}" for feature in self.config.categorical_features]

    @property
    def support_count_features(self) -> list[str]:
        return [f"support_count_{feature}" for feature in self.config.categorical_features]

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> FoldTargetEncoderTransformer:
        categories, labels = self._validate_fit_input(X, y)
        self.global_rate = float(labels.mean())
        self.support_maps = self._support_maps(categories)
        if self.categorical_features:
            self.encoder = self._make_encoder(labels)
            self.encoder.fit(categories, labels)
        self.feature_names = list(self._numeric_features(X).columns) + self.encoded_features
        if self.config.add_support_counts:
            self.feature_names.extend(self.support_count_features)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        categories, labels = self._validate_fit_input(X, y)
        self.global_rate = float(labels.mean())
        self.support_maps = self._support_maps(categories)
        numeric = self._numeric_features(X)
        if self.categorical_features:
            self.encoder = self._make_encoder(labels)
            encoded = self._encoded_frame(self.encoder.fit_transform(categories, labels), X.index)
        else:
            encoded = pd.DataFrame(index=X.index)
        transformed = pd.concat([numeric, encoded, self._support_frame(categories, X.index)], axis=1)
        self.feature_names = list(transformed.columns)
        return transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.global_rate is None or self.feature_names is None:
            raise RuntimeError("FoldTargetEncoderTransformer must be fitted before transform.")
        self._validate_columns(X)
        numeric = self._numeric_features(X)
        categories = self._categorical_frame(X)
        if self.categorical_features:
            if self.encoder is None:
                raise RuntimeError("Target encoder is unavailable after fitting.")
            encoded = self._encoded_frame(self.encoder.transform(categories), X.index)
        else:
            encoded = pd.DataFrame(index=X.index)
        transformed = pd.concat([numeric, encoded, self._support_frame(categories, X.index)], axis=1)
        if list(transformed.columns) != self.feature_names:
            raise RuntimeError("Target encoder produced a mismatched feature schema.")
        return transformed

    def save(self, path: Path, seed: int, fold: int) -> None:
        if self.feature_names is None:
            raise RuntimeError("FoldTargetEncoderTransformer must be fitted before serialization.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "transformer": self,
                "seed": seed,
                "fold": fold,
                "metadata": self.as_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> FoldTargetEncoderTransformer:
        payload = joblib.load(path)
        transformer = payload.get("transformer") if isinstance(payload, dict) else None
        if not isinstance(transformer, cls):
            raise ValueError("Target encoder artifact does not contain a compatible transformer.")
        return transformer

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.config.as_dict(),
            "encoded_features": self.encoded_features,
            "support_count_features": self.support_count_features,
            "global_rate": self.global_rate,
            "feature_names": self.feature_names,
        }

    def _validate_fit_input(
        self, X: pd.DataFrame, y: pd.Series | np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray]:
        self._validate_columns(X)
        labels = np.asarray(y, dtype=int).reshape(-1)
        if len(labels) != len(X):
            raise ValueError("X and y must have the same length.")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("y must contain only binary labels.")
        if len(labels) == 0:
            raise ValueError("X and y must not be empty.")
        return self._categorical_frame(X), labels

    def _make_encoder(self, labels: np.ndarray) -> TargetEncoder:
        class_counts = np.bincount(labels, minlength=2)
        n_splits = min(self.config.inner_n_splits, int(class_counts.min()))
        if n_splits < 2:
            raise ValueError("Each outer training partition needs at least two rows from each class.")
        return TargetEncoder(
            target_type="binary",
            smooth=self.config.smooth,
            cv=StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.config.random_state,
            ),
        )

    def _validate_columns(self, X: pd.DataFrame) -> None:
        missing = [feature for feature in self.categorical_features if feature not in X]
        if missing:
            raise ValueError(f"Categorical target-encoding features are unavailable: {missing}")
        if ID_COL in X or TARGET in X:
            raise ValueError("claim_id and label must not be target-encoder inputs.")

    def _categorical_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                feature: X[feature].astype("string").fillna(MISSING_CATEGORY).astype(str)
                for feature in self.categorical_features
            },
            index=X.index,
        )

    def _numeric_features(self, X: pd.DataFrame) -> pd.DataFrame:
        numeric = X.drop(columns=self.categorical_features).copy()
        for column in numeric:
            numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
        return numeric.astype(float)

    def _support_maps(self, categories: pd.DataFrame) -> dict[str, dict[str, int]]:
        if not self.config.add_support_counts:
            return {}
        return {
            feature: categories[feature].value_counts().astype(int).to_dict()
            for feature in self.categorical_features
        }

    def _encoded_frame(self, values: np.ndarray, index: pd.Index) -> pd.DataFrame:
        if self.global_rate is None:
            raise RuntimeError("Target encoder global rate is unavailable.")
        encoded = pd.DataFrame(values, index=index, columns=self.encoded_features, dtype=float)
        return encoded.fillna(self.global_rate)

    def _support_frame(self, categories: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
        if not self.config.add_support_counts:
            return pd.DataFrame(index=index)
        return pd.DataFrame(
            {
                output_feature: categories[source_feature]
                .map(self.support_maps[source_feature])
                .fillna(0.0)
                .astype(float)
                for source_feature, output_feature in zip(
                    self.categorical_features, self.support_count_features, strict=True
                )
            },
            index=index,
        )


def prepare_xgb_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: FeatureSpec,
    add_interaction_features: bool = False,
) -> PreparedFeatures:
    return prepare_catboost_features(
        train,
        test,
        spec,
        add_count_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features=(XGB_INTERACTION_FEATURES if add_interaction_features else None),
    )


def ensure_xgb_gpu_ready() -> str:
    probe_X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0], "count": [1, 0, 1, 0]})
    probe_y = np.array([0, 1, 0, 1])
    try:
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            n_estimators=1,
            max_depth=1,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(probe_X, probe_y, verbose=False)
        config = json.loads(model.get_booster().save_config())
        device = config["learner"]["generic_param"].get("device", "")
        if not device.startswith("cuda"):
            raise RuntimeError(f"XGBoost selected {device or 'an unknown device'} instead of CUDA.")
    except Exception as error:
        raise RuntimeError(
            "CUDA XGBoost is unavailable. Install a CUDA-enabled XGBoost build or use --task-type CPU. "
            f"Probe detail: {error}"
        ) from error
    return device


def train_xgb_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: list[str],
    cv: Any,
    params: dict[str, Any] | None = None,
    task_type: str = "GPU",
    early_stopping_rounds: int = 200,
    model_dir: Path | None = None,
    model_prefix: str = "xgb",
    verbose: int | bool = False,
    progress_callback: Callable[[str, int], None] | None = None,
    groups: np.ndarray | pd.Series | None = None,
    predict_test: bool = True,
    target_encoder_factory: Callable[[], FoldTargetEncoderTransformer] | None = None,
) -> dict[str, Any]:
    task_type = task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be positive.")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("X and X_test must have the same feature order.")
    if any(feature not in X for feature in categorical_features):
        raise ValueError("Every categorical feature must exist in X.")
    if ID_COL in X or TARGET in X:
        raise ValueError("claim_id and label must not be model features.")
    if groups is not None and len(groups) != len(X):
        raise ValueError("groups must have the same length as X.")

    model_params = {**XGB_BASE_PARAMS, **(params or {})}
    model_params.update({"tree_method": "hist", "device": "cuda" if task_type == "GPU" else "cpu"})
    y = pd.Series(y, index=X.index, dtype=int)
    oof_pred = np.zeros(len(X), dtype=float)
    fold_id = np.full(len(X), -1, dtype=int)
    test_fold_predictions: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    importance_records: list[pd.DataFrame] = []
    models: list[XGBClassifier] = []
    fold_transformers: list[FoldTargetEncoderTransformer] = []
    model_features: list[str] | None = None

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
    split_iterator = cv.split(X, y, groups) if groups is not None else cv.split(X, y)
    for fold, (train_idx, valid_idx) in enumerate(split_iterator):
        if progress_callback is not None:
            progress_callback("start", fold)
        transformer = (
            target_encoder_factory()
            if target_encoder_factory is not None
            else FoldTargetEncoderTransformer(categorical_features)
        )
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]
        X_train_model = transformer.fit_transform(X_train, y_train)
        X_valid_model = transformer.transform(X_valid)
        X_test_model = transformer.transform(X_test) if predict_test else None
        if model_features is None:
            model_features = list(X_train_model.columns)
        if list(X_train_model.columns) != list(X_valid_model.columns):
            raise RuntimeError("Target encoder produced mismatched train and validation schemas.")
        if X_test_model is not None and list(X_train_model.columns) != list(X_test_model.columns):
            raise RuntimeError("Target encoder produced mismatched train and test schemas.")
        model = XGBClassifier(
            **model_params,
            callbacks=[EarlyStopping(rounds=early_stopping_rounds, save_best=True)],
        )
        model.fit(X_train_model, y_train, eval_set=[(X_valid_model, y_valid)], verbose=verbose)
        valid_pred = model.predict_proba(X_valid_model)[:, 1]
        oof_pred[valid_idx] = valid_pred
        fold_id[valid_idx] = fold
        if predict_test:
            test_fold_predictions.append(model.predict_proba(X_test_model)[:, 1])
        best_iteration = _best_iteration(model)
        metrics = evaluate_probabilities(y_valid, valid_pred)
        metrics.update(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                "train_fraud_prevalence": float(y_train.mean()),
                "valid_fraud_prevalence": float(y_valid.mean()),
                "best_iteration": best_iteration + 1,
                "iteration_cap": int(model_params["n_estimators"]),
                "hit_iteration_cap": bool(best_iteration + 1 >= model_params["n_estimators"]),
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
                    "importance": np.asarray(model.feature_importances_, dtype=float),
                    "fold": fold,
                }
            )
        )
        if model_dir is not None:
            model.save_model(model_dir / f"{model_prefix}_fold_{fold}.json")
            transformer.save(
                model_dir / f"{model_prefix}_target_encoder_fold_{fold}.joblib",
                seed=int(model_params["random_state"]),
                fold=fold,
            )
        models.append(model)
        fold_transformers.append(transformer)
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
        "model_features": model_features or [],
        "fold_transformers": fold_transformers,
        "params": model_params,
    }


def _best_iteration(model: XGBClassifier) -> int:
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        return int(best_iteration)
    return int(model.get_booster().num_boosted_rounds() - 1)
