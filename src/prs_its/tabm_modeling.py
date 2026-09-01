from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import QuantileTransformer
import torch
from torch import nn
from torch.nn import functional as F
from tabm import TabM
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins

from prs_its.metrics import evaluate_probabilities
from prs_its.modeling import PreparedFeatures, prepare_catboost_features


TABM_VARIANTS = ("tabm_base", "tabm_piecewise")


@dataclass(frozen=True)
class TabMParams:
    variant: str
    k: int = 8
    d_block: int = 256
    n_blocks: int = 3
    dropout: float = 0.1
    learning_rate: float = 0.002
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 500
    patience: int = 20
    inner_validation_fraction: float = 0.1
    piecewise_bins: int = 16
    piecewise_embedding_dim: int = 16

    def __post_init__(self) -> None:
        if self.variant not in TABM_VARIANTS:
            raise ValueError(f"variant must be one of {TABM_VARIANTS}.")
        if self.k < 1 or self.d_block < 1 or self.n_blocks < 1:
            raise ValueError("TabM k, d_block, and n_blocks must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("TabM dropout must be within [0, 1).")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("TabM optimizer settings are invalid.")
        if self.batch_size < 1 or self.max_epochs < 1 or self.patience < 1:
            raise ValueError("TabM batch_size, max_epochs, and patience must be positive.")
        if not 0 < self.inner_validation_fraction < 0.5:
            raise ValueError("inner_validation_fraction must be within (0, 0.5).")
        if self.piecewise_bins < 2 or self.piecewise_embedding_dim < 1:
            raise ValueError("Piecewise embedding settings are invalid.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FoldTabMPreprocessor:
    categorical_features: list[str]
    random_state: int
    numeric_features: list[str] | None = None
    category_maps: dict[str, dict[str, int]] | None = None
    medians: dict[str, float] | None = None
    quantile_transformer: QuantileTransformer | None = None
    quantile_feature_indices: list[int] | None = None
    missing_indicator_indices: list[int] | None = None
    numeric_output_names: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> FoldTabMPreprocessor:
        missing = [feature for feature in self.categorical_features if feature not in X]
        if missing:
            raise ValueError(f"Categorical features are unavailable: {missing}")
        self.numeric_features = [feature for feature in X.columns if feature not in self.categorical_features]
        if not self.numeric_features and not self.categorical_features:
            raise ValueError("TabM needs at least one feature.")
        self.category_maps = {}
        for feature in self.categorical_features:
            values = _categorical_values(X[feature])
            categories = sorted(values.unique().tolist())
            self.category_maps[feature] = {
                category: index for index, category in enumerate(categories, start=1)
            }
        self.medians = {}
        if self.numeric_features:
            numeric = self._numeric_values(X)
            for index, feature in enumerate(self.numeric_features):
                values = numeric[:, index]
                median = float(np.median(values[np.isfinite(values)])) if np.isfinite(values).any() else 0.0
                self.medians[feature] = median
            imputed = self._impute_numeric(numeric)
            missing = ~np.isfinite(numeric)
            self.quantile_feature_indices = [
                index for index in range(imputed.shape[1]) if np.unique(imputed[:, index]).size > 1
            ]
            self.missing_indicator_indices = [
                index for index in range(missing.shape[1]) if np.unique(missing[:, index]).size > 1
            ]
            self.numeric_output_names = [
                self.numeric_features[index] for index in self.quantile_feature_indices
            ] + [
                f"{self.numeric_features[index]}__missing"
                for index in self.missing_indicator_indices
            ]
            if self.quantile_feature_indices:
                self.quantile_transformer = QuantileTransformer(
                    n_quantiles=min(1000, len(imputed)),
                    output_distribution="normal",
                    random_state=self.random_state,
                ).fit(imputed[:, self.quantile_feature_indices])
            else:
                self.quantile_transformer = None
        else:
            self.quantile_feature_indices = []
            self.missing_indicator_indices = []
            self.numeric_output_names = []
            self.quantile_transformer = None
        if not self.numeric_output_names and not self.categorical_features:
            raise ValueError("TabM needs at least one non-constant feature.")
        return self

    @property
    def cat_cardinalities(self) -> list[int]:
        self._require_fitted()
        assert self.category_maps is not None
        return [len(self.category_maps[feature]) + 1 for feature in self.categorical_features]

    @property
    def numeric_output_features(self) -> list[str]:
        self._require_fitted()
        assert self.numeric_output_names is not None
        return self.numeric_output_names

    @property
    def model_features(self) -> list[str]:
        return [*self.numeric_output_features, *self.categorical_features]

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_fitted()
        assert self.numeric_features is not None
        assert self.category_maps is not None
        if list(X.columns) != [*self.numeric_features, *self.categorical_features] and set(X.columns) != set(
            [*self.numeric_features, *self.categorical_features]
        ):
            raise ValueError("TabM preprocessor received an incompatible feature schema.")
        if self.numeric_features:
            numeric = self._numeric_values(X)
            missing = (~np.isfinite(numeric)).astype(np.float32)
            imputed = self._impute_numeric(numeric)
            assert self.quantile_feature_indices is not None
            assert self.missing_indicator_indices is not None
            parts = []
            if self.quantile_feature_indices:
                assert self.quantile_transformer is not None
                parts.append(
                    self.quantile_transformer.transform(imputed[:, self.quantile_feature_indices])
                )
            if self.missing_indicator_indices:
                parts.append(missing[:, self.missing_indicator_indices])
            x_num = (
                np.column_stack(parts).astype(np.float32)
                if parts
                else np.empty((len(X), 0), dtype=np.float32)
            )
        else:
            x_num = np.empty((len(X), 0), dtype=np.float32)
        if not self.categorical_features:
            return x_num, None
        encoded = np.column_stack(
            [
                _categorical_values(X[feature]).map(self.category_maps[feature]).fillna(0).to_numpy(
                    dtype=np.int64
                )
                for feature in self.categorical_features
            ]
        )
        return x_num, encoded

    def save(self, path: Path, seed: int, fold: int) -> None:
        self._require_fitted()
        joblib.dump({"preprocessor": self, "seed": seed, "fold": fold}, path)

    @classmethod
    def load(cls, path: Path) -> FoldTabMPreprocessor:
        payload = joblib.load(path)
        preprocessor = payload.get("preprocessor") if isinstance(payload, dict) else None
        if not isinstance(preprocessor, cls):
            raise ValueError("TabM preprocessor artifact is invalid.")
        return preprocessor

    def _numeric_values(self, X: pd.DataFrame) -> np.ndarray:
        assert self.numeric_features is not None
        return X.loc[:, self.numeric_features].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=float
        )

    def _impute_numeric(self, numeric: np.ndarray) -> np.ndarray:
        assert self.numeric_features is not None
        assert self.medians is not None
        imputed = numeric.copy()
        for index, feature in enumerate(self.numeric_features):
            imputed[~np.isfinite(imputed[:, index]), index] = self.medians[feature]
        return imputed

    def _require_fitted(self) -> None:
        if (
            self.numeric_features is None
            or self.category_maps is None
            or self.medians is None
            or self.quantile_feature_indices is None
            or self.missing_indicator_indices is None
            or self.numeric_output_names is None
        ):
            raise RuntimeError("FoldTabMPreprocessor must be fitted before use.")


def prepare_tabm_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec,
) -> PreparedFeatures:
    return prepare_catboost_features(
        train,
        test,
        spec,
        add_count_features=True,
        add_interaction_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features={"dati2_typeppk": ("dati2", "typeppk")},
    )


def ensure_tabm_gpu_ready() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is unavailable. Install a CUDA-enabled Torch build or use --task-type CPU.")
    device = torch.device("cuda")
    try:
        model = TabM.make(
            n_num_features=2,
            cat_cardinalities=[3],
            d_out=1,
            k=2,
            d_block=32,
            n_blocks=1,
            dropout=0.0,
        ).to(device)
        output = model(
            torch.randn(4, 2, device=device),
            torch.tensor([[0], [1], [2], [0]], device=device),
        )
        output.sum().backward()
        if output.device.type != "cuda":
            raise RuntimeError(f"TabM CUDA probe ran on {output.device.type} instead of CUDA.")
    except Exception as error:
        raise RuntimeError(f"TabM CUDA probe failed: {error}") from error
    return f"{torch.cuda.get_device_name(device)}; torch={torch.__version__}; cuda={torch.version.cuda}"


def train_tabm_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    categorical_features: list[str],
    cv: Any,
    params: TabMParams,
    seed: int,
    task_type: str = "GPU",
    model_dir: Path | None = None,
    model_prefix: str = "tabm",
    progress_callback: Callable[[str, int], None] | None = None,
    groups: np.ndarray | pd.Series | None = None,
    predict_test: bool = True,
    compute_feature_importance: bool = False,
) -> dict[str, Any]:
    task_type = task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("X and X_test must have the same feature order.")
    if any(feature not in X for feature in categorical_features):
        raise ValueError("Every categorical feature must exist in X.")
    if groups is not None and len(groups) != len(X):
        raise ValueError("groups must have the same length as X.")
    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if task_type == "GPU" else "cpu")
    labels = pd.Series(y, index=X.index, dtype=int)
    oof_pred = np.full(len(X), np.nan, dtype=float)
    fold_id = np.full(len(X), -1, dtype=int)
    test_fold_predictions: list[np.ndarray] = []
    fold_metrics: list[dict[str, Any]] = []
    importance_records: list[pd.DataFrame] = []
    models: list[nn.Module] = []
    preprocessors: list[FoldTabMPreprocessor] = []
    model_features: list[str] | None = None
    fold_model_features: dict[int, list[str]] = {}
    split_iterator = cv.split(X, labels, groups) if groups is not None else cv.split(X, labels)

    for fold, (train_idx, valid_idx) in enumerate(split_iterator):
        if progress_callback is not None:
            progress_callback("start", fold)
        _set_seed(seed + fold)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = labels.iloc[train_idx].to_numpy(), labels.iloc[valid_idx].to_numpy()
        preprocessor = FoldTabMPreprocessor(categorical_features, random_state=seed + fold).fit(X_train)
        x_train_num, x_train_cat = preprocessor.transform(X_train)
        x_valid_num, x_valid_cat = preprocessor.transform(X_valid)
        x_test_num, x_test_cat = preprocessor.transform(X_test) if predict_test else (None, None)
        if model_features is None:
            model_features = preprocessor.model_features
        fold_model_features[fold] = preprocessor.model_features
        model = _make_model(x_train_num, preprocessor.cat_cardinalities, params).to(device)
        best_epoch, best_validation_ap = _fit_model(
            model,
            x_train_num,
            x_train_cat,
            y_train,
            params,
            seed + fold,
            device,
        )
        valid_pred = _predict_probabilities(model, x_valid_num, x_valid_cat, params.batch_size, device)
        oof_pred[valid_idx] = valid_pred
        fold_id[valid_idx] = fold
        if predict_test:
            assert x_test_num is not None
            test_fold_predictions.append(
                _predict_probabilities(model, x_test_num, x_test_cat, params.batch_size, device)
            )
        metrics = evaluate_probabilities(y_valid, valid_pred)
        metrics.update(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "valid_size": len(valid_idx),
                "train_fraud_prevalence": float(y_train.mean()),
                "valid_fraud_prevalence": float(y_valid.mean()),
                "best_epoch": best_epoch,
                "best_validation_average_precision": best_validation_ap,
                "epoch_cap": params.max_epochs,
                "hit_epoch_cap": bool(best_epoch >= params.max_epochs),
            }
        )
        if groups is not None:
            train_groups = np.asarray(groups)[train_idx]
            valid_groups = np.asarray(groups)[valid_idx]
            overlap_count = len(set(train_groups).intersection(valid_groups))
            if overlap_count:
                raise RuntimeError(f"Fold {fold} has {overlap_count} overlapping groups.")
            metrics.update(
                {
                    "train_group_count": int(len(np.unique(train_groups))),
                    "valid_group_count": int(len(np.unique(valid_groups))),
                    "group_overlap_count": overlap_count,
                }
            )
        fold_metrics.append(metrics)
        if compute_feature_importance:
            importance_records.append(
                _permutation_importance(
                    model,
                    x_valid_num,
                    x_valid_cat,
                    y_valid,
                    preprocessor,
                    params.batch_size,
                    device,
                    seed + fold,
                    fold,
                )
            )
        if model_dir is not None:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "params": params.as_dict(),
                    "numeric_features": preprocessor.numeric_output_features,
                    "categorical_features": preprocessor.categorical_features,
                    "cat_cardinalities": preprocessor.cat_cardinalities,
                },
                model_dir / f"{model_prefix}_fold_{fold}.pt",
            )
            preprocessor.save(
                model_dir / f"{model_prefix}_preprocessor_fold_{fold}.joblib",
                seed=seed,
                fold=fold,
            )
        models.append(model)
        preprocessors.append(preprocessor)
        if progress_callback is not None:
            progress_callback("complete", fold)

    if (fold_id < 0).any():
        raise RuntimeError("Every training row must receive exactly one OOF prediction.")
    if not np.isfinite(oof_pred).all() or ((oof_pred < 0) | (oof_pred > 1)).any():
        raise RuntimeError("TabM OOF predictions must be finite probabilities.")
    test_fold_predictions_array = np.vstack(test_fold_predictions) if predict_test else None
    test_pred = np.mean(test_fold_predictions_array, axis=0) if predict_test else None
    feature_importance = (
        pd.concat(importance_records, ignore_index=True)
        if importance_records
        else pd.DataFrame(columns=["feature", "importance", "fold"])
    )
    return {
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "test_fold_predictions": test_fold_predictions_array,
        "models": models,
        "fold_preprocessors": preprocessors,
        "fold_metrics": pd.DataFrame(fold_metrics),
        "fold_id": fold_id,
        "feature_importance": feature_importance,
        "model_features": model_features or [],
        "fold_model_features": fold_model_features,
        "params": params.as_dict(),
        "seed": seed,
    }


def _make_model(
    x_num: np.ndarray,
    cat_cardinalities: list[int],
    params: TabMParams,
) -> nn.Module:
    num_embeddings = None
    if params.variant == "tabm_piecewise" and x_num.shape[1] > 0:
        n_bins = min(params.piecewise_bins, len(x_num))
        if n_bins < 2:
            raise ValueError("Piecewise TabM needs at least two outer-training rows.")
        bins = compute_bins(torch.as_tensor(x_num, dtype=torch.float32), n_bins=n_bins)
        num_embeddings = PiecewiseLinearEmbeddings(
            bins,
            d_embedding=params.piecewise_embedding_dim,
            activation=False,
            version="B",
        )
    return TabM.make(
        n_num_features=x_num.shape[1],
        cat_cardinalities=cat_cardinalities or None,
        d_out=1,
        num_embeddings=num_embeddings,
        k=params.k,
        d_block=params.d_block,
        n_blocks=params.n_blocks,
        dropout=params.dropout,
    )


def _fit_model(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray | None,
    labels: np.ndarray,
    params: TabMParams,
    seed: int,
    device: torch.device,
) -> tuple[int, float]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=params.inner_validation_fraction,
        random_state=seed,
    )
    inner_train_idx, inner_valid_idx = next(splitter.split(x_num, labels))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)
    random = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_score = -np.inf
    stale_epochs = 0
    for epoch in range(1, params.max_epochs + 1):
        model.train()
        for batch_idx in _batches(random.permutation(inner_train_idx), params.batch_size):
            x_num_batch = torch.as_tensor(x_num[batch_idx], dtype=torch.float32, device=device)
            x_cat_batch = _cat_tensor(x_cat, batch_idx, device)
            y_batch = torch.as_tensor(labels[batch_idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(x_num_batch, x_cat_batch).squeeze(-1)
                target = y_batch.unsqueeze(1).expand_as(logits)
                loss = F.binary_cross_entropy_with_logits(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        validation_pred = _predict_probabilities(
            model,
            x_num[inner_valid_idx],
            _select_cat(x_cat, inner_valid_idx),
            params.batch_size,
            device,
        )
        score = float(average_precision_score(labels[inner_valid_idx], validation_pred))
        if score > best_score + 1e-12:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= params.patience:
                break
    if best_state is None:
        raise RuntimeError("TabM early stopping did not retain a model state.")
    model.load_state_dict(best_state)
    return best_epoch, best_score


def _predict_probabilities(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for batch_idx in _batches(np.arange(len(x_num)), batch_size):
            x_num_batch = torch.as_tensor(x_num[batch_idx], dtype=torch.float32, device=device)
            x_cat_batch = _cat_tensor(x_cat, batch_idx, device)
            logits = model(x_num_batch, x_cat_batch).squeeze(-1).float()
            predictions.append(torch.sigmoid(logits).mean(dim=1).cpu().numpy())
    probabilities = np.concatenate(predictions)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise RuntimeError("TabM predictions must be finite probabilities.")
    return probabilities.astype(float)


def _permutation_importance(
    model: nn.Module,
    x_num: np.ndarray,
    x_cat: np.ndarray | None,
    labels: np.ndarray,
    preprocessor: FoldTabMPreprocessor,
    batch_size: int,
    device: torch.device,
    seed: int,
    fold: int,
) -> pd.DataFrame:
    random = np.random.default_rng(seed)
    sample_size = min(2000, len(labels))
    sampled = np.sort(random.choice(len(labels), size=sample_size, replace=False))
    sample_num = x_num[sampled].copy()
    sample_cat = _select_cat(x_cat, sampled)
    sample_labels = labels[sampled]
    baseline = float(average_precision_score(sample_labels, _predict_probabilities(
        model, sample_num, sample_cat, batch_size, device
    )))
    rows = []
    for index, feature in enumerate(preprocessor.numeric_output_features):
        permuted_num = sample_num.copy()
        permuted_num[:, index] = random.permutation(permuted_num[:, index])
        score = average_precision_score(
            sample_labels, _predict_probabilities(model, permuted_num, sample_cat, batch_size, device)
        )
        rows.append({"feature": feature, "importance": baseline - float(score), "fold": fold})
    if sample_cat is not None:
        for index, feature in enumerate(preprocessor.categorical_features):
            permuted_cat = sample_cat.copy()
            permuted_cat[:, index] = random.permutation(permuted_cat[:, index])
            score = average_precision_score(
                sample_labels, _predict_probabilities(model, sample_num, permuted_cat, batch_size, device)
            )
            rows.append({"feature": feature, "importance": baseline - float(score), "fold": fold})
    return pd.DataFrame(rows)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def _batches(indices: np.ndarray, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _cat_tensor(
    values: np.ndarray | None, indices: np.ndarray, device: torch.device
) -> torch.Tensor | None:
    if values is None:
        return None
    return torch.as_tensor(values[indices], dtype=torch.long, device=device)


def _select_cat(values: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    return None if values is None else values[indices]


def _categorical_values(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("__MISSING__").astype(str)
