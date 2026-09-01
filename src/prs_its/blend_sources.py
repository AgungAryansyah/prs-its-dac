from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from prs_its.metrics import validate_paired_oof
from prs_its.modeling import ID_COL, TARGET, make_feature_spec, prepare_catboost_features
from prs_its.training import CTR_INTERACTION_FEATURES


SCREEN_SEEDS = (42, 2026)
CONFIRMATION_SEED = 2718
ALL_SEEDS = (*SCREEN_SEEDS, CONFIRMATION_SEED)


@dataclass(frozen=True)
class BlendSourceArtifacts:
    ctr_run_dir: Path
    xgb_run_dir: Path
    ctr_config: dict[str, Any]
    xgb_config: dict[str, Any]
    ctr_oof_by_seed: dict[int, pd.DataFrame]
    xgb_oof_by_seed: dict[int, pd.DataFrame]
    xgb_test_by_seed: dict[int, np.ndarray]

    @property
    def n_splits(self) -> int:
        return int(self.ctr_config["cv"]["n_splits"])


def load_blend_source_artifacts(
    project_root: Path,
    ctr_run_name: str,
    xgb_run_name: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> BlendSourceArtifacts:
    if ctr_run_name == xgb_run_name:
        raise ValueError("ctr_run_name and xgb_run_name must be different.")
    ctr_run_dir = project_root / "outputs" / "runs" / ctr_run_name
    xgb_run_dir = project_root / "outputs" / "runs" / xgb_run_name
    ctr_config = _load_json(ctr_run_dir / "models" / "catboost_final_config.json")
    xgb_config = _load_json(xgb_run_dir / "models" / "xgb_final_config.json")
    _validate_ctr_config(ctr_config)
    _validate_xgb_config(xgb_config)
    if int(ctr_config["cv"]["n_splits"]) != int(xgb_config["cv"]["n_splits"]):
        raise ValueError("CTR and XGBoost source runs use different fold counts.")
    if ctr_config["cv"] != xgb_config["cv"]:
        raise ValueError("CTR and XGBoost source runs use different CV configurations.")

    expected_ids = train[ID_COL].reset_index(drop=True)
    expected_labels = train[TARGET].astype(int).reset_index(drop=True)
    expected_folds = _expected_folds(train, ctr_config["cv"])
    ctr_oof_by_seed = {
        seed: _load_oof(
            ctr_run_dir / "oof" / f"catboost_oof_seed_{seed}.csv",
            expected_ids,
            expected_labels,
            expected_folds,
        )
        for seed in ALL_SEEDS
    }
    xgb_oof_by_seed = {
        seed: _load_oof(
            xgb_run_dir / "oof" / f"xgb_oof_seed_{seed}.csv",
            expected_ids,
            expected_labels,
            expected_folds,
        )
        for seed in SCREEN_SEEDS
    }
    for seed in SCREEN_SEEDS:
        validate_paired_oof(xgb_oof_by_seed[seed], ctr_oof_by_seed[seed])
    xgb_test_by_seed = {
        seed: _load_test_fold_predictions(
            xgb_run_dir / "oof" / f"xgb_test_fold_predictions_seed_{seed}.csv",
            test[ID_COL].reset_index(drop=True),
            int(xgb_config["cv"]["n_splits"]),
        )
        for seed in SCREEN_SEEDS
    }
    _validate_ctr_model_artifacts(ctr_run_dir, ALL_SEEDS, int(ctr_config["cv"]["n_splits"]))
    return BlendSourceArtifacts(
        ctr_run_dir=ctr_run_dir,
        xgb_run_dir=xgb_run_dir,
        ctr_config=ctr_config,
        xgb_config=xgb_config,
        ctr_oof_by_seed=ctr_oof_by_seed,
        xgb_oof_by_seed=xgb_oof_by_seed,
        xgb_test_by_seed=xgb_test_by_seed,
    )


def reconstruct_ctr_test_predictions(
    sources: BlendSourceArtifacts,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[int, np.ndarray]:
    prepared = _prepare_ctr_features(train, test, sources.ctr_config)
    features = sources.ctr_config["features"]
    categorical_features = sources.ctr_config["categorical_features"]
    X_test = prepared.X_test.loc[:, features]
    test_pool = Pool(X_test, cat_features=categorical_features)
    predictions: dict[int, np.ndarray] = {}
    for seed in ALL_SEEDS:
        fold_predictions = []
        for fold in range(sources.n_splits):
            model = CatBoostClassifier()
            model.load_model(sources.ctr_run_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm")
            fold_predictions.append(np.asarray(model.predict_proba(test_pool)[:, 1], dtype=float))
        predictions[seed] = _validated_probabilities(
            np.mean(np.vstack(fold_predictions), axis=0),
            f"CTR test predictions for seed {seed}",
        )
    return predictions


def _prepare_ctr_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    ctr_config: dict[str, Any],
):
    prepared = prepare_catboost_features(
        train,
        test,
        make_feature_spec(train, test),
        add_count_features=True,
        add_interaction_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features={
            "dati2_typeppk": CTR_INTERACTION_FEATURES["dati2_typeppk"]
        },
    )
    if list(prepared.X.columns) != list(ctr_config["features"]):
        raise ValueError("Current data no longer matches the saved CTR feature schema.")
    if list(prepared.categorical_features) != list(ctr_config["categorical_features"]):
        raise ValueError("Current data no longer matches the saved CTR categorical schema.")
    return prepared


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source configuration: {path}")
    with path.open() as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Source configuration must be a JSON object: {path}")
    return payload


def _validate_ctr_config(config: dict[str, Any]) -> None:
    required = {"model", "profile", "features", "categorical_features", "experiment", "cv"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"CTR source configuration is missing required fields: {missing}")
    if config["model"] != "CatBoostClassifier" or config["profile"] != "ctr":
        raise ValueError("Blend confirmation requires a saved CTR-profile CatBoost source run.")
    if config["experiment"].get("name") != "ctr_dati2_typeppk":
        raise ValueError("Blend confirmation requires the ctr_dati2_typeppk CTR source recipe.")
    if config.get("frequency_transformer") is not None:
        raise ValueError("Blend confirmation does not support CTR frequency-transformer source runs.")
    _validate_cv_config(config["cv"], "CTR")


def _validate_xgb_config(config: dict[str, Any]) -> None:
    required = {
        "model",
        "features",
        "categorical_features",
        "params",
        "experiment",
        "target_encoder",
        "cv",
        "ensemble_seeds",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"XGBoost source configuration is missing required fields: {missing}")
    experiment = config["experiment"]
    if config["model"] != "XGBClassifier":
        raise ValueError("Blend confirmation requires an XGBClassifier source run.")
    if experiment.get("name") != "te_xgb_support" or not experiment.get("add_support_counts"):
        raise ValueError("Blend confirmation requires the te_xgb_support XGBoost source recipe.")
    if not set(SCREEN_SEEDS).issubset(set(config["ensemble_seeds"])):
        raise ValueError("XGBoost source configuration is missing required screen seeds.")
    target_encoder_required = {"inner_n_splits", "smooth"}
    target_encoder_missing = sorted(target_encoder_required - set(config["target_encoder"]))
    if target_encoder_missing:
        raise ValueError(
            "XGBoost source configuration is missing target-encoder fields: "
            f"{target_encoder_missing}"
        )
    _validate_cv_config(config["cv"], "XGBoost")


def _validate_cv_config(config: Any, source_name: str) -> None:
    if not isinstance(config, dict):
        raise ValueError(f"{source_name} CV configuration must be an object.")
    required = {"type", "n_splits", "shuffle", "random_state"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"{source_name} CV configuration is missing fields: {missing}")
    if config["type"] != "StratifiedKFold" or not config["shuffle"]:
        raise ValueError(f"{source_name} source run must use shuffled StratifiedKFold.")
    if int(config["n_splits"]) < 2:
        raise ValueError(f"{source_name} source run must use at least two folds.")


def _expected_folds(train: pd.DataFrame, cv_config: dict[str, Any]) -> np.ndarray:
    labels = train[TARGET].astype(int)
    folds = np.full(len(train), -1, dtype=int)
    cv = StratifiedKFold(
        n_splits=int(cv_config["n_splits"]),
        shuffle=True,
        random_state=int(cv_config["random_state"]),
    )
    for fold, (_, valid_idx) in enumerate(cv.split(train, labels)):
        folds[valid_idx] = fold
    return folds


def _load_oof(
    path: Path,
    expected_ids: pd.Series,
    expected_labels: pd.Series,
    expected_folds: np.ndarray,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source OOF artifact: {path}")
    frame = pd.read_csv(path)
    validate_paired_oof(frame, frame)
    if not frame[ID_COL].reset_index(drop=True).equals(expected_ids):
        raise ValueError(f"Source OOF claim_id order does not match current training data: {path}")
    labels = frame[TARGET].astype(int).reset_index(drop=True)
    if not labels.equals(expected_labels):
        raise ValueError(f"Source OOF labels do not match current training data: {path}")
    folds = frame["fold"].to_numpy(dtype=int)
    if not np.array_equal(folds, expected_folds):
        raise ValueError(f"Source OOF folds do not match the saved CV configuration: {path}")
    return frame.reset_index(drop=True)


def _load_test_fold_predictions(
    path: Path,
    expected_ids: pd.Series,
    n_splits: int,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source test prediction artifact: {path}")
    frame = pd.read_csv(path)
    expected_columns = [f"fold_{fold}" for fold in range(n_splits)]
    required = {ID_COL, *expected_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Source test prediction artifact is missing columns: {missing}")
    if frame[ID_COL].isna().any() or frame[ID_COL].duplicated().any():
        raise ValueError(f"Source test prediction claim_id values must be unique: {path}")
    if not frame[ID_COL].reset_index(drop=True).equals(expected_ids):
        raise ValueError(f"Source test prediction claim_id order does not match current test data: {path}")
    fold_predictions = frame.loc[:, expected_columns].to_numpy(dtype=float).T
    _validated_probabilities(fold_predictions.reshape(-1), f"Source test predictions in {path}")
    return np.mean(fold_predictions, axis=0)


def _validate_ctr_model_artifacts(ctr_run_dir: Path, seeds: tuple[int, ...], n_splits: int) -> None:
    missing = [
        ctr_run_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm"
        for seed in seeds
        for fold in range(n_splits)
        if not (ctr_run_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm").exists()
    ]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required CTR fold model artifacts: {rendered}")


def _validated_probabilities(values: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{name} must contain finite probabilities within [0, 1].")
    return probabilities
