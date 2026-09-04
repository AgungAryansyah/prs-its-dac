from __future__ import annotations

import argparse
import json
import platform
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm.auto import tqdm

from prs_its.calibration import (
    calibrate_test_predictions,
    calibration_curve_frame,
    cross_fit_calibration,
    prediction_distribution,
    should_select_calibration,
)
from prs_its.fairness import age_groups, fairness_across_budgets
from prs_its.foundation_modeling import prepare_foundation_features
from prs_its.metrics import (
    bootstrap_audit_intervals,
    evaluate_probabilities,
    paired_bootstrap_comparison,
    paired_fairness_comparison,
)
from prs_its.modeling import (
    ID_COL,
    TARGET,
    make_feature_spec,
    validate_train_test_schema,
)
from prs_its.submission import make_submission
from prs_its.tabicl_finetune_modeling import (
    TabICLFinetuneParams,
    adaptive_predict_probabilities,
    create_tabicl_predictor,
    ensure_tabicl_finetune_gpu_ready,
    fine_tune_with_oom_backoff,
    fit_in_context_predictor,
    release_cuda_memory,
)
from prs_its.training import find_project_root, load_competition_data

AUDIT_FRACTIONS = (0.03, 0.05, 0.07)
TABICL_FINETUNE_SEED = 42
AVERAGE_PRECISION_TOLERANCE = 0.005


@dataclass(frozen=True)
class TabICLFinetuneTrainingConfig:
    project_root: Path
    run_name: str
    incumbent_run_name: str = "ctr-v1"
    task_type: str = "GPU"
    devices: str = "0"
    n_splits: int = 3
    random_state: int = TABICL_FINETUNE_SEED
    max_runtime_minutes: float = 720.0
    n_bootstrap: int = 1000
    show_progress: bool = True
    resume: bool = False
    params: TabICLFinetuneParams = field(default_factory=TabICLFinetuneParams)

    def __post_init__(self) -> None:
        if not self.run_name or not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_name):
            raise ValueError(
                "run_name may contain only letters, numbers, underscores, and hyphens."
            )
        if self.run_name == self.incumbent_run_name:
            raise ValueError("run_name must differ from the CTR incumbent run.")
        if self.task_type.upper() != "GPU":
            raise ValueError("TabICL fine-tuning is GPU-only; task_type must be GPU.")
        if self.n_splits != 3:
            raise ValueError("TabICL fine-tuning uses exactly three outer folds.")
        if self.max_runtime_minutes <= 0 or self.n_bootstrap <= 0:
            raise ValueError("Runtime and bootstrap settings must be positive.")


def tabicl_finetune_output_paths(
    project_root: Path,
    run_name: str,
    *,
    resume: bool = False,
) -> dict[str, Path]:
    root = project_root / "outputs" / "runs" / run_name
    if resume and not root.exists():
        raise FileNotFoundError(
            f"TabICL fine-tune run does not exist and cannot be resumed: {root}"
        )
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError(
            f"TabICL fine-tune output run already contains artifacts: {root}"
        )
    paths = {
        "root": root,
        "models": root / "models",
        "oof": root / "oof",
        "metrics": root / "metrics",
        "submissions": root / "submissions",
        "cache": root / "cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def preflight_tabicl_finetune(config: TabICLFinetuneTrainingConfig) -> dict[str, Any]:
    paths = tabicl_finetune_output_paths(
        config.project_root, config.run_name, resume=config.resume
    )
    train, test, prepared = _load_prepared_data(config)
    status = ensure_tabicl_finetune_gpu_ready(config.params.min_free_vram_gib)
    cv = _make_cv(config)
    outer_train_idx, _ = next(iter(cv.split(prepared.X, prepared.y)))
    inner_train_idx, inner_val_idx = _inner_split_indices(
        prepared.y.iloc[outer_train_idx].to_numpy(),
        config.params.validation_split_ratio,
        config.random_state,
    )
    outer_train = prepared.X.iloc[outer_train_idx]
    y_outer_train = prepared.y.iloc[outer_train_idx]
    smoke_train_idx = _stratified_sample_indices(
        y_outer_train.iloc[inner_train_idx].to_numpy(), 128, config.random_state
    )
    smoke_val_idx = _stratified_sample_indices(
        y_outer_train.iloc[inner_val_idx].to_numpy(), 64, config.random_state + 1
    )
    smoke_params = replace(
        config.params,
        epochs=1,
        patience=1,
        max_data_size=1024,
        n_estimators_inference=1,
    )
    result = fine_tune_with_oom_backoff(
        outer_train.iloc[inner_train_idx].iloc[smoke_train_idx],
        y_outer_train.iloc[inner_train_idx].iloc[smoke_train_idx],
        outer_train.iloc[inner_val_idx].iloc[smoke_val_idx],
        y_outer_train.iloc[inner_val_idx].iloc[smoke_val_idx],
        smoke_params,
        checkpoint_root=paths["models"] / "preflight",
        cache_dir=paths["cache"] / "preflight",
        random_state=config.random_state,
        time_limit_seconds=300.0,
    )
    payload = {
        "run_name": config.run_name,
        "gpu": status,
        "params": config.params.as_dict(),
        "feature_columns": list(prepared.X.columns),
        "categorical_features": list(prepared.categorical_features),
        "train_rows": len(train),
        "test_rows": len(test),
        "smoke_train_rows": len(smoke_train_idx),
        "smoke_validation_rows": len(smoke_val_idx),
        "checkpoint_validation": result.as_dict(),
        "checkpoint_interface": "TabICL FinetunedTabICLClassifier.fit(output_dir=...)",
    }
    _save_json(paths["metrics"] / "tabicl_finetune_preflight.json", payload)
    return payload


def run_tabicl_finetune_training(
    config: TabICLFinetuneTrainingConfig,
) -> dict[str, Any]:
    started = time.monotonic()
    if config.resume:
        paths = tabicl_finetune_output_paths(
            config.project_root, config.run_name, resume=True
        )
        preflight = None
    else:
        preflight = preflight_tabicl_finetune(config)
        paths = tabicl_finetune_output_paths(
            config.project_root, config.run_name, resume=True
        )
    train, test, prepared = _load_prepared_data(config)
    cv = _make_cv(config)
    expected_fold_id = _expected_fold_ids(cv, prepared.y)
    if config.resume:
        preflight = _load_preflight(paths, config, prepared)
    assert preflight is not None

    fold_rows: list[dict[str, Any]] = []
    test_predictions = np.full((config.n_splits, len(test)), np.nan, dtype=float)
    raw_oof = np.full(len(train), np.nan, dtype=float)
    support_manifest_rows: list[pd.DataFrame] = []
    progress = tqdm(
        total=config.n_splits + 1,
        desc="TabICL fine-tuning",
        unit="stage",
        disable=not config.show_progress,
    )
    try:
        for fold, (outer_train_idx, outer_valid_idx) in enumerate(
            cv.split(prepared.X, prepared.y)
        ):
            saved = _load_completed_fold(
                paths,
                train,
                test,
                fold,
                outer_valid_idx,
                expected_fold_id,
            )
            if saved is not None:
                raw_oof[outer_valid_idx] = saved["oof_pred"]
                test_predictions[fold] = saved["test_pred"]
                fold_rows.append(saved["metrics"])
                progress.update(1)
                continue
            _check_budget(
                started, config.max_runtime_minutes, f"before outer fold {fold}"
            )
            outer_X = prepared.X.iloc[outer_train_idx]
            outer_y = prepared.y.iloc[outer_train_idx]
            inner_train_idx, inner_val_idx = _inner_split_indices(
                outer_y.to_numpy(),
                config.params.validation_split_ratio,
                config.random_state + fold,
            )
            fold_result = fine_tune_with_oom_backoff(
                outer_X.iloc[inner_train_idx],
                outer_y.iloc[inner_train_idx],
                outer_X.iloc[inner_val_idx],
                outer_y.iloc[inner_val_idx],
                config.params,
                checkpoint_root=paths["models"] / f"fold_{fold}",
                cache_dir=paths["cache"] / f"fold_{fold}",
                random_state=config.random_state + fold,
                time_limit_seconds=_remaining_fold_budget(
                    started, config.max_runtime_minutes, config.n_splits - fold + 1
                ),
            )
            predictor, support_profile = fit_in_context_predictor(
                lambda checkpoint_path=fold_result.checkpoint_path, predictor_cache=(paths["cache"] / f"fold_{fold}" / "predictor"), predictor_seed=config.random_state + fold: (
                    create_tabicl_predictor(
                        checkpoint_path,
                        config.params,
                        predictor_cache,
                        predictor_seed,
                    )
                ),
                outer_X,
                outer_y,
                config.params.support_cap,
                config.random_state + fold,
            )
            valid_pred, prediction_profile = adaptive_predict_probabilities(
                predictor,
                prepared.X.iloc[outer_valid_idx],
                config.params.prediction_chunk_size,
                config.params.min_prediction_chunk_size,
            )
            test_pred, test_prediction_profile = adaptive_predict_probabilities(
                predictor,
                prepared.X_test,
                config.params.prediction_chunk_size,
                config.params.min_prediction_chunk_size,
            )
            raw_oof[outer_valid_idx] = valid_pred
            test_predictions[fold] = test_pred
            metrics = {
                "fold": fold,
                "outer_train_rows": len(outer_train_idx),
                "outer_validation_rows": len(outer_valid_idx),
                "inner_train_rows": len(inner_train_idx),
                "inner_validation_rows": len(inner_val_idx),
                "outer_train_fraud_prevalence": float(outer_y.mean()),
                "outer_validation_fraud_prevalence": float(
                    prepared.y.iloc[outer_valid_idx].mean()
                ),
                "fine_tuning": fold_result.as_dict(),
                "support": support_profile.as_dict(),
                "validation_prediction": prediction_profile.as_dict(),
                "test_prediction": test_prediction_profile.as_dict(),
                "metrics": evaluate_probabilities(
                    prepared.y.iloc[outer_valid_idx], valid_pred, AUDIT_FRACTIONS
                ),
                "elapsed_seconds": time.monotonic() - started,
            }
            _save_fold_artifacts(
                paths,
                train,
                test,
                fold,
                outer_valid_idx,
                valid_pred,
                test_pred,
                metrics,
            )
            support_manifest_rows.append(
                _support_manifest_rows(
                    train.iloc[outer_train_idx],
                    support_profile.selected_indices,
                    "outer_fold",
                    fold,
                )
            )
            fold_rows.append(metrics)
            del predictor
            release_cuda_memory()
            progress.update(1)

        if (
            not np.isfinite(raw_oof).all()
            or not ((0 <= raw_oof) & (raw_oof <= 1)).all()
        ):
            raise RuntimeError(
                "Every training row must receive a valid TabICL fine-tuned OOF prediction."
            )
        if not np.isfinite(test_predictions).all():
            raise RuntimeError("Every outer fold must produce test predictions.")
        raw_oof_frame = pd.DataFrame(
            {
                ID_COL: train[ID_COL].to_numpy(),
                TARGET: train[TARGET].to_numpy(dtype=int),
                "fold": expected_fold_id,
                "fraud_probability_raw": raw_oof,
            }
        )
        raw_oof_frame.to_csv(paths["oof"] / "tabicl_ft_raw_oof.csv", index=False)
        pd.DataFrame(fold_rows).to_json(
            paths["metrics"] / "tabicl_ft_fold_metrics.json", orient="records", indent=2
        )
        _save_oof_evaluation(paths, train, raw_oof_frame, config)
        comparison, fairness = _save_ctr_comparison(paths, train, raw_oof_frame, config)
        calibration = _calibrate(raw_oof_frame)
        _save_calibration_artifacts(paths, raw_oof_frame, calibration)
        _require_complete_oof_artifacts(paths)
        _check_budget(
            started, config.max_runtime_minutes, "before final full-data fine-tuning"
        )
        final_raw_test, final_support = _final_full_data_prediction(
            config,
            paths,
            prepared.X,
            prepared.y,
            prepared.X_test,
            train,
            test[ID_COL],
            started,
        )
        support_manifest_rows.append(
            _support_manifest_rows(train, final_support.selected_indices, "final", None)
        )
        _save_support_manifest(paths, support_manifest_rows)
        final_test = final_raw_test
        if calibration["method"] != "raw":
            final_test = calibrate_test_predictions(
                raw_oof_frame["fraud_probability_raw"],
                raw_oof_frame[TARGET],
                final_raw_test,
                calibration["method"],
            )
        decision = _promotion_decision(comparison, fairness)
        status = "promoted" if decision["promoted"] else "unpromoted"
        submission_path = (
            paths["submissions"]
            / f"tabicl_ft_{calibration['method']}_{status}_submission.csv"
        )
        make_submission(test[ID_COL], final_test, submission_path)
        raw_submission_path = (
            paths["submissions"] / "tabicl_ft_raw_unpromoted_submission.csv"
        )
        if calibration["method"] != "raw":
            make_submission(test[ID_COL], final_raw_test, raw_submission_path)
        else:
            raw_submission_path = submission_path
        decision.update(
            {
                "submission_status": status,
                "submission_path": str(submission_path),
                "raw_submission_path": str(raw_submission_path),
                "calibration": calibration["method"],
                "comparison_fold_schemes": {
                    "candidate": "StratifiedKFold(n_splits=3, random_state=42)",
                    "incumbent": "stored CTR OOF; paired by claim_id and label",
                },
            }
        )
        _save_json(paths["metrics"] / "tabicl_ft_promotion_decision.json", decision)
        manifest = _run_manifest(config, preflight, started, decision)
        _save_json(paths["metrics"] / "tabicl_ft_run_manifest.json", manifest)
        progress.update(1)
        return {
            "submission_path": submission_path,
            "raw_submission_path": raw_submission_path,
            "promoted": decision["promoted"],
            "submission_status": status,
            "resumed": config.resume,
        }
    finally:
        progress.close()
        release_cuda_memory()


def _load_prepared_data(config: TabICLFinetuneTrainingConfig):
    train, test = load_competition_data(config.project_root)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError(
            "claim_id and label must not be TabICL fine-tuning features."
        )
    prepared = prepare_foundation_features(train, test, make_feature_spec(train, test))
    return train, test, prepared


def _make_cv(config: TabICLFinetuneTrainingConfig) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_state
    )


def _expected_fold_ids(cv: StratifiedKFold, labels: pd.Series) -> np.ndarray:
    fold_id = np.full(len(labels), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        fold_id[valid_idx] = fold
    if (fold_id < 0).any():
        raise RuntimeError("Outer folds must cover every training row exactly once.")
    return fold_id


def _inner_split_indices(
    labels: np.ndarray,
    validation_split_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_split_ratio,
        random_state=random_state,
    )
    train_idx, validation_idx = next(splitter.split(np.zeros(len(labels)), labels))
    if np.intersect1d(train_idx, validation_idx).size:
        raise RuntimeError("Inner training and validation rows must be disjoint.")
    return train_idx.astype(int), validation_idx.astype(int)


def _stratified_sample_indices(
    labels: np.ndarray, maximum_rows: int, random_state: int
) -> np.ndarray:
    if len(labels) <= maximum_rows:
        return np.arange(len(labels), dtype=int)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=maximum_rows, random_state=random_state
    )
    selected, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(selected.astype(int))


def _load_completed_fold(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
    outer_valid_idx: np.ndarray,
    expected_fold_id: np.ndarray,
) -> dict[str, Any] | None:
    oof_path = paths["oof"] / f"tabicl_ft_fold_{fold}_oof.csv"
    test_path = paths["oof"] / f"tabicl_ft_fold_{fold}_test.csv"
    metrics_path = paths["metrics"] / f"tabicl_ft_fold_{fold}.json"
    artifacts = (oof_path, test_path, metrics_path)
    if not any(path.exists() for path in artifacts):
        return None
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot resume outer fold {fold}; artifacts are incomplete: {missing}"
        )
    oof = pd.read_csv(oof_path)
    required_oof = {ID_COL, TARGET, "fold", "fraud_probability_raw"}
    if missing_columns := sorted(required_oof - set(oof.columns)):
        raise ValueError(
            f"Saved outer fold {fold} OOF is missing columns: {missing_columns}"
        )
    expected_train = train.iloc[outer_valid_idx]
    if not np.array_equal(oof[ID_COL].to_numpy(), expected_train[ID_COL].to_numpy()):
        raise ValueError(
            f"Saved outer fold {fold} claim IDs do not match the current validation partition."
        )
    if not np.array_equal(
        oof[TARGET].to_numpy(dtype=int), expected_train[TARGET].to_numpy(dtype=int)
    ):
        raise ValueError(
            f"Saved outer fold {fold} labels do not match the current validation partition."
        )
    if not np.all(oof["fold"].to_numpy(dtype=int) == expected_fold_id[outer_valid_idx]):
        raise ValueError(
            f"Saved outer fold {fold} fold assignments do not match the current CV split."
        )
    prediction = oof["fraud_probability_raw"].to_numpy(dtype=float)
    _validate_probabilities(prediction, f"Saved outer fold {fold} OOF")
    test_frame = pd.read_csv(test_path)
    if list(test_frame.columns) != [ID_COL, "fraud_probability_raw"]:
        raise ValueError(
            f"Saved outer fold {fold} test prediction columns are invalid."
        )
    if not np.array_equal(test_frame[ID_COL].to_numpy(), test[ID_COL].to_numpy()):
        raise ValueError(
            f"Saved outer fold {fold} test claim IDs do not match current test data."
        )
    test_prediction = test_frame["fraud_probability_raw"].to_numpy(dtype=float)
    _validate_probabilities(
        test_prediction, f"Saved outer fold {fold} test predictions"
    )
    return {
        "oof_pred": prediction,
        "test_pred": test_prediction,
        "metrics": _load_json(metrics_path),
    }


def _save_fold_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    fold: int,
    outer_valid_idx: np.ndarray,
    valid_pred: np.ndarray,
    test_pred: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    pd.DataFrame(
        {
            ID_COL: train.iloc[outer_valid_idx][ID_COL].to_numpy(),
            TARGET: train.iloc[outer_valid_idx][TARGET].to_numpy(dtype=int),
            "fold": fold,
            "fraud_probability_raw": valid_pred,
        }
    ).to_csv(paths["oof"] / f"tabicl_ft_fold_{fold}_oof.csv", index=False)
    pd.DataFrame(
        {ID_COL: test[ID_COL].to_numpy(), "fraud_probability_raw": test_pred}
    ).to_csv(paths["oof"] / f"tabicl_ft_fold_{fold}_test.csv", index=False)
    _save_json(paths["metrics"] / f"tabicl_ft_fold_{fold}.json", metrics)


def _save_oof_evaluation(
    paths: dict[str, Path],
    train: pd.DataFrame,
    raw_oof: pd.DataFrame,
    config: TabICLFinetuneTrainingConfig,
) -> None:
    metrics = evaluate_probabilities(
        raw_oof[TARGET], raw_oof["fraud_probability_raw"], AUDIT_FRACTIONS
    )
    _save_json(paths["metrics"] / "tabicl_ft_oof_metrics.json", metrics)
    pd.DataFrame(
        bootstrap_audit_intervals(
            raw_oof[TARGET],
            raw_oof["fraud_probability_raw"],
            audit_fractions=AUDIT_FRACTIONS,
            n_bootstrap=config.n_bootstrap,
        )
    ).to_csv(paths["metrics"] / "tabicl_ft_audit_bootstrap.csv", index=False)
    if {"jkpst", "umur"}.issubset(train.columns):
        fairness = pd.concat(
            [
                fairness_across_budgets(
                    train["jkpst"],
                    raw_oof[TARGET],
                    raw_oof["fraud_probability_raw"],
                    group_name="gender",
                ),
                fairness_across_budgets(
                    age_groups(train["umur"]),
                    raw_oof[TARGET],
                    raw_oof["fraud_probability_raw"],
                    group_name="age_group",
                ),
            ],
            ignore_index=True,
        )
        fairness.to_csv(paths["metrics"] / "tabicl_ft_fairness.csv", index=False)


def _save_ctr_comparison(
    paths: dict[str, Path],
    train: pd.DataFrame,
    candidate: pd.DataFrame,
    config: TabICLFinetuneTrainingConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    incumbent_path = (
        config.project_root
        / "outputs"
        / "runs"
        / config.incumbent_run_name
        / "oof"
        / f"catboost_oof_seed_{TABICL_FINETUNE_SEED}.csv"
    )
    if not incumbent_path.exists():
        raise FileNotFoundError(f"CTR incumbent OOF does not exist: {incumbent_path}")
    incumbent = pd.read_csv(incumbent_path)
    if not np.array_equal(candidate[ID_COL].to_numpy(), incumbent[ID_COL].to_numpy()):
        raise ValueError("TabICL and CTR OOF claim_id order must be identical.")
    if not np.array_equal(
        candidate[TARGET].to_numpy(dtype=int), incumbent[TARGET].to_numpy(dtype=int)
    ):
        raise ValueError("TabICL and CTR OOF labels must be identical.")
    comparison = paired_bootstrap_comparison(
        candidate[TARGET],
        candidate["fraud_probability_raw"],
        incumbent["fraud_probability_raw"],
        audit_fractions=AUDIT_FRACTIONS,
        n_bootstrap=config.n_bootstrap,
        random_state=config.random_state,
    )
    comparison.to_csv(paths["metrics"] / "tabicl_ft_vs_ctr_paired.csv", index=False)
    if not {"jkpst", "umur"}.issubset(train.columns):
        return comparison, {
            "subgroup_rate_deltas": pd.DataFrame(),
            "gap_intervals": pd.DataFrame(),
        }
    fairness = paired_fairness_comparison(
        candidate[TARGET],
        candidate["fraud_probability_raw"],
        incumbent["fraud_probability_raw"],
        gender_groups=train["jkpst"],
        age_group_values=age_groups(train["umur"]),
        audit_fractions=AUDIT_FRACTIONS,
        n_bootstrap=config.n_bootstrap,
        random_state=config.random_state,
    )
    fairness["subgroup_rate_deltas"].to_csv(
        paths["metrics"] / "tabicl_ft_vs_ctr_fairness_rates.csv", index=False
    )
    fairness["gap_intervals"].to_csv(
        paths["metrics"] / "tabicl_ft_vs_ctr_fairness_gaps.csv", index=False
    )
    return comparison, fairness


def _calibrate(candidate: pd.DataFrame) -> dict[str, Any]:
    raw = candidate["fraud_probability_raw"].to_numpy(dtype=float)
    labels = candidate[TARGET].to_numpy(dtype=int)
    folds = candidate["fold"].to_numpy(dtype=int)
    raw_metrics = evaluate_probabilities(labels, raw, AUDIT_FRACTIONS)
    rows = [{"prediction_type": "raw", **raw_metrics}]
    calibrated_predictions: dict[str, np.ndarray] = {}
    eligible: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for method in ("sigmoid", "isotonic"):
        calibrated = np.asarray(
            cross_fit_calibration(labels, raw, folds, method)["oof_pred"], dtype=float
        )
        metrics = evaluate_probabilities(labels, calibrated, AUDIT_FRACTIONS)
        calibrated_predictions[method] = calibrated
        rows.append({"prediction_type": method, **metrics})
        if should_select_calibration(raw_metrics, metrics):
            eligible.append((method, calibrated, metrics))
    if eligible:
        method, selected, selected_metrics = min(
            eligible, key=lambda row: row[2]["brier_score"]
        )
    else:
        method, selected, selected_metrics = "raw", raw, raw_metrics
    return {
        "method": method,
        "selected_oof": selected,
        "selected_metrics": selected_metrics,
        "cross_fitted_oof": calibrated_predictions,
        "rows": rows,
    }


def _save_calibration_artifacts(
    paths: dict[str, Path],
    candidate: pd.DataFrame,
    calibration: dict[str, Any],
) -> None:
    calibrated = candidate.copy()
    calibrated["fraud_probability_final"] = calibration["selected_oof"]
    calibrated.to_csv(paths["oof"] / "tabicl_ft_oof.csv", index=False)
    for method, probabilities in calibration["cross_fitted_oof"].items():
        frame = candidate.copy()
        frame["fraud_probability_raw"] = probabilities
        frame.to_csv(
            paths["oof"] / f"tabicl_ft_oof_calibrated_{method}.csv", index=False
        )
    pd.DataFrame(calibration["rows"]).to_csv(
        paths["metrics"] / "tabicl_ft_calibration_metrics.csv", index=False
    )
    calibration_curve_frame(candidate[TARGET], calibration["selected_oof"]).to_csv(
        paths["metrics"] / "tabicl_ft_calibration_curve.csv", index=False
    )
    prediction_distribution(candidate[TARGET], calibration["selected_oof"]).to_csv(
        paths["metrics"] / "tabicl_ft_prediction_distribution.csv", index=False
    )


def _require_complete_oof_artifacts(paths: dict[str, Path]) -> None:
    required = (
        paths["oof"] / "tabicl_ft_raw_oof.csv",
        paths["oof"] / "tabicl_ft_oof.csv",
        paths["metrics"] / "tabicl_ft_fold_metrics.json",
        paths["metrics"] / "tabicl_ft_oof_metrics.json",
        paths["metrics"] / "tabicl_ft_vs_ctr_paired.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Final TabICL fine-tuning requires complete OOF artifacts: {missing}"
        )


def _final_full_data_prediction(
    config: TabICLFinetuneTrainingConfig,
    paths: dict[str, Path],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    train: pd.DataFrame,
    test_claim_ids: pd.Series,
    started: float,
):
    output_path = paths["oof"] / "tabicl_ft_final_test_raw.csv"
    support_path = paths["metrics"] / "tabicl_ft_final_support.json"
    if config.resume and output_path.exists() and support_path.exists():
        saved = pd.read_csv(output_path)
        if not np.array_equal(saved[ID_COL].to_numpy(), test_claim_ids.to_numpy()):
            raise ValueError("Saved final TabICL test prediction IDs are invalid.")
        prediction = saved["fraud_probability_raw"].to_numpy(dtype=float)
        _validate_probabilities(prediction, "Saved final TabICL test predictions")
        support = _load_json(support_path)
        selected = np.asarray(support["selected_indices"], dtype=int)
        return prediction, _support_profile_from_saved(support, selected)
    inner_train_idx, inner_val_idx = _inner_split_indices(
        y.to_numpy(), config.params.validation_split_ratio, config.random_state + 10_000
    )
    result = fine_tune_with_oom_backoff(
        X.iloc[inner_train_idx],
        y.iloc[inner_train_idx],
        X.iloc[inner_val_idx],
        y.iloc[inner_val_idx],
        config.params,
        checkpoint_root=paths["models"] / "final",
        cache_dir=paths["cache"] / "final",
        random_state=config.random_state + 10_000,
        time_limit_seconds=_remaining_fold_budget(
            started, config.max_runtime_minutes, 1
        ),
    )
    predictor, support = fit_in_context_predictor(
        lambda: create_tabicl_predictor(
            result.checkpoint_path,
            config.params,
            paths["cache"] / "final" / "predictor",
            config.random_state + 10_000,
        ),
        X,
        y,
        config.params.support_cap,
        config.random_state + 10_000,
    )
    prediction, profile = adaptive_predict_probabilities(
        predictor,
        X_test,
        config.params.prediction_chunk_size,
        config.params.min_prediction_chunk_size,
    )
    pd.DataFrame(
        {ID_COL: test_claim_ids.to_numpy(), "fraud_probability_raw": prediction}
    ).to_csv(output_path, index=False)
    _save_json(
        support_path,
        {
            **support.as_dict(),
            "selected_indices": support.selected_indices.tolist(),
            "fine_tuning": result.as_dict(),
            "prediction": profile.as_dict(),
        },
    )
    del predictor
    release_cuda_memory()
    return prediction, support


def _support_manifest_rows(
    train: pd.DataFrame,
    selected_indices: np.ndarray,
    phase: str,
    fold: int | None,
) -> pd.DataFrame:
    if len(selected_indices) == len(train):
        return pd.DataFrame(columns=["phase", "fold", ID_COL])
    return pd.DataFrame(
        {
            "phase": phase,
            "fold": fold,
            ID_COL: train.iloc[selected_indices][ID_COL].to_numpy(),
        }
    )


def _save_support_manifest(paths: dict[str, Path], frames: list[pd.DataFrame]) -> None:
    output_path = paths["metrics"] / "tabicl_ft_support_manifest.csv"
    existing = pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()
    manifest = pd.concat([existing, *frames], ignore_index=True) if frames else existing
    if manifest.empty:
        manifest = pd.DataFrame(columns=["phase", "fold", ID_COL])
    else:
        manifest = manifest.drop_duplicates().reset_index(drop=True)
    manifest.to_csv(output_path, index=False)


def _promotion_decision(
    comparison: pd.DataFrame,
    fairness: dict[str, pd.DataFrame],
) -> dict[str, bool | float]:
    fraud = _comparison_row(comparison, "fraud_caught", 0.05)
    ap = _comparison_row(comparison, "average_precision", None)
    brier = _comparison_row(comparison, "brier_score", None)
    gender_fair = _fairness_not_regressed(fairness["gap_intervals"], "gender")
    age_fair = _fairness_not_regressed(fairness["gap_intervals"], "age_group")
    return {
        "promoted": bool(
            fraud["ci_lower"] > 0
            and ap["delta"] >= -AVERAGE_PRECISION_TOLERANCE
            and brier["delta"] <= 0
            and gender_fair
            and age_fair
        ),
        "fraud_caught_delta_at_5pct": float(fraud["delta"]),
        "fraud_capture_positive_interval": bool(fraud["ci_lower"] > 0),
        "average_precision_noninferior": bool(
            ap["delta"] >= -AVERAGE_PRECISION_TOLERANCE
        ),
        "brier_noninferior": bool(brier["delta"] <= 0),
        "gender_fairness_not_regressed": gender_fair,
        "age_fairness_not_regressed": age_fair,
    }


def _comparison_row(
    comparison: pd.DataFrame, metric: str, fraction: float | None
) -> pd.Series:
    rows = comparison.loc[comparison["metric"].eq(metric)]
    if fraction is None:
        rows = rows.loc[rows["audit_fraction"].isna()]
    else:
        rows = rows.loc[np.isclose(rows["audit_fraction"], fraction, equal_nan=False)]
    if len(rows) != 1:
        raise ValueError(f"Comparison is missing {metric!r} at {fraction!r}.")
    return rows.iloc[0]


def _fairness_not_regressed(intervals: pd.DataFrame, group_variable: str) -> bool:
    rows = intervals.loc[
        intervals["group_variable"].eq(group_variable)
        & np.isclose(intervals["audit_fraction"], 0.05, equal_nan=False)
    ]
    return bool(len(rows) == 1 and float(rows.iloc[0]["ci_lower"]) <= 0)


def _remaining_fold_budget(
    started: float, maximum_minutes: float, remaining_runs: int
) -> float:
    remaining = maximum_minutes * 60 - (time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("TabICL fine-tuning runtime budget is exhausted.")
    return max(1.0, remaining / remaining_runs)


def _check_budget(started: float, maximum_minutes: float, label: str) -> None:
    if time.monotonic() - started >= maximum_minutes * 60:
        raise TimeoutError(f"TabICL fine-tuning runtime budget exhausted {label}.")


def _load_preflight(
    paths: dict[str, Path], config: TabICLFinetuneTrainingConfig, prepared: Any
) -> dict[str, Any]:
    preflight = _load_json(paths["metrics"] / "tabicl_finetune_preflight.json")
    if preflight.get("params") != config.params.as_dict():
        raise ValueError(
            "Saved TabICL fine-tuning preflight parameters do not match this run."
        )
    if preflight.get("feature_columns") != list(prepared.X.columns):
        raise ValueError(
            "Current feature schema does not match the saved TabICL fine-tuning preflight."
        )
    return preflight


def _validate_probabilities(values: np.ndarray, name: str) -> None:
    if not np.isfinite(values).all() or not ((0 <= values) & (values <= 1)).all():
        raise ValueError(f"{name} must be finite probabilities.")


def _support_profile_from_saved(payload: dict[str, Any], selected_indices: np.ndarray):
    from prs_its.tabicl_finetune_modeling import SupportProfile

    return SupportProfile(
        strategy=str(payload["strategy"]),
        support_rows=int(payload["support_rows"]),
        support_cap=int(payload["support_cap"]),
        attempted_rows=tuple(int(value) for value in payload["attempted_rows"]),
        selected_indices=selected_indices,
    )


def _run_manifest(
    config: TabICLFinetuneTrainingConfig,
    preflight: dict[str, Any],
    started: float,
    decision: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "task_type": "GPU",
        "devices": config.devices,
        "resumed": config.resume,
        "runtime_budget_minutes": config.max_runtime_minutes,
        "runtime_seconds": time.monotonic() - started,
        "outer_cv": {
            "type": "StratifiedKFold",
            "n_splits": config.n_splits,
            "random_state": config.random_state,
        },
        "inner_validation": {
            "type": "StratifiedShuffleSplit",
            "ratio": config.params.validation_split_ratio,
            "early_stopping_metric": "roc_auc",
        },
        "params": config.params.as_dict(),
        "preflight": preflight,
        "promotion_decision": decision,
        "model_artifacts": "retained on the remote training server; not mirrored in local artifact exports",
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Cannot serialize {type(value)!r} to JSON.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune TabICLv2 for claims-fraud prediction."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--incumbent-run-name", default="ctr-v1")
    parser.add_argument("--task-type", default="GPU", choices=["GPU"])
    parser.add_argument("--devices", default="0")
    parser.add_argument("--max-runtime-minutes", type=float, default=720.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-data-size", type=int, choices=[4096, 2048, 1024], default=4096
    )
    parser.add_argument("--prediction-chunk-size", type=int, default=256)
    parser.add_argument("--min-prediction-chunk-size", type=int, default=64)
    parser.add_argument("--support-cap", type=int, default=100_000)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = TabICLFinetuneTrainingConfig(
        project_root=find_project_root(),
        run_name=args.run_name,
        incumbent_run_name=args.incumbent_run_name,
        task_type=args.task_type,
        devices=args.devices,
        max_runtime_minutes=args.max_runtime_minutes,
        n_bootstrap=args.n_bootstrap,
        show_progress=not args.quiet,
        resume=args.resume,
        params=TabICLFinetuneParams(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_data_size=args.max_data_size,
            prediction_chunk_size=args.prediction_chunk_size,
            min_prediction_chunk_size=args.min_prediction_chunk_size,
            support_cap=args.support_cap,
        ),
    )
    result = (
        preflight_tabicl_finetune(config)
        if args.preflight
        else run_tabicl_finetune_training(config)
    )
    print(json.dumps(result, indent=2, default=_json_default))
