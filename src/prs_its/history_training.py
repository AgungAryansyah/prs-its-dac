from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterable, Iterator

import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.auto import tqdm

from prs_its.calibration import (
    calibrate_test_predictions,
    calibration_curve_frame,
    cross_fit_calibration,
    prediction_distribution,
    should_select_calibration,
)
from prs_its.fairness import age_groups
from prs_its.history_modeling import (
    HistoryFeatureBundle,
    add_history_features,
    build_causal_history_features,
    history_feature_schema,
    history_provider_groups,
    load_claim_history,
)
from prs_its.metrics import (
    bootstrap_audit_intervals,
    evaluate_probabilities,
    paired_fairness_comparison,
    paired_oof_comparison,
)
from prs_its.modeling import (
    BASE_PARAMS,
    ID_COL,
    N_SPLITS,
    RANDOM_STATE,
    TARGET,
    PreparedFeatures,
    aggregate_feature_importance,
    ensure_gpu_ready,
    make_feature_spec,
    prepare_catboost_features,
    train_catboost_cv,
    validate_train_test_schema,
)
from prs_its.submission import make_submission
from prs_its.training import CTR_INTERACTION_FEATURES, _select_experiment, find_project_root, load_competition_data


AUDIT_FRACTIONS = (0.03, 0.05, 0.07)
SCREEN_SEED = RANDOM_STATE
ENSEMBLE_SEEDS = (SCREEN_SEED, 2026)
CONFIRMATION_SEED = 2718
WARMUP_FRACTION = 0.2
MEANINGFUL_FRAUD_CAPTURE_GAIN = 20
AVERAGE_PRECISION_TOLERANCE = 0.005
HISTORY_PARAMS = {
    **BASE_PARAMS,
    "depth": 8,
    "l2_leaf_reg": 10.0,
    "random_strength": 2.0,
    "bagging_temperature": 1.0,
    "max_ctr_complexity": 4,
}


@dataclass(frozen=True)
class HistoryTrainingConfig:
    project_root: Path
    history_path: Path
    run_name: str
    history_column_map_path: Path | None = None
    task_type: str = "GPU"
    devices: str = "0"
    n_splits: int = N_SPLITS
    random_state: int = RANDOM_STATE
    early_stopping_rounds: int = 200
    show_progress: bool = True
    catboost_verbose: int | bool = 100
    n_bootstrap: int = 1000
    warmup_fraction: float = WARMUP_FRACTION


@dataclass(frozen=True)
class HistoryExperimentSpec:
    name: str
    prepared: PreparedFeatures
    feature_groups: tuple[str, ...]
    notes: str
    stage: str = "screen"


@dataclass(frozen=True)
class _TemporalFold:
    fold: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    validation_start: pd.Timestamp


class RollingOriginCV:
    def __init__(
        self,
        event_at: pd.Series,
        adjudicated_at: pd.Series,
        n_splits: int,
        warmup_fraction: float = WARMUP_FRACTION,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not 0 < warmup_fraction < 1:
            raise ValueError("warmup_fraction must be within (0, 1).")
        self.event_at = pd.to_datetime(event_at, utc=True, errors="raise").reset_index(drop=True)
        self.adjudicated_at = pd.to_datetime(
            adjudicated_at, utc=True, errors="raise"
        ).reset_index(drop=True)
        if len(self.event_at) != len(self.adjudicated_at):
            raise ValueError("event_at and adjudicated_at must have matching lengths.")
        if self.event_at.isna().any() or self.adjudicated_at.isna().any():
            raise ValueError("Temporal CV timestamps must be complete.")
        self.n_splits = n_splits
        self.warmup_fraction = warmup_fraction
        self._folds = self._make_folds()

    @property
    def folds(self) -> tuple[_TemporalFold, ...]:
        return self._folds

    @property
    def evaluation_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.event_at), dtype=bool)
        for fold in self._folds:
            mask[fold.valid_idx] = True
        return mask

    @property
    def fold_id(self) -> np.ndarray:
        assignments = np.full(len(self.event_at), -1, dtype=int)
        for fold in self._folds:
            assignments[fold.valid_idx] = fold.fold
        return assignments

    def split(
        self, X: pd.DataFrame, y: pd.Series | None = None, groups: np.ndarray | None = None
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if len(X) != len(self.event_at):
            raise ValueError("Temporal CV input must match timestamp length.")
        for fold in self._folds:
            yield fold.train_idx.copy(), fold.valid_idx.copy()

    def validate_labels(self, labels: Iterable[int]) -> None:
        y = np.asarray(labels, dtype=int)
        if len(y) != len(self.event_at):
            raise ValueError("Labels must match temporal CV timestamp length.")
        for fold in self._folds:
            if np.unique(y[fold.train_idx]).size < 2:
                raise ValueError(f"Temporal fold {fold.fold} has a single-class training partition.")
            if np.unique(y[fold.valid_idx]).size < 2:
                raise ValueError(f"Temporal fold {fold.fold} has a single-class validation partition.")

    def _make_folds(self) -> tuple[_TemporalFold, ...]:
        total_rows = len(self.event_at)
        warmup_rows = int(np.floor(total_rows * self.warmup_fraction))
        if warmup_rows < 1 or total_rows - warmup_rows < self.n_splits:
            raise ValueError("Temporal CV needs enough rows after the warm-up period.")
        order = np.argsort(self.event_at.astype("int64").to_numpy(), kind="stable")
        validation_blocks = np.array_split(order[warmup_rows:], self.n_splits)
        folds = []
        for fold_index, valid_idx in enumerate(validation_blocks):
            if len(valid_idx) == 0:
                raise ValueError("Temporal CV produced an empty validation block.")
            validation_start = self.event_at.iloc[valid_idx].min()
            train_mask = (self.event_at < validation_start) & (
                self.adjudicated_at < validation_start
            )
            train_idx = np.flatnonzero(train_mask.to_numpy())
            if len(train_idx) == 0:
                raise ValueError(f"Temporal fold {fold_index} has no adjudicated training claims.")
            folds.append(
                _TemporalFold(
                    fold=fold_index,
                    train_idx=train_idx,
                    valid_idx=np.sort(valid_idx),
                    validation_start=validation_start,
                )
            )
        return tuple(folds)


def history_output_paths(project_root: Path, run_name: str) -> dict[str, Path]:
    if not run_name:
        raise ValueError("run_name is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
    root = project_root / "outputs" / "runs" / run_name
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"History output run already contains artifacts: {root}")
    paths = {
        "root": root,
        "models": root / "models",
        "oof": root / "oof",
        "metrics": root / "metrics",
        "submissions": root / "submissions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def preflight_history(config: HistoryTrainingConfig) -> dict[str, Any]:
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    train, test = load_competition_data(config.project_root)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    history_path = _project_path(config.project_root, config.history_path)
    column_map_path = (
        None
        if config.history_column_map_path is None
        else _project_path(config.project_root, config.history_column_map_path)
    )
    history = load_claim_history(history_path, train, test, column_map_path)
    return _history_preflight_report(
        history,
        train,
        test,
        config.n_splits,
        config.warmup_fraction,
        history_path,
        column_map_path,
    )


def run_history_training(config: HistoryTrainingConfig) -> dict[str, Any]:
    task_type = config.task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if config.early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be positive.")
    if config.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    train, test = load_competition_data(config.project_root)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    history_path = _project_path(config.project_root, config.history_path)
    column_map_path = (
        None
        if config.history_column_map_path is None
        else _project_path(config.project_root, config.history_column_map_path)
    )
    history = load_claim_history(history_path, train, test, column_map_path)
    bundle = build_causal_history_features(history)
    static_prepared = _prepare_ctr_features(train, test)
    experiments = _history_experiments(static_prepared, train, test, bundle)
    experiments_by_name = {experiment.name: experiment for experiment in experiments}
    history_indexed = history.set_index(ID_COL)
    temporal_cv = RollingOriginCV(
        history_indexed.loc[train[ID_COL], "event_at"],
        history_indexed.loc[train[ID_COL], "adjudicated_at"],
        n_splits=config.n_splits,
        warmup_fraction=config.warmup_fraction,
    )
    y = static_prepared.y
    temporal_cv.validate_labels(y)
    gpu_status = ensure_gpu_ready(config.devices) if task_type == "GPU" else "CPU explicitly selected"
    paths = history_output_paths(config.project_root, config.run_name)
    progress = tqdm(
        total=9 * config.n_splits,
        desc="History CatBoost folds",
        unit="fold",
        disable=not config.show_progress,
    )

    def fold_progress(label: str):
        def update(event: str, fold: int) -> None:
            progress.set_postfix_str(f"{label}, fold {fold + 1}/{config.n_splits}")
            if event == "complete":
                progress.update(1)

        return update

    def run_temporal(experiment: HistoryExperimentSpec, seed: int) -> dict[str, Any]:
        result = train_catboost_cv(
            experiment.prepared.X,
            y,
            experiment.prepared.X_test,
            experiment.prepared.categorical_features,
            cv=temporal_cv,
            params={**HISTORY_PARAMS, "random_seed": seed},
            task_type=task_type,
            devices=config.devices,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=config.catboost_verbose if config.show_progress else False,
            progress_callback=fold_progress(f"{experiment.name} seed {seed}"),
            require_complete_oof=False,
        )
        result.update(
            {
                "experiment_name": experiment.name,
                "experiment_stage": experiment.stage,
                "feature_groups": experiment.feature_groups,
                "notes": experiment.notes,
                "prepared": experiment.prepared,
                "evaluation_mask": temporal_cv.evaluation_mask,
                "oof_metrics": _temporal_metrics(y, result),
            }
        )
        return result

    try:
        screen_runs: dict[str, dict[str, Any]] = {}
        for experiment in experiments:
            progress.set_description(f"Screen {experiment.name}")
            screen_runs[experiment.name] = run_temporal(experiment, SCREEN_SEED)
        screen_results = _history_results_frame(screen_runs, train, y)
        screen_results.to_csv(paths["metrics"] / "history_experiments.csv", index=False)
        _save_screen_artifacts(paths, train, test, y, screen_runs)
        control = screen_runs["history_static_control"]
        selected_name = _select_experiment(
            screen_results.loc[screen_results["experiment_name"].ne("history_static_control")]
        )
        selected_spec = experiments_by_name[selected_name]
        control_spec = experiments_by_name["history_static_control"]
        selected = screen_runs[selected_name]
        screen_comparison, _ = _save_comparison(
            paths,
            "screen",
            _temporal_oof_frame(train, y, selected, SCREEN_SEED),
            _temporal_oof_frame(train, y, control, SCREEN_SEED),
            train,
            config.n_bootstrap,
        )
        screen_decision = _screen_decision(screen_comparison)
        _save_json(
            paths["metrics"] / "history_screen_decision.json",
            {"selected_experiment": selected_name, **screen_decision},
        )
        selected_seed_runs = {SCREEN_SEED: selected}
        control_seed_runs = {SCREEN_SEED: control}
        for seed in (2026,):
            progress.set_description(f"Confirmation seed {seed}")
            selected_seed_runs[seed] = run_temporal(selected_spec, seed)
            control_seed_runs[seed] = run_temporal(control_spec, seed)
        selected_ensemble = _combine_seed_runs(selected_seed_runs)
        control_ensemble = _combine_seed_runs(control_seed_runs)
        _save_seed_artifacts(paths, train, test, selected_name, selected_seed_runs)
        _save_seed_artifacts(paths, train, test, "history_static_control", control_seed_runs)
        ensemble_comparison, ensemble_fairness = _save_comparison(
            paths,
            "ensemble",
            _temporal_oof_frame(train, y, selected_ensemble, None),
            _temporal_oof_frame(train, y, control_ensemble, None),
            train,
            config.n_bootstrap,
        )

        progress.set_description(f"Fresh confirmation seed {CONFIRMATION_SEED}")
        fresh_selected = run_temporal(selected_spec, CONFIRMATION_SEED)
        fresh_control = run_temporal(control_spec, CONFIRMATION_SEED)
        _save_seed_artifacts(paths, train, test, selected_name, {CONFIRMATION_SEED: fresh_selected})
        _save_seed_artifacts(
            paths, train, test, "history_static_control", {CONFIRMATION_SEED: fresh_control}
        )
        fresh_comparison, _ = _save_comparison(
            paths,
            "fresh_seed_2718",
            _temporal_oof_frame(train, y, fresh_selected, CONFIRMATION_SEED),
            _temporal_oof_frame(train, y, fresh_control, CONFIRMATION_SEED),
            train,
            config.n_bootstrap,
        )

        progress.set_description("Provider grouped robustness")
        grouped_run = _run_grouped_robustness(
            selected,
            y,
            history_provider_groups(bundle.features, train[ID_COL]),
            config,
            task_type,
            fold_progress("provider grouped robustness"),
        )
        _save_grouped_artifacts(paths, grouped_run, y, config.n_bootstrap)

        decision = _promotion_decision(ensemble_comparison, ensemble_fairness, fresh_comparison)
        confirmation_promoted = bool(decision["promoted"])
        decision.update(
            {
                "selected_experiment": selected_name,
                "screen": screen_decision,
                "confirmation_promoted": confirmation_promoted,
                "promoted": confirmation_promoted and bool(screen_decision["eligible"]),
            }
        )
        final_test_predictions, final_iterations = _fit_final_models(
            selected,
            y,
            selected_seed_runs | {CONFIRMATION_SEED: fresh_selected},
            paths["models"],
            selected_name,
            config,
            task_type,
        )
        calibration = _calibrate_final_predictions(
            y,
            selected_ensemble,
            final_test_predictions,
        )
        _save_calibration_artifacts(paths, train, selected_ensemble, calibration)
        primary_test_predictions = calibration["test_pred"]
        submission_kind = "submission" if decision["promoted"] else "unpromoted_submission"
        submission_name = f"{selected_name}_{calibration['method']}_{submission_kind}.csv"
        submission_path = paths["submissions"] / submission_name
        make_submission(test[ID_COL], primary_test_predictions, submission_path)
        raw_submission_path = paths["submissions"] / f"{selected_name}_raw_{submission_kind}.csv"
        if calibration["method"] != "raw":
            make_submission(test[ID_COL], final_test_predictions, raw_submission_path)
        else:
            raw_submission_path = submission_path
        decision.update(
            {
                "submission_status": "promoted" if decision["promoted"] else "unpromoted",
                "submission_path": str(submission_path),
                "raw_submission_path": str(raw_submission_path),
            }
        )
        _save_json(paths["metrics"] / "history_promotion_decision.json", decision)
        _save_final_artifacts(
            paths,
            train,
            test,
            selected,
            selected_name,
            selected_ensemble,
            bundle,
            config,
            gpu_status,
            decision,
        )
        _save_json(
            paths["models"] / "history_final_fit.json",
            {"final_iterations": final_iterations, "ensemble_seeds": [42, 2026, 2718]},
        )
        return {
            "selected_experiment": selected_name,
            "promoted": bool(decision["promoted"]),
            "submission_status": decision["submission_status"],
            "submission_path": submission_path,
            "raw_submission_path": raw_submission_path,
            "calibration_method": calibration["method"],
            "promotion_decision": decision,
        }
    finally:
        progress.close()


def _project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _history_preflight_report(
    history: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    n_splits: int,
    warmup_fraction: float,
    history_path: Path,
    column_map_path: Path | None,
) -> dict[str, Any]:
    train_ids = train[ID_COL].astype("string").astype(str).to_numpy()
    test_ids = test[ID_COL].astype("string").astype(str).to_numpy()
    current_ids = set(train_ids) | set(test_ids)
    indexed = history.set_index(ID_COL)
    prior = history.loc[~history[ID_COL].isin(current_ids)].copy()
    prior_adjudicated = prior.loc[prior["adjudicated_label"].notna()].copy()
    adjudicated_times = prior_adjudicated["adjudicated_at"].to_numpy()

    def available_before(claim_ids: np.ndarray) -> int:
        events = indexed.loc[claim_ids, "event_at"].to_numpy()
        return int(sum(bool(np.any(adjudicated_times < event)) for event in events))

    temporal_folds: list[dict[str, Any]] = []
    temporal_error: str | None = None
    try:
        temporal_cv = RollingOriginCV(
            indexed.loc[train_ids, "event_at"],
            indexed.loc[train_ids, "adjudicated_at"],
            n_splits=n_splits,
            warmup_fraction=warmup_fraction,
        )
        temporal_cv.validate_labels(train[TARGET])
        temporal_folds = [
            {
                "fold": fold.fold,
                "train_rows": int(len(fold.train_idx)),
                "validation_rows": int(len(fold.valid_idx)),
                "validation_start": fold.validation_start.isoformat(),
            }
            for fold in temporal_cv.folds
        ]
    except ValueError as error:
        temporal_error = str(error)

    train_prior_coverage = available_before(train_ids)
    test_prior_coverage = available_before(test_ids)
    temporal_eligible = temporal_error is None
    prior_history_available = bool(len(prior_adjudicated))
    return {
        "history_path": str(history_path),
        "history_column_map_path": None if column_map_path is None else str(column_map_path),
        "history_rows": int(len(history)),
        "coverage": {
            "current_train_claims": int(len(train_ids)),
            "current_test_claims": int(len(test_ids)),
            "matched_current_train_claims": int(len(indexed.loc[train_ids])),
            "matched_current_test_claims": int(len(indexed.loc[test_ids])),
        },
        "prior_history": {
            "rows": int(len(prior)),
            "adjudicated_rows": int(len(prior_adjudicated)),
            "current_train_claims_with_prior_adjudicated_history": train_prior_coverage,
            "current_test_claims_with_prior_adjudicated_history": test_prior_coverage,
            "available": prior_history_available,
        },
        "temporal_eligibility": {
            "eligible": temporal_eligible,
            "reason": temporal_error,
            "evaluation_rows": int(
                sum(fold["validation_rows"] for fold in temporal_folds)
            ),
            "folds": temporal_folds,
        },
        "ready_to_train": bool(
            temporal_eligible and prior_history_available and train_prior_coverage > 0
        ),
    }


def _prepare_ctr_features(train: pd.DataFrame, test: pd.DataFrame) -> PreparedFeatures:
    return prepare_catboost_features(
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


def _history_experiments(
    static_prepared: PreparedFeatures,
    train: pd.DataFrame,
    test: pd.DataFrame,
    bundle: HistoryFeatureBundle,
) -> list[HistoryExperimentSpec]:
    return [
        HistoryExperimentSpec(
            "history_static_control",
            static_prepared,
            (),
            "Selected CTR feature recipe evaluated with rolling-origin time folds.",
        ),
        HistoryExperimentSpec(
            "history_financial",
            add_history_features(
                static_prepared, train[ID_COL], test[ID_COL], bundle.features, ("financial",)
            ),
            ("financial",),
            "Static CTR features plus current financial claim and tariff signals.",
        ),
        HistoryExperimentSpec(
            "history_behavioral",
            add_history_features(
                static_prepared,
                train[ID_COL],
                test[ID_COL],
                bundle.features,
                ("financial", "behavioral"),
            ),
            ("financial", "behavioral"),
            "Financial features plus causal provider, patient, timing, and peer-cost history.",
        ),
        HistoryExperimentSpec(
            "history_adjudication",
            add_history_features(
                static_prepared,
                train[ID_COL],
                test[ID_COL],
                bundle.features,
                ("financial", "behavioral", "adjudication"),
            ),
            ("financial", "behavioral", "adjudication"),
            "Behavioral history plus outcome rates available only after adjudication.",
        ),
    ]


def _temporal_metrics(y: pd.Series, result: dict[str, Any]) -> dict[str, float | int]:
    mask = np.asarray(result["fold_id"]) >= 0
    return evaluate_probabilities(y.iloc[mask], np.asarray(result["oof_pred"])[mask], AUDIT_FRACTIONS)


def _history_results_frame(
    experiment_runs: dict[str, dict[str, Any]], train: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    rows = []
    for name, result in experiment_runs.items():
        mask = np.asarray(result["evaluation_mask"], dtype=bool)
        fold_metrics = result["fold_metrics"]
        rows.append(
            {
                "experiment_name": name,
                "experiment_stage": result["experiment_stage"],
                "feature_groups": ",".join(result["feature_groups"]),
                "params": json.dumps(result["params"], sort_keys=True, default=str),
                "cv_strategy": "RollingOriginCV",
                "evaluation_rows": int(mask.sum()),
                **result["oof_metrics"],
                "mean_best_iteration": float(fold_metrics["best_iteration"].mean()),
                "iteration_cap": int(result["params"]["iterations"]),
                "iteration_cap_hit_rate": float(fold_metrics["hit_iteration_cap"].mean()),
                "fold_normalized_recall_5_std": float(
                    fold_metrics["normalized_recall_at_5pct"].std()
                ),
                "fairness_audit_rate_gap_5": _fairness_gap(
                    train.loc[mask], y.loc[mask], np.asarray(result["oof_pred"])[mask]
                ),
                "notes": result["notes"],
            }
        )
    return pd.DataFrame(rows)


def _fairness_gap(train: pd.DataFrame, y: pd.Series, probabilities: np.ndarray) -> float:
    if not {"jkpst", "umur"}.issubset(train.columns):
        return float("nan")
    from prs_its.fairness import fairness_across_budgets

    rates = pd.concat(
        [
            fairness_across_budgets(train["jkpst"], y, probabilities, group_name="gender"),
            fairness_across_budgets(
                age_groups(train["umur"]), y, probabilities, group_name="age_group"
            ),
        ],
        ignore_index=True,
    )
    eligible = rates.loc[
        rates["audit_fraction"].eq(0.05) & rates["eligible_for_comparison"]
    ]
    if eligible.empty:
        return float("nan")
    return float(eligible["audit_rate"].max() - eligible["audit_rate"].min())


def _temporal_oof_frame(
    train: pd.DataFrame,
    y: pd.Series,
    result: dict[str, Any],
    seed: int | None,
) -> pd.DataFrame:
    mask = np.asarray(result["fold_id"]) >= 0
    frame = pd.DataFrame(
        {
            ID_COL: train.loc[mask, ID_COL].to_numpy(),
            TARGET: y.loc[mask].to_numpy(),
            "fold": np.asarray(result["fold_id"])[mask],
            "fraud_probability_raw": np.asarray(result["oof_pred"])[mask],
        }
    )
    if seed is not None:
        frame["random_seed"] = seed
    return frame


def _save_screen_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    runs: dict[str, dict[str, Any]],
) -> None:
    fold_metrics = []
    for name, result in runs.items():
        _temporal_oof_frame(train, y, result, SCREEN_SEED).to_csv(
            paths["oof"] / f"{name}_temporal_oof_seed_{SCREEN_SEED}.csv", index=False
        )
        _save_test_fold_predictions(
            paths["oof"] / f"{name}_test_fold_predictions_seed_{SCREEN_SEED}.csv",
            test[ID_COL],
            result["test_fold_predictions"],
        )
        fold_metrics.append(
            result["fold_metrics"].assign(
                experiment_name=name,
                experiment_stage=result["experiment_stage"],
                cv_strategy="RollingOriginCV",
            )
        )
        result["models"].clear()
    pd.concat(fold_metrics, ignore_index=True).to_csv(
        paths["metrics"] / "history_experiment_fold_metrics.csv", index=False
    )


def _save_seed_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment_name: str,
    seed_runs: dict[int, dict[str, Any]],
) -> None:
    for seed, result in seed_runs.items():
        _temporal_oof_frame(train, result["prepared"].y, result, seed).to_csv(
            paths["oof"] / f"{experiment_name}_temporal_oof_seed_{seed}.csv", index=False
        )
        _save_test_fold_predictions(
            paths["oof"] / f"{experiment_name}_test_fold_predictions_seed_{seed}.csv",
            test[ID_COL],
            result["test_fold_predictions"],
        )
        result["models"].clear()


def _save_test_fold_predictions(
    path: Path, claim_ids: pd.Series, predictions: np.ndarray
) -> None:
    frame = pd.DataFrame({ID_COL: claim_ids.to_numpy()})
    for fold, values in enumerate(predictions):
        frame[f"fold_{fold}"] = values
    frame.to_csv(path, index=False)


def _combine_seed_runs(seed_runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    first_seed, first = next(iter(seed_runs.items()))
    fold_id = np.asarray(first["fold_id"])
    for seed, result in seed_runs.items():
        if not np.array_equal(fold_id, np.asarray(result["fold_id"])):
            raise RuntimeError(f"Seed {seed} did not use identical temporal validation folds.")
    oof_by_seed = np.vstack([np.asarray(result["oof_pred"]) for result in seed_runs.values()])
    oof_pred = np.full(oof_by_seed.shape[1], np.nan, dtype=float)
    evaluated = fold_id >= 0
    oof_pred[evaluated] = np.mean(oof_by_seed[:, evaluated], axis=0)
    return {
        "oof_pred": oof_pred,
        "test_pred": np.mean(
            np.vstack([np.asarray(result["test_pred"]) for result in seed_runs.values()]), axis=0
        ),
        "fold_id": fold_id,
        "fold_metrics": pd.concat(
            [
                result["fold_metrics"].assign(random_seed=seed)
                for seed, result in seed_runs.items()
            ],
            ignore_index=True,
        ),
        "feature_importance": pd.concat(
            [result["feature_importance"] for result in seed_runs.values()], ignore_index=True
        ),
        "params": first["params"],
        "prepared": first["prepared"],
        "evaluation_mask": first["evaluation_mask"],
    }


def _save_comparison(
    paths: dict[str, Path],
    name: str,
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    train: pd.DataFrame,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    comparison = paired_oof_comparison(candidate, incumbent, n_bootstrap=n_bootstrap)
    comparison.to_csv(paths["metrics"] / f"history_vs_control_{name}_paired.csv", index=False)
    if not {"jkpst", "umur"}.issubset(train.columns):
        return comparison, {"subgroup_rate_deltas": pd.DataFrame(), "gap_intervals": pd.DataFrame()}
    claim_index = train.set_index(ID_COL)
    subset = claim_index.loc[candidate[ID_COL]]
    fairness = paired_fairness_comparison(
        candidate[TARGET],
        candidate["fraud_probability_raw"],
        incumbent["fraud_probability_raw"],
        gender_groups=subset["jkpst"].reset_index(drop=True),
        age_group_values=age_groups(subset["umur"]).reset_index(drop=True),
        n_bootstrap=n_bootstrap,
    )
    fairness["subgroup_rate_deltas"].to_csv(
        paths["metrics"] / f"history_vs_control_{name}_fairness_rates.csv", index=False
    )
    fairness["gap_intervals"].to_csv(
        paths["metrics"] / f"history_vs_control_{name}_fairness_gaps.csv", index=False
    )
    return comparison, fairness


def _screen_decision(comparison: pd.DataFrame) -> dict[str, bool]:
    normalized = _comparison_row(comparison, "normalized_recall", 0.05)
    ap = _comparison_row(comparison, "average_precision", None)
    brier = _comparison_row(comparison, "brier_score", None)
    normalized_recall_noninferior = bool(float(normalized["delta"]) >= 0)
    average_precision_noninferior = bool(float(ap["delta"]) >= -AVERAGE_PRECISION_TOLERANCE)
    brier_noninferior = bool(float(brier["delta"]) <= 0)
    return {
        "eligible": normalized_recall_noninferior
        and average_precision_noninferior
        and brier_noninferior,
        "normalized_recall_noninferior": normalized_recall_noninferior,
        "average_precision_noninferior": average_precision_noninferior,
        "brier_noninferior": brier_noninferior,
    }


def _promotion_decision(
    ensemble_comparison: pd.DataFrame,
    ensemble_fairness: dict[str, pd.DataFrame],
    fresh_comparison: pd.DataFrame,
) -> dict[str, bool | float]:
    fraud = _comparison_row(ensemble_comparison, "fraud_caught", 0.05)
    normalized = _comparison_row(ensemble_comparison, "normalized_recall", 0.05)
    ap = _comparison_row(ensemble_comparison, "average_precision", None)
    brier = _comparison_row(ensemble_comparison, "brier_score", None)
    fresh = _screen_decision(fresh_comparison)
    fairness = ensemble_fairness["gap_intervals"]
    gender_fair = _fairness_nonregression(fairness, "gender")
    age_fair = _fairness_nonregression(fairness, "age_group")
    fraud_gain = float(fraud["delta"])
    fraud_capture_meaningful = fraud_gain >= MEANINGFUL_FRAUD_CAPTURE_GAIN
    fraud_capture_positive_interval = bool(float(fraud["ci_lower"]) > 0)
    normalized_recall_positive_interval = bool(float(normalized["ci_lower"]) > 0)
    average_precision_noninferior = bool(float(ap["delta"]) >= -AVERAGE_PRECISION_TOLERANCE)
    brier_noninferior = bool(float(brier["delta"]) <= 0)
    return {
        "promoted": fraud_capture_meaningful
        and fraud_capture_positive_interval
        and normalized_recall_positive_interval
        and average_precision_noninferior
        and brier_noninferior
        and gender_fair
        and age_fair
        and fresh["eligible"],
        "fraud_caught_delta_at_5pct": fraud_gain,
        "fraud_capture_meaningful": fraud_capture_meaningful,
        "fraud_capture_positive_interval": fraud_capture_positive_interval,
        "normalized_recall_positive_interval": normalized_recall_positive_interval,
        "average_precision_noninferior": average_precision_noninferior,
        "brier_noninferior": brier_noninferior,
        "gender_fairness_not_regressed": gender_fair,
        "age_fairness_not_regressed": age_fair,
        "fresh_seed_noninferior": bool(fresh["eligible"]),
    }


def _comparison_row(
    comparison: pd.DataFrame, metric: str, audit_fraction: float | None
) -> pd.Series:
    rows = comparison.loc[comparison["metric"].eq(metric)]
    if audit_fraction is None:
        rows = rows.loc[rows["audit_fraction"].isna()]
    else:
        rows = rows.loc[np.isclose(rows["audit_fraction"], audit_fraction, equal_nan=False)]
    if len(rows) != 1:
        raise ValueError(f"Comparison must contain one {metric!r} row.")
    return rows.iloc[0]


def _fairness_nonregression(fairness: pd.DataFrame, group_variable: str) -> bool:
    if fairness.empty:
        return False
    rows = fairness.loc[
        fairness["group_variable"].eq(group_variable)
        & np.isclose(fairness["audit_fraction"], 0.05, equal_nan=False)
    ]
    if len(rows) != 1 or not np.isfinite(float(rows.iloc[0]["ci_lower"])):
        return False
    return bool(float(rows.iloc[0]["ci_lower"]) <= 0)


def _run_grouped_robustness(
    selected: dict[str, Any],
    y: pd.Series,
    provider_groups: np.ndarray,
    config: HistoryTrainingConfig,
    task_type: str,
    progress_callback,
) -> dict[str, Any]:
    cv = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    result = train_catboost_cv(
        selected["prepared"].X,
        y,
        selected["prepared"].X_test,
        selected["prepared"].categorical_features,
        cv=cv,
        params={**HISTORY_PARAMS, "random_seed": SCREEN_SEED},
        task_type=task_type,
        devices=config.devices,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose=config.catboost_verbose if config.show_progress else False,
        progress_callback=progress_callback,
        groups=provider_groups,
        predict_test=False,
    )
    result["oof_metrics"] = evaluate_probabilities(y, result["oof_pred"], AUDIT_FRACTIONS)
    result["feature_group_count"] = int(len(np.unique(provider_groups)))
    result["models"].clear()
    return result


def _save_grouped_artifacts(
    paths: dict[str, Path],
    grouped: dict[str, Any],
    y: pd.Series,
    n_bootstrap: int,
) -> None:
    grouped["fold_metrics"].assign(scope="fold").to_csv(
        paths["metrics"] / "history_provider_grouped_fold_metrics.csv", index=False
    )
    pd.DataFrame([{**grouped["oof_metrics"], "scope": "overall"}]).to_csv(
        paths["metrics"] / "history_provider_grouped_metrics.csv", index=False
    )
    pd.DataFrame(
        bootstrap_audit_intervals(y, grouped["oof_pred"], n_bootstrap=n_bootstrap)
    ).to_csv(paths["metrics"] / "history_provider_grouped_bootstrap.csv", index=False)


def _save_final_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, Any],
    selected_name: str,
    ensemble: dict[str, Any],
    bundle: HistoryFeatureBundle,
    config: HistoryTrainingConfig,
    gpu_status: str,
    decision: dict[str, Any],
) -> None:
    mask = np.asarray(ensemble["fold_id"]) >= 0
    oof = _temporal_oof_frame(train, selected["prepared"].y, ensemble, None)
    oof.to_csv(paths["oof"] / "history_temporal_oof_raw.csv", index=False)
    aggregate_feature_importance(ensemble["feature_importance"]).to_csv(
        paths["metrics"] / "history_feature_importance.csv", index=False
    )
    pd.DataFrame(
        bootstrap_audit_intervals(
            selected["prepared"].y.loc[mask], ensemble["oof_pred"][mask], n_bootstrap=config.n_bootstrap
        )
    ).to_csv(paths["metrics"] / "history_temporal_audit_bootstrap.csv", index=False)
    _save_json(
        paths["models"] / "history_final_config.json",
        {
            "model": "CatBoostClassifier",
            "run_name": config.run_name,
            "history_path": str(config.history_path),
            "experiment": {
                "name": selected_name,
                "feature_groups": list(selected["feature_groups"]),
                "notes": selected["notes"],
            },
            "features": ensemble["prepared"].X.columns.tolist(),
            "categorical_features": ensemble["prepared"].categorical_features,
            "excluded_features": [ID_COL, "patient_id"],
            "params": ensemble["params"],
            "history_feature_schema": history_feature_schema(),
            "cv": {
                "type": "RollingOriginCV",
                "n_splits": config.n_splits,
                "warmup_fraction": config.warmup_fraction,
                "evaluation_rows": int(mask.sum()),
            },
            "ensemble_seeds": [*ENSEMBLE_SEEDS, CONFIRMATION_SEED],
            "fresh_confirmation_seed": CONFIRMATION_SEED,
            "promotion_decision": decision,
            "gpu_status": gpu_status,
            "versions": {
                "python": platform.python_version(),
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
                "catboost": catboost.__version__,
            },
            "test_rows": len(test),
        },
    )


def _fit_final_models(
    selected: dict[str, Any],
    y: pd.Series,
    seed_runs: dict[int, dict[str, Any]],
    model_dir: Path,
    selected_name: str,
    config: HistoryTrainingConfig,
    task_type: str,
) -> tuple[np.ndarray, dict[str, int]]:
    predictions = []
    iterations_by_seed: dict[str, int] = {}
    for seed, run in seed_runs.items():
        iterations = max(1, int(round(float(run["fold_metrics"]["best_iteration"].mean()))))
        params = {**run["params"], "iterations": iterations, "random_seed": seed}
        model = CatBoostClassifier(**params)
        model.fit(
            selected["prepared"].X,
            y,
            cat_features=selected["prepared"].categorical_features,
            verbose=config.catboost_verbose if config.show_progress else False,
        )
        model.save_model(model_dir / f"{selected_name}_final_seed_{seed}.cbm")
        predictions.append(np.asarray(model.predict_proba(selected["prepared"].X_test)[:, 1], dtype=float))
        iterations_by_seed[str(seed)] = iterations
    averaged = np.mean(np.vstack(predictions), axis=0)
    if not np.isfinite(averaged).all() or ((averaged < 0) | (averaged > 1)).any():
        raise RuntimeError("Final history predictions must be finite probabilities.")
    return averaged, iterations_by_seed


def _calibrate_final_predictions(
    y: pd.Series,
    ensemble: dict[str, Any],
    raw_test_pred: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(ensemble["fold_id"]) >= 0
    labels = y.loc[mask].to_numpy()
    raw_oof = np.asarray(ensemble["oof_pred"])[mask]
    fold_id = np.asarray(ensemble["fold_id"])[mask]
    raw_metrics = evaluate_probabilities(labels, raw_oof, AUDIT_FRACTIONS)
    candidates: list[tuple[str, np.ndarray, dict[str, float | int]]] = []
    calibration_oof: dict[str, np.ndarray] = {}
    for method in ("sigmoid", "isotonic"):
        calibrated = cross_fit_calibration(labels, raw_oof, fold_id, method)
        metrics = evaluate_probabilities(labels, calibrated["oof_pred"], AUDIT_FRACTIONS)
        candidates.append((method, calibrated["oof_pred"], metrics))
        calibration_oof[method] = calibrated["oof_pred"]
    eligible = [candidate for candidate in candidates if should_select_calibration(raw_metrics, candidate[2])]
    if not eligible:
        return {
            "method": "raw",
            "oof_pred": raw_oof,
            "test_pred": raw_test_pred,
            "raw_oof": raw_oof,
            "raw_metrics": raw_metrics,
            "calibration_oof": calibration_oof,
            "rows": [{"prediction_type": "raw", **raw_metrics}, *[
                {"prediction_type": method, **metrics} for method, _, metrics in candidates
            ]],
        }
    method, oof_pred, metrics = min(eligible, key=lambda candidate: candidate[2]["brier_score"])
    return {
        "method": method,
        "oof_pred": oof_pred,
        "test_pred": calibrate_test_predictions(raw_oof, labels, raw_test_pred, method),
        "raw_oof": raw_oof,
        "raw_metrics": raw_metrics,
        "calibration_oof": calibration_oof,
        "rows": [{"prediction_type": "raw", **raw_metrics}, *[
            {"prediction_type": name, **candidate_metrics}
            for name, _, candidate_metrics in candidates
        ]],
        "selected_metrics": metrics,
    }


def _save_calibration_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    ensemble: dict[str, Any],
    calibration: dict[str, Any],
) -> None:
    mask = np.asarray(ensemble["fold_id"]) >= 0
    oof = pd.DataFrame(
        {
            ID_COL: train.loc[mask, ID_COL].to_numpy(),
            TARGET: ensemble["prepared"].y.loc[mask].to_numpy(),
            "fold": np.asarray(ensemble["fold_id"])[mask],
            "fraud_probability_raw": calibration["raw_oof"],
            "fraud_probability_final": calibration["oof_pred"],
            **{
                f"fraud_probability_{method}": values
                for method, values in calibration["calibration_oof"].items()
            },
        }
    )
    oof.to_csv(paths["oof"] / "history_temporal_oof.csv", index=False)
    pd.DataFrame(calibration["rows"]).to_csv(
        paths["metrics"] / "history_calibration_metrics.csv", index=False
    )
    calibration_curve_frame(
        oof[TARGET], oof["fraud_probability_final"]
    ).to_csv(paths["metrics"] / "history_calibration_curve.csv", index=False)
    prediction_distribution(oof[TARGET], oof["fraud_probability_final"]).to_csv(
        paths["metrics"] / "history_prediction_distribution.csv", index=False
    )


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the causal history-enriched CatBoost challenger."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-name")
    parser.add_argument("--history-path", type=Path, required=True)
    parser.add_argument("--history-column-map", type=Path, default=None)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default=os.environ.get("PRS_ITS_HISTORY_TASK_TYPE", "GPU").upper(),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--catboost-verbose", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HistoryTrainingConfig(
        project_root=find_project_root(args.project_root),
        history_path=args.history_path,
        run_name=args.run_name or "preflight",
        history_column_map_path=args.history_column_map,
        task_type=args.task_type,
        show_progress=not args.quiet,
        catboost_verbose=args.catboost_verbose,
    )
    if args.preflight:
        result = preflight_history(config)
    else:
        if args.run_name is None:
            raise ValueError("--run-name is required unless --preflight is used.")
        result = run_history_training(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
