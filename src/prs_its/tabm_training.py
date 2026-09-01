from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Callable

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import torch
from tqdm.auto import tqdm

from prs_its.calibration import (
    calibrate_test_predictions,
    calibration_curve_frame,
    cross_fit_calibration,
    prediction_distribution,
    should_select_calibration,
)
from prs_its.fairness import age_groups, fairness_across_budgets
from prs_its.metrics import (
    bootstrap_audit_intervals,
    evaluate_probabilities,
    paired_fairness_comparison,
    paired_oof_comparison,
    validate_paired_oof,
)
from prs_its.modeling import (
    ID_COL,
    N_SPLITS,
    RANDOM_STATE,
    TARGET,
    PreparedFeatures,
    aggregate_feature_importance,
    feature_signature_groups,
    make_feature_spec,
    prepare_catboost_features,
    validate_train_test_schema,
)
from prs_its.submission import make_submission
from prs_its.tabm_modeling import (
    TABM_VARIANTS,
    TabMParams,
    ensure_tabm_gpu_ready,
    prepare_tabm_features,
    train_tabm_cv,
)
from prs_its.training import CTR_INTERACTION_FEATURES, _fairness_gap, _select_experiment, find_project_root, load_competition_data


AUDIT_FRACTIONS = (0.03, 0.05, 0.07)
SCREEN_SEED = RANDOM_STATE
ENSEMBLE_SEEDS = (SCREEN_SEED, 2026)
CONFIRMATION_SEED = 2718
TABM_BLEND_WEIGHTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MEANINGFUL_FRAUD_CAPTURE_GAIN = 20
AVERAGE_PRECISION_TOLERANCE = 0.005


@dataclass(frozen=True)
class TabMTrainingConfig:
    project_root: Path
    run_name: str
    incumbent_run_name: str = "ctr-v1"
    task_type: str = "GPU"
    n_splits: int = N_SPLITS
    random_state: int = RANDOM_STATE
    show_progress: bool = True
    n_bootstrap: int = 1000


@dataclass(frozen=True)
class CTROOFSource:
    run_dir: Path
    config: dict[str, Any]
    oof_by_seed: dict[int, pd.DataFrame]

    @property
    def n_splits(self) -> int:
        return int(self.config["cv"]["n_splits"])


@dataclass(frozen=True)
class TabMExperimentSpec:
    variant: str
    params: TabMParams
    prepared: PreparedFeatures
    notes: str


def tabm_output_paths(project_root: Path, run_name: str) -> dict[str, Path]:
    if not run_name:
        raise ValueError("run_name is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
    root = project_root / "outputs" / "runs" / run_name
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"TabM output run already contains artifacts: {root}")
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


def run_tabm_training(config: TabMTrainingConfig) -> dict[str, Any]:
    task_type = config.task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if config.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if config.run_name == config.incumbent_run_name:
        raise ValueError("run_name must differ from incumbent_run_name.")

    train, test = load_competition_data(config.project_root)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    source = load_ctr_oof_source(
        config.project_root,
        config.incumbent_run_name,
        train,
        expected_n_splits=config.n_splits,
        expected_random_state=config.random_state,
    )
    prepared = prepare_tabm_features(train, test, make_feature_spec(train, test))
    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    gpu_status = ensure_tabm_gpu_ready() if task_type == "GPU" else "CPU explicitly selected"
    paths = tabm_output_paths(config.project_root, config.run_name)
    progress = tqdm(
        total=5 * config.n_splits,
        desc="TabM folds",
        unit="fold",
        disable=not config.show_progress,
    )

    def fold_progress(label: str) -> Callable[[str, int], None]:
        def update(event: str, fold: int) -> None:
            progress.set_postfix_str(f"{label}, fold {fold + 1}/{config.n_splits}")
            if event == "complete":
                progress.update(1)

        return update

    def fit_experiment(
        experiment: TabMExperimentSpec,
        seed: int,
        label: str,
        *,
        groups: np.ndarray | None = None,
        predict_test: bool = True,
        save_models: bool = True,
        compute_feature_importance: bool = False,
    ) -> dict[str, Any]:
        result = train_tabm_cv(
            experiment.prepared.X,
            experiment.prepared.y,
            experiment.prepared.X_test,
            experiment.prepared.categorical_features,
            cv=cv if groups is None else StratifiedGroupKFold(
                n_splits=config.n_splits,
                shuffle=True,
                random_state=config.random_state,
            ),
            params=experiment.params,
            seed=seed,
            task_type=task_type,
            model_dir=paths["models"] if save_models else None,
            model_prefix=f"{experiment.variant}_seed_{seed}",
            progress_callback=fold_progress(label),
            groups=groups,
            predict_test=predict_test,
            compute_feature_importance=compute_feature_importance,
        )
        result.update(
            {
                "experiment_name": experiment.variant,
                "prepared": experiment.prepared,
                "notes": experiment.notes,
                "oof_metrics": evaluate_probabilities(
                    experiment.prepared.y, result["oof_pred"], AUDIT_FRACTIONS
                ),
            }
        )
        return result

    experiments = [
        TabMExperimentSpec(
            "tabm_base",
            TabMParams("tabm_base"),
            prepared,
            "TabM with train-fold categorical indices and quantile-normalized numeric features.",
        ),
        TabMExperimentSpec(
            "tabm_piecewise",
            TabMParams("tabm_piecewise"),
            prepared,
            "TabM with train-fold piecewise-linear numeric embeddings.",
        ),
    ]
    experiments_by_variant = {experiment.variant: experiment for experiment in experiments}
    screen_runs: dict[str, dict[str, Any]] = {}
    try:
        for experiment in experiments:
            progress.set_description(f"Screen {experiment.variant}")
            result = fit_experiment(experiment, SCREEN_SEED, f"screen {experiment.variant}")
            screen_runs[experiment.variant] = result
            _save_seed_artifact(paths, train, test, experiment.variant, SCREEN_SEED, result)
            _release_models(result)

        screen_results, screen_frames = _screen_results(
            screen_runs,
            source.oof_by_seed[SCREEN_SEED],
            train,
        )
        screen_results.to_csv(paths["metrics"] / "tabm_experiments.csv", index=False)
        _save_screen_artifacts(paths, screen_runs, screen_frames, source.oof_by_seed[SCREEN_SEED], train, config)
        candidates = screen_results.loc[screen_results["experiment_stage"].eq("screen_candidate")]
        selected_name = _select_experiment(candidates)
        selected_row = candidates.loc[candidates["experiment_name"].eq(selected_name)]
        if len(selected_row) != 1:
            raise RuntimeError("The selected TabM experiment is not uniquely identifiable.")
        selected_row = selected_row.iloc[0]
        selected_variant = str(selected_row["tabm_variant"])
        selected_weight = float(selected_row["tabm_weight"])
        control_row = screen_results.loc[screen_results["experiment_stage"].eq("screen_control")]
        if len(control_row) != 1:
            raise RuntimeError("The CTR screen control is not uniquely identifiable.")
        screen_decision = _screen_decision(selected_row, control_row.iloc[0])
        screen_decision.update(
            {
                "selected_experiment": selected_name,
                "selected_variant": selected_variant,
                "tabm_weight": selected_weight,
            }
        )
        _save_json(paths["metrics"] / "tabm_screen_decision.json", screen_decision)
        if not screen_decision["eligible"]:
            decision = {
                "selected_experiment": selected_name,
                "selected_variant": selected_variant,
                "tabm_weight": selected_weight,
                "promoted": False,
                "reason": "The selected TabM candidate did not clear the CTR screen guardrails.",
                "screen": screen_decision,
            }
            _save_json(paths["metrics"] / "tabm_promotion_decision.json", decision)
            _save_final_config(
                paths,
                config,
                source,
                experiments_by_variant[selected_variant],
                selected_name,
                selected_weight,
                {SCREEN_SEED: screen_runs[selected_variant]},
                None,
                gpu_status,
                decision,
                None,
            )
            return {
                "selected_experiment": selected_name,
                "selected_variant": selected_variant,
                "tabm_weight": selected_weight,
                "promoted": False,
                "submission_path": None,
                "screen_decision": screen_decision,
            }

        selected_experiment = experiments_by_variant[selected_variant]
        selected_seed_runs = {SCREEN_SEED: screen_runs[selected_variant]}
        progress.set_description("Confirmation seed 2026")
        selected_seed_runs[2026] = fit_experiment(
            selected_experiment,
            2026,
            "confirmation seed 2026",
            compute_feature_importance=True,
        )
        _save_seed_artifact(paths, train, test, selected_variant, 2026, selected_seed_runs[2026])
        _release_models(selected_seed_runs[2026])
        _save_seed_fold_metrics(paths, selected_variant, selected_seed_runs, "tabm_seed_fold_metrics.csv")
        selected_ensemble = _combine_seed_runs(selected_seed_runs)
        ctr_ensemble = _mean_seed_oof(source.oof_by_seed, ENSEMBLE_SEEDS)
        ensemble_candidate = _oof_frame(
            ctr_ensemble,
            _blend_probabilities(
                ctr_ensemble["fraud_probability_raw"], selected_ensemble["oof_pred"], selected_weight
            ),
        )
        ensemble_comparison, ensemble_fairness = _save_comparison(
            paths,
            f"{selected_name}_ensemble",
            ensemble_candidate,
            ctr_ensemble,
            train,
            config.n_bootstrap,
        )

        progress.set_description("Fresh confirmation seed 2718")
        fresh_run = fit_experiment(selected_experiment, CONFIRMATION_SEED, "fresh seed 2718")
        _save_seed_artifact(paths, train, test, selected_variant, CONFIRMATION_SEED, fresh_run)
        _release_models(fresh_run)
        fresh_ctr = source.oof_by_seed[CONFIRMATION_SEED]
        fresh_candidate = _oof_frame(
            fresh_ctr,
            _blend_probabilities(
                fresh_ctr["fraud_probability_raw"], fresh_run["oof_pred"], selected_weight
            ),
        )
        fresh_comparison, _ = _save_comparison(
            paths,
            f"{selected_name}_fresh_seed_{CONFIRMATION_SEED}",
            fresh_candidate,
            fresh_ctr,
            train,
            config.n_bootstrap,
        )

        progress.set_description("Feature-signature grouped robustness")
        grouped_run = fit_experiment(
            selected_experiment,
            SCREEN_SEED,
            "feature-signature grouped robustness",
            groups=feature_signature_groups(selected_experiment.prepared.X),
            predict_test=False,
            save_models=False,
        )
        _save_grouped_artifacts(paths, train, grouped_run, config.n_bootstrap)
        _release_models(grouped_run)
        calibration = _calibrate_oof(
            selected_ensemble["prepared"].y,
            ensemble_candidate,
        )
        _save_calibration_artifacts(paths, selected_name, ensemble_candidate, calibration)
        for method, calibrated_probabilities in calibration["cross_fitted_oof"].items():
            _save_comparison(
                paths,
                f"{selected_name}_ensemble_{method}_calibrated",
                _oof_frame(ctr_ensemble, calibrated_probabilities),
                ctr_ensemble,
                train,
                config.n_bootstrap,
            )
        _save_final_oof_artifacts(
            paths,
            train,
            test,
            selected_name,
            selected_ensemble,
            ensemble_candidate,
            grouped_run,
            calibration,
            config.n_bootstrap,
        )
        decision = _promotion_decision(ensemble_comparison, ensemble_fairness, fresh_comparison)
        decision.update(
            {
                "selected_experiment": selected_name,
                "selected_variant": selected_variant,
                "tabm_weight": selected_weight,
                "screen": screen_decision,
            }
        )
        _save_json(paths["metrics"] / "tabm_promotion_decision.json", decision)
        _save_final_config(
            paths,
            config,
            source,
            selected_experiment,
            selected_name,
            selected_weight,
            selected_seed_runs | {CONFIRMATION_SEED: fresh_run},
            grouped_run,
            gpu_status,
            decision,
            calibration,
        )
        if not decision["promoted"]:
            return {
                "selected_experiment": selected_name,
                "selected_variant": selected_variant,
                "tabm_weight": selected_weight,
                "promoted": False,
                "submission_path": None,
                "promotion_decision": decision,
            }

        all_seed_runs = {**selected_seed_runs, CONFIRMATION_SEED: fresh_run}
        tabm_test_by_seed = {
            seed: np.asarray(run["test_pred"], dtype=float)
            for seed, run in all_seed_runs.items()
        }
        tabm_test = np.mean(np.vstack([tabm_test_by_seed[seed] for seed in (*ENSEMBLE_SEEDS, CONFIRMATION_SEED)]), axis=0)
        if np.isclose(selected_weight, 1.0):
            raw_test_pred = tabm_test
        else:
            ctr_test_by_seed = reconstruct_ctr_test_predictions(source, train, test)
            ctr_test = np.mean(
                np.vstack([ctr_test_by_seed[seed] for seed in (*ENSEMBLE_SEEDS, CONFIRMATION_SEED)]),
                axis=0,
            )
            raw_test_pred = _blend_probabilities(ctr_test, tabm_test, selected_weight)
        final_test_pred = (
            raw_test_pred
            if calibration["method"] == "raw"
            else calibrate_test_predictions(
                ensemble_candidate["fraud_probability_raw"],
                selected_ensemble["prepared"].y,
                raw_test_pred,
                calibration["method"],
            )
        )
        raw_test_path = paths["oof"] / f"{selected_name}_test_raw.csv"
        pd.DataFrame({ID_COL: test[ID_COL], "fraud_probability_raw": raw_test_pred}).to_csv(
            raw_test_path, index=False
        )
        primary_name = f"{selected_name}_{calibration['method']}_submission.csv"
        submission_path = paths["submissions"] / primary_name
        make_submission(test[ID_COL], final_test_pred, submission_path)
        raw_submission_path = paths["submissions"] / f"{selected_name}_raw_submission.csv"
        if calibration["method"] != "raw":
            make_submission(test[ID_COL], raw_test_pred, raw_submission_path)
        else:
            raw_submission_path = submission_path
        return {
            "selected_experiment": selected_name,
            "selected_variant": selected_variant,
            "tabm_weight": selected_weight,
            "promoted": True,
            "calibration_method": calibration["method"],
            "submission_path": submission_path,
            "raw_submission_path": raw_submission_path,
            "promotion_decision": decision,
        }
    finally:
        progress.close()


def load_ctr_oof_source(
    project_root: Path,
    run_name: str,
    train: pd.DataFrame,
    *,
    expected_n_splits: int,
    expected_random_state: int,
) -> CTROOFSource:
    run_dir = project_root / "outputs" / "runs" / run_name
    source_config = _load_json(run_dir / "models" / "catboost_final_config.json")
    _validate_ctr_config(source_config)
    cv_config = source_config["cv"]
    if int(cv_config["n_splits"]) != expected_n_splits:
        raise ValueError("CTR source run uses a different number of CV folds than the TabM run.")
    if int(cv_config["random_state"]) != expected_random_state:
        raise ValueError("CTR source run uses a different CV random state than the TabM run.")
    expected_ids = train[ID_COL].reset_index(drop=True)
    expected_labels = train[TARGET].astype(int).reset_index(drop=True)
    expected_folds = _expected_folds(train, cv_config)
    oof_by_seed = {
        seed: _load_ctr_oof(
            run_dir / "oof" / f"catboost_oof_seed_{seed}.csv",
            expected_ids,
            expected_labels,
            expected_folds,
        )
        for seed in (*ENSEMBLE_SEEDS, CONFIRMATION_SEED)
    }
    return CTROOFSource(run_dir=run_dir, config=source_config, oof_by_seed=oof_by_seed)


def reconstruct_ctr_test_predictions(
    source: CTROOFSource,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[int, np.ndarray]:
    _validate_ctr_model_artifacts(
        source.run_dir,
        (*ENSEMBLE_SEEDS, CONFIRMATION_SEED),
        source.n_splits,
    )
    prepared = prepare_catboost_features(
        train,
        test,
        make_feature_spec(train, test),
        add_count_features=True,
        add_interaction_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features={"dati2_typeppk": CTR_INTERACTION_FEATURES["dati2_typeppk"]},
    )
    features = list(source.config["features"])
    categorical_features = list(source.config["categorical_features"])
    if list(prepared.X.columns) != features:
        raise ValueError("Current data no longer matches the saved CTR feature schema.")
    if list(prepared.categorical_features) != categorical_features:
        raise ValueError("Current data no longer matches the saved CTR categorical schema.")
    test_pool = Pool(prepared.X_test.loc[:, features], cat_features=categorical_features)
    predictions: dict[int, np.ndarray] = {}
    for seed in (*ENSEMBLE_SEEDS, CONFIRMATION_SEED):
        folds = []
        for fold in range(source.n_splits):
            model = CatBoostClassifier()
            model.load_model(source.run_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm")
            folds.append(np.asarray(model.predict_proba(test_pool)[:, 1], dtype=float))
        predictions[seed] = _validated_probabilities(
            np.mean(np.vstack(folds), axis=0), f"CTR test predictions for seed {seed}"
        )
    return predictions


def _screen_results(
    screen_runs: dict[str, dict[str, Any]],
    ctr_oof: pd.DataFrame,
    train: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = [_control_row(ctr_oof, train)]
    frames: dict[str, pd.DataFrame] = {}
    for variant, result in screen_runs.items():
        tabm_frame = _oof_frame_from_run(train, result, SCREEN_SEED)
        validate_paired_oof(tabm_frame, ctr_oof)
        raw_name = _experiment_name(variant, 1.0)
        frames[raw_name] = tabm_frame
        rows.append(_candidate_row(raw_name, variant, 1.0, result, tabm_frame, train, "raw"))
        for weight in TABM_BLEND_WEIGHTS:
            name = _experiment_name(variant, weight)
            candidate = _oof_frame(
                ctr_oof,
                _blend_probabilities(
                    ctr_oof["fraud_probability_raw"], tabm_frame["fraud_probability_raw"], weight
                ),
            )
            frames[name] = candidate
            rows.append(_candidate_row(name, variant, weight, result, candidate, train, "fixed_blend"))
    return pd.DataFrame(rows), frames


def _control_row(ctr_oof: pd.DataFrame, train: pd.DataFrame) -> dict[str, Any]:
    probabilities = ctr_oof["fraud_probability_raw"].to_numpy(dtype=float)
    return {
        "experiment_name": "ctr_control_raw",
        "experiment_stage": "screen_control",
        "tabm_variant": "ctr_control",
        "tabm_weight": 0.0,
        "notes": "Saved CTR incumbent OOF predictions.",
        "params": "{}",
        "mean_best_epoch": 0.0,
        "mean_best_iteration": 0.0,
        "fold_normalized_recall_5_std": _fold_normalized_recall_std(
            ctr_oof[TARGET], probabilities, ctr_oof["fold"]
        ),
        "fairness_audit_rate_gap_5": _fairness_gap(train, ctr_oof[TARGET], probabilities),
        **evaluate_probabilities(ctr_oof[TARGET], probabilities, AUDIT_FRACTIONS),
    }


def _candidate_row(
    name: str,
    variant: str,
    weight: float,
    result: dict[str, Any],
    candidate: pd.DataFrame,
    train: pd.DataFrame,
    stage: str,
) -> dict[str, Any]:
    probabilities = candidate["fraud_probability_raw"].to_numpy(dtype=float)
    fold_metrics = result["fold_metrics"]
    return {
        "experiment_name": name,
        "experiment_stage": "screen_candidate",
        "tabm_variant": variant,
        "tabm_weight": weight,
        "candidate_type": stage,
        "notes": result["notes"],
        "params": json.dumps(result["params"], sort_keys=True),
        "mean_best_epoch": float(fold_metrics["best_epoch"].mean()),
        "mean_best_iteration": float(fold_metrics["best_epoch"].mean()),
        "fold_normalized_recall_5_std": _fold_normalized_recall_std(
            candidate[TARGET], probabilities, candidate["fold"]
        ),
        "fairness_audit_rate_gap_5": _fairness_gap(train, candidate[TARGET], probabilities),
        **evaluate_probabilities(candidate[TARGET], probabilities, AUDIT_FRACTIONS),
    }


def _fold_normalized_recall_std(
    labels: pd.Series,
    probabilities: np.ndarray,
    fold_id: pd.Series,
) -> float:
    values = []
    y = np.asarray(labels, dtype=int)
    pred = np.asarray(probabilities, dtype=float)
    folds = np.asarray(fold_id, dtype=int)
    for fold in np.unique(folds):
        mask = folds == fold
        values.append(evaluate_probabilities(y[mask], pred[mask], AUDIT_FRACTIONS)["normalized_recall_at_5pct"])
    return float(pd.Series(values).std())


def _save_screen_artifacts(
    paths: dict[str, Path],
    screen_runs: dict[str, dict[str, Any]],
    screen_frames: dict[str, pd.DataFrame],
    ctr_oof: pd.DataFrame,
    train: pd.DataFrame,
    config: TabMTrainingConfig,
) -> None:
    fold_metrics = []
    for variant, result in screen_runs.items():
        fold_metrics.append(
            result["fold_metrics"].assign(
                experiment_name=variant,
                experiment_stage="screen",
                random_seed=SCREEN_SEED,
                cv_strategy="StratifiedKFold",
            )
        )
    pd.concat(fold_metrics, ignore_index=True).to_csv(
        paths["metrics"] / "tabm_screen_fold_metrics.csv", index=False
    )
    for name, frame in screen_frames.items():
        frame.to_csv(paths["oof"] / f"{name}_screen_oof_seed_{SCREEN_SEED}.csv", index=False)
        _save_comparison(paths, f"{name}_screen", frame, ctr_oof, train, config.n_bootstrap)


def _save_seed_artifact(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    variant: str,
    seed: int,
    result: dict[str, Any],
) -> None:
    _oof_frame_from_run(train, result, seed).to_csv(
        paths["oof"] / f"{variant}_oof_seed_{seed}.csv", index=False
    )
    if result["test_fold_predictions"] is not None:
        _save_test_fold_predictions(
            paths["oof"] / f"{variant}_test_fold_predictions_seed_{seed}.csv",
            test[ID_COL],
            result["test_fold_predictions"],
        )


def _save_seed_fold_metrics(
    paths: dict[str, Path],
    variant: str,
    seed_runs: dict[int, dict[str, Any]],
    filename: str,
) -> None:
    pd.concat(
        [
            result["fold_metrics"].assign(
                experiment_name=variant,
                random_seed=seed,
                cv_strategy="StratifiedKFold",
            )
            for seed, result in seed_runs.items()
        ],
        ignore_index=True,
    ).to_csv(paths["metrics"] / filename, index=False)


def _combine_seed_runs(seed_runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    first_seed, first = next(iter(seed_runs.items()))
    fold_id = np.asarray(first["fold_id"], dtype=int)
    for seed, result in seed_runs.items():
        if not np.array_equal(fold_id, np.asarray(result["fold_id"], dtype=int)):
            raise RuntimeError(f"TabM seed {seed} did not use the same validation folds.")
    return {
        "oof_pred": np.mean(
            np.vstack([np.asarray(result["oof_pred"], dtype=float) for result in seed_runs.values()]), axis=0
        ),
        "test_pred": np.mean(
            np.vstack([np.asarray(result["test_pred"], dtype=float) for result in seed_runs.values()]), axis=0
        ),
        "test_fold_predictions": np.vstack(
            [np.asarray(result["test_fold_predictions"], dtype=float) for result in seed_runs.values()]
        ),
        "fold_id": fold_id,
        "fold_metrics": pd.concat(
            [result["fold_metrics"].assign(random_seed=seed) for seed, result in seed_runs.items()],
            ignore_index=True,
        ),
        "feature_importance": pd.concat(
            [result["feature_importance"] for result in seed_runs.values()], ignore_index=True
        ),
        "params": first["params"],
        "prepared": first["prepared"],
        "fold_model_features": {
            str(seed): result["fold_model_features"] for seed, result in seed_runs.items()
        },
    }


def _mean_seed_oof(frames: dict[int, pd.DataFrame], seeds: tuple[int, ...]) -> pd.DataFrame:
    first = frames[seeds[0]].copy()
    for seed in seeds[1:]:
        validate_paired_oof(first, frames[seed])
    first["fraud_probability_raw"] = np.mean(
        np.vstack([frames[seed]["fraud_probability_raw"].to_numpy(dtype=float) for seed in seeds]), axis=0
    )
    return first


def _oof_frame_from_run(train: pd.DataFrame, result: dict[str, Any], seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: train[ID_COL].to_numpy(),
            TARGET: result["prepared"].y.to_numpy(),
            "fold": np.asarray(result["fold_id"], dtype=int),
            "random_seed": seed,
            "fraud_probability_raw": np.asarray(result["oof_pred"], dtype=float),
        }
    )


def _oof_frame(reference: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: reference[ID_COL].to_numpy(),
            TARGET: reference[TARGET].to_numpy(),
            "fold": reference["fold"].to_numpy(dtype=int),
            "fraud_probability_raw": _validated_probabilities(probabilities, "candidate probabilities"),
        }
    )


def _save_comparison(
    paths: dict[str, Path],
    name: str,
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    train: pd.DataFrame,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    comparison = paired_oof_comparison(candidate, incumbent, n_bootstrap=n_bootstrap)
    comparison.to_csv(paths["metrics"] / f"{name}_vs_ctr_paired.csv", index=False)
    if not {"jkpst", "umur"}.issubset(train.columns):
        return comparison, {"subgroup_rate_deltas": pd.DataFrame(), "gap_intervals": pd.DataFrame()}
    fairness = paired_fairness_comparison(
        candidate[TARGET],
        candidate["fraud_probability_raw"],
        incumbent["fraud_probability_raw"],
        gender_groups=train["jkpst"],
        age_group_values=age_groups(train["umur"]),
        n_bootstrap=n_bootstrap,
    )
    fairness["subgroup_rate_deltas"].to_csv(
        paths["metrics"] / f"{name}_vs_ctr_fairness_rates.csv", index=False
    )
    fairness["gap_intervals"].to_csv(
        paths["metrics"] / f"{name}_vs_ctr_fairness_gaps.csv", index=False
    )
    return comparison, fairness


def _screen_decision(selected: pd.Series, control: pd.Series) -> dict[str, bool]:
    normalized_recall_noninferior = bool(
        selected["normalized_recall_at_5pct"] >= control["normalized_recall_at_5pct"]
    )
    average_precision_noninferior = bool(
        selected["average_precision"] >= control["average_precision"] - AVERAGE_PRECISION_TOLERANCE
    )
    brier_noninferior = bool(selected["brier_score"] <= control["brier_score"])
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
    fresh = _screen_decision_from_comparison(fresh_comparison)
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
        "fresh_seed_noninferior": fresh["eligible"],
    }


def _screen_decision_from_comparison(comparison: pd.DataFrame) -> dict[str, bool]:
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
    if len(rows) != 1:
        return False
    ci_lower = float(rows.iloc[0]["ci_lower"])
    return bool(np.isfinite(ci_lower) and ci_lower <= 0)


def _run_grouped_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **result["oof_metrics"],
        "feature_group_count": int(result["feature_group_count"]),
    }


def _save_grouped_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    grouped: dict[str, Any],
    n_bootstrap: int,
) -> None:
    groups = feature_signature_groups(grouped["prepared"].X)
    grouped["feature_group_count"] = int(len(np.unique(groups)))
    grouped["fold_metrics"].assign(
        scope="fold", cv_strategy="StratifiedGroupKFold"
    ).to_csv(paths["metrics"] / "tabm_grouped_robustness.csv", index=False)
    pd.DataFrame([{"scope": "overall", **_run_grouped_metrics(grouped)}]).to_csv(
        paths["metrics"] / "tabm_grouped_robustness_metrics.csv", index=False
    )
    _oof_frame_from_run(train, grouped, SCREEN_SEED).to_csv(
        paths["oof"] / "tabm_grouped_robustness_oof_seed_42.csv", index=False
    )
    pd.DataFrame(
        bootstrap_audit_intervals(
            grouped["prepared"].y,
            grouped["oof_pred"],
            audit_fractions=AUDIT_FRACTIONS,
            n_bootstrap=n_bootstrap,
        )
    ).to_csv(paths["metrics"] / "tabm_grouped_robustness_bootstrap.csv", index=False)


def _calibrate_oof(y: pd.Series, candidate: pd.DataFrame) -> dict[str, Any]:
    raw_oof = candidate["fraud_probability_raw"].to_numpy(dtype=float)
    fold_id = candidate["fold"].to_numpy(dtype=int)
    raw_metrics = evaluate_probabilities(y, raw_oof, AUDIT_FRACTIONS)
    candidates: list[tuple[str, np.ndarray, dict[str, float | int]]] = []
    cross_fitted_oof: dict[str, np.ndarray] = {}
    for method in ("sigmoid", "isotonic"):
        calibrated = cross_fit_calibration(y, raw_oof, fold_id, method)
        metrics = evaluate_probabilities(y, calibrated["oof_pred"], AUDIT_FRACTIONS)
        candidates.append((method, calibrated["oof_pred"], metrics))
        cross_fitted_oof[method] = calibrated["oof_pred"]
    eligible = [candidate for candidate in candidates if should_select_calibration(raw_metrics, candidate[2])]
    if eligible:
        method, selected_oof, selected_metrics = min(
            eligible, key=lambda candidate: candidate[2]["brier_score"]
        )
    else:
        method, selected_oof, selected_metrics = "raw", raw_oof, raw_metrics
    return {
        "method": method,
        "raw_oof": raw_oof,
        "selected_oof": selected_oof,
        "selected_metrics": selected_metrics,
        "cross_fitted_oof": cross_fitted_oof,
        "rows": [
            {"prediction_type": "raw", **raw_metrics},
            *[{"prediction_type": name, **metrics} for name, _, metrics in candidates],
        ],
    }


def _save_calibration_artifacts(
    paths: dict[str, Path],
    selected_name: str,
    candidate: pd.DataFrame,
    calibration: dict[str, Any],
) -> None:
    frame = candidate.copy()
    frame["fraud_probability_final"] = calibration["selected_oof"]
    for method, probabilities in calibration["cross_fitted_oof"].items():
        frame[f"fraud_probability_{method}"] = probabilities
        _oof_frame(candidate, probabilities).to_csv(
            paths["oof"] / f"{selected_name}_ensemble_oof_calibrated_{method}.csv", index=False
        )
    frame.to_csv(paths["oof"] / f"{selected_name}_ensemble_oof.csv", index=False)
    pd.DataFrame(calibration["rows"]).to_csv(
        paths["metrics"] / f"{selected_name}_calibration_metrics.csv", index=False
    )
    calibration_curve_frame(candidate[TARGET], calibration["selected_oof"]).to_csv(
        paths["metrics"] / f"{selected_name}_calibration_curve.csv", index=False
    )
    prediction_distribution(candidate[TARGET], calibration["selected_oof"]).to_csv(
        paths["metrics"] / f"{selected_name}_prediction_distribution.csv", index=False
    )


def _save_final_oof_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_name: str,
    selected_ensemble: dict[str, Any],
    ensemble_candidate: pd.DataFrame,
    grouped_run: dict[str, Any],
    calibration: dict[str, Any],
    n_bootstrap: int,
) -> None:
    _save_test_fold_predictions(
        paths["oof"] / f"{selected_name}_tabm_test_fold_predictions_ensemble.csv",
        test[ID_COL],
        selected_ensemble["test_fold_predictions"],
    )
    pd.DataFrame(
        bootstrap_audit_intervals(
            ensemble_candidate[TARGET],
            ensemble_candidate["fraud_probability_raw"],
            audit_fractions=AUDIT_FRACTIONS,
            n_bootstrap=n_bootstrap,
        )
    ).to_csv(paths["metrics"] / f"{selected_name}_audit_bootstrap.csv", index=False)
    fairness = pd.concat(
        [
            fairness_across_budgets(
                train["jkpst"], ensemble_candidate[TARGET], calibration["selected_oof"], group_name="gender"
            ),
            fairness_across_budgets(
                age_groups(train["umur"]),
                ensemble_candidate[TARGET],
                calibration["selected_oof"],
                group_name="age_group",
            ),
        ],
        ignore_index=True,
    )
    fairness.to_csv(paths["metrics"] / f"{selected_name}_fairness.csv", index=False)
    aggregate_feature_importance(selected_ensemble["feature_importance"]).to_csv(
        paths["metrics"] / f"{selected_name}_feature_importance.csv", index=False
    )


def _release_models(result: dict[str, Any]) -> None:
    for model in result["models"]:
        model.to("cpu")
    result["models"].clear()
    result["fold_preprocessors"].clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _save_test_fold_predictions(path: Path, claim_ids: pd.Series, predictions: np.ndarray) -> None:
    values = np.asarray(predictions, dtype=float)
    frame = pd.DataFrame({ID_COL: claim_ids.to_numpy()})
    for fold, fold_values in enumerate(values):
        frame[f"fold_{fold}"] = fold_values
    frame.to_csv(path, index=False)


def _save_final_config(
    paths: dict[str, Path],
    config: TabMTrainingConfig,
    source: CTROOFSource,
    experiment: TabMExperimentSpec,
    selected_name: str,
    selected_weight: float,
    seed_runs: dict[int, dict[str, Any]],
    grouped_run: dict[str, Any] | None,
    gpu_status: str,
    decision: dict[str, Any],
    calibration: dict[str, Any] | None,
) -> None:
    first_run = next(iter(seed_runs.values()))
    payload = {
        "model": "TabM",
        "run_name": config.run_name,
        "incumbent_run_name": config.incumbent_run_name,
        "selected_experiment": selected_name,
        "tabm_variant": experiment.variant,
        "tabm_weight": selected_weight,
        "features": list(experiment.prepared.X.columns),
        "categorical_features": experiment.prepared.categorical_features,
        "excluded_features": [ID_COL],
        "params": experiment.params.as_dict(),
        "fold_model_features": {
            str(seed): run["fold_model_features"] for seed, run in seed_runs.items()
        },
        "preprocessor_artifacts": {
            str(seed): [
                f"{experiment.variant}_seed_{seed}_preprocessor_fold_{fold}.joblib"
                for fold in range(config.n_splits)
            ]
            for seed in seed_runs
        },
        "model_artifacts": {
            str(seed): [
                f"{experiment.variant}_seed_{seed}_fold_{fold}.pt"
                for fold in range(config.n_splits)
            ]
            for seed in seed_runs
        },
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": config.n_splits,
            "shuffle": True,
            "random_state": config.random_state,
        },
        "grouped_robustness": {
            "type": "StratifiedGroupKFold",
            "feature_group_count": grouped_run.get("feature_group_count") if grouped_run else None,
        },
        "ctr_source": {
            "run_name": config.incumbent_run_name,
            "model": source.config["model"],
            "profile": source.config["profile"],
            "experiment": source.config["experiment"],
        },
        "screen_seed": SCREEN_SEED,
        "ensemble_seeds": [*ENSEMBLE_SEEDS, CONFIRMATION_SEED],
        "fresh_confirmation_seed": CONFIRMATION_SEED,
        "calibration": calibration["method"] if calibration else "not_run",
        "promotion_decision": decision,
        "gpu_status": gpu_status,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "notes": experiment.notes,
        "first_seed_params": first_run["params"],
    }
    _save_json(paths["models"] / "tabm_final_config.json", payload)


def _experiment_name(variant: str, tabm_weight: float) -> str:
    if variant not in TABM_VARIANTS:
        raise ValueError(f"Unknown TabM variant: {variant}")
    if np.isclose(tabm_weight, 1.0):
        return f"{variant}_raw"
    percent = round(tabm_weight * 100)
    if not np.isclose(tabm_weight, percent / 100):
        raise ValueError("TabM blend weights must be representable to two decimal places.")
    return f"{variant}_ctr_blend_w{percent:02d}"


def _blend_probabilities(
    ctr_probabilities: pd.Series | np.ndarray,
    tabm_probabilities: pd.Series | np.ndarray,
    tabm_weight: float,
) -> np.ndarray:
    if not 0 <= tabm_weight <= 1:
        raise ValueError("tabm_weight must be within [0, 1].")
    ctr = _validated_probabilities(ctr_probabilities, "CTR probabilities")
    tabm = _validated_probabilities(tabm_probabilities, "TabM probabilities")
    if len(ctr) != len(tabm):
        raise ValueError("CTR and TabM probabilities must have the same length.")
    return (1 - tabm_weight) * ctr + tabm_weight * tabm


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CTR source configuration: {path}")
    with path.open() as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"CTR source configuration must be a JSON object: {path}")
    return payload


def _validate_ctr_config(source_config: dict[str, Any]) -> None:
    required = {"model", "profile", "features", "categorical_features", "experiment", "cv"}
    missing = sorted(required - set(source_config))
    if missing:
        raise ValueError(f"CTR source configuration is missing required fields: {missing}")
    if source_config["model"] != "CatBoostClassifier" or source_config["profile"] != "ctr":
        raise ValueError("TabM confirmation requires a saved CTR-profile CatBoost source run.")
    if source_config["experiment"].get("name") != "ctr_dati2_typeppk":
        raise ValueError("TabM confirmation requires the ctr_dati2_typeppk CTR source recipe.")
    if source_config.get("frequency_transformer") is not None:
        raise ValueError("TabM confirmation does not support CTR frequency-transformer source runs.")
    cv = source_config["cv"]
    required_cv = {"type", "n_splits", "shuffle", "random_state"}
    missing_cv = sorted(required_cv - set(cv)) if isinstance(cv, dict) else sorted(required_cv)
    if missing_cv:
        raise ValueError(f"CTR source CV configuration is missing fields: {missing_cv}")
    if cv["type"] != "StratifiedKFold" or not cv["shuffle"]:
        raise ValueError("CTR source run must use shuffled StratifiedKFold.")
    if int(cv["n_splits"]) < 2:
        raise ValueError("CTR source run must use at least two folds.")


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


def _load_ctr_oof(
    path: Path,
    expected_ids: pd.Series,
    expected_labels: pd.Series,
    expected_folds: np.ndarray,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CTR OOF artifact: {path}")
    frame = pd.read_csv(path)
    validate_paired_oof(frame, frame)
    if not frame[ID_COL].reset_index(drop=True).equals(expected_ids):
        raise ValueError(f"CTR OOF claim_id order does not match current training data: {path}")
    if not frame[TARGET].astype(int).reset_index(drop=True).equals(expected_labels):
        raise ValueError(f"CTR OOF labels do not match current training data: {path}")
    if not np.array_equal(frame["fold"].to_numpy(dtype=int), expected_folds):
        raise ValueError(f"CTR OOF folds do not match the saved CV configuration: {path}")
    return frame.reset_index(drop=True)


def _validate_ctr_model_artifacts(run_dir: Path, seeds: tuple[int, ...], n_splits: int) -> None:
    missing = [
        run_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm"
        for seed in seeds
        for fold in range(n_splits)
        if not (run_dir / "models" / f"catboost_seed_{seed}_fold_{fold}.cbm").exists()
    ]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required CTR fold model artifacts: {rendered}")


def _validated_probabilities(values: pd.Series | np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if len(probabilities) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{name} must contain finite probabilities within [0, 1].")
    return probabilities


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the staged TabM challenger and fixed CTR blend candidates."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--incumbent-run-name", default="ctr-v1")
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default=os.environ.get("PRS_ITS_TABM_TASK_TYPE", "GPU").upper(),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tabm_training(
        TabMTrainingConfig(
            project_root=find_project_root(args.project_root),
            run_name=args.run_name,
            incumbent_run_name=args.incumbent_run_name,
            task_type=args.task_type,
            show_progress=not args.quiet,
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
