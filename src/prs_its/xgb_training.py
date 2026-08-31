from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
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
    validate_train_test_schema,
)
from prs_its.submission import make_submission
from prs_its.training import _fairness_gap, _select_experiment, find_project_root, load_competition_data
from prs_its.xgb_modeling import (
    XGB_BASE_PARAMS,
    FoldTargetEncoderTransformer,
    ensure_xgb_gpu_ready,
    prepare_xgb_features,
    train_xgb_cv,
)


FINAL_SEEDS = (RANDOM_STATE, 2026)
AUDIT_FRACTIONS = (0.03, 0.05, 0.07)


@dataclass(frozen=True)
class XGBTrainingConfig:
    project_root: Path
    run_name: str
    incumbent_run_name: str = "ctr-v1"
    task_type: str = "GPU"
    n_splits: int = N_SPLITS
    random_state: int = RANDOM_STATE
    early_stopping_rounds: int = 200
    show_progress: bool = True
    xgb_verbose: int | bool = 100
    n_bootstrap: int = 1000


@dataclass(frozen=True)
class XGBExperimentSpec:
    name: str
    prepared: PreparedFeatures
    add_support_counts: bool
    feature_set: str
    notes: str
    stage: str


def xgb_output_paths(project_root: Path, run_name: str) -> dict[str, Path]:
    if not run_name:
        raise ValueError("run_name is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
    output_dir = project_root / "outputs" / "runs" / run_name
    paths = {
        "models": output_dir / "models",
        "oof": output_dir / "oof",
        "metrics": output_dir / "metrics",
        "submissions": output_dir / "submissions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_xgb_training(config: XGBTrainingConfig) -> dict[str, Any]:
    task_type = config.task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if config.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    train, test = load_competition_data(config.project_root)
    _validate_incumbent_paths(config.project_root, config.incumbent_run_name)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    spec = make_feature_spec(train, test)
    base_prepared = prepare_xgb_features(train, test, spec, add_interaction_features=False)
    interaction_prepared = prepare_xgb_features(train, test, spec, add_interaction_features=True)
    y = base_prepared.y
    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    gpu_status = ensure_xgb_gpu_ready() if task_type == "GPU" else "CPU explicitly selected"
    paths = xgb_output_paths(config.project_root, config.run_name)
    progress = tqdm(
        total=6 * config.n_splits,
        desc="XGBoost folds",
        unit="fold",
        disable=not config.show_progress,
    )

    def fold_progress(name: str) -> Callable[[str, int], None]:
        def update(event: str, fold: int) -> None:
            progress.set_postfix_str(f"{name}, fold {fold + 1}/{config.n_splits}")
            if event == "complete":
                progress.update(1)

        return update

    def target_encoder_factory(
        prepared: PreparedFeatures, seed: int, add_support_counts: bool
    ) -> Callable[[], FoldTargetEncoderTransformer]:
        return lambda: FoldTargetEncoderTransformer(
            prepared.categorical_features,
            random_state=seed,
            add_support_counts=add_support_counts,
        )

    def run_experiment(experiment: XGBExperimentSpec, seed: int, index: int) -> dict[str, Any]:
        progress.set_description(f"Screen {index}/3")
        result = train_xgb_cv(
            experiment.prepared.X,
            y,
            experiment.prepared.X_test,
            experiment.prepared.categorical_features,
            cv=cv,
            params={**XGB_BASE_PARAMS, "random_state": seed},
            task_type=task_type,
            early_stopping_rounds=config.early_stopping_rounds,
            verbose=config.xgb_verbose if config.show_progress else False,
            progress_callback=fold_progress(experiment.name),
            target_encoder_factory=target_encoder_factory(
                experiment.prepared, seed, experiment.add_support_counts
            ),
        )
        result.update(
            {
                "experiment_name": experiment.name,
                "experiment_stage": experiment.stage,
                "feature_set": experiment.feature_set,
                "add_support_counts": experiment.add_support_counts,
                "notes": experiment.notes,
                "prepared": experiment.prepared,
                "oof_metrics": evaluate_probabilities(y, result["oof_pred"], AUDIT_FRACTIONS),
            }
        )
        return result

    experiment_runs: dict[str, dict[str, Any]] = {}
    stage_one_specs = [
        XGBExperimentSpec(
            name="te_xgb_base",
            prepared=base_prepared,
            add_support_counts=False,
            feature_set="single target encodings with count, LOS, and bucket features",
            notes="Smoothed single-category target encodings.",
            stage="initial_screen",
        ),
        XGBExperimentSpec(
            name="te_xgb_interactions",
            prepared=interaction_prepared,
            add_support_counts=False,
            feature_set="single and targeted interaction target encodings",
            notes="Adds dati2_typeppk, diagprimer_cmg, and cmg_severitylevel target encodings.",
            stage="initial_screen",
        ),
    ]
    grouped_run: dict[str, Any] | None = None
    try:
        for index, experiment in enumerate(stage_one_specs, start=1):
            result = run_experiment(experiment, RANDOM_STATE, index)
            _discard_screen_models(result)
            experiment_runs[experiment.name] = result
        stage_one_results = _xgb_experiment_results_frame(experiment_runs, train, y)
        stage_one_winner = experiment_runs[_select_experiment(stage_one_results)]
        support_spec = XGBExperimentSpec(
            name="te_xgb_support",
            prepared=stage_one_winner["prepared"],
            add_support_counts=True,
            feature_set=f"{stage_one_winner['feature_set']} plus category support counts",
            notes=f"Fold-fitted support counts added to {stage_one_winner['experiment_name']}.",
            stage="support_follow_up",
        )
        support_result = run_experiment(support_spec, RANDOM_STATE, 3)
        _discard_screen_models(support_result)
        experiment_runs[support_spec.name] = support_result
        experiment_results = _xgb_experiment_results_frame(experiment_runs, train, y)
        experiment_results.to_csv(paths["metrics"] / "xgb_experiments.csv", index=False)
        _save_screen_artifacts(paths, train, y, experiment_runs)
        selected_name = _select_experiment(experiment_results)
        selected = experiment_runs[selected_name]

        seed_runs: dict[int, dict[str, Any]] = {}
        for seed in FINAL_SEEDS:
            progress.set_description(f"Final seed {seed}")
            seed_runs[seed] = train_xgb_cv(
                selected["prepared"].X,
                y,
                selected["prepared"].X_test,
                selected["prepared"].categorical_features,
                cv=cv,
                params={**XGB_BASE_PARAMS, "random_state": seed},
                task_type=task_type,
                early_stopping_rounds=config.early_stopping_rounds,
                model_dir=paths["models"],
                model_prefix=f"xgb_seed_{seed}",
                verbose=config.xgb_verbose if config.show_progress else False,
                progress_callback=fold_progress(f"final seed {seed}"),
                target_encoder_factory=target_encoder_factory(
                    selected["prepared"], seed, selected["add_support_counts"]
                ),
            )
        final_run = _combine_seed_runs(seed_runs, y)
        progress.set_description("Grouped robustness")
        grouped_run = _run_grouped_robustness(
            selected["prepared"],
            y,
            config,
            task_type,
            fold_progress("grouped robustness"),
            target_encoder_factory(
                selected["prepared"], RANDOM_STATE, selected["add_support_counts"]
            ),
        )
    finally:
        progress.close()

    if grouped_run is None:
        raise RuntimeError("XGBoost training did not produce grouped robustness results.")

    raw_oof_pred = final_run["oof_pred"]
    raw_test_pred = final_run["test_pred"]
    raw_metrics = evaluate_probabilities(y, raw_oof_pred, AUDIT_FRACTIONS)
    calibrated_candidates: list[tuple[str, np.ndarray, dict[str, float | int]]] = []
    calibration_oof: dict[str, np.ndarray] = {}
    for method in ("sigmoid", "isotonic"):
        calibrated = cross_fit_calibration(y, raw_oof_pred, final_run["fold_id"], method)
        calibrated_metrics = evaluate_probabilities(y, calibrated["oof_pred"], AUDIT_FRACTIONS)
        calibrated_candidates.append((method, calibrated["oof_pred"], calibrated_metrics))
        calibration_oof[method] = calibrated["oof_pred"]
    calibration_rows = [{"prediction_type": "raw", **raw_metrics}]
    calibration_rows.extend(
        {"prediction_type": method, **metrics}
        for method, _, metrics in calibrated_candidates
    )
    eligible_calibration = [
        candidate
        for candidate in calibrated_candidates
        if should_select_calibration(raw_metrics, candidate[2])
    ]
    if eligible_calibration:
        calibration_method, final_oof_pred, final_metrics = min(
            eligible_calibration, key=lambda candidate: candidate[2]["brier_score"]
        )
        final_test_pred = calibrate_test_predictions(
            raw_oof_pred, y, raw_test_pred, calibration_method
        )
    else:
        calibration_method = "raw"
        final_oof_pred = raw_oof_pred
        final_metrics = raw_metrics
        final_test_pred = raw_test_pred

    _save_final_artifacts(
        paths,
        train,
        test,
        y,
        final_run,
        grouped_run,
        raw_oof_pred,
        final_oof_pred,
        calibration_oof,
        calibration_rows,
        calibration_method,
        selected,
        selected_name,
        config,
        gpu_status,
    )
    _save_ctr_comparisons(
        paths,
        config.project_root,
        config.incumbent_run_name,
        train,
        y,
        final_run,
        raw_oof_pred,
        config.n_bootstrap,
    )
    submission_path = paths["submissions"] / "xgb_target_encoding_submission.csv"
    submission = make_submission(test[ID_COL], final_test_pred, submission_path)
    raw_submission_path = None
    if calibration_method != "raw":
        raw_submission_path = paths["submissions"] / "xgb_target_encoding_raw_submission.csv"
        make_submission(test[ID_COL], raw_test_pred, raw_submission_path)
    return {
        "selected_experiment": selected_name,
        "calibration_method": calibration_method,
        "metrics": final_metrics,
        "submission_path": submission_path,
        "submission_rows": len(submission),
        "raw_submission_path": raw_submission_path,
    }


def _discard_screen_models(result: dict[str, Any]) -> None:
    result["models"].clear()
    result["fold_transformers"].clear()


def _xgb_experiment_results_frame(
    experiment_runs: dict[str, dict[str, Any]], train: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    rows = []
    for name, result in experiment_runs.items():
        fold_metrics = result["fold_metrics"]
        rows.append(
            {
                "experiment_name": name,
                "experiment_stage": result["experiment_stage"],
                "feature_set": result["feature_set"],
                "add_support_counts": result["add_support_counts"],
                "params": json.dumps(result["params"], sort_keys=True, default=str),
                "cv_strategy": "StratifiedKFold",
                **result["oof_metrics"],
                "mean_best_iteration": fold_metrics["best_iteration"].mean(),
                "iteration_cap": int(result["params"]["n_estimators"]),
                "iteration_cap_hit_rate": fold_metrics["hit_iteration_cap"].mean(),
                "fold_normalized_recall_5_std": fold_metrics[
                    "normalized_recall_at_5pct"
                ].std(),
                "fairness_audit_rate_gap_5": _fairness_gap(train, y, result["oof_pred"]),
                "notes": result["notes"],
            }
        )
    return pd.DataFrame(rows)


def _combine_seed_runs(seed_runs: dict[int, dict[str, Any]], y: pd.Series) -> dict[str, Any]:
    first_seed, first_run = next(iter(seed_runs.items()))
    fold_id = np.asarray(first_run["fold_id"])
    for seed, run in seed_runs.items():
        if not np.array_equal(fold_id, run["fold_id"]):
            raise RuntimeError(f"Seed {seed} did not use the same validation folds.")
    oof_pred = np.mean(np.vstack([run["oof_pred"] for run in seed_runs.values()]), axis=0)
    test_pred = np.mean(np.vstack([run["test_pred"] for run in seed_runs.values()]), axis=0)
    fold_metrics = []
    for fold in np.unique(fold_id):
        valid_mask = fold_id == fold
        seed_metrics = pd.concat(
            [run["fold_metrics"].query("fold == @fold") for run in seed_runs.values()],
            ignore_index=True,
        )
        metrics = evaluate_probabilities(y.iloc[valid_mask], oof_pred[valid_mask], AUDIT_FRACTIONS)
        metrics.update(
            {
                "fold": int(fold),
                "train_size": int(seed_metrics["train_size"].iloc[0]),
                "valid_size": int(valid_mask.sum()),
                "train_fraud_prevalence": float(seed_metrics["train_fraud_prevalence"].iloc[0]),
                "valid_fraud_prevalence": float(y.iloc[valid_mask].mean()),
                "best_iteration": float(seed_metrics["best_iteration"].mean()),
                "iteration_cap": int(seed_metrics["iteration_cap"].iloc[0]),
                "hit_iteration_cap": bool(seed_metrics["hit_iteration_cap"].any()),
            }
        )
        fold_metrics.append(metrics)
    seed_fold_metrics = pd.concat(
        [run["fold_metrics"].assign(random_seed=seed) for seed, run in seed_runs.items()],
        ignore_index=True,
    )
    for seed, run in seed_runs.items():
        if seed != first_seed:
            run["models"].clear()
            run["fold_transformers"].clear()
    return {
        "oof_pred": oof_pred,
        "test_pred": test_pred,
        "test_fold_predictions": np.vstack(
            [run["test_fold_predictions"] for run in seed_runs.values()]
        ),
        "models": first_run["models"],
        "fold_metrics": pd.DataFrame(fold_metrics),
        "fold_id": fold_id,
        "feature_importance": pd.concat(
            [run["feature_importance"] for run in seed_runs.values()], ignore_index=True
        ),
        "model_features": first_run["model_features"],
        "fold_transformers": first_run["fold_transformers"],
        "params": first_run["params"],
        "seed_runs": seed_runs,
        "seed_fold_metrics": seed_fold_metrics,
    }


def _run_grouped_robustness(
    prepared: PreparedFeatures,
    y: pd.Series,
    config: XGBTrainingConfig,
    task_type: str,
    progress_callback: Callable[[str, int], None],
    target_encoder_factory: Callable[[], FoldTargetEncoderTransformer],
) -> dict[str, Any]:
    groups = feature_signature_groups(prepared.X)
    cv = StratifiedGroupKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    result = train_xgb_cv(
        prepared.X,
        y,
        prepared.X_test,
        prepared.categorical_features,
        cv=cv,
        params={**XGB_BASE_PARAMS, "random_state": RANDOM_STATE},
        task_type=task_type,
        early_stopping_rounds=config.early_stopping_rounds,
        verbose=config.xgb_verbose if config.show_progress else False,
        progress_callback=progress_callback,
        groups=groups,
        predict_test=False,
        target_encoder_factory=target_encoder_factory,
    )
    result["oof_metrics"] = evaluate_probabilities(y, result["oof_pred"], AUDIT_FRACTIONS)
    result["feature_group_count"] = int(len(np.unique(groups)))
    result["models"].clear()
    result["fold_transformers"].clear()
    return result


def _save_screen_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    y: pd.Series,
    experiment_runs: dict[str, dict[str, Any]],
) -> None:
    screen_fold_metrics = pd.concat(
        [
            result["fold_metrics"].assign(
                experiment_name=name,
                experiment_stage=result["experiment_stage"],
                feature_set=result["feature_set"],
                add_support_counts=result["add_support_counts"],
                cv_strategy="StratifiedKFold",
            )
            for name, result in experiment_runs.items()
        ],
        ignore_index=True,
    )
    screen_fold_metrics.to_csv(paths["metrics"] / "xgb_experiment_fold_metrics.csv", index=False)
    for name, result in experiment_runs.items():
        pd.DataFrame(
            {
                ID_COL: train[ID_COL],
                TARGET: y,
                "fold": result["fold_id"],
                "random_seed": RANDOM_STATE,
                "fraud_probability_raw": result["oof_pred"],
            }
        ).to_csv(paths["oof"] / f"{name}_oof_seed_{RANDOM_STATE}.csv", index=False)


def _save_final_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
    final_run: dict[str, Any],
    grouped_run: dict[str, Any],
    raw_oof_pred: np.ndarray,
    final_oof_pred: np.ndarray,
    calibration_oof: dict[str, np.ndarray],
    calibration_rows: list[dict[str, float | int | str]],
    calibration_method: str,
    selected: dict[str, Any],
    selected_name: str,
    config: XGBTrainingConfig,
    gpu_status: str,
) -> None:
    for seed, seed_run in final_run["seed_runs"].items():
        pd.DataFrame(
            {
                ID_COL: train[ID_COL],
                TARGET: y,
                "fold": seed_run["fold_id"],
                "random_seed": seed,
                "fraud_probability_raw": seed_run["oof_pred"],
            }
        ).to_csv(paths["oof"] / f"xgb_oof_seed_{seed}.csv", index=False)
        _save_test_fold_predictions(
            paths["oof"] / f"xgb_test_fold_predictions_seed_{seed}.csv",
            test[ID_COL],
            seed_run["test_fold_predictions"],
        )
    final_run["seed_fold_metrics"].to_csv(
        paths["metrics"] / "xgb_seed_fold_metrics.csv", index=False
    )
    final_run["fold_metrics"].to_csv(paths["metrics"] / "xgb_fold_metrics.csv", index=False)
    _save_test_fold_predictions(
        paths["oof"] / "xgb_test_fold_predictions.csv",
        test[ID_COL],
        final_run["test_fold_predictions"],
    )
    oof_output = pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": final_run["fold_id"],
            "fraud_probability_raw": raw_oof_pred,
            "fraud_probability_final": final_oof_pred,
            **{
                f"fraud_probability_{method}": probabilities
                for method, probabilities in calibration_oof.items()
            },
        }
    )
    oof_output.to_csv(paths["oof"] / "xgb_oof.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        paths["metrics"] / "xgb_calibration_comparison.csv", index=False
    )
    for method, probabilities in calibration_oof.items():
        pd.DataFrame(
            {
                ID_COL: train[ID_COL],
                TARGET: y,
                "fold": final_run["fold_id"],
                "fraud_probability_cross_fitted": probabilities,
            }
        ).to_csv(paths["oof"] / f"xgb_oof_calibrated_{method}.csv", index=False)
    grouped_summary = pd.concat(
        [
            grouped_run["fold_metrics"].assign(scope="fold", cv_strategy="StratifiedGroupKFold"),
            pd.DataFrame(
                [
                    {
                        "scope": "overall",
                        "cv_strategy": "StratifiedGroupKFold",
                        "feature_group_count": grouped_run["feature_group_count"],
                        **grouped_run["oof_metrics"],
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    grouped_summary.to_csv(paths["metrics"] / "xgb_grouped_robustness.csv", index=False)
    pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": grouped_run["fold_id"],
            "fraud_probability_raw": grouped_run["oof_pred"],
        }
    ).to_csv(paths["oof"] / "xgb_grouped_robustness_oof.csv", index=False)
    pd.DataFrame(
        bootstrap_audit_intervals(y, raw_oof_pred, audit_fractions=AUDIT_FRACTIONS, n_bootstrap=config.n_bootstrap)
    ).to_csv(paths["metrics"] / "xgb_audit_bootstrap.csv", index=False)
    pd.DataFrame(
        bootstrap_audit_intervals(
            y,
            grouped_run["oof_pred"],
            audit_fractions=AUDIT_FRACTIONS,
            n_bootstrap=config.n_bootstrap,
        )
    ).to_csv(paths["metrics"] / "xgb_grouped_robustness_bootstrap.csv", index=False)
    fairness_results = pd.concat(
        [
            fairness_across_budgets(train["jkpst"], y, final_oof_pred, group_name="gender"),
            fairness_across_budgets(
                age_groups(train["umur"]), y, final_oof_pred, group_name="age_group"
            ),
        ],
        ignore_index=True,
    )
    fairness_results.to_csv(paths["metrics"] / "xgb_fairness.csv", index=False)
    feature_importance = aggregate_feature_importance(final_run["feature_importance"])
    feature_importance.to_csv(paths["metrics"] / "xgb_feature_importance.csv", index=False)
    calibration_curve_frame(y, final_oof_pred).to_csv(
        paths["metrics"] / "xgb_calibration_curve.csv", index=False
    )
    prediction_distribution(y, final_oof_pred).to_csv(
        paths["metrics"] / "xgb_prediction_distribution.csv", index=False
    )
    transformer = final_run["fold_transformers"][0]
    final_config = {
        "model": "XGBClassifier",
        "run_name": config.run_name,
        "incumbent_run_name": config.incumbent_run_name,
        "features": final_run["model_features"],
        "categorical_features": selected["prepared"].categorical_features,
        "excluded_features": [ID_COL],
        "params": final_run["params"],
        "experiment": {
            "name": selected_name,
            "stage": selected["experiment_stage"],
            "add_support_counts": selected["add_support_counts"],
        },
        "target_encoder": {
            **transformer.as_dict(),
            "artifacts_by_seed": {
                str(seed): [
                    f"xgb_seed_{seed}_target_encoder_fold_{fold}.joblib"
                    for fold in range(config.n_splits)
                ]
                for seed in FINAL_SEEDS
            },
        },
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": config.n_splits,
            "shuffle": True,
            "random_state": config.random_state,
        },
        "calibration": calibration_method,
        "ensemble_seeds": list(FINAL_SEEDS),
        "iteration_diagnostics": {
            "iteration_cap": int(final_run["params"]["n_estimators"]),
            "final_fold_cap_hit_rate": float(final_run["fold_metrics"]["hit_iteration_cap"].mean()),
        },
        "gpu_status": gpu_status,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    with (paths["models"] / "xgb_final_config.json").open("w") as file:
        json.dump(final_config, file, indent=2, default=str)


def _save_test_fold_predictions(path: Path, claim_ids: pd.Series, predictions: np.ndarray) -> None:
    frame = pd.DataFrame({ID_COL: claim_ids.to_numpy()})
    for fold, values in enumerate(predictions):
        frame[f"fold_{fold}"] = values
    frame.to_csv(path, index=False)


def _save_ctr_comparisons(
    paths: dict[str, Path],
    project_root: Path,
    incumbent_run_name: str,
    train: pd.DataFrame,
    y: pd.Series,
    final_run: dict[str, Any],
    raw_oof_pred: np.ndarray,
    n_bootstrap: int,
) -> None:
    incumbent_seed_frames: dict[int, pd.DataFrame] = {}
    for seed, seed_run in final_run["seed_runs"].items():
        candidate = pd.DataFrame(
            {
                ID_COL: train[ID_COL],
                TARGET: y,
                "fold": seed_run["fold_id"],
                "fraud_probability_raw": seed_run["oof_pred"],
            }
        )
        incumbent = pd.read_csv(_incumbent_oof_path(project_root, incumbent_run_name, seed))
        incumbent_seed_frames[seed] = incumbent
        _save_single_ctr_comparison(
            paths,
            f"seed_{seed}",
            candidate,
            incumbent,
            train,
            n_bootstrap,
        )
    incumbent_ensemble = incumbent_seed_frames[FINAL_SEEDS[0]].copy()
    incumbent_ensemble["fraud_probability_raw"] = np.mean(
        np.vstack(
            [frame["fraud_probability_raw"].to_numpy() for frame in incumbent_seed_frames.values()]
        ),
        axis=0,
    )
    candidate_ensemble = pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": final_run["fold_id"],
            "fraud_probability_raw": raw_oof_pred,
        }
    )
    _save_single_ctr_comparison(
        paths,
        "ensemble",
        candidate_ensemble,
        incumbent_ensemble,
        train,
        n_bootstrap,
    )


def _save_single_ctr_comparison(
    paths: dict[str, Path],
    name: str,
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    train: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    paired_oof_comparison(candidate, incumbent, n_bootstrap=n_bootstrap).to_csv(
        paths["metrics"] / f"xgb_vs_ctr_{name}_paired.csv", index=False
    )
    fairness = paired_fairness_comparison(
        candidate[TARGET],
        candidate["fraud_probability_raw"],
        incumbent["fraud_probability_raw"],
        gender_groups=train["jkpst"],
        age_group_values=age_groups(train["umur"]),
        n_bootstrap=n_bootstrap,
    )
    fairness["subgroup_rate_deltas"].to_csv(
        paths["metrics"] / f"xgb_vs_ctr_{name}_fairness_rates.csv", index=False
    )
    fairness["gap_intervals"].to_csv(
        paths["metrics"] / f"xgb_vs_ctr_{name}_fairness_gaps.csv", index=False
    )


def _validate_incumbent_paths(project_root: Path, incumbent_run_name: str) -> None:
    missing = [
        path
        for seed in FINAL_SEEDS
        if not (path := _incumbent_oof_path(project_root, incumbent_run_name, seed)).exists()
    ]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing incumbent CTR seed OOF artifacts: {rendered}")


def _incumbent_oof_path(project_root: Path, incumbent_run_name: str, seed: int) -> Path:
    return (
        project_root
        / "outputs"
        / "runs"
        / incumbent_run_name
        / "oof"
        / f"catboost_oof_seed_{seed}.csv"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the leakage-safe target-encoded XGBoost challenger."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--incumbent-run-name", default="ctr-v1")
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default=os.environ.get("PRS_ITS_XGB_TASK_TYPE", "GPU").upper(),
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--xgb-verbose", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_xgb_training(
        XGBTrainingConfig(
            project_root=find_project_root(args.project_root),
            run_name=args.run_name,
            incumbent_run_name=args.incumbent_run_name,
            task_type=args.task_type,
            show_progress=not args.quiet,
            xgb_verbose=args.xgb_verbose,
        )
    )
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
