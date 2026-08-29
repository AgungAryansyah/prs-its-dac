from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
from typing import Any

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from catboost import Pool
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from prs_its.calibration import (
    calibrate_test_predictions,
    calibration_curve_frame,
    cross_fit_calibration,
    prediction_distribution,
    should_select_calibration,
)
from prs_its.fairness import age_groups, fairness_across_budgets
from prs_its.metrics import audit_metrics, evaluate_probabilities
from prs_its.modeling import (
    BASE_PARAMS,
    ID_COL,
    N_SPLITS,
    RANDOM_STATE,
    TARGET,
    PreparedFeatures,
    aggregate_feature_importance,
    code_like_dtypes,
    ensure_gpu_ready,
    make_feature_spec,
    prepare_catboost_features,
    train_catboost_cv,
    validate_train_test_schema,
)
from prs_its.submission import make_submission


@dataclass(frozen=True)
class TrainingConfig:
    project_root: Path
    task_type: str = "GPU"
    devices: str = "0"
    n_splits: int = N_SPLITS
    random_state: int = RANDOM_STATE
    early_stopping_rounds: int = 200
    show_progress: bool = True
    catboost_verbose: int = 100
    profile: str = "baseline"
    run_name: str | None = None
    iterations: int | None = None
    ensemble_seeds: tuple[int, ...] = (RANDOM_STATE, 2026, 2718)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    prepared: PreparedFeatures
    params: dict[str, Any]
    feature_set: str
    notes: str


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate pyproject.toml.")


def output_paths(
    project_root: Path, profile: str = "baseline", run_name: str | None = None
) -> dict[str, Path]:
    output_dir = project_root / "outputs"
    if profile == "refined":
        name = run_name or "refined"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
        output_dir = output_dir / "runs" / name
    elif profile != "baseline":
        raise ValueError("profile must be 'baseline' or 'refined'.")
    paths = {
        "models": output_dir / "models",
        "oof": output_dir / "oof",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "submissions": output_dir / "submissions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_competition_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = project_root / "data"
    train = pd.read_csv(data_dir / "train.csv", dtype=code_like_dtypes())
    test = pd.read_csv(data_dir / "test.csv", dtype=code_like_dtypes())
    if ID_COL not in train or ID_COL not in test:
        raise ValueError("Both datasets must include claim_id.")
    if TARGET not in train or TARGET in test:
        raise ValueError("Training data must include label and test data must not.")
    if not set(train[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("label must contain only 0 and 1.")
    return train, test


def _baseline_experiment_specs(
    baseline: PreparedFeatures, count_features: PreparedFeatures
) -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            "unweighted_baseline",
            baseline,
            BASE_PARAMS.copy(),
            "original",
            "Original features; no class weighting.",
        ),
        ExperimentSpec(
            "balanced_baseline",
            baseline,
            {**BASE_PARAMS, "auto_class_weights": "Balanced"},
            "original",
            "Original features; CatBoost balanced weights.",
        ),
        ExperimentSpec(
            "count_features",
            count_features,
            BASE_PARAMS.copy(),
            "count_features",
            "Original features plus diagnosis and procedure counts.",
        ),
        ExperimentSpec(
            "shallow_regularized",
            baseline,
            {
                **BASE_PARAMS,
                "depth": 4,
                "l2_leaf_reg": 10.0,
                "random_strength": 0.5,
                "bagging_temperature": 0.5,
            },
            "original",
            "Targeted shallow regularization.",
        ),
        ExperimentSpec(
            "deep_regularized",
            baseline,
            {
                **BASE_PARAMS,
                "depth": 8,
                "l2_leaf_reg": 10.0,
                "random_strength": 2.0,
                "bagging_temperature": 1.0,
            },
            "original",
            "Targeted deeper regularization.",
        ),
    ]


def _refined_experiment_specs(
    baseline: PreparedFeatures,
    count_features: PreparedFeatures,
    interaction_features: PreparedFeatures,
    combined_features: PreparedFeatures,
) -> list[ExperimentSpec]:
    deep_params = {
        **BASE_PARAMS,
        "depth": 8,
        "l2_leaf_reg": 10.0,
        "random_strength": 2.0,
        "bagging_temperature": 1.0,
    }
    return [
        ExperimentSpec(
            "deep_regularized_control",
            baseline,
            deep_params,
            "original",
            "Current deep regularized configuration.",
        ),
        ExperimentSpec(
            "deep_count_features",
            count_features,
            deep_params,
            "count_features",
            "Deep configuration plus diagnosis and procedure counts.",
        ),
        ExperimentSpec(
            "deep_interaction_features",
            interaction_features,
            deep_params,
            "interactions_and_los",
            "Deep configuration plus categorical interactions and LOS features.",
        ),
        ExperimentSpec(
            "deep_combined_features",
            combined_features,
            deep_params,
            "counts_interactions_and_los",
            "Deep configuration plus count, interaction, and LOS features.",
        ),
        ExperimentSpec(
            "deep_depth_7",
            baseline,
            {**deep_params, "depth": 7},
            "original",
            "Targeted shallower deep-model variant.",
        ),
        ExperimentSpec(
            "deep_depth_9",
            baseline,
            {**deep_params, "depth": 9},
            "original",
            "Targeted deeper-model variant.",
        ),
        ExperimentSpec(
            "deep_l2_20",
            baseline,
            {**deep_params, "l2_leaf_reg": 20.0},
            "original",
            "Stronger L2 regularization.",
        ),
        ExperimentSpec(
            "deep_random_strength_1",
            baseline,
            {**deep_params, "random_strength": 1.0},
            "original",
            "Reduced split-score randomness.",
        ),
        ExperimentSpec(
            "deep_bagging_temperature_0_5",
            baseline,
            {**deep_params, "bagging_temperature": 0.5},
            "original",
            "Reduced Bayesian bootstrap temperature.",
        ),
    ]


def _with_iteration_cap(params: dict[str, Any], iterations: int | None) -> dict[str, Any]:
    if iterations is None:
        return params.copy()
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    return {**params, "iterations": iterations}


def _validate_ensemble_seeds(seeds: tuple[int, ...]) -> None:
    if not seeds:
        raise ValueError("ensemble_seeds must not be empty.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("ensemble_seeds must be unique.")


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
        metrics = evaluate_probabilities(y.iloc[valid_mask], oof_pred[valid_mask])
        seed_metrics = pd.concat(
            [run["fold_metrics"].query("fold == @fold") for run in seed_runs.values()],
            ignore_index=True,
        )
        metrics.update(
            {
                "fold": int(fold),
                "train_size": int(seed_metrics["train_size"].iloc[0]),
                "valid_size": int(valid_mask.sum()),
                "train_fraud_prevalence": float(seed_metrics["train_fraud_prevalence"].iloc[0]),
                "valid_fraud_prevalence": float(y.iloc[valid_mask].mean()),
                "best_iteration": float(seed_metrics["best_iteration"].mean()),
            }
        )
        fold_metrics.append(metrics)

    seed_fold_metrics = pd.concat(
        [run["fold_metrics"].assign(random_seed=seed) for seed, run in seed_runs.items()],
        ignore_index=True,
    )
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
        "params": first_run["params"],
        "seed_runs": seed_runs,
        "seed_fold_metrics": seed_fold_metrics,
        "explanation_seed": first_seed,
    }


def _fairness_gap(train: pd.DataFrame, y: pd.Series, probabilities: np.ndarray) -> float:
    rates = pd.concat(
        [
            fairness_across_budgets(train["jkpst"], y, probabilities, group_name="jkpst"),
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


def _select_experiment(results: pd.DataFrame) -> str:
    best_normalized_recall = results["normalized_recall_at_5pct"].max()
    candidates = results.loc[
        results["normalized_recall_at_5pct"] >= best_normalized_recall - 0.005
    ]
    return str(
        candidates.sort_values(
            [
                "average_precision",
                "brier_score",
                "fold_normalized_recall_5_std",
                "fairness_audit_rate_gap_5",
                "mean_best_iteration",
            ],
            ascending=[False, True, True, True, True],
            na_position="last",
        ).iloc[0]["experiment_name"]
    )


def _save_figures(
    paths: dict[str, Path],
    experiment_runs: dict[str, dict[str, Any]],
    y: pd.Series,
    final_oof_pred: np.ndarray,
    calibration_curve: pd.DataFrame,
    calibration_method: str,
    feature_importance: pd.DataFrame,
) -> None:
    figure_dir = paths["figures"]
    ax = feature_importance.head(25).sort_values("mean_importance").plot.barh(
        x="feature", y="mean_importance", legend=False, figsize=(10, 8)
    )
    ax.set_title("CatBoost Predictive Contribution Across Folds")
    plt.tight_layout()
    plt.savefig(figure_dir / "catboost_feature_importance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    for name in ["unweighted_baseline", "balanced_baseline"]:
        if name not in experiment_runs:
            continue
        curve = calibration_curve_frame(y, experiment_runs[name]["oof_pred"])
        plt.plot(
            curve["mean_predicted_probability"],
            curve["observed_fraud_rate"],
            marker="o",
            label=name,
        )
    if {"unweighted_baseline", "balanced_baseline"} <= experiment_runs.keys():
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed fraud rate")
        plt.legend()
        plt.title("Class-Weighting Calibration Comparison")
        plt.tight_layout()
        plt.savefig(figure_dir / "class_weighting_calibration.png", dpi=160)
    plt.close()

    robustness = pd.DataFrame(
        [audit_metrics(y, final_oof_pred, fraction) for fraction in np.arange(0.01, 0.101, 0.01)]
    )
    ax = robustness.plot(x="audit_fraction", y="fraud_caught", marker="o", legend=False)
    ax.set_title("Fraud Capture by Audit Capacity")
    ax.set_xlabel("Audit fraction")
    ax.set_ylabel("Fraud claims captured")
    plt.tight_layout()
    plt.savefig(figure_dir / "fraud_capture_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    labels = y.to_numpy()
    plt.hist(final_oof_pred[labels == 0], bins=40, alpha=0.6, label="label = 0")
    plt.hist(final_oof_pred[labels == 1], bins=40, alpha=0.6, label="label = 1")
    plt.legend()
    plt.title("OOF Fraud-Probability Distribution by Label")
    plt.tight_layout()
    plt.savefig(figure_dir / "oof_probability_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="ideal")
    plt.plot(
        calibration_curve["mean_predicted_probability"],
        calibration_curve["observed_fraud_rate"],
        marker="o",
        label=calibration_method,
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title("OOF Reliability Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "calibration_curve.png", dpi=160)
    plt.close()


def _save_explanations(
    paths: dict[str, Path],
    train: pd.DataFrame,
    y: pd.Series,
    prepared: PreparedFeatures,
    final_run: dict[str, Any],
    final_oof_pred: np.ndarray,
    categorical_features: list[str],
) -> None:
    sample_size = min(2000, len(y))
    sample_indices = np.random.default_rng(RANDOM_STATE).choice(
        len(y), size=sample_size, replace=False
    )
    shap_rows = []
    for fold, model in enumerate(final_run["models"]):
        indices = sample_indices[final_run["fold_id"][sample_indices] == fold]
        if not len(indices):
            continue
        values = model.get_feature_importance(
            Pool(prepared.X.iloc[indices], cat_features=categorical_features), type="ShapValues"
        )[:, :-1]
        shap_rows.append(
            pd.DataFrame(
                {
                    "feature": prepared.X.columns,
                    "mean_abs_shap": np.abs(values).mean(axis=0),
                    "fold": fold,
                }
            )
        )
    if shap_rows:
        shap_summary = (
            pd.concat(shap_rows)
            .groupby("feature", as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
        )
        shap_summary.to_csv(paths["metrics"] / "catboost_shap_summary.csv", index=False)

    labels = y.to_numpy()
    case_masks = {
        "high_risk_fraud": (labels == 1) & (final_oof_pred >= np.quantile(final_oof_pred, 0.95)),
        "high_risk_legitimate": (labels == 0)
        & (final_oof_pred >= np.quantile(final_oof_pred[labels == 0], 0.95)),
        "missed_fraud": (labels == 1)
        & (final_oof_pred <= np.quantile(final_oof_pred[labels == 1], 0.05)),
        "low_risk_legitimate": (labels == 0)
        & (final_oof_pred <= np.quantile(final_oof_pred[labels == 0], 0.05)),
    }
    explanation_rows = []
    for case, mask in case_masks.items():
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        index = int(indices[0])
        model = final_run["models"][final_run["fold_id"][index]]
        values = model.get_feature_importance(
            Pool(prepared.X.iloc[[index]], cat_features=categorical_features), type="ShapValues"
        )[0, :-1]
        for feature_index in np.argsort(np.abs(values))[-5:][::-1]:
            explanation_rows.append(
                {
                    "case": case,
                    "claim_id": train.iloc[index][ID_COL],
                    "label": int(y.iloc[index]),
                    "oof_fraud_probability": final_oof_pred[index],
                    "feature": prepared.X.columns[feature_index],
                    "shap_contribution": values[feature_index],
                }
            )
    pd.DataFrame(explanation_rows).to_csv(
        paths["metrics"] / "catboost_representative_explanations.csv", index=False
    )


def run_training(config: TrainingConfig) -> dict[str, Any]:
    task_type = config.task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if config.profile not in {"baseline", "refined"}:
        raise ValueError("profile must be 'baseline' or 'refined'.")
    train, test = load_competition_data(config.project_root)
    paths = output_paths(config.project_root, profile=config.profile, run_name=config.run_name)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    spec = make_feature_spec(train, test)
    baseline = prepare_catboost_features(train, test, spec, add_count_features=False)
    count_features = prepare_catboost_features(train, test, spec, add_count_features=True)
    y = baseline.y
    cv = StratifiedKFold(
        n_splits=config.n_splits, shuffle=True, random_state=config.random_state
    )
    gpu_status = ensure_gpu_ready(config.devices) if task_type == "GPU" else "CPU explicitly selected"

    if config.profile == "baseline":
        experiment_specs = _baseline_experiment_specs(baseline, count_features)
    else:
        interaction_features = prepare_catboost_features(
            train,
            test,
            spec,
            add_interaction_features=True,
            add_los_features=True,
        )
        combined_features = prepare_catboost_features(
            train,
            test,
            spec,
            add_count_features=True,
            add_interaction_features=True,
            add_los_features=True,
        )
        experiment_specs = _refined_experiment_specs(
            baseline,
            count_features,
            interaction_features,
            combined_features,
        )
    if config.profile == "refined":
        _validate_ensemble_seeds(config.ensemble_seeds)
        final_training_runs = len(config.ensemble_seeds)
    else:
        final_training_runs = 1
    progress = tqdm(
        total=(len(experiment_specs) + final_training_runs) * config.n_splits,
        desc="CatBoost folds",
        unit="fold",
        disable=not config.show_progress,
    )

    def fold_progress(name: str):
        def update(event: str, fold: int) -> None:
            progress.set_postfix_str(f"{name}, fold {fold + 1}/{config.n_splits}")
            if event == "complete":
                progress.update(1)

        return update

    experiment_runs: dict[str, dict[str, Any]] = {}
    try:
        for index, experiment in enumerate(experiment_specs, start=1):
            name = experiment.name
            prepared = experiment.prepared
            params = _with_iteration_cap(experiment.params, config.iterations)
            progress.set_description(f"Experiment {index}/{len(experiment_specs)}")
            result = train_catboost_cv(
                prepared.X,
                y,
                prepared.X_test,
                prepared.categorical_features,
                cv=cv,
                params=params,
                task_type=task_type,
                devices=config.devices,
                early_stopping_rounds=config.early_stopping_rounds,
                verbose=config.catboost_verbose if config.show_progress else False,
                progress_callback=fold_progress(name),
            )
            result.update(
                {
                    "experiment_name": name,
                    "feature_set": experiment.feature_set,
                    "notes": experiment.notes,
                    "oof_metrics": evaluate_probabilities(y, result["oof_pred"]),
                    "prepared": prepared,
                }
            )
            result["models"].clear()
            experiment_runs[name] = result

        experiment_rows = []
        for name, result in experiment_runs.items():
            fold_metrics = result["fold_metrics"]
            experiment_rows.append(
                {
                    "experiment_name": name,
                    "feature_set": result["feature_set"],
                    "params": json.dumps(result["params"], sort_keys=True, default=str),
                    "class_weight_strategy": result["params"].get("auto_class_weights", "None"),
                    "cv_strategy": "StratifiedKFold",
                    **result["oof_metrics"],
                    "mean_best_iteration": fold_metrics["best_iteration"].mean(),
                    "fold_normalized_recall_5_std": fold_metrics[
                        "normalized_recall_at_5pct"
                    ].std(),
                    "fairness_audit_rate_gap_5": _fairness_gap(train, y, result["oof_pred"]),
                    "notes": result["notes"],
                }
            )
        experiment_results = pd.DataFrame(experiment_rows)
        experiment_results.to_csv(paths["metrics"] / "catboost_experiments.csv", index=False)
        selected_name = _select_experiment(experiment_results)
        selected = experiment_runs[selected_name]
        if config.profile == "baseline":
            progress.set_description("Final ensemble")
            final_run = train_catboost_cv(
                selected["prepared"].X,
                y,
                selected["prepared"].X_test,
                selected["prepared"].categorical_features,
                cv=cv,
                params=selected["params"],
                task_type=task_type,
                devices=config.devices,
                early_stopping_rounds=config.early_stopping_rounds,
                model_dir=paths["models"],
                model_prefix="catboost",
                verbose=config.catboost_verbose if config.show_progress else False,
                progress_callback=fold_progress("final ensemble"),
            )
        else:
            seed_runs = {}
            for seed in config.ensemble_seeds:
                progress.set_description(f"Seed ensemble {seed}")
                seed_runs[seed] = train_catboost_cv(
                    selected["prepared"].X,
                    y,
                    selected["prepared"].X_test,
                    selected["prepared"].categorical_features,
                    cv=cv,
                    params={**selected["params"], "random_seed": seed},
                    task_type=task_type,
                    devices=config.devices,
                    early_stopping_rounds=config.early_stopping_rounds,
                    model_dir=paths["models"],
                    model_prefix=f"catboost_seed_{seed}",
                    verbose=config.catboost_verbose if config.show_progress else False,
                    progress_callback=fold_progress(f"seed {seed}"),
                )
            final_run = _combine_seed_runs(seed_runs, y)
    finally:
        progress.close()
    raw_oof_pred = final_run["oof_pred"]
    raw_test_pred = final_run["test_pred"]
    raw_metrics = evaluate_probabilities(y, raw_oof_pred)
    calibrated_candidates: list[tuple[str, np.ndarray, dict[str, float | int]]] = []
    for method in ("sigmoid", "isotonic"):
        calibrated = cross_fit_calibration(y, raw_oof_pred, final_run["fold_id"], method)
        metrics = evaluate_probabilities(y, calibrated["oof_pred"])
        calibrated_candidates.append((method, calibrated["oof_pred"], metrics))
    calibration_rows = [{"prediction_type": "raw", **raw_metrics}]
    calibration_rows.extend(
        {"prediction_type": method, **metrics} for method, _, metrics in calibrated_candidates
    )
    eligible_calibration = [
        candidate for candidate in calibrated_candidates if should_select_calibration(raw_metrics, candidate[2])
    ]
    calibration_sidecar_method: str | None = None
    calibration_sidecar_pred: np.ndarray | None = None
    if config.profile == "refined":
        final_oof_pred = raw_oof_pred
        final_metrics = raw_metrics
        final_test_pred = raw_test_pred
        calibration_method = "raw"
        if eligible_calibration:
            calibration_sidecar_method, _, _ = min(
                eligible_calibration, key=lambda candidate: candidate[2]["brier_score"]
            )
            calibration_sidecar_pred = calibrate_test_predictions(
                raw_oof_pred, y, raw_test_pred, calibration_sidecar_method
            )
    elif eligible_calibration:
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

    if config.profile == "refined":
        for seed, seed_run in final_run["seed_runs"].items():
            pd.DataFrame(
                {
                    ID_COL: train[ID_COL],
                    TARGET: y,
                    "fold": seed_run["fold_id"],
                    "random_seed": seed,
                    "fraud_probability_raw": seed_run["oof_pred"],
                }
            ).to_csv(paths["oof"] / f"catboost_oof_seed_{seed}.csv", index=False)
        final_run["seed_fold_metrics"].to_csv(
            paths["metrics"] / "catboost_seed_fold_metrics.csv", index=False
        )
        pd.DataFrame(calibration_rows).to_csv(
            paths["metrics"] / "catboost_calibration_comparison.csv", index=False
        )

    oof_output = pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: y,
            "fold": final_run["fold_id"],
            "fraud_probability_raw": raw_oof_pred,
            "fraud_probability_final": final_oof_pred,
        }
    )
    oof_output.to_csv(paths["oof"] / "catboost_oof.csv", index=False)
    final_run["fold_metrics"].to_csv(paths["metrics"] / "catboost_fold_metrics.csv", index=False)
    fairness_results = pd.concat(
        [
            fairness_across_budgets(train["jkpst"], y, final_oof_pred, group_name="jkpst"),
            fairness_across_budgets(
                age_groups(train["umur"]), y, final_oof_pred, group_name="age_group"
            ),
        ],
        ignore_index=True,
    )
    fairness_results.to_csv(paths["metrics"] / "catboost_fairness.csv", index=False)
    feature_importance = aggregate_feature_importance(final_run["feature_importance"])
    feature_importance.to_csv(paths["metrics"] / "catboost_feature_importance.csv", index=False)
    calibration_curve = calibration_curve_frame(y, final_oof_pred)
    calibration_curve.to_csv(paths["metrics"] / "catboost_calibration_curve.csv", index=False)
    prediction_distribution(y, final_oof_pred).to_csv(
        paths["metrics"] / "catboost_prediction_distribution.csv", index=False
    )
    _save_figures(
        paths,
        experiment_runs,
        y,
        final_oof_pred,
        calibration_curve,
        calibration_method,
        feature_importance,
    )
    _save_explanations(
        paths,
        train,
        y,
        selected["prepared"],
        final_run,
        final_oof_pred,
        selected["prepared"].categorical_features,
    )

    final_config = {
        "model": "CatBoostClassifier",
        "profile": config.profile,
        "run_name": config.run_name,
        "features": list(selected["prepared"].X.columns),
        "categorical_features": selected["prepared"].categorical_features,
        "excluded_features": [ID_COL],
        "params": final_run["params"],
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": config.n_splits,
            "shuffle": True,
            "random_state": config.random_state,
        },
        "calibration": calibration_method,
        "calibration_sidecar": calibration_sidecar_method,
        "ensemble_seeds": list(config.ensemble_seeds) if config.profile == "refined" else None,
        "gpu_status": gpu_status,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "catboost": catboost.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    with (paths["models"] / "catboost_final_config.json").open("w") as file:
        json.dump(final_config, file, indent=2, default=str)
    submission = make_submission(
        test[ID_COL], final_test_pred, paths["submissions"] / "catboost_submission.csv"
    )
    calibrated_submission_path = None
    if calibration_sidecar_pred is not None:
        calibrated_submission_path = paths["submissions"] / "catboost_calibrated_submission.csv"
        make_submission(test[ID_COL], calibration_sidecar_pred, calibrated_submission_path)
    findings = pd.DataFrame(
        [
            {
                "Finding": "Selected CatBoost configuration",
                "Evidence": selected_name,
                "Decision": selected_name,
                "Reason": "OOF Normalized Recall@5%, AP, Brier, stability, fairness gap, and complexity.",
            },
            {
                "Finding": "Final probability treatment",
                "Evidence": calibration_method,
                "Decision": calibration_method,
                "Reason": (
                    "Raw probabilities preserve the audit ranking."
                    if config.profile == "refined"
                    else "Calibration must improve Brier without unacceptable ranking loss."
                ),
            },
            {
                "Finding": "Audit portfolio at 5%",
                "Evidence": json.dumps(audit_metrics(y, final_oof_pred, 0.05)),
                "Decision": "Human audit prioritization",
                "Reason": "The model ranks claims; investigators determine fraud.",
            },
        ]
    )
    findings.to_csv(paths["metrics"] / "catboost_final_findings.csv", index=False)
    return {
        "profile": config.profile,
        "selected_experiment": selected_name,
        "calibration_method": calibration_method,
        "metrics": final_metrics,
        "submission_path": paths["submissions"] / "catboost_submission.csv",
        "submission_rows": len(submission),
        "calibrated_submission_path": calibrated_submission_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NHPA CatBoost fraud-risk model.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default=os.environ.get("PRS_ITS_TASK_TYPE", "GPU").upper(),
    )
    parser.add_argument("--devices", default=os.environ.get("PRS_ITS_GPU_DEVICES", "0"))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--catboost-verbose", type=int, default=100)
    parser.add_argument("--profile", choices=["baseline", "refined"], default="baseline")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--ensemble-seeds", default="42,2026,2718")
    return parser.parse_args()


def parse_ensemble_seeds(raw_seeds: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(value.strip()) for value in raw_seeds.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("ensemble_seeds must be comma-separated integers.") from error
    _validate_ensemble_seeds(seeds)
    return seeds


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        project_root=find_project_root(args.project_root),
        task_type=args.task_type,
        devices=args.devices,
        show_progress=not args.quiet,
        catboost_verbose=args.catboost_verbose,
        profile=args.profile,
        run_name=args.run_name,
        iterations=args.iterations,
        ensemble_seeds=parse_ensemble_seeds(args.ensemble_seeds),
    )
    result = run_training(config)
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
