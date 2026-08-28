from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
from typing import Any

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from catboost import Pool
from sklearn.model_selection import StratifiedKFold

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


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not locate pyproject.toml.")


def output_paths(project_root: Path) -> dict[str, Path]:
    output_dir = project_root / "outputs"
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


def _experiment_specs(
    baseline: PreparedFeatures, count_features: PreparedFeatures
) -> list[tuple[str, PreparedFeatures, dict[str, Any], str]]:
    return [
        ("unweighted_baseline", baseline, BASE_PARAMS.copy(), "Original features; no class weighting."),
        (
            "balanced_baseline",
            baseline,
            {**BASE_PARAMS, "auto_class_weights": "Balanced"},
            "Original features; CatBoost balanced weights.",
        ),
        (
            "count_features",
            count_features,
            BASE_PARAMS.copy(),
            "Original features plus diagnosis and procedure counts.",
        ),
        (
            "shallow_regularized",
            baseline,
            {
                **BASE_PARAMS,
                "depth": 4,
                "l2_leaf_reg": 10.0,
                "random_strength": 0.5,
                "bagging_temperature": 0.5,
            },
            "Targeted shallow regularization.",
        ),
        (
            "deep_regularized",
            baseline,
            {
                **BASE_PARAMS,
                "depth": 8,
                "l2_leaf_reg": 10.0,
                "random_strength": 2.0,
                "bagging_temperature": 1.0,
            },
            "Targeted deeper regularization.",
        ),
    ]


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
        curve = calibration_curve_frame(y, experiment_runs[name]["oof_pred"])
        plt.plot(
            curve["mean_predicted_probability"],
            curve["observed_fraud_rate"],
            marker="o",
            label=name,
        )
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
    train, test = load_competition_data(config.project_root)
    paths = output_paths(config.project_root)
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

    experiment_runs: dict[str, dict[str, Any]] = {}
    for name, prepared, params, notes in _experiment_specs(baseline, count_features):
        result = train_catboost_cv(
            prepared.X,
            y,
            prepared.X_test,
            spec.categorical_features,
            cv=cv,
            params=params,
            task_type=task_type,
            devices=config.devices,
            early_stopping_rounds=config.early_stopping_rounds,
        )
        result.update(
            {
                "experiment_name": name,
                "notes": notes,
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
                "feature_set": "count_features" if name == "count_features" else "original",
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

    final_run = train_catboost_cv(
        selected["prepared"].X,
        y,
        selected["prepared"].X_test,
        spec.categorical_features,
        cv=cv,
        params=selected["params"],
        task_type=task_type,
        devices=config.devices,
        early_stopping_rounds=config.early_stopping_rounds,
        model_dir=paths["models"],
        model_prefix="catboost",
    )
    raw_oof_pred = final_run["oof_pred"]
    raw_test_pred = final_run["test_pred"]
    raw_metrics = evaluate_probabilities(y, raw_oof_pred)
    calibrated_candidates = []
    for method in ("sigmoid", "isotonic"):
        calibrated = cross_fit_calibration(y, raw_oof_pred, final_run["fold_id"], method)
        metrics = evaluate_probabilities(y, calibrated["oof_pred"])
        calibrated_candidates.append((method, calibrated["oof_pred"], metrics))
    eligible_calibration = [
        candidate for candidate in calibrated_candidates if should_select_calibration(raw_metrics, candidate[2])
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
        spec.categorical_features,
    )

    final_config = {
        "model": "CatBoostClassifier",
        "features": list(selected["prepared"].X.columns),
        "categorical_features": spec.categorical_features,
        "excluded_features": [ID_COL],
        "params": final_run["params"],
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": config.n_splits,
            "shuffle": True,
            "random_state": config.random_state,
        },
        "calibration": calibration_method,
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
                "Reason": "Calibration must improve Brier without unacceptable ranking loss.",
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
        "selected_experiment": selected_name,
        "calibration_method": calibration_method,
        "metrics": final_metrics,
        "submission_path": paths["submissions"] / "catboost_submission.csv",
        "submission_rows": len(submission),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NHPA CatBoost fraud-risk model.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default=os.environ.get("PRS_ITS_TASK_TYPE", "GPU").upper())
    parser.add_argument("--devices", default=os.environ.get("PRS_ITS_GPU_DEVICES", "0"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig(
        project_root=find_project_root(args.project_root),
        task_type=args.task_type,
        devices=args.devices,
    )
    result = run_training(config)
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
