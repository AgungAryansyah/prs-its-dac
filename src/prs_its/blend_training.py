from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from prs_its.blend_modeling import (
    AUDIT_FRACTIONS,
    BLEND_WEIGHTS,
    ScreenDecision,
    blend_experiment_results,
    blend_probabilities,
    promotion_decision,
    select_screen_candidate,
)
from prs_its.blend_sources import (
    ALL_SEEDS,
    CONFIRMATION_SEED,
    SCREEN_SEEDS,
    BlendSourceArtifacts,
    load_blend_source_artifacts,
    reconstruct_ctr_test_predictions,
)
from prs_its.fairness import age_groups
from prs_its.metrics import paired_fairness_comparison, paired_oof_comparison, validate_paired_oof
from prs_its.modeling import ID_COL, TARGET, aggregate_feature_importance, make_feature_spec, validate_train_test_schema
from prs_its.submission import make_submission
from prs_its.training import find_project_root, load_competition_data
from prs_its.xgb_modeling import (
    FoldTargetEncoderTransformer,
    ensure_xgb_gpu_ready,
    prepare_xgb_features,
    train_xgb_cv,
)


@dataclass(frozen=True)
class BlendTrainingConfig:
    project_root: Path
    run_name: str
    ctr_run_name: str = "ctr-v1"
    xgb_run_name: str = "xgb-v1"
    task_type: str = "GPU"
    early_stopping_rounds: int = 200
    n_bootstrap: int = 1000
    show_progress: bool = True
    xgb_verbose: int | bool = 100


def blend_output_paths(project_root: Path, run_name: str) -> dict[str, Path]:
    if not run_name:
        raise ValueError("run_name is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
    output_dir = project_root / "outputs" / "runs" / run_name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Blend output run already contains artifacts: {output_dir}")
    paths = {
        "root": output_dir,
        "models": output_dir / "models",
        "oof": output_dir / "oof",
        "metrics": output_dir / "metrics",
        "submissions": output_dir / "submissions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_blend_training(config: BlendTrainingConfig) -> dict[str, Any]:
    task_type = config.task_type.upper()
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if config.early_stopping_rounds <= 0:
        raise ValueError("early_stopping_rounds must be positive.")
    if config.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if config.run_name in {config.ctr_run_name, config.xgb_run_name}:
        raise ValueError("run_name must differ from both source run names.")

    train, test = load_competition_data(config.project_root)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    sources = load_blend_source_artifacts(
        config.project_root,
        config.ctr_run_name,
        config.xgb_run_name,
        train,
        test,
    )
    paths = blend_output_paths(config.project_root, config.run_name)
    screen = _run_screen(paths, sources, train, config.n_bootstrap)
    screen["results"].to_csv(paths["metrics"] / "blend_experiments.csv", index=False)
    _save_screen_comparisons(paths, screen, train, config.n_bootstrap)
    if not screen["decision"].eligible:
        decision = {
            "status": "screen_rejected",
            "screen": screen["decision"].as_dict(),
            "submission_path": None,
        }
        _save_final_config(paths, config, sources, decision, None, "not_run")
        _save_decision(paths, decision)
        return {
            "status": "screen_rejected",
            "selected_experiment": screen["decision"].selected_name,
            "selected_weight": screen["decision"].selected_weight,
            "submission_path": None,
        }

    gpu_status = ensure_xgb_gpu_ready() if task_type == "GPU" else "CPU explicitly selected"
    confirmation = _run_confirmation(
        paths, sources, train, test, config, task_type, screen["decision"]
    )
    confirmation["paired"].to_csv(
        paths["metrics"] / f"{confirmation['name']}_vs_ctr_confirmation_paired.csv",
        index=False,
    )
    confirmation["fairness"]["subgroup_rate_deltas"].to_csv(
        paths["metrics"] / f"{confirmation['name']}_vs_ctr_confirmation_fairness_rates.csv",
        index=False,
    )
    confirmation["fairness"]["gap_intervals"].to_csv(
        paths["metrics"] / f"{confirmation['name']}_vs_ctr_confirmation_fairness_gaps.csv",
        index=False,
    )
    decision = {
        "status": "promoted" if confirmation["decision"].promoted else "confirmation_rejected",
        "screen": screen["decision"].as_dict(),
        "confirmation": confirmation["decision"].as_dict(),
        "selected_experiment": confirmation["name"],
        "selected_weight": confirmation["weight"],
    }
    submission_path = None
    if confirmation["decision"].promoted:
        submission_path = _save_promoted_submission(paths, sources, confirmation, train, test)
        decision["submission_path"] = str(submission_path)
    else:
        decision["submission_path"] = None
    _save_final_config(paths, config, sources, decision, confirmation, gpu_status)
    _save_decision(paths, decision)
    return {
        "status": decision["status"],
        "selected_experiment": confirmation["name"],
        "selected_weight": confirmation["weight"],
        "submission_path": submission_path,
    }


def _run_screen(
    paths: dict[str, Path],
    sources: BlendSourceArtifacts,
    train: pd.DataFrame,
    n_bootstrap: int,
) -> dict[str, Any]:
    ctr_oof = _mean_seed_oof(sources.ctr_oof_by_seed, SCREEN_SEEDS)
    xgb_oof = _mean_seed_oof(sources.xgb_oof_by_seed, SCREEN_SEEDS)
    validate_paired_oof(xgb_oof, ctr_oof)
    results, predictions = blend_experiment_results(
        ctr_oof[TARGET],
        ctr_oof["fraud_probability_raw"],
        xgb_oof["fraud_probability_raw"],
        ctr_oof["fold"],
        train["jkpst"],
        age_groups(train["umur"]),
        weights=BLEND_WEIGHTS,
    )
    for name, probabilities in predictions.items():
        _oof_frame(ctr_oof, probabilities).to_csv(paths["oof"] / f"{name}_screen_oof.csv", index=False)
    return {
        "ctr_oof": ctr_oof,
        "predictions": predictions,
        "results": results,
        "decision": select_screen_candidate(results),
        "n_bootstrap": n_bootstrap,
    }


def _save_screen_comparisons(
    paths: dict[str, Path],
    screen: dict[str, Any],
    train: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    ctr_oof = screen["ctr_oof"]
    for name, probabilities in screen["predictions"].items():
        candidate = _oof_frame(ctr_oof, probabilities)
        paired = paired_oof_comparison(candidate, ctr_oof, n_bootstrap=n_bootstrap)
        paired.to_csv(paths["metrics"] / f"{name}_vs_ctr_screen_paired.csv", index=False)
        fairness = paired_fairness_comparison(
            candidate[TARGET],
            candidate["fraud_probability_raw"],
            ctr_oof["fraud_probability_raw"],
            gender_groups=train["jkpst"],
            age_group_values=age_groups(train["umur"]),
            n_bootstrap=n_bootstrap,
        )
        fairness["subgroup_rate_deltas"].to_csv(
            paths["metrics"] / f"{name}_vs_ctr_screen_fairness_rates.csv", index=False
        )
        fairness["gap_intervals"].to_csv(
            paths["metrics"] / f"{name}_vs_ctr_screen_fairness_gaps.csv", index=False
        )


def _run_confirmation(
    paths: dict[str, Path],
    sources: BlendSourceArtifacts,
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: BlendTrainingConfig,
    task_type: str,
    screen_decision: ScreenDecision,
) -> dict[str, Any]:
    prepared = _prepare_xgb_source_features(train, test, sources.xgb_config)
    xgb_run = _train_confirmation_xgb(
        paths,
        sources,
        prepared,
        config,
        task_type,
    )
    confirmation_xgb = pd.DataFrame(
        {
            ID_COL: train[ID_COL],
            TARGET: train[TARGET].astype(int),
            "fold": xgb_run["fold_id"],
            "random_seed": CONFIRMATION_SEED,
            "fraud_probability_raw": xgb_run["oof_pred"],
        }
    )
    confirmation_xgb.to_csv(
        paths["oof"] / f"xgb_oof_seed_{CONFIRMATION_SEED}.csv", index=False
    )
    _save_test_fold_predictions(
        paths["oof"] / f"xgb_test_fold_predictions_seed_{CONFIRMATION_SEED}.csv",
        test[ID_COL],
        xgb_run["test_fold_predictions"],
    )
    xgb_run["fold_metrics"].assign(random_seed=CONFIRMATION_SEED).to_csv(
        paths["metrics"] / "xgb_confirmation_fold_metrics.csv", index=False
    )
    aggregate_feature_importance(xgb_run["feature_importance"]).to_csv(
        paths["metrics"] / "xgb_confirmation_feature_importance.csv", index=False
    )
    ctr_confirmation = sources.ctr_oof_by_seed[CONFIRMATION_SEED]
    validate_paired_oof(confirmation_xgb, ctr_confirmation)
    probabilities = blend_probabilities(
        ctr_confirmation["fraud_probability_raw"],
        confirmation_xgb["fraud_probability_raw"],
        screen_decision.selected_weight,
    )
    candidate = _oof_frame(ctr_confirmation, probabilities)
    candidate.to_csv(
        paths["oof"] / f"{screen_decision.selected_name}_confirmation_oof.csv", index=False
    )
    paired = paired_oof_comparison(candidate, ctr_confirmation, n_bootstrap=config.n_bootstrap)
    fairness = paired_fairness_comparison(
        candidate[TARGET],
        candidate["fraud_probability_raw"],
        ctr_confirmation["fraud_probability_raw"],
        gender_groups=train["jkpst"],
        age_group_values=age_groups(train["umur"]),
        n_bootstrap=config.n_bootstrap,
    )
    return {
        "name": screen_decision.selected_name,
        "weight": screen_decision.selected_weight,
        "xgb_run": xgb_run,
        "paired": paired,
        "fairness": fairness,
        "decision": promotion_decision(paired, fairness["gap_intervals"]),
    }


def _prepare_xgb_source_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    xgb_config: dict[str, Any],
):
    spec = make_feature_spec(train, test)
    candidates = [
        prepare_xgb_features(train, test, spec, add_interaction_features=False),
        prepare_xgb_features(train, test, spec, add_interaction_features=True),
    ]
    expected_categorical_features = list(xgb_config["categorical_features"])
    matches = [
        prepared
        for prepared in candidates
        if prepared.categorical_features == expected_categorical_features
    ]
    if len(matches) != 1:
        raise ValueError("Saved XGBoost categorical schema does not match a supported source recipe.")
    return matches[0]


def _train_confirmation_xgb(
    paths: dict[str, Path],
    sources: BlendSourceArtifacts,
    prepared,
    config: BlendTrainingConfig,
    task_type: str,
) -> dict[str, Any]:
    cv_config = sources.xgb_config["cv"]
    cv = StratifiedKFold(
        n_splits=int(cv_config["n_splits"]),
        shuffle=True,
        random_state=int(cv_config["random_state"]),
    )
    encoder_config = sources.xgb_config["target_encoder"]
    params = {**sources.xgb_config["params"], "random_state": CONFIRMATION_SEED}
    progress = tqdm(
        total=int(cv_config["n_splits"]),
        desc="XGBoost confirmation folds",
        unit="fold",
        disable=not config.show_progress,
    )

    def progress_callback(event: str, fold: int) -> None:
        progress.set_postfix_str(f"seed {CONFIRMATION_SEED}, fold {fold + 1}/{cv_config['n_splits']}")
        if event == "complete":
            progress.update(1)

    try:
        result = train_xgb_cv(
            prepared.X,
            prepared.y,
            prepared.X_test,
            prepared.categorical_features,
            cv=cv,
            params=params,
            task_type=task_type,
            early_stopping_rounds=config.early_stopping_rounds,
            model_dir=paths["models"],
            model_prefix=f"xgb_seed_{CONFIRMATION_SEED}",
            verbose=config.xgb_verbose if config.show_progress else False,
            progress_callback=progress_callback,
            target_encoder_factory=lambda: FoldTargetEncoderTransformer(
                prepared.categorical_features,
                inner_n_splits=int(encoder_config["inner_n_splits"]),
                smooth=float(encoder_config["smooth"]),
                random_state=CONFIRMATION_SEED,
                add_support_counts=True,
            ),
        )
    finally:
        progress.close()
    if list(result["model_features"]) != list(sources.xgb_config["features"]):
        raise ValueError("Fresh XGBoost model features do not match the saved source configuration.")
    return result


def _save_promoted_submission(
    paths: dict[str, Path],
    sources: BlendSourceArtifacts,
    confirmation: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Path:
    ctr_test_by_seed = reconstruct_ctr_test_predictions(sources, train, test)
    xgb_test_by_seed = {
        **sources.xgb_test_by_seed,
        CONFIRMATION_SEED: confirmation["xgb_run"]["test_pred"],
    }
    ctr_ensemble = np.mean(np.vstack([ctr_test_by_seed[seed] for seed in ALL_SEEDS]), axis=0)
    xgb_ensemble = np.mean(np.vstack([xgb_test_by_seed[seed] for seed in ALL_SEEDS]), axis=0)
    probabilities = blend_probabilities(ctr_ensemble, xgb_ensemble, confirmation["weight"])
    name = confirmation["name"]
    pd.DataFrame(
        {ID_COL: test[ID_COL], "fraud_probability_raw": probabilities}
    ).to_csv(paths["oof"] / f"{name}_test_raw.csv", index=False)
    submission_path = paths["submissions"] / f"{name}_submission.csv"
    make_submission(test[ID_COL], probabilities, submission_path)
    return submission_path


def _mean_seed_oof(
    frames: dict[int, pd.DataFrame], seeds: tuple[int, ...]
) -> pd.DataFrame:
    first = frames[seeds[0]].copy()
    probabilities = np.vstack([frames[seed]["fraud_probability_raw"].to_numpy() for seed in seeds])
    first["fraud_probability_raw"] = np.mean(probabilities, axis=0)
    return first


def _oof_frame(reference: pd.DataFrame, probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: reference[ID_COL],
            TARGET: reference[TARGET],
            "fold": reference["fold"],
            "fraud_probability_raw": probabilities,
        }
    )


def _save_test_fold_predictions(path: Path, claim_ids: pd.Series, predictions: np.ndarray) -> None:
    frame = pd.DataFrame({ID_COL: claim_ids.to_numpy()})
    for fold, values in enumerate(predictions):
        frame[f"fold_{fold}"] = values
    frame.to_csv(path, index=False)


def _save_final_config(
    paths: dict[str, Path],
    config: BlendTrainingConfig,
    sources: BlendSourceArtifacts,
    decision: dict[str, Any],
    confirmation: dict[str, Any] | None,
    gpu_status: str,
) -> None:
    payload = {
        "model": "raw_ctr_xgb_blend",
        "run_name": config.run_name,
        "source_runs": {"ctr": config.ctr_run_name, "xgb": config.xgb_run_name},
        "screen_seeds": list(SCREEN_SEEDS),
        "confirmation_seed": CONFIRMATION_SEED,
        "final_ensemble_seeds": list(ALL_SEEDS) if confirmation is not None else [],
        "weights": list(BLEND_WEIGHTS),
        "calibration": "raw",
        "cv": sources.xgb_config["cv"],
        "task_type": config.task_type.upper(),
        "gpu_status": gpu_status,
        "decision": decision,
    }
    with (paths["models"] / "blend_final_config.json").open("w") as file:
        json.dump(payload, file, indent=2, default=str)


def _save_decision(paths: dict[str, Path], decision: dict[str, Any]) -> None:
    with (paths["metrics"] / "promotion_decision.json").open("w") as file:
        json.dump(decision, file, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm a constrained raw CTR-XGBoost blend with a fresh XGBoost seed."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ctr-run-name", default="ctr-v1")
    parser.add_argument("--xgb-run-name", default="xgb-v1")
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
    result = run_blend_training(
        BlendTrainingConfig(
            project_root=find_project_root(args.project_root),
            run_name=args.run_name,
            ctr_run_name=args.ctr_run_name,
            xgb_run_name=args.xgb_run_name,
            task_type=args.task_type,
            show_progress=not args.quiet,
            xgb_verbose=args.xgb_verbose,
        )
    )
    print(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
