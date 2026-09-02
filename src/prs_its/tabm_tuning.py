from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import pickle
import platform
import re
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import sklearn
import torch
from tqdm.auto import tqdm

from prs_its.calibration import calibrate_test_predictions
from prs_its.metrics import evaluate_probabilities, validate_paired_oof
from prs_its.modeling import (
    ID_COL,
    N_SPLITS,
    RANDOM_STATE,
    TARGET,
    feature_signature_groups,
    make_feature_spec,
    validate_train_test_schema,
)
from prs_its.submission import make_submission
from prs_its.tabm_modeling import (
    TabMParams,
    ensure_tabm_cuda_memory_ready,
    ensure_tabm_gpu_ready,
    prepare_tabm_features,
    train_tabm_cv,
)
from prs_its.tabm_training import (
    AUDIT_FRACTIONS,
    CONFIRMATION_SEED,
    ENSEMBLE_SEEDS,
    SCREEN_SEED,
    CTROOFSource,
    _blend_probabilities,
    _calibrate_oof,
    _combine_seed_runs,
    _control_row,
    _fold_normalized_recall_std,
    _mean_seed_oof,
    _oof_frame,
    _oof_frame_from_run,
    _promotion_decision,
    _release_models,
    _save_calibration_artifacts,
    _save_comparison,
    _save_final_oof_artifacts,
    _save_grouped_artifacts,
    _save_seed_artifact,
    _save_seed_fold_metrics,
    _screen_decision,
    load_ctr_oof_source,
    reconstruct_ctr_test_predictions,
)
from prs_its.training import _fairness_gap, find_project_root, load_competition_data


HPO_K_VALUES = (16, 32)
HPO_BLEND_WEIGHTS = tuple(round(weight / 100, 2) for weight in range(5, 51, 5))
HPO_BATCH_SIZE = 512


@dataclass(frozen=True)
class TabMTuningConfig:
    project_root: Path
    run_name: str
    incumbent_run_name: str = "ctr-v1"
    task_type: str = "GPU"
    trials: int = 30
    n_splits: int = N_SPLITS
    random_state: int = RANDOM_STATE
    show_progress: bool = True
    n_bootstrap: int = 1000
    resume: bool = False


def tabm_tuning_output_paths(
    project_root: Path,
    run_name: str,
    *,
    resume: bool,
) -> dict[str, Path]:
    if not run_name:
        raise ValueError("run_name is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
    root = project_root / "outputs" / "runs" / run_name
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"TabM HPO output run already contains artifacts: {root}")
    paths = {
        "root": root,
        "models": root / "models",
        "oof": root / "oof",
        "metrics": root / "metrics",
        "trial_metrics": root / "metrics" / "trials",
        "studies": root / "studies",
        "submissions": root / "submissions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def trial_allocation(trials: int) -> dict[int, int]:
    if trials <= 0:
        raise ValueError("trials must be positive.")
    if trials % len(HPO_K_VALUES):
        raise ValueError(f"trials must divide evenly across k values {HPO_K_VALUES}.")
    return {k: trials // len(HPO_K_VALUES) for k in HPO_K_VALUES}


def largest_hpo_params() -> TabMParams:
    return TabMParams(
        variant="tabm_piecewise",
        k=max(HPO_K_VALUES),
        d_block=512,
        n_blocks=4,
        dropout=0.3,
        learning_rate=0.005,
        weight_decay=0.1,
        batch_size=HPO_BATCH_SIZE,
        piecewise_bins=128,
        piecewise_embedding_dim=32,
    )


def run_tabm_tuning(config: TabMTuningConfig) -> dict[str, Any]:
    _validate_config(config)
    allocation = trial_allocation(config.trials)
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
    task_type = config.task_type.upper()
    if task_type == "GPU":
        gpu_status = ensure_tabm_gpu_ready()
        memory_status = ensure_tabm_cuda_memory_ready(
            prepared.X,
            prepared.categorical_features,
            largest_hpo_params(),
        )
    else:
        gpu_status = "CPU explicitly selected"
        memory_status = "CUDA memory preflight skipped because CPU was explicitly selected"
    paths = tabm_tuning_output_paths(config.project_root, config.run_name, resume=config.resume)
    _save_run_manifest(paths, config, source, gpu_status, memory_status, allocation)
    cv = StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    ctr_screen = source.oof_by_seed[SCREEN_SEED].copy()
    control = pd.Series(_control_row(ctr_screen, train))
    progress = tqdm(
        total=config.trials * config.n_splits + 4 * config.n_splits,
        desc="TabM HPO",
        unit="fold",
        disable=not config.show_progress,
    )

    def fold_progress(label: str):
        def update(event: str, fold: int) -> None:
            progress.set_postfix_str(f"{label}, fold {fold + 1}/{config.n_splits}")
            if event == "complete":
                progress.update(1)

        return update

    try:
        for k, target_trials in allocation.items():
            study = _open_study(paths, config, k)
            completed = _finished_trial_count(study)
            if completed > target_trials:
                raise ValueError(
                    f"Study k={k} already has {completed} finished trials, exceeding {target_trials}."
                )
            while completed < target_trials:
                progress.set_description(f"Tune k={k}, trial {completed + 1}/{target_trials}")
                study.optimize(
                    lambda trial: _run_tuning_trial(
                        trial,
                        paths,
                        config,
                        k,
                        prepared,
                        train,
                        cv,
                        ctr_screen,
                        control,
                        fold_progress(f"tune k={k}, trial {trial.number}"),
                    ),
                    n_trials=1,
                )
                _save_sampler(_sampler_path(paths, k), study.sampler)
                _save_study_summary(paths, k, study)
                _write_trial_table(paths)
                completed = _finished_trial_count(study)

        candidates = _read_trial_candidates(paths)
        _write_trial_table(paths, candidates)
        selected = _select_candidate(candidates)
        _save_json(
            paths["metrics"] / "tabm_hpo_selection.json",
            {
                "candidate_count": int(len(candidates)),
                "eligible_candidate_count": int(candidates["screen_eligible"].sum())
                if not candidates.empty
                else 0,
                "selected_candidate": None if selected is None else selected,
            },
        )
        if selected is None:
            decision = {
                "promoted": False,
                "reason": "No TabM HPO candidate cleared the predeclared CTR screen guardrails.",
                "screen_eligible": False,
            }
            _save_json(paths["metrics"] / "tabm_hpo_promotion_decision.json", decision)
            _save_final_config(
                paths,
                config,
                source,
                None,
                None,
                {},
                None,
                gpu_status,
                memory_status,
                decision,
                None,
            )
            return {
                "selected_experiment": None,
                "promoted": False,
                "submission_path": None,
                "promotion_decision": decision,
            }

        params = TabMParams(**selected["params"])
        selected_name = str(selected["experiment_name"])
        selected_weight = float(selected["tabm_weight"])
        selected_seed_runs: dict[int, dict[str, Any]] = {}
        for seed in ENSEMBLE_SEEDS:
            progress.set_description(f"Final k={params.k}, seed {seed}")
            selected_seed_runs[seed] = _fit_final_configuration(
                params,
                seed,
                prepared,
                cv,
                task_type,
                paths["models"],
                selected_name,
                fold_progress(f"final seed {seed}"),
                compute_feature_importance=seed == ENSEMBLE_SEEDS[-1],
            )
            _save_seed_artifact(paths, train, test, selected_name, seed, selected_seed_runs[seed])
            _release_models(selected_seed_runs[seed])
        _save_seed_fold_metrics(
            paths,
            selected_name,
            selected_seed_runs,
            "tabm_hpo_seed_fold_metrics.csv",
        )
        selected_ensemble = _combine_seed_runs(selected_seed_runs)
        ctr_ensemble = _mean_seed_oof(source.oof_by_seed, ENSEMBLE_SEEDS)
        ensemble_candidate = _oof_frame(
            ctr_ensemble,
            _blend_probabilities(
                ctr_ensemble["fraud_probability_raw"],
                selected_ensemble["oof_pred"],
                selected_weight,
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

        progress.set_description(f"Fresh confirmation seed {CONFIRMATION_SEED}")
        fresh_run = _fit_final_configuration(
            params,
            CONFIRMATION_SEED,
            prepared,
            cv,
            task_type,
            paths["models"],
            selected_name,
            fold_progress(f"fresh seed {CONFIRMATION_SEED}"),
        )
        _save_seed_artifact(paths, train, test, selected_name, CONFIRMATION_SEED, fresh_run)
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
        grouped_run = _fit_final_configuration(
            params,
            SCREEN_SEED,
            prepared,
            StratifiedGroupKFold(
                n_splits=config.n_splits,
                shuffle=True,
                random_state=config.random_state,
            ),
            task_type,
            None,
            selected_name,
            fold_progress("grouped robustness"),
            groups=feature_signature_groups(prepared.X),
            predict_test=False,
        )
        _save_grouped_artifacts(paths, train, grouped_run, config.n_bootstrap)
        _release_models(grouped_run)
        calibration = _calibrate_oof(selected_ensemble["prepared"].y, ensemble_candidate)
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
                "tabm_weight": selected_weight,
                "selected_trial": int(selected["trial_number"]),
                "selected_k": int(selected["k"]),
                "screen": _screen_decision(pd.Series(selected), control),
            }
        )
        _save_json(paths["metrics"] / "tabm_hpo_promotion_decision.json", decision)
        all_seed_runs = {**selected_seed_runs, CONFIRMATION_SEED: fresh_run}
        _save_final_config(
            paths,
            config,
            source,
            params,
            selected,
            all_seed_runs,
            grouped_run,
            gpu_status,
            memory_status,
            decision,
            calibration,
        )
        if not decision["promoted"]:
            return {
                "selected_experiment": selected_name,
                "tabm_weight": selected_weight,
                "promoted": False,
                "submission_path": None,
                "promotion_decision": decision,
            }

        tabm_test = np.mean(
            np.vstack(
                [np.asarray(all_seed_runs[seed]["test_pred"], dtype=float) for seed in (*ENSEMBLE_SEEDS, CONFIRMATION_SEED)]
            ),
            axis=0,
        )
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
        pd.DataFrame(
            {ID_COL: test[ID_COL], "fraud_probability_raw": raw_test_pred}
        ).to_csv(paths["oof"] / f"{selected_name}_test_raw.csv", index=False)
        submission_path = paths["submissions"] / f"{selected_name}_{calibration['method']}_submission.csv"
        make_submission(test[ID_COL], final_test_pred, submission_path)
        raw_submission_path = paths["submissions"] / f"{selected_name}_raw_submission.csv"
        if calibration["method"] != "raw":
            make_submission(test[ID_COL], raw_test_pred, raw_submission_path)
        else:
            raw_submission_path = submission_path
        return {
            "selected_experiment": selected_name,
            "tabm_weight": selected_weight,
            "promoted": True,
            "calibration_method": calibration["method"],
            "submission_path": submission_path,
            "raw_submission_path": raw_submission_path,
            "promotion_decision": decision,
        }
    finally:
        progress.close()


def _validate_config(config: TabMTuningConfig) -> None:
    if config.task_type.upper() not in {"CPU", "GPU"}:
        raise ValueError("task_type must be CPU or GPU.")
    if config.n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if config.n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if config.run_name == config.incumbent_run_name:
        raise ValueError("run_name must differ from incumbent_run_name.")


def _open_study(paths: dict[str, Path], config: TabMTuningConfig, k: int) -> optuna.Study:
    database_path = _study_database_path(paths, k)
    sampler_path = _sampler_path(paths, k)
    if database_path.exists() and config.resume and sampler_path.exists():
        with sampler_path.open("rb") as file:
            sampler = pickle.load(file)
    else:
        sampler = optuna.samplers.TPESampler(seed=config.random_state + k)
    study = optuna.create_study(
        study_name=f"tabm_piecewise_k{k}",
        storage=f"sqlite:///{database_path.resolve()}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=config.resume,
    )
    _save_sampler(sampler_path, study.sampler)
    return study


def _run_tuning_trial(
    trial: optuna.Trial,
    paths: dict[str, Path],
    config: TabMTuningConfig,
    k: int,
    prepared,
    train: pd.DataFrame,
    cv: StratifiedKFold,
    ctr_oof: pd.DataFrame,
    control: pd.Series,
    progress_callback,
) -> float:
    params = _suggest_params(trial, k)
    result = train_tabm_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv,
        params,
        seed=config.random_state + trial.number,
        task_type=config.task_type.upper(),
        model_dir=None,
        model_prefix=f"tuning_k{k}_trial_{trial.number}",
        progress_callback=progress_callback,
        predict_test=False,
        compute_feature_importance=False,
    )
    result["prepared"] = prepared
    result["notes"] = "Piecewise TabM HPO trial."
    try:
        tabm_oof = _oof_frame_from_run(train, result, config.random_state + trial.number)
        validate_paired_oof(tabm_oof, ctr_oof)
        candidates = _trial_candidate_rows(
            params,
            k,
            trial.number,
            result,
            tabm_oof,
            ctr_oof,
            train,
            control,
        )
        ranked = _rank_candidates(pd.DataFrame(candidates))
        best = ranked.iloc[0].to_dict()
        trial.set_user_attr("selected_candidate", best["experiment_name"])
        trial.set_user_attr("eligible_candidate_count", int(sum(row["screen_eligible"] for row in candidates)))
        _save_json(
            _trial_path(paths, k, trial.number),
            {
                "k": k,
                "trial_number": trial.number,
                "params": params.as_dict(),
                "candidates": candidates,
            },
        )
        return float(best["fraud_caught_at_5pct"])
    finally:
        _release_models(result)
        gc.collect()


def _suggest_params(trial: optuna.Trial, k: int) -> TabMParams:
    weight_decay_mode = trial.suggest_categorical("weight_decay_mode", ["zero", "log_uniform"])
    weight_decay = (
        0.0
        if weight_decay_mode == "zero"
        else trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    )
    return TabMParams(
        variant="tabm_piecewise",
        k=k,
        d_block=trial.suggest_int("d_block", 128, 512, step=32),
        n_blocks=trial.suggest_int("n_blocks", 2, 4),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        weight_decay=weight_decay,
        batch_size=HPO_BATCH_SIZE,
        piecewise_bins=trial.suggest_int("piecewise_bins", 8, 128),
        piecewise_embedding_dim=trial.suggest_int("piecewise_embedding_dim", 8, 32),
    )


def _trial_candidate_rows(
    params: TabMParams,
    k: int,
    trial_number: int,
    result: dict[str, Any],
    tabm_oof: pd.DataFrame,
    ctr_oof: pd.DataFrame,
    train: pd.DataFrame,
    control: pd.Series,
) -> list[dict[str, Any]]:
    rows = []
    for weight in (*HPO_BLEND_WEIGHTS, 1.0):
        probabilities = (
            tabm_oof["fraud_probability_raw"].to_numpy(dtype=float)
            if np.isclose(weight, 1.0)
            else _blend_probabilities(
                ctr_oof["fraud_probability_raw"],
                tabm_oof["fraud_probability_raw"],
                weight,
            )
        )
        metrics = evaluate_probabilities(ctr_oof[TARGET], probabilities, AUDIT_FRACTIONS)
        row = {
            "experiment_name": _candidate_name(k, trial_number, weight),
            "trial_number": int(trial_number),
            "k": int(k),
            "tabm_variant": "tabm_piecewise",
            "tabm_weight": float(weight),
            "candidate_type": "raw" if np.isclose(weight, 1.0) else "fixed_blend",
            "params": params.as_dict(),
            "mean_best_epoch": float(result["fold_metrics"]["best_epoch"].mean()),
            "fold_normalized_recall_5_std": _fold_normalized_recall_std(
                ctr_oof[TARGET], probabilities, ctr_oof["fold"]
            ),
            "fairness_audit_rate_gap_5": _fairness_gap(train, ctr_oof[TARGET], probabilities),
            **metrics,
        }
        guardrails = _screen_decision(pd.Series(row), control)
        row.update(
            {
                "screen_eligible": bool(guardrails["eligible"]),
                "normalized_recall_noninferior": bool(guardrails["normalized_recall_noninferior"]),
                "average_precision_noninferior": bool(guardrails["average_precision_noninferior"]),
                "brier_noninferior": bool(guardrails["brier_noninferior"]),
            }
        )
        rows.append(row)
    return rows


def _rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    return candidates.sort_values(
        [
            "fraud_caught_at_5pct",
            "average_precision",
            "brier_score",
            "fold_normalized_recall_5_std",
            "fairness_audit_rate_gap_5",
            "tabm_weight",
            "trial_number",
        ],
        ascending=[False, False, True, True, True, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _select_candidate(candidates: pd.DataFrame) -> dict[str, Any] | None:
    if candidates.empty:
        return None
    eligible = candidates.loc[candidates["screen_eligible"]].copy()
    if eligible.empty:
        return None
    selected = _rank_candidates(eligible).iloc[0].to_dict()
    params = selected.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("Selected TabM HPO candidate has an invalid parameter payload.")
    return selected


def _fit_final_configuration(
    params: TabMParams,
    seed: int,
    prepared,
    cv,
    task_type: str,
    model_dir: Path | None,
    selected_name: str,
    progress_callback,
    *,
    groups: np.ndarray | None = None,
    predict_test: bool = True,
    compute_feature_importance: bool = False,
) -> dict[str, Any]:
    result = train_tabm_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv,
        params,
        seed=seed,
        task_type=task_type,
        model_dir=model_dir,
        model_prefix=f"{selected_name}_seed_{seed}",
        progress_callback=progress_callback,
        groups=groups,
        predict_test=predict_test,
        compute_feature_importance=compute_feature_importance,
    )
    result.update(
        {
            "prepared": prepared,
            "notes": "Selected piecewise TabM HPO configuration.",
            "oof_metrics": evaluate_probabilities(prepared.y, result["oof_pred"], AUDIT_FRACTIONS),
        }
    )
    return result


def _finished_trial_count(study: optuna.Study) -> int:
    return sum(trial.state.is_finished() for trial in study.trials)


def _study_database_path(paths: dict[str, Path], k: int) -> Path:
    return paths["studies"] / f"tabm_piecewise_k{k}.sqlite3"


def _sampler_path(paths: dict[str, Path], k: int) -> Path:
    return paths["studies"] / f"tabm_piecewise_k{k}_sampler.pkl"


def _trial_path(paths: dict[str, Path], k: int, trial_number: int) -> Path:
    return paths["trial_metrics"] / f"tabm_piecewise_k{k}_trial_{trial_number:04d}.json"


def _candidate_name(k: int, trial_number: int, weight: float) -> str:
    prefix = f"tabm_piecewise_hpo_k{k}_t{trial_number:02d}"
    if np.isclose(weight, 1.0):
        return f"{prefix}_raw"
    return f"{prefix}_ctr_blend_w{round(weight * 100):02d}"


def _save_sampler(path: Path, sampler: optuna.samplers.BaseSampler) -> None:
    with path.open("wb") as file:
        pickle.dump(sampler, file)


def _save_study_summary(paths: dict[str, Path], k: int, study: optuna.Study) -> None:
    rows = [
        {
            "trial_number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": json.dumps(trial.params, sort_keys=True),
            "selected_candidate": trial.user_attrs.get("selected_candidate"),
            "eligible_candidate_count": trial.user_attrs.get("eligible_candidate_count"),
        }
        for trial in study.trials
    ]
    pd.DataFrame(rows).to_csv(paths["metrics"] / f"tabm_hpo_study_k{k}.csv", index=False)


def _read_trial_candidates(paths: dict[str, Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(paths["trial_metrics"].glob("tabm_piecewise_k*_trial_*.json")):
        with path.open() as file:
            payload = json.load(file)
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"TabM HPO trial artifact has invalid candidates: {path}")
        rows.extend(candidates)
    return pd.DataFrame(rows)


def _write_trial_table(paths: dict[str, Path], candidates: pd.DataFrame | None = None) -> None:
    frame = _read_trial_candidates(paths) if candidates is None else candidates.copy()
    if not frame.empty:
        frame["params"] = frame["params"].map(lambda values: json.dumps(values, sort_keys=True))
        frame = frame.sort_values(["k", "trial_number", "tabm_weight"], kind="stable")
    frame.to_csv(paths["metrics"] / "tabm_hpo_trials.csv", index=False)


def _save_run_manifest(
    paths: dict[str, Path],
    config: TabMTuningConfig,
    source: CTROOFSource,
    gpu_status: str,
    memory_status: str,
    allocation: dict[int, int],
) -> None:
    path = paths["metrics"] / "tabm_hpo_run_manifest.json"
    payload = {
        "run_name": config.run_name,
        "incumbent_run_name": config.incumbent_run_name,
        "task_type": config.task_type.upper(),
        "trials": config.trials,
        "trial_allocation": {str(k): count for k, count in allocation.items()},
        "k_values": list(HPO_K_VALUES),
        "blend_weights": list(HPO_BLEND_WEIGHTS),
        "batch_size": HPO_BATCH_SIZE,
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": config.n_splits,
            "shuffle": True,
            "random_state": config.random_state,
        },
        "ctr_source": {
            "run_name": config.incumbent_run_name,
            "model": source.config["model"],
            "profile": source.config["profile"],
            "experiment": source.config["experiment"],
        },
        "gpu_status": gpu_status,
        "cuda_memory_preflight": memory_status,
    }
    if path.exists() and config.resume:
        with path.open() as file:
            existing = json.load(file)
        if existing.get("run_name") != payload["run_name"]:
            raise ValueError("TabM HPO resume manifest does not match the requested run.")
    _save_json(path, payload)


def _save_final_config(
    paths: dict[str, Path],
    config: TabMTuningConfig,
    source: CTROOFSource,
    params: TabMParams | None,
    selected: dict[str, Any] | None,
    seed_runs: dict[int, dict[str, Any]],
    grouped_run: dict[str, Any] | None,
    gpu_status: str,
    memory_status: str,
    decision: dict[str, Any],
    calibration: dict[str, Any] | None,
) -> None:
    payload = {
        "model": "TabM",
        "profile": "tabm_hpo",
        "run_name": config.run_name,
        "incumbent_run_name": config.incumbent_run_name,
        "selected_experiment": None if selected is None else selected["experiment_name"],
        "selected_trial": None if selected is None else selected["trial_number"],
        "tabm_weight": None if selected is None else selected["tabm_weight"],
        "params": None if params is None else params.as_dict(),
        "features": None if params is None else list(next(iter(seed_runs.values()))["prepared"].X.columns),
        "categorical_features": (
            None
            if params is None
            else next(iter(seed_runs.values()))["prepared"].categorical_features
        ),
        "fold_model_features": {
            str(seed): run["fold_model_features"] for seed, run in seed_runs.items()
        },
        "model_artifacts": {
            str(seed): [
                f"{selected['experiment_name']}_seed_{seed}_fold_{fold}.pt"
                for fold in range(config.n_splits)
            ]
            for seed in seed_runs
        }
        if selected is not None
        else {},
        "preprocessor_artifacts": {
            str(seed): [
                f"{selected['experiment_name']}_seed_{seed}_preprocessor_fold_{fold}.joblib"
                for fold in range(config.n_splits)
            ]
            for seed in seed_runs
        }
        if selected is not None
        else {},
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
        "study_artifacts": {
            str(k): {
                "database": _study_database_path(paths, k).name,
                "sampler": _sampler_path(paths, k).name,
            }
            for k in HPO_K_VALUES
        },
        "screen_seed": SCREEN_SEED,
        "ensemble_seeds": [*ENSEMBLE_SEEDS, CONFIRMATION_SEED],
        "fresh_confirmation_seed": CONFIRMATION_SEED,
        "calibration": calibration["method"] if calibration else "not_run",
        "promotion_decision": decision,
        "gpu_status": gpu_status,
        "cuda_memory_preflight": memory_status,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "optuna": optuna.__version__,
        },
    }
    _save_json(paths["models"] / "tabm_hpo_final_config.json", payload)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and confirm the piecewise TabM challenger against the saved CTR incumbent."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--incumbent-run-name", default="ctr-v1")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default=os.environ.get("PRS_ITS_TABM_TASK_TYPE", "GPU").upper(),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_tabm_tuning(
        TabMTuningConfig(
            project_root=find_project_root(args.project_root),
            run_name=args.run_name,
            incumbent_run_name=args.incumbent_run_name,
            task_type=args.task_type,
            trials=args.trials,
            resume=args.resume,
            show_progress=not args.quiet,
        )
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
