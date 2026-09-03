from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
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
from prs_its.foundation_modeling import (
    DEFAULT_FOUNDATION_ESTIMATORS,
    DEFAULT_FOUNDATION_PREDICTION_CHUNK_SIZE,
    DEFAULT_MIN_FREE_VRAM_GIB,
    FOUNDATION_MODELS,
    FoundationParams,
    prepare_foundation_features,
    run_foundation_preflight,
    train_foundation_cv,
)
from prs_its.metrics import (
    bootstrap_audit_intervals,
    evaluate_probabilities,
    paired_fairness_comparison,
    paired_oof_comparison,
    validate_paired_oof,
)
from prs_its.modeling import (
    ID_COL,
    TARGET,
    PreparedFeatures,
    make_feature_spec,
    prepare_catboost_features,
    validate_train_test_schema,
)
from prs_its.submission import make_submission
from prs_its.tabm_modeling import TabMParams, prepare_tabm_features, train_tabm_cv
from prs_its.training import _fairness_gap, find_project_root, load_competition_data


AUDIT_FRACTIONS = (0.03, 0.05, 0.07)
FOUNDATION_SCREEN_SEED = 42
FOUNDATION_CONFIRMATION_SEEDS = (42, 2026)
FOUNDATION_BLEND_WEIGHTS = (0.10, 0.25, 0.50, 0.75, 1.00)
AVERAGE_PRECISION_TOLERANCE = 0.005


@dataclass(frozen=True)
class FoundationTrainingConfig:
    project_root: Path
    run_name: str
    incumbent_run_name: str = "ctr-v1"
    tabm_run_name: str = "tabm-hpo-v1"
    model: str = "tabicl-v2"
    task_type: str = "GPU"
    devices: str = "0"
    n_splits: int = 3
    random_state: int = 42
    max_runtime_minutes: float = 360.0
    n_bootstrap: int = 1000
    show_progress: bool = True
    n_estimators: int = DEFAULT_FOUNDATION_ESTIMATORS
    prediction_chunk_size: int = DEFAULT_FOUNDATION_PREDICTION_CHUNK_SIZE
    min_free_vram_gib: float = DEFAULT_MIN_FREE_VRAM_GIB
    accept_tabpfn_terms: bool = False

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name is required.")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.run_name):
            raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
        if self.run_name in {self.incumbent_run_name, self.tabm_run_name}:
            raise ValueError("run_name must differ from incumbent and TabM source runs.")
        if self.model not in FOUNDATION_MODELS:
            raise ValueError(f"model must be one of {FOUNDATION_MODELS}.")
        if self.model == "tabpfn-3" and not self.accept_tabpfn_terms:
            raise ValueError(
                "TabPFN-3 requires --confirm-tabpfn-eligibility after reviewing its model terms."
            )
        if self.task_type.upper() not in {"CPU", "GPU"}:
            raise ValueError("task_type must be CPU or GPU.")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if self.max_runtime_minutes <= 0:
            raise ValueError("max_runtime_minutes must be positive.")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive.")
        if self.n_estimators < 1 or self.prediction_chunk_size < 1:
            raise ValueError("Foundation estimator and prediction chunk sizes must be positive.")
        if self.min_free_vram_gib <= 0:
            raise ValueError("min_free_vram_gib must be positive.")


@dataclass(frozen=True)
class SourceRecipes:
    ctr_config: dict[str, Any]
    ctr_prepared: PreparedFeatures
    tabm_params: TabMParams
    tabm_selection: dict[str, Any]


def foundation_output_paths(project_root: Path, run_name: str) -> dict[str, Path]:
    if not run_name:
        raise ValueError("run_name is required.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
        raise ValueError("run_name may contain only letters, numbers, underscores, and hyphens.")
    root = project_root / "outputs" / "runs" / run_name
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Foundation output run already contains artifacts: {root}")
    paths = {
        "root": root,
        "oof": root / "oof",
        "metrics": root / "metrics",
        "submissions": root / "submissions",
        "cache": root / "cache",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_foundation_training(config: FoundationTrainingConfig) -> dict[str, Any]:
    started = time.monotonic()
    task_type = config.task_type.upper()
    train, test = load_competition_data(config.project_root)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    spec = make_feature_spec(train, test)
    sources = _load_source_recipes(config, train, test, spec)
    foundation_prepared = prepare_foundation_features(train, test, spec)
    cv = _make_cv(config)
    foundation_params = FoundationParams(
        model=config.model,
        n_estimators=config.n_estimators,
        prediction_chunk_size=config.prediction_chunk_size,
        min_free_vram_gib=config.min_free_vram_gib,
    )
    preflight = run_foundation_preflight(
        foundation_prepared.X,
        foundation_prepared.y,
        foundation_prepared.categorical_features,
        cv,
        foundation_params,
        seed=FOUNDATION_SCREEN_SEED,
        task_type=task_type,
    )
    paths = foundation_output_paths(config.project_root, config.run_name)
    _save_json(paths["metrics"] / "foundation_preflight.json", preflight)
    if config.max_runtime_minutes * 60 <= time.monotonic() - started:
        raise RuntimeError("Foundation runtime budget was exhausted during preflight.")

    if config.show_progress:
        progress = tqdm(total=5, desc="Foundation challenger", unit="stage")
    else:
        progress = None

    try:
        ctr_runs: dict[int, dict[str, Any]] = {}
        tabm_runs: dict[int, dict[str, Any]] = {}
        for seed in FOUNDATION_CONFIRMATION_SEEDS:
            _check_budget(started, config.max_runtime_minutes, f"before CTR seed {seed}")
            ctr_runs[seed] = _train_ctr_seed(
                train,
                test,
                sources.ctr_prepared,
                sources.ctr_config,
                cv,
                seed,
                task_type,
                config.devices,
            )
            _save_seed_artifact(paths, train, test, "ctr", seed, ctr_runs[seed])
            _release_result(ctr_runs[seed])
            if progress is not None:
                progress.update(1)

            _check_budget(started, config.max_runtime_minutes, f"before TabM seed {seed}")
            tabm_runs[seed] = _train_tabm_seed(
                train,
                test,
                sources,
                cv,
                seed,
                task_type,
            )
            _save_seed_artifact(paths, train, test, "tabm", seed, tabm_runs[seed])
            _release_result(tabm_runs[seed])
            if progress is not None:
                progress.update(1)

        _check_budget(started, config.max_runtime_minutes, "before foundation model")
        foundation_result = train_foundation_cv(
            foundation_prepared.X,
            foundation_prepared.y,
            foundation_prepared.X_test,
            foundation_prepared.categorical_features,
            cv,
            foundation_params,
            seed=FOUNDATION_SCREEN_SEED,
            task_type=task_type,
            cache_dir=paths["cache"],
            progress_callback=_foundation_progress(progress),
            predict_test=True,
        )
        _save_seed_artifact(paths, train, test, "foundation", FOUNDATION_SCREEN_SEED, foundation_result)
        _release_result(foundation_result)
        if progress is not None:
            progress.update(1)

        ctr = _combine_seed_runs(ctr_runs)
        tabm = _combine_seed_runs(tabm_runs)
        if not np.array_equal(ctr["fold_id"], tabm["fold_id"]):
            raise RuntimeError("CTR and TabM must use identical validation folds.")
        base = _oof_frame(train, ctr["fold_id"], _blend(ctr["oof_pred"], tabm["oof_pred"], 0.5))
        foundation_oof = _load_saved_oof(paths, "foundation", FOUNDATION_SCREEN_SEED)
        foundation_pred = foundation_oof["fraud_probability_raw"].to_numpy(dtype=float)
        validate_paired_oof(foundation_oof, base)
        candidates, candidate_frames = _screen_candidates(
            train,
            base,
            foundation_pred,
            foundation_result=None,
        )
        candidates.to_csv(paths["metrics"] / "foundation_experiments.csv", index=False)
        for name, frame in candidate_frames.items():
            frame.to_csv(paths["oof"] / f"{name}_screen_oof.csv", index=False)
            _save_pairwise_artifacts(paths, name, frame, base, train, config.n_bootstrap)

        selected_row, screen_decision = _select_candidate(candidates)
        selected_name = str(selected_row["experiment_name"])
        selected_frame = candidate_frames[selected_name]
        _save_json(paths["metrics"] / "foundation_screen_decision.json", screen_decision)

        calibration = _calibrate(selected_frame)
        _save_calibration_artifacts(paths, selected_name, selected_frame, calibration)
        _save_final_artifacts(
            paths,
            train,
            test,
            ctr,
            tabm,
            foundation_result,
            base,
            selected_frame,
            selected_name,
            calibration,
            sources,
            preflight,
            config,
            started,
            screen_decision,
        )
        final_test = _selected_test_predictions(
            ctr["test_pred"], tabm["test_pred"], foundation_result["test_pred"], float(selected_row["foundation_weight"])
        )
        if calibration["method"] != "raw":
            final_test = calibrate_test_predictions(
                selected_frame["fraud_probability_raw"],
                selected_frame[TARGET],
                final_test,
                calibration["method"],
            )
        raw_test_path = paths["oof"] / f"{selected_name}_test_raw.csv"
        pd.DataFrame({ID_COL: test[ID_COL], "fraud_probability_raw": final_test}).to_csv(
            raw_test_path, index=False
        )
        submission_path = paths["submissions"] / f"{selected_name}_{calibration['method']}_unpromoted_submission.csv"
        make_submission(test[ID_COL], final_test, submission_path)
        decision = {
            **screen_decision,
            "selected_experiment": selected_name,
            "submission_status": "unpromoted",
            "submission_path": str(submission_path),
            "promoted": False,
            "confirmation_status": "not_run",
            "runtime_seconds": time.monotonic() - started,
            "model_artifacts": "not_saved_locally; retain on remote training server",
        }
        _save_json(paths["metrics"] / "foundation_promotion_decision.json", decision)
        _save_json(
            paths["metrics"] / "foundation_run_manifest.json",
            _run_manifest(config, sources, foundation_params, preflight, decision, started),
        )
        return {
            "selected_experiment": selected_name,
            "promoted": False,
            "submission_status": "unpromoted",
            "submission_path": submission_path,
            "promotion_decision": decision,
        }
    finally:
        if progress is not None:
            progress.close()


def preflight_foundation_training(config: FoundationTrainingConfig) -> dict[str, Any]:
    train, test = load_competition_data(config.project_root)
    features = validate_train_test_schema(train, test)
    if ID_COL in features or TARGET in features:
        raise RuntimeError("claim_id and label must not be model features.")
    spec = make_feature_spec(train, test)
    _load_source_recipes(config, train, test, spec)
    prepared = prepare_foundation_features(train, test, spec)
    result = run_foundation_preflight(
        prepared.X,
        prepared.y,
        prepared.categorical_features,
        _make_cv(config),
        FoundationParams(
            model=config.model,
            n_estimators=config.n_estimators,
            prediction_chunk_size=config.prediction_chunk_size,
            min_free_vram_gib=config.min_free_vram_gib,
        ),
        seed=FOUNDATION_SCREEN_SEED,
        task_type=config.task_type,
    )
    paths = foundation_output_paths(config.project_root, config.run_name)
    result.update(
        {
            "source_runs_validated": {
                "ctr": config.incumbent_run_name,
                "tabm": config.tabm_run_name,
            },
            "fallback_model": "tabpfn-3; run separately after a TabICLv2 preflight failure",
        }
    )
    _save_json(paths["metrics"] / "foundation_preflight.json", result)
    return result


def _load_source_recipes(
    config: FoundationTrainingConfig,
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: Any,
) -> SourceRecipes:
    ctr_config = _load_json(
        config.project_root
        / "outputs"
        / "runs"
        / config.incumbent_run_name
        / "models"
        / "catboost_final_config.json"
    )
    _validate_ctr_config(ctr_config)
    ctr_prepared = prepare_catboost_features(
        train,
        test,
        spec,
        add_count_features=True,
        add_interaction_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features={"dati2_typeppk": ("dati2", "typeppk")},
    )
    if list(ctr_prepared.X.columns) != list(ctr_config["features"]):
        raise ValueError("Current data no longer matches the saved CTR feature schema.")
    if list(ctr_prepared.categorical_features) != list(ctr_config["categorical_features"]):
        raise ValueError("Current data no longer matches the saved CTR categorical schema.")
    selection = _load_json(
        config.project_root
        / "outputs"
        / "runs"
        / config.tabm_run_name
        / "metrics"
        / "tabm_hpo_selection.json"
    )
    selected = selection.get("selected_candidate")
    if not isinstance(selected, dict) or not isinstance(selected.get("params"), dict):
        raise ValueError("TabM selection artifact does not contain a selected parameter set.")
    params = TabMParams(**selected["params"])
    if params.variant != "tabm_piecewise":
        raise ValueError("Foundation challenger requires the selected tabm_piecewise recipe.")
    if float(selected.get("tabm_weight", 0.5)) != 0.5:
        raise ValueError("Foundation challenger requires the selected 50% TabM blend recipe.")
    return SourceRecipes(ctr_config, ctr_prepared, params, selection)


def _validate_ctr_config(payload: dict[str, Any]) -> None:
    required = {"model", "profile", "features", "categorical_features", "experiment", "params"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"CTR source configuration is missing required fields: {missing}")
    if payload["model"] != "CatBoostClassifier" or payload["profile"] != "ctr":
        raise ValueError("Foundation challenger requires a saved CTR-profile CatBoost source run.")
    if payload["experiment"].get("name") != "ctr_dati2_typeppk":
        raise ValueError("Foundation challenger requires the ctr_dati2_typeppk CTR source recipe.")


def _train_ctr_seed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prepared: PreparedFeatures,
    source_config: dict[str, Any],
    cv: StratifiedKFold,
    seed: int,
    task_type: str,
    devices: str,
) -> dict[str, Any]:
    from prs_its.modeling import train_catboost_cv

    params = dict(source_config["params"])
    params["random_seed"] = seed
    params["task_type"] = task_type
    params["devices"] = devices
    result = train_catboost_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv=cv,
        params=params,
        task_type=task_type,
        devices=devices,
        model_dir=None,
        model_prefix=f"foundation_ctr_seed_{seed}",
        verbose=False,
        predict_test=True,
    )
    result["prepared"] = prepared
    result["params"] = params
    result["seed"] = seed
    return result


def _train_tabm_seed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sources: SourceRecipes,
    cv: StratifiedKFold,
    seed: int,
    task_type: str,
) -> dict[str, Any]:
    prepared = prepare_tabm_features(train, test, make_feature_spec(train, test))
    result = train_tabm_cv(
        prepared.X,
        prepared.y,
        prepared.X_test,
        prepared.categorical_features,
        cv=cv,
        params=sources.tabm_params,
        seed=seed,
        task_type=task_type,
        model_dir=None,
        model_prefix=f"foundation_tabm_seed_{seed}",
        predict_test=True,
    )
    result["prepared"] = prepared
    result["params"] = sources.tabm_params.as_dict()
    result["seed"] = seed
    return result


def _make_cv(config: FoundationTrainingConfig) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.n_splits,
        shuffle=True,
        random_state=config.random_state,
    )


def _foundation_progress(progress: Any) -> Callable[[str, int], None] | None:
    if progress is None:
        return None

    def update(event: str, fold: int) -> None:
        progress.set_postfix_str(f"foundation fold {fold + 1}")

    return update


def _combine_seed_runs(runs: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("At least one seed run is required.")
    first = next(iter(runs.values()))
    fold_id = np.asarray(first["fold_id"], dtype=int)
    for seed, result in runs.items():
        if not np.array_equal(fold_id, np.asarray(result["fold_id"], dtype=int)):
            raise RuntimeError(f"Seed {seed} did not use the same validation folds.")
    return {
        "oof_pred": np.mean(np.vstack([result["oof_pred"] for result in runs.values()]), axis=0),
        "test_pred": np.mean(np.vstack([result["test_pred"] for result in runs.values()]), axis=0),
        "test_fold_predictions": np.vstack(
            [result["test_fold_predictions"] for result in runs.values()]
        ),
        "fold_id": fold_id,
        "fold_metrics": pd.concat(
            [result["fold_metrics"].assign(random_seed=seed) for seed, result in runs.items()],
            ignore_index=True,
        ),
        "params": first.get("params", {}),
        "prepared": first["prepared"],
        "seed_order": list(runs),
    }


def _oof_frame(
    train: pd.DataFrame,
    fold_id: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COL: train[ID_COL].to_numpy(),
            TARGET: train[TARGET].astype(int).to_numpy(),
            "fold": np.asarray(fold_id, dtype=int),
            "fraud_probability_raw": _validated_probabilities(probabilities, "OOF probabilities"),
        }
    )


def _screen_candidates(
    train: pd.DataFrame,
    base: pd.DataFrame,
    foundation_probabilities: np.ndarray,
    foundation_result: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    del foundation_result
    rows = [_metric_row("ctr_tabm_base_w50", "base_control", 0.0, base, train)]
    frames = {"ctr_tabm_base_w50": base.copy()}
    for order, weight in enumerate(FOUNDATION_BLEND_WEIGHTS, start=1):
        name = _foundation_experiment_name(weight)
        candidate = _oof_frame(
            train,
            base["fold"].to_numpy(dtype=int),
            _blend(base["fraud_probability_raw"].to_numpy(dtype=float), foundation_probabilities, weight),
        )
        frames[name] = candidate
        row = _metric_row(name, "foundation_candidate", weight, candidate, train)
        row["candidate_order"] = order
        rows.append(row)
    control = rows[0]
    for row in rows[1:]:
        row.update(_screen_guardrails(row, control))
    rows[0].update(
        {
            "screen_eligible": True,
            "normalized_recall_noninferior": True,
            "average_precision_noninferior": True,
            "brier_noninferior": True,
        }
    )
    return pd.DataFrame(rows), frames


def _metric_row(
    name: str,
    stage: str,
    foundation_weight: float,
    frame: pd.DataFrame,
    train: pd.DataFrame,
) -> dict[str, Any]:
    probabilities = frame["fraud_probability_raw"].to_numpy(dtype=float)
    return {
        "experiment_name": name,
        "experiment_stage": stage,
        "foundation_weight": foundation_weight,
        "fold_normalized_recall_5_std": _fold_recall_std(
            frame[TARGET], probabilities, frame["fold"]
        ),
        "fairness_audit_rate_gap_5": _fairness_gap(train, frame[TARGET], probabilities),
        **evaluate_probabilities(frame[TARGET], probabilities, AUDIT_FRACTIONS),
    }


def _screen_guardrails(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, bool]:
    normalized = candidate["normalized_recall_at_5pct"] >= control["normalized_recall_at_5pct"]
    ap = candidate["average_precision"] >= control["average_precision"] - AVERAGE_PRECISION_TOLERANCE
    brier = candidate["brier_score"] <= control["brier_score"]
    return {
        "screen_eligible": bool(normalized and ap and brier),
        "normalized_recall_noninferior": bool(normalized),
        "average_precision_noninferior": bool(ap),
        "brier_noninferior": bool(brier),
    }


def _select_candidate(candidates: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    control = candidates.loc[candidates["experiment_stage"].eq("base_control")]
    if len(control) != 1:
        raise RuntimeError("Foundation screen must contain exactly one base control.")
    candidate_rows = candidates.loc[candidates["experiment_stage"].eq("foundation_candidate")].copy()
    if candidate_rows.empty:
        raise RuntimeError("Foundation screen produced no candidates.")
    eligible = candidate_rows.loc[candidate_rows["screen_eligible"]].copy()
    fallback = eligible.empty
    ranked = eligible if not fallback else candidate_rows
    ranked = ranked.sort_values(
        [
            "fraud_caught_at_5pct",
            "average_precision",
            "brier_score",
            "fold_normalized_recall_5_std",
            "fairness_audit_rate_gap_5",
            "foundation_weight",
            "candidate_order",
        ],
        ascending=[False, False, True, True, True, True, True],
        na_position="last",
        kind="mergesort",
    )
    selected = ranked.iloc[0]
    decision = {
        "selected_experiment": str(selected["experiment_name"]),
        "selected_foundation_weight": float(selected["foundation_weight"]),
        "screen_eligible": bool(selected["screen_eligible"]),
        "screen_eligible_candidate_count": int(len(eligible)),
        "fallback_to_best_ineligible": fallback,
        "promotion_gate": "not_run_until_5-fold/fresh-seed/grouped confirmation",
        "base_control": str(control.iloc[0]["experiment_name"]),
    }
    return selected, decision


def _fold_recall_std(labels: pd.Series, probabilities: np.ndarray, folds: pd.Series) -> float:
    values = []
    labels_array = np.asarray(labels, dtype=int)
    fold_array = np.asarray(folds, dtype=int)
    for fold in np.unique(fold_array):
        mask = fold_array == fold
        values.append(
            evaluate_probabilities(labels_array[mask], probabilities[mask], AUDIT_FRACTIONS)[
                "normalized_recall_at_5pct"
            ]
        )
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _blend(left: np.ndarray, right: np.ndarray, right_weight: float) -> np.ndarray:
    if not 0 <= right_weight <= 1:
        raise ValueError("blend weight must be within [0, 1].")
    left_values = _validated_probabilities(left, "left probabilities")
    right_values = _validated_probabilities(right, "right probabilities")
    if len(left_values) != len(right_values):
        raise ValueError("Blended probability arrays must have matching lengths.")
    return (1 - right_weight) * left_values + right_weight * right_values


def _selected_test_predictions(
    ctr: np.ndarray,
    tabm: np.ndarray,
    foundation: np.ndarray,
    foundation_weight: float,
) -> np.ndarray:
    base = _blend(ctr, tabm, 0.5)
    return _blend(base, foundation, foundation_weight)


def _foundation_experiment_name(weight: float) -> str:
    if np.isclose(weight, 1.0):
        return "foundation_raw"
    return f"foundation_ctr_tabm_base_blend_w{round(weight * 100):02d}"


def _save_pairwise_artifacts(
    paths: dict[str, Path],
    name: str,
    candidate: pd.DataFrame,
    base: pd.DataFrame,
    train: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    comparison = paired_oof_comparison(candidate, base, n_bootstrap=n_bootstrap)
    comparison.to_csv(paths["metrics"] / f"{name}_vs_base_paired.csv", index=False)
    if {"jkpst", "umur"}.issubset(train.columns):
        fairness = paired_fairness_comparison(
            candidate[TARGET],
            candidate["fraud_probability_raw"],
            base["fraud_probability_raw"],
            gender_groups=train["jkpst"],
            age_group_values=age_groups(train["umur"]),
            n_bootstrap=n_bootstrap,
        )
        fairness["subgroup_rate_deltas"].to_csv(
            paths["metrics"] / f"{name}_vs_base_fairness_rates.csv", index=False
        )
        fairness["gap_intervals"].to_csv(
            paths["metrics"] / f"{name}_vs_base_fairness_gaps.csv", index=False
        )


def _calibrate(candidate: pd.DataFrame) -> dict[str, Any]:
    raw = candidate["fraud_probability_raw"].to_numpy(dtype=float)
    labels = candidate[TARGET].to_numpy(dtype=int)
    folds = candidate["fold"].to_numpy(dtype=int)
    raw_metrics = evaluate_probabilities(labels, raw, AUDIT_FRACTIONS)
    rows = [{"prediction_type": "raw", **raw_metrics}]
    cross_fitted: dict[str, np.ndarray] = {}
    eligible: list[tuple[str, np.ndarray, dict[str, Any]]] = []
    for method in ("sigmoid", "isotonic"):
        result = cross_fit_calibration(labels, raw, folds, method)
        calibrated = np.asarray(result["oof_pred"], dtype=float)
        metrics = evaluate_probabilities(labels, calibrated, AUDIT_FRACTIONS)
        cross_fitted[method] = calibrated
        rows.append({"prediction_type": method, **metrics})
        if should_select_calibration(raw_metrics, metrics):
            eligible.append((method, calibrated, metrics))
    if eligible:
        method, selected, metrics = min(eligible, key=lambda item: item[2]["brier_score"])
    else:
        method, selected, metrics = "raw", raw, raw_metrics
    return {
        "method": method,
        "raw_oof": raw,
        "selected_oof": selected,
        "selected_metrics": metrics,
        "cross_fitted_oof": cross_fitted,
        "rows": rows,
    }


def _save_calibration_artifacts(
    paths: dict[str, Path],
    selected_name: str,
    candidate: pd.DataFrame,
    calibration: dict[str, Any],
) -> None:
    frame = candidate.copy()
    frame["fraud_probability_final"] = calibration["selected_oof"]
    frame.to_csv(paths["oof"] / f"{selected_name}_oof.csv", index=False)
    for method, probabilities in calibration["cross_fitted_oof"].items():
        calibrated = candidate.copy()
        calibrated["fraud_probability_raw"] = probabilities
        calibrated.to_csv(paths["oof"] / f"{selected_name}_oof_calibrated_{method}.csv", index=False)
    pd.DataFrame(calibration["rows"]).to_csv(
        paths["metrics"] / f"{selected_name}_calibration_metrics.csv", index=False
    )
    calibration_curve_frame(candidate[TARGET], calibration["selected_oof"]).to_csv(
        paths["metrics"] / f"{selected_name}_calibration_curve.csv", index=False
    )
    prediction_distribution(candidate[TARGET], calibration["selected_oof"]).to_csv(
        paths["metrics"] / f"{selected_name}_prediction_distribution.csv", index=False
    )


def _save_final_artifacts(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    ctr: dict[str, Any],
    tabm: dict[str, Any],
    foundation: dict[str, Any],
    base: pd.DataFrame,
    selected: pd.DataFrame,
    selected_name: str,
    calibration: dict[str, Any],
    sources: SourceRecipes,
    preflight: dict[str, Any],
    config: FoundationTrainingConfig,
    started: float,
    screen_decision: dict[str, Any],
) -> None:
    pd.DataFrame(
        bootstrap_audit_intervals(
            selected[TARGET],
            selected["fraud_probability_raw"],
            audit_fractions=AUDIT_FRACTIONS,
            n_bootstrap=config.n_bootstrap,
        )
    ).to_csv(paths["metrics"] / f"{selected_name}_audit_bootstrap.csv", index=False)
    if {"jkpst", "umur"}.issubset(train.columns):
        fairness = pd.concat(
            [
                fairness_across_budgets(
                    train["jkpst"], selected[TARGET], calibration["selected_oof"], group_name="gender"
                ),
                fairness_across_budgets(
                    age_groups(train["umur"]), selected[TARGET], calibration["selected_oof"], group_name="age_group"
                ),
            ],
            ignore_index=True,
        )
        fairness.to_csv(paths["metrics"] / f"{selected_name}_fairness.csv", index=False)
    pd.DataFrame(
        {
            "component": ["ctr", "tabm", "foundation", "base"],
            "mean_prediction": [
                float(np.mean(ctr["oof_pred"])),
                float(np.mean(tabm["oof_pred"])),
                float(np.mean(foundation["oof_pred"])),
                float(np.mean(base["fraud_probability_raw"])),
            ],
        }
    ).to_csv(paths["metrics"] / "component_prediction_summary.csv", index=False)
    _save_json(
        paths["metrics"] / "foundation_final_config.json",
        {
            "model": config.model,
            "run_name": config.run_name,
            "features": list(prepare_foundation_features(train, test, make_feature_spec(train, test)).X.columns),
            "categorical_features": list(prepare_foundation_features(train, test, make_feature_spec(train, test)).categorical_features),
            "excluded_features": [ID_COL, TARGET],
            "foundation_params": preflight["params"],
            "base": {"ctr_weight": 0.5, "tabm_weight": 0.5},
            "blend_weights": list(FOUNDATION_BLEND_WEIGHTS),
            "screen_seed": FOUNDATION_SCREEN_SEED,
            "confirmation_seeds": list(FOUNDATION_CONFIRMATION_SEEDS),
            "cv": {
                "type": "StratifiedKFold",
                "n_splits": config.n_splits,
                "shuffle": True,
                "random_state": config.random_state,
            },
            "ctr_source": {
                "run_name": config.incumbent_run_name,
                "recipe": sources.ctr_config["experiment"],
                "params": sources.ctr_config["params"],
            },
            "tabm_source": {
                "run_name": config.tabm_run_name,
                "selection": sources.tabm_selection.get("selected_candidate", {}),
            },
            "screen_decision": screen_decision,
            "runtime_seconds": time.monotonic() - started,
            "model_artifacts": "not_saved_locally; retain on remote training server",
            "versions": _versions(),
        },
    )


def _save_seed_artifact(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    name: str,
    seed: int,
    result: dict[str, Any],
) -> None:
    frame = _oof_frame(train, result["fold_id"], result["oof_pred"])
    frame.to_csv(paths["oof"] / f"{name}_oof_seed_{seed}.csv", index=False)
    if result.get("test_fold_predictions") is not None:
        values = np.asarray(result["test_fold_predictions"], dtype=float)
        test_frame = pd.DataFrame({ID_COL: test[ID_COL].to_numpy()})
        for fold, predictions in enumerate(values):
            test_frame[f"fold_{fold}"] = predictions
        test_frame.to_csv(paths["oof"] / f"{name}_test_fold_predictions_seed_{seed}.csv", index=False)
    result["oof_pred"] = np.asarray(result["oof_pred"], dtype=float)


def _load_saved_oof(paths: dict[str, Path], name: str, seed: int) -> pd.DataFrame:
    path = paths["oof"] / f"{name}_oof_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing saved {name} OOF artifact: {path}")
    return pd.read_csv(path)


def _release_result(result: dict[str, Any]) -> None:
    models = result.get("models", [])
    for model in models:
        to_cpu = getattr(model, "to", None)
        if to_cpu is not None:
            to_cpu("cpu")
    models.clear()
    preprocessors = result.get("fold_preprocessors", [])
    preprocessors.clear()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _check_budget(started: float, minutes: float, label: str) -> None:
    if time.monotonic() - started >= minutes * 60:
        raise TimeoutError(f"Foundation runtime budget exhausted {label}.")


def _run_manifest(
    config: FoundationTrainingConfig,
    sources: SourceRecipes,
    foundation_params: FoundationParams,
    preflight: dict[str, Any],
    decision: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "model": config.model,
        "task_type": config.task_type.upper(),
        "devices": config.devices,
        "runtime_budget_minutes": config.max_runtime_minutes,
        "runtime_seconds": time.monotonic() - started,
        "foundation_params": foundation_params.as_dict(),
        "screen_seed": FOUNDATION_SCREEN_SEED,
        "confirmation_seeds": list(FOUNDATION_CONFIRMATION_SEEDS),
        "source_runs": {
            "ctr": config.incumbent_run_name,
            "tabm": config.tabm_run_name,
        },
        "tabm_params": sources.tabm_params.as_dict(),
        "preflight": preflight,
        "promotion_decision": decision,
        "model_artifacts": "remote-only",
        "versions": _versions(),
    }


def _versions() -> dict[str, str]:
    values = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for package in ("torch", "tabicl", "tabpfn", "scikit-learn"):
        try:
            values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            values[package] = "not-installed"
    return values


def _validated_probabilities(values: Any, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
        raise ValueError(f"{name} must contain finite probabilities within [0, 1].")
    return probabilities


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required source artifact: {path}")
    with path.open() as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w") as file:
        json.dump(payload, file, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fold-matched TabICLv2/TabPFN-3 foundation challenger."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--incumbent-run-name", default="ctr-v1")
    parser.add_argument("--tabm-run-name", default="tabm-hpo-v1")
    parser.add_argument("--model", choices=FOUNDATION_MODELS, default="tabicl-v2")
    parser.add_argument("--task-type", choices=["CPU", "GPU"], default="GPU")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-runtime-minutes", type=float, default=360.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_FOUNDATION_ESTIMATORS)
    parser.add_argument("--prediction-chunk-size", type=int, default=DEFAULT_FOUNDATION_PREDICTION_CHUNK_SIZE)
    parser.add_argument("--min-free-vram-gib", type=float, default=DEFAULT_MIN_FREE_VRAM_GIB)
    parser.add_argument("--confirm-tabpfn-eligibility", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(args.project_root)
    config = FoundationTrainingConfig(
        project_root=project_root,
        run_name=args.run_name,
        incumbent_run_name=args.incumbent_run_name,
        tabm_run_name=args.tabm_run_name,
        model=args.model,
        task_type=args.task_type,
        devices=args.devices,
        n_splits=args.n_splits,
        random_state=args.random_state,
        max_runtime_minutes=args.max_runtime_minutes,
        n_bootstrap=args.n_bootstrap,
        show_progress=not args.quiet,
        n_estimators=args.n_estimators,
        prediction_chunk_size=args.prediction_chunk_size,
        min_free_vram_gib=args.min_free_vram_gib,
        accept_tabpfn_terms=args.confirm_tabpfn_eligibility,
    )
    if args.preflight:
        result = preflight_foundation_training(config)
    else:
        result = run_foundation_training(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
