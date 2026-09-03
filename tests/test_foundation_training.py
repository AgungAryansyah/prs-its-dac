from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

import prs_its.foundation_training as foundation_training
from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec, prepare_catboost_features
from prs_its.tabm_modeling import TabMParams


def _data(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = 30
    frame = pd.DataFrame(
        {
            "claim_id": [f"CLM_{i:03d}" for i in range(rows)],
            "umur": np.arange(rows) % 70,
            "los": np.arange(rows) % 4,
            "dx2_a00_b99": np.arange(rows) % 3,
            "proc00_13": np.arange(rows) % 2,
        }
    )
    for index, column in enumerate(CATEGORICAL_CANDIDATES):
        frame[column] = [f"{index}-{i % 3}" for i in range(rows)]
    train = frame.copy()
    train["label"] = np.arange(rows) % 2
    test = frame.iloc[:8].copy()
    test["claim_id"] = [f"TST_{i:03d}" for i in range(len(test))]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    return train, test


def _sources(tmp_path: Path, train: pd.DataFrame, test: pd.DataFrame) -> None:
    spec = make_feature_spec(train, test)
    prepared = prepare_catboost_features(
        train,
        test,
        spec,
        add_count_features=True,
        add_interaction_features=True,
        add_los_features=True,
        add_count_bucket_features=True,
        additional_interaction_features={"dati2_typeppk": ("dati2", "typeppk")},
    )
    ctr_dir = tmp_path / "outputs" / "runs" / "ctr-v1" / "models"
    tabm_dir = tmp_path / "outputs" / "runs" / "tabm-hpo-v1" / "metrics"
    ctr_dir.mkdir(parents=True)
    tabm_dir.mkdir(parents=True)
    (ctr_dir / "catboost_final_config.json").write_text(
        json.dumps(
            {
                "model": "CatBoostClassifier",
                "profile": "ctr",
                "features": list(prepared.X.columns),
                "categorical_features": prepared.categorical_features,
                "experiment": {"name": "ctr_dati2_typeppk"},
                "params": {"iterations": 2, "depth": 2},
            }
        )
    )
    (tabm_dir / "tabm_hpo_selection.json").write_text(
        json.dumps(
            {
                "selected_candidate": {
                    "experiment_name": "tabm_piecewise_hpo_k16_t07_ctr_blend_w50",
                    "tabm_weight": 0.5,
                    "params": TabMParams("tabm_piecewise", k=16).as_dict(),
                }
            }
        )
    )


def _result(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    cv: StratifiedKFold,
    probability: float,
) -> dict[str, object]:
    folds = np.full(len(X), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(X, y)):
        folds[valid_idx] = fold
    probabilities = np.where(y.to_numpy() == 1, probability, 1 - probability)
    return {
        "oof_pred": probabilities,
        "test_pred": np.full(len(X_test), probability),
        "test_fold_predictions": np.full((3, len(X_test)), probability),
        "fold_id": folds,
        "fold_metrics": pd.DataFrame([{"fold": i} for i in range(3)]),
        "models": [],
        "fold_preprocessors": [],
    }


def test_source_recipes_and_output_paths_are_isolated(tmp_path: Path) -> None:
    train, test = _data(tmp_path)
    _sources(tmp_path, train, test)
    config = foundation_training.FoundationTrainingConfig(
        project_root=tmp_path, run_name="foundation-test", task_type="CPU"
    )
    recipes = foundation_training._load_source_recipes(
        config, train, test, make_feature_spec(train, test)
    )
    assert recipes.tabm_params.variant == "tabm_piecewise"
    assert recipes.tabm_params.k == 16
    paths = foundation_training.foundation_output_paths(tmp_path, config.run_name)
    assert set(paths) == {"root", "oof", "metrics", "submissions", "cache"}
    with pytest.raises(FileExistsError):
        foundation_training.foundation_output_paths(tmp_path, config.run_name)


def test_screen_selection_is_deterministic_and_uses_guardrails(tmp_path: Path) -> None:
    train, _ = _data(tmp_path)
    folds = np.arange(len(train)) % 3
    base = foundation_training._oof_frame(
        train, folds, np.where(train["label"].to_numpy() == 1, 0.7, 0.3)
    )
    foundation = np.where(train["label"].to_numpy() == 1, 0.9, 0.1)
    candidates, _ = foundation_training._screen_candidates(train, base, foundation, None)
    selected, decision = foundation_training._select_candidate(candidates)
    assert selected["experiment_name"] == "foundation_raw"
    assert decision["selected_foundation_weight"] == 1.0
    assert decision["screen_eligible_candidate_count"] == 5


def test_tabpfn_requires_explicit_terms_confirmation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confirm-tabpfn-eligibility"):
        foundation_training.FoundationTrainingConfig(
            project_root=tmp_path, run_name="foundation-tabpfn", model="tabpfn-3"
        )


def test_preflight_failure_does_not_create_run_directory(tmp_path: Path, monkeypatch) -> None:
    train, test = _data(tmp_path)
    _sources(tmp_path, train, test)
    monkeypatch.setattr(
        foundation_training,
        "run_foundation_preflight",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no GPU")),
    )
    config = foundation_training.FoundationTrainingConfig(
        project_root=tmp_path, run_name="foundation-preflight-failure", task_type="CPU"
    )
    with pytest.raises(RuntimeError, match="no GPU"):
        foundation_training.preflight_foundation_training(config)
    assert not (tmp_path / "outputs" / "runs" / config.run_name).exists()


def test_runner_writes_unpromoted_submission_without_model_artifacts(tmp_path: Path, monkeypatch) -> None:
    train, test = _data(tmp_path)
    _sources(tmp_path, train, test)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    calls: list[tuple[str, int]] = []

    def fake_ctr(train, test, prepared, source_config, cv, seed, task_type, devices):
        calls.append(("ctr", seed))
        result = _result(prepared.X, prepared.y, prepared.X_test, cv, 0.75)
        result["prepared"] = prepared
        result["params"] = source_config["params"]
        return result

    def fake_tabm(train, test, sources, cv, seed, task_type):
        calls.append(("tabm", seed))
        prepared = foundation_training.prepare_tabm_features(
            train, test, make_feature_spec(train, test)
        )
        result = _result(prepared.X, prepared.y, prepared.X_test, cv, 0.8)
        result["prepared"] = prepared
        result["params"] = sources.tabm_params.as_dict()
        return result

    def fake_preflight(*args, **kwargs):
        return {"model": "tabicl-v2", "params": foundation_training.FoundationParams().as_dict()}

    def fake_foundation(X, y, X_test, categorical_features, cv, params, seed, **kwargs):
        calls.append(("foundation", seed))
        result = _result(X, y, X_test, cv, 0.85)
        result["params"] = params.as_dict()
        return result

    monkeypatch.setattr(foundation_training, "_train_ctr_seed", fake_ctr)
    monkeypatch.setattr(foundation_training, "_train_tabm_seed", fake_tabm)
    monkeypatch.setattr(foundation_training, "run_foundation_preflight", fake_preflight)
    monkeypatch.setattr(foundation_training, "train_foundation_cv", fake_foundation)
    monkeypatch.setattr(foundation_training, "_make_cv", lambda config: cv)

    result = foundation_training.run_foundation_training(
        foundation_training.FoundationTrainingConfig(
            project_root=tmp_path,
            run_name="foundation-run",
            task_type="CPU",
            n_bootstrap=2,
            show_progress=False,
        )
    )
    run_dir = tmp_path / "outputs" / "runs" / "foundation-run"
    assert calls == [("ctr", 42), ("tabm", 42), ("ctr", 2026), ("tabm", 2026), ("foundation", 42)]
    assert result["promoted"] is False
    assert result["submission_status"] == "unpromoted"
    assert result["submission_path"].exists()
    assert list((run_dir / "models").glob("*")) == [] if (run_dir / "models").exists() else True
    assert (run_dir / "metrics" / "foundation_promotion_decision.json").exists()
    assert (run_dir / "metrics" / "foundation_final_config.json").exists()
