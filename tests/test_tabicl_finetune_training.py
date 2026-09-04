from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from prs_its.tabicl_finetune_training import (
    TabICLFinetuneTrainingConfig,
    _expected_fold_ids,
    _final_full_data_prediction,
    _inner_split_indices,
    _load_completed_fold,
    _require_complete_oof_artifacts,
    _save_fold_artifacts,
    _save_json,
    _save_support_manifest,
    tabicl_finetune_output_paths,
)


def _dataframes() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "claim_id": [f"CLM_{index}" for index in range(12)],
            "label": [0, 1] * 6,
        }
    )
    test = pd.DataFrame({"claim_id": ["TST_3", "TST_1", "TST_2"]})
    return train, test


def test_inner_split_never_uses_the_untouched_outer_validation_rows(tmp_path) -> None:
    labels = np.array([0, 1] * 15)
    config = TabICLFinetuneTrainingConfig(
        project_root=tmp_path, run_name="tabicl-ft-test"
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    outer_train, outer_valid = next(iter(cv.split(np.zeros(len(labels)), labels)))
    inner_train, inner_valid = _inner_split_indices(labels[outer_train], 0.1, 42)

    assert not np.intersect1d(inner_train, inner_valid).size
    assert not np.intersect1d(outer_train[inner_train], outer_valid).size
    assert not np.intersect1d(outer_train[inner_valid], outer_valid).size
    assert config.n_splits == 3


def test_completed_fold_resume_requires_complete_matching_artifacts(tmp_path) -> None:
    train, test = _dataframes()
    paths = tabicl_finetune_output_paths(tmp_path, "tabicl-ft-test")
    expected_folds = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    outer_valid = np.array([4, 5, 6, 7])
    metrics = {"fold": 1, "support": {"strategy": "full_support"}}
    _save_fold_artifacts(
        paths,
        train,
        test,
        fold=1,
        outer_valid_idx=outer_valid,
        valid_pred=np.array([0.1, 0.2, 0.3, 0.4]),
        test_pred=np.array([0.5, 0.6, 0.7]),
        metrics=metrics,
    )

    restored = _load_completed_fold(paths, train, test, 1, outer_valid, expected_folds)

    assert restored is not None
    np.testing.assert_allclose(restored["oof_pred"], [0.1, 0.2, 0.3, 0.4])
    assert restored["metrics"] == metrics
    (paths["metrics"] / "tabicl_ft_fold_1.json").unlink()
    with pytest.raises(FileNotFoundError, match="artifacts are incomplete"):
        _load_completed_fold(paths, train, test, 1, outer_valid, expected_folds)


def test_final_resume_preserves_test_claim_id_order(tmp_path) -> None:
    _, test = _dataframes()
    paths = tabicl_finetune_output_paths(tmp_path, "tabicl-ft-test")
    output = paths["oof"] / "tabicl_ft_final_test_raw.csv"
    pd.DataFrame(
        {"claim_id": test["claim_id"], "fraud_probability_raw": [0.3, 0.1, 0.2]}
    ).to_csv(output, index=False)
    _save_json(
        paths["metrics"] / "tabicl_ft_final_support.json",
        {
            "strategy": "full_support",
            "support_rows": 12,
            "support_cap": 100_000,
            "attempted_rows": [12],
            "selected_indices": list(range(12)),
        },
    )
    config = TabICLFinetuneTrainingConfig(
        project_root=tmp_path,
        run_name="tabicl-ft-test",
        resume=True,
    )

    prediction, support = _final_full_data_prediction(
        config,
        paths,
        pd.DataFrame({"value": np.arange(12)}),
        pd.Series([0, 1] * 6),
        pd.DataFrame({"value": np.arange(len(test))}),
        pd.DataFrame({"claim_id": [f"CLM_{index}" for index in range(12)]}),
        test["claim_id"],
        started=0.0,
    )

    np.testing.assert_allclose(prediction, [0.3, 0.1, 0.2])
    assert support.strategy == "full_support"


def test_support_manifest_is_preserved_when_a_resumed_run_adds_rows(tmp_path) -> None:
    paths = tabicl_finetune_output_paths(tmp_path, "tabicl-ft-test")
    original = pd.DataFrame(
        {"phase": ["outer_fold"], "fold": [0], "claim_id": ["CLM_1"]}
    )
    _save_support_manifest(paths, [original])
    _save_support_manifest(
        paths,
        [pd.DataFrame({"phase": ["final"], "fold": [None], "claim_id": ["CLM_2"]})],
    )

    manifest = pd.read_csv(paths["metrics"] / "tabicl_ft_support_manifest.csv")

    assert set(manifest["claim_id"]) == {"CLM_1", "CLM_2"}


def test_final_finetuning_requires_completed_oof_artifacts(tmp_path) -> None:
    paths = tabicl_finetune_output_paths(tmp_path, "tabicl-ft-test")

    with pytest.raises(RuntimeError, match="requires complete OOF artifacts"):
        _require_complete_oof_artifacts(paths)

    for path in (
        paths["oof"] / "tabicl_ft_raw_oof.csv",
        paths["oof"] / "tabicl_ft_oof.csv",
        paths["metrics"] / "tabicl_ft_fold_metrics.json",
        paths["metrics"] / "tabicl_ft_oof_metrics.json",
        paths["metrics"] / "tabicl_ft_vs_ctr_paired.csv",
    ):
        path.touch()

    _require_complete_oof_artifacts(paths)


def test_gpu_only_config_rejects_cpu(tmp_path) -> None:
    with pytest.raises(ValueError, match="GPU-only"):
        TabICLFinetuneTrainingConfig(
            project_root=tmp_path, run_name="tabicl-ft-test", task_type="CPU"
        )


def test_expected_fold_ids_cover_each_training_row(tmp_path) -> None:
    train, _ = _dataframes()
    config = TabICLFinetuneTrainingConfig(
        project_root=tmp_path, run_name="tabicl-ft-test"
    )

    fold_ids = _expected_fold_ids(
        StratifiedKFold(
            n_splits=config.n_splits, shuffle=True, random_state=config.random_state
        ),
        train["label"],
    )

    assert set(fold_ids) == {0, 1, 2}
