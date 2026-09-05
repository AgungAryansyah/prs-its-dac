from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import prs_its.tabicl_lora_checkpoint_export as checkpoint_export
from prs_its.tabicl_finetune_modeling import (
    PredictionProfile,
    SupportProfile,
    TabICLFinetuneParams,
)
from prs_its.tabicl_finetune_training import (
    TabICLFinetuneTrainingConfig,
    _expected_fold_ids,
    _make_cv,
    _save_fold_artifacts,
    tabicl_finetune_output_paths,
)


def _run_artifacts(tmp_path: Path):
    train = pd.DataFrame(
        {
            "claim_id": [f"CLM_{index}" for index in range(12)],
            "label": [0, 1] * 6,
            "feature": np.arange(12),
        }
    )
    test = pd.DataFrame({"claim_id": ["TST_3", "TST_1", "TST_2"], "feature": [3, 1, 2]})
    config = TabICLFinetuneTrainingConfig(
        project_root=tmp_path,
        run_name="tabicl-lora-test",
        resume=True,
        params=TabICLFinetuneParams(learning_rate=1e-4, offload_mode=False),
        artifact_prefix="tabicl_lora",
    )
    paths = tabicl_finetune_output_paths(tmp_path, config.run_name)
    (paths["metrics"] / "tabicl_lora_preflight.json").write_text(
        json.dumps(
            {
                "pipeline": "tabicl-lora-finetune",
                "params": config.params.as_dict(),
            }
        )
    )
    cv = _make_cv(config)
    outer_train_idx, outer_valid_idx = next(
        iter(cv.split(np.zeros(len(train)), train["label"]))
    )
    checkpoint_path = paths["models"] / "fold_0" / "attempt-01" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()
    checkpoint_path.with_name("best.adapter.ckpt").touch()
    _save_fold_artifacts(
        paths,
        train,
        test,
        fold=0,
        outer_valid_idx=outer_valid_idx,
        valid_pred=np.linspace(0.1, 0.4, len(outer_valid_idx)),
        test_pred=np.array([0.3, 0.1, 0.2]),
        metrics={"fine_tuning": {"checkpoint_path": str(checkpoint_path)}},
        artifact_prefix="tabicl_lora",
    )
    prepared = SimpleNamespace(
        X=train[["feature"]],
        y=train["label"],
        X_test=test[["feature"]],
    )
    return config, paths, train, test, prepared, outer_train_idx, outer_valid_idx


def test_selects_a_completed_fold_checkpoint_and_ignores_incomplete_later_fold(
    tmp_path: Path,
) -> None:
    config, paths, train, test, prepared, outer_train_idx, outer_valid_idx = (
        _run_artifacts(tmp_path)
    )
    incomplete = paths["oof"] / "tabicl_lora_fold_1_oof.csv"
    incomplete.write_text("partial")

    source = checkpoint_export._select_completed_fold_checkpoint(
        paths, train, test, prepared.y, config, None
    )

    assert source["fold"] == 0
    assert np.array_equal(source["outer_train_idx"], outer_train_idx)
    assert np.array_equal(source["outer_valid_idx"], outer_valid_idx)


def test_checkpoint_export_requires_a_completed_outer_fold(tmp_path: Path) -> None:
    config, paths, train, test, prepared, _, _ = _run_artifacts(tmp_path)
    for path in (
        paths["oof"] / "tabicl_lora_fold_0_oof.csv",
        paths["oof"] / "tabicl_lora_fold_0_test.csv",
        paths["metrics"] / "tabicl_lora_fold_0.json",
    ):
        path.unlink()

    with pytest.raises(RuntimeError, match="No completed TabICL LoRA outer fold"):
        checkpoint_export._select_completed_fold_checkpoint(
            paths, train, test, prepared.y, config, None
        )


def test_checkpoint_export_writes_provisional_submission_without_touching_oof(
    tmp_path: Path, monkeypatch
) -> None:
    config, paths, train, test, prepared, outer_train_idx, _ = _run_artifacts(tmp_path)
    fold_oof = paths["oof"] / "tabicl_lora_fold_0_oof.csv"
    oof_before = fold_oof.read_bytes()

    monkeypatch.setattr(
        checkpoint_export,
        "_load_prepared_data",
        lambda _: (train, test, prepared),
    )
    monkeypatch.setattr(
        checkpoint_export,
        "ensure_tabicl_finetune_gpu_ready",
        lambda _: {"device": "test-cuda", "free_vram_gib": 12.0},
    )
    monkeypatch.setattr(
        checkpoint_export,
        "_validate_lora_checkpoint",
        lambda path: {
            "merged_checkpoint": str(path),
            "merged_checkpoint_sha256": "merged-hash",
            "adapter_checkpoint": str(path.with_name("best.adapter.ckpt")),
            "adapter_checkpoint_sha256": "adapter-hash",
        },
    )
    monkeypatch.setattr(
        checkpoint_export,
        "fit_in_context_predictor",
        lambda *_args: (
            object(),
            SupportProfile(
                strategy="full_support",
                support_rows=len(outer_train_idx),
                support_cap=100_000,
                attempted_rows=(len(outer_train_idx),),
                selected_indices=np.arange(len(outer_train_idx)),
            ),
        ),
    )
    monkeypatch.setattr(
        checkpoint_export,
        "adaptive_predict_probabilities",
        lambda *_args: (
            np.array([0.3, 0.1, 0.2]),
            PredictionProfile(256, 256, (256,)),
        ),
    )

    result = checkpoint_export.export_tabicl_lora_checkpoint_submission(config)

    submission = pd.read_csv(result["submission_path"])
    manifest_path = paths["metrics"] / "tabicl_lora_checkpoint_fold_0_export.json"
    manifest = json.loads(manifest_path.read_text())
    assert submission["claim_id"].tolist() == test["claim_id"].tolist()
    assert submission["fraud_probability"].tolist() == [0.3, 0.1, 0.2]
    assert manifest["submission_status"] == "provisional_checkpoint"
    assert manifest["promotion_decision"] == "not_evaluated_incomplete_oof"
    assert manifest["source"]["fold"] == 0
    assert fold_oof.read_bytes() == oof_before

    with pytest.raises(FileExistsError, match="will not overwrite"):
        checkpoint_export.export_tabicl_lora_checkpoint_submission(config)


def test_checkpoint_export_fold_selection_rejects_out_of_range_fold(
    tmp_path: Path,
) -> None:
    config, paths, train, test, prepared, _, _ = _run_artifacts(tmp_path)

    with pytest.raises(ValueError, match="checkpoint_fold"):
        checkpoint_export._select_completed_fold_checkpoint(
            paths, train, test, prepared.y, config, 3
        )


def test_expected_fold_ids_match_exporter_reconstruction(tmp_path: Path) -> None:
    config, _, _, _, prepared, _, _ = _run_artifacts(tmp_path)
    cv = _make_cv(config)

    expected = _expected_fold_ids(cv, prepared.y)
    reconstructed = checkpoint_export._expected_fold_ids(cv, prepared.y)

    assert np.array_equal(reconstructed, expected)
