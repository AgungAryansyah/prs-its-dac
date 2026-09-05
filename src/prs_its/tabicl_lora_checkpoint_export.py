from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from prs_its.modeling import ID_COL
from prs_its.submission import make_submission
from prs_its.tabicl_finetune_modeling import (
    TabICLFinetuneParams,
    adaptive_predict_probabilities,
    create_tabicl_predictor,
    ensure_tabicl_finetune_gpu_ready,
    fit_in_context_predictor,
    release_cuda_memory,
)
from prs_its.tabicl_finetune_training import (
    TabICLFinetuneTrainingConfig,
    _load_completed_fold,
    _load_json,
    _load_prepared_data,
    _make_cv,
    _support_manifest_rows,
    tabicl_finetune_output_paths,
)

EXPORT_ARTIFACT_PREFIX = "tabicl_lora_checkpoint"


def export_tabicl_lora_checkpoint_submission(
    config: TabICLFinetuneTrainingConfig,
    checkpoint_fold: int | None = None,
) -> dict[str, Any]:
    if config.artifact_prefix != "tabicl_lora":
        raise ValueError("Checkpoint export is available only for the TabICL LoRA run.")
    paths = tabicl_finetune_output_paths(
        config.project_root, config.run_name, resume=True
    )
    preflight = _load_lora_preflight(paths)
    stored_params = _stored_lora_params(preflight)
    export_config = replace(config, params=stored_params, resume=True)
    train, test, prepared = _load_prepared_data(export_config)
    source = _select_completed_fold_checkpoint(
        paths,
        train,
        test,
        prepared.y,
        export_config,
        checkpoint_fold,
    )
    output_paths = _export_output_paths(paths, source["fold"])
    _require_new_export_artifacts(output_paths)
    gpu = ensure_tabicl_finetune_gpu_ready(stored_params.min_free_vram_gib)
    checkpoint = _validate_lora_checkpoint(source["checkpoint_path"])
    outer_train_idx = source["outer_train_idx"]
    predictor, support = fit_in_context_predictor(
        lambda: create_tabicl_predictor(
            source["checkpoint_path"],
            stored_params,
            paths["cache"] / f"checkpoint-export-fold-{source['fold']}" / "predictor",
            export_config.random_state + source["fold"],
        ),
        prepared.X.iloc[outer_train_idx],
        prepared.y.iloc[outer_train_idx],
        stored_params.support_cap,
        export_config.random_state + source["fold"],
    )
    try:
        probabilities, prediction_profile = adaptive_predict_probabilities(
            predictor,
            prepared.X_test,
            stored_params.prediction_chunk_size,
            stored_params.min_prediction_chunk_size,
        )
    finally:
        del predictor
        release_cuda_memory()
    raw_predictions = pd.DataFrame(
        {
            ID_COL: test[ID_COL].to_numpy(),
            "fraud_probability_raw": probabilities,
        }
    )
    submission = make_submission(test[ID_COL], probabilities)
    support_manifest = _support_manifest_rows(
        train.iloc[outer_train_idx],
        support.selected_indices,
        "checkpoint_export",
        source["fold"],
    )
    manifest = {
        "pipeline": "tabicl-lora-checkpoint-export",
        "run_name": config.run_name,
        "submission_status": "provisional_checkpoint",
        "promotion_decision": "not_evaluated_incomplete_oof",
        "source": {
            **checkpoint,
            "fold": source["fold"],
            "checkpoint_selection": source["selection"],
            "outer_train_rows": len(outer_train_idx),
            "outer_validation_rows": len(source["outer_valid_idx"]),
        },
        "params": stored_params.as_dict(),
        "gpu": gpu,
        "support": support.as_dict(),
        "prediction": prediction_profile.as_dict(),
        "artifacts": {key: str(path) for key, path in output_paths.items()},
    }
    _write_csv_atomically(raw_predictions, output_paths["raw_predictions"])
    _write_csv_atomically(support_manifest, output_paths["support_manifest"])
    _write_csv_atomically(submission, output_paths["submission"])
    _write_json_atomically(manifest, output_paths["manifest"])
    return {
        "submission_path": output_paths["submission"],
        "source_checkpoint": source["checkpoint_path"],
        "source_fold": source["fold"],
        "submission_status": "provisional_checkpoint",
    }


def _load_lora_preflight(paths: dict[str, Path]) -> dict[str, Any]:
    payload = _load_json(paths["metrics"] / "tabicl_lora_preflight.json")
    if payload.get("pipeline") != "tabicl-lora-finetune":
        raise ValueError("Saved preflight is not from the TabICL LoRA pipeline.")
    if not isinstance(payload.get("params"), dict):
        raise TypeError("Saved TabICL LoRA preflight does not record parameters.")
    return payload


def _stored_lora_params(preflight: dict[str, Any]) -> TabICLFinetuneParams:
    values = dict(preflight["params"])
    values["offload_mode"] = False
    return TabICLFinetuneParams(**values)


def _select_completed_fold_checkpoint(
    paths: dict[str, Path],
    train: pd.DataFrame,
    test: pd.DataFrame,
    labels: pd.Series,
    config: TabICLFinetuneTrainingConfig,
    checkpoint_fold: int | None,
) -> dict[str, Any]:
    if checkpoint_fold is not None and not 0 <= checkpoint_fold < config.n_splits:
        raise ValueError(
            f"checkpoint_fold must be between 0 and {config.n_splits - 1}."
        )
    cv = _make_cv(config)
    candidates: list[dict[str, Any]] = []
    for fold, (outer_train_idx, outer_valid_idx) in enumerate(
        cv.split(np.zeros(len(labels)), labels)
    ):
        if checkpoint_fold is not None and fold != checkpoint_fold:
            continue
        try:
            saved = _load_completed_fold(
                paths,
                train,
                test,
                fold,
                outer_valid_idx,
                _expected_fold_ids(cv, labels),
                config.artifact_prefix,
            )
        except FileNotFoundError:
            continue
        if saved is None:
            continue
        checkpoint_path = _checkpoint_path_from_fold_metrics(saved["metrics"], fold)
        _require_fold_checkpoint_location(checkpoint_path, paths["models"], fold)
        adapter_path = checkpoint_path.with_name("best.adapter.ckpt")
        if not checkpoint_path.exists() or not adapter_path.exists():
            continue
        candidates.append(
            {
                "fold": fold,
                "checkpoint_path": checkpoint_path,
                "outer_train_idx": outer_train_idx.astype(int),
                "outer_valid_idx": outer_valid_idx.astype(int),
                "selection": "completed_outer_fold",
            }
        )
    if not candidates:
        requested = (
            f" for fold {checkpoint_fold}" if checkpoint_fold is not None else ""
        )
        raise RuntimeError(
            "No completed TabICL LoRA outer fold with both merged and adapter "
            f"checkpoints is available{requested}. Resume training or select a completed fold."
        )
    return min(candidates, key=lambda candidate: candidate["fold"])


def _expected_fold_ids(cv: Any, labels: pd.Series) -> np.ndarray:
    fold_ids = np.full(len(labels), -1, dtype=int)
    for fold, (_, valid_idx) in enumerate(cv.split(np.zeros(len(labels)), labels)):
        fold_ids[valid_idx] = fold
    return fold_ids


def _checkpoint_path_from_fold_metrics(metrics: dict[str, Any], fold: int) -> Path:
    fine_tuning = metrics.get("fine_tuning")
    if not isinstance(fine_tuning, dict):
        raise TypeError(f"Completed outer fold {fold} lacks fine-tuning metadata.")
    checkpoint = fine_tuning.get("checkpoint_path")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise TypeError(f"Completed outer fold {fold} lacks a checkpoint path.")
    return Path(checkpoint)


def _require_fold_checkpoint_location(
    checkpoint_path: Path, models_path: Path, fold: int
) -> None:
    fold_path = (models_path / f"fold_{fold}").resolve()
    try:
        checkpoint_path.resolve().relative_to(fold_path)
    except ValueError as error:
        raise ValueError(
            f"Fold {fold} checkpoint is outside its model directory: {checkpoint_path}"
        ) from error


def _validate_lora_checkpoint(checkpoint_path: Path) -> dict[str, str]:
    adapter_path = checkpoint_path.with_name("best.adapter.ckpt")
    merged = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    state_dict = merged.get("state_dict") if isinstance(merged, dict) else None
    lora = adapter.get("lora") if isinstance(adapter, dict) else None
    if not isinstance(state_dict, dict) or any(
        "parametrizations" in key for key in state_dict
    ):
        raise RuntimeError("Merged checkpoint is not loadable by TabICLClassifier.")
    if not isinstance(lora, dict) or not lora.get("targets"):
        raise RuntimeError("Adapter checkpoint does not contain LoRA target metadata.")
    return {
        "merged_checkpoint": str(checkpoint_path),
        "merged_checkpoint_sha256": _sha256(checkpoint_path),
        "adapter_checkpoint": str(adapter_path),
        "adapter_checkpoint_sha256": _sha256(adapter_path),
    }


def _export_output_paths(paths: dict[str, Path], fold: int) -> dict[str, Path]:
    stem = f"{EXPORT_ARTIFACT_PREFIX}_fold_{fold}"
    return {
        "raw_predictions": paths["oof"] / f"{stem}_test_raw.csv",
        "support_manifest": paths["metrics"] / f"{stem}_support_manifest.csv",
        "submission": paths["submissions"] / f"{stem}_submission.csv",
        "manifest": paths["metrics"] / f"{stem}_export.json",
    }


def _require_new_export_artifacts(paths: dict[str, Path]) -> None:
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Checkpoint submission export already has artifacts and will not overwrite them: "
            f"{existing}"
        )


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Checkpoint export temporary file already exists: {temporary}"
        )
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json_atomically(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(
            f"Checkpoint export temporary file already exists: {temporary}"
        )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
