from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from prs_its.tabicl_finetune_modeling import (
    TabICLFinetuneParams,
    adaptive_predict_probabilities,
    create_tabicl_predictor,
    release_cuda_memory,
)
from prs_its.tabicl_finetune_training import (
    TabICLFinetuneTrainingConfig,
    _load_prepared_data,
    _save_json,
    _stratified_sample_indices,
    preflight_tabicl_finetune,
    run_tabicl_finetune_training,
)
from prs_its.tabicl_lora_modeling import TabICLLoRAConfig, lora_finetuner_factory
from prs_its.training import find_project_root


@dataclass(frozen=True)
class TabICLLoRATrainingConfig:
    project_root: Path
    run_name: str
    incumbent_run_name: str = "ctr-v1"
    full_finetune_run_name: str = "tabicl-ft-v1"
    task_type: str = "GPU"
    devices: str = "0"
    max_runtime_minutes: float = 720.0
    n_bootstrap: int = 1000
    show_progress: bool = True
    resume: bool = False
    params: TabICLFinetuneParams = field(
        default_factory=lambda: TabICLFinetuneParams(learning_rate=1e-4)
    )
    lora: TabICLLoRAConfig = field(default_factory=TabICLLoRAConfig)

    def __post_init__(self) -> None:
        if not self.run_name:
            raise ValueError("run_name is required.")
        if self.run_name == self.full_finetune_run_name:
            raise ValueError(
                "LoRA run_name must differ from the full TabICL fine-tune run."
            )
        if self.params.learning_rate != 1e-4:
            raise ValueError("Frozen LoRA fine-tuning requires learning_rate=1e-4.")


def preflight_tabicl_lora(config: TabICLLoRATrainingConfig) -> dict[str, Any]:
    base = _base_config(config, resume=config.resume)
    preflight = preflight_tabicl_finetune(base)
    checkpoint_path = Path(preflight["checkpoint_validation"]["checkpoint_path"])
    adapter_path = checkpoint_path.with_name("best.adapter.ckpt")
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA preflight did not save adapter checkpoint: {adapter_path}"
        )
    merged = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    state_dict = merged.get("state_dict")
    lora_payload = adapter.get("lora")
    if not isinstance(state_dict, dict) or any(
        "parametrizations" in key for key in state_dict
    ):
        raise RuntimeError(
            "LoRA merged checkpoint is not compatible with TabICLClassifier."
        )
    if not isinstance(lora_payload, dict) or not lora_payload.get("targets"):
        raise RuntimeError("LoRA adapter checkpoint does not record adapter targets.")
    inventory = lora_payload.get("inventory")
    if not isinstance(inventory, dict) or not inventory.get(
        "trainable_parameter_names"
    ):
        raise RuntimeError(
            "LoRA adapter checkpoint does not record the trainable parameter inventory."
        )
    train, _, prepared = _load_prepared_data(base)
    sample_idx = _stratified_sample_indices(
        prepared.y.to_numpy(), 128, base.random_state
    )
    smoke_params = replace(config.params, n_estimators_inference=1)
    predictor = create_tabicl_predictor(
        checkpoint_path,
        smoke_params,
        base.project_root
        / "outputs"
        / "runs"
        / config.run_name
        / "cache"
        / "preflight-standard",
        base.random_state,
    )
    try:
        predictor.fit(
            prepared.X.iloc[sample_idx], prepared.y.iloc[sample_idx].to_numpy()
        )
        _, profile = adaptive_predict_probabilities(
            predictor,
            prepared.X.iloc[sample_idx[: min(8, len(sample_idx))]],
            smoke_params.prediction_chunk_size,
            smoke_params.min_prediction_chunk_size,
        )
    finally:
        del predictor
        release_cuda_memory()
    payload = {
        **preflight,
        "pipeline": "tabicl-lora-finetune",
        "lora": {
            "config": config.lora.as_dict(),
            "adapter_target_count": len(lora_payload["targets"]),
            "adapter_targets": lora_payload["targets"],
            "frozen_base": True,
            "trainable_parameter_inventory": inventory,
            "adapter_checkpoint": str(adapter_path),
            "merged_checkpoint": str(checkpoint_path),
        },
        "standard_predictor": {
            "support_rows": len(sample_idx),
            "prediction_profile": profile.as_dict(),
            "checkpoint_loadable": True,
        },
        "preflight_train_rows": len(train),
    }
    _save_json(
        base.project_root
        / "outputs"
        / "runs"
        / config.run_name
        / "metrics"
        / "tabicl_lora_preflight.json",
        payload,
    )
    return payload


def run_tabicl_lora_training(config: TabICLLoRATrainingConfig) -> dict[str, Any]:
    if config.resume:
        base = _base_config(config, resume=True)
    else:
        preflight_tabicl_lora(config)
        base = _base_config(config, resume=True)
    result = run_tabicl_finetune_training(base)
    manifest = {
        "pipeline": "tabicl-lora-finetune",
        "run_name": config.run_name,
        "full_finetune_run_name": config.full_finetune_run_name,
        "params": config.params.as_dict(),
        "lora": config.lora.as_dict(),
        "artifact_prefix": "tabicl_lora",
        "base_weights": "frozen",
        "trainable_parameters": ["lora_adapters", "icl_predictor.decoder"],
        "result": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in result.items()
        },
    }
    _save_json(
        config.project_root
        / "outputs"
        / "runs"
        / config.run_name
        / "metrics"
        / "tabicl_lora_challenger.json",
        manifest,
    )
    return result


def _base_config(
    config: TabICLLoRATrainingConfig, *, resume: bool
) -> TabICLFinetuneTrainingConfig:
    return TabICLFinetuneTrainingConfig(
        project_root=config.project_root,
        run_name=config.run_name,
        incumbent_run_name=config.incumbent_run_name,
        task_type=config.task_type,
        devices=config.devices,
        max_runtime_minutes=config.max_runtime_minutes,
        n_bootstrap=config.n_bootstrap,
        show_progress=config.show_progress,
        resume=resume,
        params=config.params,
        finetuner_factory=lora_finetuner_factory(config.lora),
        artifact_prefix="tabicl_lora",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune frozen LoRA adapters for TabICLv2."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--incumbent-run-name", default="ctr-v1")
    parser.add_argument("--full-finetune-run-name", default="tabicl-ft-v1")
    parser.add_argument("--task-type", default="GPU", choices=["GPU"])
    parser.add_argument("--devices", default="0")
    parser.add_argument("--max-runtime-minutes", type=float, default=720.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--max-data-size", type=int, choices=[4096, 2048, 1024], default=4096
    )
    parser.add_argument("--prediction-chunk-size", type=int, default=256)
    parser.add_argument("--min-prediction-chunk-size", type=int, default=64)
    parser.add_argument("--support-cap", type=int, default=100_000)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = TabICLLoRATrainingConfig(
        project_root=find_project_root(),
        run_name=args.run_name,
        incumbent_run_name=args.incumbent_run_name,
        full_finetune_run_name=args.full_finetune_run_name,
        task_type=args.task_type,
        devices=args.devices,
        max_runtime_minutes=args.max_runtime_minutes,
        n_bootstrap=args.n_bootstrap,
        show_progress=not args.quiet,
        resume=args.resume,
        params=TabICLFinetuneParams(
            epochs=args.epochs,
            learning_rate=1e-4,
            weight_decay=args.weight_decay,
            max_data_size=args.max_data_size,
            prediction_chunk_size=args.prediction_chunk_size,
            min_prediction_chunk_size=args.min_prediction_chunk_size,
            support_cap=args.support_cap,
        ),
        lora=TabICLLoRAConfig(rank=args.lora_rank, alpha=args.lora_alpha),
    )
    result = (
        preflight_tabicl_lora(config)
        if args.preflight
        else run_tabicl_lora_training(config)
    )
    print(json.dumps(result, indent=2, default=str))
