from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import parametrize

from prs_its.tabicl_finetune_modeling import (
    TabICLFinetuneParams,
    tabicl_inference_kwargs,
)


@dataclass(frozen=True)
class TabICLLoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive.")
        if self.dropout != 0:
            raise ValueError("TabICL LoRA supports only zero adapter dropout.")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoRATarget:
    module_path: str
    parameter_name: str

    @property
    def state_key(self) -> str:
        return _join_path(self.module_path, self.parameter_name)

    def as_dict(self) -> dict[str, str]:
        return {"module_path": self.module_path, "parameter_name": self.parameter_name}


class LoRAWeight(nn.Module):
    def __init__(self, weight: torch.Tensor, config: TabICLLoRAConfig) -> None:
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("LoRA can only parametrize rank-two projection weights.")
        self.rank = config.rank
        self.scaling = config.alpha / config.rank
        self.lora_a = nn.Parameter(
            torch.empty(
                config.rank, weight.shape[1], dtype=weight.dtype, device=weight.device
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                weight.shape[0], config.rank, dtype=weight.dtype, device=weight.device
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return weight + self.delta().to(dtype=weight.dtype)

    def delta(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling


def install_frozen_lora_adapters(
    model: nn.Module, config: TabICLLoRAConfig
) -> tuple[LoRATarget, ...]:
    existing = getattr(model, "_prs_its_lora_targets", None)
    if existing is not None:
        return tuple(existing)
    from tabicl._model.layers import MultiheadAttentionBlock

    targets: list[LoRATarget] = []
    blocks = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, MultiheadAttentionBlock)
    ]
    if not blocks:
        raise ValueError("TabICL model does not contain transformer attention blocks.")
    for block_path, block in blocks:
        candidates = (
            (block.attn, "in_proj_weight", _join_path(block_path, "attn")),
            (block.attn.out_proj, "weight", _join_path(block_path, "attn.out_proj")),
            (block.linear1, "weight", _join_path(block_path, "linear1")),
            (block.linear2, "weight", _join_path(block_path, "linear2")),
        )
        for module, parameter_path, module_path in candidates:
            owner, parameter_name = _resolve_parameter_owner(module, parameter_path)
            if parametrize.is_parametrized(owner, parameter_name):
                raise ValueError(
                    f"LoRA target is already parametrized: {_join_path(module_path, parameter_name)}"
                )
            weight = getattr(owner, parameter_name)
            if not isinstance(weight, nn.Parameter):
                raise TypeError(
                    f"LoRA target must be an nn.Parameter: {_join_path(module_path, parameter_name)}"
                )
            parametrize.register_parametrization(
                owner, parameter_name, LoRAWeight(weight, config)
            )
            targets.append(
                LoRATarget(module_path=module_path, parameter_name=parameter_name)
            )
    model._prs_its_lora_targets = tuple(targets)
    model._prs_its_lora_config = config.as_dict()
    return tuple(targets)


def freeze_tabicl_for_lora(
    model: nn.Module, config: TabICLLoRAConfig
) -> dict[str, Any]:
    targets = install_frozen_lora_adapters(model, config)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for target in targets:
        adapter = _lora_weight(
            _resolve_module(model, target.module_path), target.parameter_name
        )
        for parameter in adapter.parameters():
            parameter.requires_grad = True
    decoder = getattr(getattr(model, "icl_predictor", None), "decoder", None)
    if not isinstance(decoder, nn.Module):
        raise TypeError(
            "TabICL model does not expose icl_predictor.decoder as the classification head."
        )
    for parameter in decoder.parameters():
        parameter.requires_grad = True
    inventory = lora_parameter_inventory(model)
    if not inventory["trainable_parameter_names"]:
        raise RuntimeError("Frozen LoRA configuration has no trainable parameters.")
    return inventory


def lora_parameter_inventory(model: nn.Module) -> dict[str, Any]:
    targets = tuple(getattr(model, "_prs_its_lora_targets", ()))
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    frozen = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    ]
    invalid = [
        name
        for name in trainable
        if ".parametrizations." not in name
        and not name.startswith("icl_predictor.decoder.")
    ]
    if invalid:
        raise RuntimeError(f"Unexpected trainable TabICL parameters: {invalid}")
    return {
        "adapter_targets": [target.as_dict() for target in targets],
        "adapter_target_count": len(targets),
        "trainable_parameter_names": trainable,
        "frozen_parameter_names": frozen,
        "trainable_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        ),
        "frozen_parameter_count": int(
            sum(
                parameter.numel()
                for parameter in model.parameters()
                if not parameter.requires_grad
            )
        ),
    }


def merged_tabicl_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    targets = tuple(getattr(model, "_prs_its_lora_targets", ()))
    if not targets:
        raise ValueError("TabICL model has no installed LoRA adapters.")
    skipped_prefixes = tuple(
        f"{target.module_path}.parametrizations.{target.parameter_name}."
        if target.module_path
        else f"parametrizations.{target.parameter_name}."
        for target in targets
    )
    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith(skipped_prefixes)
    }
    for target in targets:
        module = _resolve_module(model, target.module_path)
        state[target.state_key] = (
            getattr(module, target.parameter_name).detach().cpu().clone()
        )
    return state


def attach_frozen_lora_finetuner(estimator: Any, config: TabICLLoRAConfig) -> Any:
    def apply_freezing(self: Any, model: nn.Module) -> bool:
        self._prs_its_lora_inventory = freeze_tabicl_for_lora(model, config)
        return True

    def save_checkpoint(
        self: Any,
        output_dir: Path,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        best_metric: float,
        is_best: bool,
        save_interval: bool,
    ) -> None:
        adapter_payload = {
            "config": self.model_config_,
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.model_.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
            if scheduler is not None
            else None,
            "curr_step": epoch,
            "epoch": epoch,
            "best_metric": float(best_metric),
            "lora": {
                "config": config.as_dict(),
                "targets": [
                    target.as_dict() for target in self.model_._prs_its_lora_targets
                ],
                "inventory": self._prs_its_lora_inventory,
            },
        }
        merged_payload = {
            "config": self.model_config_,
            "state_dict": merged_tabicl_state_dict(self.model_),
            "curr_step": epoch,
            "epoch": epoch,
            "best_metric": float(best_metric),
        }
        if is_best:
            torch.save(adapter_payload, output_dir / "best.adapter.ckpt")
            torch.save(merged_payload, output_dir / "best.ckpt")
        if save_interval:
            torch.save(adapter_payload, output_dir / f"epoch{epoch}.adapter.ckpt")
            torch.save(merged_payload, output_dir / f"epoch{epoch}.ckpt")

    estimator._apply_freezing = types.MethodType(apply_freezing, estimator)
    estimator._save_checkpoint = types.MethodType(save_checkpoint, estimator)
    estimator._prs_its_lora_config = config.as_dict()
    return estimator


def create_lora_finetuner(
    *,
    params: TabICLFinetuneParams,
    max_data_size: int,
    output_dir: Path,
    cache_dir: Path,
    random_state: int,
    time_limit_seconds: float | None,
    lora_config: TabICLLoRAConfig,
) -> Any:
    from tabicl import FinetunedTabICLClassifier

    cache_dir.mkdir(parents=True, exist_ok=True)
    estimator = FinetunedTabICLClassifier(
        epochs=params.epochs,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        grad_clip=params.grad_clip,
        amp=True,
        n_estimators_finetune=params.n_estimators_finetune,
        n_estimators_validation=params.n_estimators_validation,
        n_estimators_inference=params.n_estimators_inference,
        max_data_size=max_data_size,
        finetune_ctx_query_ratio=params.finetune_ctx_query_ratio,
        validation_split_ratio=params.validation_split_ratio,
        early_stopping=True,
        patience=params.patience,
        time_limit=time_limit_seconds,
        save_interval=1,
        device="cuda",
        random_state=random_state,
        verbose=False,
        wandb_kwargs=None,
        eval_metric="roc_auc",
        extra_classifier_kwargs=tabicl_inference_kwargs(params, cache_dir),
    )
    return attach_frozen_lora_finetuner(estimator, lora_config)


def lora_finetuner_factory(config: TabICLLoRAConfig) -> Callable[..., Any]:
    def factory(**kwargs: Any) -> Any:
        return create_lora_finetuner(lora_config=config, **kwargs)

    return factory


def _lora_weight(module: nn.Module, parameter_name: str) -> LoRAWeight:
    parametrizations = getattr(module, "parametrizations", None)
    if parametrizations is None or parameter_name not in parametrizations:
        raise ValueError(f"Missing LoRA parametrization for {parameter_name}.")
    parametrization = parametrizations[parameter_name][0]
    if not isinstance(parametrization, LoRAWeight):
        raise TypeError(f"Unexpected parametrization for {parameter_name}.")
    return parametrization


def _resolve_parameter_owner(
    module: nn.Module, parameter_path: str
) -> tuple[nn.Module, str]:
    parts = parameter_path.split(".")
    owner = module
    for part in parts[:-1]:
        child = getattr(owner, part)
        if not isinstance(child, nn.Module):
            raise TypeError(f"LoRA target owner is not a module: {parameter_path}")
        owner = child
    return owner, parts[-1]


def _resolve_module(model: nn.Module, module_path: str) -> nn.Module:
    module: nn.Module = model
    if not module_path:
        return module
    for part in module_path.split("."):
        child = getattr(module, part)
        if not isinstance(child, nn.Module):
            raise TypeError(f"LoRA target is not a module: {module_path}")
        module = child
    return module


def _join_path(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}" if prefix else suffix
