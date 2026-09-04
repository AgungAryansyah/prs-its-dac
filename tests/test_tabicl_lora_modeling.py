from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from prs_its.tabicl_lora_modeling import (
    TabICLLoRAConfig,
    attach_frozen_lora_finetuner,
    freeze_tabicl_for_lora,
    merged_tabicl_state_dict,
)


class _ToyTabICL(nn.Module):
    def __init__(self, blocks: int = 1) -> None:
        super().__init__()
        from tabicl._model.layers import MultiheadAttentionBlock

        self.embedding = nn.Linear(8, 8)
        self.blocks = nn.ModuleList(
            [
                MultiheadAttentionBlock(
                    d_model=8,
                    nhead=2,
                    dim_feedforward=16,
                    zero_init=False,
                )
                for _ in range(blocks)
            ]
        )
        self.icl_predictor = nn.Module()
        self.icl_predictor.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 2)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.embedding(values)
        for block in self.blocks:
            output = block(output)
        return self.icl_predictor.decoder(output.mean(dim=1))


def _model() -> _ToyTabICL:
    torch.manual_seed(42)
    return _ToyTabICL()


def test_frozen_lora_starts_as_an_exact_base_model() -> None:
    model = _model().eval()
    values = torch.randn(4, 3, 8)
    before = model(values)

    inventory = freeze_tabicl_for_lora(model, TabICLLoRAConfig())
    after = model(values)

    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert inventory["adapter_target_count"] == 4
    assert model.embedding.weight.requires_grad is False
    assert all(
        ".parametrizations." in name or name.startswith("icl_predictor.decoder.")
        for name in inventory["trainable_parameter_names"]
    )


def test_lora_update_changes_only_adapters_and_decoder() -> None:
    model = _model()
    freeze_tabicl_for_lora(model, TabICLLoRAConfig())
    before = merged_tabicl_state_dict(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    values = torch.randn(6, 3, 8)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    loss = nn.functional.cross_entropy(model(values), labels)
    loss.backward()
    optimizer.step()
    after = merged_tabicl_state_dict(model)

    assert torch.equal(before["embedding.weight"], after["embedding.weight"])
    assert not torch.equal(
        before["icl_predictor.decoder.0.weight"],
        after["icl_predictor.decoder.0.weight"],
    )
    assert not torch.equal(
        before["blocks.0.attn.in_proj_weight"], after["blocks.0.attn.in_proj_weight"]
    )


def test_merged_state_dict_loads_into_an_unmodified_model() -> None:
    model = _model().eval()
    freeze_tabicl_for_lora(model, TabICLLoRAConfig())
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    values = torch.randn(5, 3, 8)
    labels = torch.tensor([0, 1, 0, 1, 0])
    nn.functional.cross_entropy(model(values), labels).backward()
    optimizer.step()
    adapter_prediction = model(values)
    merged = merged_tabicl_state_dict(model)
    restored = _model().eval()
    restored.load_state_dict(merged, strict=True)

    torch.testing.assert_close(
        adapter_prediction, restored(values), rtol=1e-5, atol=1e-6
    )
    assert not any("parametrizations" in key for key in merged)


def test_checkpoint_hook_writes_adapter_and_standard_merged_checkpoints(
    tmp_path: Path,
) -> None:
    class Estimator:
        def __init__(self, model: nn.Module) -> None:
            self.model_ = model
            self.model_config_ = {"embed_dim": 8}

    model = _model()
    estimator = attach_frozen_lora_finetuner(Estimator(model), TabICLLoRAConfig())
    estimator._apply_freezing(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    estimator._save_checkpoint(
        tmp_path,
        epoch=1,
        optimizer=optimizer,
        scheduler=None,
        best_metric=0.7,
        is_best=True,
        save_interval=True,
    )

    for filename in (
        "best.ckpt",
        "best.adapter.ckpt",
        "epoch1.ckpt",
        "epoch1.adapter.ckpt",
    ):
        assert (tmp_path / filename).exists()
    merged = torch.load(tmp_path / "best.ckpt", weights_only=False)
    adapter = torch.load(tmp_path / "best.adapter.ckpt", weights_only=False)
    assert not any("parametrizations" in key for key in merged["state_dict"])
    assert "lora" in adapter
    assert adapter["lora"]["inventory"]["trainable_parameter_names"]


def test_every_transformer_block_receives_all_four_adapter_targets() -> None:
    model = _ToyTabICL(blocks=2)
    inventory = freeze_tabicl_for_lora(model, TabICLLoRAConfig())

    targets = {
        target["module_path"] + "." + target["parameter_name"]
        for target in inventory["adapter_targets"]
    }

    assert len(targets) == 8
    assert "blocks.0.attn.in_proj_weight" in targets
    assert "blocks.1.linear2.weight" in targets


def test_real_tabicl_targets_column_row_and_icl_transformers_only() -> None:
    from tabicl._model.tabicl import TabICL

    model = TabICL(
        max_classes=2,
        embed_dim=8,
        col_num_blocks=1,
        col_nhead=2,
        col_num_inds=4,
        row_num_blocks=1,
        row_nhead=2,
        row_num_cls=1,
        icl_num_blocks=1,
        icl_nhead=2,
    )
    inventory = freeze_tabicl_for_lora(model, TabICLLoRAConfig())

    target_paths = [target["module_path"] for target in inventory["adapter_targets"]]

    assert any(path.startswith("col_embedder") for path in target_paths)
    assert any(path.startswith("row_interactor") for path in target_paths)
    assert any(path.startswith("icl_predictor.tf_icl") for path in target_paths)
    assert all(
        not name.startswith("icl_predictor.y_encoder")
        for name in inventory["trainable_parameter_names"]
    )


def test_merged_checkpoint_matches_lora_model_through_tabicl_classifier(
    tmp_path: Path,
) -> None:
    from tabicl import TabICLClassifier
    from tabicl._model.tabicl import TabICL

    model_config = {
        "max_classes": 2,
        "embed_dim": 8,
        "col_num_blocks": 1,
        "col_nhead": 2,
        "col_num_inds": 4,
        "row_num_blocks": 1,
        "row_nhead": 2,
        "row_num_cls": 1,
        "icl_num_blocks": 1,
        "icl_nhead": 2,
    }
    lora_config = TabICLLoRAConfig()
    model = TabICL(**model_config)
    freeze_tabicl_for_lora(model, lora_config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora_b"):
                parameter.fill_(0.001)
    adapter_state = model.state_dict()
    adapter_path = tmp_path / "adapter.ckpt"
    merged_path = tmp_path / "merged.ckpt"
    torch.save({"config": model_config, "state_dict": adapter_state}, adapter_path)
    torch.save(
        {"config": model_config, "state_dict": merged_tabicl_state_dict(model)},
        merged_path,
    )
    X_train = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [1.0, 0.0, 1.0, 0.0]})
    y_train = np.array([0, 1, 0, 1])
    X_test = pd.DataFrame({"a": [4.0, 5.0], "b": [1.0, 0.0]})
    standard = TabICLClassifier(
        model_path=merged_path,
        allow_auto_download=False,
        device="cpu",
        n_estimators=1,
        kv_cache=False,
        verbose=False,
    ).fit(X_train, y_train)
    adapter = TabICLClassifier(
        model_path=merged_path,
        allow_auto_download=False,
        device="cpu",
        n_estimators=1,
        kv_cache=False,
        verbose=False,
    )

    def load_adapter(self) -> None:
        loaded = TabICL(**model_config)
        freeze_tabicl_for_lora(loaded, lora_config)
        loaded.load_state_dict(
            torch.load(adapter_path, weights_only=False)["state_dict"]
        )
        self.model_path_ = adapter_path
        self.model_config_ = model_config
        self.model_ = loaded.eval()

    adapter._load_model = types.MethodType(load_adapter, adapter)
    adapter.fit(X_train, y_train)

    np.testing.assert_allclose(
        standard.predict_proba(X_test),
        adapter.predict_proba(X_test),
        rtol=1e-5,
        atol=1e-6,
    )


def test_lora_rejects_adapter_dropout() -> None:
    with pytest.raises(ValueError, match="zero adapter dropout"):
        TabICLLoRAConfig(dropout=0.1)
