from __future__ import annotations

import pandas as pd
import pytest

from prs_its.tabicl_finetune_training import (
    _save_support_manifest,
    tabicl_finetune_output_paths,
)
from prs_its.tabicl_lora_training import TabICLLoRATrainingConfig, _base_config


def test_lora_base_config_uses_isolated_artifacts_and_adapter_factory(tmp_path) -> None:
    config = TabICLLoRATrainingConfig(
        project_root=tmp_path, run_name="tabicl-lora-test"
    )

    base = _base_config(config, resume=False)

    assert base.run_name == "tabicl-lora-test"
    assert base.artifact_prefix == "tabicl_lora"
    assert base.params.learning_rate == 1e-4
    assert base.finetuner_factory is not None


def test_lora_config_rejects_the_full_finetune_run_name(tmp_path) -> None:
    with pytest.raises(ValueError, match="must differ"):
        TabICLLoRATrainingConfig(project_root=tmp_path, run_name="tabicl-ft-v1")


def test_lora_config_rejects_a_full_finetuning_learning_rate(tmp_path) -> None:
    from prs_its.tabicl_finetune_modeling import TabICLFinetuneParams

    with pytest.raises(ValueError, match="learning_rate=1e-4"):
        TabICLLoRATrainingConfig(
            project_root=tmp_path,
            run_name="tabicl-lora-test",
            params=TabICLFinetuneParams(learning_rate=1e-5),
        )


def test_lora_support_manifest_uses_an_isolated_artifact_prefix(tmp_path) -> None:
    paths = tabicl_finetune_output_paths(tmp_path, "tabicl-lora-test")
    _save_support_manifest(
        paths,
        [pd.DataFrame({"phase": ["final"], "fold": [None], "claim_id": ["CLM_1"]})],
        "tabicl_lora",
    )

    assert (paths["metrics"] / "tabicl_lora_support_manifest.csv").exists()
    assert not (paths["metrics"] / "tabicl_ft_support_manifest.csv").exists()
