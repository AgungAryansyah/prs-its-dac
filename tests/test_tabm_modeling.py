from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn

from prs_its.modeling import CATEGORICAL_CANDIDATES, make_feature_spec
from prs_its.tabm_modeling import (
    FoldTabMPreprocessor,
    TabMParams,
    _predict_probabilities,
    ensure_tabm_gpu_ready,
    prepare_tabm_features,
    train_tabm_cv,
)


def _competition_data(rows: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = []
    for index in range(rows):
        row = {
            "claim_id": f"TRN_{index}",
            "label": index % 2,
            "umur": 20 + index,
            "los": float(index % 6),
            "dx2_01": index % 2,
            "proc01": (index // 2) % 2,
        }
        for feature in CATEGORICAL_CANDIDATES:
            row[feature] = f"{feature}_{index % 3}"
        values.append(row)
    train = pd.DataFrame(values)
    test = train.drop(columns="label").iloc[:8].copy()
    test["claim_id"] = [f"TST_{index}" for index in range(len(test))]
    return train, test


def _small_tabm_params(variant: str) -> TabMParams:
    return TabMParams(
        variant=variant,
        k=2,
        d_block=32,
        n_blocks=1,
        batch_size=16,
        max_epochs=3,
        patience=2,
        inner_validation_fraction=0.2,
        piecewise_bins=4,
        piecewise_embedding_dim=4,
    )


def test_fold_preprocessor_uses_train_categories_and_roundtrips(tmp_path) -> None:
    train = pd.DataFrame(
        {
            "numeric": [0.0, 1.0, np.nan, 3.0],
            "constant": [1.0, 1.0, 1.0, 1.0],
            "category": ["A", "B", None, "A"],
        }
    )
    validation = pd.DataFrame(
        {
            "numeric": [2.0, np.inf],
            "constant": [1.0, 1.0],
            "category": ["UNSEEN", None],
        }
    )
    preprocessor = FoldTabMPreprocessor(["category"], random_state=42).fit(train)

    x_num, x_cat = preprocessor.transform(validation)
    assert np.isfinite(x_num).all()
    assert x_cat is not None
    assert x_cat[0, 0] == 0
    assert x_cat[1, 0] > 0
    assert "constant" not in preprocessor.numeric_output_features

    path = tmp_path / "preprocessor.joblib"
    preprocessor.save(path, seed=42, fold=0)
    restored_num, restored_cat = FoldTabMPreprocessor.load(path).transform(validation)
    np.testing.assert_allclose(restored_num, x_num)
    np.testing.assert_array_equal(restored_cat, x_cat)


def test_prepare_tabm_features_reuses_ctr_static_schema() -> None:
    train, test = _competition_data()
    prepared = prepare_tabm_features(train, test, make_feature_spec(train, test))

    assert "claim_id" not in prepared.X
    assert "secondary_diagnosis_count" in prepared.X
    assert "procedure_count_bucket" in prepared.X
    assert "dati2_typeppk" in prepared.X
    assert list(prepared.X.columns) == list(prepared.X_test.columns)


def test_tabm_cv_saves_fold_artifacts_and_completes_oof(tmp_path) -> None:
    rows = 40
    X = pd.DataFrame(
        {"numeric": np.arange(rows, dtype=float), "category": [f"C_{index % 3}" for index in range(rows)]}
    )
    X_test = pd.DataFrame({"numeric": [1.5, 10.5, 20.5], "category": ["C_0", "UNSEEN", "C_2"]})
    labels = pd.Series(np.arange(rows) % 2)
    events: list[tuple[str, int]] = []

    result = train_tabm_cv(
        X,
        labels,
        X_test,
        ["category"],
        StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        _small_tabm_params("tabm_base"),
        seed=42,
        task_type="CPU",
        model_dir=tmp_path,
        model_prefix="base",
        progress_callback=lambda status, fold: events.append((status, fold)),
    )

    assert np.isfinite(result["oof_pred"]).all()
    assert result["test_fold_predictions"].shape == (2, len(X_test))
    np.testing.assert_allclose(result["test_pred"], result["test_fold_predictions"].mean(axis=0))
    assert events == [("start", 0), ("complete", 0), ("start", 1), ("complete", 1)]
    assert len(list(tmp_path.glob("base_fold_*.pt"))) == 2
    assert len(list(tmp_path.glob("base_preprocessor_fold_*.joblib"))) == 2
    assert set(result["fold_model_features"]) == {0, 1}


def test_piecewise_tabm_handles_constant_missing_indicators() -> None:
    rows = 40
    X = pd.DataFrame(
        {"numeric": np.arange(rows, dtype=float), "category": [f"C_{index % 2}" for index in range(rows)]}
    )
    X_test = X.iloc[:3].copy()
    labels = pd.Series(np.arange(rows) % 2)

    result = train_tabm_cv(
        X,
        labels,
        X_test,
        ["category"],
        StratifiedKFold(n_splits=2, shuffle=True, random_state=42),
        _small_tabm_params("tabm_piecewise"),
        seed=42,
        task_type="CPU",
    )

    assert np.isfinite(result["oof_pred"]).all()


class _FixedMemberModel(nn.Module):
    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor | None) -> torch.Tensor:
        logits = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32, device=x_num.device)
        return logits.repeat(len(x_num), 1).unsqueeze(-1)


def test_tabm_prediction_averages_member_probabilities() -> None:
    probabilities = _predict_probabilities(
        _FixedMemberModel(),
        np.ones((3, 1), dtype=np.float32),
        None,
        batch_size=2,
        device=torch.device("cpu"),
    )

    np.testing.assert_allclose(probabilities, np.full(3, 0.625))


def test_cuda_probe_rejects_cpu_only_torch(monkeypatch) -> None:
    monkeypatch.setattr("prs_its.tabm_modeling.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA PyTorch is unavailable"):
        ensure_tabm_gpu_ready()
