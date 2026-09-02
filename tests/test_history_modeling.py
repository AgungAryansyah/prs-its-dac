from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prs_its.history_modeling import (
    HISTORY_ADJUDICATION_FEATURES,
    HISTORY_BEHAVIORAL_FEATURES,
    HISTORY_FINANCIAL_FEATURES,
    add_history_features,
    build_causal_history_features,
    history_provider_groups,
    load_claim_history,
    load_history_column_map,
    map_claim_history_columns,
    validate_claim_history,
)
from prs_its.modeling import FeatureSpec, PreparedFeatures


def _competition_and_history() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "claim_id": "HIST_0",
            "event_at": "2025-01-01T00:00:00Z",
            "adjudicated_at": "2025-01-04T00:00:00Z",
            "provider_id": "P1",
            "patient_id": "A",
            "claim_amount": 100.0,
            "tariff_amount": 80.0,
            "adjudicated_label": 1,
            "kdkc": "K1",
            "typeppk": "T1",
            "cmg": "C1",
            "severitylevel": "S1",
            "diagprimer": "D1",
        },
        {
            "claim_id": "TRN_0",
            "event_at": "2025-01-03T00:00:00Z",
            "adjudicated_at": "2025-01-06T00:00:00Z",
            "provider_id": "P1",
            "patient_id": "B",
            "claim_amount": 120.0,
            "tariff_amount": 100.0,
            "adjudicated_label": 0,
            "kdkc": "K1",
            "typeppk": "T1",
            "cmg": "C1",
            "severitylevel": "S1",
            "diagprimer": "D1",
        },
        {
            "claim_id": "TRN_1",
            "event_at": "2025-01-05T00:00:00Z",
            "adjudicated_at": "2025-01-08T00:00:00Z",
            "provider_id": "P1",
            "patient_id": "A",
            "claim_amount": 140.0,
            "tariff_amount": 100.0,
            "adjudicated_label": 1,
            "kdkc": "K1",
            "typeppk": "T1",
            "cmg": "C1",
            "severitylevel": "S1",
            "diagprimer": "D1",
        },
        {
            "claim_id": "TRN_2",
            "event_at": "2025-01-07T00:00:00Z",
            "adjudicated_at": "2025-01-09T00:00:00Z",
            "provider_id": "P2",
            "patient_id": "C",
            "claim_amount": 60.0,
            "tariff_amount": 60.0,
            "adjudicated_label": 0,
            "kdkc": "K2",
            "typeppk": "T2",
            "cmg": "C2",
            "severitylevel": "S2",
            "diagprimer": "D2",
        },
        {
            "claim_id": "TST_0",
            "event_at": "2025-01-10T00:00:00Z",
            "adjudicated_at": None,
            "provider_id": "P1",
            "patient_id": "D",
            "claim_amount": 200.0,
            "tariff_amount": 100.0,
            "adjudicated_label": None,
            "kdkc": "K1",
            "typeppk": "T1",
            "cmg": "C1",
            "severitylevel": "S1",
            "diagprimer": "D1",
        },
        {
            "claim_id": "TST_1",
            "event_at": "2025-01-11T00:00:00Z",
            "adjudicated_at": None,
            "provider_id": "P3",
            "patient_id": "E",
            "claim_amount": np.nan,
            "tariff_amount": np.nan,
            "adjudicated_label": None,
            "kdkc": "K3",
            "typeppk": "T3",
            "cmg": "C3",
            "severitylevel": "S3",
            "diagprimer": "D3",
        },
    ]
    history = pd.DataFrame(rows)
    current_columns = ["claim_id", "kdkc", "typeppk", "cmg", "severitylevel", "diagprimer"]
    train = history.loc[history["claim_id"].str.startswith("TRN"), current_columns].copy()
    train["label"] = history.loc[history["claim_id"].str.startswith("TRN"), "adjudicated_label"].astype(int).to_numpy()
    test = history.loc[history["claim_id"].str.startswith("TST"), current_columns].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True), history


def test_causal_history_features_only_use_available_events_and_outcomes() -> None:
    train, test, history = _competition_and_history()

    validated = validate_claim_history(history, train, test)
    features = build_causal_history_features(validated).features.set_index("claim_id")

    assert features.loc["HIST_0", "history_provider_prior_claim_count"] == 0
    assert features.loc["TRN_0", "history_provider_prior_claim_count"] == 1
    assert features.loc["TRN_0", "history_provider_adjudicated_count"] == 0
    assert features.loc["TRN_1", "history_provider_prior_claim_count"] == 2
    assert features.loc["TRN_1", "history_provider_adjudicated_count"] == 1
    assert features.loc["TST_0", "history_provider_adjudicated_count"] == 3
    assert features.loc["TST_0", "history_claim_amount_peer_mean_delta"] > 0
    assert features.loc["TST_1", "history_provider_adjudicated_count"] == 0
    assert features.loc["TST_1", "history_provider_fraud_rate"] == features.loc[
        "TST_1", "history_global_fraud_rate"
    ]
    assert features.loc["TST_1", "history_claim_amount_missing"] == 1


def test_history_features_attach_selected_groups_without_identifiers() -> None:
    train, test, history = _competition_and_history()
    bundle = build_causal_history_features(validate_claim_history(history, train, test))
    prepared = PreparedFeatures(
        X=pd.DataFrame({"static_feature": [1.0, 2.0, 3.0]}),
        y=train["label"].copy(),
        X_test=pd.DataFrame({"static_feature": [4.0, 5.0]}),
        spec=FeatureSpec([], [], [], [], [], ["claim_id"]),
        categorical_features=[],
    )

    augmented = add_history_features(
        prepared,
        train["claim_id"],
        test["claim_id"],
        bundle.features,
        ("financial", "behavioral", "adjudication"),
    )

    expected = {
        "static_feature",
        *HISTORY_FINANCIAL_FEATURES,
        *HISTORY_BEHAVIORAL_FEATURES,
        *HISTORY_ADJUDICATION_FEATURES,
    }
    assert set(augmented.X.columns) == expected
    assert list(augmented.X.columns) == list(augmented.X_test.columns)
    assert "claim_id" not in augmented.X
    assert augmented.categorical_features == ["history_provider_id"]
    assert augmented.X.loc[0, "history_provider_id"] == "P1"
    assert history_provider_groups(bundle.features, test["claim_id"]).tolist() == ["P1", "P3"]


def test_claim_history_loader_reads_csv_and_preserves_current_order(tmp_path) -> None:
    train, test, history = _competition_and_history()
    path = tmp_path / "claim_history.csv"
    history.to_csv(path, index=False)

    loaded = load_claim_history(path, train, test)

    assert loaded["claim_id"].tolist() == history["claim_id"].tolist()
    assert str(loaded["event_at"].dtype).startswith("datetime64")


def test_claim_history_column_map_renames_columns_and_normalizes_labels(tmp_path) -> None:
    train, test, history = _competition_and_history()
    source_columns = {column: f"source_{column}" for column in history.columns}
    source = history.rename(columns=source_columns)
    source["source_adjudicated_label"] = source["source_adjudicated_label"].map(
        {0: "legitimate", 1: "fraud"}
    )
    source_path = tmp_path / "mapped_history.csv"
    source.to_csv(source_path, index=False)
    map_path = tmp_path / "history_column_map.json"
    map_path.write_text(
        json.dumps(
            {
                "version": 1,
                "columns": {
                    column: source_columns[column]
                    for column in history.columns
                },
                "label_values": {"legitimate": 0, "fraud": 1},
            }
        )
    )

    loaded = load_claim_history(source_path, train, test, map_path)

    assert loaded.columns.tolist() == list(history.columns)
    assert loaded.loc[loaded["claim_id"].eq("TRN_0"), "adjudicated_label"].item() == 0
    assert loaded.loc[loaded["claim_id"].eq("TRN_1"), "adjudicated_label"].item() == 1


def test_claim_history_column_map_rejects_absent_or_duplicate_sources(tmp_path) -> None:
    absent_path = tmp_path / "absent.json"
    absent_path.write_text(
        json.dumps({"version": 1, "columns": {"claim_id": "missing_claim_id"}})
    )
    mapping = load_history_column_map(absent_path)
    with pytest.raises(ValueError, match="unavailable source columns"):
        map_claim_history_columns(pd.DataFrame({"claim_id": ["A"]}), mapping)

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps(
            {
                "version": 1,
                "columns": {"claim_id": "source_claim_id", "patient_id": "source_claim_id"},
            }
        )
    )
    with pytest.raises(ValueError, match="more than once"):
        load_history_column_map(duplicate_path)


def test_claim_history_column_map_preserves_id_and_temporal_rejections(tmp_path) -> None:
    train, test, history = _competition_and_history()
    source_columns = {column: f"source_{column}" for column in history.columns}
    source = history.rename(columns=source_columns)
    source_path = tmp_path / "mapped_history.csv"
    map_path = tmp_path / "history_column_map.json"
    map_path.write_text(
        json.dumps(
            {"version": 1, "columns": {column: source_columns[column] for column in history.columns}}
        )
    )

    source.loc[source["source_claim_id"].eq("TRN_0"), "source_claim_id"] = "OTHER"
    source.to_csv(source_path, index=False)
    with pytest.raises(ValueError, match="missing current competition"):
        load_claim_history(source_path, train, test, map_path)

    source = history.rename(columns=source_columns)
    source.loc[source["source_claim_id"].eq("TST_0"), "source_event_at"] = "2025-01-09T00:00:00Z"
    source.to_csv(source_path, index=False)
    with pytest.raises(ValueError, match="after every current training"):
        load_claim_history(source_path, train, test, map_path)


def test_claim_history_rejects_test_outcomes_and_noncausal_test_time() -> None:
    train, test, history = _competition_and_history()
    history.loc[history["claim_id"].eq("TST_0"), "adjudicated_label"] = 1
    history.loc[history["claim_id"].eq("TST_0"), "adjudicated_at"] = "2025-01-10T01:00:00Z"

    with pytest.raises(ValueError, match="test claims"):
        validate_claim_history(history, train, test)

    _, _, history = _competition_and_history()
    history.loc[history["claim_id"].eq("TST_0"), "event_at"] = "2025-01-09T00:00:00Z"
    with pytest.raises(ValueError, match="after every current training"):
        validate_claim_history(history, train, test)


def test_claim_history_rejects_context_mismatch() -> None:
    train, test, history = _competition_and_history()
    history.loc[history["claim_id"].eq("TRN_0"), "cmg"] = "MISMATCH"

    with pytest.raises(ValueError, match="Training cmg"):
        validate_claim_history(history, train, test)
