from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from prs_its.modeling import ID_COL, TARGET, PreparedFeatures


HISTORY_REQUIRED_COLUMNS = (
    ID_COL,
    "event_at",
    "adjudicated_at",
    "provider_id",
    "patient_id",
    "claim_amount",
    "tariff_amount",
    "adjudicated_label",
    "kdkc",
    "typeppk",
    "cmg",
    "severitylevel",
    "diagprimer",
)
HISTORY_COLUMN_MAP_VERSION = 1
HISTORY_CONTEXT_COLUMNS = ("kdkc", "typeppk", "cmg", "severitylevel", "diagprimer")
HISTORY_FINANCIAL_FEATURES = (
    "history_claim_amount",
    "history_tariff_amount",
    "history_amount_minus_tariff",
    "history_amount_to_tariff_ratio",
    "history_log_claim_amount",
    "history_log_tariff_amount",
    "history_claim_amount_missing",
    "history_tariff_amount_missing",
)
HISTORY_BEHAVIORAL_FEATURES = (
    "history_provider_id",
    "history_provider_prior_claim_count",
    "history_provider_prior_claim_count_30d",
    "history_provider_prior_claim_count_90d",
    "history_provider_prior_claim_count_365d",
    "history_provider_days_since_claim",
    "history_patient_prior_claim_count",
    "history_patient_prior_claim_count_30d",
    "history_patient_prior_claim_count_90d",
    "history_patient_prior_claim_count_365d",
    "history_patient_days_since_claim",
    "history_provider_patient_prior_claim_count",
    "history_provider_patient_prior_claim_count_30d",
    "history_provider_patient_prior_claim_count_90d",
    "history_provider_patient_prior_claim_count_365d",
    "history_provider_patient_days_since_claim",
    "history_provider_prior_unique_patients",
    "history_provider_prior_unique_cmg",
    "history_provider_prior_unique_diagnoses",
    "history_provider_cmg_prior_claim_count",
    "history_provider_diagnosis_prior_claim_count",
    "history_cmg_prior_claim_count",
    "history_diagprimer_prior_claim_count",
    "history_peer_prior_claim_count",
    "history_claim_amount_peer_mean_ratio",
    "history_tariff_amount_peer_mean_ratio",
    "history_claim_amount_peer_mean_delta",
)
HISTORY_ADJUDICATION_FEATURES = (
    "history_global_adjudicated_count",
    "history_global_fraud_rate",
    "history_provider_adjudicated_count",
    "history_provider_fraud_rate",
    "history_patient_adjudicated_count",
    "history_patient_fraud_rate",
    "history_provider_cmg_adjudicated_count",
    "history_provider_cmg_fraud_rate",
    "history_provider_diagnosis_adjudicated_count",
    "history_provider_diagnosis_fraud_rate",
)
HISTORY_CATEGORICAL_FEATURES = ("history_provider_id",)
HISTORY_FEATURE_GROUPS = {
    "financial": HISTORY_FINANCIAL_FEATURES,
    "behavioral": HISTORY_BEHAVIORAL_FEATURES,
    "adjudication": HISTORY_ADJUDICATION_FEATURES,
}
_WINDOW_DAYS = (30, 90, 365)
_DAY_NS = 24 * 60 * 60 * 1_000_000_000


@dataclass(frozen=True)
class HistoryFeatureBundle:
    features: pd.DataFrame
    feature_groups: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class HistoryColumnMap:
    version: int
    columns: dict[str, str]
    label_values: dict[str, int] | None = None


@dataclass
class _RollingCount:
    total: int = 0
    last_time: int | None = None
    windows: dict[int, deque[int]] = field(
        default_factory=lambda: {days: deque() for days in _WINDOW_DAYS}
    )

    def values(self, time_value: int) -> tuple[int, int, int, int, float]:
        counts = []
        for days in _WINDOW_DAYS:
            cutoff = time_value - days * _DAY_NS
            values = self.windows[days]
            while values and values[0] <= cutoff:
                values.popleft()
            counts.append(len(values))
        recency = (
            np.nan if self.last_time is None else float((time_value - self.last_time) / _DAY_NS)
        )
        return self.total, *counts, recency

    def add(self, time_value: int) -> None:
        self.total += 1
        self.last_time = time_value
        for values in self.windows.values():
            values.append(time_value)


@dataclass
class _NumericState:
    count: int = 0
    total: float = 0.0

    @property
    def mean(self) -> float:
        return np.nan if self.count == 0 else self.total / self.count

    def add(self, value: float) -> None:
        if np.isfinite(value):
            self.count += 1
            self.total += value


@dataclass
class _OutcomeState:
    count: int = 0
    positives: int = 0

    def add(self, label: int) -> None:
        self.count += 1
        self.positives += int(label)

    def rate(self, prior_rate: float, smoothing: float) -> float:
        return float((self.positives + smoothing * prior_rate) / (self.count + smoothing))


def load_claim_history(
    path: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    column_map_path: Path | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Claim-history file does not exist: {path}")
    if path.suffix.lower() == ".csv":
        history = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        history = pd.read_parquet(path)
    else:
        raise ValueError("history_path must be a CSV or Parquet file.")
    if column_map_path is not None:
        history = map_claim_history_columns(history, load_history_column_map(column_map_path))
    return validate_claim_history(history, train, test)


def load_history_column_map(path: Path) -> HistoryColumnMap:
    if not path.exists():
        raise FileNotFoundError(f"Claim-history column-map file does not exist: {path}")
    try:
        with path.open() as file:
            payload = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Claim-history column-map must be valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Claim-history column-map must be a JSON object.")
    if payload.get("version") != HISTORY_COLUMN_MAP_VERSION:
        raise ValueError(
            f"Claim-history column-map version must be {HISTORY_COLUMN_MAP_VERSION}."
        )
    raw_columns = payload.get("columns", {})
    if not isinstance(raw_columns, dict):
        raise ValueError("Claim-history column-map columns must be a JSON object.")
    unknown = sorted(set(raw_columns) - set(HISTORY_REQUIRED_COLUMNS))
    if unknown:
        raise ValueError(f"Claim-history column-map has unknown canonical columns: {unknown}")
    columns = {column: column for column in HISTORY_REQUIRED_COLUMNS}
    for canonical, source in raw_columns.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Claim-history column-map source for {canonical} must be non-empty.")
        columns[canonical] = source.strip()
    duplicate_sources = {
        source: [canonical for canonical, mapped in columns.items() if mapped == source]
        for source in set(columns.values())
    }
    duplicates = {source: names for source, names in duplicate_sources.items() if len(names) > 1}
    if duplicates:
        raise ValueError(f"Claim-history column-map maps source columns more than once: {duplicates}")
    return HistoryColumnMap(
        version=HISTORY_COLUMN_MAP_VERSION,
        columns=columns,
        label_values=_history_label_values(payload.get("label_values")),
    )


def map_claim_history_columns(history: pd.DataFrame, column_map: HistoryColumnMap) -> pd.DataFrame:
    if history.columns.duplicated().any():
        raise ValueError("Claim-history source columns must be unique before mapping.")
    missing = sorted(set(column_map.columns.values()) - set(history.columns))
    if missing:
        raise ValueError(
            f"Claim-history column-map references unavailable source columns: {missing}"
        )
    mapped = history.loc[:, [column_map.columns[column] for column in HISTORY_REQUIRED_COLUMNS]].copy()
    mapped.columns = HISTORY_REQUIRED_COLUMNS
    if column_map.label_values is not None:
        mapped["adjudicated_label"] = _normalize_mapped_labels(
            mapped["adjudicated_label"], column_map.label_values
        )
    return mapped


def validate_claim_history(
    history: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    missing = sorted(set(HISTORY_REQUIRED_COLUMNS) - set(history.columns))
    if missing:
        raise ValueError(f"Claim-history file is missing required columns: {missing}")
    for frame_name, frame in (("train", train), ("test", test)):
        missing_context = sorted(set((ID_COL, *HISTORY_CONTEXT_COLUMNS)) - set(frame.columns))
        if missing_context:
            raise ValueError(f"{frame_name} data is missing history-context columns: {missing_context}")

    normalized = history.loc[:, HISTORY_REQUIRED_COLUMNS].copy()
    normalized[ID_COL] = _normalized_ids(normalized[ID_COL], "Claim-history")
    if normalized[ID_COL].duplicated().any():
        raise ValueError("Claim-history claim_id values must be unique.")
    normalized["event_at"] = _timestamps(normalized["event_at"], "event_at", required=True)
    normalized["adjudicated_at"] = _timestamps(
        normalized["adjudicated_at"], "adjudicated_at", required=False
    )
    normalized["adjudicated_label"] = _labels(normalized["adjudicated_label"])
    for column in ("claim_amount", "tariff_amount"):
        normalized[column] = _amounts(normalized[column], column)
    for column in ("provider_id", "patient_id", *HISTORY_CONTEXT_COLUMNS):
        normalized[column] = normalized[column].astype("string")

    labelled = normalized["adjudicated_label"].notna()
    if normalized.loc[labelled, "adjudicated_at"].isna().any():
        raise ValueError("Every adjudicated_label requires adjudicated_at.")
    if normalized.loc[~labelled, "adjudicated_at"].notna().any():
        raise ValueError("adjudicated_at must be empty when adjudicated_label is empty.")
    if (normalized.loc[labelled, "adjudicated_at"] < normalized.loc[labelled, "event_at"]).any():
        raise ValueError("adjudicated_at must not precede event_at.")

    train_ids = _normalized_ids(train[ID_COL], "Training")
    test_ids = _normalized_ids(test[ID_COL], "Test")
    if train_ids.duplicated().any() or test_ids.duplicated().any():
        raise ValueError("Training and test claim_id values must be unique.")
    if set(train_ids).intersection(test_ids):
        raise ValueError("Training and test claim_id values must not overlap.")
    current_ids = pd.Index([*train_ids, *test_ids])
    missing_current = current_ids.difference(pd.Index(normalized[ID_COL]))
    if not missing_current.empty:
        raise ValueError(
            "Claim-history file is missing current competition claim_ids: "
            f"{missing_current[:10].tolist()}"
        )

    indexed = _indexed_by_id(normalized, "Claim-history")
    train_history = indexed.loc[train_ids]
    test_history = indexed.loc[test_ids]
    expected_train_labels = pd.to_numeric(train[TARGET], errors="raise").astype(int).to_numpy()
    history_train_labels = train_history["adjudicated_label"].astype(int).to_numpy()
    if not np.array_equal(expected_train_labels, history_train_labels):
        raise ValueError("Claim-history adjudicated_label values must match training label values.")
    if test_history["adjudicated_label"].notna().any():
        raise ValueError("Current test claims must not have adjudicated_label values.")
    if test_history["adjudicated_at"].notna().any():
        raise ValueError("Current test claims must not have adjudicated_at values.")
    if train_history["adjudicated_at"].isna().any():
        raise ValueError("Current training claims must have adjudicated_at values.")
    if train_history["adjudicated_at"].max() >= test_history["event_at"].min():
        raise ValueError(
            "Every test event_at must occur after every current training adjudicated_at."
        )

    for source_name, source, source_history in (
        ("Training", train, train_history),
        ("Test", test, test_history),
    ):
        for column in HISTORY_CONTEXT_COLUMNS:
            expected = _normalized_context_values(source[column])
            observed = _normalized_context_values(source_history[column])
            if not expected.equals(observed):
                raise ValueError(
                    f"{source_name} {column} values do not match the claim-history file."
                )
    return normalized.reset_index(drop=True)


def build_causal_history_features(
    history: pd.DataFrame,
    smoothing: float = 50.0,
) -> HistoryFeatureBundle:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive.")
    _validate_normalized_history(history)
    ordered = history.copy()
    ordered["history_order"] = np.arange(len(ordered))
    ordered = ordered.sort_values(["event_at", "history_order"], kind="stable")
    labelled = ordered.loc[ordered["adjudicated_label"].notna()].sort_values(
        ["adjudicated_at", "history_order"], kind="stable"
    )
    outcome_rows = list(labelled.itertuples(index=False))
    outcome_index = 0

    rolling_states: dict[str, dict[object, _RollingCount]] = {
        "provider": {},
        "patient": {},
        "provider_patient": {},
        "provider_cmg": {},
        "provider_diagprimer": {},
        "cmg": {},
        "diagprimer": {},
        "peer": {},
    }
    provider_patients: dict[str, set[str]] = defaultdict(set)
    provider_cmgs: dict[str, set[str]] = defaultdict(set)
    provider_diagnoses: dict[str, set[str]] = defaultdict(set)
    peer_claim_amounts: dict[object, _NumericState] = {}
    peer_tariff_amounts: dict[object, _NumericState] = {}
    global_outcomes = _OutcomeState()
    outcome_states: dict[str, dict[object, _OutcomeState]] = {
        "provider": {},
        "patient": {},
        "provider_cmg": {},
        "provider_diagprimer": {},
    }
    feature_rows: list[dict[str, object]] = []

    for _, event_rows in ordered.groupby("event_at", sort=False, dropna=False):
        event_time = event_rows["event_at"].iloc[0]
        event_time_ns = int(event_time.value)
        while outcome_index < len(outcome_rows):
            outcome = outcome_rows[outcome_index]
            outcome_time = getattr(outcome, "adjudicated_at")
            if outcome_time >= event_time:
                break
            _add_outcome(
                outcome,
                global_outcomes,
                outcome_states,
            )
            outcome_index += 1
        for row in event_rows.itertuples(index=False):
            feature_rows.append(
                _feature_row(
                    row,
                    event_time_ns,
                    rolling_states,
                    provider_patients,
                    provider_cmgs,
                    provider_diagnoses,
                    peer_claim_amounts,
                    peer_tariff_amounts,
                    global_outcomes,
                    outcome_states,
                    smoothing,
                )
            )
        for row in event_rows.itertuples(index=False):
            _add_event(
                row,
                event_time_ns,
                rolling_states,
                provider_patients,
                provider_cmgs,
                provider_diagnoses,
                peer_claim_amounts,
                peer_tariff_amounts,
            )

    features = pd.DataFrame(feature_rows)
    features = features.sort_values("history_order", kind="stable").drop(columns="history_order")
    if features[ID_COL].duplicated().any():
        raise RuntimeError("Causal history feature rows must have unique claim_id values.")
    return HistoryFeatureBundle(
        features=features.reset_index(drop=True),
        feature_groups={name: tuple(columns) for name, columns in HISTORY_FEATURE_GROUPS.items()},
    )


def add_history_features(
    prepared: PreparedFeatures,
    train_ids: Iterable[object],
    test_ids: Iterable[object],
    history_features: pd.DataFrame,
    groups: Iterable[str],
) -> PreparedFeatures:
    selected_groups = tuple(groups)
    unknown = sorted(set(selected_groups) - set(HISTORY_FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown history feature groups: {unknown}")
    selected_features = [
        column for group in selected_groups for column in HISTORY_FEATURE_GROUPS[group]
    ]
    if not selected_features:
        return prepared
    required = {ID_COL, *selected_features}
    missing = sorted(required - set(history_features.columns))
    if missing:
        raise ValueError(f"History feature frame is missing required columns: {missing}")
    indexed = _indexed_by_id(history_features, "History feature")
    train_index = pd.Index(_normalized_ids(pd.Series(train_ids), "Training"))
    test_index = pd.Index(_normalized_ids(pd.Series(test_ids), "Test"))
    if not train_index.isin(indexed.index).all() or not test_index.isin(indexed.index).all():
        raise ValueError("History feature frame is missing current competition claim_ids.")
    train_extra = indexed.loc[train_index, selected_features].reset_index(drop=True)
    test_extra = indexed.loc[test_index, selected_features].reset_index(drop=True)
    categorical = [
        feature for feature in HISTORY_CATEGORICAL_FEATURES if feature in selected_features
    ]
    for feature in categorical:
        train_extra[feature] = train_extra[feature].astype("string").fillna("__MISSING__").astype(str)
        test_extra[feature] = test_extra[feature].astype("string").fillna("__MISSING__").astype(str)
    if set(selected_features).intersection(prepared.X.columns):
        raise ValueError("History features collide with existing model features.")
    X = pd.concat([prepared.X.reset_index(drop=True), train_extra], axis=1)
    X_test = pd.concat([prepared.X_test.reset_index(drop=True), test_extra], axis=1)
    if list(X.columns) != list(X_test.columns):
        raise RuntimeError("History-augmented train and test features are not aligned.")
    return PreparedFeatures(
        X=X,
        y=prepared.y.copy(),
        X_test=X_test,
        spec=prepared.spec,
        categorical_features=[*prepared.categorical_features, *categorical],
    )


def history_provider_groups(history_features: pd.DataFrame, claim_ids: Iterable[object]) -> np.ndarray:
    if "history_provider_id" not in history_features:
        raise ValueError("History features do not contain history_provider_id.")
    indexed = _indexed_by_id(history_features, "History feature")
    values = indexed.loc[pd.Index(_normalized_ids(pd.Series(claim_ids), "Competition")), "history_provider_id"]
    return values.astype("string").fillna("__MISSING__").astype(str).to_numpy()


def history_feature_schema() -> dict[str, object]:
    return {
        "required_history_columns": list(HISTORY_REQUIRED_COLUMNS),
        "feature_groups": {name: list(columns) for name, columns in HISTORY_FEATURE_GROUPS.items()},
        "categorical_features": list(HISTORY_CATEGORICAL_FEATURES),
        "outcome_smoothing": 50.0,
    }


def _feature_row(
    row: object,
    event_time_ns: int,
    rolling_states: dict[str, dict[object, _RollingCount]],
    provider_patients: dict[str, set[str]],
    provider_cmgs: dict[str, set[str]],
    provider_diagnoses: dict[str, set[str]],
    peer_claim_amounts: dict[object, _NumericState],
    peer_tariff_amounts: dict[object, _NumericState],
    global_outcomes: _OutcomeState,
    outcome_states: dict[str, dict[object, _OutcomeState]],
    smoothing: float,
) -> dict[str, object]:
    provider = _key(getattr(row, "provider_id"))
    patient = _key(getattr(row, "patient_id"))
    cmg = _key(getattr(row, "cmg"))
    diagnosis = _key(getattr(row, "diagprimer"))
    provider_patient = _composite(provider, patient)
    provider_cmg = _composite(provider, cmg)
    provider_diagnosis = _composite(provider, diagnosis)
    peer = _composite(
        _key(getattr(row, "kdkc")),
        _key(getattr(row, "typeppk")),
        cmg,
        _key(getattr(row, "severitylevel")),
    )
    claim_amount = float(getattr(row, "claim_amount"))
    tariff_amount = float(getattr(row, "tariff_amount"))
    global_rate = global_outcomes.rate(0.5, smoothing)
    peer_claim = peer_claim_amounts.get(peer)
    peer_tariff = peer_tariff_amounts.get(peer)
    peer_mean_claim = np.nan if peer_claim is None else peer_claim.mean
    peer_mean_tariff = np.nan if peer_tariff is None else peer_tariff.mean
    provider_values = _rolling_values(rolling_states["provider"].get(provider), event_time_ns)
    patient_values = _rolling_values(rolling_states["patient"].get(patient), event_time_ns)
    provider_patient_values = _rolling_values(
        rolling_states["provider_patient"].get(provider_patient), event_time_ns
    )
    return {
        ID_COL: getattr(row, ID_COL),
        "history_order": getattr(row, "history_order"),
        "history_claim_amount": claim_amount,
        "history_tariff_amount": tariff_amount,
        "history_amount_minus_tariff": _difference(claim_amount, tariff_amount),
        "history_amount_to_tariff_ratio": _ratio(claim_amount, tariff_amount),
        "history_log_claim_amount": _log_amount(claim_amount),
        "history_log_tariff_amount": _log_amount(tariff_amount),
        "history_claim_amount_missing": int(not np.isfinite(claim_amount)),
        "history_tariff_amount_missing": int(not np.isfinite(tariff_amount)),
        "history_provider_id": provider or "__MISSING__",
        **_rolling_feature_values("history_provider", provider_values),
        **_rolling_feature_values("history_patient", patient_values),
        **_rolling_feature_values("history_provider_patient", provider_patient_values),
        "history_provider_prior_unique_patients": _unique_size(provider_patients, provider),
        "history_provider_prior_unique_cmg": _unique_size(provider_cmgs, provider),
        "history_provider_prior_unique_diagnoses": _unique_size(provider_diagnoses, provider),
        "history_provider_cmg_prior_claim_count": _rolling_total(
            rolling_states["provider_cmg"].get(provider_cmg)
        ),
        "history_provider_diagnosis_prior_claim_count": _rolling_total(
            rolling_states["provider_diagprimer"].get(provider_diagnosis)
        ),
        "history_cmg_prior_claim_count": _rolling_total(rolling_states["cmg"].get(cmg)),
        "history_diagprimer_prior_claim_count": _rolling_total(
            rolling_states["diagprimer"].get(diagnosis)
        ),
        "history_peer_prior_claim_count": _rolling_total(rolling_states["peer"].get(peer)),
        "history_claim_amount_peer_mean_ratio": _ratio(claim_amount, peer_mean_claim),
        "history_tariff_amount_peer_mean_ratio": _ratio(tariff_amount, peer_mean_tariff),
        "history_claim_amount_peer_mean_delta": _difference(claim_amount, peer_mean_claim),
        "history_global_adjudicated_count": global_outcomes.count,
        "history_global_fraud_rate": global_rate,
        **_outcome_feature_values(
            "history_provider", outcome_states["provider"].get(provider), global_rate, smoothing
        ),
        **_outcome_feature_values(
            "history_patient", outcome_states["patient"].get(patient), global_rate, smoothing
        ),
        **_outcome_feature_values(
            "history_provider_cmg",
            outcome_states["provider_cmg"].get(provider_cmg),
            global_rate,
            smoothing,
        ),
        **_outcome_feature_values(
            "history_provider_diagnosis",
            outcome_states["provider_diagprimer"].get(provider_diagnosis),
            global_rate,
            smoothing,
        ),
    }


def _add_event(
    row: object,
    event_time_ns: int,
    rolling_states: dict[str, dict[object, _RollingCount]],
    provider_patients: dict[str, set[str]],
    provider_cmgs: dict[str, set[str]],
    provider_diagnoses: dict[str, set[str]],
    peer_claim_amounts: dict[object, _NumericState],
    peer_tariff_amounts: dict[object, _NumericState],
) -> None:
    provider = _key(getattr(row, "provider_id"))
    patient = _key(getattr(row, "patient_id"))
    cmg = _key(getattr(row, "cmg"))
    diagnosis = _key(getattr(row, "diagprimer"))
    keys = {
        "provider": provider,
        "patient": patient,
        "provider_patient": _composite(provider, patient),
        "provider_cmg": _composite(provider, cmg),
        "provider_diagprimer": _composite(provider, diagnosis),
        "cmg": cmg,
        "diagprimer": diagnosis,
        "peer": _composite(
            _key(getattr(row, "kdkc")),
            _key(getattr(row, "typeppk")),
            cmg,
            _key(getattr(row, "severitylevel")),
        ),
    }
    for state_name, key in keys.items():
        if key is not None:
            _state_for(rolling_states[state_name], key).add(event_time_ns)
    if provider is not None:
        if patient is not None:
            provider_patients[provider].add(patient)
        if cmg is not None:
            provider_cmgs[provider].add(cmg)
        if diagnosis is not None:
            provider_diagnoses[provider].add(diagnosis)
    peer = keys["peer"]
    if peer is not None:
        _numeric_state_for(peer_claim_amounts, peer).add(float(getattr(row, "claim_amount")))
        _numeric_state_for(peer_tariff_amounts, peer).add(float(getattr(row, "tariff_amount")))


def _add_outcome(
    row: object,
    global_outcomes: _OutcomeState,
    outcome_states: dict[str, dict[object, _OutcomeState]],
) -> None:
    label = int(getattr(row, "adjudicated_label"))
    provider = _key(getattr(row, "provider_id"))
    patient = _key(getattr(row, "patient_id"))
    provider_cmg = _composite(provider, _key(getattr(row, "cmg")))
    provider_diagnosis = _composite(provider, _key(getattr(row, "diagprimer")))
    global_outcomes.add(label)
    for state_name, key in (
        ("provider", provider),
        ("patient", patient),
        ("provider_cmg", provider_cmg),
        ("provider_diagprimer", provider_diagnosis),
    ):
        if key is not None:
            _outcome_state_for(outcome_states[state_name], key).add(label)


def _rolling_feature_values(
    prefix: str, values: tuple[int, int, int, int, float]
) -> dict[str, float | int]:
    total, days_30, days_90, days_365, recency = values
    return {
        f"{prefix}_prior_claim_count": total,
        f"{prefix}_prior_claim_count_30d": days_30,
        f"{prefix}_prior_claim_count_90d": days_90,
        f"{prefix}_prior_claim_count_365d": days_365,
        f"{prefix}_days_since_claim": recency,
    }


def _outcome_feature_values(
    prefix: str,
    state: _OutcomeState | None,
    global_rate: float,
    smoothing: float,
) -> dict[str, float | int]:
    current = state or _OutcomeState()
    return {
        f"{prefix}_adjudicated_count": current.count,
        f"{prefix}_fraud_rate": current.rate(global_rate, smoothing),
    }


def _state_for(states: dict[object, _RollingCount], key: object) -> _RollingCount:
    state = states.get(key)
    if state is None:
        state = _RollingCount()
        states[key] = state
    return state


def _numeric_state_for(states: dict[object, _NumericState], key: object) -> _NumericState:
    state = states.get(key)
    if state is None:
        state = _NumericState()
        states[key] = state
    return state


def _outcome_state_for(states: dict[object, _OutcomeState], key: object) -> _OutcomeState:
    state = states.get(key)
    if state is None:
        state = _OutcomeState()
        states[key] = state
    return state


def _rolling_values(state: _RollingCount | None, time_value: int) -> tuple[int, int, int, int, float]:
    return (0, 0, 0, 0, np.nan) if state is None else state.values(time_value)


def _rolling_total(state: _RollingCount | None) -> int:
    return 0 if state is None else state.total


def _unique_size(values: dict[str, set[str]], key: str | None) -> int:
    return 0 if key is None else len(values.get(key, set()))


def _key(value: object) -> str | None:
    if pd.isna(value):
        return None
    rendered = str(value).strip()
    return rendered or None


def _composite(*values: str | None) -> tuple[str, ...] | None:
    if any(value is None for value in values):
        return None
    return tuple(value for value in values if value is not None)


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def _difference(left: float, right: float) -> float:
    return np.nan if not np.isfinite(left) or not np.isfinite(right) else float(left - right)


def _log_amount(value: float) -> float:
    return np.nan if not np.isfinite(value) else float(np.log1p(value))


def _normalized_ids(values: pd.Series, name: str) -> pd.Series:
    normalized = values.astype("string")
    if normalized.isna().any() or normalized.str.strip().eq("").any():
        raise ValueError(f"{name} claim_id values must be non-empty.")
    return normalized.astype(str)


def _timestamps(values: pd.Series, name: str, required: bool) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    invalid = values.notna() & parsed.isna()
    if invalid.any() or (required and parsed.isna().any()):
        raise ValueError(f"{name} values must be parseable timestamps.")
    return parsed


def _labels(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & (numeric.isna() | ~numeric.isin([0, 1]))
    if invalid.any():
        raise ValueError("adjudicated_label must contain only 0, 1, or empty values.")
    return numeric.astype("Int64")


def _history_label_values(values: object) -> dict[str, int] | None:
    if values is None:
        return None
    if not isinstance(values, dict) or not values:
        raise ValueError("Claim-history column-map label_values must be a non-empty JSON object.")
    normalized: dict[str, int] = {}
    for source, target in values.items():
        key = _label_value_key(source)
        if not key:
            raise ValueError("Claim-history column-map label_values keys must be non-empty.")
        try:
            normalized_target = int(target)
        except (TypeError, ValueError) as error:
            raise ValueError("Claim-history column-map label_values must normalize to 0 or 1.") from error
        if normalized_target not in {0, 1}:
            raise ValueError("Claim-history column-map label_values must normalize to 0 or 1.")
        if key in normalized:
            raise ValueError(f"Claim-history column-map repeats label value: {key!r}")
        normalized[key] = normalized_target
    return normalized


def _normalize_mapped_labels(values: pd.Series, label_values: dict[str, int]) -> pd.Series:
    normalized = pd.Series(pd.NA, index=values.index, dtype="Int64")
    observed = values.loc[values.notna()]
    missing = sorted({_label_value_key(value) for value in observed if _label_value_key(value) not in label_values})
    if missing:
        raise ValueError(
            f"Claim-history column-map does not normalize source label values: {missing}"
        )
    normalized.loc[observed.index] = [label_values[_label_value_key(value)] for value in observed]
    return normalized


def _label_value_key(value: object) -> str:
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and float(value).is_integer():
        return str(int(value))
    return str(value).strip()


def _amounts(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & numeric.isna()
    if invalid.any() or (numeric.dropna() < 0).any():
        raise ValueError(f"{name} values must be numeric, non-negative, or empty.")
    return numeric.astype(float)


def _normalized_context_values(values: pd.Series) -> pd.Series:
    return values.astype("string").fillna("__MISSING__").astype(str).reset_index(drop=True)


def _indexed_by_id(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if frame[ID_COL].duplicated().any():
        raise ValueError(f"{name} claim_id values must be unique.")
    return frame.set_index(ID_COL)


def _validate_normalized_history(history: pd.DataFrame) -> None:
    missing = sorted(set(HISTORY_REQUIRED_COLUMNS) - set(history.columns))
    if missing:
        raise ValueError(f"History frame is missing required columns: {missing}")
    if history[ID_COL].duplicated().any():
        raise ValueError("History frame claim_id values must be unique.")
    if history["event_at"].isna().any():
        raise ValueError("History frame event_at values must be complete.")
