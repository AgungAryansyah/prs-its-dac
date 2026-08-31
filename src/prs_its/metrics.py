from __future__ import annotations

from collections.abc import Mapping
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from prs_its.fairness import fairness_across_budgets


PAIRED_AUDIT_METRICS = (
    "fraud_caught",
    "legitimate_audits",
    "recall",
    "normalized_recall",
    "precision",
    "lift",
)


def _validated_arrays(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    probabilities = np.asarray(y_prob, dtype=float).reshape(-1)
    if len(labels) != len(probabilities):
        raise ValueError("y_true and y_prob must have the same length.")
    if len(labels) == 0:
        raise ValueError("y_true and y_prob must not be empty.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain only binary labels.")
    if not np.isfinite(probabilities).all():
        raise ValueError("y_prob must contain only finite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("y_prob must be within [0, 1].")
    return labels, probabilities


def normalized_recall_at_budget(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    audit_fraction: float = 0.05,
) -> float:
    return float(audit_metrics(y_true, y_prob, audit_fraction)["normalized_recall"])


def audit_metrics(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    audit_fraction: float = 0.05,
) -> dict[str, float | int]:
    if not 0 <= audit_fraction <= 1:
        raise ValueError("audit_fraction must be within [0, 1].")

    labels, probabilities = _validated_arrays(y_true, y_prob)
    n_rows = len(labels)
    audit_count = int(np.floor(n_rows * audit_fraction))
    selected = np.argsort(-probabilities, kind="mergesort")[:audit_count]
    total_fraud = int(labels.sum())
    fraud_caught = int(labels[selected].sum())
    prevalence = total_fraud / n_rows
    recall = fraud_caught / total_fraud if total_fraud else 0.0
    precision = fraud_caught / audit_count if audit_count else 0.0
    maximum_capturable = min(audit_count, total_fraud)
    normalized_recall = fraud_caught / maximum_capturable if maximum_capturable else 0.0
    lift = precision / prevalence if prevalence else 0.0

    return {
        "audit_fraction": float(audit_fraction),
        "n_rows": n_rows,
        "n_audited": audit_count,
        "total_fraud": total_fraud,
        "fraud_caught": fraud_caught,
        "legitimate_audits": audit_count - fraud_caught,
        "prevalence": float(prevalence),
        "recall": float(recall),
        "normalized_recall": float(normalized_recall),
        "precision": float(precision),
        "lift": float(lift),
    }


def bootstrap_audit_intervals(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> list[dict[str, float | int | str]]:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be within (0, 1).")

    labels, probabilities = _validated_arrays(y_true, y_prob)
    metrics_by_fraction = {
        fraction: audit_metrics(labels, probabilities, fraction) for fraction in audit_fractions
    }
    metric_names = (
        "fraud_caught",
        "legitimate_audits",
        "recall",
        "normalized_recall",
        "precision",
        "lift",
    )
    samples = {
        fraction: {name: np.empty(n_bootstrap, dtype=float) for name in metric_names}
        for fraction in audit_fractions
    }
    random = np.random.default_rng(random_state)
    for iteration in range(n_bootstrap):
        sampled_indices = random.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[sampled_indices]
        sorted_labels = sampled_labels[np.argsort(-probabilities[sampled_indices], kind="mergesort")]
        cumulative_fraud = np.cumsum(sorted_labels)
        total_fraud = int(sampled_labels.sum())
        prevalence = total_fraud / len(sampled_labels)
        for fraction in audit_fractions:
            audit_count = int(np.floor(len(sampled_labels) * fraction))
            fraud_caught = int(cumulative_fraud[audit_count - 1]) if audit_count else 0
            precision = fraud_caught / audit_count if audit_count else 0.0
            maximum_capturable = min(audit_count, total_fraud)
            samples[fraction]["fraud_caught"][iteration] = fraud_caught
            samples[fraction]["legitimate_audits"][iteration] = audit_count - fraud_caught
            samples[fraction]["recall"][iteration] = fraud_caught / total_fraud if total_fraud else 0.0
            samples[fraction]["normalized_recall"][iteration] = (
                fraud_caught / maximum_capturable if maximum_capturable else 0.0
            )
            samples[fraction]["precision"][iteration] = precision
            samples[fraction]["lift"][iteration] = precision / prevalence if prevalence else 0.0

    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
    return [
        {
            "audit_fraction": fraction,
            "metric": metric_name,
            "estimate": float(metrics_by_fraction[fraction][metric_name]),
            "ci_lower": float(np.quantile(samples[fraction][metric_name], lower_quantile)),
            "ci_upper": float(np.quantile(samples[fraction][metric_name], upper_quantile)),
            "n_bootstrap": n_bootstrap,
            "confidence_level": confidence_level,
        }
        for fraction in audit_fractions
        for metric_name in metric_names
    ]


def evaluate_probabilities(
    y_true: Iterable[int] | np.ndarray,
    y_prob: Iterable[float] | np.ndarray,
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
) -> dict[str, float | int]:
    labels, probabilities = _validated_arrays(y_true, y_prob)
    metrics: dict[str, float | int] = {
        "n_rows": len(labels),
        "fraud_prevalence": float(labels.mean()),
        "mean_prediction": float(probabilities.mean()),
        "brier_score": float(brier_score_loss(labels, probabilities)),
    }
    if labels.sum() in {0, len(labels)}:
        metrics["average_precision"] = float(labels.mean())
        metrics["roc_auc"] = float("nan")
        metrics["log_loss"] = float("nan")
    else:
        metrics["average_precision"] = float(average_precision_score(labels, probabilities))
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
        metrics["log_loss"] = float(log_loss(labels, probabilities, labels=[0, 1]))

    for fraction in audit_fractions:
        suffix = f"{fraction:.0%}".replace("%", "pct")
        for name, value in audit_metrics(labels, probabilities, fraction).items():
            if name not in {"audit_fraction", "n_rows", "total_fraud", "prevalence"}:
                metrics[f"{name}_at_{suffix}"] = value
    return metrics


def validate_paired_oof(
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    candidate_probability_column: str = "fraud_probability_raw",
    incumbent_probability_column: str = "fraud_probability_raw",
    id_column: str = "claim_id",
    target_column: str = "label",
    fold_column: str = "fold",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_labels, candidate_probabilities, candidate_folds = _validated_oof_frame(
        candidate,
        probability_column=candidate_probability_column,
        id_column=id_column,
        target_column=target_column,
        fold_column=fold_column,
    )
    incumbent_labels, incumbent_probabilities, incumbent_folds = _validated_oof_frame(
        incumbent,
        probability_column=incumbent_probability_column,
        id_column=id_column,
        target_column=target_column,
        fold_column=fold_column,
    )
    if not candidate[id_column].reset_index(drop=True).equals(
        incumbent[id_column].reset_index(drop=True)
    ):
        raise ValueError("Candidate and incumbent claim_id order must be identical.")
    if not np.array_equal(candidate_labels, incumbent_labels):
        raise ValueError("Candidate and incumbent label order must be identical.")
    if not np.array_equal(candidate_folds, incumbent_folds):
        raise ValueError("Candidate and incumbent fold assignments must be identical.")
    return candidate_labels, candidate_probabilities, incumbent_probabilities, candidate_folds


def paired_bootstrap_comparison(
    y_true: Iterable[int] | np.ndarray,
    candidate_probabilities: Iterable[float] | np.ndarray,
    incumbent_probabilities: Iterable[float] | np.ndarray,
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be within (0, 1).")
    if not audit_fractions or any(not 0 <= fraction <= 1 for fraction in audit_fractions):
        raise ValueError("audit_fractions must contain values within [0, 1].")

    labels, candidate_scores = _validated_arrays(y_true, candidate_probabilities)
    incumbent_labels, incumbent_scores = _validated_arrays(y_true, incumbent_probabilities)
    if not np.array_equal(labels, incumbent_labels):
        raise ValueError("Candidate and incumbent labels must be identical.")

    point_estimates = _paired_metric_rows(
        labels, candidate_scores, incumbent_scores, audit_fractions
    )
    bootstrap_deltas = np.empty((n_bootstrap, len(point_estimates)), dtype=float)
    random = np.random.default_rng(random_state)
    for iteration in range(n_bootstrap):
        sampled_indices = random.integers(0, len(labels), size=len(labels))
        sampled_rows = _paired_metric_rows(
            labels[sampled_indices],
            candidate_scores[sampled_indices],
            incumbent_scores[sampled_indices],
            audit_fractions,
        )
        bootstrap_deltas[iteration] = [row["delta"] for row in sampled_rows]

    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
    comparison = pd.DataFrame(point_estimates)
    comparison["ci_lower"] = np.quantile(bootstrap_deltas, lower_quantile, axis=0)
    comparison["ci_upper"] = np.quantile(bootstrap_deltas, upper_quantile, axis=0)
    comparison["n_bootstrap"] = n_bootstrap
    comparison["confidence_level"] = confidence_level
    comparison["comparison"] = "candidate_minus_incumbent"
    return comparison


def paired_oof_comparison(
    candidate: pd.DataFrame,
    incumbent: pd.DataFrame,
    candidate_probability_column: str = "fraud_probability_raw",
    incumbent_probability_column: str = "fraud_probability_raw",
    id_column: str = "claim_id",
    target_column: str = "label",
    fold_column: str = "fold",
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    labels, candidate_scores, incumbent_scores, _ = validate_paired_oof(
        candidate,
        incumbent,
        candidate_probability_column=candidate_probability_column,
        incumbent_probability_column=incumbent_probability_column,
        id_column=id_column,
        target_column=target_column,
        fold_column=fold_column,
    )
    return paired_bootstrap_comparison(
        labels,
        candidate_scores,
        incumbent_scores,
        audit_fractions=audit_fractions,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )


def fairness_rate_deltas(
    groups: pd.Series,
    y_true: Iterable[int] | np.ndarray,
    candidate_probabilities: Iterable[float] | np.ndarray,
    incumbent_probabilities: Iterable[float] | np.ndarray,
    group_name: str,
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
    min_legitimate_count: int = 100,
) -> pd.DataFrame:
    labels, candidate_scores = _validated_arrays(y_true, candidate_probabilities)
    _, incumbent_scores = _validated_arrays(y_true, incumbent_probabilities)
    group_series = _validated_groups(groups, len(labels), group_name)
    candidate_rates = fairness_across_budgets(
        group_series,
        labels,
        candidate_scores,
        group_name=group_name,
        audit_fractions=audit_fractions,
        min_legitimate_count=min_legitimate_count,
    )
    incumbent_rates = fairness_across_budgets(
        group_series,
        labels,
        incumbent_scores,
        group_name=group_name,
        audit_fractions=audit_fractions,
        min_legitimate_count=min_legitimate_count,
    )
    keys = ["group_variable", "group", "audit_fraction"]
    candidate_columns = [
        *keys,
        "nonfraud_count",
        "audited_nonfraud_count",
        "audit_rate",
        "eligible_for_comparison",
    ]
    candidate_rates = candidate_rates.loc[:, candidate_columns].rename(
        columns={
            "nonfraud_count": "candidate_nonfraud_count",
            "audited_nonfraud_count": "candidate_audited_nonfraud_count",
            "audit_rate": "candidate_audit_rate",
            "eligible_for_comparison": "candidate_eligible_for_comparison",
        }
    )
    incumbent_rates = incumbent_rates.loc[:, candidate_columns].rename(
        columns={
            "nonfraud_count": "incumbent_nonfraud_count",
            "audited_nonfraud_count": "incumbent_audited_nonfraud_count",
            "audit_rate": "incumbent_audit_rate",
            "eligible_for_comparison": "incumbent_eligible_for_comparison",
        }
    )
    comparison = candidate_rates.merge(
        incumbent_rates,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        raise RuntimeError("Candidate and incumbent fairness subgroup schemas differ.")
    comparison = comparison.drop(columns="_merge")
    if not comparison["candidate_nonfraud_count"].equals(
        comparison["incumbent_nonfraud_count"]
    ):
        raise RuntimeError("Candidate and incumbent legitimate subgroup counts differ.")
    comparison["audit_rate_delta"] = (
        comparison["candidate_audit_rate"] - comparison["incumbent_audit_rate"]
    )
    comparison["comparison"] = "candidate_minus_incumbent"
    return comparison.sort_values(keys).reset_index(drop=True)


def paired_fairness_gap_intervals(
    y_true: Iterable[int] | np.ndarray,
    candidate_probabilities: Iterable[float] | np.ndarray,
    incumbent_probabilities: Iterable[float] | np.ndarray,
    group_variables: Mapping[str, pd.Series],
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
    min_legitimate_count: int = 100,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be within (0, 1).")
    if not group_variables:
        raise ValueError("group_variables must not be empty.")

    labels, candidate_scores = _validated_arrays(y_true, candidate_probabilities)
    _, incumbent_scores = _validated_arrays(y_true, incumbent_probabilities)
    validated_groups = {
        group_name: _validated_groups(groups, len(labels), group_name)
        for group_name, groups in group_variables.items()
    }
    point_estimates = _fairness_gap_rows(
        labels,
        candidate_scores,
        incumbent_scores,
        validated_groups,
        audit_fractions,
        min_legitimate_count,
    )
    bootstrap_deltas = np.full((n_bootstrap, len(point_estimates)), np.nan, dtype=float)
    random = np.random.default_rng(random_state)
    for iteration in range(n_bootstrap):
        sampled_indices = random.integers(0, len(labels), size=len(labels))
        sampled_groups = {
            name: groups.iloc[sampled_indices].reset_index(drop=True)
            for name, groups in validated_groups.items()
        }
        sampled_rows = _fairness_gap_rows(
            labels[sampled_indices],
            candidate_scores[sampled_indices],
            incumbent_scores[sampled_indices],
            sampled_groups,
            audit_fractions,
            min_legitimate_count,
        )
        bootstrap_deltas[iteration] = [row["delta"] for row in sampled_rows]

    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
    comparison = pd.DataFrame(point_estimates)
    lower = []
    upper = []
    for column in bootstrap_deltas.T:
        finite = column[np.isfinite(column)]
        lower.append(float(np.quantile(finite, lower_quantile)) if len(finite) else np.nan)
        upper.append(float(np.quantile(finite, upper_quantile)) if len(finite) else np.nan)
    comparison["ci_lower"] = lower
    comparison["ci_upper"] = upper
    comparison["n_bootstrap"] = n_bootstrap
    comparison["confidence_level"] = confidence_level
    comparison["comparison"] = "candidate_minus_incumbent"
    return comparison


def paired_fairness_comparison(
    y_true: Iterable[int] | np.ndarray,
    candidate_probabilities: Iterable[float] | np.ndarray,
    incumbent_probabilities: Iterable[float] | np.ndarray,
    gender_groups: pd.Series,
    age_group_values: pd.Series,
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
    min_legitimate_count: int = 100,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    group_variables = {"gender": gender_groups, "age_group": age_group_values}
    rate_deltas = pd.concat(
        [
            fairness_rate_deltas(
                groups,
                y_true,
                candidate_probabilities,
                incumbent_probabilities,
                group_name=group_name,
                audit_fractions=audit_fractions,
                min_legitimate_count=min_legitimate_count,
            )
            for group_name, groups in group_variables.items()
        ],
        ignore_index=True,
    )
    return {
        "subgroup_rate_deltas": rate_deltas,
        "gap_intervals": paired_fairness_gap_intervals(
            y_true,
            candidate_probabilities,
            incumbent_probabilities,
            group_variables=group_variables,
            audit_fractions=audit_fractions,
            min_legitimate_count=min_legitimate_count,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        ),
    }


def _validated_oof_frame(
    frame: pd.DataFrame,
    probability_column: str,
    id_column: str,
    target_column: str,
    fold_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required_columns = {id_column, target_column, fold_column, probability_column}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"OOF frame is missing required columns: {missing_columns}")
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        raise ValueError("OOF claim_id values must be present and unique.")
    labels, probabilities = _validated_arrays(frame[target_column], frame[probability_column])
    folds = pd.to_numeric(frame[fold_column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(folds).all() or not np.equal(folds, np.floor(folds)).all():
        raise ValueError("OOF fold assignments must be finite integers.")
    return labels, probabilities, folds.astype(int)


def _paired_metric_rows(
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    incumbent_scores: np.ndarray,
    audit_fractions: tuple[float, ...],
) -> list[dict[str, float | str]]:
    candidate_metrics = evaluate_probabilities(labels, candidate_scores, audit_fractions)
    incumbent_metrics = evaluate_probabilities(labels, incumbent_scores, audit_fractions)
    rows: list[dict[str, float | str]] = []
    for metric in ("average_precision", "brier_score"):
        candidate_value = float(candidate_metrics[metric])
        incumbent_value = float(incumbent_metrics[metric])
        rows.append(
            {
                "metric": metric,
                "audit_fraction": np.nan,
                "candidate_value": candidate_value,
                "incumbent_value": incumbent_value,
                "delta": candidate_value - incumbent_value,
            }
        )
    for audit_fraction in audit_fractions:
        candidate_audit = audit_metrics(labels, candidate_scores, audit_fraction)
        incumbent_audit = audit_metrics(labels, incumbent_scores, audit_fraction)
        for metric in PAIRED_AUDIT_METRICS:
            candidate_value = float(candidate_audit[metric])
            incumbent_value = float(incumbent_audit[metric])
            rows.append(
                {
                    "metric": metric,
                    "audit_fraction": audit_fraction,
                    "candidate_value": candidate_value,
                    "incumbent_value": incumbent_value,
                    "delta": candidate_value - incumbent_value,
                }
            )
    return rows


def _validated_groups(groups: pd.Series, expected_length: int, group_name: str) -> pd.Series:
    group_series = pd.Series(groups).reset_index(drop=True)
    if len(group_series) != expected_length:
        raise ValueError(f"{group_name} groups must have the same length as predictions.")
    return group_series.rename(group_name)


def _fairness_gap_rows(
    labels: np.ndarray,
    candidate_scores: np.ndarray,
    incumbent_scores: np.ndarray,
    group_variables: Mapping[str, pd.Series],
    audit_fractions: tuple[float, ...],
    min_legitimate_count: int,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for group_name, groups in group_variables.items():
        candidate_rates = fairness_across_budgets(
            groups,
            labels,
            candidate_scores,
            group_name=group_name,
            audit_fractions=audit_fractions,
            min_legitimate_count=min_legitimate_count,
        )
        incumbent_rates = fairness_across_budgets(
            groups,
            labels,
            incumbent_scores,
            group_name=group_name,
            audit_fractions=audit_fractions,
            min_legitimate_count=min_legitimate_count,
        )
        for audit_fraction in audit_fractions:
            candidate_gap = _fairness_gap(
                candidate_rates.loc[candidate_rates["audit_fraction"].eq(audit_fraction)]
            )
            incumbent_gap = _fairness_gap(
                incumbent_rates.loc[incumbent_rates["audit_fraction"].eq(audit_fraction)]
            )
            rows.append(
                {
                    "group_variable": group_name,
                    "audit_fraction": audit_fraction,
                    "candidate_gap": candidate_gap,
                    "incumbent_gap": incumbent_gap,
                    "delta": candidate_gap - incumbent_gap,
                }
            )
    return rows


def _fairness_gap(rates: pd.DataFrame) -> float:
    eligible = rates.loc[rates["eligible_for_comparison"], "audit_rate"]
    if eligible.empty:
        return float("nan")
    return float(eligible.max() - eligible.min())
