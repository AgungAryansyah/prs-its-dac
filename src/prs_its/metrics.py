from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


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
