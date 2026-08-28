import numpy as np
import pytest

from prs_its.metrics import audit_metrics, evaluate_probabilities, normalized_recall_at_budget


def test_perfect_ranking_captures_all_fraud_at_budget() -> None:
    metrics = audit_metrics([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], audit_fraction=0.5)

    assert metrics["fraud_caught"] == 2
    assert metrics["recall"] == 1.0
    assert metrics["normalized_recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["lift"] == 2.0


def test_zero_fraud_capture_is_reported_safely() -> None:
    metrics = audit_metrics([1, 0, 0, 0], [0.1, 0.9, 0.8, 0.7], audit_fraction=0.25)

    assert metrics["fraud_caught"] == 0
    assert metrics["recall"] == 0.0
    assert metrics["normalized_recall"] == 0.0
    assert metrics["legitimate_audits"] == 1


def test_normalized_recall_handles_fewer_frauds_than_audit_slots() -> None:
    score = normalized_recall_at_budget([1, 0, 0, 0], [0.9, 0.8, 0.2, 0.1], 0.5)

    assert score == 1.0


def test_normalized_recall_handles_more_frauds_than_audit_slots() -> None:
    metrics = audit_metrics([1, 1, 1, 0], [0.9, 0.8, 0.7, 0.1], audit_fraction=0.5)

    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["normalized_recall"] == 1.0


def test_probability_evaluation_reports_required_budget_metrics() -> None:
    metrics = evaluate_probabilities([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    assert {"average_precision", "brier_score", "recall_at_5pct", "lift_at_7pct"} <= set(metrics)


def test_probability_validation_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="within"):
        audit_metrics([0, 1], [0.1, 1.1])

    with pytest.raises(ValueError, match="same length"):
        audit_metrics([0, 1], np.array([0.1]))
