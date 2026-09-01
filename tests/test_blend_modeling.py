from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prs_its.blend_modeling import (
    blend_experiment_name,
    blend_probabilities,
    promotion_decision,
    select_screen_candidate,
)


def test_blend_probabilities_uses_the_configured_xgb_weight() -> None:
    probabilities = blend_probabilities([0.2, 0.8], [0.6, 0.4], 0.05)

    np.testing.assert_allclose(probabilities, [0.22, 0.78])
    assert blend_experiment_name(0.02) == "ctr_xgb_raw_w02"


def test_blend_probabilities_rejects_incompatible_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        blend_probabilities([0.2], [0.6, 0.4], 0.02)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        blend_probabilities([0.2], [0.6], 1.1)


def test_screen_selects_only_nonzero_candidates_and_checks_control() -> None:
    results = pd.DataFrame(
        [
            {
                "experiment_name": "ctr_xgb_raw_w00",
                "xgb_weight": 0.0,
                "normalized_recall_at_5pct": 0.94,
                "average_precision": 0.90,
                "brier_score": 0.14,
                "fold_normalized_recall_5_std": 0.03,
                "fairness_audit_rate_gap_5": 0.02,
            },
            {
                "experiment_name": "ctr_xgb_raw_w02",
                "xgb_weight": 0.02,
                "normalized_recall_at_5pct": 0.95,
                "average_precision": 0.896,
                "brier_score": 0.13,
                "fold_normalized_recall_5_std": 0.02,
                "fairness_audit_rate_gap_5": 0.02,
            },
            {
                "experiment_name": "ctr_xgb_raw_w05",
                "xgb_weight": 0.05,
                "normalized_recall_at_5pct": 0.952,
                "average_precision": 0.88,
                "brier_score": 0.12,
                "fold_normalized_recall_5_std": 0.01,
                "fairness_audit_rate_gap_5": 0.01,
            },
        ]
    )

    decision = select_screen_candidate(results)

    assert decision.selected_name == "ctr_xgb_raw_w02"
    assert decision.selected_weight == 0.02
    assert decision.eligible

    results.loc[results["experiment_name"].eq("ctr_xgb_raw_w02"), "brier_score"] = 0.15
    results.loc[results["experiment_name"].eq("ctr_xgb_raw_w05"), "brier_score"] = 0.16
    rejected = select_screen_candidate(results)
    assert not rejected.eligible
    assert not rejected.brier_noninferior


def test_promotion_requires_each_predeclared_gate() -> None:
    paired = pd.DataFrame(
        [
            {"metric": "normalized_recall", "audit_fraction": 0.05, "delta": 0.01, "ci_lower": 0.002},
            {"metric": "average_precision", "audit_fraction": np.nan, "delta": -0.004, "ci_lower": -0.006},
            {"metric": "brier_score", "audit_fraction": np.nan, "delta": -0.001, "ci_lower": -0.002},
        ]
    )
    fairness = pd.DataFrame(
        [
            {"group_variable": "gender", "audit_fraction": 0.05, "ci_lower": -0.001},
            {"group_variable": "age_group", "audit_fraction": 0.05, "ci_lower": 0.0},
        ]
    )

    assert promotion_decision(paired, fairness).promoted

    fairness.loc[fairness["group_variable"].eq("age_group"), "ci_lower"] = 0.0001
    rejected = promotion_decision(paired, fairness)
    assert not rejected.promoted
    assert not rejected.age_fairness_not_regressed

    fairness.loc[fairness["group_variable"].eq("age_group"), "ci_lower"] = np.nan
    undefined = promotion_decision(paired, fairness)
    assert not undefined.promoted
    assert not undefined.age_fairness_not_regressed
