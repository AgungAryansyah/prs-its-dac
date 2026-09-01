from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from prs_its.fairness import fairness_across_budgets
from prs_its.metrics import evaluate_probabilities
from prs_its.training import _select_experiment


BLEND_WEIGHTS = (0.0, 0.02, 0.05)
AUDIT_FRACTIONS = (0.03, 0.05, 0.07)
PRIMARY_AUDIT_FRACTION = 0.05
AVERAGE_PRECISION_TOLERANCE = 0.005


@dataclass(frozen=True)
class ScreenDecision:
    selected_name: str
    selected_weight: float
    eligible: bool
    normalized_recall_noninferior: bool
    average_precision_noninferior: bool
    brier_noninferior: bool

    def as_dict(self) -> dict[str, bool | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    normalized_recall_positive: bool
    average_precision_noninferior: bool
    brier_noninferior: bool
    gender_fairness_not_regressed: bool
    age_fairness_not_regressed: bool

    def as_dict(self) -> dict[str, bool]:
        return asdict(self)


def blend_experiment_name(xgb_weight: float) -> str:
    _validate_weight(xgb_weight)
    percentage = round(xgb_weight * 100)
    if not np.isclose(xgb_weight, percentage / 100):
        raise ValueError("xgb_weight must be representable to two decimal places.")
    return f"ctr_xgb_raw_w{percentage:02d}"


def blend_probabilities(
    ctr_probabilities: Iterable[float] | np.ndarray,
    xgb_probabilities: Iterable[float] | np.ndarray,
    xgb_weight: float,
) -> np.ndarray:
    _validate_weight(xgb_weight)
    ctr = _validated_probabilities(ctr_probabilities, "ctr_probabilities")
    xgb = _validated_probabilities(xgb_probabilities, "xgb_probabilities")
    if len(ctr) != len(xgb):
        raise ValueError("ctr_probabilities and xgb_probabilities must have the same length.")
    return (1 - xgb_weight) * ctr + xgb_weight * xgb


def blend_experiment_results(
    y_true: Iterable[int] | np.ndarray,
    ctr_probabilities: Iterable[float] | np.ndarray,
    xgb_probabilities: Iterable[float] | np.ndarray,
    fold_ids: Iterable[int] | np.ndarray,
    gender_groups: pd.Series,
    age_group_values: pd.Series,
    weights: tuple[float, ...] = BLEND_WEIGHTS,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must contain at least one binary label.")
    folds = np.asarray(fold_ids, dtype=int).reshape(-1)
    if len(folds) != len(labels):
        raise ValueError("fold_ids must have the same length as y_true.")
    if len(gender_groups) != len(labels) or len(age_group_values) != len(labels):
        raise ValueError("Fairness groups must have the same length as y_true.")

    rows: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    for weight in weights:
        name = blend_experiment_name(weight)
        probabilities = blend_probabilities(ctr_probabilities, xgb_probabilities, weight)
        predictions[name] = probabilities
        metrics = evaluate_probabilities(labels, probabilities, AUDIT_FRACTIONS)
        fold_normalized_recall = [
            evaluate_probabilities(labels[folds == fold], probabilities[folds == fold], AUDIT_FRACTIONS)[
                "normalized_recall_at_5pct"
            ]
            for fold in np.unique(folds)
        ]
        rows.append(
            {
                "experiment_name": name,
                "xgb_weight": weight,
                **metrics,
                "fold_normalized_recall_5_std": float(pd.Series(fold_normalized_recall).std()),
                "fairness_audit_rate_gap_5": _fairness_gap(
                    labels, probabilities, gender_groups, age_group_values
                ),
                "mean_best_iteration": 0.0,
            }
        )
    return pd.DataFrame(rows), predictions


def select_screen_candidate(results: pd.DataFrame) -> ScreenDecision:
    required = {
        "experiment_name",
        "xgb_weight",
        "normalized_recall_at_5pct",
        "average_precision",
        "brier_score",
        "fold_normalized_recall_5_std",
        "fairness_audit_rate_gap_5",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"Blend screen results are missing required columns: {missing}")
    control = results.loc[np.isclose(results["xgb_weight"], 0.0)]
    candidates = results.loc[~np.isclose(results["xgb_weight"], 0.0)]
    if len(control) != 1:
        raise ValueError("Blend screen results must contain exactly one zero-weight control.")
    if candidates.empty:
        raise ValueError("Blend screen results must contain at least one nonzero candidate.")
    selected_name = _select_experiment(candidates.assign(mean_best_iteration=0.0))
    selected = candidates.loc[candidates["experiment_name"].eq(selected_name)]
    if len(selected) != 1:
        raise RuntimeError("Selected blend candidate is not uniquely identifiable.")
    selected_row = selected.iloc[0]
    control_row = control.iloc[0]
    normalized_recall_noninferior = bool(
        selected_row["normalized_recall_at_5pct"]
        >= control_row["normalized_recall_at_5pct"]
    )
    average_precision_noninferior = bool(
        selected_row["average_precision"]
        >= control_row["average_precision"] - AVERAGE_PRECISION_TOLERANCE
    )
    brier_noninferior = bool(selected_row["brier_score"] <= control_row["brier_score"])
    return ScreenDecision(
        selected_name=selected_name,
        selected_weight=float(selected_row["xgb_weight"]),
        eligible=(
            normalized_recall_noninferior
            and average_precision_noninferior
            and brier_noninferior
        ),
        normalized_recall_noninferior=normalized_recall_noninferior,
        average_precision_noninferior=average_precision_noninferior,
        brier_noninferior=brier_noninferior,
    )


def promotion_decision(
    paired_comparison: pd.DataFrame,
    fairness_gaps: pd.DataFrame,
) -> PromotionDecision:
    normalized_recall = _paired_metric_row(
        paired_comparison, "normalized_recall", PRIMARY_AUDIT_FRACTION
    )
    average_precision = _paired_metric_row(paired_comparison, "average_precision", None)
    brier = _paired_metric_row(paired_comparison, "brier_score", None)
    gender_fairness = _fairness_gap_row(fairness_gaps, "gender")
    age_fairness = _fairness_gap_row(fairness_gaps, "age_group")
    normalized_recall_positive = bool(normalized_recall["ci_lower"] > 0)
    average_precision_noninferior = bool(
        average_precision["delta"] >= -AVERAGE_PRECISION_TOLERANCE
    )
    brier_noninferior = bool(brier["delta"] <= 0)
    gender_fairness_not_regressed = _fairness_not_regressed(gender_fairness)
    age_fairness_not_regressed = _fairness_not_regressed(age_fairness)
    return PromotionDecision(
        promoted=(
            normalized_recall_positive
            and average_precision_noninferior
            and brier_noninferior
            and gender_fairness_not_regressed
            and age_fairness_not_regressed
        ),
        normalized_recall_positive=normalized_recall_positive,
        average_precision_noninferior=average_precision_noninferior,
        brier_noninferior=brier_noninferior,
        gender_fairness_not_regressed=gender_fairness_not_regressed,
        age_fairness_not_regressed=age_fairness_not_regressed,
    )


def _fairness_gap(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    gender_groups: pd.Series,
    age_group_values: pd.Series,
) -> float:
    rates = pd.concat(
        [
            fairness_across_budgets(
                gender_groups, y_true, probabilities, group_name="gender", audit_fractions=AUDIT_FRACTIONS
            ),
            fairness_across_budgets(
                age_group_values,
                y_true,
                probabilities,
                group_name="age_group",
                audit_fractions=AUDIT_FRACTIONS,
            ),
        ],
        ignore_index=True,
    )
    eligible = rates.loc[
        rates["audit_fraction"].eq(PRIMARY_AUDIT_FRACTION)
        & rates["eligible_for_comparison"]
    ]
    if eligible.empty:
        return float("nan")
    return float(eligible["audit_rate"].max() - eligible["audit_rate"].min())


def _paired_metric_row(
    comparison: pd.DataFrame,
    metric: str,
    audit_fraction: float | None,
) -> pd.Series:
    required = {"metric", "audit_fraction", "delta", "ci_lower"}
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"Paired comparison is missing required columns: {missing}")
    rows = comparison.loc[comparison["metric"].eq(metric)]
    if audit_fraction is None:
        rows = rows.loc[rows["audit_fraction"].isna()]
    else:
        rows = rows.loc[np.isclose(rows["audit_fraction"], audit_fraction, equal_nan=False)]
    if len(rows) != 1:
        raise ValueError(f"Paired comparison must contain one {metric!r} row.")
    row = rows.iloc[0]
    if not np.isfinite(float(row["delta"])) or not np.isfinite(float(row["ci_lower"])):
        raise ValueError(f"Paired comparison contains invalid {metric!r} values.")
    return row


def _fairness_gap_row(comparison: pd.DataFrame, group_variable: str) -> pd.Series:
    required = {"group_variable", "audit_fraction", "ci_lower"}
    missing = sorted(required - set(comparison.columns))
    if missing:
        raise ValueError(f"Fairness comparison is missing required columns: {missing}")
    rows = comparison.loc[
        comparison["group_variable"].eq(group_variable)
        & np.isclose(
            comparison["audit_fraction"], PRIMARY_AUDIT_FRACTION, equal_nan=False
        )
    ]
    if len(rows) != 1:
        raise ValueError(f"Fairness comparison must contain one {group_variable!r} gap row.")
    row = rows.iloc[0]
    return row


def _fairness_not_regressed(row: pd.Series) -> bool:
    ci_lower = float(row["ci_lower"])
    return bool(np.isfinite(ci_lower) and ci_lower <= 0)


def _validated_probabilities(
    probabilities: Iterable[float] | np.ndarray, name: str
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(values) == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError(f"{name} must contain finite probabilities within [0, 1].")
    return values


def _validate_weight(xgb_weight: float) -> None:
    if not np.isfinite(xgb_weight) or not 0 <= xgb_weight <= 1:
        raise ValueError("xgb_weight must be within [0, 1].")
