from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


AGE_BINS = [-np.inf, 17, 39, 59, np.inf]
AGE_LABELS = ["0-17", "18-39", "40-59", "60+"]


def age_groups(age: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(age, errors="coerce"), bins=AGE_BINS, labels=AGE_LABELS).astype("string")


def audit_portfolio(
    probabilities: Iterable[float] | np.ndarray, audit_fraction: float = 0.05
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if not 0 <= audit_fraction <= 1:
        raise ValueError("audit_fraction must be within [0, 1].")
    if not np.isfinite(scores).all() or not ((0 <= scores) & (scores <= 1)).all():
        raise ValueError("Probabilities must be finite values within [0, 1].")
    selected = np.zeros(len(scores), dtype=bool)
    count = int(np.floor(len(scores) * audit_fraction))
    selected[np.argsort(-scores, kind="mergesort")[:count]] = True
    return selected


def fairness_audit_rates(
    groups: pd.Series,
    y_true: Iterable[int] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    audit_fraction: float = 0.05,
    group_name: str | None = None,
    min_legitimate_count: int = 100,
) -> pd.DataFrame:
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(probabilities, dtype=float).reshape(-1)
    if not (len(groups) == len(labels) == len(scores)):
        raise ValueError("Groups, labels, and probabilities must have matching lengths.")
    group_series = groups.astype("string").fillna("<MISSING>")
    portfolio = audit_portfolio(scores, audit_fraction)
    legitimate = pd.DataFrame(
        {
            "group": group_series,
            "is_audited": portfolio,
            "is_legitimate": labels == 0,
        }
    ).loc[lambda frame: frame["is_legitimate"]]
    rates = (
        legitimate.groupby("group", dropna=False, observed=True)["is_audited"]
        .agg(nonfraud_count="size", audited_nonfraud_count="sum")
        .reset_index()
    )
    rates["audit_rate"] = rates["audited_nonfraud_count"] / rates["nonfraud_count"]
    rates["eligible_for_comparison"] = rates["nonfraud_count"] >= min_legitimate_count
    rates["audit_fraction"] = audit_fraction
    rates["group_variable"] = group_name or groups.name or "group"
    return rates.sort_values("audit_rate", ascending=False).reset_index(drop=True)


def fairness_across_budgets(
    groups: pd.Series,
    y_true: Iterable[int] | np.ndarray,
    probabilities: Iterable[float] | np.ndarray,
    group_name: str | None = None,
    audit_fractions: tuple[float, ...] = (0.03, 0.05, 0.07),
    min_legitimate_count: int = 100,
) -> pd.DataFrame:
    return pd.concat(
        [
            fairness_audit_rates(
                groups,
                y_true,
                probabilities,
                audit_fraction=fraction,
                group_name=group_name,
                min_legitimate_count=min_legitimate_count,
            )
            for fraction in audit_fractions
        ],
        ignore_index=True,
    )
