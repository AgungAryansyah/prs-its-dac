from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def validate_probabilities(probabilities: Iterable[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("fraud_probability must not contain NaN or infinity.")
    if not ((0 <= values) & (values <= 1)).all():
        raise ValueError("fraud_probability must be within [0, 1].")
    return values


def prediction_summary(probabilities: Iterable[float] | np.ndarray) -> pd.Series:
    values = validate_probabilities(probabilities)
    return pd.Series(
        {
            "minimum": values.min(),
            "mean": values.mean(),
            "median": np.median(values),
            "percentile_95": np.quantile(values, 0.95),
            "percentile_99": np.quantile(values, 0.99),
            "maximum": values.max(),
            "n_unique": np.unique(values).size,
        }
    )


def make_submission(
    claim_ids: pd.Series,
    probabilities: Iterable[float] | np.ndarray,
    output_path: Path | None = None,
) -> pd.DataFrame:
    values = validate_probabilities(probabilities)
    if len(claim_ids) != len(values):
        raise ValueError("claim_ids and probabilities must have matching lengths.")
    submission = pd.DataFrame({"claim_id": claim_ids.to_numpy(), "fraud_probability": values})
    if not submission["claim_id"].equals(claim_ids.reset_index(drop=True)):
        raise RuntimeError("Submission claim_id order changed.")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
    return submission
