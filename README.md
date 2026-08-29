# NHPA Fraud-Risk Modeling

The pipeline trains CatBoost models for fraud-risk ranking and writes a validated
`claim_id,fraud_probability` submission for `data/test.csv`.

## Baseline workflow

The default command preserves the original five-experiment workflow and writes
artifacts directly under `outputs/`.

```bash
uv run prs-its-train --task-type GPU --devices 0
```

It performs five-fold OOF evaluation, saves the selected five fold models under
`outputs/models/`, and writes the test submission to
`outputs/submissions/catboost_submission.csv`.

## Refined workflow

The refined profile is opt-in and never overwrites baseline artifacts. It screens
the deep-model candidates, averages three seeded CV ensembles, and produces a
separate grouped-CV robustness report.

```bash
uv run prs-its-train --profile refined --run-name deep-ensemble-v1 --task-type GPU --devices 0
```

Refined artifacts are written to `outputs/runs/deep-ensemble-v1/`:

- `submissions/catboost_submission.csv` contains the raw ensemble probabilities used for ranking.
- `models/catboost_seed_<seed>_fold_<fold>.cbm` contains the 15 saved seed-fold models.
- `metrics/catboost_experiments.csv` contains the refinement-screen metrics.
- `metrics/catboost_seed_fold_metrics.csv` contains seed-level fold metrics.
- `metrics/catboost_grouped_robustness.csv` contains the duplicate-aware diagnostic.
- `metrics/catboost_calibration_comparison.csv` compares raw, sigmoid, and isotonic probabilities.

Use a unique run name to retain multiple refined attempts. The defaults are a
10,000-tree cap, 200-round early stopping, and seeds `42,2026,2718`.

```bash
uv run prs-its-train --profile refined --run-name deep-20k --iterations 20000 --ensemble-seeds 42,2026,2718
```

Use `--quiet` to hide both tqdm and CatBoost iteration logs. CPU execution is
available with `--task-type CPU`.
