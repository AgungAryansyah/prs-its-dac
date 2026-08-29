# Improvement Research: NHPA Fraud-Risk Ranking

## Bottom line

- There is no universal tabular-data SOTA. For this mix of 160k rows, high-cardinality codes, counts, and a 5% audit decision, **CatBoost should remain the primary model**. Its ordered boosting and categorical statistics are designed to limit target leakage; broad tabular benchmarks also continue to find tree ensembles highly competitive or best overall.
- A reported `Precision@5% = 1.0` is a **testable local hypothesis**, not a safe target to assume. The current latest finding is `7,768 / 8,008 = 97.0%`; the saved OOF file is older than that finding and must be regenerated before making model-selection claims.
- Do not search for external copies of the data or labels. All experiments below use only the provided train/test files and leakage-safe validation.

## Evidence that applies here

- CatBoost's ordered boosting and ordered categorical handling were proposed specifically to reduce prediction shift and target leakage from boosting and categorical target statistics ([Prokhorenkova et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)). This fits `kdkc`, `dati2`, `typeppk`, `cmg`, and `diagprimer`.
- Large tabular benchmarks find tree models stronger and cheaper to tune than generic deep networks; a 2023 benchmark found CatBoost best across its 104 datasets ([McElfresh et al., 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/f06d5ebd4ff40b40dd97e30cee632123-Paper-Datasets_and_Benchmarks.pdf)).
- Insurance-fraud operations are naturally evaluated by `Precision@k`, because investigators review only the top-ranked claims ([Debener et al., 2023](https://onlinelibrary.wiley.com/doi/full/10.1111/jori.12427)). This directly supports optimizing and reporting the top 3%/5%/7% portfolios.
- TabM is a credible *diversity* candidate rather than a replacement: it was the strongest model among the tabular deep-learning methods evaluated by its authors ([Gorishniy et al., 2025](https://openreview.net/pdf?id=Sd4wYYOhmY)). It should be kept only if its OOF predictions improve a blend.

## Highest-value, dataset-specific experiments

1. **Make the evaluation artifacts canonical first.** Re-run the selected configuration once and save OOF predictions, fold metrics, fairness metrics, config, and submission from the same run. Compare 5-fold stratified CV with a grouped diagnostic split based on repeated full feature profiles; 1,737 profiles currently have mixed labels. Use the grouped result as a leakage-sensitivity check, not as an automatic replacement for the competition split.

2. **Ablate categorical representation.** The EDA found 76,189 string values affected by whitespace stripping. Compare raw codes with deterministic `strip`/case normalization and documented placeholder normalization. Keep leading zeros. Evaluate raw CatBoost categorical handling against controlled categorical-combination settings, prioritizing interactions among `kdkc`, `dati2`, `typeppk`, `cmg`, and `diagprimer`. CatBoost supports categorical combinations, but higher complexity can increase model size and overfit; select strictly from OOF results.

3. **Extend count engineering, not just total counts.** The existing `secondary_diagnosis_count` and `procedure_count` trial did not win. Test: non-zero flags, `log1p` counts, diagnosis/procedure totals separately, and a small set of clinically neutral interactions such as `los × severitylevel`, `umur × jnspelsep`, and `typeppk × jnspelsep`. The 37 grouped fields with values above one must stay numeric/count-like rather than being silently binarized.

4. **Build a diverse OOF ensemble.** Train a small seed/parameter CatBoost ensemble, then add one non-CatBoost tree baseline (LightGBM or XGBoost with encoding fitted inside every fold). Blend only OOF predictions, preferably with a simple constrained weight search. Refit selected components on all training data, average test predictions, and then cross-fit calibration again. A model is useful only if it improves AP, normalized recall@5%, and/or Brier without worsening fairness materially.

5. **Test TabM only as a challenger.** Use the same folds and the same normalized categorical/count inputs. It is worth one controlled comparison because its errors may differ from CatBoost's; retain it only when its OOF predictions add measurable blend value. Do not replace the tree baseline based on a single split or public leaderboard movement.

6. **Treat perfect top-5% precision as a proof requirement.** Report `fraud_caught`, false positives, raw recall, normalized recall, AP, Brier, and bootstrap intervals on the complete OOF vector. Accept a 1.0 claim only when every selected OOF audit slot is fraud and the result persists across seeds/folds. Do not force submitted probabilities to 1: that does not improve ranking and can damage Brier score.

7. **Keep calibration and policyholder protection as gates.** The current reliability bins are already close to the diagonal, and raw probabilities beat the tested calibrators. Revisit calibration only after a new model or blend is selected. Recompute legitimate-only audit exposure for `jkpst` and age bands at 3%/5%/7%; do not trade a small ranking gain for a material subgroup disparity without documenting it.

## Recommended order

1. Artifact synchronization and grouped-split diagnostic.
2. String normalization and categorical-combination ablations.
3. Count/interactions ablation.
4. CatBoost seed ensemble plus LightGBM/XGBoost challenger.
5. Optional TabM challenger, OOF blend, calibration, and fairness gate.

The first success criterion is a repeatable OOF improvement over the current deep-regularized CatBoost—not a public-leaderboard score or an externally recovered label.
