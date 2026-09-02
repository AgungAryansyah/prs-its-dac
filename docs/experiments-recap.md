# Experiment Recap

Updated: 2026-09-02

This log separates experiments with saved run artifacts from experiments that are
implemented but have not yet been run. Metrics are OOF estimates, not public
leaderboard scores. Selecting a winner from the same screen used to score it can
overstate its expected performance; fresh confirmation is required for a promotion.
[Varma & Simon, 2006](https://doi.org/10.1186/1471-2105-7-91)

## Evaluation convention

Completed screens used 160,174 labelled claims, normally five shuffled stratified
folds, and an audit budget of 8,008 claims at 5%. AP and Brier assess the full
probability vector; `fraud caught @5%` and normalized recall assess the top 5% risk
portfolio. The metric definitions are in [metrics.py](../src/prs_its/metrics.py).

The detailed tables below contain saved screen values, generally from seed 42.
Separate final-ensemble values are identified explicitly because they can use more
than one seed.

## Completed run summary

| Run | Selected configuration | Final OOF evidence | Outcome | Evidence |
|---|---|---|---|---|
| Original `outputs/` | `deep_regularized` | AP 0.824615; Brier 0.171754; 7,768 fraud caught @5% | Initial raw CatBoost submission | [findings](../outputs/metrics/catboost_final_findings.csv) |
| `deep-ensemble-v1` | `deep_combined_features` | AP 0.827844; Brier 0.170418; 7,791 @5% | Improved baseline, later superseded | [findings](../outputs/runs/deep-ensemble-v1/metrics/catboost_final_findings.csv) |
| `ctr-v1` | `ctr_dati2_typeppk` | AP **0.830660**; Brier **0.169182**; 7,808 @5% | Current general incumbent | [findings](../outputs/runs/ctr-v1/metrics/catboost_final_findings.csv) |
| `frequency-v1` | `frequency_control` | AP 0.830627; Brier 0.169203; 7,809 @5% | Tied CTR control; frequency features rejected | [findings](../outputs/runs/frequency-v1/metrics/catboost_final_findings.csv) |
| `clinical-shape-v1` | `clinical_shape_concentration` | Two-seed result: 7,816 @5%; AP 0.830119; Brier 0.169413 | Narrow cutoff alternative, not default | [findings](../outputs/runs/clinical-shape-v1/metrics/catboost_final_findings.csv) |
| `xgb-v1` | `te_xgb_support` | Matched-seed AP 0.806620; Brier 0.178927; 7,676 @5% | Not competitive with CTR | [paired comparison](../outputs/runs/xgb-v1/metrics/xgb_vs_ctr_ensemble_paired.csv) |
| `ctr-xgb-blend-v1` | `ctr_xgb_raw_w02` | Small screen gain did not survive fresh confirmation | Rejected; no promoted submission | [decision](../outputs/runs/ctr-xgb-blend-v1/metrics/promotion_decision.json) |
| `tabm-v1` | `tabm_piecewise_ctr_blend_w30` | AP 0.832558; Brier 0.168178; 7,818 @5% | Rejected: +7 @5% was neither meaningful nor statistically positive | [paired comparison](../outputs/runs/tabm-v1/metrics/tabm_piecewise_ctr_blend_w30_ensemble_vs_ctr_paired.csv) |

## Detailed experiment record

### Original CatBoost baseline

This established the first CatBoost reference: original variables, class weighting,
clinical-count features, and regularization changes. The selector chose
`deep_regularized` and retained raw probabilities.

Source: [screen metrics](../outputs/metrics/catboost_experiments.csv),
[final findings](../outputs/metrics/catboost_final_findings.csv), and
[training implementation](../src/prs_its/training.py).

| Experiment | Change | AP | Brier | Fraud caught @5% | Normalized recall @5% |
|---|---|---:|---:|---:|---:|
| `unweighted_baseline` | Original features, no weighting | 0.823059 | 0.172444 | 7,758 | 0.968781 |
| `balanced_baseline` | CatBoost balanced weights | 0.823650 | 0.172191 | 7,764 | 0.969530 |
| `count_features` | Diagnosis and procedure counts | 0.823722 | 0.172160 | 7,772 | 0.970529 |
| `shallow_regularized` | Shallower regularized trees | 0.821106 | 0.173301 | 7,751 | 0.967907 |
| `deep_regularized` | Depth-8 regularized trees | 0.824303 | 0.171905 | 7,774 | 0.970779 |

### Refined CatBoost screen (`deep-ensemble-v1`)

This evaluated counts, interactions, LOS, and modest depth/regularization changes.
`deep_combined_features` was selected, then trained with seeds `42`, `2026`, and
`2718`.

Source: [screen metrics](../outputs/runs/deep-ensemble-v1/metrics/catboost_experiments.csv),
[final configuration](../outputs/runs/deep-ensemble-v1/models/catboost_final_config.json), and
[final findings](../outputs/runs/deep-ensemble-v1/metrics/catboost_final_findings.csv).

| Experiment | Change | AP | Brier | Fraud caught @5% | Normalized recall @5% |
|---|---|---:|---:|---:|---:|
| `deep_regularized_control` | Depth-8 control | 0.824042 | 0.172003 | 7,766 | 0.969780 |
| `deep_count_features` | Diagnosis/procedure counts | 0.824824 | 0.171597 | 7,784 | 0.972028 |
| `deep_interaction_features` | Categorical interactions and LOS | 0.825788 | 0.171343 | 7,774 | 0.970779 |
| `deep_combined_features` | Counts, interactions, and LOS | 0.826113 | 0.171191 | 7,783 | 0.971903 |
| `deep_depth_7` | Depth 7 | 0.824104 | 0.171993 | 7,766 | 0.969780 |
| `deep_depth_9` | Depth 9 | 0.824298 | 0.171857 | 7,771 | 0.970405 |
| `deep_l2_20` | Stronger L2 regularization | 0.824163 | 0.171954 | 7,776 | 0.971029 |
| `deep_random_strength_1` | Lower split-score randomness | 0.824540 | 0.171767 | 7,764 | 0.969530 |
| `deep_bagging_temperature_0_5` | Lower Bayesian-bootstrap temperature | 0.825372 | 0.171402 | 7,780 | 0.971528 |

CatBoost's ordered boosting and categorical processing are described by
[Prokhorenkova et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html).

### Native CatBoost CTR screen (`ctr-v1`)

This is the current general incumbent. It tested native categorical-combination
complexity, count buckets, and a small set of semantic interaction hypotheses. The
winner added `dati2_typeppk` at `max_ctr_complexity=4`.

Source: [screen metrics](../outputs/runs/ctr-v1/metrics/catboost_experiments.csv),
[final configuration](../outputs/runs/ctr-v1/models/catboost_final_config.json), and
[final findings](../outputs/runs/ctr-v1/metrics/catboost_final_findings.csv).

| Experiment | Change | AP | Brier | Fraud caught @5% | Normalized recall @5% |
|---|---|---:|---:|---:|---:|
| `ctr_complexity_1` | CTR complexity 1 | 0.799067 | 0.182726 | 7,661 | 0.956668 |
| `ctr_complexity_2` | CTR complexity 2 | 0.819413 | 0.174221 | 7,756 | 0.968531 |
| `ctr_complexity_4` | CTR complexity 4 | 0.826255 | 0.171120 | 7,789 | 0.972652 |
| `ctr_complexity_6` | CTR complexity 6 | 0.826117 | 0.171122 | 7,774 | 0.970779 |
| `ctr_bucket_control` | Complexity 4 plus count buckets | 0.827682 | 0.170655 | 7,793 | 0.973152 |
| `ctr_dati2_typeppk` | Add `dati2_typeppk` | 0.829150 | 0.169887 | 7,802 | 0.974276 |
| `ctr_dati2_cmg` | Add `dati2_cmg` | 0.827662 | 0.170641 | 7,800 | 0.974026 |
| `ctr_kdkc_cmg` | Add `kdkc_cmg` | 0.827551 | 0.170774 | 7,797 | 0.973651 |
| `ctr_severitylevel_procedure_count_bucket` | Severity/procedure-bucket cross | 0.827853 | 0.170550 | 7,792 | 0.973027 |
| `ctr_diagprimer_secondary_diagnosis_count_bucket` | Diagnosis/count-bucket cross | 0.827402 | 0.170692 | 7,776 | 0.971029 |

The selected final three-seed run kept raw probabilities. Native categorical
combination controls are documented in the
[CatBoost CTR parameter reference](https://catboost.ai/docs/en/references/training-parameters/common#max_ctr_complexity).

### Fold-fitted frequency and rarity screen (`frequency-v1`)

This tested count, log-count, and rare-category signals fitted only on each outer
training fold. None beat the no-frequency control, so the final model intentionally
retained `frequency_control`.

Source: [screen metrics](../outputs/runs/frequency-v1/metrics/catboost_experiments.csv),
[final findings](../outputs/runs/frequency-v1/metrics/catboost_final_findings.csv), and
[fold transformer](../src/prs_its/modeling.py).

| Experiment | Change | AP | Brier | Fraud caught @5% | Normalized recall @5% |
|---|---|---:|---:|---:|---:|
| `frequency_control` | CTR recipe without frequency features | 0.829199 | 0.169892 | 7,806 | 0.974775 |
| `frequency_count` | Category counts | 0.828750 | 0.170069 | 7,787 | 0.972403 |
| `frequency_log_count` | Log category counts | 0.828564 | 0.170106 | 7,791 | 0.972902 |
| `frequency_rare_flag` | Rare-category flags | 0.828853 | 0.170054 | 7,798 | 0.973776 |

### Clinical-shape screen (`clinical-shape-v1`)

This added compact summaries of existing diagnosis/procedure indicators: active
groups, concentration, and joint burden. `clinical_shape_concentration` won the
screen and was confirmed with two seeds (`42`, `2026`).

Source: [screen metrics](../outputs/runs/clinical-shape-v1/metrics/catboost_experiments.csv),
[final findings](../outputs/runs/clinical-shape-v1/metrics/catboost_final_findings.csv), and
[feature implementation](../src/prs_its/modeling.py).

| Experiment | Change | AP | Brier | Fraud caught @5% | Normalized recall @5% |
|---|---|---:|---:|---:|---:|
| `clinical_shape_control` | CTR recipe without shape features | 0.829121 | 0.169910 | 7,800 | 0.974026 |
| `clinical_shape_active_groups` | Active diagnosis/procedure-group counts | 0.829180 | 0.169872 | 7,796 | 0.973526 |
| `clinical_shape_concentration` | Largest-group and concentration summaries | 0.829443 | 0.169800 | 7,796 | 0.973526 |
| `clinical_shape_joint_burden` | Total burden and procedure share | 0.828960 | 0.169971 | 7,798 | 0.973776 |

The final two-seed concentration ensemble caught five more fraud claims at 5% than
matched two-seed CTR, but had slightly lower AP and worse Brier. It is a narrow
cutoff alternative, not the default incumbent. Its directory has a configuration and
OOF evidence but no saved fold models, so it is only partially reproducible.

### Leakage-safe target-encoded XGBoost challenger (`xgb-v1`)

This changed both the categorical representation and learner: inner-cross-fitted
target encoding, selected interactions, then fold-fitted category-support counts.

Source: [screen metrics](../outputs/runs/xgb-v1/metrics/xgb_experiments.csv),
[paired CTR comparison](../outputs/runs/xgb-v1/metrics/xgb_vs_ctr_ensemble_paired.csv), and
[implementation](../src/prs_its/xgb_modeling.py).

| Experiment | Change | AP | Brier | Fraud caught @5% | Normalized recall @5% |
|---|---|---:|---:|---:|---:|
| `te_xgb_base` | Smoothed single-column target encodings | 0.747219 | 0.203925 | 7,429 | 0.927697 |
| `te_xgb_interactions` | Add three selected interaction encodings | 0.783467 | 0.188708 | 7,603 | 0.949426 |
| `te_xgb_support` | Add fold-fitted category support counts | 0.804615 | 0.179906 | 7,666 | 0.957293 |

The final matched-seed XGBoost ensemble was materially weaker than CTR: 135 fewer
fraud claims at 5%, AP 0.806620 versus 0.830331, and Brier 0.178927 versus
0.169343. Its standalone submission remains an artifact, not a promoted
replacement. The present run lacks saved models and final configuration, so rerun it
for complete reproducibility.

References: [Micci-Barreca, 2001](https://doi.org/10.1145/507533.507538),
[scikit-learn TargetEncoder documentation](https://scikit-learn.org/stable/modules/preprocessing.html#target-encoder),
and [XGBoost GPU documentation](https://xgboost.readthedocs.io/en/stable/gpu/index.html).

### CTR + XGBoost fixed-weight blend (`ctr-xgb-blend-v1`)

This predeclared a small raw-probability blend grid against matched two-seed CTR,
then fit a fresh XGBoost seed for confirmation. The 2% XGBoost weight cleared the
screen but failed the fresh normalized-recall gate; no blend submission was promoted.

Source: [blend screen](../outputs/runs/ctr-xgb-blend-v1/metrics/blend_experiments.csv),
[promotion decision](../outputs/runs/ctr-xgb-blend-v1/metrics/promotion_decision.json), and
[implementation](../src/prs_its/blend_training.py).

| Experiment | XGBoost weight | AP | Brier | Fraud caught @5% | Normalized recall @5% |
|---|---:|---:|---:|---:|---:|
| `ctr_xgb_raw_w00` | 0% (CTR control) | 0.830331 | 0.169343 | 7,811 | 0.975400 |
| `ctr_xgb_raw_w02` | 2% | 0.830330 | 0.169329 | 7,812 | 0.975524 |
| `ctr_xgb_raw_w05` | 5% | 0.830297 | 0.169323 | 7,812 | 0.975524 |

The selected 2% blend is an example of why fresh confirmation matters: its tiny
screen gain did not reproduce, and the saved decision correctly rejects promotion.

### TabM challenger (`tabm-v1`)

This tested a fold-safe TabM base model and piecewise-linear numeric embeddings,
then predeclared raw and 5–30% TabM blends with matched CTR OOF predictions. Raw
TabM was weaker than CTR: the piecewise raw run reached AP 0.818622, Brier
0.173814, and 7,703 fraud claims caught at 5%. The 30% piecewise blend improved
full-vector quality and was selected for confirmation.

| Configuration | AP | Brier | Fraud caught @5% | Decision |
|---|---:|---:|---:|---|
| `ctr_control_raw` screen control | 0.829085 | 0.169928 | 7,804 | Incumbent screen reference |
| `tabm_piecewise_raw` | 0.818622 | 0.173814 | 7,703 | Weaker raw challenger |
| `tabm_piecewise_ctr_blend_w30` screen | 0.831475 | 0.168678 | 7,809 | Selected for confirmation |
| `tabm_piecewise_ctr_blend_w30` two-seed ensemble | 0.832558 | 0.168178 | 7,818 | +7 at 5% versus matched CTR |

The ensemble improved AP by 0.002227 and Brier by 0.001166 versus matched CTR,
but the 5% capture interval was -11 to +21 claims. It therefore failed both the
minimum +20-capture rule and the statistically positive capture requirement, even
though the fresh seed was non-inferior. Raw calibration was retained because neither
cross-fitted alternative improved its Brier score.

Source: [experiment table](../outputs/runs/tabm-v1/metrics/tabm_experiments.csv),
[paired comparison](../outputs/runs/tabm-v1/metrics/tabm_piecewise_ctr_blend_w30_ensemble_vs_ctr_paired.csv),
[calibration result](../outputs/runs/tabm-v1/metrics/tabm_piecewise_ctr_blend_w30_calibration_metrics.csv), and
[promotion decision](../outputs/runs/tabm-v1/metrics/tabm_promotion_decision.json).

The model and preprocessor artifacts are intentionally retained on the remote
training server and are not mirrored in this local artifact export.

## Implemented but not yet executed

The following suites are implemented but have no completed result yet.

| Suite | Planned screen | Purpose | References |
|---|---|---|---|
| Causal history CatBoost | `history_static_control`, `history_financial`, `history_behavioral`, `history_adjudication` | Adds time-available financial, provider/patient-behaviour, and adjudication signals using rolling-origin validation; it requires an external history file and rejects post-event leakage. | [feature builder](../src/prs_its/history_modeling.py), [command](../src/prs_its/history_training.py), commits `90817bc`, `2bf95da` |
| TabM HPO challenger | 15 seeded TPE trials each for `k=16` and `k=32`, then raw and fixed 5–50% CTR blends | Tunes only the piecewise family, persists resumable SQLite studies, checks the largest GPU configuration, and retains the same confirmation and promotion gates. | [HPO command](../src/prs_its/tabm_tuning.py), [fold-safe CV](../src/prs_its/tabm_modeling.py), [Gorishniy et al., 2025](https://openreview.net/forum?id=Sd4wYYOhmY) |

## Current conclusion

`ctr-v1` with `dati2_typeppk` remains the strongest general, reproducible incumbent.
Frequency variants were ruled out, clinical shape is a small cutoff-specific trade-off,
target-encoded XGBoost was not competitive, and `tabm-v1` did not clear its
capture-promotion gate. Causal history and TabM HPO should only be considered
improvements if their paired, fresh-seed, fairness, and grouped-CV gates show a
meaningful gain over CTR.
