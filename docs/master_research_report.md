# DAC 2026 Master Research Report

Evidence synthesis date: 2026-09-06. This is the single evidence source for
writing the competition report; it is not the final paper. Values are labelled
as holdout, OOF screen, fresh confirmation, or submission evidence.

## 1. Executive Summary

NHPA has a fixed manual-audit capacity of at most 5% of claims. The modeling
objective is therefore primarily a ranking and audit-allocation problem, with
probability quality and interpretability as supporting requirements.

The strongest supported general model is the `ctr-v1` CatBoost incumbent using
the native categorical `dati2_typeppk` combination. Its five-fold OOF result is
AP **0.830660**, Brier **0.169182**, and **7,808 fraud claims captured at 5%**
(8,008 audited claims under the floor-based convention). No completed
LambdaMART, fusion, selective-reranking, or stacked-ranking candidate exceeded
it. `clinical-shape-v1` and `frequency-v1` produced narrow or screen-level
cutoff alternatives, but the team did not select them as the general incumbent.

The defensible conclusion is not that every alternative failed. Rather, the
research progressively tested feature engineering, categorical modeling,
alternative learners, ranking objectives, and ensembles against the constrained
audit objective. The evidence supports retaining the incumbent while documenting
the value of leakage controls, top-K evaluation, calibration checks, fairness
descriptions, and conservative promotion gates. Later TabM, TabM-HPO, and
TabICLv2 foundation-model screens produced competitive point estimates, but none
completed the promotion path required to replace the incumbent.

## 2. Competition Problem and Analytical Objective

The case concerns healthcare insurance claims processed by NHPA. A smaller
subset may be suspicious, while manual review is limited to 5% of incoming
claims. The system must rank claims by fraud risk so that auditors receive the
highest-value portfolio while avoiding unnecessary review of legitimate claims.

The official submission schema is `claim_id,fraud_probability`. The ranking at
3%, 5%, and 7% is operationally meaningful, with 5% as the primary decision
point because it matches NHPA's stated audit capacity. The local artifacts do
not include the organizer's mathematical definition of NormalizedRecall@5%;
the project reports the transparent proxy `captured positives / min(audited
claims, total positives)` and must describe that limitation.

## 3. Dataset and Data Characteristics

| Dataset | Rows | Columns | Missing cells | Duplicate rows | ID status |
|---|---:|---:|---:|---:|---|
| Train | 160,174 | 53 | 0 | 0 | 160,174 unique `claim_id` |
| Test | 40,043 | 52 | 0 | 0 | 40,043 unique `claim_id` |
| Sample submission | 40,043 | 2 | 0 | 0 | IDs match test |

Train and test share the same 51 predictive columns and have zero claim-ID
overlap. The train label is binary with 79,971 zeros and 80,203 ones
(positive rate 0.500724). The variables comprise administrative/categorical
codes (`kdkc`, `dati2`, `typeppk`, `jkpst`, `jnspelsep`, `cmg`, `diagprimer`),
age (`umur`), length of stay (`los`), severity, 22 diagnosis-group counts, and
19 procedure-group counts. There are no timestamp, geography-coordinate,
patient, provider, or temporal-history fields in the local train/test tables.

`los` is highly right-skewed: 68.1% of train rows are zero, with mean 1.304,
maximum 592, and skewness 48.34. Three columns are constant and 21 are
near-constant under the audit threshold. These were documented rather than
silently treated as evidence of leakage.

## 4. Analytical Strategy

The project used a staged strategy:

1. Establish a baseline using random audit, prevalence, logistic regression,
   ExtraTrees, and CatBoost on a stratified 80/20 holdout.
2. Improve CatBoost through count/aggregate, interaction, LOS, native CTR,
   frequency/rarity, and clinical-shape hypotheses.
3. Test alternative learners including target-encoded XGBoost and tabular
   neural/transformer challengers where artifacts exist.
4. Test direct ranking with LambdaMART, then rank fusion and OOF hill climbing.
5. Test boundary-oriented rescue, selective reranking, and stacked LambdaMART.
6. Apply confirmation and promotion reasoning rather than promoting the best
   screen automatically.

## 5. Experiment History and Research Progression

### A. Baseline and CatBoost development

The initial holdout established ExtraTrees as the strongest holdout Recall@5%
(0.097313; 1,561/1,602) and CatBoost as the strongest holdout AP (0.816656)
with better calibration. This is an early holdout result, not the later
`ctr-v1` incumbent result.

The refined CatBoost screen tested counts, interactions, LOS, depth, L2,
random strength, and bagging temperature. `deep_combined_features` was selected
for a later three-seed ensemble, but was subsequently superseded by the CTR
branch.

### B. Native CTR and categorical modeling

The `ctr-v1` branch tested native CTR complexity, bucket controls, and selected
categorical combinations. The selected `ctr_dati2_typeppk` configuration became
the current general incumbent. This is the strongest coherent methodological
story for the main report because it directly improved the categorical
representation while retaining CatBoost's ordered categorical treatment.

### C. Frequency, rarity, and clinical shape

Frequency/rarity features were fitted fold-wise in their branch, but the
frequency variants were weaker than the control. The clinical-shape branch
added diagnosis/procedure burden and concentration summaries. Its two-seed
result reached 7,816 at 5%, but AP and Brier were worse than the incumbent, so it
was retained as a narrow cutoff alternative rather than the default.

### D. Alternative learners and neural challengers

Leakage-safe target-encoded XGBoost was materially weaker than CTR: the matched
ensemble recorded AP 0.806620, Brier 0.178927, and 135 fewer fraud claims at
5%. The fixed CTR+XGBoost blend produced a tiny screen gain but failed fresh
confirmation and was rejected.

The later neural experiments are present in the local output tree and supersede
the earlier recap's statement that TabM was unexecuted. `tabm-v1` trained both
quantile-normalized and piecewise-numeric TabM variants with fold-fitted
categorical encodings, then screened fixed CTR blends. Its selected 30% piecewise
TabM blend had raw OOF AP 0.832558, Brier 0.168178, and 7,818 captures at 5%,
but the paired promotion gate rejected it: the estimated gain was seven claims,
without a positive paired interval for either fraud capture or normalized recall.

`tabm-hpo-v1` then searched piecewise TabM configurations at `k=16` and `k=32`.
It selected trial 7 at `k=16` with a 50% TabM blend. The raw ensemble recorded
AP 0.832018, Brier 0.168193, and 7,809 captures at 5%; fresh-seed capture was
two claims below its CTR control. It was therefore unpromoted despite satisfying
the screen AP, Brier, fairness, and fresh-seed non-inferiority checks.

`foundation-v1` screened a TabICLv2 component on top of its saved CTR--TabM
base. The selected 50% foundation blend had raw OOF AP 0.831123, Brier 0.169094,
and 7,845 captures at 5% within that screen. It is explicitly unpromoted because
the required grouped and fresh-seed confirmation was not run; its model artifacts
also remain on the remote training server. Full TabICLv2 fine-tuning and LoRA
runs each contain two completed fold artifacts only. They have no complete OOF,
calibration, fairness, paired comparison, or promotion decision, so neither can
support a final-model claim.

### E. Ranking-oriented modeling

R001–R006 tested LightGBM LambdaMART with bounded shuffled pseudo-query chunks
because no natural query ID exists. R005 was the strongest LambdaMART screen:
Recall@5% 0.081543 and 6,540 captured, still 1,268 below the incumbent. R001
and R003 have identical effective configurations and identical OOF scores.

### F. Fusion, rescue, selective reranking, and stacking

R007 average-rank fusion of the incumbent and R001 captured 7,553, below the
incumbent. R008 tested only the incumbent plus R001; the candidate was rejected
and the final ranking reproduced the incumbent. R009 showed theoretical rescue
potential around the boundary but was diagnostic only. R010–R016 selective
rerankers and R017 stacked LambdaMART all remained below the incumbent; R016
was strongest among these with 7,783 captured at 5%.

## 6. Data Quality and Leakage Findings

Completed checks found no missing cells, duplicate full rows, duplicate claim
IDs, or train/test ID overlap. However, 1,737 repeated feature signatures have
conflicting labels. The baseline split also found 2,626 exact feature-signature
overlaps between fit and validation and 3,143 validation rows whose signature
appeared in fit. This creates a material risk that random validation is
optimistic if signatures represent hidden entities or repeated claim patterns.

The local data have no documented patient/provider key, temporal field, or data
dictionary. The report must therefore say that leakage was screened, not
eliminated. Frequency features and target-encoded XGBoost were described as
fold-fitted/inner-cross-fitted in their respective branches. The target label
was not used as a feature in the local baseline code.

## 7. Feature Engineering Findings

| Category | Rationale | Evidence | Final interpretation |
|---|---|---|---|
| Core/raw | Preserve observed claim and code information | Used across CatBoost and LambdaMART | Foundational feature set |
| Counts/aggregates | Summarize diagnosis/procedure burden and LOS | Improved early CatBoost screens, but later superseded | Useful hypothesis, not independently final |
| Interactions | Capture relationships among codes, severity, and LOS | Tested in refined CatBoost screens | Include only selected, documented interactions |
| Clinical shape | Represent active groups, concentration, joint burden | Narrow clinical-shape alternative; not default | Useful for analysis, not promoted |
| Native CTR | Model categorical combinations without treating codes as continuous | `dati2_typeppk` branch selected as incumbent | Strongest supported categorical improvement |
| Frequency/rarity | Represent category support or unusualness | Fold-fitted variants rejected | Do not claim usefulness |
| Target encoding | Alternative categorical representation for XGBoost | Leakage-safe but weaker than CTR | Challenger/rejected approach |

Permutation importance from the initial ExtraTrees holdout placed `typeppk`,
`kdkc`, `cmg`, `dati2`, and `diagprimer` highest. This is predictive association
in an ExtraTrees diagnostic, not causal importance for `ctr-v1`.

## 8. Model Development Findings

The table below is refreshed directly from the saved `outputs/` OOF artifacts.
For complete runs, it evaluates the raw OOF probability column using the common
floor-based 3%, 5%, and 7% audit convention. The TabICL rows are per-fold ranges,
not pooled OOF results. Historical holdout and ranking-only evidence remains in
the surrounding sections because matching output artifacts are unavailable.

| Experiment | Model/configuration | Recall@3% | Recall@5% | Recall@7% | NormalizedRecall@5% | Captured@5% | AP | Brier | Confirmation/status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline | Final CatBoost baseline | 0.058714 | 0.096854 | 0.134010 | 0.970030 | 7,768/8,008 | 0.824615 | 0.171754 | Complete OOF |
| `deep-ensemble-v1` | Three-seed refined CatBoost ensemble | 0.058851 | 0.097141 | 0.134633 | 0.972902 | 7,791/8,008 | 0.827844 | 0.170418 | Complete OOF |
| `ctr-v1` | CatBoost native CTR + `dati2_typeppk` | 0.058925 | 0.097353 | 0.134982 | 0.975025 | 7,808/8,008 | 0.830660 | 0.169182 | Incumbent; complete OOF |
| `frequency-v1` | CatBoost CTR frequency control | 0.058888 | 0.097365 | 0.135095 | 0.975150 | 7,809/8,008 | 0.830627 | 0.169203 | Complete OOF; variants rejected |
| `clinical-shape-v1` | Two-seed clinical-shape concentration | 0.058950 | 0.097453 | 0.134808 | 0.976024 | 7,816/8,008 | 0.830119 | 0.169413 | Complete OOF; narrow alternative |
| `xgb-v1` | Target-encoded XGBoost ensemble | 0.058290 | 0.095707 | 0.131653 | 0.958541 | 7,676/8,008 | 0.806620 | 0.178927 | Complete OOF; rejected |
| `ctr-xgb-blend-v1` screen | 2% XGBoost + matched CTR | 0.058925 | 0.097403 | 0.134883 | 0.975524 | 7,812/8,008 | 0.830330 | 0.169329 | Screen-selected |
| `ctr-xgb-blend-v1` confirmation | 2% XGBoost + fresh-confirmation CTR | 0.058925 | 0.097328 | 0.135007 | 0.974775 | 7,806/8,008 | 0.829252 | 0.169780 | Confirmation rejected |
| `tabm-v1` | 30% piecewise TabM + CTR blend | 0.058988 | 0.097478 | 0.135182 | 0.976274 | 7,818/8,008 | 0.832558 | 0.168178 | Paired-capture gate rejected |
| `tabm-hpo-v1` | HPO piecewise TabM (`k=16`, trial 7) + 50% CTR blend | 0.058963 | 0.097365 | 0.134845 | 0.975150 | 7,809/8,008 | 0.832018 | 0.168193 | Fresh capture −2; unpromoted |
| `foundation-v1` | 50% TabICLv2 blend over saved CTR--TabM base | 0.059137 | 0.097814 | 0.135369 | 0.979645 | 7,845/8,008 | 0.831123 | 0.169094 | Confirmation not run; unpromoted |
| `tabicl-ft-v1` | TabICLv2 full fine-tuning, two completed folds | 0.058989–0.059099 | 0.097625–0.098040 | 0.135296–0.135553 | 0.977894–0.982016 | 2,610–2,621 / 2,669 per fold | 0.820685–0.820714 | 0.174663–0.175074 | Incomplete; not comparable to full OOF |
| `tabicl-lora-v1` | TabICLv2 LoRA fine-tuning, two completed folds | 0.058839–0.059099 | 0.097774–0.097965 | 0.135258–0.135366 | 0.979393–0.981266 | 2,614–2,619 / 2,669 per fold | 0.821212–0.821379 | 0.174484–0.174969 | Incomplete; not comparable to full OOF |

The complete OOF rows remain screening or confirmation evidence unless their
saved promotion decision says otherwise. Their point estimates must not be used
to claim a replacement for `ctr-v1` without the required paired and fresh-seed
confirmation.

## 9. Ranking-Oriented Findings

R001 had top-5% overlap of 2,425 claims with the incumbent, Jaccard similarity
0.1784, and Spearman correlation 0.4238. Thus LambdaMART supplied genuine
ranking diversity, but R007 showed that diversity did not improve the fixed
audit portfolio. R005 was the strongest standalone ranking challenger, yet was
still materially below the incumbent.

The selective branch deliberately preserved the incumbent's high-risk pool
outside local reranking. Its candidates therefore remained highly correlated
with the incumbent; R016 had top-5% overlap 6,945 and Spearman 0.999755. This
stability did not produce a gain.

## 10. Ensemble and Fusion Findings

| Ensemble | Models combined | Method | Result | Decision |
|---|---|---|---|---|
| CTR+XGB blend | CTR + XGBoost | Fixed raw-probability weights | Tiny screen gain failed fresh confirmation | Rejected |
| R007 | Incumbent + R001 | Average rank | 7,553 at 5%, −255 vs incumbent | Rejected |
| R008 | Incumbent + R001 only | Greedy rank hill climb | Candidate rejected; incumbent retained | No new ensemble |
| R010–R016 | Incumbent-local pool + LambdaMART | Selective reranking | All below incumbent | Not promoted |
| R017 | Original features + incumbent OOF score | Stacked LambdaMART | 7,770 at 5% | Not promoted |
| `tabm-v1` | CTR + piecewise TabM | Fixed 30% TabM blend | +7 captures in the saved comparison, but neither paired lower bound was positive | Rejected |
| `tabm-hpo-v1` | CTR + HPO piecewise TabM | Fixed 50% TabM blend | Fresh-seed capture −2 versus control | Rejected |
| `foundation-v1` | Saved CTR--TabM base + TabICLv2 | 50% foundation blend | Screen-eligible; confirmation not run | Unpromoted |

Hill climbing added no new model: it started from the incumbent, tested R001 at
weight 0.05 as the best attempted addition, observed negative improvement, and
stopped. This is a valid conservative outcome, but it is not a complete search
over every LambdaMART candidate.

## 11. Audit Allocation Findings

Five percent is primary because it is the stated NHPA manual-review capacity.
The 3% and 7% views test whether the ranking remains useful when operational
capacity changes.

For the final incumbent OOF screen, the portfolio captured 4,726 at 3%, 7,808
at 5%, and 10,826 at 7%. In the initial holdout, ExtraTrees captured 948, 1,561,
and 2,153 at 3%, 5%, and 7%, compared with random-sample captures of 495, 798,
and 1,103. Capture increases with capacity, but the reported tables do not
establish a causal diminishing-return curve beyond these three operating points.

At the initial holdout's 5% capacity, ExtraTrees audited 41 legitimate claims
and CatBoost audited 53; this supports the operational interpretation of
precision as audit efficiency, but these are holdout values, not `ctr-v1` OOF
values.

## 12. Calibration Findings

CatBoost outputs are genuine probability-like outputs and the final `ctr-v1`
incumbent retained raw probabilities. The initial holdout calibration analysis
found sigmoid calibration improved Brier from 0.178593 to 0.178404, while
isotonic produced 0.178507. Sigmoid preserved the top-5% ranking exactly;
isotonic overlap was 99.56%. Calibration can improve probability interpretation
without changing ranking, but no calibrated transformation should be claimed
for the final incumbent unless its own fitted calibration artifact is cited.

The selected TabM and foundation artifacts also retained raw probabilities. For
`tabm-v1`, sigmoid worsened Brier to 0.169110 and isotonic reduced AP to
0.831234. For `tabm-hpo-v1`, the corresponding values were 0.169042 and
0.830705. The `foundation-v1` isotonic sidecar captured 7,869 claims at 5%, but
it worsened AP to 0.829378 and Brier to 0.169178; it is not a promoted
probability treatment. LambdaMART, rank-fusion, selective-reranking, and stacked
outputs are ranking scores, not calibrated probabilities. Their Brier values are
intentionally N/A.

## 13. Fairness Findings

The available group analysis is descriptive and was computed for the initial
ExtraTrees holdout, not a confirmed `ctr-v1` deployment analysis.

| Group | Audit selection rate | Legitimate-claim audit rate | Fraud capture rate |
|---|---:|---:|---:|
| Gender proxy `jkpst=L` | 0.05241 | 0.00285 | 0.10070 |
| Gender proxy `jkpst=P` | 0.04792 | 0.00232 | 0.09430 |
| Age 0–17 | 0.04398 | 0.00171 | 0.08699 |
| Age 18–39 | 0.05004 | 0.00262 | 0.09750 |
| Age 40–59 | 0.05111 | 0.00297 | 0.09899 |
| Age 60+ | 0.05636 | 0.00303 | 0.10806 |

The analysis found no statistical tests and does not support a claim of
discrimination. Differences may reflect prevalence, clinical mix, coding,
geography, service patterns, or model behavior. The final report should call
for monitoring, human review, reason codes, and appeal/governance safeguards.

## 14. Explainability Findings

The available global explanation is permutation importance on an ExtraTrees
validation sample. The largest associations were `typeppk`, `kdkc`, `cmg`,
`dati2`, `diagprimer`, `jnspelsep`, `los`, and `proc80_99`. No SHAP or local
explanation artifact was found. These results should be presented as model
associations and data signals, not causes of fraud. The final report should use
the permutation-importance figure as a diagnostic and explicitly identify its
model/split provenance.

## 15. Validation and Robustness

The strongest later branches use five shuffled stratified folds and save OOF
predictions. The ranking branch uses deterministic pseudo-query chunks capped at
5,000 rows because LightGBM rejects a query larger than 10,000 rows; these are
API-compatible chunks, not real business queries. Baseline CatBoost/ExtraTrees
also have a stratified 80/20 holdout.

Fresh confirmation was used to reject the CTR+XGB blend, `tabm-v1`, and
`tabm-hpo-v1`. The foundation screen has not yet completed its required grouped
and fresh-seed confirmation, while the TabICL fine-tuning artifacts cover only
two folds. The screen artifacts R001–R017 do not establish a fresh external or
temporal confirmation for a new winner. There is no temporal field or confirmed
hidden entity key, and repeated signatures make random splits potentially
optimistic.

## 16. Final Model / Submission Decision

- Strongest standalone/general model: `ctr-v1`, CatBoost native CTR with
  `dati2_typeppk`.
- Strongest new ranking candidate: R016, but it is 25 captures below the
  incumbent at 5% and is not promoted.
- Strongest neural point estimate: the unpromoted `foundation-v1` 50% TabICLv2
  blend; its confirmation gate remains incomplete.
- Strongest ensemble: no promoted ensemble; R008 reproduces the incumbent and
  the TabM/foundation blends remain unpromoted.
- Current selected model: `ctr-v1`, pending a confirmation-complete challenger.
- Submission artifact: `submissions/catboost_submission.csv` exists and has the
  correct 40,043-row schema, but its direct linkage to the `ctr-v1` model/config
  is not recorded in a manifest. Team confirmation is required before calling it
  the final submitted file.

## 17. Key Findings

### Finding 1
Observation: `ctr-v1` records AP 0.830660, Brier 0.169182, and 7,808 frauds at
5% on the later five-fold OOF screen.

Evidence: `experiments-recap-2.md`, `submissions/catboost_oof.csv`, and
`reports/research_results_metrics.csv`.

Interpretation: native categorical combination modeling produced the strongest
general incumbent in the available evidence.

Implication: use `ctr-v1` as the main model story and control.

### Finding 2
Observation: R005 is the best standalone LambdaMART screen but captures 6,540
at 5%, versus 7,808 for the incumbent.

Evidence: `reports/research_results.md` and R005 OOF/metrics artifacts.

Interpretation: direct ranking optimization was methodologically distinct but
not competitive as a replacement.

Implication: include it as a meaningful challenger, not as the final model.

### Finding 3
Observation: R001 differs materially from the incumbent, with top-5% Jaccard
0.1784 and Spearman 0.4238.

Evidence: `reports/research_diversity_matrix.csv`.

Interpretation: complementary ordering information existed, but complementarity
alone did not improve the portfolio.

Implication: explain why rank fusion was tested and rejected.

### Finding 4
Observation: R007 and R008 did not improve the incumbent; R008 rejected its
only tested LambdaMART addition.

Evidence: `experiments/metrics/hill_climb_trace.csv` and
`reports/research_results.md`.

Interpretation: conservative OOF selection prevented a weaker challenger from
being forced into the ensemble.

Implication: do not describe R008 as a multi-model improvement.

### Finding 5
Observation: selective and stacked ranking experiments R010–R017 all remained
below the incumbent; R016 was closest at 7,783 captures.

Evidence: `reports/research_results_r009_r017_metrics.csv`.

Interpretation: boundary-focused refinement did not convert theoretical rescue
potential into an observed gain.

Implication: retain the branch as supplementary evidence and stop short of
promotion.

### Finding 6
Observation: 1,737 repeated feature signatures have conflicting labels, and
3,143 validation rows share a signature with fit data.

Evidence: `reports/leakage_scan.csv`.

Interpretation: hidden entity/repetition structure may make random validation
optimistic.

Implication: state that leakage was screened but not eliminated; grouped or
temporal validation requires organizer-approved metadata.

### Finding 7
Observation: group-level selection rates differ by gender proxy and age group
in the available holdout fairness table.

Evidence: `reports/fairness_5pct.csv`.

Interpretation: descriptive disparity is not evidence of discrimination.

Implication: recommend monitoring and human safeguards rather than causal claims.

### Finding 8
Observation: calibration transformations changed Brier while largely preserving
top-K ranking.

Evidence: `reports/calibration_metrics.csv` and
`reports/calibration_ranking_overlap.json`.

Interpretation: probability reliability and audit ordering are related but
distinct objectives.

Implication: retain raw scores for ranking and report calibration separately.

### Finding 9
Observation: `tabm-v1` and `tabm-hpo-v1` produced competitive raw OOF AP/Brier
point estimates, but their paired fraud-capture intervals did not establish a
meaningful positive gain; `foundation-v1` is screen-eligible but lacks its
required confirmation layer.

Evidence: `outputs/runs/tabm-v1/metrics/tabm_promotion_decision.json`,
`outputs/runs/tabm-hpo-v1/metrics/tabm_hpo_promotion_decision.json`, and
`outputs/runs/foundation-v1/metrics/foundation_promotion_decision.json`.

Interpretation: promising screen metrics are insufficient for promotion when the
fixed-budget gain is uncertain or the confirmation protocol is incomplete.

Implication: retain `ctr-v1` and complete the required confirmation before
reconsidering a TabM or foundation-model blend.

## 18. Limitations

- The exact official NormalizedRecall formula is not available locally.
- Random/stratified validation may be optimistic under repeated signatures.
- No temporal or grouped validation is available for the anonymized claims.
- Several recap references point to remote or absent `outputs/` and
  `src/prs_its/` artifacts.
- TabICL fine-tuning and LoRA evidence are partial; neither has a complete OOF
  or promotion decision.
- `foundation-v1` is screen-only and its model artifacts are retained remotely.
- Fairness and permutation-importance artifacts belong to the initial holdout
  ExtraTrees analysis, not necessarily the final incumbent.
- Some generated integrity/metric files contain reproducibility issues recorded
  in `reports/consistency_audit.md`.

## 19. Recommended Final Narrative

NHPA's audit constraint makes ranking quality at the 5% operating point more
important than unconstrained classification accuracy. The team first audited
the claim data and established reproducible baselines, then progressively tested
feature summaries, categorical CTR representations, alternative learners, and
ranking-oriented challengers. Native CatBoost CTR modeling with the
`dati2_typeppk` combination produced the strongest general incumbent. Alternative
models revealed useful methodological contrasts and, in LambdaMART's case,
meaningful ranking diversity, but none delivered a confirmed improvement in the
fixed audit portfolio. The final recommendation is therefore to retain the
incumbent, communicate its 3/5/7% audit behavior, use probability calibration and
feature importance cautiously, and govern the system as decision support with
human review and fairness monitoring.

## 20. Source / Artifact Map

| Evidence | Primary artifact |
|---|---|
| Official structure | `docs/ANSWER SHEET PRELIMINARY ROUND_basabasic kacian.docx` |
| Latest narrative recap | `experiments-recap-2.md` |
| Earlier research recap | `experiments-recap.md` |
| R001–R008 report | `reports/research_results.md` |
| R009–R017 report | `reports/research_results_r009_r017.md` |
| Experiment registry | `experiments/registry.csv` |
| Experiment log | `experiments/experiment_log.md` |
| Data quality | `reports/data_audit_summary.csv`, `reports/data_audit_columns.csv`, `reports/leakage_scan.csv` |
| Baseline model metrics | `reports/baseline_metrics.csv` |
| Audit-capacity metrics | `reports/audit_capacity.csv`, `reports/research_results_metrics.csv` |
| Calibration | `reports/calibration_metrics.csv`, `reports/calibration_ranking_overlap.json` |
| Fairness | `reports/fairness_5pct.csv` |
| Explainability | `reports/permutation_importance.csv` |
| Ranking diversity | `reports/research_diversity_matrix.csv`, `reports/advanced_diversity_summary.csv` |
| Rescue analysis | `experiments/metrics/rescue_candidates.csv`, `experiments/metrics/rescue_summary.csv` |
| Submission artifacts | `submissions/catboost_submission.csv`, `submissions/catboost_baseline.csv`, `submissions/extra_trees_baseline.csv`, `submissions/lambdamart_r005.csv`, `submissions/r016_selective_reranker.csv` |
| TabM screen and confirmation | `outputs/runs/tabm-v1/metrics/`, especially `tabm_experiments.csv` and `tabm_promotion_decision.json` |
| TabM HPO | `outputs/runs/tabm-hpo-v1/metrics/`, especially `tabm_hpo_trials.csv` and `tabm_hpo_promotion_decision.json` |
| Foundation-model screen | `outputs/runs/foundation-v1/metrics/`, especially `foundation_experiments.csv` and `foundation_promotion_decision.json` |
| Partial TabICL fine-tuning | `outputs/runs/tabicl-ft-v1/metrics/` and `outputs/runs/tabicl-lora-v1/metrics/` |
