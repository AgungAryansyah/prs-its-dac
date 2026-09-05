# Experiment recap

Updated: 2026-09-05

This record distinguishes completed out-of-fold (OOF) experiments from partial
runs. A partial set of outer folds is not a promotion result and must not
replace the CTR incumbent.

## Current decision

The CTR model remains the incumbent. Neither TabICL run below has all three
outer folds, a complete OOF evaluation, calibration, fairness comparison, or
final full-data checkpoint. No TabICL submission has been promoted from these
artifacts.

## TabICLv2 full fine-tuning: partial result

`tabicl-ft-v1` completed outer folds 0 and 1. It fine-tunes all pretrained
weights for up to 12 epochs with a learning rate of `1e-5`; each fold used a
4096-row training context, full outer-training support, and 256-row prediction
chunks without an OOM fallback.

| Fold | OOF AP | Brier | ROC-AUC | Fraud caught @5% | Legitimate audits @5% | Peak GPU allocation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.820714 | 0.175074 | 0.816089 | 2,610 | 59 | 6.96 GiB |
| 1 | 0.820685 | 0.174663 | 0.817306 | 2,621 | 48 | 6.74 GiB |

Sources: `outputs/runs/tabicl-ft-v1/metrics/tabicl_ft_fold_0.json` and
`outputs/runs/tabicl-ft-v1/metrics/tabicl_ft_fold_1.json`.

## Frozen-LoRA TabICLv2: partial result

`tabicl-lora-v1` also completed outer folds 0 and 1. It freezes the pretrained
base and trains rank-8, alpha-16 LoRA adapters across 84 attention/MLP
projections plus the classification decoder. The preflight recorded 1,235,978
trainable parameters and 27,016,696 frozen parameters. Inference was
GPU-resident with 256-row prediction chunks; both completed folds used full
outer-training support and did not need an OOM fallback.

| Fold | OOF AP | Brier | ROC-AUC | Fraud caught @5% | Legitimate audits @5% | Peak GPU allocation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.821379 | 0.174969 | 0.816381 | 2,614 | 55 | 8.28 GiB |
| 1 | 0.821212 | 0.174484 | 0.817957 | 2,619 | 50 | 8.28 GiB |

Against the matching full-fine-tune folds, LoRA has higher AP and lower Brier
in both available folds. At the 5% audit budget it catches four more fraud
claims in fold 0 but two fewer in fold 1. This is promising but insufficient
evidence for a promotion or calibration choice.

Sources: `outputs/runs/tabicl-lora-v1/metrics/tabicl_lora_preflight.json`,
`outputs/runs/tabicl-lora-v1/metrics/tabicl_lora_fold_0.json`, and
`outputs/runs/tabicl-lora-v1/metrics/tabicl_lora_fold_1.json`.

## Required before any TabICL decision

1. Resume each run to produce fold 2 and its test predictions.
2. Complete the three-fold OOF metric, paired CTR comparison, calibration, and
   fairness artifacts.
3. Fine-tune on all labels only after complete OOF artifacts exist, then create
   the final raw submission.
4. Apply the existing promotion gate; a partial-fold or checkpoint-derived CSV
   remains provisional.

Model checkpoints are retained on the remote training server. This local
artifact export contains metrics and OOF predictions only.
