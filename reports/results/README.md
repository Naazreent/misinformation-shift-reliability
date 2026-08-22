# Verified results

These tables were generated from 12 full-data runs: classical and neural model
families under random-stratified and source-disjoint protocols for prespecified
seeds 13, 42, and 87. `scripts/verify_run.py` independently recomputed every
saved metric from per-example predictions before aggregation.

## Primary comparison

Test values are mean ± sample standard deviation across the three seeds.

| Model | Protocol | Macro-F1 | Accuracy | ECE-10 |
|---|---|---:|---:|---:|
| TF-IDF logistic | Random stratified | 0.9609 ± 0.0017 | 0.9662 ± 0.0014 | 0.0634 ± 0.0020 |
| TF-IDF logistic | Source disjoint | **0.9594 ± 0.0051** | **0.9642 ± 0.0043** | 0.0798 ± 0.0043 |
| FNN | Random stratified | 0.9485 ± 0.0023 | 0.9551 ± 0.0021 | 0.0254 ± 0.0035 |
| FNN | Source disjoint | 0.9381 ± 0.0066 | 0.9450 ± 0.0053 | 0.0486 ± 0.0109 |
| CNN | Random stratified | 0.9558 ± 0.0021 | 0.9617 ± 0.0020 | 0.0113 ± 0.0069 |
| CNN | Source disjoint | 0.9480 ± 0.0035 | 0.9544 ± 0.0028 | **0.0364 ± 0.0167** |
| LSTM | Random stratified | 0.9566 ± 0.0029 | 0.9625 ± 0.0025 | 0.0150 ± 0.0017 |
| LSTM | Source disjoint | 0.9444 ± 0.0047 | 0.9509 ± 0.0047 | 0.0404 ± 0.0264 |
| BERT-tiny | Random stratified | **0.9662 ± 0.0034** | **0.9705 ± 0.0030** | **0.0103 ± 0.0047** |
| BERT-tiny | Source disjoint | 0.9526 ± 0.0098 | 0.9576 ± 0.0090 | 0.0411 ± 0.0163 |

`BERT-tiny` is the pinned 2-layer checkpoint named in the research contract; it
is not `bert-base-uncased` and does not reproduce the dissertation's legacy BERT
configuration.

## Descriptive protocol gaps

The gap is source-disjoint minus random-stratified, paired by seed. Negative
macro-F1 and positive ECE values indicate worse source-disjoint behaviour.

| Model | Macro-F1 gap | ECE-10 gap |
|---|---:|---:|
| TF-IDF logistic | -0.0014 ± 0.0063 | +0.0165 ± 0.0047 |
| FNN | -0.0104 ± 0.0049 | +0.0232 ± 0.0133 |
| CNN | -0.0078 ± 0.0042 | +0.0251 ± 0.0236 |
| LSTM | -0.0122 ± 0.0062 | +0.0254 ± 0.0261 |
| BERT-tiny | -0.0136 ± 0.0074 | +0.0308 ± 0.0165 |

All four neural families have a lower mean macro-F1 and higher mean ECE under
source shift. TF-IDF has a small, variable mean macro-F1 gap but higher ECE in
all three comparisons. With only three holdouts, these are descriptive effects,
not claims of statistical significance.

The source-only diagnostic supports the confounding concern: mean test macro-F1
falls from 0.9852 ± 0.0019 under a random split to 0.2452 ± 0.0038 on unseen
publishers.

## Machine-readable evidence

- `all_metrics.csv`: all validation and test metrics from the 12 verified runs.
- `comparison_summary.csv`: n, mean, sample standard deviation, range, and
  exploratory t intervals for every model/metric/protocol combination.
- `protocol_gaps.csv`: source-minus-random differences paired by seed.
- `run_manifest.json`: run IDs, data checksum, code revision, seed, protocol,
  model families, and verification status.
- `baseline_metrics.csv`: superseded single-seed v0.1 evidence retained for
  provenance; do not use it as the current headline table.
- `../figures/repeated_seed_model_comparison.png`: generated macro-F1 and ECE
  comparison using mean ± sample standard deviation.

The t intervals are included for transparency but should not be overinterpreted
at n=3. The raw CSV, per-example predictions, split assignments, and model
checkpoints are deliberately excluded from Git; authorised users regenerate
them locally with:

```bash
python scripts/reproduce_matrix.py --data data/raw/IFND.csv
```

Legacy dissertation tables are provenance evidence only and must not be copied
into this directory as verified results.
