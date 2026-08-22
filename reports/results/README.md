# Verified results

`baseline_metrics.csv` is regenerated from the two completed full-run
artifacts with:

```bash
python scripts/aggregate_results.py --artifacts artifacts --output reports/results/baseline_metrics.csv
```

Every saved value was independently recomputed from the corresponding
per-example `predictions.csv` using `scripts/verify_run.py`.

## Single-seed baseline snapshot

| Protocol | Model | Test macro-F1 | Test accuracy | Brier | ECE-10 |
|---|---|---:|---:|---:|---:|
| Random stratified | Majority | 0.4038 | 0.6772 | 0.2186 | 0.0000 |
| Random stratified | Source-only logistic | 0.9846 | 0.9864 | 0.0129 | 0.0126 |
| Random stratified | TF-IDF logistic | **0.9591** | **0.9647** | 0.0343 | 0.0612 |
| Source-disjoint | Majority | 0.4017 | 0.6715 | 0.2206 | 0.0060 |
| Source-disjoint | Source-only logistic | 0.2473 | 0.3285 | 0.2719 | 0.2266 |
| Source-disjoint | TF-IDF logistic | **0.9541** | **0.9593** | 0.0384 | 0.0798 |

These are verified **single-seed baseline results**, not final manuscript
estimates. The TF-IDF macro-F1 difference is about -0.0050 under this selected
source holdout, while calibration error increases. Multiple source holdouts,
confidence intervals, and the neural-model reconstruction remain required.

Legacy dissertation tables must not be copied into this directory as verified
results.

