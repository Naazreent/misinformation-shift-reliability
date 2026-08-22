# Research contract

## Study title

**Reliability of Misinformation Classification under Source Shift**

## Question

How much does a random record-level split overestimate text-classification
performance relative to a publisher/source-disjoint split on the supplied IFND
dataset after normalized-duplicate controls?

## Prespecified hypotheses

- **H1:** TF-IDF logistic regression will have higher macro-F1 on the random
  stratified split than on the source-disjoint split.
- **H2:** A source-only classifier will perform very strongly on the random
  split but fail to transfer to unseen sources.
- **H3:** Probability calibration will degrade on the source-disjoint test set,
  reflected in higher Brier score, log loss, or expected calibration error.

These are planned hypotheses, not findings.

## Contribution boundary

Version 0.1 is a dataset and evaluation reconstruction. It does not claim a new
model, objective truth classification, state-of-the-art performance, or safe
deployment. The original FNN/CNN/LSTM/BERT results remain legacy claims until
new implementations pass the same locked protocol and tuning budget.

## Data contract

- Required columns: `id`, `Statement`, `Web`, `Category`, `Date`, `Label`.
- Raw input is immutable and checked against the recorded SHA-256.
- Labels are converted explicitly from boolean or accepted binary strings.
- Statements are normalized with Unicode NFKC, lower-casing, and collapsed
  non-word whitespace only for duplicate grouping; models receive original text.
- All normalized-text groups with conflicting labels are excluded.
- One record is retained from each remaining normalized-text group.
- The exact retained IDs and split assignments are saved per run.
- Raw data is not redistributed until source and licence are verified.

## Split protocols

1. `random_stratified`: 70% train, 15% validation, 15% test, fixed seed.
2. `source_disjoint`: publisher/source groups are disjoint across train,
   validation, and test; deterministic candidate selection preserves both
   labels and records the selected sources.

Temporal evaluation is deferred because the CSV contains 11,321 missing dates
and an implausible 2025 date cluster that must be traced to source formatting.

## Models and fixed budget

- Majority-class dummy classifier.
- Source-only one-hot logistic regression as a confounding diagnostic.
- TF-IDF (1-2 grams) plus class-balanced logistic regression as the primary
  text baseline.

No hyperparameter search is performed in version 0.1. Neural architectures will
be added only after the baseline protocol is locked.

## Metrics

- Primary: test macro-F1.
- Secondary: accuracy, balanced accuracy, macro precision, macro recall,
  weighted F1, Brier score, log loss, expected calibration error, and confusion
  matrix counts.
- Validation metrics are reported separately and are never substituted for the
  test result.

## Repetition and uncertainty

The current deterministic reconstruction fixes seed 42. Before manuscript use,
the protocol must be extended to at least five prespecified seeds or folds and
paired/clustered bootstrap confidence intervals at source level.

## Acceptance criteria

- Dataset checksum and schema validation pass.
- Unit tests and synthetic smoke test pass.
- No normalized statement or source crosses a disjoint split boundary.
- Every result is traceable to a run ID, data checksum, resolved configuration,
  prediction file, and metric record.
- Headline tables regenerate from machine-readable artifacts.
- Unsupported legacy dissertation values are not presented as reproduced.

