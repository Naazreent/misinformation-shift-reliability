# Dissertation and Colab reconstruction audit

**Audit date:** 22 August 2026  
**Researcher:** Naazreen Tabassum  
**Scope:** supplied dissertation DOCX, shared Colab notebook, and supplied
`IFND.csv`. This audit distinguishes observable artifacts from scientific
results that have been independently rerun.

## Inputs inspected

| Artifact | SHA-256 | Observable state |
|---|---|---|
| Dissertation DOCX | `8017726e44fa83e4d2b3b4131c338f2b835edbcb7836a499898772282b834716` | 101 rendered pages, 726 paragraphs, 26 tables, 58 inline figures |
| Colab notebook | `a2a54a971d0c6f7f58022e24a7ac40e712138aeb65be01bd9ed375a198661926` | 210 cells: 208 code and 2 Markdown; outputs retained, execution counts unset |
| IFND CSV | `fafbb516c5d78a26ed1cfdf5077f989eae24c1258ad3c4de731ac6d6932505d6` | 56,714 rows and 7 columns |

The notebook metadata records a Colab T4 GPU environment. That metadata is
historical; it does not prove that every saved output came from a single clean,
top-to-bottom execution.

## Verified data observations

These values were recomputed directly from the supplied CSV:

- Columns: `id`, `Statement`, `Image`, `Web`, `Category`, `Date`, `Label`.
- Labels: 37,800 `True` and 18,914 `False`.
- Recorded sources: 29.
- Missing `Date` values: 11,321.
- Exact duplicate rows: 0; duplicate statements exist because other fields differ.
- Normalized-statement groups: 1,152 groups with more than one row.
- Conflicting-label normalized groups: 522.
- Cross-source normalized duplicate groups: 672.
- Rows belonging to a normalized duplicate group: 2,326.
- A source-majority lookup is correct for approximately 98.49% of all rows.
- Several large sources are label-pure; `MISLEADING` is also label-pure in this copy.
- Parsed dates include a large 2025 cluster even though the dissertation work
  predates that period. Date provenance and parsing must be resolved before a
  temporal claim is attempted.

The source-label association is a confounding warning, not a model result.
Because models use article text rather than the `Web` field, the exact degree to
which textual publisher signatures drive accuracy must be measured under a
source-disjoint protocol.

## Traceability of the dissertation results

| Component | What is observable in the notebook | Evidence status |
|---|---|---|
| BERT | A saved output reports 0.95 accuracy and 0.94 macro-F1 on 10,706 validation examples after a random split; later epochs show about 94.62-95.33% validation accuracy | Legacy output; not independently reproduced, no locked test set or repeated seeds |
| CNN | A saved output reports 0.9398 validation accuracy | Legacy output; conflicts with the dissertation's 88%/91% narrative values |
| LSTM | The saved classification report predicts no `Fake` examples: 0.71 accuracy and 0.41 macro-F1 | Legacy output; contradicts the dissertation's 92-93% claim |
| FNN | Architecture is defined, but no traceable training loop or measured evaluation is present | Unsupported result claim |
| Resampling | Dissertation describes large post-resampling improvements | No SMOTE, oversampling, or resampling implementation was found in notebook source |
| Five-fold cross-validation | Dissertation reports mean and standard deviation for four models | No `cross_val_score` or equivalent fold execution was found; the displayed chart uses hard-coded example values |
| Statistical hypothesis test | A t-test compares `Year` between labels | Does not test whether model performance differs; the statistical unit and hypothesis do not match the stated model-comparison claim |

## Non-evidentiary code that must not support a paper claim

Several cells explicitly use example, mock, or simulated values:

- training/validation loss and accuracy curves use manually entered example lists;
- six-statement confusion matrices use simulated labels and predictions;
- model-output confidence charts use mock probabilities;
- GloVe visualizations use random vectors in later cells;
- the cross-validation accuracy chart uses hard-coded means and standard deviations;
- a final classification-report figure is built from manually entered numbers.

These cells may remain in an output-stripped legacy archive for provenance, but
they are excluded from the reproducible pipeline and cannot be cited as results.

## Methodological risks found

1. No independent test set is preserved; validation data is repeatedly inspected.
2. Record-level random splitting does not control publisher/source shift.
3. Normalized duplicates can cross splits, including conflicting-label groups.
4. The preprocessing state changes repeatedly across 208 code cells and cannot
   be reproduced reliably by running selected cells.
5. Models do not share a clearly documented tuning budget or identical data.
6. Accuracy dominates the narrative despite class imbalance; macro-F1 and
   per-class failure are more informative.
7. Saved outputs have no run IDs, resolved configuration, environment lock,
   data checksum, or code revision.
8. The dataset paper and exact Kaggle card can be identified, but the Kaggle
   data card reports the dataset licence as `Unknown`; redistribution rights
   therefore remain unresolved.

## Reconstruction decision

The original reported model table is not copied into the repository as a
verified result. Version 0.2 establishes a data contract, duplicate policy,
random and source-disjoint splits, TF-IDF/FNN/CNN/LSTM/BERT-tiny model paths,
three prespecified seeds, saved predictions, independent metric verification,
and machine-readable aggregate results.

The neural reconstruction preserves each legacy architecture's broad family
while enforcing one controlled evaluation protocol. The FNN, CNN, and LSTM use
train-only learned embeddings rather than the notebook's incomplete or
inconsistent preprocessing state. BERT uses a pinned 2-layer BERT-tiny checkpoint
to make full repeated evaluation CPU-feasible; it is not a reproduction of the
legacy `bert-base-uncased` run. These decisions are explicit methodological
changes, not concealed equivalence claims.

Across the three verified holdouts, all four neural families have lower mean
macro-F1 and higher mean calibration error under source shift. The small sample
supports a descriptive reliability result, not a statistical-significance or
generally quantified domain-shift claim.

This approach preserves the work while preventing unsupported accuracy or
cross-validation claims from reaching a public PhD portfolio.
