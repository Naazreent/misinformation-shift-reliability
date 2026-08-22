# Ethics, intended use, and limitations

## Intended use

This repository is for research on evaluation reliability, dataset artifacts,
distribution shift, and probability calibration in misinformation benchmarks.

## Out-of-scope use

Do not use its models to:

- determine whether a real-world claim is objectively true;
- automatically remove, rank, or moderate live content;
- assess a person, publisher, political group, or community;
- make legal, employment, immigration, medical, or financial decisions;
- replace evidence gathering or professional fact-checking.

## Construct validity

`Label` represents the dataset's annotation and collection process. It may
encode source type, publisher reputation, topic, time, or augmentation choices
rather than only statement veracity. A high benchmark score therefore does not
establish real-world truth-detection ability.

## Dataset risks

- Source and label are strongly associated.
- Duplicate statements may have conflicting labels.
- Dates are missing or potentially misparsed.
- Publicly accessible text can still be copyrighted.
- The exact Kaggle data card and paper DOI are recorded, but the data-card
  licence field is `Unknown`; the CSV is therefore not redistributed.
- News statements may contain distressing, political, violent, or health-related
  content and may repeat false claims.

## Model risks

- Models may memorise source, topic, named entities, or annotation style.
- Calibration can fail under source or temporal shift.
- Subgroup performance is not yet established.
- Random-split performance may substantially overstate generalisation.
- The reconstructed BERT-tiny model is much smaller than the dissertation's
  legacy BERT-base configuration and must not be presented as an exact replica.
- Three seeds and three selected source holdouts quantify limited descriptive
  variability; they do not support a broad significance or robustness claim.
- Even abstaining models do not provide evidence for a factual conclusion.

## Reporting rules

- Label legacy metrics as historical and unverified.
- Report macro-F1, per-class metrics, variability, and source-disjoint results.
- Publish negative and failed runs.
- Do not claim novelty without a documented literature review.
- Do not publish the raw CSV until its licence and redistribution rights are clear.
- When sharing reproduction instructions, direct users to obtain their own
  authorised copy and verify its checksum; do not mirror the CSV in releases or
  supplementary archives.
