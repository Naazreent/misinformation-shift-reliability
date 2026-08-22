# Data access and provenance

The raw dataset is not stored in Git.

The dissertation describes the input as the **IFND fake-news dataset**, obtained
from Kaggle and uploaded by **Sonal Garg**. The exact Kaggle URL, dataset version,
licence, and relationship to the original research release have not yet been
independently verified. Until they are, the CSV must not be redistributed from
this repository.

Place an authorised copy at `data/raw/IFND.csv`. The audit command validates the
schema and records the checksum. See
[`manifests/ifnd_manifest.json`](manifests/ifnd_manifest.json) for the expected
researcher-supplied copy.

The pipeline never edits the raw file. Duplicate controls and split assignments
are generated as run artifacts.

