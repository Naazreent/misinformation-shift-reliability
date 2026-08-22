# Data access and provenance

The raw dataset is not stored in Git.

## Identified source

- Kaggle data card: [IFND dataset by Sonal Garg](https://www.kaggle.com/datasets/sonalgarg174/ifnd-dataset)
- Dataset paper: [Sharma and Garg, *IFND: a benchmark dataset for fake news detection*](https://doi.org/10.1007/s40747-021-00552-1)
- Data-card licence shown on 22 August 2026: **Unknown**

The paper is published under CC BY 4.0, but that article licence does not by
itself establish a redistribution licence for the scraped news records, image
links, or the Kaggle CSV. The repository therefore provides code and a checksum,
not a copy of `IFND.csv`.

## Obtain an authorised copy

1. Open the Kaggle data card and review its current terms, licence field, and any
   access conditions. Confirm that your intended research use is permitted by
   Kaggle's terms and your institution's policies.
2. Sign in to Kaggle and download the dataset through the data-card interface;
   or, after configuring the official Kaggle CLI on your own machine, run:

   ```bash
   kaggle datasets download -d sonalgarg174/ifnd-dataset -p data/raw --unzip
   ```

3. Ensure the resulting file is named `data/raw/IFND.csv`.
4. Verify that it matches the copy used for the reported experiments:

   ```bash
   python scripts/audit_data.py --data data/raw/IFND.csv --output reports/data_audit.json
   ```

The expected SHA-256 is
`fafbb516c5d78a26ed1cfdf5077f989eae24c1258ad3c4de731ac6d6932505d6`.
If the checksum differs, treat the file as a different dataset version and do
not compare its results directly with the checked-in tables without recording
the new checksum.

Do not redistribute the CSV or recovered article/image content through forks,
releases, model artifacts, or supplementary files unless a rights holder grants
permission. The pipeline never edits the raw file; duplicate controls and split
assignments are generated as local run artifacts.
