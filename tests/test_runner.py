import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from misinformation_reliability.runner import run_experiment


class RunnerTests(unittest.TestCase):
    def test_synthetic_smoke_run_writes_traceable_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for index in range(240):
                label = index % 2
                rows.append(
                    {
                        "id": index + 1,
                        "Statement": f"{'realistic' if label else 'misleading'} synthetic item {index}",
                        "Image": "none",
                        "Web": f"SOURCE_{index % 8}",
                        "Category": "SYNTHETIC",
                        "Date": "2020-01-01",
                        "Label": bool(label),
                    }
                )
            data_path = root / "synthetic.csv"
            pd.DataFrame(rows).to_csv(data_path, index=False)
            config = {
                "name": "unit_smoke",
                "seed": 42,
                "split_protocol": "random_stratified",
                "train_fraction": 0.70,
                "validation_fraction": 0.15,
                "test_fraction": 0.15,
                "deduplicate": True,
                "smoke_rows_per_class": 80,
                "tfidf": {
                    "ngram_min": 1,
                    "ngram_max": 1,
                    "min_df": 1,
                    "max_features": 500,
                    "sublinear_tf": True,
                },
                "logistic_regression": {
                    "C": 1.0,
                    "max_iter": 200,
                    "class_weight": "balanced",
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            run_dir = run_experiment(
                data_path=data_path,
                config_path=config_path,
                output_root=root / "artifacts",
                repository=root,
            )
            expected = {
                "run_record.json",
                "metrics.csv",
                "predictions.csv",
                "split_assignments.csv",
                "split_metadata.json",
                "summary.json",
                "config_resolved.json",
            }
            self.assertTrue(expected.issubset({path.name for path in run_dir.iterdir()}))
            record = json.loads((run_dir / "run_record.json").read_text())
            self.assertEqual(record["status"], "completed")


if __name__ == "__main__":
    unittest.main()

