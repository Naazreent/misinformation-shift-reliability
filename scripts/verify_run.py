#!/usr/bin/env python3
"""Independently recompute every saved metric from per-example predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from misinformation_reliability.metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    predictions = pd.read_csv(run_dir / "predictions.csv")
    recorded = pd.read_csv(run_dir / "metrics.csv")

    differences = []
    for (model, split), group in predictions.groupby(
        ["model", "evaluation_split"], sort=True
    ):
        recomputed = binary_metrics(
            group["label"].to_numpy(),
            group["prediction"].to_numpy(),
            group["probability_true"].to_numpy(),
        )
        expected = recorded.loc[
            (recorded["model"] == model)
            & (recorded["evaluation_split"] == split)
        ].set_index("metric")["value"]
        for metric, value in recomputed.items():
            if metric not in expected:
                differences.append((model, split, metric, "missing", value))
                continue
            delta = abs(float(expected.loc[metric]) - float(value))
            if not np.isfinite(delta) or delta > args.tolerance:
                differences.append((model, split, metric, float(expected.loc[metric]), value))

    if differences:
        for difference in differences:
            print(difference)
        raise SystemExit(f"Metric verification failed: {len(differences)} differences")
    print(f"verified {run_dir}")


if __name__ == "__main__":
    main()

