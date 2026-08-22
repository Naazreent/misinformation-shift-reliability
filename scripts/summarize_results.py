#!/usr/bin/env python3
"""Summarize repeated-seed metrics and protocol differences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


def interval95(values: pd.Series) -> tuple[float, float]:
    numeric = values.astype(float).to_numpy()
    mean = float(np.mean(numeric))
    if len(numeric) < 2:
        return mean, mean
    half_width = float(t.ppf(0.975, len(numeric) - 1) * np.std(numeric, ddof=1) / np.sqrt(len(numeric)))
    return mean - half_width, mean + half_width


def summarize(group: pd.DataFrame) -> pd.Series:
    values = group["value"].astype(float)
    low, high = interval95(values)
    return pd.Series(
        {
            "n_seeds": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "ci95_low_t": low,
            "ci95_high_t": high,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="reports/results/all_metrics.csv")
    parser.add_argument(
        "--summary", default="reports/results/comparison_summary.csv"
    )
    parser.add_argument("--gaps", default="reports/results/protocol_gaps.csv")
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    test = metrics.loc[metrics["evaluation_split"] == "test"].copy()
    duplicate_keys = ["split_protocol", "model", "seed", "metric"]
    if test.duplicated(duplicate_keys).any():
        duplicates = test.loc[test.duplicated(duplicate_keys, keep=False), duplicate_keys]
        raise ValueError(f"Duplicate test metric rows found:\n{duplicates.head()}")
    summary = (
        test.groupby(["split_protocol", "model", "metric"], sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    pivot = test.pivot(
        index=["model", "metric", "seed"],
        columns="split_protocol",
        values="value",
    ).reset_index()
    required = {"random_stratified", "source_disjoint"}
    if not required.issubset(pivot.columns):
        raise ValueError("Both protocols are required for protocol-gap summaries")
    pivot["value"] = pivot["source_disjoint"] - pivot["random_stratified"]
    gaps = (
        pivot.groupby(["model", "metric"], sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
        .rename(
            columns={
                "mean": "source_minus_random_mean",
                "std": "source_minus_random_std",
                "minimum": "source_minus_random_minimum",
                "maximum": "source_minus_random_maximum",
                "ci95_low_t": "source_minus_random_ci95_low_t",
                "ci95_high_t": "source_minus_random_ci95_high_t",
            }
        )
    )
    gap_path = Path(args.gaps)
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(gap_path, index=False)
    print(summary_path)
    print(gap_path)


if __name__ == "__main__":
    main()
