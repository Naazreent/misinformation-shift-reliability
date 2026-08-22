#!/usr/bin/env python3
"""Aggregate completed run metrics into a comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="reports/results/baseline_metrics.csv")
    args = parser.parse_args()

    files = sorted(Path(args.artifacts).glob("*/metrics.csv"))
    if not files:
        raise SystemExit("No completed metrics.csv artifacts found")
    metrics = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()

