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
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="Specific completed run directory; repeat for multiple runs",
    )
    args = parser.parse_args()

    files = [Path(path) / "metrics.csv" for path in args.run_dir]
    if not files:
        files = sorted(Path(args.artifacts).glob("*/metrics.csv"))
    if not args.include_smoke:
        files = [path for path in files if "smoke" not in path.parent.name]
    if not files:
        raise SystemExit("No completed metrics.csv artifacts found")
    if not args.run_dir:
        latest_by_config = {}
        for path in files:
            header = pd.read_csv(path, nrows=1)
            latest_by_config[str(header.loc[0, "config_name"])] = path
        files = sorted(latest_by_config.values())
    metrics = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
