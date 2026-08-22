#!/usr/bin/env python3
"""Generate a tiny, non-scientific dataset for CI smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=600)
    args = parser.parse_args()
    sources = ["SOURCE_A", "SOURCE_B", "SOURCE_C", "SOURCE_D", "SOURCE_E", "SOURCE_F"]
    records = []
    for index in range(args.rows):
        label = index % 2
        source = sources[index % len(sources)]
        token = "verified-style" if label else "claim-style"
        records.append(
            {
                "id": index + 1,
                "Statement": f"Synthetic {token} statement number {index}",
                "Image": "none",
                "Web": source,
                "Category": "SYNTHETIC",
                "Date": f"2020-{(index % 12) + 1:02d}-01",
                "Label": bool(label),
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    print(output)


if __name__ == "__main__":
    main()

