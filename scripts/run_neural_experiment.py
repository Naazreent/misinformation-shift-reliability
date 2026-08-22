#!/usr/bin/env python3
"""Run one controlled neural reconstruction experiment."""

from __future__ import annotations

import argparse

from misinformation_reliability.neural_runner import run_neural_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--repository", default=".")
    args = parser.parse_args()
    print(
        run_neural_experiment(
            data_path=args.data,
            config_path=args.config,
            output_root=args.output_root,
            repository=args.repository,
        )
    )


if __name__ == "__main__":
    main()
