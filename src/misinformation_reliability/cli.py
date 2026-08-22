"""Console entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import audit_ifnd, load_ifnd, sha256_file
from .runner import run_experiment


def _audit_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the supplied IFND CSV")
    parser.add_argument("--data", required=True, help="Path to IFND.csv")
    parser.add_argument("--output", default="reports/data_audit.json")
    return parser


def audit_main(argv: list[str] | None = None) -> None:
    args = _audit_parser().parse_args(argv)
    data_path = Path(args.data)
    output_path = Path(args.output)
    payload = audit_ifnd(load_ifnd(data_path))
    payload["file_name"] = data_path.name
    payload["sha256"] = sha256_file(data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a reproducible baseline experiment")
    parser.add_argument("--data", required=True, help="Path to IFND.csv")
    parser.add_argument("--config", required=True, help="Path to experiment JSON")
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--repository", default=".")
    return parser


def run_main(argv: list[str] | None = None) -> None:
    args = _run_parser().parse_args(argv)
    run_dir = run_experiment(
        data_path=args.data,
        config_path=args.config,
        output_root=args.output_root,
        repository=args.repository,
    )
    print(run_dir)

