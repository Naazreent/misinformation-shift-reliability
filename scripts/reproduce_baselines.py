#!/usr/bin/env python3
"""Run the complete deterministic baseline reproduction workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from misinformation_reliability.cli import audit_main
from misinformation_reliability.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", default="artifacts")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    data_path = Path(args.data).resolve()
    output_root = (repository / args.output_root).resolve()

    audit_main(
        [
            "--data",
            str(data_path),
            "--output",
            str(repository / "reports" / "data_audit.json"),
        ]
    )
    run_dirs = []
    for config_name in ("random_stratified.json", "source_disjoint.json"):
        run_dirs.append(
            run_experiment(
                data_path=data_path,
                config_path=repository / "configs" / config_name,
                output_root=output_root,
                repository=repository,
            )
        )

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")
    for run_dir in run_dirs:
        subprocess.check_call(
            [sys.executable, str(repository / "scripts" / "verify_run.py"), str(run_dir)],
            cwd=repository,
            env=environment,
        )
    records = [
        json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    manifest = {
        "evidence_status": "verified_single_seed_baseline",
        "data_sha256": records[0]["data_version"],
        "code_revision": records[0]["code_revision"],
        "runs": [
            {
                "run_id": record["run_id"],
                "protocol": config_name,
                "status": record["status"],
                "metrics_recomputed_from_predictions": True,
            }
            for record, config_name in zip(
                records, ("random_stratified", "source_disjoint"), strict=True
            )
        ],
        "limitations": [
            "One seed and one selected source holdout",
            "No confidence intervals",
            "Neural architectures not yet reconstructed",
            "Dataset source URL and licence still unverified",
        ],
    }
    manifest_path = repository / "reports" / "results" / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    aggregate_command = [
        sys.executable,
        str(repository / "scripts" / "aggregate_results.py"),
        "--artifacts",
        str(output_root),
        "--output",
        str(repository / "reports" / "results" / "baseline_metrics.csv"),
    ]
    for run_dir in run_dirs:
        aggregate_command.extend(["--run-dir", str(run_dir)])
    subprocess.check_call(
        aggregate_command,
        cwd=repository,
        env=environment,
    )
    subprocess.check_call(
        [
            sys.executable,
            str(repository / "scripts" / "plot_results.py"),
            "--metrics",
            str(repository / "reports" / "results" / "baseline_metrics.csv"),
            "--output",
            str(repository / "reports" / "figures" / "baseline_protocol_comparison.png"),
        ],
        cwd=repository,
        env=environment,
    )
    print("completed", *(str(path) for path in run_dirs), sep="\n")


if __name__ == "__main__":
    main()
