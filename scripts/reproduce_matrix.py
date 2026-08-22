#!/usr/bin/env python3
"""Run and verify the prespecified repeated-seed five-model matrix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from misinformation_reliability.runner import run_experiment


def resolved_config(base_path: Path, seed: int, destination: Path) -> Path:
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config["seed"] = seed
    config["name"] = f"{config['name']}_s{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{base_path.stem}-seed-{seed}.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", default="artifacts")
    parser.add_argument("--seeds", default="13,42,87")
    parser.add_argument("--skip-neural", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    data_path = Path(args.data).resolve()
    output_root = (repository / args.output_root).resolve()
    seed_values = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if len(seed_values) < 3 or len(seed_values) != len(set(seed_values)):
        raise ValueError("Provide at least three distinct prespecified seeds")
    config_cache = output_root / "_matrix_configs"
    run_dirs: list[Path] = []
    run_kinds: list[str] = []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository / "src")

    if not args.skip_neural:
        from misinformation_reliability.neural_runner import run_neural_experiment

    for seed in seed_values:
        for protocol in ("random_stratified", "source_disjoint"):
            classical_config = resolved_config(
                repository / "configs" / f"{protocol}.json", seed, config_cache
            )
            classical_dir = run_experiment(
                data_path=data_path,
                config_path=classical_config,
                output_root=output_root,
                repository=repository,
            )
            run_dirs.append(classical_dir)
            run_kinds.append("classical")
            if not args.skip_neural:
                neural_config = resolved_config(
                    repository / "configs" / f"neural_{protocol}.json",
                    seed,
                    config_cache,
                )
                neural_dir = run_neural_experiment(
                    data_path=data_path,
                    config_path=neural_config,
                    output_root=output_root,
                    repository=repository,
                )
                run_dirs.append(neural_dir)
                run_kinds.append("neural")

    for run_dir in run_dirs:
        subprocess.check_call(
            [sys.executable, str(repository / "scripts" / "verify_run.py"), str(run_dir)],
            cwd=repository,
            env=environment,
        )

    metrics = pd.concat(
        (pd.read_csv(run_dir / "metrics.csv") for run_dir in run_dirs),
        ignore_index=True,
    )
    results_dir = repository / "reports" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "all_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    subprocess.check_call(
        [
            sys.executable,
            str(repository / "scripts" / "summarize_results.py"),
            "--metrics",
            str(metrics_path),
            "--summary",
            str(results_dir / "comparison_summary.csv"),
            "--gaps",
            str(results_dir / "protocol_gaps.csv"),
            "--expected-seeds",
            ",".join(str(seed) for seed in seed_values),
        ],
        cwd=repository,
        env=environment,
    )
    subprocess.check_call(
        [
            sys.executable,
            str(repository / "scripts" / "plot_results.py"),
            "--summary",
            str(results_dir / "comparison_summary.csv"),
            "--output",
            str(repository / "reports" / "figures" / "repeated_seed_model_comparison.png"),
        ],
        cwd=repository,
        env=environment,
    )

    records = [
        json.loads((run_dir / "run_record.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    resolved_configs = [
        json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    source_selection_seeds: dict[str, int] = {}
    for run_dir, config in zip(run_dirs, resolved_configs, strict=True):
        if config["split_protocol"] != "source_disjoint":
            continue
        metadata = json.loads(
            (run_dir / "split_metadata.json").read_text(encoding="utf-8")
        )
        seed_key = str(config["seed"])
        selected_seed = int(metadata["test_selection"]["selected_seed"])
        previous = source_selection_seeds.setdefault(seed_key, selected_seed)
        if previous != selected_seed:
            raise ValueError(
                f"Classical/neural source splits differ for seed {seed_key}"
            )
    manifest = {
        "evidence_status": "verified_repeated_seed_matrix",
        "data_sha256": records[0]["data_version"],
        "code_revision": records[0]["code_revision"],
        "seeds": seed_values,
        "protocols": ["random_stratified", "source_disjoint"],
        "models": [
            "majority",
            "source_only_logreg",
            "tfidf_logreg",
            "fnn",
            "cnn",
            "lstm",
            "bert_tiny",
        ]
        if not args.skip_neural
        else ["majority", "source_only_logreg", "tfidf_logreg"],
        "runs": [
            {
                "run_id": record["run_id"],
                "kind": kind,
                "protocol": config["split_protocol"],
                "seed": record["seed"],
                "models": ["majority", "source_only_logreg", "tfidf_logreg"]
                if kind == "classical"
                else ["fnn", "cnn", "lstm", "bert_tiny"],
                "status": record["status"],
                "metrics_recomputed_from_predictions": True,
            }
            for record, kind, config in zip(
                records, run_kinds, resolved_configs, strict=True
            )
        ],
        "source_disjoint_test_selection_seeds": source_selection_seeds,
        "limitations": [
            "Three prespecified seeds provide descriptive uncertainty but limited inferential power.",
            "BERT is represented by a pinned 2-layer BERT checkpoint for CPU-feasible reconstruction, not bert-base-uncased.",
            "Source-disjoint repetitions use distinct candidate-source blocks but only three selected holdouts.",
            "The Kaggle dataset licence is marked Unknown; the CSV is not redistributed.",
        ],
    }
    (results_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("completed", *(str(path) for path in run_dirs), sep="\n")


if __name__ == "__main__":
    main()
