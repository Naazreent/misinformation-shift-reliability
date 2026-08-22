"""Configuration-driven experiment execution and artifact recording."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from . import __version__
from .data import deduplicate_statements, load_ifnd, sha256_file
from .metrics import binary_metrics
from .models import build_models, model_input, positive_probability
from .splits import make_split


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "name",
        "seed",
        "split_protocol",
        "train_fraction",
        "validation_fraction",
        "test_fraction",
        "deduplicate",
        "smoke_rows_per_class",
        "tfidf",
        "logistic_regression",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    return config


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _git_revision(repository: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "uncommitted-or-unavailable"


def _smoke_sample(frame: pd.DataFrame, rows_per_class: int, seed: int) -> pd.DataFrame:
    samples = []
    for _, group in frame.groupby("Label", sort=True):
        samples.append(group.sample(n=min(rows_per_class, len(group)), random_state=seed))
    return (
        pd.concat(samples, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(
    data_path: str | Path,
    config_path: str | Path,
    output_root: str | Path = "artifacts",
    repository: str | Path | None = None,
) -> Path:
    started_clock = time.perf_counter()
    started_at = datetime.now(timezone.utc)
    data_path = Path(data_path).resolve()
    config_path = Path(config_path).resolve()
    repository_path = Path(repository or Path.cwd()).resolve()
    output_root = Path(output_root).resolve()
    config = load_config(config_path)
    seed = int(config["seed"])
    _set_seeds(seed)

    run_id = (
        f"{started_at.strftime('%Y%m%dT%H%M%SZ')}-"
        f"{config['name']}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config_resolved.json", config)

    status = "running"
    data_digest = sha256_file(data_path)
    try:
        frame = load_ifnd(data_path)
        raw_rows = len(frame)
        if config["deduplicate"]:
            frame, deduplication = deduplicate_statements(frame)
        else:
            from .data import add_text_groups

            frame = add_text_groups(frame)
            deduplication = {
                "input_rows": raw_rows,
                "conflicting_groups": 0,
                "rows_removed_as_conflicting": 0,
                "same_label_duplicate_rows_removed": 0,
                "output_rows": raw_rows,
            }

        smoke_rows = config.get("smoke_rows_per_class")
        if smoke_rows is not None:
            frame = _smoke_sample(frame, int(smoke_rows), seed)

        split_result = make_split(frame, config)
        frame = frame.copy()
        frame["split"] = split_result.assignments
        split_export = frame[["id", "Web", "Label", "text_group_hash", "split"]].rename(
            columns={"id": "row_id", "Web": "source", "Label": "label"}
        )
        split_export.to_csv(run_dir / "split_assignments.csv", index=False)
        _write_json(
            run_dir / "split_metadata.json",
            {
                **split_result.metadata,
                "rows": {key: int(value) for key, value in frame["split"].value_counts().items()},
                "label_rates": {
                    split: float(frame.loc[frame["split"] == split, "Label"].mean())
                    for split in ("train", "validation", "test")
                },
                "raw_rows": int(raw_rows),
                "experimental_rows": int(len(frame)),
                "deduplication": deduplication,
            },
        )

        train = frame.loc[frame["split"] == "train"]
        predictions: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}

        for model_name, (model, feature_kind) in build_models(config).items():
            fit_started = time.perf_counter()
            model.fit(model_input(train, feature_kind), train["Label"].to_numpy())
            model_seconds = time.perf_counter() - fit_started
            model_path = run_dir / f"model-{model_name}.joblib"
            joblib.dump(model, model_path)
            summaries[model_name] = {"fit_seconds": model_seconds, "splits": {}}

            for split in ("validation", "test"):
                evaluation = frame.loc[frame["split"] == split]
                features = model_input(evaluation, feature_kind)
                predicted = model.predict(features).astype(int)
                probability = positive_probability(model, features)
                metrics = binary_metrics(
                    evaluation["Label"].to_numpy(), predicted, probability
                )
                summaries[model_name]["splits"][split] = metrics
                for metric, value in metrics.items():
                    metric_rows.append(
                        {
                            "run_id": run_id,
                            "config_name": config["name"],
                            "split_protocol": config["split_protocol"],
                            "model": model_name,
                            "evaluation_split": split,
                            "seed": seed,
                            "metric": metric,
                            "value": value,
                        }
                    )
                predictions.append(
                    pd.DataFrame(
                        {
                            "run_id": run_id,
                            "row_id": evaluation["id"].to_numpy(),
                            "source": evaluation["Web"].to_numpy(),
                            "split_protocol": config["split_protocol"],
                            "evaluation_split": split,
                            "model": model_name,
                            "label": evaluation["Label"].to_numpy(),
                            "prediction": predicted,
                            "probability_true": probability,
                        }
                    )
                )

        pd.DataFrame(metric_rows).to_csv(run_dir / "metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(
            run_dir / "predictions.csv", index=False
        )
        _write_json(run_dir / "summary.json", summaries)
        status = "completed"
    except Exception:
        status = "failed"
        raise
    finally:
        finished_at = datetime.now(timezone.utc)
        record = {
            "run_id": run_id,
            "status": status,
            "code_revision": _git_revision(repository_path),
            "data_version": data_digest,
            "config": "config_resolved.json",
            "seed": seed,
            "system": {
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "package_version": __version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "scipy": scipy.__version__,
                "joblib": joblib.__version__,
            },
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": time.perf_counter() - started_clock,
            "metrics": "metrics.csv" if status == "completed" else None,
            "artifact_paths": [
                "config_resolved.json",
                "split_assignments.csv",
                "split_metadata.json",
                "metrics.csv",
                "predictions.csv",
                "summary.json",
            ]
            if status == "completed"
            else ["config_resolved.json"],
            "notes": "Smoke run" if config.get("smoke_rows_per_class") else "",
        }
        _write_json(run_dir / "run_record.json", record)
    return run_dir

