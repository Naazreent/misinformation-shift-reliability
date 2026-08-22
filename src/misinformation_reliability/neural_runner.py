"""Configuration-driven neural experiments with traceable artifacts."""

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

import numpy as np
import pandas as pd
import torch
import transformers

from . import __version__
from .data import add_text_groups, deduplicate_statements, load_ifnd, sha256_file
from .metrics import binary_metrics
from .neural import (
    Vocabulary,
    build_word_model,
    predict_bert_model,
    predict_word_model,
    train_bert_model,
    train_word_model,
)
from .splits import make_split


SUPPORTED_MODELS = ("fnn", "cnn", "lstm", "bert_tiny")


def load_neural_config(path: str | Path) -> dict[str, Any]:
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
        "neural",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Configuration is missing keys: {missing}")
    models = list(config["neural"].get("models", []))
    if not models or any(model not in SUPPORTED_MODELS for model in models):
        raise ValueError(
            f"neural.models must be a non-empty subset of {list(SUPPORTED_MODELS)}"
        )
    return config


def _set_seeds(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def _append_evaluation(
    predictions: list[pd.DataFrame],
    metric_rows: list[dict[str, Any]],
    run_id: str,
    config: dict[str, Any],
    model_name: str,
    evaluation: pd.DataFrame,
    split: str,
    predicted: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(predicted, dtype=np.int64)
    # Promote float32 model outputs before metric computation and CSV writing so
    # independent verification sees the exact same probability values.
    probability = np.asarray(probability, dtype=np.float64)
    metrics = binary_metrics(evaluation["Label"].to_numpy(), predicted, probability)
    for metric, value in metrics.items():
        metric_rows.append(
            {
                "run_id": run_id,
                "config_name": config["name"],
                "split_protocol": config["split_protocol"],
                "model": model_name,
                "evaluation_split": split,
                "seed": int(config["seed"]),
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
    return metrics


def run_neural_experiment(
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
    config = load_neural_config(config_path)
    seed = int(config["seed"])
    neural_config = config["neural"]
    _set_seeds(seed, int(neural_config.get("torch_num_threads", os.cpu_count() or 1)))
    device = torch.device("cpu")

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
                "rows": {
                    key: int(value) for key, value in frame["split"].value_counts().items()
                },
                "label_rates": {
                    split: float(frame.loc[frame["split"] == split, "Label"].mean())
                    for split in ("train", "validation", "test")
                },
                "raw_rows": int(raw_rows),
                "experimental_rows": int(len(frame)),
                "deduplication": deduplication,
            },
        )

        partitions = {
            split: frame.loc[frame["split"] == split].reset_index(drop=True)
            for split in ("train", "validation", "test")
        }
        predictions: list[pd.DataFrame] = []
        metric_rows: list[dict[str, Any]] = []
        summaries: dict[str, Any] = {}

        word_models = [
            model for model in neural_config["models"] if model in {"fnn", "cnn", "lstm"}
        ]
        if word_models:
            vocabulary = Vocabulary.build(
                partitions["train"]["Statement"].tolist(),
                max_size=int(neural_config["max_vocab"]),
                min_frequency=int(neural_config["min_token_frequency"]),
            )
            max_length = int(neural_config["max_length"])
            encoded = {
                split: vocabulary.encode_many(
                    partitions[split]["Statement"].tolist(), max_length
                )
                for split in ("train", "validation", "test")
            }
            _write_json(
                run_dir / "vocabulary_metadata.json",
                {
                    "size": len(vocabulary),
                    "max_length": max_length,
                    "min_token_frequency": int(neural_config["min_token_frequency"]),
                    "fit_partition": "train",
                    "tokenizer": "unicode_nfkc_casefold_word_tokens_v1",
                },
            )
            for model_name in word_models:
                _set_seeds(
                    seed,
                    int(neural_config.get("torch_num_threads", os.cpu_count() or 1)),
                )
                model = build_word_model(model_name, len(vocabulary), neural_config)
                model, training_summary = train_word_model(
                    model,
                    encoded["train"],
                    partitions["train"]["Label"].to_numpy(),
                    encoded["validation"],
                    partitions["validation"]["Label"].to_numpy(),
                    neural_config,
                    seed,
                    device,
                )
                checkpoint = run_dir / f"model-{model_name}.pt"
                torch.save(model.state_dict(), checkpoint)
                summaries[model_name] = {**training_summary, "splits": {}}
                for split in ("validation", "test"):
                    predicted, probability, _ = predict_word_model(
                        model,
                        encoded[split],
                        partitions[split]["Label"].to_numpy(),
                        int(neural_config["batch_size"]),
                        device,
                    )
                    summaries[model_name]["splits"][split] = _append_evaluation(
                        predictions,
                        metric_rows,
                        run_id,
                        config,
                        model_name,
                        partitions[split],
                        split,
                        predicted,
                        probability,
                    )
                del model

        if "bert_tiny" in neural_config["models"]:
            _set_seeds(
                seed,
                int(neural_config.get("torch_num_threads", os.cpu_count() or 1)),
            )
            model, tokenizer, _, validation_encodings, training_summary = train_bert_model(
                partitions["train"]["Statement"].tolist(),
                partitions["train"]["Label"].to_numpy(),
                partitions["validation"]["Statement"].tolist(),
                partitions["validation"]["Label"].to_numpy(),
                neural_config,
                seed,
                device,
            )
            checkpoint = run_dir / "model-bert_tiny.pt"
            torch.save(model.state_dict(), checkpoint)
            summaries["bert_tiny"] = {**training_summary, "splits": {}}
            for split in ("validation", "test"):
                if split == "validation":
                    encodings = validation_encodings
                else:
                    encodings = tokenizer(
                        partitions[split]["Statement"].tolist(),
                        padding="max_length",
                        truncation=True,
                        max_length=int(neural_config["bert_max_length"]),
                        return_tensors="pt",
                    )
                predicted, probability, _ = predict_bert_model(
                    model,
                    encodings,
                    partitions[split]["Label"].to_numpy(),
                    int(neural_config["bert_batch_size"]),
                    device,
                )
                summaries["bert_tiny"]["splits"][split] = _append_evaluation(
                    predictions,
                    metric_rows,
                    run_id,
                    config,
                    "bert_tiny",
                    partitions[split],
                    split,
                    predicted,
                    probability,
                )
            del model

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
        artifact_paths = [
            "config_resolved.json",
            "split_assignments.csv",
            "split_metadata.json",
        ]
        if status == "completed":
            artifact_paths.extend(["metrics.csv", "predictions.csv", "summary.json"])
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
                "torch_threads": torch.get_num_threads(),
                "device": str(device),
                "package_version": __version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "deterministic_algorithms": True,
            },
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_seconds": time.perf_counter() - started_clock,
            "metrics": "metrics.csv" if status == "completed" else None,
            "artifact_paths": artifact_paths,
            "checkpoints_local_only": [
                f"model-{name}.pt" for name in neural_config.get("models", [])
            ]
            if status == "completed"
            else [],
            "notes": "Smoke run" if config.get("smoke_rows_per_class") else "",
        }
        _write_json(run_dir / "run_record.json", record)
    return run_dir
