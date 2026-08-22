"""Deterministic record-level and source-disjoint split protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitResult:
    assignments: pd.Series
    metadata: dict[str, Any]


def _validate_fractions(train: float, validation: float, test: float) -> None:
    if min(train, validation, test) <= 0:
        raise ValueError("All split fractions must be positive")
    if not np.isclose(train + validation + test, 1.0):
        raise ValueError("Split fractions must sum to 1")


def random_stratified_split(
    frame: pd.DataFrame,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> SplitResult:
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    indices = np.arange(len(frame))
    train_val, test = train_test_split(
        indices,
        test_size=test_fraction,
        random_state=seed,
        stratify=frame["Label"],
    )
    validation_within_train_val = validation_fraction / (train_fraction + validation_fraction)
    train, validation = train_test_split(
        train_val,
        test_size=validation_within_train_val,
        random_state=seed + 1,
        stratify=frame.iloc[train_val]["Label"],
    )
    assignments = pd.Series(index=frame.index, dtype="object", name="split")
    assignments.iloc[train] = "train"
    assignments.iloc[validation] = "validation"
    assignments.iloc[test] = "test"
    _validate_assignments(frame, assignments, require_source_disjoint=False)
    return SplitResult(
        assignments=assignments,
        metadata={
            "protocol": "random_stratified",
            "seed": seed,
            "fractions": {
                "train": train_fraction,
                "validation": validation_fraction,
                "test": test_fraction,
            },
        },
    )


def _best_group_holdout(
    frame: pd.DataFrame,
    holdout_fraction: float,
    seed: int,
    attempts: int = 1000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    target_rate = float(frame["Label"].mean())
    best: tuple[float, np.ndarray, np.ndarray, int] | None = None
    for offset in range(attempts):
        # Allocate a disjoint candidate-seed block to each prespecified seed.
        # This prevents repeated-seed evaluations from silently searching almost
        # the same candidate holdouts (for example seeds 13, 42, and 87).
        candidate_seed = seed * attempts + offset
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=holdout_fraction,
            random_state=candidate_seed,
        )
        retained, held_out = next(
            splitter.split(frame, y=frame["Label"], groups=frame["Web"])
        )
        retained_labels = frame.iloc[retained]["Label"]
        held_labels = frame.iloc[held_out]["Label"]
        if retained_labels.nunique() < 2 or held_labels.nunique() < 2:
            continue
        if retained_labels.value_counts().min() < 20 or held_labels.value_counts().min() < 20:
            continue
        size_error = abs(len(held_out) / len(frame) - holdout_fraction)
        prevalence_error = abs(float(held_labels.mean()) - target_rate)
        score = size_error + prevalence_error
        if best is None or score < best[0]:
            best = (score, retained, held_out, candidate_seed)
    if best is None:
        raise ValueError(
            "Could not construct a source-disjoint holdout containing both labels; "
            "inspect group-label confounding or adjust the protocol"
        )
    _, retained, held_out, selected_seed = best
    return retained, held_out, {
        "selected_seed": selected_seed,
        "candidate_attempts": attempts,
        "holdout_fraction_requested": holdout_fraction,
        "holdout_fraction_observed": float(len(held_out) / len(frame)),
        "holdout_true_rate": float(frame.iloc[held_out]["Label"].mean()),
    }


def source_disjoint_split(
    frame: pd.DataFrame,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
) -> SplitResult:
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    if frame["Web"].nunique() < 3:
        raise ValueError("Source-disjoint splitting needs at least three sources")

    train_val_positions, test_positions, test_meta = _best_group_holdout(
        frame, test_fraction, seed
    )
    train_val = frame.iloc[train_val_positions].copy()
    validation_within_train_val = validation_fraction / (train_fraction + validation_fraction)
    train_relative, validation_relative, validation_meta = _best_group_holdout(
        train_val,
        validation_within_train_val,
        seed + 10000,
    )
    train_positions = train_val_positions[train_relative]
    validation_positions = train_val_positions[validation_relative]

    assignments = pd.Series(index=frame.index, dtype="object", name="split")
    assignments.iloc[train_positions] = "train"
    assignments.iloc[validation_positions] = "validation"
    assignments.iloc[test_positions] = "test"
    _validate_assignments(frame, assignments, require_source_disjoint=True)

    sources = {
        split: sorted(frame.loc[assignments == split, "Web"].unique().tolist())
        for split in ("train", "validation", "test")
    }
    return SplitResult(
        assignments=assignments,
        metadata={
            "protocol": "source_disjoint",
            "seed": seed,
            "fractions": {
                "train": train_fraction,
                "validation": validation_fraction,
                "test": test_fraction,
            },
            "test_selection": test_meta,
            "validation_selection": validation_meta,
            "sources": sources,
        },
    )


def _validate_assignments(
    frame: pd.DataFrame,
    assignments: pd.Series,
    require_source_disjoint: bool,
) -> None:
    if assignments.isna().any():
        raise AssertionError("Every row must receive a split assignment")
    if set(assignments.unique()) != {"train", "validation", "test"}:
        raise AssertionError("Expected train, validation, and test assignments")
    for split in ("train", "validation", "test"):
        labels = frame.loc[assignments == split, "Label"]
        if labels.nunique() != 2:
            raise AssertionError(f"{split} does not contain both labels")

    if "text_group_hash" in frame.columns:
        split_count = (
            pd.DataFrame({"group": frame["text_group_hash"], "split": assignments})
            .groupby("group")["split"]
            .nunique()
        )
        if (split_count > 1).any():
            raise AssertionError("A normalized-text group crosses split boundaries")

    if require_source_disjoint:
        source_sets = [
            set(frame.loc[assignments == split, "Web"])
            for split in ("train", "validation", "test")
        ]
        if any(source_sets[i] & source_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise AssertionError("A source crosses source-disjoint split boundaries")


def make_split(frame: pd.DataFrame, config: dict[str, Any]) -> SplitResult:
    common = {
        "frame": frame,
        "seed": int(config["seed"]),
        "train_fraction": float(config["train_fraction"]),
        "validation_fraction": float(config["validation_fraction"]),
        "test_fraction": float(config["test_fraction"]),
    }
    protocol = config["split_protocol"]
    if protocol == "random_stratified":
        return random_stratified_split(**common)
    if protocol == "source_disjoint":
        return source_disjoint_split(**common)
    raise ValueError(f"Unknown split protocol: {protocol}")
