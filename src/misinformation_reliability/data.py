"""Dataset validation, normalization, deduplication, and audit helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "id",
    "Statement",
    "Image",
    "Web",
    "Category",
    "Date",
    "Label",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_binary_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("int8")
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="raise")
        values = set(numeric.dropna().astype(int).unique().tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"Label must be binary; found {sorted(values)}")
        return numeric.astype("int8")

    mapping = {
        "false": 0,
        "0": 0,
        "fake": 0,
        "true": 1,
        "1": 1,
        "real": 1,
    }
    normalized = series.astype(str).str.strip().str.casefold()
    unknown = sorted(set(normalized.unique()) - set(mapping))
    if unknown:
        raise ValueError(f"Unrecognized Label values: {unknown}")
    return normalized.map(mapping).astype("int8")


def load_ifnd(path: str | Path) -> pd.DataFrame:
    """Load IFND and enforce the repository's minimal data contract."""

    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("id must be non-missing and unique")
    if frame["Statement"].isna().any():
        raise ValueError("Statement contains missing values")

    frame = frame.copy()
    frame["Label"] = _coerce_binary_labels(frame["Label"])
    frame["Statement"] = frame["Statement"].astype(str)
    frame["Web"] = frame["Web"].astype(str).str.strip()
    frame["Category"] = frame["Category"].astype(str).str.strip()
    if (frame["Statement"].str.strip() == "").any():
        raise ValueError("Statement contains empty text")
    if (frame["Web"] == "").any():
        raise ValueError("Web contains empty source names")
    return frame


def normalize_statement(text: str) -> str:
    """Normalize text only for duplicate grouping, not model input."""

    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    normalized = re.sub(r"\W+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def add_text_groups(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_normalized_statement"] = result["Statement"].map(normalize_statement)
    if (result["_normalized_statement"] == "").any():
        raise ValueError("Normalization produced an empty statement")
    result["text_group_hash"] = result["_normalized_statement"].map(
        lambda value: hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()
    )
    return result


def deduplicate_statements(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove conflicting normalized groups, then retain one row per text."""

    grouped = add_text_groups(frame)
    label_counts = grouped.groupby("text_group_hash", sort=False)["Label"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index)
    conflict_mask = grouped["text_group_hash"].isin(conflict_hashes)
    without_conflicts = grouped.loc[~conflict_mask].copy()
    before_keep_first = len(without_conflicts)
    without_conflicts = without_conflicts.sort_values("id", kind="stable")
    deduplicated = without_conflicts.drop_duplicates("text_group_hash", keep="first")
    stats = {
        "input_rows": int(len(grouped)),
        "conflicting_groups": int(len(conflict_hashes)),
        "rows_removed_as_conflicting": int(conflict_mask.sum()),
        "same_label_duplicate_rows_removed": int(before_keep_first - len(deduplicated)),
        "output_rows": int(len(deduplicated)),
    }
    return deduplicated.reset_index(drop=True), stats


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def audit_ifnd(frame: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable audit statistics from a validated frame."""

    grouped = add_text_groups(frame)
    text_groups = grouped.groupby("text_group_hash", sort=False).agg(
        rows=("Label", "size"),
        labels=("Label", "nunique"),
        sources=("Web", "nunique"),
    )
    source_majority = grouped.groupby("Web")["Label"].transform(
        lambda values: int(values.mode().iloc[0])
    )
    parsed_dates = pd.to_datetime(grouped["Date"], errors="coerce", format="mixed")
    year_counts = parsed_dates.dt.year.value_counts(dropna=False).sort_index()

    source_rows = []
    for source, source_frame in grouped.groupby("Web", sort=True):
        source_rows.append(
            {
                "source": source,
                "rows": int(len(source_frame)),
                "true_rate": float(source_frame["Label"].mean()),
                "label_count": int(source_frame["Label"].nunique()),
            }
        )

    category_rows = []
    for category, category_frame in grouped.groupby("Category", sort=True):
        category_rows.append(
            {
                "category": category,
                "rows": int(len(category_frame)),
                "true_rate": float(category_frame["Label"].mean()),
                "label_count": int(category_frame["Label"].nunique()),
            }
        )

    years: dict[str, int] = {}
    for year, count in year_counts.items():
        key = "missing_or_invalid" if pd.isna(year) else str(int(year))
        years[key] = int(count)

    return {
        "rows": int(len(grouped)),
        "columns": list(grouped.columns[: len(frame.columns)]),
        "label_counts": {
            str(int(label)): int(count)
            for label, count in grouped["Label"].value_counts().sort_index().items()
        },
        "recorded_sources": int(grouped["Web"].nunique()),
        "recorded_categories": int(grouped["Category"].nunique()),
        "missing_values": {
            column: int(count) for column, count in frame.isna().sum().items()
        },
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "normalized_duplicate_groups": int((text_groups["rows"] > 1).sum()),
        "rows_in_normalized_duplicate_groups": int(
            text_groups.loc[text_groups["rows"] > 1, "rows"].sum()
        ),
        "conflicting_label_groups": int((text_groups["labels"] > 1).sum()),
        "cross_source_duplicate_groups": int(
            ((text_groups["rows"] > 1) & (text_groups["sources"] > 1)).sum()
        ),
        "source_majority_lookup_accuracy": float((source_majority == grouped["Label"]).mean()),
        "statement_length_characters": {
            "minimum": int(grouped["Statement"].str.len().min()),
            "median": float(grouped["Statement"].str.len().median()),
            "maximum": int(grouped["Statement"].str.len().max()),
        },
        "parsed_date_min": None if parsed_dates.notna().sum() == 0 else parsed_dates.min().isoformat(),
        "parsed_date_max": None if parsed_dates.notna().sum() == 0 else parsed_dates.max().isoformat(),
        "parsed_year_counts": years,
        "sources": source_rows,
        "categories": category_rows,
    }

