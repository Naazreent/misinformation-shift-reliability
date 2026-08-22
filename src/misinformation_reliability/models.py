"""Fixed-budget diagnostic and text baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_models(config: dict[str, Any]) -> dict[str, tuple[Any, str]]:
    seed = int(config["seed"])
    logistic = config["logistic_regression"]
    tfidf = config["tfidf"]

    def classifier() -> LogisticRegression:
        return LogisticRegression(
            C=float(logistic["C"]),
            max_iter=int(logistic["max_iter"]),
            class_weight=logistic["class_weight"],
            random_state=seed,
            solver="liblinear",
        )

    source_only = Pipeline(
        steps=[
            (
                "source",
                ColumnTransformer(
                    [("one_hot", OneHotEncoder(handle_unknown="ignore"), ["Web"])],
                    remainder="drop",
                ),
            ),
            ("classifier", classifier()),
        ]
    )
    text_model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(int(tfidf["ngram_min"]), int(tfidf["ngram_max"])),
                    min_df=int(tfidf["min_df"]),
                    max_features=int(tfidf["max_features"]),
                    sublinear_tf=bool(tfidf["sublinear_tf"]),
                    dtype=np.float32,
                ),
            ),
            ("classifier", classifier()),
        ]
    )
    return {
        "majority": (DummyClassifier(strategy="prior", random_state=seed), "Statement"),
        "source_only_logreg": (source_only, "frame"),
        "tfidf_logreg": (text_model, "Statement"),
    }


def model_input(frame, feature_kind: str):
    if feature_kind == "frame":
        return frame[["Web"]]
    if feature_kind == "Statement":
        return frame["Statement"]
    raise ValueError(f"Unknown feature kind: {feature_kind}")


def positive_probability(model, features):
    probabilities = model.predict_proba(features)
    class_positions = {int(label): position for position, label in enumerate(model.classes_)}
    if 1 not in class_positions:
        raise ValueError("Model did not expose the positive class")
    return probabilities[:, class_positions[1]]
