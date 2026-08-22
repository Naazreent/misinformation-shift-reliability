#!/usr/bin/env python3
"""Generate the baseline comparison figure from machine-readable metrics."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "misinfo-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABELS = {
    "majority": "Majority",
    "source_only_logreg": "Source only",
    "tfidf_logreg": "TF-IDF text",
    "fnn": "FNN",
    "cnn": "CNN",
    "lstm": "LSTM",
    "bert_tiny": "BERT-tiny",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="reports/results/comparison_summary.csv")
    parser.add_argument(
        "--output", default="reports/figures/baseline_protocol_comparison.png"
    )
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    protocols = ["random_stratified", "source_disjoint"]
    models = ["tfidf_logreg", "fnn", "cnn", "lstm", "bert_tiny"]
    colors = {"random_stratified": "#335C81", "source_disjoint": "#D17A22"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    x = np.arange(len(models))
    width = 0.36
    for index, protocol in enumerate(protocols):
        offset = (index - 0.5) * width
        for axis, metric, title, direction in [
            (axes[0], "macro_f1", "Predictive performance", "Higher is better"),
            (axes[1], "ece_10", "Calibration error", "Lower is better"),
        ]:
            values = []
            errors = []
            for model in models:
                selected = summary.loc[
                    (summary["split_protocol"] == protocol)
                    & (summary["model"] == model)
                    & (summary["metric"] == metric)
                ]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one {metric} value for {protocol}/{model}; "
                        f"found {len(selected)}"
                    )
                values.append(float(selected.iloc[0]["mean"]))
                errors.append(float(selected.iloc[0]["std"]))
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=protocol.replace("_", " ").title(),
                color=colors[protocol],
                yerr=errors,
                capsize=3,
            )
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
            axis.set_title(f"{title}\n{direction}", fontsize=11, weight="bold")
            axis.set_xticks(x, [LABELS[model] for model in models])
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Test macro-F1")
    axes[0].set_ylim(0, 1.08)
    axes[1].set_ylabel("ECE (10 equal-width bins)")
    axes[1].set_ylim(0, 0.26)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="outside lower center",
        ncol=2,
    )
    fig.suptitle(
        "IFND model reliability across three prespecified seeds",
        fontsize=13,
        weight="bold",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
