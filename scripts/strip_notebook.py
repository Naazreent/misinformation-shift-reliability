#!/usr/bin/env python3
"""Create an output-stripped preservation copy of a Jupyter notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook.setdefault("metadata", {})["reconstruction_note"] = (
        "Legacy Colab preserved for provenance. Outputs removed; do not use as "
        "the reproduction entry point."
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()

