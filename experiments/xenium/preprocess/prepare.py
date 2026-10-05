#!/usr/bin/env python3
"""Convert the Xenium Liver raw bundle into aligned CellAttention inputs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import io, sparse


def lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not values:
        raise ValueError(f"empty input: {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    raw, output = args.raw_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    cells = lines(raw / "cell_name.txt")
    genes = lines(raw / "gene_list.txt")
    groups = lines(raw / "annotation.csv")
    if len(set(cells)) != len(cells) or len(set(genes)) != len(genes):
        raise ValueError("cell and gene IDs must be unique")
    if len(groups) != len(cells):
        raise ValueError("annotation.csv must have one nonempty row per cell")
    samples = []
    for cell in cells:
        sample, separator, _ = cell.partition("_")
        if not sample or not separator:
            raise ValueError(f"expected '<sample>_<cell>' ID: {cell}")
        samples.append(sample)
    expression = sparse.load_npz(raw / "gene_expression.npz").tocsr()
    if expression.shape[0] != len(cells) and expression.shape[1] == len(cells):
        expression = expression.transpose().tocsr()
    if expression.shape != (len(cells), len(genes)):
        raise ValueError(f"expression shape {expression.shape} != {(len(cells), len(genes))}")
    if expression.data.size and (not np.isfinite(expression.data).all() or expression.data.min() < 0):
        raise ValueError("expression values must be finite and nonnegative")
    coordinates = np.load(raw / "spatial_coordinates.npy")
    if coordinates.shape != (len(cells), 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"invalid spatial coordinates shape or values: {coordinates.shape}")
    output.mkdir(parents=True, exist_ok=True)
    io.mmwrite(output / "expression.mtx", expression)
    np.savetxt(output / "spatial_coordinates.csv", coordinates, delimiter=",")
    for name, values in (
        ("cell_ids.txt", cells), ("gene_ids.txt", genes),
        ("sample_ids.txt", samples), ("evaluation_cell_groups.txt", groups),
        ("evaluation_cell_groups_initial.txt", groups),
    ):
        (output / name).write_text("\n".join(values) + "\n", encoding="utf-8")
    summary = {
        "dataset": "Xenium_Liver", "cells": len(cells), "genes": len(genes),
        "nonzeros": int(expression.nnz), "samples": dict(sorted(Counter(samples).items(), key=lambda item: (-item[1], item[0]))),
        "groups": len(set(groups)),
    }
    (output / "preprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
