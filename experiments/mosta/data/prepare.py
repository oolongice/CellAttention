#!/usr/bin/env python3
"""Prepare the bundled MOSTA data for CellAttention."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse
from scipy.spatial import cKDTree


SCRIPT_DIR = Path(__file__).resolve().parent
GRAPH_KNN = 20
GRAPH_SIGMA = 50.0
MAX_NEIGHBOR_DISTANCE = 150.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MOSTA raw files into CellAttention inputs.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"required input file not found: {path}")
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"input file is empty: {path}")
    return values


def require_unique(values: list[str], label: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(
            f"{label} contains {len(duplicates)} duplicates; first: {duplicates[:10]}"
        )


def count_cross_sample_neighbors(
    coordinates: np.ndarray, sample_ids: list[str]
) -> int:
    if len(coordinates) < 2:
        return 0
    distances, neighbors = cKDTree(coordinates).query(
        coordinates, k=min(GRAPH_KNN + 1, len(coordinates)), workers=-1
    )
    distances = np.atleast_2d(distances)
    neighbors = np.atleast_2d(neighbors)
    samples = np.asarray(sample_ids, dtype=object)
    cross_sample = samples[neighbors[:, 1:]] != samples[:, None]
    within_limit = distances[:, 1:] <= MAX_NEIGHBOR_DISTANCE
    return int(np.logical_and(cross_sample, within_limit).sum())


def main() -> None:
    args = parse_args()
    raw = args.raw_dir.resolve()
    output = args.output_dir.resolve()

    cells = read_lines(raw / "cell_name.txt")
    genes = read_lines(raw / "gene_list.txt")
    groups = read_lines(raw / "meta_annotation.txt")
    samples = read_lines(raw / "meta_batch.txt")
    require_unique(cells, "cell IDs")
    require_unique(genes, "gene IDs")

    expression = scipy.sparse.load_npz(raw / "gene_expression.npz").tocsr()
    if expression.shape[0] != len(cells) and expression.shape[1] == len(cells):
        expression = expression.transpose().tocsr()
    if expression.shape != (len(cells), len(genes)):
        raise ValueError(
            f"expression shape {expression.shape} does not match "
            f"{len(cells)} cells x {len(genes)} genes"
        )
    if expression.data.size and (
        not np.isfinite(expression.data).all() or expression.data.min() < 0
    ):
        raise ValueError("expression values must be finite and non-negative")
    if len(groups) != len(cells) or len(samples) != len(cells):
        raise ValueError("annotation and sample files must have one row per cell")

    coordinates = np.load(raw / "spatial_coordinates.npy")
    if coordinates.shape != (len(cells), 2):
        raise ValueError(
            f"coordinate shape {coordinates.shape} does not match "
            f"({len(cells)}, 2)"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates contain non-finite values")

    output.mkdir(parents=True, exist_ok=True)
    scipy.io.mmwrite(output / "expression.mtx", expression)
    np.savetxt(output / "spatial_coordinates.csv", coordinates, delimiter=",")
    for filename, values in (
        ("cell_ids.txt", cells),
        ("gene_ids.txt", genes),
        ("sample_ids.txt", samples),
        ("evaluation_cell_groups.txt", groups),
    ):
        (output / filename).write_text(
            "\n".join(values) + "\n", encoding="utf-8"
        )

    summary = {
        "dataset": "MOSTA",
        "raw_dir": str(raw),
        "output_dir": str(output),
        "cells": len(cells),
        "genes": len(genes),
        "nonzeros": int(expression.nnz),
        "samples": dict(
            sorted(Counter(samples).items(), key=lambda item: (-item[1], item[0]))
        ),
        "groups": len(set(groups)),
        "graph_knn": GRAPH_KNN,
        "graph_sigma": GRAPH_SIGMA,
        "max_neighbor_distance": MAX_NEIGHBOR_DISTANCE,
        "cross_sample_neighbors_within_max_distance": (
            count_cross_sample_neighbors(coordinates, samples)
        ),
    }
    (output / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
