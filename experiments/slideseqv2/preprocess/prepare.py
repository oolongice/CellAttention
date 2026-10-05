#!/usr/bin/env python3
"""Prepare the bundled Slide-seqV2 AnnData file for CellAttention."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.io import mmwrite


SCRIPT_DIR = Path(__file__).resolve().parent
BLACKLIST = re.compile(r"^(Gm[0-9]+|mt-|Rps|Rpl|AY|Mir|Linc)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Slide-seqV2 genes and export CellAttention inputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "rawdata" / "adata_raw_labeled.h5ad",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR / "tmp" / "ppdata"
    )
    parser.add_argument("--target-genes", type=int, default=500)
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--gene-list", type=Path)
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--annotation-key", default="cluster")
    parser.add_argument(
        "--sample-key",
        help="Optional adata.obs column used for stratified epoch sampling.",
    )
    parser.add_argument("--normalize-total", type=float, default=10_000.0)
    parser.add_argument("--no-log1p", action="store_true")
    parser.add_argument(
        "--coordinate-scaling", choices=("unit", "none"), default="unit"
    )
    args = parser.parse_args()
    if args.target_genes <= 0 or args.min_cells <= 0:
        parser.error("--target-genes and --min-cells must be positive")
    if args.normalize_total <= 0:
        parser.error("--normalize-total must be positive")
    return args


def read_gene_list(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    genes = {
        line.strip().split(",")[0].split("\t")[0]
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if not genes:
        raise ValueError(f"gene list is empty: {path}")
    return genes


def normalize_expression(
    matrix: sparse.csr_matrix, target_sum: float, log1p: bool
) -> sparse.csr_matrix:
    totals = np.asarray(matrix.sum(axis=1)).reshape(-1)
    factors = np.divide(
        target_sum,
        totals,
        out=np.zeros_like(totals, dtype=np.float64),
        where=totals > 0,
    )
    normalized = (sparse.diags(factors) @ matrix).tocsr().astype(np.float32)
    if log1p:
        normalized.data = np.log1p(normalized.data)
    return normalized


def filter_and_select_genes(
    matrix: sparse.csr_matrix,
    gene_ids: np.ndarray,
    min_cells: int,
    target_genes: int,
    allow_list: set[str] | None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, int]:
    expressed_cells = np.asarray((matrix > 0).sum(axis=0)).reshape(-1)
    keep = expressed_cells >= min_cells
    keep &= np.asarray([not BLACKLIST.match(gene) for gene in gene_ids])
    if allow_list is not None:
        keep &= np.asarray([gene in allow_list for gene in gene_ids])
    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError("gene filtering removed every gene")

    filtered = matrix[:, indices].tocsr()
    filtered_genes = gene_ids[indices]
    filtered_coverage = expressed_cells[indices]
    means = np.asarray(filtered.mean(axis=0)).reshape(-1)
    second_moments = np.asarray(filtered.power(2).mean(axis=0)).reshape(-1)
    variances = second_moments - means**2
    selected = np.argsort(variances)[-min(target_genes, len(indices)) :]
    selected = selected[np.argsort(variances[selected])[::-1]]
    return (
        filtered[:, selected].tocsr(),
        filtered_genes[selected],
        filtered_coverage[selected],
        len(indices),
    )


def scale_coordinates(coordinates: np.ndarray, mode: str) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("spatial coordinates must have at least two columns")
    coordinates = coordinates[:, :2]
    if not np.isfinite(coordinates).all():
        raise ValueError("spatial coordinates contain non-finite values")
    if mode == "unit":
        coordinates = coordinates - coordinates.min(axis=0)
        coordinates /= max(float(np.ptp(coordinates, axis=0).max()), 1e-8)
    return coordinates


def write_lines(path: Path, values: np.ndarray) -> None:
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input)
    if args.spatial_key not in adata.obsm:
        raise ValueError(f"missing adata.obsm[{args.spatial_key!r}]")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("cell IDs and gene IDs must be unique")

    matrix = sparse.csr_matrix(adata.X, dtype=np.float32)
    if matrix.data.size and (
        not np.isfinite(matrix.data).all() or matrix.data.min() < 0
    ):
        raise ValueError("expression values must be finite and non-negative")
    normalized = normalize_expression(
        matrix, args.normalize_total, not args.no_log1p
    )
    selected, genes, coverage, filtered_count = filter_and_select_genes(
        normalized,
        adata.var_names.astype(str).to_numpy(),
        args.min_cells,
        args.target_genes,
        read_gene_list(args.gene_list),
    )
    coordinates = scale_coordinates(
        adata.obsm[args.spatial_key], args.coordinate_scaling
    )
    has_annotations = args.annotation_key in adata.obs
    has_samples = args.sample_key is not None and args.sample_key in adata.obs
    if args.sample_key is not None and not has_samples:
        raise ValueError(f"missing adata.obs[{args.sample_key!r}]")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mmwrite(output / "expression.mtx", selected.tocoo())
    write_lines(output / "cell_ids.txt", adata.obs_names.astype(str).to_numpy())
    write_lines(output / "gene_ids.txt", genes)
    np.savetxt(
        output / "spatial_coordinates.csv",
        coordinates,
        delimiter=",",
        fmt="%.7g",
    )
    if has_annotations:
        write_lines(
            output / "evaluation_cell_groups.txt",
            adata.obs[args.annotation_key].astype(str).to_numpy(),
        )
    if has_samples:
        write_lines(
            output / "sample_ids.txt",
            adata.obs[args.sample_key].astype(str).to_numpy(),
        )
    np.savetxt(
        output / "selected_gene_coverage.csv",
        np.column_stack([genes, coverage]),
        delimiter=",",
        fmt="%s",
        header="gene_id,expressed_cells",
        comments="",
    )
    summary = {
        "dataset": "SlideSeqV2",
        "input": str(args.input.resolve()),
        "output_dir": str(output),
        "cells": int(selected.shape[0]),
        "input_genes": int(adata.n_vars),
        "genes_after_filter": filtered_count,
        "selected_genes": int(selected.shape[1]),
        "min_cells": args.min_cells,
        "normalized_total": args.normalize_total,
        "log1p": not args.no_log1p,
        "coordinate_scaling": args.coordinate_scaling,
        "annotation_key": args.annotation_key if has_annotations else None,
        "sample_key": args.sample_key if has_samples else None,
    }
    (output / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
