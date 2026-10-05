#!/usr/bin/env python3
"""Build and preprocess the independently screened marker-QC panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
from scipy.io import mmwrite
from scipy.spatial import cKDTree

from step2_build_robust_panel import CHUNK_SIZE, TARGET_SUM, as_bool, calculate_qc


def write_lines(path: Path, values) -> None:
    path.write_text("\n".join(map(str, values)) + "\n")


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=root / "SpacSeq_lung_cancer_label.h5ad")
    parser.add_argument("--panel", type=Path, default=root / "marker_qc_run/panel_selection_marker_qc/panel_gene_ids.txt")
    parser.add_argument("--output-h5ad", type=Path, default=root / "marker_qc_run/SpacSeq_lung_cancer_label_markerQC_500genes.h5ad")
    parser.add_argument("--output-dir", type=Path, default=root / "marker_qc_run/preprocessed_marker_qc")
    args = parser.parse_args()

    panel = [line.strip() for line in args.panel.read_text().splitlines() if line.strip()]
    if len(panel) != 500 or len(set(panel)) != 500:
        raise ValueError("panel must contain exactly 500 unique genes")
    backed = ad.read_h5ad(args.input, backed="r")
    missing = sorted(set(panel) - set(map(str, backed.var_names)))
    if missing:
        raise ValueError(f"panel genes absent from input: {missing}")
    total, n_genes, pct_mt = calculate_qc(backed, CHUNK_SIZE)
    max_counts = float(np.quantile(total, 0.999))
    max_genes = float(np.quantile(n_genes, 0.999))
    keep = (total >= 200) & (n_genes >= 100) & (total <= max_counts) & (n_genes <= max_genes) & (pct_mt <= 10.0)
    result = backed[np.flatnonzero(keep), panel].to_memory()
    backed.file.close()
    result.obs["total_counts"] = total[keep]
    result.obs["n_genes_by_counts"] = n_genes[keep]
    result.obs["pct_counts_mt"] = pct_mt[keep]
    counts = sparse.csr_matrix(result.X).copy()
    result.layers["counts"] = counts
    factors = np.divide(TARGET_SUM, total[keep], out=np.zeros_like(total[keep]), where=total[keep] > 0)
    normalized = sparse.diags(factors) @ counts.astype(np.float64)
    normalized.data = np.log1p(normalized.data)
    result.X = normalized.astype(np.float32)
    args.output_h5ad.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output_h5ad, compression="gzip")

    output = args.output_dir
    all_dir, control_dir, neighbor_dir = output / "all_cells", output / "control_train", output / "neighbor_context"
    for directory in (all_dir, control_dir, neighbor_dir): directory.mkdir(parents=True, exist_ok=True)
    control = as_bool(result.obs["control_training_shapes"])
    matrix = sparse.csr_matrix(result.X, dtype=np.float32)
    scale = float(result.uns["spatial"]["sample1"]["scalefactors"]["microns_per_pixel"])
    coordinates = np.asarray(result.obsm["spatial_raw"], dtype=np.float64) * scale
    annotations = result.obs["cell_annotation"].astype(str).tolist()
    cells = result.obs_names.astype(str).tolist()
    mmwrite(all_dir / "expression.mtx", matrix)
    mmwrite(control_dir / "expression.mtx", matrix[control])
    write_lines(all_dir / "cell_ids.txt", cells); write_lines(control_dir / "cell_ids.txt", np.asarray(cells)[control])
    write_lines(all_dir / "gene_ids.txt", panel); write_lines(control_dir / "gene_ids.txt", panel)
    write_lines(all_dir / "annotations.txt", annotations); write_lines(control_dir / "annotation.txt", np.asarray(annotations)[control])
    np.savetxt(all_dir / "spatial_coordinates_um.csv", coordinates, delimiter=",", fmt="%.7f")
    np.savetxt(control_dir / "spatial_coordinates_um.csv", coordinates[control], delimiter=",", fmt="%.7f")
    write_lines(all_dir / "control_shapes.txt", control.tolist())
    neighbor_mask = control.copy()
    distances, neighbors = cKDTree(coordinates).query(coordinates[control], k=21, workers=-1)
    neighbor_indices = np.unique(neighbors[:, 1:][distances[:, 1:] <= 50.0])
    neighbor_mask[neighbor_indices] = True
    mmwrite(neighbor_dir / "expression.mtx", matrix[neighbor_mask])
    write_lines(neighbor_dir / "cell_ids.txt", np.asarray(cells)[neighbor_mask])
    write_lines(neighbor_dir / "annotations.txt", np.asarray(annotations)[neighbor_mask])
    np.savetxt(neighbor_dir / "spatial_coordinates_um.csv", coordinates[neighbor_mask], delimiter=",", fmt="%.7f")
    summary = {"cells_all": result.n_obs, "cells_control": int(control.sum()), "cells_neighbor_context": int(neighbor_mask.sum()), "genes": result.n_vars, "panel": str(args.panel), "output_h5ad": str(args.output_h5ad)}
    output.mkdir(parents=True, exist_ok=True)
    (output / "preprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
