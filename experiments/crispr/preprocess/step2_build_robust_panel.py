#!/usr/bin/env python3
"""Build a 500-gene panel with mandatory sources, robust responses, and control-only HVGs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

SHAPE_ALIASES = {"Icam": "Icam1", "Pngr1": "Phgr1"}
MIN_COUNTS = 200
MIN_GENES = 100
UPPER_QUANTILE = 0.999
MAX_PCT_MT = 10.0
TARGET_SUM = 10_000.0
CHUNK_SIZE = 10_000
PANEL_SIZE = 500


def as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def calculate_qc(data: ad.AnnData, chunk_size: int):
    total = np.zeros(data.n_obs, dtype=np.float64)
    n_genes = np.zeros(data.n_obs, dtype=np.int32)
    mt_counts = np.zeros(data.n_obs, dtype=np.float64)
    mt_mask = np.asarray(data.var_names.str.upper().str.startswith("MT-"))
    for start in range(0, data.n_obs, chunk_size):
        stop = min(start + chunk_size, data.n_obs)
        matrix = sparse.csr_matrix(data.X[start:stop])
        total[start:stop] = np.asarray(matrix.sum(axis=1)).ravel()
        n_genes[start:stop] = matrix.getnnz(axis=1)
        mt_counts[start:stop] = np.asarray(matrix[:, mt_mask].sum(axis=1)).ravel()
    pct_mt = np.divide(mt_counts * 100, total, out=np.zeros_like(total), where=total > 0)
    return total, n_genes, pct_mt


def hvg_scores(data, keep_cells, total_counts):
    sums = np.zeros(data.n_vars, dtype=np.float64)
    squares = np.zeros(data.n_vars, dtype=np.float64)
    n_kept = int(keep_cells.sum())
    if n_kept == 0:
        raise ValueError("no control-training cells pass QC")
    for start in range(0, data.n_obs, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, data.n_obs)
        local = keep_cells[start:stop]
        if not local.any():
            continue
        matrix = sparse.csr_matrix(data.X[start:stop])[local].astype(np.float64)
        totals = total_counts[start:stop][local]
        factors = np.divide(TARGET_SUM, totals, out=np.zeros_like(totals), where=totals > 0)
        matrix = sparse.diags(factors) @ matrix
        matrix.data = np.log1p(matrix.data)
        sums += np.asarray(matrix.sum(axis=0)).ravel()
        squares += np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
    mean = sums / n_kept
    variance = np.maximum(squares / n_kept - mean**2, 0)
    log_mean = np.log1p(mean)
    log_dispersion = np.log1p(variance / np.maximum(mean, 1e-12))
    bins = pd.qcut(log_mean, q=20, labels=False, duplicates="drop")
    frame = pd.DataFrame({"bin": bins, "dispersion": log_dispersion})
    grouped = frame.groupby("bin", observed=True)["dispersion"]
    center = grouped.transform("mean").to_numpy()
    spread = grouped.transform("std").fillna(1).to_numpy()
    return np.divide(log_dispersion - center, spread, out=np.zeros_like(log_dispersion), where=spread > 0)


def technical_hvg(gene: str) -> bool:
    upper = gene.upper()
    return upper.startswith(("MT-", "RPL", "RPS"))


def measurable_sources(data) -> tuple[list[str], list[dict[str, object]]]:
    var_set = set(map(str, data.var_names))
    sources = []
    records = []
    for column in data.obs.columns:
        column = str(column)
        if not column.endswith("_shapes") or column.startswith("control_"):
            continue
        label = column.removesuffix("_shapes")
        gene = SHAPE_ALIASES.get(label, label)
        measurable = gene in var_set
        records.append({"shape_column": column, "shape_label": label, "source_gene": gene, "measurable": measurable})
        if measurable and gene not in sources:
            sources.append(gene)
    return sources, records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("SpacSeq_lung_cancer_label.h5ad"))
    parser.add_argument("--responses", type=Path, default=Path("panel_selection/selected_response_gene_list.txt"))
    parser.add_argument("--output", type=Path, default=Path("SpacSeq_lung_cancer_label_robustQC_500genes.h5ad"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("panel_selection"))
    args = parser.parse_args()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    responses = list(dict.fromkeys(line.strip() for line in args.responses.read_text().splitlines() if line.strip()))

    backed = ad.read_h5ad(args.input, backed="r")
    values = np.asarray(backed.X[:1000, :1000].data if sparse.issparse(backed.X[:1000, :1000]) else backed.X[:1000, :1000]).ravel()
    if values.size and (values.min() < 0 or np.mean(np.abs(values - np.rint(values)) > 1e-6) > 0.01):
        raise ValueError("input .X does not look like non-negative raw counts")
    total, n_genes, pct_mt = calculate_qc(backed, CHUNK_SIZE)
    max_counts = float(np.quantile(total, UPPER_QUANTILE))
    max_genes = float(np.quantile(n_genes, UPPER_QUANTILE))
    keep = (total >= MIN_COUNTS) & (n_genes >= MIN_GENES) & (total <= max_counts) & (n_genes <= max_genes) & (pct_mt <= MAX_PCT_MT)
    control_training = as_bool(backed.obs["control_training_shapes"])
    hvg_cells = keep & control_training
    scores = hvg_scores(backed, hvg_cells, total)
    genes = list(map(str, backed.var_names))
    var_set = set(genes)
    score_lookup = dict(zip(genes, scores))
    sources, source_records = measurable_sources(backed)
    missing_responses = sorted(set(responses) - var_set)
    if missing_responses:
        raise ValueError(f"selected response genes absent from input: {missing_responses}")

    panel = []
    origin = {}
    for gene in sources:
        panel.append(gene); origin[gene] = "mandatory_perturbation_source"
    for gene in responses:
        if gene not in origin:
            panel.append(gene); origin[gene] = "robust_perturbation_response"
    if len(panel) > PANEL_SIZE:
        raise ValueError(f"mandatory sources plus responses ({len(panel)}) exceed panel size {PANEL_SIZE}")
    hvg_order = sorted(
        (gene for gene in genes if not technical_hvg(gene)),
        key=lambda gene: score_lookup[gene], reverse=True,
    )
    for gene in hvg_order:
        if gene not in origin:
            panel.append(gene); origin[gene] = "control_training_hvg_fill"
        if len(panel) == PANEL_SIZE:
            break
    if len(panel) != PANEL_SIZE:
        raise ValueError(f"could only construct {len(panel)} panel genes")

    kept_indices = np.flatnonzero(keep)
    result = backed[kept_indices, panel].to_memory()
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
    result.var["panel_origin"] = [origin[gene] for gene in panel]
    result.var["control_training_hvg_score"] = [score_lookup[gene] for gene in panel]
    result.uns["robust_500_gene_panel"] = {
        "input_cells": int(len(total)), "output_cells": int(keep.sum()),
        "control_training_hvg_cells": int(hvg_cells.sum()), "panel_size": PANEL_SIZE,
        "mandatory_sources": len(sources), "robust_responses": sum(value == "robust_perturbation_response" for value in origin.values()),
        "hvg_fill": sum(value == "control_training_hvg_fill" for value in origin.values()),
        "normalization_total_source": f"all {len(genes)} input genes",
        "qc": {"min_counts": MIN_COUNTS, "min_genes": MIN_GENES, "upper_quantile": UPPER_QUANTILE,
               "max_counts": max_counts, "max_genes": max_genes, "max_pct_mt": MAX_PCT_MT},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output, compression="gzip")

    manifest = pd.DataFrame({
        "panel_index": np.arange(len(panel)), "gene": panel,
        "panel_origin": [origin[gene] for gene in panel],
        "control_training_hvg_score": [score_lookup[gene] for gene in panel],
    })
    manifest.to_csv(args.manifest_dir / "panel_manifest.csv", index=False)
    pd.DataFrame(source_records).to_csv(args.manifest_dir / "perturbation_source_manifest.csv", index=False)
    (args.manifest_dir / "panel_gene_ids.txt").write_text("\n".join(panel) + "\n")
    metadata = result.uns["robust_500_gene_panel"] | {"input": str(args.input), "output": str(args.output), "response_file": str(args.responses)}
    (args.manifest_dir / "panel_build_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
