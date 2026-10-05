#!/usr/bin/env python3
"""Select a fixed <=500-gene Slide-seqV2 panel for modeling and CCC comparison.

The panel is selected without using CellAttention results. It balances complete
measurable CellChat interactions, receiver-annotation markers, spatial structure,
and expression variability. All intermediate scores and selection reasons are
written for auditing.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree


BLACKLIST = re.compile(
    r"^(Gm\d+|mt-|Rps|Rpl|Mir|Snor|Hist\d|Hbb-|Hba-|Malat1$|Xist$|AY|Linc)",
    re.IGNORECASE,
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--cellchat-db", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--annotation-key", default="cluster")
    p.add_argument("--target-genes", type=int, default=500)
    p.add_argument("--min-beads", type=int, default=100)
    p.add_argument("--max-detection-fraction", type=float, default=0.90)
    p.add_argument("--ccc-gene-budget", type=int, default=160)
    p.add_argument("--marker-total", type=int, default=300)
    p.add_argument("--spatial-total", type=int, default=400)
    p.add_argument("--knn", type=int, default=12)
    return p.parse_args()


def normalized_log1p(x: sparse.csr_matrix) -> sparse.csr_matrix:
    totals = np.asarray(x.sum(axis=1)).ravel()
    scale = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    y = (sparse.diags(scale) @ x).tocsr().astype(np.float32)
    y.data = np.log1p(y.data)
    return y


def cellchat_interactions(path: Path, gene_lookup: dict[str, str]):
    table = pd.read_csv(path, sep="\t")
    records = []
    for row in table.itertuples(index=False):
        ligand = gene_lookup.get(str(row[0]).casefold())
        receptor_parts = [gene_lookup.get(part.casefold()) for part in str(row[1]).split("_")]
        if ligand and receptor_parts and all(receptor_parts):
            records.append((str(row[0]), str(row[1]), ligand, tuple(receptor_parts)))
    return records


def main() -> None:
    args = arguments()
    if not (1 <= args.target_genes <= 500):
        raise ValueError("target genes must be between 1 and 500")
    a = ad.read_h5ad(args.input)
    if args.annotation_key not in a.obs or "spatial" not in a.obsm:
        raise ValueError("required cluster annotation or spatial coordinates are missing")
    x_raw = sparse.csr_matrix(a.X, dtype=np.float32)
    x = normalized_log1p(x_raw)
    genes = np.asarray(a.var_names.astype(str))
    n = x.shape[0]
    coverage = np.asarray((x_raw > 0).sum(axis=0)).ravel()
    detection = coverage / n
    means = np.asarray(x.mean(axis=0)).ravel()
    variances = np.asarray(x.power(2).mean(axis=0)).ravel() - means**2
    dispersion = variances / np.maximum(means, 1e-3)
    eligible = (
        (coverage >= args.min_beads)
        & (detection <= args.max_detection_fraction)
        & np.asarray([not BLACKLIST.match(g) for g in genes])
    )
    eligible_idx = np.flatnonzero(eligible)
    if eligible_idx.size < args.target_genes:
        raise ValueError("fewer eligible genes than requested panel size")

    labels = a.obs[args.annotation_key].astype(str).to_numpy()
    label_names = sorted(np.unique(labels))
    marker_score = np.zeros(x.shape[1], dtype=np.float32)
    marker_label = np.full(x.shape[1], "", dtype=object)
    marker_rankings: dict[str, list[int]] = {}
    for label in label_names:
        inside = labels == label
        outside = ~inside
        mean_in = np.asarray(x[inside].mean(axis=0)).ravel()
        mean_out = np.asarray(x[outside].mean(axis=0)).ravel()
        det_in = np.asarray((x_raw[inside] > 0).mean(axis=0)).ravel()
        score = np.log2((mean_in + 0.05) / (mean_out + 0.05)) * np.sqrt(det_in)
        score[~eligible] = -np.inf
        order = np.argsort(score)[::-1]
        marker_rankings[label] = order.tolist()
        better = score > marker_score
        marker_score[better] = score[better]
        marker_label[better] = label

    gene_lookup = {g.casefold(): g for g in genes}
    gene_index = {g: i for i, g in enumerate(genes)}
    interactions = cellchat_interactions(args.cellchat_db, gene_lookup)
    interactions = [
        r for r in interactions
        if eligible[gene_index[r[2]]] and all(eligible[gene_index[g]] for g in r[3])
    ]

    hvg_z = np.zeros_like(dispersion)
    vals = np.log1p(dispersion[eligible])
    hvg_z[eligible] = (vals - vals.mean()) / max(vals.std(), 1e-6)
    marker_z = np.zeros_like(marker_score)
    vals = marker_score[eligible]
    marker_z[eligible] = (vals - vals.mean()) / max(vals.std(), 1e-6)
    gene_priority = hvg_z + marker_z

    selected: set[str] = set()
    reasons: dict[str, set[str]] = defaultdict(set)
    interaction_scores = []
    for ligand_name, receptor_name, ligand, receptor_parts in interactions:
        members = (ligand,) + receptor_parts
        score = float(np.mean([gene_priority[gene_index[g]] for g in members]))
        score += 0.25 * min(np.log1p(coverage[gene_index[g]]) for g in members)
        interaction_scores.append((score, ligand_name, receptor_name, members))
    kept_interactions = []
    for score, ligand_name, receptor_name, members in sorted(interaction_scores, reverse=True):
        additions = set(members) - selected
        if len(selected) + len(additions) > min(args.ccc_gene_budget, args.target_genes):
            continue
        selected.update(additions)
        kept_interactions.append((ligand_name, receptor_name, score, members))
        for gene in members:
            reasons[gene].add("cellchat_complete_interaction")

    # Add annotation markers round-robin so large receiver classes cannot monopolize the panel.
    marker_position = {label: 0 for label in label_names}
    while len(selected) < min(args.marker_total, args.target_genes):
        changed = False
        for label in label_names:
            ranking = marker_rankings[label]
            while marker_position[label] < len(ranking):
                idx = ranking[marker_position[label]]
                marker_position[label] += 1
                if not np.isfinite(marker_score[idx]):
                    break
                gene = genes[idx]
                reasons[gene].add(f"annotation_marker:{label}")
                if gene not in selected:
                    selected.add(gene)
                    changed = True
                    break
            if len(selected) >= min(args.marker_total, args.target_genes):
                break
        if not changed:
            break

    # Score spatial structure only for a pre-screened pool to keep memory bounded.
    pool_idx = set(np.argsort(np.where(eligible, dispersion, -np.inf))[-2000:].tolist())
    pool_idx.update(np.argsort(np.where(eligible, marker_score, -np.inf))[-1000:].tolist())
    pool_idx.update(gene_index[g] for g in selected)
    coords = np.asarray(a.obsm["spatial"], dtype=np.float32)[:, :2]
    _, neighbours = cKDTree(coords).query(coords, k=args.knn + 1)
    neighbours = neighbours[:, 1:]
    spatial_score = np.full(x.shape[1], np.nan, dtype=np.float32)
    for idx in sorted(pool_idx):
        values = x[:, idx].toarray().ravel()
        sd = values.std()
        if sd <= 1e-8:
            spatial_score[idx] = 0.0
            continue
        z = (values - values.mean()) / sd
        neighbour_mean = z[neighbours].mean(axis=1)
        spatial_score[idx] = np.corrcoef(z, neighbour_mean)[0, 1]
    spatial_order = np.argsort(np.nan_to_num(spatial_score, nan=-np.inf))[::-1]
    for idx in spatial_order:
        if len(selected) >= min(args.spatial_total, args.target_genes):
            break
        if not eligible[idx]:
            continue
        gene = genes[idx]
        selected.add(gene)
        reasons[gene].add("spatial_knn_autocorrelation")

    hvg_order = np.argsort(np.where(eligible, dispersion, -np.inf))[::-1]
    for idx in hvg_order:
        if len(selected) >= args.target_genes:
            break
        gene = genes[idx]
        selected.add(gene)
        reasons[gene].add("high_dispersion")

    panel = sorted(selected, key=lambda g: (-gene_priority[gene_index[g]], g))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "selected_genes.txt").write_text("\n".join(panel) + "\n")
    audit = pd.DataFrame({
        "gene": panel,
        "expressed_beads": [coverage[gene_index[g]] for g in panel],
        "detection_fraction": [detection[gene_index[g]] for g in panel],
        "normalized_mean": [means[gene_index[g]] for g in panel],
        "normalized_variance": [variances[gene_index[g]] for g in panel],
        "dispersion": [dispersion[gene_index[g]] for g in panel],
        "best_marker_score": [marker_score[gene_index[g]] for g in panel],
        "best_marker_annotation": [marker_label[gene_index[g]] for g in panel],
        "spatial_knn_correlation": [spatial_score[gene_index[g]] for g in panel],
        "selection_reasons": [";".join(sorted(reasons[g])) for g in panel],
    })
    audit.to_csv(out / "selected_gene_audit.csv", index=False)
    pd.DataFrame([
        {
            "ligand_db": ligand,
            "receptor_db": receptor,
            "selection_score": score,
            "panel_members": ";".join(members),
        }
        for ligand, receptor, score, members in kept_interactions
    ]).to_csv(out / "retained_complete_cellchat_interactions.csv", index=False)
    summary = {
        "input": str(args.input.resolve()),
        "cells_or_beads": int(a.n_obs),
        "input_genes": int(a.n_vars),
        "annotations": {k: int(v) for k, v in pd.Series(labels).value_counts().items()},
        "sample_fields": [],
        "selected_genes": len(panel),
        "eligible_genes": int(eligible.sum()),
        "cellchat_interactions_measurable_before_panel": len(interactions),
        "cellchat_interactions_used_to_construct_panel": len(kept_interactions),
        "single_section_no_replication": True,
        "selection_budgets": {
            "ccc_genes": args.ccc_gene_budget,
            "markers_total": args.marker_total,
            "spatial_total": args.spatial_total,
            "final_total": args.target_genes,
        },
    }
    (out / "panel_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
