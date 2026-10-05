#!/usr/bin/env python3
"""Export standard CellAttention attribution edges and 3D direction summaries."""
from pathlib import Path
import os
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(os.environ["CELLATTENTION_MERFISH_WORKDIR"]) / "analysis"
ROOT = HERE.parent
RESULTS = HERE / "results"
MODEL = ROOT / "model/transformer"
PRE = ROOT / "data/preprocessed"
GRID_SIZE = 5
LENGTH_SCALE = 50.0
MAX_DISTANCE = 150.0
MIN_DISTANCE = 2.0
TOP_FRACTION = 0.5
MAX_EDGES = 1000


def select_dense_grid(relations, size=GRID_SIZE):
    score = relations.pivot_table(index="target_gene", columns="source_gene",
                                  values="derived_attention", aggfunc="max", fill_value=0.0)
    present = (score > 0).astype(np.int8)
    target_priority = relations.groupby("target_gene").agg(
        coverage=("source_gene", "nunique"), attention=("derived_attention", "sum"),
        rank=("target_rank", "min")).sort_values(
            ["coverage", "attention", "rank"], ascending=[False, False, True])
    seeds = [target_priority.head(size).index.tolist()]
    for target in target_priority.head(80).index:
        seeds.append([target] + [x for x in target_priority.index if x != target][:size - 1])
    best = None
    for targets in seeds:
        for _ in range(12):
            source_stats = pd.DataFrame({"coverage": present.loc[targets].sum(axis=0),
                                         "attention": score.loc[targets].sum(axis=0)}).sort_values(
                                             ["coverage", "attention"], ascending=False)
            sources = source_stats.head(size).index.tolist()
            target_stats = pd.DataFrame({"coverage": present[sources].sum(axis=1),
                                         "attention": score[sources].sum(axis=1)}).sort_values(
                                             ["coverage", "attention"], ascending=False)
            targets = target_stats.head(size).index.tolist()
        candidate = (int(present.loc[targets, sources].to_numpy().sum()),
                     float(score.loc[targets, sources].to_numpy().sum()), targets, sources)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2], best[3]


def yukawa(distance):
    distance = np.maximum(np.asarray(distance, dtype=float), MIN_DISTANCE)
    values = np.exp(-distance / LENGTH_SCALE) / (4 * np.pi * LENGTH_SCALE ** 2 * distance)
    return values.astype(np.float32).astype(float)


def weighted_vector(weights, directions):
    total = weights.sum()
    return np.sum(weights[:, None] * directions, axis=0) / total if total > 0 else np.full(3, np.nan)


def top_indices(weights):
    count = int((weights > 0).sum())
    n = min(math.ceil(count * TOP_FRACTION), MAX_EDGES)
    return np.argpartition(weights, -n)[-n:] if n else np.array([], dtype=int)


def candidate_grid(relations):
    rows = []
    for cluster, group in relations.groupby("cluster", sort=True):
        targets, sources = select_dense_grid(group)
        lookup = group.sort_values(["derived_attention", "target_rank"], ascending=[False, True]).drop_duplicates(
            ["source_gene", "target_gene"]).set_index(["source_gene", "target_gene"])
        for target in targets:
            for source in sources:
                key = (source, target)
                if key in lookup.index:
                    q = lookup.loc[key]
                    rows.append(dict(cluster=int(cluster), target_gene=target, source_gene=source,
                                     modeled=True, signed_beta=float(q.signed_beta),
                                     derived_attention=float(q.derived_attention)))
                else:
                    rows.append(dict(cluster=int(cluster), target_gene=target, source_gene=source,
                                     modeled=False, signed_beta=np.nan, derived_attention=np.nan))
    grid = pd.DataFrame(rows)
    if len(grid) != 8 * GRID_SIZE * GRID_SIZE:
        raise RuntimeError(f"Expected 200 grid entries, found {len(grid)}")
    return grid


def main():
    relations = pd.read_csv(HERE / "all_selected_cluster_source_target_relations.csv")
    grid = candidate_grid(relations)
    assignments = pd.read_csv(RESULTS / "receiver_group_assignments.csv")
    coordinates = np.loadtxt(PRE / "spatial_coordinates.csv", delimiter=",").astype(np.float32).astype(float)
    raw = np.loadtxt(MODEL / "raw_expression.csv", delimiter=",", dtype=np.float32)
    residuals = np.loadtxt(MODEL / "residuals.csv", delimiter=",", dtype=np.float32)
    genes = (MODEL / "gene_ids.txt").read_text().splitlines()
    gene_index = {gene: index for index, gene in enumerate(genes)}
    if (MODEL / "cell_ids.txt").read_text().splitlines() != assignments.cell_id.tolist():
        raise RuntimeError("Model and receiver-group cell order differ")
    labels = assignments.receiver_group.to_numpy() + 1
    probability = assignments.maximum_probability.to_numpy()
    residual_z = ((residuals.astype(float) - residuals.astype(float).mean(0)) /
                  np.maximum(residuals.astype(float).std(0), 1e-8))
    cell_ids = assignments.cell_id.to_numpy()
    stats = []
    exports = []
    modeled = grid[grid.modeled]
    for number, (source, source_rows) in enumerate(modeled.groupby("source_gene"), start=1):
        source_index = gene_index[source]
        senders = np.flatnonzero(raw[:, source_index] > 0)
        distances = cKDTree(coordinates).sparse_distance_matrix(
            cKDTree(coordinates[senders]), MAX_DISTANCE, output_type="coo_matrix")
        keep = distances.row != senders[distances.col]
        receivers = distances.row[keep]
        senders_for_edge = senders[distances.col[keep]]
        distance = distances.data[keep]
        base = yukawa(distance) * raw[senders_for_edge, source_index].astype(float)
        field = np.bincount(receivers, weights=base, minlength=len(raw))
        field[raw[:, source_index] > 0] = 0
        field_sd = max(field.std(), 1e-8)
        for cluster, cluster_rows in source_rows.groupby("cluster"):
            in_cluster = labels[receivers] == cluster
            receiver = receivers[in_cluster]
            sender = senders_for_edge[in_cluster]
            edge_distance = distance[in_cluster]
            edge_base = base[in_cluster]
            directions = ((coordinates[receiver] - coordinates[sender]) /
                          np.maximum(edge_distance[:, None], MIN_DISTANCE))
            source_negative_receiver = raw[receiver, source_index] <= 0
            for relation in cluster_rows.itertuples():
                target_index = gene_index[relation.target_gene]
                contribution = (probability[receiver] * relation.signed_beta * edge_base / field_sd)
                supported = source_negative_receiver & (contribution * residual_z[receiver, target_index] > 0)
                weights = np.where(supported, np.abs(contribution) *
                                   np.minimum(np.abs(residual_z[receiver, target_index]), 3) / 3, 0.0)
                retained = top_indices(weights)
                vector = weighted_vector(weights[retained], directions[retained])
                stats.append(dict(cluster=int(cluster), target_gene=relation.target_gene,
                                  source_gene=source, modeled=True,
                                  direction_x=vector[0], direction_y=vector[1], direction_z=vector[2],
                                  directional_enrichment=float(np.linalg.norm(vector)),
                                  signed_beta=relation.signed_beta,
                                  derived_attention=relation.derived_attention,
                                  edge_weight_sum=float(weights[retained].sum()),
                                  receiver_edges=len(retained)))
                if len(retained):
                    retained = retained[np.argsort(-weights[retained])]
                    receiver_total = np.bincount(receiver, weights=weights, minlength=len(raw))
                    clipped = np.minimum(np.abs(residual_z[receiver[retained], target_index]), 3) / 3
                    signed = contribution[retained] * clipped
                    exports.append(pd.DataFrame(dict(
                        cluster=int(cluster), source_gene=source, target_gene=relation.target_gene,
                        sender_cell_id=cell_ids[sender[retained]], receiver_cell_id=cell_ids[receiver[retained]],
                        distance=edge_distance[retained], kernel_weight=yukawa(edge_distance[retained]),
                        source_expression=raw[sender[retained], source_index],
                        receiver_group_probability=probability[receiver[retained]],
                        target_residual_z=residual_z[receiver[retained], target_index],
                        signed_beta=relation.signed_beta, signed_contribution=signed,
                        influence_score=weights[retained],
                        receiver_normalized_attribution=weights[retained] / receiver_total[receiver[retained]],
                        rank_within_triplet=np.arange(1, len(retained) + 1))))
        print(f"source {number}/{modeled.source_gene.nunique()}: {source}", flush=True)
    calculated = pd.DataFrame(stats)
    final_stats = grid.merge(calculated, on=["cluster", "target_gene", "source_gene", "modeled",
                                               "signed_beta", "derived_attention"], how="left")
    final_stats.to_csv(RESULTS / "directional_edge_statistics.csv", index=False)
    edge_table = pd.concat(exports, ignore_index=True)
    edge_table.to_csv(RESULTS / "directional_edges.csv", index=False)
    metadata = pd.Series(dict(spatial_dimensions=3, length_scale=LENGTH_SCALE,
                              maximum_distance=MAX_DISTANCE, minimum_distance=MIN_DISTANCE,
                              top_fraction=TOP_FRACTION, maximum_edges_per_triplet=MAX_EDGES,
                              grid_entries=len(final_stats), modeled_triplets=len(calculated),
                              retained_edges=len(edge_table)))
    metadata.to_json(RESULTS / "directional_edge_metadata.json", indent=2)
    print(f"Wrote {len(final_stats)} grid entries and {len(edge_table)} retained 3D edges")


if __name__ == "__main__":
    main()
