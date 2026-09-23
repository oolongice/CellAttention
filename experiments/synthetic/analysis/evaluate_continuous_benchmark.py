#!/usr/bin/env python3
"""Cell-level continuous target/source benchmark for available checkpoints."""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import benchmark_core as core

ROOT = Path(os.environ["CELLATTENTION_SYNTHETIC_WORKDIR"])
DATASETS = ("shared_target", "independent_targets", "two_sources_per_target")


def available_seeds(dataset: str) -> list[int]:
    root = ROOT / "model/multiseed" / dataset
    return sorted(int(p.name.removeprefix("seed_")) for p in root.glob("seed_*") if (p / "model.mpk").is_file())


def average_precision(values: np.ndarray, positives: set[int], excluded: int) -> float:
    candidate = np.arange(values.size) != excluded
    order = np.argsort(-np.where(candidate, values, -np.inf))
    ranks = sorted(int(np.where(order == gene)[0][0]) + 1 for gene in positives)
    return float(np.mean([(index + 1) / rank for index, rank in enumerate(ranks)]))


def normalized_source_log_rank(
    values: np.ndarray, positives: set[int], excluded: int
) -> float:
    candidate = np.arange(values.size) != excluded
    order = np.argsort(-np.where(candidate, values, -np.inf))
    ranks = sorted(int(np.where(order == gene)[0][0]) + 1 for gene in positives)
    candidate_count = int(candidate.sum())
    discount = lambda rank: 1.0 - np.log(rank) / np.log(candidate_count)
    observed = sum(discount(rank) for rank in ranks)
    ideal = sum(discount(rank) for rank in range(1, len(ranks) + 1))
    return float(observed / ideal)


def main() -> None:
    rows = []
    for dataset in DATASETS:
        data = ROOT / "data/preprocessed" / dataset
        genes = np.loadtxt(data / "gene_ids.txt", dtype=str).tolist()
        gene_index = {gene: index for index, gene in enumerate(genes)}
        groups = np.loadtxt(data / "cell_groups.txt", dtype=int)
        raw = np.loadtxt(data / "expression.csv", delimiter=",")
        coordinates = np.loadtxt(data / "spatial_coordinates.csv", delimiter=",")
        truth = pd.read_csv(data / "truth.csv")
        targets = {int(row.receiver_group): gene_index[row.target_gene] for row in truth.itertuples()}
        sources = {}
        for row in truth.itertuples():
            key = (int(row.receiver_group), gene_index[row.target_gene])
            sources.setdefault(key, set()).add(gene_index[row.source_gene])
        physical, uniform = core.source_fields(coordinates, raw)
        fields = {"physical": core.standardize(physical), "uniform": core.standardize(uniform)}
        for seed in available_seeds(dataset):
            model = ROOT / "model/multiseed" / dataset / f"seed_{seed}"
            observed = np.loadtxt(model / "standardized_expression.csv", delimiter=",")
            reconstruction = np.loadtxt(model / "reconstruction.csv", delimiter=",")
            embedding = np.loadtxt(model / "cell_embeddings.csv", delimiter=",")
            labels_by_context = {
                "baseline": core.cluster(observed, seed),
                "transformer": core.cluster(core.whiten(embedding), seed),
            }
            for method, target_mode, field_mode in core.METHODS:
                labels = labels_by_context[target_mode]
                outcome = observed if target_mode == "baseline" else observed - reconstruction
                target_score, source_score = core.scores(labels, outcome, fields[field_mode], target_mode)
                target_rr_values, target_log_values, rank_values = [], [], []
                source_values, source_log_values = [], []
                joint_rr_values, joint_log_values, joint_full_log_values = [], [], []
                for cell, true_group in enumerate(groups):
                    true_group = int(true_group)
                    if true_group not in targets:
                        continue
                    receiver_group = int(labels[cell])
                    target = targets[true_group]
                    rank = 1 + int(np.sum(target_score[receiver_group] > target_score[receiver_group, target]))
                    target_rr = 1.0 / rank
                    target_log_rank = 1.0 - np.log(rank) / np.log(len(genes))
                    source_ap = average_precision(
                        source_score[receiver_group, :, target],
                        sources[(true_group, target)],
                        target,
                    )
                    source_log_rank = normalized_source_log_rank(
                        source_score[receiver_group, :, target],
                        sources[(true_group, target)],
                        target,
                    )
                    target_rr_values.append(target_rr)
                    target_log_values.append(target_log_rank)
                    rank_values.append(rank)
                    source_values.append(source_ap)
                    source_log_values.append(source_log_rank)
                    joint_rr_values.append(target_rr * source_ap)
                    joint_log_values.append(target_log_rank * source_ap)
                    joint_full_log_values.append(target_log_rank * source_log_rank)
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "method": method,
                    "cell_target_reciprocal_rank": float(np.mean(target_rr_values)),
                    "cell_target_normalized_log_rank": float(np.mean(target_log_values)),
                    "cell_target_mean_rank": float(np.mean(rank_values)),
                    "cell_target_median_rank": float(np.median(rank_values)),
                    "cell_source_average_precision_given_true_target": float(np.mean(source_values)),
                    "cell_source_normalized_log_rank_given_true_target": float(
                        np.mean(source_log_values)
                    ),
                    "cell_joint_target_source_recovery": float(np.mean(joint_rr_values)),
                    "cell_joint_target_source_log_recovery": float(np.mean(joint_log_values)),
                    "cell_joint_target_source_full_log_recovery": float(
                        np.mean(joint_full_log_values)
                    ),
                    "n_evaluated_cells": len(target_rr_values),
                })
    detail = pd.DataFrame(rows)
    output = ROOT / "analysis/continuous_cell_benchmark"
    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "run_metrics.csv", index=False)
    summary = detail.groupby(["dataset", "method"], as_index=False).agg(
        n_seeds=("seed", "nunique"),
        target_reciprocal_rank=("cell_target_reciprocal_rank", "mean"),
        target_normalized_log_rank=("cell_target_normalized_log_rank", "mean"),
        target_mean_rank=("cell_target_mean_rank", "mean"),
        target_median_rank=("cell_target_median_rank", "median"),
        source_average_precision_given_true_target=(
            "cell_source_average_precision_given_true_target", "mean"
        ),
        source_normalized_log_rank_given_true_target=(
            "cell_source_normalized_log_rank_given_true_target", "mean"
        ),
        joint_target_source_recovery=("cell_joint_target_source_recovery", "mean"),
        joint_target_source_log_recovery=(
            "cell_joint_target_source_log_recovery", "mean"
        ),
        joint_target_source_full_log_recovery=(
            "cell_joint_target_source_full_log_recovery", "mean"
        ),
    )
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
