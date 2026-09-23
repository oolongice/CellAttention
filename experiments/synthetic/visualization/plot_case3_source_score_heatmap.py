#!/usr/bin/env python3
"""Plot Case 3 source scores for target_0 across all inferred clusters."""

from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET = Path(os.environ["CELLATTENTION_SYNTHETIC_WORKDIR"])
PROJECT = DATASET.parent
DATA = DATASET / "data/preprocessed/shared_target"
MODELS = DATASET / "model/multiseed/shared_target"
BENCHMARK = DATASET / "analysis/continuous_cell_benchmark/run_metrics.csv"
FIGURES = DATASET / "visualization/figures"
OUTPUT_DATA = DATASET / "visualization/data"
TARGET_GENE = "target_0"
METHOD = "transformer_physical"
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_FIGURE_FORMAT", "png")
PNG_DPI = 600

if str(HERE.parent / "analysis") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "analysis"))
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import benchmark_core as core
from style import configure_style, mm_to_inches, save_figure


def available_seeds() -> list[int]:
    """Return seeds already represented in the maintained benchmark table."""
    results = pd.read_csv(BENCHMARK)
    selected = results.loc[
        (results["dataset"] == "shared_target") & (results["method"] == METHOD)
    ].copy()
    if selected.empty:
        raise ValueError(f"No existing benchmark rows for shared_target/{METHOD}")
    return sorted(selected["seed"].astype(int).unique().tolist())


def main() -> None:
    configure_style(PROJECT / "font", font_size=7, axes_title_size=9,
                    axes_label_size=7, tick_size=6.5, legend_size=6)
    seeds = available_seeds()
    genes = (DATA / "gene_ids.txt").read_text().splitlines()
    gene_index = {gene: index for index, gene in enumerate(genes)}
    source_genes = sorted(
        [gene for gene in genes if gene.startswith("source_")],
        key=lambda gene: int(gene.rsplit("_", 1)[1]),
    )
    missing = [gene for gene in [TARGET_GENE, *source_genes] if gene not in gene_index]
    if missing:
        raise ValueError(f"Missing genes: {missing}")

    raw = np.loadtxt(DATA / "expression.csv", delimiter=",")
    coordinates = np.loadtxt(DATA / "spatial_coordinates.csv", delimiter=",")
    planted_groups = np.loadtxt(DATA / "cell_groups.txt", dtype=int)
    physical_fields, _ = core.source_fields(coordinates, raw)
    fields = core.standardize(physical_fields)
    source_indices = [gene_index[gene] for gene in source_genes]
    target_index = gene_index[TARGET_GENE]
    seed_matrices = []
    for seed in seeds:
        model = MODELS / f"seed_{seed}"
        observed = np.loadtxt(model / "standardized_expression.csv", delimiter=",")
        reconstruction = np.loadtxt(model / "reconstruction.csv", delimiter=",")
        embedding = np.loadtxt(model / "cell_embeddings.csv", delimiter=",")
        labels = core.cluster(core.whiten(embedding), seed)
        _, source_scores = core.scores(
            labels, observed - reconstruction, fields, "transformer"
        )
        seed_matrices.append(np.stack([
            source_scores[labels[planted_groups == receiver], :, target_index][
                :, source_indices
            ].mean(axis=0)
            for receiver in range(6)
        ], axis=1))
    seed_matrices = np.asarray(seed_matrices)
    matrix = seed_matrices.mean(axis=0)
    matrix_sd = seed_matrices.std(axis=0, ddof=1 if len(seeds) > 1 else 0)

    truth = pd.read_csv(DATA / "truth.csv")
    planted = {(str(row.source_gene), int(row.receiver_group)) for row in truth.itertuples()}
    rows = []
    for receiver in range(6):
        for row, source_gene in enumerate(source_genes):
            rows.append({
                "source_gene": source_gene,
                "target_gene": TARGET_GENE,
                "receiver_cluster": receiver,
                "source_score_mean": matrix[row, receiver],
                "source_score_sd": matrix_sd[row, receiver],
                "is_planted_relation": (source_gene, receiver) in planted,
                "n_training_seeds": len(seeds),
                "method": METHOD,
            })
    output_table = pd.DataFrame(rows)
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(OUTPUT_DATA / "case3_target0_source_scores_by_cluster.csv", index=False)

    figure, axis = plt.subplots(figsize=mm_to_inches(105, 92))
    figure.subplots_adjust(left=.17, right=.82, bottom=.16, top=.84)
    soft_blues = LinearSegmentedColormap.from_list(
        "soft_blues", ["#F7FBFF", "#DEEBF7", "#9ECAE1", "#4292C6"]
    )
    image = axis.pcolormesh(
        matrix, cmap=soft_blues, vmin=0, vmax=float(np.quantile(matrix, .98)),
        shading="flat", edgecolors="white", linewidth=.8,
    )
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xticks(np.arange(6) + .5, [f"Cluster {index}" for index in range(6)])
    axis.set_yticks(np.arange(len(source_genes)) + .5,
                    [f"Source {index}" for index in range(len(source_genes))])
    axis.set_xlabel("Receiver cluster")
    axis.set_ylabel("Source gene")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    colorbar = figure.colorbar(image, ax=axis, fraction=.035, pad=.025)
    colorbar.set_label("Inferred spatial influence")
    colorbar.outline.set_linewidth(.5)
    figure.suptitle("Case 3: Source-field associations for Target",
                    x=.17, y=.96, ha="left", fontweight="bold", fontsize=9)

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / f"case3_target0_source_score_heatmap.{OUTPUT_FORMAT}"
    save_figure(figure, output, dpi=PNG_DPI)
    print(f"seeds={seeds[0]}-{seeds[-1]} (n={len(seeds)})")
    print(f"output_data={OUTPUT_DATA / 'case3_target0_source_scores_by_cluster.csv'}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
