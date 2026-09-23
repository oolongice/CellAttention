#!/usr/bin/env python3
"""Plot source scores for each synthetic receiver and its planted target."""

from pathlib import Path
import sys
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET = Path(os.environ["CELLATTENTION_SYNTHETIC_WORKDIR"])
PROJECT = DATASET.parent
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else "shared_target"
CASE_TITLES = {"independent_targets": "Case 1: Independent targets", "two_sources_per_target": "Case 2: Two sources per target", "shared_target": "Case 3: Shared target"}
CASE_DIRS = {"independent_targets": "case1_independent_targets", "two_sources_per_target": "case2_two_sources_per_target", "shared_target": "case3_shared_target"}
if DATASET_NAME not in CASE_TITLES:
    raise ValueError(f"Unknown dataset: {DATASET_NAME}")
DATA = DATASET / "data/preprocessed" / DATASET_NAME
MODELS = DATASET / "model/multiseed" / DATASET_NAME
BENCHMARK = DATASET / "analysis/continuous_cell_benchmark/run_metrics.csv"
FIGURES = DATASET / "visualization/supp_figures" / CASE_DIRS[DATASET_NAME]
OUTPUT_DATA = FIGURES / "data"
METHOD = "transformer_physical"
OUTPUT_FORMAT = "png"
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
        (results["dataset"] == DATASET_NAME) & (results["method"] == METHOD)
    ].copy()
    if selected.empty:
        raise ValueError(f"No existing benchmark rows for {DATASET_NAME}/{METHOD}")
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
    truth = pd.read_csv(DATA / "truth.csv")
    programs = (truth[["receiver_group", "target_gene"]].drop_duplicates()
                .sort_values(["receiver_group", "target_gene"]))
    target_by_receiver = dict(zip(programs.receiver_group.astype(int), programs.target_gene))
    if sorted(target_by_receiver) != list(range(6)):
        raise ValueError(f"Expected one target program for receiver groups 0-5: {target_by_receiver}")
    missing = [gene for gene in [*source_genes, *target_by_receiver.values()] if gene not in gene_index]
    if missing:
        raise ValueError(f"Missing genes: {missing}")

    raw = np.loadtxt(DATA / "expression.csv", delimiter=",")
    coordinates = np.loadtxt(DATA / "spatial_coordinates.csv", delimiter=",")
    planted_groups = np.loadtxt(DATA / "cell_groups.txt", dtype=int)
    physical_fields, _ = core.source_fields(coordinates, raw)
    fields = core.standardize(physical_fields)
    source_indices = [gene_index[gene] for gene in source_genes]
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
        columns = []
        for receiver in range(6):
            target_index = gene_index[target_by_receiver[receiver]]
            columns.append(source_scores[labels[planted_groups == receiver], :, target_index][
                :, source_indices].mean(axis=0))
        seed_matrices.append(np.stack(columns, axis=1))
    seed_matrices = np.asarray(seed_matrices)
    matrix = seed_matrices.mean(axis=0)
    matrix_sd = seed_matrices.std(axis=0, ddof=1 if len(seeds) > 1 else 0)

    planted = {(str(row.source_gene), int(row.receiver_group), str(row.target_gene))
               for row in truth.itertuples()}
    rows = []
    for receiver in range(6):
        target_gene = target_by_receiver[receiver]
        for row, source_gene in enumerate(source_genes):
            rows.append({
                "source_gene": source_gene,
                "target_gene": target_gene,
                "receiver_group": receiver,
                "source_score_mean": matrix[row, receiver],
                "source_score_sd": matrix_sd[row, receiver],
                "is_planted_relation": (source_gene, receiver, target_gene) in planted,
                "n_training_seeds": len(seeds),
                "method": METHOD,
            })
    output_table = pd.DataFrame(rows)
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(OUTPUT_DATA / "receiver_target_source_scores.csv", index=False)

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
    for row, source_gene in enumerate(source_genes):
        for receiver in range(6):
            target_gene = target_by_receiver[receiver]
            if (source_gene, receiver, target_gene) in planted:
                axis.add_patch(Rectangle((receiver + .10, row + .10), .80, .80,
                                         fill=False, edgecolor="#C83E4D",
                                         linewidth=1.0, linestyle=(0, (2, 1.5)), zorder=4))
    axis.set_xticks(
        np.arange(6) + .5,
        [f"R{index}–T{target_by_receiver[index].rsplit(chr(95), 1)[1]}" for index in range(6)],
        fontsize=6.5,
    )
    axis.set_yticks(np.arange(len(source_genes)) + .5,
                    [f"Source {index}" for index in range(len(source_genes))])
    axis.set_xlabel("Receiver–target pair")
    axis.set_ylabel("Source gene")
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.legend(handles=[Patch(facecolor="none", edgecolor="#C83E4D", linewidth=1.0,
                               linestyle=(0, (2, 1.5)), label="Planted relation")], loc="lower left",
                bbox_to_anchor=(0, 1.02), frameon=False, fontsize=6, borderaxespad=0)
    colorbar = figure.colorbar(image, ax=axis, fraction=.035, pad=.025)
    colorbar.set_label("Inferred spatial influence")
    colorbar.outline.set_linewidth(.5)
    figure.suptitle(f"{CASE_TITLES[DATASET_NAME].split(chr(58), 1)[0]}: Source recovery",
                    x=.17, y=.96, ha="left", fontweight="bold", fontsize=9)

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / f"receiver_target_source_score_heatmap.{OUTPUT_FORMAT}"
    save_figure(figure, output, dpi=PNG_DPI)
    print(f"seeds={seeds[0]}-{seeds[-1]} (n={len(seeds)})")
    print(f"output_data={OUTPUT_DATA / 'receiver_target_source_scores.csv'}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
