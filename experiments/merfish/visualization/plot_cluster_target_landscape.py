#!/usr/bin/env python3
"""MERFISH cluster target landscape matched to the Xenium liver figure."""
import os
OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "png").lower()
PNG_DPI = 600
FIGURE_WIDTH_IN = 4.2
FIGURE_HEIGHT_IN = 2.5
TOP_N = 8

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist

from merfish_common import (DATASET, FIGURES, HERE, PRE, TYPE_COLORS,
                            configure_style)

TARGETS = DATASET / "analysis/results/target_scores.csv"
ASSIGNMENTS = DATASET / "analysis/results/receiver_group_assignments.csv"
METADATA = PRE / "cell_metadata.parquet"


def main():
    configure_style()
    scores = pd.read_csv(TARGETS)
    scores["cluster"] = scores.receiver_group.astype(int) + 1
    shown = scores[scores.target_rank <= TOP_N].copy()
    clusters = sorted(shown.cluster.unique())
    genes = sorted(shown.target_gene.unique())
    matrix = (shown.pivot(index="target_gene", columns="cluster",
                          values="training_improvement")
              .reindex(index=genes, columns=clusters).fillna(0))
    distance = pdist(matrix.to_numpy(), metric="cosine")
    distance[~np.isfinite(distance)] = 1
    gene_order = matrix.index.to_numpy()[
        leaves_list(linkage(distance, method="average"))].tolist()
    xmap = {gene: index for index, gene in enumerate(gene_order)}
    ymap = {cluster: index for index, cluster in enumerate(clusters)}
    shown["x"] = shown.target_gene.map(xmap)
    shown["y"] = shown.cluster.map(ymap)
    low = float(shown.training_improvement.min())
    high = float(shown.training_improvement.max())
    denominator = max(np.sqrt(high) - np.sqrt(low), 1e-8)
    scaled = (np.sqrt(shown.training_improvement) - np.sqrt(low)) / denominator
    shown["point_size"] = 4 + 24 * scaled

    assignments = pd.read_csv(ASSIGNMENTS)
    metadata = pd.read_parquet(METADATA)
    if len(assignments) != len(metadata):
        raise ValueError("Cluster assignments and metadata row counts differ")
    if not np.array_equal(assignments.cell_id.astype(str), metadata.cell_id.astype(str)):
        raise ValueError("Cluster assignments and metadata cell orders differ")
    cells = pd.DataFrame({
        "cluster": assignments.receiver_group.to_numpy(dtype=int) + 1,
        "annotation": metadata["type"].astype(str).to_numpy(),
    })
    composition = (cells.groupby(["cluster", "annotation"]).size()
                   .rename("count").reset_index())
    composition["fraction"] = (composition["count"] /
                               composition.groupby("cluster")["count"].transform("sum"))

    output_data = HERE / "data"
    output_data.mkdir(parents=True, exist_ok=True)
    shown.to_csv(output_data / "cluster_target_landscape.csv", index=False)
    pd.DataFrame({"display_order": range(len(gene_order)),
                  "target_gene": gene_order}).to_csv(
                      output_data / "cluster_target_landscape_gene_order.csv", index=False)
    composition.to_csv(
        output_data / "cluster_target_landscape_cell_type_composition.csv", index=False)

    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    axis = figure.add_axes([.085, .34, .70, .405])
    for row in range(len(clusters)):
        axis.axhspan(row - .48, row + .48,
                    color="#F4F4F4" if row % 2 == 0 else "#FAFAFA", zorder=0)
    for x in np.arange(len(gene_order) - 1) + .5:
        axis.axvline(x, color="white", linewidth=.45, zorder=1)
    axis.scatter(shown.x, shown.y, s=shown.point_size, c=shown.target_rank,
                 cmap="Blues_r", vmin=1, vmax=TOP_N,
                 edgecolors="white", linewidths=.65, zorder=3)
    axis.set(xlim=(-.55, len(gene_order) - .45),
             ylim=(len(clusters) - .45, -.55))
    axis.set_xticks(range(len(gene_order)), gene_order, rotation=90,
                    ha="center", va="top", fontsize=4.0)
    axis.tick_params(axis="x", length=0, pad=2)
    axis.set_yticks(range(len(clusters)),
                    [f"Cluster {cluster}" for cluster in clusters], fontsize=4.2)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.spines[:].set_visible(False)
    axis.set_xlabel("Model-identified target gene", fontsize=4.6, labelpad=3)

    plot_bottom, plot_height = .34, .405
    for row, cluster in enumerate(clusters):
        center = plot_bottom + plot_height * (1 - (row + .5) / len(clusters))
        pie = figure.add_axes([.798, center - .031, .062, .062])
        part = composition[composition.cluster == cluster].sort_values(
            "fraction", ascending=False)
        colors = [TYPE_COLORS.get(annotation, "#BDBDBD")
                  for annotation in part.annotation]
        pie.pie(part.fraction, colors=colors, startangle=90, counterclock=False,
                wedgeprops={"linewidth": .22, "edgecolor": "white"}, radius=.98)
        pie.set_aspect("equal")
        pie.set_axis_off()
    figure.text(.822, .77, "Cell-type\ncomposition", ha="center", va="bottom",
                fontsize=3.6, color="#444444")

    handles = []
    for value in [.05, .15, .25]:
        scale = np.clip((np.sqrt(value) - np.sqrt(low)) / denominator, 0, 1)
        handles.append(axis.scatter([], [], s=4 + 24 * scale, color="#769CB3",
                                    edgecolor="white", linewidth=.4,
                                    label=f"{value:.2f}"))
    legend = axis.legend(
        handles=handles, title="Training improvement", frameon=False,
        loc="lower left", bbox_to_anchor=(0, 1.08), ncol=3,
        columnspacing=.65, handletextpad=.2, fontsize=3.2, title_fontsize=3.5)
    axis.add_artist(legend)
    color_axis = figure.add_axes([.655, .795, .115, .014])
    colorbar = mpl.colorbar.ColorbarBase(
        color_axis, cmap=mpl.colormaps["Blues_r"],
        norm=mpl.colors.Normalize(vmin=1, vmax=TOP_N), orientation="horizontal")
    colorbar.set_ticks([1, TOP_N])
    colorbar.ax.tick_params(labelsize=3.0, length=1.2, pad=1)
    colorbar.outline.set_linewidth(.35)
    colorbar.ax.set_title("Target rank", fontsize=3.5, pad=1.5)
    figure.suptitle("Cluster-specific target programs in 3D MERFISH embryo",
                    x=.085, y=.975, ha="left", fontsize=6.8, fontweight="bold")

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / f"cluster_target_landscape.{OUTPUT_FORMAT}"
    options = {"facecolor": "white"}
    if OUTPUT_FORMAT == "png":
        options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    figure.savefig(output, **options)
    plt.close(figure)
    print(f"targets={len(gene_order)}")
    print(f"cluster_target_entries={len(shown)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
