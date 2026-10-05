#!/usr/bin/env python3
"""Plot the leading source-target relations in every MERFISH CCC module."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from merfish_common import DATASET, HERE, configure_style


INPUT = DATASET / "analysis/results/directional_edge_statistics.csv"
CLUSTER_META = DATASET / "analysis/cluster_metadata.csv"
OUT = HERE / "supp_figures"
PDF_OUT = OUT
TOP_N = 12
WIDTH_MM = 160
HEIGHT_MM = 178
POSITIVE = "#3C8DAD"
NEGATIVE = "#D97855"


def main():
    configure_style()
    relations = pd.read_csv(INPUT)
    modeled = relations.modeled.astype(str).str.lower().eq("true")
    eligible = relations.loc[modeled & relations.directional_enrichment.gt(0)].copy()
    cluster_meta = pd.read_csv(CLUSTER_META)[["cluster", "dominant_type"]]
    eligible = eligible.merge(cluster_meta, on="cluster", how="left", validate="many_to_one")
    eligible = eligible.sort_values(
        ["cluster", "directional_enrichment"], ascending=[True, False])
    shown = eligible.groupby("cluster", sort=True, group_keys=False).head(TOP_N).copy()
    shown["relation"] = shown.source_gene.astype(str) + " → " + shown.target_gene.astype(str)
    shown["rank"] = shown.groupby("cluster").cumcount() + 1
    OUT.mkdir(parents=True, exist_ok=True)
    shown.to_csv(OUT / "cluster_source_target_ranking.csv", index=False)

    clusters = sorted(shown.cluster.unique())
    fig, axes = plt.subplots(4, 2, figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4),
                             sharex=True)
    fig.subplots_adjust(left=.205, right=.975, bottom=.085, top=.925,
                        hspace=.53, wspace=.54)
    xmax = max(float(shown.directional_enrichment.max()) * 1.08, .1)

    for ax, cluster in zip(axes.flat, clusters):
        part = shown.loc[shown.cluster.eq(cluster)].sort_values("rank", ascending=False)
        y = np.arange(len(part))
        colors = np.where(part.signed_beta.ge(0), POSITIVE, NEGATIVE)
        ax.barh(y, part.directional_enrichment, height=.66, color=colors,
                edgecolor="none")
        labels = [f"#{int(rank)}  {pair}" for rank, pair in
                  zip(part["rank"], part.relation)]
        ax.set_yticks(y, labels, fontsize=5.2)
        ax.tick_params(axis="y", length=0, pad=2)
        ax.tick_params(axis="x", labelsize=5, length=2, width=.45, pad=1.5)
        ax.set_xlim(0, xmax)
        ax.grid(axis="x", color="#E5E5E5", linewidth=.45, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_linewidth(.45)
        dominant = str(part.dominant_type.iloc[0])
        ax.set_title(f"Cluster {int(cluster)}  |  {dominant}", loc="left",
                     fontsize=6.3, fontweight="bold", pad=3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Directional enrichment", fontsize=5.5, labelpad=3)

    fig.suptitle("Cluster-specific source–target rankings", x=.205, y=.975,
                 ha="left", fontsize=8, fontweight="bold")
    fig.legend(handles=[Patch(facecolor=POSITIVE, label="Positive effect"),
                        Patch(facecolor=NEGATIVE, label="Negative effect")],
               loc="lower center", bbox_to_anchor=(.60, .018), ncol=2,
               frameon=False, fontsize=5.2, handlelength=1.3,
               columnspacing=1.4)

    output = OUT / "cluster_source_target_ranking.png"
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    pdf_output = PDF_OUT / "cluster_source_target_ranking.pdf"
    fig.savefig(output, dpi=600, facecolor="white",
                pil_kwargs={"compress_level": 6})
    fig.savefig(pdf_output, facecolor="white")
    plt.close(fig)
    print(f"clusters={len(clusters)}")
    print(f"relations={len(shown)}")
    print(output)


if __name__ == "__main__":
    main()
