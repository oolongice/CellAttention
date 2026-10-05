#!/usr/bin/env python3
import os
"""Heatmap of model-derived source-target effects in all immune receiver cells."""

SOURCE_GENES = ["Icam1", "Cxcr4"]
TARGETS_PER_SOURCE = 10
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600

from pathlib import Path
from font_helper import font_family
import sys
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
PROJECT = HERE.parent.parent
INPUT = HERE.parent / "analysis/source_focused_immune/source_target_receiver_results.csv"
FONT = HERE.parent / "font"


def configure_style():
    family = font_family(FONT)
    mpl.rcParams.update({
        "font.family": family, "font.sans-serif": [family], "font.size": 7,
        "axes.titlesize": 8, "xtick.labelsize": 6.2, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def ordered_targets(data):
    selected = data[data.physical_rank_within_source_receiver <= TARGETS_PER_SOURCE].copy()
    order = (selected.groupby("target_gene", as_index=False)
             .agg(best_rank=("physical_rank_within_source_receiver", "min"),
                  best_score=("heldout_mse_reduction_physical", "max"))
             .sort_values(["best_rank", "best_score", "target_gene"], ascending=[True, False, True]))
    return order.target_gene.tolist()


def main():
    fmt = OUTPUT_FORMAT.lower()
    if fmt not in {"png", "pdf"}:
        raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_style()
    data = pd.read_csv(INPUT)
    data = data[data.receiver_cell_type.eq("Immune_all") & data.source_gene.isin(SOURCE_GENES)].copy()
    missing = sorted(set(SOURCE_GENES) - set(data.source_gene.unique()))
    if missing:
        raise ValueError(f"Sources absent from immune model results: {missing}")
    targets = ordered_targets(data)
    if not targets:
        raise ValueError("No targets passed the requested rank threshold")
    effect = (data.pivot(index="source_gene", columns="target_gene", values="effect_median_physical")
              .reindex(index=SOURCE_GENES, columns=targets))
    best = (data.pivot(index="source_gene", columns="target_gene", values="physical_best")
            .reindex(index=SOURCE_GENES, columns=targets).astype("boolean").fillna(False).astype(bool))
    values = np.abs(effect.to_numpy(dtype=float))
    limit = max(float(np.nanquantile(values, .98)), np.finfo(float).eps)

    width_mm = max(150, 8.2 * len(targets) + 31)
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, 58 / 25.4))
    fig.subplots_adjust(left=.105, right=.88, bottom=.36, top=.80)
    sns.heatmap(effect, ax=ax, cmap="vlag", center=0, vmin=-limit, vmax=limit,
                linewidths=1.15, linecolor="white", cbar=False,
                xticklabels=targets, yticklabels=SOURCE_GENES, square=False)
    for row in range(len(SOURCE_GENES)):
        for col in range(len(targets)):
            if best.iat[row, col]:
                ax.plot(col + .5, row + .5, marker="o", markersize=2.1,
                        markerfacecolor="#151515", markeredgewidth=0, zorder=4)
    ax.set_xlabel("Target gene", labelpad=4)
    ax.set_ylabel("Source gene", labelpad=5)
    ax.set_title("Source–target effects in immune receiver cells", loc="left", fontweight="bold", pad=6)
    ax.tick_params(axis="x", rotation=38, length=0, pad=2)
    ax.tick_params(axis="y", rotation=0, length=0, pad=3)
    cax = fig.add_axes([.905, .35, .015, .38])
    norm = mpl.colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="vlag"), cax=cax)
    bar.set_label("Signed physical effect", labelpad=3)
    cax.tick_params(labelsize=6, pad=2)
    ax.text(1.0, 1.055, "●  physical > raw and KNN", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=6.2, color="#202020")

    slug = "_".join(SOURCE_GENES)
    out = HERE / "figures" / f"immune_source_target_heatmap_{slug}.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    options = {"facecolor": "white"}
    if fmt == "png":
        options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(out, **options)
    plt.close(fig)
    print(f"sources={','.join(SOURCE_GENES)}")
    print(f"targets={len(targets)}")
    print(f"target_genes={','.join(targets)}")
    print(f"output={out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
