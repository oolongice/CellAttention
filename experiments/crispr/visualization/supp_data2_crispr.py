#!/usr/bin/env python3
import os
"""Generate supplementary expression coverage and Icam1 top-100 validation plots."""

from pathlib import Path
from font_helper import font_family
import sys

import anndata as ad
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse


HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
DATASET = HERE.parent
PROJECT = DATASET.parent
RAW = DATASET / "data/raw/SpacSeq_lung_cancer_label.h5ad"
VALIDATION = HERE / "data/source_conditioned_crispr_validation.csv"
OUT = HERE / "supp_figures"
PDF_OUT = OUT
FONT = DATASET / "font"
DPI = 600
WIDTH = 160 / 25.4
ALIASES = {"Icam": "Icam1", "Pngr1": "Phgr1"}


def style():
    family = font_family(FONT)
    mpl.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def as_bool(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def perturbation_expression_fraction():
    data = ad.read_h5ad(RAW, backed="r")
    shape_columns = [
        str(column) for column in data.obs.columns
        if str(column).endswith("_shapes") and not str(column).startswith("control_")
    ]
    control = as_bool(data.obs["control_training_shapes"])
    control_indices = np.flatnonzero(control)
    gene_lookup = {str(gene): index for index, gene in enumerate(data.var_names)}
    records = []
    for shape_column in shape_columns:
        shape_label = shape_column.removesuffix("_shapes")
        gene = ALIASES.get(shape_label, shape_label)
        if gene not in gene_lookup:
            records.append({
                "shape_column": shape_column,
                "perturbation_gene": gene,
                "expressing_control_cells": np.nan,
                "control_cells": len(control_indices),
                "expression_fraction": np.nan,
                "measured": False,
            })
            continue
        values = data.X[control_indices, gene_lookup[gene]]
        if sparse.issparse(values):
            expressing = int(values.getnnz())
        else:
            expressing = int(np.count_nonzero(np.asarray(values)))
        records.append({
            "shape_column": shape_column,
            "perturbation_gene": gene,
            "expressing_control_cells": expressing,
            "control_cells": len(control_indices),
            "expression_fraction": expressing / len(control_indices),
            "measured": True,
        })
    data.file.close()
    result = pd.DataFrame(records).sort_values(
        ["measured", "expression_fraction", "perturbation_gene"],
        ascending=[False, False, True], na_position="last"
    ).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT / "perturbation_gene_control_expression_fraction.csv", index=False)

    fig, ax = plt.subplots(figsize=(WIDTH, 78 / 25.4))
    positions = np.arange(len(result))
    heights = result.expression_fraction.fillna(0).to_numpy()
    colors = np.where(result.measured, "#4C78A8", "#C9C9C9")
    ax.bar(positions, heights, width=0.76, color=colors, edgecolor="white", linewidth=0.35)
    labels = [
        gene if measured else f"{gene}\n(not measured)"
        for gene, measured in zip(result.perturbation_gene, result.measured)
    ]
    ax.set_xticks(positions, labels, rotation=90, ha="center", va="top")
    ax.set_ylabel("Fraction of control cells expressing gene")
    ax.set_title("Perturbation-gene expression in the control region",
                 loc="left", fontweight="bold", pad=5)
    ax.set_ylim(0, max(heights) * 1.12 if max(heights) else 1)
    ax.grid(axis="y", color="#E2E2E2", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.3, pad=1.5)
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.33, top=0.90)
    fig.savefig(OUT / "perturbation_gene_control_expression_fraction.png",
                dpi=DPI, facecolor="white", pil_kwargs={"compress_level": 6})
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT / "perturbation_gene_control_expression_fraction.pdf",
                format="pdf", facecolor="white")
    plt.close(fig)
    return result


def icam1_top50_validation():
    data = pd.read_csv(VALIDATION)
    part = (
        data.loc[data.source_gene.eq("Icam1")]
        .nsmallest(50, "physical_rank_within_source_receiver")
        .sort_values("physical_rank_within_source_receiver")
        .reset_index(drop=True)
    )
    if len(part) != 50:
        raise ValueError(f"Expected 50 Icam1 targets, found {len(part)}")

    vmax = max(1.0, float(data.crispr_logfc.abs().max(skipna=True)))
    norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    x = np.arange(len(part))
    y = part.heldout_mse_reduction_physical.to_numpy()
    supported = part.experimental_supported.astype(bool).to_numpy()
    evidence = part.experimental_evidence.fillna(0).to_numpy()
    sizes = np.where(supported, 13 + 5 * np.minimum(evidence, 10), 13)

    fig, ax = plt.subplots(figsize=(WIDTH, 88 / 25.4))
    ax.vlines(x, 0, y, color="#D7DCE0", linewidth=0.55, zorder=1)
    ax.scatter(x[~supported], y[~supported], s=sizes[~supported], c="#C8CDD1",
               edgecolors="white", linewidths=0.35, zorder=2)
    scatter = None
    if supported.any():
        edges = np.where(
            part.direction_concordant.astype(bool).to_numpy()[supported],
            "#202020", "#D95F02"
        )
        scatter = ax.scatter(
            x[supported], y[supported], s=sizes[supported],
            c=part.crispr_logfc.to_numpy()[supported], cmap="coolwarm", norm=norm,
            edgecolors=edges, linewidths=0.65, zorder=3
        )

    labels = [
        f"{gene} ({int(rank)})"
        for gene, rank in zip(part.target_gene, part.physical_rank_within_source_receiver)
    ]
    ax.set_xticks(x, labels, rotation=90, ha="center", va="top")
    ax.set_ylabel("Source-conditioned held-out MSE reduction")
    ax.set_xlabel("Top 50 Icam1-conditioned target genes")
    ax.set_title("Icam1 target validation", loc="left", fontweight="bold", pad=5)
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.0, pad=1.2)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#C8CDD1",
               markeredgecolor="white", markersize=4.5, label="No matched CRISPR DGE"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F7F7F7",
               markeredgecolor="#202020", markersize=4.5, label="Direction concordant"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#F7F7F7",
               markeredgecolor="#D95F02", markersize=4.5, label="Direction discordant"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.53, 1.12),
              ncol=3, frameon=False, handletextpad=0.35, columnspacing=0.8)

    cax = fig.add_axes([0.80, 0.865, 0.17, 0.022])
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"),
        cax=cax, orientation="horizontal"
    )
    bar.set_label("CRISPR logFC", labelpad=1)
    bar.ax.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
    bar.outline.set_linewidth(0.4)

    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.37, top=0.82)
    fig.savefig(OUT / "Icam1_top50_source_conditioned_target_validation.png",
                dpi=DPI, facecolor="white", pil_kwargs={"compress_level": 6})
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF_OUT / "Icam1_top50_source_conditioned_target_validation.pdf",
                format="pdf", facecolor="white")
    plt.close(fig)
    part.to_csv(OUT / "Icam1_top50_source_conditioned_target_validation.csv", index=False)


def main():
    style()
    fractions = perturbation_expression_fraction()
    icam1_top50_validation()
    print(f"perturbations={len(fractions)}")
    print(f"measured={int(fractions.measured.sum())}")
    print(f"output={OUT}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
