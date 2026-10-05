#!/usr/bin/env python3
import os
"""Compare source-conditioned physical effects with observed CRISPR effects."""

SOURCE_GENE = "Icam1"
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
LABEL_MODEL_TOP_N = 20
MAX_LABELS = 8

from pathlib import Path
from font_helper import font_family
import sys
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
PROJECT = HERE.parent.parent
INPUT = HERE / "data/source_conditioned_crispr_validation.csv"
FONT = HERE.parent / "font"


def configure_style():
    family = font_family(FONT)
    mpl.rcParams.update({
        "font.family": family, "font.sans-serif": [family], "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def main():
    fmt = OUTPUT_FORMAT.lower()
    if fmt not in {"png", "pdf"}:
        raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_style()
    data = pd.read_csv(INPUT)
    available = set(data.source_gene.unique())
    if SOURCE_GENE not in available:
        raise ValueError(f"SOURCE_GENE {SOURCE_GENE!r} is absent; available sources: {sorted(available)}")
    part = data[data.source_gene.eq(SOURCE_GENE) & data.crispr_logfc.notna()].copy()
    if part.empty:
        raise ValueError(f"No matched CRISPR effects are available for {SOURCE_GENE}")

    # A positive source effect predicts a decrease after loss of source activity.
    part["predicted_perturbation_effect"] = -part.effect_median_physical
    part["evidence"] = -np.log10(part.crispr_fdr.clip(lower=np.finfo(float).tiny))
    improvement = part.heldout_mse_reduction_physical.clip(lower=0)
    scale = max(float(improvement.max()), np.finfo(float).eps)
    sizes = 18 + 62 * np.sqrt(improvement / scale)
    rho, pvalue = spearmanr(part.predicted_perturbation_effect, part.crispr_logfc)

    fig, ax = plt.subplots(figsize=(100 / 25.4, 91 / 25.4))
    fig.subplots_adjust(left=.18, right=.79, bottom=.18, top=.87)
    norm = mpl.colors.Normalize(vmin=float(part.evidence.min()), vmax=float(part.evidence.max()))
    points = ax.scatter(
        part.predicted_perturbation_effect, part.crispr_logfc, s=sizes,
        c=part.evidence, cmap="viridis", norm=norm, alpha=.86,
        edgecolors=np.where(part.direction_concordant, "#202020", "#D95F02"),
        linewidths=.65, zorder=3,
    )
    ax.axhline(0, color="#858585", linewidth=.65, linestyle="--", zorder=1)
    ax.axvline(0, color="#858585", linewidth=.65, linestyle="--", zorder=1)
    ax.grid(color="#E8E8E8", linewidth=.45, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Predicted effect after source loss\n(− signed physical effect)")
    ax.set_ylabel("Observed CRISPR logFC")
    ax.set_title(f"{SOURCE_GENE}-conditioned perturbation effects", loc="left", fontweight="bold", pad=5)
    ax.text(.03, .97, f"n = {len(part)} matched targets\nSpearman ρ = {rho:.2f}, P = {pvalue:.2g}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2)
    ax.tick_params(length=2.5, width=.6, pad=2)

    labelled = (part[part.physical_rank_within_source_receiver <= LABEL_MODEL_TOP_N]
                .sort_values("physical_rank_within_source_receiver").head(MAX_LABELS))
    offsets = [(4, 4), (4, -9), (-4, 5), (4, 5), (-4, -9), (4, 5), (4, -9), (-4, 5)]
    for (_, row), offset in zip(labelled.iterrows(), offsets):
        ax.annotate(row.target_gene, (row.predicted_perturbation_effect, row.crispr_logfc),
                    xytext=offset, textcoords="offset points", fontsize=6,
                    ha="left" if offset[0] > 0 else "right",
                    va="bottom" if offset[1] > 0 else "top")

    cax = fig.add_axes([.84, .27, .025, .48])
    bar = fig.colorbar(points, cax=cax)
    bar.set_label("Experimental −log10 FDR", labelpad=3)
    cax.tick_params(labelsize=6, pad=2)
    out = HERE / "figures" / f"{SOURCE_GENE}_predicted_vs_crispr_effect.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    options = {"facecolor": "white"}
    if fmt == "png":
        options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(out, **options)
    plt.close(fig)
    print(f"source_gene={SOURCE_GENE}")
    print(f"matched_targets={len(part)}")
    print(f"spearman_rho={rho:.6f}")
    print(f"spearman_p={pvalue:.6g}")
    print(f"output={out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
