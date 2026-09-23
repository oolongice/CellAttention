#!/usr/bin/env python3
"""Create a publication-ready overview of one synthetic experimental design."""

# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
import os
DATASET_TYPE = int(os.environ.get("CELLATTENTION_SYNTHETIC_CASE", "1"))  # 1: independent_targets; 2: two_sources_per_target; 3: shared_target
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_FIGURE_FORMAT", "png")  # "png" or "pdf"
RECEIVER_GROUP = 0  # representative receiver program shown in panels b and c
PNG_DPI = 600

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

VARIANTS = {
    1: "independent_targets",
    2: "two_sources_per_target",
    3: "shared_target",
}
CASE_TITLES = {
    1: "Case 1: Independent targets",
    2: "Case 2: Two sources per target",
    3: "Case 3: Shared target",
}
GROUP_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
    "#332288", "#88CCEE", "#44AA99", "#AA4499", "#999933", "#882255",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ["CELLATTENTION_SYNTHETIC_WORKDIR"])
PROJECT_DIR = DATASET_DIR
FONT_DIR = PROJECT_DIR / "font"
FIGURE_DIR = DATASET_DIR / "visualization/figures"


def configure_arial() -> str:
    """Register the project-local Arial family and return its family name."""
    files = [
        FONT_DIR / "arial.ttf",
        FONT_DIR / "arialbd.ttf",
        FONT_DIR / "ariali.ttf",
        FONT_DIR / "arialbi.ttf",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if not missing:
        for path in files:
            fm.fontManager.addfont(path)
        family = fm.FontProperties(fname=str(files[0])).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })
    return family


def load_dataset(variant: str):
    directory = DATASET_DIR / "data" / "preprocessed" / variant
    required = [
        "expression.csv", "spatial_coordinates.csv", "gene_ids.txt",
        "cell_groups.txt", "truth.csv",
    ]
    missing = [str(directory / name) for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError("Run data/run_preprocessing.sh first. Missing: " + ", ".join(missing))
    genes = [line.strip() for line in (directory / "gene_ids.txt").read_text().splitlines() if line.strip()]
    expression = np.loadtxt(directory / "expression.csv", delimiter=",", dtype=np.float32)
    coordinates = np.loadtxt(directory / "spatial_coordinates.csv", delimiter=",", dtype=np.float32)
    groups = np.loadtxt(directory / "cell_groups.txt", dtype=np.int16)
    truth = pd.read_csv(directory / "truth.csv")
    if expression.shape != (coordinates.shape[0], len(genes)):
        raise ValueError("Expression, coordinate, and gene dimensions are inconsistent")
    return expression, coordinates, groups, genes, truth


def clean_spatial_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial x (µm)")
    ax.set_ylabel("Spatial y (µm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, pad=1.5)


def plot_groups(ax, coordinates, groups):
    for group in sorted(np.unique(groups)):
        mask = groups == group
        label = f"Receiver {group}" if group < 6 else f"Sender {group - 6}"
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=5.0,
                   color=GROUP_COLORS[int(group)], edgecolors="none", alpha=0.88,
                   label=label, rasterized=False)
    ax.set_title("Cell-group organization", loc="left", fontweight="bold", pad=4)
    clean_spatial_axis(ax)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.92,
              borderpad=0.35, handletextpad=0.25, columnspacing=0.55,
              markerscale=1.25, labelspacing=0.25)


def plot_expression(ax, coordinates, values, title, colorbar_label, cmap):
    upper = float(np.quantile(values, 0.99))
    upper = upper if upper > 0 else 1.0
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=4.2, color="#D9D9D9",
               edgecolors="none", alpha=0.50, rasterized=False)
    positive = values > 0
    points = ax.scatter(coordinates[positive, 0], coordinates[positive, 1],
                        c=values[positive], s=6.0, cmap=cmap, vmin=0, vmax=upper,
                        edgecolors="none", alpha=0.95, rasterized=False)
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    clean_spatial_axis(ax)
    colorbar = ax.figure.colorbar(points, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label(colorbar_label, labelpad=2)
    colorbar.outline.set_linewidth(0.5)
    colorbar.ax.tick_params(length=2, width=0.5, pad=1)


def plot_network(ax, truth, compact=False):
    sources = sorted(truth["source_gene"].unique(), key=lambda x: int(x.split("_")[-1]))
    targets = sorted(truth["target_gene"].unique(), key=lambda x: int(x.split("_")[-1]))
    source_y = dict(zip(sources, np.linspace(0.88, 0.12, len(sources))))
    target_y = dict(zip(targets, np.linspace(0.88, 0.12, len(targets))))
    source_x, target_x = (0.30, 0.70) if compact else (0.18, 0.82)
    source_size = 145 if compact else 115
    target_size = 155 if compact else 125
    gene_font = 8 if compact else 6.5
    heading_font = 8.5 if compact else None
    legend_font = 6.5 if compact else None
    title_font = 9 if compact else None
    for row in truth.itertuples(index=False):
        y0, y1 = source_y[row.source_gene], target_y[row.target_gene]
        bend = 0.08 * ((int(row.receiver_group) % 3) - 1)
        arrow = FancyArrowPatch(
            (source_x + 0.045, y0), (target_x - 0.055, y1),
            connectionstyle=f"arc3,rad={bend}", arrowstyle="-|>",
            mutation_scale=7, linewidth=1.05,
            color=GROUP_COLORS[int(row.receiver_group)], alpha=0.82,
            zorder=1,
        )
        ax.add_patch(arrow)
    ax.scatter([source_x] * len(sources), list(source_y.values()), s=source_size,
               color="#56B4E9", edgecolor="#222222", linewidth=0.55, zorder=3)
    ax.scatter([target_x] * len(targets), list(target_y.values()), s=target_size,
               marker="s", color="#E69F00", edgecolor="#222222", linewidth=0.55, zorder=3)
    for gene, y in source_y.items():
        ax.text(source_x - 0.065, y, gene, ha="right", va="center", fontsize=gene_font)
    for gene, y in target_y.items():
        ax.text(target_x + 0.065, y, gene, ha="left", va="center", fontsize=gene_font)
    for group in range(6):
        ax.plot([], [], color=GROUP_COLORS[group], linewidth=1.4,
                label=f"Receiver {group}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=3,
              frameon=False, handlelength=1.2, columnspacing=0.8, fontsize=legend_font)
    ax.text(source_x, 0.98, "Source genes", ha="center", va="top", fontweight="bold", fontsize=heading_font)
    ax.text(target_x, 0.98, "Target genes", ha="center", va="top", fontweight="bold", fontsize=heading_font)
    ax.set_title("Planted regulatory network", loc="left", fontweight="bold", pad=4, fontsize=title_font)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def main():
    if DATASET_TYPE not in VARIANTS:
        raise ValueError(f"DATASET_TYPE must be one of {sorted(VARIANTS)}")
    output_format = OUTPUT_FORMAT.lower()
    if output_format not in {"png", "pdf"}:
        raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_arial()
    variant = VARIANTS[DATASET_TYPE]
    expression, coordinates, groups, genes, truth = load_dataset(variant)
    program = truth.loc[truth["receiver_group"] == RECEIVER_GROUP]
    if program.empty:
        raise ValueError(f"No truth program for receiver group {RECEIVER_GROUP}")
    source_genes = list(dict.fromkeys(program["source_gene"]))
    target_genes = list(dict.fromkeys(program["target_gene"]))
    if len(target_genes) != 1:
        raise ValueError("The selected receiver program must contain one target gene")
    gene_index = {gene: index for index, gene in enumerate(genes)}
    source_values = expression[:, [gene_index[g] for g in source_genes]].sum(axis=1)
    target_gene = target_genes[0]
    target_values = expression[:, gene_index[target_gene]]

    # Nature double-column width: 183 mm. Height remains below the 170 mm page depth.
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 160 / 25.4),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.025, h_pad=0.025, wspace=0.08, hspace=0.10)
    fig.suptitle(CASE_TITLES[DATASET_TYPE], fontsize=10, fontweight="bold")
    plot_groups(axes[0, 0], coordinates, groups)
    plot_expression(
        axes[0, 1], coordinates, source_values,
        "Source-gene expression", "Summed expression",
        mpl.colors.LinearSegmentedColormap.from_list("source", ["#F2F2F2", "#56B4E9", "#002B5B"]),
    )
    axes[0, 1].text(0.02, 0.02, " + ".join(source_genes), transform=axes[0, 1].transAxes,
                    ha="left", va="bottom", fontsize=6.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
    plot_expression(
        axes[1, 0], coordinates, target_values,
        "Target-gene expression", "Expression",
        mpl.colors.LinearSegmentedColormap.from_list("target", ["#F2F2F2", "#E69F00", "#7A3E00"]),
    )
    axes[1, 0].text(0.02, 0.02, target_gene, transform=axes[1, 0].transAxes,
                    ha="left", va="bottom", fontsize=6.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5})
    plot_network(axes[1, 1], truth)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / f"synthetic_overview_{variant}.{output_format}"
    save_options = {"facecolor": "white"}
    if output_format == "png":
        save_options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(output, **save_options)
    plt.close(fig)
    print(f"variant={variant}")
    print(f"receiver_group={RECEIVER_GROUP}")
    print(f"source_genes={','.join(source_genes)}")
    print(f"target_gene={target_gene}")
    print(f"output={output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
