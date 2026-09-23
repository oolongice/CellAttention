#!/usr/bin/env python3
"""Redraw the shared-target overview and identification benchmarks as one figure."""

# =============================================================================
# Adjustable figure and style parameters
# =============================================================================
import os

FIGURE_WIDTH_IN = 5.0
FIGURE_HEIGHT_IN = 3.5
PNG_DPI = 600
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_FIGURE_FORMAT", "png")                 # "png" or "pdf"
OUTPUT_STEM = "synthetic_combined"

FONT_SIZE = 4.5
TITLE_FONT_SIZE = 6.0                    # keep >= 4 pt
LABEL_FONT_SIZE = 4.5                    # keep >= 3 pt
TICK_FONT_SIZE = 4.0
LEGEND_FONT_SIZE = 4.0
GENE_FONT_SIZE = 4.0

GROUP_POINT_SIZE = 1.8
BACKGROUND_POINT_SIZE = 1.2
EXPRESSION_POINT_SIZE = 2.0
NETWORK_SOURCE_NODE_SIZE = 36
NETWORK_TARGET_NODE_SIZE = 40
NETWORK_EDGE_WIDTH = 0.55
NETWORK_ARROW_SIZE = 4.2
BAR_WIDTH = 0.18
BAR_EDGE_WIDTH = 0.35
ERRORBAR_WIDTH = 0.50
ERRORBAR_CAP_SIZE = 1.2
AXIS_LINE_WIDTH = 0.45
GRID_LINE_WIDTH = 0.35

LEFT_MARGIN = 0.075
RIGHT_MARGIN = 0.985
BOTTOM_MARGIN = 0.16
TOP_MARGIN = 0.96
HSPACE = 0.42
WSPACE = 0.42
IDENTIFICATION_LEGEND_Y = 0.015

RECEIVER_GROUP = 0
BOOTSTRAP_ITERATIONS = 10_000

# =============================================================================
# Imports and fixed data definitions
# =============================================================================
from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATASET_DIR = Path(os.environ["CELLATTENTION_SYNTHETIC_WORKDIR"])
PROJECT_DIR = DATASET_DIR
FONT_DIR = PROJECT_DIR / "font"
FIGURE_DIR = DATASET_DIR / "visualization/figures"
BENCHMARK_FILE = DATASET_DIR / "analysis/continuous_cell_benchmark/run_metrics.csv"

GROUP_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00",
    "#332288", "#88CCEE", "#44AA99", "#AA4499", "#999933", "#882255",
]
DATASETS = ["independent_targets", "two_sources_per_target", "shared_target"]
DATASET_LABELS = {
    "independent_targets": "Case 1",
    "two_sources_per_target": "Case 2",
    "shared_target": "Case 3",
}
METHODS = [
    "baseline_uniform", "baseline_physical",
    "transformer_uniform", "transformer_physical",
]
METHOD_LABELS = {
    "baseline_uniform": "Baseline + uniform field",
    "baseline_physical": "Baseline + physical",
    "transformer_uniform": "Transformer + uniform field",
    "transformer_physical": "Transformer + physical",
}
METHOD_COLORS = {
    "baseline_uniform": "#D9E8F0",
    "baseline_physical": "#8EC5DA",
    "transformer_uniform": "#F7D6CA",
    "transformer_physical": "#E9957C",
}


def configure_style():
    files = [
        FONT_DIR / "arial.ttf", FONT_DIR / "arialbd.ttf",
        FONT_DIR / "ariali.ttf", FONT_DIR / "arialbi.ttf",
    ]
    if all(path.is_file() for path in files):
        for path in files:
            fm.fontManager.addfont(path)
        family = fm.FontProperties(fname=str(files[0])).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": family,
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "axes.linewidth": AXIS_LINE_WIDTH,
        "xtick.major.width": AXIS_LINE_WIDTH,
        "ytick.major.width": AXIS_LINE_WIDTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_shared_target():
    directory = DATASET_DIR / "data" / "preprocessed" / "shared_target"
    genes = [x.strip() for x in (directory / "gene_ids.txt").read_text().splitlines() if x.strip()]
    expression = np.loadtxt(directory / "expression.csv", delimiter=",", dtype=np.float32)
    coordinates = np.loadtxt(directory / "spatial_coordinates.csv", delimiter=",", dtype=np.float32)
    groups = np.loadtxt(directory / "cell_groups.txt", dtype=np.int16)
    truth = pd.read_csv(directory / "truth.csv")
    return expression, coordinates, groups, genes, truth


def clean_spatial_axis(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("Spatial x (µm)", labelpad=1)
    ax.set_ylabel("Spatial y (µm)", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=1.8, pad=1)


def plot_groups(ax, coordinates, groups):
    for group in sorted(np.unique(groups)):
        mask = groups == group
        label = f"Receiver {group}" if group < 6 else f"Sender {group - 6}"
        ax.scatter(coordinates[mask, 0], coordinates[mask, 1], s=GROUP_POINT_SIZE,
                   color=GROUP_COLORS[int(group)], edgecolors="none", alpha=0.88,
                   label=label)
    ax.set_title("Cell-group organization", loc="left", fontweight="bold", pad=2)
    clean_spatial_axis(ax)


def plot_expression(ax, coordinates, values, title, label, cmap, gene_text):
    upper = max(float(np.quantile(values, 0.99)), 1.0)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=BACKGROUND_POINT_SIZE,
               color="#D9D9D9", edgecolors="none", alpha=0.5)
    positive = values > 0
    points = ax.scatter(coordinates[positive, 0], coordinates[positive, 1],
                        c=values[positive], s=EXPRESSION_POINT_SIZE, cmap=cmap,
                        vmin=0, vmax=upper, edgecolors="none", alpha=0.95)
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    clean_spatial_axis(ax)
    ax.text(0.02, 0.02, gene_text, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=GENE_FONT_SIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.7})


def plot_network(ax, truth):
    sources = sorted(truth.source_gene.unique(), key=lambda x: int(x.split("_")[-1]))
    targets = sorted(truth.target_gene.unique(), key=lambda x: int(x.split("_")[-1]))
    source_y = dict(zip(sources, np.linspace(0.86, 0.18, len(sources))))
    target_y = dict(zip(targets, np.linspace(0.86, 0.18, len(targets))))
    source_x, target_x = 0.18, 0.82
    for row in truth.itertuples(index=False):
        bend = 0.08 * ((int(row.receiver_group) % 3) - 1)
        ax.add_patch(FancyArrowPatch(
            (source_x + 0.05, source_y[row.source_gene]),
            (target_x - 0.06, target_y[row.target_gene]),
            connectionstyle=f"arc3,rad={bend}", arrowstyle="-|>",
            mutation_scale=NETWORK_ARROW_SIZE, linewidth=NETWORK_EDGE_WIDTH,
            color=GROUP_COLORS[int(row.receiver_group)], alpha=0.82, zorder=1))
    ax.scatter([source_x] * len(sources), list(source_y.values()),
               s=NETWORK_SOURCE_NODE_SIZE, color="#56B4E9", edgecolor="#222222",
               linewidth=AXIS_LINE_WIDTH, zorder=3)
    ax.scatter([target_x] * len(targets), list(target_y.values()),
               s=NETWORK_TARGET_NODE_SIZE, marker="s", color="#E69F00",
               edgecolor="#222222", linewidth=AXIS_LINE_WIDTH, zorder=3)
    for gene, y in source_y.items():
        ax.text(source_x - 0.055, y, gene, ha="right", va="center", fontsize=GENE_FONT_SIZE)
    for gene, y in target_y.items():
        ax.text(target_x + 0.055, y, gene, ha="left", va="center", fontsize=GENE_FONT_SIZE)
    for group in range(6):
        ax.plot([], [], color=GROUP_COLORS[group], linewidth=0.9, label=f"Receiver {group}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=3,
              frameon=False, handlelength=0.9, columnspacing=0.35,
              handletextpad=0.2, labelspacing=0.15)
    ax.text(source_x, 0.98, "Source genes", ha="center", va="top", fontweight="bold")
    ax.text(target_x, 0.98, "Target genes", ha="center", va="top", fontweight="bold")
    ax.set_title("Case 3: Shared-target regulatory network", loc="left", fontweight="bold", pad=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def plot_identification(ax, table, metric, title, metric_seed):
    x = np.arange(len(DATASETS))
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * BAR_WIDTH
    for method_index, (offset, method) in enumerate(zip(offsets, METHODS)):
        means, lower, upper = [], [], []
        for dataset_index, dataset in enumerate(DATASETS):
            values = table.loc[(table.dataset == dataset) & (table.method == method), metric].to_numpy(float)
            mean = values.mean()
            rng = np.random.default_rng(20260817 + metric_seed * 1000 + method_index * 100 + dataset_index)
            boot = rng.choice(values, size=(BOOTSTRAP_ITERATIONS, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(boot, [0.025, 0.975])
            means.append(mean); lower.append(mean - low); upper.append(high - mean)
        means = np.asarray(means)
        ax.bar(x + offset, means, BAR_WIDTH * 0.86, color=METHOD_COLORS[method],
               edgecolor="white", linewidth=BAR_EDGE_WIDTH,
               label=METHOD_LABELS[method], zorder=2)
        ax.errorbar(x + offset, means, yerr=np.vstack([lower, upper]), fmt="none",
                    ecolor="#4A4A4A", elinewidth=ERRORBAR_WIDTH,
                    capsize=ERRORBAR_CAP_SIZE, capthick=ERRORBAR_WIDTH, zorder=3)
    ax.set_title(title, loc="left", fontweight="bold", pad=2)
    ax.set_xticks(x, [DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylim(0, 1.03); ax.set_yticks(np.linspace(0, 1, 6)); ax.set_ylabel("Score", labelpad=1)
    ax.grid(axis="y", color="#E5E5E5", linewidth=GRID_LINE_WIDTH, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=1.8, pad=1)


def main():
    configure_style()
    expression, coordinates, groups, genes, truth = load_shared_target()
    program = truth.loc[truth.receiver_group == RECEIVER_GROUP]
    source_genes = list(dict.fromkeys(program.source_gene))
    target_gene = list(dict.fromkeys(program.target_gene))[0]
    gene_index = {gene: index for index, gene in enumerate(genes)}
    source_values = expression[:, [gene_index[g] for g in source_genes]].sum(axis=1)
    target_values = expression[:, gene_index[target_gene]]
    benchmark = pd.read_csv(BENCHMARK_FILE)

    fig, axes = plt.subplots(2, 3, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    fig.subplots_adjust(left=LEFT_MARGIN, right=RIGHT_MARGIN, bottom=BOTTOM_MARGIN,
                        top=TOP_MARGIN, wspace=WSPACE, hspace=HSPACE)
    plot_groups(axes[0, 0], coordinates, groups)
    plot_expression(
        axes[0, 1], coordinates, source_values, "Source-gene expression",
        "Summed expression",
        mpl.colors.LinearSegmentedColormap.from_list("source", ["#F2F2F2", "#56B4E9", "#002B5B"]),
        " + ".join(source_genes))
    plot_expression(
        axes[0, 2], coordinates, target_values, "Target-gene expression", "Expression",
        mpl.colors.LinearSegmentedColormap.from_list("target", ["#F2F2F2", "#E69F00", "#7A3E00"]),
        target_gene)
    plot_network(axes[1, 0], truth)
    plot_identification(axes[1, 1], benchmark, "cell_target_normalized_log_rank",
                        "Target identification", 0)
    plot_identification(axes[1, 2], benchmark,
                        "cell_source_normalized_log_rank_given_true_target",
                        "Source identification", 1)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.68, IDENTIFICATION_LEGEND_Y),
               ncol=2, frameon=False, handlelength=1.1, handletextpad=0.3,
               columnspacing=0.6, labelspacing=0.2)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / f"{OUTPUT_STEM}.{OUTPUT_FORMAT.lower()}"
    options = {"facecolor": "white"}
    if OUTPUT_FORMAT.lower() == "png":
        options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(output, **options)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
