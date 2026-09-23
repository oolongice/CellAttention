#!/usr/bin/env python3
"""Plot separate multiseed Synthetic target, source, and joint benchmarks."""
PNG_DPI = 600
BOOTSTRAP_ITERATIONS = 10_000
from pathlib import Path
import os
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(os.environ["CELLATTENTION_SYNTHETIC_WORKDIR"])
INPUT = PROJECT / "analysis/continuous_cell_benchmark/run_metrics.csv"
OUTPUT = PROJECT / "visualization/figures"
PDF_OUTPUT = OUTPUT
DATASETS = ["independent_targets", "two_sources_per_target", "shared_target"]
DATASET_LABELS = {"independent_targets": "Case 1", "two_sources_per_target": "Case 2", "shared_target": "Case 3"}
METHODS = ["baseline_uniform", "baseline_physical", "transformer_uniform", "transformer_physical"]
LABELS = {"baseline_uniform": "Baseline + uniform field", "baseline_physical": "Baseline + physical", "transformer_uniform": "Transformer + uniform field", "transformer_physical": "Transformer + physical"}
COLORS = {"baseline_uniform": "#D9E8F0", "baseline_physical": "#8EC5DA", "transformer_uniform": "#F7D6CA", "transformer_physical": "#E9957C"}
FIGURES = [
    ("cell_target_normalized_log_rank", "Target identification", "synthetic_target_identification", "Target-gene recovery"),
    ("cell_source_normalized_log_rank_given_true_target", "Source identification", "synthetic_source_identification", "Source-gene recovery"),
    ("cell_joint_target_source_full_log_recovery", "Joint target–source identification", "synthetic_joint_recovery", "Source–target pair recovery"),
]


def configure_font():
    files = [PROJECT / "font/arial.ttf", PROJECT / "font/arialbd.ttf", PROJECT / "font/ariali.ttf", PROJECT / "font/arialbi.ttf"]
    if all(path.is_file() for path in files):
        for path in files:
            fm.fontManager.addfont(path)
        family = fm.FontProperties(fname=str(files[0])).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({"font.family": family, "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6, "legend.fontsize": 6.2, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": .6})


def draw_panel(ax, table, metric, title, ylabel):
    x = np.arange(len(DATASETS)); width = .18
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * width
    for method_index, (offset, method) in enumerate(zip(offsets, METHODS)):
        means, lower, upper = [], [], []
        for dataset_index, dataset in enumerate(DATASETS):
            values = table.loc[(table.dataset == dataset) & (table.method == method), metric].to_numpy(float)
            means.append(values.mean())
            rng = np.random.default_rng(20260817 + 1000 * FIGURES.index(next(item for item in FIGURES if item[0] == metric)) + 100 * method_index + dataset_index)
            boot = rng.choice(values, size=(BOOTSTRAP_ITERATIONS, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(boot, [.025, .975])
            lower.append(values.mean() - low); upper.append(high - values.mean())
        means = np.asarray(means)
        ax.bar(x + offset, means, width * .86, color=COLORS[method], edgecolor="white", linewidth=.6, label=LABELS[method], zorder=2)
        ax.errorbar(x + offset, means, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#4A4A4A", elinewidth=.85, capsize=2, capthick=.85, zorder=3)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xticks(x, [DATASET_LABELS[d] for d in DATASETS]); ax.set_ylim(0, 1.03)
    ax.set_yticks(np.linspace(0, 1, 6)); ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E5E5E5", linewidth=.55, zorder=0)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2.5, pad=2)


def main():
    configure_font(); table = pd.read_csv(INPUT)
    n = table.groupby(["dataset", "method"]).seed.nunique()
    if n.min() != n.max():
        raise ValueError("Unequal seed counts across dataset/method combinations")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PDF_OUTPUT.mkdir(parents=True, exist_ok=True)
    for metric, title, stem, ylabel in FIGURES:
        fig, ax = plt.subplots(figsize=(89 / 25.4, 105 / 25.4))
        fig.subplots_adjust(left=.18, right=.985, bottom=.13, top=.81)
        ax.set_box_aspect(1)
        draw_panel(ax, table, metric, title, ylabel)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .99), ncol=2, frameon=False, handlelength=1.6, handletextpad=.45, columnspacing=1.0)
        png_path = OUTPUT / f"{stem}.png"
        pdf_path = PDF_OUTPUT / f"{stem}.pdf"
        fig.savefig(png_path, facecolor="white", dpi=PNG_DPI, pil_kwargs={"compress_level": 6})
        fig.savefig(pdf_path, facecolor="white")
        plt.close(fig)
        print(png_path)
        print(pdf_path)


if __name__ == "__main__":
    main()
