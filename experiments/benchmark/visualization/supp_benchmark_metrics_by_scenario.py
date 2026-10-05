#!/usr/bin/env python3
"""Plot supplementary recovery benchmarks separately for each synthetic case."""

from pathlib import Path
import os

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ['CELLATTENTION_BENCHMARK_WORKDIR'])
INPUT = ROOT / "tables" / "scenario_benchmark" / "receiver_source_target_results_by_seed.csv"
OUTPUT = ROOT / 'visualization/supp_figures'
MM = 1 / 25.4
COMBINED_FIGSIZE = (150 * MM, 72 * MM)
FIGSIZE = (160 * MM, 88 * MM)
DPI = 600

CASES = {
    "simple": "Simple",
    "complex_lr": "Complex LR",
    "spatial_overlap": "Spatial overlap",
}

# Keep the same order, grouping, and colors as the main benchmark figure.
METHODS = [
    "cellattention",
    "misty_no_gene_role_prior",
    "misty_receiver",
    None,
    None,
    "holonet_adapted",
    "commot_partial_adapted",
    "commot_adapted",
]
POSITIONS = np.array([0, 2, 3, 4, 5, 7, 8, 9], dtype=float)
METHOD_LABELS = [
    "CellAttention",
    "MISTy\nall genes",
    "MISTy\nsource/target\nspecified",
    "HoloNet\nno database",
    "COMMOT\nno database",
    "HoloNet\nfull database",
    "COMMOT\npartial database",
    "COMMOT\nfull database",
]
COLORS = ["#7b3294", "#4c78a8", "#72b7b2", "#bdbdbd", "#bdbdbd", "#e07b39", "#f6c85f", "#f2b134"]
GROUPS = [
    (-0.55, 0.55, "Proposed"),
    (1.45, 5.55, "No ligand–receptor database"),
    (6.45, 9.55, "Ligand–receptor database"),
]
METRICS = [
    ("average_precision", "Average precision", "average_precision", True),
    ("mean_reciprocal_rank", "Mean reciprocal rank", "mean_reciprocal_rank", True),
    ("mean_true_edge_rank", "Mean rank", "mean_rank", False),
    ("top_k_recall", "Top-k recall", "top_k_recall", True),
]


def configure_font():
    font_dir = ROOT.parents[1] / "font"
    candidates = [font_dir / name for name in
                  ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")]
    missing = [str(path) for path in candidates if not path.exists()]
    if missing:
        family = "DejaVu Sans"
    else:
        for path in candidates:
            fm.fontManager.addfont(path)
        family = fm.FontProperties(fname=candidates[0]).get_name()
    mpl.rcParams.update({
        "font.family": family,
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def value_label(value, metric):
    return f"{value:.1f}" if metric == "mean_true_edge_rank" else f"{value:.3f}"


def plot_case_metric(case_data, case_label, metric, metric_label, filename, bounded):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, (left, right, label) in enumerate(GROUPS):
        ax.axvspan(left, right, color=("#f2e8f5", "#eaf2f8", "#fff3df")[index], alpha=0.78, zorder=0)
        ax.text((left + right) / 2, 1.025, label, ha="center", va="bottom",
                transform=ax.get_xaxis_transform(), fontsize=6.3, fontweight="bold", clip_on=False)

    displayed = []
    upper_values = []
    for pos, method, color in zip(POSITIONS, METHODS, COLORS):
        if method is None:
            displayed.append(None)
            continue
        values = case_data.loc[case_data["method"] == method, metric].dropna().to_numpy()
        if values.size == 0:
            displayed.append(None)
            continue
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
        ax.bar(pos, mean, width=0.72, color=color, edgecolor="white", linewidth=0.5, zorder=2)
        ax.errorbar(pos, mean, yerr=sd, color="#303030", capsize=2, lw=0.65, zorder=3)
        displayed.append(mean)
        upper_values.append(mean + sd)

    ymax_data = max(upper_values) if upper_values else 1.0
    ymax = (1.0 if bounded else ymax_data * 1.30)
    if bounded and ymax_data > 1:
        ymax = ymax_data * 1.18
    ax.set_ylim(0, ymax)
    label_offset = ymax * 0.025
    for pos, mean in zip(POSITIONS, displayed):
        if mean is None:
            ax.text(pos, ymax * 0.035, "N/A", ha="center", va="bottom", fontsize=6, color="#666666")
        else:
            ax.text(pos, min(mean + label_offset, ymax * 0.965), value_label(mean, metric),
                    ha="center", va="bottom", fontsize=5.7, color="#303030")

    if metric == "average_precision":
        random_ap = float(case_data["random_ap"].mean())
        ax.axhline(random_ap, color="#b2182b", ls="--", lw=0.8, zorder=1)
        ax.text(9.55, random_ap + ymax * 0.018, f"Random AP {random_ap:.3f}",
                ha="right", va="bottom", fontsize=5.8, color="#8e1424")

    ax.set_xticks(POSITIONS)
    ax.set_xticklabels(METHOD_LABELS, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.65, 9.65)
    direction = "Higher is better" if bounded else "Lower is better"
    ax.set_ylabel(f"{metric_label}\n({direction})")
    ax.set_title(f"{case_label}: {metric_label}", loc="left", fontweight="bold", pad=14)
    ax.grid(axis="y", color="#d9d9d9", lw=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.6)
    ax.tick_params(width=0.55, length=2.5)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.30, top=0.82)
    fig.savefig(filename, dpi=DPI, facecolor="white")
    fig.savefig(filename.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)



def plot_combined_metrics(case_data, case_label, filename):
    """Draw four horizontally aligned metric axes with one shared method axis."""
    row_positions = np.arange(len(METHODS))
    fig, axes = plt.subplots(1, len(METRICS), figsize=COMBINED_FIGSIZE, sharey=True,
        gridspec_kw={"wspace": 0.10, "width_ratios": [1, 1, 1.08, 1]})
    group_rows = [(0, 0, "#f2e8f5"), (1, 4, "#eaf2f8"), (5, 7, "#fff3df")]
    for ax, (metric, metric_label, _stem, bounded) in zip(axes, METRICS):
        for first, last, color in group_rows:
            ax.axhspan(first - 0.48, last + 0.48, color=color, alpha=0.78, zorder=0)
        for boundary in (0.5, 4.5):
            ax.axhline(boundary, color="white", lw=1.2, zorder=1)
        values_for_limit, plotted = [], []
        for row, method, color in zip(row_positions, METHODS, COLORS):
            if method is None:
                plotted.append(None)
                continue
            values = case_data.loc[case_data["method"] == method, metric].dropna().to_numpy()
            if values.size == 0:
                plotted.append(None)
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
            ax.errorbar(mean, row, xerr=sd, fmt="o", ms=4.0, mfc=color, mec="white",
                        mew=0.45, ecolor="#3f3f3f", elinewidth=0.7, capsize=2,
                        capthick=0.7, zorder=3)
            plotted.append(mean)
            values_for_limit.extend((max(0, mean - sd), mean + sd))
        xmax = 1.0 if bounded else max(values_for_limit) * 1.12
        if bounded and max(values_for_limit, default=1) > 1:
            xmax = max(values_for_limit) * 1.08
        ax.set_xlim(0, xmax)
        for row, mean in zip(row_positions, plotted):
            if mean is None:
                ax.text(xmax * 0.50, row, "N/A", ha="center", va="center",
                        fontsize=5.2, color="#777777", zorder=4)
        if metric == "average_precision":
            random_ap = float(case_data["random_ap"].mean())
            ax.axvline(random_ap, color="#b2182b", ls="--", lw=0.75, zorder=2)
            ax.text(random_ap + xmax * 0.025, 7.42, "Random", ha="left", va="bottom",
                    fontsize=5.1, color="#8e1424", rotation=90, clip_on=False)
        direction = "Higher is better" if bounded else "Lower is better"
        panel_label = {
            "Average precision": "Average\nprecision",
            "Mean reciprocal rank": "Mean reciprocal\nrank",
        }.get(metric_label, metric_label)
        ax.set_title(f"{panel_label}\n{direction}", fontsize=6.5, fontweight="bold", pad=4)
        ax.grid(axis="x", color="#d2d2d2", lw=0.4, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.55)
        ax.tick_params(axis="x", width=0.5, length=2.5, labelsize=5.7, labelrotation=90)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(row_positions)
    axes[0].set_yticklabels([label.replace("\n", " ") for label in METHOD_LABELS], fontsize=5.8)
    axes[0].invert_yaxis()
    fig.suptitle(case_label, x=0.015, y=0.985, ha="left", va="top",
                 fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.275, right=0.992, bottom=0.14, top=0.78)
    fig.savefig(filename, dpi=DPI, facecolor="white")
    fig.savefig(filename.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def main():
    configure_font()
    detail = pd.read_csv(INPUT)
    for case, case_label in CASES.items():
        case_data = detail.loc[detail["difficulty"] == case].copy()
        if case_data.empty:
            continue
        case_dir = OUTPUT / case
        case_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for metric, metric_label, stem, bounded in METRICS:
            plot_case_metric(case_data, case_label, metric, metric_label,
                             case_dir / f"benchmark_{stem}.png", bounded)
            summary = case_data.groupby("method")[metric].agg(["mean", "std", "count"]).reset_index()
            summary.insert(0, "metric", metric)
            summaries.append(summary)
        plot_combined_metrics(case_data, case_label, case_dir / "benchmark_metrics.png")
        pd.concat(summaries, ignore_index=True).to_csv(case_dir / "benchmark_metric_summary.csv", index=False)
        print(f"Wrote benchmark plots, including the combined panel, to {case_dir}")


if __name__ == "__main__":
    main()
