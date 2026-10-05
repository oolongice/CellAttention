#!/usr/bin/env python3
"""Detailed three-layer database support diagrams underlying the Slide-seqV2 UpSet plot."""

from pathlib import Path
import colorsys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
import numpy as np
import pandas as pd


from slideseq_common import HERE, DATASET, PROJECT
OUT = HERE / "supp_figures" / "database_support_relations"
PDF_OUT = HERE / "supp_figures/database_support_relations"
DPI = 600
WIDTH_MM = 160
MIN_SUPPORTED_EDGES = 10

DATABASES = {
    "cellchatdb": {
        "label": "CellChatDB",
        "input": DATASET / "analysis/cellchat_support_expanded/results/lr_distance_matched_enrichment_long.csv",
        "face": "#EAF3FA", "edge": "#6B9CBD", "text": "#24455B",
    },
    "neuronchatdb": {
        "label": "NeuronChatDB",
        "input": DATASET / "analysis/neuronchat_support_expanded/results/lr_distance_matched_enrichment_long.csv",
        "face": "#F0ECF8", "edge": "#8C78B4", "text": "#4B3D64",
    },
    "metachatdb": {
        "label": "MetaChatDB",
        "input": DATASET / "analysis/metachat_support_expanded/results/lr_distance_matched_enrichment_long.csv",
        "face": "#E9F5EE", "edge": "#6FA184", "text": "#315744",
    },
}


def style():
    font_dir = DATASET / "font"
    fonts = [font_dir / name for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")]
    if all(font.is_file() for font in fonts):
        for font in fonts:
            fm.fontManager.addfont(font)
        family = fm.FontProperties(fname=str(fonts[0])).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": family,
        "font.sans-serif": [family],
        "font.size": 5,
        "axes.titlesize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def select_supported_relations(path):
    data = pd.read_csv(path)
    supported = data.loc[
        (data.model_supported_edges >= MIN_SUPPORTED_EDGES) &
        (data.log2_odds_ratio > 0)
    ].copy()
    order = ["source_gene", "target_gene", "model_supported_edges", "log2_odds_ratio", "lr_pair"]
    supported = supported.sort_values(
        order, ascending=[True, True, False, False, True]
    )
    selected = supported.groupby(["source_gene", "target_gene"], sort=False).head(1).copy()
    return selected.sort_values(
        ["model_supported_edges", "log2_odds_ratio"],
        ascending=[False, False]
    ).reset_index(drop=True)


def categorical_colors(names):
    ordered = sorted(set(names))
    colors = {}
    golden = 0.61803398875
    for index, name in enumerate(ordered):
        hue = (0.58 + index * golden) % 1
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.48, 0.72)
        colors[name] = (red, green, blue)
    return colors


def barycentric_orders(data):
    sources = sorted(data.source_gene.unique())
    pairs = sorted(data.lr_pair.unique())
    targets = sorted(data.target_gene.unique())
    for _ in range(8):
        source_rank = {name: i for i, name in enumerate(sources)}
        target_rank = {name: i for i, name in enumerate(targets)}
        pairs = sorted(pairs, key=lambda pair: (
            data.loc[data.lr_pair.eq(pair), "source_gene"].map(source_rank).mean(),
            data.loc[data.lr_pair.eq(pair), "target_gene"].map(target_rank).mean(),
            pair,
        ))
        pair_rank = {name: i for i, name in enumerate(pairs)}
        sources = sorted(sources, key=lambda gene: (
            data.loc[data.source_gene.eq(gene), "lr_pair"].map(pair_rank).mean(), gene
        ))
        targets = sorted(targets, key=lambda gene: (
            data.loc[data.target_gene.eq(gene), "lr_pair"].map(pair_rank).mean(), gene
        ))
    return sources, pairs, targets


def positions(names):
    if len(names) == 1:
        return {names[0]: 0.5}
    values = np.linspace(0.955, 0.045, len(names))
    return dict(zip(names, values))


def curved_link(ax, x0, y0, x1, y1, color, linewidth):
    bend = (x1 - x0) * 0.46
    path = MplPath(
        [(x0, y0), (x0 + bend, y0), (x1 - bend, y1), (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, facecolor="none", edgecolor=color,
                           linewidth=linewidth, alpha=0.34, zorder=1))


def plot_database(name, spec):
    selected = select_supported_relations(spec["input"])
    sources, pairs, targets = barycentric_orders(selected)
    source_y, pair_y, target_y = map(positions, (sources, pairs, targets))
    max_nodes = max(len(sources), len(pairs), len(targets))
    height_mm = max(70, 18 + 2.35 * max_nodes)
    fig, ax = plt.subplots(figsize=(WIDTH_MM / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.94)
    xs = [0.055, 0.50, 0.945]
    source_colors = categorical_colors(sources)
    support = selected.model_supported_edges.to_numpy(float)
    lo, hi = np.sqrt(support.min()), np.sqrt(support.max())

    for row in selected.itertuples():
        scaled = 0.45 if hi == lo else 0.35 + 1.05 * (np.sqrt(row.model_supported_edges) - lo) / (hi - lo)
        color = source_colors[row.source_gene]
        curved_link(ax, xs[0] + 0.012, source_y[row.source_gene],
                    xs[1] - 0.13, pair_y[row.lr_pair], color, scaled)
        curved_link(ax, xs[1] + 0.13, pair_y[row.lr_pair],
                    xs[2] - 0.012, target_y[row.target_gene], color, scaled)

    node_height = min(0.012, 0.70 / max_nodes)
    for gene in sources:
        y = source_y[gene]
        ax.add_patch(Rectangle((xs[0] - 0.008, y - node_height / 2), 0.016, node_height,
                               facecolor=source_colors[gene], edgecolor="#333333",
                               linewidth=0.35, zorder=3))
        ax.text(xs[0] + 0.013, y, gene, ha="left", va="center", fontsize=5, zorder=4)

    for pair in pairs:
        y = pair_y[pair]
        ax.add_patch(FancyBboxPatch(
            (xs[1] - 0.13, y - node_height / 2), 0.26, node_height,
            boxstyle="round,pad=0.001,rounding_size=0.0025",
            facecolor=spec["face"], edgecolor=spec["edge"],
            linewidth=0.45, zorder=3
        ))
        label = str(pair).replace(" -> ", "  →  ").replace("->", "→")
        ax.text(xs[1], y, label, ha="center", va="center", fontsize=5,
                color=spec["text"], fontweight="semibold", zorder=4)

    for gene in targets:
        y = target_y[gene]
        ax.add_patch(Rectangle((xs[2] - 0.008, y - node_height / 2), 0.016, node_height,
                               facecolor="#E9A6B5", edgecolor="#333333",
                               linewidth=0.35, zorder=3))
        ax.text(xs[2] - 0.013, y, gene, ha="right", va="center", fontsize=5, zorder=4)

    for x, label in zip(xs, ("Source gene", "Database gene pair", "Target gene")):
        ax.text(x, 0.997, label, ha="center", va="bottom", fontsize=6,
                fontweight="bold", color=spec["edge"] if x == xs[1] else "#202020")
    ax.set_title(f"{spec['label']} supported model gene pairs (n = {len(selected)})",
                 loc="left", fontsize=7, fontweight="bold", pad=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.axis("off")

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}_source_database_pair_target.png"
    pdf = PDF_OUT / f"{name}_source_database_pair_target.pdf"
    csv = OUT / f"{name}_source_database_pair_target.csv"
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=DPI, facecolor="white", pil_kwargs={"compress_level": 6})
    fig.savefig(pdf, format="pdf", facecolor="white")
    plt.close(fig)
    selected.to_csv(csv, index=False)
    print(f"{spec['label']}: pairs={len(selected)}, size={WIDTH_MM:.0f}x{height_mm:.1f} mm")
    print(png)


def compact_gene_label(value):
    text = str(value)
    if len(text) <= 10:
        return text
    split = (len(text) + 1) // 2
    return text[:split] + "\n" + text[split:]


def plot_database_compact(name, spec):
    selected = select_supported_relations(spec["input"])
    selected = selected.sort_values(
        ["source_gene", "target_gene", "model_supported_edges", "lr_pair"],
        ascending=[True, True, False, True]
    ).reset_index(drop=True)
    columns = min(4, max(2, int(np.ceil(len(selected) / 20))))
    rows_per_column = int(np.ceil(len(selected) / columns))
    height_mm = max(52, 16 + 3.3 * rows_per_column)

    fig, axes = plt.subplots(
        1, columns, figsize=(WIDTH_MM / 25.4, height_mm / 25.4),
        squeeze=False, gridspec_kw={"wspace": 0.055}
    )
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.90)
    axes = axes.ravel()
    support = selected.model_supported_edges.to_numpy(float)
    support_min, support_max = support.min(), support.max()

    for column, ax in enumerate(axes):
        start = column * rows_per_column
        part = selected.iloc[start:start + rows_per_column].reset_index(drop=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(rows_per_column - 0.35, -1.35)
        ax.axis("off")
        for xpos, header in ((0.13, "Source"), (0.50, "Database pair"), (0.87, "Target")):
            ax.text(xpos, -0.83, header, ha="center", va="center",
                    fontsize=5.5, fontweight="bold",
                    color=spec["edge"] if xpos == 0.50 else "#202020")
        ax.plot([0.01, 0.99], [-0.38, -0.38], color="#AFAFAF", lw=0.4)

        for row_index, row in part.iterrows():
            if row_index % 2 == 0:
                ax.add_patch(Rectangle((0.005, row_index - 0.47), 0.99, 0.94,
                                       facecolor="#F7F7F7", edgecolor="none", zorder=0))
            strength = 0.35 if support_max == support_min else (
                0.24 + 0.38 * (row.model_supported_edges - support_min) /
                (support_max - support_min)
            )
            pair_text = str(row.lr_pair).replace("->", "→")
            if " → " in pair_text:
                pair_left, pair_right = pair_text.split(" → ", 1)
                pair_text = f"{pair_left} →\n{pair_right}"
            ax.add_patch(FancyBboxPatch(
                (0.285, row_index - 0.39), 0.43, 0.78,
                boxstyle="round,pad=0.004,rounding_size=0.018",
                facecolor=mpl.colors.to_rgba(spec["face"], strength + 0.35),
                edgecolor=spec["edge"], linewidth=0.4, zorder=1
            ))
            ax.text(0.13, row_index, compact_gene_label(row.source_gene),
                    ha="center", va="center", fontsize=5, linespacing=0.78)
            ax.text(0.255, row_index, "→", ha="center", va="center",
                    fontsize=5, color="#777777")
            ax.text(0.50, row_index, pair_text, ha="center", va="center",
                    fontsize=5, linespacing=0.76, color=spec["text"],
                    fontweight="semibold")
            ax.text(0.745, row_index, "→", ha="center", va="center",
                    fontsize=5, color="#777777")
            ax.text(0.87, row_index, compact_gene_label(row.target_gene),
                    ha="center", va="center", fontsize=5, linespacing=0.78)

    fig.suptitle(
        f"{spec['label']} supported model gene pairs (n = {len(selected)})",
        x=0.012, y=0.975, ha="left", va="top", fontsize=7, fontweight="bold"
    )
    fig.text(0.988, 0.972, "Darker pair boxes indicate more supported model edges",
             ha="right", va="top", fontsize=5, color="#555555")

    OUT.mkdir(parents=True, exist_ok=True)
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{name}_source_database_pair_target.png"
    pdf = PDF_OUT / f"{name}_source_database_pair_target.pdf"
    csv = OUT / f"{name}_source_database_pair_target.csv"
    fig.savefig(png, dpi=DPI, facecolor="white", pil_kwargs={"compress_level": 6})
    fig.savefig(pdf, format="pdf", facecolor="white")
    plt.close(fig)
    selected.to_csv(csv, index=False)
    print(f"{spec['label']}: pairs={len(selected)}, columns={columns}, "
          f"size={WIDTH_MM:.0f}x{height_mm:.1f} mm")
    print(png)



def main():
    style()
    for name, spec in DATABASES.items():
        plot_database_compact(name, spec)


if __name__ == "__main__":
    main()
