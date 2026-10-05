#!/usr/bin/env python3
"""Database-specific five-layer Sankey plots for model-derived spatial relations.

The middle node is a database gene pair rather than separate ligand/receptor
layers. Existing validation outputs are reused; no statistical analysis is run.
"""

OUTPUT_FORMAT = "png"
PNG_DPI = 600
FIGURE_WIDTH_IN = 3.5
FIGURE_HEIGHT_IN = 1.6
NODE_LABEL_SIZE = 4.0
LAYER_LABEL_SIZE = 5.0
TITLE_SIZE = 6.5
GENE_PAIR_NODE_WIDTH = 0.255
GENE_PAIR_NODE_MIN_HEIGHT = 0.040
MAX_MODEL_RELATIONS = 10
MAX_DATABASE_PAIRS_PER_MODEL_RELATION = 1
MAX_CELL_TYPE_PAIRS_PER_PATH = 1
MIN_MODEL_SUPPORTED_EDGES = 10
REQUIRE_POSITIVE_ENRICHMENT = True

from collections import defaultdict
from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
import numpy as np
import pandas as pd

from slideseq_common import DATASET, HERE, PROJECT, configure_style, load, palette

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Keep the spatial examples shown in the companion streamline figures in the
# Sankey selection so the two visualizations can be compared directly.
STREAMLINE_SHOWCASES = {
    "protein_signaling": [
        (1, "Mylk", "Hs3st4", "Mdk -> Ncl"),
        (4, "Itga11", "Cplx1", "Lama2 -> Sv2a"),
    ],
    "neural_signaling": [
        (4, "Snhg11", "Plp1", "Nrxn2 -> Nlgn1"),
        (1, "Pvalb", "Tshz2", "Gad1+Slc32a1 -> Gabra2"),
    ],
    "metabolite_signaling": [
        (1, "Drc7", "Igfbp4", "Nadsyn1 -> Gria2"),
        (1, "Thbs4", "Igfbp4", "Aldh7a1 -> Vdac1"),
    ],
}


DATABASES = {
    "protein_signaling": {
        "result_dir": DATASET / "analysis/cellchat_support_expanded/results",
        "title": "CellChatDB support",
        "pair_label": "Gene pair",
        "pair_face": "#EAF3FA", "pair_edge": "#6B9CBD", "pair_text": "#24455B",
    },
    "neural_signaling": {
        "result_dir": DATASET / "analysis/neuronchat_support_expanded/results",
        "title": "NeuronChatDB support",
        "pair_label": "Gene pair",
        "pair_face": "#F0ECF8", "pair_edge": "#8C78B4", "pair_text": "#4B3D64",
    },
    "metabolite_signaling": {
        "result_dir": DATASET / "analysis/metachat_support_expanded/results",
        "title": "MetaChatDB support",
        "pair_label": "Gene pair",
        "pair_face": "#E9F5EE", "pair_edge": "#6FA184", "pair_text": "#315744",
    },
}
# The expanded database validations were calculated from this expanded edge set.
# Using the smaller legacy edge table leaves most validated cell IDs unmatched.
EDGES = DATASET / "analysis/cell_cell_edges_expanded/cell_cell_interaction_edges.csv"
OUT_DATA = HERE / "data/database_pair_sankey"
OUT_FIG = HERE / "figures/database_pair_sankey"
OUT_PDF = HERE / "figures/database_pair_sankey"
KEY = ["cluster", "source_gene", "target_gene"]
EDGE_KEY = KEY + ["sender_cell_id", "receiver_cell_id"]
LAYERS = ["Sender cell", "Source gene", "Database gene pair", "Target gene", "Receiver cell"]
NODE_COLORS = ["#BFD7EA", "#A8DADC", "#F5D6A6", "#F4B6C2", "#D6C4E9"]
RIBBON_COLORS = ["#4E79A7", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2", "#EDC948", "#9C755F"]


def select_relations(enrichment, database_name):
    frame = enrichment[enrichment.model_supported_edges >= MIN_MODEL_SUPPORTED_EDGES].copy()
    if REQUIRE_POSITIVE_ENRICHMENT:
        frame = frame[frame.log2_odds_ratio > 0]
    relation_order = (frame.groupby(KEY, as_index=False)
                      .agg(best_supported=("model_supported_edges", "max"),
                           best_fraction=("model_fraction", "max"),
                           best_enrichment=("log2_odds_ratio", "max"))
                      .sort_values(["best_supported", "best_fraction", "best_enrichment"], ascending=False)
                      .head(MAX_MODEL_RELATIONS))
    ranked = frame.merge(relation_order[KEY], on=KEY, how="inner")
    ranked = ranked.sort_values(KEY + ["model_supported_edges", "log2_odds_ratio"],
                                ascending=[True, True, True, False, False])
    selected = ranked.groupby(KEY, sort=False).head(MAX_DATABASE_PAIRS_PER_MODEL_RELATION)

    # A showcased pair replaces the automatically ranked pair for the same model
    # relation. Missing showcased model relations are appended.
    showcased = []
    for cluster, source, target, lr_pair in STREAMLINE_SHOWCASES[database_name]:
        match = frame[(frame.cluster == cluster) &
                      (frame.source_gene == source) &
                      (frame.target_gene == target) &
                      (frame.lr_pair == lr_pair)]
        if match.empty:
            raise RuntimeError(
                f"Missing streamline showcase in enrichment data: {database_name} "
                f"C{cluster} {source}->{target}, {lr_pair}"
            )
        same_relation = ((selected.cluster == cluster) &
                         (selected.source_gene == source) &
                         (selected.target_gene == target))
        selected = selected.loc[~same_relation]
        showcased.append(match.sort_values(
            ["model_supported_edges", "log2_odds_ratio"], ascending=False).head(1))
    return pd.concat([selected, *showcased], ignore_index=True).drop_duplicates(KEY + ["lr_pair"])


def build_paths(selected, support, edges):
    selected_keys = selected[KEY + ["lr_pair"]].drop_duplicates()
    rows = support.merge(selected_keys, on=KEY + ["lr_pair"], how="inner")
    edge_types = edges[EDGE_KEY + ["sender_cell_type", "receiver_cell_type"]].drop_duplicates(EDGE_KEY)
    rows = rows.merge(edge_types, on=EDGE_KEY, how="left", validate="many_to_one")
    rows.sender_cell_type = rows.sender_cell_type.fillna("Unannotated")
    rows.receiver_cell_type = rows.receiver_cell_type.fillna("Unannotated")
    cols = ["cluster", "sender_cell_type", "source_gene", "lr_pair", "target_gene", "receiver_cell_type"]
    paths = rows.groupby(cols, as_index=False).size().rename(columns={"size": "supported_edge_count"})
    paths = paths.sort_values(KEY + ["lr_pair", "supported_edge_count"], ascending=[True, True, True, True, False])
    paths = paths.groupby(KEY + ["lr_pair"], sort=False).head(MAX_CELL_TYPE_PAIRS_PER_PATH)
    stats = selected[KEY + ["lr_pair", "databases", "model_supported_edges", "model_fraction",
                            "background_fraction", "log2_odds_ratio", "fisher_p", "fdr_q_within_triplet"]]
    return paths.merge(stats, on=KEY + ["lr_pair"], how="left")


def barycentric_order(paths):
    values = paths[LAYERS].astype(str).to_numpy(); weights = paths.supported_edge_count.to_numpy(float)
    orders = []
    for layer in range(len(LAYERS)):
        totals = defaultdict(float)
        for value, weight in zip(values[:, layer], weights): totals[value] += weight
        orders.append(sorted(totals, key=lambda value: (-totals[value], value)))
    for _ in range(6):
        for layer in range(1, len(LAYERS)):
            previous = {value: index for index, value in enumerate(orders[layer - 1])}; scores = defaultdict(lambda: [0., 0.])
            for row, weight in zip(values, weights): scores[row[layer]][0] += previous[row[layer - 1]] * weight; scores[row[layer]][1] += weight
            orders[layer].sort(key=lambda value: (scores[value][0] / scores[value][1], value))
        for layer in range(len(LAYERS) - 2, -1, -1):
            following = {value: index for index, value in enumerate(orders[layer + 1])}; scores = defaultdict(lambda: [0., 0.])
            for row, weight in zip(values, weights): scores[row[layer]][0] += following[row[layer + 1]] * weight; scores[row[layer]][1] += weight
            orders[layer].sort(key=lambda value: (scores[value][0] / scores[value][1], value))
    return orders


def node_layout(paths, orders):
    total = float(paths.supported_edge_count.sum()); positions = {}
    for layer, names in enumerate(orders):
        # Give the central tubes more breathing room than the compact side nodes.
        gap = 0.032 if layer == 2 else 0.012
        values = paths[LAYERS[layer]].astype(str)
        totals = {name: float(paths.loc[values == name, "supported_edge_count"].sum()) for name in names}
        available = 0.88 - gap * max(0, len(names) - 1)
        if layer == 2:
            # Prevent low-count database pairs from collapsing into unreadably
            # thin strips, while assigning the remaining height by edge count.
            minimum = min(GENE_PAIR_NODE_MIN_HEIGHT, available / len(names) * .90)
            remainder = max(0.0, available - minimum * len(names))
            heights = {name: minimum + remainder * totals[name] / total for name in names}
        else:
            heights = {name: available * totals[name] / total for name in names}
        cursor = 0.055
        for name in names:
            height = heights[name]
            positions[(layer, name)] = [cursor, cursor + height]
            cursor += height + gap
    return positions


def make_links(paths, positions, source_color):
    links = []
    for layer in range(len(LAYERS) - 1):
        group_keys = list(dict.fromkeys([LAYERS[layer], LAYERS[layer + 1], LAYERS[0]]))
        grouped = paths.groupby(group_keys, as_index=False).supported_edge_count.sum()
        grouped["next_pos"] = grouped[LAYERS[layer + 1]].map(lambda x: positions[(layer + 1, str(x))][0])
        source_cursor = {key: value[0] for key, value in positions.items() if key[0] == layer}
        target_cursor = {key: value[0] for key, value in positions.items() if key[0] == layer + 1}
        outgoing = {}
        for _, row in grouped.sort_values([LAYERS[layer], "next_pos", LAYERS[0]]).iterrows():
            source, target, color_key = str(row[LAYERS[layer]]), str(row[LAYERS[layer + 1]]), str(row[LAYERS[0]])
            weight = float(row.supported_edge_count)
            denom = paths.loc[paths[LAYERS[layer]].astype(str) == source, "supported_edge_count"].sum()
            scale = (positions[(layer, source)][1] - positions[(layer, source)][0]) / denom
            y0 = source_cursor[(layer, source)]; y1 = y0 + weight * scale; source_cursor[(layer, source)] = y1
            outgoing[(source, target, color_key)] = (y0, y1, weight)
        for _, row in grouped.sort_values([LAYERS[layer + 1], LAYERS[layer], LAYERS[0]]).iterrows():
            source, target, color_key = str(row[LAYERS[layer]]), str(row[LAYERS[layer + 1]]), str(row[LAYERS[0]])
            sy0, sy1, weight = outgoing[(source, target, color_key)]
            denom = paths.loc[paths[LAYERS[layer + 1]].astype(str) == target, "supported_edge_count"].sum()
            scale = (positions[(layer + 1, target)][1] - positions[(layer + 1, target)][0]) / denom
            ty0 = target_cursor[(layer + 1, target)]; ty1 = ty0 + weight * scale; target_cursor[(layer + 1, target)] = ty1
            links.append((layer, sy0, sy1, ty0, ty1, source_color[color_key]))
    return links


def ribbon(ax, x0, x1, sy0, sy1, ty0, ty1, color):
    bend = (x1 - x0) * .45
    vertices = [(x0, sy0), (x0+bend, sy0), (x1-bend, ty0), (x1, ty0), (x1, ty1),
                (x1-bend, ty1), (x0+bend, sy1), (x0, sy1), (x0, sy0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none", alpha=.34, zorder=1))


def plot_database(name, spec, paths, annotation_colors):
    plot_paths = paths.rename(columns={"sender_cell_type": "Sender cell", "source_gene": "Source gene",
                                      "lr_pair": "Database gene pair", "target_gene": "Target gene",
                                      "receiver_cell_type": "Receiver cell"}).copy()
    orders = barycentric_order(plot_paths); positions = node_layout(plot_paths, orders)
    source_color = {value: annotation_colors.get(value, "#AFAFAF") for value in orders[0]}
    links = make_links(plot_paths, positions, source_color)
    configure_style(); fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN)); fig.subplots_adjust(left=.035, right=.965, bottom=.055, top=.94)
    # Pull the gene layers outward and reserve a wider central channel for each
    # database pair.  Cell layers stay close to their corresponding gene layer.
    xs = np.asarray([.035, .225, .500, .775, .965])
    node_widths = np.asarray([.018, .018, GENE_PAIR_NODE_WIDTH, .018, .018])
    for layer, sy0, sy1, ty0, ty1, color in links:
        ribbon(ax, xs[layer]+node_widths[layer]/2, xs[layer+1]-node_widths[layer+1]/2, sy0, sy1, ty0, ty1, color)
    layer_headers = LAYERS.copy(); layer_headers[2] = spec["pair_label"]
    pair_text_artists = []
    for layer, names in enumerate(orders):
        for value in names:
            y0, y1 = positions[(layer, value)]
            if layer == 2:
                # Use one clean vector capsule per database pair. A single path
                # stays aligned at every height and is easy to edit in Illustrator.
                node_height = y1 - y0
                node_x = xs[layer] - node_widths[layer] / 2
                corner = min(.010, max(.003, node_height * .18))
                ax.add_patch(FancyBboxPatch(
                    (node_x, y0), node_widths[layer], node_height,
                    boxstyle=f"round,pad=0,rounding_size={corner}",
                    facecolor=spec["pair_face"], edgecolor=spec["pair_edge"],
                    linewidth=.82, alpha=1.0, zorder=3,
                ))
                label = str(value).replace(" -> ", "  →  ").replace("->", "→")
                label_artist = ax.text(
                    xs[layer], (y0 + y1) / 2, label,
                    ha="center", va="center", fontsize=NODE_LABEL_SIZE,
                    fontweight="semibold", color=spec["pair_text"], zorder=4,
                )
                pair_text_artists.append((label_artist, node_x, node_widths[layer]))
            else:
                if layer in (0, len(LAYERS)-1):
                    node_color = annotation_colors.get(value, "#AFAFAF")
                else:
                    node_color = NODE_COLORS[layer]
                ax.add_patch(Rectangle((xs[layer]-node_widths[layer]/2, y0), node_widths[layer], y1-y0,
                                       facecolor=node_color, edgecolor="#333333", linewidth=.45, zorder=3))
                # Cell identity is encoded by the annotation-matched node color;
                # cell-type names are intentionally omitted in this compact view.
                if layer not in (0, len(LAYERS)-1):
                    label = "\n".join(textwrap.wrap(str(value), width=18))
                    ha = "left" if layer == 1 else "right"
                    offset = .014 if ha == "left" else -.014
                    ax.text(xs[layer]+offset, (y0+y1)/2, label, ha=ha, va="center",
                            fontsize=NODE_LABEL_SIZE, zorder=4)
        header_color = spec["pair_edge"] if layer == 2 else "#202020"
        ax.text(xs[layer], .995, layer_headers[layer], ha="center", va="bottom",
                fontsize=LAYER_LABEL_SIZE, fontweight="bold", color=header_color)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.axis("off")

    # Fit every label against its actual rendered width. All pairs stay on one
    # line; only labels that need it receive a small, bounded font reduction.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, node_x, node_width in pair_text_artists:
        left_px = ax.transData.transform((node_x + .010, 0))[0]
        right_px = ax.transData.transform((node_x + node_width - .010, 0))[0]
        available_px = right_px - left_px
        rendered_px = artist.get_window_extent(renderer=renderer).width
        if rendered_px > available_px:
            fitted_size = max(3.2, NODE_LABEL_SIZE * available_px / rendered_px)
            artist.set_fontsize(fitted_size)

    png_output = OUT_FIG / f"{name}_model_database_pair_sankey.png"
    pdf_output = OUT_PDF / f"{name}_model_database_pair_sankey.pdf"
    OUT_PDF.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, dpi=PNG_DPI, facecolor="white",
                pil_kwargs={"compress_level": 6})
    fig.savefig(pdf_output, dpi=PNG_DPI, facecolor="white")
    plt.close(fig)
    return png_output, pdf_output


def main():
    OUT_DATA.mkdir(parents=True, exist_ok=True); OUT_FIG.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(EDGES, usecols=EDGE_KEY + ["sender_cell_type", "receiver_cell_type"])
    _, annotations, _ = load()
    annotation_colors = palette(annotations)
    for name, spec in DATABASES.items():
        enrichment = pd.read_csv(spec["result_dir"] / "lr_distance_matched_enrichment_long.csv")
        support = pd.read_csv(spec["result_dir"] / "model_edge_lr_support.csv")
        selected = select_relations(enrichment, name); paths = build_paths(selected, support, edges)
        if paths.empty:
            print(f"{name}: no supported paths", file=sys.stderr); continue
        selected.to_csv(OUT_DATA / f"{name}_selected_relations.csv", index=False)
        paths.to_csv(OUT_DATA / f"{name}_sankey_paths.csv", index=False)
        png_output, pdf_output = plot_database(name, spec, paths, annotation_colors)
        print(png_output); print(pdf_output)


if __name__ == "__main__":
    main()
