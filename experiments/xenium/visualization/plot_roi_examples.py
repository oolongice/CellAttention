#!/usr/bin/env python3
"""Render the selected CCC Module 1 ROI overview and C7–NTN4 network."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label

from roi_network_layout import stress_layout


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CELL_LABELS = {"Fibroblast/Myofibroblast": "Fibro/Myofibro",
               "Monocyte/Macrophage": "Mono/Macro",
               "High_proliferating_hepatocyte": "Prolif. hep."}


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, facecolor="white", pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return path


def overview(workdir: Path) -> Path:
    pre = workdir / "data/preprocessed"
    xy = np.loadtxt(pre / "spatial_coordinates.csv", delimiter=",")
    samples = np.asarray((pre / "sample_ids.txt").read_text().splitlines())
    clusters = pd.read_csv(workdir / "analysis/results/receiver_group_assignments.csv",
                           usecols=["receiver_group"]).receiver_group.to_numpy(dtype=int) + 1
    if not (len(xy) == len(samples) == len(clusters)):
        raise ValueError("coordinates, samples, and receiver groups have different row counts")
    cancer = samples == "livercancer"
    xy, clusters = xy[cancer], clusters[cancer]
    member = clusters == 1
    x0, x1 = xy[:, 0].min(), xy[:, 0].max()
    y0, y1 = xy[:, 1].min(), xy[:, 1].max()
    hist, _, _ = np.histogram2d(xy[member, 0], xy[member, 1], bins=360,
                                range=[[x0, x1], [y0, y1]])
    density = gaussian_filter(hist.T, 3.2)
    occupied = density[density > 0]
    threshold = max(np.quantile(occupied, .58), density.max() * .12)
    regions, count = label(density >= threshold)
    mask = np.zeros_like(regions, dtype=bool)
    for index in range(1, count + 1):
        part = regions == index
        if part.sum() >= 14:
            mask |= part

    cases = pd.read_csv(DATA / "selected_rois.csv").query("cluster == 1")
    if len(cases) != 2:
        raise ValueError("expected two selected ROIs for CCC Module 1")
    fig, ax = plt.subplots(figsize=(3.2, 1.48))
    fig.subplots_adjust(left=.015, right=.985, bottom=.025, top=.88)
    ax.scatter(xy[:, 0], xy[:, 1], s=.10, c="#D9D9D9", alpha=.38,
               linewidths=0, rasterized=True)
    ax.scatter(xy[member, 0], xy[member, 1], s=.18, c="#6A8FB3", alpha=.82,
               linewidths=0, rasterized=True)
    ax.contour(mask.astype(float), levels=[.5], extent=(x0, x1, y0, y1),
               colors=["#355F82"], linewidths=.75)
    for number, (case, color) in enumerate(zip(cases.itertuples(), ("#C75B4E", "#D39A24")), 1):
        left, top = case.center_x - 520.0, case.center_y - 520.0
        ax.add_patch(Rectangle((left, top), 1040.0, 1040.0,
                               fill=False, edgecolor=color, linewidth=1.2, zorder=8))
        ax.text(left, top - 55, f"ROI {number}", color=color, fontsize=4.5,
                fontweight="bold", ha="left", va="bottom", zorder=9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("CCC Module 1 in liver cancer with selected ROIs",
                 loc="left", fontweight="bold", fontsize=5.5)
    return save(fig, workdir / "visualization/figures/cluster_1_overview.png")


def network(workdir: Path) -> Path:
    paths = pd.read_csv(DATA / "cluster_1_C7_to_NTN4_roi_network.csv")
    nodes = []
    edges = []
    for row in paths.itertuples():
        chain = [("cell", row.sender_cell_type), ("source", row.source_gene),
                 ("target", row.target_gene), ("cell", row.receiver_cell_type)]
        for node in chain:
            if node not in nodes:
                nodes.append(node)
        edges.extend((left, right, bool(row.focal))
                     for left, right in zip(chain[:-1], chain[1:]))
    positions = stress_layout(
        nodes, edges, "1:C7:NTN4", figure_width_in=3.0, figure_height_in=1.8,
        axes_width_fraction=.98, axes_height_fraction=.70, node_font_size=4.0,
        gene_node_size=130, cell_node_size=80, cell_labels=CELL_LABELS,
        display_names={}, restarts=8, edge_length_in=.36,
        collision_padding_in=.025, collision_weight=7.0)
    fig, ax = plt.subplots(figsize=(3.0, 1.8))
    fig.subplots_adjust(left=.01, right=.99, bottom=.14, top=.84)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for focal_value in (False, True):
        for left, right, focal in edges:
            if focal != focal_value:
                continue
            x1, y1 = positions[left]
            x2, y2 = positions[right]
            color = "#C94343" if focal else "#222222"
            width = 1.35 if focal else .65
            if left[0] == "source" and right[0] == "target":
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                            arrowprops={"arrowstyle": "-|>", "color": color,
                                        "lw": width, "mutation_scale": 5.0,
                                        "shrinkA": 5.0, "shrinkB": 5.0}, zorder=1)
            else:
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width,
                        alpha=.9, solid_capstyle="round", zorder=1)
    for kind, color, size in (("cell", "#AAFFB5", 80),
                              ("source", "#58BAFB", 130),
                              ("target", "#F68E8E", 130)):
        selected = [node for node in nodes if node[0] == kind]
        xy = np.asarray([positions[node] for node in selected])
        ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color,
                   edgecolors="white" if kind == "cell" else "none",
                   linewidths=.5 if kind == "cell" else 0, zorder=2)
        for node, (x, y) in zip(selected, xy):
            text = CELL_LABELS.get(node[1], node[1]) if kind == "cell" else node[1]
            ax.text(x, y, text, ha="center", va="center", fontsize=4.0,
                    color="#111111", zorder=3,
                    path_effects=[patheffects.Stroke(linewidth=1.2, foreground="white"),
                                  patheffects.Normal()])
    return save(fig, workdir / "visualization/figures/cluster_1_C7_to_NTN4_roi_network.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True, type=Path)
    workdir = parser.parse_args().workdir.expanduser().resolve()
    print(f"output={overview(workdir)}")
    print(f"output={network(workdir)}")


if __name__ == "__main__":
    main()
