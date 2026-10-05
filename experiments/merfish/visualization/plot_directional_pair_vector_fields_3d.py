#!/usr/bin/env python3
"""Render local 3D CCI directions as spatial cone-arrow vector fields."""
import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import plot_directional_pairs_3d as shared
from directional_edges import load_edges, pair_edges
from merfish_common import (DATASET, FONT_REGULAR, FIGURES,
                            HERE, PNG_DPI, PRE, configure_style)

OUT_FIG = FIGURES / "directional_pair_vector_fields_3d"
OUT_DATA = HERE / "data/directional_pair_vector_fields_3d"

# ---- Local spatial aggregation parameters (safe to adjust) ----
LOCAL_BIN_SIZE = 30.
MIN_RECEIVERS_PER_BIN = 1
MIN_DIRECTIONAL_COHERENCE = 0.
MAX_ARROWS_PER_FIGURE = 180

# ---- Arrow display parameters (safe to adjust) ----
ARROW_MIN_LENGTH = 16.
ARROW_MAX_LENGTH = 34.
ARROW_SHAFT_RADIUS = .055
ARROW_TIP_RADIUS = .16
ARROW_TIP_LENGTH = .38
ARROW_COLOR = "#202020"
ARROW_OPACITY = .96
CLUSTER_SURFACE_OPACITY = .32
TARGET_ENVELOPE_COLOR = "#B8D8EA"

# ---- Figure, font, and legend parameters ----
WINDOW_SIZE = (1900, 1600)
CAMERA_ZOOM = 1.34
LEGEND_FONT_SIZE = 50
LEGEND_LINE_HEIGHT = 72
LEGEND_MARKER_SIZE = 32
LEGEND_LEFT = 64
LEGEND_TOP = 58
LEGEND_BACKGROUND_ALPHA = 225


def safe_name(value):
    return str(value).replace("/", "_").replace(":", "_").replace(" ", "_")


def add_legend(path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    legend_font = ImageFont.truetype(str(FONT_REGULAR), LEGEND_FONT_SIZE)
    entries = [
        (shared.TISSUE_COLOR, "Whole-tissue envelope"),
        (TARGET_ENVELOPE_COLOR, "Target-cluster envelope"),
        (ARROW_COLOR, "Supported-edge direction"),
    ]
    text_width = max(draw.textbbox((0, 0), label, font=legend_font)[2]
                     for _, label in entries)
    box_width = text_width + 92
    box_height = len(entries) * LEGEND_LINE_HEIGHT + 28
    left, top = LEGEND_LEFT, LEGEND_TOP
    draw.rounded_rectangle((left - 22, top, left + box_width, top + box_height),
                           radius=10, fill=(255, 255, 255, LEGEND_BACKGROUND_ALPHA))
    for index, (color, label) in enumerate(entries):
        y = top + 16 + index * LEGEND_LINE_HEIGHT + LEGEND_LINE_HEIGHT / 2
        half = LEGEND_MARKER_SIZE / 2
        draw.ellipse((left, y - half, left + LEGEND_MARKER_SIZE, y + half), fill=color)
        draw.text((left + LEGEND_MARKER_SIZE + 17, y), label, font=legend_font,
                  fill=(32, 32, 32, 255), anchor="lm")

    image.save(path, dpi=(PNG_DPI, PNG_DPI), compress_level=6)


def local_vectors(coordinates, display, receiver_indices, edge):
    receivers = edge.receiver_index.to_numpy(dtype=int)
    sources = edge.sender_index.to_numpy(dtype=int)
    receiver_rows = np.searchsorted(receiver_indices, receivers)
    assert np.array_equal(receiver_indices[receiver_rows], receivers)
    distance = edge.distance.to_numpy()
    weight = edge.influence_score.to_numpy()
    direction = coordinates[receivers] - coordinates[sources]
    direction /= np.maximum(distance[:, None], shared.MIN_DISTANCE)

    lower = display[receiver_indices].min(axis=0)
    bins = np.floor((display[receiver_indices] - lower) / LOCAL_BIN_SIZE).astype(int)
    dimensions = bins.max(axis=0) + 1
    receiver_bin = np.ravel_multi_index(bins.T, dimensions)
    edge_bin = receiver_bin[receiver_rows]
    number_bins = int(np.prod(dimensions))
    total_weight = np.bincount(edge_bin, weights=weight, minlength=number_bins)
    vector_sum = np.column_stack([
        np.bincount(edge_bin, weights=weight * direction[:, axis], minlength=number_bins)
        for axis in range(3)
    ])
    mean = np.divide(vector_sum, total_weight[:, None], out=np.zeros_like(vector_sum),
                     where=total_weight[:, None] > 0)
    coherence = np.linalg.norm(mean, axis=1)
    active_receiver = np.bincount(receiver_rows, weights=weight,
                                  minlength=len(receiver_indices)) > 0
    receiver_count = np.bincount(receiver_bin[active_receiver], minlength=number_bins)
    valid = ((receiver_count >= MIN_RECEIVERS_PER_BIN) &
             (coherence >= MIN_DIRECTIONAL_COHERENCE) & (total_weight > 0))
    selected_bins = np.flatnonzero(valid)
    if len(selected_bins) > MAX_ARROWS_PER_FIGURE:
        score = total_weight[selected_bins] * coherence[selected_bins]
        selected_bins = selected_bins[np.argsort(score)[-MAX_ARROWS_PER_FIGURE:]]

    bin_coordinates = np.asarray(np.unravel_index(selected_bins, dimensions)).T
    positions = lower + (bin_coordinates + .5) * LOCAL_BIN_SIZE
    vectors = mean[selected_bins]
    vectors[:, 2] *= shared.Z_DISPLAY_SCALE
    vector_norm = np.linalg.norm(vectors, axis=1)
    directions = np.divide(vectors, vector_norm[:, None], out=np.zeros_like(vectors),
                           where=vector_norm[:, None] > 0)
    coherence_selected = coherence[selected_bins]
    lengths = ARROW_MIN_LENGTH + (ARROW_MAX_LENGTH - ARROW_MIN_LENGTH) * np.clip(
        coherence_selected, 0, 1)
    return positions, directions, lengths, coherence_selected, total_weight[selected_bins], receiver_count[selected_bins]


def arrow_mesh(positions, directions, lengths):
    arrows = [pv.Arrow(start=position, direction=direction,
                       tip_length=ARROW_TIP_LENGTH, tip_radius=ARROW_TIP_RADIUS,
                       shaft_radius=ARROW_SHAFT_RADIUS, scale=float(length))
              for position, direction, length in zip(positions, directions, lengths)]
    if not arrows:
        return None
    return arrows[0].merge(arrows[1:]) if len(arrows) > 1 else arrows[0]


def main():
    configure_style()
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    statistics = pd.read_csv(shared.STATS)
    selected = (statistics[statistics.modeled & statistics.directional_enrichment.gt(0)]
                .sort_values(["cluster", "directional_enrichment"], ascending=[True, False])
                .groupby("cluster", as_index=False).head(shared.PAIRS_PER_CLUSTER).copy())
    genes = shared.read_lines(shared.MODEL / "gene_ids.txt")
    gene_index = {gene: index for index, gene in enumerate(genes)}
    coordinates = np.loadtxt(PRE / "spatial_coordinates.csv", delimiter=",").astype(np.float32).astype(float)
    expression = np.loadtxt(shared.MODEL / "raw_expression.csv", delimiter=",", dtype=np.float32)
    assignments = pd.read_csv(DATASET / "analysis/results/receiver_group_assignments.csv")
    clusters = assignments.receiver_group.to_numpy(dtype=int) + 1
    display = coordinates.copy()
    display[:, 2] = (display[:, 2] - display[:, 2].mean()) * shared.Z_DISPLAY_SCALE
    origin, _, edges = shared.grid_definition(display)
    tissue_density = shared.density_array(display, edges, shared.TISSUE_GAUSSIAN_SIGMA)
    tissue_surface = shared.isosurface(tissue_density, origin, shared.TISSUE_ISO_FRACTION)
    camera = shared.camera_for(display)
    arrow_rows = []
    edge_table = load_edges()
    selected[["cluster", "target_gene", "source_gene", "modeled",
              "direction_x", "direction_y", "direction_z",
              "directional_enrichment", "signed_beta", "derived_attention",
              "edge_weight_sum", "receiver_edges"]].to_csv(
                  OUT_DATA / "selected_pairs.csv", index=False)

    for cluster, relations in selected.groupby("cluster", sort=True):
        receiver_indices = np.flatnonzero(clusters == cluster)

        cluster_density = shared.density_array(
            display[receiver_indices], edges, shared.LOCAL_GAUSSIAN_SIGMA)
        cluster_dir = OUT_FIG / f"cluster_{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for relation in relations.itertuples(index=False):
            edge = pair_edges(edge_table, cluster, relation.source_gene, relation.target_gene)
            positions, directions, lengths, coherence, weights, counts = local_vectors(
                coordinates, display, receiver_indices, edge)
            cluster_surface = shared.isosurface(
                cluster_density, origin, shared.CLUSTER_ISO_FRACTION)
            cluster_color = TARGET_ENVELOPE_COLOR
            arrow_color = ARROW_COLOR

            plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE, border=False)
            plotter.set_background("white")
            plotter.add_mesh(tissue_surface, color=shared.TISSUE_COLOR,
                             opacity=shared.TISSUE_OPACITY, smooth_shading=True)
            if cluster_surface is not None:
                plotter.add_mesh(cluster_surface, color=cluster_color,
                                 opacity=CLUSTER_SURFACE_OPACITY, smooth_shading=True,
                                 ambient=.52, diffuse=.48)
            arrows = arrow_mesh(positions, directions, lengths)
            if arrows is not None:
                plotter.add_mesh(arrows, color=arrow_color, opacity=ARROW_OPACITY,
                                 smooth_shading=True, ambient=.55, diffuse=.45)
            plotter.add_axes(line_width=4, color="#303030", labels_off=False)
            plotter.camera_position = camera
            plotter.camera.parallel_projection = True
            plotter.reset_camera()
            plotter.camera.zoom(CAMERA_ZOOM)
            filename = (f"cluster_{cluster}_{safe_name(relation.source_gene)}_to_"
                        f"{safe_name(relation.target_gene)}_local_cci_directions_3d.png")
            output = cluster_dir / filename
            plotter.show(screenshot=str(output), auto_close=True)
            add_legend(output)
            print(output)
            for index in range(len(positions)):
                arrow_rows.append({
                    "cluster": cluster, "source_gene": relation.source_gene,
                    "target_gene": relation.target_gene, "arrow_x": positions[index, 0],
                    "arrow_y": positions[index, 1], "arrow_z": positions[index, 2],
                    "direction_x": directions[index, 0],
                    "direction_y": directions[index, 1],
                    "direction_z": directions[index, 2],
                    "directional_coherence": coherence[index],
                    "display_length": lengths[index], "edge_weight": weights[index],
                    "receiver_cells": counts[index],
                })
    pd.DataFrame(arrow_rows).to_csv(OUT_DATA / "local_cci_direction_vectors.csv", index=False)


if __name__ == "__main__":
    main()
