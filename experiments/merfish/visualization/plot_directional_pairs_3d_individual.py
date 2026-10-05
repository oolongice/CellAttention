#!/usr/bin/env python3
"""Render each selected directional source-target relation as one compact 3D figure."""
from pathlib import Path

import numpy as np
from matplotlib import colormaps
import pandas as pd
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import plot_directional_pairs_3d as shared
from directional_edges import load_edges, pair_edges
from merfish_common import DATASET, FONT_BOLD, FONT_REGULAR, FIGURES, HERE, PNG_DPI, PRE, configure_style

OUT_FIG = FIGURES / "directional_pairs_3d"
OUT_DATA = HERE / "data/directional_pairs_3d"

# ---- Compact single-panel display parameters (safe to adjust) ----
WINDOW_SIZE = (1900, 1600)
CAMERA_ZOOM = 1.34
TITLE_FONT_SIZE = 38
EXPRESSION_BAR_TITLE_FONT_SIZE = 56
EXPRESSION_BAR_LABEL_FONT_SIZE = 30
LEGEND_FONT_SIZE = 56
LEGEND_LINE_HEIGHT = 78
LEGEND_MARKER_SIZE = 34
LEGEND_LEFT = 64
LEGEND_TOP = 270
LEGEND_BACKGROUND_ALPHA = 225
EXPRESSION_BAR_WIDTH = 650
EXPRESSION_BAR_HEIGHT = 34
EXPRESSION_BAR_GAP_BELOW_LEGEND = 92
TARGET_EXPRESSION_CMAP = "Blues"
TARGET_EXPRESSION_QUANTILE = .99


def safe_name(value):
    return str(value).replace("/", "_").replace(":", "_").replace(" ", "_")


def add_image_legend(path):
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    legend_font = ImageFont.truetype(str(FONT_REGULAR), LEGEND_FONT_SIZE)
    bar_title_font = ImageFont.truetype(str(FONT_BOLD), EXPRESSION_BAR_TITLE_FONT_SIZE)
    bar_label_font = ImageFont.truetype(str(FONT_REGULAR), EXPRESSION_BAR_LABEL_FONT_SIZE)

    entries = [
        (shared.TISSUE_COLOR, "Whole-tissue envelope"),
        (shared.SOURCE_COLOR, "Supported sender domain"),
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

    bar_left = left
    bar_top = top + box_height + EXPRESSION_BAR_GAP_BELOW_LEGEND
    cmap = colormaps[TARGET_EXPRESSION_CMAP]
    for offset in range(EXPRESSION_BAR_WIDTH):
        rgba = cmap(offset / max(EXPRESSION_BAR_WIDTH - 1, 1))
        color = tuple(int(round(channel * 255)) for channel in rgba[:3]) + (255,)
        draw.line((bar_left + offset, bar_top, bar_left + offset,
                   bar_top + EXPRESSION_BAR_HEIGHT), fill=color)
    draw.rectangle((bar_left, bar_top, bar_left + EXPRESSION_BAR_WIDTH,
                    bar_top + EXPRESSION_BAR_HEIGHT), outline=(90, 90, 90, 255), width=1)
    draw.text((bar_left + EXPRESSION_BAR_WIDTH / 2, bar_top - 10),
              "Target expression", font=bar_title_font, fill=(32, 32, 32, 255),
              anchor="ms")
    draw.text((bar_left, bar_top + EXPRESSION_BAR_HEIGHT + 7),
              "Low expression", font=bar_label_font, fill=(32, 32, 32, 255),
              anchor="la")
    draw.text((bar_left + EXPRESSION_BAR_WIDTH, bar_top + EXPRESSION_BAR_HEIGHT + 7),
              "High expression", font=bar_label_font, fill=(32, 32, 32, 255),
              anchor="ra")
    image.save(path, dpi=(PNG_DPI, PNG_DPI), compress_level=6)


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
    coordinates = np.loadtxt(PRE / "spatial_coordinates.csv", delimiter=",")
    expression = np.loadtxt(shared.MODEL / "raw_expression.csv", delimiter=",", dtype=np.float32)
    assignments = pd.read_csv(DATASET / "analysis/results/receiver_group_assignments.csv")
    clusters = assignments.receiver_group.to_numpy(dtype=int) + 1
    display = coordinates.copy()
    display[:, 2] = (display[:, 2] - display[:, 2].mean()) * shared.Z_DISPLAY_SCALE
    origin, _, edges = shared.grid_definition(display)
    tissue_density = shared.density_array(display, edges, shared.TISSUE_GAUSSIAN_SIGMA)
    tissue_surface = shared.isosurface(
        tissue_density, origin, shared.TISSUE_ISO_FRACTION)
    camera = shared.camera_for(display)
    metadata_rows = []
    edge_table = load_edges()

    for cluster, relations in selected.groupby("cluster", sort=True):
        receiver_indices = np.flatnonzero(clusters == cluster)

        cluster_density = shared.density_array(
            display[receiver_indices], edges, shared.LOCAL_GAUSSIAN_SIGMA)
        cluster_dir = OUT_FIG / f"cluster_{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for relation in relations.itertuples(index=False):
            source_values = expression[:, gene_index[relation.source_gene]]
            edge = pair_edges(edge_table, cluster, relation.source_gene, relation.target_gene)
            sources = edge.sender_index.to_numpy(dtype=int)
            contributing_sources = np.unique(sources)
            target_index = gene_index[relation.target_gene]
            target_values = expression[receiver_indices, target_index]

            target_sum = shared.density_array(
                display[receiver_indices], edges, shared.LOCAL_GAUSSIAN_SIGMA,
                weights=target_values)
            target_field = np.divide(
                target_sum, cluster_density, out=np.zeros_like(target_sum),
                where=cluster_density > 1e-8)
            cluster_surface = shared.isosurface(
                cluster_density, origin, shared.CLUSTER_ISO_FRACTION,
                extra_scalars={"target_expression": target_field})
            source_density = shared.density_array(
                display[contributing_sources], edges, shared.LOCAL_GAUSSIAN_SIGMA,
                weights=source_values[contributing_sources])
            source_surface = shared.isosurface(
                source_density, origin, shared.SOURCE_ISO_FRACTION)
            limit = float(np.quantile(target_values, TARGET_EXPRESSION_QUANTILE))
            limit = max(limit, 1e-12)

            plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE, border=False)
            plotter.set_background("white")
            plotter.add_mesh(tissue_surface, color=shared.TISSUE_COLOR,
                             opacity=shared.TISSUE_OPACITY, smooth_shading=True,
                             ambient=.55, diffuse=.45, specular=shared.SURFACE_SPECULAR)
            if source_surface is not None:
                plotter.add_mesh(source_surface, color=shared.SOURCE_COLOR,
                                 opacity=shared.SOURCE_OPACITY, smooth_shading=True,
                                 ambient=.52, diffuse=.48, specular=shared.SURFACE_SPECULAR)
            if cluster_surface is not None:
                plotter.add_mesh(
                    cluster_surface, scalars="target_expression", cmap=TARGET_EXPRESSION_CMAP,
                    clim=(0, limit), opacity=shared.CLUSTER_OPACITY,
                    smooth_shading=True, ambient=.50, diffuse=.50,
                    specular=shared.SURFACE_SPECULAR, show_scalar_bar=False)
            title = (f"Cluster {cluster}: {relation.source_gene} → {relation.target_gene}\n"
                     f"Directional enrichment R={relation.directional_enrichment:.3f}; "
                     f"β={relation.signed_beta:.3f}")
            plotter.add_text(title, position="upper_left", font_size=TITLE_FONT_SIZE,
                             font_file=str(FONT_BOLD), color="#202020")
            plotter.add_axes(line_width=4, color="#303030", labels_off=False)
            plotter.camera_position = camera
            plotter.camera.parallel_projection = True
            plotter.reset_camera()
            plotter.camera.zoom(CAMERA_ZOOM)
            filename = (f"cluster_{cluster}_{safe_name(relation.source_gene)}_to_"
                        f"{safe_name(relation.target_gene)}_3d.png")
            output = cluster_dir / filename
            plotter.show(screenshot=str(output), auto_close=True)
            add_image_legend(output)
            print(output)
            metadata_rows.append({
                "cluster": cluster, "source_gene": relation.source_gene,
                "target_gene": relation.target_gene,
                "directional_enrichment": relation.directional_enrichment,
                "signed_beta": relation.signed_beta,
                "output": str(output.relative_to(HERE)),
                "retained_edges": len(edge),
                "supported_senders": len(contributing_sources),
            })

    pd.DataFrame(metadata_rows).to_csv(
        OUT_DATA / "individual_directional_pair_figures.csv", index=False)


if __name__ == "__main__":
    main()
