#!/usr/bin/env python3
"""Render MERFISH annotations and CellAttention clusters as publication-style 3D cells."""
import json
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import pyvista as pv
from PIL import Image, ImageChops, ImageDraw, ImageFont

from merfish_common import (HERE as VIS, FIGURES, PRE, PNG_DPI, FONT_BOLD, FONT_REGULAR,
                            configure_style, ordered_cluster_palette,
                            ordered_type_palette)

FIG = FIGURES / "cells_3d"
DATA = VIS / "data" / "cells_3d"
WINDOW_SIZE = (2700, 1500)
POINT_SIZE = 10.0
Z_DISPLAY_SCALE = 4.0
WHITE_MIX = .15


def softened_rgb(color):
    rgb = np.asarray(mcolors.to_rgb(color))
    return rgb * (1 - WHITE_MIX) + WHITE_MIX


def rgb_u8(labels, order, colors):
    palette = {label: softened_rgb(color) for label, color in zip(order, colors)}
    missing = sorted(set(labels) - set(palette))
    if missing:
        raise ValueError(f"Labels missing from fixed palette: {missing}")
    return (np.asarray([palette[label] for label in labels]) * 255).astype(np.uint8)


def make_cloud(xyz, labels, order, colors):
    cloud = pv.PolyData(xyz)
    cloud["rgb"] = rgb_u8(labels, order, colors)
    return cloud


def camera_for(points):
    center = points.mean(axis=0)
    distance = 1.45 * np.linalg.norm(np.ptp(points, axis=0))
    direction = np.asarray((.90, -1.35, .72), dtype=float)
    direction /= np.linalg.norm(direction)
    return [tuple(center + direction * distance), tuple(center), (0., 0., 1.)]


def add_panel(plotter, cloud, title, camera):
    plotter.add_mesh(
        cloud, scalars="rgb", rgb=True, style="points", point_size=POINT_SIZE,
        render_points_as_spheres=True, ambient=.88, diffuse=.48,
        specular=.08, specular_power=12,
    )
    plotter.add_text(title, position="upper_left", font_size=50,
                     font_file=str(FONT_BOLD), color="#202020", shadow=False)
    plotter.camera_position = camera
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(1.25)


def add_arial_legend(filename, order, colors):
    image = Image.open(filename).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.truetype(str(FONT_REGULAR), 70)
    line_height = 70
    radius = 20
    left = 54
    bottom = image.height - 600
    text_width = max(draw.textbbox((0, 0), label, font=font)[2] for label in order)
    box_width = text_width + 86
    box_height = line_height * len(order) + 28
    top = bottom - box_height
    draw.rounded_rectangle((left - 20, top, left + box_width, bottom), radius=8,
                           fill=(255, 255, 255, 224))
    for index, (label, color) in enumerate(zip(order, colors)):
        y = top + 19 + index * line_height + line_height // 2
        rgb = tuple(np.round(softened_rgb(color) * 255).astype(int))
        draw.ellipse((left, y - radius, left + 2 * radius, y + radius), fill=rgb + (255,))
        draw.text((left + 55, y), label, font=font, fill=(32, 32, 32, 255),
                  anchor="lm")
    image.save(filename, dpi=(PNG_DPI, PNG_DPI), compress_level=6)


def render_single(cloud, title, order, colors, camera, filename):
    plotter = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    plotter.set_background("white")
    add_panel(plotter, cloud, title, camera)
    plotter.enable_eye_dome_lighting()
    plotter.show(screenshot=str(filename), auto_close=True)
    add_arial_legend(filename, order, colors)


def compose_combined(left_path, right_path, output_path):
    panels = []
    for path in (left_path, right_path):
        image = Image.open(path).convert("RGB")
        difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
        bbox = difference.getbbox() or (0, 0, *image.size)
        pad = 36
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(image.width, bbox[2] + pad), min(image.height, bbox[3] + pad))
        panels.append(image.crop(bbox))
    canvas = Image.new("RGB", (3200, 1500), "white")
    for index, panel in enumerate(panels):
        panel.thumbnail((1540, 1460), Image.Resampling.LANCZOS)
        x = index * 1600 + (1600 - panel.width) // 2
        y = (1500 - panel.height) // 2
        canvas.paste(panel, (x, y))
    canvas.save(output_path, dpi=(PNG_DPI, PNG_DPI), compress_level=6)


def main():
    configure_style()
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(PRE / "cell_metadata.parquet")
    receiver_groups = pd.read_parquet(VIS / "data/cell_receiver_groups.parquet")
    xyz = np.loadtxt(PRE / "spatial_coordinates.csv", delimiter=",")
    if not (len(metadata) == len(receiver_groups) == len(xyz)):
        raise ValueError("Coordinates, annotations, and CCC Modules have different row counts")
    if not np.array_equal(metadata["cell_id"].astype(str), receiver_groups["cell_id"].astype(str)):
        raise ValueError("Cell order differs between metadata and CCC Module table")

    display_xyz = xyz.copy()
    display_xyz[:, 2] = (display_xyz[:, 2] - display_xyz[:, 2].mean()) * Z_DISPLAY_SCALE
    annotation = metadata["type"].astype(str).to_numpy()
    annotation_order, annotation_colors = ordered_type_palette(annotation)
    cluster_order, cluster_colors = ordered_cluster_palette(receiver_groups["cluster"])
    clusters = receiver_groups["cluster"].map(lambda value: f"Cluster {int(value)}").to_numpy()
    annotation_cloud = make_cloud(display_xyz, annotation, annotation_order, annotation_colors)
    cluster_cloud = make_cloud(display_xyz, clusters, cluster_order, cluster_colors)
    camera = camera_for(display_xyz)

    annotation_path = FIG / "cell_annotation_3d.png"
    cluster_path = FIG / "model_clusters_3d.png"
    render_single(annotation_cloud, "Cell annotation", annotation_order,
                  annotation_colors, camera, annotation_path)
    render_single(cluster_cloud, "CCC Modules", cluster_order,
                  cluster_colors, camera, cluster_path)
    compose_combined(annotation_path, cluster_path, FIG / "annotation_and_clusters_3d.png")
    (DATA / "render_metadata.json").write_text(json.dumps({
        "coordinate_source": "data/preprocessed/spatial_coordinates.csv",
        "annotation_field": "type",
        "cluster_source": "visualization/data/cell_receiver_groups.parquet:cluster",
        "number_of_cells": int(len(xyz)),
        "z_display_scale": Z_DISPLAY_SCALE,
        "point_size_pixels": POINT_SIZE,
        "white_mix": WHITE_MIX,
        "projection": "parallel",
        "camera_position": camera,
        "visualization_only": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
