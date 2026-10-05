#!/usr/bin/env python3
"""Shared 3D rendering helpers; selected pairs and formal attribution edges are consumed by the individual and vector renderers."""
import json

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from merfish_common import (DATASET, FONT_BOLD, FIGURES, HERE, PRE, configure_style)

MODEL = DATASET / "model/transformer"
STATS = HERE / "data/directional_arrow_heatmaps/directional_arrow_statistics.csv"
OUT_FIG = FIGURES / "directional_pairs_3d"
OUT_DATA = HERE / "data/directional_pairs_3d"

# ---- Selection and model-consistent physical-field parameters ----
PAIRS_PER_CLUSTER = 3
LENGTH_SCALE = 50.
MAX_DISTANCE = 150.
MIN_DISTANCE = 2.

# ---- Density-isosurface reconstruction parameters (safe to adjust) ----
VOXEL_SIZE = 14.
GRID_PADDING = 35.
TISSUE_GAUSSIAN_SIGMA = 2.2
LOCAL_GAUSSIAN_SIGMA = 1.65
TISSUE_ISO_FRACTION = .075
CLUSTER_ISO_FRACTION = .16
SOURCE_ISO_FRACTION = .18
SURFACE_SMOOTH_ITERATIONS = 35
SURFACE_SMOOTH_PASS_BAND = .08

# ---- 3D display parameters (safe to adjust) ----
WINDOW_SIZE = (5400, 1700)
Z_DISPLAY_SCALE = 4.
CAMERA_DIRECTION = (.90, -1.35, .72)
CAMERA_DISTANCE_SCALE = 1.45
CAMERA_ZOOM = 1.18
TISSUE_COLOR = "#B9C0C8"
TISSUE_OPACITY = .12
CLUSTER_OPACITY = .92
SOURCE_COLOR = "#F0C75E"
SOURCE_OPACITY = .72
SURFACE_SPECULAR = .10
TITLE_FONT_SIZE = 15
SCALAR_BAR_TITLE_FONT_SIZE = 22
SCALAR_BAR_LABEL_FONT_SIZE = 18
INFLUENCE_QUANTILE = .99


def read_lines(path):
    return path.read_text().splitlines()


def yukawa(distance):
    distance = np.maximum(np.asarray(distance, dtype=float), MIN_DISTANCE)
    return np.exp(-distance / LENGTH_SCALE) / (
        4 * np.pi * LENGTH_SCALE ** 2 * distance)


def grid_definition(points):
    lower = points.min(axis=0) - GRID_PADDING
    upper = points.max(axis=0) + GRID_PADDING
    dimensions = np.ceil((upper - lower) / VOXEL_SIZE).astype(int) + 1
    upper = lower + (dimensions - 1) * VOXEL_SIZE
    edges = [np.linspace(lower[axis] - VOXEL_SIZE / 2,
                         upper[axis] + VOXEL_SIZE / 2,
                         dimensions[axis] + 1) for axis in range(3)]
    return lower, dimensions, edges


def density_array(points, edges, sigma, weights=None):
    values, _ = np.histogramdd(points, bins=edges, weights=weights)
    return gaussian_filter(values.astype(np.float32), sigma=sigma, mode="constant")


def image_grid(values, origin, scalar_name, extra_scalars=None):
    grid = pv.ImageData(dimensions=values.shape, spacing=(VOXEL_SIZE,) * 3,
                        origin=tuple(origin))
    grid.point_data[scalar_name] = values.ravel(order="F")
    for name, scalar in (extra_scalars or {}).items():
        grid.point_data[name] = scalar.ravel(order="F")
    return grid


def isosurface(density, origin, fraction, extra_scalars=None):
    maximum = float(density.max())
    if maximum <= 0:
        return None
    grid = image_grid(density, origin, "density", extra_scalars)
    surface = grid.contour([fraction * maximum], scalars="density")
    if not surface.n_cells:
        return None
    return surface.smooth_taubin(n_iter=SURFACE_SMOOTH_ITERATIONS,
                                 pass_band=SURFACE_SMOOTH_PASS_BAND).clean()


def camera_for(points):
    center = points.mean(axis=0)
    distance = CAMERA_DISTANCE_SCALE * np.linalg.norm(np.ptp(points, axis=0))
    direction = np.asarray(CAMERA_DIRECTION, dtype=float)
    direction /= np.linalg.norm(direction)
    return [tuple(center + direction * distance), tuple(center), (0., 0., 1.)]
