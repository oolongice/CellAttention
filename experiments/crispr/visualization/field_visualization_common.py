import os
"""Shared V2 reaction-diffusion field calculation and plotting helpers."""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.font_manager as fm
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import k0

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
DATASET = HERE.parent
PROJECT = DATASET.parent
CHECKPOINT = DATASET / "model" / "transformer"
PREPROCESSED = DATASET / "data" / "preprocessed" / "control_train"
FONT_DIR = DATASET / "font"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from vis.spatial import reaction_diffusion_field
from vis.style import configure_style as _configure_style

LENGTH_SCALE_UM = 25.0
MAXIMUM_DISTANCE_UM = 75.0
MINIMUM_DISTANCE_UM = 2.0
DIFFUSION = LENGTH_SCALE_UM ** 2
TISSUE_SUPPORT_DISTANCE_UM = 50.0


def configure_style():
    return _configure_style(FONT_DIR, axes_title_size=8)


def read_source(gene: str):
    genes = [line.strip() for line in (CHECKPOINT / "gene_ids.txt").read_text().splitlines() if line.strip()]
    if gene not in genes:
        matches = [value for value in genes if value.lower() == gene.lower()]
        if len(matches) == 1:
            gene = matches[0]
        else:
            raise ValueError(f"Source gene {gene!r} is absent from the 500-gene checkpoint")
    index = genes.index(gene)
    expression = np.loadtxt(CHECKPOINT / "raw_expression.csv", delimiter=",",
                            dtype=np.float64, usecols=[index])
    coordinates = np.loadtxt(PREPROCESSED / "spatial_coordinates_um.csv",
                             delimiter=",", dtype=np.float64)
    if len(expression) != len(coordinates):
        raise ValueError("Expression and coordinate rows are not aligned")
    return gene, coordinates, expression


def grid_for_bounds(bounds, maximum_axis_points=420, minimum_axis_points=120):
    xmin, xmax, ymin, ymax = map(float, bounds)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError("Coordinate bounds must satisfy min < max")
    aspect = (xmax - xmin) / (ymax - ymin)
    if aspect >= 1:
        nx, ny = maximum_axis_points, max(minimum_axis_points, round(maximum_axis_points / aspect))
    else:
        ny, nx = maximum_axis_points, max(minimum_axis_points, round(maximum_axis_points * aspect))
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    mesh_x, mesh_y = np.meshgrid(x, y)
    return mesh_x, mesh_y, np.column_stack([mesh_x.ravel(), mesh_y.ravel()])


def calculate_field(coordinates, expression, query):
    positive = expression > 0
    if not positive.any():
        raise ValueError("The selected source gene has no positive-expression cells")
    source_xy = coordinates[positive]
    source_activity = expression[positive]
    query_tree = cKDTree(query)
    source_tree = cKDTree(source_xy)
    distances = query_tree.sparse_distance_matrix(
        source_tree, MAXIMUM_DISTANCE_UM, output_type="coo_matrix"
    )
    radius = np.maximum(distances.data, MINIMUM_DISTANCE_UM)
    weights = k0(radius / LENGTH_SCALE_UM) / (2.0 * np.pi * DIFFUSION)
    field = np.bincount(
        distances.row,
        weights=weights * source_activity[distances.col],
        minlength=len(query),
    )
    nearest, _ = cKDTree(coordinates).query(query, k=1, workers=-1)
    field[nearest > TISSUE_SUPPORT_DISTANCE_UM] = np.nan
    return field, positive


def calculate_cell_field(coordinates, expression):
    """Calculate the shared V2 field at cell nodes, excluding self contribution."""
    field, source_indices = reaction_diffusion_field(
        coordinates, expression, query=None, length_scale=LENGTH_SCALE_UM,
        maximum_distance=MAXIMUM_DISTANCE_UM, minimum_distance=MINIMUM_DISTANCE_UM,
        diffusion=DIFFUSION, exclude_self=True,
    )
    positive = np.zeros(len(expression), dtype=bool)
    positive[source_indices] = True
    return field, positive


def clipped_field(field, quantile=0.995):
    finite = field[np.isfinite(field)]
    if not len(finite):
        raise ValueError("No field grid points fall inside tissue support")
    maximum = float(np.quantile(finite, quantile))
    return np.minimum(field, maximum), maximum
