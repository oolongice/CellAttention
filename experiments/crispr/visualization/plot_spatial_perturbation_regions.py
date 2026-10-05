#!/usr/bin/env python3
import os
"""Plot all CRISPR cells with translucent perturbation-region envelopes."""

OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
GRID_LONG_AXIS = 720
ENVELOPE_ALPHA = 0.20
MIN_COMPONENT_BINS = 20
SMOOTHING_SIGMA = 5.0
ADJACENCY_DILATION_BINS = 5

from pathlib import Path
from font_helper import font_family
import sys
import anndata as ad
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy import ndimage

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
DATASET = HERE.parent
PROJECT = DATASET.parent
INPUT = DATASET / "data/preprocessed/SpacSeq_lung_cancer_label_integratedPanel_500genes.h5ad"
FONT = DATASET / "font"
OUT = HERE / "figures"
CONTROL = "control_training_shapes"
EXCLUDE = {"control_nontargeting_shapes"}
CONTROL_COLOR = "#D94B4B"


def configure_style():
    family = font_family(FONT)
    mpl.rcParams.update({"font.family": family, "font.sans-serif": [family],
                         "font.size": 7, "legend.fontsize": 6,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


PASTEL_COLORS = [
    "#6B9BB8",  # blue
    "#73A887",  # green
    "#9182B3",  # purple
    "#B89B63",  # ochre
    "#63A8A3",  # teal
    "#7D91B3",  # slate blue
    "#9EAB63",  # olive
    "#B582A6",  # mauve
    "#72A6C7",  # sky blue
    "#89B77D",  # leaf green
    "#A38DBD",  # lavender
    "#C0A66B",  # sand
]

def assign_adjacent_colors(names, masks):
    adjacency = {name: set() for name in names}
    expanded = {name: ndimage.binary_dilation(masks[name], iterations=ADJACENCY_DILATION_BINS)
                for name in names}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if np.any(expanded[left] & expanded[right]):
                adjacency[left].add(right)
                adjacency[right].add(left)
    assigned = {}
    unassigned = set(names)
    while unassigned:
        node = max(unassigned, key=lambda name: (
            len({assigned[n] for n in adjacency[name] if n in assigned}),
            len(adjacency[name]), name,
        ))
        forbidden = {assigned[n] for n in adjacency[node] if n in assigned}
        available = [index for index in range(len(PASTEL_COLORS)) if index not in forbidden]
        if not available:
            raise RuntimeError("The perturbation adjacency graph needs more colors")
        usage = {index: sum(value == index for value in assigned.values()) for index in available}
        # Balance usage across the full palette instead of repeatedly choosing
        # the first two colors in a sparse adjacency graph.
        assigned[node] = min(available, key=lambda index: (usage[index], index))
        unassigned.remove(node)
    conflicts = sum(assigned[a] == assigned[b] for a in names for b in adjacency[a]) // 2
    if conflicts:
        raise RuntimeError(f"Adjacent-color conflicts remain: {conflicts}")
    return assigned, adjacency


def remove_small_components(binary):
    labels, count = ndimage.label(binary)
    if count == 0:
        return binary
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_COMPONENT_BINS
    keep[0] = False
    return keep[labels]


def density_envelope(coords, selected, x_edges, y_edges):
    density, _, _ = np.histogram2d(
        coords[selected, 0], coords[selected, 1], bins=(x_edges, y_edges)
    )
    density = ndimage.gaussian_filter(density.T, sigma=SMOOTHING_SIGMA)
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    x_bins = np.clip(np.searchsorted(x_edges, coords[selected, 0], side="right") - 1, 0, nx - 1)
    y_bins = np.clip(np.searchsorted(y_edges, coords[selected, 1], side="right") - 1, 0, ny - 1)
    occupied_density = density[y_bins, x_bins]
    threshold = max(float(np.quantile(occupied_density, 0.12)), float(density.max()) * 0.055)
    mask = remove_small_components(density >= threshold)
    # Excluding small components also excludes their density from the final contour.
    density = np.where(mask, density, 0.0)
    return density, threshold, mask


def main():
    fmt = OUTPUT_FORMAT.lower()
    if fmt not in {"png", "pdf"}:
        raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_style()
    a = ad.read_h5ad(INPUT, backed="r")
    coords = np.asarray(a.obsm["spatial_aligned"], dtype=float)
    columns = [c for c in a.obs if c.endswith("_shapes") and c != CONTROL and c not in EXCLUDE]
    masks = {c: a.obs[c].astype(str).eq("True").to_numpy() for c in columns}
    control = a.obs[CONTROL].astype(str).eq("True").to_numpy()
    xs, ys = np.ptp(coords[:, 0]), np.ptp(coords[:, 1])
    x_limits = (coords[:, 0].min() - xs * 0.018, coords[:, 0].max() + xs * 0.018)
    y_limits = (coords[:, 1].min() - ys * 0.018, coords[:, 1].max() + ys * 0.018)
    nx = GRID_LONG_AXIS
    ny = max(420, round(nx * (y_limits[1] - y_limits[0]) / (x_limits[1] - x_limits[0])))
    xe = np.linspace(*x_limits, nx + 1)
    ye = np.linspace(*y_limits, ny + 1)
    xc, yc = (xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2

    plot_width = 183 / 25.4
    map_height = plot_width * (y_limits[1] - y_limits[0]) / (x_limits[1] - x_limits[0])
    fig, ax = plt.subplots(figsize=(plot_width, map_height + 1.05))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=1.05 / (map_height + 1.05), top=1.0)
    ax.scatter(coords[:, 0], coords[:, 1], s=0.38, c="#8F969D", alpha=0.42,
               linewidths=0, rasterized=True, zorder=1)
    regions = {column: density_envelope(coords, masks[column], xe, ye) for column in columns}
    region_masks = {column: regions[column][2] for column in columns}
    color_ids, adjacency = assign_adjacent_colors(columns, region_masks)
    handles = []
    for column in columns:
        color = PASTEL_COLORS[color_ids[column]]
        density, threshold, region_mask = regions[column]
        if region_mask.any() and threshold < density.max():
            ax.contourf(xc, yc, density, levels=[threshold, density.max() + np.finfo(float).eps],
                        colors=[color], alpha=0.19, antialiased=True, zorder=2)
            ax.contour(xc, yc, density, levels=[threshold], colors=[color],
                       linewidths=1.05, alpha=0.86, antialiased=True, zorder=3)
        handles.append(Patch(facecolor=mpl.colors.to_rgba(color, 0.30), edgecolor=color,
                             linewidth=0.75, label=column.removesuffix("_shapes")))
    control_density, control_threshold, control_mask = density_envelope(coords, control, xe, ye)
    ax.contourf(xc, yc, control_density,
                levels=[control_threshold, control_density.max() + np.finfo(float).eps],
                colors=[CONTROL_COLOR], alpha=0.20, antialiased=True, zorder=4)
    ax.contour(xc, yc, control_density, levels=[control_threshold], colors=[CONTROL_COLOR],
               linewidths=1.65, alpha=0.96, antialiased=True, zorder=5)
    control_patch = Patch(facecolor=mpl.colors.to_rgba(CONTROL_COLOR, 0.30),
                          edgecolor=CONTROL_COLOR, linewidth=0.8, label="Training control")
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.legend(handles=[control_patch, *handles], loc="lower center", bbox_to_anchor=(0.5, 0.012),
               ncol=6, frameon=False, handlelength=1.5, handleheight=0.8,
               columnspacing=1.0, handletextpad=0.4, labelspacing=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"crispr_spatial_perturbation_regions.{fmt}"
    options = {"facecolor": "white"}
    if fmt == "png":
        options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(output, **options)
    plt.close(fig)
    non_target = a.obs["control_nontargeting_shapes"].astype(str).eq("True").sum()
    print(f"cells={len(coords)}")
    print(f"perturbation_regions={len(columns)}")
    print(f"training_control_cells={int(control.sum())}")
    print(f"adjacency_edges={sum(map(len, adjacency.values())) // 2}")
    print("adjacent_color_conflicts=0")
    print(f"perturbation_colors_used={len(set(color_ids.values()))}")
    print(f"excluded_nontargeting_control_cells={int(non_target)}")
    print(f"output={output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
