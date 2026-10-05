#!/usr/bin/env python3
import os
"""Visualize a V2 source field inside a user-specified spatial window."""

SOURCE_GENE = "Icam1"
X_RANGE = (5250.0, 5380.0)
Y_RANGE = (3030.0, 3160.0)
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
DISPLAY_QUANTILE = 0.995

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

from field_visualization_common import (
    LENGTH_SCALE_UM, MAXIMUM_DISTANCE_UM, calculate_field, clipped_field,
    configure_style, grid_for_bounds, read_source,
)

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"


def main():
    fmt = OUTPUT_FORMAT.lower()
    if fmt not in {"png", "pdf"}:
        raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_style()
    gene, coordinates, expression = read_source(SOURCE_GENE)
    bounds = (*map(float, X_RANGE), *map(float, Y_RANGE))
    mx, my, query = grid_for_bounds(bounds, maximum_axis_points=420, minimum_axis_points=240)
    field, positive = calculate_field(coordinates, expression, query)
    shown, maximum = clipped_field(field, DISPLAY_QUANTILE)
    shown = shown.reshape(mx.shape)
    local = (coordinates[:, 0] >= bounds[0]) & (coordinates[:, 0] <= bounds[1]) & \
            (coordinates[:, 1] >= bounds[2]) & (coordinates[:, 1] <= bounds[3])
    local_source = local & positive
    fig, ax = plt.subplots(figsize=(104 / 25.4, 88 / 25.4))
    fig.subplots_adjust(left=0.19, right=0.76, bottom=0.17, top=0.89)
    image = ax.pcolormesh(mx, my, shown, cmap="Blues", vmin=0, vmax=maximum,
                          shading="auto", alpha=0.90, rasterized=True, zorder=0)
    ax.scatter(coordinates[local, 0], coordinates[local, 1], s=5, c="#BDBDBD",
               alpha=0.42, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(coordinates[local_source, 0], coordinates[local_source, 1], s=16,
               c="#2171B5", edgecolors="#08306B", linewidths=0.35,
               alpha=0.90, rasterized=True, zorder=2)
    ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[3], bounds[2]); ax.set_aspect("equal")
    ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
    ax.set_title(f"Local {gene} reaction–diffusion field", loc="left", fontweight="bold", pad=4)
    for spine in ax.spines.values(): spine.set_linewidth(0.6)
    ax.tick_params(length=2.5, width=0.6, pad=2)
    color_ax = fig.add_axes([0.81, 0.19, 0.030, 0.62])
    bar = fig.colorbar(image, cax=color_ax)
    bar.set_label(f"{gene} field", labelpad=3)
    color_ax.tick_params(pad=2)
    out = HERE / "figures" / f"{gene}_source_field_local.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    opts = {"facecolor": "white"}
    if fmt == "png": opts.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(out, **opts); plt.close(fig)
    print(f"source_gene={gene}"); print(f"x_range={X_RANGE}"); print(f"y_range={Y_RANGE}")
    print(f"local_cells={int(local.sum())}"); print(f"local_sender_cells={int(local_source.sum())}")
    print(f"length_scale_um={LENGTH_SCALE_UM}"); print(f"maximum_distance_um={MAXIMUM_DISTANCE_UM}")
    print(f"output={out}")


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr); raise
