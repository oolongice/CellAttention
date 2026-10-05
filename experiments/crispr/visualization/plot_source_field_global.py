#!/usr/bin/env python3
import os
"""Visualize the V2 reaction-diffusion field for one source across the training region."""

SOURCE_GENE = "Icam1"
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
DISPLAY_QUANTILE = 0.995
TITLE_FONT_SIZE = 25

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
    xpad, ypad = np.ptp(coordinates[:, 0]) * 0.018, np.ptp(coordinates[:, 1]) * 0.018
    bounds = (coordinates[:, 0].min() - xpad, coordinates[:, 0].max() + xpad,
              coordinates[:, 1].min() - ypad, coordinates[:, 1].max() + ypad)
    mx, my, query = grid_for_bounds(bounds)
    field, positive = calculate_field(coordinates, expression, query)
    shown, maximum = clipped_field(field, DISPLAY_QUANTILE)
    shown = shown.reshape(mx.shape)
    width = 183 / 25.4
    height = width * (bounds[3] - bounds[2]) / (bounds[1] - bounds[0])
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.94)
    image = ax.pcolormesh(mx, my, shown, cmap="Blues", vmin=0, vmax=maximum,
                          shading="auto", alpha=0.90, rasterized=True, zorder=0)
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=0.22, c="#8F969D",
               alpha=0.25, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(coordinates[positive, 0], coordinates[positive, 1], s=1.2,
               c="#2171B5", alpha=0.60, linewidths=0, rasterized=True, zorder=2)
    ax.set_xlim(bounds[:2]); ax.set_ylim(bounds[2:]); ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box"); ax.set_axis_off()
    ax.set_title(f"{gene} reaction–diffusion field", loc="left", fontweight="bold",
                 fontsize=TITLE_FONT_SIZE, pad=4)
    # Colorbar intentionally hidden for the global source-field figure.
    # bar = fig.colorbar(image, ax=ax, fraction=0.028, pad=0.015)
    # bar.set_label(f"{gene} field (q{DISPLAY_QUANTILE:.3f} clipped)")
    out = HERE / "figures" / f"{gene}_source_field_global.{fmt}"
    out.parent.mkdir(parents=True, exist_ok=True)
    opts = {"facecolor": "white"}
    if fmt == "png": opts.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    fig.savefig(out, **opts); plt.close(fig)
    print(f"source_gene={gene}"); print(f"positive_sender_cells={int(positive.sum())}")
    print(f"length_scale_um={LENGTH_SCALE_UM}"); print(f"maximum_distance_um={MAXIMUM_DISTANCE_UM}")
    print(f"output={out}")


if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr); raise
