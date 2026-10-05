#!/usr/bin/env python3
import os
"""Mark the Cxcl9 and Cxcl10 dose-response windows on the global Icam1 field."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from field_visualization_common import (
    PREPROCESSED, calculate_cell_field, calculate_field, clipped_field,
    configure_style, grid_for_bounds, read_source,
)
from plot_source_target_field_dose_response import (
    IMMUNE, LOCAL_HALF_WIDTH_UM, hotspot, read_target,
)

SOURCE_GENE = "Icam1"
TARGET_GENES = ("Cxcl9", "Cxcl10")
REGION_COLORS = {"Cxcl9": "#D55E00", "Cxcl10": "#CC79A7"}
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
DISPLAY_QUANTILE = 0.995
TITLE_FONT_SIZE = 22

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"


def selected_regions(coordinates, source_field, immune):
    rows = []
    for requested_target in TARGET_GENES:
        target, target_expression = read_target(requested_target)
        center = hotspot(coordinates, source_field, target_expression, immune)
        rows.append({
            "source_gene": SOURCE_GENE,
            "target_gene": target,
            "center_x_um": float(center[0]),
            "center_y_um": float(center[1]),
            "xmin_um": float(center[0] - LOCAL_HALF_WIDTH_UM),
            "xmax_um": float(center[0] + LOCAL_HALF_WIDTH_UM),
            "ymin_um": float(center[1] - LOCAL_HALF_WIDTH_UM),
            "ymax_um": float(center[1] + LOCAL_HALF_WIDTH_UM),
        })
    return pd.DataFrame(rows)


def main():
    fmt = OUTPUT_FORMAT.lower()
    if fmt not in {"png", "pdf"}:
        raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_style()
    gene, coordinates, expression = read_source(SOURCE_GENE)
    annotations = np.asarray([
        value.strip() for value in (PREPROCESSED / "annotation.txt").read_text().splitlines()
        if value.strip()
    ])
    immune = np.isin(annotations, list(IMMUNE))
    node_field, _ = calculate_cell_field(coordinates, expression)
    regions = selected_regions(coordinates, node_field, immune)

    xpad = np.ptp(coordinates[:, 0]) * 0.018
    ypad = np.ptp(coordinates[:, 1]) * 0.018
    bounds = (
        coordinates[:, 0].min() - xpad, coordinates[:, 0].max() + xpad,
        coordinates[:, 1].min() - ypad, coordinates[:, 1].max() + ypad,
    )
    mesh_x, mesh_y, query = grid_for_bounds(bounds)
    field, positive = calculate_field(coordinates, expression, query)
    shown, maximum = clipped_field(field, DISPLAY_QUANTILE)
    shown = shown.reshape(mesh_x.shape)

    width = 183 / 25.4
    height = width * (bounds[3] - bounds[2]) / (bounds[1] - bounds[0])
    figure, axis = plt.subplots(figsize=(width, height))
    figure.subplots_adjust(left=.015, right=.985, bottom=.02, top=.94)
    axis.pcolormesh(
        mesh_x, mesh_y, shown, cmap="Blues", vmin=0, vmax=maximum,
        shading="auto", alpha=.90, rasterized=True, zorder=0,
    )
    axis.scatter(
        coordinates[:, 0], coordinates[:, 1], s=.22, c="#8F969D",
        alpha=.25, linewidths=0, rasterized=True, zorder=1,
    )
    axis.scatter(
        coordinates[positive, 0], coordinates[positive, 1], s=1.2,
        c="#2171B5", alpha=.60, linewidths=0, rasterized=True, zorder=2,
    )

    for region in regions.itertuples(index=False):
        color = REGION_COLORS[region.target_gene]
        axis.add_patch(Rectangle(
            (region.xmin_um, region.ymin_um),
            region.xmax_um - region.xmin_um,
            region.ymax_um - region.ymin_um,
            fill=False, edgecolor=color, linewidth=1.35, zorder=5,
        ))
        axis.text(
            region.xmin_um, region.ymin_um - 18, f"{region.target_gene} region",
            ha="left", va="bottom", color=color, fontsize=7, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 1.2},
            zorder=6,
        )

    axis.set_xlim(bounds[:2])
    axis.set_ylim(bounds[2:])
    axis.invert_yaxis()
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    axis.set_title(
        f"{gene} field: selected Cxcl9 and Cxcl10 regions",
        loc="left", fontweight="bold", fontsize=TITLE_FONT_SIZE, pad=4,
    )

    data_output = HERE / "data/Icam1_selected_dose_response_regions.csv"
    data_output.parent.mkdir(parents=True, exist_ok=True)
    regions.to_csv(data_output, index=False)
    output = HERE / "figures" / f"Icam1_source_field_global_selected_regions.{fmt}"
    options = {"facecolor": "white"}
    if fmt == "png":
        options.update({"dpi": PNG_DPI, "pil_kwargs": {"compress_level": 6}})
    figure.savefig(output, **options)
    plt.close(figure)
    print(regions.to_string(index=False))
    print(f"output_data={data_output}")
    print(f"output={output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
