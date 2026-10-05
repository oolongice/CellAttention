#!/usr/bin/env python3
"""Compact 1x2 Slide-seq annotation and model-cluster spatial overview."""

PNG_DPI = 600
FIGURE_WIDTH_IN = 3.0
FIGURE_HEIGHT_IN = 2.0
MINIMUM_FONT_SIZE_PT = 3.0
CCI_CLUSTER_COLORS = ("#B9D8F2", "#F4C3A2", "#D7C4EB", "#B7DFC7", "#F2BFC4", "#DCCFA7")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from slideseq_common import HERE, configure_style, load, palette


DISPLAY_NAMES = {
    "CA1_CA2_CA3_Subiculum": "CA1/CA2/CA3/Subiculum",
    "DentatePyramids": "Dentate pyramids",
    "Endothelial_Stalk": "Endothelial stalk",
    "Endothelial_Tip": "Endothelial tip",
    "Subiculum_Entorhinal_cl2": "Subiculum/Entorhinal cl2",
    "Subiculum_Entorhinal_cl3": "Subiculum/Entorhinal cl3",
}


def spatial_panel(ax, coordinates, labels, colors, title):
    values, counts = np.unique(labels, return_counts=True)
    order = values[np.argsort(-counts)]
    for value in order:
        keep = labels == value
        ax.scatter(coordinates[keep, 0], coordinates[keep, 1], s=.23,
                   color=colors[value], alpha=.72, linewidths=0,
                   rasterized=True)
    xpad = .008 * np.ptp(coordinates[:, 0])
    ypad = .008 * np.ptp(coordinates[:, 1])
    ax.set_xlim(coordinates[:, 0].min() - xpad, coordinates[:, 0].max() + xpad)
    ax.set_ylim(coordinates[:, 1].max() + ypad, coordinates[:, 1].min() - ypad)
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title, fontsize=5.2, fontweight="bold", pad=1.6)


def main():
    configure_style()
    xy, annotations, clusters = load()
    annotations = np.asarray(annotations, dtype=object)
    clusters = np.asarray(clusters, dtype=int)
    annotation_colors = palette(annotations)
    cluster_values = sorted(np.unique(clusters))
    cluster_colors = {value: CCI_CLUSTER_COLORS[index % len(CCI_CLUSTER_COLORS)]
                      for index, value in enumerate(cluster_values)}

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    fig.subplots_adjust(left=.015, right=.985, top=.94, bottom=.43, wspace=.005)
    spatial_panel(axes[0], xy, annotations, annotation_colors, "Cell annotation")
    spatial_panel(axes[1], xy, clusters, cluster_colors, "CCC Modules")
    axes[0].set_anchor("E")
    axes[1].set_anchor("W")

    annotation_values = sorted(np.unique(annotations))
    annotation_handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                          markersize=2.4, markerfacecolor=annotation_colors[a],
                          markeredgewidth=0, label=DISPLAY_NAMES.get(a, a.replace("_", " ")))
                          for a in annotation_values]
    axes[0].legend(
        handles=annotation_handles, title="Cell annotation", ncol=2,
        loc="upper left", bbox_to_anchor=(0, -.025), frameon=False,
        fontsize=3.15, title_fontsize=3.6, handletextpad=.25,
        columnspacing=.62, labelspacing=.20, borderaxespad=0,
    )

    cluster_handles = [mlines.Line2D([], [], marker="o", linestyle="none",
                       markersize=2.5, markerfacecolor=cluster_colors[c],
                       markeredgewidth=0, label=f"CCC Module {c}")
                       for c in cluster_values]
    axes[1].legend(
        handles=cluster_handles, title="CCC Module", ncol=2,
        loc="upper right", bbox_to_anchor=(1, -.025), frameon=False,
        fontsize=3.4, title_fontsize=3.6, handletextpad=.25,
        columnspacing=.72, labelspacing=.24, borderaxespad=0,
    )

    png_output = HERE / "figures" / "slideseq_annotation_clusters_compact.png"
    pdf_output = HERE / "figures/slideseq_annotation_clusters_compact.pdf"
    png_output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, dpi=PNG_DPI, facecolor="white",
                pil_kwargs={"compress_level": 6})
    fig.savefig(pdf_output, dpi=PNG_DPI, facecolor="white")
    plt.close(fig)
    print(f"figure_inches={FIGURE_WIDTH_IN}x{FIGURE_HEIGHT_IN}")
    print(f"minimum_font_size_pt={MINIMUM_FONT_SIZE_PT}")
    print(f"png_output={png_output}")
    print(f"pdf_output={pdf_output}")


if __name__ == "__main__":
    main()
