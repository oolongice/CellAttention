#!/usr/bin/env python3
"""Shared paths, palettes, and publication style for MERFISH figures."""
from pathlib import Path
import os
import sys

CODE = Path(__file__).resolve().parent.parent
DATASET = Path(os.environ["CELLATTENTION_MERFISH_WORKDIR"])
HERE = DATASET / "visualization"
PROJECT = CODE
PRE = DATASET / "data/preprocessed"
RESULTS = DATASET / "analysis/results"
FIGURES = Path(os.environ.get("FIGURE_DIR", HERE / "figures"))
FONT = DATASET / "font"
FONT_REGULAR = FONT / "arial.ttf"
FONT_BOLD = FONT / "arialbd.ttf"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from vis.style import configure_style as _configure_style, save_figure

OUTPUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "png").lower()
PNG_DPI = int(os.environ.get("PNG_DPI", "600"))
if OUTPUT_FORMAT not in {"png", "pdf", "svg"}:
    raise ValueError("OUTPUT_FORMAT must be png, pdf, or svg")

TYPE_ORDER = [
    "Neural Ectoderm", "Non-neural Ectoderm", "Somitic Mesoderm",
    "Lateral Plate Mesoderm", "Axial Mesoderm", "Head Mesoderm",
    "Endoderm", "Extraembryonic", "Tailbud", "PGC",
]
TYPE_COLORS = {
    label: color for label, color in zip(TYPE_ORDER, [
        "#8EBAD9", "#E7A7B5", "#8FC89B", "#D9B46D", "#7CB3A5",
        "#C4A6D8", "#E99B72", "#B9B9B9", "#9B8AC4", "#D95F8D",
    ])
}
CLUSTER_COLORS = {
    1: "#77AADD", 2: "#EE8866", 3: "#E8C868", 4: "#44BB99",
    5: "#AA4499", 6: "#99DDFF", 7: "#CC6677", 8: "#6677AA",
}


def configure_style():
    return _configure_style(FONT, legend_size=6.5)


def save_publication_figure(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / f"{name}.{OUTPUT_FORMAT}"
    save_figure(fig, output, dpi=PNG_DPI)
    return output


def ordered_type_palette(labels):
    labels = set(labels)
    missing = sorted(labels - set(TYPE_COLORS))
    if missing:
        raise ValueError(f"Cell types missing from fixed palette: {missing}")
    order = [label for label in TYPE_ORDER if label in labels]
    return order, [TYPE_COLORS[label] for label in order]


def ordered_cluster_palette(clusters):
    clusters = sorted({int(cluster) for cluster in clusters})
    missing = sorted(set(clusters) - set(CLUSTER_COLORS))
    if missing:
        raise ValueError(f"Clusters missing from fixed palette: {missing}")
    labels = [f"Cluster {cluster}" for cluster in clusters]
    return labels, [CLUSTER_COLORS[cluster] for cluster in clusters]
