"""Aligned SlideSeqV2 inputs and figure style for the local experiment workspace."""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
DATASET = Path(os.environ.get("CELLATTENTION_SLIDESEQ_WORKDIR", PROJECT / ".local/slideseqv2")).expanduser().resolve()
HERE = DATASET / "visualization"
PRE = DATASET / "data/preprocessed"
RESULTS = DATASET / "analysis/results"


def lines(path):
    return np.asarray([x.strip() for x in Path(path).read_text().splitlines() if x.strip()], dtype=object)


def configure_style():
    font_dir = DATASET / "font"
    fonts = [font_dir / x for x in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")]
    if all(x.is_file() for x in fonts):
        for path in fonts:
            fm.fontManager.addfont(path)
        family = fm.FontProperties(fname=str(fonts[0])).get_name()
    else:
        family = "DejaVu Sans"
    mpl.rcParams.update({
        "font.family": family, "font.sans-serif": [family], "font.size": 7,
        "axes.titlesize": 9, "axes.labelsize": 7, "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5, "legend.fontsize": 5.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    })
    return family


def palette(labels, overrides=None):
    labels = sorted(set(labels), key=str)
    colors = []
    for name in ("Set3", "tab20", "Pastel1", "Dark2"):
        cmap = plt.get_cmap(name)
        colors.extend(cmap(np.linspace(.03, .97, cmap.N)))
    result = {label: colors[i % len(colors)] for i, label in enumerate(labels)}
    if overrides:
        result.update({key: value for key, value in overrides.items() if key in result})
    return result


def load():
    xy = np.loadtxt(PRE / "spatial_coordinates.csv", delimiter=",")
    annotations = lines(PRE / "evaluation_cell_groups.txt")
    cells = lines(PRE / "cell_ids.txt")
    assignments = pd.read_csv(RESULTS / "receiver_group_assignments.csv")
    if len(assignments) != len(cells) or not np.array_equal(assignments.cell_id.astype(str).to_numpy(), cells.astype(str)):
        raise ValueError("receiver-group assignments do not align with cell IDs")
    clusters = assignments.receiver_group.to_numpy(int) + 1
    if len({len(xy), len(annotations), len(clusters)}) != 1:
        raise ValueError("SlideSeqV2 spatial inputs have unequal row counts")
    return xy, annotations, clusters
