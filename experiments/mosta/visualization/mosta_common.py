#!/usr/bin/env python3
"""Dataset adapter over the shared Paper_results.vis library."""
from pathlib import Path
import os
import sys
CODE=Path(__file__).resolve().parent.parent
DATASET=Path(os.environ["CELLATTENTION_MOSTA_WORKDIR"])
HERE=DATASET/"visualization"
PROJECT=CODE
PRE=DATASET/"data/preprocessed"
RESULTS=DATASET/"analysis/results"
FONT=DATASET/"font"
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from vis.colors import categorical_palette
from vis.io import load_spatial_receiver_group_data,read_lines
from vis.style import configure_style as _configure_style
def lines(path):return read_lines(path)
def configure_style():return _configure_style(FONT)
def palette(labels,overrides=None):return categorical_palette(labels,overrides=overrides)
def load():return load_spatial_receiver_group_data(PRE,RESULTS,with_samples=True)

# Shared 16-cluster palette for MOSTA. Even hue spacing with alternating value so
# adjacent colors differ in hue AND lightness: no yellow, not too light, well separated.
_CLUSTER_COLORS={
    1:  '#D95F5F',  # coral red
    2:  '#5B8FC2',  # blue
    3:  "#FFE5AA",  # ochre
    4:  '#62A875',  # green
    5:  '#9A72B5',  # purple
    6:  '#E48B4A',  # orange
    7:  '#55B7B1',  # teal
    8:  '#D56B9B',  # rose
    9:  '#7EA54D',  # olive green
    10: '#6670B5',  # indigo
    11: '#D28A78',  # salmon
    12: '#4FA58D',  # jade
    13: '#B66DB0',  # magenta
    14: '#6FA8C4',  # sky blue
    15: '#B39A62',  # warm ochre
    16: '#8A7C6A',  # taupe
}
def cluster_palette(clusters=None):
    """Return the shared cluster->color mapping, ensuring every visualization uses
    identical colors (and, for scatter, the same alpha handled by the caller)."""
    if clusters is None:
        return dict(_CLUSTER_COLORS)
    return {int(c):_CLUSTER_COLORS[int(c)] for c in clusters}
