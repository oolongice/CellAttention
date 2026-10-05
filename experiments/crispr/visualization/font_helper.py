"""Use supplied Arial files for paper typography, otherwise DejaVu Sans."""
from pathlib import Path
import matplotlib.font_manager as fm

def font_family(directory):
    paths = [Path(directory) / name for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")]
    if not all(path.is_file() for path in paths):
        return "DejaVu Sans"
    for path in paths:
        fm.fontManager.addfont(path)
    return fm.FontProperties(fname=str(paths[0])).get_name()
