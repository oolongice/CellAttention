"""Project-wide journal figure style and output helpers."""
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
MM_PER_INCH=25.4
def mm_to_inches(*values):
    result=tuple(float(v)/MM_PER_INCH for v in values)
    return result[0] if len(result)==1 else result
def configure_style(font_dir,*,font_size=7,axes_title_size=9,axes_label_size=7,tick_size=6.5,legend_size=5.5,overrides=None):
    font_dir=Path(font_dir);names=("arial.ttf","arialbd.ttf","ariali.ttf","arialbi.ttf");paths=[font_dir/x for x in names];missing=[str(x) for x in paths if not x.is_file()]
    if missing:
        family="DejaVu Sans"
    else:
        for path in paths:fm.fontManager.addfont(path)
        family=fm.FontProperties(fname=str(paths[0])).get_name()
    settings={"font.family":family,"font.sans-serif":[family],"font.size":font_size,"axes.titlesize":axes_title_size,"axes.labelsize":axes_label_size,"xtick.labelsize":tick_size,"ytick.labelsize":tick_size,"legend.fontsize":legend_size,"pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"}
    if overrides:settings.update(overrides)
    mpl.rcParams.update(settings);return family
def save_figure(fig,path,*,dpi=600,close=True,transparent=False,**kwargs):
    import matplotlib.pyplot as plt
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);options={"facecolor":"none" if transparent else "white","transparent":transparent,"dpi":dpi,**kwargs}
    if path.suffix.lower()==".png":options.update({"pil_kwargs":{"compress_level":6}})
    fig.savefig(path,**options)
    if close:plt.close(fig)
    return path
