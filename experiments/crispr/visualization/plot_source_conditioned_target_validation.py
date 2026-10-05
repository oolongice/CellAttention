#!/usr/bin/env python3
import os
"""Plot model-prioritized targets and matching CRISPR evidence for two fixed sources."""

SOURCE_GENES = ["Icam1", "Cxcr4"]
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
TOP_TARGETS = 20

from pathlib import Path
from font_helper import font_family
import sys
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
PROJECT = HERE.parent.parent
INPUT = HERE / "data/source_conditioned_crispr_validation.csv"
FONT = HERE.parent / "font"


def style():
    family = font_family(FONT)
    mpl.rcParams.update({"font.family":family,"font.sans-serif":[family],"font.size":7,
                         "axes.titlesize":8,"xtick.labelsize":6.5,"ytick.labelsize":6.2,
                         "pdf.fonttype":42,"ps.fonttype":42})


def main():
    fmt=OUTPUT_FORMAT.lower(); style(); data=pd.read_csv(INPUT)
    missing=sorted(set(SOURCE_GENES)-set(data.source_gene.unique()))
    if missing: raise ValueError(f"Requested sources absent from prepared validation data: {missing}")
    if not SOURCE_GENES: raise ValueError("SOURCE_GENES must contain at least one gene")
    vmax=max(1.0,float(data.crispr_logfc.abs().max(skipna=True)))
    norm=mpl.colors.TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax)
    width=(104 if len(SOURCE_GENES)==1 else 183)/25.4
    fig,axes=plt.subplots(1,len(SOURCE_GENES),figsize=(width,112/25.4),squeeze=False)
    axes=axes.ravel()
    # Keep long target labels and the colour-bar label inside the fixed canvas.
    single_source = len(SOURCE_GENES) == 1
    left = .22 if single_source else .13
    right = .80 if single_source else .865
    fig.subplots_adjust(left=left,right=right,bottom=.14,top=.90,wspace=.62)
    scatter=None
    for ax,source in zip(axes,SOURCE_GENES):
        part=data[data.source_gene.eq(source)].nsmallest(TOP_TARGETS,"physical_rank_within_source_receiver").sort_values("physical_rank_within_source_receiver",ascending=False)
        y=np.arange(len(part)); x=part.heldout_mse_reduction_physical.to_numpy()
        ax.hlines(y,0,x,color="#D7DCE0",linewidth=.8,zorder=1)
        supported=part.experimental_supported.to_numpy(bool)
        sizes=np.where(supported,28+12*np.minimum(part.experimental_evidence.fillna(0),10),28)
        colors=np.where(supported,part.crispr_logfc.fillna(0),np.nan)
        ax.scatter(x[~supported],y[~supported],s=sizes[~supported],c="#C8CDD1",edgecolors="white",linewidths=.5,zorder=2)
        if supported.any():
            edge=np.where(part.direction_concordant.to_numpy(bool)[supported],"#202020","#D95F02")
            scatter=ax.scatter(x[supported],y[supported],s=sizes[supported],c=colors[supported],
                               cmap="coolwarm",norm=norm,edgecolors=edge,linewidths=.9,zorder=3)
        labels=[f"{g}  ({int(r)})" for g,r in zip(part.target_gene,part.physical_rank_within_source_receiver)]
        ax.set_yticks(y,labels);ax.set_xlabel("Source-conditioned held-out MSE reduction")
        ax.set_title(source,loc="left",fontweight="bold",pad=5);ax.axvline(0,color="#777",linewidth=.6)
        ax.grid(axis="x",color="#E6E6E6",linewidth=.5);ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(length=2.5,pad=2)
        if not supported.any():
            ax.text(.98,.02,"No DGE-supported target\nin model Top 20",transform=ax.transAxes,
                    ha="right",va="bottom",fontsize=6,color="#555")
    cbar_left = .84 if single_source else .895
    cax=fig.add_axes([cbar_left,.26,.012,.48]);bar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap="coolwarm"),cax=cax)
    bar.set_label("CRISPR logFC",labelpad=4)
    slug="_".join(SOURCE_GENES)
    out=HERE/"figures"/f"source_conditioned_target_validation_{slug}.{fmt}";out.parent.mkdir(parents=True,exist_ok=True)
    opts={"facecolor":"white"}
    if fmt=="png":opts.update({"dpi":PNG_DPI,"pil_kwargs":{"compress_level":6}})
    fig.savefig(out,**opts);plt.close(fig);print(f"output={out}")


if __name__=="__main__":
    try:main()
    except Exception as error:print(f"error: {error}",file=sys.stderr);raise
