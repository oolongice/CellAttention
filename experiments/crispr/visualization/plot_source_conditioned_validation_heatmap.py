#!/usr/bin/env python3
import os
"""Compact heatmap of model effects and CRISPR evidence for fixed sources."""

SOURCE_GENES = ["Icam1", "Cxcr4"]
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
MODEL_TOP_N = 10
EXPERIMENT_OVERLAPS = 5

from pathlib import Path
from font_helper import font_family
import sys
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE=Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"])/"visualization"
PROJECT=HERE.parent.parent
INPUT=HERE/"data/source_conditioned_crispr_validation.csv"
FONT=HERE.parent/"font"


def style():
    family = font_family(FONT)
    mpl.rcParams.update({"font.family":family,"font.sans-serif":[family],"font.size":7,
                         "axes.titlesize":8,"xtick.labelsize":6.2,"ytick.labelsize":6.2,
                         "pdf.fonttype":42,"ps.fonttype":42})


def select(part):
    top=part.nsmallest(MODEL_TOP_N,"physical_rank_within_source_receiver")
    overlaps=part[part.experimental_supported].nsmallest(EXPERIMENT_OVERLAPS,"physical_rank_within_source_receiver")
    return pd.concat([top,overlaps]).drop_duplicates(["target_gene"]).sort_values("physical_rank_within_source_receiver")


def main():
    fmt=OUTPUT_FORMAT.lower();style();data=pd.read_csv(INPUT)
    missing=sorted(set(SOURCE_GENES)-set(data.source_gene.unique()))
    if missing: raise ValueError(f"Requested sources absent from prepared validation data: {missing}")
    if not SOURCE_GENES: raise ValueError("SOURCE_GENES must contain at least one gene")
    width=(104 if len(SOURCE_GENES)==1 else 183)/25.4
    fig,axes=plt.subplots(1,len(SOURCE_GENES),figsize=(width,112/25.4),squeeze=False)
    axes=axes.ravel()
    # Fixed margins prevent rotated x labels and long gene labels being clipped.
    left = .25 if len(SOURCE_GENES) == 1 else .145
    fig.subplots_adjust(left=left,right=.965,bottom=.27,top=.90,wspace=.72)
    for ax,source in zip(axes,SOURCE_GENES):
        part=select(data[data.source_gene.eq(source)]).copy()
        effect_scale=max(float(part.effect_median_physical.abs().max()),1e-12)
        logfc_scale=max(float(data.crispr_logfc.abs().max(skipna=True)),1e-12)
        evidence_scale=max(float(data.experimental_evidence.max(skipna=True)),1e-12)
        matrix=np.column_stack([
            part.effect_median_physical/effect_scale,
            part.crispr_logfc/logfc_scale,
            part.experimental_evidence/evidence_scale,
        ])
        labels=[f"{g}  [{int(r)}]" for g,r in zip(part.target_gene,part.physical_rank_within_source_receiver)]
        sns.heatmap(matrix,ax=ax,cmap="vlag",center=0,vmin=-1,vmax=1,cbar=False,
                    linewidths=1,linecolor="white",
                    xticklabels=["Model effect","CRISPR logFC","−log10 FDR"],
                    yticklabels=labels,mask=np.isnan(matrix))
        for row,value in enumerate(part.crispr_fdr):
            if np.isfinite(value):
                mark="+" if bool(part.direction_concordant.iloc[row]) else "x"
                ax.text(2.5,row+.5,mark,ha="center",va="center",fontsize=7,
                        color="white" if matrix[row,2]>.45 else "#333")
        ax.set_title(source,loc="left",fontweight="bold",pad=5)
        ax.set_ylabel("Target gene [model rank]")
        ax.tick_params(axis="x",rotation=28,length=0);ax.tick_params(axis="y",rotation=0,length=0)
    slug="_".join(SOURCE_GENES)
    out=HERE/"figures"/f"source_conditioned_validation_heatmap_{slug}.{fmt}";out.parent.mkdir(parents=True,exist_ok=True)
    opts={"facecolor":"white"}
    if fmt=="png":opts.update({"dpi":PNG_DPI,"pil_kwargs":{"compress_level":6}})
    fig.savefig(out,**opts);plt.close(fig);print(f"output={out}")


if __name__=="__main__":
    try:main()
    except Exception as error:print(f"error: {error}",file=sys.stderr);raise
