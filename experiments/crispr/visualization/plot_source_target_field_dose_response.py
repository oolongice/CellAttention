#!/usr/bin/env python3
import os
"""Plot a representative local source field and target dose response."""

SOURCE_GENE = "Icam1"
TARGET_GENE = os.environ.get("CELLATTENTION_TARGET_GENE", "Cxcl10")
OUTPUT_FORMAT = os.environ.get("CELLATTENTION_PLOT_FORMAT", "png")
PNG_DPI = 600
LOCAL_HALF_WIDTH_UM = 60.0
HOTSPOT_RADIUS_UM = 55.0
DISPLAY_QUANTILE = 0.995
DOSE_RESPONSE_BINS = 5

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from field_visualization_common import (CHECKPOINT, PREPROCESSED, calculate_cell_field,
    calculate_field, clipped_field, configure_style, grid_for_bounds, read_source)

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
IMMUNE = {"Macrophages", "B", "T", "DC", "Neutrophils", "MAST", "Plasma", "NK"}


def read_target(gene):
    genes = [x.strip() for x in (CHECKPOINT / "gene_ids.txt").read_text().splitlines() if x.strip()]
    if gene not in genes:
        match = [x for x in genes if x.lower() == gene.lower()]
        if len(match) != 1: raise ValueError(f"Target gene {gene!r} is absent from checkpoint")
        gene = match[0]
    values = np.loadtxt(CHECKPOINT / "raw_expression.csv", delimiter=",", dtype=float,
                        usecols=[genes.index(gene)])
    return gene, values


def hotspot(coords, field, target, immune):
    cells = np.flatnonzero(immune)
    activity = np.maximum(field[cells], 0) * np.maximum(target[cells], 0)
    if not np.any(activity > 0): activity = np.maximum(field[cells], 0)
    tree = cKDTree(coords[cells])
    score = np.array([activity[n].sum() for n in tree.query_ball_point(coords[cells], HOTSPOT_RADIUS_UM)])
    return coords[cells[int(np.argmax(score))]]


def dose_response(field, target, immune):
    x, y = field[immune], target[immune]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, DOSE_RESPONSE_BINS + 1)))
    group = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    pos=[]; frac=[]; err=[]
    for g in range(len(edges)-1):
        keep=group==g
        if not keep.any(): continue
        f=float(np.mean(y[keep]>0)); pos.append(g+1); frac.append(f)
        err.append(1.96*np.sqrt(f*(1-f)/keep.sum()))
    return np.asarray(pos),np.asarray(frac),np.asarray(err),float(spearmanr(x,y).statistic)


def main():
    fmt=OUTPUT_FORMAT.lower()
    if fmt not in {"png","pdf"}: raise ValueError('OUTPUT_FORMAT must be "png" or "pdf"')
    configure_style()
    source,coords,source_expr=read_source(SOURCE_GENE)
    target,target_expr=read_target(TARGET_GENE)
    annotations=np.asarray([x.strip() for x in (PREPROCESSED/"annotation.txt").read_text().splitlines() if x.strip()])
    immune=np.isin(annotations,list(IMMUNE))
    node_field,positive=calculate_cell_field(coords,source_expr)
    center=hotspot(coords,node_field,target_expr,immune)
    bounds=(center[0]-LOCAL_HALF_WIDTH_UM,center[0]+LOCAL_HALF_WIDTH_UM,
            center[1]-LOCAL_HALF_WIDTH_UM,center[1]+LOCAL_HALF_WIDTH_UM)
    mx,my,query=grid_for_bounds(bounds,420,300)
    continuous,_=calculate_field(coords,source_expr,query)
    shown,field_max=clipped_field(continuous,DISPLAY_QUANTILE);shown=shown.reshape(mx.shape)
    local=((coords[:,0]>=bounds[0])&(coords[:,0]<=bounds[1])&
           (coords[:,1]>=bounds[2])&(coords[:,1]<=bounds[3]))
    source_local=local&positive&~immune; target_local=local&immune

    fig,ax=plt.subplots(figsize=(116/25.4,94/25.4))
    fig.subplots_adjust(left=0.16,right=0.74,bottom=0.16,top=0.87)
    field_art=ax.pcolormesh(mx,my,shown,cmap="Blues",vmin=0,vmax=field_max,
                            shading="auto",alpha=.88,rasterized=True,zorder=0)
    other=local&~source_local&~target_local
    ax.scatter(coords[other,0],coords[other,1],s=8,c="#BDBDBD",alpha=.42,linewidths=0,rasterized=True)
    ax.scatter(coords[source_local,0],coords[source_local,1],s=28,c="#2171B5",
               edgecolors="#08306B",linewidths=.55,label=f"{source}+ source",zorder=3)
    target_art=ax.scatter(coords[target_local,0],coords[target_local,1],s=31,
        c=target_expr[target_local],cmap="Oranges",vmin=0,edgecolors="#A63603",
        linewidths=.5,marker="^",label="Immune receiver cell",zorder=4)
    ax.set_xlim(bounds[:2]);ax.set_ylim(bounds[3],bounds[2]);ax.set_aspect("equal")
    ax.set_xlabel("x (µm)");ax.set_ylabel("y (µm)")
    ax.set_title(f"{source} → {target}\nrepresentative local diffusion field",loc="left",fontweight="bold",pad=4)
    ax.legend(loc="upper right",frameon=True,framealpha=.88,fontsize=6)
    for spine in ax.spines.values():spine.set_linewidth(.6)
    ax.tick_params(length=2.5,width=.6,pad=2)
    c1=fig.add_axes([.79,.53,.026,.28]);b1=fig.colorbar(field_art,cax=c1);b1.set_label(f"{source} field",fontsize=6,labelpad=3);c1.tick_params(labelsize=5.5,pad=2)
    c2=fig.add_axes([.79,.16,.026,.24]);b2=fig.colorbar(target_art,cax=c2);b2.set_label(f"{target} expression",fontsize=6,labelpad=3);c2.tick_params(labelsize=5.5,pad=2)

    positions,fractions,errors,rho=dose_response(node_field,target_expr,immune)
    inset=ax.inset_axes([.57,.06,.36,.36])
    inset.errorbar(positions,fractions,yerr=errors,color="#D95F02",marker="o",markersize=2.8,linewidth=1,capsize=1.8)
    inset.set_facecolor((1,1,1,.90));inset.set_xticks(positions)
    inset.set_xlabel(f"{source} field quintile",fontsize=5.5,labelpad=1)
    inset.set_ylabel(f"{target}+ fraction",fontsize=5.5,labelpad=1)
    inset.set_title(f"All immune cells; Spearman ρ={rho:.2f}",fontsize=5.5,pad=2)
    inset.tick_params(axis="both",labelbottom=False,labelleft=False,length=2,pad=1)
    inset.set_box_aspect(1)
    inset.grid(axis="y",color="#DDD",linewidth=.4);inset.set_ylim(bottom=0)

    out=HERE/"figures"/f"{source}_to_{target}_field_dose_response.{fmt}";out.parent.mkdir(parents=True,exist_ok=True)
    options={"facecolor":"white"}
    if fmt=="png":options.update({"dpi":PNG_DPI,"pil_kwargs":{"compress_level":6}})
    fig.savefig(out,**options);plt.close(fig)
    print(f"source_gene={source}");print(f"target_gene={target}")
    print(f"hotspot_center=({center[0]:.3f},{center[1]:.3f})")
    print(f"local_cells={int(local.sum())}");print(f"immune_cells={int(immune.sum())}")
    print(f"spearman_rho={rho:.6f}");print(f"output={out}")


if __name__=="__main__":
    try:main()
    except Exception as error:print(f"error: {error}",file=sys.stderr);raise
