#!/usr/bin/env python3
OUTPUT_FORMAT="png";PNG_DPI=600
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from mosta_common import HERE,PROJECT,configure_style,load,palette,cluster_palette
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from vis.composition import cell_type_composition,plot_stacked_composition
from vis.spatial import plot_spatial_categories
from vis.style import mm_to_inches,save_figure
point_size = [6.5, 3.5, 3.0, 1.7]
def main():
    configure_style();xy,samples,ann,clusters=load();figdir=HERE/"figures";datadir=HERE/"data";figdir.mkdir(exist_ok=True);datadir.mkdir(exist_ok=True);annotation_colors=palette(ann);receiver_group_colors=cluster_palette()
    for idx,sample in enumerate(sorted(set(samples))):
        keep=samples==sample
        plot_spatial_categories(xy[keep],ann[keep],annotation_colors,title=f"MOSTA embryo: {sample} cell annotations",path=figdir/f"{sample}_cell_annotations.{OUTPUT_FORMAT}",legend=True)
        plot_spatial_categories(xy[keep],clusters[keep],receiver_group_colors,title="",path=figdir/f"{sample}_model_clusters.{OUTPUT_FORMAT}",legend=False,figure_mm=(105,150),layout=dict(left=.005,right=.995,bottom=.005,top=.995),point_size=point_size[idx],alpha=1.0)
    composition=cell_type_composition(clusters,ann);composition.to_csv(datadir/"cluster_cell_type_composition.csv",index=False)
    fig,ax=plt.subplots(figsize=mm_to_inches(180,82));fig.subplots_adjust(left=.08,right=.76,bottom=.15,top=.90);plot_stacked_composition(ax,composition,annotation_colors);save_figure(fig,figdir/f"cluster_cell_type_composition.{OUTPUT_FORMAT}",dpi=PNG_DPI)
if __name__=="__main__":main()
