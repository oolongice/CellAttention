#!/usr/bin/env python3
"""Brain-development UMAP of Brain-annotated cell embeddings.

Two figures from the same 2D UMAP of the frozen-transformer 48-d embeddings for
cells annotated 'Brain':
  1. colored by embryonic stage (E10.5-E13.5)
  2. colored by CCC Module (1-16)

The 2D coordinates (with stage and cluster) are saved to
visualization/data/brain_development_umap_coordinates.csv and reused by
plot_brain_development_boundary.py for the boundary/transition analysis.
"""
OUTPUT_FORMAT='png';PNG_DPI=600
ANNOTATION='Brain'
STAGES=['E10.5','E11.5','E12.5','E13.5']
STAGE_COLORS={'E10.5':'#2C7BB6','E11.5':'#1B9E77','E12.5':'#FD8D3C','E13.5':'#D7191C'}
POINT_SIZE=2.0;ALPHA=0.7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D
from mosta_common import HERE,DATASET,PRE,RESULTS,configure_style,cluster_palette,lines
from vis.style import mm_to_inches,save_figure

def main():
    configure_style()
    figdir=HERE/'figures/brain_development';datadir=HERE/'data'
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)

    cells=lines(PRE/'cell_ids.txt')
    samples=lines(PRE/'sample_ids.txt')
    ann=lines(PRE/'evaluation_cell_groups.txt')
    embeddings=np.loadtxt(DATASET/'model/transformer/cell_embeddings.csv',delimiter=',')
    clusters=pd.read_csv(RESULTS/'receiver_group_assignments.csv')['receiver_group'].to_numpy(int)+1

    mask=ann==ANNOTATION
    n=int(mask.sum())
    X=embeddings[mask]

    reducer=umap.UMAP(n_neighbors=30,min_dist=0.3,metric='euclidean',random_state=42)
    coords=reducer.fit_transform(X)
    df=pd.DataFrame({'cell_id':cells[mask],'stage':samples[mask],'cluster':clusters[mask],
                     'umap1':coords[:,0],'umap2':coords[:,1]})
    df.to_csv(datadir/'brain_development_umap_coordinates.csv',index=False)

    stage_order=[s for s in STAGES if s in set(df.stage)]
    cluster_order=sorted(set(df.cluster))
    cpal=cluster_palette()

    # --- figure 1: colored by stage ---
    fig,ax=plt.subplots(figsize=mm_to_inches(180,150))
    fig.subplots_adjust(left=.04,right=.76,bottom=.04,top=.96)
    for s in stage_order:
        q=df[df.stage==s]
        ax.scatter(q.umap1,q.umap2,s=POINT_SIZE,c=STAGE_COLORS[s],alpha=ALPHA,
                   linewidths=0,rasterized=True,label=s)
    ax.set_aspect('equal');ax.set_axis_off()
    ax.set_title(f'Brain cells: embedding UMAP by stage (n={n:,})',loc='left',fontweight='bold',fontsize=18)
    handles=[Line2D([0],[0],marker='o',linestyle='none',markersize=11,
                    markerfacecolor=STAGE_COLORS[s],markeredgewidth=0,label=s) for s in stage_order]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.78,.5),frameon=False,
               labelspacing=.6,handletextpad=.4,fontsize=15)
    save_figure(fig,figdir/f'brain_umap_stage.{OUTPUT_FORMAT}',dpi=PNG_DPI)

    # --- figure 2: colored by cluster ---
    fig,ax=plt.subplots(figsize=mm_to_inches(180,150))
    fig.subplots_adjust(left=.04,right=.74,bottom=.04,top=.96)
    for c in cluster_order:
        q=df[df.cluster==c]
        ax.scatter(q.umap1,q.umap2,s=POINT_SIZE,c=cpal[c],alpha=ALPHA,
                   linewidths=0,rasterized=True)
    ax.set_aspect('equal');ax.set_axis_off()
    ax.set_title(f'Brain cells: embedding UMAP by cluster (n={n:,})',loc='left',fontweight='bold',fontsize=18)
    handles=[Line2D([0],[0],marker='o',linestyle='none',markersize=11,
                    markerfacecolor=cpal[c],markeredgewidth=0,label=f'C{c}') for c in cluster_order]
    fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.76,.5),frameon=False,ncol=2,
               columnspacing=.6,labelspacing=.35,handletextpad=.35,fontsize=12)
    save_figure(fig,figdir/f'brain_umap_cluster.{OUTPUT_FORMAT}',dpi=PNG_DPI)

    print(figdir/f'brain_umap_stage.{OUTPUT_FORMAT}')
    print(figdir/f'brain_umap_cluster.{OUTPUT_FORMAT}')

if __name__=='__main__':main()
