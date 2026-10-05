#!/usr/bin/env python3
"""Circular ring of cluster x model target-gene membership.

The angle axis is split into k sectors, one per distinct top-20 target gene
across all clusters. The radius axis is split into n_clusters bands (cluster
1 -> 16 from the inner to the outer edge) with the inner circle left blank.
Each (gene sector x cluster band) cell is filled with that cluster's color if
the cluster's top-20 targets contain the gene, and left blank otherwise.
Gene names are written radially along the outer circumference.
"""
OUTPUT_FORMAT='png';PNG_DPI=600;TOP_K=20
R_INNER=0.12;R_BASE=0.30;R_OUTER=0.80
GENE_FONT=8.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge,Circle
from matplotlib.lines import Line2D
from mosta_common import HERE,DATASET,configure_style,cluster_palette
from vis.style import mm_to_inches,save_figure

def main():
    configure_style()
    figdir=HERE/'figures/mosta_specific';datadir=HERE/'data'
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)

    relations=pd.read_csv(DATASET/'analysis/all_selected_cluster_source_target_relations.csv')
    rel=relations[relations.selected.astype(str).str.lower().eq('true')]
    clusters=sorted(rel.cluster.unique())
    n_clusters=len(clusters)

    # top-20 targets per cluster; membership pattern of each gene
    genes=[]
    genes_by_cluster={}
    for c in clusters:
        g=rel[rel.cluster==c]
        top=g.groupby('target_gene').target_rank.min().nsmallest(TOP_K).index.tolist()
        genes_by_cluster[c]=set(top)
        for gene in top:
            if gene not in genes:
                genes.append(gene)

    # Gene membership across clusters
    cluster_index={c:i for i,c in enumerate(clusters)}
    gene_clusters={gene:[] for gene in genes}
    for c in clusters:
        for gene in genes_by_cluster[c]:
            gene_clusters[gene].append(cluster_index[c])

    # Order so shared targets group together first (descending shared-cluster count),
    # then by the exact membership pattern so co-shared genes sit adjacent; cluster-
    # specific genes (count==1) come last.
    genes=sorted(genes,
                 key=lambda g:( -len(gene_clusters[g]),
                                tuple(sorted(gene_clusters[g])),
                                g))

    k=len(genes)
    gene_index={g:i for i,g in enumerate(genes)}

    # membership matrix (cluster x gene)
    filled=np.zeros((n_clusters,k),bool)
    for ci,c in enumerate(clusters):
        for gene in genes_by_cluster[c]:
            filled[ci,gene_index[gene]]=True

    pd.DataFrame({'cluster':clusters,
                  'target_genes':[', '.join(genes_by_cluster[c]) for c in clusters]})\
      .to_csv(datadir/'cluster_target_ring_genes.csv',index=False)
    pd.DataFrame(filled.T,index=genes,columns=[f'C{c}' for c in clusters])\
      .to_csv(datadir/'cluster_target_ring_membership.csv')

    colors=cluster_palette()

    fig,ax=plt.subplots(figsize=mm_to_inches(180,180))
    fig.subplots_adjust(left=.04,right=.96,bottom=.10,top=.95)
    ax.set_aspect('equal');ax.axis('off')
    ax.set_xlim(-1.0,1.0);ax.set_ylim(-1.0,1.0)

    dtheta=2*np.pi/k
    # angular cell boundaries: sector for gene i spans [i*dtheta,(i+1)*dtheta] (radians)
    # inner blank circle up to R_INNER, bands from R_INNER to R_OUTER
    band=(R_OUTER-R_INNER)/n_clusters
    for i in range(k):
        a0=np.rad2deg(i*dtheta); a1=np.rad2deg((i+1)*dtheta)
        for ci,c in enumerate(clusters):
            r0=R_INNER+ci*band; r1=r0+band
            if filled[ci,i]:
                ax.add_patch(Wedge((0,0),r1,a0,a1,width=r1-r0,
                                   facecolor=colors[c],edgecolor='white',linewidth=.15))
        # gene label at outer radius, middle of sector (radians); text runs radially
        mid=(i*dtheta+(i+1)*dtheta)/2
        rtext=R_OUTER+0.02
        rot=np.rad2deg(mid)
        # flip text on the left half so it reads upright
        flip=90<rot<270
        ha='left' if not flip else 'right'
        ax.text(rtext*np.cos(mid),rtext*np.sin(mid),genes[i],
                ha=ha,va='center',rotation=(rot+180) if flip else rot,
                rotation_mode='anchor',fontsize=GENE_FONT,color='#202020')

    # thin light radial separators between gene sectors
    for i in range(k+1):
        a=i*dtheta
        x0,x1=R_INNER*np.cos(a),R_OUTER*np.cos(a)
        y0,y1=R_INNER*np.sin(a),R_OUTER*np.sin(a)
        ax.plot([x0,x1],[y0,y1],color='#C8C8C8',linewidth=.2,zorder=2)

    # inner and outer circle boundaries (slightly thicker)
    ax.add_patch(Circle((0,0),R_INNER,facecolor='none',edgecolor='#202020',linewidth=1.1,zorder=3))
    ax.add_patch(Circle((0,0),R_OUTER,facecolor='none',edgecolor='#202020',linewidth=1.1,zorder=3))

    # legend for clusters (compact, horizontal at bottom)
    handles=[Line2D([0],[0],marker='o',linestyle='none',markersize=5,
                    markerfacecolor=colors[c],markeredgewidth=0,label=f'C{c}') for c in clusters]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),
               frameon=False,ncol=8,columnspacing=.55,labelspacing=.2,handletextpad=.25,
               borderaxespad=0,fontsize=8)

    fig.suptitle('Model target-gene membership across clusters (top-20 per cluster)',
                 x=.5,y=.98,ha='center',fontweight='bold',fontsize=9)
    out=figdir/f'cluster_target_ring.{OUTPUT_FORMAT}'
    save_figure(fig,out,dpi=PNG_DPI)
    print(out,f'genes={k} clusters={n_clusters} filled={int(filled.sum())}/{filled.size}')

if __name__=='__main__':main()
