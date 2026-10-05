#!/usr/bin/env python3
"""Cluster-transition boundary detection on the Brain UMAP.

Reads the UMAP coordinates saved by plot_brain_development_umap.py and:

  1. computes each cell's cluster-mixing entropy from its k nearest neighbors in
     the UMAP plane  H_i = -sum_k p_ik log p_ik ;
  2. selects cells whose normalized cluster entropy is above a quantile
     (cluster boundary / mixed cells);
  3. groups them with DBSCAN and discards regions with too few cells;
  4. draws a smooth envelope (KDE density contour) around each region, filled
     with light-grey diagonal hatching, a black number (white outline) at its
     centre, and a legend below the cluster legend noting which two clusters
     each transition is between.

Only a cluster-colored figure is produced (no stage-colored overlay).
"""
PNG_DPI=600
K_NEIGHBORS=50
ENTROPY_QUANTILE=0.85
DBSCAN_EPS=0.4
DBSCAN_MIN_SAMPLES=25
MIN_REGION_CELLS=100
POINT_SIZE=2.0;ALPHA=0.7

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.path as mpath
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
from mosta_common import HERE,DATASET,configure_style,cluster_palette
from vis.style import mm_to_inches,save_figure

def cluster_entropy(clusters, neighbors):
    H=np.zeros(len(clusters))
    cats=sorted(set(clusters))
    for cat in cats:
        p=(clusters[neighbors]==cat).sum(axis=1)/neighbors.shape[1]
        H-=p*np.log(p+1e-12)
    return H, len(cats)

def smooth_envelope(ax, pts, level_frac=0.95, grid=180):
    """Return closed contour paths of a KDE density envelope around pts."""
    kde=gaussian_kde(pts.T,bw_method=1.0)
    d=kde(pts.T)
    level=np.quantile(d,1-level_frac)
    xmin,ymin=pts.min(axis=0);xmax,ymax=pts.max(axis=0)
    pad=0.5*max(xmax-xmin,ymax-ymin,0.2)
    xs=np.linspace(xmin-pad,xmax+pad,grid);ys=np.linspace(ymin-pad,ymax+pad,grid)
    gx,gy=np.meshgrid(xs,ys)
    z=kde(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape)
    cs=ax.contour(gx,gy,z,levels=[level])
    paths=[]
    for seg in cs.allsegs[0]:
        verts=np.asarray(seg)
        if len(verts)>=3:
            paths.append(mpath.Path(np.vstack([verts,verts[:1]]),closed=True))
    cs.remove()
    return paths

def main():
    configure_style()
    figdir=HERE/'figures/brain_development';datadir=HERE/'data'
    pdfdir=figdir
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
    pdfdir.mkdir(parents=True,exist_ok=True)

    df=pd.read_csv(datadir/'brain_development_umap_coordinates.csv')
    coords=df[['umap1','umap2']].to_numpy()
    clusters=df.cluster.to_numpy()

    tree=cKDTree(coords)
    _,idx=tree.query(coords,k=K_NEIGHBORS+1)
    neighbors=idx[:,1:]
    Hc,nc=cluster_entropy(clusters,neighbors)
    Hc_norm=Hc/np.log(nc)
    df['cluster_entropy']=Hc;df['cluster_entropy_norm']=Hc_norm

    tc=np.quantile(Hc_norm,ENTROPY_QUANTILE)
    cand=Hc_norm>=tc
    df['is_cluster_boundary']=cand
    df.to_csv(datadir/'brain_development_umap_coordinates.csv',index=False)

    # Group boundary cells; keep regions with enough cells.
    cc=coords[cand];cc_clusters=clusters[cand]
    labels=DBSCAN(eps=DBSCAN_EPS,min_samples=DBSCAN_MIN_SAMPLES).fit_predict(cc)
    bdy_idx=np.where(cand)[0]
    region_id=np.full(len(df),-1,int)
    regions=[]
    for lab in sorted(set(labels)-{-1}):
        m=labels==lab
        if m.sum()<MIN_REGION_CELLS:
            continue
        pts=cc[m]
        top=pd.Series(cc_clusters[m]).value_counts().index[:2].tolist()
        region_id[bdy_idx[m]]=len(regions)+1
        regions.append(dict(n=int(m.sum()),pts=pts,center=pts.mean(axis=0),clusters=top))
    df['boundary_region_id']=region_id
    df.to_csv(datadir/'brain_development_umap_coordinates.csv',index=False)

    cpal=cluster_palette()
    cluster_order=sorted(set(clusters))

    fig,ax=plt.subplots(figsize=mm_to_inches(180,150))
    fig.subplots_adjust(left=.04,right=.72,bottom=.06,top=.96)

    for c in cluster_order:
        q=df[df.cluster==c]
        ax.scatter(q.umap1,q.umap2,s=POINT_SIZE,c=cpal[c],alpha=ALPHA,
                   linewidths=0,rasterized=True)

    # smooth hatched envelopes + numbered labels
    for i,r in enumerate(regions):
        for path in smooth_envelope(ax,r['pts']):
            ax.add_patch(PathPatch(path,facecolor='none',edgecolor='#C8C8C8',
                                   hatch='///',linewidth=.6,zorder=9))
            ax.add_patch(PathPatch(path,facecolor='none',edgecolor='black',
                                   linewidth=1.6,zorder=10))
        cx,cy=r['center']
        ax.text(cx,cy,str(i+1),ha='center',va='center',fontsize=14,fontweight='bold',
                color='black',zorder=20,
                path_effects=[pe.withStroke(linewidth=3,foreground='white')])

    ax.set_aspect('equal');ax.set_axis_off()
    #ax.set_title(f'Brain cells: cluster-transition boundaries (n={len(df):,})',
    #             loc='left',fontweight='bold',fontsize=18)

    # cluster color legend (upper right)
    handles=[Line2D([0],[0],marker='o',linestyle='none',markersize=11,
                    markerfacecolor=cpal[c],markeredgewidth=0,label=f'C{c}') for c in cluster_order]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.64,.98),frameon=False,ncol=2,
               columnspacing=.6,labelspacing=.35,handletextpad=.35,fontsize=16)

    # number legend below the cluster legend (which two clusters each transition is between)
    lines=['Cluster transitions']
    for i,r in enumerate(regions):
        a,b=r['clusters'] if len(r['clusters'])==2 else (r['clusters'][0],r['clusters'][0])
        lines.append(f'{i+1}: C{a} ↔ C{b}')
    fig.text(.74,.52,'\n'.join(lines),ha='left',va='top',fontsize=16,color='#202020')

    png_out=figdir/'brain_umap_cluster_boundary.png'
    pdf_out=pdfdir/'brain_umap_cluster_boundary.pdf'
    save_figure(fig,png_out,dpi=PNG_DPI,close=False)
    save_figure(fig,pdf_out,dpi=PNG_DPI)

    pd.DataFrame([{'region':i+1,'n_cells':r['n'],'cluster_a':r['clusters'][0],
                   'cluster_b':(r['clusters'][1] if len(r['clusters'])>1 else r['clusters'][0])}
                  for i,r in enumerate(regions)])\
      .to_csv(datadir/'brain_development_cluster_boundary_regions.csv',index=False)

    print(png_out)
    print(pdf_out)
    print(f'boundary cells={int(cand.sum())} regions={len(regions)}')

if __name__=='__main__':main()
