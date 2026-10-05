#!/usr/bin/env python3
"""Spatial diffusion fields of transition source genes over cluster maps.

Two figures: the model reaction-diffusion source field of
  C7<->C8 : Wnt3a at E11.5
  C8<->C11: Shh   at E12.5
drawn as dark contour lines over a brain-ROI crop of the stage's cluster map.
The ROI is a square window centred on the densest brain-annotated cell of the
two transition clusters. The transition clusters keep their cluster colors at
full opacity; all other clusters are faded.
"""
OUTPUT_FORMAT='png';PNG_DPI=600
LENGTH=50.0;CUTOFF=150.0;MIN_DIST=2.0
BACKGROUND_COLOR='#D0D0D0'
FADE_ALPHA=0.62
EMPH_ALPHA=0.4
TRANS_ALPHA=1.0
GRID_N=640
SMOOTH_SIGMA=4.0
MIN_AREA_FRAC=0.01
MAX_COMPONENTS=5
CONTOUR_LEVELS=[(0.40,"#9EDFFF",0.7),(0.7,"#617DFD",0.9),(0.95,"#1C60FF",1.3)]
ROI_HALF=3000.0

CASES=[
 dict(transition='C7\u2194C8',clusters=(7,8),gene='Wnt3a',stage='E11.5',point_size=3.5,roi_half=2400.0,shift=(2000.0,0.0)),
 dict(transition='C8\u2194C11',clusters=(8,11),gene='Shh',stage='E12.5',point_size=3.0),
 dict(transition='C7\u2194C8',clusters=(7,8),gene='Wnt3a',stage='E12.5',point_size=3.0),
 dict(transition='C8\u2194C11',clusters=(8,11),gene='Shh',stage='E13.5',point_size=1.7),
]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.special import k0
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba
from mosta_common import HERE,DATASET,PRE,RESULTS,configure_style,cluster_palette,lines
from vis.style import mm_to_inches,save_figure

def kernel(d):return k0(np.maximum(d,MIN_DIST)/LENGTH)/(2*np.pi*LENGTH**2)

def gridfield(xy,e,b,n=GRID_N):
    x=np.linspace(b[0],b[1],n);y=np.linspace(b[2],b[3],n)
    mx,my=np.meshgrid(x,y);q=np.c_[mx.ravel(),my.ravel()]
    pos=e>0
    coo=cKDTree(q).sparse_distance_matrix(cKDTree(xy[pos]),CUTOFF,output_type='coo_matrix')
    v=np.bincount(coo.row,weights=kernel(coo.data)*e[pos][coo.col],minlength=len(q))
    return mx,my,v.reshape(mx.shape)

def main():
    configure_style()
    figdir=HERE/'figures/brain_development';datadir=HERE/'data'
    pdfdir=figdir
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
    pdfdir.mkdir(parents=True,exist_ok=True)

    cells=lines(PRE/'cell_ids.txt')
    samples=lines(PRE/'sample_ids.txt')
    ann=lines(PRE/'evaluation_cell_groups.txt')
    xy_all=np.loadtxt(PRE/'spatial_coordinates.csv',delimiter=',')
    clusters_all=pd.read_csv(RESULTS/'receiver_group_assignments.csv')['receiver_group'].to_numpy(int)+1
    gene_ids=lines(DATASET/'model/transformer/gene_ids.txt');gi={g:i for i,g in enumerate(gene_ids)}
    expr=mmread(PRE/'expression.mtx').tocsr()
    if expr.shape[0]!=len(cells):expr=expr.T.tocsr()
    cpal=cluster_palette()

    umap_df=pd.read_csv(datadir/'brain_development_umap_coordinates.csv')
    region_by_cell=dict(zip(umap_df.cell_id.astype(str),umap_df.boundary_region_id.astype(int)))
    regions_df=pd.read_csv(datadir/'brain_development_cluster_boundary_regions.csv')
    region_lookup={(int(r.cluster_a),int(r.cluster_b)):int(r.region) for _,r in regions_df.iterrows()}

    for case in CASES:
        a,b=case['clusters'];gene=case['gene'];stage=case['stage']
        keep=samples==stage
        xy=xy_all[keep];clusters=clusters_all[keep]
        sexpr=expr[:,gi[gene]].toarray().ravel()[keep]

        # transition-region cells at this stage define the ROI centre
        region_id=region_lookup.get((a,b),-1)
        stage_cell_ids=cells[keep]
        stage_boundary=np.array([region_by_cell.get(cid,-1) for cid in stage_cell_ids])
        trans_stage=stage_boundary==region_id
        z=xy[trans_stage]
        if len(z)==0:
            z=xy[np.isin(clusters,(a,b))&(ann[keep]=='Brain')]
        tree=cKDTree(z)
        dens=np.array([len(v) for v in tree.query_ball_point(z,400)])
        center=z[dens.argmax()]+np.array(case.get('shift',(0.0,0.0)))
        half=case.get('roi_half',ROI_HALF)
        bounds=(center[0]-half,center[0]+half,
                center[1]-half,center[1]+half)

        # only draw cells inside the brain ROI
        inwin=(xy[:,0]>=bounds[0])&(xy[:,0]<=bounds[1])&(xy[:,1]>=bounds[2])&(xy[:,1]<=bounds[3])
        xy_win=xy[inwin];clusters_win=clusters[inwin]
        brain_win=ann[keep][inwin]=='Brain'
        emphasize=np.isin(clusters_win,(a,b))&brain_win
        cell_win=cells[keep][inwin]
        boundary_win=np.array([region_by_cell.get(cid,-1) for cid in cell_win])
        transition=emphasize&(boundary_win==region_id)

        mx,my,f=gridfield(xy,sexpr,bounds)
        # visualization-only Gaussian smoothing (mesoscale view; TIS/model untouched)
        f_smooth=gaussian_filter(f,sigma=SMOOTH_SIGMA)
        pos_f=f_smooth[f_smooth>0]
        levels=[float(np.quantile(pos_f,q)) for q,_,_ in CONTOUR_LEVELS] if len(pos_f) else []

        fig,ax=plt.subplots(figsize=mm_to_inches(120,120))
        fig.subplots_adjust(left=.005,right=.995,bottom=.17,top=.94)

        for c in sorted(set(clusters_win)):
            m=(clusters_win==c)&(~emphasize)
            if m.any():
                ax.scatter(xy_win[m,0],xy_win[m,1],s=case['point_size'],c=BACKGROUND_COLOR,
                           alpha=FADE_ALPHA,linewidths=0,rasterized=True,zorder=1)
        for c in (a,b):
            m=(clusters_win==c)&emphasize&(~transition)
            if m.any():
                ax.scatter(xy_win[m,0],xy_win[m,1],s=case['point_size'],c=[cpal[c]],
                           alpha=EMPH_ALPHA,linewidths=0,rasterized=True,zorder=2)
        for c in (a,b):
            m=(clusters_win==c)&transition
            if m.any():
                ax.scatter(xy_win[m,0],xy_win[m,1],s=case['point_size'],c=[cpal[c]],
                           alpha=TRANS_ALPHA,linewidths=0,rasterized=True,zorder=3)

        # contours with per-level color/width; drop small island components
        if levels:
            cs=ax.contour(mx,my,f_smooth,levels=levels)
            window_area=(bounds[1]-bounds[0])*(bounds[3]-bounds[2])
            min_area=MIN_AREA_FRAC*window_area
            info=[]
            for i,style in enumerate(CONTOUR_LEVELS):
                color,lw=style[1],style[2]
                kept=[]
                areas=[]
                for seg in cs.allsegs[i]:
                    if len(seg)<4:continue
                    xs=seg[:,0];ys=seg[:,1]
                    area=0.5*abs(float(np.sum(xs[:-1]*ys[1:]-xs[1:]*ys[:-1])))
                    areas.append(area)
                    if area>=min_area:
                        kept.append((area,seg))
                kept.sort(key=lambda x:-x[0])
                kept=[seg for _,seg in kept[:MAX_COMPONENTS]]
                if kept:
                    ax.add_collection(LineCollection(kept,colors=color,linewidths=lw,
                                                      alpha=.85,zorder=4))
                info.append(f'P{int(style[0]*100)}:{len(kept)}/{len(areas)}')
            cs.remove()
            print(f'  contours: {", ".join(info)}')

        ax.set_xlim(bounds[0],bounds[1]);ax.set_ylim(bounds[2],bounds[3])
        ax.invert_yaxis()
        ax.set_aspect('equal');ax.set_axis_off()
        ax.set_title(f'{case["transition"]} \u00b7 {gene} \u00b7 {stage}',
                     loc='left',fontweight='bold',fontsize=14)
        other_handle=Line2D([0],[0],marker='o',linestyle='none',markersize=4.3,
                   markerfacecolor=to_rgba(BACKGROUND_COLOR,FADE_ALPHA),
                   markeredgewidth=0,label='Other clusters')
        a_core=Line2D([0],[0],marker='o',linestyle='none',markersize=4.3,
                   markerfacecolor=to_rgba(cpal[a],EMPH_ALPHA),
                   markeredgewidth=0,label=f'C{a} non-transition')
        b_core=Line2D([0],[0],marker='o',linestyle='none',markersize=4.3,
                   markerfacecolor=to_rgba(cpal[b],EMPH_ALPHA),
                   markeredgewidth=0,label=f'C{b} non-transition')
        a_transition=Line2D([0],[0],marker='o',linestyle='none',markersize=4.3,
                   markerfacecolor=cpal[a],markeredgewidth=0,label=f'C{a} transition')
        b_transition=Line2D([0],[0],marker='o',linestyle='none',markersize=4.3,
                   markerfacecolor=cpal[b],markeredgewidth=0,label=f'C{b} transition')
        field_handles={
            int(q*100):Line2D([0],[0],color=color,lw=linewidth,label=f'Field P{int(q*100)}')
            for q,color,linewidth in CONTOUR_LEVELS
        }
        # Figure legends fill columns first. This order produces two logical rows.
        legend_handles=[other_handle,a_transition,a_core,b_transition,
                        b_core,field_handles[70],field_handles[40],field_handles[95]]
        fig.legend(handles=legend_handles,loc='lower center',bbox_to_anchor=(.5,.018),
                   ncol=4,frameon=False,fontsize=5.5,columnspacing=1.1,
                   handlelength=1.4,handletextpad=.4,labelspacing=.55)

        stem=f'brain_field_C{a}_C{b}_{gene}_{stage}'
        out=figdir/f'{stem}.png'
        pdf_out=pdfdir/f'{stem}.pdf'
        save_figure(fig,out,dpi=PNG_DPI,close=False)
        save_figure(fig,pdf_out,dpi=PNG_DPI)
        print(out)
        print(pdf_out,f'levels={len(levels)} roi_center=({center[0]:.0f},{center[1]:.0f}) trans_cells={int(trans_stage.sum())}')

if __name__=='__main__':main()
