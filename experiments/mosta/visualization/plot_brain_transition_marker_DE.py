#!/usr/bin/env python3
"""Marker / DE analysis of C7-core vs C7<->C8 transition vs C8-core cells.

Independent of the source-target model (no Wnt3a or model predictions are used),
so there is no circular reasoning. For every gene the group means, expression
fractions and Mann-Whitney U p-values are computed; genes are classified into
C7-high (monotonic down), C8-high (monotonic up) and transition-high (peak)
trends. A compact heatmap shows ~5 representative markers per trend (row-wise
z-scored group means), and the full table is saved for biological annotation.
"""
OUTPUT_FORMAT='png';PNG_DPI=600
TOP_PER_TREND=5
MIN_FRAC=0.01
MIN_MEAN=0.05

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.stats import mannwhitneyu
from mosta_common import HERE,DATASET,PRE,configure_style,lines
from vis.style import mm_to_inches,save_figure

def main():
    configure_style()
    figdir=HERE/'figures/brain_development';datadir=HERE/'data'
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)

    u=pd.read_csv(datadir/'brain_development_umap_coordinates.csv')
    gene_ids=lines(DATASET/'model/transformer/gene_ids.txt')
    cell_ids=lines(PRE/'cell_ids.txt')
    expr=mmread(PRE/'expression.mtx').tocsr()
    if expr.shape[0]!=len(cell_ids):expr=expr.T.tocsr()
    idx={cid:i for i,cid in enumerate(cell_ids)}

    c7_core=u[(u.cluster==7)&(u.boundary_region_id==-1)].cell_id.astype(str).tolist()
    trans=u[u.boundary_region_id==1].cell_id.astype(str).tolist()
    c8_core=u[(u.cluster==8)&(u.boundary_region_id==-1)].cell_id.astype(str).tolist()

    E7=expr[np.array([idx[c] for c in c7_core])].toarray()
    ET=expr[np.array([idx[c] for c in trans])].toarray()
    E8=expr[np.array([idx[c] for c in c8_core])].toarray()

    mu7=E7.mean(axis=0);muT=ET.mean(axis=0);mu8=E8.mean(axis=0)
    fr7=(E7>0).mean(axis=0);frT=(ET>0).mean(axis=0);fr8=(E8>0).mean(axis=0)
    n7=len(c7_core);nT=len(trans);n8=len(c8_core)

    rows=[]
    for g,gene in enumerate(gene_ids):
        x7=E7[:,g];xT=ET[:,g];x8=E8[:,g]
        m7,mT,m8=mu7[g],muT[g],mu8[g]
        f7,fT,f8=fr7[g],frT[g],fr8[g]
        p_c7c8=mannwhitneyu(x7,x8,alternative='two-sided',method='asymptotic').pvalue
        p_Tcores=mannwhitneyu(xT,np.concatenate([x7,x8]),alternative='two-sided',method='asymptotic').pvalue
        if m7>mT>m8:
            trend='C7-high';score=np.log2((m7+0.05)/(m8+0.05))
        elif m7<mT<m8:
            trend='C8-high';score=np.log2((m8+0.05)/(m7+0.05))
        elif mT>m7 and mT>m8:
            trend='transition-high';score=np.log2((mT+0.05)/(max(m7,m8)+0.05))
        else:
            trend='none';score=0.0
        if max(f7,fT,f8)<MIN_FRAC or max(m7,mT,m8)<MIN_MEAN:
            trend='none';score=0.0
        rows.append(dict(gene=gene,trend=trend,score=score,
                         mean_C7=m7,mean_T=mT,mean_C8=m8,
                         frac_C7=f7,frac_T=fT,frac_C8=f8,
                         p_C7_vs_C8=p_c7c8,p_T_vs_cores=p_Tcores))

    res=pd.DataFrame(rows)
    res.to_csv(datadir/'brain_C7_C8_marker_DE.csv',index=False)

    # representative markers per trend
    sel=[]
    for trend in ['C7-high','transition-high','C8-high']:
        sub=res[res.trend==trend].sort_values('score',ascending=False).head(TOP_PER_TREND)
        sel.append(sub)
    sel=pd.concat(sel)

    # heatmap matrix (group means, row z-score)
    M=sel[['mean_C7','mean_T','mean_C8']].to_numpy(float)
    Z=(M-M.mean(axis=1,keepdims=True))/(M.std(axis=1,keepdims=True)+1e-9)

    fig,ax=plt.subplots(figsize=mm_to_inches(52,62))
    fig.subplots_adjust(left=.40,right=.98,bottom=.26,top=.97)
    vmax=max(1.0,float(np.abs(Z).max()))
    im=ax.imshow(Z,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
    ax.set_xticks([0,1,2],['C7 core','Transition','C8 core'],fontsize=8)
    ax.set_yticks(range(len(sel)),sel.gene.tolist(),fontsize=8)
    ax.tick_params(length=0)
    ax.xaxis.set_ticks_position('bottom')
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
    # trend separators
    order=sel.trend.tolist()
    for i in range(1,len(order)):
        if order[i]!=order[i-1]:
            ax.axhline(i-0.5,color='#202020',linewidth=.8)
    cb=fig.colorbar(im,ax=ax,fraction=.04,pad=.03)
    cb.set_label('row z-score (mean expr)',fontsize=8);cb.ax.tick_params(labelsize=7)
    save_figure(fig,figdir/f'brain_C7_C8_marker_heatmap.{OUTPUT_FORMAT}',dpi=PNG_DPI)
    print(figdir/f'brain_C7_C8_marker_heatmap.{OUTPUT_FORMAT}')

    print(f'groups: C7 core={n7} transition={nT} C8 core={n8}')
    for trend in ['C7-high','transition-high','C8-high']:
        sub=res[res.trend==trend].sort_values('score',ascending=False).head(TOP_PER_TREND)
        print(f'\n=== {trend} (top {TOP_PER_TREND}) ===')
        print(sub[['gene','mean_C7','mean_T','mean_C8','frac_C7','frac_T','frac_C8','p_C7_vs_C8']].to_string(index=False))

if __name__=='__main__':main()
