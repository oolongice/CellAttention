#!/usr/bin/env python3
"""Wnt3a regulatory influence (C7<->C8): UMAP + two minimal quantitative panels.

Left: brain UMAP with non-relevant cells in light grey, C7<->C8 transition cells
highlighted, and the continuous manifold-constrained Wnt3a regulatory influence
field (receiver-specific beta x field, blue family, light-blue peak).

Right (two stacked violin panels):
  top    - bootstrap distribution of log10 median positive influence;
  bottom - bootstrap distribution of the influenced-cell fraction.
Both panels retain the observed estimate and bootstrap 95% interval.

Definitions:
  influence(i) = beta(cluster_i) * field_i(Wnt3a)
  beta(cluster) = sum over the 7 C7/C8 common Wnt3a targets of max(beta, 0)
  field_i = sum_j k0(d_ij/50)/(2*pi*50^2) * expr_j(Wnt3a), receiver hard-exclusion
  tau = 0  (any positive model-inferred influence, i.e. within the 150-um radius)
"""
OUTPUT_FORMAT='png';PNG_DPI=600
LENGTH=50.0;CUTOFF=150.0;MIN_DIST=2.0
POINT_SIZE=0.1
TRANS_COLOR="#F18AAC"
BANDWIDTH=0.3
KDE_GRID_SIZE=600
MASK_FACTOR=2.5
ALPHA_LOW=0.05
TAU=0.0
N_BOOT=2000

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from scipy.io import mmread
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde,kruskal,chi2_contingency
from mosta_common import HERE,DATASET,PRE,RESULTS,configure_style,cluster_palette,lines
from vis.spatial import reaction_diffusion_field
from vis.style import mm_to_inches,save_figure

def bootstrap_ci(x,stat_fn,n=N_BOOT,seed=2026):
    rng=np.random.default_rng(seed)
    s=np.empty(n)
    for i in range(n):
        idx=rng.integers(0,len(x),len(x))
        s[i]=stat_fn(x[idx])
    return float(np.percentile(s,2.5)),float(np.percentile(s,97.5))

def main():
    configure_style()
    figdir=HERE/'figures/brain_development';datadir=HERE/'data'
    pdfdir=figdir
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)
    pdfdir.mkdir(parents=True,exist_ok=True)

    u=pd.read_csv(datadir/'brain_development_umap_coordinates.csv')
    umap=u[['umap1','umap2']].to_numpy(float)
    clusters=u.cluster.to_numpy(int)
    region=u.boundary_region_id.to_numpy(int)

    rel=pd.read_csv(DATASET/'analysis/all_selected_cluster_source_target_relations.csv')
    rel=rel[rel.selected.astype(str).str.lower().eq('true')]
    w7=rel[(rel.cluster==7)&(rel.source_gene=='Wnt3a')]
    w8=rel[(rel.cluster==8)&(rel.source_gene=='Wnt3a')]
    common=set(w7.target_gene)&set(w8.target_gene)
    beta7=float(w7[w7.target_gene.isin(common)].signed_beta.clip(lower=0).sum())
    beta8=float(w8[w8.target_gene.isin(common)].signed_beta.clip(lower=0).sum())

    cells=lines(PRE/'cell_ids.txt')
    gene_ids=lines(DATASET/'model/transformer/gene_ids.txt');gi={g:i for i,g in enumerate(gene_ids)}
    ann=lines(PRE/'evaluation_cell_groups.txt')
    xy_all=np.loadtxt(PRE/'spatial_coordinates.csv',delimiter=',')
    expr=mmread(PRE/'expression.mtx').tocsr()
    if expr.shape[0]!=len(cells):expr=expr.T.tocsr()

    brain_mask=ann=='Brain'
    brain_xy=xy_all[brain_mask]
    wnt3a=expr[:,gi['Wnt3a']].toarray().ravel()
    field,_=reaction_diffusion_field(xy_all,wnt3a,query=brain_xy,
                                     length_scale=LENGTH,maximum_distance=CUTOFF,
                                     minimum_distance=MIN_DIST)
    field[wnt3a[brain_mask]>0]=0.0
    beta_sum=np.where(clusters==7,beta7,np.where(clusters==8,beta8,0.0))
    influence=field*beta_sum

    cpal=cluster_palette()
    tr=region==1
    g7=influence[(clusters==7)&(region==-1)]
    gT=influence[tr]
    g8=influence[(clusters==8)&(region==-1)]

    groups=[g7,gT,g8]
    positions=np.arange(3)
    labels=['C7','Trans.','C8']
    colors=[cpal[7],TRANS_COLOR,cpal[8]]

    # ---- figure 1: standalone square UMAP ----
    pos=influence>0
    kde=gaussian_kde(umap[pos].T,weights=influence[pos],bw_method=BANDWIDTH)
    xmin,xmax=umap[:,0].min(),umap[:,0].max()
    ymin,ymax=umap[:,1].min(),umap[:,1].max()
    gx,gy=np.meshgrid(
        np.linspace(xmin,xmax,KDE_GRID_SIZE),
        np.linspace(ymin,ymax,KDE_GRID_SIZE),
    )
    gridpts=np.vstack([gx.ravel(),gy.ravel()])
    z=kde(gridpts).reshape(gx.shape)
    tree=cKDTree(umap)
    nn,_=tree.query(umap,k=2)
    nn_med=float(np.median(nn[:,1]))
    dgrid,_=tree.query(gridpts.T,k=1)
    z[dgrid.reshape(gx.shape)>MASK_FACTOR*nn_med]=0.0

    cmap=LinearSegmentedColormap.from_list('inf_blue',['#08306B','#094E8B','#347EA9','#5DABD5'])
    zmax=float(np.quantile(z[z>0],0.99)) if (z>0).any() else 1.0
    zn=np.clip(z,0,zmax)/zmax
    rgba=cmap(zn)
    rgba[...,3]=np.clip((zn-ALPHA_LOW)/(1.0-ALPHA_LOW),0,1)*0.95

    fig_map,ax_map=plt.subplots(figsize=(2,2))
    fig_map.subplots_adjust(left=.06,right=.96,bottom=.05,top=.88)
    ax_map.imshow(
        rgba, extent=[xmin,xmax,ymin,ymax], origin='lower', aspect='auto',
        interpolation='bicubic', resample=True, zorder=1,
    )
    ax_map.scatter(umap[:,0],umap[:,1],s=POINT_SIZE,c='#D0D0D0',alpha=.62,
                   linewidths=0,rasterized=True,zorder=2)
    ax_map.scatter(umap[tr,0],umap[tr,1],s=.1,c=TRANS_COLOR,alpha=1.0,
                   linewidths=0,rasterized=True,zorder=3)
    ax_map.set_aspect('equal');ax_map.set_axis_off()
    ax_map.set_title('Wnt3a influence',loc='left',fontweight='bold',fontsize=7)
    map_out=figdir/'brain_C7_C8_Wnt3a_influence_umap.png'
    fig_map.savefig(map_out,dpi=PNG_DPI,facecolor='white',pil_kwargs={'compress_level':6})
    map_pdf=pdfdir/'brain_C7_C8_Wnt3a_influence_umap.pdf'
    fig_map.savefig(map_pdf,dpi=PNG_DPI,facecolor='white')
    fig_map.savefig(figdir/'brain_C7_C8_Wnt3a_influence_umap.svg',dpi=300,facecolor='white')
    plt.close(fig_map)

    # Robust statistics used by the two quantitative panels.
    def median_pos(values):
        positive=values[values>TAU]
        return float(np.median(positive)) if len(positive) else 0.0
    def frac_pos(values):
        return float((values>TAU).mean())
    def bootstrap_samples(values,stat_fn,seed):
        rng=np.random.default_rng(seed)
        samples=np.empty(N_BOOT)
        for iteration in range(N_BOOT):
            index=rng.integers(0,len(values),len(values))
            samples[iteration]=stat_fn(values[index])
        return samples

    positive=[group[group>TAU] for group in groups]
    pts=np.array([np.log10(np.median(group)) for group in positive])
    strength_boot=[
        np.log10(bootstrap_samples(group,np.median,2026+i))
        for i,group in enumerate(positive)
    ]
    lo=np.array([np.percentile(values,2.5) for values in strength_boot])
    hi=np.array([np.percentile(values,97.5) for values in strength_boot])

    fr=np.array([frac_pos(group) for group in groups])
    fraction_boot=[
        bootstrap_samples(group,frac_pos,3026+i)
        for i,group in enumerate(groups)
    ]
    flo=np.array([np.percentile(values,2.5) for values in fraction_boot])
    fhi=np.array([np.percentile(values,97.5) for values in fraction_boot])

    H,p_s=kruskal(*positive)
    tab=[[int((g7>TAU).sum()),int((g7<=TAU).sum())],
         [int((gT>TAU).sum()),int((gT<=TAU).sum())],
         [int((g8>TAU).sum()),int((g8<=TAU).sum())]]
    chi2,p_f,_,_=chi2_contingency(tab)

    # ---- figure 2: square quantitative summary, two equal-height rows ----
    fig_quant,axes=plt.subplots(2,1,figsize=(2,2),gridspec_kw={'height_ratios':[1,1]})
    fig_quant.subplots_adjust(left=.31,right=.96,bottom=.12,top=.93,hspace=.95)
    ax_strength,ax_fraction=axes

    violin=ax_strength.violinplot(
        strength_boot,positions=positions,widths=.72,vert=False,
        showmeans=False,showmedians=False,showextrema=False,points=160,
    )
    for body,color in zip(violin['bodies'],colors):
        body.set_facecolor(color);body.set_edgecolor('#555555')
        body.set_linewidth(.45);body.set_alpha(.78)
    for position,lower,upper,value in zip(positions,lo,hi,pts):
        ax_strength.plot([lower,upper],[position,position],color='#333333',lw=.75,zorder=4)
        ax_strength.plot(value,position,'o',markersize=2.4,markerfacecolor='white',
                         markeredgecolor='#222222',markeredgewidth=.55,zorder=5)
    ax_strength.set_yticks(positions,labels,fontsize=4.5)
    ax_strength.invert_yaxis()
    ax_strength.set_xlabel(r'log$_{10}$ median positive influence',fontsize=4.5,labelpad=1)
    ax_strength.set_title('Regulatory influence strength',loc='left',fontweight='bold',fontsize=5)
    ax_strength.tick_params(axis='x',labelsize=3.5)
    ax_strength.grid(axis='x',color='#E8E8E8',linewidth=.3);ax_strength.set_axisbelow(True)
    ax_strength.spines[['top','right']].set_visible(False)
    exp_s=int(np.floor(-np.log10(p_s)))
    ax_strength.text(.98,.97,rf'$P < 10^{{-{exp_s}}}$',transform=ax_strength.transAxes,
                     ha='right',va='top',fontsize=3,color='#777777')

    lower_error=fr-flo;upper_error=fhi-fr
    ax_fraction.barh(positions,fr,height=.58,color=colors,alpha=.82,
                     edgecolor='#555555',linewidth=.45,zorder=2)
    ax_fraction.errorbar(fr,positions,xerr=np.vstack([lower_error,upper_error]),fmt='none',
                         ecolor='#333333',elinewidth=.75,capsize=1.8,capthick=.75,zorder=4)
    label_x=max(fhi)*1.30
    for position,group in zip(positions,groups):
        positive_count=int((group>TAU).sum())
        ax_fraction.text(label_x,position,f'{positive_count:,}/{len(group):,}',
                         va='center',ha='right',fontsize=3.2,color='#444444')
    ax_fraction.set_yticks(positions,labels,fontsize=4.5)
    ax_fraction.invert_yaxis()
    ax_fraction.set_xlabel('Influenced cells (%)',fontsize=4.5,labelpad=2)
    ax_fraction.xaxis.set_major_formatter(lambda value,_: f'{100*value:.0f}%')
    ax_fraction.tick_params(axis='x',labelsize=3.5)
    ax_fraction.set_xlim(0,max(fhi)*1.34)
    ax_fraction.set_title('Fraction influenced',loc='left',fontweight='bold',fontsize=5,pad=6)
    ax_fraction.grid(axis='x',color='#E8E8E8',linewidth=.3);ax_fraction.set_axisbelow(True)
    ax_fraction.spines[['top','right']].set_visible(False)
    exp_f=int(np.floor(-np.log10(p_f)))
    ax_fraction.text(.98,.97,rf'$P < 10^{{-{exp_f}}}$',transform=ax_fraction.transAxes,
                     ha='right',va='top',fontsize=3,color='#777777')

    quant_out=figdir/'brain_C7_C8_Wnt3a_influence_quantification.png'
    fig_quant.savefig(quant_out,dpi=PNG_DPI,facecolor='white',pil_kwargs={'compress_level':6})
    quant_pdf=pdfdir/'brain_C7_C8_Wnt3a_influence_quantification.pdf'
    fig_quant.savefig(quant_pdf,dpi=PNG_DPI,facecolor='white')
    plt.close(fig_quant)
    print(map_out)
    print(map_pdf)
    print(quant_out)
    print(quant_pdf)
    print('--- figure legend ---')
    print('influence(i) = beta(cluster_i) * field_i(Wnt3a);')
    print('beta(cluster) = sum of max(beta,0) over 7 C7/C8 common targets (C7=%.3f, C8=%.3f)'%(beta7,beta8))
    print('field_i = sum_j k0(d_ij/50)/(2*pi*50^2) * expr_j(Wnt3a), receiver hard-exclusion')
    print('tau = 0 (any positive influence within the 150-um kernel radius)')
    print(f'groups n: C7 core={len(g7)}, Transition={len(gT)}, C8 core={len(g8)}')
    for name,p,l,h in zip(['C7 core','Transition','C8 core'],pts,lo,hi):
        print(f'  {name}: strength log10(median)={p:.3f} (95% CI {l:.3f}..{h:.3f})')
    for name,f,l,h in zip(['C7 core','Transition','C8 core'],fr,flo,fhi):
        print(f'  {name}: fraction={f:.4f} (95% CI {l:.4f}..{h:.4f})')
    print(f'strength KW P={p_s:.3e}; fraction chi2 P={p_f:.3e}')

if __name__=='__main__':main()
