#!/usr/bin/env python3
"""Transition-specific Influence Score (TIS) analysis for brain cluster transitions.

Core question: which model-inferred source-gene -> target-gene interactions are
selectively strengthened at the interface between two brain clusters?

For each transition A <-> B the cells are split into three populations:
  A_core - cells clearly in cluster A, away from the A-B boundary;
  B_core - cells clearly in cluster B, away from the A-B boundary;
  T      - mixed/transition cells at the A-B boundary (excluded from A/B core).

For every source->target pair shared by A and B, the model influence
I(i) = beta(cluster(i)) * source_field(i) is computed, x_i = |I(i)|, and

    TIS = ( mu_T - max(mu_A, mu_B) ) / ( sigma_pooled + eps )

where mu_* is the mean of |I| over the population and sigma_pooled is the pooled
standard deviation of |I| over A_core U B_core. The signed mean influence in T is
kept separately for direction interpretation. Pairs are fully ranked by TIS.
"""
OUTPUT_FORMAT='png';PNG_DPI=600
LENGTH_SCALE=50.0;MAX_DIST=150.0;MIN_DIST=2.0
EPS=1e-8
TOP_RANK=20
TOP_PROFILE=5
POINT_SIZE=2.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.io import mmread
from mosta_common import HERE,DATASET,PRE,RESULTS,configure_style,cluster_palette,lines
from vis.spatial import reaction_diffusion_field
from vis.style import mm_to_inches,save_figure

def pooled_std(a, b):
    na, nb = len(a), len(b)
    if na + nb <= 2:
        return 0.0
    va = float(np.var(a)) if na > 1 else 0.0
    vb = float(np.var(b)) if nb > 1 else 0.0
    return float(np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)))

def main():
    configure_style()
    figdir=HERE/'figures/brain_development';datadir=HERE/'data'
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)

    regions=pd.read_csv(datadir/'brain_development_cluster_boundary_regions.csv')
    rel=pd.read_csv(DATASET/'analysis/all_selected_cluster_source_target_relations.csv')
    rel=rel[rel.selected.astype(str).str.lower().eq('true')]

    df=pd.read_csv(datadir/'brain_development_umap_coordinates.csv')

    cells=lines(PRE/'cell_ids.txt')
    ann=lines(PRE/'evaluation_cell_groups.txt')
    gene_ids=lines(DATASET/'model/transformer/gene_ids.txt')
    gi={g:i for i,g in enumerate(gene_ids)}
    all_xy=np.loadtxt(PRE/'spatial_coordinates.csv',delimiter=',')
    clusters=pd.read_csv(RESULTS/'receiver_group_assignments.csv')['receiver_group'].to_numpy(int)+1
    expr=mmread(PRE/'expression.mtx').tocsr()
    if expr.shape[0]!=len(cells):expr=expr.T.tocsr()

    brain_mask=ann=='Brain'
    brain_xy=all_xy[brain_mask]
    bc=clusters[brain_mask]
    stages=df.stage.to_numpy()
    umap1=df.umap1.to_numpy();umap2=df.umap2.to_numpy()
    region_id=df.boundary_region_id.to_numpy()

    # common pairs per transition + union of source genes
    pairs_by_region={}
    source_set=set()
    for _,r in regions.iterrows():
        a=int(r.cluster_a);b=int(r.cluster_b)
        pa=rel[rel.cluster==a];pb=rel[rel.cluster==b]
        common=pd.merge(pa,pb,on=['source_gene','target_gene'],suffixes=('_a','_b'))
        common=common.drop_duplicates(['source_gene','target_gene'])
        pairs_by_region[a,b]=common
        source_set.update(common.source_gene)

    # source fields once per unique source gene
    field_cache={}
    for s in source_set:
        sexpr=expr[:,gi[s]].toarray().ravel()
        field,_=reaction_diffusion_field(all_xy,sexpr,query=brain_xy,
                                         length_scale=LENGTH_SCALE,maximum_distance=MAX_DIST,
                                         minimum_distance=MIN_DIST)
        field[sexpr[brain_mask]>0]=0.0
        field_cache[s]=field

    ranking_rows=[]
    cell_rows=[]
    for _,r in regions.iterrows():
        a=int(r.cluster_a);b=int(r.cluster_b);reg=int(r.region)
        common=pairs_by_region[a,b]
        mask_T=region_id==reg
        mask_A_core=(bc==a)&(region_id==-1)
        mask_B_core=(bc==b)&(region_id==-1)
        mask_A_all=bc==a;mask_B_all=bc==b
        pop=np.full(len(bc),'other',object)
        pop[mask_A_core]='A_core';pop[mask_B_core]='B_core';pop[mask_T]='T'

        pair_scores=[]
        for _,row in common.iterrows():
            s=row.source_gene;t=row.target_gene
            beta_a=float(row.signed_beta_a);beta_b=float(row.signed_beta_b)
            field=field_cache[s]
            inf=np.zeros(len(bc))
            inf[mask_A_all]=beta_a*field[mask_A_all]
            inf[mask_B_all]=beta_b*field[mask_B_all]
            x=np.abs(inf)
            mu_T=float(np.mean(x[mask_T]))
            mu_A=float(np.mean(x[mask_A_core]))
            mu_B=float(np.mean(x[mask_B_core]))
            sigma=pooled_std(x[mask_A_core],x[mask_B_core])
            tis=(mu_T-max(mu_A,mu_B))/(sigma+EPS)
            signed_mean_T=float(np.mean(inf[mask_T]))
            pair_scores.append(dict(source_gene=s,target_gene=t,TIS=tis,
                                    mu_T=mu_T,mu_A=mu_A,mu_B=mu_B,pooled_sd=sigma,
                                    signed_mean_T=signed_mean_T,
                                    n_T=int(mask_T.sum()),n_A=int(mask_A_core.sum()),n_B=int(mask_B_core.sum()),
                                    signed_beta_a=beta_a,signed_beta_b=beta_b,
                                    derived_attention_a=float(row.derived_attention_a),
                                    derived_attention_b=float(row.derived_attention_b),
                                    target_rank_a=int(row.target_rank_a),target_rank_b=int(row.target_rank_b),
                                    training_improvement_a=float(row.training_improvement_a),
                                    training_improvement_b=float(row.training_improvement_b),
                                    screening_score_a=float(row.screening_score_a),
                                    screening_score_b=float(row.screening_score_b)))
        pair_scores.sort(key=lambda d:-d['TIS'])
        for rank,d in enumerate(pair_scores,1):
            d['rank']=rank;d['transition']=f'C{a}-C{b}';d['cluster_a']=a;d['cluster_b']=b
            ranking_rows.append(d)

        # cell-level influence for top pairs (three populations only)
        for d in pair_scores[:TOP_RANK]:
            s=d['source_gene'];t=d['target_gene']
            beta_a=d['signed_beta_a'];beta_b=d['signed_beta_b']
            field=field_cache[s]
            inf=np.zeros(len(bc))
            inf[mask_A_all]=beta_a*field[mask_A_all]
            inf[mask_B_all]=beta_b*field[mask_B_all]
            keep=mask_T|mask_A_core|mask_B_core
            idx=np.where(keep)[0]
            for i in idx:
                cell_rows.append(dict(transition=f'C{a}-C{b}',cluster_a=a,cluster_b=b,
                                      source_gene=s,target_gene=t,TIS=d['TIS'],
                                      cell_id=df.cell_id.iloc[i],stage=stages[i],cluster=int(bc[i]),
                                      population=pop[i],influence=float(inf[i]),
                                      abs_influence=float(np.abs(inf[i])),
                                      umap1=float(umap1[i]),umap2=float(umap2[i]),
                                      spatial_x=float(brain_xy[i,0]),spatial_y=float(brain_xy[i,1])))

    ranking=pd.DataFrame(ranking_rows)
    ranking.to_csv(datadir/'brain_transition_tis_ranking.csv',index=False)
    top20=ranking[ranking['rank']<=TOP_RANK].reset_index(drop=True)
    top20.to_csv(datadir/'brain_transition_tis_top20.csv',index=False)
    cell_df=pd.DataFrame(cell_rows)
    cell_df.to_csv(datadir/'brain_transition_tis_cell_influence.csv',index=False)

    cpal=cluster_palette()

    # ---- figure type 1: TIS ranking (top 20) ----
    for _,r in regions.iterrows():
        a=int(r.cluster_a);b=int(r.cluster_b)
        q=ranking[(ranking.cluster_a==a)&(ranking.cluster_b==b)].head(TOP_RANK)[::-1]
        fig,ax=plt.subplots(figsize=mm_to_inches(160,150))
        fig.subplots_adjust(left=.34,right=.96,bottom=.10,top=.94)
        y=np.arange(len(q))
        vals=q.TIS.to_numpy()
        colors=['#D98278' if v>=0 else '#76A9C4' for v in vals]
        ax.barh(y,vals,color=colors,edgecolor='white',linewidth=.4)
        ax.axvline(0,color='#888',lw=.7)
        ax.set_yticks(y,[f'{s} \u2192 {t}' for s,t in zip(q.source_gene,q.target_gene)],fontsize=8)
        ax.set_xlabel('TIS (transition-specific influence score)',fontsize=11)
        ax.set_title(f'C{a} \u2194 C{b}: transition-specific interactions (top {TOP_RANK})',
                     loc='left',fontweight='bold',fontsize=12)
        ax.spines[['top','right']].set_visible(False)
        save_figure(fig,figdir/f'brain_transition_C{a}_C{b}_tis_ranking.{OUTPUT_FORMAT}',dpi=PNG_DPI)
        print(figdir/f'brain_transition_C{a}_C{b}_tis_ranking.{OUTPUT_FORMAT}')

    # ---- figure type 2: A-core -> T -> B-core influence profile (top candidates) ----
    rng=np.random.default_rng(0)
    for _,r in regions.iterrows():
        a=int(r.cluster_a);b=int(r.cluster_b)
        q=ranking[(ranking.cluster_a==a)&(ranking.cluster_b==b)].head(TOP_PROFILE)
        mask_T=region_id==int(r.region)
        mask_A_core=(bc==a)&(region_id==-1)
        mask_B_core=(bc==b)&(region_id==-1)
        mask_A_all=bc==a;mask_B_all=bc==b
        fig,axes=plt.subplots(TOP_PROFILE,1,figsize=mm_to_inches(150,34*TOP_PROFILE))
        fig.subplots_adjust(left=.30,right=.97,bottom=.10,top=.88,hspace=.70)
        for j,(_,d) in enumerate(q.iterrows()):
            ax=axes[j]
            s=d.source_gene;t=d.target_gene
            beta_a=d.signed_beta_a;beta_b=d.signed_beta_b
            field=field_cache[s]
            inf=np.zeros(len(bc))
            inf[mask_A_all]=beta_a*field[mask_A_all]
            inf[mask_B_all]=beta_b*field[mask_B_all]
            data_A=np.abs(inf[mask_A_core])*1e6;data_T=np.abs(inf[mask_T])*1e6;data_B=np.abs(inf[mask_B_core])*1e6
            boxes=ax.boxplot([data_A,data_T,data_B],vert=False,patch_artist=True,widths=.6,
                             showfliers=False,
                             medianprops=dict(color='#202020',linewidth=.9),
                             boxprops=dict(facecolor='#E3ECF3',edgecolor='#4A6B8A',linewidth=.7),
                             whiskerprops=dict(color='#4A6B8A',linewidth=.7),
                             capprops=dict(color='#4A6B8A',linewidth=.7))
            boxes['boxes'][1].set_facecolor('#F2B8B5');boxes['boxes'][1].set_edgecolor('#B0413E')
            for k,data in enumerate((data_A,data_T,data_B)):
                if len(data)==0:continue
                sample=data if len(data)<=250 else rng.choice(data,250,replace=False)
                jit=rng.uniform(-.18,.18,len(sample))
                ax.scatter(sample,k+1+jit,s=3,c='#7F8C8D',alpha=.35,linewidths=0,rasterized=True)
            ax.set_yticks([1,2,3],[f'A core (n={len(data_A):,})',f'T (n={len(data_T):,})',f'B core (n={len(data_B):,})'],fontsize=9)
            ax.set_title(f'{s} \u2192 {t}    TIS={d.TIS:.2f}',loc='left',fontweight='bold',fontsize=11)
            ax.spines[['top','right']].set_visible(False)
        #axes[-1].set_xlabel('|influence| = |\u03b2 \u00d7 source field| (\u00d710\u207b\u2076)',fontsize=11)
        fig.suptitle(f'C{a} \u2194 C{b}: transition influence profiles',
                     x=.30,y=.985,ha='left',fontweight='bold',fontsize=12)
        save_figure(fig,figdir/f'brain_transition_C{a}_C{b}_tis_profiles.{OUTPUT_FORMAT}',dpi=PNG_DPI)
        print(figdir/f'brain_transition_C{a}_C{b}_tis_profiles.{OUTPUT_FORMAT}')

    print('ranking pairs:',len(ranking),'cell rows:',len(cell_df))

if __name__=='__main__':main()
