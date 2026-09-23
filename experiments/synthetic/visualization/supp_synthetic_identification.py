#!/usr/bin/env python3
"""Per-case supplemental designs and model recovery plots; see README.md."""
from pathlib import Path
import sys
import os
import json
import re
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from plot_synthetic_data import (VARIANTS, CASE_TITLES, GROUP_COLORS,
                                 configure_arial, load_dataset, plot_groups, plot_network)

# User settings: fixed first training seed, never selected using recovery performance.
SEED = 2001
PNG_DPI = 600
TOP_TARGETS = 8
RING_TARGETS = 3
SOURCES_PER_TARGET = 6
ROOT = Path(os.environ["CELLATTENTION_SYNTHETIC_WORKDIR"])
OUTPUT = ROOT / 'visualization' / 'supp_figures'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'analysis'))
import benchmark_core as core


def group_name(g):
    return f'Receiver {g}' if g < 6 else f'Sender {g-6}'


def save(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=PNG_DPI, facecolor='white')
    plt.close(fig)
    print(out / f'{name}.png', flush=True)


def compute(variant, raw, xy, groups, genes, truth):
    model = ROOT / 'model/multiseed' / variant / f'seed_{SEED}'
    observed = np.loadtxt(model / 'standardized_expression.csv', delimiter=',')
    reconstruction = np.loadtxt(model / 'reconstruction.csv', delimiter=',')
    embedding = np.loadtxt(model / 'cell_embeddings.csv', delimiter=',')
    assert observed.shape == reconstruction.shape == raw.shape
    assert len(embedding) == len(groups)
    labels = core.cluster(core.whiten(embedding), SEED)
    assert len(np.unique(labels)) == core.N_CONTEXTS
    field, _ = core.source_fields(xy, raw)
    target, source = core.scores(labels, observed-reconstruction, core.standardize(field), 'transformer')
    assert np.isfinite(target).all() and np.isfinite(source).all()
    composition = pd.crosstab(pd.Series(labels, name='cluster'), pd.Series(groups, name='true_group'))
    composition = composition.div(composition.sum(axis=1), axis=0)
    dominant = composition.idxmax(axis=1)
    planted_targets = set(zip(truth.receiver_group, truth.target_gene))
    planted_edges = set(zip(truth.receiver_group, truth.source_gene, truth.target_gene))
    origins = np.argmax(np.stack([raw[groups == g].mean(axis=0) for g in range(12)]), axis=0)
    target_rows, source_rows = [], []
    for c in range(core.N_CONTEXTS):
        order = np.argsort(-target[c], kind='stable')
        for rank, t in enumerate(order, 1):
            target_rows.append(dict(cluster=c, target_gene=genes[t], target_rank=rank,
                                    target_score=target[c,t], dominant_group=int(dominant[c]),
                                    is_planted=(int(dominant[c]), genes[t]) in planted_targets))
            source_order = [s for s in np.argsort(-source[c,:,t], kind='stable') if s != t]
            for srank, s in enumerate(source_order, 1):
                source_rows.append(dict(cluster=c, target_gene=genes[t], target_rank=rank,
                    source_gene=genes[s], source_rank=srank, source_score=source[c,s,t],
                    source_origin_group=int(origins[s]), dominant_group=int(dominant[c]),
                    is_planted=(int(dominant[c]), genes[s], genes[t]) in planted_edges))
    return labels, composition, pd.DataFrame(target_rows), pd.DataFrame(source_rows)


def landscape(scores, comp, out, title):
    shown = scores[scores.target_rank <= TOP_TARGETS].copy()
    # Include true targets even when absent from every top-eight list; hollow squares expose misses.
    # Natural name ordering groups gene families and sorts every numeric component.
    genes = sorted(set(shown.target_gene) | set(scores.loc[scores.is_planted, 'target_gene']),
                   key=lambda gene: tuple(int(part) if part.isdigit() else part
                                          for part in re.split(r'(\d+)', gene)))
    display_genes = [gene.replace('marker_group','M').replace('low_noise_','N')
                     .replace('source_','S').replace('target_','T').replace('_','.')
                     for gene in genes]
    pd.DataFrame({'x_position': range(len(genes)), 'target_gene': genes,
                  'display_label': display_genes}).to_csv(
        out/'data/landscape_gene_order.csv', index=False)
    xmap = {g:i for i,g in enumerate(genes)}
    fig = plt.figure(figsize=(max(7.2, len(genes)*.13+1.6), 4.8))
    ax = fig.add_axes([.12,.25,.71,.53])
    maximum = float(shown.target_score.max())
    size = lambda v: 9+65*np.sqrt(np.asarray(v)/max(maximum,1e-12))
    for c in range(12):
        ax.axhspan(c-.48,c+.48,color='#F2F2F2' if c%2==0 else '#FAFAFA',zorder=0)
    dots = ax.scatter(shown.target_gene.map(xmap), shown.cluster, s=size(shown.target_score),
                      c=shown.target_rank,cmap='Blues_r',vmin=1,vmax=TOP_TARGETS,
                      edgecolor='white',linewidth=.5)
    planted = scores[scores.is_planted & scores.target_gene.isin(genes)]
    ax.scatter(planted.target_gene.map(xmap),planted.cluster,s=95,marker='s',
               facecolor='none',edgecolor='#222222',linewidth=.65)
    ax.set(xlim=(-.6,len(genes)-.4),ylim=(11.5,-.5))
    ax.set_xlabel('Candidate target gene (M: marker; N: noise; S: source; T: target)',fontsize=7.5,labelpad=5)
    ax.set_xticks(range(len(genes)),display_genes,rotation=75,ha='right',rotation_mode='anchor',fontsize=6.5)
    ax.set_yticks(range(12),[f'Cluster {c}' for c in range(12)],fontsize=6.5)
    ax.tick_params(length=0,pad=2); ax.spines[:].set_visible(False)
    for c in range(12):
        cy=.25+.53*(1-(c+.5)/12)
        pie=fig.add_axes([.855,cy-.016,.032,.032*fig.get_figwidth()/fig.get_figheight()])
        pie.pie(comp.loc[c],colors=[GROUP_COLORS[int(g)] for g in comp.columns],
                startangle=90,wedgeprops={'linewidth':.2,'edgecolor':'white'})
        pie.axis('equal');pie.axis('off')
    fig.text(.873,.80,'True-group\ncomposition',ha='center',fontsize=6.5)
    cb=fig.colorbar(dots,cax=fig.add_axes([.68,.86,.15,.018]),orientation='horizontal')
    cb.set_ticks([1,TOP_TARGETS]);cb.ax.set_title('Target rank',fontsize=6.5,pad=3)
    cb.ax.tick_params(labelsize=6,length=2);cb.outline.set_linewidth(.4)
    vals=np.array([1.0,5.0,10.0])
    handles=[ax.scatter([],[],s=size(v),color='#6BAED6',label=f'{v:g}') for v in vals]
    ax.legend(handles=handles,title='Target score',loc='lower left',
              bbox_to_anchor=(0,1.08),ncol=3,frameon=False,fontsize=6,title_fontsize=6.5)
    fig.suptitle(f'{title}: target identification',x=.12,y=.98,ha='left',fontsize=10.5,fontweight='bold')
    fig.text(.98,.975,f'Seed {SEED}',ha='right',va='top',fontsize=6.5,color='#555555')
    handles=[Line2D([],[],marker='s',ls='',color=GROUP_COLORS[g],label=group_name(g)) for g in range(12)]
    handles.append(Line2D([],[],marker='s',ls='',markerfacecolor='none',color='#222',label='Planted target for dominant group'))
    fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=6,frameon=False,bbox_to_anchor=(.5,.015))
    shown.to_csv(out/'data/landscape_display.csv',index=False)
    save(fig,out,'cluster_target_landscape')


def sectors(genes, inner, outer, ax, colors, label_radius, fontsize):
    positions={}
    width=2*np.pi/len(genes)
    for i,gene in enumerate(genes):
        mid=np.pi/2+(i+.5)*width
        gap=min(.055,width*.1)
        ax.add_patch(Wedge((0,0),outer,np.degrees(mid-width/2+gap),np.degrees(mid+width/2-gap),
                           width=outer-inner,facecolor=colors[gene],edgecolor='white',lw=.5,zorder=3))
        positions[gene]=mid
        degrees=np.degrees(mid)%360
        rotation=degrees if not 90<degrees<270 else degrees+180
        ha=('left' if not 90<degrees<270 else 'right') if label_radius>1 else 'center'
        ax.text(label_radius*np.cos(mid),label_radius*np.sin(mid),gene,rotation=rotation,
                rotation_mode='anchor',ha=ha,va='center',fontsize=fontsize,zorder=5)
    return positions


def rings(relations, comp, out, title):
    shown=relations[(relations.target_rank<=RING_TARGETS)&(relations.source_rank<=SOURCES_PER_TARGET)].copy()
    fig,axes=plt.subplots(4,3,figsize=(10.8,13))
    fig.subplots_adjust(left=.055,right=.95,top=.92,bottom=.10,wspace=.25,hspace=.35)
    maximum=max(float(shown.source_score.max()),1e-12)
    for c,ax in enumerate(axes.flat):
        rows=shown[shown.cluster==c]
        sources=sorted(rows.source_gene.unique());targets=list(rows.sort_values('target_rank').target_gene.drop_duplicates())
        origin=rows.set_index('source_gene').source_origin_group.to_dict()
        dominant=int(comp.loc[c].idxmax())
        sa=sectors(sources,.84,1,ax,{g:GROUP_COLORS[origin[g]] for g in sources},1.035,5)
        ta=sectors(targets,.22,.43,ax,{g:GROUP_COLORS[dominant] for g in targets},.32,5)
        for row in rows.sort_values('source_score').itertuples():
            start=.82*np.array([np.cos(sa[row.source_gene]),np.sin(sa[row.source_gene])])
            end=.45*np.array([np.cos(ta[row.target_gene]),np.sin(ta[row.target_gene])])
            ax.add_patch(FancyArrowPatch(start,end,connectionstyle='arc3,rad=.12',arrowstyle='-|>',
                mutation_scale=5,lw=.45+1.3*row.source_score/maximum,
                color=mpl.colormaps['Blues'](.25+.7*row.source_score/maximum),alpha=.85,zorder=2))
            if row.is_planted:
                mid=(start+end)/2
                ax.scatter(*mid,marker='*',s=23,c='#B2182B',edgecolor='white',lw=.3,zorder=6)
        ax.set(xlim=(-1.58,1.58),ylim=(-1.3,1.3),aspect='equal');ax.axis('off')
        ax.set_title(f'Cluster {c} | {group_name(dominant)} ({comp.loc[c,dominant]:.0%})',loc='left',fontsize=8,fontweight='bold')
    fig.suptitle(f'{title}: cluster-specific source–target identification',x=.055,y=.98,ha='left',fontsize=12,fontweight='bold')
    fig.text(.055,.945,f'Training seed {SEED} | top {RING_TARGETS} targets × top {SOURCES_PER_TARGET} sources | outer ring: sources; inner ring: targets',fontsize=8)
    handles=[Line2D([],[],marker='s',ls='',color=GROUP_COLORS[g],label=group_name(g)) for g in range(12)]
    handles.append(Line2D([],[],marker='*',ls='',color='#B2182B',label='Planted relation for dominant group'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.02),ncol=5,frameon=False,fontsize=6)
    cax=fig.add_axes([.74,.075,.16,.009])
    cb=mpl.colorbar.ColorbarBase(cax,cmap=mpl.colors.LinearSegmentedColormap.from_list('association',mpl.colormaps['Blues'](np.linspace(.25,.95,256))),norm=mpl.colors.Normalize(0,maximum),orientation='horizontal')
    cb.ax.tick_params(labelsize=5,length=2);cb.ax.set_title('Spatial association score (absolute)',fontsize=6)
    fig.text(.055,.079,'Source color: highest-mean-expression group; target color: dominant cluster group.',fontsize=6)
    shown.to_csv(out/'data/radial_display_relations.csv',index=False)
    save(fig,out,'cluster_source_target_relations')


def focused_source_recovery(relations, comp, out, title):
    """Conditional source recovery for planted targets in receiver-dominated clusters."""
    top_n = 5
    planted = relations[relations.is_planted]
    programs = planted[['cluster', 'target_gene']].drop_duplicates().sort_values(['cluster', 'target_gene'])
    if programs.empty:
        raise ValueError('No planted receiver programs available')
    rows_n = int(np.ceil(len(programs) / 3))
    fig, axes = plt.subplots(rows_n, 3, figsize=(10.8, rows_n*3.1+1.2), squeeze=False)
    fig.subplots_adjust(left=.055, right=.97, top=.88, bottom=.10, hspace=.60, wspace=.25)
    displayed, recovery = [], []
    score_max = max(float(relations[relations.source_rank <= top_n].source_score.max()), 1e-12)
    for ax, program in zip(axes.flat, programs.itertuples()):
        c, target = int(program.cluster), program.target_gene
        candidates = relations[(relations.cluster == c) & (relations.target_gene == target)].sort_values('source_rank')
        selected = candidates[candidates.source_rank <= top_n]
        positives = candidates[candidates.is_planted]
        displayed.append(selected)
        recovery.append(positives)
        dominant = int(comp.loc[c].idxmax())
        ax.scatter([0], [0], s=1800, marker='o', color='#FFF1D0', edgecolor='#E69F00', linewidth=1.2, zorder=3)
        ax.text(0, 0, target, ha='center', va='center', fontsize=9, fontweight='bold', zorder=4)
        for angle, row in zip(np.deg2rad(90 + np.arange(top_n)*360/top_n), selected.itertuples()):
            end = .31*np.array([np.cos(angle), np.sin(angle)])
            start = .88*np.array([np.cos(angle), np.sin(angle)])
            color = '#009E73' if row.is_planted else '#AAAAAA'
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=10,
                                        linewidth=.6+2.4*row.source_score/score_max, color=color, zorder=2))
            pos = 1.12*np.array([np.cos(angle), np.sin(angle)])
            ax.text(*pos, f'{row.source_gene}'+(' *' if row.is_planted else '')+f'\nRank {row.source_rank} | {row.source_score:.3f}',
                    ha='center', va='center', fontsize=8, color='#007350' if row.is_planted else '#666666',
                    bbox=dict(boxstyle='round,pad=.3', facecolor='#E5F5EF' if row.is_planted else '#F4F4F4', edgecolor='none'))
        recovered = int((positives.source_rank <= top_n).sum())
        truth_text = '\n'.join(f'{r.source_gene}: rank {r.source_rank}/{len(candidates)}, score {r.source_score:.3f}' for r in positives.itertuples())
        ax.text(.5, -.055, f'Planted source(s):\n{truth_text}\n{recovered}/{len(positives)} planted sources within displayed top {top_n}',
                transform=ax.transAxes, ha='center', va='top', fontsize=7)
        ax.set_title(f'Cluster {c} | Receiver {dominant} ({comp.loc[c,dominant]:.0%})',
                     fontsize=8, loc='left', fontweight='bold', pad=8)
        ax.set(xlim=(-1.65,1.65), ylim=(-1.3,1.5), aspect='equal'); ax.axis('off')
    for ax in list(axes.flat)[len(programs):]:
        ax.axis('off')
    fig.suptitle(f'{title}: source recovery for planted targets', x=.055, y=.975,
                 ha='left', fontsize=12, fontweight='bold')
    fig.text(.055,.935,f'Training seed {SEED} | top {top_n} source candidates per known target | all-gene source ranking',fontsize=8)
    omitted = [str(c) for c in comp.index if c not in set(programs.cluster)]
    fig.text(.055,.025,'Green / *: planted source; gray: other candidate. Labels show rank and association score; arrow width encodes score.\n'
             'Conditional on known targets; target identification is shown in the landscape.\n'
             f'Sender-dominated clusters without a dominant planted receiver program: {", ".join(omitted)}.',fontsize=7)
    pd.concat(displayed).to_csv(out/'data/focused_display_relations.csv',index=False)
    pd.concat(recovery).to_csv(out/'data/planted_source_recovery.csv',index=False)
    save(fig,out,'cluster_source_target_relations')


def source_recovery_bars(relations, comp, out, title):
    """Compact per-module bars: top five plus any planted sources outside top five."""
    from matplotlib.patches import Patch
    top_n = 5
    programs = (relations.loc[relations.is_planted, ['cluster', 'target_gene']]
                .drop_duplicates().sort_values(['cluster', 'target_gene']))
    selected_programs = []
    for row in programs.itertuples():
        candidates = relations[(relations.cluster == row.cluster) & (relations.target_gene == row.target_gene)]
        shown = candidates[(candidates.source_rank <= top_n) | candidates.is_planted].copy()
        shown = shown.sort_values(['source_score', 'source_rank'], ascending=[False, True])
        shown['outside_top_five'] = shown.source_rank > top_n
        selected_programs.append(shown)
    ncols = 4
    nrows = int(np.ceil((len(programs)+1)/ncols))
    fig, axes = plt.subplots(nrows,ncols,figsize=(160/25.4,100/25.4),squeeze=False)
    fig.subplots_adjust(left=.155,right=.95,top=.82,bottom=.12,wspace=.78,hspace=.60)
    maximum = max(float(frame.source_score.max()) for frame in selected_programs)
    for ax, shown in zip(axes.flat, selected_programs):
        c = int(shown.iloc[0].cluster)
        target = shown.iloc[0].target_gene
        g = int(comp.loc[c].idxmax())
        colors = np.where(shown.is_planted, '#C83E4D', '#4C86B7')
        ax.barh(np.arange(len(shown)),shown.source_score,color=colors,height=.65)
        ax.set_yticks(np.arange(len(shown)),
                      [f'{r.source_gene}  #{r.source_rank}' for r in shown.itertuples()],fontsize=5.2)
        ax.invert_yaxis()
        ax.set_xlim(0,maximum*1.2)
        for y,row in enumerate(shown.itertuples()):
            if row.is_planted:
                ax.text(row.source_score+maximum*.025,y,f'{row.source_score:.2f}',
                        va='center',fontsize=5,color='#A32B39')
        ax.set_title(f'CCC Module {c} | {target}\nReceiver {g} ({comp.loc[c,g]:.0%})',
                     loc='left',fontsize=6.4,fontweight='bold',pad=5)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0,pad=2)
        ax.tick_params(axis='x',labelsize=5,length=2,pad=2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        ax.set_axisbelow(True);ax.grid(axis='x',color='#EAEAEA',linewidth=.4)
    legend_ax=list(axes.flat)[len(programs)]
    legend_ax.axis('off')
    legend_ax.legend(handles=[Patch(color='#C83E4D',label='Planted source'),
                             Patch(color='#4C86B7',label='Other candidate')],
                     loc='upper left',frameon=False,fontsize=6,borderaxespad=0)
    for ax in list(axes.flat)[len(programs)+1:]:
        ax.axis('off')
    fig.suptitle(f'{title}: source identification',x=.155,y=.98,ha='left',fontsize=9,fontweight='bold')
    fig.text(.53,.035,'Spatial association score',ha='center',fontsize=7)
    pd.concat(selected_programs).to_csv(out/'data/bar_display_relations.csv',index=False)
    save(fig,out,'cluster_source_target_relations')


def main():
    configure_arial()
    for case,variant in VARIANTS.items():
        out=OUTPUT/f'case{case}_{variant}';(out/'data').mkdir(parents=True,exist_ok=True)
        raw,xy,groups,genes,truth=load_dataset(variant)
        labels,comp,targets,relations=compute(variant,raw,xy,groups,genes,truth)
        fig,ax=plt.subplots(figsize=(4.5,4.1),constrained_layout=True)
        plot_groups(ax,xy,groups);fig.suptitle(CASE_TITLES[case],fontweight='bold',fontsize=10)
        save(fig,out,'true_cell_groups')
        fig,ax=plt.subplots(figsize=(130/25.4,90/25.4),constrained_layout=True)
        plot_network(ax,truth,compact=True);fig.suptitle(CASE_TITLES[case],fontweight='bold',fontsize=10)
        save(fig,out,'planted_source_target_network')
        landscape(targets,comp,out,CASE_TITLES[case])
        source_recovery_bars(relations,comp,out,CASE_TITLES[case])
        targets.to_csv(out/'data/target_scores.csv',index=False)
        relations.to_csv(out/'data/source_target_scores.csv',index=False)
        comp.to_csv(out/'data/cluster_group_composition.csv')
        pd.DataFrame({'cluster':labels,'true_group':groups,'spatial_x':xy[:,0],'spatial_y':xy[:,1]}).to_csv(out/'data/cell_clusters.csv',index=False)
        truth.to_csv(out/'data/planted_relations.csv',index=False)
        (out/'data/provenance.json').write_text(json.dumps(dict(case=case,dataset=variant,training_seed=SEED,
            method='transformer_physical',n_clusters=core.N_CONTEXTS,target_score='mean squared reconstruction residual',
            source_score='absolute covariance / source-field standard deviation',
            ranking='all genes; exclude target itself from source candidates',
            truth_annotation='dominant true group only; truth never used in model ranking'),indent=2)+'\n')


if __name__=='__main__':
    main()
