#!/usr/bin/env python3
"""Source-level summary and 2x2 source-target network for TIS transitions.

1. Source-level summary: per transition aggregate high-TIS (TIS>0) pairs per
   source gene (S_s = sum max(TIS,0), N_s = #targets) and draw Top-10 bars.

2. 2x2 source-target network: circular layout with source genes on the left
   semicircle and target genes on the right semicircle, connected by tapered
   (thick-at-source -> thin-at-target, no arrow) lines whose width and
   black-grey shade encode TIS. A single shared scale is used across panels.
"""
OUTPUT_FORMAT='png'
TOP_NETWORK=20
TOP_SOURCES=10
W_MIN=0.006;W_MAX=0.045;TAPER=0.10
SOURCE_NODE_MIN=70;SOURCE_NODE_MAX=130;TARGET_NODE=58
GAP=20.0
CURVATURE=0.10
SOURCE_COLOR='#BFD6EA';TARGET_COLOR='#F2B8B5'
LINE_COLOR='#7E96AD'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mosta_common import HERE,configure_style
from vis.style import mm_to_inches,save_figure

def curved_tapered_line(ax, p0, p1, h0, h1, color, curvature=CURVATURE, alpha=0.9, n=60):
    """Thick-at-source, thin-at-target curved ribbon (quadratic Bezier, no arrow)."""
    p0=np.asarray(p0,float);p1=np.asarray(p1,float)
    d=p1-p0;L=np.linalg.norm(d)
    if L<1e-9:return
    mid=(p0+p1)/2
    mnorm=np.linalg.norm(mid)
    ctrl=mid+(mid/mnorm)*(curvature*L) if mnorm>1e-9 else mid
    ts=np.linspace(0,1,n)
    w0=1-ts;w1=ts
    Bx=w0*w0*p0[0]+2*w0*w1*ctrl[0]+w1*w1*p1[0]
    By=w0*w0*p0[1]+2*w0*w1*ctrl[1]+w1*w1*p1[1]
    dBx=2*w0*(ctrl[0]-p0[0])+2*w1*(p1[0]-ctrl[0])
    dBy=2*w0*(ctrl[1]-p0[1])+2*w1*(p1[1]-ctrl[1])
    Tn=np.sqrt(dBx**2+dBy**2)
    Tx=dBx/Tn;Ty=dBy/Tn
    Px=-Ty;Py=Tx
    width=h0*(1-ts)+h1*ts
    left=np.column_stack([Bx-Px*width/2, By-Py*width/2])
    right=np.column_stack([Bx+Px*width/2, By+Py*width/2])
    ax.add_patch(Polygon(np.vstack([left,right[::-1]]),closed=True,
                         facecolor=color,edgecolor='none',alpha=alpha,zorder=2))

def radial_label(ax, x, y, angle_deg, text, fontsize):
    rot=angle_deg%360
    flip=90<rot<270
    ha='left' if not flip else 'right'
    rotation=(rot+180) if flip else rot
    ax.text(x,y,text,ha=ha,va='center',rotation=rotation,rotation_mode='anchor',
            fontsize=fontsize,color='#202020',zorder=5)

def draw_network(ax, pairs, global_max, fontsize=9):
    src_S={s:g.TIS.clip(lower=0).sum() for s,g in pairs.groupby('source_gene')}
    sources=sorted(src_S,key=lambda s:-src_S[s])
    n_src=len(sources)
    max_S=max(src_S.values()) if src_S else 1.0
    src_deg={s:90+GAP+(180-2*GAP)*i/max(n_src-1,1) for i,s in enumerate(sources)}

    edges=list(pairs[['source_gene','target_gene','TIS']].itertuples(index=False))
    targets=sorted(set(pairs.target_gene))
    def bary(t):
        conn=[np.deg2rad(src_deg[s]) for s,tt,_ in edges if tt==t]
        return np.mean(conn) if conn else np.deg2rad(180)
    targets=sorted(targets,key=bary)
    n_tgt=len(targets)
    tgt_deg={t:-90+GAP+(180-2*GAP)*j/max(n_tgt-1,1) for j,t in enumerate(targets)}

    R=1.15
    def pos(deg):
        a=np.deg2rad(deg);return (R*np.cos(a),R*np.sin(a))
    src_pos={s:pos(src_deg[s]) for s in sources}
    tgt_pos={t:pos(tgt_deg[t]) for t in targets}

    for s,t,tis in edges:
        tn=min(max(tis,0.0),global_max)/global_max
        h0=W_MIN+(W_MAX-W_MIN)*tn
        h1=h0*TAPER
        curved_tapered_line(ax,src_pos[s],tgt_pos[t],h0,h1,LINE_COLOR)

    for s in sources:
        size=SOURCE_NODE_MIN+(SOURCE_NODE_MAX-SOURCE_NODE_MIN)*(src_S[s]/max_S)
        ax.scatter([src_pos[s][0]],[src_pos[s][1]],s=size,c=SOURCE_COLOR,
                   edgecolors='#333333',linewidths=.8,zorder=3)
    ax.scatter([tgt_pos[t][0] for t in targets],[tgt_pos[t][1] for t in targets],
               s=TARGET_NODE,c=TARGET_COLOR,edgecolors='#333333',linewidths=.8,zorder=3)

    RL=1.33
    for s in sources:
        radial_label(ax,RL*np.cos(np.deg2rad(src_deg[s])),RL*np.sin(np.deg2rad(src_deg[s])),
                     src_deg[s],s,fontsize)
    for t in targets:
        radial_label(ax,RL*np.cos(np.deg2rad(tgt_deg[t])),RL*np.sin(np.deg2rad(tgt_deg[t])),
                     tgt_deg[t],t,fontsize)

    ax.set_xlim(-1.9,1.9);ax.set_ylim(-1.9,1.9)
    ax.set_aspect('equal');ax.axis('off')

def main():
    configure_style()
    figdir=HERE/'figures/brain_development';datadir=HERE/'data'
    figdir.mkdir(parents=True,exist_ok=True);datadir.mkdir(parents=True,exist_ok=True)

    ranking=pd.read_csv(datadir/'brain_transition_tis_ranking.csv')
    transitions=ranking.transition.unique().tolist()

    # ---- source-level summary figures ----
    for tr in transitions:
        g=ranking[ranking.transition==tr]
        pos=g[g.TIS>0]
        agg=pos.groupby('source_gene').agg(
            S=('TIS',lambda x:float(x.clip(lower=0).sum())),
            N=('target_gene','nunique')).reset_index().sort_values('S',ascending=False).head(TOP_SOURCES)
        agg=agg.iloc[::-1]
        a,b=tr.split('-')
        fig,ax=plt.subplots(figsize=mm_to_inches(120,95))
        fig.subplots_adjust(left=.30,right=.88,bottom=.13,top=.92)
        y=np.arange(len(agg))
        ax.barh(y,agg.S,color='#4C72B0',edgecolor='white',linewidth=.4)
        ax.set_yticks(y,agg.source_gene,fontsize=10)
        ax.set_xlabel('cumulative TIS  S = \u03a3 max(TIS,0)',fontsize=10)
        ax.set_title(f'{tr}: transition-specific sources',loc='left',fontweight='bold',fontsize=12)
        for yi,(s,n) in enumerate(zip(agg.S,agg.N)):
            ax.text(s,yi,f'  n={n}',ha='left',va='center',fontsize=9,color='#333333')
        ax.spines[['top','right']].set_visible(False)
        save_figure(fig,figdir/f'brain_transition_{a}_{b}_source_summary.{OUTPUT_FORMAT}',dpi=600)
        print(figdir/f'brain_transition_{a}_{b}_source_summary.{OUTPUT_FORMAT}')

    # ---- 2x2 network ----
    top_pairs={}
    global_max=0.0
    for tr in transitions:
        g=ranking[ranking.transition==tr].head(TOP_NETWORK).copy()
        top_pairs[tr]=g
        global_max=max(global_max,float(g.TIS.max()))
    global_max=max(global_max,1e-6)

    fig,axes=plt.subplots(2,2,figsize=mm_to_inches(180,168))
    fig.subplots_adjust(left=.02,right=.98,bottom=.03,top=.91,wspace=.02,hspace=.05)
    for ax,tr in zip(axes.ravel(),transitions):
        draw_network(ax,top_pairs[tr],global_max,fontsize=10)
        ax.set_title(tr,loc='left',fontweight='bold',fontsize=13,pad=4)

    fig.suptitle('Transition-specific source\u2013target networks',
                 x=.02,y=.975,ha='left',fontweight='bold',fontsize=12.5)
    save_figure(fig,figdir/f'brain_transition_network_2x2.{OUTPUT_FORMAT}',dpi=600)
    print(figdir/f'brain_transition_network_2x2.{OUTPUT_FORMAT}')

    # ---- individual vector networks + edge tables ----
    for tr in transitions:
        a,b=tr.split('-')
        g=top_pairs[tr]
        fig,ax=plt.subplots(figsize=mm_to_inches(120,120))
        fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.92)
        draw_network(ax,g,global_max,fontsize=12)
        fig.suptitle(tr,fontweight='bold',fontsize=14,y=.97)
        svg_path=figdir/f'brain_transition_{a}_{b}_network.svg'
        fig.savefig(svg_path,facecolor='white')
        plt.close(fig)
        g[['rank','source_gene','target_gene','TIS','mu_T','mu_A','mu_B','pooled_sd',
           'signed_mean_T','signed_beta_a','signed_beta_b','derived_attention_a','derived_attention_b']]\
          .to_csv(datadir/f'brain_transition_{a}_{b}_network_edges.csv',index=False)
        print(svg_path)

    print('done')

if __name__=='__main__':main()
