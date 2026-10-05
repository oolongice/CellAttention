#!/usr/bin/env python3
"""Independent merged Wnt3a/Shh OmniPath networks with curved tapered arrows."""
from pathlib import Path
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from mosta_common import configure_style

HERE = Path(os.environ["CELLATTENTION_MOSTA_WORKDIR"]) / "visualization"
VALIDATION = HERE.parent / 'analysis/omnipath_validation'
FIGSIZE = (5.0, 2.45)
PNG_DPI = 600
SOURCE_COLOR = '#DCECF7'
TARGET_COLOR = '#F7DEDE'
INTERMEDIATE_COLOR = '#F2F2F2'
ARROW_COLOR = '#59636E'
SEPARATOR_COLOR = '#AEB7C0'
FONT_SIZE = 6.0
TITLE_SIZE = 7.0
NODE_H = .076
# Minimum center-to-center spacing used by the automatic layout.
GENERAL_NODE_MIN_GAP = .090
CONNECTED_NODE_MIN_GAP = .190
SENDER_X = .205
RECEIVER_FIRST_X = .605
TARGET_X = 1.115
X_LIMIT = 1.175
TARGET_ORDER = {
    'Wnt3a': ['Wnt1', 'Msx1', 'Ttr', 'Dmrta2', 'Bmp7', 'Ptms', 'Olig3'],
    'Shh': ['Foxa2', 'Nkx2-4', 'Sox3', 'Nkx2-1', 'Foxa1', 'Six3', 'Ptch1'],
}


def node_width(label):
    return min(.135, max(.072, .029 + .0112 * len(label)))


def quadratic(p0, p1, p2, t):
    return ((1-t)**2)[:,None]*p0 + (2*(1-t)*t)[:,None]*p1 + (t**2)[:,None]*p2


def tapered_arrow(ax, start, end, bend=0.0, color=ARROW_COLOR, zorder=1):
    """Draw a straight, constant-width arrow with rounded stroke joins."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0',
        mutation_scale=7.5,
        linewidth=1.0,
        color=color,
        capstyle='round',
        joinstyle='round',
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
        alpha=.90,
    )
    ax.add_patch(arrow)


def separator_arc(ax, x, bow, y0=.10, y1=.88):
    ym = (y0+y1)/2
    verts = [(x,y0), (x+bow,ym-.18), (x+bow,ym+.18), (x,y1)]
    path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4,
                           MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, edgecolor=SEPARATOR_COLOR,
                           linewidth=1.15, zorder=0))


def draw_node(ax, x, y, label, role):
    color = {'source':SOURCE_COLOR, 'target':TARGET_COLOR}.get(role, INTERMEDIATE_COLOR)
    w = node_width(label)
    ax.add_patch(Rectangle((x-w/2,y-NODE_H/2),w,NODE_H,
                           facecolor=color, edgecolor='none', zorder=3))
    ax.text(x,y,label,ha='center',va='center',fontsize=FONT_SIZE,zorder=4)
    return w


def paths_for(ligand, summary):
    paths=[]
    for target in TARGET_ORDER[ligand]:
        row=summary[(summary.ligand.eq(ligand)) &
                    (summary.model_target.eq(target))].iloc[0]
        if pd.notna(row.path) and str(row.path).strip():
            paths.append(str(row.path).split(' -> '))
    return paths


def build_positions(ligand, paths):
    targets=TARGET_ORDER[ligand]
    target_y={t:y for t,y in zip(targets,np.linspace(.82,.18,len(targets)))}
    receptors=sorted({p[1] for p in paths})
    positions={ligand:(SENDER_X,.50)}
    receptor_y={r:np.mean([target_y[p[-1]] for p in paths if p[1]==r]) for r in receptors}
    # Spread receptors when their descendant averages are close.
    rec_order=sorted(receptors,key=lambda r:receptor_y[r],reverse=True)
    rec_slots=np.linspace(.75,.25,len(rec_order)) if len(rec_order)>1 else [.50]
    # Receptors are receiver-expressed genes; place their centers directly
    # on the right-hand ligand-receptor interface arc.
    def right_arc_x(y):
        t=np.clip((y-.10)/(.88-.10),0,1)
        return .485 + 3*(-.070)*t*(1-t)
    positions.update({r:(right_arc_x(y),y) for r,y in zip(rec_order,rec_slots)})

    appearances=defaultdict(list)
    # Arrange only intracellular intermediates in layered columns; targets
    # have their own right-aligned column.
    max_depth=max((len(p)-3 for p in paths),default=1)
    for p in paths:
        ty=target_y[p[-1]]
        for depth,node in enumerate(p[2:-1],1):
            appearances[node].append((depth,ty))
    layers=defaultdict(list)
    for node,vals in appearances.items():
        depth=max(d for d,_ in vals)
        layers[depth].append((node,np.mean([y for _,y in vals])))
    for depth,items in layers.items():
        # Consecutive signaling layers are separated by at least the
        # connected-node gap so their arrows retain a visible shaft.
        x=RECEIVER_FIRST_X + (depth-1)*CONNECTED_NODE_MIN_GAP
        ordered=sorted(items,key=lambda z:z[1])
        # Give the first receiver-intracellular layer extra vertical room;
        # most branches originate here, so close nodes cause arrow overlap.
        min_gap=.145 if depth == 1 else GENERAL_NODE_MIN_GAP
        ys=[]
        for _,desired in ordered:
            ys.append(max(desired, ys[-1]+min_gap) if ys else desired)
        if ys and ys[-1]>.84:
            shift=ys[-1]-.84; ys=[y-shift for y in ys]
        if ys and ys[0]<.16:
            shift=.16-ys[0]; ys=[y+shift for y in ys]
        for (node,_),y in zip(ordered,ys):
            positions[node]=(x,y)
    # Targets that also regulate another model target remain in their actual
    # intracellular layer. Only terminal-only targets use the aligned column.
    intermediate_nodes={node for path in paths for node in path[2:-1]}
    for t in targets:
        if t not in intermediate_nodes and t not in receptors:
            positions[t]=(TARGET_X,target_y[t])
    return positions


def render(ligand, summary, edge_frame):
    configure_style()
    paths=paths_for(ligand,summary)
    positions=build_positions(ligand,paths)
    targets=set(TARGET_ORDER[ligand])
    receptors={p[1] for p in paths}
    edge_keys=[]
    for p in paths:
        edge_keys.extend(zip(p[:-1],p[1:]))
    edge_keys=list(dict.fromkeys(edge_keys))

    fig,ax=plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(left=.035,right=.985,bottom=.12,top=.88)
    ax.set_xlim(0,X_LIMIT);ax.set_ylim(0,1);ax.axis('off')
    ax.set_title(ligand,loc='left',fontsize=TITLE_SIZE,fontweight='bold',pad=2)
    ax.text(SENDER_X,.94,'Sender',ha='center',fontsize=4.7,color='#66717C')
    ax.text(.39,.94,'Cell–cell interaction',ha='center',fontsize=4.7,color='#66717C')
    ax.text(.86,.94,'Receiver cell',ha='center',fontsize=4.7,color='#66717C')
    separator_arc(ax,.285,.070)
    separator_arc(ax,.485,-.070)

    widths={n:node_width(n) for n in positions}
    # Draw links before nodes. A larger gap keeps arrowheads clear of node faces.
    for i,(u,v) in enumerate(edge_keys):
        x1,y1=positions[u];x2,y2=positions[v]
        vec=np.array([x2-x1,y2-y1]);dist=np.linalg.norm(vec);unit=vec/max(dist,1e-9)
        start=np.array([x1,y1])+unit*(widths[u]/2+.010)
        end=np.array([x2,y2])-unit*(widths[v]/2+.022)
        tapered_arrow(ax,start,end)
    for node,(x,y) in positions.items():
        role='source' if node==ligand else ('target' if node in targets else 'intermediate')
        draw_node(ax,x,y,node,role)

    handles=[
        Rectangle((0,0),1,1,facecolor=SOURCE_COLOR,edgecolor='none',label='Model source gene'),
        Rectangle((0,0),1,1,facecolor=TARGET_COLOR,edgecolor='none',label='Model target gene'),
        Line2D([0],[0],color=ARROW_COLOR,lw=1.0,solid_capstyle='round',label='OmniPath interaction'),
    ]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.018),
               frameon=False,ncol=3,fontsize=4.2,columnspacing=.9,handlelength=1.5)
    out=HERE/'figures'/f'omnipath_brain_{ligand.lower()}_pathway_merged.png'
    pdf=HERE/'figures'/f'omnipath_brain_{ligand.lower()}_pathway_merged.pdf'
    out.parent.mkdir(parents=True,exist_ok=True)
    pdf.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=PNG_DPI,facecolor='white',pil_kwargs={'compress_level':6})
    fig.savefig(pdf,dpi=PNG_DPI,facecolor='white')
    plt.close(fig)
    print(out)
    print(pdf)


def main():
    summary=pd.read_csv(VALIDATION/'omnipath_validation_summary.csv')
    edges=pd.read_csv(VALIDATION/'omnipath_path_edges.csv')
    summary=summary[summary.organism.eq('mouse')]
    edges=edges[edges.organism.eq('mouse')]
    render('Wnt3a',summary,edges)
    render('Shh',summary,edges)


if __name__=='__main__':
    main()
