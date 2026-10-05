#!/usr/bin/env python3
"""Six-panel dual-ring view of cluster-specific source-target programs."""
OUTPUT_FORMAT="png"
PNG_DPI=600
INNER_RING=(.25,.43)
OUTER_RING=(.84,1.00)
GROUP_GAP_DEGREES=4.0
TOP_LABEL_GAP_DEGREES=18.0
TARGETS_PER_CLUSTER=3
SOURCES_PER_TARGET=6

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch,Wedge
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
from scipy.io import mmread

from slideseq_common import HERE,PRE,configure_style,lines,palette

RELATIONS=HERE.parent/"analysis/all_selected_cluster_source_target_relations.csv"
CELL_EDGES=HERE.parent/"analysis/cell_cell_edges/cell_cell_interaction_edges.csv"



def select_display_relations(relations):
    """Select several top targets and high-attention sources per target."""
    selected=[]
    for cluster,group in relations.groupby("cluster",sort=True):
        target_order=(group.groupby("target_gene",as_index=False)
                      .agg(target_rank=("target_rank","min"),training_improvement=("training_improvement","max"))
                      .sort_values(["target_rank","training_improvement","target_gene"],ascending=[True,False,True])
                      .head(TARGETS_PER_CLUSTER).target_gene)
        for target in target_order:
            subset=group[group.target_gene.eq(target)].copy()
            subset["absolute_beta"]=subset.signed_beta.abs()
            subset=(subset.sort_values(["derived_attention","absolute_beta","source_rank"],ascending=[False,False,True])
                    .drop_duplicates("source_gene").head(SOURCES_PER_TARGET))
            selected.append(subset.drop(columns="absolute_beta"))
    return pd.concat(selected,ignore_index=True)

def expression_type_map(genes):
    """Fallback gene origin: annotation with the highest mean expression."""
    gene_ids=lines(PRE/"gene_ids.txt");gene_index={gene:i for i,gene in enumerate(gene_ids)}
    annotations=np.asarray(lines(PRE/"evaluation_cell_groups.txt"),object)
    matrix=mmread(PRE/"expression.mtx").tocsr()
    if matrix.shape[0]!=len(annotations):matrix=matrix.T.tocsr()
    present=[gene for gene in sorted(set(genes)) if gene in gene_index]
    indices=np.asarray([gene_index[gene] for gene in present],int)
    subset=matrix[:,indices]
    cell_types=sorted(set(annotations));means=np.zeros((len(cell_types),len(present)))
    for i,cell_type in enumerate(cell_types):means[i]=np.asarray(subset[annotations==cell_type].mean(axis=0)).ravel()
    winners=np.argmax(means,axis=0)
    return {gene:cell_types[int(winner)] for gene,winner in zip(present,winners)}


def dominant_type_table(edges,relations,expression_types):
    """Assign each displayed gene to its dominant model-edge endpoint type."""
    keep=edges.merge(relations[["cluster","source_gene","target_gene"]].drop_duplicates(),
                     on=["cluster","source_gene","target_gene"],how="inner")
    source=(keep.groupby(["cluster","source_gene","sender_cell_type"],as_index=False)
            .influence_score.sum().sort_values("influence_score",ascending=False)
            .drop_duplicates(["cluster","source_gene"])
            .rename(columns={"sender_cell_type":"source_type"}))
    target=(keep.groupby(["cluster","target_gene","receiver_cell_type"],as_index=False)
            .influence_score.sum().sort_values("influence_score",ascending=False)
            .drop_duplicates(["cluster","target_gene"])
            .rename(columns={"receiver_cell_type":"target_type"}))
    out=(relations.merge(source[["cluster","source_gene","source_type"]],on=["cluster","source_gene"],how="left")
         .merge(target[["cluster","target_gene","target_type"]],on=["cluster","target_gene"],how="left"))
    out["source_type"]=out.source_type.fillna(out.source_gene.map(expression_types)).fillna("Unassigned")
    out["target_type"]=out.target_type.fillna(out.target_gene.map(expression_types)).fillna(out.dominant_cell_type).fillna("Unassigned")
    return out


def angular_segments(genes,gene_types,start_degrees=90.0):
    """Equal gene sectors with gaps between cell-type groups."""
    frame=pd.DataFrame({"gene":genes,"cell_type":gene_types}).drop_duplicates()
    frame=frame.sort_values(["cell_type","gene"],kind="stable").reset_index(drop=True)
    groups=list(frame.cell_type.drop_duplicates())
    gap=np.deg2rad(GROUP_GAP_DEGREES);top_gap=np.deg2rad(TOP_LABEL_GAP_DEGREES)
    available=2*np.pi-top_gap-gap*len(groups)
    width=available/max(len(frame),1);angle=np.deg2rad(start_degrees)+top_gap/2
    segments={};group_bounds=[]
    for cell_type in groups:
        members=frame[frame.cell_type.eq(cell_type)]
        group_start=angle
        for row in members.itertuples():
            segments[row.gene]=(angle,angle+width,row.cell_type)
            angle+=width
        group_bounds.append((group_start,angle,cell_type,len(members)))
        angle+=gap
    return frame,segments,group_bounds


def polar_xy(radius,angle):
    return radius*np.cos(angle),radius*np.sin(angle)


def tangent_label_style(angle):
    """Keep tangential labels upright on the two clock-face semicircles."""
    degrees=np.rad2deg(angle)%360;rotation=degrees-90
    if 90<degrees<270:rotation+=180
    return rotation,"center"


def radial_label_style(angle):
    """Place outward radial labels with one readable direction per semicircle."""
    degrees=np.rad2deg(angle)%360
    if 90<degrees<270:return degrees+180,"right"
    return degrees,"left"


def draw_ring(ax,segments,inner,outer,colors,font_size,label_mode):
    for gene,(a0,a1,cell_type) in segments.items():
        ax.add_patch(Wedge((0,0),outer,np.rad2deg(a0),np.rad2deg(a1),
                           width=outer-inner,facecolor=colors[cell_type],
                           edgecolor="white",linewidth=.7,zorder=3))
        middle=(a0+a1)/2
        if label_mode=="outside_radial":
            radius=outer+.035;rotation,horizontal=radial_label_style(middle)
        else:
            radius=(inner+outer)/2;rotation,horizontal=tangent_label_style(middle)
        x,y=polar_xy(radius,middle)
        ax.text(x,y,gene,ha=horizontal,va="center",rotation=rotation,
                rotation_mode="anchor",fontsize=font_size,color="#202020",zorder=5)


def target_anchor_angles(relations,target_segments):
    anchors={}
    for target,group in relations.groupby("target_gene",sort=False):
        a0,a1,_=target_segments[target];ordered=group.sort_values("source_gene")
        margin=min(np.deg2rad(8),(a1-a0)*.18)
        values=np.linspace(a0+margin,a1-margin,len(ordered)+2)[1:-1]
        anchors.update({(row.source_gene,row.target_gene):angle for row,angle in zip(ordered.itertuples(),values)})
    return anchors


def draw_cluster(ax,relations,colors):
    sources=(relations[["source_gene","source_type"]].drop_duplicates()
             .sort_values(["source_type","source_gene"],kind="stable"))
    targets=(relations[["target_gene","target_type"]].drop_duplicates()
             .sort_values(["target_type","target_gene"],kind="stable"))
    _,source_segments,_=angular_segments(sources.source_gene,sources.source_type)
    _,target_segments,_=angular_segments(targets.target_gene,targets.target_type)
    draw_ring(ax,target_segments,*INNER_RING,colors,font_size=4.5,label_mode="inside_tangent")
    draw_ring(ax,source_segments,*OUTER_RING,colors,font_size=4.35,label_mode="outside_radial")

    maximum=max(float(relations.signed_beta.abs().max()),1e-12)
    target_anchors=target_anchor_angles(relations,target_segments)
    for row in relations.sort_values("signed_beta",key=lambda x:x.abs()).itertuples():
        sa0,sa1,source_type=source_segments[row.source_gene]
        source_angle=(sa0+sa1)/2
        target_angle=target_anchors[(row.source_gene,row.target_gene)]
        start_radius=OUTER_RING[0]-.018;end_radius=INNER_RING[1]+.018
        t=np.linspace(0,1,48);smooth=t*t*(3-2*t)
        delta=((target_angle-source_angle+np.pi)%(2*np.pi))-np.pi
        angles=source_angle+delta*smooth
        radii=start_radius+(end_radius-start_radius)*t
        vertices=np.column_stack([radii*np.cos(angles),radii*np.sin(angles)])
        codes=np.full(len(vertices),MplPath.LINETO,dtype=np.uint8);codes[0]=MplPath.MOVETO
        path=MplPath(vertices,codes)
        signed_value=float(row.signed_beta)/maximum
        arrow_color=plt.get_cmap("coolwarm")(.5+.5*np.clip(signed_value,-1,1))
        arrow=FancyArrowPatch(path=path,arrowstyle="-|>",mutation_scale=5.5,
                              linewidth=.82,color=arrow_color,alpha=.78,
                              zorder=2,shrinkA=0,shrinkB=0)
        ax.add_patch(arrow)
    ax.add_patch(plt.Circle((0,0),INNER_RING[0],facecolor="white",edgecolor="#D0D0D0",linewidth=.45,zorder=1))
    ax.set_xlim(-1.23,1.23);ax.set_ylim(-1.18,1.18);ax.set_aspect("equal");ax.axis("off")
    row=relations.iloc[0]
    ax.set_title(f"Cluster {int(row.cluster)}\n{row.dominant_cell_type} ({row.dominant_cell_type_fraction:.0%})",
                 loc="left",fontweight="bold",fontsize=6.5,pad=1.5)


def main():
    configure_style();relations=select_display_relations(pd.read_csv(RELATIONS));edges=pd.read_csv(CELL_EDGES)
    expression_types=expression_type_map(set(relations.source_gene)|set(relations.target_gene))
    relations=dominant_type_table(edges,relations,expression_types)
    cell_types=sorted(set(relations.source_type)|set(relations.target_type))
    colors=palette(cell_types)
    figure,axes=plt.subplots(2,3,figsize=(180/25.4,143/25.4))
    figure.subplots_adjust(left=.025,right=.975,bottom=.19,top=.91,wspace=.12,hspace=.22)
    for ax,cluster in zip(axes.ravel(),range(1,7)):
        draw_cluster(ax,relations[relations.cluster.eq(cluster)].copy(),colors)
    cell_handles=[Line2D([0],[0],marker="s",linestyle="none",markersize=4.2,
                         markerfacecolor=colors[t],markeredgewidth=0,label=t) for t in cell_types]
    relation_handles=[Line2D([0],[0],color=plt.get_cmap("coolwarm")(.88),linewidth=1.5,label="Positive β"),
                      Line2D([0],[0],color=plt.get_cmap("coolwarm")(.12),linewidth=1.5,label="Negative β")]
    figure.legend(handles=cell_handles+relation_handles,loc="lower center",bbox_to_anchor=(.5,.015),
                  ncol=5,frameon=False,columnspacing=.75,labelspacing=.35,handletextpad=.35,fontsize=4.7)
    figure.suptitle("Slide-seqV2 hippocampus cluster-specific source–target programs",
                    x=.025,y=.975,ha="left",fontweight="bold",fontsize=9)
    output=HERE/"figures"/f"cluster_source_target_relations.{OUTPUT_FORMAT}"
    figure.savefig(output,dpi=PNG_DPI,facecolor="white",pil_kwargs={"compress_level":6})
    figure.savefig(output.with_suffix(".pdf"),dpi=PNG_DPI,facecolor="white")
    plt.close(figure)
    relations.to_csv(HERE/"data/radial_source_target_relations.csv",index=False)
    print(output)


if __name__=="__main__":main()
