"""Reusable spatial maps, scale bars, and reaction-diffusion fields."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import k0
from .style import mm_to_inches,save_figure
def plot_spatial_categories(xy,labels,colors,*,title,path=None,legend=True,point_size=.45,alpha=.78,invert_y=True,figure_mm=(180,105),legend_columns=2,display_names=None,order=None,layout=None):
    labels=np.asarray(labels);xy=np.asarray(xy);fig,ax=plt.subplots(figsize=mm_to_inches(*figure_mm));layout=layout or dict(left=.025,right=.68 if legend else .97,bottom=.035,top=.90);fig.subplots_adjust(**layout);order=list(order) if order is not None else sorted(set(labels),key=str)
    for label in order:
        keep=labels==label
        if keep.any():ax.scatter(xy[keep,0],xy[keep,1],s=point_size,c=[colors[label]],alpha=alpha,linewidths=0,rasterized=True)
    ax.set_aspect("equal");ax.set_axis_off()
    if invert_y:ax.invert_yaxis()
    ax.set_title(title,loc="left",fontweight="bold")
    if legend:
        names=display_names or {};counts={x:int(np.count_nonzero(labels==x)) for x in order};handles=[mpl.lines.Line2D([0],[0],marker="o",linestyle="none",markersize=3.6,markerfacecolor=colors[x],markeredgewidth=0,label=names.get(x,x)) for x in order if counts[x]]
        fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.69,.5),frameon=False,ncol=legend_columns,columnspacing=.7,labelspacing=.38,handletextpad=.3)
    if path is not None:save_figure(fig,path)
    return fig,ax
def add_scale_bar(ax,xy,length,label=None,*,location=(.035,.045),color="#202020",linewidth=1.5):
    xy=np.asarray(xy);xmin,xmax=xy[:,0].min(),xy[:,0].max();ymin,ymax=xy[:,1].min(),xy[:,1].max();x0=xmin+location[0]*(xmax-xmin);y0=ymax-location[1]*(ymax-ymin);ax.plot([x0,x0+length],[y0,y0],color=color,linewidth=linewidth,solid_capstyle="butt",zorder=10);ax.text(x0+length/2,y0-.018*(ymax-ymin),label or str(length),ha="center",va="bottom",fontsize=6.5,color=color,zorder=10)
def reaction_diffusion_field(coordinates,expression,query=None,*,length_scale,maximum_distance,minimum_distance,diffusion=None,exclude_self=False):
    coordinates=np.asarray(coordinates,float);expression=np.asarray(expression,float);query=coordinates if query is None else np.asarray(query,float);source=np.flatnonzero(expression>0)
    if not len(source):raise ValueError("source gene has no positive-expression cells")
    coo=cKDTree(query).sparse_distance_matrix(cKDTree(coordinates[source]),maximum_distance,output_type="coo_matrix");keep=np.ones(len(coo.data),bool)
    if exclude_self and query is coordinates:keep=coo.row!=source[coo.col]
    radius=np.maximum(coo.data[keep],minimum_distance);diffusion=length_scale**2 if diffusion is None else diffusion;weights=k0(radius/length_scale)/(2*np.pi*diffusion);field=np.bincount(coo.row[keep],weights=weights*expression[source[coo.col[keep]]],minlength=len(query));return field,source


def plot_spatial_influence_map(
    ax, coordinates, *, field_x, field_y, field_values, source_mask,
    source_expression, target_mask, target_expression, edges,
    source_gene, target_gene, candidate_target_mask=None,
    source_cmap="Blues", target_cmap="Reds",
    field_cmap="Blues", background_color="#BDBDBD", background_size=5,
    cell_size=32, arrow_color="black", arrow_width=.65, arrow_alpha=.72,
    arrow_head_base=5.0, arrow_head_scale=2.0,
    arrow_shrink_source=4.5, arrow_shrink_target=5.0, arrow_curve=.025,
    edge_style="-|>", cell_edge_color="black", cell_edge_width=.38,
):
    """Render de novo field-conditioned sender-cell to receiver-cell influences.

    `edges` is an iterable of (sender_cell_index, receiver_cell_index, strength).
    Biological selection and hotspot construction remain dataset-specific.
    """
    from matplotlib.lines import Line2D
    coordinates=np.asarray(coordinates);source_mask=np.asarray(source_mask,bool);target_mask=np.asarray(target_mask,bool)
    candidate_target_mask=np.zeros(len(coordinates),bool) if candidate_target_mask is None else np.asarray(candidate_target_mask,bool)
    candidate_only=candidate_target_mask&~source_mask&~target_mask
    field_values=np.asarray(field_values,float);finite=field_values[np.isfinite(field_values)]
    maximum=float(np.quantile(finite,.995)) if len(finite) else 1.0
    artist=ax.pcolormesh(field_x,field_y,np.minimum(field_values,maximum),cmap=field_cmap,shading="auto",alpha=.48,rasterized=True,zorder=0)
    other=~(source_mask|target_mask|candidate_only);ax.scatter(coordinates[other,0],coordinates[other,1],s=background_size,c=background_color,alpha=.62,linewidths=0,rasterized=True,zorder=1)
    if candidate_only.any():ax.scatter(coordinates[candidate_only,0],coordinates[candidate_only,1],s=cell_size*.48,c="#F1B6B6",alpha=.58,marker="^",edgecolors="none",rasterized=True,zorder=2)
    edge_list=list(edges);max_strength=max((float(x[2]) for x in edge_list),default=1.0)
    for source,target,strength in edge_list:
        scale=np.sqrt(max(float(strength),0)/max(max_strength,1e-12));ax.annotate("",xy=coordinates[target],xytext=coordinates[source],arrowprops={"arrowstyle":edge_style,"color":arrow_color,"lw":arrow_width*(.65+.7*scale),"alpha":arrow_alpha,"mutation_scale":arrow_head_base+arrow_head_scale*scale,"shrinkA":arrow_shrink_source,"shrinkB":arrow_shrink_target,"connectionstyle":f"arc3,rad={arrow_curve}"},zorder=3)
    source_values=np.asarray(source_expression,float);target_values=np.asarray(target_expression,float);source_max=max(float(np.quantile(source_values[source_mask],.98)) if source_mask.any() else 0,1e-8);target_max=max(float(np.quantile(target_values[target_mask],.98)) if target_mask.any() else 0,1e-8)
    source_artist=ax.scatter(coordinates[source_mask,0],coordinates[source_mask,1],s=cell_size,c=np.clip(source_values[source_mask],0,source_max),cmap=source_cmap,vmin=0,vmax=source_max,marker="o",edgecolors=cell_edge_color,linewidths=cell_edge_width,rasterized=True,zorder=4)
    target_artist=ax.scatter(coordinates[target_mask,0],coordinates[target_mask,1],s=cell_size*1.08,c=np.clip(target_values[target_mask],0,target_max),cmap=target_cmap,vmin=0,vmax=target_max,marker="^",edgecolors=cell_edge_color,linewidths=cell_edge_width,rasterized=True,zorder=5)
    handles=[Line2D([0],[0],marker="o",linestyle="none",markersize=5,markerfacecolor="#5B9BD5",markeredgecolor=cell_edge_color,markeredgewidth=cell_edge_width,label=f"{source_gene}+ sender cell")]
    if candidate_target_mask.any():handles.append(Line2D([0],[0],marker="^",linestyle="none",markersize=4.2,markerfacecolor="#F1B6B6",markeredgewidth=0,label=f"All Cluster {target_gene}+ cells"))
    handles.extend([Line2D([0],[0],marker="^",linestyle="none",markersize=5.5,markerfacecolor="#E45B5B",markeredgecolor=cell_edge_color,markeredgewidth=cell_edge_width,label=f"Edge-supported {target_gene}+ receiver"),Line2D([0],[0],marker="o",linestyle="none",markersize=3.5,markerfacecolor=background_color,markeredgewidth=0,label="Other cell"),Line2D([0],[0],color="black",linewidth=arrow_width,label="Model-inferred edge")])
    return artist,source_artist,target_artist,handles
