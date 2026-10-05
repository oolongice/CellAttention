"""Cell-type composition tables and stacked-bar rendering."""
import numpy as np
import pandas as pd
def cell_type_composition(receiver_groups,annotations):
    rows=[]
    for receiver_group in sorted(set(receiver_groups)):
        keep=np.asarray(receiver_groups)==receiver_group
        for label,count in pd.Series(np.asarray(annotations)[keep]).value_counts().items():rows.append({"cluster":receiver_group,"cell_type":label,"count":int(count),"fraction":float(count/keep.sum())})
    return pd.DataFrame(rows)
def plot_stacked_composition(ax,table,colors,*,top_per_context=12,xlabel="Receiver group",ylabel="Cell-type fraction",legend=True):
    top=table.sort_values(["cluster","fraction"],ascending=[True,False]).groupby("cluster").head(top_per_context);pivot=top.pivot(index="cluster",columns="cell_type",values="fraction").fillna(0);bottom=np.zeros(len(pivot))
    for label in pivot.columns:ax.bar(pivot.index,pivot[label],bottom=bottom,color=colors[label],width=.78,label=label);bottom+=pivot[label].to_numpy()
    ax.set(xlabel=xlabel,ylabel=ylabel);ax.set_xticks(pivot.index);ax.spines[["top","right"]].set_visible(False)
    if legend:ax.legend(loc="center left",bbox_to_anchor=(1.01,.5),frameon=False,ncol=2)
    return pivot
