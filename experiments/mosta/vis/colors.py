"""Deterministic categorical palettes with dataset-level overrides."""
import matplotlib.pyplot as plt
import numpy as np
DEFAULT_CMAPS=("Set3","tab20","Pastel1","Dark2")
def categorical_palette(labels,*,overrides=None,order=None,cmaps=DEFAULT_CMAPS):
    labels=list(dict.fromkeys(order if order is not None else sorted(set(labels),key=str)));colors=[]
    for name in cmaps:
        cmap=plt.get_cmap(name);colors.extend(cmap(np.linspace(.03,.97,cmap.N)))
    result={label:colors[i%len(colors)] for i,label in enumerate(labels)}
    if overrides:result.update({k:v for k,v in overrides.items() if k in set(labels)})
    return result
def sequential_categories(labels,cmap="Set3"):
    labels=list(dict.fromkeys(labels));cm=plt.get_cmap(cmap);den=max(len(labels)-1,1)
    return {label:cm(i/den) for i,label in enumerate(labels)}
