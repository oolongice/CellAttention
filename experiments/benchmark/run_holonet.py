#!/usr/bin/env python3
"""HoloNet adapter: one CE tensor produces receiver-target and receiver-source-target scores."""
from pathlib import Path
import json,os,numpy as np,pandas as pd,anndata as ad
from HoloNet.tools.CE_network_edge_weighting import compute_ce_tensor_connectomeDB
ROOT=Path(os.environ["CELLATTENTION_BENCHMARK_WORKDIR"]); TAG=os.environ['HIER_VARIANT']; D=ROOT/'data'/TAG; O=ROOT/'results'/TAG/'holonet_adapted'; O.mkdir(parents=True,exist_ok=True)
genes=(D/'gene_ids.txt').read_text().splitlines(); x=np.loadtxt(D/'expression.csv',delimiter=','); xy=np.loadtxt(D/'spatial_coordinates.csv',delimiter=','); groups=np.loadtxt(D/'inferred_groups.txt',dtype=int)
adata=ad.AnnData(x); adata.var_names=genes; adata.obsm['spatial']=xy
lr=pd.read_csv(D/'candidate_lr.csv').rename(columns={'ligand':'Ligand_gene_symbol','receptor':'Receptor_gene_symbol'}); lr['LR_Pair']=lr.Ligand_gene_symbol+':'+lr.Receptor_gene_symbol; lr['Ligand_location']='secreted'
incoming=compute_ce_tensor_connectomeDB(adata,lr,w_best=90,distinguish=False).numpy().sum(1).T; rows=[]
for rg in sorted(set(groups)):
 idx=groups==rg
 for ligand in sorted(lr.Ligand_gene_symbol.unique()):
  pair_idx=np.flatnonzero(lr.Ligand_gene_symbol.to_numpy()==ligand)
  for target in [g for g in genes if g.startswith('target_')]:
   y=x[idx,genes.index(target)]; values=[]
   for k in pair_idx:
    z=incoming[idx,k]; values.append(max(0,float(np.corrcoef(z,y)[0,1])) if np.std(z)>1e-9 and np.std(y)>1e-9 else 0)
   rows.append((rg,ligand,target,max(values)))
triplet=pd.DataFrame(rows,columns=['receiver_group','source_gene','target_gene','score']); triplet.to_csv(O/'triplet_scores.csv',index=False); triplet.groupby(['receiver_group','target_gene'],as_index=False).score.max().to_csv(O/'scores.csv',index=False)
(O/'method_metadata.json').write_text(json.dumps({'method':'HoloNet CE-to-target adapted','version':'0.1.0','communication_event':'native HoloNet CE tensor','downstream_score':'maximum positive marginal correlation over candidate receptors','LR_prior':'complete 8x8 candidate universe, not true edges','full_MGC_training':False,'truth_used':False},indent=2)+'\n')
