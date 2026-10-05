#!/usr/bin/env python3
from pathlib import Path
import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import k0
from scipy.sparse import csr_matrix

DATASET_DIR=Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"])
RUN=DATASET_DIR/'model'/'transformer'
DATA=DATASET_DIR/'data'/'preprocessed'/'control_train'
PARSER=argparse.ArgumentParser()
PARSER.add_argument("--output-dir", type=Path, default=DATASET_DIR/"analysis/source_focused_immune")
PARSER.add_argument("--receiver-field-mask", choices=["none", "hard_expression_exclusion"], default="none")
PARSER.add_argument("--expression-threshold", type=float, default=0.0)
ARGS=PARSER.parse_args()
OUT=ARGS.output_dir; OUT.mkdir(parents=True,exist_ok=True)
SOURCES=['Icam1','Cxcr4']; IMMUNE=['Macrophages','B','T','DC','Neutrophils','MAST','Plasma','NK']

def load(p): return np.loadtxt(p,delimiter=',',dtype=np.float64)
def standardize(x):
    sd=x.std(0);sd[sd<1e-8]=1
    return (x-x.mean(0))/sd
def physical(coords,source):
    tree=cKDTree(coords);rows=[];cols=[];val=[]
    for i,js in enumerate(tree.query_ball_point(coords,75.0)):
        for j in js:
            if i==j:continue
            d=max(float(np.linalg.norm(coords[i]-coords[j])),2.0)
            rows.append(i);cols.append(j);val.append(k0(d/25.0)/(2*np.pi*625.0))
    field=np.asarray(csr_matrix((val,(rows,cols)),shape=(len(coords),len(coords)))@source)
    if ARGS.receiver_field_mask=="hard_expression_exclusion":
        field[source>ARGS.expression_threshold]=0.0
    return standardize(field)
def knn(coords,source):
    d,j=cKDTree(coords).query(coords,k=21);d,j=d[:,1:],j[:,1:]
    w=np.exp(-d*d/(2*25.0**2));w/=w.sum(1,keepdims=True)
    return standardize((source[j]*w[:,:,None]).sum(1))
def blocked_folds(coords):
    b=np.floor(coords/75.0).astype(np.int64)
    return np.mod((b[:,0]*73856093)^(b[:,1]*19349663),5)
def evaluate(x,y,mask,fold):
    effects=[];base_sse=np.zeros(y.shape[1]);model_sse=np.zeros(y.shape[1]);cors=[]
    for f in range(5):
        tr=mask&(fold!=f);va=mask&(fold==f)
        if tr.sum()<20 or va.sum()<5:continue
        xm=x[tr].mean();ym=y[tr].mean(0);xc=x[tr]-xm;yc=y[tr]-ym
        beta=(xc[:,None]*yc).mean(0)/(np.square(xc).mean()+1e-12);effects.append(beta)
        pred=ym+(x[va]-xm)[:,None]*beta; err=y[va]-pred;base=y[va]-ym
        model_sse+=np.square(err).sum(0);base_sse+=np.square(base).sum(0)
        xv=x[va]-x[va].mean();ys=y[va]-y[va].mean(0);den=np.sqrt(np.square(xv).sum()*np.square(ys).sum(0))+1e-12
        cors.append((xv[:,None]*ys).sum(0)/den)
    effects=np.asarray(effects);cors=np.asarray(cors)
    return (base_sse-model_sse)/(mask.sum()+1e-12),np.nanmean(cors,0),np.nanmedian(effects,0),np.mean(np.sign(effects)==np.sign(np.nanmedian(effects,0)),axis=0),len(effects)

genes=RUN.joinpath('gene_ids.txt').read_text().splitlines();idx=[genes.index(s) for s in SOURCES]
raw=load(RUN/'raw_expression.csv');residual=load(RUN/'residuals.csv');coords=load(DATA/'spatial_coordinates_um.csv');ann=np.array(DATA.joinpath('annotation.txt').read_text().splitlines());fold=blocked_folds(coords)
source=raw[:,idx];fields={'raw':standardize(source),'knn':knn(coords,source),'physical':physical(coords,source)}
groups={'Immune_all':np.isin(ann,IMMUNE)}|{a:ann==a for a in IMMUNE}
rows=[]
for receiver,mask in groups.items():
    for si,sname in enumerate(SOURCES):
        for method,matrix in fields.items():
            delta,corr,effect,consistency,nfold=evaluate(matrix[:,si],residual,mask,fold)
            for target,tname in enumerate(genes):
                if tname==sname:continue
                rows.append([sname,tname,receiver,int(mask.sum()),method,delta[target],corr[target],effect[target],consistency[target],nfold])
x=pd.DataFrame(rows,columns=['source_gene','target_gene','receiver_cell_type','receiver_cell_count','method','heldout_mse_reduction','heldout_correlation','effect_median','direction_consistency','valid_folds'])
x.to_csv(OUT/'all_method_results.csv',index=False)
p=x[x.method=='physical'].merge(x[x.method=='raw'],on=['source_gene','target_gene','receiver_cell_type','receiver_cell_count'],suffixes=('_physical','_raw')).merge(x[x.method=='knn'],on=['source_gene','target_gene','receiver_cell_type','receiver_cell_count'])
p=p.rename(columns={'heldout_mse_reduction':'heldout_mse_reduction_knn','heldout_correlation':'heldout_correlation_knn','effect_median':'effect_median_knn','direction_consistency':'direction_consistency_knn'})
p['physical_minus_raw']=p.heldout_mse_reduction_physical-p.heldout_mse_reduction_raw
p['physical_minus_knn']=p.heldout_mse_reduction_physical-p.heldout_mse_reduction_knn
p['physical_best']=(p.heldout_mse_reduction_physical>p.heldout_mse_reduction_raw)&(p.heldout_mse_reduction_physical>p.heldout_mse_reduction_knn)
p['physical_rank_within_source_receiver']=p.groupby(['source_gene','receiver_cell_type']).heldout_mse_reduction_physical.rank(method='min',ascending=False).astype(int)
p.to_csv(OUT/'source_target_receiver_results.csv',index=False)
p.sort_values(['source_gene','receiver_cell_type','physical_rank_within_source_receiver']).groupby(['source_gene','receiver_cell_type']).head(25).to_csv(OUT/'top25_targets_by_source_receiver.csv',index=False)
q=p[(p.receiver_cell_type=='Immune_all')].sort_values(['source_gene','physical_rank_within_source_receiver']).groupby('source_gene').head(30)
q.to_csv(OUT/'immune_all_top30.csv',index=False)
(OUT/"receiver_field_mask.json").write_text(json.dumps({
    "mode": ARGS.receiver_field_mask,
    "expression_threshold": ARGS.expression_threshold,
    "masked_fraction_by_source": {s: float((source[:, i] > ARGS.expression_threshold).mean()) if ARGS.receiver_field_mask == "hard_expression_exclusion" else 0.0 for i, s in enumerate(SOURCES)},
    "applied_before_standardization": True,
    "raw_and_knn_baselines_masked": False,
}, indent=2) + "\n")
