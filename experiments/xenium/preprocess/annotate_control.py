#!/usr/bin/env python3
"""Annotate nondiseased liver using the matched Leiden-pseudobulk workflow."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread

PRE=Path(sys.argv[1]).expanduser().resolve()
ACTIVE=PRE/"evaluation_cell_groups.txt"
DETAILS=PRE/"control_unsupervised_pseudobulk_annotation.csv"
GROUPS=PRE/"nondiseased_unsupervised_pseudobulk_cell_groups.csv"
PSEUDOBULK=PRE/"control_leiden_cluster_pseudobulk.csv"
SCORES=PRE/"control_leiden_cluster_annotation_scores.csv"
SUMMARY=PRE/"control_unsupervised_pseudobulk_summary.json"
SEED,N_PCS,N_NEIGHBORS=17,30,15
CORE_Z,SCORE_MARGIN=0.4,0.25

M={
"Cholangiocyte":{"core":"EPCAM KRT7 FXYD2 TM4SF4 TMC5 TFF2","aux":"SFTA2 SMIM24 AQP3 CXCL6 SCGN GPX2","anti":"ADH4 APOA5 CYP2A7 CYP2B6 CYP3A4 HPX HMGCS2 TAT PTPRC PECAM1 VWF PDGFRA ACTA2"},
"Hepatocyte":{"core":"ADH1C ADH4 APOA5 CYP2A7 CYP2B6 CYP3A4 HPX TAT","aux":"AQP9 GATM GLYATL1 HAMP HMGCS2 KNG1 PPP1R1A RBP5 CFHR3 CFB","anti":"EPCAM KRT7 FXYD2 TM4SF4 TFF2 TMC5 PTPRC PECAM1 VWF PDGFRA ACTA2"},
"Fibroblast/Myofibroblast":{"core":"PDGFRA PDGFRB ACTA2 APCDD1","aux":"C7 ADAMTS1 CAV1 CRHBP FCN2 SRPX PMP22 TFPI TCF4","anti":"PTPRC CD3D CD3E EPCAM KRT7 PECAM1 VWF ACKR1 ADH4 APOA5 CYP2B6 CYP3A4 HPX HMGCS2 TAT"},
"Endothelial":{"core":"PECAM1 VWF ACKR1 EGFL7 CLEC14A MMRN1 MMRN2","aux":"RAMP2 SOX18 GNG11 ADGRL4 BMX DNASE1L3","anti":"PTPRC CD3D CD3E EPCAM KRT7 PDGFRA ADH4 CYP3A4 HPX"},
"Monocyte/Macrophage":{"core":"PTPRC AIF1 CD68 ADGRE1 CD163 FCGR1A MPEG1 SPI1","aux":"FCGR3A MRC1 CD14 FCN1 LILRB2 LILRB4 MS4A4A MS4A6A VSIG4 MARCO MNDA C15orf48 RETN","anti":"CD3D CD3E TRAC MS4A1 CD79A EPCAM KRT7 PECAM1 VWF PDGFRA"},
"T":{"core":"CD3D CD3E TRAC CD247","aux":"CD4 CD8A IL7R CCL5 GZMA","anti":"CD68 ADGRE1 CD163 MARCO MS4A1 CD79A EPCAM KRT7 PECAM1 VWF"},
"NK":{"core":"NKG7 GNLY KLRD1 KLRC1 PRF1","aux":"GZMB FGFBP2 KLRB1 CCL5","anti":"CD3D CD3E TRAC CD79A MS4A1 CD68 ADGRE1 EPCAM KRT7"},
"B":{"core":"MS4A1 CD79A CD19 BANK1","aux":"SPIB CD27","anti":"CD3D CD3E TRAC CD68 ADGRE1 MARCO EPCAM KRT7 PECAM1 VWF"},
"Plasma":{"core":"TNFRSF17 MZB1 DERL3","aux":"CD27","anti":"MS4A1 CD19 BANK1 CD3D CD3E CD68 ADGRE1 EPCAM KRT7"},
"Erythroid":{"core":"AHSP ALAS2 GYPA GYPB HEMGN SLC4A1","aux":"SNCA","anti":"PTPRC EPCAM KRT7 PECAM1 VWF PDGFRA ACTA2"}}
for modules in M.values():
    for key in modules: modules[key]=modules[key].split()

def lines(p): return np.asarray(p.read_text().splitlines(),dtype=object)
def module(z,gi,names): return z[:,[gi[g] for g in names]].mean(1)

def main():
    resolution=1.0
    genes,cells,samples=lines(PRE/"gene_ids.txt"),lines(PRE/"cell_ids.txt"),lines(PRE/"sample_ids.txt")
    current=lines(ACTIVE)
    required={g for x in M.values() for y in x.values() for g in y}
    missing=sorted(required-set(genes))
    if missing: raise ValueError(f"Markers absent from 377 panel: {missing}")
    x=mmread(PRE/"expression.mtx").tocsr()
    if x.shape[0]!=len(cells): x=x.T.tocsr()
    control=samples=="nondiseased"; counts=x[control].astype(np.float32)
    a=ad.AnnData(counts.copy()); a.var_names=genes.astype(str)
    sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a); sc.pp.scale(a,max_value=10)
    sc.tl.pca(a,n_comps=N_PCS,random_state=SEED)
    sc.pp.neighbors(a,n_neighbors=N_NEIGHBORS,n_pcs=N_PCS,random_state=SEED)
    sc.tl.leiden(a,resolution=resolution,random_state=SEED,key_added="leiden",
                 flavor="igraph",n_iterations=2,directed=False)
    cl=a.obs.leiden.astype(str).to_numpy(); clusters=sorted(np.unique(cl),key=int)
    membership=sparse.csr_matrix((np.ones(len(cl)),(np.arange(len(cl)),cl.astype(int))),
                                  shape=(len(cl),max(map(int,clusters))+1))
    sums=(membership.T@counts).toarray()[[int(c) for c in clusters]]
    pb=np.log1p(sums*(1e6/np.maximum(sums.sum(1,keepdims=True),1)))
    z=(pb-pb.mean(0,keepdims=True))/np.maximum(pb.std(0,keepdims=True),1e-6)
    gi={g:i for i,g in enumerate(genes)}; labels=list(M)
    score=[]; core=[]; anti=[]; hits=[]
    for label in labels:
        c=module(z,gi,M[label]["core"]); u=module(z,gi,M[label]["aux"]); n=module(z,gi,M[label]["anti"])
        score.append(c+0.5*u-0.5*n); core.append(c); anti.append(n)
        hits.append((z[:,[gi[g] for g in M[label]["core"]] ]>=CORE_Z).sum(1))
    score,core,anti,hits=map(np.column_stack,(score,core,anti,hits))
    order=np.argsort(score,axis=1); row=np.arange(len(clusters)); best=order[:,-1]
    best_score=score[row,best]; margin=best_score-score[row,order[:,-2]]
    best_core,best_anti,best_hits=core[row,best],anti[row,best],hits[row,best]
    cluster_label=np.asarray([labels[i] for i in best],object)
    clear=(best_hits>=2)&(best_score>0)&(margin>=SCORE_MARGIN)&(best_core>best_anti)
    cluster_label[~clear]="unassigned"; mapping=dict(zip(clusters,cluster_label))
    final=np.asarray([mapping[c] for c in cl],object)
    detail=pd.DataFrame({"cell_id":cells[control],"leiden_cluster":cl,"previous_annotation":current[control],"predicted_annotation":final})
    detail.to_csv(DETAILS,index=False)
    detail[["cell_id","predicted_annotation"]].rename(columns={"predicted_annotation":"group"}).to_csv(GROUPS,index=False)
    pd.DataFrame(pb,index=clusters,columns=genes).rename_axis("leiden_cluster").to_csv(PSEUDOBULK)
    diag=pd.DataFrame({"leiden_cluster":clusters,"cells":[int((cl==c).sum()) for c in clusters],
      "annotation":cluster_label,"best_score":best_score,"score_margin":margin,
      "core_mean_z":best_core,"anti_mean_z":best_anti,"enriched_core_markers":best_hits})
    for i,label in enumerate(labels): diag["score_"+label.replace("/","_")]=score[:,i]
    diag.to_csv(SCORES,index=False)
    out=current.copy(); out[control]=final; ACTIVE.write_text("\n".join(out)+"\n")
    summary={"method":"Scanpy PCA-neighbors-Leiden plus cluster-level pseudobulk",
      "genes_used":len(genes),"random_seed":SEED,"n_pcs":N_PCS,"n_neighbors":N_NEIGHBORS,
      "leiden_resolution":resolution,"n_leiden_clusters":len(clusters),
      "leiden_backend":"igraph","leiden_iterations":2,
      "pseudobulk_normalization":"log1p CPM per Leiden cluster",
      "score":"mean(core z)+0.5*mean(aux z)-0.5*mean(anti z)",
      "assignment_rule":f"2 core markers z>={CORE_Z}; score>0; margin>={SCORE_MARGIN}; core mean>anti mean",
      "unresolved_label":"unassigned",
      "annotation_counts":detail.predicted_annotation.value_counts().sort_index().to_dict(),
      "cluster_annotations":diag[["leiden_cluster","cells","annotation"]].to_dict("records"),
      "active_annotation":str(ACTIVE),"details":str(DETAILS),
      "pseudobulk":str(PSEUDOBULK),"cluster_scores":str(SCORES),"active_annotation":str(ACTIVE)}
    SUMMARY.write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
