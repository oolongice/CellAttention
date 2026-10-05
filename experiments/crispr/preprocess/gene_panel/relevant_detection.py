"""Generic cell-type-aware evaluability for biological priors.

The rules operate on roles and broad cell groups, never on perturbation outcomes
or a run-specific list of desired genes.
"""
import numpy as np
import pandas as pd

def _rows(index,patterns):
 return [x for x in index if any(p.lower() in str(x).lower() for p in patterns)]
def _aggregate_detection(det_by_type,cell_counts,rows):
 if not rows: return pd.DataFrame({"rate":0.0,"n":0},index=det_by_type.columns)
 counts=pd.Series({r:cell_counts.get(r,0) for r in rows},dtype=float); total=counts.sum()
 detected=det_by_type.loc[rows].mul(counts,axis=0).sum(axis=0)
 return pd.DataFrame({"rate":detected/max(total,1),"n":detected.round().astype(int)})
def apply_celltype_aware_eligibility(master,det_by_type,cell_counts,rules=None):
 """Annotate relevant-group detection and upgrade eligible biological priors."""
 rules=rules or {}; relevant_rate=float(rules.get("relevant_min_detection_rate",.01)); relevant_n=int(rules.get("relevant_min_detected_cells",10)); marker_rate=float(rules.get("relevant_marker_min_detection_rate",.02)); marker_n=int(rules.get("relevant_marker_min_detected_cells",5)); override_global=float(rules.get("multi_axis_override_global_detection",.002)); override_n=int(rules.get("multi_axis_override_min_detected_cells",5)); anchor_global=float(rules.get("article_anchor_min_detection_rate",.005))
 groups={
  "T_NK":_aggregate_detection(det_by_type,cell_counts,[x for x in det_by_type.index if str(x).lower() in {"t","nk","t_cell","cd8_t"}]),
  "Macrophage_Monocyte":_aggregate_detection(det_by_type,cell_counts,[x for x in det_by_type.index if ("macroph" in str(x).lower() or "mono" in str(x).lower())]),
  "Tumor_Epithelial":_aggregate_detection(det_by_type,cell_counts,[x for x in det_by_type.index if str(x).lower() in {"malignant","tumor","epithelial"}]),
 }
 relevant=[]; rates=[]; nums=[]
 for gene,row in master.iterrows():
  ac=str(row.article_categories); tc=str(row.target_categories); mc=str(row.mmc12_celltypes)
  if row.source_mode=="contact_associated":
   choices=["Tumor_Epithelial"]
  elif "cd44_tcell_program" in ac or ("T_cell" in mc) or (row.target_candidate and "macrophage_response" not in tc):
   choices=["T_NK"]
  elif "macrophage_program" in ac or "Macrophage" in mc or "macrophage_response" in tc:
   choices=["Macrophage_Monocyte"]
  elif "Tumor" in mc or "Epithelial" in mc:
   choices=["Tumor_Epithelial"]
  elif row.source_candidate:
   choices=["Macrophage_Monocyte","Tumor_Epithelial"]
  elif row.article_anchor:
   choices=["T_NK","Macrophage_Monocyte","Tumor_Epithelial"]
  else:
   choices=["T_NK","Macrophage_Monocyte","Tumor_Epithelial"]
  best=max(choices,key=lambda x:groups[x].loc[gene,"rate"]); relevant.append(best); rates.append(float(groups[best].loc[gene,"rate"])); nums.append(int(groups[best].loc[gene,"n"]))
 master["relevant_celltype"]=relevant; master["max_detection_in_relevant_celltype"]=rates; master["n_detected_in_relevant_celltype"]=nums
 master["detection_in_T_NK"]=groups["T_NK"]["rate"]; master["detection_in_macrophage_monocyte"]=groups["Macrophage_Monocyte"]["rate"]; master["detection_in_tumor"]=groups["Tumor_Epithelial"]["rate"]
 rule_b=(master.max_detection_in_relevant_celltype>=relevant_rate)&(master.n_detected_in_relevant_celltype>=relevant_n)
 rule_c=master.mmc12_marker&(master.max_detection_in_relevant_celltype>=marker_rate)&(master.n_detected_in_relevant_celltype>=marker_n)
 # Generic rescue for anchors supported by three independent prior axes.
 multi_axis=master.article_anchor&master.source_candidate&master.target_candidate&(master.detection_rate>=override_global)&(master.max_detection_in_relevant_celltype>=relevant_rate)&(master.n_detected_in_relevant_celltype>=override_n)
 master["ultra_sparse_anchor"]=multi_axis&~(master.detection_rate>=anchor_global)&~rule_b&~rule_c
 master["celltype_aware_pass"]=rule_b|rule_c|multi_axis
 master["article_detection_pass"]=(master.detection_rate>=anchor_global)|master.celltype_aware_pass
 prior=master.article_anchor|master.mmc12_marker|master.source_candidate|master.target_candidate
 master["not_evaluable_reason"]=np.where(master.article_anchor&~master.article_detection_pass,"insufficient_global_and_relevant_celltype_detection",np.where(master.technical_flag,"technical_feature",""))
 return master
