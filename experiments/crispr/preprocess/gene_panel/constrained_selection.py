"""Auditable deterministic constrained selection."""
import numpy as np
import pandas as pd
from .constants import ARTICLE_BENCHMARK_GENE_SETS,SOURCE_GENE_SETS,TARGET_GENE_SETS,FALLBACK_MARKERS,inverted_categories,source_mode
def z(s):
 s=pd.Series(s,dtype=float); q=s.clip(s.quantile(.01),s.quantile(.99)); sd=q.std(); return (q-q.mean())/(sd if sd and np.isfinite(sd) else 1)
def build_master(metrics,det_by_type,marker_table,mapping):
 m=metrics.copy(); m.index.name="gene"; a=inverted_categories(ARTICLE_BENCHMARK_GENE_SETS); s=inverted_categories(SOURCE_GENE_SETS); t=inverted_categories(TARGET_GENE_SETS); f=inverted_categories(FALLBACK_MARKERS); good=mapping[mapping.status=="matched"]; raw_to=dict(zip(good.original_gene,good.matched_expression_gene)); mm={}
 for _,r in marker_table.iterrows():
  g=raw_to.get(r.gene_symbol)
  if g: mm.setdefault(g,set()).add(r.cell_type_standardized)
 m["article_categories"]=[";".join(a.get(g,[])) for g in m.index]; m["article_anchor"]=m.article_categories.ne(""); m["source_categories"]=[";".join(s.get(g,[])) for g in m.index]; m["source_candidate"]=m.source_categories.ne(""); m["source_mode"]=[source_mode(g) if g in s else "" for g in m.index]; m["target_categories"]=[";".join(t.get(g,[])) for g in m.index]; m["target_candidate"]=m.target_categories.ne(""); m["mmc12_celltypes"]=[";".join(sorted(mm.get(g,()))) for g in m.index]; m["mmc12_marker"]=m.mmc12_celltypes.ne(""); m["fallback_celltypes"]=[";".join(f.get(g,[])) for g in m.index]; m["fallback_marker"]=m.fallback_celltypes.ne("")
 m["article_detection_pass"]=m.detection_rate.ge(.005); m["quality_pass"]=m.detection_rate.ge(.01)&~m.technical_flag; m["not_evaluable_reason"]=np.where(m.article_anchor&~m.article_detection_pass,"below_article_anchor_detection_threshold",np.where(m.technical_flag,"technical_feature",""))
 m["score_article"]=100+z(m.detection_rate)+z(m.mean_expression); m["score_marker"]=20+5*m.mmc12_marker+z(m.celltype_specificity)+z(m.max_celltype_detection)+z(m.batch_consistency); mb=m.source_mode.map({"diffusible":4,"secreted_or_matrix_bound":3,"contact_associated":2,"uncertain":1}).fillna(0); m["score_source"]=30+mb+z(m.max_celltype_detection)+z(m.celltype_specificity)+z(m.mean_expression); m["score_target"]=30+z(m.max_celltype_detection)+z(m.within_type_variability)+z(m.batch_consistency); m["score_data"]=.35*z(m.celltype_specificity)+.25*z(m.within_type_variability)+.2*z(m.detection_rate)+.2*z(m.spatial_variability.fillna(m.spatial_variability.median()))
 prior=m.article_anchor|m.mmc12_marker|m.source_candidate|m.target_candidate|m.fallback_marker; m["stable_negative_score"]=z(m.detection_rate)-z(m.within_type_variability)-z(m.celltype_specificity); m.loc[prior|m.technical_flag,"stable_negative_score"]=-np.inf; return m
def select_panel(master,config):
 n=int(config["panel_size"]); q=config["quotas"]; selected=[]; role={}; reason={}
 def add(frame,count,r,score):
  used=0
  for g in frame.sort_values([score,"detection_rate"],ascending=False).index:
   if used>=count or len(selected)>=n: break
   if g not in role: selected.append(g); role[g]=r; reason[g]="selected for "+r+" by "+score; used+=1
 e=master[~master.technical_flag]; add(e[e.article_anchor&e.article_detection_pass],q["article"],"article_anchor","score_article"); add(e[e.mmc12_marker&e.quality_pass],q["marker"],"mmc12_marker","score_marker"); add(e[e.source_candidate&e.quality_pass],q["source"],"source_candidate","score_source"); add(e[e.target_candidate&e.quality_pass],q["target"],"immune_target","score_target"); add(e[e.quality_pass],q["data_driven"],"data_driven","score_data"); add(e[np.isfinite(e.stable_negative_score)],q["stable_negative"],"stable_negative_control","stable_negative_score")
 for g in e[e.quality_pass].sort_values(["score_data","detection_rate"],ascending=False).index:
  if len(selected)>=n: break
  if g not in role: selected.append(g); role[g]="quality_fill"; reason[g]="high-quality non-target data fill"
 out=master.loc[selected].copy(); out["primary_role"]=[role[g] for g in selected]; out["selection_reason"]=[reason[g] for g in selected]; out["all_roles"]=[";".join(x for x,b in [("article_anchor",r.article_anchor),("mmc12_marker",r.mmc12_marker),("fallback_marker",r.fallback_marker),("source_candidate",r.source_candidate),("immune_target",r.target_candidate),("stable_negative_control",role[g]=="stable_negative_control")] if b) for g,r in out.iterrows()]; scorecols={"article_anchor":"score_article","mmc12_marker":"score_marker","source_candidate":"score_source","immune_target":"score_target","data_driven":"score_data","stable_negative_control":"stable_negative_score","quality_fill":"score_data"}; out["selection_score"]=[out.loc[g,scorecols[role[g]]] for g in selected]; out["priority"]=out.primary_role.map({"article_anchor":0,"mmc12_marker":1,"source_candidate":2,"immune_target":3,"data_driven":4,"stable_negative_control":5,"quality_fill":6}); out.insert(0,"rank",range(1,len(out)+1)); master["selected"]=master.index.isin(selected); master["rejection_reason"]=np.where(master.selected,"",np.where(master.technical_flag,"technical_feature",np.where(~master.quality_pass,"below_detection_threshold","quota_or_lower_score"))); return out,master
def evaluate_constraints(selected,config):
 checks={"direct_communication":int(selected.article_categories.str.contains("direct_communication").sum()),"diffusible_source":int((selected.source_mode=="diffusible").sum()),"source_total":int(selected.source_candidate.sum()),"target_total":int(selected.target_candidate.sum()),"ifn_response":int(selected.target_categories.str.contains("ifn_response").sum()),"exhaustion":int(selected.target_categories.str.contains("exhaustion").sum()),"memory_stemness":int(selected.target_categories.str.contains("memory_stemness").sum()),"migration":int(selected.target_categories.str.contains("migration").sum()),"stable_negative":int((selected.primary_role=="stable_negative_control").sum())}; rows=[]
 for name,target in config["constraints"].items():
  if name.startswith("marker_"): continue
  actual=checks.get(name,0); rows.append({"constraint":name,"target":target,"actual":actual,"met":actual>=target,"reason":"" if actual>=target else "insufficient detectable eligible genes or finite prior set"})
 return pd.DataFrame(rows)
