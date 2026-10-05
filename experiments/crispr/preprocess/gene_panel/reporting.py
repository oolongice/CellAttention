"""Output tables, figures, summaries and reproducibility report."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import yaml
def write_outputs(out,selected,master,mapping,markers,parse_report,audit,config,unmet,command):
 out=Path(out); fig=out/"figures"; fig.mkdir(parents=True,exist_ok=True); selected.reset_index().to_csv(out/"selected_genes_500.csv",index=False); (out/"selected_genes_500.txt").write_text("\n".join(selected.index)+"\n"); master.reset_index().to_csv(out/"all_gene_candidates.csv",index=False); prior=master.article_anchor|master.mmc12_marker|master.source_candidate|master.target_candidate; master[prior&~master.selected].reset_index().to_csv(out/"rejected_priority_genes.csv",index=False); mapping.to_csv(out/"gene_mapping_report.csv",index=False); markers.to_csv(out/"mmc12_parsed_markers.csv",index=False)
 pd.DataFrame([{"sheet_name":x["sheet_name"],"n_rows":x["n_rows"],"n_columns":x["n_columns"],"columns":";".join(x["columns"]),"non_null":json.dumps(x["non_null"]),"selected_mode":x.get("selected_mode","")} for x in parse_report["sheets"]]).to_csv(out/"mmc12_parsing_report.csv",index=False)
 cov=[]
 for ct,g in markers.groupby("cell_type_standardized"):
  genes=set(g.matched_expression_gene[g.mapping_status=="matched"]); cov.append({"cell_type":ct,"mmc12_marker_total":len(g),"present_in_matrix":len(genes),"passed_detection":int(master.reindex(list(genes)).quality_pass.fillna(False).sum()),"selected":len(genes&set(selected.index)),"fallback":int(selected.fallback_celltypes.str.contains(ct,regex=False).sum())})
 pd.DataFrame(cov).to_csv(out/"celltype_marker_coverage.csv",index=False); cats=[]
 for name,col in [("article","article_anchor"),("marker","mmc12_marker"),("source","source_candidate"),("target","target_candidate")]: cats.append({"category":name,"target":config["quotas"].get(name,0),"actual":int(selected[col].sum())})
 cats.append({"category":"stable_negative","target":config["quotas"]["stable_negative"],"actual":int((selected.primary_role=="stable_negative_control").sum())}); pd.DataFrame(cats).to_csv(out/"category_coverage.csv",index=False); unmet.to_csv(out/"unmet_constraints.tsv",sep="\t",index=False); (out/"gene_selection_config_used.yaml").write_text(yaml.safe_dump(config,sort_keys=False))
 summary={"panel_size":len(selected),"requested_panel_size":config["panel_size"],"audit":audit,"unmet_constraints":int((~unmet.met).sum()),"leakage_prevention":"Only control_training_shapes cells with target_gene none/non-targeting were used."}; (out/"selection_summary.json").write_text(json.dumps(summary,indent=2,default=str))
 selected.primary_role.value_counts().plot.bar(); plt.ylabel("genes"); plt.tight_layout(); plt.savefig(fig/"role_composition.png",dpi=160); plt.close()
 sm=selected.source_mode.replace("",pd.NA).dropna()
 if len(sm): sm.value_counts().plot.bar(); plt.tight_layout(); plt.savefig(fig/"source_mode_composition.png",dpi=160); plt.close()
 ax=master.plot.scatter("detection_rate","celltype_specificity",c=master.selected.map({True:"tab:red",False:"0.7"}),s=5); ax.set_xscale("log"); plt.tight_layout(); plt.savefig(fig/"specificity_vs_detection.png",dpi=160); plt.close()
 article=", ".join(master.index[master.article_anchor&~master.article_detection_pass]) or "none"; text="# SPAC-seq gene selection report\n\n## Safe subset\n\nTraining region: "+audit["region_column"]+". Cells: "+f'{audit["n_before"]:,}'+" total -> "+f'{audit["n_in_training_region"]:,}'+" in region -> "+f'{audit["n_after"]:,}'+" after excluding "+str(audit["n_target_guides_excluded"])+" target-guide cells.\n\n## Panel\n\nSelected "+str(len(selected))+" unique genes. Unmet constraints: "+str(int((~unmet.met).sum()))+". Article genes below evaluability: "+article+".\n\n## Leakage prevention\n\nNo target-region DEG, fold change, enrichment, or validation result was used.\n\n## Reproduction\n\n"+command+"\n"; (out/"gene_selection_report.md").write_text(text)
