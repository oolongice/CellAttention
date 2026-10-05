"""CLI for inspect-only and full leakage-safe selection."""
import argparse,json,sys
from pathlib import Path
import anndata as ad
import scanpy as sc
import pandas as pd
import yaml
from .mmc12_parser import inspect_mmc12,parse_mmc12_markers
from .gene_mapping import map_table
from .non_target_filter import filter_training_region
from .expression_metrics import compute_metrics
from .constants import ARTICLE_BENCHMARK_GENE_SETS,SOURCE_GENE_SETS,TARGET_GENE_SETS,FALLBACK_MARKERS
from .constrained_selection import build_master,select_panel,evaluate_constraints
from .reporting import write_outputs
from .relevant_detection import apply_celltype_aware_eligibility
def make_parser():
 p=argparse.ArgumentParser(); p.add_argument("--adata",required=True); p.add_argument("--markers",required=True); p.add_argument("--config",required=True); p.add_argument("--output",required=True); p.add_argument("--training-region-column"); p.add_argument("--perturbation-column"); p.add_argument("--celltype-column"); p.add_argument("--sample-column"); p.add_argument("--non-target-label",action="append"); p.add_argument("--raw-layer"); p.add_argument("--normalized-layer"); p.add_argument("--marker-sheet"); p.add_argument("--gene-column"); p.add_argument("--marker-celltype-column"); p.add_argument("--inspect-only",action="store_true"); return p
def main(argv=None):
 args=make_parser().parse_args(argv); cfg=yaml.safe_load(Path(args.config).read_text()); f=cfg["fields"]; region=args.training_region_column or f["training_region_column"]; pert=args.perturbation_column or f["perturbation_column"]; ct=args.celltype_column or f["celltype_column"]; allowed=list(f.get("unperturbed_labels",["none"]))+(args.non_target_label or f["non_target_labels"]); a=ad.read_h5ad(args.adata,backed="r")
 inspection={"shape":a.shape,"obs_columns":list(a.obs),"var_columns":list(a.var),"layers":list(a.layers),"obsm":list(a.obsm),"region_counts":a.obs[region].astype(str).value_counts().to_dict(),"perturbation_counts":a.obs[pert].astype(str).value_counts().to_dict(),"celltype_counts":a.obs[ct].astype(str).value_counts().to_dict(),"recommended":{"training_region_column":region,"perturbation_column":pert,"celltype_column":ct,"spatial_key":"spatial"},"mmc12":inspect_mmc12(args.markers)}
 out=Path(args.output); out.mkdir(parents=True,exist_ok=True); (out/"inspection.json").write_text(json.dumps(inspection,indent=2,default=str))
 if args.inspect_only: print(json.dumps(inspection,indent=2,default=str)); return 0
 safe,audit=filter_training_region(a,region,pert,allowed); audit["cell_type_counts"]=safe.obs[ct].astype(str).value_counts().to_dict(); safe.layers["_panel_counts"]=safe.X.copy(); sc.pp.normalize_total(safe,target_sum=1e4); sc.pp.log1p(safe); safe.layers["_panel_log1p"]=safe.X.copy()
 mt,md,pr=parse_mmc12_markers(args.markers,args.marker_sheet,args.gene_column,args.marker_celltype_column); maps=[map_table(mt.gene_symbol,safe.var_names,args.markers,"mmc12")]
 for category,sets in [("article",ARTICLE_BENCHMARK_GENE_SETS),("source",SOURCE_GENE_SETS),("target",TARGET_GENE_SETS),("fallback",FALLBACK_MARKERS)]:
  for sub,genes in sets.items(): maps.append(map_table(genes,safe.var_names,"constants.py",category+":"+sub))
 mapping=pd.concat(maps,ignore_index=True).drop_duplicates(); marker_map=mapping[mapping.source_category=="mmc12"].drop_duplicates("original_gene").set_index("original_gene"); mt["matched_expression_gene"]=mt.gene_symbol.map(marker_map.matched_expression_gene).fillna(""); mt["mapping_status"]=mt.gene_symbol.map(marker_map.status).fillna("absent_from_matrix")
 metrics,dets,means=compute_metrics(safe,ct,"_panel_counts","_panel_log1p",args.sample_column or f.get("sample_column"),cfg["thresholds"]["min_cells_per_type_variability"]); master=build_master(metrics,dets,mt,mapping); master=apply_celltype_aware_eligibility(master,dets,safe.obs[ct].astype(str).value_counts().to_dict(),cfg["thresholds"]); selected,master=select_panel(master,cfg); unmet=evaluate_constraints(selected,cfg); cmd="python -m gene_panel.cli "+" ".join(sys.argv[1:]); write_outputs(out,selected,master,mapping,mt,pr,audit,cfg,unmet,cmd); print(json.dumps({"selected":len(selected),"output":str(out),"unmet_constraints":int((~unmet.met).sum())},indent=2)); return 2 if len(selected)<cfg["panel_size"] else 0
if __name__=="__main__": raise SystemExit(main())
