"""Region-first filtering with guide-level leakage guard."""
def truthy(s): return s.astype(str).str.strip().str.lower().isin({"true","1","yes"})
def assert_no_target_leakage(adata,perturbation_column="target_gene",allowed_labels=("none","non-targeting")):
 allowed={str(x).lower() for x in allowed_labels}; bad=sorted(set(adata.obs[perturbation_column].astype(str).str.lower())-allowed)
 if bad: raise ValueError(f"Target perturbation leakage detected: {bad[:10]}")
def filter_training_region(adata,region_column="control_training_shapes",perturbation_column="target_gene",allowed_labels=("none","non-targeting")):
 region=truthy(adata.obs[region_column]); labels=adata.obs[perturbation_column].astype(str).str.strip().str.lower(); allowed={str(x).lower() for x in allowed_labels}; safe=region&labels.isin(allowed)
 audit={"n_before":int(adata.n_obs),"n_in_training_region":int(region.sum()),"n_target_guides_excluded":int((region&~labels.isin(allowed)).sum()),"n_after":int(safe.sum()),"region_column":region_column,"perturbation_column":perturbation_column,"allowed_labels":sorted(allowed),"region_label_counts":labels[region].value_counts().to_dict()}
 out=adata[safe].to_memory() if getattr(adata,"isbacked",False) else adata[safe].copy(); assert_no_target_leakage(out,perturbation_column,allowed); return out,audit
