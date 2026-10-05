#!/usr/bin/env python3
import os
"""Join existing V2 source-conditioned targets to existing CRISPR DGE results."""

SOURCE_GENES = ["Icam1", "Cxcr4"]

from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(os.environ["CELLATTENTION_CRISPR_WORKDIR"]) / "visualization"
DATASET = HERE.parent
MODEL = DATASET / "analysis/source_focused_immune/source_target_receiver_results.csv"
DGE = DATASET / "data/preprocessed/immune_cells_spatial_DGE_significant.csv"
OUTPUT = HERE / "data/source_conditioned_crispr_validation.csv"
SHAPE_OVERRIDES = {"Icam1": "Icam_shapes"}

model = pd.read_csv(MODEL)
model = model.loc[
    model.receiver_cell_type.eq("Immune_all") & model.source_gene.isin(SOURCE_GENES),
    ["source_gene", "target_gene", "receiver_cell_count",
     "physical_rank_within_source_receiver", "heldout_mse_reduction_physical",
     "heldout_correlation_physical", "effect_median_physical",
     "direction_consistency_physical", "training_improvement_fraction"
     ] if "training_improvement_fraction" in model.columns else
    ["source_gene", "target_gene", "receiver_cell_count",
     "physical_rank_within_source_receiver", "heldout_mse_reduction_physical",
     "heldout_correlation_physical", "effect_median_physical",
     "direction_consistency_physical"]
].copy()
dge = pd.read_csv(DGE).reset_index(names="dge_input_order")
parts = []
available = set(model.source_gene.unique())
missing_model = sorted(set(SOURCE_GENES) - available)
if missing_model:
    raise ValueError(f"Sources absent from existing source-focused model results: {missing_model}")
for source in SOURCE_GENES:
    shape = SHAPE_OVERRIDES.get(source, f"{source}_shapes")
    if shape not in set(dge.target_shape):
        raise ValueError(f"No CRISPR DGE shape found for {source}: expected {shape}")
    part = dge.loc[dge.target_shape.eq(shape)].copy()
    part["source_gene"] = source
    part["experimental_rank_within_source"] = np.arange(1, len(part) + 1)
    parts.append(part.rename(columns={"names": "target_gene", "logfoldchanges": "crispr_logfc",
                                     "pvals_adj": "crispr_fdr", "scores": "crispr_score"}))
experimental = pd.concat(parts, ignore_index=True)
experimental = experimental[["source_gene", "target_gene", "crispr_logfc", "crispr_fdr",
                             "crispr_score", "experimental_rank_within_source"]]
result = model.merge(experimental, on=["source_gene", "target_gene"], how="left")
result["experimental_supported"] = result.crispr_fdr.notna()
result["direction_concordant"] = result.experimental_supported & (
    np.sign(result.effect_median_physical) == -np.sign(result.crispr_logfc)
)
result["experimental_evidence"] = np.where(
    result.experimental_supported, -np.log10(result.crispr_fdr.clip(lower=1e-300)), np.nan
)
result = result.sort_values(["source_gene", "physical_rank_within_source_receiver"])
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)
print(f"rows={len(result)}")
print(f"experimental_supported={int(result.experimental_supported.sum())}")
print(f"direction_concordant={int(result.direction_concordant.sum())}")
print(f"output={OUTPUT}")
