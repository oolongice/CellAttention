#!/usr/bin/env python3
"""Compare immune cells in each perturbation shape with non-targeting control cells."""
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

IMMUNE = ("Macrophages", "B", "T", "DC", "Neutrophils", "MAST", "Plasma", "NK")
CONTROL_COLUMN = "control_nontargeting_shapes"


def as_bool(series):
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.to_numpy(dtype=bool)
    return series.astype(str).str.lower().isin(("true", "1", "yes")).to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw SPAC-seq H5AD")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = ad.read_h5ad(args.input, backed="r")
    shapes = [name for name in raw.obs.columns
              if str(name).endswith("_shapes") and not str(name).startswith("control_")]
    immune = raw.obs["cell_annotation"].isin(IMMUNE).to_numpy()
    control = as_bool(raw.obs[CONTROL_COLUMN])
    active = control.copy()
    for shape in shapes:
        active |= as_bool(raw.obs[shape])
    active &= immune
    data = raw[active, :].to_memory()
    raw.file.close()
    # Scanpy's Wilcoxon scores are computed on log-normalized expression.
    sc.pp.normalize_total(data, target_sum=10_000)
    sc.pp.log1p(data)
    control = control[active]
    rows = []
    for shape in shapes:
        perturbed = as_bool(data.obs[shape])
        selected = perturbed | control
        subset = data[selected, :].copy()
        subset.obs["condition"] = np.where(perturbed[selected], "shape", "control")
        sc.tl.rank_genes_groups(subset, "condition", groups=["shape"],
                                reference="control", method="wilcoxon", use_raw=False)
        frame = sc.get.rank_genes_groups_df(subset, "shape")
        frame["target_shape"] = shape
        rows.append(frame)
        print(f"{shape}: immune_shape={int(perturbed.sum())} immune_control={int(control.sum())}", flush=True)
    complete = pd.concat(rows, ignore_index=True)
    significant = complete.loc[(complete.pvals_adj < 0.05)
                               & (complete.logfoldchanges.abs() > 0.5)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete.to_csv(args.output_dir / "immune_cells_spatial_DGE_all_shapes.csv", index=False)
    significant.to_csv(args.output_dir / "immune_cells_spatial_DGE_significant.csv", index=False)
    print(f"all_rows={len(complete)} significant_rows={len(significant)}")


if __name__ == "__main__":
    main()
