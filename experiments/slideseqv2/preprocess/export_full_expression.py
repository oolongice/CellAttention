#!/usr/bin/env python3
"""Export whole-transcriptome counts and aligned IDs for database validation."""
from __future__ import annotations

import argparse
from pathlib import Path
import anndata as ad
import numpy as np
from scipy import sparse
from scipy.io import mmwrite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    a = ad.read_h5ad(args.input)
    if not a.obs_names.is_unique or not a.var_names.is_unique:
        raise ValueError("whole-transcriptome cell and gene IDs must be unique")
    x = sparse.csr_matrix(a.X)
    if x.data.size and (not np.isfinite(x.data).all() or x.data.min() < 0):
        raise ValueError("whole-transcriptome expression must be finite and nonnegative")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    mmwrite(out / "full_expression.mtx", x.tocoo())
    (out / "full_cell_ids.txt").write_text("\n".join(a.obs_names.astype(str)) + "\n")
    (out / "full_gene_ids.txt").write_text("\n".join(a.var_names.astype(str)) + "\n")
    print(f"cells={a.n_obs} genes={a.n_vars} nonzeros={x.nnz}")


if __name__ == "__main__":
    main()
