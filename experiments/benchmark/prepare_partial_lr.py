#!/usr/bin/env python3
"""Create a controlled incomplete LR database for a generated dataset."""
from pathlib import Path
import argparse, json
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('data_dir', type=Path)
a = p.parse_args()
d = a.data_dir
truth = pd.read_csv(d / 'truth_hierarchical.csv')
full = pd.read_csv(d / 'candidate_lr.csv')
true_sources = sorted(truth.ligand.unique())
# Deterministic, seed-independent alternating selection; floor(n/2) sources are withheld.
retained = true_sources[::2]
withheld = [x for x in true_sources if x not in retained]
partial = full[full.ligand.isin(retained)].copy()
partial.to_csv(d / 'candidate_lr_partial.csv', index=False)
meta = {
    'design': 'controlled source-level LR database ablation',
    'selection': 'retain alternating sorted true source ligands and all of their candidate receptors',
    'truth_used_only_to_construct_ablation': True,
    'retained_true_sources': retained,
    'withheld_true_sources': withheld,
    'true_source_coverage': len(retained) / len(true_sources),
    'full_lr_pairs': len(full),
    'partial_lr_pairs': len(partial),
}
(d / 'candidate_lr_partial_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
