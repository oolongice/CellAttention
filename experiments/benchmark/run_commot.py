#!/usr/bin/env python3
"""COMMOT LR-to-target adapter with full or controlled partial LR coverage."""
from pathlib import Path
import json, os
import numpy as np
import pandas as pd
import anndata as ad
import commot as ct

ROOT = Path(os.environ["CELLATTENTION_BENCHMARK_WORKDIR"])
TAG = os.environ['HIER_VARIANT']
MODE = os.environ.get('COMMOT_LR_MODE', 'full')
if MODE not in {'full', 'partial'}:
    raise ValueError(f'unknown COMMOT_LR_MODE={MODE}')
D = ROOT / 'data' / TAG
method = 'commot_adapted' if MODE == 'full' else 'commot_partial_adapted'
O = ROOT / 'results' / TAG / method
O.mkdir(parents=True, exist_ok=True)
genes = (D / 'gene_ids.txt').read_text().splitlines()
x = np.loadtxt(D / 'expression.csv', delimiter=',')
xy = np.loadtxt(D / 'spatial_coordinates.csv', delimiter=',')
groups = np.loadtxt(D / 'inferred_groups.txt', dtype=int)
adata = ad.AnnData(x)
adata.var_names = genes
adata.obsm['spatial'] = xy
lr_file = 'candidate_lr.csv' if MODE == 'full' else 'candidate_lr_partial.csv'
lr = pd.read_csv(D / lr_file)
db = lr.rename(columns={'ligand': 'source_gene', 'receptor': 'target_gene'}).assign(pathway='synthetic_candidates')
ct.tl.spatial_communication(
    adata, database_name='hierarchical',
    df_ligrec=db[['source_gene', 'target_gene', 'pathway']],
    dis_thr=320., heteromeric=False,
)
incoming = np.asarray([
    np.asarray(adata.obsp[f'commot-hierarchical-{p.source_gene}-{p.target_gene}'].sum(axis=0)).ravel()
    for p in db.itertuples()
]).T
all_sources = sorted(pd.read_csv(D / 'candidate_source_target.csv').source_gene.unique())
targets = [g for g in genes if g.startswith('target_')]
supported = set(db.source_gene)
rows = []
for rg in sorted(set(groups)):
    idx = groups == rg
    for ligand in all_sources:
        pair_idx = np.flatnonzero(db.source_gene.to_numpy() == ligand)
        for target in targets:
            if ligand not in supported:
                score = 0.0
            else:
                y = x[idx, genes.index(target)]
                values = []
                for k in pair_idx:
                    z = incoming[idx, k]
                    values.append(max(0, float(np.corrcoef(z, y)[0, 1])) if np.std(z) > 1e-9 and np.std(y) > 1e-9 else 0)
                score = max(values)
            rows.append((rg, ligand, target, score))
triplet = pd.DataFrame(rows, columns=['receiver_group', 'source_gene', 'target_gene', 'score'])
triplet.to_csv(O / 'triplet_scores.csv', index=False)
triplet.groupby(['receiver_group', 'target_gene'], as_index=False).score.max().to_csv(O / 'scores.csv', index=False)
meta = {
    'method': f'COMMOT {MODE} LR-to-target adapted',
    'communication_event': 'native pair-specific collective OT',
    'downstream_score': 'maximum positive marginal correlation over available candidate receptors',
    'LR_prior': lr_file,
    'unsupported_source_score': 0.0,
    'truth_used_for_scoring': False,
}
if MODE == 'partial':
    meta.update(json.loads((D / 'candidate_lr_partial_metadata.json').read_text()))
(O / 'method_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
