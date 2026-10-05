#!/usr/bin/env python3
"""Evaluate receiver-source-target recovery with reproducible tie randomization."""
from pathlib import Path
import hashlib, json, os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

R = Path(os.environ['CELLATTENTION_BENCHMARK_WORKDIR'])
TAG = os.environ['HIER_VARIANT']
D = R / 'data' / TAG
O = R / 'results' / TAG
raw = O / 'misty_receiver' / 'raw_scores.csv'
if raw.exists():
    d = pd.read_csv(raw).rename(columns={'target_group': 'receiver_group'})
    d = d.groupby(['receiver_group', 'source_gene', 'target_gene'], as_index=False).score.max()
    d.to_csv(O / 'misty_receiver' / 'triplet_scores.csv', index=False)

true_groups = np.loadtxt(D / 'cell_groups.txt', dtype=int)
inferred = np.loadtxt(D / 'inferred_groups.txt', dtype=int)
cm = np.zeros((6, 6), int)
for t, p in zip(true_groups, inferred):
    cm[t, p] += 1
rr, cc = linear_sum_assignment(-cm)
mapping = {int(p): int(t) for t, p in zip(rr, cc)}
truth = pd.read_csv(D / 'truth_hierarchical.csv')
positives = {(int(x.receiver_group), x.ligand, x.target) for x in truth.itertuples()}


def tie_averaged_metrics(frame, positive_set, seed_key, repeats=1000):
    labels = np.asarray([
        int((int(x.evaluation_group), x.source_gene, x.target_gene) in positive_set)
        for x in frame.itertuples()
    ])
    p = int(labels.sum())
    if p == 0:
        return (np.nan, np.nan, np.nan, np.nan)
    scores = frame.score.to_numpy(float)
    base = np.argsort(-scores, kind='stable')
    sorted_scores = scores[base]
    cuts = np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(base)]
    tied = any(cuts[i + 1] - cuts[i] > 1 for i in range(len(cuts) - 1))
    nrep = repeats if tied else 1
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(nrep):
        order = base.copy()
        if tied:
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                if hi - lo > 1:
                    rng.shuffle(order[lo:hi])
        y = labels[order]
        ranks = np.flatnonzero(y) + 1
        hits = np.cumsum(y)
        ap = float(np.sum(hits / (np.arange(len(y)) + 1) * y) / p)
        values.append((ap, float(np.mean(1 / ranks)), float(y[:p].sum() / p), float(ranks.mean())))
    return tuple(np.mean(values, axis=0))

methods = ['cellattention', 'misty_receiver', 'misty_no_gene_role_prior',
           'holonet_adapted', 'commot_partial_adapted', 'commot_adapted']
rows = []
expected = len(mapping) * len(pd.read_csv(D / 'candidate_source_target.csv'))
for method in methods:
    path = O / method / 'triplet_scores.csv'
    if not path.exists():
        continue
    d = pd.read_csv(path)
    d['evaluation_group'] = d.receiver_group.map(mapping)
    if len(d) != expected:
        raise ValueError(f'{method}: expected {expected} candidates, got {len(d)}')
    exposed = sum((int(x.evaluation_group), x.source_gene, x.target_gene) in positives for x in d.itertuples())
    if exposed != len(positives):
        raise ValueError(f'{method}: exposed {exposed}/{len(positives)} positives')
    ap, mrr, top, mean_rank = tie_averaged_metrics(d, positives, f'{TAG}:{method}:full')
    coverage = 1.0
    covered_ap = np.nan
    if method == 'commot_partial_adapted':
        meta = json.loads((D / 'candidate_lr_partial_metadata.json').read_text())
        retained = set(meta['retained_true_sources'])
        coverage = float(meta['true_source_coverage'])
        covered_truth = {x for x in positives if x[1] in retained}
        covered = d[d.source_gene.isin(retained)].copy()
        covered_ap = tie_averaged_metrics(covered, covered_truth, f'{TAG}:{method}:covered')[0]
    rows.append((method, ap, mrr, top, mean_rank, len(d), len(positives), len(positives) / len(d), coverage, covered_ap))
summary = pd.DataFrame(rows, columns=[
    'method', 'average_precision', 'mean_reciprocal_rank', 'top_k_recall',
    'mean_true_edge_rank', 'candidate_scores', 'true_edges', 'random_ap',
    'true_source_coverage', 'covered_only_average_precision',
])
summary.to_csv(O / 'triplet_summary.csv', index=False)
(O / 'triplet_evaluation_metadata.json').write_text(json.dumps({
    'task': 'receiver-source-target recovery',
    'candidate_universe': f'{len(mapping)} inferred clusters x source ligands x targets',
    'cluster_alignment': 'Hungarian, evaluation only',
    'inferred_to_truth': mapping,
    'tie_handling': 'mean over 1000 deterministic random permutations within exact-score ties',
    'partial_lr': 'unsupported source candidates receive zero; full-universe and covered-only AP are both reported',
}, indent=2) + '\n')
print(summary.to_string(index=False))
