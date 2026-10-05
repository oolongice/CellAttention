#!/usr/bin/env python3
from pathlib import Path
import os
import pandas as pd
import plot_benchmark_metric as plotting

ROOT = Path(os.environ['CELLATTENTION_BENCHMARK_WORKDIR'])
SEEDS = range(int(os.environ.get('CELLATTENTION_GENERATION_SEED_START', '168')), int(os.environ.get('CELLATTENTION_GENERATION_SEED_END', '172')) + 1)
SCENARIOS = tuple(os.environ.get('CELLATTENTION_SCENARIOS', 'simple,complex_lr,spatial_overlap').split(','))
LABELS = ('Simple 1:1', 'Complex LR', 'Overlapping fields')

frames = []
for scenario in SCENARIOS:
    for seed in SEEDS:
        path = ROOT / 'results' / f'scenario_{scenario}_seed_{seed}' / 'triplet_summary.csv'
        frame = pd.read_csv(path)
        frame.insert(0, 'generation_seed', seed)
        frame.insert(0, 'difficulty', scenario)
        frames.append(frame)

detail = pd.concat(frames, ignore_index=True)
out = ROOT / 'tables' / 'scenario_benchmark'
out.mkdir(parents=True, exist_ok=True)
detail.to_csv(out / 'receiver_source_target_results_by_seed.csv', index=False)
metrics = ['average_precision', 'mean_reciprocal_rank', 'top_k_recall', 'mean_true_edge_rank', 'true_source_coverage', 'covered_only_average_precision']
summary = detail.groupby(['difficulty', 'method'], as_index=False)[metrics].agg(['mean', 'std'])
summary.columns = ['_'.join(x).rstrip('_') for x in summary.columns]
summary.to_csv(out / 'receiver_source_target_summary_across_seeds.csv', index=False)
partial = detail[detail.method == 'commot_partial_adapted']
coverage = partial.groupby('difficulty', as_index=False).agg(
    true_source_coverage_mean=('true_source_coverage', 'mean'),
    full_universe_ap_mean=('average_precision', 'mean'),
    full_universe_ap_std=('average_precision', 'std'),
    covered_only_ap_mean=('covered_only_average_precision', 'mean'),
    covered_only_ap_std=('covered_only_average_precision', 'std'),
)
full = detail[detail.method == 'commot_adapted'].groupby('difficulty', as_index=False).agg(
    full_lr_ap_mean=('average_precision', 'mean'), full_lr_ap_std=('average_precision', 'std'))
coverage.merge(full, on='difficulty').to_csv(out / 'commot_lr_coverage_summary.csv', index=False)

plotting.DIFFICULTIES = list(SCENARIOS)
plotting.PARAMETER_LABELS = list(LABELS)
items = ['cellattention', 'misty_no_gene_role_prior', 'misty_receiver', None, None,
         'holonet_adapted', 'commot_partial_adapted', 'commot_adapted']
figures = ROOT / 'figures'
figures.mkdir(exist_ok=True)
baselines = detail.groupby('difficulty').random_ap.mean().to_dict()
for metric, label, filename in (
    ('average_precision', 'Average precision', 'receiver_source_target_average_precision.png'),
    ('top_k_recall', 'Top-k recall', 'receiver_source_target_top_k_recall.png'),
):
    plotting.plot_metric_by_difficulty(
        detail, items, metric, label, 'Receiver–source–target recovery', figures / filename,
        None,
    )
print(summary.to_string(index=False))
