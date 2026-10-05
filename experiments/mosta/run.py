#!/usr/bin/env python3
"""Prepare, analyze, and render the MOSTA Figure 5 results."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
RAW_FILES = ('cell_name.txt', 'gene_list.txt', 'meta_annotation.txt', 'meta_batch.txt',
             'gene_expression.npz', 'spatial_coordinates.npy')
PLOTS = ('plot_mosta_spatial_overview.py', 'plot_cluster_target_ring.py',
         'plot_brain_development_umap.py', 'plot_brain_development_boundary.py',
         'plot_brain_transition_tis.py', 'plot_brain_transition_source_network.py',
         'plot_brain_transition_source_field.py', 'plot_brain_transition_marker_DE.py',
         'plot_brain_wnt3a_influence_umap.py',
         'plot_omnipath_brain_pathway_individual_merged.py')


def link(source, dest):
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink() and dest.resolve() == source:
        return
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(dest)
    dest.symlink_to(source)


def environment(args):
    env = os.environ.copy()
    env.update(CELLATTENTION_MOSTA_WORKDIR=str(args.workdir), PYTHONDONTWRITEBYTECODE='1',
               MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMBA_NUM_THREADS=str(args.threads),
               RAYON_NUM_THREADS=str(args.threads), PYTHONPATH=str(CODE))
    return env


def run(args, command, label):
    log = args.workdir / 'logs' / f'{label}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as output:
        subprocess.run(command, cwd=ROOT, env=environment(args), stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    print(f'{label}: {log}', flush=True)


def config(args, name, dest):
    value = json.loads((CODE / name).read_text().replace('${WORKDIR}', str(args.workdir)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(value, indent=2) + '\n')
    return dest


def binary(args, name):
    path = ROOT / 'target/release' / name
    if not path.is_file():
        run(args, ['cargo', 'build', '--release', '--locked', '--bin', name], f'build_{name}')
    return str(path)


def prepare(args):
    raw = args.workdir / 'data/raw'
    if args.raw_dir:
        for name in RAW_FILES:
            link(args.raw_dir / name, raw / name)
    for name in RAW_FILES:
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    pre = args.workdir / 'data/preprocessed'
    run(args, [args.python, str(CODE / 'data/prepare.py'), '--raw-dir', str(raw),
               '--output-dir', str(pre)], 'prepare')
    print(f'prepared={pre}')


def checkpoint(args):
    path = args.checkpoint_dir or args.workdir / 'model/transformer'
    if args.checkpoint_dir:
        dest = args.workdir / 'model/transformer'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() and dest.resolve() == path:
            pass
        elif dest.exists() or dest.is_symlink():
            raise FileExistsError(f'checkpoint path already exists: {dest}')
        else:
            dest.symlink_to(path, target_is_directory=True)
    for name in ('model.mpk', 'cell_ids.txt', 'gene_ids.txt', 'cell_embeddings.csv'):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    for name in ('cell_ids.txt', 'gene_ids.txt'):
        pre = args.workdir / 'data/preprocessed' / name
        if pre.is_file() and pre.read_bytes() != (path / name).read_bytes():
            raise ValueError(f'checkpoint {name} differs from prepared input')
    return path


def train(args):
    dest = args.workdir / 'model/train_config.json'
    cfg = json.loads((CODE / 'train_config.template.json').read_text().replace('${WORKDIR}', str(args.workdir)))
    cfg['device_index'] = args.device_index
    cfg['training']['epochs'] = args.epochs
    if (args.workdir / 'model/transformer/model.mpk').exists():
        if not dest.is_file() or json.loads(dest.read_text()) != cfg:
            raise ValueError('existing checkpoint has a different or unknown training configuration')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cfg, indent=2) + '\n')
    run(args, [binary(args, 'cell_attention'), str(dest)], 'train')


def groups(args):
    dest = config(args, 'analysis/group_count_config.template.json',
                  args.workdir / 'analysis/receiver_group_count_selection/config.json')
    cfg = json.loads(dest.read_text()); cfg['checkpoint_dir'] = str(checkpoint(args))
    dest.write_text(json.dumps(cfg, indent=2) + '\n')
    run(args, [binary(args, 'select_receiver_group_count'), str(dest)], 'groups')


def analyze(args):
    dest = config(args, 'analysis/source_target_config.template.json',
                  args.workdir / 'analysis/config.json')
    cfg = json.loads(dest.read_text()); cfg['checkpoint_dir'] = str(checkpoint(args))
    dest.write_text(json.dumps(cfg, indent=2) + '\n')
    run(args, [binary(args, 'run_source_target_analysis'), str(dest)], 'analyze')
    summarize(args)


def summarize(args):
    run(args, [args.python, str(CODE / 'analysis/summarize.py')], 'summarize')


def plot(args):
    checkpoint(args)
    if not (args.workdir / 'analysis/all_selected_cluster_source_target_relations.csv').is_file():
        summarize(args)
    for name in ('omnipath_validation_summary.csv', 'omnipath_path_edges.csv'):
        link(CODE / 'analysis/omnipath_validation' / name,
             args.workdir / 'analysis/omnipath_validation' / name)
    for name in PLOTS:
        run(args, [args.python, str(CODE / 'visualization' / name)], Path(name).stem)
    print(f'figures={args.workdir / "visualization/figures"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'groups', 'analyze', 'summarize', 'plot', 'all'))
    parser.add_argument('--workdir', type=Path, default=ROOT / '.local/mosta')
    parser.add_argument('--raw-dir', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--device-index', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.device_index < 0 or args.epochs < 1 or args.threads < 1:
        parser.error('device index must be nonnegative; epochs and threads must be positive')
    args.workdir = args.workdir.expanduser().resolve()
    args.raw_dir = args.raw_dir.expanduser().resolve() if args.raw_dir else None
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else None
    args.workdir.mkdir(parents=True, exist_ok=True)
    stages = ('prepare', 'train', 'groups', 'analyze', 'plot') if args.stage == 'all' else (args.stage,)
    if args.stage == 'all' and args.checkpoint_dir:
        stages = tuple(stage for stage in stages if stage != 'train')
    for stage in stages:
        print(f'stage={stage}', flush=True)
        globals()[stage](args)

if __name__ == '__main__':
    main()
