#!/usr/bin/env python3
"""Reproduce the selected 3D weMERFISH analysis and figures."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
ARCHIVE_NAME = 'doi_10_5061_dryad_j0zpc86v9__v20251211.zip'
PLOTS = {
    'plot-landscape': 'plot_cluster_target_landscape.py',
    'plot-supp': 'supp_cluster_source_target_ranking.py',
    'plot-cells': 'plot_cells_3d.py',
    'plot-pairs': 'plot_directional_pairs_3d_individual.py',
    'plot-vectors': 'plot_directional_pair_vector_fields_3d.py',
}


def link(source, destination):
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def environment(args):
    env = os.environ.copy()
    env.update(CELLATTENTION_MERFISH_WORKDIR=str(args.workdir), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(CODE), MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS=str(args.threads),
               VTK_DEFAULT_OPENGL_WINDOW='vtkOSOpenGLRenderWindow', LIBGL_ALWAYS_SOFTWARE='1')
    return env


def run(args, command, label):
    path = args.workdir / 'logs' / f'{label}.log'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output:
        subprocess.run(command, cwd=ROOT, env=environment(args), stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    print(f'{label}: {path}', flush=True)


def config(args, template, output):
    value = json.loads((CODE / template).read_text().replace('${WORKDIR}', str(args.workdir)))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + '\n')
    return value


def binary(args, name, *, cuda=False):
    path = ROOT / 'target/release' / name
    if cuda or not path.is_file():
        command = ['cargo', 'build', '--release', '--locked', '--bin', name]
        if cuda:
            command += ['--no-default-features', '--features', 'cuda']
        run(args, command, 'build_' + name)
    return str(path)


def prepare(args):
    archive = args.workdir / 'data/raw' / ARCHIVE_NAME
    if args.archive:
        link(args.archive, archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    run(args, [args.python, str(CODE / 'data/prepare.py')], 'prepare')
    print(f'prepared={args.workdir / "data/preprocessed"}')


def checkpoint(args):
    path = args.checkpoint_dir or args.workdir / 'model/transformer'
    if args.checkpoint_dir:
        dest = args.workdir / 'model/transformer'
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() and dest.resolve() == path:
            pass
        elif dest.exists() or dest.is_symlink():
            raise FileExistsError(dest)
        else:
            dest.symlink_to(path, target_is_directory=True)
    for name in ('model.mpk', 'cell_ids.txt', 'gene_ids.txt', 'raw_expression.csv', 'residuals.csv'):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    for name in ('cell_ids.txt', 'gene_ids.txt'):
        prepared = args.workdir / 'data/preprocessed' / name
        if prepared.is_file() and prepared.read_bytes() != (path / name).read_bytes():
            raise ValueError(f'checkpoint {name} differs from prepared input')
    return path


def train(args):
    dest = args.workdir / 'model/train_config.json'
    cfg = json.loads((CODE / 'train_config.template.json').read_text().replace('${WORKDIR}', str(args.workdir)))
    cfg['device_index'] = 0 if args.cpu else args.device_index
    cfg['training']['epochs'] = args.epochs
    if (args.workdir / 'model/transformer/model.mpk').is_file():
        if not dest.is_file() or json.loads(dest.read_text()) != cfg:
            raise ValueError('existing checkpoint has a different or unknown training configuration')
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(cfg, indent=2) + '\n')
    run(args, [binary(args, 'cell_attention', cuda=not args.cpu), str(dest)], 'train')


def groups(args):
    dest = args.workdir / 'analysis/receiver_group_count_selection/config.json'
    cfg = config(args, 'analysis/group_count_config.template.json', dest)
    cfg['checkpoint_dir'] = str(checkpoint(args))
    dest.write_text(json.dumps(cfg, indent=2) + '\n')
    run(args, [binary(args, 'select_receiver_group_count'), str(dest)], 'groups')


def analyze(args):
    dest = args.workdir / 'analysis/config.json'
    cfg = config(args, 'analysis/source_target_config.template.json', dest)
    cfg['checkpoint_dir'] = str(checkpoint(args))
    dest.write_text(json.dumps(cfg, indent=2) + '\n')
    run(args, [binary(args, 'run_source_target_analysis'), str(dest)], 'analyze')
    summarize(args)


def summarize(args):
    run(args, [args.python, str(CODE / 'analysis/summarize.py')], 'summarize')


def edges(args):
    checkpoint(args)
    if not (args.workdir / 'analysis/all_selected_cluster_source_target_relations.csv').is_file():
        summarize(args)
    run(args, [args.python, str(CODE / 'analysis/export_3d_edges.py')], 'edges')


def prepare_figures(args):
    checkpoint(args)
    if not (args.workdir / 'analysis/results/directional_edges.csv').is_file():
        raise FileNotFoundError(args.workdir / 'analysis/results/directional_edges.csv')
    run(args, [args.python, str(CODE / 'visualization/prepare_visualization_data.py')],
        'prepare_figures')


def plot_one(args, stage):
    if stage in ('plot-pairs', 'plot-vectors'):
        prepare_figures(args)
    script = PLOTS[stage]
    run(args, [args.python, str(CODE / 'visualization' / script)], Path(script).stem)


def plot(args):
    prepare_figures(args)
    for stage, script in PLOTS.items():
        run(args, [args.python, str(CODE / 'visualization' / script)], Path(script).stem)
    print(f'figures={args.workdir / "visualization/figures"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'groups', 'analyze', 'summarize',
                                          'edges', 'plot', 'all', *PLOTS))
    parser.add_argument('--workdir', type=Path, default=ROOT / '.local/merfish')
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--device-index', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if args.device_index < 0 or args.epochs < 1 or args.threads < 1:
        parser.error('device index must be nonnegative; epochs and threads must be positive')
    args.workdir = args.workdir.expanduser().resolve()
    args.archive = args.archive.expanduser().resolve() if args.archive else None
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else None
    args.workdir.mkdir(parents=True, exist_ok=True)
    stages = ('prepare', 'train', 'groups', 'analyze', 'edges', 'plot') if args.stage == 'all' else (args.stage,)
    if args.stage == 'all' and args.checkpoint_dir:
        stages = tuple(stage for stage in stages if stage != 'train')
    for stage in stages:
        print(f'stage={stage}', flush=True)
        if stage in PLOTS:
            plot_one(args, stage)
        else:
            globals()[stage](args)

if __name__ == '__main__':
    main()
