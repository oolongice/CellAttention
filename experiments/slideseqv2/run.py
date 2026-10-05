#!/usr/bin/env python3
"""Reproduce selected SlideSeqV2 analysis tables and figures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
STAGES = ("prepare", "train", "groups", "analyze", "edges", "databases", "plot")
PLOTS = (
    "plot_compact_annotation_clusters.py",
    "plot_cluster_relations.py",
    "build_database_membership.py",
    "plot_database_support_upset.py",
    "plot_database_pair_sankey.py",
    "plot_process_spatial_orientation.py",
    "plot_process_spatial_orientation_combined.py",
    "supp_database_pair_relations.py",
)
DATABASES = {
    "cellchat_support_expanded": "cellchat_relations_mouse.csv",
    "neuronchat_support_expanded": "neuronchat_relations_mouse.csv",
    "metachat_support_expanded": "metachat_relations_mouse.csv",
}


def environment(args):
    env = os.environ.copy()
    env.update(CELLATTENTION_SLIDESEQ_WORKDIR=str(args.workdir), PYTHONDONTWRITEBYTECODE="1",
               MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", RAYON_NUM_THREADS=str(args.threads))
    return env


def run(args, command, log=None):
    if log is None:
        subprocess.run(command, cwd=ROOT, env=environment(args), check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=environment(args), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)


def rust_binary(args, name):
    run(args, ["cargo", "build", "--release", "--locked", "--manifest-path",
               str(ROOT / "Cargo.toml"), "--bin", name])
    return ROOT / "target/release" / name


def config(args, template):
    source = (CODE / template).read_text()
    return json.loads(source.replace("${WORKDIR}", str(args.workdir)))


def save_config(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def link_input(source, destination):
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"existing input at {destination}")
    destination.symlink_to(source)


def prepare(args):
    raw = args.workdir / "data/raw"
    if args.raw_h5ad is not None:
        link_input(args.raw_h5ad, raw / "adata_raw_labeled.h5ad")
    if not (raw / "adata_raw_labeled.h5ad").is_file():
        raise FileNotFoundError(raw / "adata_raw_labeled.h5ad")
    if args.cellchat_db is None:
        raise ValueError("prepare requires --cellchat-db for the fixed 500-gene panel")
    pre = args.workdir / "data/preprocessed"
    panel = pre / "gene_panel"
    run(args, [args.python, str(CODE / "preprocess/select_gene_panel.py"),
               "--input", str(raw / "adata_raw_labeled.h5ad"),
               "--cellchat-db", str(args.cellchat_db), "--output-dir", str(panel)],
        args.workdir / "logs/panel.log")
    run(args, [args.python, str(CODE / "preprocess/prepare.py"),
               "--input", str(raw / "adata_raw_labeled.h5ad"), "--output-dir", str(pre),
               "--target-genes", "500", "--min-cells", "100", "--gene-list",
               str(panel / "selected_genes.txt")], args.workdir / "logs/prepare.log")
    run(args, [args.python, str(CODE / "preprocess/export_full_expression.py"),
               "--input", str(raw / "adata_raw_labeled.h5ad"), "--output-dir", str(raw)],
        args.workdir / "logs/export_full_expression.log")
    print(f"prepared {pre}")


def checkpoint(args):
    return args.checkpoint_dir or args.workdir / "model/transformer"


def check_checkpoint(args):
    path = checkpoint(args)
    for name in ("model.mpk", "cell_ids.txt", "gene_ids.txt", "raw_expression.csv", "residuals.csv", "cell_embeddings.csv"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    pre = args.workdir / "data/preprocessed"
    for name in ("cell_ids.txt", "gene_ids.txt"):
        if (path / name).read_bytes() != (pre / name).read_bytes():
            raise ValueError(f"checkpoint {name} does not match prepared data")
    return path


def train(args):
    cfg = config(args, "train_config.template.json")
    cfg["device_index"] = args.device_index
    cfg["training"]["epochs"] = args.epochs
    path = args.workdir / "model/train_config.json"
    output = args.workdir / "model/transformer/model.mpk"
    if output.is_file():
        if not path.is_file() or json.loads(path.read_text()) != cfg:
            raise ValueError("existing checkpoint has a different or unknown training configuration")
        print(f"reusing {output}")
        return
    save_config(path, cfg)
    run(args, [str(rust_binary(args, "cell_attention")), str(path)], args.workdir / "logs/train.log")
    print(f"checkpoint={output}")


def groups(args):
    cfg = config(args, "analysis/group_count_config.template.json")
    cfg["checkpoint_dir"] = str(check_checkpoint(args))
    path = save_config(args.workdir / "analysis/receiver_group_count_selection/config.json", cfg)
    run(args, [str(rust_binary(args, "select_receiver_group_count")), str(path)],
        args.workdir / "logs/groups.log")


def analyze(args):
    cfg = config(args, "analysis/source_target_config.template.json")
    cfg["checkpoint_dir"] = str(check_checkpoint(args))
    path = save_config(args.workdir / "analysis/config.json", cfg)
    run(args, [str(rust_binary(args, "run_source_target_analysis")), str(path)],
        args.workdir / "logs/analyze.log")
    summarize(args)


def summarize(args):
    run(args, [args.python, str(CODE / "analysis/summarize.py")],
        args.workdir / "logs/summarize.log")


def edges(args):
    for name, template in (("cell_cell_edges", "cell_edges_config.template.json"),
                           ("cell_cell_edges_expanded", "expanded_edges_config.template.json")):
        cfg = config(args, "analysis/" + template)
        cfg["checkpoint_dir"] = str(check_checkpoint(args))
        path = save_config(args.workdir / "analysis" / name / "config.json", cfg)
        run(args, [str(rust_binary(args, "export_cell_cell_interactions")), str(path)],
            args.workdir / "logs" / f"{name}.log")


def databases(args):
    raw = args.workdir / "data/raw"
    pre = args.workdir / "data/preprocessed"
    for name in ("full_expression.mtx", "full_cell_ids.txt", "full_gene_ids.txt"):
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    for directory, relation_file in DATABASES.items():
        base = args.workdir / "analysis" / directory
        cfg = {
            "expression_mtx": str(raw / "full_expression.mtx"),
            "cell_ids": str(raw / "full_cell_ids.txt"),
            "gene_ids": str(raw / "full_gene_ids.txt"),
            "coordinates": str(pre / "spatial_coordinates.csv"),
            "model_edges": str(args.workdir / "analysis/cell_cell_edges_expanded/cell_cell_interaction_edges.csv"),
            "database_relations": str(CODE / "database_relations" / relation_file),
            "output_dir": str(base / "results"),
            "maximum_distance": 0.1, "distance_tolerance_fraction": 0.1,
            "expression_threshold": 0.0, "random_seed": 20260812,
            "write_edge_support": True,
        }
        path = save_config(base / "config.json", cfg)
        run(args, [str(rust_binary(args, "validata_ccc_database")), str(path)],
            args.workdir / "logs" / f"{directory}.log")


def plot(args):
    for directory in ("visualization/data", "visualization/figures", "visualization/supp_figures"):
        (args.workdir / directory).mkdir(parents=True, exist_ok=True)
    summarize(args)
    for name in PLOTS:
        print(f"plot={name}", flush=True)
        run(args, [args.python, str(CODE / "visualization" / name)],
            args.workdir / "logs" / f"{Path(name).stem}.log")
    print(f"figures={args.workdir / 'visualization/figures'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "summarize", "all"))
    parser.add_argument("--workdir", type=Path, default=ROOT / ".local/slideseqv2")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--raw-h5ad", type=Path)
    parser.add_argument("--cellchat-db", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.device_index < 0 or args.epochs < 1 or args.threads < 1:
        parser.error("device index must be nonnegative; epochs and threads must be positive")
    args.workdir = args.workdir.expanduser().resolve()
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else None
    args.cellchat_db = args.cellchat_db.expanduser().resolve() if args.cellchat_db else None
    args.workdir.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else (args.stage,)
    if args.stage == "all" and args.checkpoint_dir:
        stages = tuple(stage for stage in STAGES if stage != "train")
    for stage in stages:
        print(f"stage={stage}", flush=True)
        globals()[stage](args)


if __name__ == "__main__":
    main()
