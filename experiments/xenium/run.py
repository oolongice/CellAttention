#!/usr/bin/env python3
"""Run Xenium Liver preprocessing, Transformer training, and source-target analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
STAGES = ("prepare", "annotate", "train", "analyze", "summarize")
RAW_FILES = ("cell_name.txt", "gene_list.txt", "annotation.csv", "gene_expression.npz", "spatial_coordinates.npy")


def run(command: list[str], log: Path | None = None) -> None:
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", RAYON_NUM_THREADS="8")
    if log is None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def binary(name: str) -> Path:
    run(["cargo", "build", "--release", "--locked", "--manifest-path", str(ROOT / "Cargo.toml"), "--bin", name])
    return ROOT / "target/release" / name


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def prepare(args: argparse.Namespace) -> None:
    if args.raw_dir is None:
        raise ValueError("prepare requires --raw-dir with the five Xenium raw files")
    for name in RAW_FILES:
        if not (args.raw_dir / name).is_file():
            raise FileNotFoundError(args.raw_dir / name)
    output = args.workdir / "data/preprocessed"
    run([args.python, str(CODE / "preprocess/prepare.py"), "--raw-dir", str(args.raw_dir), "--output-dir", str(output)], args.workdir / "logs/prepare.log")
    print(f"prepared {output}")


def annotate(args: argparse.Namespace) -> None:
    data = args.workdir / "data/preprocessed"
    for name in ("expression.mtx", "cell_ids.txt", "gene_ids.txt", "sample_ids.txt", "evaluation_cell_groups.txt"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
    for sample in ("control", "cancer"):
        run([args.python, str(CODE / "preprocess" / f"annotate_{sample}.py"), str(data)], args.workdir / "logs" / f"annotate_{sample}.log")
    print(f"annotated {data / 'evaluation_cell_groups.txt'}")


def training_config(args: argparse.Namespace) -> dict:
    config = json.loads((CODE / "train_config.json").read_text())
    data = args.workdir / "data/preprocessed"
    for key, name in (("expression_matrix", "expression.mtx"), ("cell_ids", "cell_ids.txt"), ("gene_ids", "gene_ids.txt"), ("sample_ids", "sample_ids.txt")):
        config[key] = str(data / name)
    config["output_dir"] = str(args.workdir / "model/transformer")
    config["device_index"] = args.device_index
    config["training"]["epochs"] = args.epochs
    config["training"]["max_cells_per_epoch"] = args.max_cells_per_epoch
    return config


def train(args: argparse.Namespace) -> None:
    config = training_config(args)
    for key in ("expression_matrix", "cell_ids", "gene_ids", "sample_ids"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    path = args.workdir / "model/train_config.json"
    checkpoint = args.workdir / "model/transformer/model.mpk"
    if checkpoint.is_file():
        if not path.is_file() or json.loads(path.read_text()) != config:
            raise ValueError("existing checkpoint has a different or unknown training configuration")
        print(f"reusing checkpoint {checkpoint}")
        return
    write_config(path, config)
    run([str(binary("cell_attention")), str(path)], args.workdir / "logs/train.log")
    print(f"trained {checkpoint}")


def checkpoint_dir(args: argparse.Namespace) -> Path:
    return args.checkpoint_dir or args.workdir / "model/transformer"


def check_alignment(args: argparse.Namespace, checkpoint: Path) -> None:
    data = args.workdir / "data/preprocessed"
    for name in ("cell_ids.txt", "gene_ids.txt"):
        expected = (data / name).read_bytes()
        observed = (checkpoint / name).read_bytes()
        if expected != observed:
            raise ValueError(f"preprocessed {name} differs from checkpoint {name}")
    for name in ("spatial_coordinates.csv", "evaluation_cell_groups_initial.txt"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)


def analysis_config(args: argparse.Namespace, checkpoint: Path) -> dict:
    data = args.workdir / "data/preprocessed"
    return {
        "dataset": "xenium_liver", "checkpoint_dir": str(checkpoint),
        "coordinates": str(data / "spatial_coordinates.csv"),
        "annotations": str(data / "evaluation_cell_groups_initial.txt"),
        "output_dir": str(args.workdir / "analysis/results"),
        "embedding_dimensions": 48, "receiver_group_count": 10,
        "receiver_group_clustering_method": "truncated_whitened_pca_kmeans",
        "receiver_group_pca_components": 10, "maximum_fit_cells": 60000,
        "length_scale": 30.0, "maximum_distance": 90.0,
        "minimum_distance": 2.0, "maximum_neighbors": 9999,
        "target_block_size": 512, "top_k_sources": 8,
        "receiver_field_mask": {"mode": "hard_expression_exclusion", "expression_threshold": 0.0},
    }


def analyze(args: argparse.Namespace) -> None:
    checkpoint = checkpoint_dir(args)
    if not (checkpoint / "model.mpk").is_file():
        raise FileNotFoundError(checkpoint / "model.mpk")
    check_alignment(args, checkpoint)
    path = args.workdir / "analysis/config.json"
    write_config(path, analysis_config(args, checkpoint))
    run([str(binary("run_source_target_analysis")), str(path)], args.workdir / "logs/analysis.log")
    summarize(args)


def summarize(args: argparse.Namespace) -> None:
    run([args.python, str(CODE / "analysis/summarize.py"), "--workdir", str(args.workdir)], args.workdir / "logs/summarize.log")
    print(f"analysis results: {args.workdir / 'analysis/results'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--workdir", type=Path, default=ROOT / ".local/xenium")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--raw-dir", type=Path, help="directory containing the five original Xenium files")
    parser.add_argument("--checkpoint-dir", type=Path, help="existing Transformer checkpoint for analyze; does not modify it")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-cells-per-epoch", type=int, default=32768)
    args = parser.parse_args()
    if args.device_index < 0 or args.epochs < 1 or args.max_cells_per_epoch < 0:
        parser.error("invalid training override")
    args.workdir = args.workdir.expanduser().resolve()
    args.raw_dir = args.raw_dir.expanduser().resolve() if args.raw_dir else None
    args.checkpoint_dir = args.checkpoint_dir.expanduser().resolve() if args.checkpoint_dir else None
    args.workdir.mkdir(parents=True, exist_ok=True)
    stages = ("prepare", "annotate", "analyze") if args.stage == "all" and args.checkpoint_dir else (STAGES[:4] if args.stage == "all" else (args.stage,))
    for stage in stages:
        print(f"stage={stage}", flush=True)
        globals()[stage](args)


if __name__ == "__main__":
    main()
