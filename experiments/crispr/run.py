#!/usr/bin/env python3
"""Reproduce the spatial CRISPR analysis from the CellAttention manuscript."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
STAGES = ("prepare", "train", "analyze", "validate", "plot")
PLOTS = (
    "plot_spatial_perturbation_regions.py",
    "plot_predicted_vs_crispr_effect.py",
    "plot_immune_source_target_heatmap.py",
    "plot_source_conditioned_target_validation.py",
    "plot_source_conditioned_validation_heatmap.py",
    "plot_target_recovery_random_baseline.py",
    "plot_source_field_global.py",
    "plot_source_field_local.py",
    "plot_source_field_global_selected_regions.py",
)


def run(argv, env, log=None):
    if log is None:
        subprocess.run(argv, cwd=ROOT, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def environment(args):
    env = os.environ.copy()
    paths = [str(CODE), str(CODE / "preprocess")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env.update(CELLATTENTION_CRISPR_WORKDIR=str(args.workdir), PYTHONPATH=os.pathsep.join(paths),
               PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", RAYON_NUM_THREADS="8")
    return env


def place_input(source, destination):
    if source is None:
        if not destination.is_file():
            raise FileNotFoundError(f"required input missing: {destination}")
        return
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"input already exists at {destination}; remove or use the current file")
    destination.symlink_to(source)


def inputs(args, *names):
    locations = {
        "raw": (args.raw_h5ad, args.workdir / "data/raw/SpacSeq_lung_cancer_label.h5ad"),
        "markers": (args.marker_xlsx, args.workdir / "data/raw/1-s2.0-S0092867426005167-mmc12.xlsx"),
        "dge_all": (args.dge_all, args.workdir / "data/preprocessed/immune_cells_spatial_DGE_all_shapes.csv"),
        "dge_significant": (args.dge_significant, args.workdir / "data/preprocessed/immune_cells_spatial_DGE_significant.csv"),
    }
    for name in names:
        place_input(*locations[name])


def prepare(args):
    inputs(args, "raw", "markers")
    env = environment(args)
    data = args.workdir / "data"
    panel = data / "preprocessed/gene_panel"
    raw = data / "raw/SpacSeq_lung_cancer_label.h5ad"
    run([args.python, "-m", "gene_panel.cli", "--adata", str(raw), "--markers",
         str(data / "raw/1-s2.0-S0092867426005167-mmc12.xlsx"), "--config",
         str(CODE / "preprocess/gene_panel/default.yaml"), "--output", str(panel)], env,
        args.workdir / "logs/panel.log")
    run([args.python, str(CODE / "preprocess/step3b_prepare_marker_qc_training.py"),
         "--input", str(raw), "--panel", str(panel / "selected_genes_500.txt"),
         "--output-h5ad", str(data / "preprocessed/SpacSeq_lung_cancer_label_integratedPanel_500genes.h5ad"),
         "--output-dir", str(data / "preprocessed")], env,
        args.workdir / "logs/preprocess.log")
    if args.dge_all is not None or args.dge_significant is not None:
        inputs(args, "dge_all", "dge_significant")
    else:
        run([args.python, str(CODE / "preprocess/build_immune_dge.py"),
             "--input", str(raw), "--output-dir", str(data / "preprocessed")], env,
            args.workdir / "logs/immune_dge.log")


def model_config(args):
    data = args.workdir / "data/preprocessed/control_train"
    return {
        "expression_matrix": str(data / "expression.mtx"),
        "cell_ids": str(data / "cell_ids.txt"),
        "gene_ids": str(data / "gene_ids.txt"),
        "output_dir": str(args.workdir / "model/transformer"),
        "device_index": args.device_index,
        "model": {"d_model": 48, "d_ff": 96, "n_heads": 4, "n_layers": 2, "dropout": 0.05},
        "training": {"epochs": args.epochs, "batch_size": 1024, "learning_rate": 0.001,
                     "mask_probability": 0.25, "seed": 168, "shuffle": True,
                     "inference_mask_chunk_size": 16,
                     "max_cells_per_epoch": args.max_cells_per_epoch,
                     "epoch_sampling": "random", "minimum_cells_per_sample": 0},
    }


def build(binary):
    run(["cargo", "build", "--release", "--locked", "--manifest-path", str(ROOT / "Cargo.toml"),
         "--bin", binary], os.environ.copy())
    return ROOT / "target/release" / binary


def train(args):
    cfg = model_config(args)
    for name in ("expression_matrix", "cell_ids", "gene_ids"):
        if not Path(cfg[name]).is_file():
            raise FileNotFoundError(cfg[name])
    binary = build("cell_attention")
    path = args.workdir / "model/train_config.json"
    checkpoint = args.workdir / "model/transformer/model.mpk"
    if checkpoint.exists():
        if not path.is_file() or json.loads(path.read_text()) != cfg:
            raise ValueError("existing checkpoint has a different or unknown training configuration")
        print(f"reusing checkpoint {checkpoint}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    run([str(binary), str(path)], environment(args), args.workdir / "logs/train.log")


def analyze(args):
    checkpoint = args.workdir / "model/transformer/model.mpk"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = {
        "dataset": "crispr_control_integrated_panel",
        "checkpoint_dir": str(args.workdir / "model/transformer"),
        "coordinates": str(args.workdir / "data/preprocessed/control_train/spatial_coordinates_um.csv"),
        "annotations": str(args.workdir / "data/preprocessed/control_train/annotation.txt"),
        "output_dir": str(args.workdir / "analysis/results"),
        "embedding_dimensions": 48, "receiver_group_count": 12,
        "maximum_fit_cells": 60000, "length_scale": 25.0,
        "maximum_distance": 75.0, "minimum_distance": 2.0,
        "maximum_neighbors": 9999, "target_block_size": 512,
        "top_k_sources": 8,
        "receiver_field_mask": {"mode": "hard_expression_exclusion", "expression_threshold": 0.0},
    }
    path = args.workdir / "analysis/config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    run([str(build("run_source_target_analysis")), str(path)], environment(args),
        args.workdir / "logs/analysis.log")
    run([args.python, str(CODE / "analysis/summarize.py")], environment(args))
    run([args.python, str(CODE / "analysis/analyze_icam1_cxcr4.py"),
         "--receiver-field-mask", "hard_expression_exclusion", "--expression-threshold", "0.0"],
        environment(args), args.workdir / "logs/source_focused.log")


def validate(args):
    inputs(args, "dge_significant")
    run([args.python, str(CODE / "visualization/prepare_source_conditioned_validation.py")],
        environment(args))


def plot(args):
    inputs(args, "raw", "dge_all", "dge_significant")
    env = environment(args)
    for format_name in ("png", "pdf"):
        env["CELLATTENTION_PLOT_FORMAT"] = format_name
        for name in PLOTS:
            print(f"plot {name} {format_name}", flush=True)
            run([args.python, str(CODE / "visualization" / name)], env,
                args.workdir / "logs" / f"{Path(name).stem}_{format_name}.log")
        for target in ("Cxcl9", "Cxcl10"):
            env["CELLATTENTION_TARGET_GENE"] = target
            run([args.python, str(CODE / "visualization/plot_source_target_field_dose_response.py")], env,
                args.workdir / "logs" / f"dose_{target}_{format_name}.log")
    run([args.python, str(CODE / "visualization/supp_data2_crispr.py")], env,
        args.workdir / "logs/supplementary.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument("--workdir", type=Path, default=ROOT / ".local/crispr")
    parser.add_argument("--python", default=sys.executable, help="Python with packages from requirements.txt")
    parser.add_argument("--raw-h5ad", type=Path)
    parser.add_argument("--marker-xlsx", type=Path)
    parser.add_argument("--dge-all", type=Path)
    parser.add_argument("--dge-significant", type=Path)
    parser.add_argument("--device-index", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max-cells-per-epoch", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_cells_per_epoch < 0 or args.device_index < 0:
        parser.error("invalid training override")
    args.workdir = args.workdir.expanduser().resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    for stage in STAGES if args.stage == "all" else (args.stage,):
        print(f"stage={stage}", flush=True)
        globals()[stage](args)


if __name__ == "__main__":
    main()
