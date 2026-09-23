#!/usr/bin/env python3
"""Generate and reproduce the three synthetic CellAttention experiments."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

EXPERIMENT = Path(__file__).resolve().parent
ROOT = EXPERIMENT.parents[1]
CASES = ("independent_targets", "two_sources_per_target", "shared_target")
CASE_NUMBERS = {name: index for index, name in enumerate(CASES, 1)}


def command(argv, *, cwd=ROOT, env=None, log=None):
    if log is None:
        subprocess.run(argv, cwd=cwd, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def generate(args):
    manifest = EXPERIMENT / "generator/Cargo.toml"
    binary = ROOT / "target" / args.profile / "cellattention-synthetic-generator"
    env = {**os.environ, "CARGO_TARGET_DIR": str(ROOT / "target")}
    command(["cargo", "build", "--locked", "--manifest-path", str(manifest),
             "--bin", "cellattention-synthetic-generator",
             *(["--release"] if args.profile == "release" else [])], env=env)
    for case in CASES:
        output = args.workdir / "data/preprocessed" / case
        command([str(binary), case, str(output), "168"])
        print(f"generated {case}: {output}")


def train(args):
    if args.seed_start > args.seed_end:
        raise ValueError("seed-start must be at most seed-end")
    for case in CASES:
        for name in ("expression.csv", "cell_ids.txt", "gene_ids.txt"):
            path = args.workdir / "data/preprocessed" / case / name
            if not path.is_file():
                raise FileNotFoundError(f"missing {path}; run generate first")
    command(["cargo", "build", "--locked", "--manifest-path", str(ROOT / "Cargo.toml"),
             "--bin", "cell_attention",
             *(["--release"] if args.profile == "release" else [])])
    binary = ROOT / "target" / args.profile / "cell_attention"
    template = json.loads((EXPERIMENT / "train_config.template.json").read_text())
    jobs = []
    for case in CASES:
        data = args.workdir / "data/preprocessed" / case
        for seed in range(args.seed_start, args.seed_end + 1):
            out = args.workdir / "model/multiseed" / case / f"seed_{seed}"
            config = json.loads(json.dumps(template))
            config.update(expression_matrix=str(data / "expression.csv"),
                          cell_ids=str(data / "cell_ids.txt"),
                          gene_ids=str(data / "gene_ids.txt"), output_dir=str(out))
            config["training"]["seed"] = seed
            if args.epochs is not None:
                config["training"]["epochs"] = args.epochs
            if args.max_cells_per_epoch is not None:
                config["training"]["max_cells_per_epoch"] = args.max_cells_per_epoch
            if args.inference_chunk_size is not None:
                config["training"]["inference_mask_chunk_size"] = args.inference_chunk_size
            out.mkdir(parents=True, exist_ok=True)
            config_path = out / "run_config.json"
            if (out / "model.mpk").is_file():
                if not config_path.is_file() or json.loads(config_path.read_text()) != config:
                    raise ValueError(f"existing checkpoint has a different or unknown config: {out}")
                print(f"reusing {case} seed {seed}")
                continue
            config_path.write_text(json.dumps(config, indent=2) + "\n")
            jobs.append((case, seed, config_path, out / "training.log"))

    def run_one(job):
        case, seed, config_path, log = job
        env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
               "RAYON_NUM_THREADS": str(args.threads_per_job)}
        command([str(binary), str(config_path)], env=env, log=log)
        if not (config_path.parent / "model.mpk").is_file():
            raise RuntimeError(f"checkpoint missing after training: {config_path.parent}")
        return case, seed

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        for future in as_completed(futures):
            case, seed = future.result()
            print(f"trained {case} seed {seed}")


def evaluate(args):
    expected = set(range(args.seed_start, args.seed_end + 1))
    for case in CASES:
        folder = args.workdir / "model/multiseed" / case
        available = {int(path.name.removeprefix("seed_")) for path in folder.glob("seed_*")
                     if (path / "model.mpk").is_file()}
        missing = sorted(expected - available)
        if missing:
            raise FileNotFoundError(f"{case}: missing checkpoints for seeds {missing}")
    env = {**os.environ, "CELLATTENTION_SYNTHETIC_WORKDIR": str(args.workdir),
           "PYTHONDONTWRITEBYTECODE": "1"}
    command([sys.executable, str(EXPERIMENT / "analysis/evaluate_continuous_benchmark.py")], env=env)


def plot(args):
    env = {**os.environ, "CELLATTENTION_SYNTHETIC_WORKDIR": str(args.workdir),
           "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    code = EXPERIMENT / "visualization"
    for case in CASES:
        for image_format in ("png", "pdf"):
            command([sys.executable, str(code / "plot_synthetic_data.py")],
                    env={**env, "CELLATTENTION_SYNTHETIC_CASE": str(CASE_NUMBERS[case]),
                         "CELLATTENTION_FIGURE_FORMAT": image_format})
    command([sys.executable, str(code / "plot_continuous_benchmark.py")], env=env)
    for image_format in ("png", "pdf"):
        figure_env = {**env, "CELLATTENTION_FIGURE_FORMAT": image_format}
        command([sys.executable, str(code / "plot_synthetic_combined.py")], env=figure_env)
        command([sys.executable, str(code / "plot_case3_source_score_heatmap.py")], env=figure_env)
    if args.supplementary:
        command([sys.executable, str(code / "supp_synthetic_identification.py")], env=env)
        for case in CASES:
            command([sys.executable, str(code / "supp_receiver_target_source_score_heatmaps.py"), case], env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("generate", "train", "evaluate", "plot", "all"))
    parser.add_argument("--workdir", type=Path, default=ROOT / ".local/synthetic")
    parser.add_argument("--profile", choices=("debug", "release"), default="release")
    parser.add_argument("--seed-start", type=int, default=2001)
    parser.add_argument("--seed-end", type=int, default=2030)
    parser.add_argument("--epochs", type=int, help="short run for debugging; paper uses 110")
    parser.add_argument("--max-cells-per-epoch", type=int, help="debug override; paper uses 500")
    parser.add_argument("--inference-chunk-size", type=int, help="debug override; paper default is 1")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads-per-job", type=int, default=8)
    parser.add_argument("--supplementary", action="store_true")
    args = parser.parse_args()
    args.workdir = args.workdir.expanduser().resolve()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if any(value is not None and value < 1 for value in (args.max_cells_per_epoch, args.inference_chunk_size)):
        parser.error("debug overrides must be positive")
    if args.jobs < 1 or args.threads_per_job < 1:
        parser.error("--jobs and --threads-per-job must be positive")
    args.workdir.mkdir(parents=True, exist_ok=True)
    stages = ("generate", "train", "evaluate", "plot") if args.step == "all" else (args.step,)
    for stage in stages:
        print(f"stage={stage}", flush=True)
        globals()[stage](args)


if __name__ == "__main__":
    main()
