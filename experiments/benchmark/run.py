#!/usr/bin/env python3
"""Reproduce the ligand–receptor–downstream benchmark in Fig. 2e–f."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]
SCENARIOS = ("simple", "complex_lr", "spatial_overlap")
METHODS = ("cellattention", "misty_all", "misty_informed", "holonet", "commot_partial", "commot_full")
RESULT_DIRS = {
    "cellattention": "cellattention",
    "misty_all": "misty_no_gene_role_prior",
    "misty_informed": "misty_receiver",
    "holonet": "holonet_adapted",
    "commot_partial": "commot_partial_adapted",
    "commot_full": "commot_adapted",
}


def run(argv, *, env=None, log=None):
    if log is None:
        subprocess.run(argv, cwd=ROOT, env=env, check=True)
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def environment(args, tag=None):
    env = {**os.environ, "CELLATTENTION_BENCHMARK_WORKDIR": str(args.workdir),
           "CELLATTENTION_BENCHMARK_ENVS": str(args.env_root),
           "CELLATTENTION_GENERATION_SEED_START": str(args.seed_start),
           "CELLATTENTION_GENERATION_SEED_END": str(args.seed_end),
           "CELLATTENTION_SCENARIOS": ",".join(args.scenarios),
           "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if tag is not None:
        env.update(HIER_VARIANT=tag, HIER_GROUPS="inferred")
    return env


def python_for(args, method):
    path = args.env_root / method / "bin/python"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run the setup stage")
    return str(path)


def tags(args):
    for scenario in args.scenarios:
        for seed in range(args.seed_start, args.seed_end + 1):
            yield scenario, seed, f"scenario_{scenario}_seed_{seed}"


def setup(args):
    env = environment(args)
    env["MAMBA_ROOT_PREFIX"] = str(ROOT / ".local/micromamba")
    for method in ("report", "holonet", "commot", "misty"):
        prefix = args.env_root / method
        if (prefix / "conda-meta").is_dir():
            print(f"reusing environment {prefix}")
            continue
        run([args.micromamba, "create", "--yes", "--prefix", str(prefix),
             "--file", str(CODE / "envs" / f"{method}.yml")], env=env)


def generate(args):
    report = python_for(args, "report")
    cluster = python_for(args, "holonet")
    common = ["--cells", "600", "--marker-strength", "3.5",
              "--type-preference", "1.8", "--spatial-separation", "80",
              "--effect-scale", "3", "--dispersion", "6", "--dropout-rate", ".10",
              "--target-background-scale", ".4", "--ligand-spatial-scale", "0",
              "--ligand-hotspot-scale", "0", "--aligned-kernel",
              "--binary-receptor-gate", "--bimodal-receptor",
              "--receptor-gate-threshold", "1.0", "--separated-ligand-domains",
              "--ligand-domain-scale", "15"]
    variants = {
        "simple": ["--communication-genes", "4", "--simple-one-to-one",
                   "--decoy-scale", "0", "--ligand-domain-radius", "175",
                   "--ligand-domain-sigma", "42"],
        "complex_lr": ["--communication-genes", "8", "--decoy-scale", ".25",
                       "--ligand-domain-radius", "175", "--ligand-domain-sigma", "42"],
        "spatial_overlap": ["--communication-genes", "4", "--simple-one-to-one",
                            "--decoy-scale", "0", "--ligand-domain-radius", "105",
                            "--ligand-domain-sigma", "78"],
    }
    for scenario, seed, tag in tags(args):
        data = args.workdir / "data" / tag
        run([report, str(CODE / "generate_shared_prior.py"), "--out", str(data),
             "--seed", str(seed), *common, *variants[scenario]], env=environment(args, tag))
        run([cluster, str(CODE / "infer_groups.py"), str(data)], env=environment(args, tag))
        run([report, str(CODE / "prepare_partial_lr.py"), str(data)], env=environment(args, tag))
        print(f"generated {tag}")


def train(args):
    report = python_for(args, "report")
    run(["cargo", "build", "--locked", "--release", "--manifest-path",
         str(ROOT / "Cargo.toml"), "--bin", "cell_attention"])
    for _, _, tag in tags(args):
        env = environment(args, tag)
        env.update(CELLATTENTION_BINARY=str(ROOT / "target/release/cell_attention"),
                   CELLATTENTION_MODEL_SEED_START=str(args.model_seed_start),
                   CELLATTENTION_MODEL_SEED_END=str(args.model_seed_end))
        if args.epochs is not None:
            env["CELLATTENTION_EPOCHS"] = str(args.epochs)
        if args.max_cells_per_epoch is not None:
            env["CELLATTENTION_MAX_CELLS_PER_EPOCH"] = str(args.max_cells_per_epoch)
        if args.inference_chunk_size is not None:
            env["CELLATTENTION_INFERENCE_CHUNK_SIZE"] = str(args.inference_chunk_size)
        run([report, str(CODE / "train_cellattention.py")], env=env)
        print(f"trained {tag}")


def methods(args):
    report = python_for(args, "report")
    for _, _, tag in tags(args):
        env = environment(args, tag)
        if "cellattention" in args.methods:
            models = args.workdir / "models" / tag
            expected = {f"seed_{seed}" for seed in range(args.model_seed_start, args.model_seed_end + 1)}
            found = {p.name for p in models.glob("seed_*") if (p / "model.mpk").is_file()}
            if found != expected:
                raise ValueError(f"{tag}: expected CellAttention model seeds {sorted(expected)}, found {sorted(found)}")
            run([report, str(CODE / "run_adapters.py"), "cellattention"], env=env)
        if "holonet" in args.methods:
            run([report, str(CODE / "run_adapters.py"), "holonet"], env=env)
        for mode in ("partial", "full"):
            if f"commot_{mode}" in args.methods:
                run([report, str(CODE / "run_adapters.py"), "commot"],
                    env={**env, "COMMOT_LR_MODE": mode})
        for method, prepare, folder in (
            ("misty_informed", "prepare_misty_candidate_informed.py", "misty_receiver"),
            ("misty_all", "prepare_misty_no_prior.py", "misty_no_gene_role_prior"),
        ):
            if method not in args.methods:
                continue
            run([report, str(CODE / prepare)], env=env)
            out = args.workdir / "results" / tag / folder
            rscript = args.env_root / "misty/bin/Rscript"
            if not rscript.is_file():
                raise FileNotFoundError(f"missing {rscript}; run the setup stage")
            run([str(rscript), str(CODE / "run_misty.R"), str(out / "manifest.json"),
                 str(out / "raw_scores.csv"), "receiver"], env=env,
                log=args.workdir / "logs" / f"{tag}_{folder}.log")
        print(f"methods completed {tag}")


def evaluate(args):
    report = python_for(args, "report")
    analysis = python_for(args, "holonet")
    for _, _, tag in tags(args):
        env = environment(args, tag)
        result = args.workdir / "results" / tag
        if (result / "misty_no_gene_role_prior/raw_scores.csv").is_file():
            run([report, str(CODE / "extract_misty_no_prior.py")], env=env)
        for method in args.methods:
            path = result / RESULT_DIRS[method] / "triplet_scores.csv"
            if method == "misty_informed" and (result / "misty_receiver/raw_scores.csv").is_file():
                continue  # evaluate_triplet.py converts these scores to triplets.
            if not path.is_file():
                raise FileNotFoundError(f"missing scores for {tag}/{method}: {path}")
        run([analysis, str(CODE / "evaluate_triplet.py")], env=env)
        print(f"evaluated {tag}")
    run([analysis, str(CODE / "aggregate_scenarios.py")], env=environment(args))
    run([analysis, str(CODE / "visualization/supp_benchmark_metrics_by_scenario.py")], env=environment(args))


def visualize(args):
    env = environment(args)
    env["CELLATTENTION_VISUALIZATION_SEED"] = str(args.seed_start)
    run([python_for(args, "report"), str(CODE / "visualization/supp_synthetic_spatial_communication.py")], env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("setup", "generate", "train", "methods", "evaluate", "visualize", "all"))
    parser.add_argument("--workdir", type=Path, default=ROOT / ".local/benchmark")
    parser.add_argument("--env-root", type=Path, default=ROOT / ".local/benchmark-envs")
    parser.add_argument("--micromamba", default="micromamba")
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument("--seed-start", type=int, default=168)
    parser.add_argument("--seed-end", type=int, default=172)
    parser.add_argument("--model-seed-start", type=int, default=3001)
    parser.add_argument("--model-seed-end", type=int, default=3005)
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--epochs", type=int, help="debug override; paper uses 110")
    parser.add_argument("--max-cells-per-epoch", type=int, help="debug override; paper uses 600")
    parser.add_argument("--inference-chunk-size", type=int, help="debug override; paper default is 1")
    args = parser.parse_args()
    args.scenarios = tuple(args.scenario or SCENARIOS)
    args.methods = tuple(args.method or METHODS)
    args.workdir = args.workdir.expanduser().resolve()
    args.env_root = args.env_root.expanduser().resolve()
    if args.seed_start > args.seed_end or args.model_seed_start > args.model_seed_end:
        parser.error("seed start must not exceed seed end")
    if any(x is not None and x < 1 for x in (args.epochs, args.max_cells_per_epoch, args.inference_chunk_size)):
        parser.error("debug overrides must be positive")
    args.workdir.mkdir(parents=True, exist_ok=True)
    stages = ("generate", "train", "methods", "evaluate", "visualize") if args.stage == "all" else (args.stage,)
    for stage in stages:
        print(f"stage={stage}", flush=True)
        globals()[stage](args)


if __name__ == "__main__":
    main()
