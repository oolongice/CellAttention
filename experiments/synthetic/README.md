# Synthetic experiments

Reproduces Fig. 2a–d and Supplementary Fig. 1 from three generated scenarios. Requires Python 3.10–3.12, Rust/Cargo, and the packages in `requirements.txt`.

From the repository root:

```bash
python3 -m venv .local/venv
.local/venv/bin/python -m pip install -r experiments/synthetic/requirements.txt
.local/venv/bin/python experiments/synthetic/run.py all --jobs 4 --threads-per-job 8 --supplementary
```

The stages are `generate`, `train`, `evaluate`, and `plot`. Generation uses seed 168; training uses seeds 2001–2030 and 110 epochs per seed. The workflow writes generated data, checkpoints, per-seed recovery tables, and PNG/PDF figures to `.local/synthetic/`.
