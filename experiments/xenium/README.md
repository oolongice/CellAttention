# Xenium liver

The workflow combines non-diseased and liver-cancer Xenium samples (401,899 cells; 377 genes), trains a Transformer with seed 168 for 200 epochs, and fits ten receiver groups for spatial source–target analysis.

Install Python dependencies from the repository root:

```bash
python3 -m venv .local/xenium-venv
.local/xenium-venv/bin/python -m pip install -r experiments/xenium/requirements.txt
```

The raw directory must contain `cell_name.txt`, `gene_list.txt`, `annotation.csv`, `gene_expression.npz`, and `spatial_coordinates.npy`. Run:

```bash
.local/xenium-venv/bin/python experiments/xenium/run.py prepare --raw-dir /path/to/xenium/raw
.local/xenium-venv/bin/python experiments/xenium/run.py annotate
.local/xenium-venv/bin/python experiments/xenium/run.py train
.local/xenium-venv/bin/python experiments/xenium/run.py analyze
```

`prepare` writes the expression matrix, aligned IDs, coordinates, and initial cell annotations to `.local/xenium/data/preprocessed/`. `annotate` updates both samples with Leiden cluster pseudobulk labels; the original labels remain in `evaluation_cell_groups_initial.txt` for the receiver-group composition table. `train` uses `train_config.json` and writes the checkpoint to `.local/xenium/model/transformer/`. `analyze` writes receiver-group assignments, target scores, source–target scores, and compact summaries to `.local/xenium/analysis/results/`.

To analyze an existing checkpoint, run `analyze` with `--checkpoint-dir /path/to/model/transformer`. The checkpoint's cell and gene IDs must match the prepared data.
