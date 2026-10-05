# MOSTA

This workflow reproduces the selected Figure 5 spatial maps, brain embedding and transition figures, target-gene ring, and Wnt3a/Shh OmniPath pathways from four [MOSTA](https://db.cngb.org/stomics/mosta/) embryo sections (E10.5–E13.5).

## Run

From the repository root:

```bash
python3 -m venv .local/mosta-venv
.local/mosta-venv/bin/python -m pip install -r experiments/mosta/requirements.txt
.local/mosta-venv/bin/python experiments/mosta/run.py all \
  --raw-dir /path/to/mosta/raw
```

The raw directory contains `cell_name.txt`, `gene_list.txt`, `meta_annotation.txt`, `meta_batch.txt`, `gene_expression.npz`, and `spatial_coordinates.npy`. Put `arial.ttf`, `arialbd.ttf`, `ariali.ttf`, and `arialbi.ttf` in `.local/mosta/font/` for the original figure style.

You can run `prepare`, `train`, `groups`, `analyze`, or `plot` individually in place of `all`. To reuse an existing checkpoint with `all`, pass `--checkpoint-dir /path/to/model/transformer`. Training uses the original 300-epoch, seed-168 settings and device index 1; use `--device-index 0` to select device 0.

## Output

All generated files go to `.local/mosta/`:

| Stage | Output |
| --- | --- |
| `prepare` | Expression matrix, coordinates, aligned IDs and annotations in `data/preprocessed/` |
| `train` | Checkpoint and embeddings in `model/transformer/` |
| `groups` | Receiver-group selection in `analysis/receiver_group_count_selection/` |
| `analyze` | Receiver-group assignments and source–target relations in `analysis/` |
| `plot` | Figure data in `visualization/data/` and figures in `visualization/figures/` |

`plot` creates four stage annotation and module maps, `mosta_specific/cluster_target_ring.png`, Wnt3a and Shh pathway plots, and the brain UMAP, transition network, influence, source-field, and marker figures in `brain_development/`.
