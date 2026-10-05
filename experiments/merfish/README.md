# 3D weMERFISH zebrafish embryo

This workflow uses the measured 6-somite E1 embryo from the [Dryad weMERFISH archive](https://doi.org/10.5061/dryad.j0zpc86v9) to reproduce the selected Figure 6 and supplementary visualizations. It analyzes 22,475 cells, 495 measured genes, and three-dimensional coordinates.

## Run

From the repository root:

```bash
python3 -m venv .local/merfish-venv
.local/merfish-venv/bin/python -m pip install -r experiments/merfish/requirements.txt
.local/merfish-venv/bin/python experiments/merfish/run.py all \
  --archive /path/to/doi_10_5061_dryad_j0zpc86v9__v20251211.zip
```

Place `arial.ttf`, `arialbd.ttf`, `ariali.ttf`, and `arialbi.ttf` in `.local/merfish/font/`. The 3D plots use software rendering through OSMesa. Training uses the original CUDA configuration: 300 epochs, seed 168, and device index 1. Use `--device-index 0` for device 0, or `--cpu` for CPU training. To reuse a matching checkpoint with `all`, pass `--checkpoint-dir /path/to/model/transformer`.

Run `prepare`, `train`, `groups`, `analyze`, `edges`, or `plot` separately in place of `all`. Individual figure stages are `plot-landscape`, `plot-supp`, `plot-cells`, `plot-pairs`, and `plot-vectors`.

## Output

All generated data, checkpoints, figures, and logs go to `.local/merfish/`.

| Stage | Output |
| --- | --- |
| `prepare` | Expression, 3D coordinates, IDs, and annotations in `data/preprocessed/` |
| `train` | Transformer checkpoint in `model/transformer/` |
| `groups` | Receiver-group count scores in `analysis/receiver_group_count_selection/` |
| `analyze` | Eight receiver groups and source–target scores in `analysis/` |
| `edges` | Directional 3D attribution edges and pair statistics in `analysis/results/` |
| `plot` | Figure input tables in `visualization/data/`; figures in `visualization/figures/` and `visualization/supp_figures/` |

`plot` writes the annotation and module views to `figures/cells_3d/`, the 24 pair views to `figures/directional_pairs_3d/`, the 24 arrow-field views to `figures/directional_pair_vector_fields_3d/`, the target program to `figures/cluster_target_landscape.png`, and the supplementary ranking to `supp_figures/`.
