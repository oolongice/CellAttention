# MOSTA mouse organogenesis

This workflow reproduces the selected Figure 5 visualizations from four MOSTA E1S1 embryo sections (E10.5–E13.5) of the [MOSTA atlas](https://db.cngb.org/stomics/mosta/). It uses the 500-gene expression panel, 177,266 aligned cell IDs, stage labels, tissue annotations, and spatial coordinates. The four stage sections are analyzed jointly. The two OmniPath plots use the compact mouse pathway tables in `analysis/omnipath_validation/`.

From the repository root, install the Python packages and run:

```bash
python3 -m venv .local/mosta-venv
.local/mosta-venv/bin/python -m pip install -r experiments/mosta/requirements.txt
.local/mosta-venv/bin/python experiments/mosta/run.py all \
  --raw-dir /path/to/mosta/raw
```

The raw directory contains `cell_name.txt`, `gene_list.txt`, `meta_annotation.txt`, `meta_batch.txt`, `gene_expression.npz`, and `spatial_coordinates.npy`. Their SHA-256 values in that order are:

```text
26614aa66ca57e5478f9038baa3a1b040201787985637537482b32781646dd5a
212de6c7ad34e7e4693f9ca83a3d0794a59db0c885f0e37b2849979fd3d27cef
e3228172fa20ebf18a8e32afcd0d11d5c48dae67f68e5a2dc4ada1233e8b68e9
6b184e44a7158690c05d8fe91f6ca8f4c232c07db8ba2b31f08b4ce68426313f
5c6a0ec00ca5753eacc420150a56dce642a137ee18aab6b6c1f8b12e14015ad8
2f59160a20998e77f395aee5608a54de9a4bae47db2389a6036098dbcf9d2b85
```

 Place `arial.ttf`, `arialbd.ttf`, `ariali.ttf`, and `arialbi.ttf` in `.local/mosta/font/` for the published figure typography. The default training uses device index 1, 300 epochs, seed 168, a 48-dimensional Transformer, stratified stage sampling, and the original 16 receiver groups. Use `--device-index 0` when the desired device is index 0.

Stages can be run separately: `prepare`, `train`, `groups`, `analyze`, and `plot`. The `prepare` stage writes aligned expression, coordinates, IDs and annotations. `train` writes the Transformer checkpoint and embeddings. `groups` writes receiver-group selection scores. `analyze` writes assignments, source–target scores, and the selected relation table. To reuse a matching checkpoint, pass `--checkpoint-dir /path/to/model/transformer`; the runner checks its cell and gene IDs against the prepared data.

`plot` writes these figures and their data under `.local/mosta/visualization/`:

| Figure | Main generated inputs |
| --- | --- |
| Four stage annotation and CCC-module spatial maps | Prepared coordinates, annotations, stage IDs, receiver-group assignments |
| CCC-module target-gene ring | Selected source–target relations; `data/cluster_target_ring_*.csv` |
| Brain stage and module UMAPs | Transformer embeddings and receiver-group assignments; `data/brain_development_umap_coordinates.csv` |
| Four brain transition networks, influence profiles, source fields, marker heatmap, and Wnt3a influence plots | The UMAP, selected relations, expression matrix and spatial coordinates; `data/brain_*` transition tables |
| Wnt3a and Shh OmniPath pathways | Bundled pathway evidence tables |

The brain figures are written to `visualization/figures/brain_development/`, the ring to `visualization/figures/mosta_specific/cluster_target_ring.png`, and the pathway plots to `visualization/figures/omnipath_brain_{wnt3a,shh}_pathway_merged.png`. The pathway and Wnt3a plots also write PDFs. Generated data, checkpoints, figures, and logs stay in the Git-ignored `.local/mosta/`.
