# Slide-seqV2 mouse hippocampus

This workflow uses the annotated Slide-seqV2 hippocampus H5AD (41,786 beads, 23,264 genes), selects a fixed 500-gene panel, trains CellAttention, and evaluates source–target relations against CellChatDB, NeuronChatDB, and MetaChatDB. The CellChat ligand–receptor table is used for panel selection; the three compact relation tables in `database_relations/` are used for validation.

The aligned H5AD combines whole-transcriptome counts from [SCP815](https://singlecell.broadinstitute.org/single_cell/study/SCP815) with the cell-type annotations provided by [Squidpy Slide-seqV2](https://squidpy.readthedocs.io/en/stable/notebooks/tutorials/tutorial_slideseqv2.html). The reference input SHA-256 values are `2858e0ffc80a747200ab1e2b071561cc7e6dbd8984452bfd1ed83f1bd09b89db` for `adata_raw_labeled.h5ad` and `7754133703b55a28476498c977143e77b84a4929079082b074fbf8ae6dd69965` for `LR.Mouse_LR.txt`.

From the repository root, install the Python packages and run:

```bash
python3 -m venv .local/slideseqv2-venv
.local/slideseqv2-venv/bin/python -m pip install -r experiments/slideseqv2/requirements.txt
.local/slideseqv2-venv/bin/python experiments/slideseqv2/run.py all \
  --raw-h5ad /path/to/adata_raw_labeled.h5ad \
  --cellchat-db /path/to/LR.Mouse_LR.txt
```

The stages can also be called separately: `prepare`, `train`, `groups`, `analyze`, `edges`, `databases`, and `plot`. `prepare` writes the 500-gene matrix, aligned IDs and coordinates, and whole-transcriptome counts. `train` uses the original 60-epoch, seed-168 model settings. `groups` evaluates candidate receiver-group counts; `analyze` fits six receiver groups and source–target effects. `edges` exports the base and expanded cell-edge selections, and `databases` evaluates the expanded edges against the three fixed relation tables. An existing checkpoint can be supplied with `--checkpoint-dir /path/to/model/transformer`; the runner checks its cell and gene IDs against the prepared inputs.

`plot` produces PNG and PDF versions of:

- the compact annotation and CCC-module map;
- the six-module source–target relation figure;
- the gene-pair database-support UpSet plot;
- three database-pair Sankey figures;
- six single-case and three combined spatial-streamline figures;
- three supplementary source–database-pair–target diagrams.

The figures use these generated data:

| Figure | Main inputs |
| --- | --- |
| Annotation and CCC modules | `data/preprocessed/spatial_coordinates.csv`, `evaluation_cell_groups.txt`, and `analysis/results/receiver_group_assignments.csv` |
| Cluster source–target relations | `analysis/all_selected_cluster_source_target_relations.csv`, base cell edges, and the 500-gene expression matrix |
| Gene-pair UpSet | Three `analysis/*_support_expanded/results/lr_distance_matched_enrichment_long.csv` tables; the plotting stage writes the pair membership and intersection counts |
| Database-pair Sankey | Expanded cell edges plus each database's enrichment and `model_edge_lr_support.csv` tables; selected relations and paths are written to `visualization/data/database_pair_sankey/` |
| Spatial streamlines | Whole-transcriptome expression, aligned IDs, coordinates, and database-supported edges; field similarities and selected cases are written to `visualization/data/database_streamline_concordance/` |
| Supplementary database-pair diagrams | The three database enrichment tables; selected pair tables are written beside the supplementary figures |

All generated data, checkpoints, figures, and logs remain in the Git-ignored `.local/slideseqv2/`. Place Arial font files in `.local/slideseqv2/font/` to use the original figure typography.
