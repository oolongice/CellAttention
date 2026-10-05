# CellAttention

CellAttention trains a masked-gene Transformer from expression data and uses spatial reaction–diffusion fields to estimate receiver-group-specific source–target relations.

## Build

From the repository root:

```bash
cargo build --release --locked
```

Binaries are written to `target/release/`. The training binary can also be built with CUDA:

```bash
cargo build --release --locked --no-default-features --features cuda
```

## Programs

| Program | Run | Input | Output |
| --- | --- | --- | --- |
| `cell_attention` | `target/release/cell_attention TRAIN_CONFIG.json` | Expression matrix, aligned cell/gene IDs, optional sample IDs | Transformer checkpoint, residuals, embeddings |
| `run_source_target_analysis` | `target/release/run_source_target_analysis CONFIG.json` | Checkpoint, coordinates, optional annotations | Receiver groups, target scores, source–target scores |
| `select_receiver_group_count` | `target/release/select_receiver_group_count CONFIG.json` | Checkpoint, coordinates, candidate group counts | Cross-validation scores and selected group count |
| `validate_selected_triplets` | `target/release/validate_selected_triplets CONFIG.json` | Checkpoint, coordinates, selected source–target triplets | Spatial holdout and stability results |
| `export_cell_cell_interactions` | `target/release/export_cell_cell_interactions CONFIG.json` | Checkpoint, coordinates, source–target scores | Ranked sender–receiver cell edges |
| `validata_ccc_database` | `target/release/validata_ccc_database CONFIG.json` | Expression, cell edges, ligand–receptor database | Distance-matched support statistics |
| `evaluate_synthetic_recovery` | `target/release/evaluate_synthetic_recovery ROOT OUTPUT MASK_MODE EXPRESSION_THRESHOLD` | Synthetic data and checkpoints | Recovery scores and relation ranks |
| `benchmark_model_components` | `target/release/benchmark_model_components ROOT OUTPUT MASK_MODE EXPRESSION_THRESHOLD SEED_START SEED_END` | Synthetic data and checkpoints | Component benchmark metrics |

Dataset workflows: [synthetic](experiments/synthetic/README.md), [external-method benchmark](experiments/benchmark/README.md), [spatial CRISPR](experiments/crispr/README.md), and [Xenium liver](experiments/xenium/README.md).
