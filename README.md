# CellAttention

CellAttention trains a masked-gene Transformer and estimates spatial source–target gene relations.

## Build and run

```bash
cargo build --release --locked
target/release/cell_attention TRAIN_CONFIG.json
target/release/run_source_target_analysis ANALYSIS_CONFIG.json
```

`cell_attention` writes a Transformer checkpoint, expression residuals, and cell embeddings. `run_source_target_analysis` uses that checkpoint and spatial coordinates to write receiver groups and source–target scores. Other analysis binaries are built in `target/release/`.

Dataset workflows: [synthetic](experiments/synthetic/README.md), [benchmark](experiments/benchmark/README.md), [spatial CRISPR](experiments/crispr/README.md), [Xenium liver](experiments/xenium/README.md), [Slide-seqV2 hippocampus](experiments/slideseqv2/README.md), [MOSTA](experiments/mosta/README.md), and [3D weMERFISH](experiments/merfish/README.md).
