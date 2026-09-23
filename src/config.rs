use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub dropout: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub mask_probability: f32,
    pub seed: u64,
    #[serde(default = "default_training_shuffle")]
    pub shuffle: bool,
    #[serde(default = "default_inference_mask_chunk_size")]
    pub inference_mask_chunk_size: usize,
    #[serde(default)]
    pub max_cells_per_epoch: usize,
    #[serde(default = "default_epoch_sampling")]
    pub epoch_sampling: String,
    #[serde(default)]
    pub minimum_cells_per_sample: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    pub expression_matrix: String,
    pub cell_ids: String,
    pub gene_ids: String,
    #[serde(default)]
    pub sample_ids: Option<String>,
    pub output_dir: String,
    #[serde(default)]
    pub device_index: usize,
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

fn default_training_shuffle() -> bool {
    true
}

fn default_inference_mask_chunk_size() -> usize {
    1
}

fn default_epoch_sampling() -> String {
    "stratified".to_string()
}

impl RunConfig {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let text = fs::read_to_string(path.as_ref())
            .map_err(|error| format!("failed to read config: {error}"))?;
        serde_json::from_str(&text).map_err(|error| format!("invalid JSON config: {error}"))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointMetadata {
    pub format_version: usize,
    pub num_cells: usize,
    pub num_genes: usize,
    pub model: ModelConfig,
    pub expression_matrix: String,
    pub cell_ids_file: String,
    pub gene_ids_file: String,
    #[serde(default)]
    pub sample_ids_file: Option<String>,
    #[serde(default)]
    pub device_index: usize,
}
