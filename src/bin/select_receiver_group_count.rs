//! Data-driven receiver-receiver_group count selection for a frozen real-data checkpoint.
use cell_attention::{
    data::{read_expression, read_ids},
    physical::{
        ReactionDiffusion, ReceiverFieldMaskConfig, SpatialDimension, SpatialKernelConfig,
        apply_receiver_field_mask_inplace, compute_source_fields,
    },
    source_target::{
        ReceiverGroupClusteringMethod, ReceiverGroupCountConfig, ReceiverGroupCountMode,
        SparseEffectConfig, select_receiver_group_count,
    },
};
use ndarray::Array2;
use serde::Deserialize;
use std::{fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    dataset: String,
    checkpoint_dir: PathBuf,
    coordinates: PathBuf,
    output_dir: PathBuf,
    #[serde(default = "two_dimensions")]
    coordinate_dimensions: usize,
    embedding_dimensions: usize,
    maximum_selection_cells: usize,
    candidates: Vec<usize>,
    #[serde(default, alias = "context_clustering_method")]
    receiver_group_clustering_method: ReceiverGroupClusteringMethod,
    #[serde(default, alias = "context_pca_components")]
    receiver_group_pca_components: Option<usize>,
    spatial_folds: usize,
    spatial_block_width: f64,
    kmeans_restarts: usize,
    kmeans_iterations: usize,
    seed: u64,
    length_scale: f64,
    maximum_distance: f64,
    minimum_distance: f64,
    maximum_neighbors: usize,
    target_block_size: usize,
    top_k_sources: usize,
    #[serde(default)]
    receiver_field_mask: ReceiverFieldMaskConfig,
}
fn two_dimensions() -> usize {
    2
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: select_receiver_group_count CONFIG.json")?;
    let config: Config =
        serde_json::from_str(&fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    fs::create_dir_all(&config.output_dir).map_err(|e| e.to_string())?;
    let cells = read_ids(config.checkpoint_dir.join("cell_ids.txt"))?;
    let genes = read_ids(config.checkpoint_dir.join("gene_ids.txt"))?;
    eprintln!(
        "dataset={} stage=load cells={} genes={}",
        config.dataset,
        cells.len(),
        genes.len()
    );
    let raw = read_expression(
        config.checkpoint_dir.join("raw_expression.csv"),
        cells.len(),
        genes.len(),
    )?;
    let residual = read_expression(
        config.checkpoint_dir.join("residuals.csv"),
        cells.len(),
        genes.len(),
    )?;
    let embedding = read_expression(
        config.checkpoint_dir.join("cell_embeddings.csv"),
        cells.len(),
        config.embedding_dimensions,
    )?;
    if !matches!(config.coordinate_dimensions, 2 | 3) {
        return Err("coordinate_dimensions must be 2 or 3".to_string());
    }
    let spatial_dimension = if config.coordinate_dimensions == 3 {
        SpatialDimension::ThreeD
    } else {
        SpatialDimension::TwoD
    };
    let coords = read_expression(
        &config.coordinates,
        cells.len(),
        config.coordinate_dimensions,
    )?;
    let selected = systematic_indices(cells.len(), config.maximum_selection_cells);
    eprintln!(
        "dataset={} stage=physical_fields all_cells={} selection_cells={}",
        config.dataset,
        cells.len(),
        selected.len()
    );
    let mut fields = compute_source_fields(
        &coords.mapv(f64::from),
        &raw,
        None,
        ReactionDiffusion {
            diffusion: config.length_scale.powi(2),
            degradation: 1.,
            production: 1.,
            min_distance: config.minimum_distance,
            dimension: spatial_dimension,
        },
        &SpatialKernelConfig {
            max_distance: config.maximum_distance,
            max_neighbors: config.maximum_neighbors,
            include_self: false,
            target_block_size: config.target_block_size,
        },
    )?;
    let mask_summary =
        apply_receiver_field_mask_inplace(&mut fields, &raw, config.receiver_field_mask)?;
    eprintln!(
        "dataset={} stage=receiver_field_mask mode={:?} masked_fraction={:.6}",
        config.dataset, config.receiver_field_mask.mode, mask_summary.masked_fraction
    );
    let x = standardize_selected(&fields, &selected);
    let y = standardize_selected_f32(&residual, &selected);
    let e = take_selected_f32(&embedding, &selected);
    let c = take_selected_f32(&coords, &selected);
    drop(fields);
    drop(raw);
    drop(residual);
    drop(embedding);
    drop(coords);
    let receiver_group_config = ReceiverGroupCountConfig {
        mode: ReceiverGroupCountMode::DataDriven,
        clustering_method: config.receiver_group_clustering_method,
        pca_components: config.receiver_group_pca_components,
        candidates: config.candidates.clone(),
        spatial_folds: config.spatial_folds,
        spatial_block_width: config.spatial_block_width,
        minimum_effective_receiver_group_mass: 30.,
        kmeans_restarts: config.kmeans_restarts,
        kmeans_iterations: config.kmeans_iterations,
        seed: config.seed,
    };
    let effect_config = SparseEffectConfig {
        top_k_sources: config.top_k_sources,
        ..Default::default()
    };
    eprintln!(
        "dataset={} stage=receiver_group_selection candidates={:?} folds={}",
        config.dataset, config.candidates, config.spatial_folds
    );
    let result =
        select_receiver_group_count(&e, &x, &y, &c, &receiver_group_config, &effect_config)?;
    let mut csv="receiver_group_count,baseline_mse,physical_mse,improvement_fraction,improvement_standard_error,mean_active_effects,minimum_effective_train_mass,eligible,selected\n".to_string();
    for row in &result.candidates {
        csv.push_str(&format!(
            "{},{:.10},{:.10},{:.10},{:.10},{:.2},{:.3},{},{}\n",
            row.receiver_group_count,
            row.baseline_mse,
            row.physical_mse,
            row.improvement_fraction,
            row.improvement_standard_error,
            row.mean_active_effects,
            row.minimum_effective_train_mass,
            row.eligible,
            row.receiver_group_count == result.selected_receiver_group_count
        ));
    }
    fs::write(
        config.output_dir.join("receiver_group_count_selection.csv"),
        csv,
    )
    .map_err(|e| e.to_string())?;
    let equation = if config.coordinate_dimensions == 3 {
        "3d_reaction_diffusion_yukawa"
    } else {
        "2d_reaction_diffusion_k0"
    };
    let metadata = serde_json::json!({"dataset":config.dataset,"cells":cells.len(),"genes":genes.len(),"selection_cells":selected.len(),"selection_indices":"deterministic_systematic_full_order","selected_receiver_group_count":result.selected_receiver_group_count,"selection_rule":result.selection_rule,"receiver_group_clustering_method":config.receiver_group_clustering_method,"receiver_group_pca_components":config.receiver_group_pca_components,"candidates":config.candidates,"spatial_folds":config.spatial_folds,"spatial_block_width":config.spatial_block_width,"kmeans_restarts":config.kmeans_restarts,"kmeans_iterations":config.kmeans_iterations,"seed":config.seed,"kernel":{"equation":equation,"coordinate_dimensions":config.coordinate_dimensions,"length_scale":config.length_scale,"maximum_distance":config.maximum_distance,"minimum_distance":config.minimum_distance,"maximum_neighbors":config.maximum_neighbors,"include_self":false},"receiver_field_mask":{"mode":config.receiver_field_mask.mode,"expression_threshold":config.receiver_field_mask.expression_threshold,"masked_fraction":mask_summary.masked_fraction},"source_target":{"top_k_sources":config.top_k_sources,"l1_penalty":effect_config.l1_penalty,"l2_penalty":effect_config.l2_penalty,"minimum_training_improvement":effect_config.minimum_training_improvement}});
    fs::write(
        config.output_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "dataset={} stage=complete selected_receiver_group_count={}",
        config.dataset, result.selected_receiver_group_count
    );
    Ok(())
}
fn systematic_indices(total: usize, maximum: usize) -> Vec<usize> {
    if total <= maximum {
        return (0..total).collect();
    }
    (0..maximum).map(|i| i * total / maximum).collect()
}
fn take_selected_f32(a: &Array2<f32>, idx: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((idx.len(), a.ncols()), |(i, j)| a[[idx[i], j]] as f64)
}
fn standardize_selected_f32(a: &Array2<f32>, idx: &[usize]) -> Array2<f64> {
    let mut z = Array2::<f64>::zeros((idx.len(), a.ncols()));
    for j in 0..a.ncols() {
        let mean = idx.iter().map(|&i| a[[i, j]] as f64).sum::<f64>() / idx.len() as f64;
        let sd = (idx
            .iter()
            .map(|&i| (a[[i, j]] as f64 - mean).powi(2))
            .sum::<f64>()
            / idx.len() as f64)
            .sqrt()
            .max(1e-8);
        for (k, &i) in idx.iter().enumerate() {
            z[[k, j]] = (a[[i, j]] as f64 - mean) / sd;
        }
    }
    z
}
fn standardize_selected(a: &Array2<f32>, idx: &[usize]) -> Array2<f64> {
    standardize_selected_f32(a, idx)
}
