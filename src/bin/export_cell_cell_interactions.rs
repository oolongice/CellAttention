//! Export model-derived cell-to-cell influence edges for selected
//! receiver_group/source/target triplets without refitting the model.

use cell_attention::{
    data::{read_expression, read_ids},
    physical::{
        ReactionDiffusion, ReceiverFieldMaskConfig, ReceiverFieldMaskMode, SparseSpatialKernel,
        SpatialDimension, SpatialKernelConfig,
    },
    source_target::{
        ReceiverGroupClusteringMethod, ReceiverGroupCountConfig, ReceiverGroupCountMode,
        fit_embedding_receiver_groups,
    },
};
use ndarray::Array2;
use rayon::prelude::*;
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Triplet {
    cluster: usize,
    source_gene: String,
    target_gene: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    checkpoint_dir: PathBuf,
    coordinates: PathBuf,
    source_target_scores: PathBuf,
    output_dir: PathBuf,
    #[serde(default)]
    annotations: Option<PathBuf>,
    embedding_dimensions: usize,
    #[serde(alias = "context_count")]
    receiver_group_count: usize,
    #[serde(default, alias = "context_clustering_method")]
    receiver_group_clustering_method: ReceiverGroupClusteringMethod,
    #[serde(default, alias = "context_pca_components")]
    receiver_group_pca_components: Option<usize>,
    maximum_fit_cells: usize,
    length_scale: f64,
    maximum_distance: f64,
    minimum_distance: f64,
    #[serde(default = "unlimited")]
    maximum_neighbors: usize,
    #[serde(default = "default_block_size")]
    target_block_size: usize,
    #[serde(default = "default_top_fraction")]
    top_fraction: f64,
    #[serde(default = "default_maximum_edges")]
    maximum_edges_per_triplet: usize,
    #[serde(default = "default_residual_clip")]
    residual_clip: f64,
    #[serde(default = "default_true")]
    require_direction_support: bool,
    #[serde(default)]
    receiver_field_mask: ReceiverFieldMaskConfig,
    triplets: Vec<Triplet>,
}

fn unlimited() -> usize {
    9999
}
fn default_block_size() -> usize {
    512
}
fn default_top_fraction() -> f64 {
    0.5
}
fn default_maximum_edges() -> usize {
    1000
}
fn default_residual_clip() -> f64 {
    3.0
}
fn default_true() -> bool {
    true
}

#[derive(Clone)]
struct Edge {
    source: usize,
    receiver: usize,
    distance: f64,
    kernel_weight: f64,
    source_expression: f64,
    receiver_group_probability: f64,
    target_residual_z: f64,
    signed_contribution: f64,
    influence_score: f64,
    receiver_normalized_attribution: f64,
}

fn main() -> Result<(), String> {
    let config_path = std::env::args()
        .nth(1)
        .ok_or("usage: export_cell_cell_interactions CONFIG.json")?;
    let config: Config =
        serde_json::from_str(&fs::read_to_string(config_path).map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
    validate_config(&config)?;
    fs::create_dir_all(&config.output_dir).map_err(|error| error.to_string())?;

    let cells = read_ids(config.checkpoint_dir.join("cell_ids.txt"))?;
    let genes = read_ids(config.checkpoint_dir.join("gene_ids.txt"))?;
    let gene_index: BTreeMap<&str, usize> = genes
        .iter()
        .enumerate()
        .map(|(index, gene)| (gene.as_str(), index))
        .collect();
    let annotations = match &config.annotations {
        Some(path) => {
            let values = read_ids(path)?;
            if values.len() != cells.len() {
                return Err(format!(
                    "annotation count {} != cell count {}",
                    values.len(),
                    cells.len()
                ));
            }
            Some(values)
        }
        None => None,
    };
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
    )?
    .mapv(f64::from);
    let coordinates = read_expression(&config.coordinates, cells.len(), 2)?.mapv(f64::from);

    eprintln!(
        "stage=receiver_groups cells={} genes={} receiver_groups={}",
        cells.len(),
        genes.len(),
        config.receiver_group_count
    );
    let receiver_group_config = ReceiverGroupCountConfig {
        mode: ReceiverGroupCountMode::Manual(config.receiver_group_count),
        clustering_method: config.receiver_group_clustering_method,
        pca_components: config.receiver_group_pca_components,
        kmeans_restarts: 8,
        kmeans_iterations: 100,
        ..Default::default()
    };
    let receiver_group_model = fit_embedding_receiver_groups(
        &embedding,
        config.receiver_group_count,
        &receiver_group_config,
    )?;
    let probabilities = receiver_group_model.probabilities(&embedding)?;
    let labels: Vec<usize> = (0..cells.len())
        .map(|cell| {
            (0..config.receiver_group_count)
                .max_by(|&left, &right| {
                    probabilities[[cell, left]].total_cmp(&probabilities[[cell, right]])
                })
                .unwrap()
        })
        .collect();
    let fit_cells = balanced_indices(
        &labels,
        config.receiver_group_count,
        config.maximum_fit_cells,
    );

    eprintln!("stage=kernel maximum_distance={}", config.maximum_distance);
    let equation = ReactionDiffusion {
        diffusion: config.length_scale.powi(2),
        degradation: 1.0,
        production: 1.0,
        min_distance: config.minimum_distance,
        dimension: SpatialDimension::TwoD,
    };
    let kernel = SparseSpatialKernel::build(
        &coordinates,
        None,
        equation,
        &SpatialKernelConfig {
            max_distance: config.maximum_distance,
            max_neighbors: config.maximum_neighbors,
            include_self: false,
            target_block_size: config.target_block_size,
        },
    )?;
    let beta = read_selected_betas(&config.source_target_scores)?;

    let edge_path = config.output_dir.join("cell_cell_interaction_edges.csv");
    let summary_path = config.output_dir.join("cell_cell_interaction_summary.csv");
    let mut edge_writer = BufWriter::new(fs::File::create(&edge_path).map_err(|e| e.to_string())?);
    writeln!(edge_writer, "cluster,source_gene,target_gene,sender_cell_id,receiver_cell_id,sender_cell_type,receiver_cell_type,distance,kernel_weight,source_expression,receiver_group_probability,target_residual_z,signed_beta,signed_contribution,influence_score,receiver_normalized_attribution,rank_within_triplet")
        .map_err(|e| e.to_string())?;
    let mut summary_writer =
        BufWriter::new(fs::File::create(&summary_path).map_err(|e| e.to_string())?);
    writeln!(summary_writer, "cluster,source_gene,target_gene,signed_beta,candidate_receiver_cells,direction_supported_edges,retained_by_fraction,retained_edges,top_fraction,maximum_edges_per_triplet,influence_score_cutoff")
        .map_err(|e| e.to_string())?;

    for triplet in &config.triplets {
        let receiver_group = triplet
            .cluster
            .checked_sub(1)
            .ok_or("clusters are one-based")?;
        if receiver_group >= config.receiver_group_count {
            return Err(format!(
                "cluster {} exceeds receiver_group count {}",
                triplet.cluster, config.receiver_group_count
            ));
        }
        let source_gene = *gene_index
            .get(triplet.source_gene.as_str())
            .ok_or_else(|| format!("unknown source gene {}", triplet.source_gene))?;
        let target_gene = *gene_index
            .get(triplet.target_gene.as_str())
            .ok_or_else(|| format!("unknown target gene {}", triplet.target_gene))?;
        let signed_beta = *beta
            .get(&(
                receiver_group,
                triplet.source_gene.clone(),
                triplet.target_gene.clone(),
            ))
            .ok_or_else(|| {
                format!(
                    "missing selected beta for cluster {}: {} -> {}",
                    triplet.cluster, triplet.source_gene, triplet.target_gene
                )
            })?;
        if signed_beta == 0.0 {
            return Err(format!(
                "selected beta is zero for cluster {}: {} -> {}",
                triplet.cluster, triplet.source_gene, triplet.target_gene
            ));
        }

        eprintln!(
            "stage=edges cluster={} source={} target={}",
            triplet.cluster, triplet.source_gene, triplet.target_gene
        );
        let receiver_cells: Vec<usize> = (0..cells.len())
            .filter(|&cell| labels[cell] == receiver_group)
            .collect();
        // Match the standardization population used by run_source_target_analysis.
        let (_field_mean, field_sd) = field_moments(
            &kernel,
            &raw,
            source_gene,
            &fit_cells,
            config.receiver_field_mask,
        );
        let (residual_mean, residual_sd) = column_moments(&residual, target_gene, &fit_cells);

        let (supported_edges, mut edges) = receiver_cells
            .par_iter()
            .map(|&receiver| {
                if config.receiver_field_mask.mode == ReceiverFieldMaskMode::HardExpressionExclusion
                    && raw[[receiver, source_gene]]
                        > config.receiver_field_mask.expression_threshold
                {
                    return (0usize, Vec::new());
                }
                let probability = probabilities[[receiver, receiver_group]];
                let residual_z =
                    (residual[[receiver, target_gene]] as f64 - residual_mean) / residual_sd;
                let support =
                    (residual_z.abs().min(config.residual_clip) / config.residual_clip).max(0.0);
                let start = kernel.row_offsets[receiver];
                let end = kernel.row_offsets[receiver + 1];
                let mut edges = Vec::new();
                for edge_index in start..end {
                    let source = kernel.source_indices[edge_index];
                    let expression = raw[[source, source_gene]] as f64;
                    if expression <= 0.0 {
                        continue;
                    }
                    let weight = kernel.weights[edge_index] as f64;
                    let contribution = probability * signed_beta * weight * expression / field_sd;
                    if config.require_direction_support && contribution * residual_z <= 0.0 {
                        continue;
                    }
                    let influence = contribution.abs() * support;
                    if influence <= 0.0 || !influence.is_finite() {
                        continue;
                    }
                    let distance = euclidean_distance(&coordinates, source, receiver);
                    edges.push(Edge {
                        source,
                        receiver,
                        distance,
                        kernel_weight: weight,
                        source_expression: expression,
                        receiver_group_probability: probability,
                        target_residual_z: residual_z,
                        signed_contribution: contribution * support,
                        influence_score: influence,
                        receiver_normalized_attribution: 0.0,
                    });
                }
                let total: f64 = edges.iter().map(|edge| edge.influence_score).sum();
                if total > 0.0 {
                    for edge in &mut edges {
                        edge.receiver_normalized_attribution = edge.influence_score / total;
                    }
                }
                let count = edges.len();
                edges.sort_unstable_by(|left, right| {
                    right.influence_score.total_cmp(&left.influence_score)
                });
                edges.truncate(config.maximum_edges_per_triplet);
                (count, edges)
            })
            .reduce(
                || (0usize, Vec::new()),
                |(left_count, left), (right_count, right)| {
                    (
                        left_count + right_count,
                        merge_top_edges(left, right, config.maximum_edges_per_triplet),
                    )
                },
            );
        edges.par_sort_unstable_by(|left, right| {
            right.influence_score.total_cmp(&left.influence_score)
        });
        let retained_by_fraction =
            ((supported_edges as f64 * config.top_fraction).ceil() as usize).min(supported_edges);
        let retained = retained_by_fraction.min(config.maximum_edges_per_triplet);
        edges.truncate(retained);
        let cutoff = edges.last().map(|edge| edge.influence_score).unwrap_or(0.0);

        for (rank, edge) in edges.iter().enumerate() {
            let source_type = annotations
                .as_ref()
                .map(|x| x[edge.source].as_str())
                .unwrap_or("");
            let receiver_type = annotations
                .as_ref()
                .map(|x| x[edge.receiver].as_str())
                .unwrap_or("");
            writeln!(edge_writer, "{},{},{},{},{},{},{},{:.8},{:.10},{:.8},{:.8},{:.8},{:.10},{:.10},{:.10},{:.10},{}",
                triplet.cluster, triplet.source_gene, triplet.target_gene,
                cells[edge.source], cells[edge.receiver], source_type, receiver_type,
                edge.distance, edge.kernel_weight, edge.source_expression,
                edge.receiver_group_probability, edge.target_residual_z, signed_beta,
                edge.signed_contribution, edge.influence_score,
                edge.receiver_normalized_attribution, rank + 1,
            ).map_err(|e| e.to_string())?;
        }
        writeln!(
            summary_writer,
            "{},{},{},{:.10},{},{},{},{},{:.6},{},{:.10}",
            triplet.cluster,
            triplet.source_gene,
            triplet.target_gene,
            signed_beta,
            receiver_cells.len(),
            supported_edges,
            retained_by_fraction,
            retained,
            config.top_fraction,
            config.maximum_edges_per_triplet,
            cutoff,
        )
        .map_err(|e| e.to_string())?;
    }
    edge_writer.flush().map_err(|e| e.to_string())?;
    summary_writer.flush().map_err(|e| e.to_string())?;
    fs::write(config.output_dir.join("metadata.json"), serde_json::to_string_pretty(&serde_json::json!({
        "definition": "model_derived_potential_cell_cell_influence",
        "receiver_rule": "hard_receiver_group_assignment",
        "receiver_group_clustering_method": config.receiver_group_clustering_method,
        "receiver_group_pca_components": config.receiver_group_pca_components,
        "edge_contribution": "p_receiver_group * signed_beta * kernel_weight * source_expression / source_field_sd",
        "residual_support": "direction_match * min(abs(target_residual_z), residual_clip) / residual_clip",
        "top_fraction": config.top_fraction,
        "maximum_edges_per_triplet": config.maximum_edges_per_triplet,
        "residual_clip": config.residual_clip,
        "require_direction_support": config.require_direction_support,
        "receiver_field_mask": config.receiver_field_mask,
        "kernel": {
            "length_scale": config.length_scale,
            "maximum_distance": config.maximum_distance,
            "minimum_distance": config.minimum_distance,
            "maximum_neighbors": config.maximum_neighbors,
            "include_self": false
        },
        "triplets": config.triplets.len(),
        "fit_cells_for_standardization": fit_cells.len(),
        "refit": false
    })).unwrap()).map_err(|e| e.to_string())?;
    eprintln!("stage=complete output={}", edge_path.display());
    Ok(())
}

fn validate_config(config: &Config) -> Result<(), String> {
    if config.receiver_group_count == 0 {
        return Err("receiver_group_count must be > 0".to_string());
    }
    if config.maximum_fit_cells == 0 {
        return Err("maximum_fit_cells must be > 0".to_string());
    }
    if !config.top_fraction.is_finite() || config.top_fraction <= 0.0 || config.top_fraction > 1.0 {
        return Err("top_fraction must be in (0, 1]".to_string());
    }
    if config.maximum_edges_per_triplet == 0 {
        return Err("maximum_edges_per_triplet must be > 0".to_string());
    }
    if !config.residual_clip.is_finite() || config.residual_clip <= 0.0 {
        return Err("residual_clip must be finite and > 0".to_string());
    }
    if config.triplets.is_empty() {
        return Err("triplets must not be empty".to_string());
    }
    Ok(())
}

fn read_selected_betas(path: &Path) -> Result<HashMap<(usize, String, String), f64>, String> {
    let text = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or("source-target score table is empty")?
        .split(',')
        .collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|value| *value == name)
            .ok_or_else(|| format!("missing column {name} in source-target score table"))
    };
    let receiver_group_col = column("receiver_group")?;
    let source_col = column("source_gene")?;
    let target_col = column("target_gene")?;
    let beta_col = column("signed_beta")?;
    let selected_col = column("selected")?;
    let mut result = HashMap::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').collect();
        if !matches!(
            fields[selected_col].trim().to_ascii_lowercase().as_str(),
            "true" | "1"
        ) {
            continue;
        }
        result.insert(
            (
                fields[receiver_group_col]
                    .parse::<usize>()
                    .map_err(|e| e.to_string())?,
                fields[source_col].to_string(),
                fields[target_col].to_string(),
            ),
            fields[beta_col].parse::<f64>().map_err(|e| e.to_string())?,
        );
    }
    Ok(result)
}

fn field_moments(
    kernel: &SparseSpatialKernel,
    expression: &Array2<f32>,
    gene: usize,
    cells: &[usize],
    receiver_field_mask: ReceiverFieldMaskConfig,
) -> (f64, f64) {
    let values: Vec<f64> = cells
        .par_iter()
        .map(|&receiver| {
            if receiver_field_mask.mode == ReceiverFieldMaskMode::HardExpressionExclusion
                && expression[[receiver, gene]] > receiver_field_mask.expression_threshold
            {
                return 0.0;
            }
            (kernel.row_offsets[receiver]..kernel.row_offsets[receiver + 1])
                .map(|edge| {
                    kernel.weights[edge] as f64
                        * expression[[kernel.source_indices[edge], gene]] as f64
                })
                .sum()
        })
        .collect();
    moments(values.into_iter())
}

fn column_moments(matrix: &Array2<f32>, column: usize, cells: &[usize]) -> (f64, f64) {
    moments(cells.iter().map(|&cell| matrix[[cell, column]] as f64))
}

fn moments(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let values: Vec<f64> = values.collect();
    let mean = values.iter().sum::<f64>() / values.len().max(1) as f64;
    let sd = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len().max(1) as f64)
        .sqrt()
        .max(1e-8);
    (mean, sd)
}

fn euclidean_distance(coordinates: &Array2<f64>, left: usize, right: usize) -> f64 {
    (0..coordinates.ncols())
        .map(|axis| {
            let difference = coordinates[[left, axis]] - coordinates[[right, axis]];
            difference * difference
        })
        .sum::<f64>()
        .sqrt()
}

fn balanced_indices(labels: &[usize], receiver_groups: usize, maximum: usize) -> Vec<usize> {
    if labels.len() <= maximum {
        return (0..labels.len()).collect();
    }
    let per_receiver_group = (maximum / receiver_groups).max(1);
    let mut selected = Vec::new();
    for receiver_group in 0..receiver_groups {
        let members: Vec<usize> = (0..labels.len())
            .filter(|&cell| labels[cell] == receiver_group)
            .collect();
        let take = per_receiver_group.min(members.len());
        for offset in 0..take {
            selected.push(members[offset * members.len() / take]);
        }
    }
    selected.sort_unstable();
    selected
}

fn merge_top_edges(mut left: Vec<Edge>, mut right: Vec<Edge>, maximum: usize) -> Vec<Edge> {
    left.append(&mut right);
    left.sort_unstable_by(|a, b| b.influence_score.total_cmp(&a.influence_score));
    left.truncate(maximum);
    left
}

#[cfg(test)]
mod tests {
    use super::*;

    fn edge(score: f64) -> Edge {
        Edge {
            source: 0,
            receiver: 0,
            distance: 1.0,
            kernel_weight: 1.0,
            source_expression: 1.0,
            receiver_group_probability: 1.0,
            target_residual_z: 1.0,
            signed_contribution: score,
            influence_score: score,
            receiver_normalized_attribution: 1.0,
        }
    }

    #[test]
    fn bounded_merge_keeps_only_highest_scores() {
        let merged = merge_top_edges(vec![edge(4.0), edge(1.0)], vec![edge(3.0), edge(2.0)], 2);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].influence_score, 4.0);
        assert_eq!(merged[1].influence_score, 3.0);
    }

    #[test]
    fn balanced_indices_match_main_analysis_sampling() {
        let labels = vec![0, 0, 0, 0, 1, 1];
        assert_eq!(balanced_indices(&labels, 2, 4), vec![0, 2, 4, 5]);
    }
}
