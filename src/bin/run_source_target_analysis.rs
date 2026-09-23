use cell_attention::{
    data::{read_expression, read_ids},
    physical::{
        ReactionDiffusion, ReceiverFieldMaskConfig, SpatialDimension, SpatialKernelConfig,
        apply_receiver_field_mask_inplace, compute_source_fields,
    },
    source_target::{
        ReceiverGroupClusteringMethod, ReceiverGroupCountConfig, ReceiverGroupCountMode,
        SparseEffectConfig, fit_embedding_receiver_groups, fit_fixed_receiver_group_effects,
    },
};
use ndarray::Array2;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    dataset: String,
    checkpoint_dir: PathBuf,
    coordinates: PathBuf,
    output_dir: PathBuf,
    #[serde(default = "two_dimensions")]
    coordinate_dimensions: usize,
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
    #[serde(default = "block_size")]
    target_block_size: usize,
    #[serde(default = "top_k")]
    top_k_sources: usize,
    #[serde(default)]
    receiver_field_mask: ReceiverFieldMaskConfig,
}
fn two_dimensions() -> usize {
    2
}
fn unlimited() -> usize {
    9999
}
fn block_size() -> usize {
    512
}
fn top_k() -> usize {
    8
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: run_source_target_analysis CONFIG.json")?;
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
    )?
    .mapv(f64::from);
    if !matches!(config.coordinate_dimensions, 2 | 3) {
        return Err("coordinate_dimensions must be 2 or 3".to_string());
    }
    let spatial_dimension = if config.coordinate_dimensions == 3 {
        SpatialDimension::ThreeD
    } else {
        SpatialDimension::TwoD
    };
    let coords_f32 = read_expression(
        &config.coordinates,
        cells.len(),
        config.coordinate_dimensions,
    )?;
    let coords = coords_f32.mapv(f64::from);
    eprintln!(
        "dataset={} stage=receiver_group_kmeans receiver_groups={}",
        config.dataset, config.receiver_group_count
    );
    let cc = ReceiverGroupCountConfig {
        mode: ReceiverGroupCountMode::Manual(config.receiver_group_count),
        clustering_method: config.receiver_group_clustering_method,
        pca_components: config.receiver_group_pca_components,
        kmeans_restarts: 8,
        kmeans_iterations: 100,
        ..Default::default()
    };
    let receiver_group_model =
        fit_embedding_receiver_groups(&embedding, config.receiver_group_count, &cc)?;
    let probabilities = receiver_group_model.probabilities(&embedding)?;
    let labels: Vec<usize> = (0..cells.len())
        .map(|i| {
            (0..config.receiver_group_count)
                .max_by(|&a, &b| probabilities[[i, a]].total_cmp(&probabilities[[i, b]]))
                .unwrap()
        })
        .collect();
    write_assignments(&config.output_dir, &cells, &labels, &probabilities)?;
    if let Some(path) = &config.annotations {
        write_annotation_composition(
            &config.output_dir,
            path,
            &labels,
            config.receiver_group_count,
        )?;
    }
    let selected = balanced_indices(
        &labels,
        config.receiver_group_count,
        config.maximum_fit_cells,
    );
    eprintln!(
        "dataset={} stage=physical_fields all_cells={} fit_cells={}",
        config.dataset,
        cells.len(),
        selected.len()
    );
    let mut fields = compute_source_fields(
        &coords,
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
        "dataset={} stage=receiver_field_mask mode={:?} threshold={} masked_fraction={:.6}",
        config.dataset,
        config.receiver_field_mask.mode,
        config.receiver_field_mask.expression_threshold,
        mask_summary.masked_fraction
    );
    let x = standardize_selected_f32(&fields, &selected);
    let y = standardize_selected_f32(&residual, &selected);
    let p = take_rows(&probabilities, &selected);
    drop(fields);
    drop(raw);
    eprintln!("dataset={} stage=fit_source_target", config.dataset);
    let ec = SparseEffectConfig {
        top_k_sources: config.top_k_sources,
        ..Default::default()
    };
    let model = fit_fixed_receiver_group_effects(&x, &y, &p, &ec)?;
    write_results(&config.output_dir, &genes, &model, 10)?;
    let equation = if config.coordinate_dimensions == 3 {
        "3d_reaction_diffusion_yukawa"
    } else {
        "2d_reaction_diffusion_k0"
    };
    let metadata = serde_json::json!({"dataset":config.dataset,"cells":cells.len(),"genes":genes.len(),"embedding_dimensions":config.embedding_dimensions,"receiver_group_count":config.receiver_group_count,"fit_cells":selected.len(),"maximum_fit_cells":config.maximum_fit_cells,"receiver_group_method":config.receiver_group_clustering_method,"receiver_group_pca_components":config.receiver_group_pca_components,"target_definition":"physical_explanation_of_transformer_residual","kernel":{"equation":equation,"coordinate_dimensions":config.coordinate_dimensions,"length_scale":config.length_scale,"maximum_distance":config.maximum_distance,"minimum_distance":config.minimum_distance,"maximum_neighbors":config.maximum_neighbors,"include_self":false},"source_target":{"top_k_sources":config.top_k_sources,"l1_penalty":ec.l1_penalty,"l2_penalty":ec.l2_penalty,"minimum_training_improvement":ec.minimum_training_improvement},"receiver_field_mask":{"mode":config.receiver_field_mask.mode,"expression_threshold":config.receiver_field_mask.expression_threshold,"masked_entries":mask_summary.masked_entries,"total_entries":mask_summary.total_entries,"masked_fraction":mask_summary.masked_fraction,"applied_before_standardization":true},"checkpoint_dir":config.checkpoint_dir,"coordinates":config.coordinates});
    fs::write(
        config.output_dir.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    eprintln!(
        "dataset={} stage=complete active_effects={}",
        config.dataset, model.active_effects
    );
    Ok(())
}

fn balanced_indices(labels: &[usize], receiver_groups: usize, maximum: usize) -> Vec<usize> {
    if labels.len() <= maximum {
        return (0..labels.len()).collect();
    }
    let per = (maximum / receiver_groups).max(1);
    let mut out = Vec::new();
    for c in 0..receiver_groups {
        let v: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == c).collect();
        let take = per.min(v.len());
        for k in 0..take {
            out.push(v[k * v.len() / take]);
        }
    }
    out.sort_unstable();
    out
}
fn take_rows(a: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((idx.len(), a.ncols()), |(i, j)| a[[idx[i], j]])
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
fn write_assignments(
    out: &Path,
    cells: &[String],
    labels: &[usize],
    p: &Array2<f64>,
) -> Result<(), String> {
    let mut s = "cell_id,receiver_group,maximum_probability\n".to_string();
    for i in 0..cells.len() {
        s.push_str(&format!(
            "{},{},{:.7}\n",
            cells[i],
            labels[i],
            p[[i, labels[i]]]
        ));
    }
    fs::write(out.join("receiver_group_assignments.csv"), s).map_err(|e| e.to_string())
}
fn write_annotation_composition(
    out: &Path,
    path: &Path,
    labels: &[usize],
    receiver_groups: usize,
) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let annotations: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .collect();
    if annotations.len() != labels.len() {
        return Err(format!(
            "annotation count {} != cell count {}",
            annotations.len(),
            labels.len()
        ));
    }
    let mut counts = BTreeMap::<(usize, String), usize>::new();
    for (i, a) in annotations.iter().enumerate() {
        *counts.entry((labels[i], (*a).to_string())).or_default() += 1;
    }
    let totals: Vec<usize> = (0..receiver_groups)
        .map(|c| labels.iter().filter(|&&x| x == c).count())
        .collect();
    let mut s = "receiver_group,annotation,count,fraction_within_context\n".to_string();
    for ((c, a), n) in counts {
        s.push_str(&format!(
            "{c},{a},{n},{:.7}\n",
            n as f64 / totals[c].max(1) as f64
        ));
    }
    fs::write(out.join("receiver_group_annotation_composition.csv"), s).map_err(|e| e.to_string())
}
fn write_results(
    out: &Path,
    genes: &[String],
    m: &cell_attention::source_target::FixedReceiverGroupEffectModel,
    per_target: usize,
) -> Result<(), String> {
    let shape = m.beta.shape();
    let (c, s, t) = (shape[0], shape[1], shape[2]);
    let mut target="receiver_group,target_gene,target_rank,training_improvement,best_source_gene,best_source_score\n".to_string();
    let mut pairs="receiver_group,target_gene,source_gene,target_rank,source_rank,screening_signed_effect,screening_score,signed_beta,derived_attention,selected,training_improvement\n".to_string();
    for g in 0..c {
        let mut tr: Vec<usize> = (0..t).collect();
        tr.sort_by(|&a, &b| {
            m.training_improvement[[g, b]].total_cmp(&m.training_improvement[[g, a]])
        });
        let ranks: BTreeMap<usize, usize> =
            tr.iter().enumerate().map(|(r, &x)| (x, r + 1)).collect();
        for target_idx in 0..t {
            let mut sr: Vec<usize> = (0..s).filter(|&x| x != target_idx).collect();
            sr.sort_by(|&a, &b| {
                m.screening_score[[g, b, target_idx]]
                    .total_cmp(&m.screening_score[[g, a, target_idx]])
            });
            let best = sr[0];
            target.push_str(&format!(
                "{g},{},{},{:.8},{},{:.8}\n",
                genes[target_idx],
                ranks[&target_idx],
                m.training_improvement[[g, target_idx]],
                genes[best],
                m.screening_score[[g, best, target_idx]]
            ));
            for (source_rank, &source) in sr.iter().take(per_target).enumerate() {
                pairs.push_str(&format!(
                    "{g},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{},{:.8}\n",
                    genes[target_idx],
                    genes[source],
                    ranks[&target_idx],
                    source_rank + 1,
                    m.screening_signed_effect[[g, source, target_idx]],
                    m.screening_score[[g, source, target_idx]],
                    m.beta[[g, source, target_idx]],
                    m.derived_attention(g, source, target_idx),
                    m.beta[[g, source, target_idx]] != 0.,
                    m.training_improvement[[g, target_idx]]
                ));
            }
        }
    }
    fs::write(out.join("target_scores.csv"), target).map_err(|e| e.to_string())?;
    fs::write(out.join("source_target_scores.csv"), pairs).map_err(|e| e.to_string())
}
