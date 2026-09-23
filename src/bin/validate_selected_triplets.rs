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
use ndarray::{Array2, Axis};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Deserialize)]
struct Triplet {
    cluster: usize,
    source_gene: String,
    target_gene: String,
}

#[derive(Deserialize)]
struct Config {
    checkpoint_dir: PathBuf,
    coordinates: PathBuf,
    output_dir: PathBuf,
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
    maximum_neighbors: usize,
    target_block_size: usize,
    top_k_sources: usize,
    #[serde(default)]
    receiver_field_mask: ReceiverFieldMaskConfig,
    spatial_folds: usize,
    spatial_block_width: f64,
    stability_repeats: usize,
    stability_fraction: f64,
    seed: u64,
    triplets: Vec<Triplet>,
}

fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: validate_selected_triplets CONFIG.json")?;
    let config: Config =
        serde_json::from_str(&fs::read_to_string(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    fs::create_dir_all(&config.output_dir).map_err(|e| e.to_string())?;
    let cells = read_ids(config.checkpoint_dir.join("cell_ids.txt"))?;
    let genes = read_ids(config.checkpoint_dir.join("gene_ids.txt"))?;
    let gene_index: BTreeMap<&str, usize> = genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
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
    let coords = read_expression(&config.coordinates, cells.len(), 2)?.mapv(f64::from);
    eprintln!(
        "stage=receiver_groups cells={} genes={}",
        cells.len(),
        genes.len()
    );
    let cc = ReceiverGroupCountConfig {
        mode: ReceiverGroupCountMode::Manual(config.receiver_group_count),
        clustering_method: config.receiver_group_clustering_method,
        pca_components: config.receiver_group_pca_components,
        kmeans_restarts: 8,
        kmeans_iterations: 100,
        seed: 20_260_810,
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
    let selected = balanced_indices(
        &labels,
        config.receiver_group_count,
        config.maximum_fit_cells,
    );
    eprintln!("stage=fields fit_cells={}", selected.len());
    let mut fields = compute_source_fields(
        &coords,
        &raw,
        None,
        ReactionDiffusion {
            diffusion: config.length_scale.powi(2),
            degradation: 1.,
            production: 1.,
            min_distance: config.minimum_distance,
            dimension: SpatialDimension::TwoD,
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
        "stage=receiver_field_mask mode={:?} threshold={} masked_fraction={:.6}",
        config.receiver_field_mask.mode,
        config.receiver_field_mask.expression_threshold,
        mask_summary.masked_fraction
    );
    let x = standardize_selected(&fields, &selected);
    let y = standardize_selected(&residual, &selected);
    let p = take_rows(&probabilities, &selected);
    let c = take_rows(&coords, &selected);
    drop(fields);
    drop(raw);
    drop(residual);
    drop(embedding);
    drop(probabilities);
    drop(coords);
    let ec = SparseEffectConfig {
        top_k_sources: config.top_k_sources,
        ..Default::default()
    };
    let resolved: Vec<(usize, usize, usize, &Triplet)> = config
        .triplets
        .iter()
        .map(|t| {
            let receiver_group = t.cluster.checked_sub(1).ok_or("clusters are one-based")?;
            let source = *gene_index
                .get(t.source_gene.as_str())
                .ok_or_else(|| format!("unknown source {}", t.source_gene))?;
            let target = *gene_index
                .get(t.target_gene.as_str())
                .ok_or_else(|| format!("unknown target {}", t.target_gene))?;
            Ok((receiver_group, source, target, t))
        })
        .collect::<Result<_, String>>()?;

    eprintln!("stage=heldout folds={}", config.spatial_folds);
    let folds = spatial_fold_labels(&c, config.spatial_folds, config.spatial_block_width);
    let mut heldout="cluster,source_gene,target_gene,fold,validation_cells,effective_receiver_mass,selected,signed_beta,source_rank,target_training_improvement,baseline_mse,physical_mse,heldout_improvement,without_source_mse,source_incremental_improvement\n".to_string();
    for fold in 0..config.spatial_folds {
        let train: Vec<usize> = (0..x.nrows()).filter(|&i| folds[i] != fold).collect();
        let valid: Vec<usize> = (0..x.nrows()).filter(|&i| folds[i] == fold).collect();
        let model = fit_fixed_receiver_group_effects(
            &take_rows(&x, &train),
            &take_rows(&y, &train),
            &take_rows(&p, &train),
            &ec,
        )?;
        for &(receiver_group, source, target, t) in &resolved {
            let weights = p.column(receiver_group);
            let mass: f64 = valid.iter().map(|&i| weights[i]).sum();
            let mut base = 0.;
            let mut full = 0.;
            let mut without = 0.;
            for &i in &valid {
                let w = weights[i];
                let truth = y[[i, target]];
                let baseline = model.intercept[[receiver_group, target]];
                let prediction = baseline
                    + (0..x.ncols())
                        .map(|s| x[[i, s]] * model.beta[[receiver_group, s, target]])
                        .sum::<f64>();
                let removed =
                    prediction - x[[i, source]] * model.beta[[receiver_group, source, target]];
                base += w * (truth - baseline).powi(2);
                full += w * (truth - prediction).powi(2);
                without += w * (truth - removed).powi(2);
            }
            base /= mass.max(1e-12);
            full /= mass.max(1e-12);
            without /= mass.max(1e-12);
            let beta = model.beta[[receiver_group, source, target]];
            let rank = source_rank(&model.beta, receiver_group, source, target);
            heldout.push_str(&format!(
                "{},{},{},{},{},{:.6},{},{:.9},{},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9}\n",
                t.cluster,
                t.source_gene,
                t.target_gene,
                fold + 1,
                valid.len(),
                mass,
                beta != 0.,
                beta,
                rank,
                model.training_improvement[[receiver_group, target]],
                base,
                full,
                (base - full) / base.max(1e-12),
                without,
                (without - full) / without.max(1e-12)
            ));
        }
    }
    fs::write(config.output_dir.join("heldout_fold_results.csv"), heldout)
        .map_err(|e| e.to_string())?;

    eprintln!("stage=stability repeats={}", config.stability_repeats);
    let hard: Vec<usize> = (0..p.nrows())
        .map(|i| {
            (0..config.receiver_group_count)
                .max_by(|&a, &b| p[[i, a]].total_cmp(&p[[i, b]]))
                .unwrap()
        })
        .collect();
    let mut stability="cluster,source_gene,target_gene,repeat,fit_cells,selected,signed_beta,direction_matches_full,source_rank,target_training_improvement,target_retained,derived_attention\n".to_string();
    let full = fit_fixed_receiver_group_effects(&x, &y, &p, &ec)?;
    for repeat in 0..config.stability_repeats {
        let sample = stratified_sample(
            &hard,
            config.receiver_group_count,
            config.stability_fraction,
            config.seed + repeat as u64,
        );
        let model = fit_fixed_receiver_group_effects(
            &take_rows(&x, &sample),
            &take_rows(&y, &sample),
            &take_rows(&p, &sample),
            &ec,
        )?;
        for &(receiver_group, source, target, t) in &resolved {
            let beta = model.beta[[receiver_group, source, target]];
            let full_beta = full.beta[[receiver_group, source, target]];
            stability.push_str(&format!(
                "{},{},{},{},{},{},{:.9},{},{},{:.9},{},{:.9}\n",
                t.cluster,
                t.source_gene,
                t.target_gene,
                repeat + 1,
                sample.len(),
                beta != 0.,
                beta,
                beta.signum() == full_beta.signum(),
                source_rank(&model.beta, receiver_group, source, target),
                model.training_improvement[[receiver_group, target]],
                model.training_improvement[[receiver_group, target]]
                    >= ec.minimum_training_improvement,
                model.derived_attention(receiver_group, source, target)
            ));
        }
    }
    fs::write(
        config.output_dir.join("stability_repeat_results.csv"),
        stability,
    )
    .map_err(|e| e.to_string())?;
    fs::write(config.output_dir.join("metadata.json"),serde_json::to_string_pretty(&serde_json::json!({"fit_cells":selected.len(),"spatial_folds":config.spatial_folds,"spatial_block_width":config.spatial_block_width,"stability_repeats":config.stability_repeats,"stability_fraction":config.stability_fraction,"triplets":config.triplets.len(),"fixed_transformer":true,"physical_fields_computed_once":true,"receiver_group_probabilities_fixed_after_full_embedding_fit":true,"receiver_group_clustering_method":config.receiver_group_clustering_method,"receiver_group_pca_components":config.receiver_group_pca_components,"source_target":{"top_k_sources":ec.top_k_sources,"l1_penalty":ec.l1_penalty,"l2_penalty":ec.l2_penalty,"minimum_training_improvement":ec.minimum_training_improvement},"receiver_field_mask":{"mode":config.receiver_field_mask.mode,"expression_threshold":config.receiver_field_mask.expression_threshold,"masked_entries":mask_summary.masked_entries,"total_entries":mask_summary.total_entries,"masked_fraction":mask_summary.masked_fraction,"applied_before_standardization":true}})).unwrap()).map_err(|e|e.to_string())?;
    eprintln!("stage=complete");
    Ok(())
}

fn take_rows<T: Copy + Default>(a: &Array2<T>, idx: &[usize]) -> Array2<T> {
    Array2::from_shape_fn((idx.len(), a.ncols()), |(i, j)| a[[idx[i], j]])
}
fn standardize_selected(a: &Array2<f32>, idx: &[usize]) -> Array2<f64> {
    let mut z = Array2::zeros((idx.len(), a.ncols()));
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
fn spatial_fold_labels(coords: &Array2<f64>, k: usize, width: f64) -> Vec<usize> {
    (0..coords.nrows())
        .map(|i| {
            let a = (coords[[i, 0]] / width).floor() as i64;
            let b = (coords[[i, 1]] / width).floor() as i64;
            ((a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663)).unsigned_abs() as usize) % k
        })
        .collect()
}
fn stratified_sample(
    labels: &[usize],
    receiver_groups: usize,
    fraction: f64,
    seed: u64,
) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::new();
    for c in 0..receiver_groups {
        let mut v: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == c).collect();
        v.shuffle(&mut rng);
        let n = ((v.len() as f64 * fraction).round() as usize)
            .max(1)
            .min(v.len());
        out.extend_from_slice(&v[..n]);
    }
    out.sort_unstable();
    out
}
fn source_rank(
    beta: &ndarray::Array3<f64>,
    receiver_group: usize,
    source: usize,
    target: usize,
) -> usize {
    let value = beta[[receiver_group, source, target]].abs();
    1 + (0..beta.len_of(Axis(1)))
        .filter(|&s| beta[[receiver_group, s, target]].abs() > value)
        .count()
}
