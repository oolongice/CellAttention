//! Minimal architecture benchmark on the fixed synthetic datasets.
//!
//! Compares group means, a frozen Transformer, physical regression alone,
//! Transformer + physical residual regression, and a row-permuted-field control.
//! Regressions use method-specific K-means receiver_groups and spatially blocked 5-fold CV.
//! True groups are reserved strictly for final recovery evaluation.

use cell_attention::{
    data::{read_expression, read_ids},
    physical::{
        ReactionDiffusion, ReceiverFieldMaskConfig, ReceiverFieldMaskMode, SpatialDimension,
        SpatialKernelConfig, apply_receiver_field_mask_inplace, compute_source_fields,
    },
    source_target::{
        ReceiverGroupClusteringMethod, ReceiverGroupCountConfig, ReceiverGroupCountMode,
        SparseEffectConfig, fit_embedding_receiver_groups, fit_fixed_receiver_group_effects,
    },
};
use ndarray::{Array2, Array3};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone)]
struct Truth {
    target: usize,
    group: usize,
    source: usize,
    source_name: String,
    target_name: String,
}

fn main() -> Result<(), String> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "examples/synthetic_data/multiseed_data".into()),
    );
    let out = PathBuf::from(std::env::args().nth(2).unwrap_or_else(|| {
        "examples/synthetic_data/analysis/minimal_architecture_benchmark".into()
    }));
    let mask_mode = ReceiverFieldMaskMode::parse(
        &std::env::args()
            .nth(3)
            .unwrap_or_else(|| "none".to_string()),
    )?;
    let expression_threshold = std::env::args()
        .nth(4)
        .map(|value| value.parse::<f32>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(0.0);
    let mask_config = ReceiverFieldMaskConfig {
        mode: mask_mode,
        expression_threshold,
    };
    let seed_start = std::env::args()
        .nth(5)
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(2001);
    let seed_end = std::env::args()
        .nth(6)
        .map(|value| value.parse::<usize>().map_err(|error| error.to_string()))
        .transpose()?
        .unwrap_or(2030);
    if seed_start > seed_end {
        return Err("seed_start must be <= seed_end".to_string());
    }
    fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    fs::write(
        out.join("receiver_field_mask.json"),
        serde_json::to_string_pretty(&mask_config).unwrap(),
    )
    .map_err(|error| error.to_string())?;
    let mut metrics = vec!["dataset,seed,method,receiver_group_ari,all_gene_mse,truth_receiver_mse,target_recall,conditional_source_recall,target_hits_at_1,target_mrr,source_hits_at_1,source_hits_at_5,source_mrr,multisource_recall_at_truth_count\n".to_string()];
    let mut ranks = vec!["dataset,seed,method,source_gene,target_gene,receiver_group,target_rank,source_rank,screening_score\n".to_string()];
    for dataset in [
        "independent_targets",
        "shared_target",
        "two_sources_per_target",
    ] {
        let fixed = root.join("data").join("preprocessed").join(dataset);
        let genes = read_ids(fixed.join("gene_ids.txt"))?;
        let cells = read_ids(fixed.join("cell_ids.txt"))?;
        let raw = read_expression(fixed.join("expression.csv"), cells.len(), genes.len())?;
        let coords_f32 = read_expression(fixed.join("spatial_coordinates.csv"), cells.len(), 2)?;
        let coords = coords_f32.mapv(f64::from);
        let true_groups = read_groups(&fixed.join("cell_groups.txt"))?;
        let truth = read_truth(&fixed.join("truth.csv"), &genes)?;
        eprintln!("dataset={dataset} stage=compute_v2_fields");
        let mut field_f32 = compute_source_fields(
            &coords,
            &raw,
            None,
            ReactionDiffusion {
                diffusion: 16_900.0,
                degradation: 1.0,
                production: 1.0,
                min_distance: 5.0,
                dimension: SpatialDimension::TwoD,
            },
            &SpatialKernelConfig {
                max_distance: 320.0,
                max_neighbors: 9999,
                include_self: false,
                target_block_size: 256,
            },
        )?;
        let mask_summary = apply_receiver_field_mask_inplace(&mut field_f32, &raw, mask_config)?;
        eprintln!(
            "dataset={dataset} stage=receiver_field_mask mode={:?} threshold={} masked_fraction={:.6}",
            mask_mode, expression_threshold, mask_summary.masked_fraction
        );
        let fields = standardize(&field_f32.mapv(f64::from));
        let folds = spatial_folds(&coords, 5, 160.0);
        for seed in seed_start..=seed_end {
            eprintln!("dataset={dataset} seed={seed}");
            let run = root
                .join("model")
                .join("multiseed")
                .join(dataset)
                .join(format!("seed_{seed}"));
            let observed = read_expression(
                run.join("standardized_expression.csv"),
                cells.len(),
                genes.len(),
            )?
            .mapv(f64::from);
            let transformer =
                read_expression(run.join("reconstruction.csv"), cells.len(), genes.len())?
                    .mapv(f64::from);
            let residual = &observed - &transformer;
            let embeddings =
                read_expression(run.join("cell_embeddings.csv"), cells.len(), 48)?.mapv(f64::from);
            let receiver_group_config = ReceiverGroupCountConfig {
                mode: ReceiverGroupCountMode::Manual(12),
                kmeans_restarts: 12,
                seed: 20_260_810u64.wrapping_add(seed as u64),
                ..Default::default()
            };
            let expression_groups = receiver_group_labels(&observed, 12, &receiver_group_config)?;
            let transformer_groups =
                receiver_group_labels(&embeddings, 12, &receiver_group_config)?;
            let mut whitened_receiver_group_config = receiver_group_config.clone();
            whitened_receiver_group_config.clustering_method =
                ReceiverGroupClusteringMethod::TruncatedWhitenedPcaKmeans;
            whitened_receiver_group_config.pca_components = Some(12);
            let transformer_whitened_groups =
                receiver_group_labels(&embeddings, 12, &whitened_receiver_group_config)?;
            let expression_ari = adjusted_rand(&expression_groups, &true_groups);
            let transformer_ari = adjusted_rand(&transformer_groups, &true_groups);
            let transformer_whitened_ari =
                adjusted_rand(&transformer_whitened_groups, &true_groups);
            let permuted = permute_rows(&fields, seed as u64 + 77_031);
            evaluate_simple(
                dataset,
                seed,
                "mean_baseline",
                &observed,
                &transformer,
                &fields,
                &expression_groups,
                &true_groups,
                expression_ari,
                &folds,
                &truth,
                None,
                &mut metrics,
                &mut ranks,
            )?;
            evaluate_simple(
                dataset,
                seed,
                "transformer_only",
                &observed,
                &transformer,
                &fields,
                &transformer_groups,
                &true_groups,
                transformer_ari,
                &folds,
                &truth,
                Some(EvalKind::Transformer),
                &mut metrics,
                &mut ranks,
            )?;
            evaluate_simple(
                dataset,
                seed,
                "transformer_whitened_only",
                &observed,
                &transformer,
                &fields,
                &transformer_whitened_groups,
                &true_groups,
                transformer_whitened_ari,
                &folds,
                &truth,
                Some(EvalKind::Transformer),
                &mut metrics,
                &mut ranks,
            )?;
            evaluate_physical(
                dataset,
                seed,
                "physical_only",
                &observed,
                &Array2::zeros(observed.raw_dim()),
                &fields,
                &expression_groups,
                &true_groups,
                expression_ari,
                &folds,
                &truth,
                &mut metrics,
                &mut ranks,
            )?;
            evaluate_physical(
                dataset,
                seed,
                "transformer_physical",
                &residual,
                &transformer,
                &fields,
                &transformer_groups,
                &true_groups,
                transformer_ari,
                &folds,
                &truth,
                &mut metrics,
                &mut ranks,
            )?;
            evaluate_physical(
                dataset,
                seed,
                "transformer_whitened_physical",
                &residual,
                &transformer,
                &fields,
                &transformer_whitened_groups,
                &true_groups,
                transformer_whitened_ari,
                &folds,
                &truth,
                &mut metrics,
                &mut ranks,
            )?;
            evaluate_physical(
                dataset,
                seed,
                "transformer_permuted_physical",
                &residual,
                &transformer,
                &permuted,
                &transformer_groups,
                &true_groups,
                transformer_ari,
                &folds,
                &truth,
                &mut metrics,
                &mut ranks,
            )?;
            evaluate_physical(
                dataset,
                seed,
                "transformer_whitened_permuted_physical",
                &residual,
                &transformer,
                &permuted,
                &transformer_whitened_groups,
                &true_groups,
                transformer_whitened_ari,
                &folds,
                &truth,
                &mut metrics,
                &mut ranks,
            )?;
        }
    }
    fs::write(out.join("run_metrics.csv"), metrics.concat()).map_err(|e| e.to_string())?;
    fs::write(out.join("truth_relation_ranks.csv"), ranks.concat()).map_err(|e| e.to_string())?;
    Ok(())
}

enum EvalKind {
    Transformer,
}

#[allow(clippy::too_many_arguments)]
fn evaluate_simple(
    dataset: &str,
    seed: usize,
    method: &str,
    y: &Array2<f64>,
    transformer: &Array2<f64>,
    _fields: &Array2<f64>,
    groups: &[usize],
    true_groups: &[usize],
    receiver_group_ari: f64,
    folds: &[usize],
    truth: &[Truth],
    kind: Option<EvalKind>,
    metrics: &mut Vec<String>,
    _ranks: &mut Vec<String>,
) -> Result<(), String> {
    let mut pred = Array2::zeros(y.raw_dim());
    for fold in 0..5 {
        if kind.is_some() {
            for i in 0..y.nrows() {
                if folds[i] == fold {
                    for j in 0..y.ncols() {
                        pred[[i, j]] = transformer[[i, j]];
                    }
                }
            }
        } else {
            let c = groups.iter().max().copied().unwrap() + 1;
            let mut sums = Array2::<f64>::zeros((c, y.ncols()));
            let mut counts = vec![0usize; c];
            for i in 0..y.nrows() {
                if folds[i] != fold {
                    counts[groups[i]] += 1;
                    for j in 0..y.ncols() {
                        sums[[groups[i], j]] += y[[i, j]];
                    }
                }
            }
            for i in 0..y.nrows() {
                if folds[i] == fold {
                    for j in 0..y.ncols() {
                        pred[[i, j]] = sums[[groups[i], j]] / (counts[groups[i]].max(1) as f64);
                    }
                }
            }
        }
    }
    push_metrics(
        dataset,
        seed,
        method,
        y,
        &pred,
        receiver_group_ari,
        true_groups,
        truth,
        None,
        metrics,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_physical(
    dataset: &str,
    seed: usize,
    method: &str,
    outcome: &Array2<f64>,
    base: &Array2<f64>,
    fields: &Array2<f64>,
    groups: &[usize],
    true_groups: &[usize],
    receiver_group_ari: f64,
    folds: &[usize],
    truth: &[Truth],
    metrics: &mut Vec<String>,
    ranks: &mut Vec<String>,
) -> Result<(), String> {
    let receiver_groups = groups.iter().max().copied().unwrap() + 1;
    let mut pred = Array2::zeros(outcome.raw_dim());
    let mut score_sum = Array3::<f64>::zeros((receiver_groups, fields.ncols(), outcome.ncols()));
    let mut target_sum = Array2::<f64>::zeros((receiver_groups, outcome.ncols()));
    let mut fits = 0.0;
    for fold in 0..5 {
        let train: Vec<usize> = (0..outcome.nrows()).filter(|&i| folds[i] != fold).collect();
        let valid: Vec<usize> = (0..outcome.nrows()).filter(|&i| folds[i] == fold).collect();
        let xt = take_rows(fields, &train);
        let yt = take_rows(outcome, &train);
        let pt = one_hot_subset(groups, &train, receiver_groups);
        let model =
            fit_fixed_receiver_group_effects(&xt, &yt, &pt, &SparseEffectConfig::default())?;
        let xv = take_rows(fields, &valid);
        let pv = one_hot_subset(groups, &valid, receiver_groups);
        let add = model.predict(&xv, &pv)?;
        for (k, &i) in valid.iter().enumerate() {
            for j in 0..outcome.ncols() {
                pred[[i, j]] = base[[i, j]] + add[[k, j]];
            }
        }
        score_sum += &model.screening_score;
        target_sum += &model.training_improvement;
        fits += 1.0;
    }
    score_sum.mapv_inplace(|x| x / fits);
    target_sum.mapv_inplace(|x| x / fits);
    let true_receiver_groups = true_groups.iter().max().copied().unwrap_or(0) + 1;
    let (evaluation_scores, evaluation_targets) = remap_receiver_group_scores(
        &score_sum,
        &target_sum,
        groups,
        true_groups,
        true_receiver_groups,
    );
    let observed = outcome + base;
    push_metrics(
        dataset,
        seed,
        method,
        &observed,
        &pred,
        receiver_group_ari,
        true_groups,
        truth,
        Some((&evaluation_scores, &evaluation_targets)),
        metrics,
    );
    for relation in truth {
        let tr = target_rank(&evaluation_scores, relation.group, relation.target);
        let sr = source_rank(
            &evaluation_scores,
            relation.group,
            relation.source,
            relation.target,
        );
        ranks.push(format!(
            "{dataset},{seed},{method},{},{},{},{tr},{sr},{:.9}\n",
            relation.source_name,
            relation.target_name,
            relation.group,
            evaluation_scores[[relation.group, relation.source, relation.target]]
        ));
    }
    Ok(())
}

fn push_metrics(
    dataset: &str,
    seed: usize,
    method: &str,
    y: &Array2<f64>,
    pred: &Array2<f64>,
    receiver_group_ari: f64,
    groups: &[usize],
    truth: &[Truth],
    scores: Option<(&Array3<f64>, &Array2<f64>)>,
    out: &mut Vec<String>,
) {
    let all = mse_iter(
        (0..y.nrows()).flat_map(|i| (0..y.ncols()).map(move |j| (y[[i, j]], pred[[i, j]]))),
    );
    let keys: BTreeSet<(usize, usize)> = truth.iter().map(|r| (r.group, r.target)).collect();
    let tm = mse_iter(keys.iter().flat_map(|&(g, t)| {
        (0..y.nrows())
            .filter(move |&i| groups[i] == g)
            .map(move |i| (y[[i, t]], pred[[i, t]]))
    }));
    let (target_recall, conditional_source_recall, th1, tmrr, sh1, sh5, smrr, rec) =
        if let Some(s) = scores {
            let (source_scores, target_scores) = s;
            let (target_recall, conditional_source_recall) =
                call_recall_metrics(source_scores, target_scores, truth);
            let (th1, tmrr, sh1, sh5, smrr, rec) =
                ranking_metrics_with_targets(source_scores, target_scores, truth);
            (
                target_recall,
                conditional_source_recall,
                th1,
                tmrr,
                sh1,
                sh5,
                smrr,
                rec,
            )
        } else {
            (
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            )
        };
    out.push(format!("{dataset},{seed},{method},{receiver_group_ari:.9},{all:.9},{tm:.9},{target_recall},{conditional_source_recall},{th1},{tmrr},{sh1},{sh5},{smrr},{rec}\n"));
}

// Match the primary recovery metrics used by the original synthetic benchmark:
// call exactly as many receiver-targets as exist in the truth, then call exactly
// as many sources per receiver-target as are planted for that target.
fn call_recall_metrics(
    s: &Array3<f64>,
    target_scores: &Array2<f64>,
    truth: &[Truth],
) -> (f64, f64) {
    let true_targets: BTreeSet<(usize, usize)> =
        truth.iter().map(|r| (r.group, r.target)).collect();
    let mut candidates = Vec::new();
    for g in 0..target_scores.nrows() {
        for t in 0..target_scores.ncols() {
            candidates.push(((g, t), target_scores[[g, t]]));
        }
    }
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    let predicted_targets: BTreeSet<(usize, usize)> = candidates
        .into_iter()
        .take(true_targets.len())
        .map(|x| x.0)
        .collect();
    let target_hits = predicted_targets.intersection(&true_targets).count();
    let conditional_source_hits = truth
        .iter()
        .filter(|relation| {
            if !predicted_targets.contains(&(relation.group, relation.target)) {
                return false;
            }
            let source_limit = truth
                .iter()
                .filter(|x| x.group == relation.group && x.target == relation.target)
                .count();
            source_rank(s, relation.group, relation.source, relation.target) <= source_limit
        })
        .count();
    (
        target_hits as f64 / true_targets.len() as f64,
        conditional_source_hits as f64 / truth.len() as f64,
    )
}

fn ranking_metrics_with_targets(
    s: &Array3<f64>,
    target_scores: &Array2<f64>,
    truth: &[Truth],
) -> (f64, f64, f64, f64, f64, f64) {
    let keys: BTreeSet<(usize, usize)> = truth.iter().map(|r| (r.group, r.target)).collect();
    let mut th = 0.;
    let mut trr = 0.;
    for &(g, t) in &keys {
        let r = 1
            + (0..target_scores.ncols())
                .filter(|&u| target_scores[[g, u]] > target_scores[[g, t]])
                .count();
        th += (r == 1) as usize as f64;
        trr += 1. / r as f64;
    }
    let mut sh = 0.;
    let mut s5 = 0.;
    let mut srr = 0.;
    for x in truth {
        let r = source_rank(s, x.group, x.source, x.target);
        sh += (r == 1) as usize as f64;
        s5 += (r <= 5) as usize as f64;
        srr += 1. / r as f64;
    }
    let mut recall = 0.;
    for &(g, t) in &keys {
        let true_s: BTreeSet<usize> = truth
            .iter()
            .filter(|x| x.group == g && x.target == t)
            .map(|x| x.source)
            .collect();
        let mut candidates: Vec<usize> = (0..s.shape()[1]).filter(|&x| x != t).collect();
        candidates.sort_by(|&a, &b| s[[g, b, t]].total_cmp(&s[[g, a, t]]));
        recall += candidates
            .iter()
            .take(true_s.len())
            .filter(|x| true_s.contains(x))
            .count() as f64
            / true_s.len() as f64;
    }
    (
        th / keys.len() as f64,
        trr / keys.len() as f64,
        sh / truth.len() as f64,
        s5 / truth.len() as f64,
        srr / truth.len() as f64,
        recall / keys.len() as f64,
    )
}

fn adjusted_rand(left: &[usize], right: &[usize]) -> f64 {
    let mut contingency = HashMap::<(usize, usize), usize>::new();
    let mut left_counts = HashMap::<usize, usize>::new();
    let mut right_counts = HashMap::<usize, usize>::new();
    for (&a, &b) in left.iter().zip(right) {
        *contingency.entry((a, b)).or_default() += 1;
        *left_counts.entry(a).or_default() += 1;
        *right_counts.entry(b).or_default() += 1;
    }
    let choose2 = |value: usize| value.saturating_mul(value.saturating_sub(1)) as f64 / 2.0;
    let cells = left.len();
    let total_pairs = choose2(cells);
    if total_pairs == 0.0 {
        return 1.0;
    }
    let observed: f64 = contingency.values().map(|&count| choose2(count)).sum();
    let left_pairs: f64 = left_counts.values().map(|&count| choose2(count)).sum();
    let right_pairs: f64 = right_counts.values().map(|&count| choose2(count)).sum();
    let expected = left_pairs * right_pairs / total_pairs;
    let maximum = 0.5 * (left_pairs + right_pairs);
    if (maximum - expected).abs() < 1e-12 {
        1.0
    } else {
        (observed - expected) / (maximum - expected)
    }
}

fn receiver_group_labels(
    features: &Array2<f64>,
    receiver_group_count: usize,
    config: &ReceiverGroupCountConfig,
) -> Result<Vec<usize>, String> {
    let model = fit_embedding_receiver_groups(features, receiver_group_count, config)?;
    let probabilities = model.probabilities(features)?;
    Ok((0..probabilities.nrows())
        .map(|cell| {
            (0..probabilities.ncols())
                .max_by(|&left, &right| {
                    probabilities[[cell, left]].total_cmp(&probabilities[[cell, right]])
                })
                .unwrap()
        })
        .collect())
}

fn remap_receiver_group_scores(
    source_scores: &Array3<f64>,
    target_scores: &Array2<f64>,
    inferred_groups: &[usize],
    true_groups: &[usize],
    true_receiver_groups: usize,
) -> (Array3<f64>, Array2<f64>) {
    let mut counts = HashMap::<(usize, usize), usize>::new();
    for (&inferred, &truth) in inferred_groups.iter().zip(true_groups) {
        *counts.entry((inferred, truth)).or_default() += 1;
    }
    let mut mapping = vec![0usize; source_scores.shape()[0]];
    for (receiver_group, mapped) in mapping.iter_mut().enumerate() {
        *mapped = (0..true_receiver_groups)
            .max_by_key(|&truth| counts.get(&(receiver_group, truth)).copied().unwrap_or(0))
            .unwrap_or(0);
    }
    let mut remapped_sources = Array3::<f64>::zeros((
        true_receiver_groups,
        source_scores.shape()[1],
        source_scores.shape()[2],
    ));
    let mut remapped_targets = Array2::<f64>::zeros((true_receiver_groups, target_scores.ncols()));
    for receiver_group in 0..source_scores.shape()[0] {
        let truth = mapping[receiver_group];
        for source in 0..source_scores.shape()[1] {
            for target in 0..source_scores.shape()[2] {
                remapped_sources[[truth, source, target]] = remapped_sources
                    [[truth, source, target]]
                .max(source_scores[[receiver_group, source, target]]);
            }
        }
        for target in 0..target_scores.ncols() {
            remapped_targets[[truth, target]] =
                remapped_targets[[truth, target]].max(target_scores[[receiver_group, target]]);
        }
    }
    (remapped_sources, remapped_targets)
}

fn target_rank(s: &Array3<f64>, g: usize, t: usize) -> usize {
    let val = (0..s.shape()[1])
        .filter(|&x| x != t)
        .map(|x| s[[g, x, t]])
        .fold(f64::NEG_INFINITY, f64::max);
    1 + (0..s.shape()[2])
        .filter(|&u| u != t)
        .filter(|&u| {
            (0..s.shape()[1])
                .filter(|&x| x != u)
                .map(|x| s[[g, x, u]])
                .fold(f64::NEG_INFINITY, f64::max)
                > val
        })
        .count()
}
fn source_rank(s: &Array3<f64>, g: usize, x: usize, t: usize) -> usize {
    1 + (0..s.shape()[1])
        .filter(|&q| q != t && s[[g, q, t]] > s[[g, x, t]])
        .count()
}
fn mse_iter<I: Iterator<Item = (f64, f64)>>(it: I) -> f64 {
    let mut s = 0.;
    let mut n = 0;
    for (a, b) in it {
        s += (a - b).powi(2);
        n += 1
    }
    if n == 0 { f64::NAN } else { s / n as f64 }
}
fn take_rows(a: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    Array2::from_shape_fn((idx.len(), a.ncols()), |(i, j)| a[[idx[i], j]])
}
fn one_hot_subset(groups: &[usize], idx: &[usize], c: usize) -> Array2<f64> {
    let mut p = Array2::zeros((idx.len(), c));
    for (i, &row) in idx.iter().enumerate() {
        p[[i, groups[row]]] = 1.;
    }
    p
}
fn permute_rows(a: &Array2<f64>, seed: u64) -> Array2<f64> {
    let mut idx: Vec<usize> = (0..a.nrows()).collect();
    idx.shuffle(&mut StdRng::seed_from_u64(seed));
    take_rows(a, &idx)
}
fn standardize(a: &Array2<f64>) -> Array2<f64> {
    let mut z = Array2::zeros(a.raw_dim());
    for j in 0..a.ncols() {
        let m = a.column(j).sum() / a.nrows() as f64;
        let sd = (a.column(j).iter().map(|x| (x - m).powi(2)).sum::<f64>() / a.nrows() as f64)
            .sqrt()
            .max(1e-8);
        for i in 0..a.nrows() {
            z[[i, j]] = (a[[i, j]] - m) / sd;
        }
    }
    z
}
fn spatial_folds(x: &Array2<f64>, k: usize, w: f64) -> Vec<usize> {
    (0..x.nrows())
        .map(|i| {
            let a = (x[[i, 0]] / w).floor() as i64;
            let b = (x[[i, 1]] / w).floor() as i64;
            ((a.wrapping_mul(73856093) ^ b.wrapping_mul(19349663)).unsigned_abs() as usize) % k
        })
        .collect()
}
fn read_groups(p: &Path) -> Result<Vec<usize>, String> {
    fs::read_to_string(p)
        .map_err(|e| e.to_string())?
        .lines()
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            x.trim()
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())
        })
        .collect()
}
fn read_truth(p: &Path, genes: &[String]) -> Result<Vec<Truth>, String> {
    let map: BTreeMap<&str, usize> = genes
        .iter()
        .enumerate()
        .map(|(i, g)| (g.as_str(), i))
        .collect();
    fs::read_to_string(p)
        .map_err(|e| e.to_string())?
        .lines()
        .skip(1)
        .filter(|x| !x.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            Ok(Truth {
                target: *map.get(f[0]).ok_or(f[0])?,
                group: f[1].parse::<usize>().map_err(|e| e.to_string())?,
                source: *map.get(f[2]).ok_or(f[2])?,
                source_name: f[2].into(),
                target_name: f[0].into(),
            })
        })
        .collect()
}
