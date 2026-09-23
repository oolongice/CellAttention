use cell_attention::{
    data::{read_expression, read_ids},
    physical::{
        ReactionDiffusion, ReceiverFieldMaskConfig, ReceiverFieldMaskMode, SpatialDimension,
        SpatialKernelConfig, apply_receiver_field_mask_inplace, compute_source_fields,
    },
    source_target::{
        ReceiverGroupCountConfig, ReceiverGroupCountMode, SparseEffectConfig,
        fit_embedding_receiver_groups, fit_fixed_receiver_group_effects,
        select_receiver_group_count,
    },
};
use ndarray::{Array2, Axis};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Clone)]
struct TruthRelation {
    target: String,
    group: usize,
    source: String,
}

fn main() -> Result<(), String> {
    let root_arg = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Paper_results/data1_Synthetic".to_string());
    let output_arg = std::env::args()
        .nth(2)
        .unwrap_or_else(|| format!("{root_arg}/analysis/source_target"));
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
    let root = Path::new(&root_arg);
    let output = Path::new(&output_arg);
    fs::create_dir_all(output).map_err(|error| error.to_string())?;
    fs::write(
        output.join("receiver_field_mask.json"),
        serde_json::to_string_pretty(&mask_config).unwrap(),
    )
    .map_err(|error| error.to_string())?;
    let datasets = [
        "independent_targets",
        "shared_target",
        "two_sources_per_target",
    ];
    let mut summaries = vec!["dataset,seed,scenario,receiver_group_count,ari,nmi,mean_receiver_group_purity,truth_triplets,target_top1,target_top3,screening_source_top1,screening_source_top3,screening_source_top5,joint_source_top1,joint_source_top3,joint_source_top5\n".to_string()];
    for dataset in datasets {
        eprintln!("dataset={dataset} stage=load_fixed_data");
        let data_root = root.join("data").join("preprocessed").join(dataset);
        let genes = read_ids(data_root.join("gene_ids.txt"))?;
        let cells = read_ids(data_root.join("cell_ids.txt"))?;
        let expression_f32 =
            read_expression(data_root.join("expression.csv"), cells.len(), genes.len())?;
        let coordinates_f32 =
            read_expression(data_root.join("spatial_coordinates.csv"), cells.len(), 2)?;
        let coordinates = coordinates_f32.mapv(f64::from);
        let groups: Vec<usize> = fs::read_to_string(data_root.join("cell_groups.txt"))
            .map_err(|e| e.to_string())?
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| line.trim().parse::<usize>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        let truth = read_truth(&data_root.join("truth.csv"))?;
        eprintln!("dataset={dataset} stage=physical_fields");
        let mut field_f32 = compute_source_fields(
            &coordinates,
            &expression_f32,
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
        let mask_summary =
            apply_receiver_field_mask_inplace(&mut field_f32, &expression_f32, mask_config)?;
        eprintln!(
            "dataset={dataset} stage=receiver_field_mask mode={:?} threshold={} masked_fraction={:.6}",
            mask_mode, expression_threshold, mask_summary.masked_fraction
        );
        let fields = standardize(&field_f32.mapv(f64::from));
        let effect_config = SparseEffectConfig::default();
        for seed in 2001..=2030 {
            eprintln!("dataset={dataset} seed={seed} stage=load_transformer");
            let checkpoint = root
                .join("model")
                .join("multiseed")
                .join(dataset)
                .join(format!("seed_{seed}"));
            let seed_output = output.join(dataset).join(format!("seed_{seed}"));
            fs::create_dir_all(&seed_output).map_err(|error| error.to_string())?;
            let residual_f32 =
                read_expression(checkpoint.join("residuals.csv"), cells.len(), genes.len())?;
            let embedding_f32 =
                read_expression(checkpoint.join("cell_embeddings.csv"), cells.len(), 48)?;
            let residuals = standardize(&residual_f32.mapv(f64::from));
            let embeddings = embedding_f32.mapv(f64::from);
            let fixed_config = ReceiverGroupCountConfig {
                mode: ReceiverGroupCountMode::Manual(12),
                kmeans_restarts: 12,
                ..Default::default()
            };
            let learned_model = fit_embedding_receiver_groups(&embeddings, 12, &fixed_config)?;
            let learned_probabilities = learned_model.probabilities(&embeddings)?;
            evaluate_scenario(
                dataset,
                seed,
                "learned_c12",
                &genes,
                &groups,
                &truth,
                &fields,
                &residuals,
                &learned_probabilities,
                &effect_config,
                &seed_output,
                &mut summaries,
            )?;
            let oracle_probabilities = one_hot(&groups);
            evaluate_scenario(
                dataset,
                seed,
                "oracle_groups",
                &genes,
                &groups,
                &truth,
                &fields,
                &residuals,
                &oracle_probabilities,
                &effect_config,
                &seed_output,
                &mut summaries,
            )?;
            let auto_config = ReceiverGroupCountConfig {
                mode: ReceiverGroupCountMode::DataDriven,
                kmeans_restarts: 4,
                kmeans_iterations: 75,
                ..Default::default()
            };
            let selection = select_receiver_group_count(
                &embeddings,
                &fields,
                &residuals,
                &coordinates,
                &auto_config,
                &effect_config,
            )?;
            let mut selection_csv = "receiver_group_count,baseline_mse,physical_mse,improvement_fraction,improvement_standard_error,mean_active_effects,minimum_effective_train_mass,eligible,selected\n".to_string();
            for row in &selection.candidates {
                selection_csv.push_str(&format!(
                    "{},{:.8},{:.8},{:.8},{:.8},{:.2},{:.3},{},{}\n",
                    row.receiver_group_count,
                    row.baseline_mse,
                    row.physical_mse,
                    row.improvement_fraction,
                    row.improvement_standard_error,
                    row.mean_active_effects,
                    row.minimum_effective_train_mass,
                    row.eligible,
                    row.receiver_group_count == selection.selected_receiver_group_count
                ));
            }
            fs::write(
                seed_output.join("receiver_group_count_selection.csv"),
                selection_csv,
            )
            .map_err(|e| e.to_string())?;
            let auto_model = fit_embedding_receiver_groups(
                &embeddings,
                selection.selected_receiver_group_count,
                &auto_config,
            )?;
            let auto_probabilities = auto_model.probabilities(&embeddings)?;
            evaluate_scenario(
                dataset,
                seed,
                "learned_auto_c",
                &genes,
                &groups,
                &truth,
                &fields,
                &residuals,
                &auto_probabilities,
                &effect_config,
                &seed_output,
                &mut summaries,
            )?;
        }
    }
    fs::write(output.join("summary.csv"), summaries.concat()).map_err(|error| error.to_string())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate_scenario(
    dataset: &str,
    seed: usize,
    scenario: &str,
    genes: &[String],
    groups: &[usize],
    truth: &[TruthRelation],
    fields: &Array2<f64>,
    residuals: &Array2<f64>,
    probabilities: &Array2<f64>,
    config: &SparseEffectConfig,
    output: &Path,
    summaries: &mut Vec<String>,
) -> Result<(), String> {
    let model = fit_fixed_receiver_group_effects(fields, residuals, probabilities, config)?;
    let labels: Vec<usize> = (0..probabilities.nrows())
        .map(|cell| {
            probabilities
                .row(cell)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0
        })
        .collect();
    fs::write(
        output.join(format!("{scenario}_assignments.csv")),
        format!(
            "receiver_group\n{}\n",
            labels
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        ),
    )
    .map_err(|e| e.to_string())?;
    let ari = adjusted_rand(&labels, groups);
    let nmi = normalized_mutual_information(&labels, groups);
    let unique_groups: BTreeSet<usize> = groups.iter().copied().collect();
    let matched: Vec<usize> = unique_groups
        .iter()
        .map(|group| best_context(*group, &labels, groups, probabilities.ncols()))
        .collect();
    let mapping = unique_groups
        .iter()
        .copied()
        .zip(matched.iter().copied())
        .collect::<std::collections::BTreeMap<_, _>>();
    let mut rows=vec!["dataset,scenario,source_gene,target_gene,true_receiver_group,matched_receiver_group,receiver_group_purity,target_rank_by_screening,target_rank_by_joint_improvement,screening_source_rank,joint_source_rank,screening_signed_effect,screening_score,joint_beta,derived_attention,target_training_improvement\n".to_string()];
    let mut target_top1 = 0;
    let mut target_top3 = 0;
    let mut screen1 = 0;
    let mut screen3 = 0;
    let mut screen5 = 0;
    let mut joint1 = 0;
    let mut joint3 = 0;
    let mut joint5 = 0;
    let mut purity_sum = 0.0;
    for relation in truth {
        let receiver_group = *mapping.get(&relation.group).unwrap();
        let source = genes
            .iter()
            .position(|gene| gene == &relation.source)
            .unwrap();
        let target = genes
            .iter()
            .position(|gene| gene == &relation.target)
            .unwrap();
        let purity = receiver_group_purity(receiver_group, relation.group, &labels, groups);
        purity_sum += purity;
        let target_score = |candidate: usize| {
            (0..fields.ncols())
                .map(|s| model.screening_score[[receiver_group, s, candidate]])
                .fold(0.0, f64::max)
        };
        let true_target_score = target_score(target);
        let target_rank = 1
            + (0..residuals.ncols())
                .filter(|candidate| target_score(*candidate) > true_target_score)
                .count();
        let true_improvement = model.training_improvement[[receiver_group, target]];
        let target_joint_rank = 1
            + (0..residuals.ncols())
                .filter(|candidate| {
                    model.training_improvement[[receiver_group, *candidate]] > true_improvement
                })
                .count();
        let true_screen = model.screening_score[[receiver_group, source, target]];
        let screen_rank = 1
            + (0..fields.ncols())
                .filter(|candidate| {
                    model.screening_score[[receiver_group, *candidate, target]] > true_screen
                })
                .count();
        let true_joint = model.beta[[receiver_group, source, target]].abs();
        let joint_rank = 1
            + (0..fields.ncols())
                .filter(|candidate| {
                    model.beta[[receiver_group, *candidate, target]].abs() > true_joint
                })
                .count();
        target_top1 += (target_rank <= 1) as usize;
        target_top3 += (target_rank <= 3) as usize;
        screen1 += (screen_rank <= 1) as usize;
        screen3 += (screen_rank <= 3) as usize;
        screen5 += (screen_rank <= 5) as usize;
        joint1 += (joint_rank <= 1) as usize;
        joint3 += (joint_rank <= 3) as usize;
        joint5 += (joint_rank <= 5) as usize;
        rows.push(format!(
            "{dataset},{scenario},{},{},{},{},{:.6},{},{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8}\n",
            relation.source,
            relation.target,
            relation.group,
            receiver_group,
            purity,
            target_rank,
            target_joint_rank,
            screen_rank,
            joint_rank,
            model.screening_signed_effect[[receiver_group, source, target]],
            true_screen,
            model.beta[[receiver_group, source, target]],
            model.derived_attention(receiver_group, source, target),
            true_improvement
        ));
    }
    fs::write(
        output.join(format!("{scenario}_truth_triplets.csv")),
        rows.concat(),
    )
    .map_err(|e| e.to_string())?;
    summaries.push(format!(
        "{dataset},{seed},{scenario},{},{:.6},{:.6},{:.6},{},{},{},{},{},{},{},{},{}\n",
        probabilities.ncols(),
        ari,
        nmi,
        purity_sum / truth.len() as f64,
        truth.len(),
        target_top1,
        target_top3,
        screen1,
        screen3,
        screen5,
        joint1,
        joint3,
        joint5
    ));
    Ok(())
}

fn standardize(matrix: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::zeros(matrix.raw_dim());
    for col in 0..matrix.ncols() {
        let mean = matrix.column(col).sum() / matrix.nrows() as f64;
        let sd = (matrix
            .column(col)
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>()
            / matrix.nrows() as f64)
            .sqrt()
            .max(1e-8);
        for row in 0..matrix.nrows() {
            out[[row, col]] = (matrix[[row, col]] - mean) / sd;
        }
    }
    out
}
fn one_hot(groups: &[usize]) -> Array2<f64> {
    let count = groups.iter().max().copied().unwrap() + 1;
    let mut p = Array2::zeros((groups.len(), count));
    for (i, &g) in groups.iter().enumerate() {
        p[[i, g]] = 1.;
    }
    p
}
fn read_truth(path: &Path) -> Result<Vec<TruthRelation>, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    text.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let f: Vec<&str> = line.split(',').collect();
            Ok(TruthRelation {
                target: f[0].to_string(),
                group: f[1].parse::<usize>().map_err(|e| e.to_string())?,
                source: f[2].to_string(),
            })
        })
        .collect()
}
fn best_context(group: usize, labels: &[usize], truth: &[usize], receiver_groups: usize) -> usize {
    (0..receiver_groups)
        .max_by_key(|receiver_group| {
            (0..labels.len())
                .filter(|&i| labels[i] == *receiver_group && truth[i] == group)
                .count()
        })
        .unwrap()
}
fn receiver_group_purity(
    receiver_group: usize,
    group: usize,
    labels: &[usize],
    truth: &[usize],
) -> f64 {
    let total = (0..labels.len())
        .filter(|&i| labels[i] == receiver_group)
        .count();
    if total == 0 {
        0.
    } else {
        (0..labels.len())
            .filter(|&i| labels[i] == receiver_group && truth[i] == group)
            .count() as f64
            / total as f64
    }
}
fn contingency(a: &[usize], b: &[usize]) -> Array2<f64> {
    let ar = a.iter().max().copied().unwrap() + 1;
    let br = b.iter().max().copied().unwrap() + 1;
    let mut table = Array2::zeros((ar, br));
    for i in 0..a.len() {
        table[[a[i], b[i]]] += 1.;
    }
    table
}
fn choose2(x: f64) -> f64 {
    x * (x - 1.) / 2.
}
fn adjusted_rand(a: &[usize], b: &[usize]) -> f64 {
    let t = contingency(a, b);
    let nij = t.iter().map(|v| choose2(*v)).sum::<f64>();
    let rows = t.sum_axis(Axis(1)).iter().map(|v| choose2(*v)).sum::<f64>();
    let cols = t.sum_axis(Axis(0)).iter().map(|v| choose2(*v)).sum::<f64>();
    let total = choose2(a.len() as f64);
    let expected = rows * cols / total;
    let maximum = (rows + cols) / 2.;
    (nij - expected) / (maximum - expected).max(1e-12)
}
fn normalized_mutual_information(a: &[usize], b: &[usize]) -> f64 {
    let t = contingency(a, b);
    let n = a.len() as f64;
    let rows = t.sum_axis(Axis(1));
    let cols = t.sum_axis(Axis(0));
    let mut mi = 0.;
    for i in 0..t.nrows() {
        for j in 0..t.ncols() {
            let value = t[[i, j]];
            if value > 0. {
                let p = value / n;
                mi += p * (p / ((rows[i] / n) * (cols[j] / n))).ln();
            }
        }
    }
    let ha = -rows
        .iter()
        .filter(|v| **v > 0.)
        .map(|v| {
            let p = *v / n;
            p * p.ln()
        })
        .sum::<f64>();
    let hb = -cols
        .iter()
        .filter(|v| **v > 0.)
        .map(|v| {
            let p = *v / n;
            p * p.ln()
        })
        .sum::<f64>();
    mi / (ha * hb).sqrt().max(1e-12)
}
