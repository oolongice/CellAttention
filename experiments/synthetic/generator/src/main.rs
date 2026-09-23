// Generate the maintained radial-sector spatial Hill benchmark dataset.
use ndarray::Array2;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use std::{
    f32::consts::TAU,
    fs,
    io::{BufWriter, Write},
    path::Path,
};

const CELLS: usize = 1000;
const RECEIVER_GROUPS: usize = 6;
const SECTORS: usize = 6;
const GROUPS: usize = RECEIVER_GROUPS + SECTORS;
const MARKERS_PER_GROUP: usize = 3;
const NOISE_GENES: usize = 12;
const CENTER_CELLS: usize = 650;
const FILLER_CELLS: usize = 120;
const SENDER_CELLS_START: usize = CENTER_CELLS + FILLER_CELLS;
const SPACE_CENTER: f32 = 500.0;
const CENTER_RADIUS_UM: f32 = 270.0;
const FILLER_OUTER_RADIUS_UM: f32 = 340.0;
const ANNULUS_OUTER_RADIUS_UM: f32 = 490.0;
const DIFFUSION_BANDWIDTH_UM: f32 = 130.0;
const TRUTH_EDGE_MAX_DISTANCE_UM: f32 = 320.0;

#[derive(Clone, Copy, Debug)]
enum Variant {
    SharedTarget,
    IndependentTargets,
    TwoSourcesPerTarget,
    VarianceMatchedMarkers,
}

impl Variant {
    fn parse(value: &str) -> Self {
        match value {
            "shared_target" => Self::SharedTarget,
            "independent_targets" => Self::IndependentTargets,
            "two_sources_per_target" => Self::TwoSourcesPerTarget,
            "variance_matched_markers" => Self::VarianceMatchedMarkers,
            _ => panic!(
                "unknown variant '{value}'; expected shared_target, independent_targets, two_sources_per_target, or variance_matched_markers"
            ),
        }
    }

    fn default_directory(self) -> &'static str {
        match self {
            Self::SharedTarget => "shared_target",
            Self::IndependentTargets => "independent_targets",
            Self::TwoSourcesPerTarget => "two_sources_per_target",
            Self::VarianceMatchedMarkers => "variance_matched_markers",
        }
    }

    fn target_count(self) -> usize {
        match self {
            Self::SharedTarget => 1,
            Self::IndependentTargets | Self::TwoSourcesPerTarget | Self::VarianceMatchedMarkers => {
                RECEIVER_GROUPS
            }
        }
    }
}

#[derive(Clone, Debug)]
struct TruthRelation {
    target_gene: String,
    target_index: usize,
    receiver_group: usize,
    source_sector: usize,
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let variant = Variant::parse(
        &arguments
            .next()
            .unwrap_or_else(|| "shared_target".to_string()),
    );
    let output_directory = arguments
        .next()
        .unwrap_or_else(|| variant.default_directory().to_string());
    let generation_seed = arguments
        .next()
        .map(|value| {
            value
                .parse::<u64>()
                .unwrap_or_else(|_| panic!("generation seed must be a non-negative integer"))
        })
        .unwrap_or(168);
    if arguments.next().is_some() {
        panic!(
            "usage: cellattention-synthetic-generator \
             [variant] [output_directory] [generation_seed]"
        );
    }
    let root = Path::new(&output_directory);
    fs::create_dir_all(root).unwrap();
    let mut rng = StdRng::seed_from_u64(generation_seed);
    let marker_noise = Normal::<f32>::new(0.0, 1.2).unwrap();
    let mut coordinates = Array2::<f32>::zeros((CELLS, 2));
    let mut groups = vec![0usize; CELLS];

    // -------------------------------------------------------------------------
    // Mixed receiver groups inside the central disk, in micrometer coordinates
    // -------------------------------------------------------------------------
    for cell in 0..CENTER_CELLS {
        let radius = CENTER_RADIUS_UM * rng.random::<f32>().sqrt();
        let angle = TAU * rng.random::<f32>();
        coordinates[[cell, 0]] = SPACE_CENTER + radius * angle.cos();
        coordinates[[cell, 1]] = SPACE_CENTER + radius * angle.sin();
        groups[cell] = rng.random_range(0..RECEIVER_GROUPS);
    }

    // -------------------------------------------------------------------------
    // Sector-group filler cells fill the former gap but do not express source genes
    // -------------------------------------------------------------------------
    for cell in CENTER_CELLS..SENDER_CELLS_START {
        let radius = (CENTER_RADIUS_UM.powi(2)
            + rng.random::<f32>() * (FILLER_OUTER_RADIUS_UM.powi(2) - CENTER_RADIUS_UM.powi(2)))
        .sqrt();
        let angle = TAU * rng.random::<f32>();
        coordinates[[cell, 0]] = SPACE_CENTER + radius * angle.cos();
        coordinates[[cell, 1]] = SPACE_CENTER + radius * angle.sin();
        let sector = ((angle / TAU * SECTORS as f32).floor() as usize).min(SECTORS - 1);
        groups[cell] = RECEIVER_GROUPS + sector;
    }

    // -------------------------------------------------------------------------
    // One dominant sender group per sector in the outer annulus
    // -------------------------------------------------------------------------
    for cell in SENDER_CELLS_START..CELLS {
        let radius = (FILLER_OUTER_RADIUS_UM.powi(2)
            + rng.random::<f32>()
                * (ANNULUS_OUTER_RADIUS_UM.powi(2) - FILLER_OUTER_RADIUS_UM.powi(2)))
        .sqrt();
        let angle = TAU * rng.random::<f32>();
        coordinates[[cell, 0]] = SPACE_CENTER + radius * angle.cos();
        coordinates[[cell, 1]] = SPACE_CENTER + radius * angle.sin();
        let sector = ((angle / TAU * SECTORS as f32).floor() as usize).min(SECTORS - 1);
        let assigned = if rng.random::<f32>() < 0.9 {
            sector
        } else if rng.random::<f32>() < 0.5 {
            (sector + 1) % SECTORS
        } else {
            (sector + SECTORS - 1) % SECTORS
        };
        groups[cell] = RECEIVER_GROUPS + assigned;
    }

    let marker_end = GROUPS * MARKERS_PER_GROUP;
    let noise_start = marker_end;
    let source_start = noise_start + NOISE_GENES;
    let target_start = source_start + SECTORS;
    let num_genes = target_start + variant.target_count();
    let mut expression = Array2::<f32>::zeros((CELLS, num_genes));
    let mut gene_ids = Vec::new();

    // -------------------------------------------------------------------------
    // Clear cell-type markers
    // -------------------------------------------------------------------------
    for group in 0..GROUPS {
        let shared_noise = (0..CELLS)
            .map(|_| marker_noise.sample(&mut rng))
            .collect::<Vec<_>>();
        for marker in 0..MARKERS_PER_GROUP {
            let gene = group * MARKERS_PER_GROUP + marker;
            gene_ids.push(format!("marker_group{}_{}", group, marker));
            for cell in 0..CELLS {
                let mean = if groups[cell] == group { 18.0 } else { 1.2 };
                let specific_noise = marker_noise.sample(&mut rng);
                let correlated_noise =
                    0.948_683_3 * shared_noise[cell] + 0.316_227_76 * specific_noise;
                expression[[cell, gene]] = (mean + correlated_noise).max(0.0);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Low-value sparse random noise genes
    // -------------------------------------------------------------------------
    for noise in 0..NOISE_GENES {
        gene_ids.push(format!("low_noise_{noise}"));
        for cell in 0..CELLS {
            if rng.random::<f32>() < 0.08 {
                expression[[cell, noise_start + noise]] =
                    (0.8 + 0.35 * marker_noise.sample(&mut rng)).max(0.0);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Strict 1-to-1 sector source pattern
    // -------------------------------------------------------------------------
    let mut true_fields = Array2::<f32>::zeros((CELLS, SECTORS));
    for sector in 0..SECTORS {
        gene_ids.push(format!("source_{sector}"));
        let gene = source_start + sector;
        for cell in SENDER_CELLS_START..CELLS {
            if groups[cell] == RECEIVER_GROUPS + sector && rng.random::<f32>() < 0.75 {
                expression[[cell, gene]] = (14.0 + 2.0 * marker_noise.sample(&mut rng)).max(0.0);
            }
        }
        let field = diffuse(&coordinates, &expression, gene, DIFFUSION_BANDWIDTH_UM);
        for cell in 0..CELLS {
            true_fields[[cell, sector]] = field[cell];
        }
    }

    // -------------------------------------------------------------------------
    // Target response variants. Independent variants use one target per receiver;
    // the two-source variant adds two separately saturating Hill contributions.
    // -------------------------------------------------------------------------
    let mut truth = "target_gene,receiver_group,source_gene,sender_group\n".to_string();
    let mut relations = Vec::new();
    for target in 0..variant.target_count() {
        gene_ids.push(format!("target_{target}"));
    }
    for receiver in 0..RECEIVER_GROUPS {
        let target_offset = match variant {
            Variant::SharedTarget => 0,
            Variant::IndependentTargets
            | Variant::TwoSourcesPerTarget
            | Variant::VarianceMatchedMarkers => receiver,
        };
        let target_index = target_start + target_offset;
        let target_name = format!("target_{target_offset}");
        let sources = match variant {
            Variant::SharedTarget
            | Variant::IndependentTargets
            | Variant::VarianceMatchedMarkers => vec![receiver],
            Variant::TwoSourcesPerTarget => vec![
                receiver,
                least_correlated_source(receiver, &groups, &true_fields),
            ],
        };
        for &source in &sources {
            truth.push_str(&format!(
                "{target_name},{receiver},source_{source},{}\n",
                RECEIVER_GROUPS + source
            ));
            relations.push(TruthRelation {
                target_gene: target_name.clone(),
                target_index,
                receiver_group: receiver,
                source_sector: source,
            });
        }
        let thresholds = sources
            .iter()
            .map(|&source| {
                let values: Vec<f32> = (0..CENTER_CELLS)
                    .filter(|&cell| groups[cell] == receiver)
                    .map(|cell| true_fields[[cell, source]])
                    .collect();
                percentile(&values, 0.82).max(1e-6)
            })
            .collect::<Vec<_>>();
        for cell in 0..CENTER_CELLS {
            if groups[cell] == receiver {
                let effect_scale = if sources.len() == 1 { 30.0 } else { 18.0 };
                let response = sources
                    .iter()
                    .zip(&thresholds)
                    .map(|(&source, &threshold)| {
                        let hill = (true_fields[[cell, source]] / threshold).powf(1.5);
                        effect_scale * hill / (1.0 + hill)
                    })
                    .sum::<f32>();
                expression[[cell, target_index]] =
                    3.0 + response + marker_noise.sample(&mut rng).max(-2.0);
            }
        }
    }

    // In every formal scenario, receiver markers form a correlated intrinsic
    // program whose within-receiver variance matches the corresponding target.
    // This prevents a simple variance ranker from identifying targets merely
    // because ordinary marker genes have much smaller variance.
    make_receiver_markers_variance_matched(
        &mut expression,
        &groups,
        target_start,
        variant,
        &mut rng,
    );

    let cell_ids: Vec<_> = (0..CELLS)
        .map(|cell| format!("cell_{}", cell + 1))
        .collect();
    write_matrix(root.join("expression.csv"), &expression).unwrap();
    write_matrix(root.join("spatial_coordinates.csv"), &coordinates).unwrap();
    write_matrix(root.join("true_source_fields.csv"), &true_fields).unwrap();
    write_ids(root.join("cell_ids.txt"), &cell_ids).unwrap();
    write_ids(root.join("gene_ids.txt"), &gene_ids).unwrap();
    write_ids(
        root.join("cell_groups.txt"),
        &groups.iter().map(usize::to_string).collect::<Vec<_>>(),
    )
    .unwrap();
    fs::write(root.join("truth.csv"), truth).unwrap();
    fs::write(
        root.join("truth_cell_edges.csv"),
        truth_cell_edges(
            &coordinates,
            &expression,
            &groups,
            source_start,
            &relations,
            &cell_ids,
        ),
    )
    .unwrap();
    println!(
        "generated variant={variant:?} generation_seed={generation_seed} \
         cells={CELLS} genes={num_genes} truth_relations={}",
        relations.len()
    );
}

fn least_correlated_source(primary: usize, groups: &[usize], fields: &Array2<f32>) -> usize {
    let cells = (0..CENTER_CELLS)
        .filter(|&cell| groups[cell] == primary)
        .collect::<Vec<_>>();
    let correlation = |left: usize, right: usize| {
        let left_mean =
            cells.iter().map(|&cell| fields[[cell, left]]).sum::<f32>() / cells.len() as f32;
        let right_mean =
            cells.iter().map(|&cell| fields[[cell, right]]).sum::<f32>() / cells.len() as f32;
        let mut covariance = 0.0_f32;
        let mut left_ss = 0.0_f32;
        let mut right_ss = 0.0_f32;
        for &cell in &cells {
            let left_centered = fields[[cell, left]] - left_mean;
            let right_centered = fields[[cell, right]] - right_mean;
            covariance += left_centered * right_centered;
            left_ss += left_centered * left_centered;
            right_ss += right_centered * right_centered;
        }
        covariance / (left_ss * right_ss).sqrt().max(1e-12)
    };
    (0..SECTORS)
        .filter(|&candidate| candidate != primary)
        .min_by(|&left, &right| {
            correlation(primary, left)
                .abs()
                .total_cmp(&correlation(primary, right).abs())
        })
        .unwrap()
}

fn make_receiver_markers_variance_matched(
    expression: &mut Array2<f32>,
    groups: &[usize],
    target_start: usize,
    variant: Variant,
    rng: &mut StdRng,
) {
    use rand::seq::SliceRandom;
    let loadings = [0.95_f32, 1.05, 1.10];
    for receiver in 0..RECEIVER_GROUPS {
        let cells = (0..CELLS)
            .filter(|&cell| groups[cell] == receiver)
            .collect::<Vec<_>>();
        let target_offset = match variant {
            Variant::SharedTarget => 0,
            Variant::IndependentTargets
            | Variant::TwoSourcesPerTarget
            | Variant::VarianceMatchedMarkers => receiver,
        };
        let target = cells
            .iter()
            .map(|&cell| expression[[cell, target_start + target_offset]])
            .collect::<Vec<_>>();
        let mean = target.iter().sum::<f32>() / target.len() as f32;
        let mut intrinsic = target.clone();
        intrinsic.shuffle(rng);
        for marker in 0..MARKERS_PER_GROUP {
            let gene = receiver * MARKERS_PER_GROUP + marker;
            let mut marker_specific = target.clone();
            marker_specific.shuffle(rng);
            for cell in 0..CELLS {
                expression[[cell, gene]] = 0.0;
            }
            for ((&cell, &latent), &specific) in cells.iter().zip(&intrinsic).zip(&marker_specific)
            {
                let program = 0.948_683_3 * (latent - mean) + 0.316_227_76 * (specific - mean);
                expression[[cell, gene]] = (mean + loadings[marker] * program).max(0.0);
            }
        }
    }
}

fn truth_cell_edges(
    coordinates: &Array2<f32>,
    expression: &Array2<f32>,
    groups: &[usize],
    source_start: usize,
    relations: &[TruthRelation],
    cell_ids: &[String],
) -> String {
    let mut output = "source_gene,target_gene,sender_cell_index,receiver_cell_index,sender_cell_id,receiver_cell_id,sender_group,receiver_group,distance,true_weight\n".to_string();
    for relation in relations {
        let sector = relation.source_sector;
        let sender_group = RECEIVER_GROUPS + relation.source_sector;
        for source in SENDER_CELLS_START..CELLS {
            let source_activity = expression[[source, source_start + sector]];
            if groups[source] != sender_group || source_activity <= 0.0 {
                continue;
            }
            for target in 0..CENTER_CELLS {
                if groups[target] != relation.receiver_group
                    || expression[[target, relation.target_index]] <= 0.0
                {
                    continue;
                }
                let dx = coordinates[[target, 0]] - coordinates[[source, 0]];
                let dy = coordinates[[target, 1]] - coordinates[[source, 1]];
                let distance = (dx * dx + dy * dy).sqrt();
                if distance > TRUTH_EDGE_MAX_DISTANCE_UM {
                    continue;
                }
                let true_weight = (-distance / DIFFUSION_BANDWIDTH_UM).exp() * source_activity;
                output.push_str(&format!(
                    "source_{sector},{},{source},{target},{},{},{sender_group},{},{distance:.4},{true_weight:.6}\n",
                    relation.target_gene,
                    cell_ids[source],
                    cell_ids[target],
                    groups[target],
                ));
            }
        }
    }
    output
}

fn diffuse(
    coordinates: &Array2<f32>,
    expression: &Array2<f32>,
    gene: usize,
    bandwidth: f32,
) -> Vec<f32> {
    (0..CELLS)
        .map(|target| {
            (0..CELLS)
                .filter(|&source| expression[[source, gene]] > 0.0)
                .map(|source| {
                    let dx = coordinates[[target, 0]] - coordinates[[source, 0]];
                    let dy = coordinates[[target, 1]] - coordinates[[source, 1]];
                    (-((dx * dx + dy * dy).sqrt()) / bandwidth).exp() * expression[[source, gene]]
                })
                .sum()
        })
        .collect()
}

fn write_matrix(path: impl AsRef<Path>, matrix: &Array2<f32>) -> Result<(), String> {
    let file =
        fs::File::create(path).map_err(|error| format!("failed to create matrix: {error}"))?;
    let mut output = BufWriter::new(file);
    for row in matrix.rows() {
        for (column, value) in row.iter().enumerate() {
            if column > 0 {
                output
                    .write_all(b",")
                    .map_err(|error| format!("failed to write matrix: {error}"))?;
            }
            write!(output, "{value:.7}")
                .map_err(|error| format!("failed to write matrix: {error}"))?;
        }
        output
            .write_all(b"\n")
            .map_err(|error| format!("failed to write matrix: {error}"))?;
    }
    output
        .flush()
        .map_err(|error| format!("failed to flush matrix: {error}"))
}

fn write_ids(path: impl AsRef<Path>, ids: &[String]) -> Result<(), String> {
    fs::write(path, ids.join("\n") + "\n").map_err(|error| format!("failed to write IDs: {error}"))
}

fn percentile(values: &[f32], probability: f32) -> f32 {
    let mut values = values.to_vec();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[((values.len() - 1) as f32 * probability) as usize]
}
