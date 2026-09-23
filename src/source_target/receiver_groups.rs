use super::effects::{SparseEffectConfig, fit_fixed_receiver_group_effects};
use ndarray::{Array1, Array2};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverGroupCountMode {
    Manual(usize),
    DataDriven,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverGroupClusteringMethod {
    #[default]
    DirectKmeans,
    TruncatedWhitenedPcaKmeans,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReceiverGroupCountConfig {
    pub mode: ReceiverGroupCountMode,
    pub candidates: Vec<usize>,
    pub spatial_folds: usize,
    pub spatial_block_width: f64,
    pub minimum_effective_receiver_group_mass: f64,
    pub clustering_method: ReceiverGroupClusteringMethod,
    /// Number of whitened PCs; None uses the fitted receiver_group count.
    pub pca_components: Option<usize>,
    pub kmeans_restarts: usize,
    pub kmeans_iterations: usize,
    pub seed: u64,
}

impl Default for ReceiverGroupCountConfig {
    fn default() -> Self {
        Self {
            mode: ReceiverGroupCountMode::DataDriven,
            candidates: vec![2, 4, 6, 8, 10, 12, 14, 16],
            spatial_folds: 5,
            spatial_block_width: 160.0,
            minimum_effective_receiver_group_mass: 30.0,
            clustering_method: ReceiverGroupClusteringMethod::DirectKmeans,
            pca_components: None,
            kmeans_restarts: 6,
            kmeans_iterations: 100,
            seed: 20_260_810,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmbeddingReceiverGroupModel {
    pub embedding_means: Array1<f64>,
    pub embedding_standard_deviations: Array1<f64>,
    /// Optional input-feature by whitened-PC projection.
    pub projection: Option<Array2<f64>>,
    pub clustering_method: ReceiverGroupClusteringMethod,
    pub centers: Array2<f64>,
    pub temperature: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReceiverGroupCountCandidate {
    pub receiver_group_count: usize,
    pub baseline_mse: f64,
    pub physical_mse: f64,
    pub improvement_fraction: f64,
    pub improvement_standard_error: f64,
    pub mean_active_effects: f64,
    pub minimum_effective_train_mass: f64,
    pub eligible: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ReceiverGroupCountSelection {
    pub selected_receiver_group_count: usize,
    pub selection_rule: String,
    pub candidates: Vec<ReceiverGroupCountCandidate>,
}

pub fn fit_embedding_receiver_groups(
    embeddings: &Array2<f64>,
    receiver_group_count: usize,
    config: &ReceiverGroupCountConfig,
) -> Result<EmbeddingReceiverGroupModel, String> {
    validate_receiver_group_config(embeddings, receiver_group_count, config)?;
    let (transformed, means, standard_deviations, projection) = match config.clustering_method {
        ReceiverGroupClusteringMethod::DirectKmeans => {
            let (matrix, means, standard_deviations) = standardize_fit(embeddings);
            (matrix, means, standard_deviations, None)
        }
        ReceiverGroupClusteringMethod::TruncatedWhitenedPcaKmeans => {
            let components = config.pca_components.unwrap_or(receiver_group_count);
            whitened_pca_fit(embeddings, components)?
        }
    };
    let candidates: Vec<(f64, Array2<f64>)> = (0..config.kmeans_restarts)
        .into_par_iter()
        .map(|restart| {
            kmeans(
                &transformed,
                receiver_group_count,
                config.kmeans_iterations,
                config.seed.wrapping_add(restart as u64),
            )
        })
        .collect();
    let (_, centers) = candidates
        .into_iter()
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .ok_or_else(|| "k-means produced no candidate".to_string())?;
    let distances = squared_distances(&transformed, &centers);
    let mut gaps = Vec::with_capacity(embeddings.nrows());
    for row in distances.rows() {
        let mut values = row.to_vec();
        values.sort_unstable_by(f64::total_cmp);
        gaps.push(if values.len() > 1 {
            values[1] - values[0]
        } else {
            1.0
        });
    }
    gaps.sort_unstable_by(f64::total_cmp);
    let temperature = gaps[gaps.len() / 2].max(1e-3);
    Ok(EmbeddingReceiverGroupModel {
        embedding_means: means,
        embedding_standard_deviations: standard_deviations,
        projection,
        clustering_method: config.clustering_method,
        centers,
        temperature,
    })
}

impl EmbeddingReceiverGroupModel {
    pub fn probabilities(&self, embeddings: &Array2<f64>) -> Result<Array2<f64>, String> {
        if embeddings.ncols() != self.embedding_means.len() {
            return Err("embedding dimension does not match receiver_group model".to_string());
        }
        let transformed = if let Some(projection) = &self.projection {
            center_and_project(embeddings, &self.embedding_means, projection)
        } else {
            let mut matrix = Array2::<f64>::zeros(embeddings.raw_dim());
            for cell in 0..embeddings.nrows() {
                for feature in 0..embeddings.ncols() {
                    matrix[[cell, feature]] = (embeddings[[cell, feature]]
                        - self.embedding_means[feature])
                        / self.embedding_standard_deviations[feature];
                }
            }
            matrix
        };
        let distances = squared_distances(&transformed, &self.centers);
        let mut probabilities = Array2::<f64>::zeros(distances.raw_dim());
        for cell in 0..distances.nrows() {
            let maximum = (0..distances.ncols())
                .map(|receiver_group| -distances[[cell, receiver_group]] / self.temperature)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut total = 0.0;
            for receiver_group in 0..distances.ncols() {
                let value = (-distances[[cell, receiver_group]] / self.temperature - maximum)
                    .clamp(-60.0, 0.0)
                    .exp();
                probabilities[[cell, receiver_group]] = value;
                total += value;
            }
            for receiver_group in 0..distances.ncols() {
                probabilities[[cell, receiver_group]] /= total;
            }
        }
        Ok(probabilities)
    }
}

pub fn select_receiver_group_count(
    embeddings: &Array2<f64>,
    fields: &Array2<f64>,
    residuals: &Array2<f64>,
    coordinates: &Array2<f64>,
    receiver_group_config: &ReceiverGroupCountConfig,
    effect_config: &SparseEffectConfig,
) -> Result<ReceiverGroupCountSelection, String> {
    if embeddings.nrows() != fields.nrows()
        || fields.nrows() != residuals.nrows()
        || fields.nrows() != coordinates.nrows()
    {
        return Err("receiver_group-selection inputs must share the same cells".to_string());
    }
    if let ReceiverGroupCountMode::Manual(count) = receiver_group_config.mode {
        validate_receiver_group_config(embeddings, count, receiver_group_config)?;
        return Ok(ReceiverGroupCountSelection {
            selected_receiver_group_count: count,
            selection_rule: "manual".to_string(),
            candidates: Vec::new(),
        });
    }
    let folds = spatial_folds(coordinates, receiver_group_config)?;
    let mut candidates = Vec::new();
    for &count in &receiver_group_config.candidates {
        validate_receiver_group_config(embeddings, count, receiver_group_config)?;
        let mut baseline_losses = Vec::new();
        let mut physical_losses = Vec::new();
        let mut active_effects = Vec::new();
        let mut minimum_masses = Vec::new();
        for (fold_index, validation_indices) in folds.iter().enumerate() {
            let training_indices = complement_indices(embeddings.nrows(), validation_indices);
            let training_embeddings = take_rows(embeddings, &training_indices);
            let validation_embeddings = take_rows(embeddings, validation_indices);
            let mut fold_config = receiver_group_config.clone();
            fold_config.seed = fold_config
                .seed
                .wrapping_add((count * 100 + fold_index) as u64);
            let receiver_group_model =
                fit_embedding_receiver_groups(&training_embeddings, count, &fold_config)?;
            let training_probabilities =
                receiver_group_model.probabilities(&training_embeddings)?;
            let validation_probabilities =
                receiver_group_model.probabilities(&validation_embeddings)?;
            let training_fields = take_rows(fields, &training_indices);
            let training_residuals = take_rows(residuals, &training_indices);
            let validation_fields = take_rows(fields, validation_indices);
            let validation_residuals = take_rows(residuals, validation_indices);
            let effect_model = fit_fixed_receiver_group_effects(
                &training_fields,
                &training_residuals,
                &training_probabilities,
                effect_config,
            )?;
            let prediction = effect_model.predict(&validation_fields, &validation_probabilities)?;
            let baseline = validation_probabilities.dot(&effect_model.intercept);
            physical_losses.push(mean_squared_error(&validation_residuals, &prediction));
            baseline_losses.push(mean_squared_error(&validation_residuals, &baseline));
            active_effects.push(effect_model.active_effects as f64);
            minimum_masses.push(
                training_probabilities
                    .sum_axis(ndarray::Axis(0))
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min),
            );
        }
        let fold_improvements: Vec<f64> = baseline_losses
            .iter()
            .zip(&physical_losses)
            .map(|(baseline, physical)| (baseline - physical) / baseline.max(1e-12))
            .collect();
        let improvement = mean(&fold_improvements);
        let standard_error =
            sample_standard_deviation(&fold_improvements) / (fold_improvements.len() as f64).sqrt();
        let minimum_mass = mean(&minimum_masses);
        candidates.push(ReceiverGroupCountCandidate {
            receiver_group_count: count,
            baseline_mse: mean(&baseline_losses),
            physical_mse: mean(&physical_losses),
            improvement_fraction: improvement,
            improvement_standard_error: standard_error,
            mean_active_effects: mean(&active_effects),
            minimum_effective_train_mass: minimum_mass,
            eligible: minimum_mass >= receiver_group_config.minimum_effective_receiver_group_mass,
        });
    }
    let eligible: Vec<&ReceiverGroupCountCandidate> =
        candidates.iter().filter(|row| row.eligible).collect();
    let pool: Vec<&ReceiverGroupCountCandidate> = if eligible.is_empty() {
        candidates.iter().collect()
    } else {
        eligible
    };
    let best = pool
        .iter()
        .copied()
        .max_by(|left, right| {
            left.improvement_fraction
                .total_cmp(&right.improvement_fraction)
        })
        .ok_or_else(|| "no receiver_group-count candidates".to_string())?;
    let threshold = best.improvement_fraction - best.improvement_standard_error;
    let selected = pool
        .into_iter()
        .filter(|row| row.improvement_fraction >= threshold)
        .map(|row| row.receiver_group_count)
        .min()
        .unwrap_or(best.receiver_group_count);
    Ok(ReceiverGroupCountSelection {
        selected_receiver_group_count: selected,
        selection_rule: "spatial_cv_one_standard_error".to_string(),
        candidates,
    })
}

fn validate_receiver_group_config(
    embeddings: &Array2<f64>,
    receiver_group_count: usize,
    config: &ReceiverGroupCountConfig,
) -> Result<(), String> {
    if embeddings.nrows() < receiver_group_count
        || embeddings.ncols() == 0
        || receiver_group_count < 1
    {
        return Err("invalid embedding shape or receiver_group count".to_string());
    }
    if embeddings.iter().any(|value| !value.is_finite()) {
        return Err("embeddings contain a non-finite value".to_string());
    }
    if config.kmeans_restarts == 0 || config.kmeans_iterations == 0 {
        return Err("k-means restarts and iterations must be > 0".to_string());
    }
    if config.clustering_method == ReceiverGroupClusteringMethod::TruncatedWhitenedPcaKmeans {
        let components = config.pca_components.unwrap_or(receiver_group_count);
        if components == 0 || components > embeddings.ncols() || components >= embeddings.nrows() {
            return Err(
                "PCA components must be in 1..=embedding dimensions and less than cell count"
                    .to_string(),
            );
        }
    }
    Ok(())
}

fn whitened_pca_fit(
    matrix: &Array2<f64>,
    components: usize,
) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>, Option<Array2<f64>>), String> {
    let means = Array1::from_iter(
        (0..matrix.ncols()).map(|column| matrix.column(column).sum() / matrix.nrows() as f64),
    );
    let mut covariance = Array2::<f64>::zeros((matrix.ncols(), matrix.ncols()));
    let denominator = (matrix.nrows() - 1) as f64;
    for row in 0..matrix.nrows() {
        for left in 0..matrix.ncols() {
            let x = matrix[[row, left]] - means[left];
            for right in left..matrix.ncols() {
                covariance[[left, right]] += x * (matrix[[row, right]] - means[right]);
            }
        }
    }
    for left in 0..matrix.ncols() {
        for right in left..matrix.ncols() {
            let value = covariance[[left, right]] / denominator;
            covariance[[left, right]] = value;
            covariance[[right, left]] = value;
        }
    }
    let (eigenvalues, eigenvectors) = symmetric_eigen(covariance);
    let largest = eigenvalues[0].max(1e-12);
    let floor = largest * 1e-10;
    let mut projection = Array2::<f64>::zeros((matrix.ncols(), components));
    for component in 0..components {
        let scale = eigenvalues[component].max(floor).sqrt();
        for feature in 0..matrix.ncols() {
            projection[[feature, component]] = eigenvectors[[feature, component]] / scale;
        }
    }
    let transformed = center_and_project(matrix, &means, &projection);
    Ok((
        transformed,
        means,
        Array1::ones(matrix.ncols()),
        Some(projection),
    ))
}

fn center_and_project(
    matrix: &Array2<f64>,
    means: &Array1<f64>,
    projection: &Array2<f64>,
) -> Array2<f64> {
    let mut output = Array2::<f64>::zeros((matrix.nrows(), projection.ncols()));
    for row in 0..matrix.nrows() {
        for component in 0..projection.ncols() {
            output[[row, component]] = (0..matrix.ncols())
                .map(|feature| {
                    (matrix[[row, feature]] - means[feature]) * projection[[feature, component]]
                })
                .sum();
        }
    }
    output
}

fn symmetric_eigen(mut matrix: Array2<f64>) -> (Vec<f64>, Array2<f64>) {
    let dimensions = matrix.nrows();
    let mut vectors = Array2::<f64>::eye(dimensions);
    for _ in 0..100 {
        let mut maximum: f64 = 0.0;
        for left in 0..dimensions {
            for right in (left + 1)..dimensions {
                let off_diagonal = matrix[[left, right]];
                maximum = maximum.max(off_diagonal.abs());
                if off_diagonal.abs() <= 1e-12 {
                    continue;
                }
                let tau = (matrix[[right, right]] - matrix[[left, left]]) / (2.0 * off_diagonal);
                let tangent = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
                let sine = tangent * cosine;
                let left_diagonal = matrix[[left, left]];
                let right_diagonal = matrix[[right, right]];
                for index in 0..dimensions {
                    if index == left || index == right {
                        continue;
                    }
                    let a = matrix[[index, left]];
                    let b = matrix[[index, right]];
                    matrix[[index, left]] = cosine * a - sine * b;
                    matrix[[left, index]] = matrix[[index, left]];
                    matrix[[index, right]] = sine * a + cosine * b;
                    matrix[[right, index]] = matrix[[index, right]];
                }
                matrix[[left, left]] = cosine * cosine * left_diagonal
                    - 2.0 * sine * cosine * off_diagonal
                    + sine * sine * right_diagonal;
                matrix[[right, right]] = sine * sine * left_diagonal
                    + 2.0 * sine * cosine * off_diagonal
                    + cosine * cosine * right_diagonal;
                matrix[[left, right]] = 0.0;
                matrix[[right, left]] = 0.0;
                for index in 0..dimensions {
                    let a = vectors[[index, left]];
                    let b = vectors[[index, right]];
                    vectors[[index, left]] = cosine * a - sine * b;
                    vectors[[index, right]] = sine * a + cosine * b;
                }
            }
        }
        if maximum <= 1e-10 {
            break;
        }
    }
    let mut order = (0..dimensions).collect::<Vec<_>>();
    order.sort_by(|&left, &right| matrix[[right, right]].total_cmp(&matrix[[left, left]]));
    let values = order
        .iter()
        .map(|&index| matrix[[index, index]].max(0.0))
        .collect();
    let sorted_vectors = Array2::from_shape_fn((dimensions, dimensions), |(row, column)| {
        vectors[[row, order[column]]]
    });
    (values, sorted_vectors)
}

fn standardize_fit(matrix: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let mut means = Array1::zeros(matrix.ncols());
    let mut standard_deviations = Array1::zeros(matrix.ncols());
    let mut standardized = Array2::zeros(matrix.raw_dim());
    for column in 0..matrix.ncols() {
        means[column] = matrix.column(column).sum() / matrix.nrows() as f64;
        standard_deviations[column] = (matrix
            .column(column)
            .iter()
            .map(|value| (value - means[column]).powi(2))
            .sum::<f64>()
            / matrix.nrows() as f64)
            .sqrt()
            .max(1e-8);
        for row in 0..matrix.nrows() {
            standardized[[row, column]] =
                (matrix[[row, column]] - means[column]) / standard_deviations[column];
        }
    }
    (standardized, means, standard_deviations)
}

fn kmeans(matrix: &Array2<f64>, count: usize, iterations: usize, seed: u64) -> (f64, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut centers = Array2::<f64>::zeros((count, matrix.ncols()));
    let first = rng.random_range(0..matrix.nrows());
    centers.row_mut(0).assign(&matrix.row(first));
    let mut nearest = vec![f64::INFINITY; matrix.nrows()];
    for center in 1..count {
        for cell in 0..matrix.nrows() {
            nearest[cell] =
                nearest[cell].min(squared_row_distance(matrix, cell, &centers, center - 1));
        }
        let total: f64 = nearest.iter().sum();
        let selected = if total > 0.0 {
            let mut draw = rng.random::<f64>() * total;
            let mut chosen = matrix.nrows() - 1;
            for (cell, distance) in nearest.iter().enumerate() {
                draw -= distance;
                if draw <= 0.0 {
                    chosen = cell;
                    break;
                }
            }
            chosen
        } else {
            rng.random_range(0..matrix.nrows())
        };
        centers.row_mut(center).assign(&matrix.row(selected));
    }
    let mut assignments = vec![0usize; matrix.nrows()];
    for _ in 0..iterations {
        let new_assignments: Vec<usize> = (0..matrix.nrows())
            .into_par_iter()
            .map(|cell| {
                (0..count)
                    .min_by(|left, right| {
                        squared_row_distance(matrix, cell, &centers, *left)
                            .total_cmp(&squared_row_distance(matrix, cell, &centers, *right))
                    })
                    .unwrap()
            })
            .collect();
        if new_assignments == assignments {
            break;
        }
        assignments = new_assignments;
        let mut next = Array2::<f64>::zeros(centers.raw_dim());
        let mut sizes = vec![0usize; count];
        for cell in 0..matrix.nrows() {
            sizes[assignments[cell]] += 1;
            for feature in 0..matrix.ncols() {
                next[[assignments[cell], feature]] += matrix[[cell, feature]];
            }
        }
        for receiver_group in 0..count {
            if sizes[receiver_group] == 0 {
                next.row_mut(receiver_group)
                    .assign(&matrix.row(rng.random_range(0..matrix.nrows())));
            } else {
                for feature in 0..matrix.ncols() {
                    next[[receiver_group, feature]] /= sizes[receiver_group] as f64;
                }
            }
        }
        centers = next;
    }
    let loss = (0..matrix.nrows())
        .map(|cell| {
            (0..count)
                .map(|receiver_group| squared_row_distance(matrix, cell, &centers, receiver_group))
                .fold(f64::INFINITY, f64::min)
        })
        .sum();
    (loss, centers)
}

fn squared_distances(matrix: &Array2<f64>, centers: &Array2<f64>) -> Array2<f64> {
    let rows: Vec<Vec<f64>> = (0..matrix.nrows())
        .into_par_iter()
        .map(|cell| {
            (0..centers.nrows())
                .map(|receiver_group| squared_row_distance(matrix, cell, centers, receiver_group))
                .collect()
        })
        .collect();
    Array2::from_shape_vec(
        (matrix.nrows(), centers.nrows()),
        rows.into_iter().flatten().collect(),
    )
    .unwrap()
}

fn squared_row_distance(
    matrix: &Array2<f64>,
    row: usize,
    centers: &Array2<f64>,
    center: usize,
) -> f64 {
    (0..matrix.ncols())
        .map(|feature| (matrix[[row, feature]] - centers[[center, feature]]).powi(2))
        .sum()
}

fn spatial_folds(
    coordinates: &Array2<f64>,
    config: &ReceiverGroupCountConfig,
) -> Result<Vec<Vec<usize>>, String> {
    if coordinates.ncols() < 2
        || !config.spatial_block_width.is_finite()
        || config.spatial_block_width <= 0.0
        || config.spatial_folds < 2
    {
        return Err("invalid coordinates or spatial fold configuration".to_string());
    }
    let mut folds = vec![Vec::new(); config.spatial_folds];
    for cell in 0..coordinates.nrows() {
        let x = (coordinates[[cell, 0]] / config.spatial_block_width).floor() as i64;
        let y = (coordinates[[cell, 1]] / config.spatial_block_width).floor() as i64;
        let z = if coordinates.ncols() >= 3 {
            (coordinates[[cell, 2]] / config.spatial_block_width).floor() as i64
        } else {
            0
        };
        let hash =
            x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663) ^ z.wrapping_mul(83_492_791);
        folds[hash.rem_euclid(config.spatial_folds as i64) as usize].push(cell);
    }
    if folds.iter().any(Vec::is_empty) {
        folds = vec![Vec::new(); config.spatial_folds];
        for cell in 0..coordinates.nrows() {
            folds[cell % config.spatial_folds].push(cell);
        }
    }
    Ok(folds)
}

fn complement_indices(cells: usize, excluded: &[usize]) -> Vec<usize> {
    let mut mask = vec![true; cells];
    for &index in excluded {
        mask[index] = false;
    }
    mask.into_iter()
        .enumerate()
        .filter_map(|(index, keep)| keep.then_some(index))
        .collect()
}
fn take_rows(matrix: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let mut output = Array2::zeros((indices.len(), matrix.ncols()));
    for (row, &source) in indices.iter().enumerate() {
        output.row_mut(row).assign(&matrix.row(source));
    }
    output
}
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
fn sample_standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let center = mean(values);
    (values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}
fn mean_squared_error(left: &Array2<f64>, right: &Array2<f64>) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / left.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn fixed_embedding_probabilities_sum_to_one() {
        let embeddings = array![[0., 0.], [0.1, 0.], [5., 5.], [5.1, 5.]];
        let config = ReceiverGroupCountConfig {
            kmeans_restarts: 2,
            ..Default::default()
        };
        let model = fit_embedding_receiver_groups(&embeddings, 2, &config).unwrap();
        let probabilities = model.probabilities(&embeddings).unwrap();
        for row in probabilities.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-10);
        }
    }
    #[test]
    fn truncated_whitened_pca_probabilities_sum_to_one() {
        let embeddings = array![
            [0., 0., 0.],
            [0.1, 0.2, 0.],
            [4.9, 5.1, 1.],
            [5.0, 5.0, 1.1]
        ];
        let config = ReceiverGroupCountConfig {
            clustering_method: ReceiverGroupClusteringMethod::TruncatedWhitenedPcaKmeans,
            pca_components: Some(2),
            kmeans_restarts: 2,
            ..Default::default()
        };
        let model = fit_embedding_receiver_groups(&embeddings, 2, &config).unwrap();
        assert_eq!(model.centers.ncols(), 2);
        assert!(model.projection.is_some());
        let probabilities = model.probabilities(&embeddings).unwrap();
        for row in probabilities.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn manual_receiver_group_count_does_not_fit_or_reassign() {
        let embeddings = array![[0., 0.], [1., 1.], [2., 2.]];
        let fields = array![[0., 1.], [1., 0.], [2., 1.]];
        let residuals = array![[0.], [1.], [2.]];
        let coordinates = array![[0., 0.], [1., 0.], [2., 0.]];
        let config = ReceiverGroupCountConfig {
            mode: ReceiverGroupCountMode::Manual(2),
            ..Default::default()
        };
        let selected = select_receiver_group_count(
            &embeddings,
            &fields,
            &residuals,
            &coordinates,
            &config,
            &SparseEffectConfig {
                top_k_sources: 1,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(selected.selected_receiver_group_count, 2);
        assert!(selected.candidates.is_empty());
    }
}
