use ndarray::{Array2, Array3, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SparseEffectConfig {
    pub top_k_sources: usize,
    pub l1_penalty: f64,
    pub l2_penalty: f64,
    pub maximum_iterations: usize,
    pub convergence_tolerance: f64,
    pub minimum_training_improvement: f64,
}

impl Default for SparseEffectConfig {
    fn default() -> Self {
        Self {
            top_k_sources: 8,
            l1_penalty: 0.03,
            l2_penalty: 0.05,
            maximum_iterations: 60,
            convergence_tolerance: 1e-6,
            minimum_training_improvement: 0.005,
        }
    }
}

impl SparseEffectConfig {
    pub fn validate(&self, sources: usize) -> Result<(), String> {
        if self.top_k_sources == 0 || self.top_k_sources >= sources {
            return Err(format!("top_k_sources must be in 1..{sources}"));
        }
        for (name, value) in [
            ("l1_penalty", self.l1_penalty),
            ("l2_penalty", self.l2_penalty),
            ("convergence_tolerance", self.convergence_tolerance),
            (
                "minimum_training_improvement",
                self.minimum_training_improvement,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and >= 0"));
            }
        }
        if self.maximum_iterations == 0 {
            return Err("maximum_iterations must be > 0".to_string());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct FixedReceiverGroupEffectModel {
    /// [receiver_group, source, target]. This is the only trained source-target parameter.
    pub beta: Array3<f64>,
    /// Independent one-source effect used for candidate discovery.
    pub screening_signed_effect: Array3<f64>,
    /// Absolute standardized independent association score.
    pub screening_score: Array3<f64>,
    /// Corrected intercept: mean residual - mean field dot beta.
    pub intercept: Array2<f64>,
    pub training_improvement: Array2<f64>,
    pub active_effects: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct EffectRecord {
    pub receiver_group: usize,
    pub source: usize,
    pub target: usize,
    pub screening_signed_effect: f64,
    pub screening_score: f64,
    pub signed_beta: f64,
    pub derived_attention: f64,
    pub selected: bool,
    pub training_improvement: f64,
}

pub fn fit_fixed_receiver_group_effects(
    fields: &Array2<f64>,
    residuals: &Array2<f64>,
    receiver_group_probabilities: &Array2<f64>,
    config: &SparseEffectConfig,
) -> Result<FixedReceiverGroupEffectModel, String> {
    validate_inputs(fields, residuals, receiver_group_probabilities, config)?;
    let cells = fields.nrows();
    let sources = fields.ncols();
    let targets = residuals.ncols();
    let receiver_groups = receiver_group_probabilities.ncols();
    let mut beta = Array3::<f64>::zeros((receiver_groups, sources, targets));
    let mut screening_signed_effect = Array3::<f64>::zeros((receiver_groups, sources, targets));
    let mut screening_score = Array3::<f64>::zeros((receiver_groups, sources, targets));
    let mut intercept = Array2::<f64>::zeros((receiver_groups, targets));
    let mut improvements = Array2::<f64>::zeros((receiver_groups, targets));

    for receiver_group in 0..receiver_groups {
        let weights = receiver_group_probabilities
            .column(receiver_group)
            .to_owned();
        let mass = weights.sum();
        if mass <= 1e-8 {
            return Err(format!(
                "receiver_group {receiver_group} has zero effective mass"
            ));
        }
        let mean_fields = weighted_column_means(fields, &weights, mass);
        let mean_residuals = weighted_column_means(residuals, &weights, mass);
        let mut centered_fields = Array2::<f64>::zeros(fields.raw_dim());
        for cell in 0..cells {
            for source in 0..sources {
                centered_fields[[cell, source]] = fields[[cell, source]] - mean_fields[source];
            }
        }
        let mut weighted_residuals = Array2::<f64>::zeros(residuals.raw_dim());
        for cell in 0..cells {
            for target in 0..targets {
                weighted_residuals[[cell, target]] =
                    weights[cell] * (residuals[[cell, target]] - mean_residuals[target]);
            }
        }
        let covariance = centered_fields.t().dot(&weighted_residuals) / mass;
        let source_variances: Vec<f64> = (0..sources)
            .map(|source| {
                (0..cells)
                    .map(|cell| weights[cell] * centered_fields[[cell, source]].powi(2))
                    .sum::<f64>()
                    / mass
                    + 1e-12
            })
            .collect();

        for source in 0..sources {
            for target in 0..targets {
                if !(sources == targets && source == target) {
                    screening_signed_effect[[receiver_group, source, target]] =
                        covariance[[source, target]] / source_variances[source];
                    screening_score[[receiver_group, source, target]] =
                        covariance[[source, target]].abs() / source_variances[source].sqrt();
                }
            }
        }

        let fitted: Vec<TargetFit> = (0..targets)
            .into_par_iter()
            .map(|target| {
                fit_target(
                    target,
                    &centered_fields,
                    residuals,
                    &weights,
                    &mean_residuals,
                    &covariance,
                    &source_variances,
                    config,
                )
            })
            .collect();
        for fit in fitted {
            improvements[[receiver_group, fit.target]] = fit.improvement;
            if fit.improvement >= config.minimum_training_improvement {
                for (source, coefficient) in fit.coefficients {
                    beta[[receiver_group, source, fit.target]] = coefficient;
                }
            }
        }
        // The correction is exact even though fitting used receiver_group-centered fields.
        for target in 0..targets {
            let field_shift = (0..sources)
                .map(|source| mean_fields[source] * beta[[receiver_group, source, target]])
                .sum::<f64>();
            intercept[[receiver_group, target]] = mean_residuals[target] - field_shift;
        }
    }
    let active_effects = beta.iter().filter(|value| value.abs() > 0.0).count();
    Ok(FixedReceiverGroupEffectModel {
        beta,
        screening_signed_effect,
        screening_score,
        intercept,
        training_improvement: improvements,
        active_effects,
    })
}

impl FixedReceiverGroupEffectModel {
    pub fn predict(
        &self,
        fields: &Array2<f64>,
        receiver_group_probabilities: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let receiver_groups = self.beta.len_of(Axis(0));
        let sources = self.beta.len_of(Axis(1));
        let targets = self.beta.len_of(Axis(2));
        if fields.ncols() != sources || receiver_group_probabilities.ncols() != receiver_groups {
            return Err("field/receiver_group dimensions do not match fitted model".to_string());
        }
        if fields.nrows() != receiver_group_probabilities.nrows() {
            return Err(
                "fields and receiver_group probabilities have different cell counts".to_string(),
            );
        }
        let rows: Vec<Vec<f64>> = (0..fields.nrows())
            .into_par_iter()
            .map(|cell| {
                let mut result = vec![0.0; targets];
                for receiver_group in 0..receiver_groups {
                    let probability = receiver_group_probabilities[[cell, receiver_group]];
                    if probability == 0.0 {
                        continue;
                    }
                    for target in 0..targets {
                        let mut value = self.intercept[[receiver_group, target]];
                        for source in 0..sources {
                            value += fields[[cell, source]]
                                * self.beta[[receiver_group, source, target]];
                        }
                        result[target] += probability * value;
                    }
                }
                result
            })
            .collect();
        Array2::from_shape_vec(
            (fields.nrows(), targets),
            rows.into_iter().flatten().collect(),
        )
        .map_err(|error| error.to_string())
    }

    pub fn derived_attention(&self, receiver_group: usize, source: usize, target: usize) -> f64 {
        let denominator = self
            .beta
            .slice(ndarray::s![receiver_group, .., target])
            .iter()
            .map(|value| value.abs())
            .sum::<f64>();
        if denominator > 0.0 {
            self.beta[[receiver_group, source, target]].abs() / denominator
        } else {
            0.0
        }
    }

    pub fn effective_beta(
        &self,
        receiver_group_probabilities: &Array2<f64>,
        cell: usize,
        source: usize,
        target: usize,
    ) -> f64 {
        (0..self.beta.len_of(Axis(0)))
            .map(|receiver_group| {
                receiver_group_probabilities[[cell, receiver_group]]
                    * self.beta[[receiver_group, source, target]]
            })
            .sum()
    }

    pub fn cell_contribution(
        &self,
        fields: &Array2<f64>,
        receiver_group_probabilities: &Array2<f64>,
        cell: usize,
        source: usize,
        target: usize,
    ) -> f64 {
        fields[[cell, source]]
            * self.effective_beta(receiver_group_probabilities, cell, source, target)
    }

    pub fn effect_records(&self) -> Vec<EffectRecord> {
        let receiver_groups = self.beta.len_of(Axis(0));
        let sources = self.beta.len_of(Axis(1));
        let targets = self.beta.len_of(Axis(2));
        let mut records = Vec::with_capacity(receiver_groups * sources * targets);
        for receiver_group in 0..receiver_groups {
            for target in 0..targets {
                for source in 0..sources {
                    let value = self.beta[[receiver_group, source, target]];
                    records.push(EffectRecord {
                        receiver_group,
                        source,
                        target,
                        screening_signed_effect: self.screening_signed_effect
                            [[receiver_group, source, target]],
                        screening_score: self.screening_score[[receiver_group, source, target]],
                        signed_beta: value,
                        derived_attention: self.derived_attention(receiver_group, source, target),
                        selected: value != 0.0,
                        training_improvement: self.training_improvement[[receiver_group, target]],
                    });
                }
            }
        }
        records
    }
}

struct TargetFit {
    target: usize,
    coefficients: Vec<(usize, f64)>,
    improvement: f64,
}

#[allow(clippy::too_many_arguments)]
fn fit_target(
    target: usize,
    centered_fields: &Array2<f64>,
    residuals: &Array2<f64>,
    weights: &ndarray::Array1<f64>,
    mean_residuals: &[f64],
    covariance: &Array2<f64>,
    source_variances: &[f64],
    config: &SparseEffectConfig,
) -> TargetFit {
    let cells = centered_fields.nrows();
    let sources = centered_fields.ncols();
    let mut scored: Vec<(usize, f64)> = (0..sources)
        .filter(|source| !(sources == residuals.ncols() && *source == target))
        .map(|source| {
            (
                source,
                covariance[[source, target]].abs() / source_variances[source].sqrt(),
            )
        })
        .collect();
    scored.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    scored.truncate(config.top_k_sources.min(scored.len()));
    let selected: Vec<usize> = scored.into_iter().map(|value| value.0).collect();
    let mut coefficients = vec![0.0; selected.len()];
    let mut prediction = vec![0.0; cells];
    let mass = weights.sum();
    for _ in 0..config.maximum_iterations {
        let mut maximum_change = 0.0f64;
        for (slot, source) in selected.iter().copied().enumerate() {
            let old = coefficients[slot];
            let mut rho = 0.0;
            let mut denominator = config.l2_penalty;
            for cell in 0..cells {
                let value = centered_fields[[cell, source]];
                let residual =
                    residuals[[cell, target]] - mean_residuals[target] - prediction[cell]
                        + value * old;
                rho += weights[cell] * value * residual / mass;
                denominator += weights[cell] * value * value / mass;
            }
            let new = soft_threshold(rho, config.l1_penalty) / denominator.max(1e-12);
            let change = new - old;
            if change != 0.0 {
                for cell in 0..cells {
                    prediction[cell] += centered_fields[[cell, source]] * change;
                }
            }
            coefficients[slot] = new;
            maximum_change = maximum_change.max(change.abs());
        }
        if maximum_change < config.convergence_tolerance {
            break;
        }
    }
    let mut baseline = 0.0;
    let mut fitted = 0.0;
    for cell in 0..cells {
        let centered = residuals[[cell, target]] - mean_residuals[target];
        baseline += weights[cell] * centered * centered / mass;
        fitted += weights[cell] * (centered - prediction[cell]).powi(2) / mass;
    }
    let improvement = if baseline > 1e-12 {
        (baseline - fitted) / baseline
    } else {
        0.0
    };
    TargetFit {
        target,
        coefficients: selected.into_iter().zip(coefficients).collect(),
        improvement,
    }
}

fn soft_threshold(value: f64, penalty: f64) -> f64 {
    value.signum() * (value.abs() - penalty).max(0.0)
}

fn weighted_column_means(
    matrix: &Array2<f64>,
    weights: &ndarray::Array1<f64>,
    mass: f64,
) -> Vec<f64> {
    (0..matrix.ncols())
        .map(|column| {
            (0..matrix.nrows())
                .map(|row| weights[row] * matrix[[row, column]])
                .sum::<f64>()
                / mass
        })
        .collect()
}

fn validate_inputs(
    fields: &Array2<f64>,
    residuals: &Array2<f64>,
    probabilities: &Array2<f64>,
    config: &SparseEffectConfig,
) -> Result<(), String> {
    if fields.nrows() == 0 || fields.ncols() < 2 || residuals.ncols() == 0 {
        return Err("fields/residuals must be non-empty and have at least two sources".to_string());
    }
    if fields.nrows() != residuals.nrows() || fields.nrows() != probabilities.nrows() {
        return Err(
            "fields, residuals, and receiver_group probabilities must share cells".to_string(),
        );
    }
    if probabilities.ncols() == 0 {
        return Err("at least one receiver_group is required".to_string());
    }
    if fields
        .iter()
        .chain(residuals.iter())
        .chain(probabilities.iter())
        .any(|v| !v.is_finite())
    {
        return Err("input contains a non-finite value".to_string());
    }
    for (cell, row) in probabilities.rows().into_iter().enumerate() {
        if row.iter().any(|value| *value < 0.0) {
            return Err(format!(
                "receiver_group probabilities are negative at cell {cell}"
            ));
        }
        if (row.sum() - 1.0).abs() > 1e-6 {
            return Err(format!(
                "receiver_group probabilities at cell {cell} do not sum to one"
            ));
        }
    }
    config.validate(fields.ncols())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn corrected_intercept_gives_zero_weighted_mean_residual() {
        let fields = array![[0., 1., 2.], [1., 2., 0.], [2., 0., 1.], [3., 1., 1.]];
        let residuals = array![[1., 0.], [2., 1.], [3., 0.], [4., 1.]];
        let probabilities = array![[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]];
        let config = SparseEffectConfig {
            top_k_sources: 2,
            l1_penalty: 0.0,
            minimum_training_improvement: 0.0,
            ..Default::default()
        };
        let model =
            fit_fixed_receiver_group_effects(&fields, &residuals, &probabilities, &config).unwrap();
        for receiver_group in 0..2 {
            let weights = probabilities.column(receiver_group);
            for target in 0..2 {
                let weighted_residual = (0..4)
                    .map(|cell| {
                        let fitted = model.intercept[[receiver_group, target]]
                            + (0..3)
                                .map(|source| {
                                    fields[[cell, source]]
                                        * model.beta[[receiver_group, source, target]]
                                })
                                .sum::<f64>();
                        weights[cell] * (residuals[[cell, target]] - fitted)
                    })
                    .sum::<f64>()
                    / weights.sum();
                assert!(weighted_residual.abs() < 1e-8);
            }
        }
    }

    #[test]
    fn attention_is_derived_and_sums_to_one() {
        let model = FixedReceiverGroupEffectModel {
            beta: Array3::from_shape_vec((1, 3, 1), vec![1.0, -2.0, 0.0]).unwrap(),
            screening_signed_effect: Array3::zeros((1, 3, 1)),
            screening_score: Array3::zeros((1, 3, 1)),
            intercept: Array2::zeros((1, 1)),
            training_improvement: Array2::zeros((1, 1)),
            active_effects: 2,
        };
        let sum: f64 = (0..3)
            .map(|source| model.derived_attention(0, source, 0))
            .sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!((model.derived_attention(0, 1, 0) - 2.0 / 3.0).abs() < 1e-12);
    }
}
