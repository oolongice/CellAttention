use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Dimensionality of the extracellular space used by the Green's function.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpatialDimension {
    TwoD,
    ThreeD,
}

/// Parameters of the steady-state point-source reaction-diffusion equation
/// `D ∇²c - λc + qδ(x) = 0`.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReactionDiffusion {
    /// Diffusion coefficient D (> 0), in distance² / time.
    pub diffusion: f64,
    /// First-order degradation rate λ (> 0), in 1 / time.
    pub degradation: f64,
    /// Constant point-source production q (>= 0), in amount / time.
    pub production: f64,
    /// Regularizes the point-source singularity: r_eff = max(r, min_distance).
    pub min_distance: f64,
    pub dimension: SpatialDimension,
}

impl ReactionDiffusion {
    pub fn length_scale(&self) -> Result<f64, String> {
        self.validate()?;
        Ok((self.diffusion / self.degradation).sqrt())
    }

    pub fn validate(&self) -> Result<(), String> {
        if !self.diffusion.is_finite() || self.diffusion <= 0.0 {
            return Err("diffusion must be finite and > 0".to_string());
        }
        if !self.degradation.is_finite() || self.degradation <= 0.0 {
            return Err("degradation must be finite and > 0".to_string());
        }
        if !self.production.is_finite() || self.production < 0.0 {
            return Err("production must be finite and >= 0".to_string());
        }
        if !self.min_distance.is_finite() || self.min_distance <= 0.0 {
            return Err("min_distance must be finite and > 0".to_string());
        }
        Ok(())
    }

    /// Analytic Green's function evaluated at a distance. The singular point
    /// source is regularized at `min_distance` to represent finite-sized cells.
    pub fn kernel(&self, distance: f64) -> Result<f64, String> {
        self.validate()?;
        if !distance.is_finite() || distance < 0.0 {
            return Err("distance must be finite and >= 0".to_string());
        }
        let radius = distance.max(self.min_distance);
        let length = (self.diffusion / self.degradation).sqrt();
        let value = match self.dimension {
            SpatialDimension::TwoD => {
                self.production * bessel_k0(radius / length) / (2.0 * PI * self.diffusion)
            }
            SpatialDimension::ThreeD => {
                self.production * (-radius / length).exp() / (4.0 * PI * self.diffusion * radius)
            }
        };
        Ok(value)
    }
}

/// Radius-neighborhood and block execution settings.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct SpatialKernelConfig {
    /// Only sources at distance <= max_distance are included.
    pub max_distance: f64,
    /// 9999 is the unlimited sentinel; otherwise retain the nearest N sources.
    pub max_neighbors: usize,
    /// Whether a cell's expression contributes to its own field.
    pub include_self: bool,
    /// Number of receiver cells materialized per parallel block.
    pub target_block_size: usize,
}

impl Default for SpatialKernelConfig {
    fn default() -> Self {
        Self {
            max_distance: 200.0,
            max_neighbors: 9999,
            include_self: false,
            target_block_size: 4096,
        }
    }
}

impl SpatialKernelConfig {
    fn validate(&self) -> Result<(), String> {
        if !self.max_distance.is_finite() || self.max_distance <= 0.0 {
            return Err("max_distance must be finite and > 0".to_string());
        }
        if self.max_neighbors == 0 {
            return Err("max_neighbors must be > 0 (9999 means unlimited)".to_string());
        }
        if self.target_block_size == 0 {
            return Err("target_block_size must be > 0".to_string());
        }
        Ok(())
    }

    fn neighbor_limit(&self) -> Option<usize> {
        (self.max_neighbors != 9999).then_some(self.max_neighbors)
    }
}

/// CSR matrix with rows = receiver cells and columns = sender cells.
#[derive(Clone, Debug)]
pub struct SparseSpatialKernel {
    pub num_cells: usize,
    pub row_offsets: Vec<usize>,
    pub source_indices: Vec<usize>,
    pub weights: Vec<f32>,
}

impl SparseSpatialKernel {
    pub fn build(
        coordinates: &Array2<f64>,
        source_mask: Option<&[bool]>,
        equation: ReactionDiffusion,
        config: &SpatialKernelConfig,
    ) -> Result<Self, String> {
        let validated = ValidatedInput::new(coordinates, source_mask, equation, config)?;
        let index = UniformGrid::new(
            coordinates,
            validated.dimensions,
            config.max_distance,
            source_mask,
        );
        let rows: Vec<Vec<(usize, f32)>> = (0..coordinates.nrows())
            .into_par_iter()
            .map(|receiver| neighbors_for(receiver, coordinates, &index, equation, config))
            .collect();

        let mut row_offsets = Vec::with_capacity(coordinates.nrows() + 1);
        let total = rows.iter().map(Vec::len).sum();
        let mut source_indices = Vec::with_capacity(total);
        let mut weights = Vec::with_capacity(total);
        row_offsets.push(0);
        for row in rows {
            for (source, weight) in row {
                source_indices.push(source);
                weights.push(weight);
            }
            row_offsets.push(source_indices.len());
        }
        Ok(Self {
            num_cells: coordinates.nrows(),
            row_offsets,
            source_indices,
            weights,
        })
    }

    /// Computes K × X where X is cells × genes.
    pub fn multiply(&self, source_expression: &Array2<f32>) -> Result<Array2<f32>, String> {
        if source_expression.nrows() != self.num_cells {
            return Err(format!(
                "expression has {} cells; kernel has {}",
                source_expression.nrows(),
                self.num_cells
            ));
        }
        let genes = source_expression.ncols();
        let rows: Vec<Vec<f32>> = (0..self.num_cells)
            .into_par_iter()
            .map(|receiver| {
                let mut field = vec![0.0f32; genes];
                for edge in self.row_offsets[receiver]..self.row_offsets[receiver + 1] {
                    let source = self.source_indices[edge];
                    let weight = self.weights[edge];
                    for gene in 0..genes {
                        field[gene] += weight * source_expression[[source, gene]];
                    }
                }
                field
            })
            .collect();
        Array2::from_shape_vec(
            (self.num_cells, genes),
            rows.into_iter().flatten().collect(),
        )
        .map_err(|error| error.to_string())
    }
}

/// Memory-bounded field calculation. A spatial hash is built once, receiver cells
/// are processed in blocks, and each block's sparse rows are multiplied without
/// retaining the full cell-by-cell kernel.
pub fn compute_source_fields(
    coordinates: &Array2<f64>,
    source_expression: &Array2<f32>,
    source_mask: Option<&[bool]>,
    equation: ReactionDiffusion,
    config: &SpatialKernelConfig,
) -> Result<Array2<f32>, String> {
    let validated = ValidatedInput::new(coordinates, source_mask, equation, config)?;
    if source_expression.nrows() != coordinates.nrows() {
        return Err(format!(
            "expression has {} cells but coordinates have {}",
            source_expression.nrows(),
            coordinates.nrows()
        ));
    }
    let index = UniformGrid::new(
        coordinates,
        validated.dimensions,
        config.max_distance,
        source_mask,
    );
    let cells = coordinates.nrows();
    let genes = source_expression.ncols();
    let mut output = Array2::<f32>::zeros((cells, genes));

    for block_start in (0..cells).step_by(config.target_block_size) {
        let block_end = (block_start + config.target_block_size).min(cells);
        let block: Vec<Vec<f32>> = (block_start..block_end)
            .into_par_iter()
            .map(|receiver| {
                let neighbors = neighbors_for(receiver, coordinates, &index, equation, config);
                let mut field = vec![0.0f32; genes];
                for (source, weight) in neighbors {
                    for gene in 0..genes {
                        field[gene] += weight * source_expression[[source, gene]];
                    }
                }
                field
            })
            .collect();
        for (offset, row) in block.into_iter().enumerate() {
            for (gene, value) in row.into_iter().enumerate() {
                output[[block_start + offset, gene]] = value;
            }
        }
    }
    Ok(output)
}

struct ValidatedInput {
    dimensions: usize,
}

impl ValidatedInput {
    fn new(
        coordinates: &Array2<f64>,
        source_mask: Option<&[bool]>,
        equation: ReactionDiffusion,
        config: &SpatialKernelConfig,
    ) -> Result<Self, String> {
        equation.validate()?;
        config.validate()?;
        let dimensions = match equation.dimension {
            SpatialDimension::TwoD => 2,
            SpatialDimension::ThreeD => 3,
        };
        if coordinates.ncols() != dimensions {
            return Err(format!(
                "{:?} requires {dimensions} coordinate columns; found {}",
                equation.dimension,
                coordinates.ncols()
            ));
        }
        if coordinates.iter().any(|value| !value.is_finite()) {
            return Err("coordinates contain a non-finite value".to_string());
        }
        if let Some(mask) = source_mask {
            if mask.len() != coordinates.nrows() {
                return Err(format!(
                    "source mask has {} entries; expected {}",
                    mask.len(),
                    coordinates.nrows()
                ));
            }
        }
        Ok(Self { dimensions })
    }
}

struct UniformGrid {
    bins: HashMap<[i64; 3], Vec<usize>>,
    dimensions: usize,
    cell_width: f64,
}

impl UniformGrid {
    fn new(
        coordinates: &Array2<f64>,
        dimensions: usize,
        cell_width: f64,
        source_mask: Option<&[bool]>,
    ) -> Self {
        let mut bins: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
        for source in 0..coordinates.nrows() {
            if source_mask.is_some_and(|mask| !mask[source]) {
                continue;
            }
            let key = grid_key(coordinates, source, dimensions, cell_width);
            bins.entry(key).or_default().push(source);
        }
        Self {
            bins,
            dimensions,
            cell_width,
        }
    }

    fn candidates(&self, coordinates: &Array2<f64>, receiver: usize) -> Vec<usize> {
        let center = grid_key(coordinates, receiver, self.dimensions, self.cell_width);
        let mut candidates = Vec::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                let z_range = if self.dimensions == 3 { -1..=1 } else { 0..=0 };
                for dz in z_range {
                    if let Some(indices) =
                        self.bins
                            .get(&[center[0] + dx, center[1] + dy, center[2] + dz])
                    {
                        candidates.extend(indices.iter().copied());
                    }
                }
            }
        }
        candidates
    }
}

fn grid_key(coordinates: &Array2<f64>, cell: usize, dimensions: usize, width: f64) -> [i64; 3] {
    let mut key = [0i64; 3];
    for axis in 0..dimensions {
        key[axis] = (coordinates[[cell, axis]] / width).floor() as i64;
    }
    key
}

fn neighbors_for(
    receiver: usize,
    coordinates: &Array2<f64>,
    index: &UniformGrid,
    equation: ReactionDiffusion,
    config: &SpatialKernelConfig,
) -> Vec<(usize, f32)> {
    let max_squared = config.max_distance * config.max_distance;
    let mut neighbors: Vec<(usize, f64)> = index
        .candidates(coordinates, receiver)
        .into_iter()
        .filter(|&source| config.include_self || source != receiver)
        .filter_map(|source| {
            let squared = (0..index.dimensions)
                .map(|axis| {
                    let delta = coordinates[[receiver, axis]] - coordinates[[source, axis]];
                    delta * delta
                })
                .sum::<f64>();
            (squared <= max_squared).then_some((source, squared))
        })
        .collect();
    if let Some(limit) = config.neighbor_limit() {
        if neighbors.len() > limit {
            neighbors.select_nth_unstable_by(limit, |left, right| left.1.total_cmp(&right.1));
            neighbors.truncate(limit);
        }
    }
    neighbors.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    neighbors
        .into_iter()
        .map(|(source, squared)| {
            (
                source,
                equation.kernel(squared.sqrt()).expect("validated equation") as f32,
            )
        })
        .collect()
}

// Numerical Recipes / Cephes-style approximations; relative accuracy is
// sufficient for f32 kernel storage and avoids a heavyweight special-function dependency.
fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let y = (x / 3.75).powi(2);
        1.0 + y
            * (3.515_622_9
                + y * (3.089_942_4
                    + y * (1.206_749_2 + y * (0.265_973_2 + y * (0.036_076_8 + y * 0.004_581_3)))))
    } else {
        let y = 3.75 / ax;
        ax.exp() / ax.sqrt()
            * (0.398_942_28
                + y * (0.013_285_92
                    + y * (0.002_253_19
                        + y * (-0.001_575_65
                            + y * (0.009_162_81
                                + y * (-0.020_577_06
                                    + y * (0.026_355_37
                                        + y * (-0.016_476_33 + y * 0.003_923_77))))))))
    }
}

fn bessel_k0(x: f64) -> f64 {
    debug_assert!(x > 0.0);
    if x <= 2.0 {
        let y = x * x / 4.0;
        -(x / 2.0).ln() * bessel_i0(x)
            + (-0.577_215_66
                + y * (0.422_784_20
                    + y * (0.230_697_56
                        + y * (0.034_885_90
                            + y * (0.002_626_98 + y * (0.000_107_50 + y * 0.000_007_40))))))
    } else {
        let y = 2.0 / x;
        (-x).exp() / x.sqrt()
            * (1.253_314_14
                + y * (-0.078_323_58
                    + y * (0.021_895_68
                        + y * (-0.010_624_46
                            + y * (0.005_878_72 + y * (-0.002_515_40 + y * 0.000_532_08))))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn equation(dimension: SpatialDimension) -> ReactionDiffusion {
        ReactionDiffusion {
            diffusion: 2.0,
            degradation: 0.5,
            production: 3.0,
            min_distance: 0.1,
            dimension,
        }
    }

    #[test]
    fn analytic_kernels_are_positive_and_decrease() {
        for dimension in [SpatialDimension::TwoD, SpatialDimension::ThreeD] {
            let eq = equation(dimension);
            assert!(eq.kernel(0.5).unwrap() > eq.kernel(2.0).unwrap());
            assert!(eq.kernel(2.0).unwrap() > 0.0);
        }
    }

    #[test]
    fn radius_search_includes_all_sources_by_default() {
        let coordinates = array![[0.0, 0.0], [1.0, 0.0], [1.5, 0.0], [10.0, 0.0]];
        let config = SpatialKernelConfig {
            max_distance: 2.0,
            ..Default::default()
        };
        let kernel = SparseSpatialKernel::build(
            &coordinates,
            None,
            equation(SpatialDimension::TwoD),
            &config,
        )
        .unwrap();
        assert_eq!(kernel.row_offsets[1] - kernel.row_offsets[0], 2);
        assert_eq!(kernel.row_offsets[4] - kernel.row_offsets[3], 0);
    }

    #[test]
    fn neighbor_limit_keeps_nearest_source() {
        let coordinates = array![[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]];
        let config = SpatialKernelConfig {
            max_distance: 3.0,
            max_neighbors: 1,
            ..Default::default()
        };
        let kernel = SparseSpatialKernel::build(
            &coordinates,
            None,
            equation(SpatialDimension::TwoD),
            &config,
        )
        .unwrap();
        assert_eq!(
            &kernel.source_indices[kernel.row_offsets[0]..kernel.row_offsets[1]],
            &[2]
        );
    }

    #[test]
    fn streaming_and_csr_multiplication_agree() {
        let coordinates = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [4.0, 0.0]];
        let expression = array![[1.0f32, 0.0], [0.0, 2.0], [3.0, 1.0], [2.0, 4.0]];
        let config = SpatialKernelConfig {
            max_distance: 2.1,
            target_block_size: 2,
            ..Default::default()
        };
        let eq = equation(SpatialDimension::TwoD);
        let csr = SparseSpatialKernel::build(&coordinates, None, eq, &config)
            .unwrap()
            .multiply(&expression)
            .unwrap();
        let streaming =
            compute_source_fields(&coordinates, &expression, None, eq, &config).unwrap();
        for (left, right) in csr.iter().zip(streaming.iter()) {
            assert!((left - right).abs() < 1e-6);
        }
    }

    #[test]
    fn source_mask_is_respected() {
        let coordinates = array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let expression = array![[1.0f32], [100.0], [3.0]];
        let mask = [true, false, true];
        let config = SpatialKernelConfig {
            max_distance: 3.0,
            include_self: true,
            ..Default::default()
        };
        let field = compute_source_fields(
            &coordinates,
            &expression,
            Some(&mask),
            equation(SpatialDimension::TwoD),
            &config,
        )
        .unwrap();
        assert!(field[[1, 0]] < 10.0);
    }
}
