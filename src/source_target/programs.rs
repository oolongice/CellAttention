use ndarray::Array2;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SourceProgram {
    pub program_index: usize,
    pub source_indices: Vec<usize>,
}

/// Connected components of source fields with absolute Pearson correlation at least
/// `threshold`. Programs record statistical non-identifiability, not biological identity.
pub fn find_correlated_source_programs(
    fields: &Array2<f64>,
    threshold: f64,
) -> Result<Vec<SourceProgram>, String> {
    if fields.nrows() < 2 || fields.ncols() == 0 {
        return Err("fields must contain at least two cells and one source".to_string());
    }
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err("correlation threshold must be in [0, 1]".to_string());
    }
    let sources = fields.ncols();
    let means: Vec<f64> = (0..sources)
        .map(|source| fields.column(source).sum() / fields.nrows() as f64)
        .collect();
    let norms: Vec<f64> = (0..sources)
        .map(|source| {
            fields
                .column(source)
                .iter()
                .map(|value| (value - means[source]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    let mut parent: Vec<usize> = (0..sources).collect();
    for left in 0..sources {
        if norms[left] <= 1e-12 {
            continue;
        }
        for right in 0..left {
            if norms[right] <= 1e-12 {
                continue;
            }
            let numerator = (0..fields.nrows())
                .map(|cell| {
                    (fields[[cell, left]] - means[left]) * (fields[[cell, right]] - means[right])
                })
                .sum::<f64>();
            let correlation = numerator / (norms[left] * norms[right]);
            if correlation.abs() >= threshold {
                union(&mut parent, left, right);
            }
        }
    }
    let mut grouped = std::collections::BTreeMap::<usize, Vec<usize>>::new();
    for source in 0..sources {
        let root = find(&mut parent, source);
        grouped.entry(root).or_default().push(source);
    }
    let mut groups: Vec<Vec<usize>> = grouped.into_values().collect();
    groups.sort_by(|left, right| {
        right
            .len()
            .cmp(&left.len())
            .then_with(|| left[0].cmp(&right[0]))
    });
    Ok(groups
        .into_iter()
        .enumerate()
        .map(|(program_index, source_indices)| SourceProgram {
            program_index,
            source_indices,
        })
        .collect())
}
fn find(parent: &mut [usize], value: usize) -> usize {
    if parent[value] != value {
        parent[value] = find(parent, parent[value]);
    }
    parent[value]
}
fn union(parent: &mut [usize], left: usize, right: usize) {
    let a = find(parent, left);
    let b = find(parent, right);
    if a != b {
        parent[b] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn perfectly_correlated_fields_share_program() {
        let fields = array![[1., 2., 0.], [2., 4., 1.], [3., 6., 0.], [4., 8., 1.]];
        let programs = find_correlated_source_programs(&fields, 0.99).unwrap();
        assert!(
            programs
                .iter()
                .any(|program| program.source_indices == vec![0, 1])
        );
    }
}
