//! Optional receiver-side masking of already computed spatial source fields.

use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReceiverFieldMaskMode {
    #[default]
    None,
    HardExpressionExclusion,
}

impl ReceiverFieldMaskMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(Self::None),
            "hard_expression_exclusion" | "hard" => Ok(Self::HardExpressionExclusion),
            other => Err(format!(
                "unknown receiver field mask mode {other:?}; expected none or hard_expression_exclusion"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReceiverFieldMaskConfig {
    pub mode: ReceiverFieldMaskMode,
    pub expression_threshold: f32,
}

impl Default for ReceiverFieldMaskConfig {
    fn default() -> Self {
        Self {
            mode: ReceiverFieldMaskMode::None,
            expression_threshold: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ReceiverFieldMaskSummary {
    pub total_entries: usize,
    pub masked_entries: usize,
    pub masked_fraction: f64,
}

/// Applies the receiver mask in place, before any field standardization.
pub fn apply_receiver_field_mask_inplace(
    fields: &mut Array2<f32>,
    receiver_expression: &Array2<f32>,
    config: ReceiverFieldMaskConfig,
) -> Result<ReceiverFieldMaskSummary, String> {
    if fields.raw_dim() != receiver_expression.raw_dim() {
        return Err(format!(
            "field dimensions {:?} != receiver expression dimensions {:?}",
            fields.raw_dim(),
            receiver_expression.raw_dim()
        ));
    }
    if !config.expression_threshold.is_finite() {
        return Err("receiver mask expression threshold must be finite".to_string());
    }
    let total_entries = fields.len();
    if config.mode == ReceiverFieldMaskMode::None {
        return Ok(ReceiverFieldMaskSummary {
            total_entries,
            masked_entries: 0,
            masked_fraction: 0.0,
        });
    }
    let field_values = fields
        .as_slice_mut()
        .ok_or("field matrix must be contiguous for receiver masking")?;
    let expression_values = receiver_expression
        .as_slice()
        .ok_or("receiver expression matrix must be contiguous for receiver masking")?;
    let threshold = config.expression_threshold;
    let masked_entries: usize = field_values
        .par_iter_mut()
        .zip(expression_values.par_iter())
        .map(|(field, expression)| {
            if *expression > threshold {
                *field = 0.0;
                1
            } else {
                0
            }
        })
        .sum();
    Ok(ReceiverFieldMaskSummary {
        total_entries,
        masked_entries,
        masked_fraction: masked_entries as f64 / total_entries.max(1) as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn none_preserves_fields() {
        let mut fields = array![[1.0f32, 2.0], [3.0, 4.0]];
        let expression = array![[1.0f32, 0.0], [0.0, 2.0]];
        let before = fields.clone();
        let summary = apply_receiver_field_mask_inplace(
            &mut fields,
            &expression,
            ReceiverFieldMaskConfig::default(),
        )
        .unwrap();
        assert_eq!(fields, before);
        assert_eq!(summary.masked_entries, 0);
    }

    #[test]
    fn hard_mask_removes_only_entries_above_threshold() {
        let mut fields = array![[1.0f32, 2.0], [3.0, 4.0]];
        let expression = array![[1.0f32, 0.0], [0.5, 2.0]];
        let summary = apply_receiver_field_mask_inplace(
            &mut fields,
            &expression,
            ReceiverFieldMaskConfig {
                mode: ReceiverFieldMaskMode::HardExpressionExclusion,
                expression_threshold: 0.5,
            },
        )
        .unwrap();
        assert_eq!(fields, array![[0.0, 2.0], [3.0, 0.0]]);
        assert_eq!(summary.masked_entries, 2);
        assert_eq!(summary.masked_fraction, 0.5);
    }
}
