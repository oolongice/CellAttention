//! Identifiable source-target effects with fixed receiver receiver_groups.
//!
//! This module is deliberately independent from Transformer training and spatial-field
//! construction. Callers provide frozen residuals, source fields, and fixed receiver_group
//! probabilities. Receiver group reassignment is not implemented.

mod effects;
mod programs;
mod receiver_groups;

pub use effects::{
    EffectRecord, FixedReceiverGroupEffectModel, SparseEffectConfig,
    fit_fixed_receiver_group_effects,
};
pub use programs::{SourceProgram, find_correlated_source_programs};
pub use receiver_groups::{
    EmbeddingReceiverGroupModel, ReceiverGroupClusteringMethod, ReceiverGroupCountCandidate,
    ReceiverGroupCountConfig, ReceiverGroupCountMode, ReceiverGroupCountSelection,
    fit_embedding_receiver_groups, select_receiver_group_count,
};
