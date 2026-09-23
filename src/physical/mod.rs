//! Spatial reaction-diffusion fields, independent of Transformer training.

mod reaction_diffusion;
mod receiver_mask;

pub use reaction_diffusion::{
    ReactionDiffusion, SparseSpatialKernel, SpatialDimension, SpatialKernelConfig,
    compute_source_fields,
};
pub use receiver_mask::{
    ReceiverFieldMaskConfig, ReceiverFieldMaskMode, ReceiverFieldMaskSummary,
    apply_receiver_field_mask_inplace,
};
