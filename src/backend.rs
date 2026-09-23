use burn::backend::Autodiff;

#[cfg(all(feature = "cuda", feature = "metal"))]
compile_error!("features 'cuda' and 'metal' cannot be enabled together");

#[cfg(feature = "cuda")]
pub type InnerBackend = burn::backend::Cuda<f32, i32>;

#[cfg(all(not(feature = "cuda"), feature = "metal"))]
pub type InnerBackend = burn::backend::Metal<f32, i32>;

#[cfg(not(any(feature = "cuda", feature = "metal")))]
pub type InnerBackend = burn::backend::NdArray<f32>;

pub type TrainingBackend = Autodiff<InnerBackend>;
pub type InferenceBackend = InnerBackend;

pub fn device(index: usize) -> <InnerBackend as burn::tensor::backend::BackendTypes>::Device {
    #[cfg(feature = "cuda")]
    {
        burn::backend::cuda::CudaDevice { index }
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        let _ = index;
        Default::default()
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        let _ = index;
        Default::default()
    }
}

pub fn backend_name() -> &'static str {
    #[cfg(feature = "cuda")]
    {
        "cuda"
    }
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    {
        "metal"
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        "ndarray"
    }
}
