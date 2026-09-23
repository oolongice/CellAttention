use crate::config::ModelConfig;
use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    tensor::{Bool, Int, Tensor, activation, backend::Backend},
};

#[derive(Module, Debug)]
pub struct CellGeneTransformer<B: Backend> {
    gene_embedding: Embedding<B>,
    mask_embedding: Embedding<B>,
    expression_encoder: Linear<B>,
    transformer: TransformerEncoder<B>,
    pool_score: Linear<B>,
    reconstruction_head: Linear<B>,
}

impl<B: Backend> CellGeneTransformer<B> {
    pub fn new(num_genes: usize, config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            gene_embedding: EmbeddingConfig::new(num_genes, config.d_model).init(device),
            mask_embedding: EmbeddingConfig::new(2, config.d_model).init(device),
            expression_encoder: LinearConfig::new(1, config.d_model).init(device),
            transformer: TransformerEncoderConfig::new(
                config.d_model,
                config.d_ff,
                config.n_heads,
                config.n_layers,
            )
            .with_dropout(config.dropout)
            .with_norm_first(true)
            .init(device),
            pool_score: LinearConfig::new(config.d_model, 1).init(device),
            reconstruction_head: LinearConfig::new(config.d_model * 2, 1).init(device),
        }
    }

    pub fn forward(
        &self,
        expression: Tensor<B, 2>,
        mask: Tensor<B, 2, Bool>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [num_cells, num_genes] = expression.dims();
        let ids: Vec<i64> = (0..num_cells)
            .flat_map(|_| (0..num_genes).map(|gene| gene as i64))
            .collect();
        let gene_ids = Tensor::<B, 2, Int>::from_data(
            burn::tensor::TensorData::new(ids, [num_cells, num_genes]),
            &expression.device(),
        );
        let mask_float = mask.clone().float();
        let visible =
            expression * (Tensor::ones([num_cells, num_genes], &mask_float.device()) - mask_float);
        let tokens = self.expression_encoder.forward(visible.unsqueeze_dim(2))
            + self.gene_embedding.forward(gene_ids)
            + self.mask_embedding.forward(mask.int());
        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(tokens));
        let weights = activation::softmax(self.pool_score.forward(encoded.clone()), 1);
        let cell_embedding = (weights * encoded.clone())
            .sum_dim(1)
            .reshape([num_cells, encoded.dims()[2]]);
        let repeated = cell_embedding
            .clone()
            .unsqueeze_dim(1)
            .repeat_dim(1, num_genes);
        let reconstruction = self
            .reconstruction_head
            .forward(Tensor::cat(vec![encoded, repeated], 2))
            .reshape([num_cells, num_genes]);
        (reconstruction, cell_embedding)
    }
}
