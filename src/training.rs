use crate::{
    backend::{InferenceBackend, TrainingBackend, backend_name, device},
    config::{CheckpointMetadata, RunConfig},
    data::{ExpressionData, load_and_standardize, read_ids, write_ids, write_matrix},
    model::CellGeneTransformer,
};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::CompactRecorder,
    tensor::{
        Bool, Tensor, TensorData,
        backend::{Backend, BackendTypes},
    },
};
use ndarray::{Array2, Axis, s};
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    io::{self, IsTerminal, Write},
    path::Path,
    time::{Duration, Instant},
};

struct ProgressLine {
    label: String,
    total: usize,
    started: Instant,
    last_draw: Instant,
    enabled: bool,
}

impl ProgressLine {
    fn new(label: impl Into<String>, total: usize) -> Self {
        let now = Instant::now();
        Self {
            label: label.into(),
            total: total.max(1),
            started: now,
            last_draw: now - Duration::from_secs(1),
            enabled: io::stderr().is_terminal(),
        }
    }

    fn draw(&mut self, current: usize, detail: &str) {
        if !self.enabled
            || (current < self.total && self.last_draw.elapsed() < Duration::from_millis(100))
        {
            return;
        }
        self.last_draw = Instant::now();
        let current = current.min(self.total);
        let fraction = current as f64 / self.total as f64;
        let filled = (fraction * 24.0).round() as usize;
        let elapsed = self.started.elapsed().as_secs_f64();
        let eta = if current > 0 {
            elapsed * (self.total - current) as f64 / current as f64
        } else {
            0.0
        };
        eprint!(
            "\r\x1b[2K{} [{}{}] {:>3.0}% {}/{} elapsed={} eta={} {}",
            self.label,
            "#".repeat(filled),
            "-".repeat(24 - filled),
            fraction * 100.0,
            current,
            self.total,
            format_duration(elapsed),
            format_duration(eta),
            detail,
        );
        let _ = io::stderr().flush();
    }

    fn finish(&mut self, detail: &str) {
        self.draw(self.total, detail);
        if self.enabled {
            eprintln!();
        }
    }
}

fn format_duration(seconds: f64) -> String {
    let seconds = seconds.max(0.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

pub fn train_from_config(config: &RunConfig) -> Result<(), String> {
    let total_start = Instant::now();
    let mut effective_config = config.clone();
    let config = &mut effective_config;
    let load_start = Instant::now();
    let data = load_and_standardize(
        &config.expression_matrix,
        &config.cell_ids,
        &config.gene_ids,
    )?;
    println!(
        "stage=data_load seconds={:.3}",
        load_start.elapsed().as_secs_f64()
    );
    fs::create_dir_all(&config.output_dir)
        .map_err(|error| format!("failed to create output directory: {error}"))?;
    let device = device(config.device_index);
    println!(
        "backend={} device_index={}",
        backend_name(),
        config.device_index
    );
    TrainingBackend::seed(&device, config.training.seed);
    let mut model =
        CellGeneTransformer::<TrainingBackend>::new(data.gene_ids.len(), &config.model, &device);
    let mut optimizer = AdamConfig::new().init();
    let batch_size = config.training.batch_size.max(1);
    let sample_ids = config.sample_ids.as_ref().map(read_ids).transpose()?;
    if let Some(sample_ids) = &sample_ids
        && sample_ids.len() != data.cell_ids.len()
    {
        return Err(format!(
            "sample ID count {} does not match cell count {}",
            sample_ids.len(),
            data.cell_ids.len()
        ));
    }
    let sampling = config.training.epoch_sampling.as_str();
    if sampling != "random" && sampling != "stratified" {
        return Err(format!(
            "training.epoch_sampling must be 'random' or 'stratified'; received '{sampling}'"
        ));
    }

    let transformer_start = Instant::now();
    for epoch in 0..config.training.epochs {
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        let order = epoch_cell_order(
            data.cell_ids.len(),
            config.training.seed,
            epoch,
            config.training.shuffle,
            config.training.max_cells_per_epoch,
            sampling,
            sample_ids.as_deref(),
            config.training.minimum_cells_per_sample,
        );
        let total_batches = order.len().div_ceil(batch_size);
        let mut progress = ProgressLine::new(
            format!("train epoch {}/{}", epoch + 1, config.training.epochs),
            total_batches,
        );
        for (batch_index, indices) in order.chunks(batch_size).enumerate() {
            let batch = data.matrix.select(Axis(0), indices);
            let expression = matrix_tensor::<TrainingBackend>(&batch, &device);
            let mask = random_mask(
                indices.len(),
                data.gene_ids.len(),
                config.training.mask_probability,
                config.training.seed + epoch as u64 * 10_000 + batch_index as u64,
                &device,
            );
            let mask_float = mask.clone().float();
            let (prediction, _) = model.forward(expression.clone(), mask);
            let loss = ((prediction - expression).square() * mask_float.clone()).sum()
                / mask_float.sum().add_scalar(1e-8);
            epoch_loss += loss.clone().into_scalar();
            batches += 1;
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(config.training.learning_rate, model, gradients);
            progress.draw(
                batch_index + 1,
                &format!(
                    "cells={} loss={:.6}",
                    ((batch_index + 1) * batch_size).min(order.len()),
                    epoch_loss / batches as f32
                ),
            );
        }
        progress.finish(&format!("loss={:.6}", epoch_loss / batches as f32));
        if epoch % 10 == 0 || epoch + 1 == config.training.epochs {
            println!(
                "epoch={epoch:04} sampled_cells={} masked_reconstruction_mse={:.6}",
                order.len(),
                epoch_loss / batches as f32,
            );
        }
    }
    println!(
        "stage=transformer_training seconds={:.3}",
        transformer_start.elapsed().as_secs_f64()
    );

    let inference_model = model.valid();
    let inference_start = Instant::now();
    let (reconstruction, embeddings) = infer_in_batches(
        &inference_model,
        &data.matrix,
        batch_size,
        config.model.d_model,
        config.training.inference_mask_chunk_size,
        &device,
    );
    println!(
        "stage=target_masked_inference seconds={:.3}",
        inference_start.elapsed().as_secs_f64()
    );
    let residuals = &data.matrix - &reconstruction;
    let save_start = Instant::now();
    save_checkpoint(
        config,
        &data,
        &model,
        &reconstruction,
        &residuals,
        &embeddings,
    )?;
    println!(
        "stage=checkpoint_save seconds={:.3}",
        save_start.elapsed().as_secs_f64()
    );
    println!(
        "stage=total seconds={:.3}",
        total_start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn epoch_cell_order(
    cells: usize,
    seed: u64,
    epoch: usize,
    shuffle: bool,
    max_cells_per_epoch: usize,
    sampling: &str,
    sample_ids: Option<&[String]>,
    minimum_cells_per_sample: usize,
) -> Vec<usize> {
    let limit = if max_cells_per_epoch == 0 {
        cells
    } else {
        max_cells_per_epoch.min(cells)
    };
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(epoch as u64));
    let mut order = (0..cells).collect::<Vec<_>>();
    if limit < cells {
        if sampling == "stratified" {
            if let Some(sample_ids) = sample_ids {
                order =
                    stratified_epoch_sample(sample_ids, limit, minimum_cells_per_sample, &mut rng);
            } else {
                order.shuffle(&mut rng);
                order.truncate(limit);
            }
        } else {
            order.shuffle(&mut rng);
            order.truncate(limit);
        }
    }
    if shuffle {
        order.shuffle(&mut rng);
    } else {
        order.sort_unstable();
    }
    order
}

fn stratified_epoch_sample(
    sample_ids: &[String],
    limit: usize,
    minimum_cells_per_sample: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let mut grouped = BTreeMap::<&str, Vec<usize>>::new();
    for (cell, sample) in sample_ids.iter().enumerate() {
        grouped.entry(sample.as_str()).or_default().push(cell);
    }
    let mut groups = grouped.into_values().collect::<Vec<_>>();
    for cells in &mut groups {
        cells.shuffle(rng);
    }
    groups.shuffle(rng);

    let mut selected = Vec::with_capacity(limit);
    let minimum = minimum_cells_per_sample;
    for layer in 0..minimum {
        for cells in &groups {
            if selected.len() == limit {
                return selected;
            }
            if let Some(cell) = cells.get(layer) {
                selected.push(*cell);
            }
        }
    }

    let mut remaining = groups
        .iter()
        .flat_map(|cells| cells.iter().skip(minimum).copied())
        .collect::<Vec<_>>();
    remaining.shuffle(rng);
    selected.extend(remaining.into_iter().take(limit - selected.len()));
    selected
}

fn infer_in_batches(
    model: &CellGeneTransformer<InferenceBackend>,
    matrix: &Array2<f32>,
    batch_size: usize,
    embedding_size: usize,
    inference_mask_chunk_size: usize,
    device: &<InferenceBackend as BackendTypes>::Device,
) -> (Array2<f32>, Array2<f32>) {
    let mut reconstruction = Array2::<f32>::zeros(matrix.raw_dim());
    let mut embeddings = Array2::<f32>::zeros((matrix.nrows(), embedding_size));
    let mask_chunk_size = inference_mask_chunk_size.max(1);
    let total_batches = matrix.nrows().div_ceil(batch_size);
    let mut progress = ProgressLine::new("target-masked inference", total_batches);
    for start in (0..matrix.nrows()).step_by(batch_size) {
        let end = (start + batch_size).min(matrix.nrows());
        let batch = matrix.slice(s![start..end, ..]).to_owned();
        let expression = matrix_tensor::<InferenceBackend>(&batch, device);
        let no_mask = Tensor::<InferenceBackend, 2, Bool>::from_data(
            TensorData::new(
                vec![false; (end - start) * matrix.ncols()],
                [end - start, matrix.ncols()],
            ),
            device,
        );
        let (_, batch_embeddings) = model.forward(expression.clone(), no_mask);
        embeddings
            .slice_mut(s![start..end, ..])
            .assign(&tensor_matrix(batch_embeddings));
        for target_start in (0..matrix.ncols()).step_by(mask_chunk_size) {
            let target_end = (target_start + mask_chunk_size).min(matrix.ncols());
            let mut mask_values = vec![false; (end - start) * matrix.ncols()];
            for cell in 0..(end - start) {
                for target_gene in target_start..target_end {
                    mask_values[cell * matrix.ncols() + target_gene] = true;
                }
            }
            let target_mask = Tensor::<InferenceBackend, 2, Bool>::from_data(
                TensorData::new(mask_values, [end - start, matrix.ncols()]),
                device,
            );
            let (masked_prediction, _) = model.forward(expression.clone(), target_mask);
            let masked_prediction = tensor_matrix(masked_prediction);
            for target_gene in target_start..target_end {
                for cell in 0..(end - start) {
                    reconstruction[[start + cell, target_gene]] =
                        masked_prediction[[cell, target_gene]];
                }
            }
        }
        let completed = end.div_ceil(batch_size);
        progress.draw(completed, &format!("cells={end}/{}", matrix.nrows()));
    }
    progress.finish("complete");
    (reconstruction, embeddings)
}

fn save_checkpoint(
    config: &RunConfig,
    data: &ExpressionData,
    model: &CellGeneTransformer<TrainingBackend>,
    reconstruction: &Array2<f32>,
    residuals: &Array2<f32>,
    embeddings: &Array2<f32>,
) -> Result<(), String> {
    let root = Path::new(&config.output_dir);
    model
        .clone()
        .save_file(root.join("model"), &CompactRecorder::new())
        .map_err(|error| format!("failed to save Burn model: {error}"))?;
    let metadata = CheckpointMetadata {
        format_version: 1,
        num_cells: data.cell_ids.len(),
        num_genes: data.gene_ids.len(),
        model: config.model.clone(),
        expression_matrix: config.expression_matrix.clone(),
        cell_ids_file: config.cell_ids.clone(),
        gene_ids_file: config.gene_ids.clone(),
        sample_ids_file: config.sample_ids.clone(),
        device_index: config.device_index,
    };
    fs::write(
        root.join("metadata.json"),
        serde_json::to_string_pretty(&metadata).unwrap(),
    )
    .map_err(|error| format!("failed to write metadata: {error}"))?;
    fs::write(
        root.join("normalization.json"),
        serde_json::to_string_pretty(&json!({
            "means": data.means,
            "standard_deviations": data.standard_deviations
        }))
        .unwrap(),
    )
    .map_err(|error| format!("failed to write normalization: {error}"))?;
    write_ids(root.join("cell_ids.txt"), &data.cell_ids)?;
    write_ids(root.join("gene_ids.txt"), &data.gene_ids)?;
    write_matrix(root.join("raw_expression.csv"), &data.raw_matrix)?;
    write_matrix(root.join("standardized_expression.csv"), &data.matrix)?;
    write_matrix(root.join("reconstruction.csv"), reconstruction)?;
    write_matrix(root.join("residuals.csv"), residuals)?;
    write_matrix(root.join("cell_embeddings.csv"), embeddings)?;
    Ok(())
}

fn matrix_tensor<B: burn::tensor::backend::Backend>(
    matrix: &Array2<f32>,
    device: &B::Device,
) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(
            matrix.iter().copied().collect::<Vec<_>>(),
            [matrix.nrows(), matrix.ncols()],
        ),
        device,
    )
}

fn tensor_matrix<B: Backend>(tensor: Tensor<B, 2>) -> Array2<f32> {
    let shape = tensor.dims();
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .expect("tensor should contain f32 values");
    Array2::from_shape_vec((shape[0], shape[1]), values).unwrap()
}

fn random_mask(
    cells: usize,
    genes: usize,
    probability: f32,
    seed: u64,
    device: &<TrainingBackend as BackendTypes>::Device,
) -> Tensor<TrainingBackend, 2, Bool> {
    let mut rng = StdRng::seed_from_u64(seed);
    let values = (0..cells * genes)
        .map(|_| rng.random::<f32>() < probability)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(values, [cells, genes]), device)
}

#[cfg(test)]
mod tests {
    use super::epoch_cell_order;

    #[test]
    fn epoch_shuffle_is_reproducible_and_complete() {
        let first = epoch_cell_order(100, 42, 3, true, 0, "stratified", None, 0);
        let second = epoch_cell_order(100, 42, 3, true, 0, "stratified", None, 0);
        assert_eq!(first, second);
        let mut sorted = first;
        sorted.sort_unstable();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn epoch_shuffle_changes_with_epoch_and_can_be_disabled() {
        assert_ne!(
            epoch_cell_order(100, 42, 3, true, 0, "stratified", None, 0),
            epoch_cell_order(100, 42, 4, true, 0, "stratified", None, 0)
        );
        assert_eq!(
            epoch_cell_order(10, 42, 3, false, 0, "stratified", None, 0),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn epoch_subsample_is_reproducible_and_changes_between_epochs() {
        let first = epoch_cell_order(100, 42, 3, true, 25, "random", None, 0);
        let second = epoch_cell_order(100, 42, 3, true, 25, "random", None, 0);
        let next = epoch_cell_order(100, 42, 4, true, 25, "random", None, 0);
        assert_eq!(first, second);
        assert_ne!(first, next);
        assert_eq!(first.len(), 25);
        let mut unique = first;
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 25);
    }

    #[test]
    fn stratified_subsample_keeps_small_samples_present() {
        let samples = (0..90)
            .map(|_| "large".to_string())
            .chain((0..10).map(|_| "small".to_string()))
            .collect::<Vec<_>>();
        let selected = epoch_cell_order(
            samples.len(),
            42,
            0,
            false,
            20,
            "stratified",
            Some(&samples),
            5,
        );
        assert_eq!(selected.len(), 20);
        assert!(
            selected
                .iter()
                .filter(|&&cell| samples[cell] == "small")
                .count()
                >= 5
        );
    }
}
