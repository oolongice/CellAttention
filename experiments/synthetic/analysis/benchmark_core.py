"""Shared numerical utilities for the maintained Synthetic benchmark."""

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.special import k0

N_CONTEXTS = 12
PCA_COMPONENTS = 12
MAX_DISTANCE = 320.0
LENGTH_SCALE = 130.0
MIN_DISTANCE = 5.0
KMEANS_RESTARTS = 12

METHODS = (
    ("transformer_physical", "transformer", "physical"),
    ("baseline_physical", "baseline", "physical"),
    ("transformer_uniform", "transformer", "uniform"),
    ("baseline_uniform", "baseline", "uniform"),
)


def cluster(matrix: np.ndarray, seed: int) -> np.ndarray:
    """Return the lowest-inertia K-means assignment across fixed restarts."""
    best = None
    for restart in range(KMEANS_RESTARTS):
        centers, labels = kmeans2(
            matrix,
            N_CONTEXTS,
            iter=100,
            minit="++",
            seed=np.random.default_rng(seed * 101 + restart),
        )
        loss = float(np.sum((matrix - centers[labels]) ** 2))
        if best is None or loss < best[0]:
            best = (loss, labels)
    return best[1]


def whiten(embedding: np.ndarray) -> np.ndarray:
    """Apply truncated PCA whitening before Transformer-receiver_group clustering."""
    centered = embedding - embedding.mean(axis=0)
    _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    reference = singular[0] / np.sqrt(len(embedding) - 1)
    scale = np.maximum(
        singular[:PCA_COMPONENTS] / np.sqrt(len(embedding) - 1),
        reference * 1e-5,
    )
    return centered @ vectors[:PCA_COMPONENTS].T / scale


def source_fields(
    coordinates: np.ndarray, expression: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute reaction-diffusion and uniform-radius fields for all genes."""
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(delta * delta, axis=2))
    inside = (distance <= MAX_DISTANCE) & (distance > 0)
    physical_kernel = np.where(
        inside,
        k0(np.maximum(distance, MIN_DISTANCE) / LENGTH_SCALE),
        0.0,
    )
    uniform_kernel = inside.astype(float)
    uniform_kernel /= np.maximum(uniform_kernel.sum(axis=1, keepdims=True), 1.0)
    return physical_kernel @ expression, uniform_kernel @ expression


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Column-standardize a field matrix with a zero-variance safeguard."""
    scale = matrix.std(axis=0)
    return (matrix - matrix.mean(axis=0)) / np.where(scale > 1e-8, scale, 1.0)


def scores(
    labels: np.ndarray,
    outcome: np.ndarray,
    fields: np.ndarray,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute receiver_group target scores and source-to-target association scores."""
    n_genes = outcome.shape[1]
    target = np.zeros((N_CONTEXTS, n_genes))
    source = np.zeros((N_CONTEXTS, n_genes, n_genes))
    for receiver_group in range(N_CONTEXTS):
        selected = labels == receiver_group
        values = outcome[selected]
        residual = (
            values - values.mean(axis=0, keepdims=True)
            if target_mode == "baseline"
            else values
        )
        target[receiver_group] = np.mean(residual * residual, axis=0)
        selected_fields = fields[selected]
        centered_fields = selected_fields - selected_fields.mean(axis=0, keepdims=True)
        centered_outcome = residual - residual.mean(axis=0, keepdims=True)
        covariance = centered_fields.T @ centered_outcome / max(len(values), 1)
        variance = np.mean(centered_fields * centered_fields, axis=0)
        source[receiver_group] = np.abs(covariance) / np.sqrt(
            np.maximum(variance[:, None], 1e-12)
        )
    return target, source
