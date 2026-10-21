"""Exact energy-MMD distance computations."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm


def energy_distance_to_background(
    probabilities: np.ndarray,
    background: np.ndarray,
    coords: np.ndarray,
    chunk_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute exact sqrt energy-MMD from every gene distribution to background."""

    chunk_size = max(1, int(chunk_size))
    n_genes, n_spots = probabilities.shape
    probs_t = probabilities.T.astype(np.float32, copy=False)
    background = background.astype(np.float32, copy=False)
    gene_self = np.zeros(n_genes, dtype=np.float64)
    gene_bg = np.zeros(n_genes, dtype=np.float64)
    bg_self = 0.0

    # Distance blocks are generated on demand, so the result stays exact without
    # materializing the full n_spots x n_spots distance matrix.
    starts = range(0, n_spots, chunk_size)
    for start in tqdm(
        starts,
        desc="Energy-MMD to background",
        disable=not show_progress,
        leave=False,
    ):
        end = min(start + chunk_size, n_spots)
        block = cdist(coords[start:end], coords, metric="euclidean").astype(np.float32)
        weighted_genes = block @ probs_t
        weighted_bg = block @ background
        gene_self += np.sum(
            probabilities[:, start:end].astype(np.float64) * weighted_genes.T.astype(np.float64),
            axis=1,
        )
        gene_bg += probabilities[:, start:end].astype(np.float64) @ weighted_bg.astype(np.float64)
        bg_self += float(background[start:end].astype(np.float64) @ weighted_bg.astype(np.float64))

    mmd2 = gene_bg - 0.5 * gene_self - 0.5 * bg_self
    return np.sqrt(np.maximum(mmd2, 0.0)).astype(np.float32)


def energy_mmd2(
    distributions: np.ndarray,
    coords: np.ndarray,
    chunk_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    """Compute exact pairwise squared energy-MMD between gene distributions."""

    dist_expect = weighted_distance_expectations(
        distributions=distributions,
        coords=coords,
        chunk_size=chunk_size,
        show_progress=show_progress,
    )
    self_expect = np.diag(dist_expect)
    mmd2 = dist_expect - 0.5 * self_expect[:, None] - 0.5 * self_expect[None, :]
    return np.maximum((mmd2 + mmd2.T) / 2.0, 0.0).astype(np.float32)


def weighted_distance_expectations(
    distributions: np.ndarray,
    coords: np.ndarray,
    chunk_size: int,
    show_progress: bool = True,
) -> np.ndarray:
    chunk_size = max(1, int(chunk_size))
    n_distributions, n_spots = distributions.shape
    expectations = np.zeros((n_distributions, n_distributions), dtype=np.float64)
    probs_t = distributions.T.astype(np.float32, copy=False)

    starts = range(0, n_spots, chunk_size)
    for start in tqdm(
        starts,
        desc="Gene-gene Energy-MMD",
        disable=not show_progress,
        leave=False,
    ):
        end = min(start + chunk_size, n_spots)
        block = cdist(coords[start:end], coords, metric="euclidean").astype(np.float32)
        weighted = block @ probs_t
        expectations += distributions[:, start:end].astype(np.float64) @ weighted.astype(
            np.float64
        )
    return expectations
