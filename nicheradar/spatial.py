"""Spatial graph, smoothing, and module-pattern utilities."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def robust_scale(values: np.ndarray, clip: float = 8.0, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median[None, :]), axis=0)
    scaled = (values - median[None, :]) / (mad[None, :] + eps)
    if clip is not None and clip > 0:
        scaled = np.clip(scaled, -clip, clip)
    return np.nan_to_num(scaled).astype(np.float32)


def spatial_knn_graph(coords: np.ndarray, k: int) -> sparse.csr_matrix:
    rows, cols, dist = knn_edges(coords, k)
    sigma = robust_sigma(dist)
    weights = np.exp(-(dist**2) / (sigma**2 + 1e-8))
    return symmetric_graph(rows, cols, weights, coords.shape[0])


def knn_edges(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_obs = values.shape[0]
    n_neighbors = min(max(int(k) + 1, 2), n_obs)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(values)
    dist, ind = nbrs.kneighbors(values)
    dist = dist[:, 1:]
    ind = ind[:, 1:]
    rows = np.repeat(np.arange(n_obs), ind.shape[1])
    cols = ind.reshape(-1)
    return rows.astype(int), cols.astype(int), dist.reshape(-1).astype(np.float64)


def symmetric_graph(
    rows: np.ndarray, cols: np.ndarray, weights: np.ndarray, n_obs: int
) -> sparse.csr_matrix:
    graph = sparse.csr_matrix((weights, (rows, cols)), shape=(n_obs, n_obs))
    graph = graph.maximum(graph.T)
    return row_normalize(graph)


def row_normalize(graph: sparse.csr_matrix) -> sparse.csr_matrix:
    graph = graph.tocsr()
    row_sum = np.asarray(graph.sum(axis=1)).ravel()
    inv = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    return sparse.diags(inv) @ graph


def smooth_features(
    features: np.ndarray,
    graph: sparse.csr_matrix,
    steps: int,
    weight: float,
) -> np.ndarray:
    smoothed = np.asarray(features, dtype=np.float64)
    weight = float(np.clip(weight, 0.0, 1.0))
    for _ in range(max(int(steps), 0)):
        smoothed = (1.0 - weight) * smoothed + weight * (graph @ smoothed)
    return robust_scale(smoothed)


def smooth_score_matrix(
    values: np.ndarray,
    graph: sparse.csr_matrix,
    steps: int,
    weight: float,
) -> np.ndarray:
    smoothed = np.asarray(values, dtype=np.float64)
    weight = float(np.clip(weight, 0.0, 1.0))
    for _ in range(max(int(steps), 0)):
        smoothed = (1.0 - weight) * smoothed + weight * (graph @ smoothed)
    return np.nan_to_num(smoothed)


def quantile_normalize_columns(
    values: np.ndarray, q_low: float, q_high: float
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not 0 <= q_low < q_high <= 100:
        raise ValueError("score quantiles must satisfy 0 <= q_low < q_high <= 100.")
    lo = np.nanpercentile(values, q_low, axis=0)
    hi = np.nanpercentile(values, q_high, axis=0)
    scale = np.maximum(hi - lo, 1e-8)
    normalized = (values - lo[None, :]) / scale[None, :]
    return np.clip(np.nan_to_num(normalized), 0.0, 1.0)


def morans_i(values: np.ndarray, coords: np.ndarray, spatial_k: int = 12) -> float:
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.mean(values)
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return 0.0
    graph = spatial_knn_graph(coords, k=spatial_k)
    s0 = float(graph.sum())
    if s0 <= 0:
        return 0.0
    return float(values.size / s0 * centered.dot(graph @ centered) / denom)


def module_similarity(activity: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(np.asarray(activity, dtype=np.float64), rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def mean_upper(matrix: np.ndarray, empty_value: float = 1.0) -> float:
    if matrix.shape[0] < 2:
        return float(empty_value)
    idx = np.triu_indices(matrix.shape[0], k=1)
    return float(np.mean(matrix[idx]))


def robust_sigma(values: np.ndarray) -> float:
    positive = np.asarray(values, dtype=np.float64)
    positive = positive[positive > 0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive) + 1e-8)
