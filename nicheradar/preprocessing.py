"""AnnData input handling and spatial probability construction."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse


TransformName = Literal["identity", "log1p", "sqrt"]


def get_spatial_coordinates(adata) -> np.ndarray:
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
    elif {"x", "y"}.issubset(adata.obs.columns):
        coords = adata.obs[["x", "y"]].to_numpy(dtype=float)
    else:
        raise ValueError("Missing spatial coordinates: need obsm['spatial'] or obs['x','y'].")
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must be an n_spots x 2 array.")
    return coords[:, :2]


def standardize_coordinates(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    scale = coords.std(axis=0)
    scale[scale == 0] = 1.0
    return (coords - coords.mean(axis=0)) / scale


def get_expression_matrix(adata, layer: str | None = None, use_raw: bool = False):
    if layer is not None and use_raw:
        raise ValueError("Use either layer or use_raw, not both.")
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"AnnData layer not found: {layer}")
        return adata.layers[layer], f"adata.layers['{layer}']"
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True but adata.raw is empty.")
        return adata.raw.X, "adata.raw.X"
    return adata.X, "adata.X"


def get_var_names(adata, use_raw: bool) -> np.ndarray:
    if use_raw:
        return np.asarray(adata.raw.var_names)
    return np.asarray(adata.var_names)


def select_expression_matrix(
    X_input,
    var_names,
    expression_source: str,
    transform: TransformName,
    min_total_expression: float | None,
    min_mean_expression: float | None,
    min_detected_spots: int | None,
    min_detected_fraction: float | None,
    min_variance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    X = transform_expression(to_dense_float(X_input), transform)
    totals = np.asarray(X.sum(axis=0)).ravel()
    means = np.asarray(X.mean(axis=0)).ravel()
    variances = np.asarray(X.var(axis=0)).ravel()
    detected_counts = np.asarray((X > 0).sum(axis=0)).ravel()

    # Positive mass is required to convert each gene into a probability distribution.
    keep = totals > 0
    if min_total_expression is not None:
        keep &= totals > float(min_total_expression)
    if min_mean_expression is not None:
        keep &= means >= float(min_mean_expression)
    min_detected_values = []
    if min_detected_spots is not None:
        min_detected_values.append(int(min_detected_spots))
    if min_detected_fraction is not None:
        min_detected_values.append(int(np.ceil(float(min_detected_fraction) * X.shape[0])))
    min_detected = max(min_detected_values) if min_detected_values else None
    if min_detected is not None:
        keep &= detected_counts >= min_detected
    if min_variance is not None:
        keep &= variances >= float(min_variance)

    indices = np.flatnonzero(keep)
    if indices.size == 0:
        raise ValueError("No genes passed basic expression filters.")

    stats = {
        "expression_source": expression_source,
        "n_input_genes": int(X.shape[1]),
        "n_positive_expression_genes": int(np.sum(totals > 0)),
        "n_pass_basic_filter": int(indices.size),
        "min_detected_spots_effective": _optional_int(min_detected),
        "min_total_expression": _optional_float(min_total_expression),
        "min_mean_expression": _optional_float(min_mean_expression),
        "min_detected_spots": _optional_int(min_detected_spots),
        "min_detected_fraction": _optional_float(min_detected_fraction),
        "min_variance": _optional_float(min_variance),
    }
    return X[:, indices], np.asarray(var_names)[indices], indices, stats


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_int(value) -> int | None:
    return None if value is None else int(value)


def to_dense_float(X) -> np.ndarray:
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    X[X < 0] = 0.0
    return X


def transform_expression(X: np.ndarray, transform: TransformName) -> np.ndarray:
    if transform == "identity":
        return X
    if transform == "log1p":
        return np.log1p(X)
    if transform == "sqrt":
        return np.sqrt(X)
    raise ValueError(f"Unknown expression transform: {transform}")


def gene_probability_matrix(X: np.ndarray) -> np.ndarray:
    mass = X.sum(axis=0, keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("Selected genes must have positive total expression.")
    return (X / mass).T.astype(np.float32, copy=False)


def mean_gene_probability_background(gene_probabilities: np.ndarray) -> np.ndarray:
    """Average gene-wise spatial probability distributions into a background."""

    probabilities = np.asarray(gene_probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("gene_probabilities must be a genes x spots matrix.")
    background = probabilities.mean(axis=0)
    total = background.sum()
    if total <= 0:
        return np.full(
            probabilities.shape[1],
            1.0 / probabilities.shape[1],
            dtype=np.float32,
        )
    return (background / total).astype(np.float32)


def background_probability(X: np.ndarray) -> np.ndarray:
    return mean_gene_probability_background(gene_probability_matrix(X))


def probability_concentration(probability: np.ndarray) -> float:
    p = np.asarray(probability, dtype=float)
    p = p / (p.sum() + 1e-12)
    if p.size <= 1:
        return 0.0
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.clip(1.0 - entropy / np.log(p.size), 0.0, 1.0))
