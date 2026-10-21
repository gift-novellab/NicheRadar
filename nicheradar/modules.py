"""SVG selection, gene embedding, and module-level spatial scores."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .preprocessing import probability_concentration
from .spatial import mean_upper


LOGGER = logging.getLogger(__name__)
_EARLY_STOP_REL_TOL = 1e-5
_SPECTRAL_COSINE_TARGET_MEAN_MAX = 0.22
_SPECTRAL_HYPEREDGE_INIT_SCALE = 1.1


def select_svg_mask(
    svg_distance: np.ndarray,
    svg_zscore: np.ndarray,
    n_svgs: int | None,
    zscore_threshold: float | None,
    quantile: float | None,
    min_distance: float | None,
) -> np.ndarray:
    keep = np.ones(svg_distance.size, dtype=bool)
    if zscore_threshold is not None:
        keep &= svg_zscore >= float(zscore_threshold)
    if quantile is not None:
        q = float(np.clip(quantile, 0.0, 1.0))
        keep &= svg_distance >= np.quantile(svg_distance, q)
    if min_distance is not None:
        keep &= svg_distance >= float(min_distance)

    if n_svgs is not None:
        selected_pool = np.flatnonzero(keep)
        if selected_pool.size > n_svgs:
            order = np.argsort(svg_distance[selected_pool])[::-1][: int(n_svgs)]
            new_keep = np.zeros(svg_distance.size, dtype=bool)
            new_keep[selected_pool[order]] = True
            keep = new_keep
        elif selected_pool.size == 0:
            top = np.argsort(svg_distance)[::-1][: int(n_svgs)]
            keep[top] = True
    return keep


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.std(values))
    if scale == 0 or not np.isfinite(scale):
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(np.mean(values))) / scale).astype(np.float32)


def learn_soft_hyperedge_modules(
    gene_distance: np.ndarray,
    n_modules: int,
    random_state: int,
    local_k: int = 15,
    n_iter: int = 200,
    learning_rate: float = 1e-2,
    diversity_weight: float = 1e-3,
    device: str = "auto",
    gene_mmd2: np.ndarray | None = None,
    enable_unassigned_hyperedge: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Infer gene modules by reconstructing Energy-MMD-derived similarity.

    The exact Energy-MMD distance matrix is converted into a local-scale
    similarity target S. A row-normalized soft incidence matrix H and a learned
    hyperedge embedding U are optimized so that (H @ U) @ (H @ U).T
    reconstructs S under a weighted squared loss. A small spatial MMD
    diversity penalty discourages different hyperedge distributions from
    becoming too similar.
    """

    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for soft hyperedge module learning. "
            "Please run this method in the NicheRadar conda environment."
        ) from exc

    distance = np.asarray(gene_distance, dtype=np.float64)
    if distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("gene_distance must be a square matrix.")
    diversity_mmd2 = prepare_gene_mmd2_for_hyperedge_diversity(gene_mmd2, distance)
    diversity_scale = hyperedge_mmd_similarity_scale(diversity_mmd2)
    n_genes = distance.shape[0]
    n_modules = int(np.clip(n_modules, 2, n_genes))
    n_hyperedges = n_modules + int(bool(enable_unassigned_hyperedge))
    unassigned_index = n_modules if enable_unassigned_hyperedge else None
    n_iter = max(0, int(n_iter))
    diversity_weight = max(0.0, float(diversity_weight))

    target_similarity = local_scaled_similarity(distance, local_k=local_k)
    pair_weights = similarity_reconstruction_weights(
        distance=distance,
        similarity=target_similarity,
        local_k=local_k,
    )
    init_real_logits, init_hyperedge_embedding, init_representatives = (
        spectral_cosine_hyperedge_initialization(
            target_similarity=target_similarity,
            n_modules=n_modules,
            random_state=random_state,
        )
    )
    init_logits_unscaled = (
        append_unassigned_logits(init_real_logits)
        if enable_unassigned_hyperedge
        else init_real_logits
    )
    init_logits, init_logit_scale = calibrate_initial_logits(init_logits_unscaled)
    init_membership = softmax_numpy(init_logits)
    init_real_membership = init_membership[:, :n_modules]
    init_reconstruction_loss = weighted_reconstruction_loss_numpy(
        target_similarity,
        pair_weights,
        init_real_membership,
        init_hyperedge_embedding,
    )
    init_labels = hard_labels_with_unassigned(
        init_membership,
        n_modules=n_modules,
        unassigned_index=unassigned_index,
    )
    init_labels = enforce_nonempty_labels(
        init_labels,
        init_membership[:, :n_modules],
        n_modules,
    )

    torch.manual_seed(int(random_state))
    run_device = resolve_torch_device(device, torch)
    target_t = torch.as_tensor(target_similarity, dtype=torch.float32, device=run_device)
    weight_t = torch.as_tensor(pair_weights, dtype=torch.float32, device=run_device)
    gene_mmd2_t = torch.as_tensor(diversity_mmd2, dtype=torch.float32, device=run_device)
    weight_sum = torch.clamp(weight_t.sum(), min=1.0)
    min_iter = min(n_iter, max(100, min(300, n_iter // 3)))
    patience = min(40, max(10, n_iter // 25))
    LOGGER.info(
        "Soft hyperedge: genes=%d modules=%d hyperedges=%d local_k=%d max_iter=%d min_iter=%d patience=%d diversity_weight=%.3g unassigned=%s device=%s.",
        n_genes,
        n_modules,
        n_hyperedges,
        int(local_k),
        n_iter,
        min_iter,
        patience,
        diversity_weight,
        bool(enable_unassigned_hyperedge),
        str(run_device),
    )
    LOGGER.info(
        "Soft hyperedge init: mode=spectral_cosine target_mean_max=%.3f logit_scale=%.3f embedding_scale=%.2f reconstruction_loss=%.6f mean_max_membership=%.3f.",
        initial_assignment_target_mean_max(n_hyperedges),
        init_logit_scale,
        _SPECTRAL_HYPEREDGE_INIT_SCALE,
        init_reconstruction_loss,
        float(np.max(init_membership, axis=1).mean()),
    )

    # Softmax keeps H row-normalized. U lets modules have learned similarities,
    # so reconstruction follows sim = H_real U U.T H_real.T. If enabled, the
    # final H column is an unassigned sink and is excluded from Z.
    logits = torch.nn.Parameter(
        torch.as_tensor(init_logits, dtype=torch.float32, device=run_device)
    )
    hyperedge_embedding = torch.nn.Parameter(
        torch.as_tensor(init_hyperedge_embedding, dtype=torch.float32, device=run_device)
    )
    optimizer = torch.optim.Adam([logits, hyperedge_embedding], lr=float(learning_rate))

    losses: list[float] = []
    previous = np.inf
    stable_steps = 0
    log_every = max(1, min(50, n_iter // 5 if n_iter >= 5 else 1))
    for iteration in range(1, n_iter + 1):
        optimizer.zero_grad()
        membership_t = F.softmax(logits, dim=1)
        real_membership_t = membership_t[:, :n_modules]
        gene_embedding_t = real_membership_t @ hyperedge_embedding
        reconstructed_t = gene_embedding_t @ gene_embedding_t.T
        reconstruction_loss = (weight_t * (target_t - reconstructed_t).pow(2)).sum() / weight_sum
        diversity_loss, _ = hyperedge_spatial_mmd_diversity_loss(
            membership=real_membership_t,
            gene_mmd2=gene_mmd2_t,
            scale=diversity_scale,
        )
        loss = reconstruction_loss + diversity_weight * diversity_loss
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if iteration == 1 or iteration % log_every == 0 or iteration == n_iter:
            LOGGER.info(
                "Soft hyperedge: iter=%d loss=%.6f reconstruction_loss=%.6f spatial_diversity_loss=%.6f.",
                iteration,
                loss_value,
                float(reconstruction_loss.detach().cpu()),
                float(diversity_loss.detach().cpu()),
            )
        if np.isfinite(previous):
            threshold = _EARLY_STOP_REL_TOL * max(1.0, abs(previous))
            if abs(previous - loss_value) <= threshold:
                stable_steps += 1
            else:
                stable_steps = 0
            if iteration >= min_iter and stable_steps >= patience:
                LOGGER.info(
                    "Soft hyperedge: early stopped at iter=%d loss=%.6f stable_steps=%d threshold=%.3g.",
                    iteration,
                    loss_value,
                    stable_steps,
                    threshold,
                )
                break
        previous = loss_value

    with torch.no_grad():
        membership = F.softmax(logits, dim=1)
        real_membership = membership[:, :n_modules]
        gene_embedding = real_membership @ hyperedge_embedding
        reconstructed = gene_embedding @ gene_embedding.T
        final_reconstruction_loss = (
            weight_t * (target_t - reconstructed).pow(2)
        ).sum() / weight_sum
        final_diversity_loss, final_hyperedge_mmd2_t = hyperedge_spatial_mmd_diversity_loss(
            membership=real_membership,
            gene_mmd2=gene_mmd2_t,
            scale=diversity_scale,
        )
        final_loss = final_reconstruction_loss + diversity_weight * final_diversity_loss

    membership_np = membership.detach().cpu().numpy().astype(np.float32)
    real_hyperedge_embedding_np = hyperedge_embedding.detach().cpu().numpy().astype(np.float32)
    hyperedge_embedding_np = append_unassigned_hyperedge_embedding(
        real_hyperedge_embedding_np
    ) if enable_unassigned_hyperedge else real_hyperedge_embedding_np
    gene_embedding_np = gene_embedding.detach().cpu().numpy().astype(np.float32)
    reconstruction_np = reconstructed.detach().cpu().numpy().astype(np.float32)
    labels = hard_labels_with_unassigned(membership_np, n_modules, unassigned_index)
    labels = enforce_nonempty_labels(labels, membership_np[:, :n_modules], n_modules)
    final_loss_value = float(final_loss.detach().cpu())
    final_reconstruction_value = float(final_reconstruction_loss.detach().cpu())
    final_diversity_value = float(final_diversity_loss.detach().cpu())
    final_hyperedge_mmd2 = final_hyperedge_mmd2_t.detach().cpu().numpy().astype(np.float32)
    final_offdiag_mmd2 = final_hyperedge_mmd2[~np.eye(n_modules, dtype=bool)]
    final_offdiag_similarity = np.exp(
        -final_offdiag_mmd2 / max(diversity_scale, 1e-12)
    )
    info = {
        "hyperedge_loss": final_loss_value,
        "hyperedge_reconstruction_loss": final_reconstruction_value,
        "hyperedge_initial_reconstruction_loss": float(init_reconstruction_loss),
        "hyperedge_reconstruction_improvement": float(
            init_reconstruction_loss - final_reconstruction_value
        ),
        "hyperedge_diversity_loss": final_diversity_value,
        "hyperedge_spatial_diversity_loss": final_diversity_value,
        "hyperedge_diversity_weight": diversity_weight,
        "hyperedge_mean_max_membership": float(np.max(membership_np, axis=1).mean()),
        "hyperedge_initial_mean_max_membership": float(
            np.max(init_membership, axis=1).mean()
        ),
        "hyperedge_label_change_fraction_from_initial": float(
            np.mean(labels != init_labels)
        ),
        "hyperedge_initialization": "spectral_cosine",
        "hyperedge_initial_target_mean_max_membership": float(
            initial_assignment_target_mean_max(n_hyperedges)
        ),
        "hyperedge_initial_logit_scale": float(init_logit_scale),
        "hyperedge_initial_embedding_scale": float(_SPECTRAL_HYPEREDGE_INIT_SCALE),
        "hyperedge_initial_representatives": [int(idx) for idx in init_representatives],
        "hyperedge_total_count": int(n_hyperedges),
        "hyperedge_unassigned_enabled": bool(enable_unassigned_hyperedge),
        "hyperedge_unassigned_index": -1 if unassigned_index is None else int(unassigned_index),
        "hyperedge_initial_unassigned_gene_count": int(np.sum(init_labels < 0)),
        "hyperedge_unassigned_gene_count": int(np.sum(labels < 0)),
        "hyperedge_unassigned_gene_fraction": float(np.mean(labels < 0)),
        "hyperedge_real_assignment_mass_mean": float(np.sum(membership_np[:, :n_modules], axis=1).mean()),
        "hyperedge_unassigned_mass_mean": (
            0.0 if unassigned_index is None else float(membership_np[:, unassigned_index].mean())
        ),
        "hyperedge_mmd_similarity_scale": float(diversity_scale),
        "hyperedge_mean_offdiag_mmd2": float(np.mean(final_offdiag_mmd2)),
        "hyperedge_median_offdiag_mmd2": float(np.median(final_offdiag_mmd2)),
        "hyperedge_min_offdiag_mmd2": float(np.min(final_offdiag_mmd2)),
        "hyperedge_mean_offdiag_mmd_similarity": float(np.mean(final_offdiag_similarity)),
        "hyperedge_n_iter": len(losses),
        "hyperedge_local_k": int(local_k),
        "hyperedge_weighted_pair_fraction": float(np.mean(pair_weights > 0.05)),
        "hyperedge_target_similarity_mean": float(np.mean(target_similarity)),
        "hyperedge_target_similarity_max": float(np.max(target_similarity)),
        "hyperedge_learning_rate": float(learning_rate),
        "hyperedge_device": str(run_device),
        "hyperedge_embedding_dim": int(hyperedge_embedding_np.shape[1]),
        "hyperedge_real_count": int(n_modules),
    }
    LOGGER.info(
        "Soft hyperedge: finished iter=%d final_loss=%.6f final_reconstruction=%.6f final_spatial_diversity=%.6f mean_max_membership=%.3f unassigned_genes=%d/%d (%.2f%%).",
        len(losses),
        final_loss_value,
        info["hyperedge_reconstruction_loss"],
        info["hyperedge_diversity_loss"],
        info["hyperedge_mean_max_membership"],
        info["hyperedge_unassigned_gene_count"],
        n_genes,
        100.0 * info["hyperedge_unassigned_gene_fraction"],
    )
    return (
        labels.astype(int),
        membership_np,
        hyperedge_embedding_np,
        gene_embedding_np,
        reconstruction_np,
        info,
    )


def local_scaled_similarity(distance: np.ndarray, local_k: int) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    n = distance.shape[0]
    k = int(np.clip(local_k, 1, max(1, n - 1)))
    masked = distance.copy()
    np.fill_diagonal(masked, np.inf)
    sigma = np.partition(masked, kth=k - 1, axis=1)[:, k - 1]
    finite = sigma[np.isfinite(sigma) & (sigma > 0)]
    fallback = float(np.median(finite)) if finite.size else 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    scale = np.maximum(np.outer(sigma, sigma), 1e-12)
    similarity = np.exp(-(distance**2) / scale)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def prepare_gene_mmd2_for_hyperedge_diversity(
    gene_mmd2: np.ndarray | None,
    distance: np.ndarray,
) -> np.ndarray:
    if gene_mmd2 is None:
        mmd2 = np.asarray(distance, dtype=np.float64) ** 2
    else:
        mmd2 = np.asarray(gene_mmd2, dtype=np.float64)
        if mmd2.shape != distance.shape:
            raise ValueError("gene_mmd2 must have the same shape as gene_distance.")
    mmd2 = 0.5 * (mmd2 + mmd2.T)
    mmd2 = np.maximum(mmd2, 0.0)
    np.fill_diagonal(mmd2, 0.0)
    return mmd2.astype(np.float32)


def hyperedge_mmd_similarity_scale(gene_mmd2: np.ndarray) -> float:
    mmd2 = np.asarray(gene_mmd2, dtype=np.float64)
    off_diagonal = mmd2[~np.eye(mmd2.shape[0], dtype=bool)]
    positive = off_diagonal[np.isfinite(off_diagonal) & (off_diagonal > 1e-12)]
    if positive.size == 0:
        return 1.0
    return float(max(np.median(positive), 1e-12))


def hyperedge_gene_distribution_weights(membership: np.ndarray) -> np.ndarray:
    """Return K x N weights that mix gene spatial distributions into hyperedges."""

    membership = np.asarray(membership, dtype=np.float64)
    if membership.ndim != 2:
        raise ValueError("membership must be a 2D array.")
    mass = np.maximum(membership.sum(axis=0), 1e-12)
    return (membership.T / mass[:, None]).astype(np.float32)


def hyperedge_spatial_mmd2(membership: np.ndarray, gene_mmd2: np.ndarray) -> np.ndarray:
    """Exact hyperedge-hyperedge MMD² from H-weighted gene distributions."""

    weights = hyperedge_gene_distribution_weights(membership).astype(np.float64)
    mmd2 = np.asarray(gene_mmd2, dtype=np.float64)
    if mmd2.ndim != 2 or mmd2.shape[0] != mmd2.shape[1] or mmd2.shape[0] != weights.shape[1]:
        raise ValueError("gene_mmd2 must be a square matrix matching membership rows.")

    # For mixtures P_a=sum_g A_ag P_g, energy-MMD² is obtained exactly by
    # cross_ab - 0.5 * cross_aa - 0.5 * cross_bb using the gene-level MMD² matrix.
    cross = weights @ mmd2 @ weights.T
    diagonal = np.diag(cross)
    module_mmd2 = cross - 0.5 * diagonal[:, None] - 0.5 * diagonal[None, :]
    module_mmd2 = np.maximum(0.5 * (module_mmd2 + module_mmd2.T), 0.0)
    np.fill_diagonal(module_mmd2, 0.0)
    return module_mmd2.astype(np.float32)


def hyperedge_spatial_mmd_similarity(
    membership: np.ndarray,
    gene_mmd2: np.ndarray,
    scale: float | None = None,
) -> np.ndarray:
    """Similarity R=exp(-MMD²/scale) between learned hyperedge distributions."""

    module_mmd2 = hyperedge_spatial_mmd2(membership, gene_mmd2)
    if scale is None:
        scale = hyperedge_mmd_similarity_scale(gene_mmd2)
    similarity = np.exp(-module_mmd2 / max(float(scale), 1e-12))
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def hyperedge_spatial_mmd_diversity_loss(membership, gene_mmd2, scale: float):
    # Each hyperedge distribution is the H-weighted mixture of gene spatial distributions.
    module_mass = membership.sum(dim=0).clamp_min(1e-12)
    module_weights = membership.T / module_mass[:, None]

    cross = module_weights @ gene_mmd2 @ module_weights.T
    diagonal = cross.diagonal()
    module_mmd2 = cross - 0.5 * diagonal[:, None] - 0.5 * diagonal[None, :]
    module_mmd2 = (0.5 * (module_mmd2 + module_mmd2.T)).clamp_min(0.0)

    off_diagonal_mask = module_mmd2.new_ones(module_mmd2.shape, dtype=bool)
    off_diagonal_mask.fill_diagonal_(False)
    off_diagonal = module_mmd2[off_diagonal_mask]
    scale_t = module_mmd2.new_tensor(float(scale)).clamp_min(1e-12)
    diversity_loss = (-off_diagonal / scale_t).exp().mean()
    return diversity_loss, module_mmd2


def similarity_reconstruction_weights(
    distance: np.ndarray,
    similarity: np.ndarray,
    local_k: int,
) -> np.ndarray:
    """Weight local and high-similarity pairs more strongly in S reconstruction."""

    n = distance.shape[0]
    k = int(np.clip(local_k, 1, max(1, n - 1)))
    local_mask = nearest_distance_mask(distance, k=k)
    local_union = local_mask | local_mask.T

    weights = np.full_like(similarity, 0.05, dtype=np.float32)
    if np.any(local_union):
        local_values = similarity[local_union]
        high_cutoff = float(np.quantile(local_values, 0.5))
        weights[local_union | (similarity >= high_cutoff)] = 1.0
    np.fill_diagonal(weights, 1.0)
    return weights


def nearest_distance_mask(distance: np.ndarray, k: int) -> np.ndarray:
    n = distance.shape[0]
    k = int(np.clip(k, 1, max(1, n - 1)))
    masked = distance.copy()
    np.fill_diagonal(masked, np.inf)
    indices = np.argpartition(masked, kth=k - 1, axis=1)[:, :k]
    mask = np.zeros_like(distance, dtype=bool)
    mask[np.arange(n)[:, None], indices] = True
    np.fill_diagonal(mask, False)
    return mask


def spectral_cosine_hyperedge_initialization(
    target_similarity: np.ndarray,
    n_modules: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """GAMULE-style initialization from gene embeddings and representative hyperedges."""

    gene_embedding = spectral_gene_embedding(target_similarity, n_modules=n_modules)
    representatives = embedding_representative_indices(
        gene_embedding,
        n_modules=n_modules,
        random_state=random_state,
    )
    gene_embedding_norm = normalize_rows_l2(gene_embedding)
    hyperedge_embedding_norm = gene_embedding_norm[representatives].copy()
    hyperedge_embedding = _SPECTRAL_HYPEREDGE_INIT_SCALE * hyperedge_embedding_norm
    logits = gene_embedding_norm @ hyperedge_embedding_norm.T
    logits = logits - logits.mean(axis=1, keepdims=True)
    return logits.astype(np.float32), hyperedge_embedding.astype(np.float32), representatives


def calibrate_initial_logits(logits: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale spectral logits to a soft-but-informative initial assignment."""

    logits = np.asarray(logits, dtype=np.float64)
    target = initial_assignment_target_mean_max(logits.shape[1])
    low, high = 0.0, 1.0
    while mean_max_softmax(high * logits) < target and high < 5.0:
        high *= 2.0
    high = min(high, 5.0)
    for _ in range(30):
        mid = 0.5 * (low + high)
        if mean_max_softmax(mid * logits) < target:
            low = mid
        else:
            high = mid
    scaled = high * logits
    return scaled.astype(np.float32), float(high)


def initial_assignment_target_mean_max(n_hyperedges: int) -> float:
    uniform = 1.0 / max(1, int(n_hyperedges))
    return float(min(0.35, max(_SPECTRAL_COSINE_TARGET_MEAN_MAX, uniform + 0.05)))


def mean_max_softmax(logits: np.ndarray) -> float:
    return float(np.max(softmax_numpy(logits), axis=1).mean())


def spectral_gene_embedding(target_similarity: np.ndarray, n_modules: int) -> np.ndarray:
    similarity = np.asarray(target_similarity, dtype=np.float64)
    similarity = 0.5 * (similarity + similarity.T)
    eigenvalues, eigenvectors = np.linalg.eigh(similarity)
    order = np.argsort(eigenvalues)[::-1][:n_modules]
    values = np.clip(eigenvalues[order], 1e-8, None)
    embedding = eigenvectors[:, order] * np.sqrt(values)[None, :]
    if embedding.shape[1] < n_modules:
        pad = np.zeros((embedding.shape[0], n_modules - embedding.shape[1]))
        embedding = np.concatenate([embedding, pad], axis=1)
    return embedding.astype(np.float32)


def embedding_representative_indices(
    embedding: np.ndarray,
    n_modules: int,
    random_state: int,
) -> list[int]:
    rng = np.random.default_rng(int(random_state))
    points = normalize_rows_l2(embedding)
    n_genes = points.shape[0]
    representatives = [int(rng.integers(n_genes))]
    min_distance = np.sum((points - points[representatives[0]]) ** 2, axis=1)
    for _ in range(1, n_modules):
        next_idx = int(np.argmax(min_distance))
        representatives.append(next_idx)
        distance = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return representatives


def append_unassigned_logits(real_logits: np.ndarray) -> np.ndarray:
    """Append a neutral unassigned sink logit without changing real hyperedge scale."""

    real_logits = np.asarray(real_logits, dtype=np.float64)
    real_logits = real_logits - real_logits.mean(axis=1, keepdims=True)
    sink = np.zeros((real_logits.shape[0], 1), dtype=np.float64)
    return np.concatenate([real_logits, sink], axis=1).astype(np.float32)


def append_unassigned_hyperedge_embedding(real_embedding: np.ndarray) -> np.ndarray:
    """Append a zero embedding row for the unassigned hyperedge."""

    real_embedding = np.asarray(real_embedding, dtype=np.float32)
    null_embedding = np.zeros((1, real_embedding.shape[1]), dtype=np.float32)
    return np.concatenate([real_embedding, null_embedding], axis=0)


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return normalize_rows(exp_logits).astype(np.float32)


def normalize_rows_l2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norm, 1e-12)).astype(np.float32)


def weighted_reconstruction_loss_numpy(
    target_similarity: np.ndarray,
    pair_weights: np.ndarray,
    membership: np.ndarray,
    hyperedge_embedding: np.ndarray,
) -> float:
    gene_embedding = membership @ hyperedge_embedding
    reconstruction = gene_embedding @ gene_embedding.T
    weight_sum = max(float(np.sum(pair_weights)), 1.0)
    return float(
        np.sum(pair_weights * (target_similarity - reconstruction) ** 2) / weight_sum
    )


def resolve_torch_device(device: str, torch_module) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 1e-12)
    return values / np.maximum(values.sum(axis=1, keepdims=True), 1e-12)


def hard_labels_with_unassigned(
    membership: np.ndarray,
    n_modules: int,
    unassigned_index: int | None,
) -> np.ndarray:
    labels = np.argmax(membership, axis=1).astype(int)
    if unassigned_index is not None:
        labels[labels == unassigned_index] = -1
    return labels


def enforce_nonempty_labels(
    labels: np.ndarray,
    membership: np.ndarray,
    n_modules: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    missing = [module for module in range(n_modules) if not np.any(labels == module)]
    if not missing:
        return labels
    confidence = np.max(membership, axis=1)
    used = set()
    for module in missing:
        order = np.argsort(membership[:, module] - confidence)[::-1]
        gene = next(
            (int(idx) for idx in order if idx not in used and labels[idx] >= 0),
            int(order[0]),
        )
        labels[gene] = module
        used.add(gene)
    return labels


def module_normalized_expression_scores(
    X: np.ndarray,
    module_labels: np.ndarray,
    module_ids: np.ndarray,
    vote_rate: float,
) -> np.ndarray:
    vote_rate = float(np.clip(vote_rate, 0.0, 1.0))
    gene_totals = X.sum(axis=0).astype(np.float64)
    X_norm = np.divide(
        X.astype(np.float64),
        gene_totals[None, :],
        out=np.zeros_like(X, dtype=np.float64),
        where=gene_totals[None, :] > 0,
    )
    X_norm *= 100.0
    scores = np.zeros((X.shape[0], module_ids.size), dtype=np.float32)
    for j, module_id in enumerate(module_ids):
        idx = module_labels == module_id
        coverage = (X[:, idx] > 0).mean(axis=1)
        scores[:, j] = X_norm[:, idx].sum(axis=1)
        scores[coverage < vote_rate, j] = 0.0
    return scores


def module_coverage_scores(
    X: np.ndarray,
    module_labels: np.ndarray,
    module_ids: np.ndarray,
) -> np.ndarray:
    detected = X > 0
    scores = np.zeros((X.shape[0], module_ids.size), dtype=np.float32)
    for j, module_id in enumerate(module_ids):
        idx = module_labels == module_id
        scores[:, j] = detected[:, idx].mean(axis=1)
    return scores


def module_summary_table(
    module_labels: np.ndarray,
    module_ids: np.ndarray,
    gene_names: np.ndarray,
    svg_distance: np.ndarray,
    svg_zscore: np.ndarray,
    gene_distance: np.ndarray,
    probabilities: np.ndarray,
    pattern_scores: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for col, module_id in enumerate(module_ids):
        idx = np.flatnonzero(module_labels == module_id)
        top = idx[np.argsort(svg_distance[idx])[::-1][:5]]
        rows.append(
            {
                "module": int(module_id),
                "n_genes": int(idx.size),
                "mean_svg_distance_to_bg": float(np.mean(svg_distance[idx])),
                "median_svg_distance_to_bg": float(np.median(svg_distance[idx])),
                "mean_svg_zscore": float(np.mean(svg_zscore[idx])),
                "mean_within_gene_distance": mean_upper(
                    gene_distance[np.ix_(idx, idx)], empty_value=0.0
                ),
                "spatial_concentration": probability_concentration(
                    probabilities[idx].mean(axis=0)
                ),
                "pattern_total_intensity": float(np.sum(pattern_scores[:, col])),
                "top_genes": ", ".join(map(str, gene_names[top])),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_svg_distance_to_bg", ascending=False).reset_index(
        drop=True
    )
