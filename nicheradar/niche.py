"""Meta-module construction and domain assignment from NicheRadar modules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .preprocessing import get_spatial_coordinates, standardize_coordinates
from .spatial import (
    mean_upper,
    module_similarity,
    morans_i,
    quantile_normalize_columns,
    robust_scale,
    smooth_features,
    smooth_score_matrix,
    spatial_knn_graph,
)


LOGGER = logging.getLogger(__name__)

ActivityKey = Literal["normalized_expression", "coverage"]


@dataclass
class NicheRadarNiche:
    """Infer spatial domains from gene module and meta-module activity."""

    activity_key: ActivityKey = "coverage"
    robust_clip: float = 8.0
    min_reliability: float = 0.05

    def fit(
        self,
        adata,
        *,
        similarity_threshold: float = 0.75,
        module_smoothing_k: int = 12,
        module_smoothing_steps: int = 2,
        module_smoothing_weight: float = 0.45,
        reliability_spatial_k: int = 18,
        selected_meta_modules: tuple[int, ...] | list[int] | None = None,
        q_low: float = 1.0,
        q_high: float = 99.0,
        score_smoothing_k: int = 18,
        score_smoothing_steps: int = 1,
        score_smoothing_weight: float = 0.35,
    ):
        """Fit meta-modules and assign domains in one call."""

        self.fit_meta_modules(
            adata,
            similarity_threshold=similarity_threshold,
            module_smoothing_k=module_smoothing_k,
            module_smoothing_steps=module_smoothing_steps,
            module_smoothing_weight=module_smoothing_weight,
            reliability_spatial_k=reliability_spatial_k,
        )
        self.assign_domains(
            selected_meta_modules=selected_meta_modules,
            q_low=q_low,
            q_high=q_high,
            score_smoothing_k=score_smoothing_k,
            score_smoothing_steps=score_smoothing_steps,
            score_smoothing_weight=score_smoothing_weight,
        )
        return self

    def fit_meta_modules(
        self,
        adata,
        *,
        similarity_threshold: float = 0.75,
        module_smoothing_k: int = 12,
        module_smoothing_steps: int = 2,
        module_smoothing_weight: float = 0.45,
        reliability_spatial_k: int = 18,
    ):
        """Merge spatially similar gene modules into meta-modules."""

        LOGGER.info("NicheRadarNiche: building meta-modules from module activity.")
        coords_raw = get_spatial_coordinates(adata)
        coords = standardize_coordinates(coords_raw)
        activity = _get_module_activity(adata, self.activity_key)
        module_ids = _module_ids(adata, activity.shape[1])

        scaled_activity = robust_scale(activity, clip=self.robust_clip)
        module_table = _module_table(adata)
        reliability, reliability_table = module_reliability(
            scaled_activity=scaled_activity,
            coords=coords,
            module_ids=module_ids,
            module_table=module_table,
            min_reliability=self.min_reliability,
            spatial_k=reliability_spatial_k,
        )
        smooth_graph = spatial_knn_graph(coords, k=module_smoothing_k)
        denoised_activity = smooth_features(
            scaled_activity,
            graph=smooth_graph,
            steps=module_smoothing_steps,
            weight=module_smoothing_weight,
        )
        weighted_activity = denoised_activity * reliability[None, :]

        feature_matrix, feature_names, meta_modules = build_meta_module_features(
            weighted_activity=weighted_activity,
            module_ids=module_ids,
            reliability=reliability,
            similarity_threshold=similarity_threshold,
            robust_clip=self.robust_clip,
        )

        self.coords_ = coords_raw
        self.standardized_coords_ = coords
        self.module_ids_ = module_ids
        self.module_activity_ = activity
        self.scaled_activity_ = scaled_activity
        self.denoised_activity_ = denoised_activity
        self.module_reliability_ = reliability
        self.module_reliability_table_ = reliability_table
        self.meta_modules_ = meta_modules
        self.feature_matrix_ = feature_matrix
        self.feature_names_ = feature_names
        self.meta_module_params_ = {
            "activity_key": self.activity_key,
            "robust_clip": self.robust_clip,
            "min_reliability": self.min_reliability,
            "similarity_threshold": similarity_threshold,
            "module_smoothing_k": module_smoothing_k,
            "module_smoothing_steps": module_smoothing_steps,
            "module_smoothing_weight": module_smoothing_weight,
            "reliability_spatial_k": reliability_spatial_k,
        }
        LOGGER.info(
            "NicheRadarNiche: merged %d gene modules into %d meta-modules.",
            len(module_ids),
            len(meta_modules),
        )
        return self

    def assign_domains(
        self,
        *,
        selected_meta_modules: tuple[int, ...] | list[int] | None = None,
        q_low: float = 1.0,
        q_high: float = 99.0,
        score_smoothing_k: int = 18,
        score_smoothing_steps: int = 1,
        score_smoothing_weight: float = 0.35,
    ):
        """Assign each spot to the strongest normalized meta-module pattern."""

        if not hasattr(self, "feature_matrix_"):
            raise RuntimeError("Run fit_meta_modules() before assign_domains().")

        LOGGER.info("NicheRadarNiche: assigning domains from normalized meta-module scores.")
        graph = spatial_knn_graph(self.standardized_coords_, k=score_smoothing_k)
        niche_result = dominant_meta_module_domains(
            features=self.feature_matrix_,
            feature_names=self.feature_names_,
            meta_modules=self.meta_modules_,
            graph=graph,
            selected_meta_modules=selected_meta_modules,
            q_low=q_low,
            q_high=q_high,
            smoothing_steps=score_smoothing_steps,
            smoothing_weight=score_smoothing_weight,
        )
        signature = niche_module_signature(
            niche_result["labels"],
            self.scaled_activity_,
            self.module_ids_,
            label_names=niche_result["label_names"],
        )

        self.labels_ = niche_result["labels"]
        self.niche_label_names_ = niche_result["label_names"]
        self.selected_meta_modules_ = niche_result["selected_meta_modules"]
        self.meta_common_scores_ = niche_result["raw_scores"]
        self.meta_common_scores_norm_ = niche_result["normalized_scores"]
        self.meta_common_scores_display_ = niche_result["display_scores"]
        self.meta_common_score_names_ = niche_result["score_names"]
        self.niche_module_signature_ = signature
        self.niche_signature_table_ = niche_signature_table(signature)
        self.domain_params_ = {
            "selected_meta_modules": selected_meta_modules,
            "q_low": q_low,
            "q_high": q_high,
            "score_smoothing_k": score_smoothing_k,
            "score_smoothing_steps": score_smoothing_steps,
            "score_smoothing_weight": score_smoothing_weight,
        }
        LOGGER.info(
            "NicheRadarNiche: assigned %d spots to %d domains.",
            len(self.labels_),
            len(self.niche_label_names_),
        )
        return self

    def fit_predict(self, adata, copy: bool = False, **kwargs):
        target = adata.copy() if copy else adata
        self.fit(target, **kwargs)
        self.write_meta_modules(target, copy=False)
        self.write_niches(target, copy=False)
        return target

    def write_meta_modules(self, adata, copy: bool = False):
        target = adata.copy() if copy else adata
        if not hasattr(self, "feature_matrix_"):
            raise RuntimeError("Run fit_meta_modules() before write_meta_modules().")
        target.obsm["X_nicheradar_niche_features"] = self.feature_matrix_
        target.uns["nicheradar_niche_params"] = self._params_dict()
        target.uns["nicheradar_niche_feature_names"] = list(self.feature_names_)
        target.uns["nicheradar_niche_module_reliability"] = (
            self.module_reliability_table_.to_dict("list")
        )
        target.uns["nicheradar_niche_meta_modules"] = self.meta_modules_
        return target

    def write_niches(self, adata, copy: bool = False):
        target = adata.copy() if copy else adata
        if not hasattr(self, "labels_"):
            raise RuntimeError("Run assign_domains() before write_niches().")
        niche_categories = list(self.niche_label_names_)
        target.obs["nicheradar_niche"] = pd.Categorical(
            [niche_categories[label] for label in self.labels_],
            categories=niche_categories,
        )
        target.obs["nicheradar_dominant_meta_module"] = target.obs["nicheradar_niche"].copy()
        target.obsm["X_nicheradar_meta_common_scores"] = self.meta_common_scores_
        target.obsm["X_nicheradar_meta_common_scores_norm"] = self.meta_common_scores_norm_
        target.obsm["X_nicheradar_meta_common_scores_display"] = self.meta_common_scores_display_
        target.uns["nicheradar_niche_params"] = self._params_dict()
        target.uns["nicheradar_meta_common_score_names"] = list(self.meta_common_score_names_)
        target.uns["nicheradar_dominant_meta_modules_used"] = list(self.selected_meta_modules_)
        target.uns["nicheradar_niche_module_signature"] = (
            self.niche_module_signature_.to_dict("list")
        )
        target.uns["nicheradar_niche_signature_table"] = (
            self.niche_signature_table_.to_dict("list")
        )
        return target

    def _params_dict(self) -> dict:
        params = {}
        params.update(getattr(self, "meta_module_params_", {}))
        params.update(getattr(self, "domain_params_", {}))
        return params


def _get_module_activity(adata, activity_key: ActivityKey) -> np.ndarray:
    if activity_key == "normalized_expression":
        key = "X_nicheradar_modules"
    elif activity_key == "coverage":
        key = "X_nicheradar_module_coverage"
    else:
        raise ValueError("activity_key must be 'normalized_expression' or 'coverage'.")
    if key not in adata.obsm:
        raise KeyError(f"Missing adata.obsm[{key!r}]. Run NicheRadar first.")
    activity = np.asarray(adata.obsm[key], dtype=np.float64)
    if activity.ndim != 2 or activity.shape[1] == 0:
        raise ValueError(f"adata.obsm[{key!r}] must be a non-empty 2D matrix.")
    return np.nan_to_num(activity, copy=False)


def _module_ids(adata, n_modules: int) -> np.ndarray:
    table = _module_table(adata)
    if "module" in table and table.shape[0] == n_modules:
        return np.sort(table["module"].astype(int).to_numpy())
    return np.arange(n_modules, dtype=int)


def _module_table(adata) -> pd.DataFrame:
    value = adata.uns.get("nicheradar_module_table", {})
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, dict) and value:
        return pd.DataFrame(value)
    return pd.DataFrame()


def module_reliability(
    scaled_activity: np.ndarray,
    coords: np.ndarray,
    module_ids: np.ndarray,
    module_table: pd.DataFrame,
    min_reliability: float,
    spatial_k: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    moran = np.array(
        [morans_i(scaled_activity[:, i], coords, spatial_k=spatial_k) for i in range(scaled_activity.shape[1])],
        dtype=np.float64,
    )
    d_bg = _table_values(
        module_table, module_ids, "mean_svg_distance_to_bg", default=1.0
    )
    within = _table_values(
        module_table, module_ids, "mean_within_gene_distance", default=np.nan
    )

    d_score = _scale01(np.log1p(np.maximum(d_bg, 0.0)), default=1.0)
    moran_score = _scale01(np.maximum(moran, 0.0), default=1.0)
    if np.all(np.isnan(within)):
        coherence_score = np.ones_like(d_score)
    else:
        filled = np.nan_to_num(within, nan=np.nanmedian(within))
        coherence_score = 1.0 - _scale01(filled, default=0.0)

    reliability = (d_score + moran_score + coherence_score) / 3.0
    reliability = np.clip(reliability, min_reliability, 1.0)
    table = pd.DataFrame(
        {
            "module": module_ids.astype(int),
            "reliability": reliability,
            "d_bg_score": d_score,
            "moran_score": moran_score,
            "coherence_score": coherence_score,
            "morans_i": moran,
            "mean_svg_distance_to_bg": d_bg,
            "mean_within_gene_distance": within,
        }
    )
    return reliability.astype(np.float32), table


def build_meta_module_features(
    weighted_activity: np.ndarray,
    module_ids: np.ndarray,
    reliability: np.ndarray,
    similarity_threshold: float,
    robust_clip: float,
) -> tuple[np.ndarray, list[str], list[dict]]:
    similarity = module_similarity(weighted_activity)
    # Connected components avoid double-counting modules with highly similar
    # spatial activity while keeping dissimilar modules independent.
    groups = connected_module_groups(similarity, threshold=similarity_threshold)
    features = []
    names = []
    meta_modules = []

    for group_id, group in enumerate(groups):
        group_modules = module_ids[group]
        group_weights = reliability[group].astype(np.float64)
        common = weighted_average_columns(weighted_activity[:, group], group_weights)
        common_name = "meta_%d_common_%s" % (
            group_id,
            "_".join(f"m{m}" for m in group_modules),
        )
        features.append(common)
        names.append(common_name)
        meta_modules.append(
            {
                "meta_module": int(group_id),
                "modules": [int(m) for m in group_modules],
                "mean_similarity": mean_upper(similarity[np.ix_(group, group)]),
            }
        )

    return robust_scale(np.column_stack(features), clip=robust_clip), names, meta_modules


def connected_module_groups(similarity: np.ndarray, threshold: float) -> list[np.ndarray]:
    n_modules = similarity.shape[0]
    visited = np.zeros(n_modules, dtype=bool)
    groups = []
    adjacency = similarity >= threshold
    np.fill_diagonal(adjacency, True)
    for start in range(n_modules):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        group = []
        while stack:
            node = stack.pop()
            group.append(node)
            for nxt in np.flatnonzero(adjacency[node]):
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(int(nxt))
        groups.append(np.array(sorted(group), dtype=int))
    return groups


def weighted_average_columns(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 1e-8)
    return np.average(values.astype(np.float64), axis=1, weights=weights)


def dominant_meta_module_domains(
    features: np.ndarray,
    feature_names: list[str],
    meta_modules: list[dict],
    graph,
    selected_meta_modules: tuple[int, ...] | list[int] | None,
    q_low: float,
    q_high: float,
    smoothing_steps: int,
    smoothing_weight: float,
) -> dict:
    """Assign spots by winner-take-all normalized meta-module common scores."""

    meta_ids = _selected_meta_ids(meta_modules, selected_meta_modules)
    score_cols, score_names = _meta_common_columns(feature_names, meta_ids)
    raw_scores = np.asarray(features[:, score_cols], dtype=np.float64)

    # Scores are normalized column-wise before winner-take-all assignment so
    # high-range meta-modules do not dominate solely by scale.
    normalized = quantile_normalize_columns(raw_scores, q_low=q_low, q_high=q_high)
    display_scores = smooth_score_matrix(
        normalized,
        graph=graph,
        steps=smoothing_steps,
        weight=smoothing_weight,
    )
    labels = np.nanargmax(display_scores, axis=1).astype(int)
    label_names = [f"Meta {int(meta_id)}" for meta_id in meta_ids]

    return {
        "labels": labels,
        "label_names": label_names,
        "selected_meta_modules": [int(meta_id) for meta_id in meta_ids],
        "score_names": score_names,
        "raw_scores": raw_scores.astype(np.float32),
        "normalized_scores": normalized.astype(np.float32),
        "display_scores": display_scores.astype(np.float32),
    }


def _selected_meta_ids(
    meta_modules: list[dict],
    selected_meta_modules: tuple[int, ...] | list[int] | None,
) -> list[int]:
    available = [int(row["meta_module"]) for row in meta_modules]
    if selected_meta_modules is None:
        return available
    selected = {int(value) for value in selected_meta_modules}
    missing = sorted(selected.difference(available))
    if missing:
        raise ValueError(
            f"Selected meta modules are not available: {missing}. Available: {available}"
        )
    meta_ids = [meta_id for meta_id in available if meta_id in selected]
    if not meta_ids:
        raise ValueError("selected_meta_modules selected no meta modules.")
    return meta_ids


def _meta_common_columns(feature_names: list[str], meta_ids: list[int]) -> tuple[list[int], list[str]]:
    cols = []
    names = []
    for meta_id in meta_ids:
        prefix = f"meta_{int(meta_id)}_common_"
        matches = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one common-score column for Meta {meta_id}, found {len(matches)}."
            )
        cols.append(matches[0])
        names.append(f"Meta {int(meta_id)}")
    return cols, names


def niche_module_signature(
    labels: np.ndarray,
    scaled_activity: np.ndarray,
    module_ids: np.ndarray,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    scaled_activity = np.asarray(scaled_activity, dtype=np.float64)
    n_clusters = len(label_names) if label_names is not None else int(np.max(labels) + 1)
    signature = np.zeros((n_clusters, scaled_activity.shape[1]), dtype=np.float64)
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        if np.any(mask):
            signature[cluster_id] = np.mean(scaled_activity[mask], axis=0)
    columns = [f"module_{int(module_id)}" for module_id in module_ids]
    index = label_names if label_names is not None else [f"N{k + 1}" for k in range(n_clusters)]
    return pd.DataFrame(signature, index=index, columns=columns).reset_index(
        names="niche"
    )


def niche_signature_table(signature: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    module_cols = [col for col in signature.columns if col != "niche"]
    rows = []
    for _, row in signature.iterrows():
        values = row[module_cols].astype(float)
        high = values.sort_values(ascending=False).head(top_n)
        low = values.sort_values(ascending=True).head(top_n)
        rows.append(
            {
                "niche": row["niche"],
                "high_modules": ", ".join(
                    f"{name}={value:.3g}" for name, value in high.items()
                ),
                "low_modules": ", ".join(
                    f"{name}={value:.3g}" for name, value in low.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def _table_values(
    table: pd.DataFrame, module_ids: np.ndarray, column: str, default: float
) -> np.ndarray:
    if table.empty or "module" not in table or column not in table:
        return np.full(module_ids.size, default, dtype=np.float64)
    values = table.set_index("module")[column]
    return np.array([values.get(int(module_id), default) for module_id in module_ids])


def _scale01(values: np.ndarray, default: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.nan_to_num(values, nan=np.nanmedian(values))
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.full(values.shape, default, dtype=np.float64)
    return (values - lo) / (hi - lo)
