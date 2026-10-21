"""Gene-first spatial gene module discovery with exact energy-MMD."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .energy_distance import energy_distance_to_background, energy_mmd2
from .modules import (
    learn_soft_hyperedge_modules,
    module_coverage_scores,
    module_normalized_expression_scores,
    module_summary_table,
    select_svg_mask,
    zscore,
)
from .preprocessing import (
    TransformName,
    gene_probability_matrix,
    get_expression_matrix,
    get_spatial_coordinates,
    get_var_names,
    mean_gene_probability_background,
    select_expression_matrix,
    standardize_coordinates,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class NicheRadar:
    """Identify spatial gene modules from gene-level spatial distributions."""

    n_gene_modules: int = 6
    n_svgs: int | None = None
    transform: TransformName = "identity"
    layer: str | None = None
    use_raw: bool = False
    min_total_expression: float | None = None
    min_mean_expression: float | None = None
    min_detected_spots: int | None = None
    min_detected_fraction: float | None = None
    min_variance: float | None = None
    svg_zscore_threshold: float | None = 1.0
    svg_quantile: float | None = None
    min_svg_distance: float | None = None
    chunk_size: int = 512
    hyperedge_local_k: int = 15
    hyperedge_n_iter: int = 200
    hyperedge_learning_rate: float = 1e-2
    hyperedge_diversity_weight: float = 1e-3
    enable_unassigned_hyperedge: bool = True
    hyperedge_device: str = "auto"
    module_vote_rate: float = 0.0
    random_state: int = 43
    show_progress: bool = True

    def fit(self, adata):
        """Fit the gene-first energy-MMD module model."""

        LOGGER.info("NicheRadar: loading spatial coordinates and expression matrix.")
        coords_raw = get_spatial_coordinates(adata)
        coords = standardize_coordinates(coords_raw)

        X_all, expression_source = get_expression_matrix(
            adata, layer=self.layer, use_raw=self.use_raw
        )
        X, gene_names, gene_indices, filter_stats = select_expression_matrix(
            X_all,
            var_names=get_var_names(adata, use_raw=self.use_raw),
            expression_source=expression_source,
            transform=self.transform,
            min_total_expression=self.min_total_expression,
            min_mean_expression=self.min_mean_expression,
            min_detected_spots=self.min_detected_spots,
            min_detected_fraction=self.min_detected_fraction,
            min_variance=self.min_variance,
        )
        LOGGER.info(
            "NicheRadar: %d/%d genes passed expression filters.",
            X.shape[1],
            X_all.shape[1],
        )

        probabilities = gene_probability_matrix(X)
        background = mean_gene_probability_background(probabilities)
        LOGGER.info("NicheRadar: computing gene-to-background Energy-MMD.")
        svg_distance = energy_distance_to_background(
            probabilities=probabilities,
            background=background,
            coords=coords,
            chunk_size=self.chunk_size,
            show_progress=self.show_progress,
        )
        svg_zscore = zscore(np.log1p(svg_distance))
        svg_mask = select_svg_mask(
            svg_distance=svg_distance,
            svg_zscore=svg_zscore,
            n_svgs=self.n_svgs,
            zscore_threshold=self.svg_zscore_threshold,
            quantile=self.svg_quantile,
            min_distance=self.min_svg_distance,
        )
        if svg_mask.sum() < self.n_gene_modules:
            raise ValueError(
                "Too few SVGs were retained for the requested number of modules. "
                "Lower svg_zscore_threshold/svg_quantile/min_svg_distance or reduce "
                "n_gene_modules."
            )
        LOGGER.info(
            "NicheRadar: selected %d SVGs for gene-gene Energy-MMD.",
            int(svg_mask.sum()),
        )

        svg_indices = np.flatnonzero(svg_mask)
        svg_order = np.argsort(svg_distance[svg_indices])[::-1]
        svg_indices = svg_indices[svg_order]
        X_svg = X[:, svg_indices]
        probabilities_svg = probabilities[svg_indices]
        gene_names_svg = np.asarray(gene_names)[svg_indices]
        gene_indices_svg = np.asarray(gene_indices)[svg_indices]
        svg_distance_svg = svg_distance[svg_indices]
        svg_zscore_svg = svg_zscore[svg_indices]

        LOGGER.info("NicheRadar: computing SVG gene-gene Energy-MMD matrix.")
        mmd2 = energy_mmd2(
            distributions=probabilities_svg,
            coords=coords,
            chunk_size=self.chunk_size,
            show_progress=self.show_progress,
        )
        gene_distance = np.sqrt(np.maximum(mmd2, 0.0)).astype(np.float32)
        LOGGER.info(
            "NicheRadar: learning soft gene-module incidence from Energy-MMD."
        )
        (
            module_labels,
            module_membership,
            hyperedge_embedding,
            gene_module_embedding,
            module_similarity_reconstruction,
            hyperedge_info,
        ) = (
            learn_soft_hyperedge_modules(
                gene_distance=gene_distance,
                n_modules=self.n_gene_modules,
                random_state=self.random_state,
                local_k=self.hyperedge_local_k,
                n_iter=self.hyperedge_n_iter,
                learning_rate=self.hyperedge_learning_rate,
                diversity_weight=self.hyperedge_diversity_weight,
                device=self.hyperedge_device,
                gene_mmd2=mmd2,
                enable_unassigned_hyperedge=self.enable_unassigned_hyperedge,
            )
        )
        module_ids = np.arange(self.n_gene_modules, dtype=int)
        LOGGER.info(
            "NicheRadar: unassigned hyperedge contains %d/%d SVGs (%.2f%%).",
            int(hyperedge_info.get("hyperedge_unassigned_gene_count", 0)),
            int(len(module_labels)),
            100.0 * float(hyperedge_info.get("hyperedge_unassigned_gene_fraction", 0.0)),
        )

        module_scores = module_normalized_expression_scores(
            X_svg, module_labels, module_ids, vote_rate=self.module_vote_rate
        )
        module_coverage = module_coverage_scores(X_svg, module_labels, module_ids)
        module_table = module_summary_table(
            module_labels=module_labels,
            module_ids=module_ids,
            gene_names=gene_names_svg,
            svg_distance=svg_distance_svg,
            svg_zscore=svg_zscore_svg,
            gene_distance=gene_distance,
            probabilities=probabilities_svg,
            pattern_scores=module_scores,
        )

        self.expression_source_ = expression_source
        self.filter_stats_ = {
            **filter_stats,
            "n_svg_candidates": int(len(gene_names)),
            "n_svgs": int(svg_mask.sum()),
            "svg_distance_cutoff": float(np.min(svg_distance_svg)),
            "svg_zscore_cutoff": float(np.min(svg_zscore_svg)),
        }
        self.coords_ = coords_raw
        self.standardized_coords_ = coords
        self.gene_names_all_ = np.asarray(gene_names)
        self.gene_indices_all_ = np.asarray(gene_indices)
        self.svg_distance_all_ = svg_distance
        self.svg_zscore_all_ = svg_zscore
        self.svg_mask_ = svg_mask
        self.gene_names_ = gene_names_svg
        self.gene_indices_ = gene_indices_svg
        self.svg_distance_ = svg_distance_svg
        self.svg_zscore_ = svg_zscore_svg
        self.svg_expression_ = X_svg
        self.svg_probabilities_ = probabilities_svg
        self.background_probability_ = background
        self.gene_mmd2_ = mmd2
        self.gene_distance_ = gene_distance
        self.gene_module_membership_ = module_membership
        self.hyperedge_embedding_ = hyperedge_embedding
        self.gene_module_similarity_reconstruction_ = module_similarity_reconstruction
        self.hyperedge_info_ = hyperedge_info
        self.gene_module_embedding_ = gene_module_embedding
        self.module_ids_ = module_ids
        self.module_labels_ = module_labels
        self.module_coverage_ = module_coverage
        self.module_scores_ = module_scores
        self.module_table_ = module_table
        return self

    def fit_predict(self, adata, copy: bool = False):
        """Fit the model and write SVG/module outputs into AnnData."""

        target = adata.copy() if copy else adata
        self.fit(target)

        target.obsm["X_nicheradar_modules"] = self.module_scores_
        target.obsm["X_nicheradar_module_coverage"] = self.module_coverage_
        target.uns["nicheradar_module_table"] = self.module_table_.to_dict("list")
        target.uns["nicheradar_svg_table"] = self.svg_table_.to_dict("list")
        target.uns["nicheradar_hyperedge_info"] = self.hyperedge_info_
        target.uns["nicheradar_params"] = self._params_dict()
        target.uns["nicheradar_expression_source"] = self.expression_source_
        target.uns["nicheradar_filter_stats"] = self.filter_stats_

        svg_distance_by_var = pd.Series(np.nan, index=target.var_names, dtype=float)
        svg_flag_by_var = pd.Series(False, index=target.var_names, dtype=bool)
        module_by_var = pd.Series(-1, index=target.var_names, dtype=int)

        all_dist = pd.Series(self.svg_distance_all_, index=self.gene_names_all_)
        annotated_all = all_dist.index.intersection(target.var_names)
        svg_distance_by_var.loc[annotated_all] = all_dist.loc[annotated_all].to_numpy()

        selected_modules = pd.Series(self.module_labels_, index=self.gene_names_)
        annotated_svg = selected_modules.index.intersection(target.var_names)
        svg_flag_by_var.loc[annotated_svg] = True
        module_by_var.loc[annotated_svg] = selected_modules.loc[annotated_svg].to_numpy()

        target.var["nicheradar_svg_distance"] = svg_distance_by_var.to_numpy()
        target.var["nicheradar_is_svg"] = svg_flag_by_var.to_numpy()
        target.var["nicheradar_module"] = module_by_var.to_numpy()
        return target

    @property
    def svg_table_(self) -> pd.DataFrame:
        """Per-gene SVG ranking table for all genes passing basic filters."""

        table = pd.DataFrame(
            {
                "gene": self.gene_names_all_,
                "svg_distance_to_background": self.svg_distance_all_,
                "svg_zscore": self.svg_zscore_all_,
                "is_svg": self.svg_mask_,
            }
        )
        selected_modules = pd.Series(self.module_labels_, index=self.gene_names_)
        table["module"] = table["gene"].map(selected_modules).fillna(-1).astype(int)
        return table.sort_values(
            ["is_svg", "svg_distance_to_background"],
            ascending=[False, False],
        ).reset_index(drop=True)

    def _params_dict(self) -> dict:
        params = {
            "n_gene_modules": self.n_gene_modules,
            "n_svgs": self.n_svgs,
            "expression_layer": self.layer,
            "use_raw": self.use_raw,
            "expression_transform": self.transform,
            "min_total_expression": self.min_total_expression,
            "min_mean_expression": self.min_mean_expression,
            "min_detected_spots": self.min_detected_spots,
            "min_detected_fraction": self.min_detected_fraction,
            "min_variance": self.min_variance,
            "svg_zscore_threshold": self.svg_zscore_threshold,
            "svg_quantile": self.svg_quantile,
            "min_svg_distance": self.min_svg_distance,
            "exact_chunk_size": self.chunk_size,
            "hyperedge_local_k": self.hyperedge_local_k,
            "hyperedge_n_iter": self.hyperedge_n_iter,
            "hyperedge_learning_rate": self.hyperedge_learning_rate,
            "hyperedge_diversity_weight": self.hyperedge_diversity_weight,
            "enable_unassigned_hyperedge": self.enable_unassigned_hyperedge,
            "hyperedge_device": self.hyperedge_device,
            "module_vote_rate": self.module_vote_rate,
            "random_state": self.random_state,
            "show_progress": self.show_progress,
        }
        return {key: _uns_safe_value(value) for key, value in params.items()}


def _uns_safe_value(value):
    return "None" if value is None else value
