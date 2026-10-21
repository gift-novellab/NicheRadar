# NicheRadar

NicheRadar is a gene-first framework for functional niche discovery in spatial transcriptomics. It represents each gene as a spatial probability distribution, selects background-deviating spatially variable genes, compares genes with exact Energy-MMD, and consolidates co-localized genes into interpretable niche modules through soft hypergraph learning.

## Repository Layout

- `nicheradar/`: core Python implementation.
- `notebooks/NicheRadar_lymph_node.ipynb`: standard lymph node workflow.
- `data/`: local input data directory. Large files are ignored by git.
- `output/`: generated tables, figures, and optional processed AnnData files.
- `environment.yml`: conda environment for reproducing the notebook.
- `pyproject.toml`: package metadata for editable installation.

## Installation

Create the environment:

```bash
conda env create -f environment.yml
conda activate nicheradar
pip install -e .
python -m ipykernel install --user --name nicheradar --display-name "NicheRadar"
```

If you already have a compatible environment with `anndata`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, and notebook plotting packages, install only the local package:

```bash
pip install -e .
```

## Input Requirements

NicheRadar expects an `AnnData` object with:

- expression in `adata.X`, a named layer, or `adata.raw.X`;
- spatial coordinates in `adata.obsm["spatial"]` or `adata.obs[["x", "y"]]`.

For the lymph node notebook, place the dataset at:

```text
data/lymph_node_niche_annotated.h5ad
```

or set:

```bash
export NICHERADAR_LYMPH_NODE_H5AD=/path/to/lymph_node_niche_annotated.h5ad
```

## Quick Start

```python
import anndata as ad
from nicheradar import NicheRadar, NicheRadarNiche

adata = ad.read_h5ad("data/lymph_node_niche_annotated.h5ad")

model = NicheRadar(
    transform="identity",
    min_total_expression=10.0,
    min_mean_expression=1e-4,
    min_detected_spots=20,
    min_detected_fraction=0.01,
    min_variance=1e-8,
    n_svgs=700,
    svg_zscore_threshold=None,
    n_gene_modules=18,
    hyperedge_local_k=13,
    hyperedge_n_iter=700,
    hyperedge_learning_rate=5e-3,
    hyperedge_diversity_weight=1e-2,
    enable_unassigned_hyperedge=False,
    hyperedge_device="cpu",
    module_vote_rate=0.15,
    random_state=303,
)
adata = model.fit_predict(adata, copy=False)

niche = NicheRadarNiche(activity_key="normalized_expression")
niche.fit_meta_modules(adata, similarity_threshold=0.57)
niche.assign_domains(selected_meta_modules=None)
adata = niche.write_meta_modules(adata)
adata = niche.write_niches(adata)
```

## Main Outputs

`NicheRadar.fit_predict` writes:

- `adata.obsm["X_nicheradar_modules"]`: normalized-expression module activity.
- `adata.obsm["X_nicheradar_module_coverage"]`: module gene coverage activity.
- `adata.uns["nicheradar_module_table"]`: module-level summary table.
- `adata.uns["nicheradar_svg_table"]`: SVG ranking and module assignments.
- `adata.var["nicheradar_svg_distance"]`: gene-to-background Energy-MMD distance.
- `adata.var["nicheradar_is_svg"]`: selected SVG flag.
- `adata.var["nicheradar_module"]`: assigned module label, with `-1` for unassigned genes.

`NicheRadarNiche` writes:

- `adata.obs["nicheradar_niche"]`: dominant meta-module niche label.
- `adata.obsm["X_nicheradar_niche_features"]`: meta-module feature matrix.
- `adata.obsm["X_nicheradar_meta_common_scores_display"]`: smoothed normalized scores used for final assignment.
- `adata.uns["nicheradar_niche_meta_modules"]`: merged module groups.
- `adata.uns["nicheradar_niche_signature_table"]`: module signatures for each niche.

## Lymph Node Notebook

Open `notebooks/NicheRadar_lymph_node.ipynb` with the `NicheRadar` kernel. The notebook:

1. loads the lymph node AnnData object;
2. runs NicheRadar module discovery with the paper-aligned naming and output keys;
3. exports module and SVG tables;
4. visualizes spatial module activity;
5. optionally compares NicheRadar niches with manual annotations when `adata.obs["niche"]` is present.

Set `QUICK_TEST = True` in the notebook only for a fast smoke test on a subset. Keep it `False` for the full lymph node run.
