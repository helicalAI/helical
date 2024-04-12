# This tutorial covers the zero-shot integration with continual pre-trained scGPT. 
# This particular workflow works for scRNA-seq datasets without fine-tuning (or any extensive training) of scGPT.
# Continual pre-trained scGPT (scGPT_CP) is a model that inherits the pre-trained scGPT whole-human model checkpoint, 
# and is further supervised by extra cell type labels (using the [Tabula Sapiens](https://tabula-sapiens-portal.ds.czbiohub.org/) dataset) 
# during the continual pre-training stage. We observed that the scGPT_CP model can achieve comparable or better zero-shot performance 
# on cell embedding related tasks compared to the original checkpoint, especially on datasets with observable technical batch effects.
# This tutorial will show how to use the latent space of scGPT to integrate scRNA-seq datasets. 
# We use the `scGPT_CP` model to provide embeddings out of the box. 
# You may download it from [here](https://drive.google.com/drive/folders/1_GROJTzXiAV8HB4imruOTk6PEGuNOcgB).

# We will use the [scIB](https://www.nature.com/articles/s41592-021-01336-8) pancreas dataset as an example. 
# This dataset is publicly accessible via [here](https://figshare.com/ndownloader/files/24539828). You may place the dataset under `data` directory at the outer level.

# The zero-shot integration workflow is as follows:
#  1. [Load and pre-process the dataset](#prepare-the-datasets)
#  2. [Generate scGPT embeddings for each cell](#generate-the-cell-embeddings)

import os
from pathlib import Path
import scanpy as sc
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{dir_path}/scGPT")

import scgpt as scg
import logging
logger = logging.getLogger(__name__)

os.environ['KMP_DUPLICATE_LIB_OK']='True'
dir_path = os.path.dirname(os.path.realpath(__file__))

model_dir = Path("./scGPT/save/scGPT_CP")

# Load the Pancreas dataset (download it from [here](https://figshare.com/ndownloader/files/24539828)), and we set the columns storing gene name columns, 
# batch key and cell type key (optional, this is for evaluation).
smaple_data_path = './scGPT/data/human_pancreas_norm_complexBatch.h5ad'
adata = sc.read_h5ad(smaple_data_path)

gene_col = "gene_symbols"
N_HVG = 1800

adata.var[gene_col] = adata.var.index.values

# Preprocess the dataset and select `N_HVG` highly variable genes for downstream analysis.
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]

# Generate the cell embeddings for the dataset using `embed_data` function. 
# `embed_data` calculates the cell embedding for each cell with the given scGPT model. 
# The extracted embedding is stored in the `X_scGPT` field of `obsm` in AnnData.

embed_adata = scg.tasks.embed_data(
    adata,
    model_dir,
    gene_col=gene_col,
    batch_size=64,
)