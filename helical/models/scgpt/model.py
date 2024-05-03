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
import scanpy as sc
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{dir_path}/scGPT")
from helical.models.helical import HelicalBaseModel
from helical.models.scgpt.scgpt_config import scGPTConfig
import numpy as np
from anndata import AnnData
import scgpt as scg
import logging
import json
from typing import Union
from pathlib import Path
from accelerate import Accelerator
from helical.services.downloader import Downloader

os.environ['KMP_DUPLICATE_LIB_OK']='True'
dir_path = os.path.dirname(os.path.realpath(__file__))

class scGPT(HelicalBaseModel):
    default_config = scGPTConfig()

    def __init__(self, model_dir=None, model_config: scGPTConfig = default_config) -> None:
                
        super().__init__()
        self.model_config = model_config.config
        self.downloader = Downloader()
        
        if model_dir is None:
            self.downloader.download_via_name("scgpt/scGPT_CP/vocab.json")
            self.downloader.download_via_name("scgpt/scGPT_CP/best_model.pt")
            self.model_dir = Path(os.path.join(self.downloader.CACHE_DIR_HELICAL,'scgpt/scGPT_CP'))
        else:
            self.model_dir = Path(model_dir)

        self.log = logging.getLogger("scGPT-Model")

        if self.model_config["accelerator"]:
            self.accelerator = Accelerator(project_dir=self.model_dir, cpu=self.model_config["accelerator"]["cpu"])
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None

    def get_embeddings(self, gene_column_name: str = "gene_col") -> np.array:
        self.log.info(f"Inference started")
        # The extracted embedding is stored in the `X_scGPT` field of `obsm` in AnnData.
        # for local development, only get embeddings for the first 100 entries
        return scg.tasks.embed_data(
            self.adata,
            self.model_dir,
            self.model_config,
            gene_col=gene_column_name,
        )
    
    def process_data(self, adata: AnnData, n_top_genes: int = 1800, flavor: str = "seurat_v3"):

        self.adata = adata
        self.adata.var[self.data_config['scgpt']['gene_column_name']] = self.adata.var.index.values

        # Preprocess the dataset and select `N_HVG` highly variable genes for downstream analysis.
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)

        # highly variable genes
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes, flavor=flavor)
        self.adata = self.adata[:, self.adata.var['highly_variable']]
