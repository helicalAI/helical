import os
import scanpy as sc
from helical.models.helical import HelicalBaseModel
from helical.models.scgpt.scgpt_config import scGPTConfig
import numpy as np
from anndata import AnnData
import logging
from typing import Literal
from accelerate import Accelerator
from helical.models.scgpt.scgpt_utils import load_model, get_embedding
from helical.services.downloader import Downloader

LOGGER = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class scGPT(HelicalBaseModel):
    """scGPT Model. 
        The scGPT Model is a transformer-based model that can be used to extract gene embeddings from single-cell RNA-seq data.
        Currently we load the continous pre-training model from the scGPT repository as default model which works best on zero-shot tasks.


        Example
        -------
        >>> from helical.models import scGPT,scGPTConfig
        >>> import anndata as ad
        >>> scgpt_config=scGPTConfig(batch_size=10)
        >>> scgpt = scGPT(configurer=scgpt_config)
        >>> ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
        >>> dataset = scgpt.process_data(ann_data[:100])
        >>> embeddings = scgpt.get_embeddings(dataset)


        Parameters
        ----------
        configurer : scGPTConfig, optional, default = configurer
            The model configuration.

        Returns
        -------
        None

        Notes
        -----
        We use the implementation from this `repository <https://github.com/bowang-lab/scGPT>`_ , which comes from the original authors. You can find the description of the method in this `paper <https://www.nature.com/articles/s41592-024-02201-0>`_.
        """
    configurer = scGPTConfig()

    def __init__(self, configurer: scGPTConfig = configurer) -> None:
          
        super().__init__()
        self.config = configurer.config
        
        downloader = Downloader()
        for file in self.config["list_of_files_to_download"]:
            downloader.download_via_name(file)

        self.model, self.vocab = load_model(self.config)
        
        if self.config["accelerator"]:
            self.accelerator = Accelerator(project_dir=self.config["model_path"].parent, cpu=self.config["accelerator"]["cpu"])
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None
        LOGGER.info(f"Model finished initializing.")
        
    def get_embeddings(self, data: AnnData) -> np.array:
        """Gets the gene embeddings
        
        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        LOGGER.info(f"Inference started:")
        # The extracted embedding is stored in the `X_scGPT` field of `obsm` in AnnData.
        # for local development, only get embeddings for the first 100 entries

        embeddings = get_embedding(data,
            model = self.model,
            vocab = self.vocab,
            batch_size=self.config["batch_size"],
            model_configs=self.config,
            gene_col=self.gene_column_name,
            device=self.config["device"])
        
        return embeddings
    
    def process_data(self, 
                     adata: AnnData, 
                     gene_column_name: str = "gene_symbols", 
                     fine_tuning: bool = False,
                     n_top_genes: int = 1800, 
                     flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = "seurat_v3") -> AnnData:
        """Processes the data for the scGPT model

        Parameters 
        ----------
        data : AnnData
            The AnnData object containing the data to be processed. 
            The Anndata requires the expression counts as the data matrix and the column with the gene symbols is defined by the argument gene_column_name.
        gene_column_name: str, optional, default = "gene_symbols"
            The name of the column containing the genes in the data.
        fine_tuning: bool, optional, default = False
            If you intend to use the data to fine-tune the model on a downstream task, set this to True.
        n_top_genes: int, optional, default = 1800
           Only taken into account if you use the dataset for fine-tuning the model. Number of highly-variable genes to keep. Mandatory if flavor='seurat_v3'.
        flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"], optional, default = "seurat_v3",
            Only taken into account if you use the dataset for fine-tuning the model.
            Choose the flavor for identifying highly variable genes. 
            For the dispersion based methods in their default workflows, 
            Seurat passes the cutoffs whereas Cell Ranger passes n_top_genes.

        Returns
        -------
        AnnData
            The processed AnnData object
        """
        self.gene_column_name = gene_column_name
        self.adata = adata
        # self.adata.var[self.gene_column_name] = self.adata.var.index.values

        if fine_tuning:
            # Preprocess the dataset and select `N_HVG` highly variable genes for downstream analysis.
            sc.pp.normalize_total(self.adata, target_sum=1e4)
            sc.pp.log1p(self.adata)

            # highly variable genes
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes, flavor=flavor)
            self.adata = self.adata[:, self.adata.var['highly_variable']]
        return self.adata
