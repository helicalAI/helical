from helical.models.base_models import HelicalRNAModel
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
from helical.utils.downloader import Downloader
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModel
from helical.models.geneformer.geneformer_utils import get_embs,quant_layers
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from helical.models.genept.genept_config import GenePTConfig
from helical.utils.mapping import map_ensembl_ids_to_gene_symbols
from datasets import Dataset
from typing import Optional
from accelerate import Accelerator
import logging
import scanpy as sc
import torch
import json
import pandas as pd
from tqdm import tqdm 
logger = logging.getLogger(__name__)

class GenePT(HelicalRNAModel):
    """GenePT Model. 
    
    ```

    Parameters
    ----------
    configurer : GenePTConfig, optional, default = default_configurer
        The model configuration

    Notes
    -----


    """
    default_configurer = GenePTConfig()
    def __init__(self, configurer: GenePTConfig = default_configurer):
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])

        with open("/home/maxime/git/helical/helical/models/genept/genept_embeddings.json","r") as f:
            self.embeddings = json.load(f)
    
        # self.model.post_init()
        logger.info("GenePT initialized successfully.")

    def process_data(self, 
                     adata: AnnData,  
                     gene_names: str = "index", 
                     output_path: Optional[str] = None,
                     use_raw_counts: bool = True,
                     ) -> Dataset:   
        """
        Processes the data for the Geneformer model.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. Geneformer uses Ensembl IDs to identify genes 
            and currently supports only human genes. If the AnnData object already has an 'ensembl_id' column, 
            the mapping step can be skipped.
        gene_names : str, optional, default="index"
            The column in `adata.var` that contains the gene names. If set to a value other than "ensembl_id", 
            the gene symbols in that column will be mapped to Ensembl IDs using the 'pyensembl' package, 
            which retrieves mappings from the Ensembl FTP server and loads them into a local database.
            - If set to "index", the index of the AnnData object will be used and mapped to Ensembl IDs.
            - If set to "ensembl_id", no mapping will occur.
            Special case:
                If the index of `adata` already contains Ensembl IDs, setting this to "index" will result in 
                invalid mappings. In such cases, create a new column containing Ensembl IDs and pass "ensembl_id" 
                as the value of `gene_names`.
        output_path : str, optional, default=None
            If specified, saves the tokenized dataset to the given output path.
        use_raw_counts : bool, optional, default=True
            Determines whether raw counts should be used.

        Returns
        -------
        Dataset
            The tokenized dataset in the form of a Huggingface Dataset object.
        """

        # self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        # map gene symbols to ensemble ids if provided
        if gene_names == "ensembl_id":
            if (adata.var[gene_names].str.startswith("ENS").all()) or (adata.var[gene_names].str.startswith("None").any()):
                message = "It seems an anndata with 'ensemble ids' and/or 'None' was passed. " \
                "Please set gene_names='ensembl_id' and remove 'None's to skip mapping."
                logger.info(message)
                raise ValueError(message)
            adata = map_ensembl_ids_to_gene_symbols(adata, gene_names)


        sc.pp.highly_variable_genes(adata,n_top_genes=1000,flavor='seurat_v3')
        sc.pp.normalize_total(adata,target_sum=1e4)
        sc.pp.log1p(adata)

        genes_names = adata.var_names[adata.var['highly_variable']].tolist()

        adata = adata[:,genes_names]
        # input_ids = self.tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True, return_tensors="pt",truncation=True)
        
        return adata
        
    def get_text_embeddings(self, dataset: AnnData) -> np.array:
        """Gets the gene embeddings from the Geneformer model   

        Parameters
        ----------
        dataset : Dataset
            The tokenized dataset containing the processed data

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        logger.info(f"Inference started:")
        # Generate a response 
        raw_embeddings = dataset.var_names
        weights = []
        count_missed = 0
        gene_list = []
        for i,emb in enumerate(raw_embeddings):
            gene = self.embeddings.get(emb.upper(),None)
            if gene is not None:
                weights.append(gene['embeddings'])
                gene_list.append(emb)
            else:
                count_missed += 1
        logger.info("Couln't find {} genes in embeddings".format(count_missed))

        weights = torch.Tensor(weights)
        # weights = torch.rand(weights.shape)
        embeddings = torch.matmul(torch.Tensor(dataset[:,gene_list].X.toarray()),weights)
        # embeddings = (embeddings/(np.linalg.norm(embeddings,axis=1)).reshape(-1,1))
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) --> Same as above
        return embeddings
    
    def get_embeddings(self, dataset: AnnData) -> np.array:
        """Gets the gene embeddings from the Geneformer model   

        Parameters
        ----------
        dataset : Dataset
            The tokenized dataset containing the processed data

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        logger.info(f"Inference started:")
        # Generate a response 
        embeddings = self.get_text_embeddings(dataset)
        embeddings = (embeddings/(np.linalg.norm(embeddings,axis=1)).reshape(-1,1))
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1) --> Same as above
        return embeddings