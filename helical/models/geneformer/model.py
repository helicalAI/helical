from helical.models.base_models import HelicalRNAModel
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
from helical.services.downloader import Downloader
import pickle
from transformers import BertForMaskedLM
from helical.models.geneformer.geneformer_utils import get_embs,quant_layers
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from helical.models.geneformer.geneformer_config import GeneformerConfig
from helical.services.mapping import map_gene_symbols_to_ensembl_ids
from datasets import Dataset
from typing import Optional
from accelerate import Accelerator
import tempfile
import os 

LOGGER = logging.getLogger(__name__)
class Geneformer(HelicalRNAModel):
    """Geneformer Model. 
    The Geneformer Model is a transformer-based model that can be used to extract gene embeddings from single-cell RNA-seq data. Both versions are made available through this interface.
    For a detailed explanation of the difference between both versions, please refer to the model card, here: https://helical.readthedocs.io/en/latest/docs/Geneformer.html

    Example
    -------
    >>> from helical.models import Geneformer, GeneformerConfig
    >>> import anndata as ad
    >>> 
    >>> # For Version 1.0
    >>> geneformer_config_v1 = GeneformerConfig(model_name="gf-12L-30M-i2048", batch_size=10)
    >>> geneformer_v1 = Geneformer(configurer=geneformer_config_v1)
    >>> 
    >>> # For Version 2.0
    >>> geneformer_config_v2 = GeneformerConfig(model_name="gf-12L-95M-i4096", batch_size=10)
    >>> geneformer_v2 = Geneformer(configurer=geneformer_config_v2)
    >>> 
    >>> # Example usage (same for both versions)
    >>> ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
    >>> dataset = geneformer_v2.process_data(ann_data[:100])
    >>> embeddings = geneformer_v2.get_embeddings(dataset)

    Parameters
    ----------
    configurer : GeneformerConfig, optional, default = default_configurer
        The model configration

    Returns
    -------
    None

    Notes
    -----
    The first version of the model is published in this `Nature Paper <https://www.nature.com/articles/s41586-023-06139-9.epdf?sharing_token=u_5LUGVkd3A8zR-f73lU59RgN0jAjWel9jnR3ZoTv0N2UB4yyXENUK50s6uqjXH69sDxh4Z3J4plYCKlVME-W2WSuRiS96vx6t5ex2-krVDS46JkoVvAvJyWtYXIyj74pDWn_DutZq1oAlDaxfvBpUfSKDdBPJ8SKlTId8uT47M%3D>`_. 
    The second version of the model is available at https://www.biorxiv.org/content/10.1101/2024.08.16.608180v1.full.pdf
    We use the implementation from the `Geneformer <https://huggingface.co/ctheodoris/Geneformer/tree/main>`_ repository.

    """
    default_configurer = GeneformerConfig()
    def __init__(self, configurer: GeneformerConfig = default_configurer) -> None:
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config
        self.files_config = configurer.files_config
        self.device = self.config['device']

        downloader = Downloader()
        for file in configurer.list_of_files_to_download:
            downloader.download_via_name(file)

        self.model =  BertForMaskedLM.from_pretrained(self.files_config['model_files_dir'], output_hidden_states=True, output_attentions=False)
        self.model.eval()
        self.model = self.model.to(self.device)


        self.layer_to_quant = quant_layers(self.model) + self.config['emb_layer']
        self.emb_mode = self.config['emb_mode']
        self.forward_batch_size = self.config['batch_size']
        
        if self.config['accelerator']:
            self.accelerator = Accelerator(project_dir=configurer.model_dir)
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None
        LOGGER.info(f"Model finished initializing.")
        
    def process_data(self, 
                     adata: AnnData,  
                     gene_names: str = "index", 
                     nproc: int = 1, 
                     output_path: Optional[str] = None,
                     custom_attr_name_dict: Optional[dict] = None) -> Dataset:   
        """Processes the data for the Geneformer model

        Parameters 
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. It is important to note that the Geneformer uses Ensembl IDs to identify genes.
            Currently the Geneformer only supports human genes.
            If you already have the ensembl_id column, you can skip the mapping step.
        gene_names: str, optional, default = "index"
            The column in adata.var that contains the gene names. If you set this string to something other than "ensembl_id", 
            we will map the gene symbols in that column to Ensembl IDs with a mapping taken from the 'pyensembl' package, which ultimately gets the mappings from 
            the Ensembl FTP server and loads them into a local database.
            If this variable is left at "index", the index of the AnnData object will be used and mapped to Ensembl IDs.
            If it is changes to "ensembl_id", there will be no mapping.
            In the special case where the data has Ensemble IDs as the index, and you pass "index". This would result in invalid mappings.
            In that case, it is recommended to create a new column with the Ensemble IDs in the data and pass "ensembl_id" as the gene_names.
        nproc : int, optional, default = 1
            Number of processes to use for dataset processing.
        output_path : str, default = None
            Whether to save the tokenized dataset to the specified output_path.
        custom_attr_name_dict : dict, optional, default = None
            A dictionary that contains the names of the custom attributes to be added to the dataset. 
            The keys of the dictionary are the names of the custom attributes, and the values are the names of the columns in adata.obs. 
            For example, if you want to add a custom attribute called "cell_type" to the dataset, you would pass custom_attr_name_dict = {"cell_type": "cell_type"}.
            If you do not want to add any custom attributes, you can leave this parameter as None.

        Returns
        -------
        Dataset
            The tokenized dataset in the form of a Hugginface Dataset object.
            
        """ 
        self.ensure_data_validity(adata, gene_names)

        # map gene symbols to ensemble ids if provided
        if gene_names != "ensembl_id":
            if (adata.var[gene_names].str.startswith("ENSG").all()) or (adata.var[gene_names].str.startswith("None").any()):
                message = "It seems an anndata with 'ensemble ids' and/or 'None' was passed. " \
                "Please set gene_names='ensembl_id' and remove 'None's to skip mapping."
                LOGGER.info(message)
                raise ValueError(message)
            adata = map_gene_symbols_to_ensembl_ids(adata, gene_names)

        # load token dictionary (Ensembl IDs:token)
        with open(self.files_config["token_path"], "rb") as f:
            self.gene_token_dict = pickle.load(f)
            self.pad_token_id = self.gene_token_dict.get("<pad>")

        self.tk = TranscriptomeTokenizer(custom_attr_name_dict=custom_attr_name_dict,
                                         nproc=nproc, 
                                         model_input_size=self.config["input_size"],
                                         special_token=self.config["special_token"],
                                         gene_median_file = self.files_config["gene_median_path"], 
                                         token_dictionary_file = self.files_config["token_path"],
                                         gene_mapping_file = self.files_config["ensembl_dict_path"],
                                        )
        # Pass the path to the ann_data file instead of passing the anndata object
        with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
            adata.write(tmp.name)
            tmp_path = tmp.name
        try:
            tokenized_cells, cell_metadata = self.tk.tokenize_anndata(tmp_path)
        finally:
            os.unlink(tmp_path) 

        # tokenized_cells, cell_metadata =  self.tk.tokenize_anndata(adata)
        tokenized_dataset = self.tk.create_dataset(tokenized_cells, cell_metadata, use_generator=False)
        
        if output_path:
            output_path = Path(output_path).with_suffix(".dataset")
            tokenized_dataset.save_to_disk(output_path)
        return tokenized_dataset

    def get_embeddings(self, dataset: Dataset) -> np.array:
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
        LOGGER.info(f"Inference started:")
        embeddings = get_embs(
            self.model,
            dataset,
            self.emb_mode,
            self.layer_to_quant,
            self.pad_token_id,
            self.forward_batch_size,
            self.gene_token_dict,
            self.device
        ).cpu().detach().numpy()

        return embeddings


    def ensure_data_validity(self, adata: AnnData, gene_names: str) -> None:
        """Ensure that the data is eligible for processing by the Geneformer model. This checks 
        if the data contains the gene_names, and sets the total_counts column in adata.obs.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed.
        gene_names: str
            The column in adata.var that contains the gene names.

        Raises
        ------
        KeyError
            If the data is missing column names.
        """
        self.ensure_rna_data_validity(adata, gene_names)
