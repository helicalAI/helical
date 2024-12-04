from helical.models.base_models import HelicalRNAModel
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
from helical.utils.downloader import Downloader
import pickle
from transformers import BertForMaskedLM
from helical.models.geneformer.geneformer_utils import get_embs,quant_layers
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from helical.models.geneformer.geneformer_config import GeneformerConfig
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids
from datasets import Dataset
from typing import Optional
from accelerate import Accelerator

LOGGER = logging.getLogger(__name__)
class Geneformer(HelicalRNAModel):
    """Geneformer Model. 
    The Geneformer Model is a transformer-based model that can be used to extract gene embeddings from single-cell RNA-seq data. Both versions are made available through this interface.
    Both versions of Geneformer (v1 and v2) have different sub-models with varying numbers of layers, context size and pretraining set. The available models are the following:

    Version 1.0:
    - gf-12L-30M-i2048
    - gf-6L-30M-i2048

    Version 2.0:
    - gf-12L-95M-i4096
    - gf-12L-95M-i4096-CLcancer
    - gf-20L-95M-i4096

    For a detailed explanation of the differences between these models and versions, please refer to the Geneformer model card: https://helical.readthedocs.io/en/latest/docs/Geneformer.html

    Example
    -------
    ```python
     from helical.models import Geneformer, GeneformerConfig
     import anndata as ad
     
     # For Version 2.0
     geneformer_config_v2 = GeneformerConfig(model_name="gf-12L-95M-i4096", batch_size=10)
     geneformer_v2 = Geneformer(configurer=geneformer_config_v2)
    
     # You can use other model names in the config, such as:
     # "gf-12L-30M-i2048" (Version 1.0)
     # "gf-12L-95M-i4096-CLcancer" (Version 2.0, Cancer-tuned)
     # "gf-20L-95M-i4096" (Version 2.0, 20-layer model)
     
     # Example usage for base pretrained model (for general transcriptomic analysis, v1 and v2)
     ann_data = ad.read_h5ad("general_dataset.h5ad")
     dataset = geneformer_v2.process_data(ann_data)
     embeddings = geneformer_v2.get_embeddings(dataset)
     print("Base model embeddings shape:", embeddings.shape)
    
     # Example usage for cancer-tuned model (for cancer-specific analysis)
     cancer_ann_data = ad.read_h5ad("cancer_dataset.h5ad")
     cancer_dataset = geneformer_v2_cancer.process_data(cancer_ann_data)
     cancer_embeddings = geneformer_v2_cancer.get_embeddings(cancer_dataset)
     print("Cancer-tuned model embeddings shape:", cancer_embeddings.shape)
    ```

    Parameters
    ----------
    configurer : GeneformerConfig, optional, default = default_configurer
        The model configration

    Notes
    -----
    The first version of the model is published in this <a href="https://www.nature.com/articles/s41586-023-06139-9.epdf?sharing_token=u_5LUGVkd3A8zR-f73lU59RgN0jAjWel9jnR3ZoTv0N2UB4yyXENUK50s6uqjXH69sDxh4Z3J4plYCKlVME-W2WSuRiS96vx6t5ex2-krVDS46JkoVvAvJyWtYXIyj74pDWn_DutZq1oAlDaxfvBpUfSKDdBPJ8SKlTId8uT47M%3D">Nature Paper</a>. 
    The second version of the model is available at <a href="https://pubmed.ncbi.nlm.nih.gov/39229018/">NIH</a>.
    We use the implementation from the <a href="https://huggingface.co/ctheodoris/Geneformer/tree/main">Geneformer</a> repository.

    """
    default_configurer = GeneformerConfig()
    def __init__(self, configurer: GeneformerConfig = default_configurer) -> None:
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config
        self.files_config = configurer.files_config
        self.device = self.config['device']

        downloader = Downloader()
        for file in self.configurer.list_of_files_to_download:
            downloader.download_via_name(file)

        self.model =  BertForMaskedLM.from_pretrained(self.files_config['model_files_dir'], output_hidden_states=True, output_attentions=False)
        self.model.eval()
        self.model = self.model.to(self.device)


        self.layer_to_quant = quant_layers(self.model) + self.config['emb_layer']
        self.emb_mode = self.config['emb_mode']
        self.forward_batch_size = self.config['batch_size']
        
        if self.config['accelerator']:
            self.accelerator = Accelerator(project_dir=self.configurer.model_dir)
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None

        # load token dictionary (Ensembl IDs:token)
        with open(self.files_config["token_path"], "rb") as f:
            self.gene_token_dict = pickle.load(f)
            self.pad_token_id = self.gene_token_dict.get("<pad>")

        self.tk = TranscriptomeTokenizer(custom_attr_name_dict=self.config["custom_attr_name_dict"],
                                         nproc=self.config['nproc'], 
                                         model_input_size=self.config["input_size"],
                                         special_token=self.config["special_token"],
                                         gene_median_file = self.files_config["gene_median_path"], 
                                         token_dictionary_file = self.files_config["token_path"],
                                         gene_mapping_file = self.files_config["ensembl_dict_path"],
                                        )
        
        LOGGER.info(f"Model finished initializing.")
        
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

        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        # map gene symbols to ensemble ids if provided
        if gene_names != "ensembl_id":
            if (adata.var[gene_names].str.startswith("ENS").all()) or (adata.var[gene_names].str.startswith("None").any()):
                message = "It seems an anndata with 'ensemble ids' and/or 'None' was passed. " \
                "Please set gene_names='ensembl_id' and remove 'None's to skip mapping."
                LOGGER.info(message)
                raise ValueError(message)
            adata = map_gene_symbols_to_ensembl_ids(adata, gene_names)

        
        tokenized_cells, cell_metadata = self.tk.tokenize_anndata(adata)

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
