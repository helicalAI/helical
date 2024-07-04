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
from datasets import Dataset
from typing import Optional
from accelerate import Accelerator
import pickle as pkl

LOGGER = logging.getLogger(__name__)
class Geneformer(HelicalRNAModel):
    """Geneformer Model. 
    The Geneformer Model is a transformer-based model that can be used to extract gene embeddings from single-cell RNA-seq data. 

    Example
    -------
    >>> from helical.models import Geneformer,GeneformerConfig
    >>> import anndata as ad
    >>> geneformer_config=GeneformerConfig(batch_size=10)
    >>> geneformer = Geneformer(configurer=geneformer_config)
    >>> ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
    >>> dataset = geneformer.process_data(ann_data[:100])
    >>> embeddings = geneformer.get_embeddings(dataset)

    Parameters
    ----------
    configurer : GeneformerConfig, optional, default = default_configurer
        The model configration

    Returns
    -------
    None

    Notes
    -----
    It has been published in this `Nature Paper <https://www.nature.com/articles/s41586-023-06139-9.epdf?sharing_token=u_5LUGVkd3A8zR-f73lU59RgN0jAjWel9jnR3ZoTv0N2UB4yyXENUK50s6uqjXH69sDxh4Z3J4plYCKlVME-W2WSuRiS96vx6t5ex2-krVDS46JkoVvAvJyWtYXIyj74pDWn_DutZq1oAlDaxfvBpUfSKDdBPJ8SKlTId8uT47M%3D>`_. 
    We use the implementation from the `Geneformer <https://huggingface.co/ctheodoris/Geneformer/tree/main>`_ repository.

    """
    default_configurer = GeneformerConfig()
    def __init__(self, configurer: GeneformerConfig = default_configurer) -> None:
        super().__init__()
        self.config = configurer

        self.device = self.config.device

        downloader = Downloader()
        for file in self.config.list_of_files_to_download:
            downloader.download_via_name(file)

        self.model =  BertForMaskedLM.from_pretrained(self.config.model_dir / self.config.model_name, output_hidden_states=True, output_attentions=False)
        self.model.eval()#.to("cuda:0")
        self.model = self.model.to(self.device)

        self.layer_to_quant = quant_layers(self.model) + self.config.emb_layer
        self.emb_mode = self.config.emb_mode
        self.forward_batch_size = self.config.batch_size
        
        if self.config.accelerator:
            self.accelerator = Accelerator(project_dir=self.config.model_dir)
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None
        LOGGER.info(f"Model finished initializing.")
        
    def process_data(self, 
                     adata: AnnData,  
                     gene_column_name: str = "ensembl_id", 
                     nproc: int = 1, 
                     output_path: Optional[str] = None) -> Dataset:   
        """Processes the data for the UCE model

        Parameters 
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. It is important to note that the Geneformer uses Ensembl IDs to identify genes.
            Currently the Geneformer only supports human genes.
            If you already have the ensembl_id column, you can skip the mapping step.
        gene_column_name: str, optional, default = "ensembl_id"
            The column in adata.var that contains the gene names. If you set this string to something other than "ensembl_id", 
            we will map the gene symbols in that column to Ensembl IDs with a mapping taken from the `Ensembl Website https://www.ensembl.org/`.
            If it is left at "ensembl_id", there will be no mapping.
            If this variable is set to "index", the index of the AnnData object will be used and mapped to Ensembl IDs.
            In the special case where the data has Ensemble IDs as the index, and you pass "index". This would result in invalid mappings.
            In that case, it is recommended to create a new column with the Ensemble IDs in the data and pass "ensembl_id" as the gene_column_name.
        nproc : int, optional, default = 1
            Number of processes to use for dataset processing.
        output_path : str, default = None
            Whether to save the tokenized dataset to the specified output_path.


        Returns
        -------
        Dataset
            The tokenized dataset in the form of a Hugginface Dataset object.
            
        """ 
        self.check_data_validity(adata, gene_column_name)
        self.adata = adata

        files_config = {
            "mapping_path": self.config.model_dir / "human_gene_to_ensemble_id.pkl",
            "gene_median_path": self.config.model_dir / "gene_median_dictionary.pkl",
            "token_path": self.config.model_dir / "token_dictionary.pkl"
        }

        # map gene symbols to ensemble ids if provided
        if gene_column_name != "ensembl_id":          
            mappings = pkl.load(open(files_config["mapping_path"], 'rb'))
            self.adata.var['ensembl_id'] = self.adata.var[gene_column_name].apply(lambda x: mappings.get(x,{"id":None})['id'])

        # load token dictionary (Ensembl IDs:token)
        with open(files_config["token_path"], "rb") as f:
            self.gene_token_dict = pickle.load(f)
            self.pad_token_id = self.gene_token_dict.get("<pad>")

        self.tk = TranscriptomeTokenizer(nproc=nproc, 
                                         gene_median_file = files_config["gene_median_path"], 
                                         token_dictionary_file = files_config["token_path"])

        tokenized_cells, cell_metadata =  self.tk.tokenize_anndata(self.adata)
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
            self.device
        ).cpu().detach().numpy()

        # save the embeddings in the adata object
        self.adata.obsm[self.config.embed_obsm_name] = embeddings
        return embeddings


    def check_data_validity(self, adata: AnnData, gene_column_name: str) -> None:
        """Checks if the data is eligible for processing by the Geneformer model  

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed.
        gene_column_name: str
            The column in adata.var that contains the gene names.

        Raises
        ------
        KeyError
            If the data is missing column names.
        """
        self.check_rna_data_validity(adata, gene_column_name)

        if not 'n_counts' in adata.obs.columns.to_list():
            message = f"Data must have the 'obs' keys 'n_counts' to be processed by the Geneformer model."
            LOGGER.error(message)
            raise KeyError(message)