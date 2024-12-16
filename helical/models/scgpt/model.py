import os
from typing import Literal
import scanpy as sc
from helical.models.base_models import HelicalRNAModel
from helical.models.scgpt.scgpt_config import scGPTConfig
import numpy as np
import pandas as pd
from anndata import AnnData
import logging
from accelerate import Accelerator
from helical.models.scgpt.scgpt_utils import load_model
from helical.models.scgpt.dataset import Dataset
from helical.utils.downloader import Downloader
from helical.models.scgpt.data_collator import DataCollator
from torch.utils.data import DataLoader, SequentialSampler
import torch
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class scGPT(HelicalRNAModel):
    """
    scGPT Model.
    
    The scGPT Model is a transformer-based model that can be used to extract gene embeddings from single-cell RNA-seq data.
    Currently we load the continous pre-training model from the scGPT repository as default model which works best on zero-shot tasks.


    Example
    -------
    ```python
    from helical import scGPT,scGPTConfig
    from datasets import load_dataset
    from helical.utils import get_anndata_from_hf_dataset
    import anndata as ad

    scgpt_config=scGPTConfig(batch_size=10)
    scgpt = scGPT(configurer=scgpt_config)

    hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:25%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    ann_data = get_anndata_from_hf_dataset(hf_dataset)

    dataset = scgpt.process_data(ann_data[:100])

    embeddings = scgpt.get_embeddings(dataset)
    print("scGPT embeddings: ", embeddings[:10])
    ```

    Parameters
    ----------
    configurer : scGPTConfig, optional, default=configurer
        The model configuration.

    Notes
    -----
    We use the implementation from [this repository](https://github.com/bowang-lab/scGPT), which comes from the original authors. You can find the description of the method in [this paper](https://www.nature.com/articles/s41592-024-02201-0).
    """
    configurer = scGPTConfig()

    def __init__(self, configurer: scGPTConfig = configurer) -> None:
          
        super().__init__()
        self.config = configurer.config
        
        downloader = Downloader()
        for file in self.config["list_of_files_to_download"]:
            downloader.download_via_name(file)

        self.model, self.vocab = load_model(self.config)

        self.model.eval()
        
        if self.config["accelerator"]:
            self.accelerator = Accelerator(project_dir=self.config["model_path"].parent)
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None
        LOGGER.info(f"Model finished initializing.")
        
    def get_embeddings(self, dataset: Dataset) -> np.array:
        """Gets the gene embeddings

        Parameters 
        ----------
        dataset : Dataset
            The processed dataset to get the embeddings from.

        Returns
        -------
        np.ndarray | List[pd.Series]
            The embeddings produced by the model. 
            The return type depends on the `emb_mode` parameter in the configuration.
            If `emb_mode` is set to "gene", the embeddings are returned as a list of pd.Series which contain a mapping of gene_name:embedding for each cell.
        """
        LOGGER.info(f"Inference started:")
                
        # fix seeds
        np.random.seed(self.config["binning_seed"])
        torch.manual_seed(self.config["binning_seed"])
        
        self.model.eval()

        try:
            use_batch_labels = dataset.batch_ids is not None
        except:
            use_batch_labels = False
            
        collator = DataCollator(
            do_padding=True,
            pad_token_id=self.vocab[self.config["pad_token"]],
            pad_value=self.config["pad_value"],
            do_mlm=False,
            do_binning=True,
            max_length=self.config["MAX_LENGTH"],
            sampling=True,
            keep_first_n_tokens=1,
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            sampler=SequentialSampler(dataset),
            collate_fn=collator,
            drop_last=False,
            pin_memory=True,
        )

        device = next(self.model.parameters()).device

        if self.config["emb_mode"] == "gene":
            self.id_gene_dict = {i: gene for i, gene in enumerate(self.vocab.get_itos())}
        
        resulting_embeddings = []

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True): #torch.autocast(device_type=str(device),enabled=True): # torch.cuda.amp.autocast(enabled=True):
            for data_dict in tqdm(data_loader, desc="Embedding cells"):
                input_gene_ids = data_dict["gene"].to(device)

                src_key_padding_mask = input_gene_ids.eq(
                    self.vocab[self.config["pad_token"]]
                )
                embeddings = self.model._encode(
                    input_gene_ids,
                    data_dict["expr"].to(device),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=data_dict["batch_labels"].to(device)
                    if use_batch_labels
                    else None,
                )

                resulting_embeddings.extend(self._compute_embeddings_depending_on_mode(embeddings, data_dict))

        resulting_embeddings = self._normalize_embeddings(resulting_embeddings)
        return resulting_embeddings
        
    def _normalize_embeddings(self, resulting_embeddings: torch.tensor) -> np.ndarray:
        """
        Divides each element of each embedding by the norm of that embedding
        """
        if self.config["emb_mode"] != "gene":
            resulting_embeddings = resulting_embeddings / np.linalg.norm(resulting_embeddings, axis=1, keepdims=True)
        else:
            for series in resulting_embeddings:
                for gene in series.keys():
                    series[gene] = series[gene] / np.linalg.norm(series[gene])   

        return resulting_embeddings

    def _compute_embeddings_depending_on_mode(self, embeddings: torch.tensor, data_dict: dict) -> np.ndarray:
        """
        Compute the embeddings depending on the mode set in the configuration.

        Parameters
        ----------
        embeddings : torch.tensor
            The embeddings to be processed.
        data_dict : dict
            The data dictionary containing the data to be processed.
        
        Returns
        -------
        np.ndarray
            The embeddings corresponding to the mode selected
        """
        if self.config["emb_mode"] == "cls":
            embeddings = embeddings[:, 0, :]  # get the <cls> position embedding
            embeddings = embeddings.cpu().numpy()
            return embeddings

        elif self.config["emb_mode"] == "cell":
            embeddings = embeddings[:, 1:, :] # get all embeddings except the <cls> position
            embeddings = torch.mean(embeddings, dim=1) # mean embeddings to get cell embedding
            embeddings = embeddings.cpu().numpy()
            return embeddings
        
        elif self.config["emb_mode"] == "gene":
            embeddings = embeddings[:, 1:, :].cpu().numpy() # get all embeddings except the <cls> position
            gene_ids = data_dict["gene"].cpu().numpy()

            # create a dictionary with gene name to gene embedding mappings and create pd series for each cell in batch
            batch_embeddings = []
            for i, embedding in enumerate(embeddings):
                dict = {}
                for j, gene in enumerate(embedding, 1):
                    if data_dict["gene"][i][j] != self.vocab[self.config["pad_token"]]:
                        dict[self.id_gene_dict[gene_ids[i][j]]] = gene
                
                batch_embeddings.append(pd.Series(dict))
            
            return batch_embeddings
    
    def process_data(self,
                     adata: AnnData, 
                     gene_names: str = "index", 
                     fine_tuning: bool = False,
                     n_top_genes: int = 1800, 
                     flavor: Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"] = "seurat_v3",
                     use_batch_labels: bool = False,
                     use_raw_counts: bool = True
    ) -> Dataset:
        """
        Processes the data for the scGPT model.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. 
            The AnnData requires the expression counts as the data matrix, and the column with 
            the gene symbols is defined by the argument `gene_names`.
        gene_names : str, optional, default="index"
            The column in `adata.var` that contains the gene names. Default is to use the index column.
        fine_tuning : bool, optional, default=False
            If you intend to use the data to fine-tune the model on a downstream task, set this to True.
        n_top_genes : int, optional, default=1800
            Only taken into account if you use the dataset for fine-tuning the model. 
            Number of highly-variable genes to keep. Mandatory if `flavor='seurat_v3'`.
        flavor : Literal["seurat", "cell_ranger", "seurat_v3", "seurat_v3_paper"], optional, default="seurat_v3"
            Only taken into account if you use the dataset for fine-tuning the model. 
            Choose the flavor for identifying highly variable genes. 
            For the dispersion-based methods in their default workflows, 
            Seurat passes the cutoffs whereas Cell Ranger passes `n_top_genes`.
        use_batch_labels : bool, optional, default=False
            Whether to use batch labels. Defaults to False.
        use_raw_counts : bool, optional, default=True
            Whether to use raw counts or not.

        Returns
        -------
        Dataset
            The processed dataset.
        """

 
        self.ensure_data_validity(adata, gene_names, use_batch_labels, use_raw_counts)
        self.gene_names = gene_names
        if fine_tuning:
            # Preprocess the dataset and select `N_HVG` highly variable genes for downstream analysis.
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # highly variable genes
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor=flavor)
            adata = adata[:, adata.var['highly_variable']]

        # filtering
        adata.var["id_in_vocab"] = [ self.vocab[gene] if gene in self.vocab else -1 for gene in adata.var[self.gene_names] ]
        LOGGER.info(f"Filtering out {np.sum(adata.var['id_in_vocab'] < 0)} genes to a total of {np.sum(adata.var['id_in_vocab'] >= 0)} genes with an id in the scGPT vocabulary.")
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # Binning will be applied after tokenization. A possible way to do is to use the unified way of binning in the data collator.

        self.vocab.set_default_index(self.vocab["<pad>"])
        genes = adata.var[self.gene_names].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        count_matrix = (adata.X if isinstance(adata.X, np.ndarray) else adata.X.A)

        # gene vocabulary ids
        if gene_ids is None:
            gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

        if use_batch_labels:
            batch_ids = np.array(adata.obs["batch_id"].tolist())

        dataset = Dataset(
            count_matrix, gene_ids, self.vocab, self.config, batch_ids if use_batch_labels else None
        )
        return dataset


    def ensure_data_validity(self, adata: AnnData, gene_names: str, use_batch_labels: bool, use_raw_counts = True) -> None:
        """Checks if the data is eligible for processing by the scGPT model  

        Parameters
        ----------
        data : AnnData
            The AnnData object containing the data to be validated. 
        gene_names : str
            The name of the column containing gene names.
        use_batch_labels : bool
            Wheter to use batch labels.
        use_raw_counts : bool, default = True
            Whether to use raw counts or not.

        Raises
        ------
        KeyError
            If the data is missing column names.
        """
        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        if use_batch_labels:
            if not "batch_id" in adata.obs:
                message = "Data must have the 'obs' key 'batch_id' to be processed by the scGPT model."
                LOGGER.error(message)
                raise KeyError(message)