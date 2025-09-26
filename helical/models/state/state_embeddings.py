'''
This package is based on STATE by Adduri et al., 
available at https://www.biorxiv.org/content/10.1101/2025.06.26.661135v1. 
Licensed under CC BY-NC-SA 4.0.  
Modifications in this package: 
- Updated API call wrappers 
- Minor code restructuring

'''

import os
import logging
import torch
import anndata
import h5py as h5
import numpy as np
from typing import Optional

from tqdm import tqdm
from torch import nn

from .model_dir.embed_utils import StateEmbeddingModel
from .model_dir.embed_utils import create_dataloader
from .model_dir.embed_utils.utils import get_embedding_cfg, get_precision_config

from helical.models.base_models import HelicalBaseFoundationModel
from helical.models.state.state_config import StateConfig
from helical.utils.downloader import Downloader
from omegaconf import OmegaConf
from helical.utils.converter import convert_to_csr

LOGGER = logging.getLogger(__name__)

class StateEmbed(HelicalBaseFoundationModel):
    """
    State Embedding Model.

    The State Embedding Model is a transformer-based model that can be used to extract 
    cell embeddings from single-cell RNA-seq data. This model 
    leverages pre-trained ESM2 gene embeddings to create rich representations of gene 
    expression data that can be used for various downstream tasks including perturbation 
    prediction and cell state analysis.

    Example
    -------
    ```python
    from helical.models.state import StateEmbed, StateConfig
    import anndata as ad

    config = StateConfig(batch_size=16)
    state_embed = StateEmbed(configurer=config)

    # Process your data
    dataloader = state_embed.process_data(adata)

    # Get embeddings
    embeddings = state_embed.get_embeddings(dataloader)
    print("State embeddings shape:", embeddings.shape)
    ```

    Parameters
    ----------
    configurer : StateConfig, optional, default=None
        The model configuration. If None, uses default StateConfig.

    Notes
    -----
    This model uses protein embeddings from ESM2 and a transformer architecture to 
    create cell embeddings. The model can also perform reverse engineering to 
    reconstruct gene expression from cell embeddings.
    """

    def __init__(self, configurer: StateConfig = None) -> None:
        super().__init__()

        self.model = None
        self.collator = None

        if configurer is None:
            configurer = StateConfig()
        self.config = configurer.config

        downloader = Downloader()
        for file in self.config["embed_files_to_download"]:
            downloader.download_via_name(file)

        self.model_dir = os.path.join(self.config["model_path"], "state_embed")
        self.ckpt_path = os.path.join(self.model_dir, "se600m_model_weights.pt")
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_conf = OmegaConf.load(os.path.join(self.model_dir, "config.yaml"))
        self.batch_size = self.config["batch_size"]

        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.ckpt_path}")
        LOGGER.info(f"Using model checkpoint: {self.ckpt_path}")

        embedding_file = os.path.join(self.model_dir, "protein_embeddings.pt")
        self.protein_embeds = (
            torch.load(embedding_file, weights_only=False, map_location=self.device_type)
            if os.path.exists(embedding_file)
            else None
        )
        self.load_model()

    def load_model(self):
        """
        Load and initialize the State Embedding model.

        This method initializes the transformer model with the configuration parameters,
        loads the pre-trained weights, and sets up the gene embeddings. The model
        is moved to the appropriate device (GPU if available) and set to evaluation mode.

        Raises
        ------
        ValueError
            If the model is already initialized.
        FileNotFoundError
            If the model checkpoint file is not found.
        """
        if self.model:
            raise ValueError("Model already initialized")

        # First, initialize the model with the config
        self.model = StateEmbeddingModel(
            token_dim=self.model_conf.tokenizer.token_dim,  
            d_model=self.model_conf.model.emsize,
            nhead=self.model_conf.model.nhead,
            d_hid=self.model_conf.model.d_hid,
            nlayers=self.model_conf.model.nlayers,
            output_dim=self.model_conf.model.output_dim,
            compiled=self.model_conf.experiment.compiled,
            cfg=self.model_conf,
        )

        # LOGGER.info("number of free parameters: %s", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        loaded_weights = torch.load(self.ckpt_path, weights_only=False)

        # missing_keys are keys that are in the model but NOT in the loaded weights, so they would not be initialized for inference properly.
        missing_keys, _ = self.model.load_state_dict(loaded_weights, strict=False)
        # LOGGER.info(f"Missing keys: {missing_keys}")

        precision = get_precision_config(device_type=self.device_type)
        self.model = self.model.to(precision)

        all_pe = self.protein_embeds or StateEmbed.load_esm2_embeddings(self.model_conf)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))

        self.model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
        self.model.pe_embedding.to(self.model.device, dtype=precision)
        self.model.to(self.device_type)
        self.model.binary_decoder.requires_grad = False
        self.model.eval()

        if self.protein_embeds is None:
            self.protein_embeds = torch.load(
                get_embedding_cfg(self.model_conf).all_embeddings, weights_only=False
            )
        LOGGER.info("Successfully loaded model")

    def process_data(
        self,
        adata: anndata.AnnData,
    ):
        """
        Process AnnData object for embedding generation.

        This method converts the input AnnData object into a format suitable for the
        State Embedding model. It handles data preprocessing, gene column detection,
        and creates a dataloader for efficient batch processing.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object containing single-cell RNA-seq data. The data should
            have gene expression counts in the X matrix and gene names in var.

        Returns
        -------
        DataLoader
            A PyTorch DataLoader object that can be used with get_embeddings() method.
        """

        shape_dict = self.__load_dataset_meta_from_adata(adata)
        adata = convert_to_csr(adata)
        gene_column: Optional[str] = self._auto_detect_gene_column(adata)

        precision = get_precision_config(device_type=self.device_type)

        dataloader = create_dataloader(
            self.model_conf,
            adata=adata,
            adata_name="inference",
            shape_dict=shape_dict,
            data_dir=None,
            shuffle=False,
            protein_embeds=self.protein_embeds,
            precision=precision,
            gene_column=gene_column,
            batch_size=self.batch_size,
        )

        return dataloader

    def get_embeddings(self, dataloader):
        """
        Generate cell embeddings from processed data.

        This method processes the dataloader through the State Embedding model to
        generate cell embeddings. It handles both regular embeddings and dataset-specific
        embeddings, concatenating them if both are available.

        Parameters
        ----------
        dataloader : DataLoader
            The processed dataloader from process_data() method.

        Returns
        -------
        np.ndarray
            A numpy array of shape (n_cells, embedding_dim) containing the cell embeddings.
            If dataset embeddings are available, they are concatenated to the regular embeddings.
        """
        all_embeddings = []
        all_ds_embeddings = []
        for embeddings, ds_embeddings in tqdm(
            self.encode(dataloader), total=len(dataloader), desc="Encoding"
        ):
            all_embeddings.append(embeddings)
            if ds_embeddings is not None:
                all_ds_embeddings.append(ds_embeddings)

        all_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        if len(all_ds_embeddings) > 0:
            all_ds_embeddings = np.concatenate(all_ds_embeddings, axis=0).astype(
                np.float32
            )

            all_embeddings = np.concatenate(
                [all_embeddings, all_ds_embeddings], axis=-1
            )

        return all_embeddings

    def __load_dataset_meta_from_adata(self, adata, dataset_name=None):
        """
        Extract dataset metadata directly from an AnnData object.

        This helper method extracts basic metadata (number of cells and genes) from
        an AnnData object and returns it in the format expected by the model.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object to extract metadata from.
        dataset_name : str, optional, default=None
            Optional name for the dataset. If None, uses 'inference'.

        Returns
        -------
        dict
            Dictionary with dataset name as key and (num_cells, num_genes) as value.
        """
        num_cells = adata.n_obs
        num_genes = adata.n_vars

        if dataset_name is None:
            dataset_name = "inference"

        return {dataset_name: (num_cells, num_genes)}

    def get_gene_embedding(self, genes):
        """
        Get gene embeddings for a list of genes.

        This method retrieves protein embeddings for the specified genes and processes
        them through the gene embedding layer to create gene-specific embeddings.

        Parameters
        ----------
        genes : list
            List of gene names to get embeddings for.

        Returns
        -------
        torch.Tensor
            Tensor containing gene embeddings of shape (n_genes, embedding_dim).
        """
        protein_embeds = [
            self.protein_embeds[x] if x in self.protein_embeds else torch.zeros(5120)
            for x in genes
        ]
        precision = get_precision_config(device_type=self.device_type)
        protein_embeds = torch.stack(protein_embeds).to(
            self.model.device, dtype=precision
        )
        return self.model.gene_embedding_layer(protein_embeds)

    def encode(self, dataloader, rda=None):
        """
        Encode data through the model to generate embeddings.

        This method processes batches of data through the State Embedding model
        to generate cell embeddings and dataset embeddings.

        Parameters
        ----------
        dataloader : DataLoader
            The dataloader containing processed data.
        rda : optional
            Optional parameter for read depth adjustment (currently unused).

        Yields
        ------
        tuple
            Tuple containing (embeddings, dataset_embeddings) for each batch.
        """
        with torch.no_grad():
            precision = get_precision_config(device_type=self.device_type)
            with torch.autocast(device_type=self.device_type, dtype=precision):
                for i, batch in enumerate(dataloader):

                    _, _, _, emb, ds_emb = self.model._compute_embedding_for_batch(
                        batch
                    )
                    embeddings = emb.detach().cpu().float().numpy()

                    ds_emb = self.model.dataset_embedder(ds_emb)
                    ds_embeddings = ds_emb.detach().cpu().float().numpy()

                    yield embeddings, ds_embeddings

    def _auto_detect_gene_column(self, adata):
        """
        Auto-detect the gene column with highest overlap with protein embeddings.

        This method automatically identifies which column in adata.var contains gene names
        that have the best overlap with the available gene embeddings. It checks the
        index and all string columns in var.

        Parameters
        ----------
        adata : anndata.AnnData
            The AnnData object to analyze.

        Returns
        -------
        str or None
            The name of the column with the best gene overlap, or None if using index.
        """
        if self.protein_embeds is None:
            LOGGER.warning(
                "No protein embeddings available for auto-detection, using index"
            )
            return None

        protein_genes = set(self.protein_embeds.keys())
        best_column = None
        best_overlap = 0
        best_overlap_pct = 0

        # Check index first
        if hasattr(adata.var, "index"):
            index_genes = set(adata.var.index)
            overlap = len(protein_genes.intersection(index_genes))
            overlap_pct = overlap / len(index_genes) if len(index_genes) > 0 else 0
            if overlap > best_overlap:
                best_overlap = overlap
                best_overlap_pct = overlap_pct
                best_column = None  # None means use index

        # Check all columns in var
        for col in adata.var.columns:
            if adata.var[col].dtype == "object" or adata.var[col].dtype.name.startswith(
                "str"
            ):
                col_genes = set(adata.var[col].dropna().astype(str))
                overlap = len(protein_genes.intersection(col_genes))
                overlap_pct = overlap / len(col_genes) if len(col_genes) > 0 else 0
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_overlap_pct = overlap_pct
                    best_column = col

        if best_column is None:
            LOGGER.info(
                f"Auto-detected gene column: var.index (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )
        else:
            LOGGER.info(
                f"Auto-detected gene column: var.{best_column} (overlap: {best_overlap}/{len(protein_genes)} protein embeddings, {best_overlap_pct:.1%} of genes)"
            )

        return best_column

    def decode_from_file(
        self, adata_path, emb_key: str, read_depth=None, batch_size=64
    ):
        """
        Decode gene expression from embeddings stored in a file.

        This method reads an h5ad file, extracts embeddings, and reconstructs gene
        expression profiles from the embeddings.

        Parameters
        ----------
        adata_path : str
            Path to the h5ad file containing embeddings.
        emb_key : str
            Key in obsm where embeddings are stored.
        read_depth : float, optional, default=None
            Read depth parameter for reconstruction.
        batch_size : int, optional, default=64
            Batch size for processing.

        Yields
        ------
        np.ndarray
            Reconstructed gene expression profiles for each batch.
        """
        adata = anndata.read_h5ad(adata_path)
        genes = adata.var.index
        yield from self.decode_from_adata(adata, genes, emb_key, read_depth, batch_size)

    @torch.no_grad()
    def decode_from_adata(
        self, adata, genes, emb_key: str, read_depth=None, batch_size=64
    ):
        """
        Decode gene expression from embeddings in AnnData object.

        This method performs reverse engineering by taking pre-computed cell embeddings
        and reconstructing the original gene expression profiles from them.

        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing embeddings.
        genes : list
            List of gene names to reconstruct.
        emb_key : str
            Key in obsm where embeddings are stored.
        read_depth : float, optional, default=None
            Read depth parameter for reconstruction. If None and RDA is enabled,
            defaults to 4.0.
        batch_size : int, optional, default=64
            Batch size for processing.

        Yields
        ------
        np.ndarray
            Reconstructed gene expression profiles for each batch.
        """
        try:
            cell_embs = adata.obsm[emb_key]
        except:
            cell_embs = adata.X

        precision = get_precision_config(device_type=self.device_type)
        cell_embs = torch.Tensor(cell_embs).to(self.model.device, dtype=precision)

        use_rda = getattr(self.model.cfg.model, "rda", False)
        if use_rda and read_depth is None:
            read_depth = 4.0

        gene_embeds = self.get_gene_embedding(genes)
        with torch.autocast(device_type=self.device_type, dtype=precision):
            for i in tqdm(
                range(0, cell_embs.size(0), batch_size),
                total=int(cell_embs.size(0) // batch_size),
            ):
                cell_embeds_batch = cell_embs[i : i + batch_size]
                task_counts = torch.full(
                    (cell_embeds_batch.shape[0],),
                    read_depth,
                    device=self.model.device,
                    dtype=precision,
                )

                ds_emb = cell_embeds_batch[
                    :, -self.model.z_dim_ds :
                ]  # last ten columns are the dataset embeddings
                merged_embs = StateEmbeddingModel.resize_batch(
                    cell_embeds_batch,
                    gene_embeds,
                    task_counts=task_counts,
                    ds_emb=ds_emb,
                )
                logprobs_batch = self.model.binary_decoder(merged_embs)
                logprobs_batch = logprobs_batch.detach().cpu().float().numpy()
                yield logprobs_batch.squeeze()

    @staticmethod
    def load_esm2_embeddings(cfg):
        """
        Load ESM2 embeddings and special tokens.

        This static method loads pre-computed ESM2 protein embeddings from the
        configuration file and prepares them for use with the model.

        Parameters
        ----------
        cfg : OmegaConf
            Configuration object containing embedding paths.

        Returns
        -------
        torch.Tensor
            Tensor containing ESM2 embeddings moved to GPU if available.
        """
        # Load in ESM2 embeddings and special tokens
        all_pe = torch.load(get_embedding_cfg(cfg).all_embeddings, weights_only=False)
        if isinstance(all_pe, dict):
            all_pe = torch.vstack(list(all_pe.values()))

        all_pe = all_pe.cuda()
        return all_pe
