from helical.models.base_models import HelicalRNAModel
import logging
from pathlib import Path
import numpy as np
from anndata import AnnData
import torch
from typing import Optional
from datasets import Dataset
from helical.models.tahoe.tahoe_config import TahoeConfig
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids

LOGGER = logging.getLogger(__name__)


class Tahoe(HelicalRNAModel):
    """Tahoe-1x Model.

    The Tahoe-1x Model is a transformer-based foundation model designed for single-cell
    RNA-seq data. It can extract cell and gene embeddings from raw count data.
    The model is available in three sizes:

    - 70m: 12-layer transformer with 512 embedding dimensions
    - 1b: 24-layer transformer with 1024 embedding dimensions (coming soon)
    - 3b: 36-layer transformer with 1536 embedding dimensions (coming soon)

    Example
    -------
    ```python
    from helical.models.tahoe import Tahoe, TahoeConfig
    import anndata as ad

    # Example configuration
    tahoe_config = TahoeConfig(model_size="70m", batch_size=8)
    tahoe = Tahoe(configurer=tahoe_config)

    # Load and process data
    ann_data = ad.read_h5ad("anndata_file.h5ad")
    dataset = tahoe.process_data(ann_data)

    # Get embeddings
    embeddings = tahoe.get_embeddings(dataset)
    print("Tahoe embeddings shape:", embeddings.shape)

    # Get both cell and gene embeddings
    cell_embeddings, gene_embeddings = tahoe.get_embeddings(dataset, return_gene_embeddings=True)
    print("Cell embeddings shape:", cell_embeddings.shape)
    print("Gene embeddings shape:", gene_embeddings.shape)
    ```

    Parameters
    ----------
    configurer : TahoeConfig, optional
        The model configuration. Defaults to TahoeConfig() with default parameters.

    Notes
    -----
    The Tahoe-1x model uses Ensembl IDs to identify genes and currently supports only
    human genes. The model is published by Tahoe Therapeutics and available on
    Hugging Face at https://huggingface.co/tahoebio/Tahoe-x1.
    """

    default_configurer = TahoeConfig()

    def __init__(self, configurer: TahoeConfig = default_configurer) -> None:
        super().__init__()
        self.configurer = configurer
        self.config = configurer.config
        self.device = torch.device(self.config["device"])

        # Import tahoe_x1 modules
        try:
            from tahoe_x1.model import ComposerTX
        except ImportError:
            raise ImportError(
                "The tahoe_x1 package is required to use the Tahoe model. "
                "Please install it from the tahoe-x1 repository."
            )

        LOGGER.info(
            f"Loading Tahoe model (size: {self.config['model_size']}) from Hugging Face..."
        )

        # Load model from Hugging Face
        self.model, self.vocab, self.model_cfg, self.collator_cfg = (
            ComposerTX.from_hf(
                repo_id=self.config["hf_repo_id"],
                model_size=self.config["model_size"],
                return_gene_embeddings=(self.config["emb_mode"] == "gene"),
            )
        )

        self.model.to(self.device)
        self.model.eval()

        LOGGER.info(
            f"Model loaded with {self.model.model.n_layers} transformer layers."
        )
        LOGGER.info(
            f"Tahoe model is in 'eval' mode, on device '{self.device}' with embedding mode '{self.config['emb_mode']}'."
        )

    def process_data(
        self,
        adata: AnnData,
        gene_names: str = "index",
        use_raw_counts: bool = True,
    ) -> AnnData:
        """
        Processes the data for the Tahoe model.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. Tahoe uses Ensembl IDs
            to identify genes and currently supports only human genes. If the AnnData object
            already has an 'ensembl_id' column, the mapping step can be skipped.
        gene_names : str, optional, default="index"
            The column in `adata.var` that contains the gene names. If set to a value other
            than "ensembl_id", the gene symbols in that column will be mapped to Ensembl IDs
            using the 'pyensembl' package.
            - If set to "index", the index of the AnnData object will be used and mapped to Ensembl IDs.
            - If set to "ensembl_id", no mapping will occur.
        use_raw_counts : bool, optional, default=True
            Determines whether raw counts should be used.

        Returns
        -------
        AnnData
            The processed AnnData object with gene IDs mapped to the vocabulary.
        """
        LOGGER.info("Processing data for Tahoe.")
        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        # Map gene symbols to Ensembl IDs if provided
        if gene_names != "ensembl_id":
            if (adata.var[gene_names].str.startswith("ENS").all()) or (
                adata.var[gene_names].str.startswith("None").any()
            ):
                message = (
                    "It seems an anndata with 'ensemble ids' and/or 'None' was passed. "
                    "Please set gene_names='ensembl_id' and remove 'None's to skip mapping."
                )
                LOGGER.error(message)
                raise ValueError(message)
            adata = map_gene_symbols_to_ensembl_ids(adata, gene_names)

            if adata.var["ensembl_id"].isnull().all():
                message = "All gene symbols could not be mapped to Ensembl IDs. Please check the input data."
                LOGGER.error(message)
                raise ValueError(message)

        gene_id_key = "ensembl_id"

        # Map genes to vocabulary
        adata.var["id_in_vocab"] = [
            self.vocab[gene] if gene in self.vocab else -1
            for gene in adata.var[gene_id_key]
        ]

        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        n_matched = np.sum(gene_ids_in_vocab >= 0)
        n_total = len(gene_ids_in_vocab)

        LOGGER.info(
            f"Matched {n_matched}/{n_total} genes in vocabulary of size {len(self.vocab)}."
        )

        # Filter to genes in vocabulary
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        genes = adata.var[gene_id_key].tolist()
        gene_ids = np.array([self.vocab[gene] for gene in genes], dtype=int)

        if not np.all(gene_ids >= 0):
            raise ValueError("Some genes are not in the vocabulary after filtering.")

        # Store gene_ids in adata for later use
        adata.uns["tahoe_gene_ids"] = gene_ids

        LOGGER.info("Successfully processed the data for Tahoe.")
        return adata

    def get_embeddings(
        self,
        adata: AnnData,
        return_gene_embeddings: bool = False,
    ) -> np.ndarray:
        """Gets the embeddings from the Tahoe model.

        Parameters
        ----------
        adata : AnnData
            The processed AnnData object containing the data.
        return_gene_embeddings : bool, optional, default=False
            Whether to return gene embeddings in addition to cell embeddings.

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            If return_gene_embeddings is False:
                - Cell embeddings as a numpy array of shape (n_cells, embedding_dim)
            If return_gene_embeddings is True:
                - Tuple of (cell_embeddings, gene_embeddings)
                - Cell embeddings: shape (n_cells, embedding_dim)
                - Gene embeddings: shape (n_genes_in_vocab, embedding_dim)
        """
        LOGGER.info("Extracting embeddings from Tahoe model...")

        # Import the embedding extraction function
        from tahoe_x1.tasks import get_batch_embeddings

        # Get gene_ids from the processed data
        if "tahoe_gene_ids" not in adata.uns:
            raise ValueError(
                "Data must be processed with process_data() before calling get_embeddings()"
            )

        gene_ids = adata.uns["tahoe_gene_ids"]

        # Get embeddings using the tahoe_x1 function
        result = get_batch_embeddings(
            adata=adata,
            model=self.model.model,
            vocab=self.vocab,
            model_cfg=self.model_cfg,
            collator_cfg=self.collator_cfg,
            gene_ids=gene_ids,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            prefetch_factor=self.config["prefetch_factor"],
            max_length=self.config["max_length"],
            return_gene_embeddings=return_gene_embeddings,
        )

        if return_gene_embeddings:
            cell_embeddings, gene_embeddings = result
            LOGGER.info(
                f"Finished extracting embeddings. Cell shape: {cell_embeddings.shape}, Gene shape: {gene_embeddings.shape}"
            )
            return cell_embeddings, gene_embeddings
        else:
            LOGGER.info(
                f"Finished extracting embeddings. Shape: {result.shape}"
            )
            return result
