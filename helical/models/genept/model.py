from helical.models.base_models import HelicalRNAModel
import logging
import numpy as np
from anndata import AnnData
from helical.utils.downloader import Downloader
from helical.models.genept.genept_config import GenePTConfig
from helical.utils.mapping import ensure_gene_symbols
import scanpy as sc
import torch
import json

LOGGER = logging.getLogger(__name__)


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

        downloader = Downloader()
        for file in self.config["list_of_files_to_download"]:
            downloader.download_via_name(file)

        with open(self.config["embeddings_path"], "r") as f:
            self.embeddings = json.load(f)

        LOGGER.info("GenePT initialized successfully.")

    def process_data(
        self,
        adata: AnnData,
        gene_names: str = "index",
        use_raw_counts: bool = True,
    ) -> AnnData:
        """
        Processes the data for the GenePT model.

        Parameters
        ----------
        adata : AnnData
            The AnnData object containing the data to be processed. GenePT identifies genes by
            **gene symbol** -- its embedding table is keyed on symbols (see `get_text_embeddings`)
            -- and currently supports only human genes. Ensembl gene IDs are mapped to symbols
            automatically.
        gene_names : str, optional, default="index"
            The column in `adata.var` that holds the gene identifiers.
            - "index" (default): the index of the AnnData object is used.
            - any other column name: that column is used.
            Either way, if the identifiers are Ensembl gene IDs they are mapped to gene symbols
            and `var_names` is set to the result; genes with no symbol are dropped. Identifiers
            that are already symbols are left untouched.
        use_raw_counts : bool, optional, default=True
            Determines whether raw counts should be used.

        Returns
        -------
        Dataset
            The tokenized dataset in the form of a Huggingface Dataset object.
        """
        LOGGER.info("Processing data for GenePT.")
        self.ensure_rna_data_validity(adata, gene_names, use_raw_counts)

        # GenePT's embedding table is keyed on gene symbols, so Ensembl IDs have to
        # be mapped *to* symbols -- the opposite direction from the Ensembl-keyed
        # models. This condition used to read `== "ensembl_id"` (copied from
        # Geneformer, which needs `!=`), which made the mapping unreachable on every
        # input where it would have been correct: gene_names="ensembl_id" with real
        # Ensembl IDs raised an error telling the caller to set the flag they had
        # just set, and the default gene_names="index" skipped mapping altogether
        # and looked Ensembl IDs up in a symbol-keyed table.
        # GenePT's embedding table is keyed on gene symbols, so Ensembl IDs must be
        # mapped *to* symbols -- the opposite direction from the Ensembl-keyed
        # models. The guard here used to read `== "ensembl_id"` (copied from
        # Geneformer, which needs `!=`), which made the mapping unreachable on
        # every input where it would have been correct.
        adata = ensure_gene_symbols(adata, gene_names)

        sc.pp.highly_variable_genes(adata, flavor="seurat_v3")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        genes_names = adata.var_names[adata.var["highly_variable"]].tolist()
        adata = adata[:, genes_names]

        LOGGER.info("Successfully processed the data for GenePT.")
        return adata

    def get_text_embeddings(self, dataset: AnnData) -> np.array:
        """Gets the gene embeddings from the GenePT model

        Parameters
        ----------
        dataset : AnnData
            The tokenized dataset containing the processed data

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        # Generate a response
        raw_embeddings = dataset.var_names
        weights = []
        gene_list = []
        count_missed = 0

        for emb in raw_embeddings:
            gene = self.embeddings.get(emb.upper(), None)
            if gene is not None:
                weights.append(gene["embeddings"])
                gene_list.append(emb)
            else:
                count_missed += 1

        LOGGER.info(f"Couln't find {count_missed} genes in embeddings")

        weights = torch.Tensor(weights)
        embeddings = torch.matmul(
            torch.Tensor(dataset[:, gene_list].X.toarray()), weights
        )
        return embeddings

    def get_embeddings(self, dataset: AnnData) -> torch.Tensor:
        """Gets the gene embeddings from the GenePT model

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
        # Generate a response
        embeddings = self.get_text_embeddings(dataset)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1)).reshape(-1, 1)
        return embeddings
