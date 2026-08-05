from helical.models.base_models import HelicalRNAModel
import logging
import numpy as np
from anndata import AnnData
from helical.utils.downloader import Downloader
from helical.models.genept.genept_config import GenePTConfig
from helical.utils.mapping import convert_list_ensembl_ids_to_gene_symbols
import scanpy as sc
import torch
import json

LOGGER = logging.getLogger(__name__)

# Ensembl gene IDs, optionally version-suffixed (ENSG00000141510.17).
_ENSEMBL_GENE_ID_PATTERN = r"^ENS[A-Z]{0,4}G\d{11}(\.\d+)?$"


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
        # and looked Ensembl IDs up in a symbol-keyed table (bio-agent#1117).
        # Matched per entry with an anchored Ensembl *gene* ID pattern rather than
        # `.startswith("ENS")`, which also matches real gene symbols (ENSA) and
        # transcript/protein IDs (ENST.., ENSP..), and rather than `.all()`, which
        # would skip a var index that is only mostly Ensembl IDs.
        identifiers = adata.var[gene_names].astype(str)
        is_ensembl = identifiers.str.match(_ENSEMBL_GENE_ID_PATTERN).fillna(False)
        if is_ensembl.any():
            # The mapping table is keyed on bare gene IDs, so the version suffix
            # the pattern above accepts has to come off before lookup -- versioned
            # IDs (ENSG00000141510.17) are the GENCODE/CellRanger default, and
            # looking them up verbatim resolves nothing. Only Ensembl entries are
            # stripped: real gene symbols contain dots too (AC000068.10).
            bare = identifiers.str.split(".").str[0].where(is_ensembl, identifiers)
            to_convert = sorted(set(bare[is_ensembl]))
            symbols = convert_list_ensembl_ids_to_gene_symbols(to_convert)
            # Excludes unmapped ids rather than keeping a None value, so no entry
            # can later resolve through a bogus key.
            mapping = {
                ensembl_id: symbol
                for ensembl_id, symbol in zip(to_convert, symbols)
                if symbol
            }
            # Ensembl entries become their symbol; anything already a symbol is
            # kept as-is, so a mixed var index does not lose its symbols.
            resolved = [
                mapping.get(bare_id) if flag else original
                for bare_id, flag, original in zip(bare, is_ensembl, identifiers)
            ]

            # Ensembl -> symbol is many-to-one: 10616 of the 48698 symbol-bearing
            # rows in hsapiens_pybiomart.csv share a gene_name (3369 symbols
            # carried by >=2 ids, e.g. PRPF31, 5S_rRNA, Y_RNA). Without collapsing
            # them the var index comes out non-unique and the `adata[:, genes_names]`
            # subset below raises InvalidIndexError. Of a colliding set keep the
            # copy carrying the most counts, so an all-zero alt-scaffold copy can
            # never displace the expressed one (which picking positionally would).
            totals = np.asarray(adata.X.sum(axis=0)).ravel()
            best_for_symbol: dict = {}
            for index, symbol in enumerate(resolved):
                if symbol is None:
                    continue
                incumbent = best_for_symbol.get(symbol)
                if incumbent is None or totals[index] > totals[incumbent]:
                    best_for_symbol[symbol] = index
            kept_indices = set(best_for_symbol.values())
            keep = np.array(
                [
                    symbol is not None and index in kept_indices
                    for index, symbol in enumerate(resolved)
                ]
            )

            if not keep.any():
                message = (
                    "None of the Ensembl IDs could be mapped to gene symbols, which "
                    "GenePT's embeddings are keyed on. Please check the gene "
                    "identifiers in .var of the anndata input object."
                )
                LOGGER.error(message)
                raise ValueError(message)

            n_unmapped = sum(1 for symbol in resolved if symbol is None)
            n_duplicate = int((~keep).sum()) - n_unmapped
            if n_unmapped or n_duplicate:
                LOGGER.info(
                    f"Mapped Ensembl IDs to gene symbols for GenePT: "
                    f"{len(resolved)} gene(s) in -> {int(keep.sum())} out "
                    f"({n_unmapped} dropped with no symbol, "
                    f"{n_duplicate} dropped as duplicate symbols)."
                )
            adata = adata[:, keep].copy()
            adata.var_names = [symbol for symbol, kept in zip(resolved, keep) if kept]

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
