import logging
from typing import List, Optional, Sequence

import pandas as pd
from anndata import AnnData
import pybiomart

LOGGER = logging.getLogger(__name__)


def _get_ensembl_mart_df(species: str = "hsapiens") -> pd.DataFrame:
    """
    Fetch a (species)_gene_ensembl table via pybiomart.

    Parameters
    ----------
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart (e.g., "hsapiens", "mmusculus").

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns "ensembl_id" and "gene_name".
    """
    server = pybiomart.Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets[f"{species}_gene_ensembl"]
    df = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
    df = df.rename(columns={"Gene stable ID": "ensembl_id", "Gene name": "gene_name"})
    return df.sort_values(by="ensembl_id")


def map_gene_symbols_to_ensembl_ids(
    adata: AnnData, gene_names: Optional[str] = None, species: str = "hsapiens"
) -> AnnData:
    """
    Map gene symbols to Ensembl Gene IDs using pybiomart.

    Due to duplication events, some symbols map to multiple Ensembl IDs; we take
    the first occurrence after de-duplication.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene metadata in `adata.var`.
    gene_names : str, optional
        Column in `adata.var` containing gene symbols. If None, uses `adata.var_names`.
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart (e.g., "hsapiens", "mmusculus").

    Returns
    -------
    AnnData
        Same object with `adata.var["ensembl_id"]` populated.
    """
    var_names = adata.var[gene_names] if gene_names is not None else pd.Series(adata.var_names, index=adata.var_names)
    adata.var["ensembl_id"] = convert_list_gene_symbols_to_ensembl_ids(var_names, species=species)
    non_none_mappings = adata.var["ensembl_id"].notnull().sum()
    LOGGER.info("Mapped %d / %d genes to Ensembl IDs.", non_none_mappings, adata.var.shape[0])
    return adata


def map_ensembl_ids_to_gene_symbols(
    adata: AnnData, ensembl_id_key: str = "ensembl_id", species: str = "hsapiens"
) -> AnnData:
    """
    Map Ensembl Gene IDs to gene symbols using pybiomart.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene metadata in `adata.var`.
    ensembl_id_key : str, default "ensembl_id"
        Column in `adata.var` containing Ensembl Gene IDs.
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart (e.g., "hsapiens", "mmusculus").

    Returns
    -------
    AnnData
        Same object with `adata.var["gene_names"]` populated.
    """
    adata.var["gene_names"] = convert_list_ensembl_ids_to_gene_symbols(adata.var[ensembl_id_key], species=species)
    non_none_mappings = adata.var["gene_names"].notnull().sum()
    LOGGER.info("Mapped %d / %d Ensembl IDs to gene names.", non_none_mappings, adata.var.shape[0])
    return adata


def convert_list_ensembl_ids_to_gene_symbols(
    ensembl_ids: Sequence[str], species: str = "hsapiens"
) -> List[Optional[str]]:
    """
    Map a list/sequence of Ensembl Gene IDs to gene symbols using pybiomart.

    Parameters
    ----------
    ensembl_ids : Sequence[str]
        Ensembl Gene IDs (e.g., "ENSG00000139618").
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart.

    Returns
    -------
    List[Optional[str]]
        Gene symbols aligned to the input order (None if not found).
    """
    df = _get_ensembl_mart_df(species=species)
    mapping = df.drop_duplicates(subset="ensembl_id").set_index("ensembl_id")["gene_name"]
    return list(pd.Series(ensembl_ids, dtype="object").map(mapping))


def convert_list_gene_symbols_to_ensembl_ids(
    gene_symbols: Sequence[str], species: str = "hsapiens"
) -> List[Optional[str]]:
    """
    Map a list/sequence of gene symbols to Ensembl Gene IDs using pybiomart.

    Parameters
    ----------
    gene_symbols : Sequence[str]
        Gene symbols (e.g., "BRCA2", "KANSL2").
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart.

    Returns
    -------
    List[Optional[str]]
        Ensembl Gene IDs aligned to the input order (None if not found).
    """
    df = _get_ensembl_mart_df(species=species)
    mapping = df.drop_duplicates(subset="gene_name").set_index("gene_name")["ensembl_id"]
    return list(pd.Series(gene_symbols, dtype="object").map(mapping))
