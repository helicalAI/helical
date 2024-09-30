import logging 
from pyensembl import genome_for_reference_name
import pandas as pd
from pyensembl import EnsemblRelease
from pyensembl.species import human
from pyensembl.species import Species
from anndata import AnnData

LOGGER = logging.getLogger(__name__)


def map_gene_symbols_to_ensembl_ids(adata: AnnData, gene_names: str, species: Species = human) -> AnnData:
    """
    Map gene names to Ensembl IDs using the pyensembl library.
    Due to copy events, there might be multiple genes per name. We take always take the fist one.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing the gene expression data.
    gene_names : str
        Column name in adata.var containing the gene names.
    species : pyensembl.species.Species, optional, default = human
        Species for which the gene names should be mapped.
        For the provided species, we take the first 'reference_assembly' as default to do the mapping.
        For humans, this is the GRCh38 genome for example.
    
    Returns
    -------
    anndata.AnnData
        Anndata object with the gene names mapped to Ensembl IDs in adata.var["ensembl_id"]
    """
    # this is one time only
    ensembl_release = EnsemblRelease(species=species) 
    ensembl_release.download(overwrite=False)
    ensembl_release.index(overwrite=False)

    adata.var['ensembl_id'] = pd.Series([None] * len(adata.var), index=adata.var.index)
    # we take the first reference assembly from the provided dictionary as default
    genome_reference = genome_for_reference_name(next(iter(species.reference_assemblies)))
    for index, name in adata.var[gene_names].items():
        try:
            adata.var.at[index, 'ensembl_id'] = genome_reference.gene_ids_of_gene_name(name)[0]
        except:
            continue
    non_none_mappings = adata.var['ensembl_id'].notnull().sum()
    LOGGER.info(f"Mapped {non_none_mappings} genes to Ensembl IDs from a total of {adata.var.shape[0]} genes.")
    return adata


def map_ensembl_ids_to_gene_symbols(adata: AnnData, ensembl_id_key: str = "ensembl_id", species: Species = human) -> AnnData:
    """
    Map Ensembl IDs to gene names using the pyensembl library.
    We use the GRCh38 genome for mapping.

    Parameters
    ----------
    adata : anndata.AnnData
        Anndata object containing the gene expression data.
    ensembl_id_key : str, optional, default = "ensembl_id"
        Column name in adata.var containing the ensemble ids.
    species : pyensembl.species.Species, optional, default = human
        Species for which the Ensembl IDs should be mapped.
        For the provided species, we take the first reference genome as default to do the mapping.
        For humans, this is the GRCh38 genome for example.
    
    Returns
    -------
    anndata.AnnData
        Anndata object with the Ensembl IDs mapped to gene names in adata.var["gene_names"]
    """
    # this is one time only
    ensembl_release = EnsemblRelease(species=species) 
    ensembl_release.download(overwrite=False)
    ensembl_release.index(overwrite=False)

    adata.var['gene_names'] = pd.Series([None] * len(adata.var), index=adata.var.index)
    # we take the first reference assembly from the provided dictionary as default
    genome_reference = genome_for_reference_name(next(iter(species.reference_assemblies)))
    for index, ensembl_id in adata.var[ensembl_id_key].items():
        try:
            adata.var.at[index, 'gene_names'] = genome_reference.gene_name_of_gene_id(ensembl_id)
        except:
            continue
    non_none_mappings = adata.var['gene_names'].notnull().sum()
    LOGGER.info(f"Mapped {non_none_mappings} genes to Gene names from a total of {adata.var.shape[0]} Ensembl IDs.")
    return adata