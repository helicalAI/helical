"""
Helper functions for loading pretrained gene embeddings.
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
from scanpy import AnnData
import logging
from helical.models.uce.uce_config import SPECIES_GENE_EMBEDDINGS

LOGGER = logging.getLogger(__name__)


def get_gene_embedding_paths(embedding_path: Path) -> Dict[str, Dict[str, Path]]:
    """
    Get the paths to the gene embeddings for all species and models by prepending the embedding path to the file names.

    :param embedding_path: The path to the directory containing the gene embeddings.
    :return: A dictionary mapping model names to dictionaries mapping species names to the paths of the gene embeddings.
    """
    res = {}
    for model, species_dict in SPECIES_GENE_EMBEDDINGS.items():
        res[model] = {}
        for species, file in species_dict.items():
            res[model][species] = embedding_path / file
    return res


# TODO Add new function to add embeddings
# extra_species = pd.read_csv("./UCE/model_files/new_species_protein_embeddings.csv").set_index("species").to_dict()["path"]
# MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH["ESM2"].update(extra_species) # adds new species


def load_gene_embeddings_adata(
    adata: AnnData, species: list, embedding_model: str, embeddings_path: Path
) -> Tuple[AnnData, Dict[str, list[str]]]:
    """
    Loads gene embeddings for all the species/genes in the provided data.

    :param adata: An AnnData object containing gene expression data for cells.
    :param species: Species corresponding to this adata
    :param embedding_model: The gene embedding model whose embeddings will be loaded.
    :param embedding_model: The embedding paths for the model
    :return: A tuple containing:
               - A subset of the data only containing the gene expression for genes with embeddings in all species.
               - A dictionary mapping species name to the names of genes.
    """
    # Get species names
    species_names = species
    species_names_set = set(species_names)

    # Get embedding paths for the model
    embedding_paths = get_gene_embedding_paths(embeddings_path)
    species_to_gene_embedding_path = embedding_paths[embedding_model]
    available_species = set(species_to_gene_embedding_path)

    # Ensure embeddings are available for all species
    if not (species_names_set <= available_species):
        LOGGER.error(f"Missing gene embeddings here: {embeddings_path}")
        raise ValueError(
            f"The following species do not have gene embeddings: {species_names_set - available_species}"
        )
    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(
                species_to_gene_embedding_path[species]
            ).items()
        }
        for species in species_names
    }

    LOGGER.info(
        f"Finished loading gene embeddings for '{', '.join(map(str, species_names_set))}' from {embeddings_path} for model '{embedding_model}'."
    )

    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(
        *[
            set(gene_symbol_to_embedding)
            for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()
        ]
    )
    genes_to_use = {
        gene for gene in adata.var_names if gene.lower() in genes_with_embeddings
    }

    # Subset data to only use genes with embeddings
    filtered_adata = adata[:, adata.var_names.isin(genes_to_use)]
    filtered = adata.var_names.shape[0] - filtered_adata.var_names.shape[0]
    LOGGER.info(
        f"Filtered out {filtered} genes to a total of {filtered_adata.var_names.shape[0]} genes with embeddings."
    )

    if filtered_adata.var_names.shape[0] == 0:
        message = "No matching genes found between input data and UCE gene embedding vocabulary. Please check the gene names in .var of the anndata input object."
        LOGGER.error(message)
        raise ValueError(message)

    # Load gene symbols for desired species for later use with indexes
    species_to_all_gene_symbols = {
        species: [
            gene_symbol.lower()
            for gene_symbol, _ in torch.load(
                species_to_gene_embedding_path[species]
            ).items()
        ]
        for species in species_names
    }
    return filtered_adata, species_to_all_gene_symbols
