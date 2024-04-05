"""
    Helper functions for loading pretrained gene embeddings.
"""

from pathlib import Path
from typing import Dict, Tuple
import torch
from scanpy import AnnData
import logging
logger = logging.getLogger(__name__)

def get_gene_embedding_paths(embedding_path: Path):
    return {
        'ESM2': {
            'human': embedding_path / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
            'mouse': embedding_path / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
            'frog': embedding_path / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
            'zebrafish': embedding_path / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
            "mouse_lemur": embedding_path / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
            "pig": embedding_path / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
            "macaca_fascicularis": embedding_path / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
            "macaca_mulatta": embedding_path / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
        }
    }

#TODO Add new function to add embeddings
# extra_species = pd.read_csv("./UCE/model_files/new_species_protein_embeddings.csv").set_index("species").to_dict()["path"]
# MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH["ESM2"].update(extra_species) # adds new species


def load_gene_embeddings_adata(adata: AnnData, species: list, embedding_model: str, embeddings_path: Path) -> Tuple[AnnData, Dict[str, list[str]]]:
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
        logger.error(f'Missing gene embeddings here: {embeddings_path}')
        raise ValueError(f'The following species do not have gene embeddings: {species_names_set - available_species}')
    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(species_to_gene_embedding_path[species]).items()
        }
        for species in species_names
    }

    logger.info(f'Finished loading gene embeddings for {species_names_set} from {embeddings_path}')

    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(*[
        set(gene_symbol_to_embedding)
        for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()
    ])
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}
    
    # Subset data to only use genes with embeddings
    filtered_adata = adata[:, adata.var_names.isin(genes_to_use)]
    filtered = adata.var_names.shape[0] - filtered_adata.var_names.shape[0]
    logger.info(f'Filtered out {filtered} genes to a total of {filtered_adata.var_names.shape[0]} genes with embeddings.')

    # Load gene symbols for desired species for later use with indexes
    species_to_all_gene_symbols = {
        species: [
            gene_symbol.lower()
            for gene_symbol, _ in torch.load(species_to_gene_embedding_path[species]).items()
        ]
        for species in species_names
    }
    return filtered_adata, species_to_all_gene_symbols
