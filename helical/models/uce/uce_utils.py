import scanpy as sc
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union
import logging
from helical.models.uce.uce_model import TransformerModel

LOGGER = logging.getLogger(__name__)

def get_positions(species_chrom_csv_path: Path, species: str, adata: sc.AnnData) -> Tuple[pd.Series, np.array]:
    '''
    Get the chromosomes to which the genes in adata belong (encoded with cat.codes) and the start positions of the genes
    
    Args:
        species_chrom_csv_path: Path to the csv file
        species: The species for which to get the mapping
        adata: The filtered data with the genes for which there are embeddings.

    Returns:
        A tuple with a pandas series specifying to which chromosome a gene belongs and an array with the start positions per gene.
    '''
    genes_to_chroms_pos =  pd.read_csv(species_chrom_csv_path)
    genes_to_chroms_pos["spec_chrom"] = pd.Categorical(genes_to_chroms_pos["species"] + "_" +  genes_to_chroms_pos["chromosome"]) # add the spec_chrom list
    spec_gene_chrom_pos = genes_to_chroms_pos[genes_to_chroms_pos["species"] == species].set_index("gene_symbol")
    
    filtered_spec_gene_chrom_pos = spec_gene_chrom_pos.loc[[k.upper() for k in adata.var_names]]
    dataset_chroms = filtered_spec_gene_chrom_pos["spec_chrom"].cat.codes
    dataset_start = filtered_spec_gene_chrom_pos["start"].values
    
    return dataset_chroms, dataset_start

def get_ESM2_embeddings(token_file: Union[Path, str], token_dim: int) -> torch.Tensor:
    '''
    Loads the token file specified in the config file.

    Args:
        files_config: A dictionary with 'token_file' and 'token_dim' as keys. 

    Returns:
        The token file loaded as a torch.Tensor.
    '''

    all_pe = torch.load(token_file)

    # TODO: Why this if clause and why this magic number 143574? 
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, token_dim))
        # 1895 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS))  # Add the chrom tensors to the end
        all_pe.requires_grad = False

    return all_pe

def get_protein_embeddings_idxs(offset_pkl_path: str, species: str, species_to_all_gene_symbols: Dict[str, list[str]], adata: sc.AnnData)-> torch.tensor:
    '''
    For a given species, look up the offset. Find the index per gene from the list of all available genes.
    Only use the indexes of the genes for which there are embeddings (in adata).
    Add up the index and offset for the end result.

    Args:
        offset_pkl_path: Path to the offset file
        species: The species for which to get the offsets
        species_to_all_gene_symbols: A dictionary with all species and their gene symbols
        adata: The filtered data with the genes for which there are embeddings.
    
    Returns:
        A tensor with the indexes of the used genes
    '''
    with open(offset_pkl_path, "rb") as f:
        species_to_offsets = pickle.load(f)
    offset = species_to_offsets[species]
    spec_all_genes = species_to_all_gene_symbols[species]
    return torch.tensor([spec_all_genes.index(k.lower()) + offset for k in adata.var_names]).long()

def prepare_expression_counts_file(gene_expression: np.array, name: str, folder_path: str = "./") -> None:
    '''
    Creates a .npz file and writes the contents of the expression array into this file. 
    This allows handling arrays that are too large to fit entirely in memory. 
    The array is stored on disk, but it can be accessed and manipulated like a regular in-memory array. 
    Changes made to the array are written directly to disk.

    Args:
        expression: The array to write to the file
        name: The prefix of the file eventually called {name}_counts.npz
        folder_path: The folder path of the npz file
    '''
    filename = folder_path + f"{name}_counts.npz"
    try:
        shape = gene_expression.shape
        fp = np.memmap(filename, dtype='int64', mode='w+', shape=shape)
        fp[:] = gene_expression[:]
        fp.flush()
        LOGGER.info(f"Passed the gene expressions (with shape={shape} and max gene count data {gene_expression.max()}) to {filename}")
    except:
        LOGGER.error(f"Error during preparation of npz file {filename}.")
        raise Exception
    
## writing a funciton to load the model 
def load_model(model_path: Union[str, Path], model_config: Dict[str, str], all_pe: torch.Tensor) -> TransformerModel:
    '''
    Load the UCE Transformer Model based on configurations from the model_config file.

    Args:
        model_path: A path to the model to load
        model_config: A dictionary with 'token_dim', 'd_hid', 'n_layers' and 'output_dim' as keys. 
        all_pe: The token file loaded as a torch.Tensor.

    Returns:
        The TransformerModel.
    '''
    model = TransformerModel(token_dim = model_config["token_dim"], 
                             d_model = 1280, # each cell is represented as a d-dimensional vector, where d = 1280, see UCE paper. TODO: Can we use `output_dim` from the model_config?
                             nhead = 20,  # number of heads in nn.MultiheadAttention,
                             d_hid = model_config['d_hid'],
                             nlayers = model_config['n_layers'], 
                             dropout = 0.05,
                             output_dim = model_config['output_dim'])

    # empty_pe = torch.zeros(50000, 5120)
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = torch.nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(model_path, map_location=model_config["device"]), strict=True)

    # This will make sure that you don't overwrite the tokens in case you're embedding species from the training data
    # We avoid doing that just in case the random seeds are different across different versions. 
    if all_pe.shape[0] != 145469: 
        all_pe.requires_grad = False
        model.pe_embedding = torch.nn.Embedding.from_pretrained(all_pe)
    
    return model
