import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import lil_matrix
from datasets import Dataset
import logging

LOGGER = logging.getLogger(__name__)

def get_anndata_from_hf_dataset(dataset: Dataset) -> ad.AnnData:
    """
    Convert a HuggingFace dataset to an AnnData object
    It assumes that the dataset has the following columns:
    - raw_counts (specifying the raw counts of the data, excluding 0s)
    - rows (specifying the row indices of the raw_counts)
    - Any other column that will be used as obs

    The id of the raw_counts are the gene names as a string separated by commas.

    Parameters
    ----------
    dataset : Dataset
        A HuggingFace dataset object

    Returns
    -------
    AnnData
        An AnnData object containing the data from the input dataset
    """
    # obs
    excluded_features = ['raw_counts', 'rows', 'size']
    observation_names = [obs for obs in dataset.features.keys() if obs not in excluded_features]
    obs_data = pd.DataFrame(dataset.select_columns(observation_names).data.to_pandas(),columns=observation_names)
    
    # raw counts
    lil = lil_matrix((len(dataset),dataset[0]['size']))
    lil.data = np.array(dataset['raw_counts'],dtype="object")
    lil.rows = np.array(dataset['rows'],dtype="object")
    
    # create AnnData object
    ann_data = ad.AnnData(lil.tocsr(),obs=obs_data)

    # add gene names
    var_names = dataset.features['raw_counts'].id.split(",")
    if len(var_names) != ann_data.shape[1]:
        message = f"Number of gene names ({len(var_names)}) does not match the number of genes ({ann_data.shape[1]})"
        LOGGER.error(message)
        raise ValueError(message)

    ann_data.var_names = [name.upper() for name in var_names]
    ann_data.var['gene_name'] = ann_data.var_names

    return ann_data