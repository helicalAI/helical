import pandas as pd
import anndata as ad
import numpy as np
from scipy.sparse import lil_matrix

def get_anndata_from_hf_dataset(dataset):
    """
    Convert a HuggingFace dataset to an AnnData object

    Parameters
    ----------
    dataset : Dataset
        A HuggingFace dataset object
    """
    observation_columns = [obs for obs in list(dataset.features.keys()) if not obs == 'raw_counts' and not obs == "rows"]
    obs_data = pd.DataFrame(dataset.select_columns(observation_columns).data.to_pandas(),columns=observation_columns)
    lil = lil_matrix((len(dataset),dataset[0]['size']))
    lil.data = np.array(dataset['raw_counts'],dtype="object")
    lil.rows = np.array(dataset['rows'],dtype="object")
    ann_data = ad.AnnData(lil.tocsr(),obs=obs_data)
    ann_data.var_names = dataset.features['raw_counts'].id.split(",")
    ann_data.var['gene_name'] = ann_data.var_names.str.upper()

    return ann_data