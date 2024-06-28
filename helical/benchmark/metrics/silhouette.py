# This code was adapted and modified from scIB.
# https://github.com/theislab/scib/tree/main
# The original code is licensed under the MIT License.
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import silhouette_samples
import logging 
from anndata import AnnData
from typing import Union, Tuple
from pandas import DataFrame

LOGGER = logging.getLogger(__name__)

def silhouette_batch(
    adata: AnnData,
    batch_obs: str,
    label_obs: str,
    embedding_obsm: str,
    metric: str = "euclidean",
    scale: bool = True,
    return_all: bool = False,
) -> Union[int, Tuple[int, DataFrame, DataFrame]]:
    """
    This metric measures the average silhouette width (ASW) score of a given batch.
    It assumes that a silhouette width close to 0 represents perfect overlap of the batches, thus the absolute value of
    the silhouette width is used to measure how well batches are mixed.
    For all cells :math:`i` of a cell type :math:`C_j`, the batch ASW of that cell type is:

    .. math::

        batch \\, ASW_j = \\frac{1}{|C_j|} \\sum_{i \\in C_j} |silhouette(i)|

    The final score is the average of the absolute silhouette widths computed per cell type :math:`M`.

    .. math::

        batch \\, ASW = \\frac{1}{|M|} \\sum_{i \\in M} batch \\, ASW_j

    For a scaled metric (which is the default), the absolute ASW per group is subtracted from 1 before averaging, so that
    0 indicates suboptimal label representation and 1 indicates optimal label representation.

    .. math::

        batch \\, ASW_j = \\frac{1}{|C_j|} \\sum_{i \\in C_j} 1 - |silhouette(i)|

    Parameters
    ----------
    adata: AnnData 
        The anndata object.
    batch_obs: string 
        The batch labels to be compared against in adata.obs.
    label_obs: string
        The group labels to be subset by e.g. cell type
    embedding_obsm: str
        The name of metadata observation in adata.obsm.
    metric: str, default = 'euclidean'
        See sklearn pairwise distance metrics. 
    scale: bool, default = True
        If True, scale between 0 and 1.
    return_all: bool, default = False
        If True, return all silhouette scores and label means.
        By default False, returning only the average width silhouette (ASW).
    
    Returns
    -------
        Batch ASW: int (always)
        Additionally, if return_all=True, return mean silhouette per group in pd.DataFrame.
        as well as the absolute silhouette scores per group label.
    """
    if embedding_obsm not in adata.obsm.keys():
        message = f"{embedding_obsm} not in Adata.obsm!"
        LOGGER.error(message)
        raise KeyError(message)

    sil_per_label = []
    for group in adata.obs[label_obs].unique():
        adata_group = adata[adata.obs[label_obs] == group]
        n_batches = adata_group.obs[batch_obs].nunique()

        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue

        sil = silhouette_samples(
            adata_group.obsm[embedding_obsm], adata_group.obs[batch_obs], metric=metric
        )

        # take only absolute value
        sil = [abs(i) for i in sil]

        if scale:
            # scale s.t. highest number is optimal
            sil = [1 - i for i in sil]

        sil_per_label.extend([(group, score) for score in sil])

    sil_df = pd.DataFrame.from_records(
        sil_per_label, columns=["group", "silhouette_score"]
    )

    if len(sil_per_label) == 0:
        sil_means = np.nan
        asw = np.nan
    else:
        sil_means = sil_df.groupby("group").mean()
        asw = sil_means["silhouette_score"].mean()

    LOGGER.info(f"Mean silhouette per group: {sil_means}")

    if return_all:
        return asw, sil_means, sil_df

    return asw