import anndata as ad
import numpy as np


def guess_is_lognorm(
    adata: ad.AnnData,
    n_cells: int | float = 5e2,
    epsilon: float = 1e-2,
) -> bool:
    """Guess if the input is integer counts or log-normalized.

    This is an _educated guess_ based on whether the fractional component of cell sums.
    This _will not be able_ to distinguish between normalized input and log-normalized input.

    Returns:
        bool: True if the input is lognorm, False otherwise
    """
    # Determine the number of cells to use for the guess
    n_cells = int(min(adata.shape[0], n_cells))

    # Pick a random subset of cells
    cell_mask = np.random.choice(adata.shape[0], n_cells, replace=False)

    # Sum the counts for each cell
    cell_sums = adata.X[cell_mask].sum(axis=1)  # type: ignore (can be float but super unlikely)

    # Check if any cell sum's fractional part is greater than epsilon
    return bool(np.any(np.abs((cell_sums - cell_sums.round())) > epsilon))


def split_anndata_on_celltype(
    adata: ad.AnnData,
    celltype_col: str,
) -> dict[str, ad.AnnData]:
    """Split anndata on celltype column.

    Args:
        adata: AnnData object
        celltype_col: Column name in adata.obs that contains the celltype labels

    Returns:
        dict[str, AnnData]: Dictionary of AnnData objects, keyed by celltype
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Celltype column {celltype_col} not found in adata.obs: {adata.obs.columns}"
        )

    return {
        ct: adata[adata.obs[celltype_col] == ct]
        for ct in adata.obs[celltype_col].unique()
    }
