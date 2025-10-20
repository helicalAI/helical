import numpy as np
import pandas as pd
import scanpy as sc
from helical.models.state import StatePerturb, StateConfig
import random
import logging
import pytest

config = StateConfig(
    pert_col="target_gene",
    control_pert="non-targeting",
    batch_col="batch_var",
    celltype_col="cell_type",
)
state_perturb = StatePerturb(configurer=config)


n_cells = 20
n_genes = 2000

X = np.random.poisson(3, size=(n_cells, n_genes))
obs_data = {
    "cell_id": [f"cell_{i:03d}" for i in range(n_cells)],
    "target_gene": ["non-targeting"] * 15
    + ["GENE_A_knockout"] * 3
    + ["GENE_B_overexpression"] * 2,
    "batch_var": ["batch_1"] * 10 + ["batch_2"] * 10,
    "cell_type": ["type_A"] * 12 + ["type_B"] * 8,
}

adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
adata.var["gene_name"] = [
    random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)
]


def test_initialization():
    assert hasattr(state_perturb, "cell_set_len")
    assert state_perturb.device is not None
    assert state_perturb.model is not None


def test_process_data():
    processed_adata = state_perturb.process_data(adata)

    # Basic structure
    assert processed_adata is not None
    assert hasattr(processed_adata, 'n_obs')
    assert hasattr(processed_adata, 'n_vars')
    assert hasattr(processed_adata, 'X')
    assert hasattr(processed_adata, 'obs')
    assert hasattr(processed_adata, 'var')
    
    # Dimensions
    assert processed_adata.n_obs == n_cells
    assert processed_adata.n_vars == n_genes
    
    # Data types
    assert isinstance(processed_adata.X, np.ndarray) or hasattr(processed_adata.X, 'toarray')
    assert isinstance(processed_adata.obs, pd.DataFrame)
    assert isinstance(processed_adata.var, pd.DataFrame)
    
    # Data validity
    X_data = processed_adata.X.toarray() if hasattr(processed_adata.X, 'toarray') else processed_adata.X
    assert np.all(X_data >= 0)  # Non-negative values
    assert np.all(np.isfinite(X_data))  # Finite values
    
    # Batch indices
    assert hasattr(state_perturb, "batch_indices_all")
    assert isinstance(state_perturb.batch_indices_all, (list, np.ndarray))
    assert len(state_perturb.batch_indices_all) == n_cells
    
    # Data preservation - check that original data is preserved
    original_X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    processed_X = processed_adata.X.toarray() if hasattr(processed_adata.X, 'toarray') else processed_adata.X
    assert np.allclose(original_X, processed_X)
    assert processed_adata.obs.equals(adata.obs)
    assert processed_adata.var.equals(adata.var)

def test_celltype_processing():
    config = StateConfig(
        pert_col="target_gene",
        control_pert="non-targeting",
        celltype_col="cell_type",
        celltypes="type_A,type_B",
    )
    state_perturb = StatePerturb(configurer=config)
    processed_adata = state_perturb.process_data(adata)
    assert processed_adata is not None
    assert processed_adata.n_obs <= n_cells


def test_pert_col_missing_error_handling():

    logging.basicConfig(level=logging.DEBUG)
    adata_no_pert = sc.AnnData(X=np.random.poisson(2, size=(10, 5)))
    adata_no_pert.obs = pd.DataFrame({"cell_id": [f"cell_{i:03d}" for i in range(10)]})
    adata_no_pert.var["gene_name"] = [
        random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(5)
    ]
    config_custom = StateConfig(pert_col="non-existent-column", control_pert="non-targeting")
    state_perturb = StatePerturb(configurer=config_custom)
    with pytest.raises(KeyError):
        state_perturb.process_data(adata_no_pert)


if __name__ == "__main__":
    test_initialization()
    test_process_data()
    test_celltype_processing()
    test_pert_col_missing_error_handling()
