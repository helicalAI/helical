import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import os
from helical.models.state.model_dir.perturb_utils.utils import pad_adata_with_tsv

def test_initialization():

    from helical.models.state import StatePerturb, StateConfig

    config = StateConfig(
        pert_col="target_gene",
        control_pert="non-targeting",
        batch_col="batch_var"
    )
    
    state_perturb = StatePerturb(configurer=config)

    assert hasattr(state_perturb, 'cell_set_len')
    assert state_perturb.device is not None
    assert state_perturb.model is not None


def test_process_data():

    from helical.models.state import StatePerturb, StateConfig
    import random
    n_cells = 20
    n_genes = 50
    
    X = np.random.poisson(3, size=(n_cells, n_genes))
    
    gene_names = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)]
    
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'target_gene': ['non-targeting'] * 15 + ['GENE_A_knockout'] * 3 + ['GENE_B_overexpression'] * 2,
        'batch_var': ['batch_1'] * 10 + ['batch_2'] * 10,
        'cell_type': ['type_A'] * 12 + ['type_B'] * 8
    }
    
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var['gene_name'] = gene_names
    
    config = StateConfig(
        pert_col="target_gene",
        control_pert="non-targeting",
        batch_col="batch_var",
        celltype_col="cell_type"
    )
    
    state_perturb = StatePerturb(configurer=config)
    
    # Test process_data method
    processed_adata = state_perturb.process_data(adata)
    
    # Check that processed data is returned
    assert processed_adata is not None
    assert processed_adata.n_obs == n_cells
    assert processed_adata.n_vars == n_genes
    assert hasattr(state_perturb, 'batch_indices_all')


def test_celltype_processing():
    from helical.models.state import StatePerturb, StateConfig
    import random
    n_cells = 12
    n_genes = 25
    
    X = np.random.poisson(3, size=(n_cells, n_genes))
    
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'target_gene': ['non-targeting'] * n_cells,
        'cell_type': ['T_cell'] * 5 + ['B_cell'] * 4 + ['NK_cell'] * 3
    }
    
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var['gene_name'] = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)]
    
    config = StateConfig(
        pert_col="target_gene",
        control_pert="non-targeting",
        celltype_col="cell_type",
        celltypes="T_cell,B_cell"
    )
    
    state_perturb = StatePerturb(configurer=config)
    processed_adata = state_perturb.process_data(adata)

    assert processed_adata is not None
    assert processed_adata.n_obs <= n_cells


def test_embedding_processing():
    from helical.models.state import StatePerturb, StateConfig
    
    import random
    n_cells = 10
    n_genes = 20
    embed_dim = 128
    
    X = np.random.poisson(2, size=(n_cells, n_genes))
    embeddings = np.random.normal(0, 1, size=(n_cells, embed_dim))
    
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'target_gene': ['non-targeting'] * n_cells
    }
    
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var['gene_name'] = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)]
    adata.obsm['state_emb'] = embeddings  # Add embeddings
    
    config = StateConfig(
        pert_col="target_gene",
        control_pert="non-targeting",
        embed_key="state_emb"  # Use embeddings instead of expression
    )
    
    state_perturb = StatePerturb(configurer=config)
        
    processed_adata = state_perturb.process_data(adata)
    
    assert processed_adata is not None
    assert processed_adata.n_obs == n_cells


def test_get_embeddings():
    import random
    from helical.models.state import StatePerturb, StateConfig
    
    n_cells = 6
    n_genes = 2000
    
    X = np.random.poisson(2, size=(n_cells, n_genes))
    
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'target_gene': ['non-targeting'] * n_cells,
        'batch_var': ['batch_1'] * n_cells
    }
    
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var['gene_name'] = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)]
    
    config = StateConfig(
        pert_col="target_gene",
        control_pert="non-targeting"
    )
    
    state_perturb = StatePerturb(configurer=config)
    
    processed_adata = state_perturb.process_data(adata)
    embeddings = state_perturb.get_embeddings(processed_adata)     

    assert embeddings is not None
    assert hasattr(embeddings, 'shape')
    assert embeddings.shape[0] == n_cells


def test_pert_col_missing_error_handling():

    from helical.models.state import StatePerturb, StateConfig
    import random
    import logging

    logging.basicConfig(level=logging.DEBUG)
    adata_no_pert = sc.AnnData(X=np.random.poisson(2, size=(10, 5)))
    adata_no_pert.obs = pd.DataFrame({'cell_id': [f"cell_{i:03d}" for i in range(10)]})
    adata_no_pert.var['gene_name'] = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(5)]
    
    config = StateConfig(
        pert_col="non-existent-column",
        control_pert="non-targeting"
    )
    
    state_perturb = StatePerturb(configurer=config)
        
    try:
        state_perturb.process_data(adata_no_pert)
    except Exception as e:
        logging.info(f"StatePerturb raised appropriate error for missing perturbation column: {type(e).__name__}")

if __name__ == "__main__":
    test_initialization()
    test_process_data()
    test_celltype_processing()
    test_embedding_processing()
    test_get_embeddings()
    test_pert_col_missing_error_handling()
