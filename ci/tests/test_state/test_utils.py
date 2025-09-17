import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
from helical.models.state.model_dir.perturb_utils.utils import pad_adata_with_tsv
import random

def test_pad_adata_with_tsv():
    # Create dummy AnnData object
    n_cells = 100
    n_genes = 50
    X = np.random.poisson(5, size=(n_cells, n_genes))
    
    gene_names = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)]
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'perturbation': ['control'] * 80 + ['GENE_A_knockout'] * 10 + ['GENE_B_overexpression'] * 10
    }
    
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var['gene_name'] = gene_names
    
    tsv_data = {
        'perturbation': ['GENE_C_knockdown', 'GENE_D_knockout'],
        'num_cells': [15, 25]
    }
    tsv_df = pd.DataFrame(tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        tsv_df.to_csv(f.name, sep='\t', index=False)
        tsv_path = f.name
    
    rng = np.random.RandomState(42)
    result_adata = pad_adata_with_tsv(
        adata=adata,
        tsv_path=tsv_path,
        pert_col='perturbation',
        control_pert='control',
        rng=rng,
        quiet=True
    )
    
    assert adata.n_obs == 100
    assert result_adata.n_obs == 140
    assert result_adata.n_vars == n_genes
    
    unique_perturbations = result_adata.obs['perturbation'].unique()
    assert 'GENE_C_knockdown' in unique_perturbations
    assert 'GENE_D_knockout' in unique_perturbations
    
    gene_c_count = (result_adata.obs['perturbation'] == 'GENE_C_knockdown').sum()
    gene_d_count = (result_adata.obs['perturbation'] == 'GENE_D_knockout').sum()
    
    assert gene_c_count == 15
    assert gene_d_count == 25

    control_count = (result_adata.obs['perturbation'] == 'control').sum()
    assert control_count == 80 
    
    gene_a_count = (result_adata.obs['perturbation'] == 'GENE_A_knockout').sum()
    gene_b_count = (result_adata.obs['perturbation'] == 'GENE_B_overexpression').sum()
    
    assert gene_a_count == 10
    assert gene_b_count == 10
        
    gene_c_mask = result_adata.obs['perturbation'] == 'GENE_C_knockdown'
    gene_c_expression = result_adata.X[gene_c_mask]
    
    original_control_mask = adata.obs['perturbation'] == 'control'
    original_control_expression = adata.X[original_control_mask]
    
    assert gene_c_expression.shape[0] == 15
    assert gene_c_expression.shape[1] == n_genes


def test_pad_adata_with_tsv_edge_cases():

    n_cells = 50
    n_genes = 20
    
    X = np.random.poisson(3, size=(n_cells, n_genes))
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'perturbation': ['control'] * 50
    }
    
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var['gene_name'] = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)]
    
    tsv_data = {
        'perturbation': ['GENE_E_knockout'],
        'num_cells': [0]
    }
    tsv_df = pd.DataFrame(tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        tsv_df.to_csv(f.name, sep='\t', index=False)
        tsv_path = f.name

    rng = np.random.RandomState(123)
    
    result_adata = pad_adata_with_tsv(
        adata=adata,
        tsv_path=tsv_path,
        pert_col='perturbation',
        control_pert='control',
        rng=rng,
        quiet=True
    )
    
    assert result_adata.n_obs == adata.n_obs
    assert 'GENE_E_knockout' not in result_adata.obs['perturbation'].values


def test_pad_adata_with_tsv_errors():

    adata = sc.AnnData(X=np.random.poisson(2, size=(10, 5)))
    adata.obs['perturbation'] = ['control'] * 10
    adata.var['gene_name'] = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(5)]
    
    rng = np.random.RandomState(42)
    
    with pytest.raises(FileNotFoundError):
        pad_adata_with_tsv(
            adata=adata,
            tsv_path="non_existent_file.tsv",
            pert_col='perturbation',
            control_pert='control',
            rng=rng,
            quiet=True
        )
    
    invalid_tsv_data = {
        'wrong_column': ['GENE_F'],
        'another_wrong_column': [5]
    }
    invalid_tsv_df = pd.DataFrame(invalid_tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        invalid_tsv_df.to_csv(f.name, sep='\t', index=False)
        invalid_tsv_path = f.name
    
    with pytest.raises(ValueError, match="TSV file missing required columns"):
        pad_adata_with_tsv(
            adata=adata,
            tsv_path=invalid_tsv_path,
            pert_col='perturbation',
            control_pert='control',
            rng=rng,
            quiet=True
        )

    adata_no_control = sc.AnnData(X=np.random.poisson(2, size=(10, 5)))
    adata_no_control.obs['perturbation'] = ['GENE_G_knockout'] * 10
    adata_no_control.var['gene_name'] = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(5)]
    
    tsv_data = {
        'perturbation': ['GENE_H_knockout'],
        'num_cells': [5]
    }
    tsv_df = pd.DataFrame(tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        tsv_df.to_csv(f.name, sep='\t', index=False)
        tsv_path = f.name
    
    with pytest.raises(ValueError, match="No control cells found"):
        pad_adata_with_tsv(
            adata=adata_no_control,
            tsv_path=tsv_path,
            pert_col='perturbation',
            control_pert='control',
            rng=rng,
            quiet=True
        )

if __name__ == "__main__":
    test_pad_adata_with_tsv()
    test_pad_adata_with_tsv_edge_cases()
    test_pad_adata_with_tsv_errors()
    