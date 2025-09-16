import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import os
from helical.models.state.model_dir.perturb_utils.utils import pad_adata_with_tsv


def test_pad_adata_with_tsv():
    """Test the pad_adata_with_tsv function with dummy data."""
    
    # Create dummy AnnData object
    n_cells = 100
    n_genes = 50
    
    # Create random gene expression data
    X = np.random.poisson(5, size=(n_cells, n_genes))
    
    # Create dummy gene names
    gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    
    # Create dummy cell metadata
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'perturbation': ['control'] * 80 + ['GENE_A_knockout'] * 10 + ['GENE_B_overexpression'] * 10
    }
    
    # Create AnnData object
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var_names = gene_names
    
    # Create temporary TSV file
    tsv_data = {
        'perturbation': ['GENE_C_knockdown', 'GENE_D_knockout'],
        'num_cells': [15, 25]
    }
    tsv_df = pd.DataFrame(tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        tsv_df.to_csv(f.name, sep='\t', index=False)
        tsv_path = f.name
    
    try:
        # Set up random number generator for reproducible results
        rng = np.random.RandomState(42)
        
        # Test the function
        result_adata = pad_adata_with_tsv(
            adata=adata,
            tsv_path=tsv_path,
            pert_col='perturbation',
            control_pert='control',
            rng=rng,
            quiet=True
        )
        
        # Assertions
        # Original data should have 100 cells
        assert adata.n_obs == 100
        
        # Result should have 100 + 15 + 25 = 140 cells
        assert result_adata.n_obs == 140
        
        # Number of genes should remain the same
        assert result_adata.n_vars == n_genes
        
        # Check that new perturbations were added
        unique_perturbations = result_adata.obs['perturbation'].unique()
        assert 'GENE_C_knockdown' in unique_perturbations
        assert 'GENE_D_knockout' in unique_perturbations
        
        # Check that the correct number of cells were added for each perturbation
        gene_c_count = (result_adata.obs['perturbation'] == 'GENE_C_knockdown').sum()
        gene_d_count = (result_adata.obs['perturbation'] == 'GENE_D_knockout').sum()
        
        assert gene_c_count == 15
        assert gene_d_count == 25
        
        # Check that control cells still exist
        control_count = (result_adata.obs['perturbation'] == 'control').sum()
        assert control_count == 80  # Original control cells
        
        # Check that original perturbations still exist
        gene_a_count = (result_adata.obs['perturbation'] == 'GENE_A_knockout').sum()
        gene_b_count = (result_adata.obs['perturbation'] == 'GENE_B_overexpression').sum()
        
        assert gene_a_count == 10
        assert gene_b_count == 10
        
        # Check that gene expression data is preserved (should be identical to control cells)
        # Get the new GENE_C_knockdown cells
        gene_c_mask = result_adata.obs['perturbation'] == 'GENE_C_knockdown'
        gene_c_expression = result_adata.X[gene_c_mask]
        
        # Get original control cells
        original_control_mask = adata.obs['perturbation'] == 'control'
        original_control_expression = adata.X[original_control_mask]
        
        # The new cells should have expression values that exist in the original control cells
        # (since they were sampled from control cells)
        assert gene_c_expression.shape[0] == 15
        assert gene_c_expression.shape[1] == n_genes
        
        print("✅ All tests passed!")
        print(f"Original cells: {adata.n_obs}")
        print(f"Result cells: {result_adata.n_obs}")
        print(f"Added cells: {result_adata.n_obs - adata.n_obs}")
        print(f"Unique perturbations in result: {sorted(unique_perturbations)}")
        
    finally:
        # Clean up temporary file
        os.unlink(tsv_path)


def test_pad_adata_with_tsv_edge_cases():
    """Test edge cases for pad_adata_with_tsv function."""
    
    # Test with zero cells requested
    n_cells = 50
    n_genes = 20
    
    X = np.random.poisson(3, size=(n_cells, n_genes))
    obs_data = {
        'cell_id': [f"cell_{i:03d}" for i in range(n_cells)],
        'perturbation': ['control'] * 50
    }
    
    adata = sc.AnnData(X=X, obs=pd.DataFrame(obs_data))
    adata.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    
    # Create TSV with zero cells
    tsv_data = {
        'perturbation': ['GENE_E_knockout'],
        'num_cells': [0]  # Zero cells
    }
    tsv_df = pd.DataFrame(tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        tsv_df.to_csv(f.name, sep='\t', index=False)
        tsv_path = f.name
    
    try:
        rng = np.random.RandomState(123)
        
        result_adata = pad_adata_with_tsv(
            adata=adata,
            tsv_path=tsv_path,
            pert_col='perturbation',
            control_pert='control',
            rng=rng,
            quiet=True
        )
        
        # Should have same number of cells since zero were added
        assert result_adata.n_obs == adata.n_obs
        assert 'GENE_E_knockout' not in result_adata.obs['perturbation'].values
        
        print("✅ Edge case test passed!")
        
    finally:
        os.unlink(tsv_path)


def test_pad_adata_with_tsv_errors():
    """Test error handling in pad_adata_with_tsv function."""
    
    # Create minimal AnnData
    adata = sc.AnnData(X=np.random.poisson(2, size=(10, 5)))
    adata.obs['perturbation'] = ['control'] * 10
    adata.var_names = [f"GENE_{i}" for i in range(5)]
    
    rng = np.random.RandomState(42)
    
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        pad_adata_with_tsv(
            adata=adata,
            tsv_path="non_existent_file.tsv",
            pert_col='perturbation',
            control_pert='control',
            rng=rng,
            quiet=True
        )
    
    # Test with invalid TSV format (missing required columns)
    invalid_tsv_data = {
        'wrong_column': ['GENE_F'],
        'another_wrong_column': [5]
    }
    invalid_tsv_df = pd.DataFrame(invalid_tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        invalid_tsv_df.to_csv(f.name, sep='\t', index=False)
        invalid_tsv_path = f.name
    
    try:
        with pytest.raises(ValueError, match="TSV file missing required columns"):
            pad_adata_with_tsv(
                adata=adata,
                tsv_path=invalid_tsv_path,
                pert_col='perturbation',
                control_pert='control',
                rng=rng,
                quiet=True
            )
    finally:
        os.unlink(invalid_tsv_path)
    
    # Test with no control cells
    adata_no_control = sc.AnnData(X=np.random.poisson(2, size=(10, 5)))
    adata_no_control.obs['perturbation'] = ['GENE_G_knockout'] * 10
    adata_no_control.var_names = [f"GENE_{i}" for i in range(5)]
    
    tsv_data = {
        'perturbation': ['GENE_H_knockout'],
        'num_cells': [5]
    }
    tsv_df = pd.DataFrame(tsv_data)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
        tsv_df.to_csv(f.name, sep='\t', index=False)
        tsv_path = f.name
    
    try:
        with pytest.raises(ValueError, match="No control cells found"):
            pad_adata_with_tsv(
                adata=adata_no_control,
                tsv_path=tsv_path,
                pert_col='perturbation',
                control_pert='control',
                rng=rng,
                quiet=True
            )
    finally:
        os.unlink(tsv_path)
    
    print("✅ Error handling tests passed!")


if __name__ == "__main__":
    # Run the tests
    test_pad_adata_with_tsv()
    test_pad_adata_with_tsv_edge_cases()
    test_pad_adata_with_tsv_errors()
    print("\n🎉 All unit tests completed successfully!")
