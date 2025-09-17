import pytest
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import os
from helical.models.state.model_dir.perturb_utils.utils import pad_adata_with_tsv

def test_initialization():
    from helical.models.state import StateEmbed, StateConfig
    
    config = StateConfig(batch_size=8)
    state_embed = StateEmbed(configurer=config)

    assert state_embed.batch_size == 8
    assert state_embed.device_type in ["cuda", "cpu"]
    assert state_embed.model is not None
            

def test_process_data():

    from helical.models.state import StateEmbed, StateConfig
    import random
    
    # dummy data
    n_cells = 15
    n_genes = 50
    X = np.random.poisson(3, size=(n_cells, n_genes))
    gene_names = [random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)]
        
    adata = sc.AnnData(X=X, obs=pd.DataFrame({'cell_id': [f"cell_{i:03d}" for i in range(n_cells)]}))
    adata.var['gene_name'] = gene_names
    
    config = StateConfig(batch_size=4)
    state_embed = StateEmbed(configurer=config)
    dataloader = state_embed.process_data(adata)

    assert dataloader is not None
    assert hasattr(dataloader, '__len__')
    assert hasattr(dataloader, '__iter__')

def test_auto_detect_gene_column():

    from helical.models.state import StateEmbed, StateConfig

    n_cells = 10
    n_genes = 20    
    X = np.random.poisson(2, size=(n_cells, n_genes))
    adata_index = sc.AnnData(X=X, obs=pd.DataFrame({'cell_id': [f"cell_{i:03d}" for i in range(n_cells)]}))
    adata_index.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    
    adata_column = sc.AnnData(X=X, obs=pd.DataFrame({'cell_id': [f"cell_{i:03d}" for i in range(n_cells)]}))
    adata_column.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    adata_column.var['gene_symbol'] = [f"GENE_SYMBOL_{i:03d}" for i in range(n_genes)]
    
    config = StateConfig(batch_size=4)
    state_embed = StateEmbed(configurer=config)

    gene_col_index = state_embed._auto_detect_gene_column(adata_index)
    assert gene_col_index is None


def test_dataset_metadata():

    from helical.models.state import StateEmbed, StateConfig

    n_cells = 25
    n_genes = 30    
    X = np.random.poisson(4, size=(n_cells, n_genes))
    adata = sc.AnnData(X=X, obs=pd.DataFrame({'cell_id': [f"cell_{i:03d}" for i in range(n_cells)]}))
    adata.var_names = [f"GENE_{i:03d}" for i in range(n_genes)]
    
    config = StateConfig(batch_size=4)
    
    state_embed = StateEmbed(configurer=config)
    
    metadata = state_embed._StateEmbed__load_dataset_meta_from_adata(adata)
    
    assert isinstance(metadata, dict)
    assert "inference" in metadata
    assert metadata["inference"] == (n_cells, n_genes)
    
    custom_metadata = state_embed._StateEmbed__load_dataset_meta_from_adata(adata, "test_dataset")
    assert "test_dataset" in custom_metadata
    assert custom_metadata["test_dataset"] == (n_cells, n_genes)


def test_gene_embedding():

    from helical.models.state import StateEmbed, StateConfig
    
    # dummy non-existent gene list
    genes = ["GENE_001", "GENE_002", "GENE_003"]
    
    config = StateConfig(batch_size=4)
    
    state_embed = StateEmbed(configurer=config)
    
    gene_embeddings = state_embed.get_gene_embedding(genes)
    assert gene_embeddings is not None
    assert hasattr(gene_embeddings, 'shape')
    assert gene_embeddings.shape[0] == len(genes)

def test_config_validation():

    from helical.models.state import StateConfig
    
    config_default = StateConfig()
    assert config_default.config["batch_size"] == 16
    assert config_default.config["output_path"] == "prediction.h5ad"
    
    config_custom = StateConfig(
        batch_size=32,
        output_path="custom_output.h5ad",
        seed=123
    )
    assert config_custom.config["batch_size"] == 32
    assert config_custom.config["output_path"] == "custom_output.h5ad"
    assert config_custom.config["seed"] == 123

def test_process_data_error():

    from helical.models.state import StateEmbed, StateConfig
    import logging
    logging.basicConfig(level=logging.DEBUG)
    # dummy data
    adata_no_genes = sc.AnnData(X=np.random.poisson(2, size=(10, 0)))
    adata_no_genes.obs = pd.DataFrame({'cell_id': [f"cell_{i:03d}" for i in range(10)]})
    
    config = StateConfig(batch_size=4)
    state_embed = StateEmbed(configurer=config)
    
    # we expect an error as no gene columns defined 
    try:
        dataloader = state_embed.process_data(adata_no_genes)
    except Exception as e:
        logging.info(e)

if __name__ == "__main__":
    test_initialization()
    test_process_data()
    test_auto_detect_gene_column()
    test_dataset_metadata()
    test_gene_embedding()
    test_config_validation()
    test_process_data_error()