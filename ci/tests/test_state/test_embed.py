import numpy as np
import pandas as pd
import scanpy as sc
from helical.models.state.model_dir.perturb_utils.utils import pad_adata_with_tsv
from helical.models.state import StateEmbed, StateConfig
import random
import pytest
import logging
import torch

config = StateConfig(batch_size=8)
state_embed = StateEmbed(configurer=config)

# dummy data
n_cells = 15
n_genes = 1000
X = np.random.poisson(3, size=(n_cells, n_genes))
adata = sc.AnnData(
    X=X, obs=pd.DataFrame({"cell_id": [f"cell_{i:03d}" for i in range(n_cells)]})
)

adata.var["gene_name"] = [
    random.choice(["ABCC3", "ACOX1", "AKIP1", "ANGPTL4"]) for i in range(n_genes)
]


def test_initialization():
    assert state_embed.batch_size == 8
    assert state_embed.device_type in ["cuda", "cpu"]
    assert state_embed.model is not None


def test_process_data():
    """Test process_data method with immediate assertions."""
    dataloader = state_embed.process_data(adata)
    
    # Basic structure
    assert dataloader is not None
    assert hasattr(dataloader, "__len__")
    assert hasattr(dataloader, "__iter__")
    
    # Dataloader length
    dataloader_length = len(dataloader)
    assert isinstance(dataloader_length, int)
    assert dataloader_length > 0
    
    # Get first batch by iteration
    first_batch = next(iter(dataloader))
    assert isinstance(first_batch, tuple)
    assert len(first_batch) >= 3  # Should have at least X, gene_ids, total_counts
    
    X_tensor = first_batch[0]
    gene_ids_tensor = first_batch[1] 
    total_counts_tensor = first_batch[2]

    assert isinstance(X_tensor, torch.Tensor)
    assert isinstance(gene_ids_tensor, torch.Tensor)
    assert isinstance(total_counts_tensor, torch.Tensor)
    
    # Dimensions
    batch_size = X_tensor.shape[0]
    assert X_tensor.shape[1] == 2048
    assert gene_ids_tensor.shape[0] == batch_size
    assert total_counts_tensor.shape[0] == batch_size
    
    # Data validity
    assert torch.all(X_tensor >= 0)
    assert torch.all(total_counts_tensor >= 0)
    assert torch.all(torch.isfinite(X_tensor))
    assert torch.all(torch.isfinite(total_counts_tensor))
    assert torch.all(gene_ids_tensor >= 0)
    

def test_auto_detect_gene_column():
    gene_col_index = state_embed._auto_detect_gene_column(adata)
    assert gene_col_index == "gene_name"


def test_dataset_metadata():
    metadata = state_embed._StateEmbed__load_dataset_meta_from_adata(adata)
    assert isinstance(metadata, dict)
    assert "inference" in metadata
    assert metadata["inference"] == (n_cells, n_genes)

    custom_metadata = state_embed._StateEmbed__load_dataset_meta_from_adata(
        adata, "test_dataset"
    )
    assert "test_dataset" in custom_metadata
    assert custom_metadata["test_dataset"] == (n_cells, n_genes)


def test_gene_embedding():
    # dummy non-existent list of genes
    genes = ["GENE_001", "GENE_002", "GENE_003"]
    gene_embeddings = state_embed.get_gene_embedding(genes)

    assert gene_embeddings is not None
    assert hasattr(gene_embeddings, "shape")
    assert gene_embeddings.shape[0] == len(genes)


def test_config_validation():

    assert config.config["batch_size"] == 8
    assert config.config["output_path"] == "prediction.h5ad"

    config_custom = StateConfig(
        batch_size=32, output_path="custom_output.h5ad", seed=123
    )
    assert config_custom.config["batch_size"] == 32
    assert config_custom.config["output_path"] == "custom_output.h5ad"
    assert config_custom.config["seed"] == 123


def test_process_data_error():

    logging.basicConfig(level=logging.DEBUG)
    # dummy data with no genes
    adata_no_genes = sc.AnnData(X=np.random.poisson(2, size=(10, 10)))
    adata_no_genes.obs = pd.DataFrame({"cell_id": [f"cell_{i:03d}" for i in range(10)]})

    with pytest.raises(AssertionError):
        state_embed.process_data(adata_no_genes)

if __name__ == "__main__":
    test_initialization()
    test_process_data()
    test_auto_detect_gene_column()
    test_dataset_metadata()
    test_gene_embedding()
    test_config_validation()
    test_process_data_error()
