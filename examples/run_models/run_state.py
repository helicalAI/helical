from helical.models.state import stateEmbeddingsModel, stateConfig
import hydra
from omegaconf import DictConfig
import scanpy as sc
import os

@hydra.main(version_base=None, config_path="configs", config_name="state_config")
def run(cfg: DictConfig):
    # Get device parameter
    device = cfg.get("device", "cuda")
    
    # Load data with error handling
    data_path = "competition_support_set/competition_val_template.h5ad"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Please check the path.")
        return
    
    try:
        adata = sc.read_h5ad(data_path)
        print(f"Loaded data with shape: {adata.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Use only first 2 samples for testing
    adata = adata[:2].copy()
    print(f"Using subset with shape: {adata.shape}")

    # Create config and model
    state_config = stateConfig(**cfg)
    state_embed = stateEmbeddingsModel(configurer=state_config)

    # Process data and get embeddings
    try:
        processed_data = state_embed.process_data(adata=adata)
        embeddings = state_embed.get_embeddings(processed_data)
        print(f"Generated embeddings with shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error processing data: {e}")
        return


if __name__ == "__main__":
    run()
