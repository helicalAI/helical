from helical.models.uce.model import UCE, UCEConfig
import anndata as ad
import hydra
from omegaconf import DictConfig
import numpy as np
"""
Because UCE requires a lot of RAM usage, this shows an example of how to run UCE in batches.
"""

@hydra.main(version_base=None, config_path="configs", config_name="uce_config")
def run(cfg: DictConfig):
    configurer=UCEConfig(**cfg)
    uce = UCE(configurer=configurer)
    ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad", backed='r')
    
    batch_size = 10

    # Initialize a list to store embeddings from each batch
    all_embeddings = []

    # Iterate over the data in batches
    for start in range(0, ann_data.shape[0], batch_size):
        end = min(start + batch_size, ann_data.shape[0])
        ann_data_batch = ann_data[start:end].to_memory()
        
        dataset_batch = uce.process_data(ann_data_batch)
        embeddings_batch = uce.get_embeddings(dataset_batch)
        
        all_embeddings.append(embeddings_batch)
        
        # Only run the first batch to show how it works
        break

    # Concatenate the embeddings from each batch
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(all_embeddings.shape)
if __name__ == "__main__":
    run()