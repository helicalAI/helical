from helical.models.uce.model import UCE, UCEConfig
import anndata as ad
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="uce_config")
def run(cfg: DictConfig):
    configurer=UCEConfig(**cfg)
    uce = UCE(configurer=configurer)
    ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
    batch_size = 1000

    # Initialize a list to store embeddings from each batch
    all_embeddings = []

    # Iterate over the data in batches
    for start in range(0, ann_data.shape[0], batch_size):
        end = min(start + batch_size, ann_data.shape[0])
        ann_data_batch = ann_data[start:end].copy()
        dataset_batch = uce.process_data(ann_data_batch)
        embeddings_batch = uce.get_embeddings(dataset_batch)
        all_embeddings.append(embeddings_batch)
        
        # Optionally, you can print progress
        print(f"Processed batch {start} to {end}")

    print(all_embeddings.shape)

if __name__ == "__main__":
    run()