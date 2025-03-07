from helical.models.uce import UCE, UCEConfig
import hydra
from omegaconf import DictConfig
import numpy as np
import anndata as ad
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset

"""
Because UCE requires a lot of RAM usage, this shows an example of how to run UCE in batches.
"""


@hydra.main(version_base=None, config_path="configs", config_name="uce_config")
def run(cfg: DictConfig):
    configurer = UCEConfig(**cfg)
    uce = UCE(configurer=configurer)
    # either load via huggingface

    # hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly
    ann_data = ad.read_h5ad("./yolksac_human.h5ad")

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
