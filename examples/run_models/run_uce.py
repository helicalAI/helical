from helical.models.uce.model import UCE, UCEConfig
import anndata as ad
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="uce_config")
def run(cfg: DictConfig):
    configurer=UCEConfig(**cfg)
    uce = UCE(configurer=configurer)
    ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
    dataset = uce.process_data(ann_data[:10])
    embeddings = uce.get_embeddings(dataset)

    print(embeddings.shape)

if __name__ == "__main__":
    run()