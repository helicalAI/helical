from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="scgpt_config")
def run(cfg: DictConfig):
    scgpt_config = scGPTConfig(**cfg)
    scgpt = scGPT(configurer = scgpt_config)

    adata = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
    data = scgpt.process_data(adata[:10])
    embeddings = scgpt.get_embeddings(data)

    print(embeddings.shape)

if __name__ == "__main__":
    run()