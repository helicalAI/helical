from helical.models.geneformer.model import Geneformer,GeneformerConfig
import anndata as ad
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="geneformer_config")
def run(cfg: DictConfig):
    geneformer_config = GeneformerConfig(**cfg)
    geneformer = Geneformer(configurer = geneformer_config)

    ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
    dataset = geneformer.process_data(ann_data[:10])
    embeddings = geneformer.get_embeddings(dataset)

    print(embeddings.shape)



if __name__ == "__main__":
    run()