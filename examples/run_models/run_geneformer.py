from helical import Geneformer,GeneformerConfig
import hydra
from omegaconf import DictConfig
import anndata as ad
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset


@hydra.main(version_base=None, config_path="configs", config_name="geneformer_config")
def run(cfg: DictConfig):
    geneformer_config = GeneformerConfig(**cfg)
    geneformer = Geneformer(configurer = geneformer_config)

    # either load via huggingface
    # hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly
    ann_data = ad.read_h5ad("./yolksac_human.h5ad")

    dataset = geneformer.process_data(ann_data[:10])
    embeddings = geneformer.get_embeddings(dataset)

    print(embeddings.shape)

if __name__ == "__main__":
    run()