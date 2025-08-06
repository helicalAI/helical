from helical.models.scgpt import scGPT, scGPTConfig
import hydra
from omegaconf import DictConfig
import anndata as ad
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset


@hydra.main(version_base=None, config_path="configs", config_name="scgpt_config")
def run(cfg: DictConfig):
    scgpt_config = scGPTConfig(**cfg)
    scgpt = scGPT(configurer=scgpt_config)

    # either load via huggingface
    # hf_dataset = load_dataset(
    #     "helical-ai/yolksac_human",
    #     split="train[:5%]",
    #     trust_remote_code=True,
    #     download_mode="reuse_cache_if_exists",
    # )
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly
    ann_data = ad.read_h5ad("./yolksac_human.h5ad")

    data = scgpt.process_data(ann_data[:10])
    embeddings, input_genes = scgpt.get_embeddings(data, output_genes=True)
    print(embeddings, input_genes)
    embeddings, attn_weights = scgpt.get_embeddings(data, output_attentions=True)

    print(embeddings)
    print(attn_weights.shape)


if __name__ == "__main__":
    run()
