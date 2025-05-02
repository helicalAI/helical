import hydra
from omegaconf import DictConfig
from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import (
    TranscriptFormerConfig,
)
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset
import anndata as ad


@hydra.main(
    version_base=None, config_path="configs", config_name="transcriptformer_config"
)
def run(cfg: DictConfig):
    configurer = TranscriptFormerConfig(**cfg)
    model = TranscriptFormer(configurer)

    # # either load via huggingface
    # hf_dataset = load_dataset("helical-ai/yolksac_human",split="train[:5%]", trust_remote_code=True, download_mode="reuse_cache_if_exists")
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly
    ann_data = ad.read_h5ad("./yolksac_human.h5ad")

    dataset = model.process_data([ann_data[:10]])
    embeddings = model.get_embeddings(dataset)
    print(embeddings)
    log_likelihoods = model.get_output_adata().uns["llh"]
    print(log_likelihoods)


if __name__ == "__main__":
    run()
