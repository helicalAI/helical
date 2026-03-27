import hydra
import anndata as ad
from omegaconf import DictConfig
from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import (
    TranscriptFormerConfig,
)
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="transcriptformer_metazoa_config",
)
def run(cfg: DictConfig):
    configurer = TranscriptFormerConfig(**cfg)
    model = TranscriptFormer(configurer)

    ann_data = ad.read_h5ad("./yolksac_human.h5ad")[0:5]

    dataset = model.process_data([ann_data])
    embeddings = model.get_embeddings(dataset)
    print(embeddings)
    log_likelihoods = model.get_output_adata().uns["llh"]
    print(log_likelihoods)


if __name__ == "__main__":
    run()
