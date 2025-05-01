import json
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import TranscriptFormerConfig

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="transcriptformer_config")
def run(cfg: DictConfig):
    logging.debug(OmegaConf.to_yaml(cfg))

    config_path = os.path.join(cfg.model.checkpoint_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    mlflow_cfg = OmegaConf.create(config_dict)

    # Merge the MLflow config with the main config
    cfg = OmegaConf.merge(mlflow_cfg, cfg)

    configurer = TranscriptFormerConfig(cfg)
    model = TranscriptFormer(configurer)
    dataset = model.process_data(["/home/benoit/Documents/helical/examples/run_models/adjusted_17_04_24_YolkSacRaw_F158_WE_annots.h5ad"])
    embeddings = model.get_embeddings(dataset)
    print(embeddings)

if __name__ == "__main__":
    run()
