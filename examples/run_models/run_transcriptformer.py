import json
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from helical.models.transcriptformer.model import TranscriptFormer

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="transcriptformer_config")
def main(cfg: DictConfig):
    logging.debug(OmegaConf.to_yaml(cfg))

    config_path = os.path.join(cfg.model.checkpoint_path, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    mlflow_cfg = OmegaConf.create(config_dict)

    # Merge the MLflow config with the main config
    cfg = OmegaConf.merge(mlflow_cfg, cfg)

    # Set the checkpoint paths based on the unified checkpoint_path
    cfg.model.inference_config.load_checkpoint = os.path.join(cfg.model.checkpoint_path, "model_weights.pt")
    # cfg.model.data_config.aux_vocab_path = os.path.join(cfg.model.checkpoint_path, "vocabs")
    cfg.model.data_config.aux_vocab_path = None
    cfg.model.data_config.aux_cols = None
    cfg.model.data_config.esm2_mappings_path = os.path.join(cfg.model.checkpoint_path, "vocabs")

    model = TranscriptFormer(cfg)
    dataset = model.process_data(cfg.model.inference_config.data_files)
    embeddings = model.get_embeddings(dataset)
    print(embeddings)

if __name__ == "__main__":
    main()
