import logging
import hydra
from omegaconf import DictConfig
from helical.models.transcriptformer.model import TranscriptFormer
from helical.models.transcriptformer.transcriptformer_config import TranscriptFormerConfig

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="transcriptformer_config")
def run(cfg: DictConfig):
    configurer = TranscriptFormerConfig(cfg)
    model = TranscriptFormer(configurer)
    dataset = model.process_data(["/home/benoit/Documents/helical/examples/run_models/adjusted_17_04_24_YolkSacRaw_F158_WE_annots.h5ad"])
    embeddings = model.get_embeddings(dataset)
    print(embeddings)

if __name__ == "__main__":
    run()
