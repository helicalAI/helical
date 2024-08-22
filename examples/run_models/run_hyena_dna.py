from helical.models.hyena_dna.model import HyenaDNA, HyenaDNAConfig
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="hyena_dna_config")
def run(cfg: DictConfig):

    hyena_config = HyenaDNAConfig(**cfg)
    model = HyenaDNA(configurer = hyena_config)   
    sequence = 'ACTG' * int(1024/4)
    tokenized_sequence = model.process_data(sequence)
    embeddings = model.get_embeddings(tokenized_sequence)
    print(embeddings.shape)

if __name__ == "__main__":
    run()