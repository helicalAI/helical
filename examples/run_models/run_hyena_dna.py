from helical.models.hyena_dna import HyenaDNA, HyenaDNAConfig
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="hyena_dna_config")
def run(cfg: DictConfig):
    hyena_config = HyenaDNAConfig(**cfg)
    model = HyenaDNA(configurer=hyena_config)

    sequence = ["A", "CC", "TTTT", "ACGTN", "ACGT"]

    dataset = model.process_data(sequence)
    embeddings = model.get_embeddings(dataset)
    print(embeddings.shape)


if __name__ == "__main__":
    run()
