from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
import hydra
from omegaconf import DictConfig
from pandas import DataFrame


@hydra.main(
    version_base=None,
    config_path="../run_models/configs",
    config_name="helix_mrna_config",
)
def run(cfg: DictConfig):
    input_sequences = DataFrame(
        {"Sequence": ["EACU" * 20, "EAUG" * 20, "EAUG" * 20, "EACU" * 20, "EAUU" * 20]}
    )

    helix_mrna_config = HelixmRNAConfig(**cfg)
    helix_mrna = HelixmRNA(helix_mrna_config)

    processed_input_data = helix_mrna.process_data(input_sequences)

    embeddings = helix_mrna.get_embeddings(processed_input_data)
    print(embeddings.shape)


if __name__ == "__main__":
    run()
