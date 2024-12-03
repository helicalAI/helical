from helical import HelixmRNA, HelixmRNAConfig
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../run_models/configs", config_name="helix_mrna_config")
def run(cfg: DictConfig):
    input_sequences = ["ACUG"*20, "AUGC"*20, "AUGC"*20, "ACUG"*20, "AUUG"*20]

    helix_mrna_config = HelixmRNAConfig(**cfg)
    helix_mrna = HelixmRNA(helix_mrna_config)

    processed_input_data = helix_mrna.process_data(input_sequences)

    embeddings = helix_mrna.get_embeddings(processed_input_data)
    print(embeddings.shape)

if __name__ == "__main__":
    run()
