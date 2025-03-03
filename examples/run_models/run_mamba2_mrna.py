from helical.models.mamba2_mrna import Mamba2mRNA, Mamba2mRNAConfig
import hydra
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../run_models/configs",
    config_name="mamba2_mrna_config",
)
def run(cfg: DictConfig):
    input_sequences = ["ACUG" * 20, "AUGC" * 20, "AUGC" * 20, "ACUG" * 20, "AUUG" * 20]

    mamba2_mrna_config = Mamba2mRNAConfig(**cfg)
    mamba2_mrna = Mamba2mRNA(mamba2_mrna_config)

    processed_input_data = mamba2_mrna.process_data(input_sequences)

    embeddings = mamba2_mrna.get_embeddings(processed_input_data)
    print(embeddings.shape)


if __name__ == "__main__":
    run()
