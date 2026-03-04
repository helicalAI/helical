from helical.models.nicheformer import Nicheformer, NicheformerConfig
import hydra
from omegaconf import DictConfig
import scanpy as sc


@hydra.main(version_base=None, config_path="configs", config_name="nicheformer_config")
def run(cfg: DictConfig):
    nicheformer_config = NicheformerConfig(**cfg)
    nicheformer = Nicheformer(configurer=nicheformer_config)

    # PBMC 1K v3 — human dissociated 10x dataset with raw integer counts.
    # Preprocessing follows the Nicheformer data preparation pattern (Lu_2021):
    # assign the three obs columns that the tokenizer uses to prepend context tokens.
    ann_data = sc.read_10x_h5("./nicheformer_pbmc_1k_v3.h5")
    ann_data.var_names_make_unique()
    ann_data.obs["modality"] = "dissociated"
    ann_data.obs["specie"] = "human"
    ann_data.obs["assay"] = "10x 3' v3"

    dataset = nicheformer.process_data(ann_data[:10])

    cell_embeddings = nicheformer.get_embeddings(dataset)
    print(f"Cell embeddings shape: {cell_embeddings.shape}")


if __name__ == "__main__":
    run()
