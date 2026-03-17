from helical.models.nicheformer import Nicheformer, NicheformerConfig
import anndata as ad
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="nicheformer_config")
def run(cfg: DictConfig):
    nicheformer_config = NicheformerConfig(**cfg)
    nicheformer = Nicheformer(configurer=nicheformer_config)

    ann_data = ad.read_h5ad("./nicheformer_human_bladder.h5ad")
    # CELLxGENE datasets store normalized values in X and raw integer counts in raw.X.
    ann_data = ann_data.raw.to_adata()
    ann_data.obs["modality"] = "dissociated"
    ann_data.obs["specie"] = "human"
    ann_data.obs["assay"] = "10x 3' v3"

    dataset = nicheformer.process_data(ann_data[:10])

    cell_embeddings = nicheformer.get_embeddings(dataset)
    print(f"Cell embeddings shape: {cell_embeddings.shape}")


if __name__ == "__main__":
    run()
