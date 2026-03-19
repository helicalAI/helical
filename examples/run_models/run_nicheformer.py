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
    print(f"Cell embeddings shape (Ensembl IDs): {cell_embeddings.shape}")

    # yolksac uses gene symbols — exercises the symbol-to-Ensembl mapping path.
    ann_data_yolksac = ad.read_h5ad("./yolksac_human.h5ad")
    ann_data_yolksac.obs["modality"] = "dissociated"
    ann_data_yolksac.obs["specie"] = "human"
    ann_data_yolksac.obs["assay"] = "10x 3' v3"

    dataset_yolksac = nicheformer.process_data(ann_data_yolksac[:10])

    cell_embeddings_yolksac = nicheformer.get_embeddings(dataset_yolksac)
    print(f"Cell embeddings shape (gene symbols): {cell_embeddings_yolksac.shape}")


if __name__ == "__main__":
    run()
