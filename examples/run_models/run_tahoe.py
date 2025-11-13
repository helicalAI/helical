from helical.models.tahoe import Tahoe, TahoeConfig
import hydra
from omegaconf import DictConfig
import anndata as ad
from datasets import load_dataset
from helical.utils import get_anndata_from_hf_dataset


@hydra.main(version_base=None, config_path="configs", config_name="tahoe_config")
def run(cfg: DictConfig):
    tahoe_config = TahoeConfig(**cfg)
    tahoe = Tahoe(configurer=tahoe_config)

    print(f"Loaded Tahoe model with {tahoe.model.model.n_layers} layers")
    print(f"Embedding dimension: {tahoe.config['d_model']}")

    # either load via huggingface
    # hf_dataset = load_dataset(
    #     "helical-ai/yolksac_human",
    #     split="train[:5%]",
    #     trust_remote_code=True,
    #     download_mode="reuse_cache_if_exists",
    # )
    # ann_data = get_anndata_from_hf_dataset(hf_dataset)

    # or load directly
    ann_data = ad.read_h5ad("./yolksac_human.h5ad")

    # Process data - returns a DataLoader
    dataloader = tahoe.process_data(ann_data[:10])

    # Get cell embeddings from the DataLoader
    cell_embeddings = tahoe.get_embeddings(dataloader)
    print(f"Cell embeddings shape: {cell_embeddings.shape}")

    # Get both cell and gene embeddings
    cell_embeddings, gene_embeddings = tahoe.get_embeddings(
        dataloader, return_gene_embeddings=True
    )
    print(f"Cell embeddings shape: {cell_embeddings.shape}")
    print(f"Gene embeddings shape: {gene_embeddings.shape}")


if __name__ == "__main__":
    run()
