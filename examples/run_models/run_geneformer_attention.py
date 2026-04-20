from helical.models.geneformer import Geneformer, GeneformerConfig
import hydra
from omegaconf import DictConfig
import anndata as ad


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="geneformer_attention_config",
)
def run(cfg: DictConfig):
    geneformer_config = GeneformerConfig(**cfg)
    geneformer = Geneformer(configurer=geneformer_config)

    ann_data = ad.read_h5ad("./yolksac_human.h5ad")

    dataset = geneformer.process_data(ann_data[:10, :100])
    embeddings, attention_weights = geneformer.get_embeddings(
        dataset, output_attentions=True
    )

    print(embeddings)
    print(attention_weights)


if __name__ == "__main__":
    run()
