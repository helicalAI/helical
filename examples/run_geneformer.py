from helical.models.geneformer.model import Geneformer,GeneformerConfig
import anndata as ad


model_config=GeneformerConfig(batch_size=10)
geneformer = Geneformer(model_config=model_config)

ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
dataset = geneformer.process_data(ann_data[:100], "./data/config.json")
embeddings = geneformer.get_embeddings(dataset)

print(embeddings.shape)
