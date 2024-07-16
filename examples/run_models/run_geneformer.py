from helical.models.geneformer.model import Geneformer,GeneformerConfig
import anndata as ad

geneformer_config = GeneformerConfig(batch_size = 10)
geneformer = Geneformer(configurer = geneformer_config)

ann_data = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
dataset = geneformer.process_data(ann_data[:10])
embeddings = geneformer.get_embeddings(dataset)

print(embeddings.shape)
