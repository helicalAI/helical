from helical.models.geneformer.model import Geneformer
import anndata as ad

geneformer = Geneformer("./data/geneformer")
ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
dataset = geneformer.process_data(ann_data[:100], "./data/config.json")
embeddings = geneformer.get_embeddings(dataset)

print(embeddings.shape)
