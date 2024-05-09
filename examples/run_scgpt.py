from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad

model_config = scGPTConfig(batch_size=10)
scgpt = scGPT(model_config=model_config)

adata = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
scgpt.process_data(adata[:10])
embeddings = scgpt.get_embeddings()

print(embeddings.shape)