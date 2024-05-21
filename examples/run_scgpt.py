from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad

scgpt_config = scGPTConfig(batch_size=10)
scgpt = scGPT(configurer = scgpt_config)

adata = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
data = scgpt.process_data(adata[:10])
embeddings = scgpt.get_embeddings(data)

print(embeddings.shape)