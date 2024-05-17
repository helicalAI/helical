from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad

model_config = scGPTConfig(batch_size=10)
scgpt = scGPT(model_config=model_config)

adata = ad.read_h5ad("./10k_pbmcs_proc.h5ad")
data = scgpt.process_data(adata[:100])
embeddings = scgpt.get_embeddings(data)

print(embeddings.shape)