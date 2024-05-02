from helical.models.scgpt.model import scGPT, scGPTConfig
import anndata as ad

model_config = scGPTConfig(batch_size=10)
scgpt = scGPT("./data/scgpt/scGPT_CP",model_config=model_config)
adata = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
scgpt.process_data(adata[:100], "./data/config.json")
embeddings = scgpt.get_embeddings()

print(embeddings.shape)