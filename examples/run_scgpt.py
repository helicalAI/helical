from helical.models.scgpt.model import scGPT
import anndata as ad
import json

with open('./scgpt_config.json') as f:
    config = json.load(f)

scgpt = scGPT("./data/scgpt/scGPT_CP", "./data/config.json")
adata = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
scgpt.process_data(adata[:100])
embeddings = scgpt.get_embeddings()

print(embeddings.shape)