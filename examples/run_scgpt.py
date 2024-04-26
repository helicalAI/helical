from helical.models.scgpt.scgpt_helical import SCGPT
import anndata as ad

import json

with open('./scgpt_config.json') as f:
    config = json.load(f)

scgpt = SCGPT(config["model_config"],
          config["data_config"])
adata = ad.read_h5ad("./data/human_pancreas_norm_complexBatch.h5ad")
scgpt.process_data(adata[:100])
embeddings = scgpt.get_embeddings()

print(embeddings.shape)