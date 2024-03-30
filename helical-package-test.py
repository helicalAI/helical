from helical.models.uce.uce import UCE
from helical.models.geneformer.geneformer import Geneformer

import json
from accelerate import Accelerator
import anndata as ad
from pathlib import Path
import pandas as pd
import pickle as pkl

uce = UCE()

with open('./uce_config.json') as f:
    config = json.load(f)

accelerator = Accelerator(project_dir=config["data_config"]["dir"])
model = uce.get_model(config["model_config"], 
                      config["data_config"],  
                      config["files_config"], 
                      accelerator=accelerator)

ann_data = ad.read_h5ad(config["data_config"]["adata_path"])
data_loader = uce.process_data(ann_data[:100])
embeddings = uce.get_embeddings(data_loader)

print(embeddings.shape)


geneformer = Geneformer()

with open('./geneformer_config.json') as f:
    config = json.load(f)

accelerator = Accelerator(project_dir=config["data_config"]["dir"], cpu=True)
model = geneformer.get_model(config["model_config"], 
                             config["data_config"],  
                             config["files_config"], 
                             accelerator=accelerator)

ann_data = ad.read_h5ad("./data/10k_pbmcs_proc.h5ad")
mappings = pkl.load(open('./helical/models/geneformer/human_gene_to_ensemble_id.pkl', 'rb'))
ann_data.var['ensembl_id'] = ann_data.var['gene_symbols'].apply(lambda x: mappings.get(x,{"id":None})['id'])
dataset = geneformer.process_data(ann_data[:100], species=config["data_config"]["species"])
embeddings = geneformer.get_embeddings(dataset)

print(embeddings.shape)
