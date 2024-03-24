from helical.models.uce.uce import UCE
import json
from accelerate import Accelerator
import anndata as ad
from pathlib import Path
import pandas as pd

uce = UCE()

with open('uce_config.json') as f:
    config = json.load(f)

accelerator = Accelerator(project_dir=config["data_config"]["dir"])
model = uce.get_model(config["model_config"], config["data_config"],  config["files_config"], accelerator=accelerator)
df = pd.read_csv(Path('/Users/bputzeys/Documents/Helical/Code Repos/helical-package/out.csv'))
ann_data = ad.AnnData(X = df)
data_loader = uce.process_data(ann_data)
embeddings = uce.get_embeddings(data_loader)

print(embeddings.shape)
