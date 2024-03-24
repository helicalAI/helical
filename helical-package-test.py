from helical.models.uce.uce import UCE
from uce_config import model_config, files_config, data_config
from accelerate import Accelerator
import anndata as ad
from pathlib import Path
accelerator = Accelerator(project_dir=data_config["dir"])
import pandas as pd

uce = UCE()

model = uce.get_model(model_config, data_config, files_config, accelerator=accelerator)
df = pd.read_csv(Path('/Users/bputzeys/Documents/Helical/Code Repos/helical-package/out.csv'))
ann_data = ad.AnnData(X = df)
data_loader = uce.process_data(ann_data)
embeddings = uce.get_embeddings(data_loader)

print(embeddings.shape)
