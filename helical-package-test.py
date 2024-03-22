from helical.models.uce.uce import UCE
from uce_config import model_config, files_config, data_config
from accelerate import Accelerator
import anndata as ad

accelerator = Accelerator(project_dir=data_config["dir"])

uce = UCE()

model_path = uce.get_model(model_config, data_config, files_config, accelerator=accelerator)
ann_data = ad.read_h5ad("./data/full_cells_macaca.h5ad")
data_loader = uce.process_data(ann_data)
embeddings = uce.get_embeddings(data_loader)

print(embeddings.shape)
