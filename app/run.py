from helical.models.uce.uce import UCE
from accelerate import Accelerator
import anndata as ad

class Run:
    def __init__(self) -> None: 
        self.uce = UCE()

    def init_data(self, df):
        self.ann_data = ad.AnnData(X = df)

    def init_model(self, model_config, data_config, files_config):
        accelerator = Accelerator(project_dir=data_config["dir"])
        model = self.uce.get_model(model_config, data_config, files_config, accelerator=accelerator)

    def run_uce(self):
        data_loader = self.uce.process_data(self.ann_data)
        embeddings = self.uce.get_embeddings(data_loader)
        return embeddings
