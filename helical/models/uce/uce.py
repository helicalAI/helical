import logging
import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader

from helical.models.uce.uce_model import TransformerModel
from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
from helical.models.uce.uce_utils import get_ESM2_embeddings, load_model, process_data, get_gene_embeddings

class UCE(HelicalBaseModel):
    
    def __init__(self, logging_type = LoggingType.CONSOLE, level = LoggingLevel.INFO) -> None:
        super().__init__(logging_type, level)
        self.log = logging.getLogger("UCE-Model")

    def get_model(self, model_config, data_config, files_config, accelerator=None) -> TransformerModel:

        self.model_config = model_config
        self.data_config = data_config
        self.files_config = files_config

        self.embeddings = get_ESM2_embeddings(self.files_config)
        self.model =  load_model(self.model_config, self.embeddings)
        self.model = self.model.eval()

        self.accelerator = accelerator
        if accelerator is not None:
           self.model = accelerator.prepare(self.model)

        return self.model

    def process_data(self, data: AnnData, species="macaca_fascicularis") -> DataLoader:
        loader = process_data(data, 
                              model_config=self.model_config, 
                              files_config=self.files_config,
                              species=species, 
                              accelerator=self.accelerator)
        return loader

    def run(self, dataloader: DataLoader) -> np.array:
        
        self.log.info(f"Inference started")
        embeddings = get_gene_embeddings(self.model, dataloader, self.accelerator)
        return embeddings

    def get_embeddings(self, dataloader: DataLoader) -> np.array:
        
        embeddings = self.run(dataloader)
        return embeddings
