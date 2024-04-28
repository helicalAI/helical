import logging
import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader
import json
from pathlib import Path
from helical.models.uce.uce_model import TransformerModel
from helical.models.helical import HelicalBaseModel
from helical.constants.enums import LoggingType, LoggingLevel
# from helical.services.downloader import Downloader
from helical.models.uce.uce_utils import get_ESM2_embeddings, load_model, process_data, get_gene_embeddings
from typing import Union
from accelerate import Accelerator

class UCE(HelicalBaseModel):
    
    def __init__(self,
                 model_dir, 
                 use_accelerator=True, 
                 logging_type = LoggingType.CONSOLE, 
                 level = LoggingLevel.INFO) -> None:
        
        super().__init__(logging_type, level)
        self.log = logging.getLogger("UCE-Model")
        # self.downloader = Downloader()

        # load model configs via model_dir input
        self.model_dir = model_dir
        with open(self.model_dir / "args.json", "r") as f:
            model_config = json.load(f)

        self.model_config = model_config
        # self.downloader.download_via_link(Path(self.model_config["model_loc"]), "https://figshare.com/ndownloader/files/42706576")
        # self.downloader.download_via_link(Path(self.files_config["token_file"]), "https://figshare.com/ndownloader/files/42706585")

        token_file = self.model_dir / "all_tokens.torch"
        model_path = self.model_dir / "4layer_model.torch"
        self.embeddings = get_ESM2_embeddings(token_file, self.model_config["token_dim"])
        self.model =  load_model(model_path, self.model_config, self.embeddings)
        self.model = self.model.eval()

        if use_accelerator:
            self.accelerator = Accelerator(project_dir=self.model_dir, cpu=self.model_config["accelerator"]["cpu"])
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None

    def process_data(self, data: AnnData, data_config_path: Union[str, Path]) -> DataLoader:
        
        with open(data_config_path) as f:
            config = json.load(f)
        self.data_config = config

        files_config = {
            "spec_chrom_csv_path": self.model_dir / "species_chrom.csv",
            "protein_embeddings_dir": self.model_dir / "protein_embeddings/",
            "offset_pkl_path": self.model_dir / "species_offsets.pkl"
        }

        data_loader = process_data(data, 
                              model_config=self.model_config, 
                              files_config=files_config,
                              data_config=self.data_config,
                              accelerator=self.accelerator)
        return data_loader

    def get_embeddings(self, dataloader: DataLoader) -> np.array:
        self.log.info(f"Inference started")
        embeddings = get_gene_embeddings(self.model, dataloader, self.accelerator)
        return embeddings
