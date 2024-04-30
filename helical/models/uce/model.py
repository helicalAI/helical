import logging
import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader
import json
from pathlib import Path
from helical.models.uce.uce_config import UCEConfig
from helical.models.helical import HelicalBaseModel
# from helical.services.downloader import Downloader
from helical.models.uce.uce_utils import get_ESM2_embeddings, load_model, process_data, get_gene_embeddings
from typing import Union
from accelerate import Accelerator

class UCE(HelicalBaseModel):
    default_config = UCEConfig()

    def __init__(self, model_dir, model_config: UCEConfig = default_config) -> None:
        """Initializes the UCE class

        Parameters
        ----------
        model_dir : str
            The path to the model directory
        model_config : UCEConfig, optional
            The model configuration.

        Returns
        -------
        None
        """
        
        super().__init__()
        self.model_config = model_config.config
        self.model_dir = Path(model_dir)
        self.log = logging.getLogger("UCE-Model")
        # self.downloader = Downloader()

        # self.downloader.download_via_link(Path(self.model_config["model_loc"]), "https://figshare.com/ndownloader/files/42706576")
        # self.downloader.download_via_link(Path(self.files_config["token_file"]), "https://figshare.com/ndownloader/files/42706585")

        token_file = self.model_dir / "all_tokens.torch"
        model_path = self.model_dir / "4layer_model.torch"
        self.embeddings = get_ESM2_embeddings(token_file, self.model_config["token_dim"])
        self.model =  load_model(model_path, self.model_config, self.embeddings)
        self.model = self.model.eval()

        if self.model_config["accelerator"]:
            self.accelerator = Accelerator(project_dir=self.model_dir, cpu=self.model_config["accelerator"]["cpu"])
            self.model = self.accelerator.prepare(self.model)
        else:
            self.accelerator = None

    def process_data(self, data: AnnData, data_config_path: Union[str, Path]) -> DataLoader:
        """Processes the data for the UCE model

        Parameters 
        ----------
        data : AnnData
            The AnnData object containing the data to be processed
        data_config_path : Union[str, Path]
            The path to the data configuration file

        Returns
        -------
        DataLoader
            The DataLoader object containing the processed data
        """
        
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
        """Gets the gene embeddings from the UCE model

        Parameters
        ----------
        dataloader : DataLoader
            The DataLoader object containing the processed data

        Returns
        -------
        np.array
            The gene embeddings in the form of a numpy array
        """
        self.log.info(f"Inference started")
        embeddings = get_gene_embeddings(self.model, dataloader, self.accelerator)
        return embeddings
