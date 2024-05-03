import logging
import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader
from helical import CACHE_DIR_HELICAL
import os
from pathlib import Path
from helical.models.uce.uce_config import UCEConfig
from helical.models.helical import HelicalBaseModel
from helical.models.uce.uce_utils import get_ESM2_embeddings, load_model, process_data, get_gene_embeddings
from accelerate import Accelerator
from helical.services.downloader import Downloader
from typing import Optional
class UCE(HelicalBaseModel):
    default_config = UCEConfig()

    def __init__(self, model_dir: Optional[str] = None, model_config: UCEConfig = default_config) -> None:
        """Initializes the UCE class

        Parameters
        ----------
        model_dir : str, optional, default = None
            The path to the model directory. None by default, which will download the model if not present.
        model_config : UCEConfig, optional, default = default_config
            The model configuration.

        Returns
        -------
        None
        """
        
        super().__init__()
        self.model_config = model_config.config
        self.log = logging.getLogger("UCE-Model")
        self.downloader = Downloader()
        
        if model_dir is None:
            self.downloader.download_via_name("uce/4layer_model.torch")
            self.downloader.download_via_name("uce/all_tokens.torch")
            self.downloader.download_via_name("uce/species_chrom.csv")
            self.downloader.download_via_name("uce/species_offsets.pkl")
            self.downloader.download_via_name("uce/protein_embeddings/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt")
            self.downloader.download_via_name("uce/protein_embeddings/Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt")
            self.model_dir = Path(os.path.join(self.downloader.CACHE_DIR_HELICAL,'uce'))
        else:
            self.model_dir = Path(model_dir)
        


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

    def process_data(self, data: AnnData, 
                     species: str = "human", 
                     filter_genes: bool = False, 
                     embedding_model: str = "ESM2" ) -> DataLoader:
        """Processes the data for the UCE model

        Parameters 
        ----------
        data : AnnData
            The AnnData object containing the data to be processed
        species: str, optional, default = "human"
            The species of the data. 
        filter_genes: bool, optional, default = False
            Wheter to filter genes or not.
        embedding_model: str, optional, default = "ESM2"
            The name of the gene embedding model.

        Returns
        -------
        DataLoader
            The DataLoader object containing the processed data
        """
        
        files_config = {
            "spec_chrom_csv_path": self.model_dir / "species_chrom.csv",
            "protein_embeddings_dir": self.model_dir / "protein_embeddings/",
            "offset_pkl_path": self.model_dir / "species_offsets.pkl"
        }

        data_loader = process_data(data, 
                              model_config=self.model_config, 
                              files_config=files_config,
                              species=species,
                              filter_genes=filter_genes,
                              embedding_model=embedding_model,
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
